
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "common.hpp"
#include "cumsum.hpp"

#define BLOCKSIZE 1024
// TODO: add an assert, to limit the dimsize less than 256, also limit the number of logits.numel() within limit of int32
// TODO: check when to multiply grad_output to the logits_grad, method is add weights to different categories
// TODO: test case should cover, n_class from 3 to 256


// compare function for sort
template <typename idxT, typename T>
struct CompareSegmentGT {
    CompareSegmentGT(int64_t segment_size): seg_size(segment_size) {}
    __device__ bool operator()(const thrust::tuple<idxT, T, T> &lv, const thrust::tuple<idxT, T, T> &rv) {
        idxT segl = thrust::get<0>(lv) / seg_size;
        idxT segr = thrust::get<0>(rv) / seg_size;
        if (segl == segr) {
            return thrust::get<1>(lv) > thrust::get<1>(rv);
        } else {
            return segl < segr;
        }
    }
const int64_t seg_size;
};


// reduce function for shared memory
template<typename T>
class sum_op {
public:
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a + b;
    }
};

template<template<typename> class Reduction, typename scalar_t>
__device__ __forceinline__ void reduce_op(
        scalar_t* sdata, int blocksize,
        const Reduction<scalar_t>& oper) {
    int tid = threadIdx.x;
    __syncthreads();
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = oper(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}


// kernel function for forward and backward
// TODO: function name here
template<typename scalar_t>
__global__ void compute_errs(const int n_size,
                            const int dimsize, const int m_size,
                            const int ignore_index,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *errs, 
                            scalar_t *one_hot) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n_samples = m_size * n_size;
    const scalar_t one(1.);
    const scalar_t minus_one(-1.);

    for (int i{tid}; i < n_samples; i+=stride) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int e_idx;

        // if ignore index, set values to minus, to send it rear
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            for (int j = 0; j < dimsize; ++j) {
                e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
                errs[e_idx] = minus_one;
            }
            continue;
        }

        // set one hot values
        e_idx = lb * m_size * n_size + n_idx * m_size + m_idx;
        one_hot[e_idx] = one;


        // compute errs: 
        // errs = abs(lb_one_hot - softmax(logits.transpose(0, 1).view(c, -1)))
        scalar_t max_val(-10000.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            if (val > max_val) max_val = val;
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            errs[e_idx] = val;
        }

        scalar_t exp_sum_val(0.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            scalar_t val = errs[e_idx];
            exp_sum_val += math_ops::Exp(val - max_val);
        }
        exp_sum_val =  one / exp_sum_val;

        for (int j{0}; j < dimsize; ++j) {
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            scalar_t val = errs[e_idx];
            errs[e_idx] = math_ops::Exp(val - max_val) * exp_sum_val;
        }
        // (lb_one_hot - probs).abs()
        e_idx = lb * n_size * m_size + n_idx * m_size + m_idx;
        errs[e_idx] = one - errs[e_idx];
    }

}



template<typename scalar_t>
__global__ void compute_n_pos_vals(scalar_t *n_pos, 
        const scalar_t *output, const int n_size, const int m_size) {

    int tid = threadIdx.x;
    int strd = blockDim.x;
    for (int i{tid}; i < n_size; i += strd) {
        int ind = (i + 1) * m_size - 1; 
        n_pos[i] = output[ind];
    }
}


template<typename scalar_t>
__global__ void compute_jacc_iou(const scalar_t *n_pos, 
        scalar_t *output, scalar_t *tmp,
        const int n_size, const int m_size) {

    int n_samples = n_size * m_size;
    int t_size = gridDim.x * blockDim.x;
    const scalar_t one(1);
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i{tid}; i < n_samples; i += t_size) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;
        scalar_t val = output[i];
        scalar_t n_pos_val = n_pos[n_ind];
        scalar_t int_val = n_pos_val - val;
        scalar_t uni_val = n_pos_val - val + scalar_t(m_ind + 1);
        tmp[i] = one - int_val / uni_val;
    }
}


template<typename scalar_t>
__global__ void compute_jacc_diff(scalar_t *errs, scalar_t *output,
        scalar_t *tmp, const int *index, 
        const int n_size, const int m_size) {

    int n_samples = n_size * m_size;
    int t_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i{tid}; i < n_samples; i += t_size) {
        int m_ind = i % m_size;
        scalar_t val;
        if (m_ind == 0) {
            val = tmp[i];
        } else {
            val = tmp[i] - tmp[i - 1];
        }
        int ind = index[i];
        output[ind] = val;
    }
}


template<typename scalar_t>
__global__ void reorder_errs(const scalar_t *errs,
        scalar_t *tmp, const int *index, 
        const int n_size, const int m_size) {

    int n_samples = n_size * m_size;
    int t_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i{tid}; i < n_samples; i += t_size) {
        tmp[index[i]] = errs[i];
    }
}


template<typename scalar_t>
__global__ void reorder_copy_back(scalar_t *errs, const scalar_t *tmp, 
        const int n_size, const int m_size) {

    int n_samples = n_size * m_size;
    int t_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i{tid}; i < n_samples; i += t_size) {
        errs[i] = tmp[i];
    }
}


template<typename scalar_t>
__global__ void mul_reduce_sum_by_row_per_block(scalar_t *errs,
        const scalar_t *jacc, scalar_t *buf, 
        const int n_size, const int m_size) {
    const scalar_t zero(0);

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);

    int bid = blockIdx.y;
    int b_size = gridDim.y;
    int tstride = blockDim.x * gridDim.x;
    for (int i{bid}; i < n_size; i += b_size) {
        shared[threadIdx.x] = zero;
        __syncthreads();
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
        for (int j{tid}; j < m_size; j += tstride) {
            int ind = m_size * i + j;
            scalar_t err_val = errs[ind];
            if (err_val < zero) err_val = zero; // bypass ignore index
            shared[threadIdx.x] += err_val * jacc[ind];
        }
        __syncthreads();
        reduce_op<sum_op, scalar_t>(shared, blockDim.x, sum_op<scalar_t>());
        if (threadIdx.x == 0) {
            int ind = i * gridDim.x + blockIdx.x;
            buf[ind] = shared[0];
        }
    }
}


template<typename scalar_t>
__global__ void reduce_sum_by_row(const scalar_t *buf, scalar_t *loss ,
        const int n_size, const int m_size) {
    const scalar_t zero(0);

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);

    int bid = blockIdx.y;
    int bstrd = gridDim.y;
    for (int i{bid}; i < n_size; i += bstrd) {
        shared[threadIdx.x] = zero;
        __syncthreads();
        int tid = threadIdx.x;
        int tstrd = blockDim.x;
        for (int j{tid}; j < m_size; j += tstrd) {
            int ind = m_size * i + j;
            shared[threadIdx.x] += buf[ind];
        }
        __syncthreads();
        reduce_op<sum_op, scalar_t>(shared, blockDim.x, sum_op<scalar_t>());
        if (threadIdx.x == 0) {
            loss[i] = shared[0];
        }
    }
}


template<typename scalar_t>
__global__ void compute_probs_grad(scalar_t *jacc, 
                            const int64_t *labels, const int ignore_index,
                            const int n_size, const int m_size) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t zero(0.);

    for (int i{tid}; i < m_size; i += stride) {
        int e_idx;

        // set grad to zero if it is ignored index
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            for (int j = 0; j < n_size; ++j) {
                e_idx = j * m_size + i;
                jacc[e_idx] = zero;
            }
            continue;
        }

        // grad = -1 if j == lb else 1
        e_idx = lb * m_size + i;
        jacc[e_idx] = - jacc[e_idx];
    }
}


template<typename scalar_t>
__global__ void compute_softmax(const int n_size, const int dimsize, 
                        const int m_size, const int ignore_index, 
                        const scalar_t *logits, const int64_t *labels,
                        scalar_t *softmax) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n_samples = m_size * n_size;
    const scalar_t one(1.);

    for (int i{tid}; i < n_samples; i+=stride) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int e_idx;

        // if ignore index, set values to minus, to send it rear
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) continue;

        // find max val
        scalar_t max_val(-10000.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            if (val > max_val) max_val = val;
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            softmax[e_idx] = val;
        }

        // compute exp sum
        scalar_t exp_sum_val(0.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            scalar_t val = softmax[e_idx];
            exp_sum_val += math_ops::Exp(val - max_val);
        }
        exp_sum_val =  one / exp_sum_val;

        // compute softmax
        for (int j{0}; j < dimsize; ++j) {
            e_idx = j * n_size * m_size + n_idx * m_size + m_idx;
            scalar_t val = softmax[e_idx];
            softmax[e_idx] = math_ops::Exp(val - max_val) * exp_sum_val;
        }
    }
}


// TODO: there is generally two methods to do it, all depends on first compute S = sum(jac * s), then compute s(jac - S) 
// The first method should be let one thread loop along the dimsize, and compute sum value, and let another loop to to compute the grad, this does not require too much shared memory
// The second method should be depend on shared memory to compute the sum, and let each thread to compute grad
// Current method is more close to the second method
template<typename scalar_t>
__global__ void compute_logits_grad(const int n_size, const int dimsize, 
                        const int m_size, const int ignore_index, 
                        const scalar_t *logits, scalar_t *jacc,
                        scalar_t *grad_logits, const int64_t *labels) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);

    const scalar_t zero(0.);
    const int samplesize = n_size * m_size;
    const int shm_offset = blockDim.y * threadIdx.x * 2;

    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // compute grad of logits, store in jacc
    for (int i{sid}; i < samplesize; i += stride) {

        // TODO: see if we need to shrink blockDim.y to dimsize
        if (threadIdx.y >= dimsize) continue;
        int e_ind = threadIdx.y * samplesize + i;

        // set grad of ignored index to be 0
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            jacc[e_ind] = zero;
            __syncthreads();
        }

        // read to shared memory
        scalar_t s_val(grad_logits[e_ind]); // s
        shared[shm_offset + blockDim.y + threadIdx.y] = jacc[e_ind]; // jac
        shared[shm_offset + threadIdx.y] = shared[shm_offset + blockDim.y + threadIdx.y] * s_val; // s * jac
        __syncthreads();

        // compute softmax grad
        scalar_t g_val(0);
        for (int j{0}; j < dimsize; ++j) {
            if (threadIdx.y == j) {
                g_val += shared[shm_offset + j + blockDim.y] - shared[shm_offset + j]; // (1-s) * jac
            } else {
                g_val += - shared[shm_offset + j]; // -s * jac
            }
        }
        jacc[e_ind] = g_val * s_val; // s * g_val
        __syncthreads();
    }
}


template<typename scalar_t>
__global__ void transpose_logits_grad(const int n_size, const int dimsize, 
                        const int m_size, const scalar_t *jacc,
                        scalar_t *grad_logits) {

    const int samplesize = n_size * dimsize * m_size;
    const int dm_size = dimsize * m_size;

    int tid = blockIdx.x * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.y * blockDim.x * gridDim.x;

    for (int i{tid}; i < samplesize; i += stride) {
        int n_ind = i / dm_size; 
        int d_ind = i % dm_size;
        int m_ind = d_ind % m_size;
        d_ind = d_ind / m_size;
        int e_ind = d_ind * n_size * m_size + n_ind * m_size + m_ind;
        grad_logits[i] = jacc[e_ind];
    }
}



void LovaszComputeJacc(at::Tensor &errs, at::Tensor &output) {

    int n_samples = errs.size(1);
    int dimsize = errs.size(0);
    auto tmp = at::empty_like(errs);
    auto n_pos = at::zeros({dimsize}, errs.options());

    dim3 block(BLOCKSIZE);
    dim3 grid(max(min((int)tmp.numel() / BLOCKSIZE, 4096), 1));

    // sort errs, together with one hot and obtain the order index
    thrust::device_vector<int> index(n_samples * dimsize);
    thrust::sequence(thrust::device, index.begin(), index.end(), 0, 1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "jacc sort", [&] {

        thrust::device_ptr<scalar_t> errs_ptr(errs.data_ptr<scalar_t>());
        thrust::device_ptr<scalar_t> output_ptr(output.data_ptr<scalar_t>());
        auto begin = thrust::make_zip_iterator(thrust::make_tuple(
                    index.begin(), errs_ptr, output_ptr));
        thrust::sort(
                thrust::device, begin, begin + errs.numel(), 
                CompareSegmentGT<int, scalar_t>(n_samples));
    });

    // cumsum
    cumsum_2d_by_row_v2(output);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "jacc forward steps", [&] {

        // set n_pos vals, obtained directly from last number of each line in cumsum
        compute_n_pos_vals<scalar_t><<<dim3(1), block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_pos.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                dimsize, n_samples);

        // compute iou, store in temp memory of tmp
        // TODO: try to use shared memory to store n_pos, so that we could better use bandwidth
        compute_jacc_iou<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_pos.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                dimsize, n_samples);

        // compute iou difference from tmp and store at output, then copy errs to tmp
        // to prepare for re-order of errs
        compute_jacc_diff<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                errs.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                thrust::raw_pointer_cast(&index[0]),
                dimsize, n_samples);

        // re-order errs and copy to tmp
        reorder_errs<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                errs.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                thrust::raw_pointer_cast(&index[0]),
                dimsize, n_samples);
        // copy back from tmp to errs
        reorder_copy_back<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                errs.data_ptr<scalar_t>(),
                tmp.data_ptr<scalar_t>(),
                dimsize, n_samples);
    });
}


at::Tensor LovaszComputeLoss(const at::Tensor &errs, const at::Tensor &jacc) {
    const int n_size = errs.size(0);
    const int m_size = errs.size(1);

    // parallel strategy
    int gridy = 2;
    while (gridy < n_size && gridy <= 32) gridy <<= 1;
    gridy >>= 1;
    gridy = std::max(1, gridy); // limit the parallel number of rows within 1 and 32
    int gridx = std::max(std::min(m_size / BLOCKSIZE, 4096 / gridy), 1);

    dim3 block(BLOCKSIZE);
    dim3 grid(gridx, gridy);

    // allocate memory and cuda grid/block
    auto buf = at::empty({n_size, gridx}, errs.options());
    auto loss = at::empty({n_size}, errs.options());

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "compute loss", [&] {

        // multiply and reduce within each kernel
        int shm = sizeof(scalar_t) * BLOCKSIZE;
        mul_reduce_sum_by_row_per_block<scalar_t><<<grid, block, shm, at::cuda::getCurrentCUDAStream()>>>(
                errs.data_ptr<scalar_t>(),
                jacc.data_ptr<scalar_t>(),
                buf.data_ptr<scalar_t>(),
                n_size, m_size);

        // reduce sum among blocks
        // TODO: bring this parallel settings outside of the lambda
        int blockx = 2;
        while (blockx < gridx) blockx <<= 1;
        shm = sizeof(scalar_t) * blockx;
        reduce_sum_by_row<scalar_t><<<dim3(1, gridy), dim3(blockx), shm, at::cuda::getCurrentCUDAStream()>>>(
                buf.data_ptr<scalar_t>(),
                loss.data_ptr<scalar_t>(),
                n_size, gridx);
    });

    return loss;
}


/* Method */
std::tuple<at::Tensor, at::Tensor> Lovasz_softmax_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");


    // TODO: check n_classes to determine parallel method
    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();
    dim3 grid(std::min(
        samplesize / BLOCKSIZE, 4096));
    dim3 block(BLOCKSIZE);
    // allocate memory and cuda grid/block
    auto errs = at::empty_like(logits).reshape({dimsize, -1});
    auto jacc = at::zeros_like(logits).reshape({dimsize, -1});
    if (errs.numel() == 0 | jacc.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(errs, jacc);
    }

    // call kernel to compute errs
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "errors forward", [&] {
        int shm = sizeof(scalar_t) * dimsize;
        compute_errs<scalar_t><<<grid, block, shm, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, ignore_index,
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<int64_t>(), 
            errs.contiguous().data_ptr<scalar_t>(),
            jacc.contiguous().data_ptr<scalar_t>() // jacc is one hot here
        );
    });
    // compute jacc index, which is re-ordered to the original order
    // so that we could re-use it in backward pass
    LovaszComputeJacc(errs, jacc);

    // reduce sum operation
    // TODO: define the loss tensor outsize, and pass it as an arg of the function
    auto loss = LovaszComputeLoss(errs, jacc);

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(loss, jacc);
}


at::Tensor Lovasz_softmax_backward_cuda(const at::Tensor &grad, const at::Tensor &logits,
                                  const at::Tensor &labels, const at::Tensor jacc,
                                  const int64_t ignore_index) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto grad_logits = at::empty_like(logits);

    // call kernel
    dim3 block(BLOCKSIZE);
    dim3 grid(std::max(1, std::min(samplesize / BLOCKSIZE, 4096)));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lovasz backward probs", [&] {
        // compute grad of probs, store in jacc
        compute_probs_grad<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            jacc.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<int64_t>(), 
            ignore_index, dimsize, samplesize);
        compute_softmax<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, ignore_index,
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<int64_t>(), 
            grad_logits.contiguous().data_ptr<scalar_t>() // store softmax
        );

    });

    int blocky = 32;
    while (blocky < dimsize) blocky += 32;
    int blockx = BLOCKSIZE / blocky;
    int gridx = std::min(4096, std::max(0, samplesize / blockx));
    block = dim3(blockx, blocky);
    grid = dim3(gridx);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lovasz backward logits", [&] {
        // compute grad of logits, store it jacc
        int shm_size = sizeof(scalar_t) * BLOCKSIZE * 2;
        compute_logits_grad<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, ignore_index,
            logits.contiguous().data_ptr<scalar_t>(),
            jacc.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<int64_t>());

        // transpose back to nchw
        transpose_logits_grad<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size,
            jacc.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>());
    });


    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}


// python inferface

std::tuple<at::Tensor, at::Tensor> Lovasz_softmax_forward(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index) {
    if (logits.device().type() != c10::kCUDA) {
        AT_ERROR("this lovasz softmax function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return Lovasz_softmax_forward_cuda(logits, labels, ignore_index);
}

at::Tensor Lovasz_softmax_backward(const at::Tensor &grad, const at::Tensor &logits,
                                  const at::Tensor &labels, at::Tensor jacc,
                                  const int64_t ignore_index) {
    // TODO: try AT_ASSERTM
    if (logits.device().type() != c10::kCUDA || labels.device().type() != c10::kCUDA) {
        AT_ERROR("this lovasz softmax function only supports gpu mode\n");
    }
    at::DeviceGuard guard(logits.device());
    return Lovasz_softmax_backward_cuda(grad, logits, labels, jacc, ignore_index);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lovasz_softmax_forward", &Lovasz_softmax_forward, "lovasz softmax forward");
    m.def("lovasz_softmax_backward", &Lovasz_softmax_backward, "lovasz softmax backward");
}

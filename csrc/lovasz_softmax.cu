
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

#define BLOCKSIZE 512


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

template<typename T>
class gt_op {
public:
    __device__ __forceinline__ T operator()(T a, T b) const {
        /* if (a > b) return a; */
        /* else return b; */
        return (a > b) ? a : b;
    }
};

template<template<typename> class Reduction, typename scalar_t>
__device__ __forceinline__ void reduce_op(
        scalar_t* sdata, int blocksize, const int tid,
        const Reduction<scalar_t>& oper) {
    __syncthreads();
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = oper(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}


// kernel function for forward and backward
template<typename scalar_t>
__global__ void compute_errs(const int n_size, const int m_size,
                            const int ignore_index, const int64_t *labels,
                            scalar_t *errs, scalar_t *one_hot) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t minus_one(-1.);

    for (int i{tid}; i < m_size; i+=stride) {
        int e_ind;

        // if ignore index, set values to minus, to send it rear
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            for (int j = 0; j < n_size; ++j) {
                e_ind = j * m_size + i;
                errs[e_ind] = minus_one;
            }
            continue;
        }
        e_ind = lb * m_size + i;

        // set one hot values
        one_hot[e_ind] = one;

        // compute errs: 
        // errs = abs(lb_one_hot - softmax(logits.transpose(0, 1).view(c, -1)))
        // (lb_one_hot - probs).abs()
        errs[e_ind] = one - errs[e_ind];
    }
}



template<typename scalar_t>
__global__ void compute_jacc_iou(scalar_t *output, scalar_t *tmp,
                    const int n_size, const int m_size) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);
    // load n_pos to shm, n_pos is the last column of cumsum
    if (threadIdx.x < n_size) {
        shared[threadIdx.x] = output[(threadIdx.x + 1) * m_size - 1];
    }
    __syncthreads();

    int n_samples = n_size * m_size;
    int t_size = gridDim.x * blockDim.x;
    const scalar_t one(1);
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i{tid}; i < n_samples; i += t_size) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;
        scalar_t val = output[i];
        scalar_t int_val = shared[n_ind] - val;
        scalar_t uni_val = shared[n_ind] - val + scalar_t(m_ind + 1);
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
        reduce_op<sum_op, scalar_t>(shared, blockDim.x, threadIdx.x, sum_op<scalar_t>());
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
        reduce_op<sum_op, scalar_t>(shared, blockDim.x, threadIdx.x, sum_op<scalar_t>());
        if (threadIdx.x == 0) {
            loss[i] = shared[0];
        }
    }
}


template<typename scalar_t>
__global__ void compute_probs_grad_and_transpose(const scalar_t *jacc, 
                            const scalar_t *grad, scalar_t *grad_logits, 
                            const int64_t *labels, const int n_size,
                            const int dimsize, const int m_size) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const int samplesize = n_size * dimsize * m_size;
    const int dm_size = dimsize * m_size;

    // read to shared memory to save bandwidth
    if (threadIdx.x < dimsize) {
        shared[threadIdx.x] = grad[threadIdx.x];
    }
    __syncthreads();

    int e_ind;
    for (int i{tid}; i < samplesize; i += stride) {
        int n_ind = i / dm_size; 
        int d_ind = i % dm_size;
        int m_ind = d_ind % m_size;
        d_ind = d_ind / m_size;

        e_ind = n_ind * m_size + m_ind;
        int lb = static_cast<int>(labels[e_ind]);
        int e_ind = d_ind * n_size * m_size + n_ind * m_size + m_ind;
        // grad = -1 if j == lb else 1
        if (lb == d_ind) {
            grad_logits[i] = - jacc[e_ind] * shared[d_ind];
        } else {
            grad_logits[i] = jacc[e_ind] * shared[d_ind];
        }
    }
}



template<typename scalar_t>
__global__ void compute_softmax_shallow(const int n_size, const int dimsize, 
                        const int m_size, const scalar_t *logits,
                        scalar_t *softmax) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    int n_samples = m_size * n_size;
    const scalar_t one(1.);

    for (int i{tid}; i < n_samples; i+=stride) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int e_idx;

        // find max val
        scalar_t max_val(-10000.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            if (val > max_val) max_val = val;
        }

        // compute exp sum
        scalar_t exp_sum_val(0.);
        for (int j{0}; j < dimsize; ++j) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            exp_sum_val += math_ops::Exp(val - max_val);
        }
        exp_sum_val =  one / exp_sum_val;

        // compute softmax
        for (int j{0}; j < dimsize; ++j) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            softmax[e_idx] = math_ops::Exp(val - max_val) * exp_sum_val;
        }
    }
}


template<typename scalar_t>
__global__ void compute_softmax_deep(const int n_size, const int dimsize, 
                        const int m_size, const scalar_t *logits,
                        scalar_t *softmax) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);
    shared += blockDim.y * threadIdx.x;

    const int samplesize = n_size * m_size;
    const scalar_t one(1.);

    int tid = threadIdx.y;
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i{sid}; i < samplesize; i += stride) {
        int e_idx;
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        // find max val
        shared[tid] = scalar_t(-10000.);
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.y) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[e_idx];
            if (val > shared[tid]) shared[tid] = val;
        }
        __syncthreads();
        reduce_op<gt_op, scalar_t>(shared, blockDim.y, threadIdx.y, gt_op<scalar_t>());
        scalar_t max_val = shared[0];
        __syncthreads();

        // find exp sum val
        shared[tid] = scalar_t(0.);
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.y) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            shared[tid] += math_ops::Exp(logits[e_idx] - max_val);
        }
        __syncthreads();
        reduce_op<sum_op, scalar_t>(shared, blockDim.y, threadIdx.y, sum_op<scalar_t>());
        if (tid == 0) shared[0] = one / shared[0];
        __syncthreads();

        // compute softmax
        for (int j{tid}; j < dimsize; j += blockDim.y) {
            e_idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            softmax[e_idx] = math_ops::Exp(logits[e_idx] - max_val) * shared[0];
        }
    }
}


template<typename scalar_t>
__global__ void compute_logits_grad_shallow(const int n_size, const int dimsize, 
                        const int m_size, const int ignore_index, 
                        const scalar_t *jacc, scalar_t *grad_logits, 
                        const int64_t *labels) {

    const scalar_t zero(0.);
    const int samplesize = n_size * m_size;

    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // compute grad of logits, store in grad_logits, jacc is softmax
    for (int i{sid}; i < samplesize; i += stride) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;
        int e_ind;

        // set grad of ignored index to be 0
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            for (int j{0}; j < dimsize; ++j) {
                e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                grad_logits[e_ind] = zero;
            }
            continue;
        }

        scalar_t sum(0);
        for (int j{0}; j < dimsize; ++j) {
            e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            sum -= jacc[e_ind] * grad_logits[e_ind];
        }
        for (int j{0}; j < dimsize; ++j) {
            e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            grad_logits[e_ind] = jacc[e_ind] * (sum + grad_logits[e_ind]);
        }
    }
}


template<typename scalar_t>
__global__ void compute_logits_grad_deep(const int n_size, const int dimsize, 
                        const int m_size, const int ignore_index,
                        const scalar_t *jacc, scalar_t *grad_logits,
                        const int64_t *labels) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *shared = reinterpret_cast<scalar_t*>(sdata_raw);

    const scalar_t zero(0.);
    const int samplesize = n_size * m_size;
    const int shm_offset = blockDim.y * threadIdx.x;
    shared += shm_offset;

    int tid = threadIdx.y;
    int sid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // compute grad of logits, store in grad_logits, jacc is softmax
    for (int i{sid}; i < samplesize; i += stride) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;
        int e_ind;

        // set grad of ignored index to be 0
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            for (int j{tid}; j < dimsize; j += blockDim.y) {
                e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                grad_logits[e_ind] = zero;
            }
            continue;
        }

        shared[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.y) {
            e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            shared[tid] -= jacc[e_ind] * grad_logits[e_ind];
        }
        __syncthreads();
        reduce_op<sum_op, scalar_t>(shared, blockDim.y, threadIdx.y, sum_op<scalar_t>());
        for (int j{tid}; j < dimsize; j += blockDim.y) {
            e_ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            grad_logits[e_ind] = jacc[e_ind] * (grad_logits[e_ind] + shared[0]);
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__global__ void transpose_softmax(const int n_size, const int dimsize, 
                        const int m_size, scalar_t *from, scalar_t *to) {

    const int samplesize = n_size * dimsize * m_size;
    const int dm_size = dimsize * m_size;
    const scalar_t zero(0.);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i{tid}; i < samplesize; i += stride) {
        int n_ind = i / dm_size; 
        int d_ind = i % dm_size;
        int m_ind = d_ind % m_size;
        d_ind = d_ind / m_size;
        int e_ind = d_ind * n_size * m_size + n_ind * m_size + m_ind;
        to[e_ind] = from[i];
        from[i] = zero;
    }
}



void LovaszComputeErrsOneHot(const at::Tensor &logits, const at::Tensor &labels, 
                                at::Tensor &errs, at::Tensor &jacc,
                                const int ignore_index) {
    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    int blockx, blocky, gridx;
    dim3 block, grid;
    if (dimsize < 32) {
        block = dim3(BLOCKSIZE);
        grid = dim3(std::max(1, std::min(samplesize / BLOCKSIZE, 4096)));
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "lovasz forward softmax", [&] {
            compute_softmax_shallow<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                logits.contiguous().data_ptr<scalar_t>(), 
                jacc.contiguous().data_ptr<scalar_t>() // store softmax
            );
        });
    } else {
        blocky = 32;
        while (blocky < dimsize) blocky <<= 1;
        blocky >>= 1;
        blocky = std::min(std::max(1, blocky), BLOCKSIZE);
        blockx = BLOCKSIZE / blocky;
        gridx = std::min(4096, std::max(1, samplesize / blockx));
        block = dim3(blockx, blocky);
        grid = dim3(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "lovasz forward softmax", [&] {
            int shm_size = sizeof(scalar_t) * BLOCKSIZE;
            compute_softmax_deep<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                logits.contiguous().data_ptr<scalar_t>(), 
                jacc.contiguous().data_ptr<scalar_t>() // store softmax
            );
        });
    }

    block = dim3(BLOCKSIZE);
    grid = dim3(std::max(1, std::min(samplesize * dimsize / BLOCKSIZE, 4096)));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "lovasz transpose softmax", [&] {
        transpose_softmax<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size,
            jacc.contiguous().data_ptr<scalar_t>(),  // set jacc to all 0
            errs.contiguous().data_ptr<scalar_t>());
    });

    grid = dim3(std::max(1, std::min(samplesize / BLOCKSIZE, 4096)));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "lovasz forwarderrs and one hot", [&] {
        compute_errs<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            dimsize, samplesize, ignore_index,
            labels.contiguous().data_ptr<int64_t>(),
            errs.contiguous().data_ptr<scalar_t>(),
            jacc.contiguous().data_ptr<scalar_t>() // jacc is one hot here
        );
    });
}


void LovaszComputeJacc(at::Tensor &errs, at::Tensor &output) {

    int n_samples = errs.size(1);
    int dimsize = errs.size(0);
    auto tmp = at::empty_like(errs);

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

        // compute iou, store in temp memory of tmp, n_pos is the last colum of cumsum
        int shm = sizeof(scalar_t) * BLOCKSIZE;
        compute_jacc_iou<scalar_t><<<grid, block, shm, at::cuda::getCurrentCUDAStream()>>>(
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


void LovaszComputeLoss(const at::Tensor &errs, const at::Tensor &jacc, const at::Tensor &loss) {
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

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "compute loss per block", [&] {

        // multiply and reduce within each kernel
        int shm = sizeof(scalar_t) * BLOCKSIZE;
        mul_reduce_sum_by_row_per_block<scalar_t><<<grid, block, shm, at::cuda::getCurrentCUDAStream()>>>(
                errs.data_ptr<scalar_t>(),
                jacc.data_ptr<scalar_t>(),
                buf.data_ptr<scalar_t>(),
                n_size, m_size);
    });

    int blockx = 2;
    while (blockx < gridx) blockx <<= 1;
    if (blockx > BLOCKSIZE) blockx = BLOCKSIZE;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(errs.scalar_type(), "compute loss reduce block", [&] {
        // reduce sum among blocks
        int shm = sizeof(scalar_t) * blockx;
        reduce_sum_by_row<scalar_t><<<dim3(1, gridy), dim3(blockx), shm, at::cuda::getCurrentCUDAStream()>>>(
                buf.data_ptr<scalar_t>(),
                loss.data_ptr<scalar_t>(),
                n_size, gridx);
    });
}


/* Method */
std::tuple<at::Tensor, at::Tensor> Lovasz_softmax_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");
    AT_ASSERTM(logits.numel() < (1L << 31), "input tensor too large, int32 type will overflow");
    AT_ASSERTM(logits.size(1) < BLOCKSIZE, "num of classes should be less than BLOCKSIZE");


    // allocate memory and cuda grid/block
    const int dimsize = logits.size(1);
    auto errs = at::empty_like(logits).reshape({dimsize, -1});
    auto jacc = at::empty_like(logits).reshape({dimsize, -1});
    auto loss = at::empty({dimsize}, logits.options());
    if (errs.numel() == 0 | jacc.numel() == 0 | loss.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(errs, jacc);
    }

    // Compute errs and one hot
    LovaszComputeErrsOneHot(logits, labels, errs, jacc, ignore_index);

    // compute jacc index, which is re-ordered to the original order
    // so that we could re-use it in backward pass
    LovaszComputeJacc(errs, jacc);

    // reduce sum operation
    LovaszComputeLoss(errs, jacc, loss);

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
    int blockx, blocky, gridx;
    dim3 block, grid;

    gridx = std::max(1, std::min(samplesize * dimsize / BLOCKSIZE, 4096));
    block = dim3(BLOCKSIZE);
    grid = dim3(gridx);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lovasz backward probs", [&] {
        // compute grad of probs, just multiply to jacc
        // store at grad_logits and change from dnm to ndm layout
        int shm = BLOCKSIZE * sizeof(scalar_t);
        compute_probs_grad_and_transpose<scalar_t><<<grid, block, shm, at::cuda::getCurrentCUDAStream()>>>(
            jacc.contiguous().data_ptr<scalar_t>(),
            grad.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<int64_t>(), 
            n_size, dimsize, m_size);
    });

    // from now on, grad_probs is stored in grad_logits, softmax is on jacc
    // compute grad of logits, store it grad_logits
    if (dimsize < 32) {
        block = dim3(BLOCKSIZE);
        grid = dim3(std::max(1, std::min(samplesize / BLOCKSIZE, 4096)));
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lovasz backward logits", [&] {
            compute_softmax_shallow<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                logits.contiguous().data_ptr<scalar_t>(), 
                jacc.contiguous().data_ptr<scalar_t>() // store softmax
            );
            compute_logits_grad_shallow<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, ignore_index,
                jacc.contiguous().data_ptr<scalar_t>(),
                grad_logits.contiguous().data_ptr<scalar_t>(),
                labels.contiguous().data_ptr<int64_t>()
            );
        });
    } else {
        blocky = 32;
        while (blocky < dimsize) blocky <<= 1;
        blocky >>= 1;
        blocky = std::min(std::max(1, blocky), BLOCKSIZE);
        blockx = BLOCKSIZE / blocky;
        gridx = std::min(4096, std::max(1, samplesize / blockx));
        block = dim3(blockx, blocky);
        grid = dim3(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lovasz backward logits", [&] {
            int shm_size = sizeof(scalar_t) * BLOCKSIZE;
            compute_softmax_deep<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                logits.contiguous().data_ptr<scalar_t>(), 
                jacc.contiguous().data_ptr<scalar_t>() // store softmax
            );
            compute_logits_grad_deep<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, ignore_index,
                jacc.contiguous().data_ptr<scalar_t>(),
                grad_logits.contiguous().data_ptr<scalar_t>(),
                labels.contiguous().data_ptr<int64_t>()
            );
        });
    }

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

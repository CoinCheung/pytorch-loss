
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include <iostream>
#include "common.hpp"

using std::cout;
using std::endl;

#define BLOCKSIZE 512

// TODO: 
// at::numeric_limits<scalar_t>::lowest;

// implement like pytorch-softmax: two kernels: one is for inner size to be 1, and the other is for spatial. Besides, in the spatial kernel method, we should use threadIdx.x and threadIdx.y for dimsize and inner size parallelization
// define spatial kernel block like this: 
/* 
 * inline dim3 SpatialSoftMax_getBlockSize(
 *   uint64_t outer_size, uint64_t dim_size, uint64_t inner_size) {
 *   uint32_t inner_threads = inner_size;
const int max_threads = 1024;
 *   inner_threads = std::min(inner_threads, static_cast<uint32_t>(max_threads));
 *   uint32_t dim_threads = 1;
 *   if (inner_threads <= 64 && dim_size >= 64) {
 *     while (inner_threads * dim_threads <= max_threads && dim_threads <= dim_size)
 *       dim_threads *= 2;
 *     dim_threads /= 2;
 *   }
 *   return dim3(dim_threads, inner_threads);
 * }
 *  */
// consider max_active_blocks when assign grid blocks, the total number of blocks should not be greater than max_active_blocks which is multiProcessCount


namespace large_margin_space {

template<typename scalar_t>
__forceinline__ __device__ void reduce_max(scalar_t* sdata, int tid) {
    __syncthreads();
    for (unsigned int s{blockDim.x / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__forceinline__ __device__ void reduce_sum(scalar_t* sdata, int tid) {
    __syncthreads();
    for (unsigned int s{blockDim.x / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}

template<typename scalar_t>
__forceinline__ __device__ void compute_reduce_values(
        const scalar_t* logits, scalar_t* sdata,
        const int dimsize, const int m_size, 
        int n_idx, int m_idx, int64_t lb, int tid) {
    // b is max logits without target 
    // b+1 is max logits with target 
    // b+2 is sum of exp without target 
    // b+3 is sum of exp with target 

    // compute max with and without label index
    const scalar_t zero(0.);
    __syncthreads();
    sdata[tid] = scalar_t(-10000.);
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        if (j == lb) continue;
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        if (val > sdata[tid]) sdata[tid] = val;
    }
    reduce_max(sdata, tid);
    if (tid == 0) {
        sdata[blockDim.x] = sdata[0];
        sdata[blockDim.x + 1] = sdata[0];
        int idx = n_idx * dimsize * m_size + lb * m_size + m_idx;
        scalar_t val = logits[idx];
        if (val > sdata[0]) sdata[blockDim.x + 1] = val;
    }
    __syncthreads();

    // compute sum of exp with and without label index
    sdata[tid] = zero;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        if (j == lb) continue;
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        sdata[tid] += exp(val - sdata[blockDim.x]);
    }
    reduce_sum<scalar_t>(sdata, tid);
    if (tid == 0) sdata[blockDim.x + 2] = sdata[0];
    __syncthreads();

    sdata[tid] = zero;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        sdata[tid] += exp(val - sdata[blockDim.x + 1]);
    }
    reduce_sum<scalar_t>(sdata, tid);
    if (tid == 0) sdata[blockDim.x + 3] = sdata[0];
}


template<typename scalar_t>
__forceinline__ __device__ void compute_sum_of_qx(
        const scalar_t* logits, scalar_t* sdata,
        const int dimsize, const int m_size, 
        int n_idx, int m_idx, int64_t lb, int tid) {
    // compute sum of q * x to sdata[blockDim.x + 5]
    const scalar_t zero(0.);
    __syncthreads();
    sdata[tid] = zero;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        if (j == lb) continue;
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
        scalar_t val = logits[idx];
        sdata[tid] += val * exp(val - sdata[blockDim.x]);
    }
    reduce_sum<scalar_t>(sdata, tid);
    if (tid == 0) {
        sdata[blockDim.x + 5] = sdata[0] / sdata[blockDim.x + 2]; 
    }
}

}



// kernel function for forward and backward
template<typename scalar_t>
__global__ void LMarginLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float lam) {
    // shared memory
    // b+4 is coeff of 1/(dimsize - 1)
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + (blockDim.x + 8) * threadIdx.y;
    scalar_t zero(0.f);

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    int sample_offset = gridDim.x * blockDim.y;

    if (tid == 0) {
        sdata[blockDim.x + 4] = scalar_t(1.) / (dimsize - 1);
    }

    int samplesize = n_size * m_size;
    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        int64_t lb = labels[i];
        if (lb == ignore_index) {
            if (tid == 0) losses[i] = zero;
            continue;
        } 
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        large_margin_space::compute_reduce_values<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            scalar_t term(0);
            if (j == lb) {
                term = -(dval - sdata[blockDim.x + 1]);
                term += log(sdata[blockDim.x + 3]);
            } else {
                dval -= sdata[blockDim.x];
                term = exp(dval) / sdata[blockDim.x + 2];
                term -= sdata[blockDim.x + 4];
                term *= (dval - log(sdata[blockDim.x + 2]));
                term *= scalar_t(lam / 2.f);
            }
            sdata[tid] += term;
        }
        large_margin_space::reduce_sum<scalar_t>(sdata, tid);
        if (tid == 0) losses[i] = sdata[0];
    }
}


template<typename scalar_t>
__global__ void LMarginLossBackward(const int n_size,
                            const int dimsize, const int m_size,
                            scalar_t *grad_logits,
                            const scalar_t *logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float lam) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + (blockDim.x + 8) * threadIdx.y;
    scalar_t zero(0.f);

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    int sample_offset = gridDim.x * blockDim.y;

    if (tid == 0) {
        sdata[blockDim.x + 4] = 1. / (dimsize - 1);
    }

    int samplesize = n_size * m_size;
    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        int64_t lb = labels[i];
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        if (lb == ignore_index) {
            for (int j{tid}; j < dimsize; j += blockDim.x) {
                int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
                grad_logits[idx] = zero;
            }
            continue;
        } 
        large_margin_space::compute_reduce_values<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);
        large_margin_space::compute_sum_of_qx<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);

        const scalar_t one(1.f);
        for (int j{tid}; j < dimsize; j += blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t val = logits[idx];
            scalar_t pc = exp(val - sdata[blockDim.x + 1]) / sdata[blockDim.x + 3];
            scalar_t gval;
            if (j == lb) {
                gval = pc - one;
            } else {
                gval = val - sdata[blockDim.x + 5] + one;
                gval *= exp(val - sdata[blockDim.x]) / sdata[blockDim.x + 2];
                gval = pc + (gval - sdata[blockDim.x + 4]) * scalar_t(lam / 2.);
            }
            grad_logits[idx] = gval;
        }
    }
}


template<typename scalar_t>
__global__ void SpatialLMarginLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float lam) {
    // shared memory
    __shared__ int sdata[BLOCKSIZE];

    sdata[0] = blockIdx.x * blockDim.x + threadIdx.x; //tid 
    sdata[1] = n_size * m_size; // samplesize
    sdata[2] = gridDim.x * blockDim.x; // sample_offset

    for (int i{sdata[0]}; i < sdata[1]; i += sdata[2]) {
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            losses[i] = scalar_t(0.f);
            continue;
        } 
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        // compute max
        scalar_t max_with_lb(-10000.f);
        scalar_t max_no_lb(-10000.f);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (val > max_with_lb) max_with_lb = val;
            if (j == lb) continue;
            if (val > max_no_lb) max_no_lb = val;
        }
        // compute sum of exp
        scalar_t sum_with_lb(0.);
        scalar_t sum_no_lb(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            sum_with_lb += exp(val - max_with_lb);
            if (j == lb) continue;
            sum_no_lb += exp(val - max_no_lb);
        }
        // compute loss
        scalar_t loss_val(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (j == lb) {
                loss_val += - (val - max_with_lb) + log(sum_with_lb); 
            } else {
                loss_val += scalar_t(lam / 2.) * (exp(val - max_no_lb) / sum_no_lb - (scalar_t(1.) / (dimsize - 1))) * (val - max_no_lb - log(sum_no_lb));
            }
        }
        losses[i] = loss_val;
    }
}


template<typename scalar_t>
__global__ void SpatialLMarginLossBackward(const int n_size,
                            const int dimsize, const int m_size,
                            scalar_t *grad_logits,
                            const scalar_t *logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float lam) {
    // shared memory
    __shared__ int sdata[BLOCKSIZE];

    sdata[0] = blockIdx.x * blockDim.x + threadIdx.x; //tid 
    sdata[1] = n_size * m_size; // samplesize
    sdata[2] = gridDim.x * blockDim.x; // sample_offset

    const scalar_t one(1.);

    for (int i{sdata[0]}; i < sdata[1]; i += sdata[2]) {
        int lb = static_cast<int>(labels[i]);
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        if (lb == ignore_index) {
            for (int j{0}; j < dimsize; ++j) {
                int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
                grad_logits[idx] = scalar_t(0.f);
            }
            continue;
        } 

        // compute max
        scalar_t max_with_lb(-10000.);
        scalar_t max_no_lb(-10000.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (val > max_with_lb) max_with_lb = val;
            if (j == lb) continue;
            if (val > max_no_lb) max_no_lb = val;
        }
        // compute sum of exp
        scalar_t sum_with_lb(0.);
        scalar_t sum_no_lb(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            sum_with_lb += exp(val - max_with_lb);
            if (j == lb) continue;
            sum_no_lb += exp(val - max_no_lb);
        }
        // compute sum of qx
        scalar_t sum_qx(0.);
        for (int j{0}; j < dimsize; ++j) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            sum_qx += val * exp(val - max_no_lb) / sum_no_lb;
        }
        // compute grads
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (lb == j) {
                grad_logits[idx] = exp(val - max_with_lb) / sum_with_lb - one;
            } else {
                grad_logits[idx] = exp(val - max_with_lb) / sum_with_lb + scalar_t(lam / 2.) * ((val + one - sum_qx) * exp(val - max_no_lb) / sum_no_lb - (one / (dimsize - 1)));
            }
        }
    }
}

// cuda forward and backward
at::Tensor large_margin_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto losses = torch::empty_like(labels, logits.options());
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }

    // call kernel
    if (dimsize < 32 && samplesize > 4096) {
        int gridx = std::max(std::min(4096, samplesize / BLOCKSIZE), 1);
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "large margin forward", [&] {
            int shm_size = BLOCKSIZE * sizeof(scalar_t);
            SpatialLMarginLossForward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                losses.contiguous().data_ptr<scalar_t>(),
                ignore_index, lam 
            );
        });
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(BLOCKSIZE, blockx / 2), 32);
        int blocky = std::max(std::min(samplesize, BLOCKSIZE / blockx), 1);
        int gridx = std::max(std::min(4096, samplesize / blocky), 1);
        int n_shm = (blockx + 8) * blocky;
        dim3 block(blockx, blocky);
        dim3 grid(gridx);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "large margin forward", [&] {
            int shm_size = n_shm * sizeof(scalar_t);
            LMarginLossForward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                losses.contiguous().data_ptr<scalar_t>(),
                ignore_index, lam 
            );
        });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return losses;
}


at::Tensor large_margin_backward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);
    if (grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_logits;
    }

    if (dimsize < 32 && samplesize > 4096) {
        int gridx = std::max(std::min(4096, samplesize / BLOCKSIZE), 1);
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "large margin backwrd", [&] {
            int shm_size = BLOCKSIZE * sizeof(scalar_t);
            SpatialLMarginLossBackward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                grad_logits.contiguous().data_ptr<scalar_t>(),
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                ignore_index, lam 
            );
        });
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(BLOCKSIZE, blockx / 2), 32);
        int blocky = std::max(std::min(samplesize, BLOCKSIZE / blockx), 1);
        int gridx = std::max(std::min(4096, samplesize / blocky), 1);
        int n_shm = (blockx + 8) * blocky;
        dim3 block(blockx, blocky);
        dim3 grid(gridx);

        // call kernel
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "large margin backwrd", [&] {
            int shm_size = n_shm * sizeof(scalar_t); 
            LMarginLossBackward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                grad_logits.contiguous().data_ptr<scalar_t>(),
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                ignore_index, lam 
            );
        });
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor large_margin_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float lam,
                             const int64_t ignore_index) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_forward_cuda(logits, labels, ignore_index, lam);
}


at::Tensor large_margin_backward(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float lam,
                                  const int64_t ignore_index) {
    // TODO: try AT_ASSERTM
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_backward_cuda(logits, labels, ignore_index, lam);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l_margin_forward", &large_margin_forward, "large margin forward");
    m.def("l_margin_backward", &large_margin_backward, "large margin backward");
}

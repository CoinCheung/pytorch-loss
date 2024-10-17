
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


#define BLOCKSIZE 256
#define GRIDSIZE_MAX 4096

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


using block_ops::reduce_max_shm;
using block_ops::reduce_sum_shm;
using block_ops::reduce_max_shfl;
using block_ops::reduce_sum_shfl;

using math_ops::Exp;
using math_ops::Log;
using math_ops::Log1p;


namespace large_margin_space {


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
    reduce_max_shm(sdata, tid);
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
    reduce_sum_shm<scalar_t>(sdata, tid);
    if (tid == 0) sdata[blockDim.x + 2] = sdata[0];
    __syncthreads();

    sdata[tid] = zero;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        sdata[tid] += exp(val - sdata[blockDim.x + 1]);
    }
    reduce_sum_shm<scalar_t>(sdata, tid);
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
    reduce_sum_shm<scalar_t>(sdata, tid);
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
                term = exp(dval) / sdata[blockDim.x + 2]; // q
                term -= sdata[blockDim.x + 4]; // q - coeff
                term *= (dval - log(sdata[blockDim.x + 2])); // (q - coeff) * log(q)
                term *= scalar_t(lam / 2.f);
            }
            sdata[tid] += term;
        }
        reduce_sum_shm<scalar_t>(sdata, tid);
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
            scalar_t pc = exp(val - sdata[blockDim.x + 1]) / sdata[blockDim.x + 3]; // p
            scalar_t gval;
            if (j == lb) {
                gval = pc - one; // p - 1
            } else {
                gval = val - sdata[blockDim.x + 5] + one; // x - 1 - sum_qx
                gval *= exp(val - sdata[blockDim.x]) / sdata[blockDim.x + 2]; // q * (x - 1 - sum_qx)
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


template<typename scalar_t>
__global__ void SpatialLMarginLossForwardBackward(const int n_size,
                                            const int dimsize, const int m_size,
                                            scalar_t *losses,
                                            scalar_t *grad_logits,
                                            int64_t *valid_cnt,
                                            const scalar_t *logits,
                                            const int64_t *labels,
                                            const int64_t ignore_index,
                                            const float lam) {
    // const int samplesize = n_size * m_size;
    // const int stride = gridDim.x * blockDim.x;
    __shared__ int samplesize;
    __shared__ int stride;
    __shared__ scalar_t coeff;
    samplesize = n_size * m_size;
    stride = gridDim.x * blockDim.x;
    coeff = scalar_t(1. / (dimsize - 1));

    int64_t cnt = 0;
    for (int i{static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)};
            i < samplesize; i += stride) {
        const int lb = static_cast<int>(labels[i]);
        // int n_idx = i / m_size;
        // int m_idx = i % m_size;
        // int idx = n_idx * dimsize * m_size + j * m_size + m_idx;

        if (lb == ignore_index) {
            losses[i] = scalar_t(0.f);
            for (int j{0}; j < dimsize; ++j) {
                int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size);
                grad_logits[idx] = scalar_t(0.f);
            }
            continue;
        }
        ++cnt;

        // compute max, sum exp and sum_qx
        scalar_t max_val(-10000.);
        scalar_t sum_with_lb(0.);
        scalar_t sum_no_lb(0.);
        scalar_t sum_qx(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size);
            scalar_t val = logits[idx];
            if (val > max_val) {
                sum_with_lb *= Exp(max_val - val);
                sum_no_lb *= Exp(max_val - val);
                sum_qx *= Exp(max_val - val);
                max_val = val;
            }
            sum_with_lb += Exp(val - max_val);
            sum_no_lb = (j == lb) ? sum_no_lb : sum_no_lb + Exp(val - max_val);
            sum_qx = (j == lb) ? sum_qx : sum_qx + val * Exp(val - max_val);
        }
        sum_qx /= sum_no_lb;

        // compute losses and grads
        scalar_t loss_val(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size);
            scalar_t val = logits[idx];

            /* 
             * Method: 
             * p = exp(x) / sum_exp_with_lb
             * q = exp(x) / sum_exp_no_lb, only negative
             * coeff = 1 / (n_classes - 1)
             * positive:
             *     loss = log(p)
             *     grad = p - 1
             * negative:
             *     loss = (lam / 2) * (q - coeff) * log(q)
             *     grad = p + (lam / 2) * ( (x + 1 - sum(q * x)) * q - coeff ) */

            const scalar_t p = Exp(val - max_val) / sum_with_lb;
            const scalar_t q = Exp(val - max_val) / sum_no_lb;

            if (lb == j) {
                loss_val += - (val - max_val) + Log(sum_with_lb);
                val = p - scalar_t(1.);
            } else {
                loss_val += scalar_t(lam / 2.) * (q - coeff) * (val - max_val - Log(sum_no_lb));
                val = p + scalar_t(lam / 2.) * ((val + scalar_t(1.) - sum_qx) * q - coeff);
            }
            grad_logits[idx] = val;
        }
        losses[i] = loss_val;
    }

    reduce_sum_shfl(cnt, false);
    if (threadIdx.x == 0) {
        atomicAdd(valid_cnt, cnt);
    }
}


template<typename scalar_t>
__global__ void LMarginLossForwardBackward(const int n_size,
                            const int dimsize, const int m_size,
                            scalar_t *losses,
                            scalar_t *grad_logits,
                            int64_t *valid_cnt,
                            const scalar_t *logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float lam) {
    
    __shared__ int samplesize;
    __shared__ int64_t cnt;
    __shared__ int lb;
    __shared__ scalar_t coeff;
    if (threadIdx.x == 0) cnt = 0;
    samplesize = n_size * m_size;
    coeff = scalar_t(1. / (dimsize - 1));

    for (int i{static_cast<int>(blockIdx.x)}; i < samplesize; i += gridDim.x) {

        if (threadIdx.x == 0) {
            lb = static_cast<int>(labels[i]);
        }
        __syncthreads();

        // int n_idx = i / m_size;
        // int m_idx = i % m_size;
        // int idx = n_idx * dimsize * m_size + j * m_size + m_idx;

        if (lb == ignore_index) {
            if (threadIdx.x == 0) {
                losses[i] = scalar_t(0.f);
            }

            for (int j{threadIdx.x}; j < dimsize; j += blockDim.x) {
                int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size); 
                grad_logits[idx] = scalar_t(0.f);
            }
            continue;
        } 

        if (threadIdx.x == 0) {
            ++cnt;
        }

        // compute max, sum exp and sum_qx
        scalar_t max_val(-10000.);
        scalar_t sum_with_lb(0.);
        scalar_t sum_no_lb(0.);
        scalar_t sum_qx(0.);
        for (int j{threadIdx.x}; j < dimsize; j += blockDim.x) {
            int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size);
            scalar_t val = logits[idx];
            if (val > max_val) {
                sum_with_lb *= Exp(max_val - val);
                sum_no_lb *= Exp(max_val - val);
                sum_qx *= Exp(max_val - val);
                max_val = val;
            }
            sum_with_lb += Exp(val - max_val);
            sum_no_lb = (j == lb) ? sum_no_lb : sum_no_lb + Exp(val - max_val);
            sum_qx = (j == lb) ? sum_qx : sum_qx + val * Exp(val - max_val);
        }
        scalar_t tmp = max_val;
        reduce_max_shfl(tmp, true); // max of whole block
        sum_with_lb *= Exp(max_val - tmp);
        sum_no_lb *= Exp(max_val - tmp);
        sum_qx *= Exp(max_val - tmp);
        max_val = tmp;
        reduce_sum_shfl(sum_with_lb, true);
        reduce_sum_shfl(sum_no_lb, true);
        reduce_sum_shfl(sum_qx, true);
        sum_qx /= sum_no_lb;

        // compute losses and grads
        scalar_t loss_val(0.);
        for (int j{threadIdx.x}; j < dimsize; j += blockDim.x) {
            int idx = (i / m_size) * dimsize * m_size + j * m_size + (i % m_size);
            scalar_t val = logits[idx];

            /* 
             * Method: 
             * p = exp(x) / sum_exp_with_lb
             * q = exp(x) / sum_exp_no_lb, only negative
             * coeff = 1 / (n_classes - 1)
             * positive:
             *     loss = log(p)
             *     grad = p - 1
             * negative:
             *     loss = (lam / 2) * (q - coeff) * log(q)
             *     grad = p + (lam / 2) * ( (x + 1 - sum(q * x)) * q - coeff ) */


            const scalar_t p = Exp(val - max_val) / sum_with_lb;
            const scalar_t q = Exp(val - max_val) / sum_no_lb;

            if (lb == j) {
                loss_val += - (val - max_val) + Log(sum_with_lb); 
                val = p - scalar_t(1.);
            } else {
                loss_val += scalar_t(lam / 2.) * (q - coeff) * (val - max_val - Log(sum_no_lb));
                val = p + scalar_t(lam / 2.) * ((val + scalar_t(1.) - sum_qx) * q - coeff);
            }
            grad_logits[idx] = val;
        }
        reduce_sum_shfl(loss_val, false);
        if (threadIdx.x == 0) {
            losses[i] = loss_val;
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(valid_cnt, cnt);
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

    if (dimsize < 64) {
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


std::tuple<at::Tensor, at::Tensor, at::Tensor> large_margin_forward_backward_cuda(const at::Tensor &logits,
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
    auto grad_logits = torch::empty_like(logits);
    auto options_cnt = torch::TensorOptions()
                    .dtype(torch::kInt64)
                    .device(logits.options().device())
                    .requires_grad(false);
    auto valid_cnt = torch::zeros({1}, options_cnt);
    if (grad_logits.numel() == 0 or losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(losses, grad_logits, valid_cnt);
    }

    if (dimsize < 64) {
    // if (false) {
        int gridx = std::max(std::min(GRIDSIZE_MAX, samplesize / BLOCKSIZE), 1);
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);

        AT_DISPATCH_FLOATING_TYPES(grad_logits.scalar_type(), "spatial large margin forward backwrd", [&] {
            SpatialLMarginLossForwardBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                losses.contiguous().data_ptr<scalar_t>(),
                grad_logits.contiguous().data_ptr<scalar_t>(),
                valid_cnt.data_ptr<int64_t>(),
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                ignore_index, lam 
            );
        });
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(BLOCKSIZE, blockx / 4), 32);
        int gridx = std::max(std::min(4096, samplesize), 1);
        dim3 block(blockx);
        dim3 grid(gridx);

        // call kernel
        AT_DISPATCH_FLOATING_TYPES(grad_logits.scalar_type(), "large margin forward backwrd", [&] {
            LMarginLossForwardBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                losses.contiguous().data_ptr<scalar_t>(),
                grad_logits.contiguous().data_ptr<scalar_t>(),
                valid_cnt.data_ptr<int64_t>(),
                logits.contiguous().data_ptr<scalar_t>(),
                labels.contiguous().data_ptr<int64_t>(),
                ignore_index, lam
            );
        });
    }
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(losses, grad_logits, valid_cnt);
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


std::tuple<at::Tensor, at::Tensor, at::Tensor> large_margin_forward_backward(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float lam,
                                  const int64_t ignore_index) {
    // TODO: try AT_ASSERTM
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_forward_backward_cuda(logits, labels, ignore_index, lam);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l_margin_forward", &large_margin_forward, "large margin forward");
    m.def("l_margin_backward", &large_margin_backward, "large margin backward");
    m.def("l_margin_forward_backward", &large_margin_forward_backward, "large margin forward backward");
}

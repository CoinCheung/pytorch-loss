
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


namespace lsr_space {

template<typename scalar_t>
__forceinline__ __device__ void reduce_sum(scalar_t *sdata, int blocksize, int tid) {
    __syncthreads();
    // NOTE: block size should be 2 ** x
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // // reduce between warps
    // if (blocksize >= 1024) {
    //     if (tid < 512) sdata[tid] += sdata[tid + 512];
    //     __syncthreads();
    // }
    // if (blocksize >= 512) {
    //     if (tid < 256) sdata[tid] += sdata[tid + 256];
    //     __syncthreads();
    // }
    // if (blocksize >= 256) {
    //     if (tid < 128) sdata[tid] += sdata[tid + 128];
    //     __syncthreads();
    // }
    // if (blocksize >= 128) {
    //     if (tid < 64) sdata[tid] += sdata[tid + 64];
    //     __syncthreads();
    // }
    // // reduce within warps
    // if (tid < 32) {
    //     if (blocksize >= 64) sdata[tid] += sdata[tid + 32];
    //     if (blocksize >= 32) sdata[tid] += sdata[tid + 16];
    //     if (blocksize >= 16) sdata[tid] += sdata[tid +  8];
    //     if (blocksize >=  8) sdata[tid] += sdata[tid +  4];
    //     if (blocksize >=  4) sdata[tid] += sdata[tid +  2];
    //     if (blocksize >=  2) sdata[tid] += sdata[tid +  1];
    // }
}

}

// kernel function for forward and backward
template<typename scalar_t>
__global__ void LSRLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *log_scores,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float smooth) {
    // shared memory
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    int shm_offset = blockDim.x;
    int sample_offset = gridDim.x * blockDim.y;
    sdata = sdata + shm_offset * threadIdx.y;
    scalar_t zero(0.f);

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    const scalar_t lb_pos(1.f - smooth);
    const scalar_t lb_neg = smooth / dimsize;
    int samplesize = n_size * m_size;

    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int64_t lb = labels[i];

        if (lb == ignore_index) {
            if (tid == 0) losses[i] = zero;
            continue;
        }

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            if (j == lb) {
                sdata[tid] += -log_scores[idx] * lb_pos;
            } else {
                sdata[tid] += -log_scores[idx] * lb_neg;
            }
        }
        lsr_space::reduce_sum<scalar_t>(sdata, blockDim.x, tid);
        if (tid == 0) losses[i] = sdata[0];
        __syncthreads();
    }
}


template<typename scalar_t>
__global__ void LSRLossBackward(const int n_size,
                            const int dimsize, const int m_size,
                            scalar_t *grad_logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float smooth) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t lb_pos(1.f - smooth);
    const scalar_t lb_neg = smooth / dimsize;
    const scalar_t sumy = lb_neg * (dimsize - 1) + lb_pos;

    int samplesize = n_size * dimsize * m_size;
    int n_offset = dimsize * m_size;
    for (int i{tid}; i < samplesize; i += stride) {
        int n_idx = i / n_offset;
        int dim_idx = (i % n_offset) / m_size;
        int m_idx = (i % n_offset) % m_size;
        int64_t lb = labels[n_idx * m_size + m_idx];

        scalar_t gradval(0);
        if (lb != ignore_index) {
            if (lb == dim_idx) {
                gradval = sumy * grad_logits[i] - lb_pos;
            } else {
                gradval = sumy * grad_logits[i] - lb_neg;
            }
        }
        grad_logits[i] = gradval;
    }
}


template<typename scalar_t>
__global__ void SpatialLSRLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *log_scores,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float smooth) {
    // shared memory
    // TODO: check this setting
    __shared__ int sdata[BLOCKSIZE];
    sdata[0] = blockIdx.x * blockDim.x + threadIdx.x; //tid 
    sdata[1] = n_size * m_size; // samplesize
    sdata[2] = gridDim.x * blockDim.x; // sample_offset

    const scalar_t lb_pos(1.f - smooth);
    const scalar_t lb_neg = smooth / dimsize;

    for (int i{sdata[0]}; i < sdata[1]; i += sdata[2]) {
        int lb = static_cast<int>(labels[i]);
        if (lb == ignore_index) {
            losses[i] = scalar_t(0.);
            continue;
        }
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        scalar_t loss_val(0);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            if (j == lb) {
                loss_val -= lb_pos * log_scores[idx];
            } else {
                loss_val -= lb_neg * log_scores[idx];
            }
        }
        losses[i] = loss_val;
    }

}




// cuda forward and backward
at::Tensor LSR_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto losses = torch::zeros_like(labels, logits.options());
    auto log_scores = torch::log_softmax(logits, 1);
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }

    if (dimsize < 32 && samplesize > 4096) {
        int blockx = 32;
        while (blockx < samplesize && blockx < BLOCKSIZE) blockx *= 2;
        int gridx = std::max(std::min(4096, samplesize / BLOCKSIZE), 1);
        dim3 block(blockx);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "lsr forward", [&] {
            int shm_size = BLOCKSIZE * sizeof(scalar_t); 
            SpatialLSRLossForward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                log_scores.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                losses.contiguous().data_ptr<scalar_t>(),
                ignore_index, smooth
            );
        });
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min((int)BLOCKSIZE, blockx / 2), (int)32);
        int blocky = std::min(samplesize, (int)(BLOCKSIZE / blockx));
        int gridx = std::max(1, std::min(4096, (int)(samplesize / blocky)));
        int n_shm = blockx * blocky;
        dim3 block(blockx, blocky);
        dim3 grid(gridx);

        // call kernel
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "lsr forward", [&] {
            int shm_size = n_shm * sizeof(scalar_t); 
            LSRLossForward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                log_scores.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<int64_t>(), 
                losses.contiguous().data_ptr<scalar_t>(),
                ignore_index, smooth
            );
        });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return losses;
}


at::Tensor LSR_backward_cuda(const at::Tensor &logits,
                              const at::Tensor &labels,
                              const int64_t ignore_index,
                              const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int log_size = logits.numel();

    // allocate memory and cuda grid/block
    auto grad_logits = torch::softmax(logits, 1);
    if (grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_logits;
    }

    int blockx = 32;
    while (blockx < log_size && blockx < BLOCKSIZE) blockx *= 2;
    dim3 block(blockx);
    int gridx = std::max(std::min(log_size / BLOCKSIZE, (int)4096), 1);
    dim3 grid(gridx);

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lsr backwrd", [&] {
        LSRLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            grad_logits.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<int64_t>(), 
            ignore_index, smooth
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor LSR_forward(const at::Tensor &logits,
                         const at::Tensor &labels,
                         const int64_t ignore_index,
                         const float smooth) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this LSR loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return LSR_forward_cuda(logits, labels, ignore_index, smooth);
}

at::Tensor LSR_backward(const at::Tensor &logits,
                      const at::Tensor &labels,
                      const int64_t ignore_index,
                      const float smooth) {
    // TODO: try AT_ASSERTM
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this LSR loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return LSR_backward_cuda(logits, labels, ignore_index, smooth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lsr_forward", &LSR_forward, "lsr forward");
    m.def("lsr_backward", &LSR_backward, "lsr backward");
}

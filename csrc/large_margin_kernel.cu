
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include <iostream>

using std::cout;
using std::endl;

#define BLOCKSIZE 1024

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
    __syncthreads();
    sdata[tid] = -1000;
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

    // compute sum of exp with and without label index
    sdata[tid] = 0.;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        if (j == lb) continue;
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        sdata[tid] += expf(val - sdata[blockDim.x]);
    }
    reduce_sum<scalar_t>(sdata, tid);
    if (tid == 0) sdata[blockDim.x + 2] = sdata[0];

    sdata[tid] = 0.;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
        scalar_t val = logits[idx];
        sdata[tid] += expf(val - sdata[blockDim.x + 1]);
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
    __syncthreads();
    sdata[tid] = 0.;
    __syncthreads();
    for (int j{tid}; j < dimsize; j += blockDim.x) {
        if (j == lb) continue;
        int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
        scalar_t val = logits[idx];
        sdata[tid] += val * expf(val - sdata[blockDim.x]);
    }
    reduce_sum<scalar_t>(sdata, tid);
    if (tid == 0) {
        sdata[blockDim.x + 5] = sdata[0] / sdata[blockDim.x + 2]; 
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

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    int sample_offset = gridDim.x * blockDim.y;

    if (tid == 0) {
        sdata[blockDim.x + 4] = 1. / (dimsize - 1);
    }

    int samplesize = n_size * m_size;
    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        int64_t lb = labels[i];
        if (lb == ignore_index) {
            if (tid == 0) losses[i] = 0;
            continue;
        } 
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        compute_reduce_values<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);

        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            scalar_t term{0};
            if (j == lb) {
                term = -(dval - sdata[blockDim.x + 1]);
                term += logf(sdata[blockDim.x + 3]);
            } else {
                dval -= sdata[blockDim.x];
                term = expf(dval) / sdata[blockDim.x + 2];
                term -= sdata[blockDim.x + 4];
                term *= (dval - logf(sdata[blockDim.x + 2]));
                term *= lam / 2.;
            }
            sdata[tid] += term;
        }
        reduce_sum<scalar_t>(sdata, tid);
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
                grad_logits[idx] = 0;
            }
            continue;
        } 
        compute_reduce_values<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);
        compute_sum_of_qx<scalar_t>(logits, sdata,
                dimsize, m_size, n_idx, m_idx, lb, tid);

        for (int j{tid}; j < dimsize; j += blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t val = logits[idx];
            scalar_t pc = expf(val - sdata[blockDim.x + 1]) / sdata[blockDim.x + 3];
            scalar_t gval;
            if (j == lb) {
                gval = pc - 1.;
            } else {
                gval = val - sdata[blockDim.x + 5] + 1.;
                gval *= expf(val - sdata[blockDim.x]) / sdata[blockDim.x + 2];
                gval = pc + (gval - sdata[blockDim.x + 4]) * lam / 2.;
            }
            grad_logits[idx] = gval;
        }
    }
}


// cuda forward and backward
at::Tensor large_margin_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto losses = torch::empty_like(labels, logits.options());
    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }

    int blockx = 32;
    while (blockx < dimsize) blockx *= 2;
    blockx = std::max(std::min((int)BLOCKSIZE, blockx / 2), (int)32);
    int blocky = std::min(samplesize, (int)(BLOCKSIZE / blockx));
    int gridx = std::min(4096, (int)(samplesize / blocky));
    int n_shm = (blockx + 8) * blocky;
    dim3 block(blockx, blocky);
    dim3 grid(gridx);

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "large margin forward", [&] {
        int shm_size = n_shm * sizeof(scalar_t);
        LMarginLossForward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            losses.contiguous().data<scalar_t>(),
            ignore_index, lam 
        );
    });
    THCudaCheck(cudaGetLastError());
    return losses;
}


at::Tensor large_margin_backward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);
    if (grad_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_logits;
    }

    int blockx = 32;
    while (blockx < dimsize) blockx *= 2;
    blockx = std::max(std::min((int)BLOCKSIZE, blockx / 2), (int)32);
    int blocky = std::min(samplesize, (int)(BLOCKSIZE / blockx));
    int gridx = std::min(4096, (int)(samplesize / blocky));
    int n_shm = (blockx + 8) * blocky;
    dim3 block(blockx, blocky);
    dim3 grid(gridx);

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "large margin backwrd", [&] {
        int shm_size = n_shm * sizeof(scalar_t); 
        LMarginLossBackward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            grad_logits.contiguous().data<scalar_t>(),
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            ignore_index, lam 
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor large_margin_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float lam,
                             const int64_t ignore_index) {
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
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
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_backward_cuda(logits, labels, ignore_index, lam);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l_margin_forward", &large_margin_forward, "large margin forward");
    m.def("l_margin_backward", &large_margin_backward, "large margin backward");
}

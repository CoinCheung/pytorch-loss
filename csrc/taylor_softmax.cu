
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <thrust/reduce.h>

#include "common.hpp"


#define BLOCKSIZE 512


// kernel function to compute taylor series
template<typename scalar_t>
__forceinline__ __device__ scalar_t taylor_series(const scalar_t val, const int64_t n) {
    scalar_t res{1.}, mid{val}, denor{1.};
    res += val;
    for (int i{2}; i < n + 1; ++i) {
        denor *= static_cast<scalar_t>(i);
        mid *= val;
        res += mid / denor;
    }
    return res;
}


// reduce sum over shared memory, within blockx
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


// kernel function for forward and backward
template<typename scalar_t>
__global__ void SpatialTaylorSoftmaxForward(const int n_size,
                            const int dimsize,
                            const int m_size,
                            const scalar_t *feat,
                            scalar_t *activations, 
                            const int64_t n, const bool use_log) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int samplesize = n_size * m_size;
    for (int i{tid}; i < samplesize; i+=stride) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;

        const scalar_t one(1.);
        scalar_t sum{0.};
        for (int j{0}; j < dimsize; ++j) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            sum += taylor_series<scalar_t>(feat[ind], n);
        }
        if (use_log) {
            sum = math_ops::Log(sum);
        } else {
            sum = one / sum;
        }

        for (int j{0}; j < dimsize; ++j) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            scalar_t val = taylor_series<scalar_t>(feat[ind], n);
            if (use_log) {
                activations[ind] = math_ops::Log(val) - sum;
            } else {
                activations[ind] = val * sum;
            }
        }
    }
}


template<typename scalar_t>
__global__ void TaylorSoftmaxForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *feat, 
                            scalar_t *activations,  
                            const int64_t n, bool use_log) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata += blockDim.x * threadIdx.y;

    const int tid = threadIdx.x;
    const int bid = blockDim.y * blockIdx.x + threadIdx.y;
    const int t_stride = blockDim.x;
    const int b_stride = blockDim.y * gridDim.x;
    const int samplesize = n_size * m_size;

    const scalar_t zero(0.f);

    for (int k{bid}; k < samplesize; k += b_stride) {
        int n_ind = k / m_size;
        int m_ind = k % m_size;

        sdata[tid] = zero;
        __syncthreads();

        for (int j{tid}; j < dimsize; j += t_stride) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            sdata[tid] += taylor_series<scalar_t>(feat[ind], n);
        }
        reduce_sum<scalar_t>(sdata, tid);
        if (use_log && tid == 0) {
            sdata[0] = math_ops::Log(sdata[0]);
        }
        __syncthreads();
        for (int j{tid}; j < dimsize; j += t_stride) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            scalar_t val = taylor_series<scalar_t>(feat[ind], n);
            if (use_log) {
                activations[ind] = math_ops::Log(val) - sdata[0];
            } else {
                activations[ind] = val / sdata[0];
                /* if (isinf(activations[ind]) || isnan(activations[ind])) { */
                /*     printf("nan or inf"); */
                /* } */
            }
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__global__ void SpatialTaylorSoftmaxBackward(const int n_size,
                             const int dimsize, const int m_size,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat,
                             const int64_t n, const bool use_log) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const int samplesize = n_size * m_size;

    const scalar_t one(1.);
    for (int i{tid}; i < samplesize; i+=stride) {
        int n_ind = i / m_size;
        int m_ind = i % m_size;

        scalar_t sum{0.};
        for (int j{0}; j < dimsize; ++j) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            sum += taylor_series<scalar_t>(feat[ind], n);
        }
        sum = one / sum;

        scalar_t mid{0.};
        if (use_log) {
            for (int j{0}; j < dimsize; ++j) {
                int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                mid += grad[ind];
            }
            mid *= sum;
        } else {
            for (int j{0}; j < dimsize; ++j) {
                int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                mid += taylor_series<scalar_t>(feat[ind], n) * grad[ind];
            }
            mid *= (sum * sum);
        }

        for (int j{0}; j < dimsize; ++j) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            if (use_log) {
                scalar_t val1, val2;
                val1 = feat[ind];
                val2 = taylor_series<scalar_t>(val1, n - 1);
                val1 = taylor_series<scalar_t>(val1, n);
                grad_feat[ind] = - mid * val2 + val2 / val1 * grad[ind];
            } else {
                scalar_t val = feat[ind];
                val = taylor_series<scalar_t>(val, n - 1);
                grad_feat[ind] = - val * mid + val * sum * grad[ind];
            }
        }
    }
}


template<typename scalar_t>
__global__ void TaylorSoftmaxBackward(const int n_size,
                             const int dimsize, const int m_size,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat,
                             const int64_t n, const bool use_log) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata += blockDim.x * threadIdx.y;

    const int tid = threadIdx.x;
    const int bid = blockDim.y * blockIdx.x + threadIdx.y;
    const int t_stride = blockDim.x;
    const int b_stride = blockDim.y * gridDim.x;
    const int samplesize = n_size * m_size;

    const scalar_t one(1.f);
    const scalar_t zero(0.f);

    for (int k{bid}; k < samplesize; k += b_stride) {
        int n_ind = k / m_size;
        int m_ind = k % m_size;

        sdata[tid] = zero;
        __syncthreads();

        scalar_t sum{0.};
        for (int j{tid}; j < dimsize; j += t_stride) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            sdata[tid] += taylor_series<scalar_t>(feat[ind], n);
        }
        reduce_sum<scalar_t>(sdata, tid);
        sum = one / sdata[0];
        __syncthreads();

        sdata[tid] = zero;
        __syncthreads();

        scalar_t mid{0};
        if (use_log) {
            mid *= sum;
            for (int j{tid}; j < dimsize; j += t_stride) {
                int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                sdata[tid] += grad[ind];
            }
            reduce_sum<scalar_t>(sdata, tid);
            mid = sdata[0] * sum;
        } else {
            for (int j{tid}; j < dimsize; j += t_stride) {
                int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
                sdata[tid] += taylor_series<scalar_t>(feat[ind], n) * grad[ind];
            }
            reduce_sum<scalar_t>(sdata, tid);
            mid = sdata[0] * (sum * sum);
        }

        for (int j{tid}; j < dimsize; j += t_stride) {
            int ind = n_ind * dimsize * m_size + j * m_size + m_ind;
            if (use_log) {
                scalar_t val1, val2;
                val1 = feat[ind];
                val2 = taylor_series<scalar_t>(val1, n - 1);
                val1 = taylor_series<scalar_t>(val1, n);
                grad_feat[ind] = - mid * val2 + val2 / val1 * grad[ind];
            } else {
                scalar_t val = feat[ind];
                val = taylor_series<scalar_t>(val, n - 1);
                grad_feat[ind] = - val * mid + val * sum * grad[ind];
            }
        }
        __syncthreads();
    }
}


// cuda forward and backward
at::Tensor TaylorSoftmax_forward_cuda(const at::Tensor &feat,
                const int64_t dim, const int64_t n, const bool use_log) {
    // CHECK type and shape
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto activations = at::empty_like(feat);

    int n_size = 1;
    int dimsize = feat.size(dim);
    int m_size = 1;
    int n_dims = feat.dim();
    int dim_ = static_cast<int>(dim);
    for (int i{0}; i < dim_; ++i) {n_size *= feat.size(i);}
    for (int i{dim_ + 1}; i < n_dims; ++i) {m_size *= feat.size(i);}
    const int samplesize = n_size * m_size;

    if (dimsize < 32 && samplesize > 4096) { 
        // one thread process one sample
        int gridx = std::max(std::min(4096, samplesize / BLOCKSIZE), 1);
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.scalar_type(), "spatial taylor softmax forward", [&] {
            SpatialTaylorSoftmaxForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                feat.contiguous().data_ptr<scalar_t>(),
                activations.contiguous().data_ptr<scalar_t>(), 
                n, use_log
            );
        });

    } else {
        // one blockx process one sample
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(BLOCKSIZE, blockx / 2), 32);
        int blocky = std::max(std::min(samplesize, BLOCKSIZE / blockx), 1);
        int gridx = std::max(std::min(4096, samplesize / blocky), 1);
        int n_shm = blockx * blocky;
        dim3 block(blockx, blocky);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.scalar_type(), "softmax forward", [&] {
            TaylorSoftmaxForward<scalar_t><<<grid, block, n_shm * sizeof(scalar_t), at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                feat.contiguous().data_ptr<scalar_t>(),
                activations.contiguous().data_ptr<scalar_t>(),
                n, use_log
            );
        });
    }

    return activations;
}


at::Tensor TaylorSoftmax_backward_cuda(const at::Tensor &grad,
            const at::Tensor &feat,
            const int64_t dim, 
            const int64_t n, const bool use_log) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto grad_feat = at::empty_like(feat);
    int n_size = 1;
    int dimsize = feat.size(dim);
    int m_size = 1;
    int n_dims = feat.dim();
    int dim_ = static_cast<int>(dim);
    for (int i{0}; i < dim_; ++i) {n_size *= feat.size(i);}
    for (int i{dim_ + 1}; i < n_dims; ++i) {m_size *= feat.size(i);}
    const int samplesize = n_size * m_size;

    /* if (dimsize < 32 && samplesize > 4096) { */
    if (1) {
        // one thread process one sample
        int gridx = std::max(std::min(4096, samplesize / BLOCKSIZE), 1);
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.scalar_type(), "spatial taylor softmax backward", [&] {
            SpatialTaylorSoftmaxBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                feat.contiguous().data_ptr<scalar_t>(),
                grad.contiguous().data_ptr<scalar_t>(),
                grad_feat.contiguous().data_ptr<scalar_t>(), 
                n, use_log
            );
        });
    } else {
        // one blockx process one sample
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(BLOCKSIZE, blockx / 2), 32);
        int blocky = std::max(std::min(samplesize, BLOCKSIZE / blockx), 1);
        int gridx = std::max(std::min(4096, samplesize / blocky), 1);
        int n_shm = blockx * blocky;
        dim3 block(blockx, blocky);
        dim3 grid(gridx);
        /* int blockx = 32; */
        /* while (blockx < dimsize) blockx *= 2; */
        /* blockx = std::max(std::min(BLOCKSIZE, blockx / 2), 32); */
        /* int blocky = std::max(std::min(samplesize, BLOCKSIZE / blockx), 1); */
        /* gridx = std::max(std::min(4096, samplesize / blocky), 1); */
        /* int n_shm = blockx * blocky; */
        /* block = dim3(blockx, blocky); */
        /* grid = dim3(gridx); */
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feat.scalar_type(), "softmax forward", [&] {
            TaylorSoftmaxBackward<scalar_t><<<grid, block, n_shm * sizeof(scalar_t), at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size,
                feat.contiguous().data_ptr<scalar_t>(),
                grad.contiguous().data_ptr<scalar_t>(),
                grad_feat.contiguous().data_ptr<scalar_t>(), 
                n, use_log
            );
        });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return grad_feat;
}

// python inferface
at::Tensor TaylorSoftmax_forward(const at::Tensor &feat,
            const int64_t dim, 
            const int64_t n,
            const bool use_log) {
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this taylor softmax function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return TaylorSoftmax_forward_cuda(feat, dim, n, use_log);
}

at::Tensor TaylorSoftmax_backward(const at::Tensor &grad,
            const at::Tensor &feat, 
            const int64_t dim, 
            const int64_t n,
            const bool use_log) {
    // TODO: try AT_ASSERTM
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this taylor softmax function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return TaylorSoftmax_backward_cuda(grad, feat, dim, n, use_log);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("taylor_softmax_forward", &TaylorSoftmax_forward, "taylor softmax forward");
    m.def("taylor_softmax_backward", &TaylorSoftmax_backward, "taylor softmax backward");
}

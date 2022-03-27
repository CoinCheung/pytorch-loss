
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "common.hpp"

using math_ops::Pow;
using math_ops::Sqrt;
using math_ops::Rsqrt;


#define BLOCKSIZE 512


namespace layer_norm_space {

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

}


// kernel function for forward and backward
template<typename scalar_t>
__global__ void LayerNormForward(const int N,
            const int C, const int M,
            const scalar_t *x, scalar_t *res,
            const double eps) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + threadIdx.y * blockDim.x;

    int tid = threadIdx.x;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int bstrd = gridDim.x * blockDim.y;
    int n_samples = N * M;
    const scalar_t zero(0.);
    const scalar_t two(2.);
    const scalar_t scale = scalar_t(1./C);

    for (int i{bid}; i < n_samples; i += bstrd) {
        int n = i / M;
        int m = i % M;

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < C; j += blockDim.x) {
            sdata[tid] += x[n * C * M + j * M + m];
        }
        __syncthreads();
        layer_norm_space::reduce_op<layer_norm_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                layer_norm_space::sum_op<scalar_t>());
        scalar_t sum_x = sdata[0]; 
        __syncthreads();

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < C; j += blockDim.x) {
            sdata[tid] += Pow(x[n * C * M + j * M + m], two);
        }
        __syncthreads();
        layer_norm_space::reduce_op<layer_norm_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                layer_norm_space::sum_op<scalar_t>());
        scalar_t sum_x2 = sdata[0];
        __syncthreads();

        // mean = sum(x) / C
        sum_x = sum_x * scale;
        // var = 1/c * sum(x**2) - mean(x) ** 2)
        sum_x2 = sum_x2 * scale - sum_x * sum_x;
        // 1/std = rsqrt(var + eps)
        sum_x2 = Rsqrt(sum_x2 + scalar_t(eps));
        // res = (x - mean) / std
        for (int j{tid}; j < C; j += blockDim.x) {
            res[n * C * M + j * M + m] = (x[n * C * M + j * M + m] - sum_x) * sum_x2;
        }
    }
}

template<typename scalar_t>
__global__ void SpatialLayerNormForward(const int N,
            const int C, const int M,
            scalar_t *x, scalar_t *res,
            const double eps) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int strd = gridDim.x * blockDim.x;
    int n_samples = N * M;
    const scalar_t scale = scalar_t(1./C);

    for (int i{tid}; i < n_samples; i += strd) {
        int n = i / M;
        int m = i % M;

        scalar_t sum_x{0}, sum_x2{0};
        for (int j{0}; j < C; ++j) {
            scalar_t val = x[n * C * M + j * M + m];
            sum_x += val;
            // sum_x2 += Pow(val, two);
            sum_x2 += val * val;
        }

        // mean = sum(x) / C
        sum_x = sum_x * scale;
        // var = 1/c * (sum(x**2) - 1/c * sum(x) ** 2)
        sum_x2 = sum_x2 * scale - sum_x * sum_x;
        // 1/std = rsqrt(var + eps)
        sum_x2 = Rsqrt(sum_x2 + scalar_t(eps));
        // res = (x - mean) / std
        for (int j{0}; j < C; ++j) {
            int ind = n * C * M + j * M + m;
            res[ind] = (x[ind] - sum_x) * sum_x2;
        }
    }
}


template<typename scalar_t>
__global__ void LayerNormBackward(const int N,
            const int C, const int M,
            const scalar_t *x, const scalar_t *grad, 
            scalar_t *res, const double eps) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + threadIdx.y * blockDim.x;

    int tid = threadIdx.x;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int bstrd = gridDim.x * blockDim.y;
    int n_samples = N * M;
    const scalar_t zero(0.);
    const scalar_t scale = scalar_t(1./C);
    // const float scale = 1./C;

    for (int i{bid}; i < n_samples; i += bstrd) {
        int n = i / M;
        int m = i % M;

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < C; j += blockDim.x) {
            sdata[tid] += x[n * C * M + j * M + m];
        }
        __syncthreads();
        layer_norm_space::reduce_op<layer_norm_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                layer_norm_space::sum_op<scalar_t>());
        // float sum_x = static_cast<float>(sdata[0]);
        scalar_t sum_x = sdata[0]; 
        __syncthreads();

        sdata[tid] = zero;
        __syncthreads();
        for (int j{tid}; j < C; j += blockDim.x) {
            sdata[tid] += x[n * C * M + j * M + m] * x[n * C * M + j * M + m];
        }
        __syncthreads();
        layer_norm_space::reduce_op<layer_norm_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                layer_norm_space::sum_op<scalar_t>());
        // float sum_x2 = static_cast<float>(sdata[0]);
        scalar_t sum_x2 = sdata[0]; 
        __syncthreads();

        // mean = sum(x) / C
        sum_x = sum_x * scale;
        // var + eps = 1/c * sum(x**2) - mean(x) ** 2) + eps
        sum_x2 = sum_x2 * scale - sum_x * sum_x + scalar_t(eps);
        // sum_x2 = sum_x2 * scale - sum_x * sum_x + static_cast<float>(eps);

        for (int j{tid}; j < C; j += blockDim.x) {
            scalar_t val = x[n * C * M + j * M + m];
            // - rsqrt(var + eps) * (1/C + (xi - mean) * (xi - 1/C) * mean(x) * (var + eps))
            res[n * C * M + j * M + m] = -Rsqrt(sum_x2) * (scale + (val - sum_x) * (val - scale) * sum_x * sum_x2) * grad[n * C * M + j * M + m];

            // float val = static_cast<float>(x[n * C * M + j * M + m]);
            // float g = -Rsqrt(sum_x2) * (scale + (val - sum_x) * (val - scale) * sum_x * sum_x2);
            // res[n * C * M + j * M + m] = scalar_t(g * grad[n * C * M + j * M + m]);
        }
    }
}

template<typename scalar_t>
__global__ void SpatialLayerNormBackward(const int N,
            const int C, const int M,
            const scalar_t *x, const scalar_t *grad, 
            scalar_t *res, const double eps) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int strd = gridDim.x * blockDim.x;
    int n_samples = N * M;
    const scalar_t one(1.);
    const scalar_t scale = scalar_t(1./C);

    for (int i{tid}; i < n_samples; i += strd) {
        int n = i / M;
        int m = i % M;

        scalar_t sum_x{0}, sum_x2{0};
        for (int j{0}; j < C; ++j) {
            scalar_t val = x[n * C * M + j * M + m];
            sum_x += val;
            sum_x2 += val * val;
        }

        // mean
        sum_x = sum_x * scale;
        // var + eps = 1/c * sum(x**2) - mean(x) ** 2) + eps
        sum_x2 = sum_x2 * scale - sum_x * sum_x + scalar_t(eps);
        // var + eps = 1/c * (sum(x**2) - 1/c * sum(x) ** 2) + eps
        // sum_x2 = (sum_x2 - scalar_t(1./C) * Pow(sum_x, two)) / scalar_t(C) + scalar_t(eps);

        for (int j{0}; j < C; ++j) {
            int ind = n * C * M + j * M + m;
            scalar_t val = x[ind];
            // rsqrt(var + eps) * (-1/C) * (1 + (xi - mean) * (xi - 1/C) * sum(x) * (var + eps))
            // res[ind] = -Rsqrt(sum_x2) * scalar_t(1./C) * (one + (val - sum_x / scalar_t(C)) * (val - scalar_t(1./C) * sum_x * sum_x2)) * grad[ind];

            // - rsqrt(var + eps) * (1/C + (xi - mean) * (xi - 1/C) * mean(x) * (var + eps))
            res[ind] = -Rsqrt(sum_x2) * (scale + (val - sum_x) * (val - scale) * sum_x * sum_x2) * grad[ind];
        }
    }
}


// cuda forward and backward
at::Tensor LayerNorm_forward_cuda(const at::Tensor &x,
                                  const double eps) {
    // CHECK type and shape
    AT_ASSERTM(x.device().type() == c10::kCUDA, "x should be cuda");

    const int N = x.size(0);
    const int C = x.size(1);
    const int M = x.size(2);
    const int sample_size = N * M;

    auto res = torch::empty_like(x);

    // parallel method 
    if (C > 32) {
        int blockx1 = 32;
        while (blockx1 < C && blockx1 <= BLOCKSIZE) blockx1 <<= 1;
        blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
        int blocky1 = std::max(1, BLOCKSIZE / blockx1);
        int gridx1 = std::max(1, std::min(4096, sample_size / blocky1));
        dim3 block1(blockx1, blocky1);
        dim3 grid1(gridx1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "layernorm forward", [&] {
            int shm_size = blockx1 * blocky1 * sizeof(scalar_t);
            LayerNormForward<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                N, C, M,
                x.contiguous().data_ptr<scalar_t>(),
                res.contiguous().data_ptr<scalar_t>(),
                eps
            );
        });
    } else {
        int blockx1 = 32;
        while (blockx1 < sample_size && blockx1 <= BLOCKSIZE) blockx1 <<= 1;
        blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
        int gridx1 = sample_size / blockx1;
        dim3 block1(blockx1);
        dim3 grid1(gridx1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "layernorm forward", [&] {
            SpatialLayerNormForward<scalar_t><<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
                N, C, M,
                x.contiguous().data_ptr<scalar_t>(),
                res.contiguous().data_ptr<scalar_t>(),
                eps
            );
        });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return res;
}


at::Tensor LayerNorm_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &x,
                                  const float eps) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(x.device().type() == c10::kCUDA, "x should be cuda");

    const int N = x.size(0);
    const int C = x.size(1);
    const int M = x.size(2);
    const int sample_size = N * M;

    auto res = torch::empty_like(x);

    // parallel method 
    if (C > 32) {
        int blockx1 = 32;
        while (blockx1 < C && blockx1 <= BLOCKSIZE) blockx1 <<= 1;
        blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
        int blocky1 = std::max(1, BLOCKSIZE / blockx1);
        int gridx1 = std::max(1, std::min(4096, sample_size / blocky1));
        dim3 block1(blockx1, blocky1);
        dim3 grid1(gridx1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "layernorm forward", [&] {
            int shm_size = blockx1 * blocky1 * sizeof(scalar_t);
            LayerNormBackward<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                N, C, M,
                x.contiguous().data_ptr<scalar_t>(),
                grad.contiguous().data_ptr<scalar_t>(),
                res.contiguous().data_ptr<scalar_t>(),
                eps
            );
        });
    } else {
        int blockx1 = 32;
        while (blockx1 < sample_size && blockx1 <= BLOCKSIZE) blockx1 <<= 1;
        blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
        int gridx1 = sample_size / blockx1;
        dim3 block1(blockx1);
        dim3 grid1(gridx1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "layernorm forward", [&] {
            SpatialLayerNormBackward<scalar_t><<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
                N, C, M,
                x.contiguous().data_ptr<scalar_t>(),
                grad.contiguous().data_ptr<scalar_t>(),
                res.contiguous().data_ptr<scalar_t>(),
                eps
            );
        });
    }

    AT_CUDA_CHECK(cudaGetLastError());
    return res;
}

// python inferface
at::Tensor LayerNorm_forward(const at::Tensor &x,
                             const double eps) {
    if (x.device().type() != c10::kCUDA) {
        AT_ERROR("this layernorm loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(x.device());
    return LayerNorm_forward_cuda(x, eps);
}

at::Tensor LayerNorm_backward(const at::Tensor &grad,
                                  const at::Tensor &x,
                                  const double eps) {
    if ((x.device().type() != c10::kCUDA)) {
        AT_ERROR("this layernorm loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(x.device());
    return LayerNorm_backward_cuda(grad, x, eps);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("layer_norm_forward", &LayerNorm_forward, "layernorm forward");
    m.def("layer_norm_backward", &LayerNorm_backward, "layernorm backward");
}

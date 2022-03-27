
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "common.hpp"

using math_ops::Exp;
using math_ops::Pow;


#define BLOCKSIZE 512


namespace soft_dice_space {

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
__global__ void compute_numer_denor(const int batchsize, 
                            const int sample_size,
                            const int n_blockxs_sample,
                            const scalar_t *logits,
                            const scalar_t *labels,
                            scalar_t *numer,
                            scalar_t *denor,
                            const float p) {
    /* Tips about shared memory:
     * 1. torch will instantiate the template with three types: double, float, half;
     * 2. these three types should not share same definitions of shared memory;
     * 3. so one method is to use static shared memory with memory size explicitly assigned, and another method is to allocate shared memory with same raw type, such as unsigned char here, and then cast the pointer according to different template types */

    // method1: use static sized shared memory
    // __shared__ scalar_t sdata[BLOCKSIZE * 2];
    // method2: allocate with raw uchar type and then cast in different kernel
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + threadIdx.y * blockDim.x;

    int tid = threadIdx.x;
    int tstrd = blockDim.x * n_blockxs_sample;

    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int bstrd = gridDim.x * blockDim.y;
    int n_sample_blocks = n_blockxs_sample * batchsize;

    const scalar_t one(1.);
    for (int i{bid}; i < n_sample_blocks; i += bstrd) {
        int sample_start = (i / n_blockxs_sample) * sample_size;
        int local_tid = (i % n_blockxs_sample) * blockDim.x + tid;

        scalar_t v_numer{0}, v_denor{0};
        for (int j{local_tid}; j < sample_size; j += tstrd) {
            scalar_t prob = one / (one + Exp(-logits[j + sample_start]));
            scalar_t lb = labels[j + sample_start];
            v_numer += prob * lb * scalar_t(2.);
            v_denor += Pow(prob, scalar_t(p)) + lb;
        }
        __syncthreads();
        sdata[tid] = v_numer;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata, 
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        if (tid == 0) {
            numer[i] = sdata[0];
        }
        __syncthreads();
        sdata[tid] = v_denor;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata, 
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        if (tid == 0) {
            denor[i] = sdata[0];
        }
    }
}


template<typename scalar_t>
__global__ void SoftDiceForward(const int batchsize, const int n_blockxs_sample,
                            const scalar_t *numer,
                            const scalar_t *denor,
                            scalar_t *losses,
                            const float smooth) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + threadIdx.y * blockDim.x;

    int tid = threadIdx.x;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int bstrd = gridDim.x * blockDim.y;

    const scalar_t one(1.);
    for (int i{bid}; i < batchsize; i += bstrd) {
        scalar_t v_numer{0}, v_denor{0};
        int t_start = i * n_blockxs_sample;
        for (int j{tid}; j < n_blockxs_sample; j += blockDim.x) {
            v_numer += numer[j + t_start];
            v_denor += denor[j + t_start];
        }

        // reduce numer
        __syncthreads();
        sdata[tid] = v_numer;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata, 
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        v_numer = sdata[0];

        // reduce denorm
        __syncthreads();
        sdata[tid] = v_denor;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata, 
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        v_denor = sdata[0];
        if (tid == 0) {
            losses[bid] = one - (v_numer + smooth) / (v_denor + smooth);
        }
    } 
}



template<typename scalar_t>
__global__ void reduce_numer_denor(const int batchsize, const int n_blockxs_sample,
                            scalar_t *numer,
                            scalar_t *denor,
                            const float smooth) {

    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    sdata = sdata + threadIdx.y * blockDim.x;

    int tid = threadIdx.x;

    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int bstrd = gridDim.x * blockDim.y;

    for (int i{bid}; i < batchsize; i += bstrd) {
        scalar_t v_numer{0}, v_denor{0};
        int t_start = i * n_blockxs_sample;
        for (int j{tid}; j < n_blockxs_sample; j += blockDim.x) {
            v_numer += numer[j + t_start];
            v_denor += denor[j + t_start];
        }

        // reduce numer
        __syncthreads();
        sdata[tid] = v_numer;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        if (tid == 0) {
            numer[t_start] = sdata[0] + smooth;
        }

        // reduce denorm
        __syncthreads();
        sdata[tid] = v_denor;
        __syncthreads();
        soft_dice_space::reduce_op<soft_dice_space::sum_op, scalar_t>(
                sdata,
                blockDim.x,
                soft_dice_space::sum_op<scalar_t>());
        if (tid == 0) {
            denor[t_start] = sdata[0] + smooth;
        }
    }
}


template<typename scalar_t>
__global__ void SoftDiceBackward(const int batchsize, const int sample_size, 
                            const int n_blockxs_sample,
                             const scalar_t *logits,
                             const scalar_t *labels,
                             const scalar_t *grad,
                             const scalar_t *numer,
                             const scalar_t *denor,
                             scalar_t *grad_logits,
                             const float p) {
    int tid = threadIdx.x;
    int tstrd = blockDim.x * n_blockxs_sample;
    int bid = blockIdx.x * blockDim.y + threadIdx.y;
    int bstrd = blockDim.y * gridDim.x;

    const scalar_t one(1.);
    const scalar_t two(2.);
    const scalar_t v_p(p);
    int n_sample_blocks = n_blockxs_sample * batchsize;
    for (int i{bid}; i < n_sample_blocks; i += bstrd) {
        int sample_idx = i / n_blockxs_sample;
        int sample_start = sample_idx * sample_size;
        int local_tid = (i % n_blockxs_sample) * blockDim.x + tid;

        scalar_t v_numer = numer[sample_idx * n_blockxs_sample];
        scalar_t v_denor = denor[sample_idx * n_blockxs_sample];
        scalar_t grad_val = grad[sample_idx];

        for (int j{local_tid}; j < sample_size; j += tstrd) {
            scalar_t prob = one / (one + Exp(-logits[j + sample_start]));
            scalar_t lb = labels[j + sample_start];

            scalar_t term1 = v_p * Pow(prob, scalar_t(p)) * (one - prob) * v_numer / Pow(v_denor, two);
            scalar_t term2 = two * lb * prob * (one - prob) / v_denor;

            grad_logits[j + sample_start] = grad_val * (term1 - term2);
        }
    }
}


// cuda forward and backward
at::Tensor SoftDice_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float p,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int batchsize = logits.size(0);
    const int num_samples = logits.numel();
    const int sample_size = num_samples / batchsize;

    // parallel method for numer/denor
    int blockx1 = 32;
    while (blockx1 < sample_size) blockx1 *= 2;
    blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
    int n_blockxs_sample = std::max(1, sample_size / blockx1);
    int blocky1 = std::max(1, BLOCKSIZE / blockx1);
    if (blocky1 > batchsize) blocky1 = batchsize;
    int gridx1 = batchsize * n_blockxs_sample / blocky1;
    gridx1 = std::max(1, std::min(4096, gridx1));
    dim3 block1(blockx1, blocky1);
    dim3 grid1(gridx1);

    // parallel method for loss
    int blockx2 = 32;
    while (blockx2 < n_blockxs_sample) blockx2 *= 2;
    blockx2 = std::max(32, std::min(BLOCKSIZE, blockx2 / 2));
    int blocky2 = std::max(1, BLOCKSIZE / blockx2);
    int gridx2 = std::min(batchsize / blocky2, 4096);
    gridx2 = std::max(1, gridx2);
    dim3 block2(blockx2, blocky2);
    dim3 grid2(gridx2);

    // allocate memory and cuda grid/block
    // Note: should use torch::zeros rather than at::zeros, torch::zeros is variable
    // and at::zeros is tensor
    auto losses = torch::empty({batchsize}, logits.options());
    auto numer = torch::zeros(
            {batchsize * n_blockxs_sample},
            logits.options());
    auto denor = torch::zeros(
            {batchsize * n_blockxs_sample}, 
            logits.options());
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }
    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "soft dice forward", [&] {
        int shm_size = blockx1 * blocky1 * sizeof(scalar_t);
        compute_numer_denor<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            batchsize, sample_size, n_blockxs_sample,
            logits.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<scalar_t>(),
            numer.contiguous().data_ptr<scalar_t>(),
            denor.contiguous().data_ptr<scalar_t>(),
            p
        );

        shm_size = blockx2 * blocky2 * sizeof(scalar_t);
        SoftDiceForward<scalar_t><<<grid2, block2, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            batchsize, n_blockxs_sample,
            numer.contiguous().data_ptr<scalar_t>(),
            denor.contiguous().data_ptr<scalar_t>(),
            losses.contiguous().data_ptr<scalar_t>(),
            smooth
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return losses;
}


at::Tensor SoftDice_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float p,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int batchsize = logits.size(0);
    const int num_samples = logits.numel();
    const int sample_size = num_samples / batchsize;

    // parallel settings for numer/denor
    int blockx1 = 32;
    while (blockx1 < sample_size) blockx1 *= 2;
    blockx1 = std::max(32, std::min(BLOCKSIZE, blockx1 / 2));
    int n_blockxs_sample = sample_size / blockx1;
    int blocky1 = std::max(1, BLOCKSIZE / blockx1);
    if (blocky1 > batchsize) blocky1 = batchsize;
    int gridx1 = batchsize * n_blockxs_sample / blocky1;
    gridx1 = std::max(1, std::min(4096, gridx1));
    dim3 block1(blockx1, blocky1);
    dim3 grid1(gridx1);

    // parallel settings for reduce numer/denor
    int blockx2 = 32;
    while (blockx2 < n_blockxs_sample) blockx2 *= 2;
    blockx2 = std::max(32, std::min(BLOCKSIZE, blockx2 / 2));
    int blocky2 = std::max(1, BLOCKSIZE / blockx2);
    int gridx2 = std::min(batchsize / blocky2, 4096);
    gridx2 = std::max(1, gridx2);
    dim3 block2(blockx2, blocky2);
    dim3 grid2(gridx2);

    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);
    auto numer = torch::zeros(
            {batchsize * n_blockxs_sample},
            logits.options());
    auto denor = torch::zeros(
            {batchsize * n_blockxs_sample}, 
            logits.options());
    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "soft dice backwrd", [&] {
        int shm_size = blockx1 * blocky1 * sizeof(scalar_t);
        compute_numer_denor<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            batchsize, sample_size, n_blockxs_sample,
            logits.contiguous().data_ptr<scalar_t>(),
            labels.contiguous().data_ptr<scalar_t>(),
            numer.contiguous().data_ptr<scalar_t>(),
            denor.contiguous().data_ptr<scalar_t>(),
            p
        );
        shm_size = blockx2 * blocky2 * sizeof(scalar_t);
        reduce_numer_denor<scalar_t><<<grid2, block2, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            batchsize, n_blockxs_sample,
            numer.contiguous().data_ptr<scalar_t>(),
            denor.contiguous().data_ptr<scalar_t>(),
            smooth
        );
        
        SoftDiceBackward<scalar_t><<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
            batchsize, sample_size, n_blockxs_sample,
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<scalar_t>(),
            grad.contiguous().data_ptr<scalar_t>(),
            numer.contiguous().data_ptr<scalar_t>(),
            denor.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>(),
            p
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor SoftDice_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float p,
                             const float smooth) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this dice loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return SoftDice_forward_cuda(logits, labels, p, smooth);
}

at::Tensor SoftDice_backward(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float p,
                                  const float smooth) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this dice loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return SoftDice_backward_cuda(grad, logits, labels, p, smooth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_dice_forward", &SoftDice_forward, "soft-dice forward");
    m.def("soft_dice_backward", &SoftDice_backward, "soft-dice backward");
}


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


template<typename scalar_t>
__global__ void compute_numer_denor(const int nthreads,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *numer,
                            scalar_t *denor,
                            const float p, const float smooth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    int batch_size = gridDim.y;
    int sample_idx = blockIdx.y;
    int sample_size = nthreads / batch_size;
/* 
 *     Tips about shared memory:
 *     1. torch will instantiate the template with three types: double, float, half;
 *     2. these three types should not share same definitions of shared memory;
 *     3. so one method is to use static shared memory with memory size explicitly assigned, and another method is to allocate shared memory with same raw type, such as unsigned char here, and then cast the pointer according to different template types
 *  */
    // method1: use static sized shared memory
    // __shared__ scalar_t sdata[BLOCKSIZE * 2];
    // method2: allocate with raw uchar type and then cast in different kernel
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    sdata[threadIdx.x] = 0; // numer
    sdata[threadIdx.x + blockDim.x] = 0; // denor
    __syncthreads();

    for (int i{tid}; i < sample_size; i+=stride) {
        int idx = sample_idx * sample_size + i;
        scalar_t prob = 1. / (1. + expf(-logits[idx]));
        scalar_t lb = (scalar_t)labels[idx];

        sdata[threadIdx.x] += 2 * prob * lb;
        sdata[threadIdx.x + blockDim.x] += powf(prob, p) + lb;
    }
    __syncthreads();

    for (int s=1; s < blockDim.x; s*=2) {
        int idx = 2 * s * threadIdx.x;
        if (idx < blockDim.x && idx + s < blockDim.x) {
            sdata[idx] += sdata[idx + s];
        }
        idx += blockDim.x;
        if (idx < (blockDim.x + blockDim.x) && idx + s < (blockDim.x + blockDim.x)) {
            sdata[idx] += sdata[idx + s];
        }
        __syncthreads();
    }

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        sdata[0] += smooth;
        sdata[blockDim.x] += smooth;
    }
    if (threadIdx.x == 0) {
        atomicAdd(&numer[sample_idx], sdata[0]);
        atomicAdd(&denor[sample_idx], sdata[blockDim.x]);
    }
}

// kernel function for forward and backward
template<typename scalar_t>
__global__ void SoftDiceForward(const int nthreads,
                            scalar_t *loss,
                            const scalar_t *numer,
                            const scalar_t *denor) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (int i{tid}; i < nthreads; i+=stride) {
        loss[i] = 1. - numer[i] / denor[i];
    }
}


template<typename scalar_t>
__global__ void SoftDiceBackward(const int nthreads,
                             const scalar_t *logits,
                             const int64_t *labels,
                             const scalar_t *grad,
                             const scalar_t *numer,
                             const scalar_t *denor,
                             scalar_t *grad_logits,
                             const float p, const float smooth) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int sample_idx = blockIdx.y;
    int stride = gridDim.x * blockDim.x;
    int batch_size = gridDim.y;
    int sample_size = nthreads / batch_size;

    scalar_t numer_val = numer[sample_idx];
    scalar_t denor_val = denor[sample_idx];
    scalar_t grad_val = grad[sample_idx];
    for (int i{tid}; i < sample_size; i+=stride) {
        int idx = sample_idx * sample_size + i;
        scalar_t prob = 1. / (1. + expf(-logits[idx]));
        scalar_t lb = (scalar_t)labels[idx];
        scalar_t m = numer_val - 2. * (prob * lb);
        scalar_t n = denor_val - powf(prob, p);
        scalar_t g = -powf(prob, p - 1.) * p * m;
        if (lb == 1) {
            g += powf(prob, p) * 2. * (1. - p) + (n * 2.);
        }
        g = - (g / powf(powf(prob, p) + n, 2.)) * prob * (1. - prob);
        grad_logits[idx] = grad_val * g;
    }
}


// cuda forward and backward
at::Tensor SoftDice_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float p,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int batchsize = logits.size(0);
    const int num_samples = logits.numel();
    const int sample_size = num_samples / batchsize;
    // allocate memory and cuda grid/block
    auto numer = torch::zeros({batchsize}, logits.options());
    auto denor = torch::zeros({batchsize}, logits.options());
    auto losses = torch::empty({batchsize}, logits.options());
    // Note: should use torch::zeros rather than at::zeros, torch::zeros is variable
    // and at::zeros is tensor

    dim3 grid1(std::min(
        THCCeilDiv((int64_t)sample_size, (int64_t)BLOCKSIZE), (int64_t)4096
    ), batchsize);
    dim3 block1(BLOCKSIZE);
    dim3 grid2(1);
    dim3 block2(BLOCKSIZE);
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "soft dice forward", [&] {
        int shm_size = BLOCKSIZE * sizeof(scalar_t) * 2;
        compute_numer_denor<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            numer.contiguous().data<scalar_t>(),
            denor.contiguous().data<scalar_t>(),
            p, smooth
        );
        SoftDiceForward<scalar_t><<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            batchsize,
            losses.contiguous().data<scalar_t>(),
            numer.contiguous().data<scalar_t>(),
            denor.contiguous().data<scalar_t>()
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
    AT_ASSERTM(grad.type().is_cuda(), "grad should be cuda");
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int batchsize = logits.size(0);
    const int num_samples = logits.numel();
    const int sample_size = num_samples / batchsize;
    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);
    auto numer = torch::zeros({batchsize}, logits.options());
    auto denor = torch::zeros({batchsize}, logits.options());

    dim3 grid(std::min(
        THCCeilDiv((int64_t)sample_size, (int64_t)BLOCKSIZE), (int64_t)4096
    ), batchsize);
    dim3 block(BLOCKSIZE);
    if (grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "soft dice backwrd", [&] {
        int shm_size = BLOCKSIZE * sizeof(scalar_t) * 2;
        compute_numer_denor<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            numer.contiguous().data<scalar_t>(),
            denor.contiguous().data<scalar_t>(),
            p, smooth
        );
        SoftDiceBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(),
            grad.contiguous().data<scalar_t>(),
            numer.contiguous().data<scalar_t>(),
            denor.contiguous().data<scalar_t>(),
            grad_logits.contiguous().data<scalar_t>(),
            p, smooth
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
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
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
    // TODO: try AT_ASSERTM
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this dice loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return SoftDice_backward_cuda(grad, logits, labels, p, smooth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("soft_dice_forward", &SoftDice_forward, "soft-dice forward");
    m.def("soft_dice_backward", &SoftDice_backward, "soft-dice backward");
}

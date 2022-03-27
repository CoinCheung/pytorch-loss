
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "common.hpp"


#define EXP_THRESH 20.

// kernel function for forward and backward
template<typename scalar_t>
__global__ void MishForward(const int nthreads,
                            const scalar_t *feat,
                            scalar_t *activations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t val = feat[i];
        if (val > scalar_t(EXP_THRESH)) {
            activations[i] = val * tanh(val);
        } else {
            activations[i] = val * tanh(log1p(exp(val)));
        }
    }
}

template<typename scalar_t>
__global__ void MishBackward(const int nthreads,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t two(2.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t val = feat[i];
        scalar_t xtanh;
        if (val > scalar_t(EXP_THRESH)) {
            xtanh = tanh(val);
        } else {
            xtanh = tanh(log1p(exp(val)));
        }
        grad_feat[i] = grad[i] * (xtanh + val * (one - powf(xtanh, two)) * one / (one + exp(-val)));
    }
}


// cuda forward and backward
at::Tensor Mish_forward_cuda(const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto activations = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (activations.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return activations;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activations.scalar_type(), "mish forward", [&] {
        MishForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data_ptr<scalar_t>(), 
            activations.contiguous().data_ptr<scalar_t>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return activations;
}


at::Tensor Mish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto grad_feat = at::empty_like(feat);
    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (grad_feat.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_feat;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feat.scalar_type(), "mish backwrd", [&] {
        MishBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data_ptr<scalar_t>(), 
            grad.contiguous().data_ptr<scalar_t>(),
            grad_feat.contiguous().data_ptr<scalar_t>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_feat;
}

// python inferface
at::Tensor Mish_forward(const at::Tensor &feat) {
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this mish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Mish_forward_cuda(feat);
}

at::Tensor Mish_backward(const at::Tensor &grad, const at::Tensor &feat) {
    // TODO: try AT_ASSERTM
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this mish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Mish_backward_cuda(grad, feat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mish_forward", &Mish_forward, "mish forward");
    m.def("mish_backward", &Mish_backward, "mish backward");
}

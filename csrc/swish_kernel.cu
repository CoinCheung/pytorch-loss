
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include "common.hpp"

#define BLOCKSIZE 512


// NOTE: If use constant number such as 1. or 2., must use scalar_t(1.) or scalar_t(2.), or the values will be casted into double type.

// kernel function for forward and backward
template<typename scalar_t>
__global__ void SwishForward(const int nthreads,
                            const scalar_t *feat,
                            scalar_t *activations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        const scalar_t one(1.);
        scalar_t val = feat[i];
        activations[i] = val / (one + expf(-val));
    }
}

template<typename scalar_t>
__global__ void SwishBackward(const int nthreads,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        const scalar_t one(1.);
        scalar_t val = feat[i];

        grad_feat[i] = (one + val / (one + expf(val))) / (one + expf(-val));
        grad_feat[i] *= grad[i];

    }
}


namespace swish_space {
template<typename scalar_t>
__forceinline__ __device__ scalar_t ReLU6(scalar_t val) {
    const scalar_t zero(0.);
    const scalar_t six(6.);
    scalar_t res = val; 
    if (res < zero) res = zero;
    if (res > six) res = six;
    return res;
}
}



template<typename scalar_t>
__global__ void HSwishForward(const int nthreads,
                            const scalar_t *feat,
                            scalar_t *activations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        const scalar_t three(3.);
        const scalar_t one_six(1. / 6.);
        scalar_t val = feat[i];
        activations[i] = val * swish_space::ReLU6(val + three) * one_six;
    }
}

template<typename scalar_t>
__global__ void HSwishBackward(const int nthreads,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        const scalar_t zero(0.);
        const scalar_t _three(-3.);
        const scalar_t three(3.);
        const scalar_t one_six(1. / 6.);
        scalar_t val = feat[i];
        grad_feat[i] = (swish_space::ReLU6(val + three) * one_six + ((val > _three && val < three) ? one_six : zero) * val) * grad[i];
    }
}


// cuda forward and backward
at::Tensor Swish_forward_cuda(const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto activations = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv(num_samples, 2 * BLOCKSIZE), 4096
    ));
    dim3 block(BLOCKSIZE);
    if (activations.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return activations;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activations.scalar_type(), "swish forward", [&] {
        SwishForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data_ptr<scalar_t>(), 
            activations.contiguous().data_ptr<scalar_t>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return activations;
}


at::Tensor Swish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto grad_feat = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        // THCCeilDiv(num_samples, BLOCKSIZE), 4096
        THCCeilDiv(num_samples, 2 * BLOCKSIZE), 4096
    ));
    dim3 block(BLOCKSIZE);
    if (grad_feat.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_feat;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feat.scalar_type(), "swish backwrd", [&] {
        SwishBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data_ptr<scalar_t>(), 
            grad.contiguous().data_ptr<scalar_t>(),
            grad_feat.contiguous().data_ptr<scalar_t>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_feat;
}


at::Tensor HSwish_forward_cuda(const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto activations = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv(num_samples, 2 * BLOCKSIZE), 4096
    ));
    dim3 block(BLOCKSIZE);
    if (activations.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return activations;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activations.scalar_type(), "hswish forward", [&] {
        HSwishForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data_ptr<scalar_t>(), 
            activations.contiguous().data_ptr<scalar_t>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return activations;
}


at::Tensor HSwish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(grad.device().type() == c10::kCUDA, "grad should be cuda");
    AT_ASSERTM(feat.device().type() == c10::kCUDA, "feat should be cuda");

    // allocate memory and cuda grid/block
    auto grad_feat = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv(num_samples, 2 * BLOCKSIZE), 4096
    ));
    dim3 block(BLOCKSIZE);
    if (grad_feat.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_feat;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feat.scalar_type(), "hswish backwrd", [&] {
        HSwishBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
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
at::Tensor Swish_forward(const at::Tensor &feat) {
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Swish_forward_cuda(feat);
}

at::Tensor Swish_backward(const at::Tensor &grad, const at::Tensor &feat) {
    // TODO: try AT_ASSERTM
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Swish_backward_cuda(grad, feat);
}

at::Tensor HSwish_forward(const at::Tensor &feat) {
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return HSwish_forward_cuda(feat);
}

at::Tensor HSwish_backward(const at::Tensor &grad, const at::Tensor &feat) {
    // TODO: try AT_ASSERTM
    if (feat.device().type() != c10::kCUDA) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return HSwish_backward_cuda(grad, feat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_forward", &Swish_forward, "swish forward");
    m.def("swish_backward", &Swish_backward, "swish backward");
    m.def("hswish_forward", &HSwish_forward, "hswish forward");
    m.def("hswish_backward", &HSwish_backward, "hswish backward");
}

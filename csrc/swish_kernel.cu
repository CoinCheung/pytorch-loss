
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>


// kernel function for forward and backward
template<typename scalar_t>
__global__ void SwishForward(const int nthreads,
                            const scalar_t *feat,
                            scalar_t *activations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t sigmoid = 1. / (1. + expf(-feat[i]));
        activations[i] = feat[i] * sigmoid;
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
        scalar_t sigmoid = 1. / (1. + expf(-feat[i]));
        grad_feat[i] = sigmoid * (1. + feat[i] * (1. - sigmoid));
        grad_feat[i] *= grad[i];
    }
}


// cuda forward and backward
at::Tensor Swish_forward_cuda(const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(feat.type().is_cuda(), "feat should be cuda");

    // allocate memory and cuda grid/block
    auto activations = at::empty_like(feat);

    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (activations.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return activations;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activations.scalar_type(), "swish forward", [&] {
        SwishForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data<scalar_t>(), 
            activations.contiguous().data<scalar_t>()
        );
    });
    THCudaCheck(cudaGetLastError());
    return activations;
}


at::Tensor Swish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat) {
    // CHECK type and shape
    AT_ASSERTM(grad.type().is_cuda(), "grad should be cuda");
    AT_ASSERTM(feat.type().is_cuda(), "feat should be cuda");

    // allocate memory and cuda grid/block
    auto grad_feat = at::empty_like(feat);
    const int num_samples = feat.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (grad_feat.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_feat;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feat.scalar_type(), "swish backwrd", [&] {
        SwishBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data<scalar_t>(), 
            grad.contiguous().data<scalar_t>(),
            grad_feat.contiguous().data<scalar_t>()
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_feat;
}

// python inferface
at::Tensor Swish_forward(const at::Tensor &feat) {
    if (!feat.type().is_cuda()) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Swish_forward_cuda(feat);
}

at::Tensor Swish_backward(const at::Tensor &grad, const at::Tensor &feat) {
    // TODO: try AT_ASSERTM
    if (!feat.type().is_cuda()) {
        AT_ERROR("this swish function only supports gpu mode\n");
    } 
    at::DeviceGuard guard(feat.device());
    return Swish_backward_cuda(grad, feat);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("swish_forward", &Swish_forward, "swish forward");
    m.def("swish_backward", &Swish_backward, "swish backward");
}


#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>


template<typename scalar_t>
__global__ void MishForward(const int nthreads,
                            const scalar_t *feat,
                            scalar_t *activations) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t s2 = powf(1 + expf(feat[i]), 2);
        activations[i] = feat[i] * (s2 - 1.) / (s2 + 1.);
    }
}

template<typename scalar_t>
__global__ void MishBackward(const int nthreads,
                             const scalar_t *feat,
                             const scalar_t *grad,
                             scalar_t *grad_feat) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t s2 = powf(1 + expf(feat[i]), 2);
        scalar_t tanh = (s2 - 1.) / (s2 + 1.);
        scalar_t sigmoid = 1. / (1. + expf(-feat[i]));
        grad_feat[i] = tanh + feat[i] * (1. - powf(tanh, 2)) * sigmoid;
        grad_feat[i] *= grad[i];
    }
}


at::Tensor Mish_forward_cuda(const at::Tensor &feat) {
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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(activations.scalar_type(), "mish forward", [&] {
        MishForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data<scalar_t>(), 
            activations.contiguous().data<scalar_t>()
        );
    });
    THCudaCheck(cudaGetLastError());
    return activations;
}


at::Tensor Mish_backward_cuda(const at::Tensor &grad, const at::Tensor &feat) {
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
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_feat.scalar_type(), "mish backwrd", [&] {
        MishBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            feat.contiguous().data<scalar_t>(), 
            grad.contiguous().data<scalar_t>(),
            grad_feat.contiguous().data<scalar_t>()
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_feat;
}

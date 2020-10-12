
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include <torch/extension.h>



template<typename scalar_t>
__global__ void FocalLossForward(const int nthreads,
                                 const scalar_t *logits,
                                 const int64_t *labels,
                                 scalar_t *loss,
                                 const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t lgt = logits[i];
        scalar_t prob = one / (one + expf(-lgt));
        scalar_t log_p, log_1_p;
        if (lgt >= 0) {
            log_p = -logf(one + expf(-lgt));
            log_1_p = -lgt + log_p;
        } else {
            log_1_p = -logf(one + expf(lgt));
            log_p = lgt + log_1_p;
        }
        scalar_t term1 = powf(one - prob, gamma) * log_p;
        scalar_t term2 = powf(prob, gamma) * log_1_p;
        loss[i] = -alpha * term1 * labels[i] - (one - alpha) * term2 * (one - labels[i]);
    }
}

template<typename scalar_t>
__global__ void FocalLossBackward(const int nthreads,
                                  const scalar_t *logits,
                                  const int64_t *labels,
                                  const scalar_t *grad_loss,
                                  scalar_t *grad_logits,
                                  const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t lgt = logits[i];
        scalar_t prob = one / (one + expf(-lgt));
        scalar_t log_p, log_1_p;
        if (lgt >=0) {
            log_p = -logf(one + expf(-lgt));
            log_1_p = -lgt + log_p;
        } else {
            log_1_p = -logf(one + expf(lgt));
            log_p = lgt + log_1_p;
        }
        scalar_t term1 = powf(one - prob, gamma) * (one - prob - gamma * prob * log_p);
        scalar_t term2 = powf(prob, gamma) * (gamma * (one - prob) * log_1_p - prob);
        grad_logits[i] = -alpha * term1 * labels[i] - (one - alpha) * term2 * (one - labels[i]);
        grad_logits[i] = grad_logits[i] * grad_loss[i];
    }
}


at::Tensor FocalLoss_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    // allocate memory and cuda grid/block
    auto losses = at::empty_like(logits);

    const int num_samples = logits.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal forward", [&] {
        FocalLossForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(),
            losses.contiguous().data<scalar_t>(),
            scalar_t(gamma), scalar_t(alpha)
        );
    });
    THCudaCheck(cudaGetLastError());
    return losses;
}


at::Tensor FocalLoss_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(grad.type().is_cuda(), "grad should be cuda");
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");
    // AT_ASSERTM(logits.size() == labels.size(), "should have same shape");

    /* allocate memory and cuda grid/block */
    auto grad_logits = at::empty_like(logits);
    const int num_samples = logits.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (grad_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal backwrd", [&] {
        FocalLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(),
            grad.contiguous().data<scalar_t>(),
            grad_logits.contiguous().data<scalar_t>(),
            scalar_t(gamma), scalar_t(alpha)
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor FocalLoss_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    if (!logits.type().is_cuda() || !labels.type().is_cuda()) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_forward_cuda(logits, labels, gamma, alpha);
}

at::Tensor FocalLoss_backward(const at::Tensor &grad,
                             const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    // TODO: try AT_ASSERTM
    if (!logits.type().is_cuda()) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_backward_cuda(grad, logits, labels, gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("focalloss_forward", &FocalLoss_forward, "focal loss forward");
    m.def("focalloss_backward", &FocalLoss_backward, "focal loss backward");
}

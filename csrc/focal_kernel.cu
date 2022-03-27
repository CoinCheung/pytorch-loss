
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
using math_ops::Log;
using math_ops::Log1p;
using math_ops::Pow;
using math_ops::Abs;




template<typename scalar_t>
__global__ void FocalLossForward(const int nthreads,
                                 const scalar_t *logits,
                                 const scalar_t *labels,
                                 scalar_t *loss,
                                 const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t zero(0.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t lgt = logits[i];
        scalar_t lb = labels[i];
        scalar_t prob = one / (one + Exp(-lgt));
        scalar_t log_p, log_1_p;
        if (lgt >= zero) {
            // log_p = -Log(one + Exp(-lgt));
            log_p = -Log1p(Exp(-lgt));
            log_1_p = -lgt + log_p;
        } else {
            // log_1_p = -Log(one + Exp(lgt));
            log_1_p = -Log1p(Exp(lgt));
            log_p = lgt + log_1_p;
        }
        scalar_t coeff = - Pow(Abs(lb - prob), gamma);
        scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * log_1_p;
        loss[i] = coeff * ce;
    }
}


template<typename scalar_t>
__global__ void FocalLossBackward(const int nthreads,
                                  const scalar_t *logits,
                                  const scalar_t *labels,
                                  const scalar_t *grad_loss,
                                  scalar_t *grad_logits,
                                  const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t zero(0.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t lgt = logits[i];
        scalar_t lb = labels[i];
        scalar_t prob = one / (one + Exp(-lgt));
        scalar_t log_p, log_1_p;
        if (lgt >= zero) {
            // log_p = -Log(one + Exp(-lgt));
            log_p = -Log1p(Exp(-lgt));
            log_1_p = -lgt + log_p;
        } else {
            /* log_1_p = -Log(one + Exp(lgt)); */
            log_1_p = -Log1p(Exp(lgt));
            log_p = lgt + log_1_p;
        }
        scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * log_1_p;
        scalar_t coeff = - Pow(Abs(lb - prob), gamma);
        scalar_t d_ce = lb * alpha - prob * (one - lb - alpha + scalar_t(2) * lb * alpha);
        scalar_t d_coeff = gamma * Pow(Abs(lb - prob), gamma - one) * prob * (one - prob);
        if (lb < prob) {
            d_coeff = - d_coeff;
        }

        scalar_t grad = d_coeff * ce + d_ce * coeff;

        grad_logits[i] = grad * grad_loss[i];
    }
}


at::Tensor FocalLoss_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");
    AT_ASSERTM(labels.scalar_type() == logits.scalar_type(), "labels and logits should be half/float/double");

    // allocate memory and cuda grid/block
    auto losses = at::empty_like(logits);

    const int num_samples = logits.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal forward", [&] {
        FocalLossForward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<scalar_t>(),
            losses.contiguous().data_ptr<scalar_t>(),
            scalar_t(gamma), scalar_t(alpha)
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return losses;
}


at::Tensor FocalLoss_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    /* allocate memory and cuda grid/block */
    auto grad_logits = at::empty_like(logits);
    const int num_samples = logits.numel();
    dim3 grid(std::min(
        THCCeilDiv((int64_t)num_samples, (int64_t)512), (int64_t)4096
    ));
    dim3 block(512);
    if (grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal backwrd", [&] {
        FocalLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<scalar_t>(),
            grad.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>(),
            scalar_t(gamma), scalar_t(alpha)
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor FocalLoss_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
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
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_backward_cuda(grad, logits, labels, gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("focalloss_forward", &FocalLoss_forward, "focal loss forward");
    m.def("focalloss_backward", &FocalLoss_backward, "focal loss backward");
}


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


#define BLOCKSIZE 256
#define GRIDSIZE_MAX 4096
#define VEC_TYPE float4



template<typename scalar_t>
__global__ void FocalLossForward(const int64_t nthreads,
                                 const scalar_t *logits,
                                 const scalar_t *labels,
                                 scalar_t *losses,
                                 const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t zero(0.);
    for (int i{tid}; i < nthreads; i+=stride) {
        scalar_t lgt = logits[i];
        scalar_t lb = labels[i];

        scalar_t prob = one / (one + Exp(-lgt));
        // log(p) = - log[ 1 + exp(-x) ] = x - log[ exp(x) + 1 ]
        // log(1 - p) = log(p) - x
        scalar_t log_p = (lgt >= zero) ? -Log1p(Exp(-lgt)) : lgt - Log1p(Exp(lgt));

        scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * (log_p - lgt);
        scalar_t coeff = - Pow(Abs(lb - prob), gamma);
        losses[i] = coeff * ce;
    }
}


template<typename scalar_t>
__global__ void FocalLossBackward(const int64_t nthreads,
                                  const scalar_t *logits,
                                  const scalar_t *labels,
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
        // log(p) = - log[ 1 + exp(-x) ] = x - log[ exp(x) + 1 ]
        // log(1 - p) = log(p) - x
        scalar_t log_p = (lgt >= zero) ? -Log1p(Exp(-lgt)) : lgt - Log1p(Exp(lgt));

        scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * (log_p - lgt);
        scalar_t coeff = - Pow(Abs(lb - prob), gamma);
        scalar_t d_ce = lb * alpha - prob * (one - lb - alpha + scalar_t(2) * lb * alpha);
        scalar_t d_coeff = gamma * Pow(Abs(lb - prob), gamma - one) * prob * (one - prob);
        if (lb < prob) {
            d_coeff = - d_coeff;
        }

        grad_logits[i] = d_coeff * ce + d_ce * coeff;
    }
}


template<typename scalar_t>
__global__ void FocalLossForwardBackward(const int64_t nthreads,
                                  const scalar_t *logits,
                                  const scalar_t *labels,
                                  scalar_t *losses,
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
        // log(p) = - log[ 1 + exp(-x) ] = x - log[ exp(x) + 1 ]
        // log(1 - p) = log(p) - x
        scalar_t log_p = (lgt >= zero) ? -Log1p(Exp(-lgt)) : lgt - Log1p(Exp(lgt));

        scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * (log_p - lgt);
        scalar_t coeff = - Pow(Abs(lb - prob), gamma);
        scalar_t d_ce = lb * alpha - prob * (one - lb - alpha + scalar_t(2) * lb * alpha);
        scalar_t d_coeff = gamma * Pow(Abs(lb - prob), gamma - one) * prob * (one - prob);
        if (lb < prob) {
            d_coeff = - d_coeff;
        }

        losses[i] = coeff * ce;
        grad_logits[i] = d_coeff * ce + d_ce * coeff;
    }
}


template<typename scalar_t>
__global__ void FocalLossForwardBackwardVec(const int64_t nthreads,
                                  const VEC_TYPE *logits,
                                  const VEC_TYPE *labels,
                                  VEC_TYPE *losses,
                                  VEC_TYPE *grad_logits,
                                  const scalar_t gamma, const scalar_t alpha) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    const scalar_t one(1.);
    const scalar_t zero(0.);

    constexpr int vec_size = sizeof(VEC_TYPE) / sizeof(scalar_t);

    for (int i{tid}; i < nthreads / vec_size; i+=stride) {

        VEC_TYPE vec1 = logits[i];
        VEC_TYPE vec2 = labels[i];

        for (int j{0}; j < vec_size; ++j) {
            scalar_t lgt = reinterpret_cast<scalar_t*>(&vec1)[j];
            scalar_t lb  = reinterpret_cast<scalar_t*>(&vec2)[j];

            scalar_t prob = one / (one + Exp(-lgt));
            // log(p) = - log[ 1 + exp(-x) ] = x - log[ exp(x) + 1 ]
            // log(1 - p) = log(p) - x
            scalar_t log_p = (lgt >= zero) ? -Log1p(Exp(-lgt)) : lgt - Log1p(Exp(lgt));

            scalar_t ce = lb * alpha * log_p + (one - lb) * (one - alpha) * (log_p - lgt);
            scalar_t coeff = - Pow(Abs(lb - prob), gamma);
            scalar_t d_ce = lb * alpha - prob * (one - lb - alpha + scalar_t(2) * lb * alpha);
            scalar_t d_coeff = gamma * Pow(Abs(lb - prob), gamma - one) * prob * (one - prob);
            if (lb < prob) {
                d_coeff = - d_coeff;
            }

            reinterpret_cast<scalar_t*>(&vec1)[j] = coeff * ce;
            reinterpret_cast<scalar_t*>(&vec2)[j] = d_coeff * ce + d_ce * coeff;
        }

        losses[i] = vec1;
        grad_logits[i] = vec2;
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
    if (losses.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return losses;
    }

    const int64_t num_samples = logits.numel();
    int64_t n_blocks = std::min(
        THCCeilDiv(static_cast<int64_t>(num_samples), 
                   static_cast<int64_t>(BLOCKSIZE)), 
        static_cast<int64_t>(GRIDSIZE_MAX)
    );
    dim3 block(BLOCKSIZE);
    dim3 grid(n_blocks);

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


at::Tensor FocalLoss_backward_cuda(
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    /* allocate memory and cuda grid/block */
    auto grad_logits = at::empty_like(logits);
    if (grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return grad_logits;
    }

    const int64_t num_samples = logits.numel();
    int64_t n_blocks = std::min(
        THCCeilDiv(static_cast<int64_t>(num_samples), 
                   static_cast<int64_t>(BLOCKSIZE)), 
        static_cast<int64_t>(GRIDSIZE_MAX)
    );
    dim3 block(BLOCKSIZE);
    dim3 grid(n_blocks);

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal backwrd", [&] {
        FocalLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            num_samples, 
            logits.contiguous().data_ptr<scalar_t>(), 
            labels.contiguous().data_ptr<scalar_t>(),
            grad_logits.contiguous().data_ptr<scalar_t>(),
            scalar_t(gamma), scalar_t(alpha)
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_logits;
}


std::tuple<at::Tensor, at::Tensor> FocalLoss_forward_backward_cuda(
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float gamma,
                                  const float alpha) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    /* allocate memory and cuda grid/block */
    auto losses = at::empty_like(logits);
    auto grad_logits = at::empty_like(logits);
    if (losses.numel() == 0 or grad_logits.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return std::make_tuple(losses, grad_logits);
    }

    dim3 block(BLOCKSIZE);
    int64_t num_samples = logits.numel();
    int64_t n_blocks = THCCeilDiv(num_samples, static_cast<int64_t>(BLOCKSIZE)); 

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "focal forward backwrd", [&] {

        constexpr int vec_size = sizeof(VEC_TYPE) / sizeof(scalar_t);

        if (num_samples % 8 == 0 and n_blocks >= 384 * vec_size) {
            n_blocks = n_blocks / vec_size;
            dim3 grid(std::min(n_blocks, static_cast<int64_t>(GRIDSIZE_MAX)));
            FocalLossForwardBackwardVec<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_samples, 
                reinterpret_cast<VEC_TYPE*>(logits.contiguous().data_ptr<scalar_t>()),
                reinterpret_cast<VEC_TYPE*>(labels.contiguous().data_ptr<scalar_t>()), 
                reinterpret_cast<VEC_TYPE*>(losses.contiguous().data_ptr<scalar_t>()), 
                reinterpret_cast<VEC_TYPE*>(grad_logits.contiguous().data_ptr<scalar_t>()), 
                scalar_t(gamma), scalar_t(alpha)
            );
        } else {
            dim3 grid(std::min(n_blocks, static_cast<int64_t>(GRIDSIZE_MAX)));
            FocalLossForwardBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
                num_samples, 
                logits.contiguous().data_ptr<scalar_t>(), 
                labels.contiguous().data_ptr<scalar_t>(),
                losses.contiguous().data_ptr<scalar_t>(),
                grad_logits.contiguous().data_ptr<scalar_t>(),
                scalar_t(gamma), scalar_t(alpha)
            );
        }
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(losses, grad_logits);
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

at::Tensor FocalLoss_backward(const at::Tensor &logits,
                              const at::Tensor &labels,
                              const float gamma,
                              const float alpha) {
    // TODO: try AT_ASSERTM
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this focal loss only support gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return FocalLoss_backward_cuda(logits, labels, gamma, alpha);
}

std::tuple<at::Tensor, at::Tensor> FocalLoss_forward_backward(
                             const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float gamma,
                             const float alpha) {
    // TODO: try AT_ASSERTM
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this focal loss only support gpu mode\n");
    }
    at::DeviceGuard guard(logits.device());
    return FocalLoss_forward_backward_cuda(logits, labels, gamma, alpha);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("focalloss_forward", &FocalLoss_forward, "focal loss forward");
    m.def("focalloss_backward", &FocalLoss_backward, "focal loss backward");
    m.def("focalloss_forward_backward", &FocalLoss_forward_backward, "focal loss forward backward");
}


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

#include <iostream>

using std::cout;
using std::endl;

#define BLOCKSIZE 512

// TODO: whether need to care about length of gridDim.y


// kernel function for forward and backward
template<typename scalar_t>
__global__ void LSRLossForward(const int dimsize, const int dimstride,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index,
                            const float smooth) {
    int n_idx = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tstride = gridDim.x * blockDim.x;
    int64_t lgts_offset = n_idx * dimstride * dimsize;
    int64_t lb_offset = n_idx * dimstride;
    scalar_t lb_pos = 1. - smooth;
    scalar_t lb_neg = smooth / dimsize;
    scalar_t sumy = lb_neg * (dimsize - 1) + lb_pos;

    for (int i{tid}; i < dimstride; i+=tstride) {
        scalar_t lgts_max{-10000.};
        int idx;
        for (int j{0}; j < dimsize; ++j) {
            idx = lgts_offset + i + j * dimstride;
            if (lgts_max < logits[idx]) {
                lgts_max = logits[idx];
            }
        }
        scalar_t exp_sumval{0};
        scalar_t sumval{0};
        for (int j{0}; j < dimsize; ++j) {
            idx = lgts_offset + i + j * dimstride;
            scalar_t val = logits[idx] - lgts_max;
            exp_sumval += expf(val);
            sumval += val;
        }
        scalar_t lossval{0};
        int64_t lb{labels[lb_offset + i]};
        if (lb != ignore_index) {
            idx = lgts_offset + i + lb * dimstride;
            lossval = sumval * lb_neg + (logits[idx] - lgts_max) * (lb_pos - lb_neg);
            lossval = -lossval + logf(exp_sumval) * sumy;
        }
        losses[lb_offset + i] = lossval;
    }
}


template<typename scalar_t>
__global__ void LSRLossBackward(const int dimsize, const int dimstride,
                            const scalar_t *grad,
                            scalar_t *grad_logits,
                            const scalar_t *logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float smooth) {
    int n_idx = blockIdx.y;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tstride = gridDim.x * blockDim.x;
    int64_t lgts_offset = n_idx * dimstride * dimsize;
    int64_t lb_offset = n_idx * dimstride;
    scalar_t lb_pos = 1. - smooth;
    scalar_t lb_neg = smooth / dimsize;
    scalar_t sumy = lb_neg * (dimsize - 1) + lb_pos;

    // if (tid == 0 && n_idx == 0)
    // printf("in backward kernel\n");

    for (int i{tid}; i < dimstride; i+=tstride) {
        scalar_t lgts_max{-10000.};
        int idx;
        for (int j{0}; j < dimsize; ++j) {
            idx = lgts_offset + i + j * dimstride;
            if (lgts_max < logits[idx]) {
                lgts_max = logits[idx];
            }
        }
        scalar_t exp_sumval{0};
        for (int j{0}; j < dimsize; ++j) {
            idx = lgts_offset + i + j * dimstride;
            exp_sumval += expf(logits[idx] - lgts_max);
        }
        scalar_t gradval{0};
        int64_t lb{labels[lb_offset + i]};
        for (int j{0}; j < dimsize; ++j) {
            idx = lgts_offset + i + j * dimstride;
            if (lb == ignore_index) {
                gradval = 0;
            } else {
                gradval = sumy * (expf(logits[idx] - lgts_max) / exp_sumval);
                gradval -= lb_neg;
                if (j == lb) {
                    gradval += lb_neg - lb_pos;
                }
            }
            grad_logits[idx] = gradval * grad[i + lb_offset];
        }
    }
    // if (tid == 0 && n_idx == 0)
    // printf("before quit backward kernel\n");

}


// cuda forward and backward
at::Tensor LSR_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int dimstride = logits.numel() / (n_size * dimsize);

    // allocate memory and cuda grid/block
    auto losses = torch::zeros_like(labels, logits.options());
    // Note: should use torch::zeros rather than at::zeros, torch::zeros is variable
    // and at::zeros is tensor

    // TODO: use softmax-2d or softmax-nd for better
    dim3 grid1(std::min(
        THCCeilDiv((int64_t)dimstride, (int64_t)BLOCKSIZE), (int64_t)4096
    ), n_size);
    dim3 block1(BLOCKSIZE);
    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }

    // cout << "call forward kernel\n";
    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "lsr forward", [&] {
        LSRLossForward<scalar_t><<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
            dimsize, dimstride, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            losses.contiguous().data<scalar_t>(),
            ignore_index, smooth
        );
    });
    THCudaCheck(cudaGetLastError());
    return losses;
}


at::Tensor LSR_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float smooth) {
    // CHECK type and shape
    AT_ASSERTM(grad.type().is_cuda(), "grad should be cuda");
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int dimstride = logits.numel() / (n_size * dimsize);

    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);

    dim3 grid(std::min(
        THCCeilDiv((int64_t)dimstride, (int64_t)BLOCKSIZE), (int64_t)4096
    ), n_size);
    dim3 block(BLOCKSIZE);
    if (grad_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lsr backwrd", [&] {
        LSRLossBackward<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            dimsize, dimstride, 
            grad.contiguous().data<scalar_t>(), 
            grad_logits.contiguous().data<scalar_t>(),
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            ignore_index, smooth
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor LSR_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const int64_t ignore_index,
                             const float smooth) {
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this LSR loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return LSR_forward_cuda(logits, labels, ignore_index, smooth);
}

at::Tensor LSR_backward(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float smooth) {
    // TODO: try AT_ASSERTM
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this LSR loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return LSR_backward_cuda(grad, logits, labels, ignore_index, smooth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lsr_forward", &LSR_forward, "lsr forward");
    m.def("lsr_backward", &LSR_backward, "lsr backward");
}


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


// kernel function for forward and backward
template<typename scalar_t>
__global__ void LSRLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *log_scores,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float smooth) {
    // shared memory
    __shared__ scalar_t sdata[BLOCKSIZE + 2];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    scalar_t lb_pos = 1. - smooth;
    scalar_t lb_neg = smooth / dimsize;

    int samplesize = n_size * m_size;
    for (int i{bid}; i < samplesize; i+=gridDim.x) {
        sdata[tid] = 0;
        __syncthreads();
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int64_t lb = labels[i];
        if (lb == ignore_index) {
            continue;
        } 
        // compute each element and add to shared memory
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval;
            if (j == lb) {
                dval = -log_scores[idx] * lb_pos;
                sdata[tid] += dval;
            } else {
                dval = -log_scores[idx] * lb_neg;
                sdata[tid] += dval;
            }
        }
        __syncthreads();
        // sum up 
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        losses[i] = sdata[0];
    }
}


template<typename scalar_t>
__global__ void LSRLossBackward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *grad,
                            scalar_t *grad_logits,
                            const scalar_t *scores,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float smooth) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    scalar_t lb_pos = 1. - smooth;
    scalar_t lb_neg = smooth / dimsize;
    scalar_t sumy = lb_neg * (dimsize - 1) + lb_pos;

    int samplesize = n_size * m_size;
    for (int i{bid}; i < samplesize; i+=gridDim.x) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int64_t lb{labels[i]};
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t gradval = 0; 
            if (lb != ignore_index) {
                gradval = sumy * scores[idx];
                if (j == lb) {
                    gradval -= lb_pos;
                } else {
                    gradval -= lb_neg;
                }
            }
            grad_logits[idx] = gradval * grad[i];
        }
    }
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
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto losses = torch::zeros_like(labels, logits.options());
    auto log_scores = torch::log_softmax(logits, 1);

    dim3 grid1(std::min(samplesize, (int)4096));
    dim3 block1(std::min(dimsize, (int)BLOCKSIZE));
    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }

    // cout << "call forward kernel\n";
    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "lsr forward", [&] {
        int shm_size = BLOCKSIZE * sizeof(scalar_t) * 2; 
        LSRLossForward<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            log_scores.contiguous().data<scalar_t>(), 
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
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    // allocate memory and cuda grid/block
    auto grad_logits = torch::empty_like(logits);
    auto scores = torch::softmax(logits, 1);

    dim3 grid(std::min(samplesize, (int)4096));
    dim3 block(std::min(dimsize, (int)BLOCKSIZE));
    if (grad_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lsr backwrd", [&] {
        int shm_size = BLOCKSIZE * sizeof(scalar_t) * 2; 
        LSRLossBackward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            grad.contiguous().data<scalar_t>(), 
            grad_logits.contiguous().data<scalar_t>(),
            scores.contiguous().data<scalar_t>(), 
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

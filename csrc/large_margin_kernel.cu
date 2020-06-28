
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
__global__ void LMarginLossForward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *logits,
                            const int64_t *labels,
                            scalar_t *losses,
                            const int64_t ignore_index, const float lam) {
    // shared memory
    // b is max logits without target 
    // b+1 is max logits with target 
    // b+2 is sum of exp without target 
    // b+3 is sum of exp with target 
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    scalar_t coeff = 1. / (dimsize - 1);

    int samplesize = n_size * m_size;
    for (int i{bid}; i < samplesize; i+=gridDim.x) {
        int64_t lb = labels[i];
        if (lb == ignore_index) {
            if (tid == 0) losses[i] = 0;
            continue;
        } 
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        // compute max value for each vector for softmax
        sdata[tid] = -1000;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            if (dval > sdata[tid]) sdata[tid] = dval;
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                if (sdata[idx] < sdata[idx + s]) sdata[idx] = sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x] = sdata[0]; // max logits without label
            sdata[blockDim.x + 1] = sdata[0]; // max logits with label
            scalar_t dval = logits[lb];
            if (dval > sdata[0]) sdata[blockDim.x + 1] = dval;
        }

        // compute exp sum for softmax
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            sdata[tid] += expf(dval - sdata[blockDim.x]);
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x + 2] = sdata[0]; // exp sum without label
        }
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            sdata[tid] += expf(dval - sdata[blockDim.x + 1]);
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x + 3] = sdata[0]; // exp sum with label
        }

        // if (i == 0 && tid == 0) {
        //     for (int i {0}; i < dimsize; ++i) {
        //         printf("%f, ", sdata[i]);
        //     }
        //     // printf("%f, %f, %f, %f\n",
        //     //         sdata[blockDim.x + 0],
        //     //         sdata[blockDim.x + 1],
        //     //         sdata[blockDim.x + 2],
        //     //         sdata[blockDim.x + 3]
        //     //         );
        // }

        // compute extra term
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            scalar_t term{0};
            if (j == lb) {
                term = -(dval - sdata[blockDim.x + 1]);
                term += logf(sdata[blockDim.x + 3]);
            } else {
                // if (i == 0) {
                //     printf("dval: %f, ", dval);
                // }
                dval -= sdata[blockDim.x];
                term = expf(dval) / sdata[blockDim.x + 2] - coeff;
                // if (i == 0) {
                //     printf("term: %f, coeff: %f, ", term, coeff);
                // }
                term *= (dval - logf(sdata[blockDim.x + 2]));
                term *= lam / 2.;
            }
            sdata[tid] += term;
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) losses[i] = sdata[0];
    }
}


template<typename scalar_t>
__global__ void LMarginLossBackward(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *grad,
                            scalar_t *grad_logits,
                            const scalar_t *logits,
                            const int64_t *labels,
                            const int64_t ignore_index,
                            const float lam) {
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    scalar_t coeff = 1. / (dimsize - 1);

    int samplesize = n_size * m_size;
    for (int i{bid}; i < samplesize; i+=gridDim.x) {
        int64_t lb = labels[i];
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        if (lb == ignore_index) {
            for (int j{tid}; j < dimsize; j+=blockDim.x) {
                int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
                grad_logits[idx] = 0;
            }
            continue;
        } 

        // compute max value for each vector for softmax
        sdata[tid] = -1000;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            if (dval > sdata[tid]) sdata[tid] = dval;
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                if (sdata[idx] < sdata[idx + s]) sdata[idx] = sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x] = sdata[0]; // max logits without label
            sdata[blockDim.x + 1] = sdata[0]; // max logits with label
            scalar_t dval = logits[lb];
            if (dval > sdata[0]) sdata[blockDim.x + 1] = dval;
        }

        // compute exp sum for softmax
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            sdata[tid] += expf(dval - sdata[blockDim.x]);
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x + 2] = sdata[0]; // exp sum without label
        }
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            sdata[tid] += expf(dval - sdata[blockDim.x + 1]);
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x + 3] = sdata[0]; // exp sum with label
        }

        // compute sum of q * x
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            if (j == lb) continue;
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            scalar_t tmp = dval * expf(dval - sdata[blockDim.x]);
            sdata[tid] += tmp;
            if (i == 0) {
                if (tid == 0) printf("qx: ");
                printf("%f, ", tmp / sdata[blockDim.x + 2]);
            }
        }
        __syncthreads();
        for (int s=1; s < blockDim.x; s*=2) {
            int idx = 2 * s * threadIdx.x;
            if (idx < blockDim.x && idx + s < blockDim.x) {
                sdata[idx] += sdata[idx + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sdata[blockDim.x + 4] = sdata[0] / sdata[blockDim.x + 2]; 
            if (i == 0)
            printf("\nsum of qx: %f\n", sdata[blockDim.x + 4]);
        }
        for (int j{tid}; j < dimsize; j+=blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
            scalar_t dval = logits[idx];
            scalar_t pc = expf(dval - sdata[blockDim.x + 1]) / sdata[blockDim.x + 3];
            if (i == 0) {
                if (tid == 0) printf("p: ");
                printf("%f, ", pc);
            }
            scalar_t gval;
            if (j == lb) {
                gval = pc - 1.;
            } else {
                gval = dval - sdata[blockDim.x + 4] + 1.;
                gval *= expf(dval - sdata[blockDim.x]) / sdata[blockDim.x + 2]; 
                gval = pc + (gval - coeff) * lam / 2.;
            }
            grad_logits[idx] = gval * grad[idx];
            // sdata[tid] += dval * expf(dval - sdata[blockDim.x]);
            if (i == 0 && tid == 0) printf("\n gval: ");
            if (i == 0) {
                printf("%f, ", gval);
            }
            if (i == 0 && tid == 0) printf("\n grad_output: " );
            if (i == 0 && tid == 0) {
                printf("%f, ", grad[idx]);
            }
            if (i == 0 && tid == 0) printf("\n");
        }

    }
}


// cuda forward and backward
at::Tensor large_margin_forward_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
    // CHECK type and shape
    AT_ASSERTM(logits.type().is_cuda(), "logits should be cuda");
    AT_ASSERTM(labels.type().is_cuda(), "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    cout << "n_size: " << n_size << endl;
    cout << "dimsize: " << dimsize << endl;
    cout << "m_size: " << m_size << endl;

    // allocate memory and cuda grid/block
    auto losses = torch::empty_like(labels, logits.options());

    dim3 grid1(std::min(samplesize, (int)4096));
    dim3 block1(std::min(dimsize, (int)BLOCKSIZE));
    if (losses.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return losses;
    }

    // cout << "call forward kernel\n";
    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(losses.scalar_type(), "large margin forward", [&] {
        int shm_size = (std::min((int)BLOCKSIZE, dimsize) * 2 + 8) * sizeof(scalar_t); 
        cout << "shm_size: " << shm_size << endl;
        LMarginLossForward<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            losses.contiguous().data<scalar_t>(),
            ignore_index, lam 
        );
    });
    THCudaCheck(cudaGetLastError());
    return losses;
}


at::Tensor large_margin_backward_cuda(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float lam) {
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

    dim3 grid(std::min(samplesize, (int)4096));
    dim3 block(std::min(dimsize, (int)BLOCKSIZE));
    if (grad_logits.numel() == 0) {
        THCudaCheck(cudaGetLastError());
        return grad_logits;
    }

    // call kernel
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_logits.scalar_type(), "lsr backwrd", [&] {
        int shm_size = (std::min((int)BLOCKSIZE, dimsize) * 2 + 8) * sizeof(scalar_t); 
        LMarginLossBackward<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            grad.contiguous().data<scalar_t>(), 
            grad_logits.contiguous().data<scalar_t>(),
            logits.contiguous().data<scalar_t>(), 
            labels.contiguous().data<int64_t>(), 
            ignore_index,lam 
        );
    });
    THCudaCheck(cudaGetLastError());
    return grad_logits;
}

// python inferface
at::Tensor large_margin_forward(const at::Tensor &logits,
                             const at::Tensor &labels,
                             const float lam,
                             const int64_t ignore_index) {
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_forward_cuda(logits, labels, ignore_index, lam);
}


at::Tensor large_margin_backward(const at::Tensor &grad,
                                  const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const float lam,
                                  const int64_t ignore_index) {
    // TODO: try AT_ASSERTM
    if (!(logits.type().is_cuda() && labels.type().is_cuda())) {
        AT_ERROR("this large margin loss only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return large_margin_backward_cuda(grad, logits, labels, ignore_index, lam);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("l_margin_forward", &large_margin_forward, "large margin forward");
    m.def("l_margin_backward", &large_margin_backward, "large margin backward");
}


#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cfloat>

#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

#define BLOCKSIZE 1024


namespace one_hot_space {

template<typename T>
class max_op {
public:
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a > b ? a : b;
    }
};


template<typename T>
class min_op {
public:
    __device__ __forceinline__ T operator()(T a, T b) const {
        return a < b ? a : b;
    }
};


template<template<typename> class Reduction, typename scalar_t>
__device__ __forceinline__ void reduce_op(
        scalar_t* sdata, int blocksize,
        const Reduction<scalar_t>& oper) {
    int tid = threadIdx.x;
    __syncthreads();
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = oper(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }
}


__global__ void find_max_min(int64_t *data, int64_t *buffer, int samplesize,
        int64_t ignore_index) {
    __shared__ int64_t sdata[BLOCKSIZE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int sample_offset = gridDim.x * blockDim.x;

    // find max
    sdata[threadIdx.x] = -100000;
    __syncthreads();

    int64_t min{100000};
    for (int i{tid}; i < samplesize; i += sample_offset) {
        int64_t val = data[i];
        if (val == ignore_index) continue;
        if (sdata[threadIdx.x] < val) {
            sdata[threadIdx.x] = val;
        }
        if (min > val) min = val;
    }
    reduce_op<max_op, int64_t>(sdata, BLOCKSIZE, max_op<int64_t>());
    if (threadIdx.x == 0) {
        buffer[blockIdx.x] = sdata[0];
    }
    if (blockIdx.x == 0) {
        sdata[threadIdx.x] = -100000;
        __syncthreads();
        for (int i{static_cast<int>(threadIdx.x)}; i < gridDim.x; i += blockDim.x) {
            if (sdata[threadIdx.x] < buffer[i]) {
                sdata[threadIdx.x] = buffer[i];
            }
        }
        reduce_op<max_op, int64_t>(sdata, BLOCKSIZE, max_op<int64_t>());
        if (threadIdx.x == 0) {
            buffer[gridDim.x] = sdata[0];
        }
    }

    // find min
    sdata[threadIdx.x] = min;
    __syncthreads();

    reduce_op<min_op, int64_t>(sdata, BLOCKSIZE, min_op<int64_t>());
    if (threadIdx.x == 0) {
        buffer[blockIdx.x] = sdata[0];
    }
    if (blockIdx.x == 0) {
        sdata[threadIdx.x] = 100000;
        __syncthreads();
        for (int i{static_cast<int>(threadIdx.x)}; i < gridDim.x; i += blockDim.x) {
            if (sdata[threadIdx.x] > buffer[i]) {
                sdata[threadIdx.x] = buffer[i];
            }
        }
        __syncthreads();
        reduce_op<min_op, int64_t>(sdata, BLOCKSIZE, min_op<int64_t>());
        if (threadIdx.x == 0) {
            buffer[gridDim.x + 1] = sdata[0];
        }
    }
}

}


// kernel functions
__global__ void OneHotEncoder(const int n_size,
                            const int dimsize, const int m_size, 
                            float *one_hot,
                            int64_t *labels,
                            const int ignore_index,
                            const float lb_pos, 
                            const float lb_neg) {

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    int samplesize = n_size * m_size;
    int sample_offset = gridDim.x * blockDim.y;

    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int lb = static_cast<int>(labels[i]);

        if (lb == ignore_index) {
            for (int j{tid}; j < dimsize; j += blockDim.x) {
                int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
                one_hot[idx] = 0;
            }
        } else {
            for (int j{tid}; j < dimsize; j += blockDim.x) {
                int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
                if (j == lb) {
                    one_hot[idx] = lb_pos;
                } else {
                    one_hot[idx] = lb_neg;
                }
            }
        }
    }
}


__global__ void OneHotEncoderSpatial(const int n_size, 
                            const int dimsize, const int m_size, 
                            float *one_hot,
                            int64_t *labels,
                            const int ignore_index,
                            const float lb_pos, 
                            const float lb_neg) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int sample_offset = gridDim.x * blockDim.x;
    const int datasize = n_size * m_size;

    for (int i{tid}; i < datasize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        const int lb = static_cast<int>(labels[i]);
            
        if (lb == ignore_index) {
            for (int j{0}; j < dimsize; ++j) {
                const int dst_idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
                one_hot[dst_idx] = 0;
            }
        } else {
            for (int j{0}; j < dimsize; ++j) {
                const int dst_idx = n_idx * dimsize * m_size + j * m_size + m_idx; 
                if (j == lb) {
                    one_hot[dst_idx] = lb_pos;
                } else {one_hot[dst_idx] = lb_neg;}
            }
        }
    }
}


// cuda functions
at::Tensor Label_one_hot_cuda(const at::Tensor &labels,
                              const int64_t ignore_index,
                              const float smooth,
                              const int64_t min_len) {
    // CHECK type and shape
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");
    TORCH_CHECK(labels.dtype() == torch::kLong, "date type should be kLong");

    // auto shape = labels.sizes().vec(); // shape is std::vector

    vector<int64_t> size;
    for (int i{0}; i < labels.dim(); ++i) {
        size.push_back(labels.size(i));
        if (i == 0) {
            size.push_back(min_len);
        }
    }
    // TODO: chekc whether options is changed inplace -- no
    auto options = labels.options().dtype(torch::kFloat);
    auto one_hot = torch::empty(size, options);

    const int samplesize = labels.numel();
    const int n_size = one_hot.size(0);
    const int dimsize = one_hot.size(1);
    const int datasize = one_hot.numel();
    const int m_size = datasize / (n_size * dimsize);

    // find max and min and check
    int block0x = 32;
    while (block0x < samplesize) block0x *= 2;
    block0x = std::max(std::min(block0x / 2, BLOCKSIZE), 32);
    int grid0x = std::min(std::max(1, samplesize / block0x), 4096);
    dim3 block0(block0x);
    dim3 grid0(grid0x);
    int shm_size = block0x * sizeof(int64_t);
    auto buffer = torch::empty({grid0x + 2}, labels.options()); // +1 is max, +2 is min
    one_hot_space::find_max_min<<<grid0, block0, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            labels.contiguous().data_ptr<int64_t>(),
            buffer.contiguous().data_ptr<int64_t>(),
            samplesize, ignore_index);

    int64_t max{buffer[grid0x].item().toLong()};
    int64_t min{buffer[grid0x + 1].item().toLong()};
    TORCH_CHECK(max < min_len && min >= 0, "label should be within 0 and min_len\n");

    // set values
    const float lb_pos{1.f - smooth};
    const float lb_neg{smooth / dimsize};
    if (dimsize < 32 && samplesize > (4 * 1024)) {
        int gridx = std::max(1, std::min(4096, int(datasize / BLOCKSIZE)));
        dim3 block(BLOCKSIZE);
        dim3 grid(gridx);
        OneHotEncoderSpatial<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            one_hot.contiguous().data_ptr<float>(), 
            labels.contiguous().data_ptr<int64_t>(), 
            static_cast<int>(ignore_index), lb_pos, lb_neg
        );
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min(blockx / 2, BLOCKSIZE), 32);
        int blocky = std::min(BLOCKSIZE / blockx, samplesize);
        int gridx = std::min(std::max(1, samplesize / blocky), 4096);
        dim3 block(blockx, blocky);
        dim3 grid(gridx);
        OneHotEncoder<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            n_size, dimsize, m_size, 
            one_hot.contiguous().data_ptr<float>(), 
            labels.contiguous().data_ptr<int64_t>(), 
            static_cast<int>(ignore_index), lb_pos, lb_neg
        );
    }

    THCudaCheck(cudaGetLastError());
    return one_hot;
}


// python inferface
at::Tensor Label_one_hot(const at::Tensor &labels,
                         const int64_t ignore_index,
                         const float smooth,
                         const int64_t min_len) {
    if (labels.device().type() != c10::kCUDA) {
        AT_ERROR("this onehot method only supports gpu mode\n");
    } 
    at::DeviceGuard guard(labels.device());
    return Label_one_hot_cuda(labels, ignore_index, smooth, min_len);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("label_one_hot", &Label_one_hot, "label one hot");
}

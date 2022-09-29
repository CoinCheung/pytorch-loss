
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>


#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <cfloat>

#include <iostream>
#include "common.hpp"

using std::cout;
using std::endl;

#define BLOCKSIZE 1024


namespace ohem_space {

template<typename scalar_t>
__forceinline__ __device__ void reduce_sum(scalar_t *sdata, int blocksize, int tid) {
    __syncthreads();
    // NOTE: block size should be 2 ** x
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}


template<typename scalar_t>
__forceinline__ __device__ void reduce_max(scalar_t* sdata, int blocksize, int tid) {
    __syncthreads();
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
}
}


// kernel functions
template<typename scalar_t>
__global__ void OHEMGetScores(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *logits,
                            scalar_t *scores,
                            const int64_t *labels,
                            int *indices,
                            const int64_t ignore_index) {
    // shared memory
    extern __shared__ __align__(sizeof(scalar_t)) unsigned char sdata_raw[];
    scalar_t *sdata = reinterpret_cast<scalar_t*>(sdata_raw);

    int sample_offset = gridDim.x * blockDim.y;
    sdata = sdata + blockDim.x * threadIdx.y;

    int tid = threadIdx.x;
    int sample_id = blockIdx.x * blockDim.y + threadIdx.y;
    int samplesize = n_size * m_size;

    for (int i{sample_id}; i < samplesize; i += sample_offset) {
        indices[i] = i;
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int64_t lb = labels[i];

        if (lb == ignore_index) {
            if (tid == 0) scores[i] = scalar_t(1.);
            continue;
        }

        // obtain max
        sdata[tid] = scalar_t(-10000.);
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (val > sdata[tid]) sdata[tid] = val;
        }
        __syncthreads();
        ohem_space::reduce_max<scalar_t>(sdata, blockDim.x, tid);
        scalar_t max_val = sdata[0];

        // obtain exp sum
        sdata[tid] = 0.;
        __syncthreads();
        for (int j{tid}; j < dimsize; j += blockDim.x) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            sdata[tid] += expf(logits[idx] - max_val);
        }
        __syncthreads();
        ohem_space::reduce_sum<scalar_t>(sdata, blockDim.x, tid);
        if (tid == 0) {
            int idx = n_idx * dimsize * m_size + lb * m_size + m_idx;
            scores[i] = expf(logits[idx] - max_val) / sdata[0];
        }
    }
}


template<typename scalar_t>
__global__ void OHEMGetScoresSpatial(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *logits,
                            scalar_t *scores,
                            const int64_t *labels,
                            int *indices,
                            const int64_t ignore_index) {
    int sample_offset = gridDim.x * blockDim.x;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int samplesize = n_size * m_size;

    for (int i{tid}; i < samplesize; i += sample_offset) {
        indices[i] = i;
        int n_idx = i / m_size;
        int m_idx = i % m_size;
        int lb = static_cast<int>(labels[i]);

        if (lb == ignore_index) {
            scores[i] = scalar_t(1.);
            continue;
        }

        // obtain max
        scalar_t max_val = scalar_t(-10000.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            scalar_t val = logits[idx];
            if (val > max_val) max_val = val;
        }
        // obtain sum exp
        scalar_t sum_exp = scalar_t(0.);
        for (int j{0}; j < dimsize; ++j) {
            int idx = n_idx * dimsize * m_size + j * m_size + m_idx;
            sum_exp += expf(logits[idx] - max_val);
        }
        int idx = n_idx * dimsize * m_size + lb * m_size + m_idx;
        scores[i] = expf(logits[idx] - max_val) / sum_exp;
    }
}


template<typename scalar_t>
__global__ void OHEMSetLabels(const int samplesize,
                            const int *idx,
                            const scalar_t *scores,
                            int64_t *ohem_label,
                            const int64_t ignore_index,
                            const float score_thresh, 
                            const int64_t n_min) {
    int sample_offset = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i{static_cast<int>(n_min) + tid}; i < samplesize; i += sample_offset) {
        if (scores[i] > score_thresh) ohem_label[idx[i]] = ignore_index;
    }
}

// cuda functions
at::Tensor Score_ohem_label_cuda(const at::Tensor &logits,
                                  const at::Tensor &labels,
                                  const int64_t ignore_index,
                                  const float score_thresh,
                                  const int64_t n_min) {
    // CHECK type and shape
    AT_ASSERTM(logits.device().type() == c10::kCUDA, "logits should be cuda");
    AT_ASSERTM(labels.device().type() == c10::kCUDA, "labels should be cuda");

    const int n_size = logits.size(0);
    const int dimsize = logits.size(1);
    const int m_size = logits.numel() / (n_size * dimsize);
    const int samplesize = labels.numel();

    if (n_min >= samplesize) return labels;

    // allocate memory and cuda grid/block
    auto ohem_label = labels.clone();
    auto scores = torch::empty_like(labels, logits.options());
    thrust::device_vector<int> idx(samplesize);
    if (ohem_label.numel() == 0) {
        AT_CUDA_CHECK(cudaGetLastError());
        return ohem_label;
    }

    // call kernel
    if (dimsize < 32 && samplesize > (4 * 1024)) {
        int gridx = std::min((int)4096, int(samplesize / BLOCKSIZE));
        gridx = std::max((int)1, gridx);
        dim3 block1(BLOCKSIZE);
        dim3 grid1(gridx);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "ohem score label", [&] {
        
            OHEMGetScoresSpatial<scalar_t><<<grid1, block1, 0, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                logits.contiguous().data_ptr<scalar_t>(), 
                scores.contiguous().data_ptr<scalar_t>(),
                labels.contiguous().data_ptr<int64_t>(), 
                thrust::raw_pointer_cast(&idx[0]),
                ignore_index
            );
        });
    } else {
        int blockx = 32;
        while (blockx < dimsize) blockx *= 2;
        blockx = std::max(std::min((int)BLOCKSIZE, blockx / 2), (int)32);
        int blocky = std::min(samplesize, (int)(BLOCKSIZE / blockx));
        blocky = std::max((int)1, blocky);
        int gridx = std::min(4096, (int)(samplesize / blocky));
        gridx = std::max((int)1, gridx);
        int n_shm = blockx * blocky;
        dim3 block1(blockx, blocky);
        dim3 grid1(gridx);

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "ohem score label", [&] {
        
            int shm_size = n_shm * sizeof(scalar_t); 
            OHEMGetScores<scalar_t><<<grid1, block1, shm_size, at::cuda::getCurrentCUDAStream()>>>(
                n_size, dimsize, m_size, 
                logits.contiguous().data_ptr<scalar_t>(), 
                scores.contiguous().data_ptr<scalar_t>(),
                labels.contiguous().data_ptr<int64_t>(), 
                thrust::raw_pointer_cast(&idx[0]),
                ignore_index
            );
        });
    }


    int grid2_num = std::min(4096, (int)(samplesize / BLOCKSIZE));
    grid2_num = std::max((int)1, grid2_num);
    dim3 block2(BLOCKSIZE);
    dim3 grid2(grid2_num);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(logits.scalar_type(), "ohem score label", [&] {

        thrust::sort_by_key(
            thrust::device,
            scores.contiguous().data_ptr<scalar_t>(),
            scores.contiguous().data_ptr<scalar_t>() + samplesize,
            &idx[0]
        );

        OHEMSetLabels<scalar_t><<<grid2, block2, 0, at::cuda::getCurrentCUDAStream()>>>(
            samplesize, thrust::raw_pointer_cast(&idx[0]), 
            scores.contiguous().data_ptr<scalar_t>(),
            ohem_label.contiguous().data_ptr<int64_t>(), 
            ignore_index, score_thresh, n_min
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());
    return ohem_label;
}


// python inferface
at::Tensor Score_ohem_label(const at::Tensor &logits,
                         const at::Tensor &labels,
                         const int64_t ignore_index,
                         const float score_thresh,
                         const int64_t n_min) {
    if ((logits.device().type() != c10::kCUDA) || (labels.device().type() != c10::kCUDA)) {
        AT_ERROR("this ohem method only supports gpu mode\n");
    } 
    at::DeviceGuard guard(logits.device());
    return Score_ohem_label_cuda(logits, labels, ignore_index, score_thresh, n_min);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("score_ohem_label", &Score_ohem_label, "ohem by score on label");
}

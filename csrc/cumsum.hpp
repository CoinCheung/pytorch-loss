#include <vector>
#include <numeric>
#include <algorithm>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cooperative_groups.h>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <torch/types.h>

#include <cuda.h>
#include <cfloat>


#define BLOCKSIZE 512



template <typename T>
__global__ void cumsum_by_row_kernel_v2(T *arr, int m_size, int n_size,
        int n_blockx_per_row, int n_blockx_total) {
    extern __shared__ __align__(sizeof(T)) unsigned char sdata_raw[];
    T *shared = reinterpret_cast<T*>(sdata_raw);
    int b_offset = blockDim.x << 1;
    int shm_off = b_offset * threadIdx.y;
    int shm_ind = shm_off + threadIdx.x;

    int n_sm_blocks = gridDim.x * blockDim.y;

    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    for (int block_idx{tid}; block_idx < n_blockx_total; block_idx += n_sm_blocks) {

        int m_block = block_idx / n_blockx_per_row;
        int n_block = block_idx % n_blockx_per_row;
        int n_ind = n_block * b_offset + threadIdx.x;
        int arr_ind = m_block * n_size + n_ind; 
        if (n_ind < n_size) {
            shared[shm_ind] = arr[arr_ind];
        } else {
            shared[shm_ind] = 0;
        }
        if (n_ind + blockDim.x < n_size) {
            shared[shm_ind + blockDim.x] = arr[arr_ind + blockDim.x];
        } else {
            shared[shm_ind + blockDim.x] = 0;
        }
        __syncthreads();

        for (int s{1}; s < b_offset; s <<= 1) {
            int idx = s * 2 * (threadIdx.x + 1) - 1 - s;
            if (idx + s < b_offset) {
                shared[shm_off + idx + s] += shared[shm_off + idx];
            }
            __syncthreads();
        }

        int bsize = blockDim.x;
        for (int s{bsize}; s > 0; s >>= 1) {
            int idx = s * 2 * (threadIdx.x + 1) - 1;
            if (idx + s < b_offset) {
                shared[shm_off + idx + s] += shared[shm_off + idx];
            }
            __syncthreads();
        }

        if (n_ind < n_size) {
            arr[arr_ind] = shared[shm_ind];
        }
        if (n_ind + blockDim.x < n_size) {
            arr[arr_ind + blockDim.x] = shared[shm_ind + blockDim.x];
        }
    }
}


template <typename T>
__global__ void cumsum_merge_kernel_v2(T *arr, int m_size, int n_size) {
    int n_blockx_per_row = n_size / blockDim.x;
    int remain = n_size % blockDim.x;
    int tid = blockIdx.x * blockDim.y + threadIdx.y;
    int tstrd = gridDim.x * blockDim.y;
    for (int i{tid}; i < m_size; i += tstrd) {
        int base_idx = i * n_size;
        int j = 1;

        for (; j < n_blockx_per_row; ++j) {
            int idx_curr = base_idx + j * blockDim.x + threadIdx.x;
            int idx_prev = base_idx + j * blockDim.x - 1;
            arr[idx_curr] += arr[idx_prev];
            __syncthreads();
        }
        int idx_curr = base_idx + n_blockx_per_row * blockDim.x + threadIdx.x;
        int idx_prev = base_idx + n_blockx_per_row * blockDim.x - 1;
        int val = 0;
        if (n_blockx_per_row > 0) val = arr[idx_prev]; 
        if (threadIdx.x < remain) {
            arr[idx_curr] += val;
        }
        __syncthreads();

    }
}




void cumsum_2d_by_row_v2(at::Tensor arr) {
    int m_size = arr.size(0);
    int n_size = arr.size(1);
    int blockx = 64;
    while (blockx < n_size && blockx < BLOCKSIZE) blockx <<= 1;
    blockx = min(BLOCKSIZE, blockx);
    int n_blockx_per_row = n_size / blockx;
    if (n_size % blockx > 0) n_blockx_per_row += 1;
    int n_blockx_total = m_size * n_blockx_per_row;
    int blocky = max(min(n_blockx_total, BLOCKSIZE / blockx), 1);
    int gridx = max(min(4096, n_blockx_total / blocky), 1);

    blockx >>= 1;

    dim3 block(blockx, blocky);
    dim3 grid(gridx);
    // compute by blockx

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(arr.scalar_type(), "cumsum by cuda block", [&] {
        int shm_size = BLOCKSIZE * sizeof(scalar_t) * 2;
        cumsum_by_row_kernel_v2<scalar_t><<<grid, block, shm_size, at::cuda::getCurrentCUDAStream()>>>(
            arr.contiguous().data_ptr<scalar_t>(), 
            m_size, n_size, n_blockx_per_row, n_blockx_total);
    });

    // try to merge blockx
    blockx <<= 1;
    if (blockx >= n_size) return;
    blocky = max(min(m_size, BLOCKSIZE / blockx), 1);
    gridx = max(1, min(m_size / blocky, 4096));
    block = dim3(blockx, blocky);
    grid = dim3(gridx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(arr.scalar_type(), "cumsum merge", [&] {
        cumsum_merge_kernel_v2<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            arr.contiguous().data_ptr<scalar_t>(), m_size, n_size);
    });
}


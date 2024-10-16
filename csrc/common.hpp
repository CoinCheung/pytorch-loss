
#include <cuda.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace math_ops {


// exp func
template<typename scalar_t>
__forceinline__ __device__
scalar_t Exp(scalar_t x) {
    return expf(static_cast<float>(x));
}

template<>
__forceinline__ __device__
double Exp(double x) {
    return exp(x);
}


template<>
__forceinline__ __device__
c10::Half Exp(c10::Half x) {
    // return expf(static_cast<float>(x));
    return hexp(static_cast<__half>(x));
}


//
// log func
template<typename scalar_t>
__forceinline__ __device__
scalar_t Log(scalar_t x) {
    return logf(static_cast<float>(x));
}

template<>
__forceinline__ __device__
double Log(double x) {
    return log(x);
}

template<>
__forceinline__ __device__
c10::Half Log(c10::Half x) {
    return hlog(static_cast<__half>(x));
}


// 
// log1p
template<typename scalar_t>
__forceinline__ __device__
scalar_t Log1p(scalar_t x) {
    return log1pf(static_cast<float>(x));
}


template<>
__forceinline__ __device__
double Log1p(double x) {
    return log1p(x);
}


// 
// pow
template<typename scalar_t>
__forceinline__ __device__
scalar_t Pow(scalar_t x, scalar_t y) {
    return powf(static_cast<float>(x), static_cast<float>(y));
}

template<>
__forceinline__ __device__
double Pow(double x, double y) {
    return pow(x, y);
}

// sqrt
template<typename scalar_t>
__forceinline__ __device__
scalar_t Sqrt(scalar_t x) {
    return sqrt(x);
}

template<>
__forceinline__ __device__
c10::Half Sqrt(c10::Half x) {
    return sqrtf(static_cast<float>(x));
}

// rsqrt
template<typename scalar_t>
__forceinline__ __device__
scalar_t Rsqrt(scalar_t x) {
    return rsqrt(x);
}

template<>
__forceinline__ __device__
c10::Half Rsqrt(c10::Half x) {
    return rsqrtf(static_cast<float>(x));
}


// abs func
template<typename scalar_t>
__forceinline__ __device__
scalar_t Abs(scalar_t x) {
    return x > scalar_t(0.) ? x : -x;
}

}


/**
   Computes ceil(a / b)
*/
template <typename T>
__host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

/**
   Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
   multiple of b
*/
template <typename T>
__host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
  return THCCeilDiv(a, b) * b;
}


namespace block_ops {


template<typename scalar_t>
__forceinline__ __device__
void broadcast_block(scalar_t& val, int src_id) {
    __shared__ scalar_t shm; 
    if (threadIdx.x == src_id) {
        shm = val;
    }
    __syncthreads();
    val = shm;
}

template<typename scalar_t>
__forceinline__ __device__ 
void reduce_max_shm(scalar_t* sdata, int tid) {
    __syncthreads();
    for (unsigned int s{blockDim.x / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid] < sdata[tid + s]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__forceinline__ __device__ 
void reduce_sum_shm(scalar_t* sdata, int tid) {
    __syncthreads();
    for (unsigned int s{blockDim.x / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__forceinline__ __device__
void reduce_sum_shfl(scalar_t& val, bool broadcast) {
    /* this requires:
     * 1. warp layout is along x axis
     * 2. blockDim.x should be divisble by 32
     * 3. blockDim.x should be less or equal to 1024
     * 4. warpSize should be 32
     * 5. only thread with threadIdx.x == 0 obtains correct answer */

    __syncthreads();
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    __shared__ scalar_t shm[32];

    if (threadIdx.x % 32 == 0) {
        shm[threadIdx.x >> 5] = val;
    }
    __syncthreads();

    val = scalar_t(0.);

    /* from here actually only one warp work */
    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shm[threadIdx.x];
    }

    if (threadIdx.x < 32) {
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }

    if (broadcast) {
        broadcast_block(val, 0);
    }
}


/* logic is same as above, but must write this if we want to make it work */
template<> __forceinline__ __device__
void reduce_sum_shfl(c10::Half& val, bool broadcast) {
    __syncthreads();

    /* here we should cast to __half explicitly */
    val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 16);
    val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 8);
    val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 4);
    val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 2);
    val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 1);

    __shared__ __half shm[32];

    if (threadIdx.x % 32 == 0) {
        shm[threadIdx.x >> 5] = val;
    }
    __syncthreads();

    /* here we should use __double2half to assign val into zero */
    val = __double2half(0.);

    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shm[threadIdx.x];
    }

    if (threadIdx.x < 32) {
        val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 16);
        val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 8);
        val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 4);
        val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 2);
        val += __shfl_down_sync(0xffffffff, static_cast<__half>(val), 1);
    }

    if (broadcast) {
        broadcast_block(val, 0);
    }
}


template<typename scalar_t>
__forceinline__ __device__
void reduce_max_shfl(scalar_t& val, bool broadcast) {
    /* same as reduce_sum_shfl, this requires:
     * 1. warp layout is along x axis
     * 2. blockDim.x should be divisble by 32
     * 3. blockDim.x should be less or equal to 1024
     * 4. warpSize should be 32
     * 5. only thread with threadIdx.x == 0 obtains correct answer */

    __syncthreads();
    scalar_t tmp;
    tmp = __shfl_down_sync(0xffffffff, val, 16);
    if (tmp > val) val = tmp;
    tmp = __shfl_down_sync(0xffffffff, val, 8);
    if (tmp > val) val = tmp;
    tmp = __shfl_down_sync(0xffffffff, val, 4);
    if (tmp > val) val = tmp;
    tmp = __shfl_down_sync(0xffffffff, val, 2);
    if (tmp > val) val = tmp;
    tmp = __shfl_down_sync(0xffffffff, val, 1);
    if (tmp > val) val = tmp;

    __shared__ scalar_t shm[32];

    if (threadIdx.x % 32 == 0) {
        shm[threadIdx.x >> 5] = val;
    }
    __syncthreads();

    /* from here actually only one warp work */
    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shm[threadIdx.x];
    }

    if (threadIdx.x < 32) {
        tmp = __shfl_down_sync(0xffffffff, val, 16);
        if (tmp > val) val = tmp;
        tmp = __shfl_down_sync(0xffffffff, val, 8);
        if (tmp > val) val = tmp;
        tmp = __shfl_down_sync(0xffffffff, val, 4);
        if (tmp > val) val = tmp;
        tmp = __shfl_down_sync(0xffffffff, val, 2);
        if (tmp > val) val = tmp;
        tmp = __shfl_down_sync(0xffffffff, val, 1);
        if (tmp > val) val = tmp;
    }

    if (broadcast) {
        broadcast_block(val, 0);
    }
}


}


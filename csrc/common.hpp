
#include <cuda.h>
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

namespace math_ops {

// exp func
template<typename scalar_t>
__forceinline__ __device__
scalar_t Exp(scalar_t x) {
    return exp(x);
}

template<>
__forceinline__ __device__
c10::Half Exp(c10::Half x) {
    // return expf(static_cast<float>(x));
    return hexp(x);
}

//
// log func
template<typename scalar_t>
__forceinline__ __device__
scalar_t Log(scalar_t x) {
    return log(x);
}

template<>
__forceinline__ __device__
c10::Half Log(c10::Half x) {
    // return logf(static_cast<float>(x));
    return hlog(x);
}

// 
// log1p
template<typename scalar_t>
__forceinline__ __device__
scalar_t Log1p(scalar_t x) {
    return log1p(x);
}

template<>
__forceinline__ __device__
c10::Half Log1p(c10::Half x) {
    return log1p(static_cast<float>(x));
}

// 
// pow
template<typename scalar_t>
__forceinline__ __device__
scalar_t Pow(scalar_t x, scalar_t y) {
    return pow(x, y);
}

template<>
__forceinline__ __device__
c10::Half Pow(c10::Half x, c10::Half y) {
    return pow(static_cast<float>(x), static_cast<float>(y));
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


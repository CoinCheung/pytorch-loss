
#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>


template<typename scalar_t>
scalar_t Exp(scalar_t x) {
    return exp(x);
}

template<>
c10::Half Exp(c10::Half x) {
    return exp(x);
}


#ifndef __BRAHLE_CUDA_UTILS
#define __BRAHLE_CUDA_UTILS

#include "cub/cub.cuh"
#include <iostream>

#define CUDA_CHECK(call)                                                            \
    if((call) != cudaSuccess) {                                                     \
        cudaError_t err = cudaGetLastError();                                       \
        std::cerr << "CUDA error calling ""#call"", code is " << err << std::endl;  \
        exit(err); }


struct IndexedValue {
  int value, index;
  __device__ __host__ __forceinline__ bool operator< (const IndexedValue &other) const {
    if (index != other.index) return index > other.index;
    return value < other.value;
  }
};

template< typename T > struct MaxOperator {
  __device__ __host__ MaxOperator() {}

  __device__ __host__ T operator() (const T &A, const T &B) const {
    if (A < B) return B;
    return A;
  }
};

inline __device__ int max2(int a, int b) {
  if (a >= b) { 
    return a;
  }
  return b;
}

inline __device__ int min2(int a, int b) {
  if (a < b) { 
    return a;
  }
  return b;
}

inline __device__ __host__ int max3(int a, int b, int c) {
  if (a >= b) {
    if (a >= c) {
      return a;
    } else {
      return c;
    }
  }
  if (b >= c) {
    return b;
  }
  return c;
}

#endif

#ifndef __BRAHLE_CUDA_UTILS
#define __BRAHLE_CUDA_UTILS

#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <climits>

#include "cub/cub.cuh"

struct DeviceMaxOperator {
  __device__ DeviceMaxOperator() {}

  __device__ int operator() (const int &A, const int &B) const {
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

inline __device__ int max3(int a, int b, int c) {
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

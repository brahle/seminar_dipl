#include "utils.cuh"
#include "utils.h"

using namespace std;

const int INF = 0x3f3f3f3f;

const int THREADS = 2;
const int ITEMS_PER_THREAD = 1000;
const int N = THREADS * ITEMS_PER_THREAD;

const int H = 3; // penalty for the start of the gap
const int G = 1; // penalty for each element of the gap

inline __device__ int f(char A, char B) {
  return (A==B);
}

__global__ void calculateT1(int *T1_new, int *T1_old, int *T2_old, int *T3_old, char A, char *B) {
  for (int j = ITEMS_PER_THREAD*threadIdx.x; j < ITEMS_PER_THREAD*(threadIdx.x+1); ++j) {
    if (j) {
      T1_new[j] = f(A, B[j-1]) + max3(T1_old[j-1], T2_old[j-1], T3_old[j-1]);
    } else {
      T1_new[j] = -INF;
    }
  }
}

__global__ void calculateT3(int *T3_new, int *T1_old, int *T2_old, int *T3_old, int i) {
  for (int j = ITEMS_PER_THREAD*threadIdx.x; j < ITEMS_PER_THREAD*(threadIdx.x+1); ++j) {
    if (j) {
      T3_new[j] = max3(T1_old[j]-H, T2_old[j]-H, T3_old[j])-G;
    } else {
      T3_new[j] = -H - G*i;
    }
  }
}

__global__ void calculateT2(int *T2_new, int *T1_new, int *T2_old, int *T3_new) {
  int W[ITEMS_PER_THREAD];
  for (int j = ITEMS_PER_THREAD*threadIdx.x, jj = 0; j < ITEMS_PER_THREAD*(threadIdx.x+1); ++j, ++jj) {
    if (j) {
      W[jj] = max2(T1_new[j-1], T3_new[j-1])-G-H;
    } else {
      W[jj] = -INF; 
    }
    W[jj] += j*G;
  }

  typedef cub::BlockScan<int, THREADS> BlockScan;
  __shared__ typename BlockScan::SmemStorage smem_storage;
  BlockScan::ExclusiveScan(smem_storage, W, W, 0, MaxOperator());

  for (int j = ITEMS_PER_THREAD*threadIdx.x, jj = 0; j < ITEMS_PER_THREAD*(threadIdx.x+1); ++j, ++jj) {
    if (j) {
      T2_new[j] = W[jj] - j*G;
    } else {
      T2_new[j] = -INF;
    }
  }
}

int T1[2][N];
int T2[2][N];
int T3[2][N];
char A[N], B[N];

int main(void) {
	for (int i = 0; i+1 < N; ++i) {
    A[i] = "ABCD"[rand()%4];
    B[i] = "ABCD"[rand()%4];
  }
  A[N-1] = 0;
  B[N-1] = 0;
//  puts(A);
//  puts(B);

  int now = 0, old = 1;
  for (int j = 0; j < N; ++j) {
    T1[0][j] = -INF;
    T2[0][j] = -H - G*j;
    T3[0][j] = -INF;
    T1[1][j] = -INF;
    T2[1][j] = -INF;
    T3[1][j] = -INF;
  }
  T1[0][0] = 0;
  T3[0][0] = -H;

  int *dev_T1[2];
  int *dev_T2[2];
  int *dev_T3[2];
  char *dev_A, *dev_B;

  for (int i = 0; i < 2; ++i) {
    cudaMalloc(&dev_T1[i], N*sizeof(int));
    cudaMemcpy(dev_T1[i], T1[i], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_T2[i], N*sizeof(int));
    cudaMemcpy(dev_T2[i], T2[i], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_T3[i], N*sizeof(int));
    cudaMemcpy(dev_T3[i], T3[i], N*sizeof(int), cudaMemcpyHostToDevice);
  }
  cudaMalloc(&dev_A, N*sizeof(char));
  cudaMalloc(&dev_B, N*sizeof(char));
  cudaMemcpy(dev_A, A, N*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, N*sizeof(char), cudaMemcpyHostToDevice);

  clock_t start_time = clock();

  for (int i = 1; i < N; ++i) {
    now ^= 1;
    old = (now ^ 1);

    calculateT1<<<1, THREADS>>>(dev_T1[now], dev_T1[old], dev_T2[old], dev_T3[old], A[i-1], dev_B);
    calculateT3<<<1, THREADS>>>(dev_T3[now], dev_T1[old], dev_T2[old], dev_T3[old], i);
    calculateT2<<<1, THREADS>>>(dev_T2[now], dev_T1[now], dev_T2[old], dev_T3[now]);
  }

  cudaMemcpy(T1[0], dev_T1[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T2[0], dev_T2[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T3[0], dev_T3[now], N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("Maximum complete alignment is %d.\n", max3(T1[0][N-1], T2[0][N-1], T3[0][N-1]));

  clock_t end_time = clock();
  printf("Clock = %d ticks = %.4gs\n", end_time-start_time, (end_time-start_time)/(double)CLK_TCK);

  for (int i = 0; i < 2; ++i) {
    cudaFree(dev_T1[i]);
    cudaFree(dev_T2[i]);
    cudaFree(dev_T3[i]);
  }
  cudaFree(dev_A);
  cudaFree(dev_B);

	system("pause");
	return 0;
}

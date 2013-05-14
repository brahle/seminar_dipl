#include "utils.cuh"

using namespace std;

const int INF = 0x3f3f3f3f;

const int THREADS = 5;
const int ITEMS_PER_THREAD = 1;
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

  BlockScan::ExclusiveScan(smem_storage, W, W, 0, DeviceMaxOperator());

  for (int j = ITEMS_PER_THREAD*threadIdx.x, jj = 0; j < ITEMS_PER_THREAD*(threadIdx.x+1); ++j, ++jj) {
    if (j) {
      T2_new[j] = W[jj] - j*G;
    } else {
      T2_new[j] = -INF;
    }
  }
}

int main(void) {
  char A[N], B[N];
	for (int i = 0; i+1 < N; ++i) {
    A[i] = "ABCD"[rand()%4];
    B[i] = "ABCD"[rand()%4];
  }
  A[N-1] = 0;
  B[N-1] = 0;
  puts(A);
  puts(B);

  int T1[2][N];
  int T2[2][N];
  int T3[2][N];

  int now = 0;
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

  int *T1_old, *T1_new;
  int *T2_old, *T2_new;
  int *T3_old, *T3_new;
  char *dev_A, *dev_B;

  cudaMalloc(&T1_new, N*sizeof(int));
  cudaMalloc(&T2_new, N*sizeof(int));
  cudaMalloc(&T3_new, N*sizeof(int));
  cudaMalloc(&T1_old, N*sizeof(int));
  cudaMalloc(&T2_old, N*sizeof(int));
  cudaMalloc(&T3_old, N*sizeof(int));
  cudaMalloc(&dev_A, N*sizeof(char));
  cudaMalloc(&dev_B, N*sizeof(char));

  cudaMemcpy(dev_A, A, N*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, B, N*sizeof(char), cudaMemcpyHostToDevice);

  for (int i = 1; i < N; ++i) {
    if (i == 1) {
      for (int j = 0; j < N; ++j) {
        printf("%12d%12d%12d; ", T1[!now][j], T2[!now][j], T3[!now][j]);
      }
      printf("\n");
    }

    printf("Gledam za znak %c\n", A[i-1]);
    now ^= 1;
    cudaMemcpy(T1_old, T1[!now], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(T2_old, T2[!now], N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(T3_old, T3[!now], N*sizeof(int), cudaMemcpyHostToDevice);

    calculateT1<<<1, THREADS>>>(T1_new, T1_old, T2_old, T3_old, A[i-1], dev_B);
    calculateT3<<<1, THREADS>>>(T3_new, T1_old, T2_old, T3_old, i);
    calculateT2<<<1, THREADS>>>(T2_new, T1_new, T2_old, T3_new);

    cudaMemcpy(T1[now], T1_new, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(T2[now], T2_new, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(T3[now], T3_new, N*sizeof(int), cudaMemcpyDeviceToHost);

    for (int j = 0; j < N; ++j) {
      printf("%12d%12d%12d; ", T1[now][j], T2[now][j], T3[now][j]);
    }
    printf("\n");
  }

	system("pause");
	return 0;
}

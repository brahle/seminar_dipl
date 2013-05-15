#include "utils.cuh"
#include "utils.h"

using namespace std;

const int INF = 0x3f3f3f3f;

const int THREADS = 1000;
const int ITEMS_PER_THREAD = 2;
const int M = 2000;
const int N = THREADS * ITEMS_PER_THREAD;


const int H = 3; // penalty for the start of the gap
const int G = 1; // penalty for each element of the gap

inline __device__ __host__ int f(char A, char B) {
  return (A==B);
}

__global__ void calculate_T1_and_T3(int *T1_new, int *T3_new, int *T1_old, int *T2_old, int *T3_old, char A, char *B, int i) {
  register int j = ITEMS_PER_THREAD*threadIdx.x;
  register int end = ITEMS_PER_THREAD*(threadIdx.x+1);
  if (j == 0) {
    T1_new[j] = 0;
    T3_new[j] = 0;
    ++j;
  }
  for (; j < end; ++j) {
    T1_new[j] = max2(f(A, B[j-1]) + max3(T1_old[j-1], T2_old[j-1], T3_old[j-1]), 0);
    T3_new[j] = max2(max3(T1_old[j]-H, T2_old[j]-H, T3_old[j])-G, 0);
  }
}

__global__ void calculate_T2(int *T2_new, int *T1_new, int *T2_old, int *T3_new) {
  int W[ITEMS_PER_THREAD];
  register int j = ITEMS_PER_THREAD*threadIdx.x, jj = 0;
  register int end = ITEMS_PER_THREAD*(threadIdx.x+1);
  if (j == 0) {
    W[j] = 0;
    ++j; ++jj;
  }
  for (; j < end; ++j, ++jj) {
    W[jj] = max2(T1_new[j-1], T3_new[j-1]) - G - H + j*G;
  }

  typedef cub::BlockScan<int, THREADS> BlockScan;
  __shared__ typename BlockScan::SmemStorage smem_storage;
  BlockScan::ExclusiveScan(smem_storage, W, W, 0, MaxOperator());

  j = ITEMS_PER_THREAD*threadIdx.x;
  jj = 0;
  if (j == 0) {
    T2_new[j] = 0;
    ++j; ++jj;
  }
  for (; j < end; ++j, ++jj) {
    T2_new[j] = max2(0, W[jj] - j*G);
  }
}

int T1[2][N];
int T2[2][N];
int T3[2][N];
char A[M], B[N];

// MOCK function, for now
void read_protein(char *A, int n) {
  for (int i = 0; i+1 < n; ++i) {
    A[i] = "ABCD"[rand()%4];
  }
  A[n] = 0;
}

void init() {
  for (int i = 0; i < 2; ++i) {
    fill(T1[i], T1[i]+N, -INF);
    fill(T3[i], T3[i]+N, -INF);
  }
  for (int j = 0; j < N; ++j) {
    T2[0][j] = -H - G*j;
  }
  fill(T2[1], T2[1]+N, -INF);
  T1[0][0] = 0;
  T3[0][0] = -H;
}

template< typename T > 
inline void copy_to_device(T **dest, T *src, int n) {
  CUDA_CHECK(cudaMalloc(dest, n*sizeof(T)));
  CUDA_CHECK(cudaMemcpy(*dest, src, n*sizeof(T), cudaMemcpyHostToDevice));
}

template< typename T >
inline void copy_to_host(T **dest, T *src, int n) {
  cudaMemcpy(dest, src, n*sizeof(T), cudaMemcpyDeviceToHost);
  cudaFree(src);
}

int main(void) {
  read_protein(A, M);
  read_protein(B, N);

  int now = 0, old = 1;
  init();

  int *dev_T1[2];
  int *dev_T2[2];
  int *dev_T3[2];
  char *dev_A, *dev_B;
  
  for (int i = 0; i < 2; ++i) {
    copy_to_device(&dev_T1[i], T1[i], N);
    copy_to_device(&dev_T2[i], T2[i], N);
    copy_to_device(&dev_T3[i], T3[i], N);
  }
  copy_to_device(&dev_A, A, N);
  copy_to_device(&dev_B, B, N);

  clock_t start_time = clock();

  for (int i = 1; i < M; ++i) {
    now ^= 1;
    old = (now ^ 1);

    calculate_T1_and_T3<<<1, THREADS>>>(dev_T1[now], dev_T3[now], dev_T1[old], dev_T2[old], dev_T3[old], A[i-1], dev_B, i);
    calculate_T2<<<1, THREADS>>>(dev_T2[now], dev_T1[now], dev_T2[old], dev_T3[now]);
  }

  cudaMemcpy(T1[0], dev_T1[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T2[0], dev_T2[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T3[0], dev_T3[now], N*sizeof(int), cudaMemcpyDeviceToHost);

  printf("Maximum complete alignment is %d.\n", max3(T1[0][N-1], T2[0][N-1], T3[0][N-1]));

  clock_t end_time = clock();
  printf("Clock = %d ticks; %.4gs\n", end_time-start_time, (end_time-start_time)/(double)CLK_TCK);

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

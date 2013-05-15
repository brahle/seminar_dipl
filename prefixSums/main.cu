#include "utils.cuh"
#include "utils.h"

using namespace std;

const int INF = 0x3f3f3f3f;

const int THREADS = 1000;
int M = 5000;
int N = 5000;
int ITEMS_PER_THREAD = (N+THREADS-1) / THREADS;

const int H = 3; // penalty for the start of the gap
const int G = 1; // penalty for each element of the gap

inline __device__ __host__ int f(char A, char B) {
  return (A==B);
}

__global__ void calculate_T1_and_T3(int *T1_new, int *T3_new,
                                    int *T1_old, int *T2_old, int *T3_old,
                                    char A, char *B, int *total_max,
                                    int ITEMS_PER_THREAD, int N) {
  register int j = ITEMS_PER_THREAD*threadIdx.x;
  register int end = min2(ITEMS_PER_THREAD*(threadIdx.x+1), N);
  int max_value = 0;
  if (j == 0) {
    T1_new[j] = 0;
    T3_new[j] = 0;
    ++j;
  }
  for (; j < end; ++j) {
    T1_new[j] = max2(f(A, B[j-1]) + max3(T1_old[j-1], T2_old[j-1], T3_old[j-1]), 0);
    T3_new[j] = max2(max3(T1_old[j]-H, T2_old[j]-H, T3_old[j])-G, 0);
    if (T1_new[j] > max_value) {
      max_value = T1_new[j];
    }
  }

  typedef cub::BlockReduce<int, THREADS> BlockReduce;
  __shared__ typename BlockReduce::SmemStorage smem_storage;
  max_value = BlockReduce::Reduce(smem_storage, max_value, MaxOperator());
  if (threadIdx.x && *total_max < max_value) {
    *total_max = max_value;
  }
}

__global__ void calculate_T2(int *T2_new, int *T1_new, int *T2_old, int *T3_new, int ITEMS_PER_THREAD, int N) {
  register int start = ITEMS_PER_THREAD*threadIdx.x;
  register int j = start;
  register int end = min2(ITEMS_PER_THREAD*(threadIdx.x+1), N);
  int max_value = 0;

  if (j == 0) {
    T2_new[j] = 0;
    ++j;
  }
  for (; j < end; ++j) {
    T2_new[j] = max2(T1_new[j-1], T3_new[j-1]) - G - H + j*G;
    if (T2_new[j] > max_value) {
      max_value = T2_new[j];
    }
  }

  typedef cub::BlockScan<int, THREADS> BlockScan;
  __shared__ typename BlockScan::SmemStorage smem_storage;
  BlockScan::ExclusiveScan(smem_storage, max_value, max_value, 0, MaxOperator());

  j = start;
  T2_new[j] = max2(0, max_value - j*G);
  ++j;
  for (; j < end; ++j) {
    T2_new[j] = max2(0, T2_new[j-1] - j*G);
  }
}

int *T1[2];
int *T2[2];
int *T3[2];
char *A, *B;

// MOCK function, for now
void read_protein(char *A, int n) {
  for (int i = 0; i+1 < n; ++i) {
    A[i] = "ABCD"[rand()%4];
  }
  A[n-1] = 0;
}

void init() {
  for (int i = 0; i < 2; ++i) {
    T1[i] = new int[N];
    T2[i] = new int[N];
    T3[i] = new int[N];
  }
  A = new char[M];
  B = new char[N];

  for (int i = 0; i < 2; ++i) {
    fill(T1[i], T1[i]+N, 0);
    fill(T2[i], T2[i]+N, 0);
    fill(T3[i], T3[i]+N, 0);
  }
}

void finalize() {
  for (int i = 0; i < 2; ++i) {
    delete [] T1[i];
    delete [] T2[i];
    delete [] T3[i];
  }
  delete [] A;
  delete [] B;
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
  init();
  read_protein(A, M);
  read_protein(B, N);
  puts(A);
  puts(B);

  int now = 0, old = 1;

  int *dev_T1[2];
  int *dev_T2[2];
  int *dev_T3[2];
  char *dev_A, *dev_B;
  
  for (int i = 0; i < 2; ++i) {
    copy_to_device(&dev_T1[i], T1[i], N);
    copy_to_device(&dev_T2[i], T2[i], N);
    copy_to_device(&dev_T3[i], T3[i], N);
  }
  copy_to_device(&dev_A, A, M);
  copy_to_device(&dev_B, B, N);

  int max_alignment = -8;
  int *dev_max_alignment;
  cudaMalloc(&dev_max_alignment, sizeof(int));
  cudaMemcpy(dev_max_alignment, &max_alignment, sizeof(int), cudaMemcpyHostToDevice);

  /*for (int j = 0; j < N; ++j) {
    printf("%d %d %d;  ", T1[0][j], T2[0][j], T3[0][j]);
  }
  printf("\n");*/

  clock_t start_time = clock();

  for (int i = 1; i < M; ++i) {
    now ^= 1;
    old = (now ^ 1);

    calculate_T1_and_T3<<<1, THREADS>>>(dev_T1[now], dev_T3[now], dev_T1[old], dev_T2[old], dev_T3[old], A[i-1], dev_B, dev_max_alignment, ITEMS_PER_THREAD, N);
    calculate_T2<<<1, THREADS>>>(dev_T2[now], dev_T1[now], dev_T2[old], dev_T3[now], ITEMS_PER_THREAD, N);

/*    cudaMemcpy(T1[now], dev_T1[now], N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(T2[now], dev_T2[now], N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(T3[now], dev_T3[now], N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("%c s: ", A[i-1]);
    for (int j = 0; j < N; ++j) {
      printf("%c=%d: %d %d %d;  ", j?B[j-1]:' ', j?f(A[i-1],B[j-1]):0, T1[now][j], T2[now][j], T3[now][j]);
    }
    printf("\n"); */
  }

  cudaMemcpy(T1[0], dev_T1[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T2[0], dev_T2[now], N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(T3[0], dev_T3[now], N*sizeof(int), cudaMemcpyDeviceToHost);

  cudaMemcpy(&max_alignment, dev_max_alignment, sizeof(int), cudaMemcpyDeviceToHost);
  printf("Maximum local alignment is %d.\n", max_alignment);

  clock_t end_time = clock();
  printf("Clock = %d ticks; %.4gs\n", end_time-start_time, (end_time-start_time)/(double)CLK_TCK);

  for (int i = 0; i < 2; ++i) {
    cudaFree(dev_T1[i]);
    cudaFree(dev_T2[i]);
    cudaFree(dev_T3[i]);
  }
  cudaFree(dev_A);
  cudaFree(dev_B);
  cudaFree(dev_max_alignment);

	system("pause");
	return 0;
}

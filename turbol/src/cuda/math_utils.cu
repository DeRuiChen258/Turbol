#include <cuda_runtime.h>
#include <math.h>

extern "C" {

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

cudaError_t gpu_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

}

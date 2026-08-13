#include <cuda_runtime.h>

// ============================================================================
// TurboRL CUDA math primitives — element-wise kernels used by TensorUtils
// and the Policy/Reward engines.
// ============================================================================

extern "C" {

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void sub_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] - b[i];
}

__global__ void fill_kernel(float* out, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = value;
}

__global__ void add_scalar_kernel(float* inout, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inout[i] += scalar;
}

__global__ void scale_kernel(float* inout, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inout[i] *= scalar;
}

// ---- Host launch helpers ----

cudaError_t gpu_add(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t gpu_sub(const float* a, const float* b, float* c, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sub_kernel<<<grid, block, 0, stream>>>(a, b, c, n);
    return cudaGetLastError();
}

cudaError_t gpu_fill(float* out, float value, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    fill_kernel<<<grid, block, 0, stream>>>(out, value, n);
    return cudaGetLastError();
}

cudaError_t gpu_add_scalar(float* inout, float scalar, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_scalar_kernel<<<grid, block, 0, stream>>>(inout, scalar, n);
    return cudaGetLastError();
}

cudaError_t gpu_scale(float* inout, float scalar, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    scale_kernel<<<grid, block, 0, stream>>>(inout, scalar, n);
    return cudaGetLastError();
}

} // extern "C"

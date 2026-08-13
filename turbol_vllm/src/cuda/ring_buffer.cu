#include <cuda_runtime.h>

// ============================================================================
// Ring-buffer push kernel — vectorized (float4) copy of `count` elements into
// a circular buffer.
//
//   Atomic reservation: thread 0 reserves a contiguous block of `count` slots
//   via a single atomicAdd; all threads then write to `(base + idx) % capacity`.
//   Element storage is contiguous (elem_size floats per element), so float4
//   vectorization is valid whenever elem_size % 4 == 0 (cudaMalloc returns
//   256-byte-aligned pointers, keeping every element 16-byte aligned).
// ============================================================================

extern "C" {

__global__ void ring_buffer_push_kernel(
    float* __restrict__ buffer,
    const float* __restrict__ data,
    int* head,
    int capacity,
    int elem_size,
    int count)
{
    __shared__ int base;
    if (threadIdx.x == 0) base = atomicAdd(head, count);
    __syncthreads();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const int dst = (base + idx) % capacity;
    const int src = idx;

    if (elem_size % 4 == 0) {
        // Vectorized path: float4 (16-byte) loads/stores.
        float4* buf4 = reinterpret_cast<float4*>(buffer);
        const float4* data4 = reinterpret_cast<const float4*>(data);
        const int vec = elem_size / 4;
        for (int i = 0; i < vec; ++i)
            buf4[dst * vec + i] = data4[src * vec + i];
    } else {
        for (int i = 0; i < elem_size; ++i)
            buffer[dst * elem_size + i] = data[src * elem_size + i];
    }
}

cudaError_t ring_buffer_push(
    float* buffer,
    const float* data,
    int* head,
    int capacity,
    int elem_size,
    int count,
    cudaStream_t stream)
{
    const int block = 256;
    const int grid = (count + block - 1) / block;
    ring_buffer_push_kernel<<<grid, block, 0, stream>>>(
        buffer, data, head, capacity, elem_size, count);
    return cudaGetLastError();
}

} // extern "C"

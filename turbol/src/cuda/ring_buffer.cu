#include <cuda_runtime.h>

extern "C" {

__global__ void ring_buffer_push_kernel(
    float* buffer,
    const float* data,
    int* head,
    int capacity,
    int elem_size,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int pos = atomicAdd(head, count);
    int actual_pos = (pos + idx) % capacity;
    
    for (int i = 0; i < elem_size; i++) {
        buffer[actual_pos * elem_size + i] = data[idx * elem_size + i];
    }
}

cudaError_t ring_buffer_push(
    float* buffer,
    const float* data,
    int* head,
    int capacity,
    int elem_size,
    int count,
    cudaStream_t stream
) {
    int block = 256;
    int grid = (count + block - 1) / block;
    ring_buffer_push_kernel<<<grid, block, 0, stream>>>(
        buffer, data, head, capacity, elem_size, count);
    return cudaGetLastError();
}

}

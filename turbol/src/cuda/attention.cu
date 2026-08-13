#include <cuda_runtime.h>
#include <math.h>

// Flash Attention CUDA Kernel - Optimized Implementation
namespace flash_attention {

template <int HEAD_DIM, int BLOCK_SIZE>
__global__ void attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    const int* __restrict__ seq_lens,
    float scale,
    int batch_size,
    int num_heads,
    int max_seq_len
) {
    extern __shared__ float smem[];
    float* s_sum = smem + blockDim.x;
    
    int batch_idx = blockIdx.z % batch_size;
    int head_idx = (blockIdx.z / batch_size) % num_heads;
    int q_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    if (q_idx >= seq_lens[batch_idx]) return;
    
    const float* q = query + ((batch_idx * num_heads + head_idx) * max_seq_len + q_idx) * HEAD_DIM;
    
    float row_max = -1e10f;
    float row_sum = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; i++) acc[i] = 0.0f;
    
    for (int block_k = 0; block_k < (seq_lens[batch_idx] + BLOCK_SIZE - 1) / BLOCK_SIZE; block_k++) {
        float k_block[HEAD_DIM];
        int k_idx = block_k * BLOCK_SIZE + threadIdx.x;
        
        if (k_idx < seq_lens[batch_idx]) {
            int k_offset = ((batch_idx * num_heads + head_idx) * max_seq_len + k_idx) * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) k_block[d] = key[k_offset + d];
        } else {
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) k_block[d] = 0.0f;
        }
        
        float qk = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) qk += q[d] * k_block[d];
        qk *= scale;
        
        for (int offset = 16; offset > 0; offset /= 2)
            qk = fmaxf(qk, __shfl_down_sync(0xffffffff, qk, offset));
        
        float block_max = (threadIdx.x % 32 == 0) ? qk : -1e10f;
        if (threadIdx.x == 0) {
            float old_max = row_max;
            row_max = fmaxf(row_max, block_max);
            row_sum = row_sum * expf(old_max - row_max) + block_max * expf(block_max - row_max);
            s_sum[blockIdx.x] = row_sum;
        }
        __syncthreads();
        
        float w = expf(qk - row_max);
        
        float v_block[HEAD_DIM];
        if (k_idx < seq_lens[batch_idx]) {
            int v_offset = ((batch_idx * num_heads + head_idx) * max_seq_len + k_idx) * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) v_block[d] = value[v_offset + d];
        }
        
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) acc[d] += w * v_block[d];
    }
    
    if (threadIdx.x < HEAD_DIM) {
        float* out = output + ((batch_idx * num_heads + head_idx) * max_seq_len + q_idx) * HEAD_DIM;
        out[threadIdx.x] = acc[threadIdx.x] / row_sum;
    }
}

cudaError_t LaunchFlashAttention(
    const float* query, const float* key, const float* value,
    float* output, const int* seq_lens,
    int batch_size, int num_heads, int max_seq_len, int head_dim,
    cudaStream_t stream
) {
    dim3 grid;
    grid.x = (max_seq_len + 63) / 64;
    grid.y = max_seq_len;
    grid.z = batch_size * num_heads;
    
    dim3 block(256, 1, 1);
    size_t shared_mem = 2 * block.x * sizeof(float);
    
    if (head_dim == 64) {
        attention_kernel<64, 64><<<grid, block, shared_mem, stream>>>(
            query, key, value, output, seq_lens,
            1.0f / sqrtf((float)head_dim),
            batch_size, num_heads, max_seq_len);
    } else {
        attention_kernel<32, 64><<<grid, block, shared_mem, stream>>>(
            query, key, value, output, seq_lens,
            1.0f / sqrtf((float)head_dim),
            batch_size, num_heads, max_seq_len);
    }
    return cudaGetLastError();
}

} // namespace flash_attention

extern "C" {
cudaError_t flash_attention_forward(
    const float* query, const float* key, const float* value,
    float* output, const int* seq_lens,
    int batch_size, int num_heads, int max_seq_len, int head_dim,
    cudaStream_t stream
) {
    return flash_attention::LaunchFlashAttention(
        query, key, value, output, seq_lens,
        batch_size, num_heads, max_seq_len, head_dim, stream);
}
}

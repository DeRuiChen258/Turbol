#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// Flash-Attention-style scaled dot-product attention with *online softmax*
// (numerically stable running max / running-sum normalization).
//
//   One thread computes one query row: it walks every key of its sequence,
//   maintaining running max `m`, running sum `l`, and the rescaled
//   accumulator `acc`. This is the standard FlashAttention-2 normalization —
//   stable, no materialized attention matrix, correct under variable seq lens.
// ============================================================================

namespace flash_attention {

template <int HEAD_DIM>
__global__ void attention_forward_kernel(
    const float* __restrict__ query,   // [batch, heads, max_seq_len, head_dim]
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    const int* __restrict__ seq_lens,  // [batch]
    float scale,
    int batch_size,
    int num_heads,
    int max_seq_len)
{
    const int batch = blockIdx.z / num_heads;
    const int head = blockIdx.z % num_heads;
    const int q_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batch_size || q_idx >= max_seq_len) return;
    const int seq_len = seq_lens[batch];
    if (q_idx >= seq_len) return;

    const int base = (batch * num_heads + head) * max_seq_len;
    const float* q = query + (base + q_idx) * HEAD_DIM;

    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    float m = -INFINITY;
    float l = 0.0f;

    for (int j = 0; j < seq_len; ++j) {
        const float* k = key + (base + j) * HEAD_DIM;
        const float* v = value + (base + j) * HEAD_DIM;

        float qk = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) qk += q[d] * k[d];
        qk *= scale;

        const float m_new = fmaxf(m, qk);
        const float alpha = expf(m - m_new);   // rescale previous accumulator
        const float p = expf(qk - m_new);      // unnormalized attention weight

        l = l * alpha + p;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] = acc[d] * alpha + p * v[d];
        m = m_new;
    }

    float* out = output + (base + q_idx) * HEAD_DIM;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) out[d] = acc[d] * inv;
}

cudaError_t LaunchFlashAttention(
    const float* query, const float* key, const float* value,
    float* output, const int* seq_lens,
    int batch_size, int num_heads, int max_seq_len, int head_dim,
    cudaStream_t stream)
{
    const int block = 128;
    const int grid_x = (max_seq_len + block - 1) / block;
    dim3 grid(grid_x, 1, batch_size * num_heads);
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    if (head_dim == 64) {
        attention_forward_kernel<64><<<grid, block, 0, stream>>>(
            query, key, value, output, seq_lens, scale,
            batch_size, num_heads, max_seq_len);
    } else if (head_dim == 32) {
        attention_forward_kernel<32><<<grid, block, 0, stream>>>(
            query, key, value, output, seq_lens, scale,
            batch_size, num_heads, max_seq_len);
    } else if (head_dim == 128) {
        attention_forward_kernel<128><<<grid, block, 0, stream>>>(
            query, key, value, output, seq_lens, scale,
            batch_size, num_heads, max_seq_len);
    } else {
        return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

} // namespace flash_attention

extern "C" {
cudaError_t flash_attention_forward(
    const float* query, const float* key, const float* value,
    float* output, const int* seq_lens,
    int batch_size, int num_heads, int max_seq_len, int head_dim,
    cudaStream_t stream)
{
    return flash_attention::LaunchFlashAttention(
        query, key, value, output, seq_lens,
        batch_size, num_heads, max_seq_len, head_dim, stream);
}
} // extern "C"

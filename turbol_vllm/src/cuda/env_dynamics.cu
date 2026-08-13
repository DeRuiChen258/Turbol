#include <cuda_runtime.h>

// ============================================================================
// TurboRL vectorized environment dynamics kernels.
//   Backend: linear dynamical system  s' = decay·s + force·a,  r = -||s'||².
//   One thread per environment instance.
// ============================================================================

extern "C" {

__global__ void env_step_linear_kernel(
    const float* __restrict__ state,    // [num_envs, obs_dim]
    const float* __restrict__ action,   // [num_envs, act_dim]
    float* __restrict__ next_state,     // [num_envs, obs_dim]
    float* __restrict__ reward,         // [num_envs]
    bool*  __restrict__ done,           // [num_envs]
    int num_envs, int obs_dim, int act_dim,
    float decay, float force, float done_threshold)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;

    float cost = 0.0f;
    float first = 0.0f;
    for (int d = 0; d < obs_dim; ++d) {
        int a = (d < act_dim) ? d : (act_dim - 1);
        float ns = decay * state[i * obs_dim + d] + force * action[i * act_dim + a];
        next_state[i * obs_dim + d] = ns;
        cost += ns * ns;
        if (d == 0) first = ns;
    }
    reward[i] = -cost;
    done[i] = fabsf(first) > done_threshold;
}

// Deterministic pseudo-random reset (xorshift-multiply mix, no RNG state).
__global__ void env_reset_kernel(
    float* __restrict__ state, int num_envs, int obs_dim, unsigned seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_envs) return;

    for (int d = 0; d < obs_dim; ++d) {
        unsigned h = seed ^ (static_cast<unsigned>(i) * 73856093u)
                          ^ (static_cast<unsigned>(d) * 19349663u);
        h ^= h >> 13;
        h *= 2654435761u;
        h ^= h >> 16;
        float u = static_cast<float>(h & 0xFFFFu) / 65535.0f;
        state[i * obs_dim + d] = (u * 2.0f - 1.0f) * 0.1f;  // [-0.1, 0.1]
    }
}

cudaError_t env_step_linear(
    const float* state, const float* action,
    float* next_state, float* reward, bool* done,
    int num_envs, int obs_dim, int act_dim,
    float decay, float force, float done_threshold, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_envs + block - 1) / block;
    env_step_linear_kernel<<<grid, block, 0, stream>>>(
        state, action, next_state, reward, done,
        num_envs, obs_dim, act_dim, decay, force, done_threshold);
    return cudaGetLastError();
}

cudaError_t env_reset(
    float* state, int num_envs, int obs_dim, unsigned seed, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_envs + block - 1) / block;
    env_reset_kernel<<<grid, block, 0, stream>>>(state, num_envs, obs_dim, seed);
    return cudaGetLastError();
}

} // extern "C"

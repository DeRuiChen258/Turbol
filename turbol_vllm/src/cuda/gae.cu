#include <cuda_runtime.h>

// ============================================================================
// Generalized Advantage Estimation (GAE) — one thread per environment.
//
//   δ_t = r_t + γ·(1 - done_t)·V(s_{t+1}) - V(s_t)
//   A_t = δ_t + (γλ)·(1 - done_t)·A_{t+1}
//   G_t = A_t + V(s_t)
//
//   Rewards/values/dones are laid out [T, num_envs] (values [T+1, num_envs]
//   with the terminal bootstrap in the last row). The recurrence is backward
//   over time, independent per environment, so it parallelizes cleanly.
// ============================================================================

extern "C" {

__global__ void gae_kernel(
    const float* __restrict__ rewards,   // [T, num_envs]
    const float* __restrict__ values,    // [T+1, num_envs]
    const bool*  __restrict__ dones,     // [T, num_envs]
    float* __restrict__ advantages,      // [T, num_envs]
    float* __restrict__ returns,         // [T, num_envs]
    int T, int num_envs, float gamma, float lambda)
{
    const int env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= num_envs) return;

    float next_adv = 0.0f;
    float next_value = values[T * num_envs + env];

    for (int t = T - 1; t >= 0; --t) {
        const bool done = dones[t * num_envs + env];
        const float r = rewards[t * num_envs + env];
        const float v = values[t * num_envs + env];

        const float delta = r + (done ? 0.0f : gamma * next_value) - v;
        const float adv = delta + (done ? 0.0f : gamma * lambda * next_adv);

        advantages[t * num_envs + env] = adv;
        returns[t * num_envs + env] = adv + v;

        next_adv = adv;
        next_value = v;
    }
}

cudaError_t gae_compute(
    const float* rewards, const float* values, const bool* dones,
    float* advantages, float* returns,
    int T, int num_envs, float gamma, float lambda, cudaStream_t stream)
{
    const int block = 256;
    const int grid = (num_envs + block - 1) / block;
    gae_kernel<<<grid, block, 0, stream>>>(
        rewards, values, dones, advantages, returns,
        T, num_envs, gamma, lambda);
    return cudaGetLastError();
}

} // extern "C"

#include "turbol/env/gpu_environment.hpp"

#include <cmath>

namespace turborl {
namespace env {

namespace {

// CPU mirror of the CUDA `env_reset_kernel` (deterministic xorshift-multiply
// mix). Keeps CPU and CUDA resets bit-identical for a given seed.
float DeterministicUnit(unsigned i, unsigned d, unsigned seed) {
    unsigned h = seed ^ (i * 73856093u) ^ (d * 19349663u);
    h ^= h >> 13;
    h *= 2654435761u;
    h ^= h >> 16;
    return static_cast<float>(h & 0xFFFFu) / 65535.0f;
}

} // namespace

// CUDA dynamics kernels (src/cuda/env_dynamics.cu). Only present in CUDA builds.
#ifdef TURBORL_HAS_CUDA
extern "C" {
cudaError_t env_step_linear(const float* state, const float* action,
                            float* next_state, float* reward, bool* done,
                            int num_envs, int obs_dim, int act_dim,
                            float decay, float force, float done_threshold,
                            cudaStream_t stream);
cudaError_t env_reset(float* state, int num_envs, int obs_dim, unsigned seed,
                      cudaStream_t stream);
}
#endif

GPUEnvironment::GPUEnvironment(int num_envs, int obs_dim, int act_dim, Device device)
    : num_envs_(num_envs),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      device_(device),
      obs_shape_(std::vector<int64_t>{static_cast<int64_t>(num_envs),
                                       static_cast<int64_t>(obs_dim)}),
      action_shape_(std::vector<int64_t>{static_cast<int64_t>(num_envs),
                                          static_cast<int64_t>(act_dim)}),
      states_(obs_shape_, DataType::kFloat32, device),
      rng_(42) {
    Reset();
}

Status GPUEnvironment::Reset() {
#ifdef TURBORL_HAS_CUDA
    if (device_.is_cuda()) {
        const unsigned seed = static_cast<unsigned>(rng_());
        cudaError_t err = env_reset(states_.data<float>(), num_envs_, obs_dim_,
                                    seed, nullptr);
        if (err != cudaSuccess)
            return Status::CudaError(cudaGetErrorString(err));
        return Status::Ok();
    }
#endif
    const unsigned seed = static_cast<unsigned>(rng_());
    float* s = states_.data<float>();
    for (int i = 0; i < num_envs_; ++i) {
        for (int d = 0; d < obs_dim_; ++d) {
            const float u = DeterministicUnit(static_cast<unsigned>(i),
                                              static_cast<unsigned>(d), seed);
            s[i * obs_dim_ + d] = (u * 2.0f - 1.0f) * 0.1f;  // [-0.1, 0.1]
        }
    }
    return Status::Ok();
}

Status GPUEnvironment::Step(const Tensor& action, Tensor* observation,
                            Tensor* reward, Tensor* done) {
    if (!observation || !reward || !done)
        return Status::InvalidArgument("Step output tensors must be non-null");
    if (action.dtype() != DataType::kFloat32)
        return Status::InvalidArgument("action must be Float32");
    if (action.numel() != static_cast<int64_t>(num_envs_) * act_dim_)
        return Status::InvalidArgument("action has wrong number of elements");
    if (action.device() != device_)
        return Status::InvalidArgument("action device must match environment device");

    *reward = Tensor(Shape(std::vector<int64_t>{num_envs_}), DataType::kFloat32, device_);
    *done = Tensor(Shape(std::vector<int64_t>{num_envs_}), DataType::kBool, device_);

    const float* a = action.data<float>();
    float* s = states_.data<float>();   // updated in-place (per-index, no cross-dependency)
    float* r = reward->data<float>();
    bool* d = done->data<bool>();

#ifdef TURBORL_HAS_CUDA
    if (device_.is_cuda()) {
        cudaError_t err = env_step_linear(s, a, s, r, d, num_envs_, obs_dim_,
                                          act_dim_, decay_, force_,
                                          done_threshold_, nullptr);
        if (err != cudaSuccess)
            return Status::CudaError(cudaGetErrorString(err));
    } else
#endif
    {
        for (int i = 0; i < num_envs_; ++i) {
            float cost = 0.0f;
            float first = 0.0f;
            for (int dd = 0; dd < obs_dim_; ++dd) {
                const int aa = (dd < act_dim_) ? dd : (act_dim_ - 1);
                const float v = decay_ * s[i * obs_dim_ + dd] + force_ * a[i * act_dim_ + aa];
                s[i * obs_dim_ + dd] = v;
                cost += v * v;
                if (dd == 0) first = v;
            }
            r[i] = -cost;
            d[i] = std::fabs(first) > done_threshold_;
        }
    }

    // Hand the caller a snapshot of the new state (deep copy on the env device).
    *observation = states_.Clone();
    return Status::Ok();
}

Tensor GPUEnvironment::Observe() const {
    return states_.Clone();
}

} // namespace env
} // namespace turborl

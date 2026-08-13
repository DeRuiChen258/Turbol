#pragma once
#include "../common.hpp"

namespace turborl {
namespace env {

// ============================================================================
// GPUEnvironment — vectorized, device-resident environment.
//
//   A batch of `num_envs` independent instances, all stepped together. The
//   default backend is a linear dynamical system
//       s' = decay·s + force·a
//       r  = -||s'||²        (quadratic LQR-style cost)
//       done = |s'₀| > threshold
//   which exercises the full reset/step/observe pipeline on CPU or CUDA.
//   State stays on the configured device end-to-end (no host round-trips).
// ============================================================================
class GPUEnvironment {
public:
    GPUEnvironment(int num_envs = 1, int obs_dim = 4, int act_dim = 2,
                   Device device = Device::CPU());

    // Reinitialize all environments to their initial state.
    Status Reset();

    // Advance every environment by one step.
    //   action: [num_envs, act_dim]
    //   observation: [num_envs, obs_dim] (written with next state)
    //   reward:      [num_envs]
    //   done:        [num_envs] bool
    Status Step(const Tensor& action, Tensor* observation,
                Tensor* reward, Tensor* done);

    // Current state [num_envs, obs_dim].
    Tensor Observe() const;

    int NumEnvs() const { return num_envs_; }
    const Shape& ObsShape() const { return obs_shape_; }
    const Shape& ActionShape() const { return action_shape_; }
    Device device() const { return device_; }

    // Dynamics parameters (readable for diagnostics / tests).
    float decay() const { return decay_; }
    float force() const { return force_; }
    float done_threshold() const { return done_threshold_; }

private:
    int num_envs_;
    int obs_dim_;
    int act_dim_;
    Device device_;

    Shape obs_shape_;
    Shape action_shape_;

    Tensor states_;  // [num_envs, obs_dim]

    float decay_ = 0.99f;
    float force_ = 0.1f;
    float done_threshold_ = 5.0f;

    std::mt19937 rng_;
};

} // namespace env
} // namespace turborl

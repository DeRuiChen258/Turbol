#pragma once
#include "../common.hpp"

#ifdef TURBORL_LIBTORCH
#include <torch/torch.h>
#endif

namespace turborl {
namespace policy {

// ============================================================================
// PolicyEngine — a small Gaussian policy trained with a REINFORCE-style
// advantage-weighted update, plus a linear value head for GAE bootstrapping.
//
//   actor  : mean = tanh(W_a · obs + b_a)        (action in [-1, 1])
//   critic : value = W_c · obs + b_c             (scalar state value)
//   a ~ N(mean, σ²) with fixed diagonal std `std_`.
//
//   Forward(obs, &action, &logprob)  — deterministic mean action + log-prob
//                                      of that action under the Gaussian.
//   GetValue(obs, &value)            — critic state-value estimate.
//   Update(batch)                    — batch layout [N, obs_dim + act_dim + 2]:
//                                      [ observation | action | advantage | return ]
//                                      loss = -(adv·logprob) + 0.5·MSE(value, return)
// ============================================================================
class PolicyEngine {
public:
    PolicyEngine(int obs_dim = 4, int act_dim = 2, Device device = Device::CPU());
    ~PolicyEngine();

    Status Initialize();
    Status Forward(const Tensor& observation, Tensor* action, Tensor* logprob);
    Status GetValue(const Tensor& observation, Tensor* value);
    Status Update(const Tensor& batch);

    int ObsDim() const { return obs_dim_; }
    int ActDim() const { return act_dim_; }
    Device device() const { return device_; }
    float StdDev() const { return std_; }

private:
    int obs_dim_;
    int act_dim_;
    Device device_;
    bool initialized_ = false;
    float std_ = 0.5f;

#ifdef TURBORL_LIBTORCH
    std::optional<torch::nn::Linear> actor_;
    std::optional<torch::nn::Linear> critic_;
    std::optional<torch::optim::Adam> optimizer_;
    std::optional<torch::Device> torch_device_;
#endif
};

} // namespace policy
} // namespace turborl

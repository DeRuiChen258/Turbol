#pragma once
#include "../common.hpp"

#ifdef TURBORL_LIBTORCH
#include <torch/torch.h>
#endif

#include <vector>
#include <string>
#include <optional>

namespace turborl {
namespace reward {

// ============================================================================
// RewardEngine — scoring module for RLHF training.
//
//   Computes scalar rewards from rollout responses.  The engine supports:
//   1. Rule-based rewards   (format, length, keyword, regex)
//   2. Model-based scores  (reward model inference via libtorch)
//   3. KL penalty          (against a reference policy, on GPU)
//   4. Reward normalization (running mean/std, per-batch, or fixed)
//
//   Output layout:  [batch_size]  (total reward per sample)
//   Also exposes per-component tensors for diagnostics.
// ============================================================================

enum class RewardMode {
    kScalar,    // one score per complete sequence
    kPerToken, // score per token (e.g. token-level RLHF)
};

struct RewardConfig {
    // KL penalty
    float  kl_coef        = 0.01f;   // coefficient for -kl penalty term
    float  kl_target       = 0.1f;    // target KL (for adaptive KL)
    bool   adaptive_kl    = false;
    float  kl_adapt_rate  = 0.01f;

    // Normalization
    bool   normalize       = true;
    float  reward_clip     = 10.0f;   // absolute value cap before normalization
    float  gamma           = 1.0f;    // discount for trajectory-level rewards

    // Reward mode
    RewardMode mode = RewardMode::kScalar;

    // Device
    Device device = Device::CPU();

    // Rule weights
    float  format_bonus   = 0.1f;
    float  length_penalty  = -0.01f;
    int    target_length   = 256;
};

// Result of one scoring call
struct RewardResult {
    Tensor total;          // [batch]  final reward
    Tensor model_scores;   // [batch]  raw reward model output
    Tensor kl_penalties;   // [batch]  KL(π || π_ref) penalties
    Tensor rule_scores;    // [batch]  sum of rule-based rewards
};

class RewardEngine {
public:
    explicit RewardEngine(const RewardConfig& config);
    ~RewardEngine();

    Status Initialize();

    // Score a batch of responses (text strings).
    // If reward_model_path is empty → rule-based scoring only.
    // responses: [batch_size]
    // Returns per-sample total reward.
    Status ScoreBatch(const std::vector<std::string>& responses,
                      const std::optional<Tensor>& ref_logprobs,
                      RewardResult* result);

    // Score a batch of pre-computed token sequences (GPU-native path).
    // token_ids:  [batch, seq_len]
    // masks:      [batch, seq_len]  (1 = valid token)
    // ref_logprobs: [batch, seq_len]  (optional KL reference)
    Status ScoreTokens(const Tensor& token_ids,
                       const Tensor& masks,
                       const std::optional<Tensor>& ref_logprobs,
                       RewardResult* result);

    // Update adaptive KL coefficient (call after each training step).
    Status UpdateAdaptiveKL(float observed_kl);

    const RewardConfig& config() const { return config_; }
    float CurrentKLCoef() const { return current_kl_coef_; }

private:
    RewardConfig config_;
    bool initialized_ = false;
    float current_kl_coef_;

    // Running reward statistics for normalization
    Tensor reward_mean_;   // [1]
    Tensor reward_var_;    // [1]  (stored as variance, not std²)
    Tensor reward_count_;  // [1]  int64
    int64_t update_count_ = 0;

#ifdef TURBORL_LIBTORCH
    std::optional<torch::nn::Linear> reward_model_;
    std::optional<torch::nn::Linear> ref_model_;
    std::optional<torch::Device>     torch_device_;
    std::optional<torch::optim::Adam> optimizer_;
#endif

    // Internal helpers
    Status ScoreWithRules(const std::vector<std::string>& responses,
                          Tensor* out);
    Status ComputeKLPenalty(const Tensor& logprobs,
                            const Tensor& ref_logprobs,
                            const Tensor& masks,
                            Tensor* out_kl);
    Status Normalize(Tensor* rewards);
    Status UpdateStats(const Tensor& rewards);
};

} // namespace reward
} // namespace turbol

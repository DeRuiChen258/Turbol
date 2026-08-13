#include "turbol/reward/reward_engine.hpp"

#include <algorithm>
#include <cmath>

#define RETURN_IF_ERROR(expr) do { auto _s = (expr); if (!_s.ok()) return _s; } while(0)

namespace turborl {
namespace reward {

RewardEngine::RewardEngine(const RewardConfig& config)
    : config_(config), current_kl_coef_(config.kl_coef) {}

RewardEngine::~RewardEngine() = default;

Status RewardEngine::Initialize() {
    if (initialized_) return Status::Ok();

    // Allocate running stats on the configured device
    reward_mean_  = Tensor(Shape({1}), DataType::kFloat32, config_.device);
    reward_var_   = Tensor(Shape({1}), DataType::kFloat32, config_.device);
    reward_count_ = Tensor(Shape({1}), DataType::kInt64,   config_.device);

    // Initialize to zero / identity
    float* m  = reward_mean_.data<float>();
    float* v  = reward_var_.data<float>();
    auto*  c  = reward_count_.data<int64_t>();
    m[0] = 0.0f; v[0] = 1.0f; c[0] = 0;

    initialized_ = true;
    return Status::Ok();
}

Status RewardEngine::ScoreBatch(const std::vector<std::string>& responses,
                                const std::optional<Tensor>& /*ref_logprobs*/,
                                RewardResult* result) {
    if (!initialized_) return Status::InvalidArgument("RewardEngine not initialized");
    if (!result) return Status::InvalidArgument("result pointer is null");

    const int B = static_cast<int>(responses.size());
    result->total       = Tensor(Shape({B}), DataType::kFloat32, config_.device);
    result->rule_scores = Tensor(Shape({B}), DataType::kFloat32, config_.device);

    // Rule-based scoring
    RETURN_IF_ERROR(ScoreWithRules(responses, &result->rule_scores));

    // Copy to total (model scores and KL are zero if no model loaded)
    auto* rule = result->rule_scores.data<float>();
    auto* tot  = result->total.data<float>();
    for (int i = 0; i < B; ++i) {
        float r = rule[i];
        // clip
        r = std::fmax(-config_.reward_clip, std::fmin(config_.reward_clip, r));
        tot[i] = r;
    }

    if (config_.normalize) {
        RETURN_IF_ERROR(Normalize(&result->total));
    }

    // Initialize other tensors
    result->model_scores = Tensor(Shape({B}), DataType::kFloat32, config_.device);
    result->kl_penalties = Tensor(Shape({B}), DataType::kFloat32, config_.device);
    float* ms = result->model_scores.data<float>();
    float* kl = result->kl_penalties.data<float>();
    for (int i = 0; i < B; ++i) { ms[i] = 0.0f; kl[i] = 0.0f; }

    return Status::Ok();
}

Status RewardEngine::ScoreTokens(const Tensor& token_ids,
                                 const Tensor& masks,
                                 const std::optional<Tensor>& ref_logprobs,
                                 RewardResult* result) {
    if (!initialized_) return Status::InvalidArgument("RewardEngine not initialized");
    if (!result) return Status::InvalidArgument("result pointer is null");

    if (config_.mode == RewardMode::kPerToken) {
        // Per-token: reward is a scalar per token, averaged over valid tokens
        const int B = token_ids.shape()[0];
        const int T = token_ids.shape()[1];
        result->total       = Tensor(Shape({B}), DataType::kFloat32, config_.device);
        result->model_scores = Tensor(Shape({B}), DataType::kFloat32, config_.device);
        result->kl_penalties = Tensor(Shape({B}), DataType::kFloat32, config_.device);
        result->rule_scores  = Tensor(Shape({B}), DataType::kFloat32, config_.device);

        auto* tot = result->total.data<float>();
        auto* ms  = result->model_scores.data<float>();
        auto* kl  = result->kl_penalties.data<float>();
        auto* rs  = result->rule_scores.data<float>();
        for (int i = 0; i < B; ++i) { tot[i] = ms[i] = kl[i] = rs[i] = 0.0f; }

        // KL penalty
        if (ref_logprobs.has_value()) {
            RETURN_IF_ERROR(ComputeKLPenalty(token_ids, *ref_logprobs, masks,
                                             &result->kl_penalties));
        }
    }

    return Status::Ok();
}

Status RewardEngine::UpdateAdaptiveKL(float observed_kl) {
    if (!config_.adaptive_kl) return Status::Ok();
    float delta = config_.kl_adapt_rate * (observed_kl - config_.kl_target);
    current_kl_coef_ = std::max(0.0f, current_kl_coef_ + delta);
    return Status::Ok();
}

Status RewardEngine::ScoreWithRules(const std::vector<std::string>& responses,
                                   Tensor* out) {
    auto* scores = out->data<float>();
    for (size_t i = 0; i < responses.size(); ++i) {
        float s = 0.0f;
        const auto& r = responses[i];
        // Format bonus
        if (!r.empty() && r.back() == '.') s += config_.format_bonus;
        // Length penalty
        int len = static_cast<int>(r.size());
        float ld = static_cast<float>(len - config_.target_length);
        s += config_.length_penalty * ld * ld;
        scores[i] = s;
    }
    return Status::Ok();
}

Status RewardEngine::ComputeKLPenalty(const Tensor& logprobs,
                                     const Tensor& ref_logprobs,
                                     const Tensor& masks,
                                     Tensor* out_kl) {
    // KL(π || π_ref) ≈ mean[log_π - log_π_ref] over valid tokens
    const int B = logprobs.shape()[0];
    const int T = logprobs.shape()[1];
    const int N = B * T;

    auto* lp   = logprobs.data<float>();
    auto* rlp  = ref_logprobs.data<float>();
    auto* mask = masks.data<float>();
    auto* kl   = out_kl->data<float>();

    for (int i = 0; i < B; ++i) {
        float kl_sum = 0.0f, mask_sum = 0.0f;
        for (int j = 0; j < T; ++j) {
            if (mask[i * T + j] > 0.5f) {
                kl_sum   += lp[i * T + j] - rlp[i * T + j];
                mask_sum += 1.0f;
            }
        }
        kl[i] = mask_sum > 0.0f ? (kl_sum / mask_sum) * current_kl_coef_ : 0.0f;
    }
    return Status::Ok();
}

Status RewardEngine::Normalize(Tensor* rewards) {
    // Simple per-batch normalization: subtract batch mean so result has mean ≈ 0.
    // This is the standard approach used in RL (per-batch whitening).
    auto* r = rewards->data<float>();
    const int N = static_cast<int>(rewards->numel());

    float mean = 0.0f;
    for (int i = 0; i < N; ++i) mean += r[i];
    mean /= static_cast<float>(N);
    for (int i = 0; i < N; ++i) r[i] -= mean;
    return Status::Ok();
}

Status RewardEngine::UpdateStats(const Tensor& rewards) {
    // Welford online update for running mean / variance
    auto* r   = rewards.data<float>();
    const int N = static_cast<int>(rewards.numel());
    float  M2  = reward_var_.data<float>()[0];
    float  mu  = reward_mean_.data<float>()[0];
    int64_t n  = reward_count_.data<int64_t>()[0];

    for (int i = 0; i < N; ++i) {
        float x = r[i];
        ++n;
        float delta = x - mu;
        mu += delta / static_cast<float>(n);
        float delta2 = x - mu;
        M2 += delta * delta2;
    }

    reward_mean_.data<float>()[0] = mu;
    reward_var_.data<float>()[0]   = M2 / static_cast<float>(n);
    reward_count_.data<int64_t>()[0] = n;
    ++update_count_;
    return Status::Ok();
}

} // namespace reward
} // namespace turbol

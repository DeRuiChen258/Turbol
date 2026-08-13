#include "turbol/policy/policy_engine.hpp"

#include "turbol/utils/torch_interop.hpp"

#include <cmath>

namespace turborl {
namespace policy {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Copy a contiguous torch tensor into a pre-sized turborl::Tensor living on the
// same device (no dtype/shape conversion — caller guarantees layout parity).
void CopyTorchToTensor(const torch::Tensor& src, Tensor* dst) {
    const size_t bytes = dst->size_bytes();
#ifdef TURBORL_HAS_CUDA
    if (dst->is_cuda())
        cudaMemcpy(dst->data(), src.data_ptr(), bytes, cudaMemcpyDeviceToDevice);
    else
#endif
        std::memcpy(dst->data(), src.data_ptr(), bytes);
}

} // namespace

PolicyEngine::PolicyEngine(int obs_dim, int act_dim, Device device)
    : obs_dim_(obs_dim), act_dim_(act_dim), device_(device) {}

PolicyEngine::~PolicyEngine() = default;

Status PolicyEngine::Initialize() {
#ifdef TURBORL_LIBTORCH
    torch_device_ = device_.is_cuda()
                        ? torch::Device(torch::kCUDA, device_.device_id)
                        : torch::Device(torch::kCPU);

    actor_.emplace(obs_dim_, act_dim_);
    critic_.emplace(obs_dim_, 1);
    (*actor_)->to(*torch_device_);
    (*critic_)->to(*torch_device_);

    auto params = (*actor_)->parameters();
    const auto critic_params = (*critic_)->parameters();
    params.insert(params.end(), critic_params.begin(), critic_params.end());
    optimizer_.emplace(params, torch::optim::AdamOptions(1e-3));

    initialized_ = true;
    return Status::Ok();
#else
    (void)obs_dim_; (void)act_dim_;
    return Status::Unimplemented("PolicyEngine requires libtorch (TURBORL_LIBTORCH)");
#endif
}

Status PolicyEngine::Forward(const Tensor& observation, Tensor* action,
                             Tensor* logprob) {
#ifdef TURBORL_LIBTORCH
    if (!initialized_) return Status::InvalidArgument("PolicyEngine not initialized");
    if (!action || !logprob) return Status::InvalidArgument("null output in Forward");
    if (observation.dtype() != DataType::kFloat32)
        return Status::InvalidArgument("observation must be Float32");
    if (observation.ndim() != 2 || observation.dim(1) != obs_dim_)
        return Status::InvalidArgument("observation must be [N, obs_dim]");

    const int64_t n = observation.dim(0);
    torch::NoGradGuard no_grad;

    torch::Tensor obs = torch_interop::ToTorch(observation).to(*torch_device_);
    torch::Tensor mean = torch::tanh((*actor_)(obs));  // [N, act_dim]

    // Sample a ~ N(mean, σ²), clamp to the valid action range.
    torch::Tensor sample = (mean + std_ * torch::randn_like(mean)).clamp(-1.0f, 1.0f);
    const float var = std_ * std_;
    torch::Tensor lp = (-0.5f * ((sample - mean).pow(2) / var + std::log(2.0f * kPi * var)))
                           .sum(/*dim=*/-1);  // [N]

    *action = Tensor(Shape(std::vector<int64_t>{n, act_dim_}), DataType::kFloat32, device_);
    *logprob = Tensor(Shape(std::vector<int64_t>{n}), DataType::kFloat32, device_);
    CopyTorchToTensor(sample.contiguous(), action);
    CopyTorchToTensor(lp.contiguous(), logprob);
    return Status::Ok();
#else
    (void)observation; (void)action; (void)logprob;
    return Status::Unimplemented("PolicyEngine requires libtorch (TURBORL_LIBTORCH)");
#endif
}

Status PolicyEngine::GetValue(const Tensor& observation, Tensor* value) {
#ifdef TURBORL_LIBTORCH
    if (!initialized_) return Status::InvalidArgument("PolicyEngine not initialized");
    if (!value) return Status::InvalidArgument("null output in GetValue");
    if (observation.dtype() != DataType::kFloat32)
        return Status::InvalidArgument("observation must be Float32");
    if (observation.ndim() != 2 || observation.dim(1) != obs_dim_)
        return Status::InvalidArgument("observation must be [N, obs_dim]");

    const int64_t n = observation.dim(0);
    torch::NoGradGuard no_grad;
    torch::Tensor obs = torch_interop::ToTorch(observation).to(*torch_device_);
    torch::Tensor v = (*critic_)(obs).squeeze(-1);  // [N]

    *value = Tensor(Shape(std::vector<int64_t>{n}), DataType::kFloat32, device_);
    CopyTorchToTensor(v.contiguous(), value);
    return Status::Ok();
#else
    (void)observation; (void)value;
    return Status::Unimplemented("PolicyEngine requires libtorch (TURBORL_LIBTORCH)");
#endif
}

Status PolicyEngine::Update(const Tensor& batch) {
#ifdef TURBORL_LIBTORCH
    if (!initialized_) return Status::InvalidArgument("PolicyEngine not initialized");
    if (batch.dtype() != DataType::kFloat32)
        return Status::InvalidArgument("batch must be Float32");

    const int64_t row = static_cast<int64_t>(obs_dim_) + act_dim_ + 2;  // obs|action|adv|ret
    if (batch.numel() % row != 0)
        return Status::InvalidArgument("batch size must be a multiple of obs+act+adv+ret");
    const int64_t n = batch.numel() / row;

    torch::Tensor b = torch_interop::ToTorch(batch).to(*torch_device_).reshape({n, row});
    torch::Tensor obs = b.narrow(/*dim=*/1, 0, obs_dim_);
    torch::Tensor act = b.narrow(1, obs_dim_, act_dim_);
    torch::Tensor adv = b.narrow(1, obs_dim_ + act_dim_, 1).squeeze(1);
    torch::Tensor ret = b.narrow(1, obs_dim_ + act_dim_ + 1, 1).squeeze(1);

    torch::Tensor mean = torch::tanh((*actor_)(obs));
    const float var = std_ * std_;
    torch::Tensor logp = (-0.5f * ((act - mean).pow(2) / var + std::log(2.0f * kPi * var)))
                             .sum(/*dim=*/-1);  // [N]

    torch::Tensor value = (*critic_)(obs).squeeze(-1);  // [N]
    torch::Tensor policy_loss = -(adv * logp).mean();
    torch::Tensor value_loss = torch::mse_loss(value, ret);
    torch::Tensor loss = policy_loss + 0.5f * value_loss;

    optimizer_->zero_grad();
    loss.backward();
    optimizer_->step();
    return Status::Ok();
#else
    (void)batch;
    return Status::Unimplemented("PolicyEngine requires libtorch (TURBORL_LIBTORCH)");
#endif
}

} // namespace policy
} // namespace turborl

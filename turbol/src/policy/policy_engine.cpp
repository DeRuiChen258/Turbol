#include "turbol/policy/policy_engine.hpp"

namespace turborl {
namespace policy {

PolicyEngine::PolicyEngine() {}
PolicyEngine::~PolicyEngine() {}

Status PolicyEngine::Initialize() {
    return Status::Ok();
}

Status PolicyEngine::Forward(const Tensor& observation, Tensor* action, Tensor* logprob) {
    return Status::Ok();
}

Status PolicyEngine::Update(const Tensor& batch) {
    return Status::Ok();
}

} // namespace policy
} // namespace turbol

#include "turbol/rollout/rollout_engine.hpp"

namespace turborl {
namespace rollout {

RolloutEngine::RolloutEngine(const RolloutConfig& config)
    : config_(config), initialized_(false) {}

RolloutEngine::~RolloutEngine() {
    Shutdown();
}

Status RolloutEngine::Initialize() {
    if (config_.inference_backend == "vllm") {
        backend_ = std::make_unique<VLLMBackend>(
            config_.model_path.empty() ? "/models/llama" : config_.model_path,
            config_.tensor_parallel_size);
        
        auto status = backend_->Initialize();
        if (!status.ok()) {
            return status;
        }
    }
    initialized_ = true;
    return Status::Ok();
}

Status RolloutEngine::Shutdown() {
    backend_.reset();
    initialized_ = false;
    return Status::Ok();
}

Status RolloutEngine::Generate(const std::string& prompt, std::string* output) {
    if (!initialized_) return Status::InvalidArgument("RolloutEngine not initialized");
    if (!output) return Status::InvalidArgument("Output pointer is null");
    
    if (backend_ && backend_->IsAvailable()) {
        return backend_->Generate(prompt, output);
    }
    
    // Fallback to simple echo
    *output = "[RolloutEngine] " + prompt;
    return Status::Ok();
}

Status RolloutEngine::GenerateBatch(const std::vector<std::string>& prompts,
                                     std::vector<std::string>* outputs) {
    if (!initialized_) return Status::InvalidArgument("RolloutEngine not initialized");
    if (!outputs) return Status::InvalidArgument("Outputs pointer is null");
    
    outputs->clear();
    for (const auto& p : prompts) {
        std::string out;
        auto status = Generate(p, &out);
        if (!status.ok()) return status;
        outputs->push_back(out);
    }
    return Status::Ok();
}

bool RolloutEngine::IsVLLMAvailable() const {
    return backend_ && backend_->IsAvailable();
}

} // namespace rollout
} // namespace turbol

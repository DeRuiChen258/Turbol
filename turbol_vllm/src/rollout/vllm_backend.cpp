#include "turbol/rollout/vllm_backend.hpp"

namespace turborl {
namespace rollout {

VLLMBackend::VLLMBackend(const std::string& model_path, int tensor_parallel_size)
    : model_path_(model_path), tensor_parallel_size_(tensor_parallel_size) {
    available_ = false;
#ifdef TURBORL_VLLM_ENABLED
    // Only claim vLLM availability when a real model directory and a CUDA
    // device are both present; otherwise we degrade to the CPU echo fallback.
    if (!model_path_.empty() && std::filesystem::exists(model_path_)) {
#ifdef TURBORL_HAS_CUDA
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
            available_ = true;
#endif
    }
#endif
    (void)tensor_parallel_size;
}

VLLMBackend::~VLLMBackend() {
    Shutdown();
}

Status VLLMBackend::Initialize() {
    if (!available_) {
        // CPU fallback mode
        initialized_ = true;
        return Status::Ok();
    }
    
#ifdef TURBORL_VLLM_ENABLED
    try {
        vllm::EngineConfig config;
        config.model_path = model_path_;
        engine_ = std::make_unique<vllm::Engine>(config);
        
        if (!engine_->initialize()) {
            return Status::InternalError("vLLM engine initialization failed");
        }
        
        initialized_ = true;
        return Status::Ok();
    } catch (const std::exception& e) {
        return Status::InternalError(std::string("vLLM init exception: ") + e.what());
    }
#else
    initialized_ = true;
    return Status::Ok();
#endif
}

Status VLLMBackend::Shutdown() {
#ifdef TURBORL_VLLM_ENABLED
    engine_.reset();
#endif
    initialized_ = false;
    return Status::Ok();
}

Status VLLMBackend::Generate(const std::string& prompt, std::string* output) {
    if (!initialized_) return Status::InvalidArgument("VLLMBackend not initialized");
    if (!output) return Status::InvalidArgument("Output pointer is null");

#ifdef TURBORL_VLLM_ENABLED
    if (available_ && engine_) {
        try {
            engine_->add_request(prompt, max_output_len_, temperature_, top_p_, top_k_);

            // Collect generated token ids across the streaming callbacks, then
            // decode the full sequence once (single-token decodes can mangle
            // sentencepiece pieces, so we never concatenate per-token text).
            std::vector<vllm::TokenId> tokens;
            engine_->run([&tokens](const vllm::GenerationOutput& g) {
                if (!g.tokens.empty())
                    tokens.insert(tokens.end(), g.tokens.begin(), g.tokens.end());
            });

            *output = engine_->decode(tokens);
            return Status::Ok();
        } catch (const std::exception& e) {
            return Status::InternalError(std::string("vLLM generate exception: ") + e.what());
        }
    }
#endif

    // CPU fallback (no vLLM library / no CUDA device).
    *output = "[RolloutEngine] " + prompt;
    return Status::Ok();
}

Status VLLMBackend::GenerateBatch(const std::vector<std::string>& prompts,
                                   std::vector<std::string>* outputs) {
    if (!initialized_) return Status::InvalidArgument("VLLMBackend not initialized");
    if (!outputs) return Status::InvalidArgument("Outputs pointer is null");
    
    outputs->clear();
    for (const auto& prompt : prompts) {
        std::string out;
        auto status = Generate(prompt, &out);
        if (!status.ok()) return status;
        outputs->push_back(out);
    }
    return Status::Ok();
}

} // namespace rollout
} // namespace turbol

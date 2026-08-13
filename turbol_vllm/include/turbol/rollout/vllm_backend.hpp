#pragma once
#include "../common.hpp"

#ifdef TURBORL_VLLM_ENABLED
#include "vllm/vllm.hpp"
#endif

namespace turborl {
namespace rollout {

class VLLMBackend {
public:
    VLLMBackend(const std::string& model_path, int tensor_parallel_size = 1);
    ~VLLMBackend();
    
    Status Initialize();
    Status Shutdown();
    
    Status Generate(const std::string& prompt, std::string* output);
    Status GenerateBatch(const std::vector<std::string>& prompts, 
                         std::vector<std::string>* outputs);
    
    bool IsAvailable() const { return available_; }

private:
    std::string model_path_;
    int tensor_parallel_size_;
    bool available_ = false;
    bool initialized_ = false;

    // Sampling defaults forwarded to the vLLM engine.
    size_t max_output_len_ = 128;
    float temperature_ = 1.0f;
    float top_p_ = 1.0f;
    int32_t top_k_ = -1;

#ifdef TURBORL_VLLM_ENABLED
    std::unique_ptr<vllm::Engine> engine_;
#endif
};

} // namespace rollout
} // namespace turbol

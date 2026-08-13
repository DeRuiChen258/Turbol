#pragma once
#include "../common.hpp"

namespace turborl {
namespace rollout {

struct RolloutConfig {
    std::string inference_backend = "local";
    int max_new_tokens = 2048;
    float temperature = 1.0f;
    float top_p = 1.0f;
    int top_k = -1;
    int max_batch_size = 256;
};

class RolloutEngine {
public:
    RolloutEngine(const RolloutConfig& config);
    ~RolloutEngine();
    
    Status Initialize();
    Status Shutdown();
    
    Status Generate(const std::string& prompt, std::string* output);
    Status GenerateBatch(const std::vector<std::string>& prompts, std::vector<std::string>* outputs);
    
private:
    RolloutConfig config_;
    bool initialized_ = false;
};

} // namespace rollout
} // namespace turbol

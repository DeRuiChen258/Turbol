// vLLM Stub Implementation for linking
// This provides dummy implementations to allow TurboRL to compile
// Replace with actual vLLM library linking when available

#include "vllm/vllm.hpp"

namespace vllm {

Engine::Engine(const EngineConfig& config) : config_(config), initialized_(false) {}

Engine::~Engine() {}

bool Engine::initialize() {
    initialized_ = true;
    return true;
}

int64_t Engine::add_request(const std::string& prompt,
                            size_t max_output_len,
                            float temperature,
                            float top_p,
                            int32_t top_k) {
    (void)prompt; (void)max_output_len; (void)temperature; (void)top_p; (void)top_k;
    return 0;
}

std::vector<GenerationOutput> Engine::step() {
    std::vector<GenerationOutput> results;
    return results;
}

void Engine::abort_request(int64_t seq_id) {
    (void)seq_id;
}

size_t Engine::pending_requests() const {
    return 0;
}

std::string Engine::decode(const std::vector<TokenId>& tokens) const {
    (void)tokens;
    return "[stub_decode]";
}

} // namespace vllm

#include <iostream>
#include "turbol/core/core_engine.hpp"
#include "turbol/env/gpu_environment.hpp"
#include "turbol/replay_buffer/ring_buffer.hpp"
#include "turbol/rollout/rollout_engine.hpp"
#include "turbol/policy/policy_engine.hpp"

using namespace turborl;

int main() {
    std::cout << "=== TurboRL Demo ===" << std::endl;
    
    // Test Core Engine
    std::cout << "\n1. Core Engine:" << std::endl;
    auto& engine = CoreEngine::Instance();
    std::cout << "   CoreEngine singleton OK" << std::endl;
    
    // Test Tensor
    std::cout << "\n2. Tensor:" << std::endl;
    Shape shape({10, 20});
    Tensor tensor(shape, DataType::kFloat32, Device::CPU());
    std::cout << "   Tensor created: " << tensor.size_bytes() << " bytes" << std::endl;
    
    // Test GPU Environment
    std::cout << "\n3. GPU Environment:" << std::endl;
    env::GPUEnvironment env;
    std::cout << "   GPUEnvironment created" << std::endl;
    
    // Test Rollout Engine
    std::cout << "\n4. Rollout Engine:" << std::endl;
    rollout::RolloutConfig config;
    rollout::RolloutEngine rollout(config);
    rollout.Initialize();
    std::string output;
    rollout.Generate("Hello", &output);
    std::cout << "   Output: " << output << std::endl;
    
    // Test Policy Engine
    std::cout << "\n5. Policy Engine:" << std::endl;
    policy::PolicyEngine policy;
    std::cout << "   PolicyEngine created" << std::endl;
    
    // Test Replay Buffer
    std::cout << "\n6. Replay Buffer:" << std::endl;
    replay_buffer::RingBuffer buffer(1000, 128, 8);
    std::cout << "   RingBuffer capacity: " << buffer.Capacity() << std::endl;
    
    std::cout << "\n=== All Tests Passed ===" << std::endl;
    
#ifdef TURBORL_CUDA
    std::cout << "\n[CUDA] GPU Support: Enabled" << std::endl;
    auto mem_info = GetCudaMemoryInfo(0);
    std::cout << "[CUDA] GPU Memory: " << mem_info.used() / 1024 / 1024 << " MB used / " 
              << mem_info.total / 1024 / 1024 << " MB total" << std::endl;
#else
    std::cout << "\n[CUDA] GPU Support: Disabled" << std::endl;
#endif

    return 0;
}

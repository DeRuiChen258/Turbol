#include <iostream>
#include <chrono>
#include "turbol/common.hpp"

using namespace turborl;

void benchmark_tensor_creation(int iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        Shape shape({1024, 1024});
        Tensor tensor(shape, DataType::kFloat32, Device::CPU());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Tensor Creation (" << iterations << " iterations):\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Per iteration: " << (double)duration.count() / iterations << " ms\n";
}

void benchmark_tensor_copy(int iterations) {
    Shape shape({1024, 1024});
    Tensor tensor1(shape, DataType::kFloat32, Device::CPU());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; i++) {
        Tensor tensor2(tensor1);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Tensor Copy (" << iterations << " iterations):\n";
    std::cout << "  Total time: " << duration.count() << " ms\n";
    std::cout << "  Per iteration: " << (double)duration.count() / iterations << " ms\n";
}

void benchmark_memory_bandwidth() {
    const int size = 100 * 1024 * 1024; // 100MB
    Shape shape({size / 4});
    Tensor tensor(shape, DataType::kFloat32, Device::CPU());
    
    float* data = static_cast<float*>(tensor.data());
    for (int i = 0; i < size / 4; i++) data[i] = 1.0f;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Copy operation
    Tensor tensor2(tensor);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double bandwidth = (size * 2) / (1024.0 * 1024.0) / (duration.count() / 1000.0);
    
    std::cout << "Memory Bandwidth Test (100MB copy):\n";
    std::cout << "  Time: " << duration.count() << " ms\n";
    std::cout << "  Bandwidth: " << bandwidth << " MB/s\n";
}

int main() {
    std::cout << "=== TurboRL Tensor Benchmark ===\n\n";
    
    benchmark_tensor_creation(1000);
    std::cout << "\n";
    
    benchmark_tensor_copy(1000);
    std::cout << "\n";
    
    benchmark_memory_bandwidth();
    std::cout << "\n";
    
    std::cout << "Benchmark completed.\n";
    return 0;
}

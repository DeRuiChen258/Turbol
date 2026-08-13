#include <iostream>
#include <chrono>
#include <cstdlib>

#ifdef TURBORL_CUDA
#include <cuda_runtime.h>
#include "turbol/common.hpp"

using namespace turborl;

// Check CUDA availability
bool check_cuda() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB\n";
    return true;
}

void benchmark_attention_kernel(int batch_size, int num_heads, int seq_len, int head_dim) {
    std::cout << "Attention Kernel Benchmark:\n";
    std::cout << "  Batch: " << batch_size << ", Heads: " << num_heads 
              << ", SeqLen: " << seq_len << ", HeadDim: " << head_dim << "\n";
    
    size_t q_size = batch_size * num_heads * seq_len * head_dim;
    size_t bytes = q_size * 4 * 3; // Q, K, V
    
    std::cout << "  Total memory: " << bytes / (1024*1024) << " MB\n";
    std::cout << "  Note: CUDA attention kernel compilation would require\n";
    std::cout << "        linking against actual Flash Attention library\n";
}
#else
void benchmark_attention_kernel(int, int, int, int) {
    std::cout << "Attention Benchmark: CUDA not enabled\n";
}
#endif

void benchmark_attention_cpu(int batch_size, int num_heads, int seq_len, int head_dim) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate attention computation
    int iterations = 10;
    for (int iter = 0; iter < iterations; iter++) {
        // QK^T multiplication would happen here
        volatile double sum = 0;
        for (int b = 0; b < batch_size; b++) {
            for (int h = 0; h < num_heads; h++) {
                for (int i = 0; i < seq_len; i++) {
                    for (int j = 0; j < seq_len; j++) {
                        sum += 0.001; // Placeholder computation
                    }
                }
            }
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "Attention CPU Benchmark:\n";
    std::cout << "  Batch: " << batch_size << ", Heads: " << num_heads 
              << ", SeqLen: " << seq_len << ", HeadDim: " << head_dim << "\n";
    std::cout << "  Time (" << iterations << " iterations): " << duration.count() << " ms\n";
}

int main() {
    std::cout << "=== TurboRL Attention Benchmark ===\n\n";
    
#ifdef TURBORL_CUDA
    if (check_cuda()) {
        benchmark_attention_kernel(4, 8, 512, 64);
        std::cout << "\n";
    }
#endif
    
    benchmark_attention_cpu(1, 4, 128, 64);
    std::cout << "\n";
    
    std::cout << "Benchmark completed.\n";
    return 0;
}

# TurboRL Full - CUDA + vLLM + libtorch Version

The full-featured version of TurboRL with CUDA acceleration, vLLM integration, and libtorch support.

## Overview

This is the production-ready version of TurboRL with:

- CUDA GPU acceleration
- vLLM LLM inference integration
- libtorch support for neural network operations
- Flash Attention CUDA kernels
- Comprehensive testing and benchmarks

## Features

- **CUDA Acceleration**: GPU-native tensor operations and memory management
- **vLLM Integration**: High-performance LLM inference backend
- **libtorch Support**: PyTorch C++ API for model operations
- **Flash Attention**: Optimized CUDA attention kernels
- **NCCL Ready**: Multi-GPU distributed training support
- **Full Testing**: Unit tests and performance benchmarks

## Requirements

### Required

- CMake 3.28+
- C++17 compiler (GCC 9+, Clang 10+)
- CUDA Toolkit 13.x
- NVIDIA Driver 535+
- NVIDIA GPU (Compute Capability 7.0+)

### Optional

- [libtorch](https://pytorch.org/cppdocs/installing.html) (for neural network support)
- [vLLM](https://github.com/vllm-project/vllm) (for LLM inference)

## Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure with all features
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DTURBORL_ENABLE_CUDA=ON \
    -DTURBORL_USE_LIBTORCH=ON \
    -DTURBORL_ENABLE_TESTS=ON \
    -DTURBORL_ENABLE_BENCHMARKS=ON

# Build
cmake --build . -j$(nproc)
```

### Build Options

| Option                      | Description                          | Default |
| --------------------------- | ------------------------------------ | ------- |
| `TURBORL_ENABLE_CUDA`       | Enable CUDA support                  | ON      |
| `TURBORL_USE_NCCL`          | Enable NCCL distributed training     | OFF     |
| `TURBORL_USE_LIBTORCH`      | Enable libtorch support              | OFF     |
| `TURBORL_ENABLE_TESTS`      | Build unit tests                     | ON      |
| `TURBORL_ENABLE_BENCHMARKS` | Build benchmarks                     | ON      |
| `TORCH_CUDA_ARCH_LIST`      | CUDA architectures (e.g., "8.0;9.0") | -       |

## Run

```bash
# Run demo
./turbol_example

# Run unit tests
./turbol_test

# Run benchmarks
./benchmark_tensor
./benchmark_attention
```

## Project Structure

```
turbol_vllm/
├── CMakeLists.txt           # Build configuration with all options
├── include/turbol/          # Public headers
│   ├── common.hpp          # Core types (Tensor, Status, Device)
│   ├── core/               # Core engine
│   ├── cuda/               # CUDA kernels
│   │   ├── attention.cu    # Flash Attention
│   │   ├── ring_buffer.cu  # GPU ring buffer
│   │   └── math_utils.cu   # CUDA math utilities
│   ├── env/                # GPU environment
│   ├── policy/             # Policy engine
│   ├── replay_buffer/      # Ring buffer implementation
│   ├── rollout/            # Rollout engine + vLLM backend
│   │   ├── rollout_engine.hpp
│   │   ├── rollout_engine.cpp
│   │   ├── vllm_backend.hpp
│   │   └── vllm_backend.cpp
│   ├── distributed/        # Distributed training
│   │   ├── nccl_communicator.hpp
│   │   └── distributed_trainer.hpp
│   └── utils/             # Utility functions
├── src/                    # Implementation files
├── tests/                  # GoogleTest unit tests
│   ├── test_tensor.cpp
│   ├── test_ring_buffer.cpp
│   └── test_rollout.cpp
├── benchmarks/             # Performance benchmarks
│   ├── benchmark_tensor.cpp
│   └── benchmark_attention.cpp
└── examples/              # Example programs
```

## Testing

### Unit Tests

```bash
./turbol_test
```

Expected output:

```
[==========] Running 15 tests from 3 test suites.
[  PASSED  ] 15 tests.
```

Test suites:

- `TensorTest`: Tensor creation, copy, move, data types, devices
- `RingBufferTest`: Replay buffer operations
- `RolloutTest`: Rollout engine initialization and generation

### Performance Benchmarks

```bash
# Tensor operations
./benchmark_tensor

# Attention operations
./benchmark_attention
```

Sample output:

```
=== TurboRL Tensor Benchmark ===

Tensor Creation (1000 iterations):
  Total time: 0 ms
  Per iteration: 0 ms

Tensor Copy (1000 iterations):
  Total time: 57 ms
  Per iteration: 0.057 ms

Memory Bandwidth Test (100MB copy):
  Time: 18 ms
  Bandwidth: 11111.1 MB/s
```

## vLLM Integration

The vLLM backend provides high-performance LLM inference:

```cpp
#include "turbol/rollout/rollout_engine.hpp"

using namespace turborl::rollout;

// Create engine with vLLM backend
RolloutConfig config;
config.inference_backend = "vllm";
config.model_path = "/path/to/model";
config.max_new_tokens = 256;

RolloutEngine engine(config);
engine.Initialize();

// Generate text
std::string output;
engine.Generate("Hello, how are you?", &output);
```

## CUDA Configuration

### GPU Architecture

Set `TORCH_CUDA_ARCH_LIST` to match your GPU:

| GPU Series | Compute Capability |
| ---------- | ------------------ |
| RTX 30xx   | SM_86              |
| RTX 40xx   | SM_90              |
| A100       | SM_80              |
| H100       | SM_90              |

Example:

```bash
cmake .. -DTORCH_CUDA_ARCH_LIST="8.6;9.0" ...
```

### Multi-GPU Training

Enable NCCL for distributed training:

```bash
cmake .. -DTURBORL_USE_NCCL=ON -DTURBORL_ENABLE_CUDA=ON
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TurboRL Core                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │
│  │ Core    │  │ Policy   │  │ Rollout  │  │  Replay     │  │
│  │ Engine  │  │ Engine   │  │ Engine   │  │  Buffer     │  │
│  └─────────┘  └──────────┘  └──────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                    GPU Acceleration                          │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   Tensor     │  │ Flash Attention │  │ Ring Buffer    │   │
│  │   (CUDA)     │  │   (CUDA)       │  │   (CUDA)      │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                    Backend Integrations                     │
│  ┌──────────────┐  ┌────────────────┐  ┌────────────────┐   │
│  │   vLLM       │  │    libtorch     │  │     NCCL       │   │
│  │  (Inference)  │  │ (Neural Nets)   │  │(Distributed)   │   │
│  └──────────────┘  └────────────────┘  └────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Performance Optimization Tips

1. **Memory Pool**: Pre-allocate CUDA memory to reduce allocation overhead
2. **Batch Processing**: Use `GenerateBatch()` for multiple prompts
3. **Paged Attention**: vLLM's KV cache management reduces memory fragmentation
4. **Tensor Parallelism**: For large models, enable multi-GPU inference

## Limitations

- CUDA driver must be installed and functional
- GPU memory requirements depend on model size
- vLLM integration requires compatible model weights

## License

Apache License 2.0

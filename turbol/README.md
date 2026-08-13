# TurboRL Original - CPU Version

A minimal, dependency-free version of TurboRL for basic RL development and testing.

## Overview

This is the lightweight CPU-only version of TurboRL. It provides the core infrastructure with no external dependencies beyond standard C++ libraries.

## Features

- **Zero External Dependencies**: Only requires CMake and a C++17 compiler
- **Core Infrastructure**: Tensor operations, replay buffer, rollout engine
- **Modular Design**: Easy to extend and customize
- **CI/CD Friendly**: Simple build process, no GPU required

## Requirements

- CMake 3.28+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Threads (pthread, standard on Linux/macOS)

## Build

```bash
# Create build directory
mkdir -p build && cd build

# Configure (CPU-only, no external dependencies)
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DTURBORL_ENABLE_CUDA=OFF \
    -DTURBORL_ENABLE_TESTS=ON \
    -DTURBORL_ENABLE_BENCHMARKS=ON

# Build
cmake --build . -j$(nproc)
```

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

## Build Options

| Option                      | Description      | Default |
| --------------------------- | ---------------- | ------- |
| `TURBORL_ENABLE_TESTS`      | Build unit tests | ON      |
| `TURBORL_ENABLE_BENCHMARKS` | Build benchmarks | ON      |

## Project Structure

```
turbol/
├── CMakeLists.txt           # Build configuration
├── include/turbol/          # Public headers
│   ├── common.hpp          # Core types (Tensor, Status, Device)
│   ├── core/               # Core engine
│   ├── env/                # GPU environment (CPU simulation)
│   ├── policy/             # Policy engine
│   ├── replay_buffer/      # Ring buffer implementation
│   ├── rollout/            # Rollout engine
│   ├── distributed/         # Distributed training (stub)
│   └── utils/              # Utility functions
├── src/                    # Implementation files
├── tests/                  # GoogleTest unit tests
├── benchmarks/             # Performance benchmarks
└── examples/               # Example programs
```

## Testing

The CPU version includes comprehensive unit tests:

```bash
./turbol_test
```

Expected output:

```
[==========] Running 15 tests from 3 test suites.
[  PASSED  ] 15 tests.
```

## Benchmarks

```bash
# Tensor operations benchmark
./benchmark_tensor

# Attention operations benchmark  
./benchmark_attention
```

## Use Cases

1. **Algorithm Development**: Quickly prototype RL algorithms without GPU overhead
2. **CI/CD Pipelines**: Fast builds and tests in continuous integration
3. **Educational**: Learn RL infrastructure concepts
4. **Minimal Deployment**: Deploy to CPU-only environments

## Limitations

- No CUDA acceleration
- No GPU memory optimization
- No vLLM inference integration
- No multi-GPU distributed training

For full-featured version, see [turbol_vllm](../turbol_vllm/).

## License

Apache License 2.0

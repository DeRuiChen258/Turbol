# TurboRL — Minimal CPU Version

TurboRL 的**最小 CPU 版**：零外部依赖（除测试用 GoogleTest），用于算法原型、CI 与教学。
完整功能（CUDA 内核、libtorch 策略网络、vLLM 推理、Python API）见
[`../turbol_vllm/`](../turbol_vllm/) 与[根 README](../README.md)。

## 特性

- **零外部依赖**：只需要 CMake 与 C++17 编译器，无需 CUDA / GPU / libtorch
- **核心基础设施**：张量、环形回放缓冲、向量化环境、rollout 引擎、配置系统
- **CI 友好**：秒级构建，可在无 GPU runner 上运行

## 环境要求

- CMake ≥ 3.28
- C++17 编译器（GCC 9+ / Clang 10+ / MSVC 2019+）
- pthread
- 单元测试需要 GoogleTest（`libgtest-dev`）

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DTURBORL_ENABLE_CUDA=OFF \
      -DTURBORL_ENABLE_TESTS=ON \
      -DTURBORL_ENABLE_BENCHMARKS=ON
cmake --build build -j"$(nproc)"
```

## 运行

```bash
./build/turbol_example                          # 演示程序
ctest --test-dir build --output-on-failure      # 单元测试
./build/turbol_test                             # 或直接运行测试二进制
./build/benchmark_tensor                        # 张量算子基准
```

测试通过时的输出：

```text
[==========] 17 tests from 3 test suites ran.
[  PASSED  ] 17 tests.
```

## 构建选项

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `TURBORL_ENABLE_CUDA` | 启用 CUDA（本版本无 CUDA 实现，保持 OFF） | `OFF` |
| `TURBORL_ENABLE_TESTS` | 构建单元测试 | `ON` |
| `TURBORL_ENABLE_BENCHMARKS` | 构建基准程序 | `ON` |

## 目录结构

```text
turbol/
├── CMakeLists.txt
├── include/turbol/
│   ├── common.hpp                   # Status / Shape / Device / DataType / Tensor
│   ├── core/                        # CoreEngine、Config
│   ├── env/                         # GPUEnvironment（本版本为 CPU 实现）
│   ├── policy/                      # PolicyEngine（轻量实现）
│   ├── replay_buffer/               # RingBuffer（定容环形缓冲）
│   ├── rollout/                     # RolloutEngine
│   ├── distributed/                 # 分布式（占位实现）
│   └── utils/                       # 工具函数
├── src/                             # 与 include 对应的实现
├── tests/                           # GoogleTest 用例（17 个）
├── benchmarks/                      # 基准程序
└── examples/                        # 示例程序
```

## RingBuffer 语义

- 存储布局：`observations [capacity, obs_dim]`、`actions [capacity, act_dim]`、
  `rewards [capacity]`、`dones [capacity]`
- `Push` 为 O(1) 槽位写入；写满后按环形覆盖最旧样本，`Size()` 上限为 `Capacity()`
- `Sample` 在 `[0, Size())` 上**有放回**均匀采样，输出张量形状为 `[batch, dim]`

## 限制

- 无 CUDA 加速、无 GPU 显存优化
- 无 vLLM 推理集成、无多卡分布式训练
- 无策略网络训练（libtorch）、无 Profiler / RewardEngine / Dataset

需要完整能力请使用 [`../turbol_vllm/`](../turbol_vllm/)。

## 许可证

Apache License 2.0，见 [`../LICENSE`](../LICENSE)。

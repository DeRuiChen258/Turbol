# TurboRL

由 CUDA 和 C++ 驱动的高性能强化学习与 RLHF 基础设施

## 概述

TurboRL 是一个工业级、高性能的强化学习（RL）和 RLHF（基于人类反馈的强化学习）基础设施项目。它为整个 RLHF 流程提供 GPU 原生解决方案。

## 特性

- **GPU 原生**：全流程 GPU 加速 — 环境、回放缓冲区、 rollout、奖励、策略
- **LLM 集成**：与 vLLM 原生集成，实现高性能 LLM 推理
- **分布式训练**：基于 NCCL 的多 GPU 训练支持
- **模块化设计**：松耦合架构，模块可互换
- **Flash Attention**：CUDA 优化的注意力内核
- **C++17 + CUDA 13.x**：现代 C++ 配合最新 CUDA 功能
- **全面测试**：包含单元测试和基准测试

## 项目结构

```
TurboRL/
├── TurboRL_Specification.md    # 详细规格说明文档
├── turbol/                     # 原版（仅 CPU，最小依赖）
│   ├── CMakeLists.txt
│   ├── include/turbol/        # 头文件
│   ├── src/                   # 源代码
│   ├── tests/                 # 单元测试
│   ├── benchmarks/            # 性能基准测试
│   └── examples/              # 示例程序
│
├── turbol_vllm/               # 完整版（CUDA + vLLM + libtorch）
│   ├── CMakeLists.txt
│   ├── include/turbol/        # 头文件
│   ├── src/                  # 源代码
│   ├── tests/                 # 单元测试
│   ├── benchmarks/           # 性能基准测试
│   └── examples/             # 示例程序
│
└── README.md                  # 本文件
```

## 快速开始

### 构建要求

- CMake 3.28+
- C++17 兼容编译器（GCC 9+、Clang 10+）
- CUDA Toolkit 13.x（CUDA 版本）
- NVIDIA GPU，计算能力 7.0+（CUDA 版本）

### 构建说明

#### 原版（仅 CPU）

```bash
cd turbol
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DTURBORL_ENABLE_CUDA=OFF
cmake --build . -j$(nproc)
./turbol_example
```

#### 完整版（CUDA + vLLM + libtorch）

```bash
cd turbol_vllm
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DTURBORL_ENABLE_CUDA=ON \
    -DTURBORL_USE_LIBTORCH=ON \
    -DTURBORL_ENABLE_TESTS=ON \
    -DTURBORL_ENABLE_BENCHMARKS=ON
cmake --build . -j$(nproc)
./turbol_example
```

### 依赖项

#### 原版

- CMake 3.28+
- C++17 编译器
- Threads（pthread）

#### 完整版

- CMake 3.28+
- C++17 编译器
- CUDA Toolkit 13.x
- NVIDIA 驱动
- [libtorch](https://pytorch.org/)（可选）
- [vLLM](https://github.com/vllm-project/vllm)（可选）

## 使用方法

### 运行测试

```bash
cd turbol_vllm/build

# 单元测试
./turbol_test

# 张量基准测试
./benchmark_tensor

# 注意力基准测试
./benchmark_attention
```

### 示例输出

```
=== TurboRL 演示 ===

1. 核心引擎：
   CoreEngine 单例 OK

2. 张量：
   张量已创建：800 字节

3. GPU 环境：
   GPUEnvironment 已创建

4. Rollout 引擎：
   输出：[RolloutEngine] Hello

5. 策略引擎：
   PolicyEngine 已创建

6. 回放缓冲区：
   RingBuffer 容量：1000

=== 所有测试通过 ===

[CUDA] GPU 支持：已启用
```

## 技术规格

| 规格                     | 值                     |
| ------------------------ | ---------------------- |
| C++ 标准                 | C++17                  |
| CUDA 版本                | 13.x                   |
| 计算能力                 | SM_90（RTX 40xx）      |
| 分布式通信               | NCCL 2.21+（可选）     |
| 构建系统                 | CMake 3.28+            |
| 测试框架                 | GoogleTest             |

## 核心模块

| 模块               | 描述                                     |
| ------------------ | ---------------------------------------- |
| 核心引擎           | 全局状态管理和同步                       |
| 张量               | 统一 CPU/CUDA 张量接口                   |
| Rollout 引擎       | LLM 推理编排（vLLM 集成）                |
| 策略引擎           | 策略网络管理                             |
| 回放缓冲区         | 基于环形缓冲区的经验回放                 |
| GPU 环境           | GPU 加速的环境模拟                       |
| NCCL 通信器        | 多 GPU 通信                              |
| Flash Attention    | CUDA 优化的注意力内核                    |

## 构建选项

| 选项                       | 描述                         | 默认值 |
| -------------------------- | ---------------------------- | ------ |
| `TURBORL_ENABLE_CUDA`      | 启用 CUDA 支持               | ON     |
| `TURBORL_USE_NCCL`         | 启用 NCCL 分布式训练         | OFF    |
| `TURBORL_USE_LIBTORCH`     | 启用 libtorch 支持           | OFF    |
| `TURBORL_ENABLE_TESTS`     | 启用单元测试                 | ON     |
| `TURBORL_ENABLE_BENCHMARKS` | 启用基准测试                 | ON     |

## 性能基准

```
张量创建：     ~0.057 毫秒/次（100 万元素）
张量复制：     ~57 毫秒/1000 次（100 万元素）
内存带宽：     ~11 GB/s
注意力：       ~1 毫秒/10 次（CPU 回退）
```

## 路线图

- [ ] 完成 vLLM 集成，支持真实模型加载
- [ ] 启用 NCCL 多 GPU 训练
- [ ] 集成 Flash Attention 2/3
- [ ] 添加流水线并行
- [ ] Python API（pybind11）
- [ ] TensorRT-LLM 后端支持
- [ ] SGLang 后端支持

## 贡献

欢迎贡献！请随时提交 Pull Request。

## 许可证

Apache License 2.0

## 引用

如果您在研究中使用 TurboRL，请引用：

```bibtex
@software{turborl,
  title = {TurboRL: High Performance RL & RLHF Infrastructure},
  author = {TurboRL Team},
  year = {2026},
  version = {1.0.0}
}
```

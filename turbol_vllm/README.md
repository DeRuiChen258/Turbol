# TurboRL — Full Version (CUDA + libtorch + vLLM)

TurboRL 的**完整版主开发目录**：CUDA 内核 + libtorch 策略网络 + vLLM 推理 +
NCCL 集合通信 + pybind11 Python 绑定。

项目总览、架构图、基准数据与路线图见[根 README](../README.md)，
详细设计见 [`../docs/SPECIFICATION.md`](../docs/SPECIFICATION.md)。

## 能力一览

- **CUDA 内核**：FlashAttention（在线 softmax）、环形缓冲写入、元素级算子、环境动力学、GAE
- **GPU 环境**：`GPUEnvironment` 向量化 step/observe，状态驻留显存
- **经验回放**：`RingBuffer` 定容环形缓冲，CPU/CUDA 双路径，O(1) 写入 + 均匀采样
- **策略网络**：`PolicyEngine` 基于 libtorch（高斯 actor + critic + Adam）
- **RLHF 组件**：`RewardEngine`（规则打分 + KL 惩罚 + 自适应 KL）、`Dataset`（流式 JSONL）
- **推理后端**：`VLLMBackend` 接入 vLLM-CPP；Python 侧可用 vLLM Python API
- **分布式**：`DistributedTrainer`（NCCL AllReduce/Broadcast/AllGather）、`PipelineTrainer`（GPipe 微批）
- **可观测性**：`Profiler`（CPU/CUDA 计时、Chrome Trace）+ TensorBoard / Prometheus Writer
- **Python API**：pybind11 扩展 `turbol_core` + `turbol` 包，示例含 PPO / GRPO

## 环境要求

必需：

- CMake ≥ 3.28、Ninja
- 支持 **C++23** 的 GCC / Clang
- **CUDA Toolkit 13.2**
- NVIDIA 驱动 + 计算能力 ≥ 7.0 的 GPU

可选：

- libtorch（`-DTURBORL_USE_LIBTORCH=ON`，策略网络）：优先读 `-DLIBTORCH_ROOT=...`，
  其次读 `$LIBTORCH_ROOT` 环境变量，最后走 CMake 默认搜索路径
- NCCL（`-DTURBORL_USE_NCCL=ON`）
- vLLM-CPP 静态库 `libvllm.a`（放在同级 `<repo>/../vllm/build/`，自动探测）
- pybind11 + torch（`-DTURBORL_ENABLE_PYTHON=ON`）

## 构建

```bash
export LIBTORCH_ROOT=/opt/libtorch        # 或 pip 安装的 torch 目录

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBORL_ENABLE_CUDA=ON \
      -DTURBORL_USE_LIBTORCH=ON \
      -DTURBORL_ENABLE_TESTS=ON \
      -DTURBORL_ENABLE_BENCHMARKS=ON
cmake --build build -j"$(nproc)"
```

默认编译 `sm_120`（RTX 50 系列）；其它显卡用 `-DCMAKE_CUDA_ARCHITECTURES="86;90"` 覆盖。

### 构建选项

| 选项 | 说明 | 默认 |
| --- | --- | --- |
| `TURBORL_ENABLE_CUDA` | 启用 CUDA | `ON` |
| `TURBORL_USE_LIBTORCH` | 启用 libtorch 策略网络 | `ON` |
| `TURBORL_USE_NCCL` | 启用 NCCL 分布式 | `OFF` |
| `TURBORL_ENABLE_TESTS` | 构建单元测试 | `ON` |
| `TURBORL_ENABLE_BENCHMARKS` | 构建基准程序 | `ON` |
| `TURBORL_ENABLE_PYTHON` | 构建 pybind11 绑定 | `OFF` |
| `LIBTORCH_ROOT` | libtorch/torch 路径（缓存变量） | 自动探测 |
| `CMAKE_CUDA_ARCHITECTURES` | 目标计算能力 | `120` |

## 运行

```bash
./build/turbol_example
ctest --test-dir build --output-on-failure        # 38 个用例 / 7 个套件

./build/benchmark_components --device cuda --num-envs 4096 --steps 500 --batch 256
./build/benchmark_tensor
./build/benchmark_attention
```

## Python 绑定

```bash
pip install pybind11 torch numpy
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DTURBORL_USE_LIBTORCH=ON -DTURBORL_ENABLE_PYTHON=ON \
      -DCMAKE_PREFIX_PATH="$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
      -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build --target turbol_core -j"$(nproc)"

PYTHONPATH=python python -c "import turbol; print(turbol.get_version())"
```

`turbol` 包 re-export 了 `Tensor / GPUEnvironment / RingBuffer / PolicyEngine / Config /
RolloutEngine / RewardEngine / DistributedTrainer / Profiler / TensorBoardWriter /
PrometheusWriter`，并提供 `make()`、`from_torch()`、`to_torch()` 等便捷函数。

示例：

```bash
python examples/py/ppo_example.py --num-envs 256 --rollout-steps 32 --batch-size 64 \
       --num-updates 200 --device cuda:0
python examples/py/grpo_example.py
```

## 目录结构

```text
turbol_vllm/
├── CMakeLists.txt
├── include/turbol/                 # 公共头文件（core/env/replay_buffer/rollout/reward/
│                                   # policy/dataset/distributed/profiler/utils）
├── src/
│   ├── cuda/                       # attention / ring_buffer / math_utils / env_dynamics / gae
│   └── ...                         # 与 include 对应的实现
├── python/                         # pybind11 绑定（turbol_core）+ turbol 包
├── tests/                          # GoogleTest（38 个用例）
├── benchmarks/                     # benchmark_tensor / benchmark_attention / benchmark_components
└── examples/                       # C++ 示例 + examples/py（PPO / GRPO）
```

## 已知限制

- CUDA 路径下 `RingBuffer::PushBatch/Sample` 为逐元素 `cudaMemcpy` + 同步，吞吐低于 CPU 路径
  （实测约 1×10⁵ transitions/s），修复项在路线图中（批量搬运 kernel）。
- 小 batch 的策略前向/更新存在每次调用的同步开销，CUDA 优势需更大 batch 才体现。
- vLLM-CPP 不随仓库分发；`libvllm.a` 需以 `-fPIC` 编译才能链接进 Python 扩展模块。
- 只支持单机多卡，暂无多机启动器。

## 许可证

Apache License 2.0，见 [`../LICENSE`](../LICENSE)。

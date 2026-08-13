# TurboRL

> 由 CUDA 和 C++ 驱动的高性能强化学习（RL）与 RLHF 基础设施

TurboRL 是一个面向下一代强化学习与 RLHF 工作负载的工业级、GPU 原生基础设施。它把环境交互、经验回放、rollout 推理、奖励计算、策略训练到分布式通信这一整条链路都搬到 GPU 上执行，消除传统 RL 框架中反复在 CPU-GPU 之间搬运数据的瓶颈，并与 vLLM-CPP 深度集成以提供高性能大模型推理能力。

---

## 目录

- [1. 项目介绍](#1-项目介绍)
- [2. 框架](#2-框架)
  - [2.1 总体架构](#21-总体架构)
  - [2.2 目录结构](#22-目录结构)
  - [2.3 技术栈](#23-技术栈)
- [3. vLLM-CPP](#3-vllm-cpp)
  - [3.1 特点](#31-特点)
  - [3.2 用途](#32-用途)
  - [3.3 使用方法](#33-使用方法)
- [4. TurboRL](#4-turborl)
  - [4.1 特点](#41-特点)
  - [4.2 用途](#42-用途)
  - [4.3 核心模块与使用方法](#43-核心模块与使用方法)
  - [4.4 构建与测试](#44-构建与测试)
- [5. 技术规格](#5-技术规格)
- [6. 当前状态与路线图](#6-当前状态与路线图)

---

## 1. 项目介绍

当前 RL/RLHF 开源工具链存在明显痛点：

- **碎片化严重**：Gymnasium 环境、Ray/RLlib 分布式、vLLM 推理、DeepSpeed 训练各自独立，缺少统一基础设施。
- **CPU 瓶颈**：Replay Buffer、环境 step 等关键路径仍依赖 CPU，GPU 利用率常低于 60%。
- **扩展性不足**：缺少对大规模 RLHF 训练的原生支持。
- **工程标准参差**：缺少统一的 Profiler、Benchmark、CI/CD 体系。

TurboRL 的目标是填补这一空白，提供从「环境交互 → 经验回放 → 策略推理 → 分布式训练」的全链路 **GPU Native** 解决方案。

```text
TurboRL = RL Infrastructure + RLHF Infrastructure + CUDA HPC + LLM Inference + Distributed Training
```

核心定位：

- 支持传统 RL（DQN、PPO、SAC、TD3）、Offline RL（CQL、IQL）、RLHF（PPO、DPO、GRPO）、Agent RL；
- 全链路 GPU 加速：Environment → Replay Buffer → Rollout → Reward → Policy 均运行在 GPU 上；
- 以 C++ / CUDA 为性能内核，libtorch 提供神经网络运算，vLLM-CPP 提供大模型推理。

---

## 2. 框架

### 2.1 总体架构

TurboRL 采用**松耦合的模块化架构**，各模块既可组合成完整 RLHF 流水线，也可单独使用：

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         RLHF Pipeline (Trainer)                       │
└─────────────────────────────────────────────────────────────────────┘
        │            │            │             │            │
        ▼            ▼            ▼             ▼            ▼
┌──────────┐ ┌──────────┐ ┌────────────┐ ┌───────────┐ ┌──────────────┐
│  GPU      │ │ CUDA      │ │ RLHF       │ │  Policy    │ │ Distributed  │
│ Environment│ │ Replay    │ │ Rollout    │ │  Engine    │ │  Engine      │
│ (env)     │ │ Buffer    │ │ Engine     │ │ (libtorch) │ │  (NCCL)      │
└──────────┘ └──────────┘ └─────┬──────┘ └───────────┘ └──────────────┘
                                │  Generate
                                ▼
                        ┌──────────────┐
                        │  VLLMBackend │  →  vllm::Engine（大模型推理）
                        └──────────────┘
        ▲
        │ 统一张量 / 工具
┌───────┴───────────────────────────────────────────────┐
│  common.hpp（Status / Shape / Device / DataType / Tensor）  │
│  utils（TensorUtils、torch_interop）                      │
│  cuda/（attention、ring_buffer、math_utils、env_dynamics、gae）│
└──────────────────────────────────────────────────────────┘
```

模块职责：

| 模块                      | 职责                                                     |
| ----------------------- | ------------------------------------------------------ |
| **Core Engine**         | 全局单例编排器，模块注册、初始化、运行、同步                                 |
| **GPU Environment**     | 向量化环境模拟，一个线程对应一个环境实例，状态全程驻留 GPU                        |
| **CUDA Replay Buffer**  | 固定容量环形缓冲，O(1) 写入 + 均匀采样，CPU/CUDA 双路径                   |
| **RLHF Rollout Engine** | LLM 推理编排，通过 VLLMBackend 调用真实 vLLM 引擎                   |
| **Policy Engine**       | 策略网络推理与更新（libtorch 高斯 actor/critic）                    |
| **Distributed Engine**  | 基于 NCCL 的多 GPU 集合通信（AllReduce / Broadcast / AllGather） |
| **Tensor / utils**      | 统一 CPU/CUDA 张量、元素级算子、零拷贝 torch 互操作                     |

### 2.2 目录结构

```
TurboRL/
├── TurboRL_Specification.md    # 详细规格文档（设计蓝图）
├── README.md                   # 本文件
│
├── turbol/                     # 原版（仅 CPU，最小依赖，纯 C++17 骨架）
│   ├── CMakeLists.txt
│   ├── include/turbol/         # 头文件
│   ├── src/                    # 源代码
│   ├── tests/                  # 单元测试
│   ├── benchmarks/             # 性能基准
│   └── examples/               # 示例程序
│
└── turbol_vllm/                # 完整版（CUDA + vLLM + libtorch，主开发目录）
    ├── CMakeLists.txt
    ├── include/turbol/
    │   ├── common.hpp              # 基础类型：Status/Shape/Device/DataType/Tensor
    │   ├── core/                   # CoreEngine、Config
    │   ├── env/                    # GPUEnvironment
    │   ├── replay_buffer/          # RingBuffer
    │   ├── rollout/                # RolloutEngine、VLLMBackend
    │   ├── policy/                 # PolicyEngine
    │   ├── distributed/            # DistributedTrainer、NcclCommunicator
    │   └── utils/                  # TensorUtils、torch_interop
    ├── src/
    │   ├── cuda/                   # CUDA 内核：attention/ring_buffer/math_utils/env_dynamics/gae
    │   ├── core/ env/ replay_buffer/ rollout/ policy/ distributed/ utils/
    ├── tests/                  # 25 个单元测试
    ├── benchmarks/             # 张量 / 注意力基准
    └── examples/               # main.cpp 演示
```

> **说明**：`turbol/` 是仅 CPU 的最小骨架版本；`turbol_vllm/` 是功能完整的优化版本。两个目录各自独立构建。

### 2.3 技术栈

| 层级     | 技术选型                          | 说明                                |
| ------ | ----------------------------- | --------------------------------- |
| 核心语言   | C++23（主机端）/ C17（C 代码）         | 现代 C++ 特性                         |
| GPU 编程 | CUDA 13.2（设备端 C++20）          | 目标 Blackwell `sm_120`（RTX 5070）   |
| 神经网络   | libtorch 2.12（PyTorch C++ 前端） | 策略网络、优化器                          |
| 分布式通信  | NCCL 2.x                      | AllReduce / Broadcast / AllGather |
| 推理引擎   | vLLM-CPP                      | 大模型推理后端                           |
| 构建系统   | CMake 3.28+ / Ninja           | 现代化构建                             |
| 测试框架   | GoogleTest                    | 单元测试                              |

---

## 3. vLLM-CPP

### 3.1 特点

vLLM-CPP 是一个**纯 C++17 + CUDA** 实现的轻量级高性能 LLM 推理加速框架，针对 NVIDIA GeForce RTX 5070（Blackwell，`sm_120`）与 CUDA 13.2 优化。与 TurboRL 在同一工作区内，以静态库 `libvllm.a` 形式被 TurboRL 链接使用。

核心特性：

| 特性                      | 说明                                                      |
| ----------------------- | ------------------------------------------------------- |
| **PagedAttention**      | 块粒度 KV Cache 管理，块大小 16/32/64 可配置，避免碎片化                  |
| **Continuous Batching** | 连续批处理，动态增删请求，最大化 GPU 利用率                                |
| **Chunked Prefill**     | Prefill 与 Decode 解耦调度，token 预算感知                        |
| **单一 GPU 内存池**          | 所有权重、KV Cache、缓冲区一次 `cudaMalloc`                        |
| **BF16 原生 GEMM**        | `cublasGemmEx` 直接读取 BF16 权重，零 F32 转换开销                  |
| **CUDA 算子**             | FlashAttention、RMSNorm、RoPE、SiLU/SwiGLU，针对 Blackwell 优化 |
| **量化**                  | W8A8 INT8、W4A16（GPTQ/AWQ）、FP8 原生                        |
| **OpenAI 兼容 API**       | `/v1/completions`、`/v1/chat/completions`                |
| **多架构**                 | LLaMA 3/4、Qwen 2.5/3、Gemma 3、Mistral                    |

### 3.2 用途

vLLM-CPP 在 TurboRL 生态中扮演 **LLM 推理引擎** 角色，主要用途：

1. **RLHF rollout 采样**：在 PPO/GRPO 训练中，用 LLM 批量生成响应（rollout），作为奖励模型与策略更新的输入；
2. **独立推理服务**：以 OpenAI 兼容 HTTP 服务对外提供文本补全/对话能力；
3. **嵌入式推理**：作为 C++ 库被其他程序直接调用（如 TurboRL 的 `VLLMBackend`）。

### 3.3 使用方法

#### 3.3.1 命令行 / HTTP 服务

```bash
cd vllm

# 1. 下载模型
./scripts/download_model.sh Qwen/Qwen2.5-0.5B-Instruct

# 2. 构建并启动服务
MODEL_PATH=./models/Qwen2.5-0.5B-Instruct MODEL_ARCH=qwen ./scripts/run_server.sh

# 3. 调用推理接口
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The capital of France is", "max_tokens": 20}'
```

健康检查：

```bash
curl http://localhost:8000/health
# → {"status": "ok"}
```

#### 3.3.2 C++ 引擎 API（TurboRL 内部使用）

vLLM 的核心引擎接口（`include/vllm/core/engine.hpp`）：

```cpp
#include "vllm/vllm.hpp"

// 1. 配置
vllm::EngineConfig config;
config.model_path = "./models/Qwen2.5-0.5B-Instruct";
config.model_arch = "qwen";
config.temperature = 1.0f;
config.top_p = 1.0f;
config.top_k = -1;

// 2. 初始化引擎
vllm::Engine engine(config);
engine.initialize();

// 3. 添加生成请求（返回 seq_id）
int64_t id = engine.add_request("Hello", /*max_output_len=*/128,
                                /*temperature=*/1.0f, /*top_p=*/1.0f, /*top_k=*/-1);

// 4a. 单步驱动（返回本步新生成的 token）
std::vector<vllm::GenerationOutput> outs = engine.step();

// 4b. 或循环直到全部完成，每生成一个 token 回调一次
engine.run([](const vllm::GenerationOutput& out) {
    // out.seq_id / out.tokens / out.text / out.finished / out.logprobs
});

// 5. 将 token id 解码为文本
std::string text = engine.decode(outs[0].tokens);
```

其中 `GenerationOutput` 结构为：

```cpp
struct GenerationOutput {
    int64_t seq_id;                       // 请求序列 id
    std::vector<TokenId> tokens;          // 新生成的 token id
    std::string text;                     // 解码后的文本
    bool finished = false;                // 是否生成结束
    std::vector<LogprobEntry> logprobs;   // 每个 token 的 log 概率
};
```

---

## 4. TurboRL

### 4.1 特点

| 特性              | 说明                                                     |
| --------------- | ------------------------------------------------------ |
| **GPU 原生**      | 环境、回放、rollout、策略全链路 GPU 加速，避免 CPU-GPU 往返               |
| **统一张量抽象**      | `turborl::Tensor` 同时支持 CPU 与 CUDA 存储，类型化访问 `data<T>()` |
| **CUDA 内核优化**   | 在线 softmax 的 FlashAttention、float4 向量化环形缓冲、GAE 内核      |
| **libtorch 集成** | 策略网络用 PyTorch C++ 前端，支持零拷贝 `torch::from_blob` 互操作      |
| **vLLM 深度集成**   | Rollout 引擎直连真实 `vllm::Engine`，无模型/GPU 时优雅降级 CPU        |
| **分布式优先**       | 基于 NCCL 的 AllReduce / Broadcast / AllGather，单机多进程文件引导  |
| **工程标准**        | C++23 / C17 / CUDA 13.2，GoogleTest 单元测试 + 基准测试         |

### 4.2 用途

- **RL 训练**：DQN / PPO / SAC 等经典算法的环境交互、经验回放、策略更新；
- **RLHF**：LLM rollout → 奖励打分 → PPO/DPO/GRPO 策略优化；
- **高性能推理**：作为推理后端嵌入其它 C++ 服务；
- **教学与二次开发**：松耦合模块可作为构建块复用。

### 4.3 核心模块与使用方法

> 所有代码示例均使用 `namespace turborl`。

#### 4.3.1 张量与基础类型（`common.hpp`）

```cpp
#include "turbol/common.hpp"
using namespace turborl;

// 形状、设备、类型
Shape shape({4, 8});
Tensor t(shape, DataType::kFloat32, Device::CPU());

// 类型化访问
float* p = t.data<float>();
for (int i = 0; i < t.numel(); ++i) p[i] = 1.0f;

// 迁移到 GPU（按需）
Tensor g = t.Clone();
g.ToDevice(Device::CUDA(0));

// 数据类型大小
size_t s = GetDataTypeSize(DataType::kInt32);  // 4
```

#### 4.3.2 配置系统（`core/config.hpp`）

```cpp
#include "turbol/core/config.hpp"
turborl::Config cfg;

cfg.Set("learning_rate", 1e-4f);
cfg.Set("num_layers", 24);
cfg.Set("use_amp", true);

float lr = cfg.Get<float>("learning_rate", 3e-4f);   // 缺省时返回默认值
cfg.Save("/tmp/train.cfg");
cfg.Load("/tmp/train.cfg");                            // key=value 文本格式
```

#### 4.3.3 GPU 环境（`env/gpu_environment.hpp`）

```cpp
#include "turbol/env/gpu_environment.hpp"
using namespace turborl::env;

// 256 个并行环境，观测维 4，动作维 2，运行在 GPU
GPUEnvironment env(/*num_envs=*/256, /*obs_dim=*/4, /*act_dim=*/2, Device::CUDA(0));

env.Reset();                                  // 重置所有环境
Tensor obs = env.Observe();                   // [256, 4] 当前状态

Tensor action(Shape({256, 2}), DataType::kFloat32, Device::CUDA(0));
Tensor next_obs, reward, done;
env.Step(action, &next_obs, &reward, &done);  // 动力学：s' = decay·s + force·a，r = -||s'||²
```

#### 4.3.4 回放缓冲区（`replay_buffer/ring_buffer.hpp`）

```cpp
#include "turbol/replay_buffer/ring_buffer.hpp"
using namespace turborl::replay_buffer;

RingBuffer buffer(/*capacity=*/1000, /*obs_dim=*/128, /*act_dim=*/8, Device::CUDA(0));

buffer.Push(obs, act, reward, /*done=*/false);   // 单条写入 O(1)
buffer.PushBatch(obs_batch, act_batch, reward_batch, done_batch);  // 批量写入

Tensor obs, act, reward, done;
buffer.Sample(/*batch_size=*/64, &obs, &act, &reward, &done);  // 均匀采样（有放回）
```

#### 4.3.5 策略引擎（`policy/policy_engine.hpp`，libtorch）

```cpp
#include "turbol/policy/policy_engine.hpp"
using namespace turborl::policy;

PolicyEngine policy(/*obs_dim=*/4, /*act_dim=*/2, Device::CUDA(0));
policy.Initialize();                          // 构建 actor/critic + Adam 优化器

Tensor action, logprob;
policy.Forward(obs, &action, &logprob);       // 采样动作 a~N(tanh(W·obs+b), σ²) + log 概率

Tensor value;
policy.GetValue(obs, &value);                 // critic 状态价值 V(s)

// 更新：batch 布局为 [N, obs_dim + act_dim + 2] = [obs | action | advantage | return]
policy.Update(batch);                          // REINFORCE 风格 + 价值 MSE 联合损失
```

#### 4.3.6 Rollout 引擎与 vLLM 后端（`rollout/`）

```cpp
#include "turbol/rollout/rollout_engine.hpp"
using namespace turborl::rollout;

RolloutConfig config;
config.inference_backend = "vllm";
config.model_path = "./models/Qwen2.5-0.5B-Instruct";
config.max_new_tokens = 256;
config.temperature = 1.0f;

RolloutEngine engine(config);
engine.Initialize();

std::string out;
engine.Generate("Hello, world!", &out);       // 有模型+GPU 走真实 vLLM，否则回退 CPU echo

std::vector<std::string> prompts = {"a", "b", "c"};
std::vector<std::string> outs;
engine.GenerateBatch(prompts, &outs);
```

#### 4.3.7 分布式通信（`distributed/`，NCCL）

```cpp
#include "turbol/distributed/distributed_trainer.hpp"
using namespace turborl::distributed;

DistributedConfig dcfg;
dcfg.rank = 0;
dcfg.world_size = 2;
dcfg.use_nccl = true;

DistributedTrainer trainer(dcfg);
trainer.Initialize();

Tensor t(Shape({100}), DataType::kFloat32, Device::CUDA(0));
trainer.AllReduce(&t, "mean");                // 跨 rank 平均
trainer.Broadcast(&t, /*root_rank=*/0);
trainer.Shutdown();
```

#### 4.3.8 张量工具与 torch 互操作（`utils/`）

```cpp
#include "turbol/utils/tensor_utils.hpp"
#include "turbol/utils/torch_interop.hpp"
using namespace turborl::utils;

TensorUtils::Fill(&t, 2.0f);
TensorUtils::AddScalar(&t, 1.0f);
TensorUtils::Scale(&t, 0.5f);
TensorUtils::Add(a, b, &out);

// 零拷贝视图（不复制内存，torch 借用 turborl::Tensor 的缓冲区）
torch::Tensor view = torch_interop::AsTorchView(t);
// 拥有所有权的拷贝
torch::Tensor owned = torch_interop::ToTorch(t);
```

#### 4.3.9 CUDA 内核（`src/cuda/`）

| 内核文件              | 功能      | 说明                            |
| ----------------- | ------- | ----------------------------- |
| `attention.cu`    | 缩放点积注意力 | 在线 softmax，数值稳定，支持变长序列        |
| `ring_buffer.cu`  | 环形缓冲写入  | 单次原子预留 + float4 向量化           |
| `math_utils.cu`   | 元素级算子   | add/sub/fill/add_scalar/scale |
| `env_dynamics.cu` | 环境动力学   | 每环境一线程，线性系统 step/reset        |
| `gae.cu`          | 广义优势估计  | 每环境一线程，反向递推计算 GAE             |

#### 4.3.10 奖励引擎（`reward/`，RLHF）

```cpp
#include "turbol/reward/reward_engine.hpp"
using namespace turborl::reward;

RewardConfig cfg;                 // kl_coef / kl_target / adaptive_kl /
                                  // normalize / reward_clip / 规则权重
cfg.device = Device::CUDA(0);
cfg.normalize = true;
cfg.format_bonus = 0.1f;

RewardEngine engine(cfg);
engine.Initialize();

RewardResult result;
std::vector<std::string> responses = {"...", "..."};
engine.ScoreBatch(responses, std::nullopt, &result);
// result.total / rule_scores / model_scores / kl_penalties 均为 [batch] 张量
engine.UpdateAdaptiveKL(/*observed_kl=*/0.12f);   // 自适应 KL 系数
float coef = engine.CurrentKLCoef();
```

#### 4.3.11 数据集（`dataset/`）

```cpp
#include "turbol/dataset/dataset.hpp"
using namespace turborl::dataset;

DatasetConfig cfg;                 // data_paths / max_seq_length / shuffle
Dataset ds(cfg);
ds.Initialize();                   // 流式 JSONL 加载，字节偏移 seek
while (ds.HasNext()) {
    PreprocessedBatch b = ds.NextBatch();   // input_ids / attention_mask / labels
    // 送入 Rollout / Policy
}
ds.Reset();                        // 回到数据集起点
```

#### 4.3.12 性能剖析器（`profiler/`）

```cpp
#include "turbol/profiler/profiler.hpp"
using namespace turborl::profiler;

Profiler profiler;
profiler.Enable();

auto id = profiler.BeginSpan("rollout", "compute", /*device_id=*/0);
// ... 计时区间 ...
profiler.EndSpan(id);

// 或使用 RAII Guard 自动收尾
{
    Profiler::Guard g(&profiler, "policy_update");
    // ...
}

profiler.ExportChromeTrace("/tmp/trace.json");   // chrome://tracing 可视化
profiler.PrintSummary();                          // 聚合统计摘要
```

配套的 `TensorBoardWriter`（写入最小化 protobuf Event，无外部 TB 依赖）与
`PrometheusWriter`（text exposition 格式）位于 `profiler/tensorboard_writer.hpp`。

#### 4.3.13 Pipeline 训练（`distributed/pipeline_trainer.hpp`）

```cpp
#include "turbol/distributed/pipeline_trainer.hpp"
using namespace turborl::distributed;

PipelineConfig cfg;                // num_pipeline_stages / num_microbatches / stage_id
PipelineTrainer trainer(cfg);
trainer.Initialize();

float avg_loss = 0.0f;
trainer.TrainingStep(&avg_loss);   // GPipe 风格微批前向/反向调度
trainer.Shutdown();
```

### 4.4 构建与测试

#### 4.4.1 环境要求

- CMake 3.28+、Ninja
- GCC / Clang 支持 C++23
- CUDA Toolkit 13.2（`nvcc`）
- NVIDIA 驱动 + 计算能力 `sm_120` 的 GPU（RTX 5070 系列）
- libtorch 2.12.0+cu132（`-DTURBORL_USE_LIBTORCH=ON` 时需要）
- vLLM-CPP 构建产物 `libvllm.a`（自动探测）

#### 4.4.2 构建完整版

```bash
cd turbol_vllm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake 会自动探测 libtorch（`LIBTORCH_ROOT` 环境变量或默认路径）与 vLLM 库。常用选项：

| 选项                          | 描述          | 默认值 |
| --------------------------- | ----------- | --- |
| `TURBORL_ENABLE_CUDA`       | 启用 CUDA     | ON  |
| `TURBORL_USE_LIBTORCH`      | 启用 libtorch | ON  |
| `TURBORL_USE_NCCL`          | 启用 NCCL 分布式 | OFF |
| `TURBORL_ENABLE_TESTS`      | 启用单元测试      | ON  |
| `TURBORL_ENABLE_BENCHMARKS` | 启用基准测试      | ON  |

#### 4.4.3 运行示例与测试

```bash
cd build
./turbol_example      # 演示程序
ctest --output-on-failure   # 38 个单元测试
./turbol_test         # 或直接运行测试二进制
```

#### 4.4.4 运行微基准测试（Benchmark）

`benchmark_components` 对训练循环中的关键热路径做稳态吞吐量测量（先预热、后计时），覆盖
环境 step、经验回放 push/sample、策略 forward/update（libtorch）、GAE（CUDA）：

```bash
cd build
./benchmark_components                     # 默认 CPU，4096 环境
./benchmark_components --device cuda       # 走 CUDA 路径
./benchmark_components --num-envs 8192 --steps 2000 --batch 512
```

示例输出（RTX 5070，`sm_120`）：

```text
[1/6] GPUEnvironment::Step (4096 envs, 16x4)   => 1.16 亿 env-steps/s
[2/6] RingBuffer::PushBatch (256/batch)        => 6766 万 transitions/s
[3/6] RingBuffer::Sample (256/batch)           => 4447 万 transitions/s
[4/6] PolicyEngine::Forward (batch=256)        => 1242 万 samples/s
[5/6] PolicyEngine::Update (batch=256)         => 217 万 samples/s
[6/6] GAE (T=256, envs=4096)                   => 447 亿 env-steps/s
```

> 注：CUDA 路径下 `RingBuffer::PushBatch/Sample` 当前为逐元素 `cudaMemcpy` 实现，吞吐量受
> 同步拷贝限制；未来可用定制 kernel 一次性搬移整批（见「规划中」的 CUDA 内核项）。

#### 4.4.5 构建 Python 绑定（pybind11）

```bash
pip install pybind11 torch numpy   # 运行时依赖（torch 为可选项，未安装则无法 import turbol）
cd turbol_vllm
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)     # 生成 build/python/turbol_core*.so
```

绑定模块 `turbol_core` 暴露 `Tensor / GPUEnvironment / RingBuffer / PolicyEngine /
RolloutEngine / RewardEngine / DistributedTrainer / Profiler / TensorBoardWriter / PrometheusWriter`
等全部核心类型，`python/turbol/__init__.py` 负责 re-export 并封装成 `import turbol`。
示例位于 `examples/py/ppo_example.py` 与 `examples/py/grpo_example.py`。

> **已知限制**：当前环境未安装 Python `torch`，且 `libvllm.a` 以非 PIC 的 `local-exec`
> TLS 模型编译，链接为共享模块（`.so`）时需 `-fPIC` 重编 vLLM。二者满足后即可 `import turbol`
> 并运行 Python 示例。

#### 4.4.6 构建原版（仅 CPU）

```bash
cd turbol
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTURBORL_ENABLE_CUDA=OFF
cmake --build build -j$(nproc)
./build/turbol_example
```

---

## 5. 技术规格

| 规格         | 值                            |
| ---------- | ---------------------------- |
| 主机端 C++ 标准 | C++23                        |
| C 标准       | C17                          |
| CUDA 设备端标准 | C++20                        |
| CUDA 版本    | 13.2                         |
| 目标计算能力     | `sm_120`（Blackwell，RTX 5070） |
| 神经网络库      | libtorch 2.12.0+cu132        |
| 分布式通信      | NCCL 2.x（可选）                 |
| 推理引擎       | vLLM-CPP                     |
| 构建系统       | CMake 3.28+ / Ninja          |
| 测试框架       | GoogleTest                   |

---

## 6. 当前状态与路线图

### 已完成

- [x] 构建系统现代化（C++23 / C17 / CUDA 13.2 / `sm_120`）
- [x] `common.hpp` 基础类型修复与类型化张量访问
- [x] Config、RingBuffer、TensorUtils、DistributedTrainer 真实实现（CPU + CUDA 双路径）
- [x] GPUEnvironment 向量化环境（Reset / Step / Observe）
- [x] PolicyEngine（libtorch actor/critic，Forward / GetValue / Update）
- [x] VLLMBackend 接入真实 `vllm::Engine`，无模型/GPU 时优雅降级
- [x] CUDA 内核：修复 FlashAttention 在线 softmax、向量化环形缓冲、新增 GAE 内核
- [x] RewardEngine（规则打分 + KL 惩罚 + 自适应 KL + 归一化）
- [x] Dataset 模块（流式 JSONL 加载、字节偏移 seek）
- [x] Profiler（CPU/CUDA 计时、Chrome Trace / JSON 导出）+ TensorBoard / Prometheus Writer
- [x] PipelineTrainer（GPipe 风格微批调度）
- [x] Python API 层（pybind11 绑定 `turbol_core` + `turbol` 包）
- [x] 微基准测试套件（`benchmark_components`，吞吐量报告）
- [x] 单元测试（38 个全部通过）

### 规划中

- [ ] 更多 CUDA 内核（CUTLASS GEMM、FlashAttention-2/3、批量回放搬移 kernel）
- [ ] 多机分布式与 Pipeline Parallel 运行时（NCCL 后端）
- [ ] 奖励模型推理（Reward Model 前向接入 libtorch）
- [ ] Python 绑定端到端验证（需 `torch` + `-fPIC` 重编 `libvllm.a`）

---

## 许可证

Apache License 2.0

## 引用

```bibtex
@software{turborl,
  title = {TurboRL: High Performance RL & RLHF Infrastructure},
  author = {TurboRL Team},
  year = {2026},
  version = {1.0.0}
}
```

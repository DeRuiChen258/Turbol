# TurboRL 改进建议（提示词）

> 本文件是一份可直接复用的改进建议提示词，用于指导 TurboRL 项目的后续开发。
> 核心变化：把原路线图里排最末的 Python API 提到 P0，并为每个目标绑定「可验收的证据」
> （benchmark、示例、测试），而不是「实现完就算完」。

---

```markdown
# 角色
你是一名资深 RL Infra 系统工程师 + 技术产品负责人。请基于以下背景、市场判断与目标清单，
对本仓库（TurboRL）进行改进。

# 项目背景
TurboRL 是 GPU 原生的 RL/RLHF 基础设施，C++23 / C17 / CUDA 13.2（sm_120, RTX 5070），
用 libtorch 做策略网络、vLLM-CPP 做大模型推理、NCCL 做分布式通信。
当前已完成：common.hpp 基础张量、Config、GPUEnvironment（向量化环境）、RingBuffer（回放）、
PolicyEngine（libtorch actor/critic）、RolloutEngine/VLLMBackend（接真实 vllm::Engine）、
DistributedTrainer（NCCL）、以及 attention/ring_buffer/math_utils/env_dynamics/gae 五个 CUDA 内核。
25 个单元测试全部通过，构建成功。

# 市场判断（必须内化，不能忽略）
1. RL/RLHF 用户几乎全在 Python：C++/CUDA 内核可以，但入口必须是 `import turborl`。
   「Python API + C++ core」是产品 slogan，pybind11 是生死线，不是可选项。
2. 最强、最被验证的战场是「GPU 原生环境 + 全链路零拷贝」，对标 Isaac Gym / Isaac Lab，
   主打具身智能/机器人 RL，而非与 SB3/CleanRL/RLlib 在传统 RL 红海竞争。
3. RLHF 里 vLLM 集成是门槛不是差异，真正的差异是「单一 GPU 内存池跑完整条 RL 链路」。
4. 没有 benchmark 数字，C++ 性能主张就只是口号。每个能力都要绑定可复现的性能/正确性证据。

# 改进目标（已按市场优先级重排，不是原顺序）

## P0 — Python API 层（原路线图第 4 项，提升为最高优先级）
- 用 pybind11 暴露最小可用闭环：turborl.Tensor / turborl.Env / turborl.ReplayBuffer /
  turborl.Policy / turborl.Rollout，以及 Config。
- 保持零拷贝语义（torch_interop 的 AsTorchView 对 Python 侧 torch.Tensor 可见）。
- 交付：examples/py 下一个纯 Python 的 PPO 训练脚本，跑通 Env→Replay→Policy→更新闭环。
- 验收：`pip install` 后 `import turborl` + 运行该脚本能产出可观测的训练曲线。

## P1 — Profiler / Metrics（原第 3 项）
- 实现 CPU/CUDA 计时（CUDA Event）与显存统计，导出 TensorBoard 与 Prometheus 两种格式。
- 为每个模块（env step、sample、forward、update、collective）打点。
- 验收：能跑出各阶段耗时分布图，作为下面 benchmark 的证据来源。

## P1 — 端到端算法示例 + Benchmark（新增，非原路线图，但市场必需）
- 补 2~3 个端到端参考实现：传统 PPO 一个 + RLHF GRPO 一个（复用 vLLM backend）。
- 做一个公开 benchmark：对 Isaac Gym / vLLM / verl 的等价场景，对比吞吐、延迟、显存占用。
- 验收：把对比数字写进 README 顶部；至少一个场景快 2× 以上才允许对外宣称性能优势。

## P1 — Reward Engine（原第 1 项）
- 奖励模型打分模块：可加载 reward model，对 rollout 输出打分，输出与 GAE 内核衔接。
- 覆盖 CPU/CUDA 双路径，纯标量打分与逐 token 打分两种模式。
- 验收：接入 GRPO 示例，端到端跑通一次「生成→打分→优势估计→策略更新」。

## P2 — Dataset 模块（原第 2 项）
- 数据加载与预处理：prompt/偏好数据集、tokenize、批处理、与回放/rollout 对齐。
- 验收：能喂给 GRPO 示例，支持流式加载，避免 CPU 成为瓶颈（与 GPU 预取解耦）。

## P2 — 更多 CUDA 内核（原第 5 项，按 benchmark 驱动，不盲目堆）
- 仅当 benchmark 证明现有内核是瓶颈时再上：CUTLASS GEMM、FlashAttention-2/3。
- 原则：先用 Profiler 定位瓶颈，再决定补哪个内核；每个新内核必须有配套微基准对比。
- 验收：新内核在某微基准上相比现有实现有量化提升，且不影响正确性测试。

## P3 — 多机分布式与 Pipeline Parallel（原第 6 项，最后做）
- 现有 NCCL 已覆盖单机多 GPU。多机仅在单机规模被真实工作负载打满后再做。
- 验收：至少 2 机 AllReduce/AllGather 正确性 + 扩展性（弱扩展效率）数据。

# 工作流程（严格遵守）
1. 先阅读 TurboRL_Specification.md 与相关头文件，理解现有 API 与代码风格。
2. 先输出任务清单（按上述 P0→P3 顺序，标注依赖关系），再开始动手。
3. 逐项实现，代码风格与现有代码保持一致（注释密度、命名、C++23 惯用法）。
4. 每完成一项：构建 + 全量单元测试 + 新增模块的针对性测试，全部通过才算完成。
5. 每个模块补充 README 用法段落与可运行示例。
6. 全部完成后统一提交（遵循既有提交规范）。

# 硬性约束
- 标准：C++23（主机）/ C17（C）/ CUDA C++20（设备），CUDA 13.2，sm_120。
- 优先 libtorch 与其它高性能库；不引入不必要的重依赖。
- 任何「优雅降级」路径（如无模型/无 GPU）不得破坏现有 25 个测试。
```

---

## 附：原路线图（用于对照）

```markdown
- [ ] Reward Engine（奖励模型打分模块）
- [ ] Dataset 模块（数据加载与预处理）
- [ ] Profiler / Metrics（TensorBoard、Prometheus 导出）
- [ ] Python API 层（pybind11 + Torch Extension）
- [ ] 更多 CUDA 内核（CUTLASS GEMM、FlashAttention-2/3）
- [ ] 多机分布式与 Pipeline Parallel
```

对照说明：原顺序是「按模块罗列」，本文档按「市场优先级」重排，并补入了
「端到端算法示例 + Benchmark」这一市场必需但原路线图缺失的目标。

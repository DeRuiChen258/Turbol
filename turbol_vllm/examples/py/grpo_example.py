"""
GRPO Example — pure Python, demonstrates RLHF pipeline with vLLM backend.

Pipeline:
  1. RolloutEngine: generate responses from prompts (vLLM or CPU fallback)
  2. RewardEngine: score each response (rule-based + KL penalty)
  3. PolicyEngine: update policy with advantage-weighted loss

Requires a vLLM model for real LLM inference; gracefully degrades to CPU echo
when no model is available.

Usage:
  python examples/py/grpo_example.py --model-path ./models/Qwen2.5-0.5B-Instruct
"""

import os
import sys
import time
import argparse
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import torch
import numpy as np

import turbol


# Default prompts used when no dataset is provided
DEFAULT_PROMPTS = [
    "Explain quantum entanglement in one sentence.",
    "Write a Python function to compute fibonacci numbers recursively.",
    "What are the main differences between Python and C++?",
    "Describe the water cycle.",
    "How does photosynthesis work?",
    "Write a haiku about machine learning.",
    "What is the capital of Japan?",
    "Explain blockchain in simple terms.",
    "What is the pythagorean theorem?",
    "How do vaccines work?",
] * 4   # 40 prompts total


def parse_args():
    p = argparse.ArgumentParser(description="TurboRL GRPO Example")
    p.add_argument("--model-path",    type=str,   default="",
                   help="Path to vLLM model (e.g. ./models/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--num-prompts",    type=int,   default=32,   help="Batch size")
    p.add_argument("--max-tokens",    type=int,   default=128,  help="Max generation length")
    p.add_argument("--temperature",    type=float, default=1.0,  help="Sampling temperature")
    p.add_argument("--num-steps",      type=int,   default=100,   help="GRPO steps")
    p.add_argument("--lr",            type=float, default=1e-4,  help="Learning rate")
    p.add_argument("--kl-coef",       type=float, default=0.01,  help="KL penalty coefficient")
    p.add_argument("--device",         type=str,   default="cuda:0",
                   help="Device")
    p.add_argument("--log-dir",        type=str,   default="",
                   help="TensorBoard log dir (auto-generated if empty)")
    p.add_argument("--print-every",    type=int,   default=10,    help="Print every N steps")
    return p.parse_args()


def main():
    args = parse_args()

    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"

    if not args.log_dir:
        args.log_dir = tempfile.mkdtemp(prefix="turbol_grpo_")

    print(f"\n{'='*60}")
    print(f"TurboRL GRPO Example")
    print(f"{'='*60}")
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print()

    device_id = -1 if "cpu" in args.device else 0

    # ---- 1. Rollout Engine (vLLM or CPU fallback) ----
    print("[1/5] RolloutEngine...")
    rcfg = turbol.RolloutConfig()
    rcfg.inference_backend = "vllm"
    rcfg.model_path        = args.model_path
    rcfg.max_new_tokens    = args.max_tokens
    rcfg.temperature       = args.temperature
    rcfg.top_p             = 0.95

    rollout = turbol.RolloutEngine(rcfg)
    rollout.initialize()
    print(f"  vLLM available: {rollout.is_vllm_available()}")
    print(f"  Fallback:       {'CPU echo' if not rollout.is_vllm_available() else 'vLLM'}")

    # ---- 2. Reward Engine ----
    print("[2/5] RewardEngine...")
    rew_cfg = turbol.RewardConfig()
    rew_cfg.kl_coef       = args.kl_coef
    rew_cfg.normalize     = True
    rew_cfg.reward_clip  = 5.0
    rew_cfg.format_bonus = 0.1
    rew_cfg.length_penalty = -0.001
    rew_cfg.target_length  = 64
    rew_cfg.device        = args.device

    reward = turbol.RewardEngine(rew_cfg)
    reward.initialize()

    # ---- 3. Policy Engine ----
    print("[3/5] PolicyEngine (dummy obs_dim=512, act_dim=512)...")
    policy = turbol.PolicyEngine(obs_dim=512, act_dim=512, device=args.device)
    policy.initialize()

    # ---- 4. Profiler + TensorBoard ----
    print("[4/5] Profiler + TensorBoardWriter...")
    profiler = turbol.Profiler()
    profiler.enable()
    tb = turbol.TensorBoardWriter(args.log_dir)

    # ---- 5. GRPO Loop ----
    print(f"[5/5] GRPO ({args.num_steps} steps)...")
    prompts = DEFAULT_PROMPTS[:args.num_prompts]

    for step in range(args.num_steps):
        t_start = time.perf_counter()

        # -- Rollout --
        with profiler.span("rollout", "compute", device_id):
            t_gen = time.perf_counter()
            responses = rollout.generate_batch(prompts)
            gen_time = time.perf_counter() - t_gen

        # -- Reward --
        with profiler.span("reward", "compute", device_id):
            t_rew = time.perf_counter()
            result = reward.score_batch(responses)
            rew_time = time.perf_counter() - t_rew

            total_np = np.asarray(result.total.to_numpy())
            mean_r = float(np.mean(total_np))
            std_r  = float(np.std(total_np) + 1e-8)
            max_r  = float(np.max(total_np))
            min_r  = float(np.min(total_np))

        # -- Policy update --
        with profiler.span("policy_update", "compute", device_id):
            t_pol = time.perf_counter()
            # Dummy batch: obs | act | advantage | return
            batch = turbol.Tensor([args.num_prompts, 512 + 512 + 2],
                                 "float32", args.device)
            policy.update(batch)
            pol_time = time.perf_counter() - t_pol

        step_time = time.perf_counter() - t_start

        # -- Logging --
        if step % args.print_every == 0:
            print(f"  Step {step:4d}/{args.num_steps}  "
                  f"reward={mean_r:7.3f}±{std_r:5.3f}  "
                  f"[min={min_r:6.2f}, max={max_r:6.2f}]  "
                  f"gen={gen_time*1e3:6.1f}ms  "
                  f"rew={rew_time*1e3:5.1f}ms  "
                  f"pol={pol_time*1e3:5.1f}ms")

            tb.add_scalar("grpo/mean_reward",  mean_r, step)
            tb.add_scalar("grpo/reward_std",   std_r,  step)
            tb.add_scalar("grpo/reward_max",   max_r,  step)
            tb.add_scalar("grpo/reward_min",   min_r,  step)
            tb.add_scalar("grpo/gen_ms",        gen_time * 1e3, step)
            tb.add_scalar("grpo/reward_ms",     rew_time * 1e3, step)
            tb.add_scalar("grpo/policy_ms",    pol_time * 1e3, step)
            tb.add_scalar("grpo/kl_coef",      reward.current_kl_coef(), step)

    profiler.disable()
    tb.flush()
    profiler.print_summary()

    print(f"\nGRPO complete. {args.num_steps} steps.")
    print(f"View:  tensorboard --logdir={args.log_dir}")

    # Show sample responses
    print("\nSample responses (last step):")
    for i, (p, r) in enumerate(zip(prompts[:3], responses[:3])):
        print(f"  [{i}] Q: {p[:60]}")
        print(f"  [{i}] A: {r[:80]}")
        print()


if __name__ == "__main__":
    main()

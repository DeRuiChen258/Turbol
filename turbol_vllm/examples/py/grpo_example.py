"""
GRPO Example — pure Python, demonstrates RLHF pipeline with vLLM backend.

Pipeline:
  1. RolloutEngine: generate responses from prompts (vLLM or CPU fallback)
  2. RewardEngine: score each response (rule-based + KL penalty)
  3. GAE: compute advantages on GPU
  4. PolicyEngine: update policy with advantage-weighted loss

Requires a vLLM model for real LLM inference; gracefully degrades to CPU echo
when no model is available.

Run:
  python examples/py/grpo_example.py --model-path ./models/Qwen2.5-0.5B-Instruct
"""

import os
import sys
import time
import argparse
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import torch
import numpy as np

import turbol


def parse_args():
    p = argparse.ArgumentParser(description="TurboRL GRPO Example")
    p.add_argument("--model-path",      type=str,   default="",
                   help="Path to vLLM model (e.g. ./models/Qwen2.5-0.5B-Instruct)")
    p.add_argument("--num-prompts",     type=int,   default=32,   help="Batch size")
    p.add_argument("--max-tokens",     type=int,   default=128,  help="Max generation length")
    p.add_argument("--temperature",     type=float, default=1.0,  help="Sampling temperature")
    p.add_argument("--num-steps",      type=int,   default=100,   help="GRPO steps")
    p.add_argument("--lr",            type=float, default=1e-4,  help="Learning rate")
    p.add_argument("--kl-coef",       type=float, default=0.01,  help="KL penalty coefficient")
    p.add_argument("--device",         type=str,   default="cuda:0", help="Device")
    p.add_argument("--log-dir",        type=str,   default="/tmp/turbol_runs/grpo",
                   help="TensorBoard log dir")
    p.add_argument("--print-every",    type=int,   default=10,    help="Print every N steps")
    return p.parse_args()


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
] * 4  # repeat to reach 40 prompts


def main():
    args = parse_args()
    device_str = args.device if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"TurboRL GRPO Example")
    print(f"{'='*60}")
    print(f"  model_path     : {args.model_path or '(none — CPU fallback)'}")
    print(f"  num_prompts    : {args.num_prompts}")
    print(f"  max_tokens     : {args.max_tokens}")
    print(f"  temperature    : {args.temperature}")
    print(f"  num_steps      : {args.num_steps}")
    print(f"  kl_coef        : {args.kl_coef}")
    print(f"  device         : {device_str}")
    print()

    # 1. Rollout Engine (vLLM or CPU fallback)
    print("[1/5] Initializing RolloutEngine...")
    rcfg = turbol.RolloutConfig()
    rcfg.inference_backend = "vllm"
    rcfg.model_path        = args.model_path
    rcfg.max_new_tokens    = args.max_tokens
    rcfg.temperature       = args.temperature
    rcfg.top_p             = 0.95

    rollout = turbol.RolloutEngine(rcfg)
    rollout.initialize()
    print(f"  vLLM available : {rollout.is_vllm_available()}")

    # 2. Reward Engine (rule-based scoring)
    print("[2/5] Initializing RewardEngine...")
    rew_cfg = turbol.RewardConfig()
    rew_cfg.kl_coef      = args.kl_coef
    rew_cfg.normalize     = True
    rew_cfg.reward_clip  = 5.0
    rew_cfg.format_bonus = 0.1
    rew_cfg.length_penalty = -0.001
    rew_cfg.target_length = 64
    rew_cfg.device        = device_str

    reward = turbol.RewardEngine(rew_cfg)
    reward.initialize()
    print(f"  KL coef        : {reward.current_kl_coef():.4f}")

    # 3. Policy Engine
    print("[3/5] Initializing PolicyEngine...")
    # For GRPO on text, we use obs_dim=512 (token embedding dim placeholder)
    policy = turbol.PolicyEngine(obs_dim=512, act_dim=512, device=device_str)
    policy.initialize()

    # 4. Profiler + TensorBoard
    prof = turbol.Profiler()
    prof.enable()
    tb = turbol.TensorBoardWriter(args.log_dir)

    # 5. GRPO Loop
    print(f"[4/5] Starting GRPO ({args.num_steps} steps)...")
    prompts = DEFAULT_PROMPTS[:args.num_prompts]

    for step in range(args.num_steps):
        step_start = time.perf_counter()

        # --- Rollout: generate responses ---
        with prof.Guard(prof, "rollout", "compute"):
            gen_start = time.perf_counter()
            responses = rollout.generate_batch(prompts)
            gen_time = time.perf_counter() - gen_start

        # --- Reward: score responses ---
        with prof.Guard(prof, "reward", "compute"):
            rew_start = time.perf_counter()
            # Rule-based scoring (KL + format + length)
            result = reward.score_batch(responses)
            rew_time = time.perf_counter() - rew_start

            # Extract rewards as numpy
            total_np = result.total.to_numpy() if hasattr(result.total, 'to_numpy') else \
                       np.zeros(len(responses), dtype=np.float32)

        # --- Compute advantages (simplified GAE: advantage = reward) ---
        with prof.Guard(prof, "advantage", "compute"):
            adv_np = total_np  # placeholder for real GAE
            mean_reward = float(np.mean(total_np))

        # --- Policy update (advantage-weighted REINFORCE) ---
        with prof.Guard(prof, "policy_update", "compute"):
            # Dummy batch: in a real implementation this would be
            # obs | action | advantage | return stacked together
            batch = turbol.Tensor([args.num_prompts, 512 + 512 + 2],
                                  "float32", device_str)
            policy.update(batch)

        step_time = time.perf_counter() - step_start

        # --- Logging ---
        if step % args.print_every == 0:
            eps = 1e-8
            reward_std = float(np.std(total_np) + eps)
            reward_max = float(np.max(total_np))
            reward_min = float(np.min(total_np))
            print(f"  Step {step:4d}/{args.num_steps}  "
                  f"reward={mean_reward:7.3f}±{reward_std:5.3f}  "
                  f"[min={reward_min:6.2f}, max={reward_max:6.2f}]  "
                  f"gen={gen_time*1000:6.1f}ms  rew={rew_time*1000:5.1f}ms")

            tb.add_scalar("grpo/mean_reward", mean_reward, step)
            tb.add_scalar("grpo/reward_std", reward_std, step)
            tb.add_scalar("grpo/reward_max", reward_max, step)
            tb.add_scalar("grpo/reward_min", reward_min, step)
            tb.add_scalar("grpo/gen_time_ms", gen_time * 1000, step)
            tb.add_scalar("grpo/reward_time_ms", rew_time * 1000, step)
            tb.add_scalar("grpo/kl_coef", reward.current_kl_coef(), step)

    prof.disable()
    tb.flush()

    print(f"\n[5/5] GRPO complete.")
    prof.print_summary()
    print(f"\nView with: tensorboard --logdir={args.log_dir}")
    print(f"Sample responses from step {args.num_steps-1}:")
    for i, (p, r) in enumerate(zip(prompts[:3], responses[:3])):
        print(f"  [{i}] Prompt : {p[:60]}...")
        print(f"  [{i}] Response: {r[:100]}...")
        print()


if __name__ == "__main__":
    main()

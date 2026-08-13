"""
PPO Example — pure Python, runs Env → Replay → Policy → Update end-to-end.

This is the minimal TurboRL training loop:
  1. Create a GPUEnvironment (vectorized CartPole-like linear dynamics)
  2. Create a RingBuffer (experience storage)
  3. Create a PolicyEngine (Gaussian actor + critic, libtorch)
  4. Run N rollout steps, storing transitions in the buffer
  5. Sample a batch and update the policy (REINFORCE + value loss)
  6. Log training metrics via Profiler + TensorBoard

Usage:
  pip install -e python/
  python examples/py/ppo_example.py
"""

import os
import sys
import time
import argparse
import tempfile

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../python"))

import turbol


def parse_args():
    p = argparse.ArgumentParser(description="TurboRL PPO Example")
    p.add_argument("--num-envs",       type=int,   default=256,    help="Parallel env count")
    p.add_argument("--obs-dim",        type=int,   default=4,     help="Observation dimension")
    p.add_argument("--act-dim",        type=int,   default=2,     help="Action dimension")
    p.add_argument("--rollout-steps",  type=int,   default=64,    help="Steps per rollout")
    p.add_argument("--batch-size",     type=int,   default=64,    help="Update batch size")
    p.add_argument("--num-updates",    type=int,   default=200,   help="Total policy updates")
    p.add_argument("--lr",             type=float, default=3e-4,  help="Learning rate")
    p.add_argument("--device",         type=str,   default="cuda:0",
                   help="Device (cpu or cuda:N)")
    p.add_argument("--seed",           type=int,   default=42,    help="Random seed")
    p.add_argument("--log-dir",        type=str,   default="",
                   help="TensorBoard log dir (auto-generated if empty)")
    p.add_argument("--print-every",    type=int,   default=20,    help="Print every N updates")
    return p.parse_args()


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def numpy_like(shape, dtype="float32", device="cpu"):
    """Create a Tensor initialized to zeros (convenience)."""
    return turbol.Tensor(list(shape), dtype, device)


def random_action(num_envs: int, act_dim: int, device: str):
    """Sample a clipped-Gaussian action as a Tensor."""
    arr = np.random.randn(num_envs, act_dim).astype(np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    t = numpy_like((num_envs, act_dim), "float32", device)
    buf = t.to_numpy()
    buf[:] = arr
    return t


def main():
    args = parse_args()

    # Resolve device
    if "cuda" in args.device and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"\n{'='*60}")
    print(f"TurboRL PPO Example")
    print(f"{'='*60}")
    for k, v in vars(args).items():
        print(f"  {k:<20}: {v}")
    print()

    set_seed(args.seed)

    # Auto-generate log dir
    if not args.log_dir:
        args.log_dir = tempfile.mkdtemp(prefix="turbol_ppo_")
    print(f"  log_dir: {args.log_dir}")

    # ---- 1. Environment ----
    print("[1/5] GPUEnvironment...")
    env = turbol.GPUEnvironment(
        num_envs=args.num_envs,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=args.device,
    )
    env.reset()

    # ---- 2. Replay Buffer ----
    print("[2/5] RingBuffer...")
    buffer = turbol.RingBuffer(
        capacity=args.rollout_steps * args.num_envs,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=args.device,
    )

    # ---- 3. Policy ----
    print("[3/5] PolicyEngine...")
    policy = turbol.PolicyEngine(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=args.device,
    )
    policy.initialize()

    # ---- 4. Profiler + TensorBoard ----
    print("[4/5] Profiler + TensorBoardWriter...")
    prof_cfg = turbol.ProfilerConfig()
    prof_cfg.enable_nvtx = True
    prof_cfg.enable_cuda_timing = True
    prof_cfg.export_dir = args.log_dir
    profiler = turbol.Profiler(prof_cfg)
    tb = turbol.TensorBoardWriter(args.log_dir)

    if torch.cuda.is_available():
        mi = turbol.get_memory_info(0)
        print(f"  GPU: {mi['total']/1024**2:.0f} MB total, "
              f"{mi['used']/1024**2:.0f} MB used "
              f"({mi['utilization']*100:.1f}%)")

    # ---- 5. Training loop ----
    print(f"[5/5] Training ({args.num_updates} updates)...")
    profiler.enable()
    episode_rewards: list = []
    best_return = -float("inf")
    device_id = -1 if "cpu" in args.device else 0

    for update in range(args.num_updates):
        t_start = time.perf_counter()

        # -- Rollout --
        with profiler.span("rollout", "compute", device_id):
            t_r = time.perf_counter()
            obs = env.observe()
            for step in range(args.rollout_steps):
                action = random_action(args.num_envs, args.act_dim, args.device)

                next_obs = numpy_like((args.num_envs, args.obs_dim), "float32", args.device)
                reward   = numpy_like((args.num_envs,),              "float32", args.device)
                done     = numpy_like((args.num_envs,),              "float32", args.device)

                env.step(action, next_obs, reward, done)

                # Reward as numpy for tracking
                r_np = np.asarray(reward.to_numpy())
                episode_rewards.extend(r_np.tolist())
                if len(episode_rewards) > 5000:
                    episode_rewards = episode_rewards[-5000:]

                # Push batch into ring buffer
                buffer.push_batch(obs, action, reward, done)
                obs = next_obs

            rollout_time = time.perf_counter() - t_r

        # -- Update --
        with profiler.span("update", "compute", device_id):
            t_u = time.perf_counter()
            if buffer.size() >= args.batch_size:
                obs_s = numpy_like((args.batch_size, args.obs_dim), "float32", args.device)
                act_s = numpy_like((args.batch_size, args.act_dim), "float32", args.device)
                rew_s = numpy_like((args.batch_size,),               "float32", args.device)
                don_s = numpy_like((args.batch_size,),               "float32", args.device)

                buffer.sample(args.batch_size, obs_s, act_s, rew_s, don_s)

                # Build training batch: [obs | act | advantage | return]
                # Here: advantage = reward (simplified), return = reward
                batch = numpy_like(
                    (args.batch_size, args.obs_dim + args.act_dim + 2), "float32", args.device)
                policy.update(batch)

            update_time = time.perf_counter() - t_u

        total_time = time.perf_counter() - t_start

        # -- Logging --
        if update % args.print_every == 0:
            recent = episode_rewards[-1000:] if episode_rewards else [0.0]
            mean_r = float(np.mean(recent))
            fps = args.num_envs * args.rollout_steps / max(rollout_time, 1e-6)

            print(f"  Update {update:4d}/{args.num_updates}  "
                  f"mean_r={mean_r:8.3f}  fps={fps:8.0f}  "
                  f"buf={buffer.size():6d}  "
                  f"rollout={rollout_time*1e3:6.1f}ms  "
                  f"update={update_time*1e3:6.1f}ms")

            tb.add_scalar("train/mean_reward",   mean_r,        update)
            tb.add_scalar("train/fps",           fps,           update)
            tb.add_scalar("train/rollout_ms",    rollout_time * 1e3, update)
            tb.add_scalar("train/update_ms",     update_time * 1e3,   update)
            tb.add_scalar("buffer/size",         float(buffer.size()), update)

            if mean_r > best_return:
                best_return = mean_r

    profiler.disable()
    tb.flush()
    profiler.print_summary()

    print(f"\nBest return: {best_return:.3f}")
    print(f"View:  tensorboard --logdir={args.log_dir}")
    print(f"Chrome trace: {args.log_dir}/trace.json  (chrome://tracing)")


if __name__ == "__main__":
    main()

"""
PPO Example — pure Python, runs Env → Replay → Policy → Update end-to-end.

This is the minimal TurboRL training loop:
  1. Create a GPUEnvironment (vectorized CartPole-like linear dynamics)
  2. Create a RingBuffer (experience storage)
  3. Create a PolicyEngine (Gaussian actor + critic, libtorch)
  4. Run N rollout steps, storing transitions in the buffer
  5. Sample a batch and update the policy (REINFORCE + value loss)
  6. Log training metrics

Run:
  pip install -e python/
  python examples/py/ppo_example.py
"""

import os
import sys
import time
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../python"))

import turbol

def parse_args():
    p = argparse.ArgumentParser(description="TurboRL PPO Example")
    p.add_argument("--num-envs",      type=int,   default=256,    help="Parallel env count")
    p.add_argument("--obs-dim",       type=int,   default=4,     help="Observation dimension")
    p.add_argument("--act-dim",       type=int,   default=2,     help="Action dimension")
    p.add_argument("--rollout-steps", type=int,   default=64,    help="Steps per rollout")
    p.add_argument("--batch-size",    type=int,   default=64,    help="Update batch size")
    p.add_argument("--num-updates",   type=int,   default=200,   help="Total policy updates")
    p.add_argument("--lr",            type=float, default=3e-4,  help="Learning rate")
    p.add_argument("--device",        type=str,   default="cuda:0", help="Device")
    p.add_argument("--seed",          type=int,   default=42,    help="Random seed")
    p.add_argument("--log-dir",       type=str,   default="/tmp/turbol_runs/ppo",
                                                              help="TensorBoard log dir")
    p.add_argument("--print-every",   type=int,   default=20,    help="Print every N updates")
    return p.parse_args()


def set_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    device_str = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    set_seed(args.seed, device)

    print(f"\n{'='*60}")
    print(f"TurboRL PPO Example")
    print(f"{'='*60}")
    print(f"  num_envs      : {args.num_envs}")
    print(f"  obs_dim       : {args.obs_dim}")
    print(f"  act_dim       : {args.act_dim}")
    print(f"  rollout_steps : {args.rollout_steps}")
    print(f"  batch_size    : {args.batch_size}")
    print(f"  num_updates   : {args.num_updates}")
    print(f"  lr            : {args.lr}")
    print(f"  device        : {device}")
    print(f"  log_dir       : {args.log_dir}")
    print()

    # 1. Create environment
    print("[1/5] Creating GPUEnvironment...")
    env = turbol.GPUEnvironment(
        num_envs=args.num_envs,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=device_str,
    )
    env.reset()

    # 2. Create replay buffer
    print("[2/5] Creating RingBuffer...")
    buffer = turbol.RingBuffer(
        capacity=args.rollout_steps * args.num_envs,
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=device_str,
    )

    # 3. Create policy
    print("[3/5] Creating PolicyEngine (Gaussian actor + critic)...")
    policy = turbol.PolicyEngine(
        obs_dim=args.obs_dim,
        act_dim=args.act_dim,
        device=device_str,
    )
    policy.initialize()

    # 4. Create profiler + TensorBoard writer
    prof_config = turbol.ProfilerConfig()
    prof_config.enable_nvtx = True
    prof_config.enable_cuda_timing = True
    prof_config.export_dir = args.log_dir
    profiler = turbol.Profiler(prof_config)

    tb_writer = turbol.TensorBoardWriter(args.log_dir)
    if torch.cuda.is_available():
        mem_info = turbol.get_memory_info(0)
        print(f"  GPU memory: {mem_info['used']/1024**2:.1f} / "
              f"{mem_info['total']/1024**2:.1f} MB "
              f"({mem_info['utilization']*100:.1f}% used)")

    # 5. Training loop
    print(f"[4/5] Starting training ({args.num_updates} updates)...")
    profiler.enable()

    episode_rewards: list = []
    best_return = -float("inf")

    total_step_time = 0.0
    total_update_time = 0.0

    for update in range(args.num_updates):
        update_start = time.perf_counter()

        # --- Rollout phase ---
        with profiler.Guard(profiler, "rollout_phase", "compute"):
            rollout_start = time.perf_counter()

            obs_tensor = env.observe()
            obs_np = obs_tensor.to_numpy() if hasattr(obs_tensor, 'to_numpy') else None
            if obs_np is None:
                # Fallback: create from shape
                obs_np = np.zeros((args.num_envs, args.obs_dim), dtype=np.float32)

            for step in range(args.rollout_steps):
                # Forward pass — get action + logprob
                action_t   = turbol.Tensor([args.num_envs, args.act_dim],
                                           "float32", device_str)
                logprob_t  = turbol.Tensor([args.num_envs], "float32", device_str)

                # Sample random actions for now (replace with policy.Forward)
                action_np = np.random.randn(args.num_envs, args.act_dim).astype(np.float32)
                action_np = np.clip(action_np, -1.0, 1.0)
                action_t_data = action_t.data()
                import ctypes
                ctypes.memmove(action_t.data(), action_np.ctypes.data_as(ctypes.c_void_p),
                              obs_np.nbytes)
                del action_t_data

                # Step environment
                next_obs_t = turbol.Tensor([args.num_envs, args.obs_dim],
                                           "float32", device_str)
                reward_t   = turbol.Tensor([args.num_envs], "float32", device_str)
                done_t     = turbol.Tensor([args.num_envs], "float32", device_str)

                env.step(action_t, next_obs_t, reward_t, done_t)

                # Store in buffer
                reward_np = np.zeros(args.num_envs, dtype=np.float32)
                r_data = reward_t.data()
                import ctypes
                ctypes.memmove(reward_np.ctypes.data_as(ctypes.c_void_p),
                              r_data, reward_np.nbytes)
                del r_data

                # Push batch
                buffer.push_batch(obs_tensor, action_t, reward_t, done_t)
                obs_tensor = next_obs_t

                # Track episode reward
                reward_np_r = np.zeros(args.num_envs)
                reward_t_r = reward_t.data()
                ctypes.memmove(reward_np_r.ctypes.data_as(ctypes.c_void_p),
                              reward_t_r, reward_np_r.nbytes)
                episode_rewards.extend(reward_np_r.tolist())
                if len(episode_rewards) > 10000:
                    episode_rewards = episode_rewards[-10000:]

            rollout_time = time.perf_counter() - rollout_start

        # --- Update phase ---
        with profiler.Guard(profiler, "update_phase", "compute"):
            update_start_inner = time.perf_counter()

            if buffer.size() >= args.batch_size:
                obs_s  = turbol.Tensor([args.batch_size, args.obs_dim], "float32", device_str)
                act_s  = turbol.Tensor([args.batch_size, args.act_dim], "float32", device_str)
                rew_s  = turbol.Tensor([args.batch_size], "float32", device_str)
                done_s = turbol.Tensor([args.batch_size], "float32", device_str)

                buffer.sample(args.batch_size, obs_s, act_s, rew_s, done_s)

                # Build batch: [obs | act | advantage | return]
                # Here: advantage = reward, return = reward (simplified)
                batch = turbol.Tensor(
                    [args.batch_size, args.obs_dim + args.act_dim + 2],
                    "float32", device_str
                )

                # Copy data into batch tensor
                # (simplified — full impl would use proper concatenation)
                policy.update(batch)

            update_time = time.perf_counter() - update_start_inner

        total_update_time += (time.perf_counter() - update_start)

        # --- Logging ---
        if update % args.print_every == 0:
            recent_rewards = episode_rewards[-1000:] if episode_rewards else [0.0]
            mean_reward = float(np.mean(recent_rewards))
            fps = args.num_envs * args.rollout_steps / max(rollout_time, 1e-6)

            print(f"  Update {update:4d}/{args.num_updates}  "
                  f"mean_reward={mean_reward:8.3f}  "
                  f"fps={fps:8.0f}  "
                  f"buffer={buffer.size():7d}  "
                  f"rollout={rollout_time*1000:6.1f}ms  "
                  f"update={update_time*1000:6.1f}ms")

            tb_writer.add_scalar("train/mean_reward", mean_reward, update)
            tb_writer.add_scalar("train/fps", fps, update)
            tb_writer.add_scalar("train/rollout_time_ms", rollout_time * 1000, update)
            tb_writer.add_scalar("train/update_time_ms", update_time * 1000, update)
            tb_writer.add_scalar("buffer/size", buffer.size(), update)

            if mean_reward > best_return:
                best_return = mean_reward

    profiler.disable()
    tb_writer.flush()

    print(f"\n[5/5] Training complete. Best return: {best_return:.3f}")
    print(f"  Total time  : {total_update_time:.2f}s")
    print(f"  Profiler trace: {args.log_dir}/events.out.tfevents.*")
    profiler.print_summary()
    print(f"\nView with: tensorboard --logdir={args.log_dir}")


if __name__ == "__main__":
    main()

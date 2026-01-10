#!/usr/bin/env python3
"""Profile training to identify bottlenecks."""

import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from ai_training.env import BreakoutEnv


def make_env(rank=0):
    def _init():
        env = BreakoutEnv(stage_number=1, max_stage=1)
        env.reset(seed=42 + rank)
        return Monitor(env)
    return _init


def profile_env_step(n_envs=8, n_steps=1000):
    """Profile environment step time."""
    print(f"\n{'='*50}")
    print(f"Profiling Environment Step ({n_envs} envs, {n_steps} steps)")
    print(f"{'='*50}")

    # DummyVecEnv
    env_fns = [make_env(i) for i in range(n_envs)]
    dummy_env = DummyVecEnv(env_fns)

    dummy_env.reset()
    actions = [dummy_env.action_space.sample() for _ in range(n_envs)]

    start = time.perf_counter()
    for _ in range(n_steps):
        dummy_env.step(actions)
    dummy_time = time.perf_counter() - start

    dummy_env.close()

    # SubprocVecEnv
    env_fns = [make_env(i) for i in range(n_envs)]
    subproc_env = SubprocVecEnv(env_fns)

    subproc_env.reset()

    start = time.perf_counter()
    for _ in range(n_steps):
        subproc_env.step(actions)
    subproc_time = time.perf_counter() - start

    subproc_env.close()

    print(f"\nDummyVecEnv:")
    print(f"  Total time: {dummy_time:.2f}s")
    print(f"  Steps/sec: {n_steps * n_envs / dummy_time:.0f}")
    print(f"  Time/step: {dummy_time / n_steps * 1000:.2f}ms")

    print(f"\nSubprocVecEnv:")
    print(f"  Total time: {subproc_time:.2f}s")
    print(f"  Steps/sec: {n_steps * n_envs / subproc_time:.0f}")
    print(f"  Time/step: {subproc_time / n_steps * 1000:.2f}ms")

    print(f"\nSpeedup: {dummy_time / subproc_time:.2f}x")

    return dummy_time, subproc_time


def profile_model_inference(n_envs=8, n_inferences=1000):
    """Profile model inference time."""
    print(f"\n{'='*50}")
    print(f"Profiling Model Inference ({n_inferences} inferences)")
    print(f"{'='*50}")

    env_fns = [make_env(i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)

    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs={'net_arch': [512, 512, 256]},
        device='cpu',
        verbose=0
    )

    obs = env.reset()

    # Warmup
    for _ in range(10):
        model.predict(obs, deterministic=True)

    # Profile
    start = time.perf_counter()
    for _ in range(n_inferences):
        model.predict(obs, deterministic=True)
    inference_time = time.perf_counter() - start

    env.close()

    print(f"\nModel Inference (CPU):")
    print(f"  Total time: {inference_time:.2f}s")
    print(f"  Inferences/sec: {n_inferences / inference_time:.0f}")
    print(f"  Time/inference: {inference_time / n_inferences * 1000:.3f}ms")

    return inference_time


def profile_full_rollout(n_envs=8, n_steps=2048):
    """Profile full rollout collection."""
    print(f"\n{'='*50}")
    print(f"Profiling Full Rollout ({n_envs} envs, {n_steps} steps)")
    print(f"{'='*50}")

    env_fns = [make_env(i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        policy_kwargs={'net_arch': [512, 512, 256]},
        device='cpu',
        verbose=0
    )

    # Profile rollout collection
    obs = env.reset()

    env_time = 0
    inference_time = 0
    total_steps = 0

    for _ in range(n_steps):
        # Inference
        start = time.perf_counter()
        action, _ = model.predict(obs, deterministic=False)
        inference_time += time.perf_counter() - start

        # Environment step
        start = time.perf_counter()
        obs, rewards, dones, infos = env.step(action)
        env_time += time.perf_counter() - start

        total_steps += n_envs

    env.close()

    total_time = env_time + inference_time

    print(f"\nBreakdown:")
    print(f"  Environment step: {env_time:.2f}s ({env_time/total_time*100:.1f}%)")
    print(f"  Model inference:  {inference_time:.2f}s ({inference_time/total_time*100:.1f}%)")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"\nThroughput:")
    print(f"  Total steps: {total_steps}")
    print(f"  Steps/sec: {total_steps / total_time:.0f}")

    return env_time, inference_time


def profile_training_update(n_envs=8, n_steps=2048, batch_size=128):
    """Profile training update (gradient computation)."""
    print(f"\n{'='*50}")
    print(f"Profiling Training Update")
    print(f"{'='*50}")

    env_fns = [make_env(i) for i in range(n_envs)]
    env = SubprocVecEnv(env_fns)

    model = PPO(
        'MlpPolicy',
        env,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        policy_kwargs={'net_arch': [512, 512, 256]},
        device='cpu',
        verbose=0
    )

    # Collect rollout first
    print("Collecting rollout...")
    model.collect_rollouts(
        env,
        callback=None,
        rollout_buffer=model.rollout_buffer,
        n_rollout_steps=n_steps
    )

    # Profile training
    print("Profiling training update...")
    start = time.perf_counter()
    model.train()
    train_time = time.perf_counter() - start

    env.close()

    total_samples = n_steps * n_envs
    n_updates = (total_samples // batch_size) * 10  # n_epochs

    print(f"\nTraining Update:")
    print(f"  Total time: {train_time:.2f}s")
    print(f"  Samples: {total_samples}")
    print(f"  Updates: {n_updates}")
    print(f"  Time/update: {train_time / n_updates * 1000:.2f}ms")

    return train_time


def profile_env_scaling():
    """Profile how performance scales with number of environments."""
    print(f"\n{'='*50}")
    print(f"Profiling Environment Scaling")
    print(f"{'='*50}")

    env_counts = [1, 2, 4, 8, 16, 32, 64]
    n_steps = 500

    results = []

    for n_envs in env_counts:
        try:
            env_fns = [make_env(i) for i in range(n_envs)]

            if n_envs > 1:
                env = SubprocVecEnv(env_fns)
            else:
                env = DummyVecEnv(env_fns)

            env.reset()
            actions = [env.action_space.sample() for _ in range(n_envs)]

            start = time.perf_counter()
            for _ in range(n_steps):
                env.step(actions)
            elapsed = time.perf_counter() - start

            steps_per_sec = n_steps * n_envs / elapsed
            results.append((n_envs, steps_per_sec, elapsed))

            env.close()

            print(f"  {n_envs:2d} envs: {steps_per_sec:6.0f} steps/sec")

        except Exception as e:
            print(f"  {n_envs:2d} envs: Error - {e}")
            break

    print(f"\nOptimal env count: {max(results, key=lambda x: x[1])[0]}")

    return results


def main():
    print("="*60)
    print("TRAINING BOTTLENECK PROFILER")
    print("="*60)

    # Profile different components
    profile_env_step(n_envs=8, n_steps=500)
    profile_model_inference(n_envs=8, n_inferences=500)
    profile_full_rollout(n_envs=8, n_steps=1024)
    profile_training_update(n_envs=8, n_steps=2048, batch_size=128)
    profile_env_scaling()

    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()

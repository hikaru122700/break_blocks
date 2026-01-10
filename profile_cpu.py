#!/usr/bin/env python3
"""Profile CPU utilization and optimization options."""

import time
import os
import torch
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


def check_system_info():
    """Check system configuration."""
    print("="*60)
    print("SYSTEM INFO")
    print("="*60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CPU threads available: {os.cpu_count()}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"PyTorch interop threads: {torch.get_num_interop_threads()}")

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Check MKL/OpenBLAS
    print(f"\nBLAS info: {torch.__config__.show()[:500]}...")


def profile_thread_scaling():
    """Profile how inference scales with thread count."""
    print("\n" + "="*60)
    print("THREAD SCALING")
    print("="*60)

    env_fns = [make_env(i) for i in range(8)]
    env = DummyVecEnv(env_fns)

    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs={'net_arch': [512, 512, 256]},
        device='cpu',
        verbose=0
    )

    obs = env.reset()
    n_inferences = 500

    thread_counts = [1, 2, 4, 8, 16]

    for n_threads in thread_counts:
        torch.set_num_threads(n_threads)

        # Warmup
        for _ in range(10):
            model.predict(obs, deterministic=True)

        start = time.perf_counter()
        for _ in range(n_inferences):
            model.predict(obs, deterministic=True)
        elapsed = time.perf_counter() - start

        print(f"  {n_threads:2d} threads: {n_inferences/elapsed:.0f} inferences/sec "
              f"({elapsed/n_inferences*1000:.3f}ms/inference)")

    env.close()


def profile_batch_size_scaling():
    """Profile how inference scales with batch size."""
    print("\n" + "="*60)
    print("BATCH SIZE SCALING (Inference)")
    print("="*60)

    # Create a simple model for testing
    model = torch.nn.Sequential(
        torch.nn.Linear(216, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3)
    )
    model.eval()

    batch_sizes = [1, 8, 16, 32, 64, 128]
    n_samples = 4096  # Total samples to process

    for batch_size in batch_sizes:
        n_batches = n_samples // batch_size
        x = torch.randn(batch_size, 216)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                model(x)

        start = time.perf_counter()
        for _ in range(n_batches):
            with torch.no_grad():
                model(x)
        elapsed = time.perf_counter() - start

        samples_per_sec = n_samples / elapsed
        print(f"  Batch {batch_size:3d}: {samples_per_sec:.0f} samples/sec")


def profile_torch_compile():
    """Profile torch.compile optimization (PyTorch 2.0+)."""
    print("\n" + "="*60)
    print("TORCH.COMPILE OPTIMIZATION")
    print("="*60)

    # Create model
    model = torch.nn.Sequential(
        torch.nn.Linear(216, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3)
    )
    model.eval()

    x = torch.randn(8, 216)
    n_inferences = 1000

    # Without compile
    for _ in range(10):
        with torch.no_grad():
            model(x)

    start = time.perf_counter()
    for _ in range(n_inferences):
        with torch.no_grad():
            model(x)
    base_time = time.perf_counter() - start

    print(f"\nWithout torch.compile:")
    print(f"  {n_inferences/base_time:.0f} inferences/sec")

    # With compile
    try:
        compiled_model = torch.compile(model, mode="reduce-overhead")

        # Warmup (compile happens here)
        print("\nCompiling model (this may take a moment)...")
        for _ in range(10):
            with torch.no_grad():
                compiled_model(x)

        start = time.perf_counter()
        for _ in range(n_inferences):
            with torch.no_grad():
                compiled_model(x)
        compiled_time = time.perf_counter() - start

        print(f"\nWith torch.compile:")
        print(f"  {n_inferences/compiled_time:.0f} inferences/sec")
        print(f"  Speedup: {base_time/compiled_time:.2f}x")

    except Exception as e:
        print(f"\ntorch.compile not available or failed: {e}")


def profile_gpu_vs_cpu():
    """Compare GPU vs CPU inference."""
    print("\n" + "="*60)
    print("GPU vs CPU INFERENCE")
    print("="*60)

    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    # Create model
    model_cpu = torch.nn.Sequential(
        torch.nn.Linear(216, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3)
    )
    model_cpu.eval()

    model_gpu = torch.nn.Sequential(
        torch.nn.Linear(216, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 3)
    ).cuda()
    model_gpu.eval()

    batch_sizes = [8, 64, 256, 1024]
    n_inferences = 500

    print(f"\n{'Batch':>6} | {'CPU (inf/s)':>12} | {'GPU (inf/s)':>12} | {'Winner':>8}")
    print("-" * 50)

    for batch_size in batch_sizes:
        x_cpu = torch.randn(batch_size, 216)
        x_gpu = x_cpu.cuda()

        # CPU
        for _ in range(10):
            with torch.no_grad():
                model_cpu(x_cpu)

        start = time.perf_counter()
        for _ in range(n_inferences):
            with torch.no_grad():
                model_cpu(x_cpu)
        cpu_time = time.perf_counter() - start

        # GPU
        torch.cuda.synchronize()
        for _ in range(10):
            with torch.no_grad():
                model_gpu(x_gpu)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_inferences):
            with torch.no_grad():
                model_gpu(x_gpu)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        cpu_rate = n_inferences / cpu_time
        gpu_rate = n_inferences / gpu_time
        winner = "GPU" if gpu_rate > cpu_rate else "CPU"

        print(f"{batch_size:>6} | {cpu_rate:>12.0f} | {gpu_rate:>12.0f} | {winner:>8}")


def main():
    check_system_info()
    profile_thread_scaling()
    profile_batch_size_scaling()
    profile_torch_compile()
    profile_gpu_vs_cpu()

    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("""
Based on the profiling results:

1. If thread scaling shows improvement:
   - Increase torch threads: torch.set_num_threads(N)

2. If batch size scaling shows improvement:
   - Increase n_envs to match optimal batch size

3. If torch.compile shows speedup:
   - Can be integrated into training

4. If GPU wins for your batch size:
   - Use GPU despite the warning (it still works)

5. General optimizations:
   - Use SubprocVecEnv (already faster)
   - Increase frame_skip to reduce inference frequency
""")


if __name__ == '__main__':
    main()

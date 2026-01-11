#!/usr/bin/env python3
"""
CNN-based training for Break Blocks.

Uses image observations like Atari Breakout.
"""

import argparse
import os

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from ai_training.env import ImageBreakoutEnv


class WinRateCallback(BaseCallback):
    """Track win rate during training."""

    def __init__(self, log_freq: int = 50000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.episodes = 0
        self.recent_wins = []

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals.get('dones', [])):
            if done:
                self.episodes += 1
                info = self.locals['infos'][i]
                is_win = info.get('is_stage_clear', False)
                self.recent_wins.append(1 if is_win else 0)
                if is_win:
                    self.wins += 1
                if len(self.recent_wins) > 100:
                    self.recent_wins.pop(0)

        if self.num_timesteps % self.log_freq == 0 and self.episodes > 0:
            total_wr = self.wins / self.episodes * 100
            recent_wr = sum(self.recent_wins) / len(self.recent_wins) * 100 if self.recent_wins else 0
            print(f"[{self.num_timesteps:,}] Episodes: {self.episodes}, "
                  f"Win Rate: {total_wr:.1f}% (recent: {recent_wr:.1f}%)")

        return True


def make_env(rank: int, seed: int = 42):
    """Create image-based environment."""
    def _init():
        env = ImageBreakoutEnv(stage_number=1, max_stage=1)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    output_dir: str = './models/cnn',
    device: str = 'auto'
):
    """Train CNN policy like Atari."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    print(f"Creating {n_envs} image-based environments...")

    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Transpose for PyTorch (HWC -> CHW)
    env = VecTransposeImage(env)

    # Frame stacking (4 frames like Atari)
    env = VecFrameStack(env, n_stack=4)

    print("Environment observation space:", env.observation_space)

    # Atari-style hyperparameters
    print("\nCreating CNN model with Atari hyperparameters...")
    print("  Policy: CnnPolicy")
    print("  n_steps: 128")
    print("  batch_size: 256")
    print("  n_epochs: 4")
    print("  learning_rate: 2.5e-4")
    print("  clip_range: 0.1")
    print("  Frame stack: 4")

    model = PPO(
        'CnnPolicy',
        env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=os.path.join(output_dir, 'logs'),
        device=device,
        verbose=1
    )

    # Callbacks
    callbacks = [
        WinRateCallback(log_freq=50000),
        CheckpointCallback(
            save_freq=500_000 // n_envs,
            save_path=os.path.join(output_dir, 'checkpoints'),
            name_prefix='cnn'
        )
    ]

    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"Device: {device}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted")

    model.save(os.path.join(output_dir, 'final_model'))
    print(f"\nModel saved to {output_dir}/final_model")

    env.close()


def evaluate(model_path: str, n_episodes: int = 100):
    """Evaluate CNN model."""
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = DummyVecEnv([make_env(0)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = PPO.load(model_path)

    wins = 0
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done = done[0]

        if info[0].get('is_stage_clear', False):
            wins += 1
            print(f"Episode {ep+1}: WIN")
        else:
            print(f"Episode {ep+1}: LOSE")

    env.close()
    print(f"\nWin Rate: {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")


def main():
    parser = argparse.ArgumentParser(description='CNN training for Break Blocks')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--timesteps', type=int, default=10_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--output', type=str, default='./models/cnn')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.mode == 'train':
        train(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            output_dir=args.output,
            device=args.device
        )
    else:
        if not args.model:
            print("Error: --model required for eval")
            return
        evaluate(args.model)


if __name__ == '__main__':
    main()

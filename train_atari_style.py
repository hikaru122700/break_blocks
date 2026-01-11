#!/usr/bin/env python3
"""
Training script using Atari Breakout-style hyperparameters.

Based on successful sb3/ppo-BreakoutNoFrameskip-v4 model:
- Mean reward: 398 on Atari Breakout
- 10M timesteps training
"""

import argparse
import os
from datetime import datetime

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import get_linear_fn

from ai_training.env import BreakoutEnv


class WinRateCallback(BaseCallback):
    """Track and log win rate during training."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.episodes = 0
        self.recent_wins = []
        self.window = 100

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'is_stage_clear' in info:
                self.episodes += 1
                is_win = info.get('is_stage_clear', False)
                self.recent_wins.append(1 if is_win else 0)
                if is_win:
                    self.wins += 1

                # Keep window size
                if len(self.recent_wins) > self.window:
                    self.recent_wins.pop(0)

        # Log progress
        if self.num_timesteps % self.log_freq == 0 and self.episodes > 0:
            total_wr = self.wins / self.episodes * 100
            recent_wr = sum(self.recent_wins) / len(self.recent_wins) * 100 if self.recent_wins else 0
            print(f"[{self.num_timesteps:,} steps] Episodes: {self.episodes}, "
                  f"Win Rate: {total_wr:.1f}% (recent {self.window}: {recent_wr:.1f}%)")

        return True


def make_env(rank: int, seed: int = 42):
    """Create environment."""
    def _init():
        env = BreakoutEnv(stage_number=1, max_stage=1)
        env.reset(seed=seed + rank)
        return env
    return _init


def train(
    total_timesteps: int = 10_000_000,
    n_envs: int = 8,
    output_dir: str = './models/atari_style',
    resume_from: str = None,
    device: str = 'auto'
):
    """
    Train using Atari Breakout-style hyperparameters.

    Key settings from successful Atari model:
    - n_steps=128 (frequent updates)
    - n_epochs=4 (fewer epochs)
    - Linear decay on learning rate and clip range
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Create environments
    print(f"Creating {n_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Atari-style hyperparameters with linear decay
    learning_rate = get_linear_fn(2.5e-4, 0, 1.0)  # Decay from 2.5e-4 to 0
    clip_range = get_linear_fn(0.1, 0, 1.0)  # Decay from 0.1 to 0

    if resume_from:
        print(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
        # Update learning rate schedule for continued training
        model.learning_rate = learning_rate
        model.clip_range = clip_range
    else:
        print("Creating new model with Atari-style hyperparameters...")
        print(f"  n_steps: 128")
        print(f"  n_epochs: 4")
        print(f"  batch_size: 256")
        print(f"  learning_rate: 2.5e-4 (linear decay)")
        print(f"  clip_range: 0.1 (linear decay)")
        print(f"  ent_coef: 0.01")

        # Use larger network for our vector observation
        policy_kwargs = {
            'net_arch': [256, 256]  # 2 layers of 256 (simpler than before)
        }

        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=128,  # Key: frequent updates like Atari
            batch_size=256,
            n_epochs=4,  # Key: fewer epochs
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=clip_range,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
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
            name_prefix='atari_style'
        )
    ]

    # Train
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model.save(os.path.join(output_dir, f'interrupted_{timestamp}'))
        print(f"Saved to {output_dir}/interrupted_{timestamp}")
        env.close()
        return

    # Save final model
    model.save(os.path.join(output_dir, 'final_model'))
    print(f"\nTraining complete! Model saved to {output_dir}/final_model")

    env.close()


def evaluate(model_path: str, n_episodes: int = 100):
    """Evaluate trained model."""
    model = PPO.load(model_path)
    env = BreakoutEnv(stage_number=1, max_stage=1)

    wins = 0
    total_reward = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_reward += ep_reward
        if info.get('is_stage_clear', False):
            wins += 1
            print(f"Episode {ep+1}: WIN (reward: {ep_reward:.1f})")
        else:
            print(f"Episode {ep+1}: LOSE (reward: {ep_reward:.1f})")

    env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation Results")
    print(f"{'='*50}")
    print(f"Win Rate: {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")
    print(f"Average Reward: {total_reward/n_episodes:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Atari-style PPO training')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                        help='Total timesteps (default: 10M like Atari)')
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--output', type=str, default='./models/atari_style')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, help='Model path for eval')
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.mode == 'train':
        train(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            output_dir=args.output,
            resume_from=args.resume,
            device=args.device
        )
    else:
        if not args.model:
            print("Error: --model required for evaluation")
            return
        evaluate(args.model, args.eval_episodes)


if __name__ == '__main__':
    main()

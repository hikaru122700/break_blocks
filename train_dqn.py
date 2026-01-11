#!/usr/bin/env python3
"""
DQN-based training for Break Blocks.

DQN is historically the best algorithm for Breakout-style games.
Uses CNN policy with image observations like Atari.
"""

import argparse
import os
from datetime import datetime

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

from ai_training.env import ImageBreakoutEnv


class WinRateCallback(BaseCallback):
    """Track win rate during training."""

    def __init__(self, log_freq: int = 10000, verbose: int = 1):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.wins = 0
        self.episodes = 0
        self.recent_wins = []
        self.best_recent_wr = 0

    def _on_step(self) -> bool:
        # Check for episode completion
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

            # Track best
            if recent_wr > self.best_recent_wr:
                self.best_recent_wr = recent_wr

            print(f"[{self.num_timesteps:,}] Episodes: {self.episodes}, "
                  f"Win Rate: {total_wr:.1f}% (recent: {recent_wr:.1f}%, best: {self.best_recent_wr:.1f}%)")

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
    output_dir: str = './models/dqn',
    resume_from: str = None,
    device: str = 'auto'
):
    """
    Train DQN with CNN policy.

    DQN hyperparameters based on DeepMind Atari paper:
    - Experience replay buffer: 100K-1M
    - Target network update: every 1000-10000 steps
    - Exploration: epsilon 1.0 -> 0.01 over 10% of training
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    print("Creating image-based environment...")

    # DQN uses single env (not vectorized for standard impl)
    # But we can use DummyVecEnv for frame stacking
    env = DummyVecEnv([make_env(0)])

    # Transpose for PyTorch (HWC -> CHW)
    env = VecTransposeImage(env)

    # Frame stacking (4 frames like Atari)
    env = VecFrameStack(env, n_stack=4)

    print("Environment observation space:", env.observation_space)

    if resume_from:
        print(f"Resuming from {resume_from}")
        model = DQN.load(resume_from, env=env, device=device)
    else:
        # DQN hyperparameters (Atari-style)
        print("\nCreating DQN model with Atari hyperparameters...")
        print("  Policy: CnnPolicy")
        print("  buffer_size: 100,000")
        print("  learning_starts: 10,000")
        print("  batch_size: 32")
        print("  target_update_interval: 1,000")
        print("  exploration_fraction: 0.1")
        print("  exploration_final_eps: 0.01")
        print("  Frame stack: 4")

        model = DQN(
            'CnnPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=100_000,          # Experience replay buffer
            learning_starts=10_000,        # Start learning after this many steps
            batch_size=32,                 # Minibatch size
            tau=1.0,                       # Hard update (no soft update)
            target_update_interval=1_000,  # Update target network every N steps
            train_freq=4,                  # Update every 4 steps
            gradient_steps=1,              # Gradient steps per update
            exploration_fraction=0.1,      # Fraction of training for exploration decay
            exploration_initial_eps=1.0,   # Initial epsilon
            exploration_final_eps=0.01,    # Final epsilon
            max_grad_norm=10,
            tensorboard_log=os.path.join(output_dir, 'logs'),
            device=device,
            verbose=1
        )

    # Callbacks
    callbacks = [
        WinRateCallback(log_freq=10000),
        CheckpointCallback(
            save_freq=100_000,
            save_path=os.path.join(output_dir, 'checkpoints'),
            name_prefix='dqn'
        )
    ]

    print(f"\nStarting DQN training for {total_timesteps:,} timesteps...")
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

    model.save(os.path.join(output_dir, 'final_model'))
    print(f"\nModel saved to {output_dir}/final_model")

    env.close()


def evaluate(model_path: str, n_episodes: int = 100):
    """Evaluate DQN model."""
    env = DummyVecEnv([make_env(0)])
    env = VecTransposeImage(env)
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load(model_path)

    wins = 0
    total_reward = 0

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            done = done[0]

        total_reward += ep_reward
        if info[0].get('is_stage_clear', False):
            wins += 1
            print(f"Episode {ep+1}: WIN (reward: {ep_reward:.1f})")
        else:
            print(f"Episode {ep+1}: LOSE (reward: {ep_reward:.1f})")

    env.close()

    print(f"\n{'='*50}")
    print(f"Evaluation Results (DQN)")
    print(f"{'='*50}")
    print(f"Win Rate: {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")
    print(f"Average Reward: {total_reward/n_episodes:.1f}")


def main():
    parser = argparse.ArgumentParser(description='DQN training for Break Blocks')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--timesteps', type=int, default=10_000_000,
                        help='Total timesteps (default: 10M)')
    parser.add_argument('--output', type=str, default='./models/dqn')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to model to resume from')
    parser.add_argument('--model', type=str, default=None,
                        help='Model path for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    if args.mode == 'train':
        train(
            total_timesteps=args.timesteps,
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

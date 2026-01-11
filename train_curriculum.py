#!/usr/bin/env python3
"""Curriculum Learning training for Break Blocks.

Starts with easy mode and gradually increases difficulty.
"""

import argparse
import os
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from ai_training.env import EasyBreakoutEnv, BreakoutEnv


# Curriculum phases: (block_rows, ball_speed_multiplier, time_multiplier, steps)
CURRICULUM_PHASES = [
    # Phase 0: Very easy - 1 row, very slow ball
    {"block_rows": 1, "ball_speed": 0.6, "time_mult": 3.0, "steps": 500_000},
    # Phase 1: Easy - 1 row, slow ball
    {"block_rows": 1, "ball_speed": 0.8, "time_mult": 2.0, "steps": 500_000},
    # Phase 2: Medium easy - 2 rows, slow ball
    {"block_rows": 2, "ball_speed": 0.8, "time_mult": 2.0, "steps": 1_000_000},
    # Phase 3: Medium - 2 rows, normal ball
    {"block_rows": 2, "ball_speed": 1.0, "time_mult": 1.5, "steps": 1_000_000},
    # Phase 4: Full game - 3 rows, normal speed
    {"block_rows": 3, "ball_speed": 1.0, "time_mult": 1.0, "steps": 2_000_000},
]


class CurriculumCallback(BaseCallback):
    """Callback to track wins and log curriculum progress."""

    def __init__(self, phase_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.phase_name = phase_name
        self.wins = 0
        self.episodes = 0

    def _on_step(self) -> bool:
        # Check for episode ends
        for info in self.locals.get('infos', []):
            if 'is_stage_clear' in info:
                self.episodes += 1
                if info.get('is_stage_clear', False):
                    self.wins += 1

                if self.episodes > 0 and self.episodes % 100 == 0:
                    win_rate = self.wins / self.episodes * 100
                    print(f"[{self.phase_name}] Episodes: {self.episodes}, Win Rate: {win_rate:.1f}%")
        return True


def make_easy_env(block_rows: int, ball_speed: float, time_mult: float, rank: int, seed: int = 0):
    """Create an easy environment with specified difficulty."""
    def _init():
        env = EasyBreakoutEnv(
            block_rows=block_rows,
            ball_speed_multiplier=ball_speed,
            time_multiplier=time_mult,
            stage_number=1,
            max_stage=1
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def make_normal_env(rank: int, seed: int = 0):
    """Create normal environment."""
    def _init():
        env = BreakoutEnv(stage_number=1, max_stage=1)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_curriculum(
    output_dir: str = './models/curriculum',
    n_envs: int = 8,
    seed: int = 42,
    resume_phase: int = 0,
    resume_model: str = None,
    device: str = 'auto'
):
    """
    Train using curriculum learning.

    Args:
        output_dir: Directory to save models
        n_envs: Number of parallel environments
        seed: Random seed
        resume_phase: Phase to resume from (0-4)
        resume_model: Path to model to resume from
        device: Device to use
    """
    set_random_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    model = None
    total_steps = 0

    for phase_idx, phase in enumerate(CURRICULUM_PHASES):
        if phase_idx < resume_phase:
            continue

        phase_name = f"Phase_{phase_idx}"
        print(f"\n{'='*60}")
        print(f"Starting {phase_name}")
        print(f"  Blocks: {phase['block_rows']} rows")
        print(f"  Ball Speed: {phase['ball_speed']}x")
        print(f"  Time: {phase['time_mult']}x")
        print(f"  Steps: {phase['steps']:,}")
        print(f"{'='*60}\n")

        # Create environments for this phase
        if phase['block_rows'] < 3:
            env = SubprocVecEnv([
                make_easy_env(
                    phase['block_rows'],
                    phase['ball_speed'],
                    phase['time_mult'],
                    i, seed
                ) for i in range(n_envs)
            ])
        else:
            # Full game
            env = SubprocVecEnv([
                make_normal_env(i, seed) for i in range(n_envs)
            ])

        # Create or continue model
        if model is None:
            if resume_model and phase_idx == resume_phase:
                print(f"Resuming from {resume_model}")
                model = PPO.load(resume_model, env=env, device=device)
            else:
                model = PPO(
                    'MlpPolicy',
                    env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    verbose=1,
                    tensorboard_log=os.path.join(output_dir, 'logs'),
                    device=device
                )
        else:
            # Update environment for existing model
            model.set_env(env)

        # Callbacks
        curriculum_callback = CurriculumCallback(phase_name)
        checkpoint_callback = CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=os.path.join(output_dir, 'checkpoints'),
            name_prefix=f"phase_{phase_idx}"
        )

        # Train
        model.learn(
            total_timesteps=phase['steps'],
            callback=[curriculum_callback, checkpoint_callback],
            reset_num_timesteps=False,
            tb_log_name=f"phase_{phase_idx}",
            progress_bar=True
        )

        total_steps += phase['steps']

        # Save phase model
        phase_model_path = os.path.join(output_dir, f"phase_{phase_idx}_model")
        model.save(phase_model_path)
        print(f"\nSaved {phase_name} model to {phase_model_path}")

        # Final stats
        if curriculum_callback.episodes > 0:
            final_win_rate = curriculum_callback.wins / curriculum_callback.episodes * 100
            print(f"{phase_name} Final Stats:")
            print(f"  Episodes: {curriculum_callback.episodes}")
            print(f"  Win Rate: {final_win_rate:.1f}%")

        env.close()

    # Save final model
    final_path = os.path.join(output_dir, 'final_model')
    model.save(final_path)
    print(f"\n{'='*60}")
    print(f"Curriculum training complete!")
    print(f"Total steps: {total_steps:,}")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}")


def evaluate_model(model_path: str, n_episodes: int = 100, easy_mode: bool = False):
    """Evaluate a trained model."""
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    if easy_mode:
        env = EasyBreakoutEnv(block_rows=1, ball_speed_multiplier=0.7, stage_number=1, max_stage=1)
    else:
        env = BreakoutEnv(stage_number=1, max_stage=1)

    wins = 0
    total_reward = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward
        if info.get('is_stage_clear', False):
            wins += 1

    env.close()

    print(f"\nEvaluation Results ({'Easy Mode' if easy_mode else 'Normal Mode'}):")
    print(f"  Episodes: {n_episodes}")
    print(f"  Win Rate: {wins/n_episodes*100:.1f}%")
    print(f"  Average Reward: {total_reward/n_episodes:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Curriculum Learning for Break Blocks')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval'],
                        help='Mode: train or eval')
    parser.add_argument('--output', type=str, default='./models/curriculum',
                        help='Output directory for models')
    parser.add_argument('--envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--resume-phase', type=int, default=0,
                        help='Phase to resume from (0-4)')
    parser.add_argument('--resume-model', type=str, default=None,
                        help='Model path to resume from')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Model path for evaluation')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Evaluation episodes')
    parser.add_argument('--easy', action='store_true',
                        help='Evaluate on easy mode')

    args = parser.parse_args()

    if args.mode == 'train':
        train_curriculum(
            output_dir=args.output,
            n_envs=args.envs,
            seed=args.seed,
            resume_phase=args.resume_phase,
            resume_model=args.resume_model,
            device=args.device
        )
    elif args.mode == 'eval':
        if not args.model_path:
            print("Error: --model-path required for evaluation")
            return
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.eval_episodes,
            easy_mode=args.easy
        )


if __name__ == '__main__':
    main()

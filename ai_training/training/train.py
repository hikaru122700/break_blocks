"""Main training script for Break Blocks RL agent."""

import os
import argparse
from typing import Optional
import yaml

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

from ..env import BreakoutEnv, CurriculumBreakoutEnv
from .callbacks import TrainingCallback, CurriculumCallback, BestModelCallback


def make_env(
    stage_number: int = 1,
    max_stage: int = 10,
    time_penalty_scale: float = 1.0,
    rank: int = 0,
    seed: int = 0
):
    """
    Create environment factory function.

    Args:
        stage_number: Starting stage
        max_stage: Maximum stage for curriculum
        time_penalty_scale: Time penalty multiplier
        rank: Environment rank for seeding
        seed: Base random seed

    Returns:
        Function that creates environment
    """
    def _init():
        env = BreakoutEnv(
            stage_number=stage_number,
            max_stage=max_stage,
            time_penalty_scale=time_penalty_scale
        )
        env.reset(seed=seed + rank)
        return Monitor(env)
    return _init


def train(
    config_path: Optional[str] = None,
    total_timesteps: int = 10000000,
    n_envs: int = 8,
    save_path: str = './models',
    log_path: str = './logs',
    resume_from: Optional[str] = None,
    device: str = 'auto',
    verbose: int = 1
):
    """
    Train PPO agent on Break Blocks.

    Args:
        config_path: Path to config YAML file
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_path: Path to save models
        log_path: Path for tensorboard logs
        resume_from: Path to resume training from
        device: Device to use ('auto', 'cuda', 'cpu')
        verbose: Verbosity level
    """
    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

    # Default hyperparameters
    ppo_params = config.get('ppo', {
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
    })

    policy_kwargs = config.get('policy_kwargs', {
        'net_arch': [256, 256, 128]
    })

    curriculum = config.get('curriculum', {
        'initial_max_stage': 1,
        'phase_steps': [500000, 1500000, 3500000, 6500000, 10000000],
        'time_penalty_scales': [1.0, 1.0, 1.0, 1.0, 2.0]
    })

    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'best_model'), exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Create environments
    if verbose:
        print(f"Creating {n_envs} parallel environments...")

    env_fns = [
        make_env(
            stage_number=1,
            max_stage=curriculum['initial_max_stage'],
            time_penalty_scale=curriculum['time_penalty_scales'][0],
            rank=i,
            seed=42
        )
        for i in range(n_envs)
    ]

    # Use SubprocVecEnv for parallel processing (faster but more memory)
    # Use DummyVecEnv for debugging
    if n_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(stage_number=1, max_stage=10)])

    # Create or load model
    if resume_from and os.path.exists(resume_from):
        if verbose:
            print(f"Resuming from {resume_from}")
        model = PPO.load(resume_from, env=env, device=device)
    else:
        if verbose:
            print("Creating new model...")
            print(f"PPO params: {ppo_params}")
            print(f"Policy architecture: {policy_kwargs}")

        model = PPO(
            'MlpPolicy',
            env,
            **ppo_params,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_path,
            device=device,
            verbose=verbose
        )

    # Create callbacks
    callbacks = CallbackList([
        TrainingCallback(
            log_freq=5000,
            save_freq=100000,
            save_path=os.path.join(save_path, 'checkpoints'),
            verbose=verbose
        ),
        CurriculumCallback(
            phase_steps=curriculum['phase_steps'],
            verbose=verbose
        ),
        BestModelCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(save_path, 'best_model'),
            log_path=log_path,
            eval_freq=20000,
            n_eval_episodes=10,
            verbose=verbose
        )
    ])

    # Train
    if verbose:
        print(f"\nStarting training for {total_timesteps:,} timesteps...")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        # Save final model
        final_path = os.path.join(save_path, 'final_model')
        model.save(final_path)
        if verbose:
            print(f"\nFinal model saved to {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train Break Blocks RL agent')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file')
    parser.add_argument('--timesteps', type=int, default=10000000,
                        help='Total training timesteps')
    parser.add_argument('--envs', type=int, default=8,
                        help='Number of parallel environments')
    parser.add_argument('--save-path', type=str, default='./models',
                        help='Path to save models')
    parser.add_argument('--log-path', type=str, default='./logs',
                        help='Path for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to resume training from')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Verbosity level')

    args = parser.parse_args()

    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_path=args.save_path,
        log_path=args.log_path,
        resume_from=args.resume,
        device=args.device,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()

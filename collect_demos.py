#!/usr/bin/env python3
"""Collect expert demonstrations for imitation learning."""

import argparse
import os
import numpy as np
from tqdm import tqdm
from typing import List, Tuple

from ai_training.env import BreakoutEnv
from ai_training.expert import ExpertAI, ExpertAIWithNoise


def collect_demonstrations(
    n_episodes: int = 1000,
    stage: int = 1,
    noise_level: float = 0.1,
    output_path: str = './demos',
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect demonstrations from expert AI.

    Args:
        n_episodes: Number of episodes to collect
        stage: Stage to play
        noise_level: Random action probability (for diversity)
        output_path: Where to save demonstrations
        verbose: Print progress

    Returns:
        Tuple of (observations, actions) arrays
    """
    env = BreakoutEnv(stage_number=stage, max_stage=stage)
    expert = ExpertAIWithNoise(noise_level=noise_level)

    all_observations = []
    all_actions = []

    wins = 0
    total_reward = 0

    episodes = tqdm(range(n_episodes), desc="Collecting demos") if verbose else range(n_episodes)

    for ep in episodes:
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            # Get expert action
            action = expert.get_action(env.game)

            # Store observation and action
            all_observations.append(obs.copy())
            all_actions.append(action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward
        if info.get('is_stage_clear', False):
            wins += 1

        if verbose and (ep + 1) % 100 == 0:
            tqdm.write(f"Episode {ep+1}: Win rate: {wins/(ep+1)*100:.1f}%, "
                      f"Avg reward: {total_reward/(ep+1):.1f}")

    env.close()

    # Convert to numpy arrays
    observations = np.array(all_observations, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.int64)

    # Save demonstrations
    os.makedirs(output_path, exist_ok=True)

    obs_path = os.path.join(output_path, 'observations.npy')
    act_path = os.path.join(output_path, 'actions.npy')

    np.save(obs_path, observations)
    np.save(act_path, actions)

    if verbose:
        print(f"\n{'='*50}")
        print(f"Demonstration Collection Complete")
        print(f"{'='*50}")
        print(f"Episodes: {n_episodes}")
        print(f"Total transitions: {len(observations)}")
        print(f"Win rate: {wins/n_episodes*100:.1f}%")
        print(f"Average reward: {total_reward/n_episodes:.1f}")
        print(f"Saved to: {output_path}")
        print(f"  - observations.npy: {observations.shape}")
        print(f"  - actions.npy: {actions.shape}")

    return observations, actions


def evaluate_expert(n_episodes: int = 100, stage: int = 1, verbose: bool = True):
    """Evaluate expert AI performance."""
    env = BreakoutEnv(stage_number=stage, max_stage=stage)
    expert = ExpertAI()  # No noise for evaluation

    wins = 0
    total_reward = 0
    total_blocks = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = expert.get_action(env.game)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward
        total_blocks += info.get('blocks_destroyed', 0)
        if info.get('is_stage_clear', False):
            wins += 1

        if verbose:
            status = "WIN" if info.get('is_stage_clear', False) else "LOSE"
            print(f"Episode {ep+1}: {status} | Reward: {episode_reward:.1f} | "
                  f"Blocks: {info.get('blocks_destroyed', 0)}")

    env.close()

    print(f"\n{'='*50}")
    print(f"Expert AI Evaluation (Stage {stage})")
    print(f"{'='*50}")
    print(f"Win Rate: {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")
    print(f"Average Reward: {total_reward/n_episodes:.1f}")
    print(f"Average Blocks: {total_blocks/n_episodes:.1f}")


def main():
    parser = argparse.ArgumentParser(description='Collect expert demonstrations')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to collect')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage to play')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Random action probability (0-1)')
    parser.add_argument('--output', type=str, default='./demos',
                        help='Output directory')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate expert, do not collect demos')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Episodes for evaluation')

    args = parser.parse_args()

    if args.eval_only:
        evaluate_expert(
            n_episodes=args.eval_episodes,
            stage=args.stage
        )
    else:
        collect_demonstrations(
            n_episodes=args.episodes,
            stage=args.stage,
            noise_level=args.noise,
            output_path=args.output
        )


if __name__ == '__main__':
    main()

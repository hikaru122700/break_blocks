#!/usr/bin/env python3
"""Evaluate trained Break Blocks RL agent."""

import argparse
from stable_baselines3 import PPO
from ai_training.env import BreakoutEnv


def evaluate(
    model_path: str = './models/final_model',
    n_episodes: int = 10,
    stage: int = 1,
    verbose: bool = True
):
    """
    Evaluate trained model.

    Args:
        model_path: Path to saved model
        n_episodes: Number of episodes to run
        stage: Stage to evaluate on
        verbose: Print detailed info
    """
    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = BreakoutEnv(stage_number=stage, max_stage=stage)

    # Statistics
    total_rewards = []
    wins = 0
    total_blocks = 0

    for ep in range(n_episodes):
        obs, info = env.reset(options={'stage': stage})
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1

        total_rewards.append(episode_reward)
        if info.get('is_stage_clear', False):
            wins += 1
        total_blocks += info.get('blocks_destroyed', 0)

        if verbose:
            status = "WIN" if info.get('is_stage_clear', False) else "LOSE"
            print(f"Episode {ep+1}: {status} | Reward: {episode_reward:.2f} | "
                  f"Blocks: {info.get('blocks_destroyed', 0)} | Steps: {steps}")

    # Summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    win_rate = wins / n_episodes
    avg_blocks = total_blocks / n_episodes

    print(f"\n{'='*50}")
    print(f"Results for Stage {stage} ({n_episodes} episodes)")
    print(f"{'='*50}")
    print(f"Win Rate: {win_rate:.1%} ({wins}/{n_episodes})")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Blocks Destroyed: {avg_blocks:.1f}")

    env.close()
    return {'win_rate': win_rate, 'avg_reward': avg_reward, 'avg_blocks': avg_blocks}


def main():
    parser = argparse.ArgumentParser(description='Evaluate Break Blocks RL agent')
    parser.add_argument('--model', type=str, default='./models/final_model',
                        help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage to evaluate on')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress per-episode output')

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        n_episodes=args.episodes,
        stage=args.stage,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

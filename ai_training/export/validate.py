"""Validate ONNX model against PyTorch model."""

import argparse
import json
from typing import Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
from stable_baselines3 import PPO

from ..env.constants import OBSERVATION_DIM
from ..env import BreakoutEnv


def validate_onnx(
    pytorch_model_path: str,
    onnx_model_path: str,
    n_samples: int = 100,
    tolerance: float = 1e-5,
    verbose: bool = True
) -> Tuple[bool, float]:
    """
    Validate ONNX model outputs match PyTorch model.

    Args:
        pytorch_model_path: Path to PyTorch model
        onnx_model_path: Path to ONNX model
        n_samples: Number of random samples to test
        tolerance: Maximum allowed difference
        verbose: Print progress

    Returns:
        Tuple of (is_valid, max_difference)
    """
    if verbose:
        print("Loading PyTorch model...")

    # Load PyTorch model
    pytorch_model = PPO.load(pytorch_model_path, device='cpu')
    policy = pytorch_model.policy
    policy.eval()

    if verbose:
        print("Loading ONNX model...")

    # Load ONNX model
    ort_session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider']
    )

    if verbose:
        print(f"Running validation with {n_samples} samples...")

    max_diff = 0.0
    all_match = True

    # Generate random observations
    np.random.seed(42)
    for i in range(n_samples):
        # Random observation
        obs = np.random.randn(1, OBSERVATION_DIM).astype(np.float32)
        obs_tensor = torch.from_numpy(obs)

        # PyTorch inference
        with torch.no_grad():
            features = policy.extract_features(obs_tensor)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            action_logits = policy.action_net(latent_pi)
            pytorch_probs = torch.softmax(action_logits, dim=-1).numpy()

        # ONNX inference
        onnx_probs = ort_session.run(
            ['action_probs'],
            {'observation': obs}
        )[0]

        # Compare
        diff = np.abs(pytorch_probs - onnx_probs).max()
        max_diff = max(max_diff, diff)

        if diff > tolerance:
            all_match = False
            if verbose:
                print(f"Sample {i}: Difference {diff:.6f} exceeds tolerance")

    if verbose:
        print(f"\nValidation complete:")
        print(f"  Max difference: {max_diff:.6e}")
        print(f"  Tolerance: {tolerance:.6e}")
        print(f"  Status: {'PASSED' if all_match else 'FAILED'}")

    return all_match, max_diff


def validate_gameplay(
    onnx_model_path: str,
    n_episodes: int = 5,
    verbose: bool = True
) -> dict:
    """
    Validate ONNX model by playing actual games.

    Args:
        onnx_model_path: Path to ONNX model
        n_episodes: Number of episodes to play
        verbose: Print progress

    Returns:
        Dictionary with gameplay statistics
    """
    if verbose:
        print("Loading ONNX model for gameplay validation...")

    # Load ONNX model
    ort_session = ort.InferenceSession(
        onnx_model_path,
        providers=['CPUExecutionProvider']
    )

    # Create environment
    env = BreakoutEnv(stage_number=1, max_stage=10)

    results = {
        'episodes': n_episodes,
        'wins': 0,
        'total_score': 0,
        'total_blocks': 0,
        'avg_time': 0.0,
    }

    total_time = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset(options={'stage': 1})
        done = False
        ep_score = 0
        ep_blocks = 0

        while not done:
            # Get action from ONNX model
            action_probs = ort_session.run(
                ['action_probs'],
                {'observation': obs.reshape(1, -1)}
            )[0][0]

            # Select action (deterministic: argmax)
            action = int(np.argmax(action_probs))

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_score += info.get('score', 0) - ep_score
            ep_blocks = info.get('blocks_destroyed', 0)

        # Record results
        if info.get('is_stage_clear', False):
            results['wins'] += 1

        results['total_score'] += ep_score
        results['total_blocks'] += ep_blocks
        total_time += info.get('time_elapsed', 0)

        if verbose:
            status = 'WIN' if info.get('is_stage_clear') else 'LOSS'
            print(f"Episode {ep + 1}: {status}, Score: {ep_score}, Blocks: {ep_blocks}")

    env.close()

    results['avg_time'] = total_time / n_episodes if n_episodes > 0 else 0
    results['win_rate'] = results['wins'] / n_episodes if n_episodes > 0 else 0

    if verbose:
        print(f"\nGameplay validation complete:")
        print(f"  Win rate: {results['win_rate']:.2%}")
        print(f"  Avg score: {results['total_score'] / n_episodes:.0f}")
        print(f"  Avg blocks: {results['total_blocks'] / n_episodes:.1f}")

    return results


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(description='Validate ONNX model')
    parser.add_argument('--pytorch', type=str, required=True,
                        help='Path to PyTorch model')
    parser.add_argument('--onnx', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--samples', type=int, default=100,
                        help='Number of samples for numerical validation')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes for gameplay validation')
    parser.add_argument('--tolerance', type=float, default=1e-5,
                        help='Tolerance for numerical comparison')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    # Numerical validation
    is_valid, max_diff = validate_onnx(
        pytorch_model_path=args.pytorch,
        onnx_model_path=args.onnx,
        n_samples=args.samples,
        tolerance=args.tolerance,
        verbose=not args.quiet
    )

    # Gameplay validation
    if is_valid:
        validate_gameplay(
            onnx_model_path=args.onnx,
            n_episodes=args.episodes,
            verbose=not args.quiet
        )


if __name__ == '__main__':
    main()

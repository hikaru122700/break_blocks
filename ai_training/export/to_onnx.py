"""Export trained model to ONNX format for browser deployment."""

import os
import json
import argparse
from typing import Optional

import numpy as np
import torch
import onnx
from stable_baselines3 import PPO

from ..env.constants import OBSERVATION_DIM, ACTION_DIM


class PolicyWrapper(torch.nn.Module):
    """
    Wrapper to extract policy network from PPO for ONNX export.

    The wrapper takes observation as input and outputs action probabilities.
    """

    def __init__(self, policy):
        """
        Initialize wrapper.

        Args:
            policy: SB3 policy object
        """
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning action probabilities.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            Action probabilities [batch, action_dim]
        """
        # Get features from policy
        features = self.policy.extract_features(obs)

        # Get action distribution
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        action_logits = self.policy.action_net(latent_pi)

        # Convert to probabilities
        action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs


def export_to_onnx(
    model_path: str,
    output_path: str = './models/onnx/breakout_agent.onnx',
    opset_version: int = 14,
    verbose: bool = True
) -> str:
    """
    Export trained PPO model to ONNX format.

    Args:
        model_path: Path to trained SB3 model
        output_path: Output path for ONNX model
        opset_version: ONNX opset version
        verbose: Print progress

    Returns:
        Path to exported ONNX model
    """
    if verbose:
        print(f"Loading model from {model_path}...")

    # Load model
    model = PPO.load(model_path, device='cpu')
    policy = model.policy

    # Set to evaluation mode
    policy.eval()

    # Create wrapper
    wrapped_policy = PolicyWrapper(policy)
    wrapped_policy.eval()

    # Create dummy input
    dummy_input = torch.zeros(1, OBSERVATION_DIM, dtype=torch.float32)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if verbose:
        print(f"Exporting to ONNX (opset {opset_version})...")

    # Export to ONNX
    torch.onnx.export(
        wrapped_policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_probs'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_probs': {0: 'batch_size'}
        }
    )

    if verbose:
        print(f"Model exported to {output_path}")

    # Verify the model
    if verbose:
        print("Verifying ONNX model...")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    if verbose:
        print("ONNX model verified successfully!")

    # Save normalization stats (if using VecNormalize)
    stats_path = output_path.replace('.onnx', '_stats.json')
    stats = {
        'observation_dim': OBSERVATION_DIM,
        'action_dim': ACTION_DIM,
        'observation_mean': [0.0] * OBSERVATION_DIM,  # Not using normalization
        'observation_std': [1.0] * OBSERVATION_DIM,
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"Stats saved to {stats_path}")

    return output_path


def main():
    """Main entry point for ONNX export."""
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--output', type=str,
                        default='./models/onnx/breakout_agent.onnx',
                        help='Output path for ONNX model')
    parser.add_argument('--opset', type=int, default=14,
                        help='ONNX opset version')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        opset_version=args.opset,
        verbose=not args.quiet
    )


if __name__ == '__main__':
    main()

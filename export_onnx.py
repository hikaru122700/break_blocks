#!/usr/bin/env python3
"""Export trained PPO model to ONNX format for browser inference."""

import os
import argparse
import torch
import torch.nn as nn
from stable_baselines3 import PPO


class PolicyNetwork(nn.Module):
    """Wrapper for extracting policy network from PPO model."""

    def __init__(self, model: PPO):
        super().__init__()
        self.features_extractor = model.policy.features_extractor
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass returning action probabilities."""
        features = self.features_extractor(observation)
        latent_pi, _ = self.mlp_extractor(features)
        action_logits = self.action_net(latent_pi)
        action_probs = torch.softmax(action_logits, dim=-1)
        return action_probs


def export_to_onnx(
    model_path: str = './models/final_model',
    output_path: str = './models/onnx/breakout_agent.onnx',
    observation_dim: int = 216
):
    """
    Export PPO model to ONNX format.

    Args:
        model_path: Path to saved PPO model
        output_path: Path for output ONNX file
        observation_dim: Dimension of observation space
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device='cpu')

    print("Creating policy network wrapper...")
    policy_net = PolicyNetwork(model)
    policy_net.eval()
    policy_net.cpu()  # Ensure model is on CPU

    # Create dummy input
    dummy_input = torch.randn(1, observation_dim)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        policy_net,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_probs'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_probs': {0: 'batch_size'}
        }
    )

    print(f"ONNX model saved to {output_path}")

    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification: OK")
    except ImportError:
        print("Note: Install 'onnx' package to verify the exported model")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    # Print model info
    file_size = os.path.getsize(output_path) / 1024
    print(f"\nModel info:")
    print(f"  File size: {file_size:.1f} KB")
    print(f"  Input shape: [batch, {observation_dim}]")
    print(f"  Output shape: [batch, 3] (action probabilities)")


def main():
    parser = argparse.ArgumentParser(description='Export PPO model to ONNX')
    parser.add_argument('--model', type=str, default='./models/final_model',
                        help='Path to saved model')
    parser.add_argument('--output', type=str, default='./models/onnx/breakout_agent.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--obs-dim', type=int, default=216,
                        help='Observation dimension')

    args = parser.parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        observation_dim=args.obs_dim
    )


if __name__ == '__main__':
    main()

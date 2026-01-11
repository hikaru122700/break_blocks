#!/usr/bin/env python3
"""Behavioral Cloning training for Break Blocks."""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ai_training.env import BreakoutEnv
from ai_training.env.constants import OBSERVATION_DIM, ACTION_DIM


class PolicyNetwork(nn.Module):
    """Policy network for behavioral cloning."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list = [512, 512, 256]):
        super().__init__()

        layers = []
        prev_size = obs_dim

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def get_action(self, obs):
        """Get action from observation."""
        with torch.no_grad():
            logits = self.forward(obs)
            return torch.argmax(logits, dim=-1)


def load_demonstrations(demo_path: str):
    """Load demonstrations from disk."""
    obs_path = os.path.join(demo_path, 'observations.npy')
    act_path = os.path.join(demo_path, 'actions.npy')

    observations = np.load(obs_path)
    actions = np.load(act_path)

    print(f"Loaded demonstrations:")
    print(f"  Observations: {observations.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Action distribution: {np.bincount(actions)}")

    return observations, actions


def train_bc(
    demo_path: str = './demos',
    output_path: str = './models/bc_model',
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    hidden_sizes: list = [512, 512, 256],
    device: str = 'auto',
    verbose: bool = True
):
    """
    Train policy using Behavioral Cloning.

    Args:
        demo_path: Path to demonstration data
        output_path: Where to save trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        hidden_sizes: Hidden layer sizes
        device: Device to use
        verbose: Print progress
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")

    # Load demonstrations
    observations, actions = load_demonstrations(demo_path)

    # Convert to tensors
    obs_tensor = torch.FloatTensor(observations).to(device)
    act_tensor = torch.LongTensor(actions).to(device)

    # Create dataset and dataloader
    dataset = TensorDataset(obs_tensor, act_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = PolicyNetwork(
        obs_dim=OBSERVATION_DIM,
        action_dim=ACTION_DIM,
        hidden_sizes=hidden_sizes
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_accuracy = 0
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_obs, batch_act in dataloader:
            optimizer.zero_grad()

            logits = model(batch_obs)
            loss = criterion(logits, batch_act)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predicted = torch.argmax(logits, dim=-1)
            correct += (predicted == batch_act).sum().item()
            total += batch_act.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total * 100

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")

        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'obs_dim': OBSERVATION_DIM,
                'action_dim': ACTION_DIM,
                'hidden_sizes': hidden_sizes,
                'accuracy': accuracy,
            }, output_path + '.pt')

    print(f"\nTraining complete!")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print(f"Model saved to: {output_path}.pt")

    return model


def evaluate_bc_model(
    model_path: str,
    n_episodes: int = 100,
    stage: int = 1,
    device: str = 'auto',
    verbose: bool = True
):
    """Evaluate BC model performance."""
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = PolicyNetwork(
        obs_dim=checkpoint['obs_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_sizes=checkpoint['hidden_sizes']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model with accuracy: {checkpoint['accuracy']:.2f}%")

    # Create environment
    env = BreakoutEnv(stage_number=stage, max_stage=stage)

    wins = 0
    total_reward = 0
    total_blocks = 0

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = model.get_action(obs_tensor).item()

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
    print(f"BC Model Evaluation (Stage {stage})")
    print(f"{'='*50}")
    print(f"Win Rate: {wins/n_episodes*100:.1f}% ({wins}/{n_episodes})")
    print(f"Average Reward: {total_reward/n_episodes:.1f}")
    print(f"Average Blocks: {total_blocks/n_episodes:.1f}")


def convert_to_ppo(
    bc_model_path: str,
    output_path: str = './models/bc_ppo',
    device: str = 'auto'
):
    """
    Convert BC model to PPO format for fine-tuning.

    This creates a PPO model with the BC policy as initialization.
    """
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load BC model
    checkpoint = torch.load(bc_model_path, map_location=device)

    # Create environment
    env = DummyVecEnv([lambda: BreakoutEnv(stage_number=1, max_stage=1)])

    # Create PPO model with same architecture
    policy_kwargs = {
        'net_arch': checkpoint['hidden_sizes']
    }

    model = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1
    )

    # Copy weights from BC model to PPO policy network
    bc_model = PolicyNetwork(
        obs_dim=checkpoint['obs_dim'],
        action_dim=checkpoint['action_dim'],
        hidden_sizes=checkpoint['hidden_sizes']
    )
    bc_model.load_state_dict(checkpoint['model_state_dict'])

    # Get PPO policy network parameters
    ppo_policy = model.policy

    # Copy the shared layers (this is approximate, architecture must match)
    # PPO has: features_extractor -> mlp_extractor -> action_net
    # We copy to mlp_extractor

    bc_layers = list(bc_model.network.children())
    ppo_mlp = ppo_policy.mlp_extractor

    # Copy policy layers
    bc_idx = 0
    for i, (name, param) in enumerate(ppo_mlp.policy_net.named_parameters()):
        # Find corresponding BC layer
        while bc_idx < len(bc_layers) and not isinstance(bc_layers[bc_idx], nn.Linear):
            bc_idx += 1
        if bc_idx >= len(bc_layers) - 1:  # Skip last layer (action head)
            break

        bc_layer = bc_layers[bc_idx]
        if 'weight' in name and hasattr(bc_layer, 'weight'):
            param.data.copy_(bc_layer.weight.data)
        elif 'bias' in name and hasattr(bc_layer, 'bias'):
            param.data.copy_(bc_layer.bias.data)
            bc_idx += 1  # Move to next layer after bias

    # Save the model
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    model.save(output_path)
    print(f"Converted PPO model saved to: {output_path}")

    env.close()
    return model


def main():
    parser = argparse.ArgumentParser(description='Behavioral Cloning for Break Blocks')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'convert'],
                        help='Mode: train, eval, or convert to PPO')
    parser.add_argument('--demo-path', type=str, default='./demos',
                        help='Path to demonstration data')
    parser.add_argument('--model-path', type=str, default='./models/bc_model',
                        help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--eval-episodes', type=int, default=100,
                        help='Evaluation episodes')
    parser.add_argument('--stage', type=int, default=1,
                        help='Stage for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, cpu)')

    args = parser.parse_args()

    if args.mode == 'train':
        train_bc(
            demo_path=args.demo_path,
            output_path=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=args.device
        )
    elif args.mode == 'eval':
        evaluate_bc_model(
            model_path=args.model_path + '.pt',
            n_episodes=args.eval_episodes,
            stage=args.stage,
            device=args.device
        )
    elif args.mode == 'convert':
        convert_to_ppo(
            bc_model_path=args.model_path + '.pt',
            output_path=args.model_path + '_ppo',
            device=args.device
        )


if __name__ == '__main__':
    main()

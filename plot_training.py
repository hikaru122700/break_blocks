#!/usr/bin/env python3
"""Plot training curves from TensorBoard logs and save as PNG."""

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_logs(log_dir: str) -> dict:
    """
    Load training data from TensorBoard logs.

    Args:
        log_dir: Path to TensorBoard log directory

    Returns:
        Dictionary with metric names as keys and (steps, values) tuples
    """
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        }
    )
    ea.Reload()

    data = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = (np.array(steps), np.array(values))

    return data


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_training_curves(
    log_dir: str,
    output_dir: str = './plots',
    smoothing: float = 0.9
):
    """
    Plot training curves and save as PNG files.

    Args:
        log_dir: Path to TensorBoard log directory
        output_dir: Directory to save PNG files
        smoothing: Smoothing factor (0-1, higher = smoother)
    """
    # Find the latest PPO run
    log_path = Path(log_dir)
    ppo_dirs = sorted(log_path.glob('PPO_*'))

    if not ppo_dirs:
        print(f"No PPO logs found in {log_dir}")
        return

    latest_run = ppo_dirs[-1]
    print(f"Loading logs from: {latest_run}")

    # Load data
    data = load_tensorboard_logs(str(latest_run))

    if not data:
        print("No scalar data found in logs")
        return

    print(f"Found metrics: {list(data.keys())}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define plots to create
    plot_configs = [
        {
            'title': 'Episode Reward',
            'metrics': ['rollout/ep_rew_mean'],
            'ylabel': 'Reward',
            'filename': 'reward.png'
        },
        {
            'title': 'Episode Length',
            'metrics': ['rollout/ep_len_mean'],
            'ylabel': 'Steps',
            'filename': 'episode_length.png'
        },
        {
            'title': 'Win Rate',
            'metrics': ['train/win_rate', 'train/stage_1_win_rate'],
            'ylabel': 'Win Rate',
            'filename': 'win_rate.png'
        },
        {
            'title': 'Policy Loss',
            'metrics': ['train/policy_gradient_loss'],
            'ylabel': 'Loss',
            'filename': 'policy_loss.png'
        },
        {
            'title': 'Value Loss',
            'metrics': ['train/value_loss'],
            'ylabel': 'Loss',
            'filename': 'value_loss.png'
        },
        {
            'title': 'Entropy',
            'metrics': ['train/entropy_loss'],
            'ylabel': 'Entropy',
            'filename': 'entropy.png'
        },
        {
            'title': 'Learning Metrics',
            'metrics': ['train/approx_kl', 'train/clip_fraction'],
            'ylabel': 'Value',
            'filename': 'learning_metrics.png'
        },
    ]

    # Create individual plots
    for config in plot_configs:
        fig, ax = plt.subplots(figsize=(10, 6))

        has_data = False
        for metric in config['metrics']:
            if metric in data:
                steps, values = data[metric]
                if len(steps) > 0:
                    smoothed = smooth(values, smoothing)
                    label = metric.split('/')[-1]
                    ax.plot(steps, values, alpha=0.3)
                    ax.plot(steps, smoothed, label=label, linewidth=2)
                    has_data = True

        if has_data:
            ax.set_xlabel('Timesteps')
            ax.set_ylabel(config['ylabel'])
            ax.set_title(config['title'])
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Format x-axis with millions
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M')
            )

            filepath = os.path.join(output_dir, config['filename'])
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")

        plt.close(fig)

    # Create combined summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    if 'rollout/ep_rew_mean' in data:
        steps, values = data['rollout/ep_rew_mean']
        axes[0, 0].plot(steps, values, alpha=0.3, color='blue')
        axes[0, 0].plot(steps, smooth(values, smoothing), color='blue', linewidth=2)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].grid(True, alpha=0.3)

    # Win Rate
    if 'train/win_rate' in data:
        steps, values = data['train/win_rate']
        axes[0, 1].plot(steps, values, alpha=0.3, color='green')
        axes[0, 1].plot(steps, smooth(values, smoothing), color='green', linewidth=2, label='Overall')
    if 'train/stage_1_win_rate' in data:
        steps, values = data['train/stage_1_win_rate']
        axes[0, 1].plot(steps, values, alpha=0.3, color='orange')
        axes[0, 1].plot(steps, smooth(values, smoothing), color='orange', linewidth=2, label='Stage 1')
    axes[0, 1].set_title('Win Rate')
    axes[0, 1].set_xlabel('Timesteps')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Episode Length
    if 'rollout/ep_len_mean' in data:
        steps, values = data['rollout/ep_len_mean']
        axes[1, 0].plot(steps, values, alpha=0.3, color='purple')
        axes[1, 0].plot(steps, smooth(values, smoothing), color='purple', linewidth=2)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].grid(True, alpha=0.3)

    # Losses
    if 'train/value_loss' in data:
        steps, values = data['train/value_loss']
        axes[1, 1].plot(steps, smooth(values, smoothing), label='Value Loss', linewidth=2)
    if 'train/policy_gradient_loss' in data:
        steps, values = data['train/policy_gradient_loss']
        axes[1, 1].plot(steps, smooth(np.abs(values), smoothing), label='|Policy Loss|', linewidth=2)
    axes[1, 1].set_title('Losses')
    axes[1, 1].set_xlabel('Timesteps')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Format all x-axes
    for ax in axes.flat:
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M')
        )

    plt.suptitle(f'Training Summary - {latest_run.name}', fontsize=14)
    plt.tight_layout()

    summary_path = os.path.join(output_dir, 'training_summary.png')
    fig.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {summary_path}")
    plt.close(fig)

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from TensorBoard logs')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='TensorBoard log directory')
    parser.add_argument('--output', type=str, default='./plots',
                        help='Output directory for PNG files')
    parser.add_argument('--smoothing', type=float, default=0.9,
                        help='Smoothing factor (0-1)')

    args = parser.parse_args()

    plot_training_curves(
        log_dir=args.log_dir,
        output_dir=args.output,
        smoothing=args.smoothing
    )


if __name__ == '__main__':
    main()

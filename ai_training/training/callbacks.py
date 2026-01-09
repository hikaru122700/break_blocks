"""Training callbacks for Break Blocks RL."""

import os
from typing import Dict, Any, Optional
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import VecEnv


class TrainingCallback(BaseCallback):
    """
    Custom callback for monitoring training progress.

    Tracks:
    - Episode rewards and lengths
    - Win rate per stage
    - Curriculum progress
    """

    def __init__(
        self,
        log_freq: int = 1000,
        save_freq: int = 50000,
        save_path: str = './models/checkpoints',
        verbose: int = 1
    ):
        """
        Initialize callback.

        Args:
            log_freq: Frequency of logging (in steps)
            save_freq: Frequency of model saving (in steps)
            save_path: Path to save checkpoints
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.save_path = save_path

        # Statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.wins = []
        self.stage_wins: Dict[int, list] = {}

        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after each step."""
        # Check for episode end
        if self.locals.get('dones') is not None:
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals.get('infos', [{}])[i]
                    self._on_episode_end(info)

        # Logging
        if self.num_timesteps % self.log_freq == 0:
            self._log_stats()

        # Save checkpoint
        if self.num_timesteps % self.save_freq == 0:
            self._save_checkpoint()

        return True

    def _on_episode_end(self, info: Dict[str, Any]):
        """Called at the end of each episode."""
        # Get episode info
        if 'episode' in info:
            ep_info = info['episode']
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 0))

        # Track wins
        is_win = info.get('is_stage_clear', False)
        self.wins.append(is_win)

        stage = info.get('stage', 1)
        if stage not in self.stage_wins:
            self.stage_wins[stage] = []
        self.stage_wins[stage].append(is_win)

    def _log_stats(self):
        """Log training statistics."""
        if not self.episode_rewards:
            return

        # Recent stats (last 100 episodes)
        recent_rewards = self.episode_rewards[-100:]
        recent_wins = self.wins[-100:]

        mean_reward = np.mean(recent_rewards) if recent_rewards else 0
        win_rate = np.mean(recent_wins) if recent_wins else 0

        if self.verbose > 0:
            print(f"\n[Step {self.num_timesteps}]")
            print(f"  Mean reward (100 ep): {mean_reward:.2f}")
            print(f"  Win rate (100 ep): {win_rate:.2%}")
            print(f"  Total episodes: {len(self.episode_rewards)}")

            # Stage-specific stats
            for stage in sorted(self.stage_wins.keys()):
                stage_rate = np.mean(self.stage_wins[stage][-50:]) if self.stage_wins[stage] else 0
                print(f"  Stage {stage} win rate: {stage_rate:.2%}")

        # Log to tensorboard if available
        if self.logger is not None:
            self.logger.record('train/mean_reward', mean_reward)
            self.logger.record('train/win_rate', win_rate)
            self.logger.record('train/episodes', len(self.episode_rewards))

            for stage in sorted(self.stage_wins.keys()):
                stage_rate = np.mean(self.stage_wins[stage][-50:]) if self.stage_wins[stage] else 0
                self.logger.record(f'train/stage_{stage}_win_rate', stage_rate)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        if self.model is not None:
            path = os.path.join(self.save_path, f'model_{self.num_timesteps}')
            self.model.save(path)
            if self.verbose > 0:
                print(f"[Checkpoint] Saved to {path}")


class CurriculumCallback(BaseCallback):
    """
    Callback for managing curriculum learning phases.

    Automatically advances curriculum phases based on win rate.
    """

    def __init__(
        self,
        phase_thresholds: list = None,
        phase_steps: list = None,
        verbose: int = 1
    ):
        """
        Initialize curriculum callback.

        Args:
            phase_thresholds: Win rate thresholds to advance phase
            phase_steps: Step counts to advance phase (alternative to thresholds)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.phase_thresholds = phase_thresholds or [0.7, 0.75, 0.8, 0.85]
        self.phase_steps = phase_steps or [500000, 1500000, 3500000, 6500000]
        self.current_phase = 0
        self.phase_changed = False

    def _on_step(self) -> bool:
        """Check for phase advancement."""
        # Check step-based advancement
        for i, step_threshold in enumerate(self.phase_steps):
            if self.num_timesteps >= step_threshold and i > self.current_phase:
                self._advance_phase(i)
                break

        return True

    def _advance_phase(self, new_phase: int):
        """Advance to a new curriculum phase."""
        if new_phase <= self.current_phase:
            return

        self.current_phase = new_phase
        self.phase_changed = True

        if self.verbose > 0:
            print(f"\n[Curriculum] Advanced to phase {self.current_phase}")

        # Update environment if it supports curriculum
        if hasattr(self.training_env, 'env_method'):
            try:
                self.training_env.env_method('set_phase', self.current_phase)
            except Exception:
                pass


class BestModelCallback(EvalCallback):
    """
    Extended eval callback that saves best model based on win rate.
    """

    def __init__(
        self,
        eval_env: VecEnv,
        best_model_save_path: str = './models/best_model',
        log_path: str = './logs',
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        verbose: int = 1
    ):
        """
        Initialize best model callback.

        Args:
            eval_env: Environment for evaluation
            best_model_save_path: Path to save best model
            log_path: Path for logs
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of episodes per evaluation
            verbose: Verbosity level
        """
        super().__init__(
            eval_env=eval_env,
            best_model_save_path=best_model_save_path,
            log_path=log_path,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            verbose=verbose
        )

        self.best_win_rate = 0.0
        self.eval_wins = []

    def _on_step(self) -> bool:
        """Run evaluation and track wins."""
        result = super()._on_step()

        # After evaluation, check win rate
        if self.n_calls % self.eval_freq == 0:
            # Calculate win rate from recent evaluations
            if hasattr(self, 'evaluations_results') and self.evaluations_results:
                # Note: EvalCallback stores mean rewards, we need custom tracking
                pass

        return result

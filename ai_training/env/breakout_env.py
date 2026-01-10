"""Gymnasium environment for Break Blocks game."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

from .constants import (
    OBSERVATION_DIM, ACTION_DIM, FRAME_SKIP,
    MAX_STAGES, PowerUpType
)
from .game_simulation import GameSimulation


class BreakoutEnv(gym.Env):
    """
    Gymnasium environment for Break Blocks.

    Observation space: 215-dimensional Box
    Action space: Discrete(3) - Left, Stay, Right
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        stage_number: int = 1,
        max_stage: int = 10,
        frame_skip: int = FRAME_SKIP,
        time_penalty_scale: float = 1.0,
        render_mode: Optional[str] = None
    ):
        """
        Initialize the environment.

        Args:
            stage_number: Starting stage (1-10)
            max_stage: Maximum stage to train on (for curriculum learning)
            frame_skip: Number of frames to repeat action
            time_penalty_scale: Scale factor for time penalty (for curriculum)
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()

        self.stage_number = stage_number
        self.max_stage = min(max_stage, MAX_STAGES)
        self.frame_skip = frame_skip
        self.time_penalty_scale = time_penalty_scale
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.Discrete(ACTION_DIM)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_DIM,),
            dtype=np.float32
        )

        # Game simulation
        self.game: Optional[GameSimulation] = None

        # Episode statistics
        self.episode_score = 0
        self.episode_blocks_destroyed = 0
        self.episode_time_elapsed = 0.0
        self.max_combo = 0
        self._close_bonus_given = False

        # Delta time per frame (16.67ms for 60fps)
        self.dt = 16.67

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed
            options: Optional reset options (e.g., 'stage' to specify stage)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Determine stage
        if options and 'stage' in options:
            stage = options['stage']
        else:
            # Random stage selection for curriculum
            stage = self.np_random.integers(1, self.max_stage + 1)

        self.stage_number = stage

        # Create new game
        self.game = GameSimulation(stage_number=stage)

        # Reset statistics
        self.episode_score = 0
        self.episode_blocks_destroyed = 0
        self.episode_time_elapsed = 0.0
        self.max_combo = 0
        self._close_bonus_given = False

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Args:
            action: 0=left, 1=stay, 2=right

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.game is not None, "Environment not reset"

        total_reward = 0.0
        events_accumulated = {
            'blocks_destroyed': 0,
            'block_scores': [],
            'powerups_collected': [],
            'paddle_hits': 0,
            'life_lost': False,
            'stage_clear': False,
            'game_over': False,
            'time_up': False,
        }

        # Frame skip: repeat action for multiple frames
        for _ in range(self.frame_skip):
            events = self.game.step(action, self.dt)

            # Accumulate events
            events_accumulated['blocks_destroyed'] += events['blocks_destroyed']
            events_accumulated['block_scores'].extend(events['block_scores'])
            events_accumulated['powerups_collected'].extend(events['powerups_collected'])
            events_accumulated['paddle_hits'] += events['paddle_hits']
            events_accumulated['life_lost'] = events_accumulated['life_lost'] or events['life_lost']
            events_accumulated['stage_clear'] = events_accumulated['stage_clear'] or events['stage_clear']
            events_accumulated['game_over'] = events_accumulated['game_over'] or events['game_over']
            events_accumulated['time_up'] = events_accumulated['time_up'] or events['time_up']

            # Update statistics
            self.episode_time_elapsed += self.dt
            if events['combo'] > self.max_combo:
                self.max_combo = events['combo']

            if self.game.is_game_over or self.game.is_stage_clear:
                break

        # Calculate reward
        total_reward = self._calculate_reward(events_accumulated)

        # Update episode score
        self.episode_score += sum(events_accumulated['block_scores'])
        self.episode_blocks_destroyed += events_accumulated['blocks_destroyed']

        # Check termination
        terminated = self.game.is_game_over or self.game.is_stage_clear
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info['events'] = events_accumulated

        return obs, total_reward, terminated, truncated, info

    def _calculate_reward(self, events: Dict[str, Any]) -> float:
        """
        Calculate reward based on events.

        Reward design for 100% win rate:
        - Paddle hit bonus (critical for survival)
        - Block destroy bonus
        - Combo bonus
        - Stage clear bonus (large)
        - Power-up bonuses
        - Life loss penalty (large)
        - Game over penalty (very large)
        - Small time penalty
        """
        reward = 0.0

        # Paddle hit reward - CRITICAL for learning to keep ball in play
        reward += events['paddle_hits'] * 5.0

        # Block destruction rewards
        for score in events['block_scores']:
            reward += 20.0

        # Combo bonus (exponential for higher combos)
        combo = self.game.combo if self.game else 0
        if combo > 0:
            reward += 0.5 * combo

        # Power-up bonuses
        for powerup in events['powerups_collected']:
            if powerup == PowerUpType.TIME_EXTEND:
                reward += 20.0
            elif powerup == PowerUpType.PENETRATE:
                reward += 15.0
            elif powerup == PowerUpType.MULTI_BALL:
                reward += 10.0
            elif powerup == PowerUpType.SPEED_DOWN:
                reward += 8.0
            elif powerup == PowerUpType.SPEED_UP:
                reward += 3.0

        # Close to clear bonus (5 blocks or fewer remaining)
        if self.game and not self._close_bonus_given:
            active_blocks = sum(1 for b in self.game.blocks if not b.is_destroyed)
            if active_blocks <= 5 and active_blocks > 0:
                reward += 50.0
                self._close_bonus_given = True

        # Stage clear bonus - make this the primary goal
        if events['stage_clear']:
            reward += 1000.0
            # Time bonus: reward for remaining time
            if self.game:
                time_ratio = self.game.time_remaining / self.game.time_limit
                reward += 200.0 * time_ratio

        # Penalties
        if events['life_lost']:
            reward -= 100.0

        if events['game_over']:
            reward -= 500.0

        # Small time penalty (less aggressive)
        reward -= 0.05 * self.time_penalty_scale

        return reward

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        if self.game is None:
            return np.zeros(OBSERVATION_DIM, dtype=np.float32)

        obs = self.game.get_observation()
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        """Get info dictionary."""
        if self.game is None:
            return {}

        return {
            'stage': self.stage_number,
            'score': self.episode_score,
            'lives': self.game.lives,
            'blocks_destroyed': self.episode_blocks_destroyed,
            'time_remaining': self.game.time_remaining,
            'time_elapsed': self.episode_time_elapsed,
            'max_combo': self.max_combo,
            'is_stage_clear': self.game.is_stage_clear,
            'is_game_over': self.game.is_game_over,
        }

    def set_phase(self, phase: int):
        """Set curriculum phase (for compatibility with CurriculumCallback)."""
        # Base class does nothing - override in CurriculumBreakoutEnv for full support
        pass

    def render(self):
        """Render the environment (not implemented for training)."""
        pass

    def close(self):
        """Clean up resources."""
        self.game = None


class CurriculumBreakoutEnv(BreakoutEnv):
    """
    Breakout environment with curriculum learning support.

    Automatically adjusts difficulty based on training progress.
    """

    def __init__(
        self,
        initial_max_stage: int = 1,
        stage_unlock_threshold: float = 0.8,  # Win rate to unlock next stage
        time_penalty_phases: List[float] = None,
        **kwargs
    ):
        """
        Initialize curriculum environment.

        Args:
            initial_max_stage: Starting max stage
            stage_unlock_threshold: Win rate needed to unlock next stage
            time_penalty_phases: List of time penalty scales for each phase
            **kwargs: Additional arguments for BreakoutEnv
        """
        kwargs['max_stage'] = initial_max_stage
        super().__init__(**kwargs)

        self.initial_max_stage = initial_max_stage
        self.stage_unlock_threshold = stage_unlock_threshold
        self.time_penalty_phases = time_penalty_phases or [1.0, 1.5, 2.0]
        self.current_phase = 0

        # Win tracking
        self.episode_count = 0
        self.win_count = 0
        self.recent_wins: List[bool] = []
        self.window_size = 100

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Step with curriculum tracking."""
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated:
            self.episode_count += 1
            is_win = self.game.is_stage_clear
            self.recent_wins.append(is_win)
            if is_win:
                self.win_count += 1

            # Keep window size
            if len(self.recent_wins) > self.window_size:
                self.recent_wins.pop(0)

            # Check for stage unlock
            if len(self.recent_wins) >= self.window_size:
                win_rate = sum(self.recent_wins) / len(self.recent_wins)
                if win_rate >= self.stage_unlock_threshold:
                    self._unlock_next_stage()

        return obs, reward, terminated, truncated, info

    def _unlock_next_stage(self):
        """Unlock the next stage in curriculum."""
        if self.max_stage < MAX_STAGES:
            self.max_stage += 1
            self.recent_wins = []  # Reset tracking
            print(f"[Curriculum] Unlocked stage {self.max_stage}")

    def set_phase(self, phase: int):
        """Set curriculum phase (affects time penalty)."""
        if 0 <= phase < len(self.time_penalty_phases):
            self.current_phase = phase
            self.time_penalty_scale = self.time_penalty_phases[phase]
            print(f"[Curriculum] Phase {phase}, time penalty scale: {self.time_penalty_scale}")

    def get_curriculum_stats(self) -> dict:
        """Get curriculum training statistics."""
        return {
            'episode_count': self.episode_count,
            'win_count': self.win_count,
            'win_rate': sum(self.recent_wins) / len(self.recent_wins) if self.recent_wins else 0.0,
            'max_stage': self.max_stage,
            'current_phase': self.current_phase,
            'time_penalty_scale': self.time_penalty_scale,
        }

"""Reward calculation utilities for Break Blocks RL training."""

from typing import Dict, Any, Optional
from ..env.constants import PowerUpType


class RewardCalculator:
    """
    Reward calculator with configurable weights.

    Default reward structure optimized for fastest stage clear.
    """

    def __init__(
        self,
        time_penalty: float = -0.01,
        block_destroy: float = 10.0,
        combo_bonus: float = 0.2,
        stage_clear: float = 100.0,
        time_bonus_max: float = 50.0,
        life_loss: float = -50.0,
        game_over: float = -200.0,
        powerup_time: float = 15.0,
        powerup_penetrate: float = 10.0,
        powerup_multiball: float = 8.0,
        powerup_speed_down: float = 5.0,
        powerup_speed_up: float = 2.0,
        time_penalty_scale: float = 1.0
    ):
        """
        Initialize reward calculator with weights.

        Args:
            time_penalty: Penalty per frame (encourages speed)
            block_destroy: Reward for destroying a block
            combo_bonus: Bonus per combo level
            stage_clear: Bonus for clearing stage
            time_bonus_max: Maximum time bonus (scaled by remaining time)
            life_loss: Penalty for losing a life
            game_over: Penalty for game over
            powerup_*: Rewards for collecting various power-ups
            time_penalty_scale: Multiplier for time penalty (curriculum learning)
        """
        self.time_penalty = time_penalty
        self.block_destroy = block_destroy
        self.combo_bonus = combo_bonus
        self.stage_clear = stage_clear
        self.time_bonus_max = time_bonus_max
        self.life_loss = life_loss
        self.game_over = game_over
        self.powerup_time = powerup_time
        self.powerup_penetrate = powerup_penetrate
        self.powerup_multiball = powerup_multiball
        self.powerup_speed_down = powerup_speed_down
        self.powerup_speed_up = powerup_speed_up
        self.time_penalty_scale = time_penalty_scale

    def calculate(
        self,
        events: Dict[str, Any],
        combo: int = 0,
        time_remaining: float = 0.0,
        time_limit: float = 1.0
    ) -> float:
        """
        Calculate total reward for a step.

        Args:
            events: Dictionary of events from game step
            combo: Current combo count
            time_remaining: Remaining time in ms
            time_limit: Total time limit in ms

        Returns:
            Total reward
        """
        reward = 0.0

        # Time penalty (scaled)
        reward += self.time_penalty * self.time_penalty_scale

        # Block destruction
        num_blocks = events.get('blocks_destroyed', 0)
        reward += self.block_destroy * num_blocks

        # Combo bonus
        if combo > 0:
            reward += self.combo_bonus * combo

        # Power-up rewards
        for powerup in events.get('powerups_collected', []):
            if powerup == PowerUpType.TIME_EXTEND:
                reward += self.powerup_time
            elif powerup == PowerUpType.PENETRATE:
                reward += self.powerup_penetrate
            elif powerup == PowerUpType.MULTI_BALL:
                reward += self.powerup_multiball
            elif powerup == PowerUpType.SPEED_DOWN:
                reward += self.powerup_speed_down
            elif powerup == PowerUpType.SPEED_UP:
                reward += self.powerup_speed_up

        # Stage clear bonus
        if events.get('stage_clear', False):
            reward += self.stage_clear
            # Time bonus proportional to remaining time
            if time_limit > 0:
                time_ratio = time_remaining / time_limit
                reward += self.time_bonus_max * time_ratio

        # Penalties
        if events.get('life_lost', False):
            reward += self.life_loss

        if events.get('game_over', False):
            reward += self.game_over

        return reward

    def set_time_penalty_scale(self, scale: float):
        """Set time penalty scale for curriculum learning."""
        self.time_penalty_scale = scale

    def get_config(self) -> Dict[str, float]:
        """Get current reward configuration."""
        return {
            'time_penalty': self.time_penalty,
            'block_destroy': self.block_destroy,
            'combo_bonus': self.combo_bonus,
            'stage_clear': self.stage_clear,
            'time_bonus_max': self.time_bonus_max,
            'life_loss': self.life_loss,
            'game_over': self.game_over,
            'powerup_time': self.powerup_time,
            'powerup_penetrate': self.powerup_penetrate,
            'powerup_multiball': self.powerup_multiball,
            'powerup_speed_down': self.powerup_speed_down,
            'powerup_speed_up': self.powerup_speed_up,
            'time_penalty_scale': self.time_penalty_scale,
        }


# Default reward calculator instance
default_reward_calculator = RewardCalculator()

"""Simplified environment for bootstrapping paddle control learning."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any

from .constants import (
    ACTION_DIM, FRAME_SKIP, CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_Y_OFFSET
)
from .game_simulation import GameSimulation


class SimpleBreakoutEnv(gym.Env):
    """
    Simplified Breakout environment with minimal observation space.

    Only observes:
    - Ball X, Y position (normalized)
    - Ball VX, VY velocity (normalized)
    - Paddle X position (normalized)

    Total: 5 dimensions (much easier to learn than 507!)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        block_rows: int = 1,
        ball_speed_multiplier: float = 0.6,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.block_rows = block_rows
        self.ball_speed_multiplier = ball_speed_multiplier
        self.render_mode = render_mode
        self.frame_skip = FRAME_SKIP

        # Simple 5D observation space
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(5,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_DIM)

        self.game: Optional[GameSimulation] = None
        self.dt = 16.67

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.game = GameSimulation(stage_number=1)

        # Reduce blocks
        from .constants import BLOCK_OFFSET_TOP, BLOCK_HEIGHT, BLOCK_PADDING
        max_y = BLOCK_OFFSET_TOP + self.block_rows * (BLOCK_HEIGHT + BLOCK_PADDING)
        for block in self.game.blocks:
            if block.y >= max_y:
                block.is_destroyed = True

        # Slow down ball
        for ball in self.game.balls:
            ball.speed *= self.ball_speed_multiplier

        # Extend time
        self.game.time_remaining *= 3.0
        self.game.time_limit *= 3.0
        self.game.ball_speed_multiplier *= self.ball_speed_multiplier

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.game is not None

        total_reward = 0.0
        paddle_hits = 0
        blocks_destroyed = 0
        stage_clear = False
        game_over = False
        life_lost = False

        for _ in range(self.frame_skip):
            events = self.game.step(action, self.dt)
            paddle_hits += events['paddle_hits']
            blocks_destroyed += events['blocks_destroyed']
            stage_clear = stage_clear or events['stage_clear']
            game_over = game_over or events['game_over']
            life_lost = life_lost or events['life_lost']

            if self.game.is_game_over or self.game.is_stage_clear:
                break

        # Simple reward focused on paddle hits
        reward = paddle_hits * 10.0  # Big reward for hitting ball
        reward += blocks_destroyed * 5.0

        if stage_clear:
            reward += 200.0
        if life_lost:
            reward -= 20.0
        if game_over:
            reward -= 50.0

        # Tiny time penalty
        reward -= 0.01

        terminated = self.game.is_game_over or self.game.is_stage_clear
        info = {
            'is_stage_clear': self.game.is_stage_clear,
            'is_game_over': self.game.is_game_over,
            'paddle_hits': paddle_hits,
            'blocks_destroyed': blocks_destroyed,
        }

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        """Get simple 5D observation."""
        if self.game is None or not self.game.balls:
            return np.zeros(5, dtype=np.float32)

        ball = self.game.balls[0]
        paddle = self.game.paddle

        # Normalize to roughly [-1, 1] range
        obs = np.array([
            (ball.x / CANVAS_WIDTH) * 2 - 1,  # Ball X: -1 to 1
            (ball.y / CANVAS_HEIGHT) * 2 - 1,  # Ball Y: -1 to 1
            ball.vx / 10.0,  # Ball VX normalized
            ball.vy / 10.0,  # Ball VY normalized
            ((paddle.x + PADDLE_WIDTH / 2) / CANVAS_WIDTH) * 2 - 1,  # Paddle center X
        ], dtype=np.float32)

        return obs

    def close(self):
        self.game = None

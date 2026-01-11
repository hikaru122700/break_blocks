"""Image-based environment for CNN policy training."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple

from .constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_Y_OFFSET,
    BALL_RADIUS,
    BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_PADDING, BLOCK_OFFSET_TOP, BLOCK_OFFSET_LEFT,
    ACTION_DIM, FRAME_SKIP
)
from .game_simulation import GameSimulation


class ImageBreakoutEnv(gym.Env):
    """
    Image-based Breakout environment for CNN policy.

    Returns grayscale image observations like Atari.
    Uses 84x84 resolution (standard for Atari).
    """

    metadata = {'render_modes': ['rgb_array'], 'render_fps': 60}

    def __init__(
        self,
        stage_number: int = 1,
        max_stage: int = 1,
        image_size: Tuple[int, int] = (84, 84),
        grayscale: bool = True,
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.stage_number = stage_number
        self.max_stage = max_stage
        self.image_size = image_size
        self.grayscale = grayscale
        self.render_mode = render_mode
        self.frame_skip = FRAME_SKIP

        # Image observation space
        if grayscale:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(image_size[0], image_size[1], 1),
                dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(image_size[0], image_size[1], 3),
                dtype=np.uint8
            )

        self.action_space = spaces.Discrete(ACTION_DIM)

        self.game: Optional[GameSimulation] = None
        self.dt = 16.67

        # Scale factors for rendering
        self.scale_x = image_size[1] / CANVAS_WIDTH
        self.scale_y = image_size[0] / CANVAS_HEIGHT

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        stage = self.stage_number
        if options and 'stage' in options:
            stage = options['stage']

        self.game = GameSimulation(stage_number=stage)

        return self._render_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.game is not None

        total_reward = 0.0
        stage_clear = False
        game_over = False

        for _ in range(self.frame_skip):
            events = self.game.step(action, self.dt)

            # Simple reward
            total_reward += events['paddle_hits'] * 1.0
            total_reward += events['blocks_destroyed'] * 1.0

            if events['stage_clear']:
                stage_clear = True
                total_reward += 10.0
            if events['life_lost']:
                total_reward -= 1.0
            if events['game_over']:
                game_over = True

            if self.game.is_game_over or self.game.is_stage_clear:
                break

        terminated = self.game.is_game_over or self.game.is_stage_clear
        info = {
            'is_stage_clear': self.game.is_stage_clear,
            'is_game_over': self.game.is_game_over,
        }

        return self._render_observation(), total_reward, terminated, False, info

    def _render_observation(self) -> np.ndarray:
        """Render game state as image."""
        if self.grayscale:
            img = np.zeros((self.image_size[0], self.image_size[1], 1), dtype=np.uint8)
        else:
            img = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)

        if self.game is None:
            return img

        # Draw blocks (white = 255)
        for block in self.game.blocks:
            if block.is_destroyed:
                continue
            x1 = int(block.x * self.scale_x)
            y1 = int(block.y * self.scale_y)
            x2 = int((block.x + block.width) * self.scale_x)
            y2 = int((block.y + block.height) * self.scale_y)
            x1, x2 = max(0, x1), min(self.image_size[1], x2)
            y1, y2 = max(0, y1), min(self.image_size[0], y2)
            if self.grayscale:
                img[y1:y2, x1:x2, 0] = 128  # Gray for blocks
            else:
                img[y1:y2, x1:x2] = [128, 128, 128]

        # Draw paddle (white)
        paddle = self.game.paddle
        px1 = int(paddle.x * self.scale_x)
        py1 = int(paddle.y * self.scale_y)
        px2 = int((paddle.x + paddle.width) * self.scale_x)
        py2 = int((paddle.y + paddle.height) * self.scale_y)
        px1, px2 = max(0, px1), min(self.image_size[1], px2)
        py1, py2 = max(0, py1), min(self.image_size[0], py2)
        if self.grayscale:
            img[py1:py2, px1:px2, 0] = 255
        else:
            img[py1:py2, px1:px2] = [255, 255, 255]

        # Draw balls (white)
        for ball in self.game.balls:
            bx = int(ball.x * self.scale_x)
            by = int(ball.y * self.scale_y)
            br = max(1, int(ball.radius * min(self.scale_x, self.scale_y)))

            # Draw circle (simple square approximation)
            for dy in range(-br, br + 1):
                for dx in range(-br, br + 1):
                    if dx*dx + dy*dy <= br*br:
                        px, py = bx + dx, by + dy
                        if 0 <= px < self.image_size[1] and 0 <= py < self.image_size[0]:
                            if self.grayscale:
                                img[py, px, 0] = 255
                            else:
                                img[py, px] = [255, 255, 255]

        return img

    def render(self):
        return self._render_observation()

    def close(self):
        self.game = None

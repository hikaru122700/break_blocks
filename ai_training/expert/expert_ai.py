"""Rule-based Expert AI for demonstration collection."""

from typing import Optional, List
from ..env.game_simulation import GameSimulation, Ball
from ..env.constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_SPEED,
    BALL_RADIUS,
    PowerUpType
)


class ExpertAI:
    """
    Simple but effective rule-based AI.

    Core strategy: Follow the ball's X position.
    This is simple but works because:
    - Ball moves predictably
    - Paddle is wide enough to catch if positioned correctly
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2

    def get_action(self, game: GameSimulation) -> int:
        """Get action: just follow the lowest/most urgent ball."""
        paddle = game.paddle
        balls = game.balls

        if not balls:
            return 1

        # Find the ball closest to paddle (most dangerous)
        target_ball = self._get_target_ball(balls, paddle)

        if target_ball:
            # Simply follow ball's current X position
            self.target_x = target_ball.x
        else:
            # No ball moving down, stay center
            self.target_x = CANVAS_WIDTH / 2

        # Move toward target
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Small dead zone
        if diff < -5:
            return 0  # Left
        elif diff > 5:
            return 2  # Right
        else:
            return 1  # Stay

    def _get_target_ball(self, balls: List[Ball], paddle) -> Optional[Ball]:
        """Get the ball we should focus on."""
        paddle_y = paddle.y

        # First priority: balls moving down (vy > 0)
        down_balls = [b for b in balls if b.vy > 0]

        if down_balls:
            # Pick the one closest to paddle
            return min(down_balls, key=lambda b: paddle_y - b.y)

        # No balls moving down - pick lowest ball
        return max(balls, key=lambda b: b.y)


class ExpertAIWithNoise(ExpertAI):
    """Expert AI with optional noise for data augmentation."""

    def __init__(self, noise_level: float = 0.0):
        super().__init__()
        self.noise_level = noise_level

    def get_action(self, game: GameSimulation) -> int:
        import random
        if random.random() < self.noise_level:
            return random.randint(0, 2)
        return super().get_action(game)

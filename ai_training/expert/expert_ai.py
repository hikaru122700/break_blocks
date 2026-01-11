"""Rule-based Expert AI for demonstration collection."""

from typing import Optional, List
from ..env.game_simulation import GameSimulation, Ball
from ..env.constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_SPEED,
    BALL_RADIUS, BLOCK_OFFSET_TOP, BLOCK_HEIGHT, BLOCK_PADDING, BLOCK_ROWS,
    PowerUpType
)

# Y position where blocks end
BLOCK_BOTTOM_Y = BLOCK_OFFSET_TOP + BLOCK_ROWS * (BLOCK_HEIGHT + BLOCK_PADDING) + 20


class ExpertAI:
    """
    Expert AI with smart trajectory prediction.

    - When ball is in block area: tracks ball's X position
    - When ball is below blocks: predicts landing position with wall bounces
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2

    def get_action(self, game: GameSimulation) -> int:
        """Get action based on ball state and trajectory."""
        paddle = game.paddle
        balls = game.balls

        if not balls:
            return 1  # Stay

        # Find the most urgent ball
        target_ball = self._get_target_ball(balls, paddle)

        if target_ball:
            if target_ball.vy > 0:
                # Ball moving down
                if target_ball.y > BLOCK_BOTTOM_Y:
                    # Ball is below blocks - predict landing
                    self.target_x = self._predict_landing_x(target_ball, paddle.y)
                else:
                    # Ball is in block area - just track it
                    self.target_x = target_ball.x
            else:
                # Ball moving up - anticipate where it might come back
                # Stay closer to center but bias toward ball's X
                self.target_x = target_ball.x * 0.7 + CANVAS_WIDTH / 2 * 0.3
        else:
            self.target_x = CANVAS_WIDTH / 2

        # Move toward target - be aggressive
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Smaller dead zone for precision
        dead_zone = 8

        if diff < -dead_zone:
            return 0  # Left
        elif diff > dead_zone:
            return 2  # Right
        else:
            return 1  # Stay

    def _get_target_ball(self, balls: List[Ball], paddle) -> Optional[Ball]:
        """Get the most dangerous ball."""
        paddle_y = paddle.y

        # Balls moving down
        down_balls = [b for b in balls if b.vy > 0 and b.is_launched]

        if down_balls:
            # Pick the one closest to paddle (most urgent)
            return max(down_balls, key=lambda b: b.y)

        # No balls moving down - pick lowest launched ball
        launched_balls = [b for b in balls if b.is_launched]
        if launched_balls:
            return max(launched_balls, key=lambda b: b.y)

        return balls[0] if balls else None

    def _predict_landing_x(self, ball: Ball, paddle_y: float) -> float:
        """
        Predict where ball will land at paddle Y.
        Only accounts for wall bounces (no blocks).
        """
        if ball.vy <= 0:
            return ball.x

        x = ball.x
        y = ball.y
        vx = ball.vx
        vy = ball.vy

        max_iterations = 100
        iteration = 0

        while y < paddle_y and iteration < max_iterations:
            iteration += 1

            # Time to reach paddle
            time_to_paddle = (paddle_y - y) / vy

            # Time to hit wall
            if vx < 0:
                time_to_wall = (BALL_RADIUS - x) / vx
            elif vx > 0:
                time_to_wall = (CANVAS_WIDTH - BALL_RADIUS - x) / vx
            else:
                time_to_wall = float('inf')

            if time_to_wall > 0 and time_to_wall < time_to_paddle:
                # Hit wall first
                x += vx * time_to_wall
                y += vy * time_to_wall
                vx = -vx
                x = max(BALL_RADIUS, min(CANVAS_WIDTH - BALL_RADIUS, x))
            else:
                # Reach paddle
                x += vx * time_to_paddle
                break

        return max(BALL_RADIUS, min(CANVAS_WIDTH - BALL_RADIUS, x))


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

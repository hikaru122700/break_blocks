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
    Aggressive Expert AI.

    Always moves aggressively toward predicted landing position.
    No hesitation, minimal dead zone.
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2

    def get_action(self, game: GameSimulation) -> int:
        """Get action - always move aggressively toward target."""
        paddle = game.paddle
        balls = game.balls

        if not balls:
            return 1  # Stay

        # Find the most urgent ball
        target_ball = self._get_target_ball(balls, paddle)

        if target_ball:
            if target_ball.vy > 0:
                # Ball moving down - predict landing
                self.target_x = self._predict_landing_x(target_ball, paddle.y)
            else:
                # Ball moving up - stay near its X position
                self.target_x = target_ball.x
        else:
            self.target_x = CANVAS_WIDTH / 2

        # Move toward target - very aggressive, tiny dead zone
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Minimal dead zone - only stop when very close
        if diff < -3:
            return 0  # Left
        elif diff > 3:
            return 2  # Right
        else:
            return 1  # Stay

    def _get_target_ball(self, balls: List[Ball], paddle) -> Optional[Ball]:
        """Get the most urgent ball (closest to paddle, moving down)."""
        paddle_y = paddle.y

        # Filter balls moving down
        down_balls = [b for b in balls if b.vy > 0 and b.is_launched]

        if down_balls:
            # Pick the one that will reach paddle soonest
            def time_to_paddle(ball):
                if ball.vy <= 0:
                    return float('inf')
                return (paddle_y - ball.y) / ball.vy

            return min(down_balls, key=time_to_paddle)

        # No balls moving down - pick lowest ball
        launched_balls = [b for b in balls if b.is_launched]
        if launched_balls:
            return max(launched_balls, key=lambda b: b.y)

        return balls[0] if balls else None

    def _predict_landing_x(self, ball: Ball, paddle_y: float) -> float:
        """Predict where ball will be at paddle Y level."""
        if ball.vy <= 0:
            return ball.x

        x = ball.x
        y = ball.y
        vx = ball.vx
        vy = ball.vy

        max_bounces = 50
        bounces = 0

        while y < paddle_y and bounces < max_bounces:
            # Time to reach paddle
            t_paddle = (paddle_y - y) / vy

            # Time to hit left/right wall
            if vx < 0:
                t_wall = (BALL_RADIUS - x) / vx
            elif vx > 0:
                t_wall = (CANVAS_WIDTH - BALL_RADIUS - x) / vx
            else:
                t_wall = float('inf')

            if t_wall > 0 and t_wall < t_paddle:
                # Bounce off wall
                x += vx * t_wall
                y += vy * t_wall
                vx = -vx
                x = max(BALL_RADIUS, min(CANVAS_WIDTH - BALL_RADIUS, x))
                bounces += 1
            else:
                # Reach paddle
                x += vx * t_paddle
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

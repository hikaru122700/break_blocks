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
    Expert AI with trajectory prediction.

    Predicts where the ball will land at paddle height,
    accounting for wall bounces.
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2

    def get_action(self, game: GameSimulation) -> int:
        """Get action based on ball trajectory prediction."""
        paddle = game.paddle
        balls = game.balls

        if not balls:
            return 1  # Stay

        # Find the most dangerous ball (moving toward paddle)
        target_ball = self._get_target_ball(balls, paddle)

        if target_ball and target_ball.vy > 0:
            # Predict where ball will land
            self.target_x = self._predict_landing_x(target_ball, paddle.y)
        elif target_ball:
            # Ball moving up - just track its current x
            self.target_x = target_ball.x
        else:
            # Default to center
            self.target_x = CANVAS_WIDTH / 2

        # Move toward target
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Adjust dead zone based on urgency
        dead_zone = 10
        if target_ball and target_ball.vy > 0:
            # Ball is coming - smaller dead zone for precision
            time_to_paddle = (paddle.y - target_ball.y) / target_ball.vy if target_ball.vy > 0 else 999
            if time_to_paddle < 30:  # Ball is close
                dead_zone = 5

        if diff < -dead_zone:
            return 0  # Left
        elif diff > dead_zone:
            return 2  # Right
        else:
            return 1  # Stay

    def _get_target_ball(self, balls: List[Ball], paddle) -> Optional[Ball]:
        """Get the ball we should focus on (most dangerous)."""
        paddle_y = paddle.y

        # Priority: balls moving down toward paddle
        down_balls = [b for b in balls if b.vy > 0 and b.is_launched]

        if down_balls:
            # Pick the one that will reach paddle first
            def time_to_paddle(ball):
                if ball.vy <= 0:
                    return float('inf')
                return (paddle_y - ball.y) / ball.vy

            return min(down_balls, key=time_to_paddle)

        # No balls moving down - pick lowest launched ball
        launched_balls = [b for b in balls if b.is_launched]
        if launched_balls:
            return max(launched_balls, key=lambda b: b.y)

        return balls[0] if balls else None

    def _predict_landing_x(self, ball: Ball, paddle_y: float) -> float:
        """
        Predict where the ball will be when it reaches paddle Y.
        Accounts for wall bounces.
        """
        if ball.vy <= 0:
            return ball.x  # Ball not moving down

        # Simulate ball trajectory
        x = ball.x
        y = ball.y
        vx = ball.vx
        vy = ball.vy

        # Maximum iterations to prevent infinite loop
        max_iterations = 1000
        iteration = 0

        while y < paddle_y and iteration < max_iterations:
            iteration += 1

            # Calculate time to reach paddle Y
            time_to_paddle = (paddle_y - y) / vy

            # Calculate time to hit left/right wall
            if vx < 0:
                time_to_wall = (BALL_RADIUS - x) / vx
            elif vx > 0:
                time_to_wall = (CANVAS_WIDTH - BALL_RADIUS - x) / vx
            else:
                time_to_wall = float('inf')

            if time_to_wall > 0 and time_to_wall < time_to_paddle:
                # Ball hits wall before reaching paddle
                x += vx * time_to_wall
                y += vy * time_to_wall
                vx = -vx  # Reflect off wall

                # Clamp x to valid range
                x = max(BALL_RADIUS, min(CANVAS_WIDTH - BALL_RADIUS, x))
            else:
                # Ball reaches paddle level
                x += vx * time_to_paddle
                y = paddle_y
                break

        # Clamp to canvas
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

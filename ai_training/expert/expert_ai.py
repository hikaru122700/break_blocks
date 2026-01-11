"""Rule-based Expert AI for demonstration collection."""

import math
from typing import Optional, List
from ..env.game_simulation import GameSimulation, Ball
from ..env.constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED, PADDLE_Y_OFFSET,
    BALL_RADIUS,
    PowerUpType
)


class ExpertAI:
    """
    High-performance rule-based AI for Break Blocks.

    Strategy (prioritized):
    1. ALWAYS intercept the most urgent ball (survival first)
    2. Only chase powerups if ball is safe
    3. Use paddle center for better control
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2
        self.last_action = 1

    def get_action(self, game: GameSimulation) -> int:
        """
        Get optimal action based on current game state.

        Returns:
            Action: 0=left, 1=stay, 2=right
        """
        paddle = game.paddle
        balls = game.balls
        powerups = [p for p in game.powerups if p.is_active]

        if not balls:
            return 1

        # Get the most urgent ball
        urgent_ball, time_to_impact = self._get_most_urgent_ball(balls, paddle)

        if urgent_ball is None:
            # All balls moving up, we're safe
            self.target_x = self._get_powerup_target(powerups, paddle) or CANVAS_WIDTH / 2
        else:
            # Calculate where ball will land
            landing_x = self._predict_landing_precise(urgent_ball, paddle.y)

            # Only chase powerup if ball is far away (> 1 second)
            if time_to_impact > 60:  # 60 frames = 1 second
                powerup_target = self._get_safe_powerup_target(powerups, paddle, landing_x)
                if powerup_target is not None:
                    self.target_x = powerup_target
                else:
                    self.target_x = landing_x
            else:
                # Ball is close, focus on interception only
                self.target_x = landing_x

        # Move paddle toward target
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Dynamic dead zone based on urgency
        dead_zone = 3

        if diff < -dead_zone:
            return 0  # Left
        elif diff > dead_zone:
            return 2  # Right
        else:
            return 1  # Stay

    def _get_most_urgent_ball(self, balls: List[Ball], paddle) -> tuple:
        """
        Find the ball that will reach paddle level soonest.

        Returns:
            (ball, time_to_impact) or (None, inf) if no ball is moving down
        """
        paddle_y = paddle.y
        most_urgent = None
        min_time = float('inf')

        for ball in balls:
            if ball.vy <= 0:
                # Ball moving up, not urgent
                continue

            # Time to reach paddle level
            distance = paddle_y - ball.y - BALL_RADIUS
            if distance <= 0:
                # Ball already at or past paddle
                time = 0
            else:
                time = distance / ball.vy

            if time < min_time:
                min_time = time
                most_urgent = ball

        return most_urgent, min_time

    def _predict_landing_precise(self, ball: Ball, paddle_y: float) -> float:
        """
        Precisely predict where ball will land using physics simulation.
        Accounts for wall bounces accurately.
        """
        x = ball.x
        y = ball.y
        vx = ball.vx
        vy = ball.vy

        if vy <= 0:
            # Ball moving up, return current x
            return x

        # Simulate ball trajectory
        max_iterations = 2000
        dt = 0.5  # Small time step for accuracy

        for _ in range(max_iterations):
            # Move ball
            x += vx * dt
            y += vy * dt

            # Wall bounces (left and right walls)
            if x - BALL_RADIUS < 0:
                x = BALL_RADIUS - (x - BALL_RADIUS)
                vx = abs(vx)
            elif x + BALL_RADIUS > CANVAS_WIDTH:
                x = CANVAS_WIDTH - BALL_RADIUS - (x + BALL_RADIUS - CANVAS_WIDTH)
                vx = -abs(vx)

            # Check if reached paddle level
            if y + BALL_RADIUS >= paddle_y:
                # Clamp to valid range
                return max(PADDLE_WIDTH/2, min(CANVAS_WIDTH - PADDLE_WIDTH/2, x))

        return x

    def _get_powerup_target(self, powerups: list, paddle) -> Optional[float]:
        """Get target for powerup when safe."""
        if not powerups:
            return None

        # Priority based on value
        priority = {
            PowerUpType.MULTI_BALL: 100,
            PowerUpType.PENETRATE: 90,
            PowerUpType.TIME_EXTEND: 60,
            PowerUpType.SPEED_DOWN: 40,
            PowerUpType.SPEED_UP: 10,
        }

        paddle_y = paddle.y
        best_powerup = None
        best_score = 0

        for p in powerups:
            # Only consider powerups above paddle and falling
            if p.y > paddle_y - 50:  # Very close
                continue

            prio = priority.get(p.powerup_type, 0)
            if prio > best_score:
                best_score = prio
                best_powerup = p

        if best_powerup:
            return best_powerup.x

        return None

    def _get_safe_powerup_target(
        self,
        powerups: list,
        paddle,
        ball_landing_x: float
    ) -> Optional[float]:
        """
        Get powerup target only if we can get it AND get back to ball.
        """
        if not powerups:
            return None

        priority = {
            PowerUpType.MULTI_BALL: 100,
            PowerUpType.PENETRATE: 90,
            PowerUpType.TIME_EXTEND: 60,
            PowerUpType.SPEED_DOWN: 40,
            PowerUpType.SPEED_UP: 10,
        }

        paddle_y = paddle.y
        paddle_center = paddle.x + PADDLE_WIDTH / 2

        best_powerup = None
        best_score = 0

        for p in powerups:
            # Skip powerups that already passed or too far
            if p.y > paddle_y - 30 or p.y < paddle_y - 300:
                continue

            # Calculate if we can reach powerup and get back to ball
            distance_to_powerup = abs(p.x - paddle_center)
            distance_powerup_to_ball = abs(p.x - ball_landing_x)
            total_distance = distance_to_powerup + distance_powerup_to_ball

            # Estimate time available (powerup falling speed ~3 pixels/frame)
            powerup_fall_time = (paddle_y - p.y) / 3
            # We can move ~10 pixels/frame
            time_needed = total_distance / PADDLE_SPEED

            # Only go for it if we have enough time with margin
            if time_needed < powerup_fall_time * 0.7:
                prio = priority.get(p.powerup_type, 0)
                if prio > best_score:
                    best_score = prio
                    best_powerup = p

        if best_powerup and best_score >= 40:  # Only valuable powerups
            return best_powerup.x

        return None


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

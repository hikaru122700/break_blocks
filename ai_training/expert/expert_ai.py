"""Rule-based Expert AI for demonstration collection."""

from typing import Optional, Tuple
from ..env.game_simulation import GameSimulation
from ..env.constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_SPEED,
    PowerUpType
)


class ExpertAI:
    """
    Rule-based AI that plays Break Blocks optimally.

    Strategy:
    1. Predict where ball will land on paddle level
    2. Move to intercept ball
    3. Prioritize valuable powerups (multi-ball, penetrate)
    4. Use ball trajectory to aim at blocks
    """

    def __init__(self):
        self.target_x = CANVAS_WIDTH / 2

    def get_action(self, game: GameSimulation) -> int:
        """
        Get optimal action based on current game state.

        Args:
            game: Current game simulation state

        Returns:
            Action: 0=left, 1=stay, 2=right
        """
        paddle = game.paddle
        balls = game.balls
        powerups = [p for p in game.powerups if p.is_active]

        # If no balls, stay
        if not balls:
            return 1

        # Calculate target position
        self.target_x = self._calculate_target(game, balls, powerups, paddle)

        # Determine action based on target
        paddle_center = paddle.x + PADDLE_WIDTH / 2
        diff = self.target_x - paddle_center

        # Dead zone to prevent jitter
        dead_zone = 5

        if diff < -dead_zone:
            return 0  # Left
        elif diff > dead_zone:
            return 2  # Right
        else:
            return 1  # Stay

    def _calculate_target(
        self,
        game: GameSimulation,
        balls: list,
        powerups: list,
        paddle
    ) -> float:
        """Calculate optimal paddle target position."""

        # Priority 1: Catch valuable powerups near paddle
        powerup_target = self._get_powerup_target(powerups, paddle)
        if powerup_target is not None:
            return powerup_target

        # Priority 2: Intercept the most dangerous ball
        return self._get_ball_intercept(balls, paddle)

    def _get_powerup_target(self, powerups: list, paddle) -> Optional[float]:
        """
        Get target position for catching powerups.

        Only targets powerups that are:
        1. Valuable (multi_ball, penetrate, time_extend)
        2. Close to paddle (within reach)
        """
        if not powerups:
            return None

        # Powerup priority (higher = more important)
        priority = {
            PowerUpType.MULTI_BALL: 100,
            PowerUpType.PENETRATE: 90,
            PowerUpType.TIME_EXTEND: 70,
            PowerUpType.SPEED_DOWN: 50,
            PowerUpType.SPEED_UP: 10,
        }

        paddle_y = paddle.y
        best_powerup = None
        best_score = 0

        for p in powerups:
            # Only consider powerups that are falling towards paddle
            if p.y < paddle_y - 200:  # Too far
                continue
            if p.y > paddle_y:  # Already passed
                continue

            # Calculate score based on priority and proximity
            prio = priority.get(p.powerup_type, 0)
            distance_factor = 1.0 - (paddle_y - p.y) / 200.0  # Closer = higher
            score = prio * distance_factor

            if score > best_score:
                best_score = score
                best_powerup = p

        # Only go for powerup if it's valuable enough
        if best_powerup and best_score > 30:
            return best_powerup.x

        return None

    def _get_ball_intercept(self, balls: list, paddle) -> float:
        """
        Predict where the most dangerous ball will land.
        """
        if not balls:
            return CANVAS_WIDTH / 2

        paddle_y = paddle.y
        best_ball = None
        best_urgency = -1

        for ball in balls:
            # Only consider balls moving downward
            if ball.vy <= 0:
                continue

            # Calculate time to reach paddle level
            if ball.vy > 0:
                time_to_paddle = (paddle_y - ball.y) / ball.vy
            else:
                time_to_paddle = float('inf')

            # Urgency: inverse of time (closer = more urgent)
            urgency = 1.0 / max(time_to_paddle, 0.1)

            if urgency > best_urgency:
                best_urgency = urgency
                best_ball = ball

        # If no ball is moving down, pick the lowest one
        if best_ball is None:
            best_ball = max(balls, key=lambda b: b.y)

        # Predict landing position
        landing_x = self._predict_landing(best_ball, paddle_y)

        return landing_x

    def _predict_landing(self, ball, paddle_y: float) -> float:
        """
        Predict where the ball will land on paddle level.
        Accounts for wall bounces.
        """
        x = ball.x
        y = ball.y
        vx = ball.vx
        vy = ball.vy

        # If ball is moving up, use current x
        if vy <= 0:
            return x

        # Simulate ball trajectory
        max_steps = 1000
        dt = 1.0  # Time step for simulation

        for _ in range(max_steps):
            # Move ball
            x += vx * dt
            y += vy * dt

            # Wall bounces
            if x < 0:
                x = -x
                vx = -vx
            elif x > CANVAS_WIDTH:
                x = 2 * CANVAS_WIDTH - x
                vx = -vx

            # Check if reached paddle level
            if y >= paddle_y:
                return max(0, min(CANVAS_WIDTH, x))

        return x


class ExpertAIWithNoise(ExpertAI):
    """
    Expert AI with optional noise for data augmentation.
    """

    def __init__(self, noise_level: float = 0.0):
        """
        Args:
            noise_level: Probability of taking a random action (0-1)
        """
        super().__init__()
        self.noise_level = noise_level

    def get_action(self, game: GameSimulation) -> int:
        import random

        # Occasionally take random action
        if random.random() < self.noise_level:
            return random.randint(0, 2)

        return super().get_action(game)

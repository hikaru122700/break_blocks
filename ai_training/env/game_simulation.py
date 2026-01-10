"""Game simulation for RL training - ports JS game logic to Python."""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from .constants import (
    CANVAS_WIDTH, CANVAS_HEIGHT,
    PADDLE_WIDTH, PADDLE_HEIGHT, PADDLE_SPEED, PADDLE_Y_OFFSET,
    BALL_RADIUS, BALL_BASE_SPEED,
    BLOCK_WIDTH, BLOCK_HEIGHT, BLOCK_PADDING, BLOCK_OFFSET_TOP, BLOCK_OFFSET_LEFT,
    BLOCK_COLS, BLOCK_ROWS,
    INITIAL_LIVES, MAX_STAGES, STAGE_TIME_LIMIT, TIME_EXTEND_AMOUNT,
    BlockType, BLOCK_CONFIG,
    PowerUpType, POWERUP_CONFIG, POWERUP_WEIGHTS,
    get_stage_data, FRAME_SKIP, PADDLE_MOVE_AMOUNT,
    MAX_BALLS, MAX_POWERUPS
)


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value between min and max."""
    return max(min_val, min(max_val, value))


@dataclass
class Ball:
    """Ball entity."""
    x: float
    y: float
    radius: float = BALL_RADIUS
    speed: float = BALL_BASE_SPEED
    vx: float = 0.0
    vy: float = 0.0
    is_active: bool = True
    is_launched: bool = False
    is_penetrating: bool = False
    penetrate_timer: float = 0.0

    def launch(self):
        """Launch the ball at a random upward angle."""
        if self.is_launched:
            return
        self.is_launched = True
        # Random angle between -45 and -135 degrees (upward)
        angle = random.uniform(-math.pi / 4, -3 * math.pi / 4)
        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed

    def update(self, dt: float, paddle: 'Paddle'):
        """Update ball position."""
        if not self.is_launched:
            # Ball follows paddle before launch
            self.x = paddle.x + paddle.width / 2
            self.y = paddle.y - self.radius - 2
            return

        # Update position (normalize to 60fps)
        time_factor = dt / 16.67
        self.x += self.vx * time_factor
        self.y += self.vy * time_factor

        # Update penetrate timer
        if self.is_penetrating and self.penetrate_timer > 0:
            self.penetrate_timer -= dt
            if self.penetrate_timer <= 0:
                self.is_penetrating = False

    def reflect(self, normal_x: float, normal_y: float):
        """Reflect ball velocity off a surface."""
        # Reflection formula: v' = v - 2(v.n)n
        dot = self.vx * normal_x + self.vy * normal_y
        self.vx -= 2 * dot * normal_x
        self.vy -= 2 * dot * normal_y
        self.normalize_velocity()

    def reflect_from_paddle(self, normalized_position: float):
        """Reflect ball from paddle based on hit position."""
        # normalized_position: -1 (left edge) to 1 (right edge)
        # Map to angle: -120 degrees to -60 degrees (upward)
        min_angle = -math.pi * 2 / 3  # -120 degrees
        max_angle = -math.pi / 3       # -60 degrees
        angle = min_angle + (normalized_position + 1) / 2 * (max_angle - min_angle)

        self.vx = math.cos(angle) * self.speed
        self.vy = math.sin(angle) * self.speed

        # Ensure ball is moving upward
        if self.vy > 0:
            self.vy = -self.vy

    def normalize_velocity(self):
        """Normalize velocity to maintain consistent speed."""
        current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
        if current_speed > 0:
            self.vx = (self.vx / current_speed) * self.speed
            self.vy = (self.vy / current_speed) * self.speed

    def set_speed(self, new_speed: float):
        """Set ball speed and normalize velocity."""
        self.speed = new_speed
        self.normalize_velocity()

    def set_penetrating(self, duration: float):
        """Enable penetrating mode."""
        self.is_penetrating = True
        self.penetrate_timer = duration

    def reset(self, paddle: 'Paddle'):
        """Reset ball to paddle position."""
        self.x = paddle.x + paddle.width / 2
        self.y = paddle.y - self.radius - 2
        self.vx = 0.0
        self.vy = 0.0
        self.is_launched = False
        self.is_penetrating = False
        self.penetrate_timer = 0.0
        self.speed = BALL_BASE_SPEED

    def clone(self) -> 'Ball':
        """Create a copy of this ball for multi-ball."""
        new_ball = Ball(x=self.x, y=self.y)
        new_ball.speed = self.speed
        new_ball.is_launched = True
        new_ball.is_penetrating = self.is_penetrating
        new_ball.penetrate_timer = self.penetrate_timer
        return new_ball


@dataclass
class Paddle:
    """Paddle entity."""
    x: float = field(default_factory=lambda: (CANVAS_WIDTH - PADDLE_WIDTH) / 2)
    y: float = CANVAS_HEIGHT - PADDLE_Y_OFFSET
    width: float = PADDLE_WIDTH
    height: float = PADDLE_HEIGHT
    speed: float = PADDLE_SPEED
    velocity: float = 0.0  # Current movement velocity

    def move_left(self, dt: float):
        """Move paddle left."""
        time_factor = dt / 16.67
        self.x -= self.speed * time_factor
        self.x = clamp(self.x, 0, CANVAS_WIDTH - self.width)
        self.velocity = -self.speed

    def move_right(self, dt: float):
        """Move paddle right."""
        time_factor = dt / 16.67
        self.x += self.speed * time_factor
        self.x = clamp(self.x, 0, CANVAS_WIDTH - self.width)
        self.velocity = self.speed

    def stay(self):
        """Stop paddle movement."""
        self.velocity = 0.0

    def get_collision_normal(self, ball_x: float) -> float:
        """Get normalized hit position on paddle (-1 to 1)."""
        relative_x = (ball_x - self.x) / self.width
        return clamp(relative_x * 2 - 1, -1, 1)

    def reset(self):
        """Reset paddle to center."""
        self.x = (CANVAS_WIDTH - self.width) / 2
        self.width = PADDLE_WIDTH
        self.velocity = 0.0


@dataclass
class Block:
    """Block entity."""
    x: float
    y: float
    block_type: int
    width: float = BLOCK_WIDTH
    height: float = BLOCK_HEIGHT
    is_destroyed: bool = False
    hit_points: int = 1
    max_hit_points: int = 1
    score: int = 100

    def __post_init__(self):
        config = BLOCK_CONFIG.get(self.block_type)
        if config:
            self.score = config.score
            self.hit_points = config.hit_points
            self.max_hit_points = config.hit_points

    def hit(self) -> Tuple[int, bool]:
        """Hit the block. Returns (score, destroyed)."""
        self.hit_points -= 1
        if self.hit_points <= 0:
            self.is_destroyed = True
            return (self.score, True)
        return (0, False)


class DurableBlock(Block):
    """Durable block that requires multiple hits."""
    pass


@dataclass
class PowerUp:
    """Power-up entity."""
    x: float
    y: float
    powerup_type: str
    width: float = 20
    height: float = 20
    speed: float = 3.0
    is_active: bool = True

    def update(self, dt: float):
        """Update power-up position (falling)."""
        time_factor = dt / 16.67
        self.y += self.speed * time_factor

        # Deactivate if off screen
        if self.y > CANVAS_HEIGHT:
            self.is_active = False


class CollisionSystem:
    """Collision detection system."""

    @staticmethod
    def circle_rect_intersect(cx: float, cy: float, radius: float,
                               rx: float, ry: float, rw: float, rh: float) -> bool:
        """Check circle-rectangle intersection."""
        closest_x = clamp(cx, rx, rx + rw)
        closest_y = clamp(cy, ry, ry + rh)

        dist_x = cx - closest_x
        dist_y = cy - closest_y
        dist_sq = dist_x ** 2 + dist_y ** 2

        return dist_sq < radius ** 2

    @staticmethod
    def rect_intersect(x1: float, y1: float, w1: float, h1: float,
                       x2: float, y2: float, w2: float, h2: float) -> bool:
        """Check rectangle-rectangle intersection."""
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)

    @staticmethod
    def get_block_collision_normal(ball: Ball, block: Block) -> Tuple[float, float]:
        """Get collision normal for ball vs block."""
        dx = ball.x - (block.x + block.width / 2)
        dy = ball.y - (block.y + block.height / 2)

        half_width = block.width / 2 + ball.radius
        half_height = block.height / 2 + ball.radius

        overlap_x = half_width - abs(dx)
        overlap_y = half_height - abs(dy)

        if overlap_x < overlap_y:
            return (1 if dx > 0 else -1, 0)
        else:
            return (0, 1 if dy > 0 else -1)


class GameSimulation:
    """Full game simulation for RL training."""

    def __init__(self, stage_number: int = 1):
        self.stage_number = stage_number
        self.lives = INITIAL_LIVES
        self.score = 0
        self.combo = 0
        self.combo_timer = 0.0
        self.combo_timeout = 2000.0

        # Time tracking
        stage_data = get_stage_data(stage_number)
        self.time_remaining = stage_data['time_limit']
        self.time_limit = stage_data['time_limit']
        self.ball_speed_multiplier = stage_data['ball_speed_multiplier']
        self.powerup_chance = stage_data['powerup_chance']

        # Entities
        self.paddle = Paddle()
        self.balls: List[Ball] = []
        self.blocks: List[Block] = []
        self.powerups: List[PowerUp] = []

        # Game state
        self.is_game_over = False
        self.is_stage_clear = False
        self.is_time_up = False

        # Speed modifier from power-ups
        self.speed_modifier = 1.0
        self.speed_modifier_timer = 0.0

        # Collision system
        self.collision = CollisionSystem()

        # Initialize
        self._load_stage(stage_number)
        self._spawn_ball()

    def _load_stage(self, stage_number: int):
        """Load stage layout."""
        stage_data = get_stage_data(stage_number)
        layout = stage_data['layout']

        self.blocks = []
        for row_idx, row in enumerate(layout):
            for col_idx, block_type in enumerate(row):
                if block_type != BlockType.NONE:
                    x = BLOCK_OFFSET_LEFT + col_idx * (BLOCK_WIDTH + BLOCK_PADDING)
                    y = BLOCK_OFFSET_TOP + row_idx * (BLOCK_HEIGHT + BLOCK_PADDING)

                    if block_type in (BlockType.DURABLE_1, BlockType.DURABLE_2):
                        block = DurableBlock(x=x, y=y, block_type=block_type)
                    else:
                        block = Block(x=x, y=y, block_type=block_type)
                    self.blocks.append(block)

    def _spawn_ball(self):
        """Spawn a new ball on the paddle."""
        ball = Ball(
            x=self.paddle.x + self.paddle.width / 2,
            y=self.paddle.y - BALL_RADIUS - 2
        )
        ball.speed = BALL_BASE_SPEED * self.ball_speed_multiplier
        self.balls = [ball]

    def step(self, action: int, dt: float = 16.67) -> Dict[str, Any]:
        """
        Execute one step of the simulation.

        Args:
            action: 0=left, 1=stay, 2=right
            dt: Delta time in milliseconds

        Returns:
            Dictionary with step results (rewards, events, etc.)
        """
        events = {
            'blocks_destroyed': 0,
            'block_scores': [],
            'combo': self.combo,
            'life_lost': False,
            'powerups_collected': [],
            'paddle_hits': 0,
            'stage_clear': False,
            'game_over': False,
            'time_up': False,
        }

        if self.is_game_over or self.is_stage_clear:
            return events

        # Apply action
        if action == 0:
            self.paddle.move_left(dt)
        elif action == 2:
            self.paddle.move_right(dt)
        else:
            self.paddle.stay()

        # Auto-launch ball if not launched
        for ball in self.balls:
            if not ball.is_launched:
                ball.launch()

        # Update entities
        for ball in self.balls:
            ball.update(dt, self.paddle)

        for powerup in self.powerups:
            powerup.update(dt)

        # Update timers
        self._update_timers(dt)

        # Check collisions
        self._check_ball_wall_collisions()
        events['paddle_hits'] = self._check_ball_paddle_collisions()
        block_events = self._check_ball_block_collisions()
        events['blocks_destroyed'] = block_events['destroyed']
        events['block_scores'] = block_events['scores']

        powerup_events = self._check_powerup_collisions()
        events['powerups_collected'] = powerup_events

        # Check ball lost
        self._check_balls_lost(events)

        # Check stage clear
        active_blocks = sum(1 for b in self.blocks if not b.is_destroyed)
        if active_blocks == 0:
            self.is_stage_clear = True
            events['stage_clear'] = True

        # Check time up
        if self.time_remaining <= 0:
            self.is_time_up = True
            events['time_up'] = True
            self._lose_life()
            if self.is_game_over:
                events['game_over'] = True

        events['combo'] = self.combo
        return events

    def _update_timers(self, dt: float):
        """Update various timers."""
        # Time remaining (only if ball is launched)
        if self.balls and self.balls[0].is_launched:
            self.time_remaining -= dt
            if self.time_remaining < 0:
                self.time_remaining = 0

        # Combo timer
        if self.combo_timer > 0:
            self.combo_timer -= dt
            if self.combo_timer <= 0:
                self.combo = 0

        # Speed modifier timer
        if self.speed_modifier_timer > 0:
            self.speed_modifier_timer -= dt
            if self.speed_modifier_timer <= 0:
                self.speed_modifier = 1.0
                for ball in self.balls:
                    ball.set_speed(BALL_BASE_SPEED * self.ball_speed_multiplier)

    def _check_ball_wall_collisions(self):
        """Check and handle ball-wall collisions."""
        for ball in self.balls:
            if not ball.is_launched:
                continue

            # Left wall
            if ball.x - ball.radius <= 0:
                ball.x = ball.radius
                ball.reflect(1, 0)

            # Right wall
            elif ball.x + ball.radius >= CANVAS_WIDTH:
                ball.x = CANVAS_WIDTH - ball.radius
                ball.reflect(-1, 0)

            # Top wall
            if ball.y - ball.radius <= 0:
                ball.y = ball.radius
                ball.reflect(0, 1)

    def _check_ball_paddle_collisions(self) -> int:
        """Check and handle ball-paddle collisions.

        Returns:
            Number of paddle hits this frame
        """
        paddle_hits = 0
        for ball in self.balls:
            if not ball.is_launched or ball.vy <= 0:
                continue

            if self.collision.circle_rect_intersect(
                ball.x, ball.y, ball.radius,
                self.paddle.x, self.paddle.y, self.paddle.width, self.paddle.height
            ):
                # Get normalized hit position
                norm_pos = self.paddle.get_collision_normal(ball.x)
                ball.reflect_from_paddle(norm_pos)

                # Ensure ball is above paddle
                ball.y = self.paddle.y - ball.radius - 1
                paddle_hits += 1

        return paddle_hits

    def _check_ball_block_collisions(self) -> Dict[str, Any]:
        """Check and handle ball-block collisions."""
        result = {'destroyed': 0, 'scores': []}

        for ball in self.balls:
            if not ball.is_launched:
                continue

            for block in self.blocks:
                if block.is_destroyed:
                    continue

                if self.collision.circle_rect_intersect(
                    ball.x, ball.y, ball.radius,
                    block.x, block.y, block.width, block.height
                ):
                    # Hit the block
                    score, destroyed = block.hit()

                    if destroyed:
                        result['destroyed'] += 1
                        result['scores'].append(score)

                        # Update combo
                        self.combo += 1
                        self.combo_timer = self.combo_timeout

                        # Maybe spawn power-up
                        if random.random() < self.powerup_chance:
                            self._spawn_powerup(block.x + block.width / 2, block.y)

                    # Reflect ball (unless penetrating)
                    if not ball.is_penetrating:
                        nx, ny = self.collision.get_block_collision_normal(ball, block)
                        ball.reflect(nx, ny)
                        break  # One collision per ball per frame

        return result

    def _check_powerup_collisions(self) -> List[str]:
        """Check and handle power-up collisions with paddle."""
        collected = []

        for powerup in self.powerups[:]:
            if not powerup.is_active:
                continue

            if self.collision.rect_intersect(
                powerup.x - powerup.width / 2, powerup.y,
                powerup.width, powerup.height,
                self.paddle.x, self.paddle.y,
                self.paddle.width, self.paddle.height
            ):
                self._apply_powerup(powerup.powerup_type)
                collected.append(powerup.powerup_type)
                powerup.is_active = False

        # Remove inactive power-ups
        self.powerups = [p for p in self.powerups if p.is_active]

        return collected

    def _spawn_powerup(self, x: float, y: float):
        """Spawn a random power-up at given position."""
        # Weighted random selection
        total_weight = sum(POWERUP_WEIGHTS.values())
        rand = random.random() * total_weight
        cumulative = 0

        for ptype, weight in POWERUP_WEIGHTS.items():
            cumulative += weight
            if rand <= cumulative:
                self.powerups.append(PowerUp(x=x, y=y, powerup_type=ptype))
                break

    def _apply_powerup(self, powerup_type: str):
        """Apply power-up effect."""
        if powerup_type == PowerUpType.MULTI_BALL:
            # Clone existing balls
            new_balls = []
            for ball in self.balls:
                if ball.is_launched:
                    clone = ball.clone()
                    # Rotate velocity slightly
                    angle = math.atan2(clone.vy, clone.vx) + math.pi / 6
                    clone.vx = math.cos(angle) * clone.speed
                    clone.vy = math.sin(angle) * clone.speed
                    new_balls.append(clone)

                    clone2 = ball.clone()
                    angle2 = math.atan2(clone2.vy, clone2.vx) - math.pi / 6
                    clone2.vx = math.cos(angle2) * clone2.speed
                    clone2.vy = math.sin(angle2) * clone2.speed
                    new_balls.append(clone2)

            self.balls.extend(new_balls)

        elif powerup_type == PowerUpType.SPEED_UP:
            config = POWERUP_CONFIG[PowerUpType.SPEED_UP]
            self.speed_modifier = config.modifier
            self.speed_modifier_timer = config.duration
            for ball in self.balls:
                ball.set_speed(BALL_BASE_SPEED * self.ball_speed_multiplier * self.speed_modifier)

        elif powerup_type == PowerUpType.SPEED_DOWN:
            config = POWERUP_CONFIG[PowerUpType.SPEED_DOWN]
            self.speed_modifier = config.modifier
            self.speed_modifier_timer = config.duration
            for ball in self.balls:
                ball.set_speed(BALL_BASE_SPEED * self.ball_speed_multiplier * self.speed_modifier)

        elif powerup_type == PowerUpType.PENETRATE:
            config = POWERUP_CONFIG[PowerUpType.PENETRATE]
            for ball in self.balls:
                ball.set_penetrating(config.duration)

        elif powerup_type == PowerUpType.TIME_EXTEND:
            self.time_remaining += TIME_EXTEND_AMOUNT

    def _check_balls_lost(self, events: Dict[str, Any]):
        """Check if balls are lost (fell below paddle)."""
        for ball in self.balls[:]:
            if ball.y - ball.radius > CANVAS_HEIGHT:
                ball.is_active = False

        # Remove inactive balls
        self.balls = [b for b in self.balls if b.is_active]

        # If all balls lost, lose a life
        if not self.balls:
            events['life_lost'] = True
            self._lose_life()
            if self.is_game_over:
                events['game_over'] = True

    def _lose_life(self):
        """Handle losing a life."""
        self.lives -= 1
        self.combo = 0
        self.combo_timer = 0

        if self.lives <= 0:
            self.is_game_over = True
        else:
            # Respawn ball
            self._spawn_ball()

    def get_observation(self) -> List[float]:
        """
        Get current observation vector.

        Returns 216-dimensional observation:
        - Ball 1: x, y, vx, vy, speed, is_penetrating (6)
        - Ball 2: same (6) - zeros if no second ball
        - Paddle: x, velocity (2)
        - Time: remaining / limit (1)
        - Block existence: 12x8 grid (96)
        - Block HP: 12x8 grid normalized (96)
        - Power-up: speed_modifier, penetrating, ball_count, nearest_powerup_x, nearest_powerup_y (5)
        - Game state: lives, stage, remaining_blocks_ratio (3)
        - Stage speed: ball_speed_multiplier (1)
        """
        obs = []

        # Ball 1 (6 dimensions)
        if self.balls:
            ball = self.balls[0]
            obs.extend([
                ball.x / CANVAS_WIDTH,
                ball.y / CANVAS_HEIGHT,
                ball.vx / 10.0,  # Normalize velocity
                ball.vy / 10.0,
                ball.speed / 10.0,
                1.0 if ball.is_penetrating else 0.0
            ])
        else:
            obs.extend([0.0] * 6)

        # Ball 2 (6 dimensions)
        if len(self.balls) > 1:
            ball = self.balls[1]
            obs.extend([
                ball.x / CANVAS_WIDTH,
                ball.y / CANVAS_HEIGHT,
                ball.vx / 10.0,
                ball.vy / 10.0,
                ball.speed / 10.0,
                1.0 if ball.is_penetrating else 0.0
            ])
        else:
            obs.extend([0.0] * 6)

        # Paddle (2 dimensions)
        obs.extend([
            self.paddle.x / CANVAS_WIDTH,
            self.paddle.velocity / PADDLE_SPEED
        ])

        # Time (1 dimension)
        obs.append(self.time_remaining / self.time_limit if self.time_limit > 0 else 0.0)

        # Block existence grid (96 dimensions)
        block_grid = [[0.0] * BLOCK_COLS for _ in range(BLOCK_ROWS)]
        hp_grid = [[0.0] * BLOCK_COLS for _ in range(BLOCK_ROWS)]

        for block in self.blocks:
            if block.is_destroyed:
                continue
            col = int((block.x - BLOCK_OFFSET_LEFT) / (BLOCK_WIDTH + BLOCK_PADDING))
            row = int((block.y - BLOCK_OFFSET_TOP) / (BLOCK_HEIGHT + BLOCK_PADDING))
            if 0 <= row < BLOCK_ROWS and 0 <= col < BLOCK_COLS:
                block_grid[row][col] = 1.0
                hp_grid[row][col] = block.hit_points / block.max_hit_points

        for row in block_grid:
            obs.extend(row)

        # Block HP grid (96 dimensions)
        for row in hp_grid:
            obs.extend(row)

        # Power-up state (4 dimensions)
        obs.append(self.speed_modifier)

        # Check if any ball is penetrating
        any_penetrating = any(b.is_penetrating for b in self.balls)
        obs.append(1.0 if any_penetrating else 0.0)

        obs.append(len(self.balls) / 5.0)  # Normalized ball count

        # Nearest falling power-up position (X and Y)
        nearest_powerup_x = 0.5  # Default to center
        nearest_powerup_y = 1.0  # Default to off-screen (bottom)
        for powerup in self.powerups:
            if powerup.is_active:
                normalized_y = powerup.y / CANVAS_HEIGHT
                if normalized_y < nearest_powerup_y:
                    nearest_powerup_y = normalized_y
                    nearest_powerup_x = powerup.x / CANVAS_WIDTH
        obs.append(nearest_powerup_x)
        obs.append(nearest_powerup_y)

        # Game state (3 dimensions)
        obs.append(self.lives / INITIAL_LIVES)
        obs.append(self.stage_number / MAX_STAGES)

        total_blocks = len(self.blocks)
        active_blocks = sum(1 for b in self.blocks if not b.is_destroyed)
        obs.append(active_blocks / total_blocks if total_blocks > 0 else 0.0)

        # Stage speed multiplier (1 dimension)
        obs.append(self.ball_speed_multiplier)

        return obs

    def reset(self, stage_number: Optional[int] = None):
        """Reset game to initial state."""
        if stage_number is not None:
            self.stage_number = stage_number

        stage_data = get_stage_data(self.stage_number)

        self.lives = INITIAL_LIVES
        self.score = 0
        self.combo = 0
        self.combo_timer = 0.0
        self.time_remaining = stage_data['time_limit']
        self.time_limit = stage_data['time_limit']
        self.ball_speed_multiplier = stage_data['ball_speed_multiplier']
        self.powerup_chance = stage_data['powerup_chance']

        self.paddle.reset()
        self.powerups = []
        self.is_game_over = False
        self.is_stage_clear = False
        self.is_time_up = False
        self.speed_modifier = 1.0
        self.speed_modifier_timer = 0.0

        self._load_stage(self.stage_number)
        self._spawn_ball()

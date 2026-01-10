"""Game constants ported from JavaScript."""

import math
from enum import IntEnum
from dataclasses import dataclass
from typing import Dict, Optional

# Canvas settings
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# Paddle settings
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 15
PADDLE_SPEED = 10
PADDLE_Y_OFFSET = 40

# Ball settings
BALL_RADIUS = 8
BALL_BASE_SPEED = 5

# Block settings
BLOCK_WIDTH = 58
BLOCK_HEIGHT = 20
BLOCK_PADDING = 4
BLOCK_OFFSET_TOP = 80
BLOCK_OFFSET_LEFT = 35
BLOCK_COLS = 12
BLOCK_ROWS = 8

# Game settings
INITIAL_LIVES = 3
MAX_STAGES = 10
STAGE_TIME_LIMIT = 120000  # 120 seconds in milliseconds
TIME_EXTEND_AMOUNT = 15000  # 15 seconds


class BlockType(IntEnum):
    """Block types matching JavaScript BLOCK_TYPE."""
    NONE = 0
    NORMAL_BLUE = 1
    NORMAL_GREEN = 2
    NORMAL_YELLOW = 3
    NORMAL_ORANGE = 4
    NORMAL_RED = 5
    DURABLE_1 = 10
    DURABLE_2 = 11


@dataclass
class BlockConfig:
    """Block configuration data."""
    color: str
    score: int
    hit_points: int = 1


BLOCK_CONFIG: Dict[int, BlockConfig] = {
    BlockType.NORMAL_BLUE: BlockConfig(color='#00ffff', score=100),
    BlockType.NORMAL_GREEN: BlockConfig(color='#00ff00', score=150),
    BlockType.NORMAL_YELLOW: BlockConfig(color='#ffff00', score=200),
    BlockType.NORMAL_ORANGE: BlockConfig(color='#ff8800', score=250),
    BlockType.NORMAL_RED: BlockConfig(color='#ff0044', score=300),
    BlockType.DURABLE_1: BlockConfig(color='#8888ff', score=400, hit_points=2),
    BlockType.DURABLE_2: BlockConfig(color='#ff88ff', score=600, hit_points=3),
}


class PowerUpType:
    """Power-up types."""
    MULTI_BALL = 'multi_ball'
    SPEED_UP = 'speed_up'
    SPEED_DOWN = 'speed_down'
    PENETRATE = 'penetrate'
    TIME_EXTEND = 'time_extend'


@dataclass
class PowerUpConfig:
    """Power-up configuration data."""
    color: str
    icon: str
    duration: int  # ms
    modifier: float = 1.0


POWERUP_CONFIG: Dict[str, PowerUpConfig] = {
    PowerUpType.MULTI_BALL: PowerUpConfig(color='#ff00ff', icon='M', duration=0),
    PowerUpType.SPEED_UP: PowerUpConfig(color='#ff4444', icon='>', duration=8000, modifier=1.3),
    PowerUpType.SPEED_DOWN: PowerUpConfig(color='#4444ff', icon='<', duration=8000, modifier=0.7),
    PowerUpType.PENETRATE: PowerUpConfig(color='#ffff00', icon='P', duration=5000),
    PowerUpType.TIME_EXTEND: PowerUpConfig(color='#00ff88', icon='T', duration=0),
}

# Power-up spawn weights
POWERUP_WEIGHTS: Dict[str, int] = {
    PowerUpType.MULTI_BALL: 25,
    PowerUpType.SPEED_DOWN: 20,
    PowerUpType.SPEED_UP: 15,
    PowerUpType.PENETRATE: 20,
    PowerUpType.TIME_EXTEND: 25,
}

# Combo multipliers (sorted by combo threshold descending)
COMBO_MULTIPLIERS = [
    (20, 3.0),
    (15, 2.5),
    (10, 2.0),
    (5, 1.5),
    (3, 1.2),
    (0, 1.0),
]


def get_combo_multiplier(combo: int) -> float:
    """Get score multiplier for given combo count."""
    for threshold, multiplier in COMBO_MULTIPLIERS:
        if combo >= threshold:
            return multiplier
    return 1.0


# Stage data - layouts and configurations
B = BlockType  # Shorthand

STAGE_DATA = [
    # Stage 1: Introduction - Simple rows (36 hits)
    {
        'layout': [
            [B.NORMAL_BLUE] * 12,
            [B.NORMAL_GREEN] * 12,
            [B.NORMAL_YELLOW] * 12,
        ],
        'ball_speed_multiplier': 1.0,
        'powerup_chance': 0.18,
        'time_limit': 90000,
    },

    # Stage 2: Pyramid (30 hits)
    {
        'layout': [
            [B.NONE]*5 + [B.NORMAL_RED]*2 + [B.NONE]*5,
            [B.NONE]*4 + [B.NORMAL_ORANGE]*4 + [B.NONE]*4,
            [B.NONE]*3 + [B.NORMAL_YELLOW]*6 + [B.NONE]*3,
            [B.NONE]*2 + [B.NORMAL_GREEN]*8 + [B.NONE]*2,
            [B.NONE]*1 + [B.NORMAL_BLUE]*10 + [B.NONE]*1,
        ],
        'ball_speed_multiplier': 1.0,
        'powerup_chance': 0.20,
        'time_limit': 90000,
    },

    # Stage 3: Durable blocks introduction (56 hits)
    {
        'layout': [
            [B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE],
            [B.NORMAL_GREEN] * 12,
            [B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW],
            [B.NORMAL_ORANGE] * 12,
        ],
        'ball_speed_multiplier': 1.05,
        'powerup_chance': 0.18,
        'time_limit': 120000,
    },

    # Stage 4: Checkerboard (36 hits)
    {
        'layout': [
            [B.NORMAL_RED, B.NONE] * 6,
            [B.NONE, B.NORMAL_ORANGE] * 6,
            [B.NORMAL_YELLOW, B.NONE] * 6,
            [B.NONE, B.NORMAL_GREEN] * 6,
            [B.NORMAL_BLUE, B.NONE] * 6,
            [B.NONE, B.NORMAL_BLUE] * 6,
        ],
        'ball_speed_multiplier': 1.1,
        'powerup_chance': 0.15,
        'time_limit': 90000,
    },

    # Stage 5: Center hole pattern (48 hits)
    {
        'layout': [
            [B.NORMAL_RED]*4 + [B.NONE]*4 + [B.NORMAL_RED]*4,
            [B.NORMAL_ORANGE]*3 + [B.NONE]*6 + [B.NORMAL_ORANGE]*3,
            [B.NORMAL_YELLOW]*2 + [B.NONE]*3 + [B.DURABLE_2]*2 + [B.NONE]*3 + [B.NORMAL_YELLOW]*2,
            [B.NORMAL_GREEN]*2 + [B.NONE]*3 + [B.DURABLE_2]*2 + [B.NONE]*3 + [B.NORMAL_GREEN]*2,
            [B.NORMAL_BLUE]*3 + [B.NONE]*6 + [B.NORMAL_BLUE]*3,
            [B.NORMAL_BLUE]*4 + [B.NONE]*4 + [B.NORMAL_BLUE]*4,
        ],
        'ball_speed_multiplier': 1.15,
        'powerup_chance': 0.2,
        'time_limit': 120000,
    },

    # Stage 6: Wave pattern (26 hits)
    {
        'layout': [
            [B.NORMAL_BLUE]*2 + [B.NONE]*8 + [B.NORMAL_BLUE]*2,
            [B.NONE] + [B.NORMAL_GREEN]*3 + [B.NONE]*4 + [B.NORMAL_GREEN]*3 + [B.NONE],
            [B.NONE]*3 + [B.NORMAL_YELLOW]*6 + [B.NONE]*3,
            [B.NONE] + [B.NORMAL_ORANGE]*3 + [B.NONE]*4 + [B.NORMAL_ORANGE]*3 + [B.NONE],
            [B.NORMAL_RED]*2 + [B.NONE]*8 + [B.NORMAL_RED]*2,
        ],
        'ball_speed_multiplier': 1.2,
        'powerup_chance': 0.18,
        'time_limit': 90000,
    },

    # Stage 7: Fortress with durable walls (96 hits)
    {
        'layout': [
            [B.NORMAL_BLUE] * 12,
            [B.DURABLE_2] * 12,
            [B.NORMAL_GREEN] * 12,
            [B.DURABLE_1] * 12,
            [B.NORMAL_RED] * 12,
        ],
        'ball_speed_multiplier': 1.25,
        'powerup_chance': 0.22,
        'time_limit': 180000,
    },

    # Stage 8: Diamond pattern (50 hits)
    {
        'layout': [
            [B.NONE]*5 + [B.DURABLE_2]*2 + [B.NONE]*5,
            [B.NONE]*4 + [B.DURABLE_1] + [B.NORMAL_RED]*2 + [B.DURABLE_1] + [B.NONE]*4,
            [B.NONE]*3 + [B.DURABLE_1] + [B.NORMAL_ORANGE]*4 + [B.DURABLE_1] + [B.NONE]*3,
            [B.NONE]*2 + [B.DURABLE_1] + [B.NORMAL_YELLOW]*6 + [B.DURABLE_1] + [B.NONE]*2,
            [B.NONE]*3 + [B.DURABLE_1] + [B.NORMAL_GREEN]*4 + [B.DURABLE_1] + [B.NONE]*3,
            [B.NONE]*4 + [B.DURABLE_1] + [B.NORMAL_BLUE]*2 + [B.DURABLE_1] + [B.NONE]*4,
            [B.NONE]*5 + [B.DURABLE_2]*2 + [B.NONE]*5,
        ],
        'ball_speed_multiplier': 1.3,
        'powerup_chance': 0.2,
        'time_limit': 150000,
    },

    # Stage 9: Maze-like (66 hits)
    {
        'layout': [
            [B.DURABLE_1]*4 + [B.NONE]*4 + [B.DURABLE_1]*4,
            [B.NONE]*3 + [B.NORMAL_YELLOW]*6 + [B.NONE]*3,
            [B.NORMAL_GREEN]*2 + [B.NONE]*3 + [B.DURABLE_2]*2 + [B.NONE]*3 + [B.NORMAL_GREEN]*2,
            [B.NORMAL_GREEN]*4 + [B.NONE]*4 + [B.NORMAL_GREEN]*4,
            [B.NONE]*3 + [B.NORMAL_BLUE]*6 + [B.NONE]*3,
            [B.DURABLE_1]*2 + [B.NONE]*2 + [B.DURABLE_1] + [B.NORMAL_RED]*2 + [B.DURABLE_1] + [B.NONE]*2 + [B.DURABLE_1]*2,
        ],
        'ball_speed_multiplier': 1.35,
        'powerup_chance': 0.22,
        'time_limit': 180000,
    },

    # Stage 10: Final Challenge (112 hits)
    {
        'layout': [
            [B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2],
            [B.NORMAL_ORANGE, B.DURABLE_2, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_1, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_1, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_2, B.NORMAL_ORANGE],
            [B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW],
            [B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1],
            [B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE],
            [B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.DURABLE_2, B.DURABLE_2, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2],
        ],
        'ball_speed_multiplier': 1.4,
        'powerup_chance': 0.25,
        'time_limit': 240000,
    },
]


def get_stage_data(stage_number: int) -> dict:
    """Get stage data by stage number (1-indexed)."""
    index = min(stage_number - 1, len(STAGE_DATA) - 1)
    return STAGE_DATA[index]


# RL-specific constants
FRAME_SKIP = 4  # Number of frames to repeat same action
MAX_BALLS = 32  # Maximum number of balls to observe
MAX_POWERUPS = 5  # Maximum number of powerups to observe
# Observation dimensions:
# - Balls: 32 * 6 = 192 (x, y, vx, vy, speed, is_penetrating)
# - Paddle: 2 (x, velocity)
# - Time: 1
# - Block existence: 96 (12x8 grid)
# - Block HP: 96 (12x8 grid)
# - Block type: 96 (12x8 grid, normalized type)
# - Powerups: 5 * 4 = 20 (x, y, type, active)
# - Game state: 3 (lives, stage, remaining_blocks_ratio)
# - Stage speed: 1
OBSERVATION_DIM = 507
ACTION_DIM = 3  # Left, Stay, Right

# Paddle movement per action
PADDLE_MOVE_AMOUNT = PADDLE_SPEED * FRAME_SKIP

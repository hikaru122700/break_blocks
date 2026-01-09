// Canvas settings
const CANVAS_WIDTH = 800;
const CANVAS_HEIGHT = 600;

// Paddle settings
const PADDLE_WIDTH = 100;
const PADDLE_HEIGHT = 15;
const PADDLE_SPEED = 10;
const PADDLE_Y_OFFSET = 40;

// Ball settings
const BALL_RADIUS = 8;
const BALL_BASE_SPEED = 5;

// Block settings
const BLOCK_WIDTH = 58;
const BLOCK_HEIGHT = 20;
const BLOCK_PADDING = 4;
const BLOCK_OFFSET_TOP = 80;
const BLOCK_OFFSET_LEFT = 35;
const BLOCK_COLS = 12;
const BLOCK_ROWS = 8;

// Game settings
const INITIAL_LIVES = 3;
const MAX_STAGES = 10;
const STAGE_TIME_LIMIT = 120000; // 120 seconds in milliseconds
const TIME_EXTEND_AMOUNT = 15000; // 15 seconds added per time power-up

// Block types
const BLOCK_TYPE = {
    NONE: 0,
    NORMAL_BLUE: 1,
    NORMAL_GREEN: 2,
    NORMAL_YELLOW: 3,
    NORMAL_ORANGE: 4,
    NORMAL_RED: 5,
    DURABLE_1: 10,
    DURABLE_2: 11
};

// Block colors and scores
const BLOCK_CONFIG = {
    [BLOCK_TYPE.NORMAL_BLUE]: { color: '#00ffff', score: 100 },
    [BLOCK_TYPE.NORMAL_GREEN]: { color: '#00ff00', score: 150 },
    [BLOCK_TYPE.NORMAL_YELLOW]: { color: '#ffff00', score: 200 },
    [BLOCK_TYPE.NORMAL_ORANGE]: { color: '#ff8800', score: 250 },
    [BLOCK_TYPE.NORMAL_RED]: { color: '#ff0044', score: 300 },
    [BLOCK_TYPE.DURABLE_1]: { color: '#8888ff', score: 400, hitPoints: 2 },
    [BLOCK_TYPE.DURABLE_2]: { color: '#ff88ff', score: 600, hitPoints: 3 }
};

// PowerUp types
const POWERUP_TYPE = {
    MULTI_BALL: 'multi_ball',
    SPEED_UP: 'speed_up',
    SPEED_DOWN: 'speed_down',
    PENETRATE: 'penetrate',
    TIME_EXTEND: 'time_extend'
};

// PowerUp configurations
const POWERUP_CONFIG = {
    [POWERUP_TYPE.MULTI_BALL]: {
        color: '#ff00ff',
        icon: 'M',
        duration: 0
    },
    [POWERUP_TYPE.SPEED_UP]: {
        color: '#ff4444',
        icon: '>',
        duration: 8000,
        modifier: 1.3
    },
    [POWERUP_TYPE.SPEED_DOWN]: {
        color: '#4444ff',
        icon: '<',
        duration: 8000,
        modifier: 0.7
    },
    [POWERUP_TYPE.PENETRATE]: {
        color: '#ffff00',
        icon: 'P',
        duration: 5000
    },
    [POWERUP_TYPE.TIME_EXTEND]: {
        color: '#00ff88',
        icon: 'T',
        duration: 0
    }
};

// PowerUp spawn weights
const POWERUP_WEIGHTS = {
    [POWERUP_TYPE.MULTI_BALL]: 25,
    [POWERUP_TYPE.SPEED_DOWN]: 20,
    [POWERUP_TYPE.SPEED_UP]: 15,
    [POWERUP_TYPE.PENETRATE]: 20,
    [POWERUP_TYPE.TIME_EXTEND]: 20
};

// Game states
const GAME_STATE = {
    LOADING: 'loading',
    MENU: 'menu',
    PLAYING: 'playing',
    PAUSED: 'paused',
    GAME_OVER: 'game_over',
    STAGE_CLEAR: 'stage_clear',
    ALL_CLEAR: 'all_clear',
    RANKING: 'ranking'
};

// Combo multipliers
const COMBO_MULTIPLIERS = [
    { combo: 20, multiplier: 3.0 },
    { combo: 15, multiplier: 2.5 },
    { combo: 10, multiplier: 2.0 },
    { combo: 5, multiplier: 1.5 },
    { combo: 3, multiplier: 1.2 },
    { combo: 0, multiplier: 1.0 }
];

// Colors for retro theme
const COLORS = {
    BACKGROUND: '#1a0a2e',
    PRIMARY: '#00ffff',
    SECONDARY: '#ff00ff',
    TEXT: '#ffffff',
    TEXT_DIM: '#888888',
    GRID: 'rgba(0, 255, 255, 0.1)'
};

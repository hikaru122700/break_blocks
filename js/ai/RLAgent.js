/**
 * Reinforcement Learning Agent for Break Blocks.
 * Uses ONNX model trained with PPO for decision making.
 */

// Constants for observation space
const MAX_BALLS = 32;
const MAX_POWERUPS = 5;
const OBSERVATION_DIM = 507;

class RLAgent {
    constructor(game) {
        this.game = game;
        this.onnx = new ONNXInference();
        this.isEnabled = false;
        this.isLoaded = false;

        // Frame skip (match training)
        this.frameSkip = 4;
        this.frameCount = 0;
        this.currentAction = 1; // Stay

        // Auto launch
        this.autoLaunchDelay = 500;
        this.autoLaunchTimer = 0;

        // Observation buffer (507 dimensions)
        this.observation = new Float32Array(OBSERVATION_DIM);

        // Performance tracking
        this.inferenceTime = 0;
    }

    /**
     * Initialize the RL agent by loading the ONNX model.
     * @returns {Promise<boolean>} True if loaded successfully
     */
    async initialize() {
        try {
            this.isLoaded = await this.onnx.load();
            if (this.isLoaded) {
                console.log('[RLAgent] Initialized successfully');
            }
            return this.isLoaded;
        } catch (error) {
            console.error('[RLAgent] Failed to initialize:', error);
            return false;
        }
    }

    enable() {
        this.isEnabled = true;
        this.autoLaunchTimer = this.autoLaunchDelay;
        this.frameCount = 0;
        this.currentAction = 1;
    }

    disable() {
        this.isEnabled = false;
        this.autoLaunchTimer = 0;
    }

    /**
     * Update the agent (called every frame).
     * @param {number} deltaTime - Time since last frame in ms
     */
    async update(deltaTime) {
        if (!this.isEnabled || !this.isLoaded) return;

        const paddle = this.game.paddle;
        const balls = this.game.balls;

        if (!paddle || balls.length === 0) return;

        // Auto-launch ball
        const targetBall = balls[0];
        if (!targetBall.isLaunched) {
            this.autoLaunchTimer -= deltaTime;
            if (this.autoLaunchTimer <= 0) {
                targetBall.launch(paddle);
                this.game.soundManager.playSFX('hit_paddle');
                this.autoLaunchTimer = this.autoLaunchDelay;
            }
            return;
        }

        // Frame skip: only get new action every N frames
        this.frameCount++;
        if (this.frameCount >= this.frameSkip) {
            this.frameCount = 0;

            // Build observation
            this._buildObservation();

            // Get action from model
            const startTime = performance.now();
            try {
                this.currentAction = await this.onnx.getAction(this.observation);
            } catch (error) {
                console.error('[RLAgent] Inference error:', error);
                this.currentAction = 1; // Default to stay
            }
            this.inferenceTime = performance.now() - startTime;
        }

        // Apply action
        this._applyAction(this.currentAction, deltaTime);
    }

    /**
     * Build observation vector from game state.
     * Matches the 507-dimensional observation space from training.
     */
    _buildObservation() {
        let idx = 0;

        // Balls (up to MAX_BALLS, 6 dimensions each = 192 total)
        for (let i = 0; i < MAX_BALLS; i++) {
            if (i < this.game.balls.length) {
                const ball = this.game.balls[i];
                this.observation[idx++] = ball.x / CANVAS_WIDTH;
                this.observation[idx++] = ball.y / CANVAS_HEIGHT;
                this.observation[idx++] = ball.velocity.x / 10.0;
                this.observation[idx++] = ball.velocity.y / 10.0;
                this.observation[idx++] = ball.speed / 10.0;
                this.observation[idx++] = ball.isPenetrating ? 1.0 : 0.0;
            } else {
                for (let j = 0; j < 6; j++) this.observation[idx++] = 0.0;
            }
        }

        // Paddle (2 dimensions)
        const paddle = this.game.paddle;
        this.observation[idx++] = paddle.x / CANVAS_WIDTH;
        this.observation[idx++] = (paddle.movingLeft ? -1 : (paddle.movingRight ? 1 : 0));

        // Time (1 dimension)
        const timeLimit = this.game.stageManager.getTimeLimit(this.game.currentStage);
        this.observation[idx++] = this.game.stageTimeRemaining / timeLimit;

        // Block grids (96 dimensions each = 288 total)
        const blockGrid = new Array(BLOCK_ROWS).fill(null).map(() => new Array(BLOCK_COLS).fill(0));
        const hpGrid = new Array(BLOCK_ROWS).fill(null).map(() => new Array(BLOCK_COLS).fill(0));
        const typeGrid = new Array(BLOCK_ROWS).fill(null).map(() => new Array(BLOCK_COLS).fill(0));

        for (const block of this.game.blocks) {
            if (block.isDestroyed) continue;
            const col = Math.floor((block.x - BLOCK_OFFSET_LEFT) / (BLOCK_WIDTH + BLOCK_PADDING));
            const row = Math.floor((block.y - BLOCK_OFFSET_TOP) / (BLOCK_HEIGHT + BLOCK_PADDING));
            if (row >= 0 && row < BLOCK_ROWS && col >= 0 && col < BLOCK_COLS) {
                blockGrid[row][col] = 1.0;
                const maxHP = block.maxHitPoints || 1;
                const currentHP = block.hitPoints || 1;
                hpGrid[row][col] = currentHP / maxHP;
                // Normalize block type (0-11 range -> 0-1)
                typeGrid[row][col] = (block.type || 0) / 11.0;
            }
        }

        // Flatten block grid
        for (let row = 0; row < BLOCK_ROWS; row++) {
            for (let col = 0; col < BLOCK_COLS; col++) {
                this.observation[idx++] = blockGrid[row][col];
            }
        }

        // Flatten HP grid
        for (let row = 0; row < BLOCK_ROWS; row++) {
            for (let col = 0; col < BLOCK_COLS; col++) {
                this.observation[idx++] = hpGrid[row][col];
            }
        }

        // Flatten type grid
        for (let row = 0; row < BLOCK_ROWS; row++) {
            for (let col = 0; col < BLOCK_COLS; col++) {
                this.observation[idx++] = typeGrid[row][col];
            }
        }

        // Power-ups (up to MAX_POWERUPS, 4 dimensions each = 20 total)
        // Encode powerup type: multi_ball=1, penetrate=0.8, time_extend=0.6, speed_down=0.4, speed_up=0.2
        const powerupTypeEncoding = {
            'multi_ball': 1.0,
            'penetrate': 0.8,
            'time_extend': 0.6,
            'speed_down': 0.4,
            'speed_up': 0.2,
        };

        const powerUpSystem = this.game.powerUpSystem;
        let activePowerups = [];
        if (powerUpSystem && powerUpSystem.activePowerUps) {
            activePowerups = powerUpSystem.activePowerUps
                .filter(p => p.isActive !== false)
                .sort((a, b) => b.y - a.y);  // Sort by Y (closest to paddle first)
        }

        for (let i = 0; i < MAX_POWERUPS; i++) {
            if (i < activePowerups.length) {
                const p = activePowerups[i];
                this.observation[idx++] = p.x / CANVAS_WIDTH;
                this.observation[idx++] = p.y / CANVAS_HEIGHT;
                this.observation[idx++] = powerupTypeEncoding[p.type] || 0.0;
                this.observation[idx++] = 1.0;  // is_active
            } else {
                this.observation[idx++] = 0.5;  // x: center
                this.observation[idx++] = 1.0;  // y: off-screen
                this.observation[idx++] = 0.0;  // type: none
                this.observation[idx++] = 0.0;  // inactive
            }
        }

        // Game state (3 dimensions)
        this.observation[idx++] = this.game.lives / INITIAL_LIVES;
        this.observation[idx++] = this.game.currentStage / MAX_STAGES;

        const totalBlocks = this.game.blocks.length;
        const activeBlocks = this.game.blocks.filter(b => !b.isDestroyed).length;
        this.observation[idx++] = totalBlocks > 0 ? activeBlocks / totalBlocks : 0.0;

        // Stage speed multiplier (1 dimension)
        const stageConfig = this.game.stageManager.getStageConfig(this.game.currentStage);
        this.observation[idx++] = stageConfig.ballSpeedMultiplier || 1.0;
    }

    /**
     * Apply action to paddle.
     * @param {number} action - 0=left, 1=stay, 2=right
     * @param {number} deltaTime - Time since last frame
     */
    _applyAction(action, deltaTime) {
        const paddle = this.game.paddle;
        const dt = deltaTime / 16.67;

        switch (action) {
            case 0: // Left
                paddle.x -= PADDLE_SPEED * dt;
                break;
            case 2: // Right
                paddle.x += PADDLE_SPEED * dt;
                break;
            // case 1: Stay - do nothing
        }

        // Clamp to bounds
        paddle.x = Utils.clamp(paddle.x, 0, CANVAS_WIDTH - paddle.width);
        paddle.targetX = paddle.x;
    }

    /**
     * Check if agent is ready.
     * @returns {boolean}
     */
    isReady() {
        return this.isLoaded && this.onnx.isReady();
    }

    /**
     * Get performance stats.
     * @returns {Object}
     */
    getStats() {
        return {
            inferenceTime: this.inferenceTime,
            currentAction: ['Left', 'Stay', 'Right'][this.currentAction],
            isLoaded: this.isLoaded
        };
    }
}

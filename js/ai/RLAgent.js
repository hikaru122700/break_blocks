/**
 * Reinforcement Learning Agent for Break Blocks.
 * Uses ONNX model trained with PPO for decision making.
 */
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

        // Observation buffer (216 dimensions - includes powerup X position)
        this.observation = new Float32Array(216);

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
     * Matches the observation space from training.
     */
    _buildObservation() {
        let idx = 0;

        // Ball 1 (6 dimensions)
        if (this.game.balls.length > 0) {
            const ball = this.game.balls[0];
            this.observation[idx++] = ball.x / CANVAS_WIDTH;
            this.observation[idx++] = ball.y / CANVAS_HEIGHT;
            this.observation[idx++] = ball.velocity.x / 10.0;
            this.observation[idx++] = ball.velocity.y / 10.0;
            this.observation[idx++] = ball.speed / 10.0;
            this.observation[idx++] = ball.isPenetrating ? 1.0 : 0.0;
        } else {
            for (let i = 0; i < 6; i++) this.observation[idx++] = 0.0;
        }

        // Ball 2 (6 dimensions)
        if (this.game.balls.length > 1) {
            const ball = this.game.balls[1];
            this.observation[idx++] = ball.x / CANVAS_WIDTH;
            this.observation[idx++] = ball.y / CANVAS_HEIGHT;
            this.observation[idx++] = ball.velocity.x / 10.0;
            this.observation[idx++] = ball.velocity.y / 10.0;
            this.observation[idx++] = ball.speed / 10.0;
            this.observation[idx++] = ball.isPenetrating ? 1.0 : 0.0;
        } else {
            for (let i = 0; i < 6; i++) this.observation[idx++] = 0.0;
        }

        // Paddle (2 dimensions)
        const paddle = this.game.paddle;
        this.observation[idx++] = paddle.x / CANVAS_WIDTH;
        this.observation[idx++] = (paddle.movingLeft ? -1 : (paddle.movingRight ? 1 : 0));

        // Time (1 dimension)
        const timeLimit = this.game.stageManager.getTimeLimit(this.game.currentStage);
        this.observation[idx++] = this.game.stageTimeRemaining / timeLimit;

        // Block existence grid (96 dimensions - 12x8)
        const blockGrid = new Array(BLOCK_ROWS).fill(null).map(() => new Array(BLOCK_COLS).fill(0));
        const hpGrid = new Array(BLOCK_ROWS).fill(null).map(() => new Array(BLOCK_COLS).fill(0));

        for (const block of this.game.blocks) {
            if (block.isDestroyed) continue;
            const col = Math.floor((block.x - BLOCK_OFFSET_LEFT) / (BLOCK_WIDTH + BLOCK_PADDING));
            const row = Math.floor((block.y - BLOCK_OFFSET_TOP) / (BLOCK_HEIGHT + BLOCK_PADDING));
            if (row >= 0 && row < BLOCK_ROWS && col >= 0 && col < BLOCK_COLS) {
                blockGrid[row][col] = 1.0;
                const maxHP = block.maxHitPoints || 1;
                const currentHP = block.hitPoints || 1;
                hpGrid[row][col] = currentHP / maxHP;
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

        // Power-up state (4 dimensions)
        const powerUpSystem = this.game.powerUpSystem;
        this.observation[idx++] = powerUpSystem ? powerUpSystem.speedModifier : 1.0;

        // Any ball penetrating
        const anyPenetrating = this.game.balls.some(b => b.isPenetrating);
        this.observation[idx++] = anyPenetrating ? 1.0 : 0.0;

        // Ball count
        this.observation[idx++] = this.game.balls.length / 5.0;

        // Nearest falling power-up position (X and Y)
        let nearestX = 0.5;  // Default to center
        let nearestY = 1.0;  // Default to off-screen (bottom)
        if (powerUpSystem && powerUpSystem.activePowerUps) {
            for (const pu of powerUpSystem.activePowerUps) {
                const normalizedY = pu.y / CANVAS_HEIGHT;
                if (normalizedY < nearestY) {
                    nearestY = normalizedY;
                    nearestX = pu.x / CANVAS_WIDTH;
                }
            }
        }
        this.observation[idx++] = nearestX;
        this.observation[idx++] = nearestY;

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

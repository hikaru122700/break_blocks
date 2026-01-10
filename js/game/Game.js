class Game {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.state = new GameState();

        // Managers
        this.inputManager = new InputManager(canvas);
        this.soundManager = new SoundManager();
        this.storageManager = new StorageManager();
        this.stageManager = new StageManager();

        // Systems
        this.collisionSystem = new CollisionSystem();
        this.scoreSystem = new ScoreSystem();
        this.powerUpSystem = new PowerUpSystem(this);
        this.particleSystem = new ParticleSystem();

        // Game entities
        this.paddle = null;
        this.balls = [];
        this.blocks = [];

        // Game state
        this.lives = INITIAL_LIVES;
        this.currentStage = 1;

        // UI screens
        this.menuScreen = new MenuScreen(this);
        this.hud = new HUD(this);
        this.pauseScreen = new PauseScreen(this);
        this.rankingScreen = new RankingScreen(this);

        // Game loop
        this.lastTime = 0;
        this.deltaTime = 0;
        this.isRunning = false;

        // Timers
        this.stageClearTimer = 0;
        this.gameOverTimer = 0;
        this.stageTimeRemaining = STAGE_TIME_LIMIT;

        // AI Mode
        this.aiController = new AIController(this);
        this.isAIMode = false;
        this.timeScale = 1; // 1x, 2x, 3x, 4x, 5x
    }

    async init() {
        // Load sounds
        await this.soundManager.loadSounds();

        // Load saved data
        this.storageManager.loadSettings();

        // Initialize game
        this.state.setState(GAME_STATE.MENU);

        // Hide loading screen
        const loadingScreen = document.getElementById('loadingScreen');
        if (loadingScreen) {
            loadingScreen.classList.add('hidden');
        }

        // Start game loop
        this.startGameLoop();
    }

    startGameLoop() {
        this.isRunning = true;
        this.lastTime = performance.now();
        requestAnimationFrame((time) => this.gameLoop(time));
    }

    gameLoop(currentTime) {
        if (!this.isRunning) return;

        this.deltaTime = currentTime - this.lastTime;
        this.lastTime = currentTime;

        // Cap delta time to prevent large jumps
        this.deltaTime = Math.min(this.deltaTime, 50);

        this.update(this.deltaTime);
        this.render();

        requestAnimationFrame((time) => this.gameLoop(time));
    }

    update(deltaTime) {
        switch (this.state.getState()) {
            case GAME_STATE.MENU:
                this.updateMenu(deltaTime);
                break;

            case GAME_STATE.PLAYING:
                this.updateGameplay(deltaTime);
                break;

            case GAME_STATE.PAUSED:
                this.updatePaused(deltaTime);
                break;

            case GAME_STATE.STAGE_CLEAR:
                this.updateStageClear(deltaTime);
                break;

            case GAME_STATE.GAME_OVER:
                this.updateGameOver(deltaTime);
                break;

            case GAME_STATE.ALL_CLEAR:
                this.updateAllClear(deltaTime);
                break;

            case GAME_STATE.RANKING:
                this.updateRanking(deltaTime);
                break;
        }

        this.inputManager.clearPressedKeys();
    }

    updateMenu(deltaTime) {
        this.menuScreen.update(deltaTime);

        if (this.inputManager.isConfirmPressed()) {
            this.startGame();
        }

        // Start AI Mode
        if (this.inputManager.isKeyPressed('KeyA')) {
            this.startAIMode();
        }

        // Show ranking screen
        if (this.inputManager.isKeyPressed('KeyR')) {
            this.rankingScreen.show();
            this.state.setState(GAME_STATE.RANKING);
        }
    }

    updateGameplay(deltaTime) {
        // Handle pause
        if (this.inputManager.isPausePressed()) {
            this.pause();
            return;
        }

        // Time scale controls (1-5 keys) - works in both normal and AI mode
        if (this.inputManager.isKeyPressed('Digit1')) this.timeScale = 1;
        if (this.inputManager.isKeyPressed('Digit2')) this.timeScale = 2;
        if (this.inputManager.isKeyPressed('Digit3')) this.timeScale = 3;
        if (this.inputManager.isKeyPressed('Digit4')) this.timeScale = 4;
        if (this.inputManager.isKeyPressed('Digit5')) this.timeScale = 5;

        // Apply time scale
        const scaledDeltaTime = deltaTime * this.timeScale;

        if (this.isAIMode) {
            // AI controls the paddle
            this.aiController.update(scaledDeltaTime);
        } else {
            // Player controls the paddle
            this.paddle.moveLeft(this.inputManager.isLeftDown());
            this.paddle.moveRight(this.inputManager.isRightDown());
            this.paddle.setTargetX(this.inputManager.mouseX);
        }

        this.paddle.update(scaledDeltaTime);

        // Launch ball (player mode only - AI auto-launches)
        if (!this.isAIMode && !this.balls[0].isLaunched && this.inputManager.isLaunchPressed()) {
            this.balls[0].launch(this.paddle);
            this.soundManager.playSFX('hit_paddle');
        }

        // Update balls
        for (const ball of this.balls) {
            ball.update(scaledDeltaTime, this.paddle);

            if (ball.isLaunched) {
                // Wall collision
                const wallResult = this.collisionSystem.checkBallWall(ball, CANVAS_WIDTH, CANVAS_HEIGHT);
                if (wallResult.hit) {
                    ball.reflect(wallResult.normal);
                    this.soundManager.playSFX('hit_wall');
                }

                // Check if ball fell
                if (ball.y > CANVAS_HEIGHT + ball.radius) {
                    ball.isActive = false;
                }

                // Paddle collision
                if (this.collisionSystem.checkBallPaddle(ball, this.paddle)) {
                    const normalizedPos = this.paddle.getCollisionNormal(ball.x);
                    ball.reflectFromPaddle(normalizedPos);
                    ball.y = this.paddle.y - ball.radius;
                    this.soundManager.playSFX('hit_paddle');
                    this.scoreSystem.resetCombo();
                }

                // Block collision
                for (const block of this.blocks) {
                    if (block.isDestroyed) continue;

                    if (this.collisionSystem.checkBallBlock(ball, block)) {
                        const result = block.hit();
                        this.scoreSystem.addScore(result.score, block.x + block.width / 2, block.y);
                        this.soundManager.playSFX('hit_block');

                        // Reflect ball (unless penetrating)
                        if (!ball.isPenetrating) {
                            const normal = this.collisionSystem.getBlockCollisionNormal(ball, block);
                            ball.reflect(normal);
                        }

                        // Block destroyed
                        if (result.destroyed) {
                            this.particleSystem.emit(
                                block.x + block.width / 2,
                                block.y + block.height / 2,
                                block.color,
                                15
                            );
                            this.powerUpSystem.trySpawnPowerUp(
                                block.x + block.width / 2,
                                block.y + block.height / 2
                            );
                        }
                    }
                }
            }
        }

        // Remove inactive balls
        this.balls = this.balls.filter(ball => ball.isActive);

        // Check if all balls lost
        if (this.balls.length === 0) {
            this.loseLife();
        }

        // Update power-ups
        this.powerUpSystem.update(scaledDeltaTime);

        // Update particles
        this.particleSystem.update(scaledDeltaTime);

        // Update score system
        this.scoreSystem.update(scaledDeltaTime);

        // Update stage timer (only when ball is launched)
        if (this.balls.length > 0 && this.balls[0].isLaunched) {
            this.stageTimeRemaining -= scaledDeltaTime;
            if (this.stageTimeRemaining <= 0) {
                this.stageTimeRemaining = 0;
                this.timeUp();
                return;
            }
        }

        // Check stage clear
        const remainingBlocks = this.blocks.filter(b => !b.isDestroyed).length;
        if (remainingBlocks === 0) {
            this.stageClear();
        }
    }

    updatePaused(deltaTime) {
        this.pauseScreen.update(deltaTime);

        if (this.inputManager.isPausePressed()) {
            this.resume();
        }

        // Return to menu
        if (this.inputManager.isKeyPressed('KeyM')) {
            this.returnToMenu();
        }
    }

    updateStageClear(deltaTime) {
        this.stageClearTimer -= deltaTime;
        this.particleSystem.update(deltaTime);

        if (this.stageClearTimer <= 0 || this.inputManager.isConfirmPressed()) {
            if (this.currentStage >= MAX_STAGES) {
                this.allClear();
            } else {
                this.nextStage();
            }
        }
    }

    updateGameOver(deltaTime) {
        this.gameOverTimer -= deltaTime;
        this.particleSystem.update(deltaTime);

        if (this.gameOverTimer <= 0 || this.inputManager.isConfirmPressed()) {
            if (this.storageManager.isHighScore(this.scoreSystem.score)) {
                this.rankingScreen.showNameEntry(this.scoreSystem.score, this.currentStage);
                this.state.setState(GAME_STATE.RANKING);
            } else {
                this.state.setState(GAME_STATE.MENU);
            }
        }
    }

    updateAllClear(deltaTime) {
        this.particleSystem.update(deltaTime);

        if (this.inputManager.isConfirmPressed()) {
            if (this.storageManager.isHighScore(this.scoreSystem.score)) {
                this.rankingScreen.showNameEntry(this.scoreSystem.score, this.currentStage);
                this.state.setState(GAME_STATE.RANKING);
            } else {
                this.state.setState(GAME_STATE.MENU);
            }
        }
    }

    updateRanking(deltaTime) {
        this.rankingScreen.update(deltaTime);
    }

    render() {
        // Clear canvas
        this.ctx.fillStyle = COLORS.BACKGROUND;
        this.ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Draw grid
        Utils.drawGrid(this.ctx, CANVAS_WIDTH, CANVAS_HEIGHT);

        switch (this.state.getState()) {
            case GAME_STATE.MENU:
                this.menuScreen.draw(this.ctx);
                break;

            case GAME_STATE.PLAYING:
            case GAME_STATE.PAUSED:
            case GAME_STATE.STAGE_CLEAR:
            case GAME_STATE.GAME_OVER:
            case GAME_STATE.ALL_CLEAR:
                this.renderGameplay();
                if (this.state.isPaused()) {
                    this.pauseScreen.draw(this.ctx);
                } else if (this.state.is(GAME_STATE.STAGE_CLEAR)) {
                    this.drawStageClear();
                } else if (this.state.isGameOver()) {
                    this.drawGameOver();
                } else if (this.state.is(GAME_STATE.ALL_CLEAR)) {
                    this.drawAllClear();
                }
                break;

            case GAME_STATE.RANKING:
                this.rankingScreen.draw(this.ctx);
                break;
        }
    }

    renderGameplay() {
        // Draw blocks
        for (const block of this.blocks) {
            if (!block.isDestroyed) {
                block.draw(this.ctx);
            }
        }

        // Draw paddle
        this.paddle.draw(this.ctx);

        // Draw balls
        for (const ball of this.balls) {
            ball.draw(this.ctx);
        }

        // Draw power-ups
        this.powerUpSystem.draw(this.ctx);

        // Draw particles
        this.particleSystem.draw(this.ctx);

        // Draw HUD
        this.hud.draw(this.ctx);

        // Draw score popups
        this.scoreSystem.drawPopups(this.ctx);
    }

    drawStageClear() {
        this.ctx.save();
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        this.ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        Utils.drawPixelText(this.ctx, 'STAGE CLEAR!', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 - 30, 32, COLORS.PRIMARY);
        Utils.drawPixelText(this.ctx, `STAGE ${this.currentStage} COMPLETED`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 20, 16, COLORS.TEXT);
        Utils.drawPixelText(this.ctx, 'Press SPACE to continue', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 80, 12, COLORS.TEXT_DIM);

        this.ctx.restore();
    }

    drawGameOver() {
        this.ctx.save();
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        Utils.drawPixelText(this.ctx, 'GAME OVER', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 - 50, 40, '#ff0044');
        Utils.drawPixelText(this.ctx, `FINAL SCORE: ${Utils.formatNumber(this.scoreSystem.score)}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 10, 16, COLORS.TEXT);
        Utils.drawPixelText(this.ctx, `STAGE: ${this.currentStage}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 40, 14, COLORS.TEXT_DIM);
        Utils.drawPixelText(this.ctx, 'Press SPACE to continue', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 100, 12, COLORS.TEXT_DIM);

        this.ctx.restore();
    }

    drawAllClear() {
        this.ctx.save();
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        this.ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        Utils.drawPixelText(this.ctx, 'ALL CLEAR!', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 - 60, 40, COLORS.SECONDARY);
        Utils.drawPixelText(this.ctx, 'CONGRATULATIONS!', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, 20, COLORS.PRIMARY);
        Utils.drawPixelText(this.ctx, `FINAL SCORE: ${Utils.formatNumber(this.scoreSystem.score)}`, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 50, 16, COLORS.TEXT);
        Utils.drawPixelText(this.ctx, 'Press SPACE to continue', CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2 + 110, 12, COLORS.TEXT_DIM);

        this.ctx.restore();
    }

    // Game flow methods
    startGame() {
        this.isAIMode = false;
        this.timeScale = 1;
        this.aiController.disable();
        this.currentStage = 1;
        this.lives = INITIAL_LIVES;
        this.scoreSystem.reset();
        this.loadStage(this.currentStage);
        this.state.setState(GAME_STATE.PLAYING);
        this.soundManager.playBGM('game');
    }

    startAIMode() {
        this.isAIMode = true;
        this.timeScale = 1;
        this.aiController.enable();
        this.currentStage = 1;
        this.lives = INITIAL_LIVES;
        this.scoreSystem.reset();
        this.loadStage(this.currentStage);
        this.state.setState(GAME_STATE.PLAYING);
        this.soundManager.playBGM('game');
    }

    loadStage(stageNumber) {
        this.currentStage = stageNumber;
        this.blocks = this.stageManager.loadStage(stageNumber);
        this.paddle = new Paddle();
        this.balls = [new Ball(0, 0)];
        this.balls[0].reset(this.paddle);
        this.powerUpSystem.reset();
        this.stageTimeRemaining = this.stageManager.getTimeLimit(stageNumber);
    }

    loseLife() {
        this.lives--;
        this.soundManager.playSFX('lose_ball');

        if (this.lives <= 0) {
            this.gameOver();
        } else {
            // Respawn ball
            this.balls = [new Ball(0, 0)];
            this.balls[0].reset(this.paddle);
            this.powerUpSystem.clearActiveEffects();
        }
    }

    stageClear() {
        this.soundManager.playSFX('clear');
        this.stageClearTimer = 3000;
        this.state.setState(GAME_STATE.STAGE_CLEAR);

        // Emit celebration particles
        for (let i = 0; i < 5; i++) {
            setTimeout(() => {
                this.particleSystem.emit(
                    Utils.random(100, CANVAS_WIDTH - 100),
                    Utils.random(100, 300),
                    COLORS.PRIMARY,
                    20
                );
            }, i * 200);
        }
    }

    nextStage() {
        this.currentStage++;
        this.loadStage(this.currentStage);
        this.state.setState(GAME_STATE.PLAYING);
    }

    gameOver() {
        this.soundManager.playSFX('game_over');
        this.soundManager.stopBGM();
        this.gameOverTimer = 3000;
        this.state.setState(GAME_STATE.GAME_OVER);
    }

    timeUp() {
        // Time ran out - game over
        this.soundManager.playSFX('game_over');
        this.soundManager.stopBGM();
        this.gameOverTimer = 3000;
        this.state.setState(GAME_STATE.GAME_OVER);
    }

    addTime(amount) {
        this.stageTimeRemaining += amount;
        // Show time added effect
        this.scoreSystem.addPopup(
            CANVAS_WIDTH / 2,
            CANVAS_HEIGHT / 2,
            `+${Math.floor(amount / 1000)}s`,
            '#00ff88'
        );
    }

    allClear() {
        this.soundManager.playBGM('clear');
        this.state.setState(GAME_STATE.ALL_CLEAR);

        // Big celebration
        for (let i = 0; i < 10; i++) {
            setTimeout(() => {
                this.particleSystem.emit(
                    Utils.random(100, CANVAS_WIDTH - 100),
                    Utils.random(100, 400),
                    i % 2 === 0 ? COLORS.PRIMARY : COLORS.SECONDARY,
                    30
                );
            }, i * 150);
        }
    }

    pause() {
        this.state.setState(GAME_STATE.PAUSED);
        this.soundManager.pauseBGM();
    }

    resume() {
        this.state.setState(GAME_STATE.PLAYING);
        this.soundManager.resumeBGM();
    }

    returnToMenu() {
        this.soundManager.stopBGM();
        this.isAIMode = false;
        this.timeScale = 1;
        this.aiController.disable();
        this.state.setState(GAME_STATE.MENU);
    }
}

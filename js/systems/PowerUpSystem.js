class PowerUpSystem {
    constructor(game) {
        this.game = game;
        this.fallingItems = [];
        this.activeEffects = new Map();
    }

    trySpawnPowerUp(x, y) {
        // Get spawn chance from stage config
        const chance = this.game.stageManager.getPowerUpChance(this.game.currentStage);

        if (Math.random() < chance) {
            const type = this.selectRandomType();
            const powerUp = new PowerUp(x, y, type);
            this.fallingItems.push(powerUp);
        }
    }

    selectRandomType() {
        return Utils.weightedRandom(POWERUP_WEIGHTS);
    }

    update(deltaTime) {
        // Update falling items
        for (const item of this.fallingItems) {
            item.update(deltaTime);

            // Check collision with paddle
            if (this.game.collisionSystem.checkPowerUpPaddle(item, this.game.paddle)) {
                this.applyEffect(item.type);
                item.isActive = false;
                this.game.soundManager.playSFX('powerup');
            }
        }

        // Remove inactive items
        this.fallingItems = this.fallingItems.filter(item => item.isActive);

        // Update active effect timers
        for (const [type, effect] of this.activeEffects) {
            effect.timer -= deltaTime;
            if (effect.timer <= 0) {
                this.removeEffect(type);
            }
        }
    }

    applyEffect(type) {
        switch (type) {
            case POWERUP_TYPE.MULTI_BALL:
                this.applyMultiBall();
                break;

            case POWERUP_TYPE.SPEED_UP:
                this.applySpeedChange(POWERUP_CONFIG[type].modifier);
                this.activeEffects.set(type, {
                    timer: POWERUP_CONFIG[type].duration,
                    modifier: POWERUP_CONFIG[type].modifier
                });
                break;

            case POWERUP_TYPE.SPEED_DOWN:
                this.applySpeedChange(POWERUP_CONFIG[type].modifier);
                this.activeEffects.set(type, {
                    timer: POWERUP_CONFIG[type].duration,
                    modifier: POWERUP_CONFIG[type].modifier
                });
                break;

            case POWERUP_TYPE.PENETRATE:
                this.applyPenetrate();
                this.activeEffects.set(type, {
                    timer: POWERUP_CONFIG[type].duration
                });
                break;

            case POWERUP_TYPE.TIME_EXTEND:
                this.game.addTime(TIME_EXTEND_AMOUNT);
                break;
        }
    }

    applyMultiBall() {
        const newBalls = [];

        for (const ball of this.game.balls) {
            if (!ball.isLaunched) continue;

            // Create two new balls at angles
            const angles = [Math.PI / 6, -Math.PI / 6]; // 30 degrees spread

            for (const angleOffset of angles) {
                const newBall = ball.clone();
                const currentAngle = Math.atan2(ball.velocity.y, ball.velocity.x);
                const newAngle = currentAngle + angleOffset;

                newBall.velocity.x = Math.cos(newAngle) * ball.speed;
                newBall.velocity.y = Math.sin(newAngle) * ball.speed;

                newBalls.push(newBall);
            }
        }

        this.game.balls.push(...newBalls);
    }

    applySpeedChange(modifier) {
        // Remove conflicting speed effects
        if (modifier > 1 && this.activeEffects.has(POWERUP_TYPE.SPEED_DOWN)) {
            this.removeEffect(POWERUP_TYPE.SPEED_DOWN);
        } else if (modifier < 1 && this.activeEffects.has(POWERUP_TYPE.SPEED_UP)) {
            this.removeEffect(POWERUP_TYPE.SPEED_UP);
        }

        // Apply new speed to all balls
        const newSpeed = BALL_BASE_SPEED * modifier;
        for (const ball of this.game.balls) {
            ball.setSpeed(newSpeed);
        }
    }

    applyPenetrate() {
        const duration = POWERUP_CONFIG[POWERUP_TYPE.PENETRATE].duration;
        for (const ball of this.game.balls) {
            ball.setPenetrating(duration);
        }
    }

    removeEffect(type) {
        switch (type) {
            case POWERUP_TYPE.SPEED_UP:
            case POWERUP_TYPE.SPEED_DOWN:
                // Reset speed to base
                for (const ball of this.game.balls) {
                    ball.setSpeed(BALL_BASE_SPEED);
                }
                break;

            case POWERUP_TYPE.PENETRATE:
                for (const ball of this.game.balls) {
                    ball.isPenetrating = false;
                    ball.color = COLORS.PRIMARY;
                }
                break;
        }

        this.activeEffects.delete(type);
    }

    clearActiveEffects() {
        for (const type of this.activeEffects.keys()) {
            this.removeEffect(type);
        }
        this.activeEffects.clear();
    }

    reset() {
        this.fallingItems = [];
        this.clearActiveEffects();
    }

    draw(ctx) {
        for (const item of this.fallingItems) {
            item.draw(ctx);
        }
    }

    // Get active effects info for HUD
    getActiveEffectsInfo() {
        const info = [];
        for (const [type, effect] of this.activeEffects) {
            info.push({
                type,
                remainingTime: effect.timer,
                icon: POWERUP_CONFIG[type].icon,
                color: POWERUP_CONFIG[type].color
            });
        }
        return info;
    }
}

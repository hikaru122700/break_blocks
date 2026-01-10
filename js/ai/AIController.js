class AIController {
    constructor(game) {
        this.game = game;
        this.isEnabled = false;
        this.autoLaunchDelay = 300; // ms before auto-launching ball (faster)
        this.autoLaunchTimer = 0;

        // Strategic aiming - more aggressive
        this.aimOffset = 0; // -1 to 1: where to hit the ball on paddle
        this.aimUpdateTimer = 0;
        this.aimUpdateInterval = 300; // Update aim more frequently

        // Target block tracking
        this.targetBlockX = CANVAS_WIDTH / 2;
    }

    enable() {
        this.isEnabled = true;
        this.autoLaunchTimer = this.autoLaunchDelay;
        this.aimOffset = 0;
    }

    disable() {
        this.isEnabled = false;
        this.autoLaunchTimer = 0;
    }

    update(deltaTime) {
        if (!this.isEnabled) return;

        const paddle = this.game.paddle;
        const balls = this.game.balls;

        if (!paddle || balls.length === 0) return;

        // Update strategic aim periodically
        this.aimUpdateTimer -= deltaTime;
        if (this.aimUpdateTimer <= 0) {
            this.updateStrategicAim();
            this.aimUpdateTimer = this.aimUpdateInterval;
        }

        // Find the most dangerous ball (closest to paddle)
        const targetBall = this.findTargetBall(balls);

        if (targetBall) {
            if (targetBall.isLaunched) {
                // Track the ball with strategic offset
                this.trackBallWithStrategy(targetBall, paddle);
            } else {
                // Ball not launched yet - auto launch after delay
                this.autoLaunchTimer -= deltaTime;
                if (this.autoLaunchTimer <= 0) {
                    targetBall.launch(paddle);
                    this.game.soundManager.playSFX('hit_paddle');
                    this.autoLaunchTimer = this.autoLaunchDelay;
                }
            }
        }
    }

    findTargetBall(balls) {
        if (balls.length === 0) return null;
        if (balls.length === 1) return balls[0];

        // Find the ball with the highest Y (closest to bottom/paddle)
        // and moving downward (most dangerous)
        let targetBall = null;
        let maxDanger = -Infinity;

        for (const ball of balls) {
            if (!ball.isActive) continue;

            // Calculate danger score:
            // - Higher Y = more dangerous
            // - Moving down (positive vy) = more dangerous
            let danger = ball.y;

            // If ball is moving down, it's more dangerous
            if (ball.velocity.y > 0) {
                danger += 200; // Bonus for downward movement
            }

            if (danger > maxDanger) {
                maxDanger = danger;
                targetBall = ball;
            }
        }

        return targetBall || balls[0];
    }

    updateStrategicAim() {
        const blocks = this.game.blocks.filter(b => !b.isDestroyed);
        if (blocks.length === 0) {
            this.aimOffset = 0;
            this.targetBlockX = CANVAS_WIDTH / 2;
            return;
        }

        // Find the best target block cluster
        // Prioritize: lowest blocks (easiest to hit), then clusters
        const centerX = CANVAS_WIDTH / 2;

        // Group blocks by column region (left, center-left, center-right, right)
        let leftBlocks = [];
        let rightBlocks = [];

        for (const block of blocks) {
            const blockCenterX = block.x + block.width / 2;
            if (blockCenterX < centerX) {
                leftBlocks.push(block);
            } else {
                rightBlocks.push(block);
            }
        }

        // Calculate weighted center of blocks on each side
        // Lower blocks get higher weight (easier to reach)
        const calcWeightedCenter = (blockList) => {
            if (blockList.length === 0) return null;
            let totalWeight = 0;
            let weightedX = 0;
            for (const block of blockList) {
                // Weight by Y position (lower = higher weight) and HP
                const yWeight = (block.y / CANVAS_HEIGHT) * 2 + 1;
                const hpWeight = (block.hitPoints || 1);
                const weight = yWeight * hpWeight;
                weightedX += (block.x + block.width / 2) * weight;
                totalWeight += weight;
            }
            return totalWeight > 0 ? weightedX / totalWeight : null;
        };

        const leftCenter = calcWeightedCenter(leftBlocks);
        const rightCenter = calcWeightedCenter(rightBlocks);

        // Decide which side to target based on block count and position
        let targetSide = 0; // -1 = left, 0 = center, 1 = right

        if (leftBlocks.length > 0 && rightBlocks.length > 0) {
            // Both sides have blocks - target the side with more blocks
            if (leftBlocks.length > rightBlocks.length * 1.2) {
                targetSide = -1;
                this.targetBlockX = leftCenter;
            } else if (rightBlocks.length > leftBlocks.length * 1.2) {
                targetSide = 1;
                this.targetBlockX = rightCenter;
            } else {
                // Similar count - alternate or pick randomly
                targetSide = Math.random() > 0.5 ? -1 : 1;
                this.targetBlockX = targetSide < 0 ? leftCenter : rightCenter;
            }
        } else if (leftBlocks.length > 0) {
            targetSide = -1;
            this.targetBlockX = leftCenter;
        } else if (rightBlocks.length > 0) {
            targetSide = 1;
            this.targetBlockX = rightCenter;
        }

        // Calculate aim offset to hit ball towards target
        // To send ball RIGHT: hit ball with LEFT side of paddle (paddle moves RIGHT of ball)
        // To send ball LEFT: hit ball with RIGHT side of paddle (paddle moves LEFT of ball)
        if (targetSide !== 0) {
            // More aggressive offset
            this.aimOffset = -targetSide * 0.85;
        } else {
            this.aimOffset = 0;
        }

        // Small randomness for variety
        this.aimOffset += (Math.random() - 0.5) * 0.1;
        this.aimOffset = Utils.clamp(this.aimOffset, -0.95, 0.95);
    }

    trackBallWithStrategy(ball, paddle) {
        // Use predictive tracking for better positioning
        let targetX = ball.x;

        // Predict where ball will be when it reaches paddle level
        if (ball.velocity.y > 0) {
            const timeToReach = (paddle.y - ball.y) / ball.velocity.y;
            if (timeToReach > 0 && timeToReach < 100) {
                const predicted = this.predictBallPosition(ball, timeToReach);
                targetX = predicted.x;
            }
        }

        // Apply strategic offset more aggressively
        // Apply when ball is moving down OR when we have time to position
        const ballDistanceFromPaddle = paddle.y - ball.y;
        const applyStrategy = ballDistanceFromPaddle > 100 || ball.velocity.y > 0;

        if (applyStrategy) {
            // More aggressive offset - use 45% of paddle width
            const offsetAmount = this.aimOffset * (paddle.width * 0.45);
            targetX = targetX + offsetAmount;
        }

        // Calculate where paddle should be for ball to hit at target position
        const newPaddleX = targetX - paddle.width / 2;

        // Clamp to canvas bounds
        paddle.x = Utils.clamp(newPaddleX, 0, CANVAS_WIDTH - paddle.width);
        paddle.targetX = paddle.x;
    }

    // Predictive tracking with wall bounce simulation
    predictBallPosition(ball, timeAhead) {
        let x = ball.x;
        let vx = ball.velocity.x;
        const dt = timeAhead;

        // Simulate movement with bounces
        let remainingTime = dt;
        const maxBounces = 5;
        let bounces = 0;

        while (remainingTime > 0 && bounces < maxBounces) {
            // Calculate time to hit each wall
            let timeToLeftWall = vx < 0 ? (BALL_RADIUS - x) / vx : Infinity;
            let timeToRightWall = vx > 0 ? (CANVAS_WIDTH - BALL_RADIUS - x) / vx : Infinity;

            // Time to next wall (minimum positive time)
            let timeToWall = Math.min(
                timeToLeftWall > 0 ? timeToLeftWall : Infinity,
                timeToRightWall > 0 ? timeToRightWall : Infinity
            );

            if (timeToWall > remainingTime) {
                // No wall hit in remaining time
                x += vx * remainingTime;
                remainingTime = 0;
            } else {
                // Hit wall and bounce
                x += vx * timeToWall;
                vx = -vx; // Reverse direction
                remainingTime -= timeToWall;
                bounces++;
            }
        }

        // Clamp to valid range
        x = Utils.clamp(x, BALL_RADIUS, CANVAS_WIDTH - BALL_RADIUS);

        return { x: x, y: ball.y + ball.velocity.y * dt };
    }
}

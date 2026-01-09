class Ball {
    constructor(x, y) {
        this.x = x;
        this.y = y;
        this.radius = BALL_RADIUS;
        this.speed = BALL_BASE_SPEED;
        this.velocity = { x: 0, y: 0 };
        this.isActive = true;
        this.isLaunched = false;

        // Power-up states
        this.isPenetrating = false;
        this.penetrateTimer = 0;

        // Visual effects
        this.trail = [];
        this.maxTrailLength = 10;
        this.color = COLORS.PRIMARY;
    }

    launch(paddle) {
        if (this.isLaunched) return;

        this.isLaunched = true;
        // Launch at a slight random angle upward
        const angle = Utils.random(-Math.PI / 4, -3 * Math.PI / 4);
        this.velocity.x = Math.cos(angle) * this.speed;
        this.velocity.y = Math.sin(angle) * this.speed;
    }

    update(deltaTime, paddle) {
        if (!this.isLaunched) {
            // Ball follows paddle before launch
            this.x = paddle.x + paddle.width / 2;
            this.y = paddle.y - this.radius - 2;
            return;
        }

        // Store position for trail
        this.trail.unshift({ x: this.x, y: this.y });
        if (this.trail.length > this.maxTrailLength) {
            this.trail.pop();
        }

        // Update position
        const dt = deltaTime / 16.67; // Normalize to 60fps
        this.x += this.velocity.x * dt;
        this.y += this.velocity.y * dt;

        // Update penetrate timer
        if (this.isPenetrating && this.penetrateTimer > 0) {
            this.penetrateTimer -= deltaTime;
            if (this.penetrateTimer <= 0) {
                this.isPenetrating = false;
                this.color = COLORS.PRIMARY;
            }
        }
    }

    reflect(normal) {
        // Reflection formula: v' = v - 2(v.n)n
        const dot = this.velocity.x * normal.x + this.velocity.y * normal.y;
        this.velocity.x -= 2 * dot * normal.x;
        this.velocity.y -= 2 * dot * normal.y;

        // Normalize velocity to maintain consistent speed
        this.normalizeVelocity();
    }

    reflectFromPaddle(normalizedPosition) {
        // normalizedPosition: -1 (left edge) to 1 (right edge)
        // Map to angle: -120 degrees to -60 degrees (upward)
        const minAngle = -Math.PI * 2 / 3; // -120 degrees
        const maxAngle = -Math.PI / 3;      // -60 degrees
        const angle = minAngle + (normalizedPosition + 1) / 2 * (maxAngle - minAngle);

        this.velocity.x = Math.cos(angle) * this.speed;
        this.velocity.y = Math.sin(angle) * this.speed;

        // Ensure ball is moving upward
        if (this.velocity.y > 0) {
            this.velocity.y = -this.velocity.y;
        }
    }

    normalizeVelocity() {
        const currentSpeed = Math.sqrt(
            this.velocity.x * this.velocity.x +
            this.velocity.y * this.velocity.y
        );
        if (currentSpeed > 0) {
            this.velocity.x = (this.velocity.x / currentSpeed) * this.speed;
            this.velocity.y = (this.velocity.y / currentSpeed) * this.speed;
        }
    }

    setSpeed(newSpeed) {
        this.speed = newSpeed;
        this.normalizeVelocity();
    }

    setPenetrating(duration) {
        this.isPenetrating = true;
        this.penetrateTimer = duration;
        this.color = POWERUP_CONFIG[POWERUP_TYPE.PENETRATE].color;
    }

    draw(ctx) {
        // Draw trail
        if (this.isLaunched) {
            for (let i = 0; i < this.trail.length; i++) {
                const alpha = (1 - i / this.trail.length) * 0.3;
                const trailRadius = this.radius * (1 - i / this.trail.length * 0.5);
                ctx.save();
                ctx.globalAlpha = alpha;
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(this.trail[i].x, this.trail[i].y, trailRadius, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            }
        }

        // Draw main ball with glow
        Utils.drawNeonCircle(ctx, this.x, this.y, this.radius, this.color, this.isPenetrating ? 25 : 15);

        // Penetrating effect
        if (this.isPenetrating) {
            ctx.save();
            ctx.strokeStyle = this.color;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.5 + Math.sin(Date.now() / 100) * 0.3;
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius + 5, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
        }
    }

    reset(paddle) {
        this.x = paddle.x + paddle.width / 2;
        this.y = paddle.y - this.radius - 2;
        this.velocity = { x: 0, y: 0 };
        this.isLaunched = false;
        this.isPenetrating = false;
        this.penetrateTimer = 0;
        this.color = COLORS.PRIMARY;
        this.trail = [];
        this.speed = BALL_BASE_SPEED;
    }

    clone() {
        const newBall = new Ball(this.x, this.y);
        newBall.speed = this.speed;
        newBall.isLaunched = true;
        newBall.isPenetrating = this.isPenetrating;
        newBall.penetrateTimer = this.penetrateTimer;
        newBall.color = this.color;
        return newBall;
    }
}

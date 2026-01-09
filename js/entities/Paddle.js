class Paddle {
    constructor() {
        this.width = PADDLE_WIDTH;
        this.height = PADDLE_HEIGHT;
        this.x = (CANVAS_WIDTH - this.width) / 2;
        this.y = CANVAS_HEIGHT - PADDLE_Y_OFFSET;
        this.speed = PADDLE_SPEED;
        this.targetX = this.x;
        this.color = COLORS.PRIMARY;

        // Movement state
        this.movingLeft = false;
        this.movingRight = false;
    }

    update(deltaTime) {
        const dt = deltaTime / 16.67; // Normalize to 60fps

        // Keyboard movement
        if (this.movingLeft) {
            this.x -= this.speed * dt;
        }
        if (this.movingRight) {
            this.x += this.speed * dt;
        }

        // Mouse/touch following (smooth interpolation)
        const diff = this.targetX - this.x;
        if (Math.abs(diff) > 1) {
            this.x += diff * 0.2;
        }

        // Clamp to canvas bounds
        this.x = Utils.clamp(this.x, 0, CANVAS_WIDTH - this.width);
    }

    setTargetX(x) {
        this.targetX = x - this.width / 2;
        this.targetX = Utils.clamp(this.targetX, 0, CANVAS_WIDTH - this.width);
    }

    moveLeft(pressed) {
        this.movingLeft = pressed;
    }

    moveRight(pressed) {
        this.movingRight = pressed;
    }

    getCollisionNormal(ballX) {
        // Calculate where ball hit on paddle (0 to 1)
        const relativeX = (ballX - this.x) / this.width;
        // Map to -1 to 1
        const normalizedX = Utils.clamp(relativeX * 2 - 1, -1, 1);
        return normalizedX;
    }

    getBoundingBox() {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height
        };
    }

    draw(ctx) {
        // Main paddle body with glow
        Utils.drawNeonRect(ctx, this.x, this.y, this.width, this.height, this.color);

        // Center indicator
        ctx.save();
        ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        const indicatorWidth = 10;
        ctx.fillRect(
            this.x + (this.width - indicatorWidth) / 2,
            this.y + 3,
            indicatorWidth,
            this.height - 6
        );
        ctx.restore();
    }

    reset() {
        this.x = (CANVAS_WIDTH - this.width) / 2;
        this.targetX = this.x;
        this.width = PADDLE_WIDTH;
        this.movingLeft = false;
        this.movingRight = false;
    }
}

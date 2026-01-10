class PowerUp {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.type = type;
        this.width = 32;
        this.height = 18;
        this.fallSpeed = 2.5;
        this.isActive = true;
        this.pulseTimer = 0;

        // Get config
        const config = POWERUP_CONFIG[type];
        this.color = config.color;
        this.icon = config.icon;
    }

    update(deltaTime) {
        // Fall down
        const dt = deltaTime / 16.67;
        this.y += this.fallSpeed * dt;

        // Pulse animation
        this.pulseTimer += deltaTime * 0.005;

        // Deactivate if off screen
        if (this.y > CANVAS_HEIGHT + this.height) {
            this.isActive = false;
        }
    }

    draw(ctx) {
        if (!this.isActive) return;

        const pulse = Math.sin(this.pulseTimer * 5) * 0.3 + 0.7;

        ctx.save();

        // Glow effect
        ctx.shadowBlur = 15;
        ctx.shadowColor = this.color;

        // Background
        ctx.globalAlpha = pulse;
        ctx.fillStyle = this.color;
        ctx.fillRect(
            this.x - this.width / 2,
            this.y,
            this.width,
            this.height
        );

        // Border
        ctx.globalAlpha = 1;
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            this.x - this.width / 2,
            this.y,
            this.width,
            this.height
        );

        // Icon
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(this.icon, this.x, this.y + this.height / 2);

        ctx.restore();
    }

    getBoundingBox() {
        return {
            x: this.x - this.width / 2,
            y: this.y,
            width: this.width,
            height: this.height
        };
    }
}

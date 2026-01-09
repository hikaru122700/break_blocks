class DurableBlock extends Block {
    constructor(x, y, type) {
        super(x, y, type);

        const config = BLOCK_CONFIG[type];
        this.maxHitPoints = config.hitPoints || 2;
        this.hitPoints = this.maxHitPoints;
        this.crackLevel = 0;

        // Higher score for durable blocks
        this.score = config.score;
        this.hitScore = 50; // Score for each hit
    }

    hit() {
        this.hitPoints--;
        this.crackLevel = this.maxHitPoints - this.hitPoints;

        if (this.hitPoints <= 0) {
            this.isDestroyed = true;
            return { score: this.score, destroyed: true };
        }

        return { score: this.hitScore, destroyed: false };
    }

    draw(ctx) {
        if (this.isDestroyed) return;

        // Calculate color based on remaining hit points
        const healthRatio = this.hitPoints / this.maxHitPoints;
        const baseColor = this.color;

        // Draw main block
        Utils.drawNeonRect(ctx, this.x, this.y, this.width, this.height, baseColor, 10);

        // Draw hit point indicator
        this.drawHitPointIndicator(ctx);

        // Draw cracks based on damage
        if (this.crackLevel > 0) {
            this.drawCracks(ctx);
        }
    }

    drawHitPointIndicator(ctx) {
        ctx.save();

        // Draw small circles for remaining hit points
        const indicatorSize = 4;
        const spacing = 8;
        const totalWidth = this.maxHitPoints * spacing;
        const startX = this.x + (this.width - totalWidth) / 2 + spacing / 2;
        const y = this.y + this.height - 6;

        for (let i = 0; i < this.maxHitPoints; i++) {
            const x = startX + i * spacing;
            ctx.beginPath();
            ctx.arc(x, y, indicatorSize / 2, 0, Math.PI * 2);

            if (i < this.hitPoints) {
                ctx.fillStyle = '#ffffff';
                ctx.fill();
            } else {
                ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        }

        ctx.restore();
    }

    drawCracks(ctx) {
        ctx.save();
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.lineWidth = 2;

        const centerX = this.x + this.width / 2;
        const centerY = this.y + this.height / 2;

        if (this.crackLevel >= 1) {
            // First crack pattern
            ctx.beginPath();
            ctx.moveTo(centerX - 10, centerY - 5);
            ctx.lineTo(centerX, centerY);
            ctx.lineTo(centerX + 8, centerY + 6);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            ctx.lineTo(centerX - 5, centerY + 8);
            ctx.stroke();
        }

        if (this.crackLevel >= 2) {
            // Additional cracks
            ctx.beginPath();
            ctx.moveTo(centerX + 5, centerY - 8);
            ctx.lineTo(centerX + 2, centerY - 2);
            ctx.lineTo(centerX + 12, centerY + 2);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(centerX - 15, centerY + 2);
            ctx.lineTo(centerX - 8, centerY);
            ctx.stroke();
        }

        ctx.restore();

        // Add debris particles effect
        if (this.crackLevel > 0) {
            ctx.save();
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            for (let i = 0; i < this.crackLevel * 2; i++) {
                const px = this.x + Math.random() * this.width;
                const py = this.y + Math.random() * this.height;
                ctx.fillRect(px, py, 2, 2);
            }
            ctx.restore();
        }
    }
}

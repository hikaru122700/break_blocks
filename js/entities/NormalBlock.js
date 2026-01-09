class NormalBlock extends Block {
    constructor(x, y, type) {
        super(x, y, type);
    }

    hit() {
        this.isDestroyed = true;
        return { score: this.score, destroyed: true };
    }

    draw(ctx) {
        if (this.isDestroyed) return;

        // Draw block with neon effect
        Utils.drawNeonRect(ctx, this.x, this.y, this.width, this.height, this.color, 10);

        // Add subtle pattern based on color
        ctx.save();
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = '#ffffff';

        // Pixel-style highlight
        const pixelSize = 4;
        for (let i = 0; i < 3; i++) {
            ctx.fillRect(
                this.x + 4 + i * pixelSize,
                this.y + 4,
                pixelSize - 1,
                pixelSize - 1
            );
        }

        ctx.restore();
    }
}

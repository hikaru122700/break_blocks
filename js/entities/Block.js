class Block {
    constructor(x, y, type) {
        this.x = x;
        this.y = y;
        this.width = BLOCK_WIDTH;
        this.height = BLOCK_HEIGHT;
        this.type = type;
        this.isDestroyed = false;

        // Get config from constants
        const config = BLOCK_CONFIG[type] || BLOCK_CONFIG[BLOCK_TYPE.NORMAL_BLUE];
        this.color = config.color;
        this.score = config.score;
        this.powerUpChance = 0.15; // 15% chance to drop power-up
    }

    hit() {
        // Override in subclasses
        this.isDestroyed = true;
        return { score: this.score, destroyed: true };
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
        if (this.isDestroyed) return;

        Utils.drawNeonRect(ctx, this.x, this.y, this.width, this.height, this.color, 10);
    }
}

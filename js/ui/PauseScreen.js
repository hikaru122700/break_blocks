class PauseScreen {
    constructor(game) {
        this.game = game;
        this.pulseTimer = 0;
    }

    update(deltaTime) {
        this.pulseTimer += deltaTime * 0.003;
    }

    draw(ctx) {
        // Dark overlay
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
        ctx.restore();

        // Pause title
        this.drawTitle(ctx);

        // Options
        this.drawOptions(ctx);

        // Game stats
        this.drawStats(ctx);
    }

    drawTitle(ctx) {
        const pulse = Math.sin(this.pulseTimer * 3) * 0.2 + 0.8;

        ctx.save();

        ctx.globalAlpha = pulse;
        ctx.shadowBlur = 20;
        ctx.shadowColor = COLORS.PRIMARY;

        ctx.font = 'bold 40px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.PRIMARY;
        ctx.fillText('PAUSED', CANVAS_WIDTH / 2, 180);

        ctx.restore();
    }

    drawOptions(ctx) {
        ctx.save();

        const centerX = CANVAS_WIDTH / 2;
        let y = 280;

        ctx.font = '14px "Press Start 2P", monospace';
        ctx.textAlign = 'center';

        // Resume option
        const resumeBlink = Math.sin(this.pulseTimer * 5) > 0 ? 1 : 0.5;
        ctx.globalAlpha = resumeBlink;
        ctx.fillStyle = COLORS.TEXT;
        ctx.fillText('Press ESC or P to Resume', centerX, y);

        // Menu option
        ctx.globalAlpha = 1;
        ctx.fillStyle = COLORS.TEXT_DIM;
        y += 50;
        ctx.fillText('Press M for Menu', centerX, y);

        ctx.restore();
    }

    drawStats(ctx) {
        ctx.save();

        const centerX = CANVAS_WIDTH / 2;
        const startY = 400;

        ctx.font = '10px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;

        // Current stats
        ctx.fillText(`SCORE: ${Utils.formatNumber(this.game.scoreSystem.score)}`, centerX, startY);
        ctx.fillText(`STAGE: ${this.game.currentStage} / ${MAX_STAGES}`, centerX, startY + 25);
        ctx.fillText(`LIVES: ${this.game.lives}`, centerX, startY + 50);
        ctx.fillText(`MAX COMBO: ${this.game.scoreSystem.maxCombo}`, centerX, startY + 75);

        ctx.restore();
    }
}

class MenuScreen {
    constructor(game) {
        this.game = game;
        this.titlePulse = 0;
        this.starField = this.createStarField();
    }

    createStarField() {
        const stars = [];
        for (let i = 0; i < 50; i++) {
            stars.push({
                x: Math.random() * CANVAS_WIDTH,
                y: Math.random() * CANVAS_HEIGHT,
                size: Math.random() * 2 + 1,
                speed: Math.random() * 0.5 + 0.2,
                alpha: Math.random()
            });
        }
        return stars;
    }

    update(deltaTime) {
        this.titlePulse += deltaTime * 0.003;

        // Update stars
        for (const star of this.starField) {
            star.y += star.speed;
            star.alpha = Math.sin(this.titlePulse + star.x) * 0.5 + 0.5;

            if (star.y > CANVAS_HEIGHT) {
                star.y = 0;
                star.x = Math.random() * CANVAS_WIDTH;
            }
        }
    }

    draw(ctx) {
        // Draw star field
        this.drawStarField(ctx);

        // Draw title
        this.drawTitle(ctx);

        // Draw menu options
        this.drawMenuOptions(ctx);

        // Draw controls help
        this.drawControls(ctx);

        // Draw high score
        this.drawHighScore(ctx);
    }

    drawStarField(ctx) {
        ctx.save();
        for (const star of this.starField) {
            ctx.globalAlpha = star.alpha;
            ctx.fillStyle = COLORS.PRIMARY;
            ctx.beginPath();
            ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    drawTitle(ctx) {
        const pulse = Math.sin(this.titlePulse * 2) * 0.2 + 0.8;

        ctx.save();

        // Glow effect
        ctx.shadowBlur = 30;
        ctx.shadowColor = COLORS.PRIMARY;

        // Main title
        ctx.font = 'bold 48px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText('BLOCK', CANVAS_WIDTH / 2 + 3, 150 + 3);
        ctx.fillText('BREAKER', CANVAS_WIDTH / 2 + 3, 210 + 3);

        // Gradient text
        const gradient = ctx.createLinearGradient(0, 120, 0, 240);
        gradient.addColorStop(0, COLORS.PRIMARY);
        gradient.addColorStop(1, COLORS.SECONDARY);

        ctx.globalAlpha = pulse;
        ctx.fillStyle = gradient;
        ctx.fillText('BLOCK', CANVAS_WIDTH / 2, 150);
        ctx.fillText('BREAKER', CANVAS_WIDTH / 2, 210);

        ctx.restore();
    }

    drawMenuOptions(ctx) {
        const blinkAlpha = Math.sin(this.titlePulse * 4) > 0 ? 1 : 0.3;

        ctx.save();
        ctx.globalAlpha = blinkAlpha;
        ctx.font = '16px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT;
        ctx.fillText('PRESS SPACE TO START', CANVAS_WIDTH / 2, 340);
        ctx.restore();

        // AI Mode option
        ctx.save();
        ctx.font = '12px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.SECONDARY;
        ctx.fillText('Press A for AI Demo', CANVAS_WIDTH / 2, 385);
        ctx.restore();

        // Ranking option
        ctx.save();
        ctx.font = '12px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('Press R for Ranking', CANVAS_WIDTH / 2, 420);
        ctx.restore();
    }

    drawControls(ctx) {
        ctx.save();
        ctx.font = '10px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;

        const y = 480;
        ctx.fillText('CONTROLS', CANVAS_WIDTH / 2, y);
        ctx.fillText('Mouse or Arrow Keys: Move Paddle', CANVAS_WIDTH / 2, y + 25);
        ctx.fillText('Space or Click: Launch Ball', CANVAS_WIDTH / 2, y + 45);
        ctx.fillText('ESC or P: Pause', CANVAS_WIDTH / 2, y + 65);

        ctx.restore();
    }

    drawHighScore(ctx) {
        const highScores = this.game.storageManager.loadHighScores();

        if (highScores.length > 0) {
            ctx.save();
            ctx.font = '12px "Press Start 2P", monospace';
            ctx.textAlign = 'center';
            ctx.fillStyle = COLORS.SECONDARY;
            ctx.fillText(`HIGH SCORE: ${Utils.formatNumber(highScores[0].score)}`, CANVAS_WIDTH / 2, 300);
            ctx.restore();
        }
    }
}

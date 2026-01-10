class HUD {
    constructor(game) {
        this.game = game;
    }

    draw(ctx) {
        this.drawScore(ctx);
        this.drawLives(ctx);
        this.drawStage(ctx);
        this.drawTimer(ctx);
        this.drawCombo(ctx);
        this.drawActiveEffects(ctx);

        // Speed indicator (always shown during gameplay)
        this.drawSpeedIndicator(ctx);

        // AI Mode indicator
        if (this.game.isAIMode) {
            this.drawAIModeIndicator(ctx);
        }
    }

    drawScore(ctx) {
        ctx.save();

        // Score label
        ctx.font = '10px "Press Start 2P", monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('SCORE', 20, 25);

        // Score value
        ctx.font = '16px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.PRIMARY;
        ctx.shadowBlur = 10;
        ctx.shadowColor = COLORS.PRIMARY;
        ctx.fillText(Utils.formatNumber(this.game.scoreSystem.score), 20, 50);

        ctx.restore();
    }

    drawLives(ctx) {
        ctx.save();

        const startX = CANVAS_WIDTH - 20;
        const y = 35;
        const heartSize = 15;
        const spacing = 25;

        ctx.font = '10px "Press Start 2P", monospace';
        ctx.textAlign = 'right';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('LIVES', startX, 25);

        // Draw hearts for lives
        for (let i = 0; i < this.game.lives; i++) {
            const x = startX - i * spacing - heartSize / 2;
            this.drawHeart(ctx, x, y, heartSize);
        }

        ctx.restore();
    }

    drawHeart(ctx, x, y, size) {
        ctx.save();
        ctx.fillStyle = '#ff0066';
        ctx.shadowBlur = 10;
        ctx.shadowColor = '#ff0066';

        ctx.beginPath();
        ctx.moveTo(x, y + size / 4);
        ctx.bezierCurveTo(x, y, x - size / 2, y, x - size / 2, y + size / 4);
        ctx.bezierCurveTo(x - size / 2, y + size / 2, x, y + size * 0.7, x, y + size);
        ctx.bezierCurveTo(x, y + size * 0.7, x + size / 2, y + size / 2, x + size / 2, y + size / 4);
        ctx.bezierCurveTo(x + size / 2, y, x, y, x, y + size / 4);
        ctx.fill();

        ctx.restore();
    }

    drawStage(ctx) {
        ctx.save();

        ctx.font = '12px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText(`STAGE ${this.game.currentStage}`, CANVAS_WIDTH / 2, 25);

        // Progress bar
        const barWidth = 100;
        const barHeight = 6;
        const barX = (CANVAS_WIDTH - barWidth) / 2;
        const barY = 35;

        // Background
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fillRect(barX, barY, barWidth, barHeight);

        // Progress
        const totalBlocks = this.game.blocks.length;
        const destroyedBlocks = this.game.blocks.filter(b => b.isDestroyed).length;
        const progress = totalBlocks > 0 ? destroyedBlocks / totalBlocks : 0;

        ctx.fillStyle = COLORS.PRIMARY;
        ctx.shadowBlur = 5;
        ctx.shadowColor = COLORS.PRIMARY;
        ctx.fillRect(barX, barY, barWidth * progress, barHeight);

        ctx.restore();
    }

    drawTimer(ctx) {
        ctx.save();

        const timeRemaining = Math.max(0, this.game.stageTimeRemaining);
        const seconds = Math.ceil(timeRemaining / 1000);
        const minutes = Math.floor(seconds / 60);
        const secs = seconds % 60;
        const timeText = `${minutes}:${secs.toString().padStart(2, '0')}`;

        // Position below stage info
        const x = CANVAS_WIDTH / 2;
        const y = 55;

        // Warning color when time is low
        let color = COLORS.TEXT;
        if (seconds <= 10) {
            // Flashing red when very low
            const flash = Math.sin(Date.now() / 100) > 0;
            color = flash ? '#ff0044' : '#ff4444';
        } else if (seconds <= 30) {
            color = '#ffaa00';
        }

        ctx.font = '14px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = color;

        if (seconds <= 30) {
            ctx.shadowBlur = 10;
            ctx.shadowColor = color;
        }

        ctx.fillText(timeText, x, y);

        ctx.restore();
    }

    drawCombo(ctx) {
        const comboInfo = this.game.scoreSystem.getComboInfo();

        if (comboInfo.combo >= 3) {
            ctx.save();

            const x = 20;
            const y = 80;

            // Combo text
            ctx.font = '14px "Press Start 2P", monospace';
            ctx.textAlign = 'left';

            // Color based on combo
            let color = '#ffff00';
            if (comboInfo.combo >= 20) {
                color = COLORS.SECONDARY;
            } else if (comboInfo.combo >= 10) {
                color = '#ff8800';
            }

            ctx.fillStyle = color;
            ctx.shadowBlur = 10;
            ctx.shadowColor = color;

            ctx.fillText(`${comboInfo.combo} COMBO!`, x, y);

            // Multiplier
            ctx.font = '10px "Press Start 2P", monospace';
            ctx.fillStyle = COLORS.TEXT;
            ctx.shadowBlur = 0;
            ctx.fillText(`x${comboInfo.multiplier.toFixed(1)}`, x, y + 18);

            // Timer bar
            const timerWidth = 60;
            const timerHeight = 4;
            const timerProgress = comboInfo.timeRemaining / 2000;

            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.fillRect(x, y + 25, timerWidth, timerHeight);

            ctx.fillStyle = color;
            ctx.fillRect(x, y + 25, timerWidth * timerProgress, timerHeight);

            ctx.restore();
        }
    }

    drawActiveEffects(ctx) {
        const effects = this.game.powerUpSystem.getActiveEffectsInfo();

        if (effects.length === 0) return;

        ctx.save();

        const startX = CANVAS_WIDTH - 50;
        const startY = 80;
        const spacing = 30;

        for (let i = 0; i < effects.length; i++) {
            const effect = effects[i];
            const y = startY + i * spacing;

            // Background
            ctx.fillStyle = effect.color;
            ctx.globalAlpha = 0.3;
            ctx.fillRect(startX - 20, y - 10, 40, 25);

            // Icon
            ctx.globalAlpha = 1;
            ctx.font = 'bold 14px "Press Start 2P", monospace';
            ctx.textAlign = 'center';
            ctx.fillStyle = effect.color;
            ctx.shadowBlur = 10;
            ctx.shadowColor = effect.color;
            ctx.fillText(effect.icon, startX, y + 5);

            // Timer
            ctx.shadowBlur = 0;
            ctx.font = '8px "Press Start 2P", monospace';
            ctx.fillStyle = COLORS.TEXT;
            const seconds = Math.ceil(effect.remainingTime / 1000);
            ctx.fillText(`${seconds}s`, startX, y + 18);
        }

        ctx.restore();
    }

    drawSpeedIndicator(ctx) {
        ctx.save();

        const scaleY = CANVAS_HEIGHT - 30;
        ctx.font = '10px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('SPEED: 1  2  3  4  5', CANVAS_WIDTH / 2, scaleY);

        // Highlight current speed
        const speeds = [1, 2, 3, 4, 5];
        const startX = CANVAS_WIDTH / 2 - 60;
        const spacing = 30;

        for (let i = 0; i < speeds.length; i++) {
            const x = startX + i * spacing;
            if (speeds[i] === this.game.timeScale) {
                ctx.fillStyle = COLORS.PRIMARY;
                ctx.shadowBlur = 10;
                ctx.shadowColor = COLORS.PRIMARY;
                ctx.fillRect(x - 8, scaleY - 12, 16, 16);
                ctx.shadowBlur = 0;
                ctx.fillStyle = COLORS.BACKGROUND;
                ctx.fillText(speeds[i].toString(), x, scaleY);
            }
        }

        // Instructions
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.font = '8px "Press Start 2P", monospace';
        ctx.fillText('Press 1-5 to change speed', CANVAS_WIDTH / 2, scaleY + 18);

        ctx.restore();
    }

    drawAIModeIndicator(ctx) {
        ctx.save();

        const pulse = Math.sin(Date.now() / 300) * 0.3 + 0.7;

        // AI Mode badge at top-left
        const badgeX = 150;
        const badgeY = 15;

        // Background
        ctx.fillStyle = COLORS.SECONDARY;
        ctx.globalAlpha = 0.3;
        ctx.fillRect(badgeX, badgeY, 120, 25);

        // Text
        ctx.globalAlpha = pulse;
        ctx.font = 'bold 10px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.SECONDARY;
        ctx.shadowBlur = 10;
        ctx.shadowColor = COLORS.SECONDARY;
        ctx.fillText('AI PLAYING', badgeX + 60, badgeY + 17);

        ctx.restore();
    }
}

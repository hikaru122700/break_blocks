class ScoreSystem {
    constructor() {
        this.score = 0;
        this.combo = 0;
        this.maxCombo = 0;
        this.comboTimer = 0;
        this.comboTimeout = 2000; // 2 seconds to maintain combo
        this.scorePopups = [];
    }

    addScore(baseScore, x, y) {
        // Increment combo
        this.combo++;
        this.comboTimer = this.comboTimeout;

        // Track max combo
        if (this.combo > this.maxCombo) {
            this.maxCombo = this.combo;
        }

        // Calculate final score with combo multiplier
        const multiplier = this.getComboMultiplier();
        const finalScore = Math.floor(baseScore * multiplier);
        this.score += finalScore;

        // Add score popup
        if (x !== undefined && y !== undefined) {
            this.scorePopups.push({
                x,
                y,
                score: finalScore,
                combo: this.combo,
                alpha: 1,
                offsetY: 0,
                scale: 1 + Math.min(this.combo * 0.05, 0.5)
            });
        }

        return finalScore;
    }

    getComboMultiplier() {
        for (const tier of COMBO_MULTIPLIERS) {
            if (this.combo >= tier.combo) {
                return tier.multiplier;
            }
        }
        return 1.0;
    }

    update(deltaTime) {
        // Update combo timer
        if (this.comboTimer > 0) {
            this.comboTimer -= deltaTime;
            if (this.comboTimer <= 0) {
                this.combo = 0;
            }
        }

        // Update score popups
        for (const popup of this.scorePopups) {
            popup.offsetY -= 1.5;
            popup.alpha -= 0.02;
        }

        // Remove faded popups
        this.scorePopups = this.scorePopups.filter(p => p.alpha > 0);
    }

    resetCombo() {
        this.combo = 0;
        this.comboTimer = 0;
    }

    addPopup(x, y, text, color = COLORS.TEXT) {
        this.scorePopups.push({
            x,
            y,
            text: text,
            score: null,
            combo: 0,
            alpha: 1,
            offsetY: 0,
            scale: 1.2,
            customColor: color
        });
    }

    reset() {
        this.score = 0;
        this.combo = 0;
        this.maxCombo = 0;
        this.comboTimer = 0;
        this.scorePopups = [];
    }

    drawPopups(ctx) {
        ctx.save();

        for (const popup of this.scorePopups) {
            ctx.globalAlpha = popup.alpha;

            // Color based on combo or custom color
            let color = popup.customColor || COLORS.TEXT;
            if (!popup.customColor) {
                if (popup.combo >= 10) {
                    color = COLORS.SECONDARY;
                } else if (popup.combo >= 5) {
                    color = '#ffff00';
                }
            }

            // Build text
            let text;
            if (popup.text) {
                text = popup.text;
            } else {
                text = `+${popup.score}`;
                if (popup.combo > 1) {
                    text += ` x${popup.combo}`;
                }
            }

            // Draw with scale effect
            const fontSize = Math.floor(14 * popup.scale);
            ctx.font = `bold ${fontSize}px "Press Start 2P", monospace`;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Shadow
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillText(text, popup.x + 1, popup.y + popup.offsetY + 1);

            // Main text
            ctx.fillStyle = color;
            ctx.shadowBlur = 10;
            ctx.shadowColor = color;
            ctx.fillText(text, popup.x, popup.y + popup.offsetY);
        }

        ctx.restore();
    }

    // Get combo info for HUD display
    getComboInfo() {
        return {
            combo: this.combo,
            multiplier: this.getComboMultiplier(),
            timeRemaining: this.comboTimer,
            maxCombo: this.maxCombo
        };
    }
}

class RankingScreen {
    constructor(game) {
        this.game = game;
        this.scores = [];
        this.isEnteringName = false;
        this.playerName = '';
        this.pendingScore = 0;
        this.pendingStage = 0;
        this.newScoreRank = -1;
        this.cursorBlink = 0;

        // Setup keyboard input for name entry
        this.setupKeyboardInput();
    }

    setupKeyboardInput() {
        document.addEventListener('keydown', (e) => {
            if (!this.isEnteringName) return;

            if (e.key === 'Enter' && this.playerName.length > 0) {
                this.submitScore();
            } else if (e.key === 'Backspace') {
                this.playerName = this.playerName.slice(0, -1);
            } else if (e.key.length === 1 && this.playerName.length < 8) {
                // Allow alphanumeric characters only
                if (/[a-zA-Z0-9]/.test(e.key)) {
                    this.playerName += e.key.toUpperCase();
                }
            }
        });
    }

    show() {
        this.scores = this.game.storageManager.loadHighScores();
        this.isEnteringName = false;
        this.newScoreRank = -1;
    }

    showNameEntry(score, stage) {
        this.isEnteringName = true;
        this.pendingScore = score;
        this.pendingStage = stage;
        this.playerName = '';
        this.newScoreRank = -1;
    }

    submitScore() {
        this.newScoreRank = this.game.storageManager.addHighScore(
            this.playerName,
            this.pendingScore,
            this.pendingStage
        );
        this.isEnteringName = false;
        this.scores = this.game.storageManager.loadHighScores();
    }

    update(deltaTime) {
        this.cursorBlink += deltaTime * 0.005;

        // Check for exit
        if (!this.isEnteringName && this.game.inputManager.isConfirmPressed()) {
            this.game.state.setState(GAME_STATE.MENU);
        }
    }

    draw(ctx) {
        // Background
        ctx.fillStyle = COLORS.BACKGROUND;
        ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

        // Grid
        Utils.drawGrid(ctx, CANVAS_WIDTH, CANVAS_HEIGHT);

        if (this.isEnteringName) {
            this.drawNameEntry(ctx);
        } else {
            this.drawScoreList(ctx);
        }
    }

    drawNameEntry(ctx) {
        ctx.save();

        // Title
        ctx.font = 'bold 28px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.SECONDARY;
        ctx.shadowBlur = 20;
        ctx.shadowColor = COLORS.SECONDARY;
        ctx.fillText('NEW HIGH SCORE!', CANVAS_WIDTH / 2, 120);

        // Score display
        ctx.shadowBlur = 0;
        ctx.font = '20px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.TEXT;
        ctx.fillText(Utils.formatNumber(this.pendingScore), CANVAS_WIDTH / 2, 180);

        // Stage
        ctx.font = '14px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText(`STAGE ${this.pendingStage}`, CANVAS_WIDTH / 2, 220);

        // Name entry prompt
        ctx.font = '14px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.TEXT;
        ctx.fillText('ENTER YOUR NAME:', CANVAS_WIDTH / 2, 300);

        // Name input box
        const boxWidth = 250;
        const boxHeight = 50;
        const boxX = (CANVAS_WIDTH - boxWidth) / 2;
        const boxY = 330;

        ctx.strokeStyle = COLORS.PRIMARY;
        ctx.lineWidth = 3;
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // Name text
        const cursor = Math.sin(this.cursorBlink * 5) > 0 ? '_' : '';
        ctx.font = 'bold 24px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.PRIMARY;
        ctx.shadowBlur = 10;
        ctx.shadowColor = COLORS.PRIMARY;
        ctx.fillText(this.playerName + cursor, CANVAS_WIDTH / 2, boxY + 35);

        // Instructions
        ctx.shadowBlur = 0;
        ctx.font = '10px "Press Start 2P", monospace';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('Press ENTER to confirm', CANVAS_WIDTH / 2, 420);
        ctx.fillText('(Max 8 characters)', CANVAS_WIDTH / 2, 445);

        ctx.restore();
    }

    drawScoreList(ctx) {
        ctx.save();

        // Title
        ctx.font = 'bold 32px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.PRIMARY;
        ctx.shadowBlur = 20;
        ctx.shadowColor = COLORS.PRIMARY;
        ctx.fillText('HIGH SCORES', CANVAS_WIDTH / 2, 80);

        // Score list
        ctx.shadowBlur = 0;
        const startY = 140;
        const lineHeight = 45;

        for (let i = 0; i < 10; i++) {
            const y = startY + i * lineHeight;
            const score = this.scores[i];
            const isNew = i === this.newScoreRank - 1;

            // Rank number
            ctx.font = '14px "Press Start 2P", monospace';
            ctx.textAlign = 'right';
            ctx.fillStyle = isNew ? COLORS.SECONDARY : COLORS.TEXT_DIM;
            ctx.fillText(`${i + 1}.`, 180, y);

            if (score) {
                // Highlight new score
                if (isNew) {
                    ctx.fillStyle = 'rgba(255, 0, 255, 0.2)';
                    ctx.fillRect(190, y - 15, 420, 25);
                }

                // Name
                ctx.textAlign = 'left';
                ctx.fillStyle = isNew ? COLORS.SECONDARY : COLORS.TEXT;
                ctx.fillText(score.name.padEnd(8, ' '), 200, y);

                // Score
                ctx.textAlign = 'right';
                ctx.fillText(Utils.formatNumber(score.score).padStart(10, ' '), 500, y);

                // Stage
                ctx.fillStyle = COLORS.TEXT_DIM;
                ctx.font = '10px "Press Start 2P", monospace';
                ctx.fillText(`ST${score.stage}`, 560, y);
            } else {
                // Empty slot
                ctx.textAlign = 'left';
                ctx.fillStyle = COLORS.TEXT_DIM;
                ctx.fillText('--------', 200, y);
                ctx.textAlign = 'right';
                ctx.fillText('--------', 500, y);
            }
        }

        // Instructions
        ctx.font = '12px "Press Start 2P", monospace';
        ctx.textAlign = 'center';
        ctx.fillStyle = COLORS.TEXT_DIM;
        ctx.fillText('Press SPACE to return', CANVAS_WIDTH / 2, 560);

        ctx.restore();
    }
}

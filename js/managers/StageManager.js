class StageManager {
    constructor() {
        this.stageData = new StageData();
        this.currentStage = 1;
    }

    loadStage(stageNumber) {
        this.currentStage = stageNumber;
        const stage = this.stageData.getStage(stageNumber);
        return this.createBlocksFromLayout(stage.layout);
    }

    createBlocksFromLayout(layout) {
        const blocks = [];

        for (let row = 0; row < layout.length; row++) {
            for (let col = 0; col < layout[row].length; col++) {
                const type = layout[row][col];

                if (type !== BLOCK_TYPE.NONE) {
                    const x = BLOCK_OFFSET_LEFT + col * (BLOCK_WIDTH + BLOCK_PADDING);
                    const y = BLOCK_OFFSET_TOP + row * (BLOCK_HEIGHT + BLOCK_PADDING);

                    const block = this.createBlock(x, y, type);
                    blocks.push(block);
                }
            }
        }

        return blocks;
    }

    createBlock(x, y, type) {
        // Check if it's a durable block type
        if (type === BLOCK_TYPE.DURABLE_1 || type === BLOCK_TYPE.DURABLE_2) {
            return new DurableBlock(x, y, type);
        }

        // Normal block
        return new NormalBlock(x, y, type);
    }

    getStageConfig(stageNumber) {
        return this.stageData.getStage(stageNumber);
    }

    getBallSpeedMultiplier(stageNumber) {
        const stage = this.stageData.getStage(stageNumber);
        return stage.ballSpeedMultiplier || 1.0;
    }

    getPowerUpChance(stageNumber) {
        const stage = this.stageData.getStage(stageNumber);
        return stage.powerUpChance || 0.15;
    }

    getTimeLimit(stageNumber) {
        const stage = this.stageData.getStage(stageNumber);
        return stage.timeLimit || STAGE_TIME_LIMIT;
    }

    isLastStage(stageNumber) {
        return stageNumber >= MAX_STAGES;
    }

    getNextStage() {
        if (this.currentStage < MAX_STAGES) {
            return this.currentStage + 1;
        }
        return null;
    }
}

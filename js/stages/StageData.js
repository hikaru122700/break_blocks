class StageData {
    constructor() {
        this.stages = this.defineStages();
    }

    getStage(stageNumber) {
        const index = Math.min(stageNumber - 1, this.stages.length - 1);
        return this.stages[index];
    }

    defineStages() {
        const B = BLOCK_TYPE;

        return [
            // Stage 1: Introduction - Simple rows (36 hits)
            {
                layout: [
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW],
                ],
                ballSpeedMultiplier: 1.0,
                powerUpChance: 0.18,
                timeLimit: 90000 // 90 seconds
            },

            // Stage 2: Pyramid (30 hits)
            {
                layout: [
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_RED, B.NORMAL_RED, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NONE, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE, B.NONE],
                    [B.NONE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NONE],
                ],
                ballSpeedMultiplier: 1.0,
                powerUpChance: 0.15,
                timeLimit: 90000 // 90 seconds
            },

            // Stage 3: Durable blocks introduction (56 hits)
            {
                layout: [
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NORMAL_YELLOW],
                    [B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE],
                ],
                ballSpeedMultiplier: 1.05,
                powerUpChance: 0.18,
                timeLimit: 120000 // 120 seconds
            },

            // Stage 4: Checkerboard (36 hits)
            {
                layout: [
                    [B.NORMAL_RED, B.NONE, B.NORMAL_RED, B.NONE, B.NORMAL_RED, B.NONE, B.NORMAL_RED, B.NONE, B.NORMAL_RED, B.NONE, B.NORMAL_RED, B.NONE],
                    [B.NONE, B.NORMAL_ORANGE, B.NONE, B.NORMAL_ORANGE, B.NONE, B.NORMAL_ORANGE, B.NONE, B.NORMAL_ORANGE, B.NONE, B.NORMAL_ORANGE, B.NONE, B.NORMAL_ORANGE],
                    [B.NORMAL_YELLOW, B.NONE, B.NORMAL_YELLOW, B.NONE, B.NORMAL_YELLOW, B.NONE, B.NORMAL_YELLOW, B.NONE, B.NORMAL_YELLOW, B.NONE, B.NORMAL_YELLOW, B.NONE],
                    [B.NONE, B.NORMAL_GREEN, B.NONE, B.NORMAL_GREEN, B.NONE, B.NORMAL_GREEN, B.NONE, B.NORMAL_GREEN, B.NONE, B.NORMAL_GREEN, B.NONE, B.NORMAL_GREEN],
                    [B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE],
                    [B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE, B.NONE, B.NORMAL_BLUE],
                ],
                ballSpeedMultiplier: 1.1,
                powerUpChance: 0.15,
                timeLimit: 90000 // 90 seconds
            },

            // Stage 5: Center hole pattern (48 hits)
            {
                layout: [
                    [B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED],
                    [B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE],
                    [B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NONE, B.NONE, B.NONE, B.DURABLE_2, B.DURABLE_2, B.NONE, B.NONE, B.NONE, B.NORMAL_YELLOW, B.NORMAL_YELLOW],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE, B.NONE, B.NONE, B.DURABLE_2, B.DURABLE_2, B.NONE, B.NONE, B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE],
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE],
                ],
                ballSpeedMultiplier: 1.15,
                powerUpChance: 0.2,
                timeLimit: 120000 // 120 seconds
            },

            // Stage 6: Wave pattern (26 hits)
            {
                layout: [
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_BLUE, B.NORMAL_BLUE],
                    [B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NONE],
                    [B.NORMAL_RED, B.NORMAL_RED, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_RED, B.NORMAL_RED],
                ],
                ballSpeedMultiplier: 1.2,
                powerUpChance: 0.18,
                timeLimit: 90000 // 90 seconds
            },

            // Stage 7: Fortress with durable walls (96 hits)
            {
                layout: [
                    [B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE],
                    [B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2, B.DURABLE_2],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1],
                    [B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED],
                ],
                ballSpeedMultiplier: 1.25,
                powerUpChance: 0.22,
                timeLimit: 180000 // 180 seconds
            },

            // Stage 8: Diamond pattern (50 hits)
            {
                layout: [
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.DURABLE_2, B.DURABLE_2, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_1, B.NONE, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_1, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_1, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NONE, B.NONE, B.NONE, B.NONE],
                    [B.NONE, B.NONE, B.NONE, B.NONE, B.NONE, B.DURABLE_2, B.DURABLE_2, B.NONE, B.NONE, B.NONE, B.NONE, B.NONE],
                ],
                ballSpeedMultiplier: 1.3,
                powerUpChance: 0.2,
                timeLimit: 150000 // 150 seconds
            },

            // Stage 9: Maze-like (66 hits)
            {
                layout: [
                    [B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.NONE, B.NONE, B.NONE, B.NONE, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1, B.DURABLE_1],
                    [B.NONE, B.NONE, B.NONE, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.NONE, B.NONE, B.NONE],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE, B.NONE, B.NONE, B.DURABLE_2, B.DURABLE_2, B.NONE, B.NONE, B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NONE, B.NONE, B.NONE, B.NONE, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN],
                    [B.NONE, B.NONE, B.NONE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NORMAL_BLUE, B.NONE, B.NONE, B.NONE],
                    [B.DURABLE_1, B.DURABLE_1, B.NONE, B.NONE, B.DURABLE_1, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_1, B.NONE, B.NONE, B.DURABLE_1, B.DURABLE_1],
                ],
                ballSpeedMultiplier: 1.35,
                powerUpChance: 0.22,
                timeLimit: 180000 // 180 seconds
            },

            // Stage 10: Final Challenge (112 hits)
            {
                layout: [
                    [B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2, B.NORMAL_RED, B.NORMAL_RED, B.DURABLE_2],
                    [B.NORMAL_ORANGE, B.DURABLE_2, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_1, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_1, B.NORMAL_ORANGE, B.NORMAL_ORANGE, B.DURABLE_2, B.NORMAL_ORANGE],
                    [B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW, B.DURABLE_2, B.NORMAL_YELLOW, B.NORMAL_YELLOW],
                    [B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1, B.NORMAL_GREEN, B.NORMAL_GREEN, B.DURABLE_1],
                    [B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_1, B.NORMAL_BLUE],
                    [B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.DURABLE_2, B.DURABLE_2, B.NORMAL_BLUE, B.DURABLE_2, B.NORMAL_BLUE, B.NORMAL_BLUE, B.DURABLE_2],
                ],
                ballSpeedMultiplier: 1.4,
                powerUpChance: 0.25,
                timeLimit: 240000 // 240 seconds (4 minutes)
            }
        ];
    }
}

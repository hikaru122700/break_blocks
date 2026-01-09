class StorageManager {
    constructor() {
        this.prefix = 'blockBreaker_';
    }

    // High Scores
    saveHighScores(scores) {
        try {
            localStorage.setItem(
                this.prefix + 'highScores',
                JSON.stringify(scores)
            );
            return true;
        } catch (e) {
            console.warn('Failed to save high scores:', e);
            return false;
        }
    }

    loadHighScores() {
        try {
            const data = localStorage.getItem(this.prefix + 'highScores');
            return data ? JSON.parse(data) : [];
        } catch (e) {
            console.warn('Failed to load high scores:', e);
            return [];
        }
    }

    addHighScore(name, score, stage) {
        const scores = this.loadHighScores();

        const entry = {
            name: name.substring(0, 8).toUpperCase(),
            score,
            stage,
            date: new Date().toISOString(),
            id: Date.now()
        };

        scores.push(entry);
        scores.sort((a, b) => b.score - a.score);
        scores.splice(10); // Keep top 10 only

        this.saveHighScores(scores);

        // Return rank (1-based)
        return scores.findIndex(s => s.id === entry.id) + 1;
    }

    isHighScore(score) {
        const scores = this.loadHighScores();
        if (scores.length < 10) return true;
        return score > scores[scores.length - 1].score;
    }

    // Progress
    saveProgress(progress) {
        try {
            localStorage.setItem(
                this.prefix + 'progress',
                JSON.stringify(progress)
            );
            return true;
        } catch (e) {
            console.warn('Failed to save progress:', e);
            return false;
        }
    }

    loadProgress() {
        try {
            const data = localStorage.getItem(this.prefix + 'progress');
            return data ? JSON.parse(data) : this.getDefaultProgress();
        } catch (e) {
            console.warn('Failed to load progress:', e);
            return this.getDefaultProgress();
        }
    }

    getDefaultProgress() {
        return {
            unlockedStage: 1,
            bestStage: 1,
            totalGamesPlayed: 0,
            totalPlayTime: 0
        };
    }

    // Settings
    saveSettings(settings) {
        try {
            localStorage.setItem(
                this.prefix + 'settings',
                JSON.stringify(settings)
            );
            return true;
        } catch (e) {
            console.warn('Failed to save settings:', e);
            return false;
        }
    }

    loadSettings() {
        try {
            const data = localStorage.getItem(this.prefix + 'settings');
            return data ? JSON.parse(data) : this.getDefaultSettings();
        } catch (e) {
            return this.getDefaultSettings();
        }
    }

    getDefaultSettings() {
        return {
            bgmVolume: 0.3,
            sfxVolume: 0.5,
            isMuted: false
        };
    }

    // Clear all data
    clearAllData() {
        const keys = ['highScores', 'progress', 'settings'];
        for (const key of keys) {
            try {
                localStorage.removeItem(this.prefix + key);
            } catch (e) {
                console.warn(`Failed to remove ${key}:`, e);
            }
        }
    }

    // Check if storage is available
    isAvailable() {
        try {
            const test = '__storage_test__';
            localStorage.setItem(test, test);
            localStorage.removeItem(test);
            return true;
        } catch (e) {
            return false;
        }
    }
}

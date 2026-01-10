class SoundManager {
    constructor() {
        this.audioContext = null;
        this.isMuted = false;
        this.bgmVolume = 0.3;
        this.sfxVolume = 0.5;
        this.isInitialized = false;
        this.currentBGM = null;
        this.bgmGain = null;
    }

    async loadSounds() {
        // Initialize on first user interaction
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.bgmGain = this.audioContext.createGain();
            this.bgmGain.connect(this.audioContext.destination);
            this.bgmGain.gain.value = this.bgmVolume;
            this.isInitialized = true;
        } catch (e) {
            console.warn('Web Audio API not supported:', e);
        }
    }

    // Generate sound effects using Web Audio API (no external files needed)
    playSFX(name) {
        if (this.isMuted || !this.isInitialized) return;

        // Resume context if suspended
        if (this.audioContext.state === 'suspended') {
            this.audioContext.resume();
        }

        switch (name) {
            case 'hit_paddle':
                this.playTone(200, 0.1, 'square', 0.3);
                break;
            case 'hit_block':
                this.playTone(400 + Math.random() * 200, 0.08, 'square', 0.25);
                break;
            case 'hit_wall':
                this.playTone(150, 0.05, 'sine', 0.2);
                break;
            case 'powerup':
                this.playArpeggio([400, 500, 600, 800], 0.08, 'square', 0.3);
                break;
            case 'lose_ball':
                this.playDescending(400, 100, 0.3, 'sawtooth', 0.4);
                break;
            case 'clear':
                this.playVictory();
                break;
            case 'game_over':
                this.playGameOver();
                break;
            case 'menu_select':
                this.playTone(600, 0.1, 'square', 0.2);
                break;
        }
    }

    playTone(frequency, duration, type = 'sine', volume = 0.5) {
        if (!this.isInitialized) return;

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        oscillator.type = type;
        oscillator.frequency.value = frequency;

        gainNode.gain.setValueAtTime(volume * this.sfxVolume, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);

        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    playArpeggio(frequencies, noteDuration, type = 'square', volume = 0.3) {
        if (!this.isInitialized) return;

        frequencies.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, noteDuration, type, volume);
            }, index * noteDuration * 1000 * 0.8);
        });
    }

    playDescending(startFreq, endFreq, duration, type = 'sawtooth', volume = 0.4) {
        if (!this.isInitialized) return;

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(this.audioContext.destination);

        oscillator.type = type;
        oscillator.frequency.setValueAtTime(startFreq, this.audioContext.currentTime);
        oscillator.frequency.exponentialRampToValueAtTime(endFreq, this.audioContext.currentTime + duration);

        gainNode.gain.setValueAtTime(volume * this.sfxVolume, this.audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, this.audioContext.currentTime + duration);

        oscillator.start(this.audioContext.currentTime);
        oscillator.stop(this.audioContext.currentTime + duration);
    }

    playVictory() {
        this.playArpeggio([523, 659, 784, 1047], 0.15, 'square', 0.3);
    }

    playGameOver() {
        const notes = [400, 350, 300, 200];
        notes.forEach((freq, index) => {
            setTimeout(() => {
                this.playTone(freq, 0.3, 'sawtooth', 0.3);
            }, index * 200);
        });
    }

    // Simple BGM using oscillators
    playBGM(name) {
        if (!this.isInitialized) return;

        this.stopBGM();

        // Create a simple looping BGM pattern
        this.currentBGM = {
            isPlaying: true,
            intervalId: null
        };

        const playPattern = () => {
            if (!this.currentBGM || !this.currentBGM.isPlaying || this.isMuted) return;

            // Simple bass pattern
            const bassNotes = name === 'game' ?
                [110, 110, 138, 110, 165, 110, 138, 110] :
                [130, 130, 165, 130, 196, 130, 165, 130];

            bassNotes.forEach((freq, index) => {
                setTimeout(() => {
                    if (this.currentBGM && this.currentBGM.isPlaying && !this.isMuted) {
                        this.playTone(freq, 0.15, 'sine', 0.15);
                    }
                }, index * 200);
            });
        };

        playPattern();
        this.currentBGM.intervalId = setInterval(playPattern, 1600);
    }

    stopBGM() {
        if (this.currentBGM) {
            this.currentBGM.isPlaying = false;
            if (this.currentBGM.intervalId) {
                clearInterval(this.currentBGM.intervalId);
            }
            this.currentBGM = null;
        }
    }

    pauseBGM() {
        if (this.currentBGM) {
            this.currentBGM.isPlaying = false;
        }
    }

    resumeBGM() {
        if (this.currentBGM) {
            this.currentBGM.isPlaying = true;
        }
    }

    toggleMute() {
        this.isMuted = !this.isMuted;

        if (this.isMuted) {
            this.pauseBGM();
        } else {
            this.resumeBGM();
        }

        return this.isMuted;
    }

    setBGMVolume(volume) {
        this.bgmVolume = Utils.clamp(volume, 0, 1);
        if (this.bgmGain) {
            this.bgmGain.gain.value = this.bgmVolume;
        }
    }

    setSFXVolume(volume) {
        this.sfxVolume = Utils.clamp(volume, 0, 1);
    }
}

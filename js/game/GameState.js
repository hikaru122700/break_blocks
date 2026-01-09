class GameState {
    constructor() {
        this.currentState = GAME_STATE.LOADING;
        this.previousState = null;
        this.listeners = [];
    }

    setState(newState) {
        if (this.currentState !== newState) {
            this.previousState = this.currentState;
            this.currentState = newState;
            this.notifyListeners();
        }
    }

    getState() {
        return this.currentState;
    }

    getPreviousState() {
        return this.previousState;
    }

    is(state) {
        return this.currentState === state;
    }

    isPlaying() {
        return this.currentState === GAME_STATE.PLAYING;
    }

    isPaused() {
        return this.currentState === GAME_STATE.PAUSED;
    }

    isMenu() {
        return this.currentState === GAME_STATE.MENU;
    }

    isGameOver() {
        return this.currentState === GAME_STATE.GAME_OVER;
    }

    addListener(callback) {
        this.listeners.push(callback);
    }

    removeListener(callback) {
        this.listeners = this.listeners.filter(cb => cb !== callback);
    }

    notifyListeners() {
        for (const listener of this.listeners) {
            listener(this.currentState, this.previousState);
        }
    }
}

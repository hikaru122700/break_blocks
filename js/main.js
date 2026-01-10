// Entry point
let game = null;

window.addEventListener('load', async () => {
    const canvas = document.getElementById('gameCanvas');

    if (!canvas) {
        console.error('Canvas element not found');
        return;
    }

    // Update loading progress
    const loadingBarFill = document.getElementById('loadingBarFill');
    if (loadingBarFill) {
        loadingBarFill.style.width = '30%';
    }

    // Create game instance
    game = new Game(canvas);

    if (loadingBarFill) {
        loadingBarFill.style.width = '60%';
    }

    // Initialize game
    await game.init();

    if (loadingBarFill) {
        loadingBarFill.style.width = '100%';
    }

    console.log('Block Breaker initialized successfully!');
});

// Handle visibility change (pause when tab is not visible)
document.addEventListener('visibilitychange', () => {
    if (game && document.hidden && game.state.isPlaying()) {
        game.pause();
    }
});

// Prevent accidental page close during gameplay
window.addEventListener('beforeunload', (e) => {
    if (game && game.state.isPlaying() && game.scoreSystem.score > 0) {
        e.preventDefault();
        e.returnValue = '';
    }
});

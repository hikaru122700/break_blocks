class InputManager {
    constructor(canvas) {
        this.canvas = canvas;
        this.mouseX = CANVAS_WIDTH / 2;
        this.mouseY = CANVAS_HEIGHT / 2;
        this.isMouseDown = false;
        this.keys = {};
        this.keysPressed = {}; // For single press detection

        this.setupEventListeners();
    }

    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        this.canvas.addEventListener('mouseup', (e) => this.handleMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handleMouseUp(e));

        // Prevent context menu on right click
        this.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Keyboard events
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
        document.addEventListener('keyup', (e) => this.handleKeyUp(e));

        // Prevent scrolling with arrow keys
        window.addEventListener('keydown', (e) => {
            if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space'].includes(e.code)) {
                e.preventDefault();
            }
        });
    }

    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        this.mouseX = (e.clientX - rect.left) * scaleX;
        this.mouseY = (e.clientY - rect.top) * scaleY;
    }

    handleMouseDown(e) {
        this.isMouseDown = true;
    }

    handleMouseUp(e) {
        this.isMouseDown = false;
    }

    handleKeyDown(e) {
        if (!this.keys[e.code]) {
            this.keysPressed[e.code] = true;
        }
        this.keys[e.code] = true;
    }

    handleKeyUp(e) {
        this.keys[e.code] = false;
    }

    // Check if key is currently held down
    isKeyDown(code) {
        return this.keys[code] === true;
    }

    // Check if key was just pressed (single press)
    isKeyPressed(code) {
        if (this.keysPressed[code]) {
            this.keysPressed[code] = false;
            return true;
        }
        return false;
    }

    // Get mouse position
    getMousePosition() {
        return { x: this.mouseX, y: this.mouseY };
    }

    // Reset pressed keys (call at end of frame)
    clearPressedKeys() {
        this.keysPressed = {};
    }

    // Check for common game inputs
    isLaunchPressed() {
        return this.isKeyPressed('Space') || this.isMouseDown;
    }

    isPausePressed() {
        return this.isKeyPressed('Escape') || this.isKeyPressed('KeyP');
    }

    isLeftDown() {
        return this.isKeyDown('ArrowLeft') || this.isKeyDown('KeyA');
    }

    isRightDown() {
        return this.isKeyDown('ArrowRight') || this.isKeyDown('KeyD');
    }

    isConfirmPressed() {
        return this.isKeyPressed('Enter') || this.isKeyPressed('Space');
    }

    isBackPressed() {
        return this.isKeyPressed('Escape') || this.isKeyPressed('Backspace');
    }
}

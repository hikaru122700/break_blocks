// Utility functions
const Utils = {
    // Clamp value between min and max
    clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    },

    // Linear interpolation
    lerp(start, end, t) {
        return start + (end - start) * t;
    },

    // Random number between min and max
    random(min, max) {
        return Math.random() * (max - min) + min;
    },

    // Random integer between min and max (inclusive)
    randomInt(min, max) {
        return Math.floor(Math.random() * (max - min + 1)) + min;
    },

    // Distance between two points
    distance(x1, y1, x2, y2) {
        const dx = x2 - x1;
        const dy = y2 - y1;
        return Math.sqrt(dx * dx + dy * dy);
    },

    // Normalize vector
    normalize(x, y) {
        const length = Math.sqrt(x * x + y * y);
        if (length === 0) return { x: 0, y: 0 };
        return { x: x / length, y: y / length };
    },

    // Draw neon rectangle with glow effect
    drawNeonRect(ctx, x, y, width, height, color, glowSize = 15) {
        ctx.save();

        // Outer glow
        ctx.shadowBlur = glowSize;
        ctx.shadowColor = color;

        ctx.fillStyle = color;
        ctx.fillRect(x, y, width, height);

        // Inner highlight
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.fillRect(x + 2, y + 2, width - 4, Math.min(4, height / 3));

        // Border
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, width, height);

        ctx.restore();
    },

    // Draw neon circle with glow effect
    drawNeonCircle(ctx, x, y, radius, color, glowSize = 15) {
        ctx.save();

        // Outer glow
        ctx.shadowBlur = glowSize;
        ctx.shadowColor = color;

        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();

        // Inner highlight
        ctx.shadowBlur = 0;
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.beginPath();
        ctx.arc(x - radius * 0.3, y - radius * 0.3, radius * 0.3, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    },

    // Draw pixel-style text
    drawPixelText(ctx, text, x, y, size = 16, color = '#ffffff', align = 'center') {
        ctx.save();
        ctx.font = `${size}px "Press Start 2P", monospace`;
        ctx.textAlign = align;
        ctx.textBaseline = 'middle';

        // Shadow
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillText(text, x + 2, y + 2);

        // Main text
        ctx.fillStyle = color;
        ctx.fillText(text, x, y);

        ctx.restore();
    },

    // Draw grid background
    drawGrid(ctx, width, height, gridSize = 40, color = COLORS.GRID) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;

        // Vertical lines
        for (let x = 0; x <= width; x += gridSize) {
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, height);
            ctx.stroke();
        }

        // Horizontal lines
        for (let y = 0; y <= height; y += gridSize) {
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }

        ctx.restore();
    },

    // Format number with commas
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    // Weighted random selection
    weightedRandom(weights) {
        const entries = Object.entries(weights);
        const totalWeight = entries.reduce((sum, [, weight]) => sum + weight, 0);
        let random = Math.random() * totalWeight;

        for (const [key, weight] of entries) {
            random -= weight;
            if (random <= 0) {
                return key;
            }
        }
        return entries[0][0];
    }
};

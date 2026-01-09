class ParticleSystem {
    constructor() {
        this.particles = [];
    }

    emit(x, y, color, count = 10) {
        for (let i = 0; i < count; i++) {
            const angle = (Math.PI * 2 * i) / count + Utils.random(-0.3, 0.3);
            const speed = Utils.random(2, 6);

            this.particles.push({
                x,
                y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                color,
                size: Utils.random(3, 8),
                alpha: 1,
                life: Utils.random(0.5, 1),
                decay: Utils.random(0.02, 0.04),
                gravity: 0.1,
                rotation: Utils.random(0, Math.PI * 2),
                rotationSpeed: Utils.random(-0.2, 0.2)
            });
        }
    }

    emitSpark(x, y, color, count = 5) {
        for (let i = 0; i < count; i++) {
            const angle = Utils.random(-Math.PI, Math.PI);
            const speed = Utils.random(3, 8);

            this.particles.push({
                x,
                y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed - 2,
                color,
                size: Utils.random(2, 4),
                alpha: 1,
                life: 1,
                decay: Utils.random(0.03, 0.06),
                gravity: 0.15,
                isSpark: true
            });
        }
    }

    update(deltaTime) {
        const dt = deltaTime / 16.67;

        for (const particle of this.particles) {
            // Update position
            particle.x += particle.vx * dt;
            particle.y += particle.vy * dt;

            // Apply gravity
            particle.vy += particle.gravity * dt;

            // Decay
            particle.alpha -= particle.decay * dt;
            particle.life -= particle.decay * dt;

            // Rotation
            if (particle.rotation !== undefined) {
                particle.rotation += particle.rotationSpeed * dt;
            }

            // Shrink
            particle.size *= 0.98;
        }

        // Remove dead particles
        this.particles = this.particles.filter(p => p.alpha > 0 && p.life > 0);
    }

    draw(ctx) {
        ctx.save();

        for (const particle of this.particles) {
            ctx.globalAlpha = particle.alpha;

            if (particle.isSpark) {
                // Draw spark as line
                ctx.strokeStyle = particle.color;
                ctx.lineWidth = particle.size;
                ctx.shadowBlur = 10;
                ctx.shadowColor = particle.color;
                ctx.beginPath();
                ctx.moveTo(particle.x, particle.y);
                ctx.lineTo(particle.x - particle.vx * 2, particle.y - particle.vy * 2);
                ctx.stroke();
            } else {
                // Draw as rotated square (pixel style)
                ctx.fillStyle = particle.color;
                ctx.shadowBlur = 5;
                ctx.shadowColor = particle.color;

                ctx.save();
                ctx.translate(particle.x, particle.y);
                if (particle.rotation !== undefined) {
                    ctx.rotate(particle.rotation);
                }
                ctx.fillRect(-particle.size / 2, -particle.size / 2, particle.size, particle.size);
                ctx.restore();
            }
        }

        ctx.restore();
    }

    clear() {
        this.particles = [];
    }

    getParticleCount() {
        return this.particles.length;
    }
}

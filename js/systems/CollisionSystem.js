class CollisionSystem {
    // Circle-Rectangle intersection (ball vs block/paddle)
    circleRectIntersect(circle, rect) {
        // Find the closest point on the rectangle to the circle
        const closestX = Utils.clamp(circle.x, rect.x, rect.x + rect.width);
        const closestY = Utils.clamp(circle.y, rect.y, rect.y + rect.height);

        // Calculate distance between circle center and closest point
        const distanceX = circle.x - closestX;
        const distanceY = circle.y - closestY;
        const distanceSquared = distanceX * distanceX + distanceY * distanceY;

        return distanceSquared < circle.radius * circle.radius;
    }

    // Rectangle-Rectangle intersection (AABB)
    rectIntersect(rect1, rect2) {
        return rect1.x < rect2.x + rect2.width &&
               rect1.x + rect1.width > rect2.x &&
               rect1.y < rect2.y + rect2.height &&
               rect1.y + rect1.height > rect2.y;
    }

    // Check ball vs wall collision
    checkBallWall(ball, canvasWidth, canvasHeight) {
        const result = { hit: false, normal: { x: 0, y: 0 }, type: null };

        // Left wall
        if (ball.x - ball.radius <= 0) {
            result.hit = true;
            result.normal = { x: 1, y: 0 };
            result.type = 'left';
            ball.x = ball.radius;
        }
        // Right wall
        else if (ball.x + ball.radius >= canvasWidth) {
            result.hit = true;
            result.normal = { x: -1, y: 0 };
            result.type = 'right';
            ball.x = canvasWidth - ball.radius;
        }

        // Top wall
        if (ball.y - ball.radius <= 0) {
            result.hit = true;
            result.normal = { x: 0, y: 1 };
            result.type = 'top';
            ball.y = ball.radius;
        }

        return result;
    }

    // Check ball vs paddle collision
    checkBallPaddle(ball, paddle) {
        // Only check if ball is moving downward
        if (ball.velocity.y <= 0) return false;

        const paddleBox = paddle.getBoundingBox();
        const collision = this.circleRectIntersect(
            { x: ball.x, y: ball.y, radius: ball.radius },
            paddleBox
        );

        return collision;
    }

    // Check ball vs block collision
    checkBallBlock(ball, block) {
        if (block.isDestroyed) return false;

        return this.circleRectIntersect(
            { x: ball.x, y: ball.y, radius: ball.radius },
            { x: block.x, y: block.y, width: block.width, height: block.height }
        );
    }

    // Get collision normal for ball vs block (determines bounce direction)
    getBlockCollisionNormal(ball, block) {
        const ballCenterX = ball.x;
        const ballCenterY = ball.y;
        const blockCenterX = block.x + block.width / 2;
        const blockCenterY = block.y + block.height / 2;

        // Relative position
        const dx = ballCenterX - blockCenterX;
        const dy = ballCenterY - blockCenterY;

        // Half sizes including ball radius
        const halfWidth = block.width / 2 + ball.radius;
        const halfHeight = block.height / 2 + ball.radius;

        // Calculate overlap on each axis
        const overlapX = halfWidth - Math.abs(dx);
        const overlapY = halfHeight - Math.abs(dy);

        // Return normal based on smallest overlap (most likely collision surface)
        if (overlapX < overlapY) {
            // Horizontal collision (left or right of block)
            return { x: dx > 0 ? 1 : -1, y: 0 };
        } else {
            // Vertical collision (top or bottom of block)
            return { x: 0, y: dy > 0 ? 1 : -1 };
        }
    }

    // Check power-up vs paddle collision
    checkPowerUpPaddle(powerUp, paddle) {
        const paddleBox = paddle.getBoundingBox();
        const powerUpBox = {
            x: powerUp.x - powerUp.width / 2,
            y: powerUp.y,
            width: powerUp.width,
            height: powerUp.height
        };

        return this.rectIntersect(powerUpBox, paddleBox);
    }

    // Advanced: Get exact collision point (for effects)
    getCollisionPoint(ball, rect) {
        const closestX = Utils.clamp(ball.x, rect.x, rect.x + rect.width);
        const closestY = Utils.clamp(ball.y, rect.y, rect.y + rect.height);
        return { x: closestX, y: closestY };
    }
}

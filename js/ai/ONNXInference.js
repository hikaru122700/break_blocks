/**
 * ONNX.js inference wrapper for RL agent.
 * Uses ONNX Runtime Web for browser-based neural network inference.
 */
class ONNXInference {
    constructor() {
        this.session = null;
        this.isLoaded = false;
        this.modelPath = 'models/onnx/breakout_agent.onnx';
    }

    /**
     * Load the ONNX model.
     * @returns {Promise<boolean>} True if loaded successfully
     */
    async load() {
        try {
            console.log('[ONNX] Loading model from', this.modelPath);

            // Check if ONNX Runtime is available
            if (typeof ort === 'undefined') {
                console.error('[ONNX] ONNX Runtime not loaded. Add ort.min.js to your page.');
                return false;
            }

            // Create inference session
            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });

            this.isLoaded = true;
            console.log('[ONNX] Model loaded successfully');
            console.log('[ONNX] Input names:', this.session.inputNames);
            console.log('[ONNX] Output names:', this.session.outputNames);

            return true;
        } catch (error) {
            console.error('[ONNX] Failed to load model:', error);
            this.isLoaded = false;
            return false;
        }
    }

    /**
     * Run inference on observation.
     * @param {Float32Array} observation - 215-dimensional observation vector
     * @returns {Promise<Float32Array>} Action probabilities (3 values)
     */
    async predict(observation) {
        if (!this.isLoaded || !this.session) {
            throw new Error('Model not loaded');
        }

        // Create input tensor
        const inputTensor = new ort.Tensor('float32', observation, [1, observation.length]);

        // Run inference
        const feeds = { observation: inputTensor };
        const results = await this.session.run(feeds);

        // Get action probabilities
        const actionProbs = results.action_probs.data;
        return new Float32Array(actionProbs);
    }

    /**
     * Get the best action (greedy).
     * @param {Float32Array} observation - Observation vector
     * @returns {Promise<number>} Action index (0=left, 1=stay, 2=right)
     */
    async getAction(observation) {
        const probs = await this.predict(observation);

        // Argmax
        let maxIdx = 0;
        let maxProb = probs[0];
        for (let i = 1; i < probs.length; i++) {
            if (probs[i] > maxProb) {
                maxProb = probs[i];
                maxIdx = i;
            }
        }

        return maxIdx;
    }

    /**
     * Sample action from probability distribution.
     * @param {Float32Array} observation - Observation vector
     * @returns {Promise<number>} Sampled action index
     */
    async sampleAction(observation) {
        const probs = await this.predict(observation);

        // Sample from distribution
        const rand = Math.random();
        let cumulative = 0;
        for (let i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (rand < cumulative) {
                return i;
            }
        }

        return probs.length - 1;
    }

    /**
     * Check if model is ready for inference.
     * @returns {boolean}
     */
    isReady() {
        return this.isLoaded && this.session !== null;
    }

    /**
     * Dispose of the session.
     */
    dispose() {
        if (this.session) {
            this.session = null;
        }
        this.isLoaded = false;
    }
}

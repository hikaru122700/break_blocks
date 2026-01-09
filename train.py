#!/usr/bin/env python3
"""Entry point for Break Blocks RL training."""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_training.training.train import main

if __name__ == '__main__':
    main()

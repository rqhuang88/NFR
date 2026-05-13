#!/bin/bash
# Train Partial-DFR feature extractor on S&F dataset
cd "$(dirname "$0")/../train" || exit 1
python train_partial_dfr.py --config train_partial_sf

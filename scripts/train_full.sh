#!/bin/bash
# Train Full DFR feature extractor on S&F dataset
cd "$(dirname "$0")/../train" || exit 1
python train_full_dfr.py --config train_full_sf

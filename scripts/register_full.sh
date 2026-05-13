#!/bin/bash
# Run full-to-full registration
cd "$(dirname "$0")/../registration" || exit 1
python test_full_register.py --config full_register

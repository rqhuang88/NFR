#!/bin/bash
# Run partial-to-full registration
cd "$(dirname "$0")/../registration" || exit 1
python test_partial_register.py --config partial_register

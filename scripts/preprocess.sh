#!/bin/bash
# Preprocess dataset: compute partial views
cd "$(dirname "$0")/../registration" || exit 1
python dataset_preprocess.py --data_dir ../data/faust_raw/ --save_path ../results/data --save_name faust_raw

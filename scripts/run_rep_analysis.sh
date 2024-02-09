#!/bin/bash

declare -a seeds=(111 222 333 444 555 666 777 888 999 1709)

for seed in ${seeds[@]}
do
    
    python src/pca_analysis.py --results_file data/results/childes/smolm-autoreg-bpe-seed_${seed}/cat_abs_results.csv
done
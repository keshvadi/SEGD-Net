#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 {breakhis|bach} DATA_ROOT CONFIG [OUTPUT_DIR]" >&2
    exit 2
fi

dataset="$1"
data_root="$2"
config="$3"
output_dir="${4:-runs/${dataset}}"

python scripts/train_eval_ssl.py \
    --dataset "$dataset" --data-root "$data_root" --config "$config" \
    --output "$output_dir/main"
python scripts/run_noise_sweep.py \
    --dataset "$dataset" --data-root "$data_root" --config "$config" \
    --output "$output_dir/noise"
python scripts/run_ablation.py \
    --dataset "$dataset" --data-root "$data_root" --config "$config" \
    --output "$output_dir/ablation"
python scripts/generate_xai_fig8.py --output ../images/Fig8.png

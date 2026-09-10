#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SEQUENCES_CSV="${1:-Input/sequences.csv}"
OUTPUT_DIR="${2:-predictions}"

python scripts/predict.py \
  --modeldir results \
  --seqs_csv "$SEQUENCES_CSV" \
  --carbons_smiles outputs_smiles/carbons_smiles.csv \
  --outdir "$OUTPUT_DIR" \
  --esm_cache_dir results/esm_cache_compare \
  --esm_device cuda \
  --esm_model_name esm2_t6_8M_UR50D \
  --esm_layer 6 \
  --esm_pooling mean \
  --esm_batch_size 2 \
  --esm_max_len 512 \
  --esm_pca_dim 11 \
  --fp_pca_dim 13

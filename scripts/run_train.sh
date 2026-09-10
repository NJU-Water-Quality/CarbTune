#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -e results/esm_cache_compare ]]; then
  echo "results/esm_cache_compare already exists. Use an empty output path for a fresh ESM run." >&2
  exit 2
fi

python scripts/validate_inputs.py --report input_validation.json
mkdir -p results

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/CarbTune.py \
  --curves Input/curves_template.csv \
  --seqs Input/sequences.csv \
  --carbons_smiles outputs_smiles/carbons_smiles.csv \
  --outdir results \
  --growth_threshold 1.0 \
  --plateau_tol_frac 0.02 \
  --plateau_abs_min 0.05 \
  --plateau_value_mode median \
  --pauc_u_max 20.0 \
  --aa_pca_dim 11 \
  --fp_bits 2048 \
  --fp_radius 2 \
  --fp_pca_dim 13 \
  --esm_on \
  --esm_device cuda \
  --esm_model_name esm2_t6_8M_UR50D \
  --esm_layer 6 \
  --esm_pooling mean \
  --esm_batch_size 2 \
  --esm_max_len 512 \
  --esm_pca_dim 11 \
  --esm_cache_dir results/esm_cache_compare \
  --cv_splits 5

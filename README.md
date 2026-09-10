# CarbTune

The water treatment industry is facing increasing pressure to remove emerging organic pollutants, but popular biological methods are still limited by heuristic external carbon additions that ignore competition for microbial resources. This restriction hinders precise enrichment in pollutant degradation communities and limits the efficiency of biological purification. Here, we introduce CarbTune, a mechanical artificial intelligence framework that integrates genome-scale metabolic models, protein language model embeddings, and receptor molecular descriptors to calculate pUC-a low-receptor, flux-based competition phenotype formally related to Monod-R* resource competition theory-and the carbon competition overlap index for 45 carbon sources. By embedding the principles of competition for ecological resources into an interpretable deep learning workflow, this work redefines carbon management strategies in biological wastewater treatment and provides a scalable, economically beneficial way to robust control of emerging pollutants. 

## Model inputs

Each strain–carbon pair is represented by 44 features.

| Feature block | Construction | Dimensions |
|---|---|---:|
| Amino-acid composition | Proteome-wide frequencies of 20 canonical amino acids, standardized and reduced by PCA | 11 |
| ESM-2 | `esm2_t6_8M_UR50D`, layer 6, residue mean pooling per protein, mean aggregation across the proteome, standardization and PCA | 11 |
| Molecular descriptors | MolWt, LogP, TPSA, HBD, HBA, rotatable bonds, ring count, heavy-atom count and fraction Csp3 | 9 |
| Morgan fingerprint | Radius 2, 2,048 bits, standardized and reduced by PCA | 13 |

Protein sequences longer than 512 amino acids are truncated to 512 residues for ESM-2 encoding. Protein-length and other proteome summary statistics are excluded. No missingness indicator is used.

The classifier label is determined from the plateau growth rate using a default threshold of 1.0. pAUCg is calculated up to an uptake rate of 20 and is learned only from growth-positive pairs. Both stages use the same 44-dimensional feature matrix.

## Repository structure

```text
CarbTune/
├── Data/                         protein FASTA files for 12 strains
├── Input/
│   ├── curves_template.csv       strain–carbon growth curves
│   └── sequences.csv             strain-to-FASTA mapping
├── outputs_smiles/
│   ├── carbons_smiles.csv        canonical carbon-source structures
│   └── carbons_smiles_manual_template.csv
├── results/                      trained models and evaluation outputs
├── scripts/
│   ├── CarbTune.py               XGBoost training and evaluation
│   ├── feature_functions.py      shared feature and label functions
│   ├── predict.py                inference for protein FASTA inputs
│   ├── make_sequences_from_folder.py
│   ├── validate_inputs.py
│   └── validate_normalize_smiles.py
├── DATA_DICTIONARY.md
├── DATA_MANIFEST.csv
├── environment.yml
└── requirements.txt
```

## Installation

The recommended GPU environment is:

```bash
conda env create -f environment.yml
conda activate carbtune
```

Alternatively, install the Python packages into an existing CUDA-enabled environment:

```bash
python -m pip install -r requirements.txt
```

The first ESM-2 run downloads the pretrained model weights. CUDA is recommended for proteome-scale embedding.

## Input validation

Run from the repository root:

```bash
python scripts/validate_inputs.py --report input_validation.json
```

The validator checks required columns, strain and carbon identifiers, FASTA paths, grid coverage, duplicate curve records, sampling-point counts and uptake limits. Add `--strict` to return a nonzero exit code for warnings.

## Training

For a fresh run, use an empty `results/esm_cache_compare` path. The supplied shell script performs input validation before training:

```bash
bash scripts/run_train.sh
```

Equivalent command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/CarbTune.py \
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
```

The classifier uses pair-level fivefold stratified cross-validation. The regressor uses pair-level fivefold shuffled cross-validation among growth-positive pairs. A final classifier and regressor are then fitted on all eligible data.

The run writes out-of-fold predictions, metrics, plots, fitted PCA objects, trained models, the exact command-line configuration, and installed software versions to `results/`.

## Prediction

Prepare a CSV with columns `strain`, `sequence`, and `fasta_path`. Either an inline protein sequence or a FASTA path can be supplied for each strain. Relative FASTA paths are resolved from the directory in which the command is run.

```bash
python scripts/predict.py \
  --modeldir results \
  --seqs_csv Input/sequences.csv \
  --carbons_smiles outputs_smiles/carbons_smiles.csv \
  --outdir predictions \
  --esm_cache_dir results/esm_cache_compare \
  --esm_device cuda \
  --esm_model_name esm2_t6_8M_UR50D \
  --esm_layer 6 \
  --esm_pooling mean \
  --esm_batch_size 2 \
  --esm_max_len 512 \
  --esm_pca_dim 11 \
  --fp_pca_dim 13
```

Prediction reuses the AA-composition, ESM-2 and Morgan-fingerprint PCA transformations fitted during training. The main output is `predictions/real_predictions.csv`.

## Reproducibility metadata

`DATA_MANIFEST.csv` records file sizes, record counts and SHA-256 checksums for the distributed inputs. Each training run additionally creates `run_config.json` and `software_versions.json`. The random seed is fixed at 42 for PCA, cross-validation and both XGBoost models.

## License

The source code is released under the MIT License. Protein sequence and chemical database records remain subject to the terms of their source databases.

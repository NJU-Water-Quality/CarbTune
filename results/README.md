# Training outputs

Run `bash scripts/run_train.sh` from the repository root. The training process writes the following principal artifacts to this directory:

- `clf_XGB_Clf.pkl` and `reg_XGB_Reg.pkl`: final two-stage XGBoost models.
- `clf_XGB_Clf_oof.csv` and `reg_XGB_Reg_oof.csv`: out-of-fold predictions.
- `clf_metrics.json` and `reg_metrics.json`: cross-validation metrics.
- `aa_pca_pipe.joblib`, `fp_pca_pipe.joblib`, and `esm_cache_compare/esm_pca_pipe.joblib`: fitted feature transformations.
- `strain_features_used.csv`, `carbon_features_used.csv`, and `training_pairs.csv`: derived model inputs.
- `feature_contract.json`: ordered 44-feature model contract.
- `run_config.json` and `software_versions.json`: exact run configuration and installed package versions.

The ESM embedding cache is excluded from Git by default because it can be regenerated and may be large.

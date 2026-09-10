#!/usr/bin/env python

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from feature_functions import (
    AA,
    Chem,
    STRUCT_DESC_COLS,
    aa_features_aggregate,
    build_carbon_struct_features,
    build_seq_df_from_sequences_csv,
    esm_embed_dataframe,
    norm_text,
)


AA_COMPOSITION_COLS = [f"aa_{amino_acid}" for amino_acid in AA]


def read_fasta(path):
    sequences = []
    current = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append(re.sub(r"[^A-Za-z]", "", "".join(current)).upper())
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append(re.sub(r"[^A-Za-z]", "", "".join(current)).upper())
    return [sequence for sequence in sequences if sequence]


def load_sequences(seqs_csv, seqs_dir):
    if seqs_csv:
        path = Path(seqs_csv)
        if not path.exists():
            raise FileNotFoundError(f"Sequence table not found: {path}")
        return build_seq_df_from_sequences_csv(path)
    directory = Path(seqs_dir)
    if not directory.exists():
        raise FileNotFoundError(f"Sequence directory not found: {directory}")
    rows = []
    extensions = {".faa", ".fasta", ".fa", ".fna", ".fas"}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        for sequence in read_fasta(path):
            rows.append({"strain": path.stem, "sequence": sequence})
    if not rows:
        raise RuntimeError(f"No protein sequences found in: {directory}")
    return pd.DataFrame(rows)


def required_columns(model):
    columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        try:
            columns = model.named_steps["pre"].transformers_[0][2]
        except Exception as error:
            raise RuntimeError("The model does not expose its training feature columns") from error
    return [str(column) for column in columns]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", default="results")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seqs_csv", default="")
    source.add_argument("--seqs_dir", default="")
    parser.add_argument("--carbons_smiles", default="outputs_smiles/carbons_smiles.csv")
    parser.add_argument("--outdir", default="predictions")
    parser.add_argument("--esm_cache_dir", default="")
    parser.add_argument("--esm_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--esm_model_name", default="esm2_t6_8M_UR50D")
    parser.add_argument("--esm_layer", type=int, default=6)
    parser.add_argument("--esm_pooling", choices=["mean", "median"], default="mean")
    parser.add_argument("--esm_batch_size", type=int, default=2)
    parser.add_argument("--esm_max_len", type=int, default=512)
    parser.add_argument("--esm_pca_dim", type=int, default=11)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_pca_dim", type=int, default=13)
    args = parser.parse_args()

    modeldir = Path(args.modeldir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    artifact_paths = {
        "classifier": modeldir / "clf_XGB_Clf.pkl",
        "regressor": modeldir / "reg_XGB_Reg.pkl",
        "aa_pca": modeldir / "aa_pca_pipe.joblib",
        "fp_pca": modeldir / "fp_pca_pipe.joblib",
    }
    missing_artifacts = [str(path) for path in artifact_paths.values() if not path.exists()]
    if missing_artifacts:
        raise FileNotFoundError("Missing training artifacts: " + ", ".join(missing_artifacts))

    if args.esm_cache_dir:
        esm_cache = Path(args.esm_cache_dir)
    else:
        candidates = [modeldir / "esm_cache_compare", modeldir / "esm_cache"]
        esm_cache = next((path for path in candidates if path.exists()), candidates[0])
    if not (esm_cache / "esm_pca_pipe.joblib").exists():
        raise FileNotFoundError(f"Training ESM PCA pipeline not found: {esm_cache / 'esm_pca_pipe.joblib'}")

    classifier = joblib.load(artifact_paths["classifier"])
    regressor = joblib.load(artifact_paths["regressor"])
    classifier_columns = required_columns(classifier)
    regressor_columns = required_columns(regressor)
    if classifier_columns != regressor_columns:
        raise RuntimeError("Classifier and regressor feature contracts differ")

    raw_sequences = load_sequences(args.seqs_csv, args.seqs_dir)
    raw_sequences["strain"] = raw_sequences["strain"].apply(norm_text)
    amino_acid_rows = []
    for strain, group in raw_sequences.groupby("strain", sort=False):
        row = aa_features_aggregate(group["sequence"].astype(str).tolist())
        row["strain"] = strain
        amino_acid_rows.append(row)
    amino_acid = pd.DataFrame(amino_acid_rows)
    aa_pipe = joblib.load(artifact_paths["aa_pca"])
    aa_values = aa_pipe.transform(amino_acid[AA_COMPOSITION_COLS].astype(float))
    aa_features = pd.DataFrame(aa_values, columns=[f"aa_pca_{index}" for index in range(aa_values.shape[1])])
    aa_features.insert(0, "strain", amino_acid["strain"].to_numpy())

    esm_features = esm_embed_dataframe(
        raw_sequences[["strain", "sequence"]].copy(),
        out_cache=esm_cache,
        model_name=args.esm_model_name,
        layer=args.esm_layer,
        pooling=args.esm_pooling,
        batch_size=args.esm_batch_size,
        device=args.esm_device,
        max_len=args.esm_max_len,
        pca_dim=args.esm_pca_dim,
    )
    strain_features = aa_features.merge(esm_features, on="strain", how="inner")
    expected_strains = raw_sequences["strain"].nunique()
    if strain_features["strain"].nunique() != expected_strains:
        raise RuntimeError("AA-composition and ESM-2 features do not cover all input strains")

    if Chem is None:
        raise RuntimeError("RDKit is required")
    smiles = pd.read_csv(args.carbons_smiles)
    if not {"carbon", "smiles"}.issubset(smiles.columns):
        raise ValueError("Carbon table must contain carbon and smiles columns")
    smiles["carbon"] = smiles["carbon"].apply(norm_text)
    descriptors, fingerprints, report = build_carbon_struct_features(
        smiles,
        pca_dim=args.fp_pca_dim,
        fp_bits=args.fp_bits,
        radius=args.fp_radius,
        pca_pipe_path=artifact_paths["fp_pca"],
    )
    if report["n_fail"]:
        raise RuntimeError(f"RDKit failed to parse {report['n_fail']} carbon structures")
    carbon_features = descriptors.merge(fingerprints, on="carbon", how="inner")
    descriptor_columns = [column for column in STRUCT_DESC_COLS if column in carbon_features.columns]
    fingerprint_columns = sorted(
        [column for column in carbon_features.columns if column.startswith("fp_")],
        key=lambda value: int(value.rsplit("_", 1)[1]),
    )
    if len(descriptor_columns) != 9 or len(fingerprint_columns) != 13:
        raise RuntimeError("Carbon feature dimensions do not match the trained model")

    pairs = pd.MultiIndex.from_product(
        [sorted(strain_features["strain"].unique()), sorted(carbon_features["carbon"].unique())],
        names=["strain", "carbon"],
    ).to_frame(index=False)
    feature_table = pairs.merge(strain_features, on="strain", how="left").merge(carbon_features, on="carbon", how="left")
    missing_columns = [column for column in classifier_columns if column not in feature_table.columns]
    if missing_columns:
        raise RuntimeError("Missing model features: " + ", ".join(missing_columns))
    model_input = feature_table[classifier_columns].copy()

    probability = classifier.predict_proba(model_input)[:, 1].astype(float)
    growth_prediction = classifier.predict(model_input).astype(int)
    pauc_prediction = regressor.predict(model_input).astype(float)
    pauc_prediction = np.where(growth_prediction == 1, pauc_prediction, np.nan)

    predictions = pairs.copy()
    predictions["proba_grow_pred"] = probability
    predictions["pred_grow"] = growth_prediction
    predictions["pauc_g_pred"] = pauc_prediction
    predictions.to_csv(outdir / "real_predictions.csv", index=False, encoding="utf-8")
    summary = predictions.groupby("strain", as_index=False).agg(
        num_carbons=("carbon", "count"),
        num_pred_grow=("pred_grow", "sum"),
        mean_pauc_pred=("pauc_g_pred", "mean"),
        max_pauc_pred=("pauc_g_pred", "max"),
    )
    summary.to_csv(outdir / "real_summary_by_strain.csv", index=False, encoding="utf-8")
    strain_features.to_csv(outdir / "strain_features_used.csv", index=False, encoding="utf-8")
    carbon_features.to_csv(outdir / "carbon_features_used.csv", index=False, encoding="utf-8")
    metadata = {
        "modeldir": str(modeldir),
        "sequence_source": args.seqs_csv or args.seqs_dir,
        "num_strains": int(strain_features["strain"].nunique()),
        "num_carbons": int(carbon_features["carbon"].nunique()),
        "feature_dimensions": len(classifier_columns),
        "esm_cache_dir": str(esm_cache),
    }
    (outdir / "run_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Completed: {outdir / 'real_predictions.csv'}")


if __name__ == "__main__":
    main()

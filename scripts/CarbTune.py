import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, average_precision_score, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from feature_functions import (
    Chem,
    ESM_NAME_MAP,
    STRUCT_DESC_COLS,
    aa_features_aggregate,
    build_actual_labels_with_plateau,
    build_carbon_struct_features,
    build_seq_df_from_sequences_csv,
    esm_embed_dataframe,
    norm_text,
    normalize_columns,
    read_table_safely,
)


AA_COMPOSITION_COLS = [f"aa_{aa}" for aa in "ACDEFGHIKLMNPQRSTVWY"]


def compute_pauc_g(u: np.ndarray, g: np.ndarray, u_cap: float) -> float:
    u = np.asarray(u, dtype=float)
    g = np.asarray(g, dtype=float)
    mask = np.isfinite(u) & np.isfinite(g)
    u = u[mask]
    g = g[mask]
    if u.size < 2 or not np.isfinite(u_cap):
        return 0.0
    order = np.argsort(u)
    u = u[order]
    g = g[order]
    u_min = float(np.min(u))
    u_cap_eff = float(min(np.max(u), u_cap))
    span = u_cap_eff - u_min
    g_max = float(np.max(g))
    if not np.isfinite(span) or span <= 0 or not np.isfinite(g_max) or g_max <= 0:
        return 0.0
    keep = u <= u_cap_eff
    uu = u[keep]
    gg = g[keep]
    if uu.size < 2:
        return 0.0
    auc = float(np.trapz(gg, uu))
    denom = span * g_max
    if not np.isfinite(auc) or not np.isfinite(denom) or denom <= 0:
        return 0.0
    value = auc / denom
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.5))


def reduce_aa_composition(aa_df: pd.DataFrame, outdir: Path, pca_dim: int) -> pd.DataFrame:
    required = ["strain"] + AA_COMPOSITION_COLS
    missing = [c for c in required if c not in aa_df.columns]
    if missing:
        raise ValueError(f"AA feature table is missing columns: {missing}")
    if len(aa_df) < 2:
        raise RuntimeError("AA PCA requires at least two strains")
    requested_dim = int(pca_dim)
    if requested_dim < 1:
        raise ValueError("--aa_pca_dim must be at least 1")
    n_comp = min(requested_dim, len(AA_COMPOSITION_COLS), len(aa_df) - 1)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_comp, random_state=42)),
    ])
    transformed = pipe.fit_transform(aa_df[AA_COMPOSITION_COLS].astype(float))
    outdir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, outdir / "aa_pca_pipe.joblib")
    columns = [f"aa_pca_{i}" for i in range(n_comp)]
    meta = {
        "input_columns": AA_COMPOSITION_COLS,
        "output_columns": columns,
        "pca_dim_requested": requested_dim,
        "pca_dim_actual": n_comp,
        "n_training_strains": int(len(aa_df)),
    }
    (outdir / "aa_pca_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    reduced = pd.DataFrame(transformed, columns=columns)
    reduced.insert(0, "strain", aa_df["strain"].to_numpy())
    return reduced


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    return ColumnTransformer([("num", numeric, feature_cols)], remainder="drop")


def build_classifier(feature_cols: list[str], args) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("clf", XGBClassifier(
            n_estimators=args.xgb_clf_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=args.xgb_n_jobs,
        )),
    ])


def build_regressor(feature_cols: list[str], args) -> Pipeline:
    return Pipeline([
        ("pre", build_preprocessor(feature_cols)),
        ("reg", XGBRegressor(
            n_estimators=args.xgb_reg_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
            n_jobs=args.xgb_n_jobs,
        )),
    ])


def prepare_dataset(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw, _ = read_table_safely(Path(args.curves))
    curves = normalize_columns(raw)
    required = ["strain", "carbon", "uptake", "growth"]
    missing = [c for c in required if c not in curves.columns]
    if missing:
        raise ValueError(f"Curves table is missing columns: {missing}")
    curves["strain"] = curves["strain"].apply(norm_text)
    curves["carbon"] = curves["carbon"].apply(norm_text)
    curves["uptake"] = pd.to_numeric(curves["uptake"], errors="coerce")
    curves["growth"] = pd.to_numeric(curves["growth"], errors="coerce")
    curves = curves.dropna(subset=required)
    labels = build_actual_labels_with_plateau(
        curves,
        growth_thr=args.growth_threshold,
        tol_frac=args.plateau_tol_frac,
        tol_abs_min=args.plateau_abs_min,
        value_mode=args.plateau_value_mode,
    )
    pauc = (
        curves.groupby(["strain", "carbon"], sort=False)[["uptake", "growth"]]
        .apply(lambda g: compute_pauc_g(g["uptake"].to_numpy(), g["growth"].to_numpy(), args.pauc_u_max))
        .rename("pauc_g_actual")
        .reset_index()
    )
    labels = labels.merge(pauc, on=["strain", "carbon"], how="left")
    labels.to_csv(outdir / "labels_used.csv", index=False, encoding="utf-8")
    carbons = sorted(labels["carbon"].dropna().unique().tolist())
    (outdir / "carbons_from_curves.txt").write_text("\n".join(carbons), encoding="utf-8")
    seqs_path = Path(args.seqs)
    if not seqs_path.exists():
        raise FileNotFoundError(f"Sequence table not found: {seqs_path}")
    raw_seqs = build_seq_df_from_sequences_csv(seqs_path)
    strains = labels["strain"].drop_duplicates().tolist()
    raw_seqs = raw_seqs[raw_seqs["strain"].isin(strains)].copy()
    present = set(raw_seqs["strain"].dropna().unique())
    missing_strains = [s for s in strains if s not in present]
    if missing_strains:
        raise RuntimeError("No usable protein sequences were found for: " + ", ".join(missing_strains))
    aa_rows = []
    for strain, group in raw_seqs.groupby("strain", sort=False):
        row = aa_features_aggregate(group["sequence"].astype(str).tolist())
        row["strain"] = strain
        aa_rows.append(row)
    aa_reduced = reduce_aa_composition(pd.DataFrame(aa_rows), outdir, args.aa_pca_dim)
    if not args.esm_on:
        raise ValueError("--esm_on is required because XGB uses ESM features")
    cache_dir = Path(args.esm_cache_dir) if args.esm_cache_dir else outdir / "esm_cache"
    esm_features = esm_embed_dataframe(
        raw_seqs[["strain", "sequence"]].copy(),
        out_cache=cache_dir,
        model_name=args.esm_model_name,
        layer=args.esm_layer,
        pooling=args.esm_pooling,
        batch_size=args.esm_batch_size,
        device=args.esm_device,
        max_len=args.esm_max_len,
        pca_dim=args.esm_pca_dim,
    )
    if esm_features.empty:
        raise RuntimeError("ESM feature extraction returned no features")
    strain_features = aa_reduced.merge(esm_features, on="strain", how="inner")
    if strain_features["strain"].nunique() != len(strains):
        raise RuntimeError("AA and ESM features do not cover all training strains")
    strain_features.to_csv(outdir / "strain_features_used.csv", index=False, encoding="utf-8")
    if Chem is None:
        raise RuntimeError("RDKit is required")
    smiles = pd.read_csv(args.carbons_smiles)
    if not {"carbon", "smiles"}.issubset(smiles.columns):
        raise ValueError("Carbon table must contain carbon and smiles columns")
    smiles["carbon"] = smiles["carbon"].apply(norm_text)
    smiles = smiles[smiles["carbon"].isin(carbons)].copy()
    desc, fp, report = build_carbon_struct_features(
        smiles,
        pca_dim=args.fp_pca_dim,
        fp_bits=args.fp_bits,
        radius=args.fp_radius,
        pca_pipe_path=outdir / "fp_pca_pipe.joblib",
    )
    pd.DataFrame([report]).to_csv(outdir / "carbon_struct_report.csv", index=False, encoding="utf-8")
    carbon_features = desc.merge(fp, on="carbon", how="outer")
    carbon_features["carbon"] = carbon_features["carbon"].apply(norm_text)
    structural_cols = [c for c in carbon_features.columns if c in STRUCT_DESC_COLS or c.startswith("fp_")]
    for column in structural_cols:
        carbon_features[column] = carbon_features[column].fillna(0.0)
    carbon_features.to_csv(outdir / "carbon_features_used.csv", index=False, encoding="utf-8")
    train = labels.merge(strain_features, on="strain", how="left").merge(carbon_features, on="carbon", how="left")
    aa_cols = sorted([c for c in train.columns if c.startswith("aa_pca_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    esm_cols = sorted([c for c in train.columns if c.startswith("esm_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    fp_cols = sorted([c for c in train.columns if c.startswith("fp_")], key=lambda x: int(x.rsplit("_", 1)[1]))
    descriptor_cols = [c for c in STRUCT_DESC_COLS if c in train.columns]
    feature_cols = aa_cols + esm_cols + descriptor_cols + fp_cols
    expected = args.aa_pca_dim + args.esm_pca_dim + len(STRUCT_DESC_COLS) + args.fp_pca_dim
    if len(feature_cols) != expected:
        raise RuntimeError(f"Expected {expected} features but constructed {len(feature_cols)}")
    feature_contract = {
        "total_dimensions": len(feature_cols),
        "feature_columns": feature_cols,
        "blocks": {
            "amino_acid_composition_pca": aa_cols,
            "esm2_pca": esm_cols,
            "rdkit_descriptors": descriptor_cols,
            "morgan_fingerprint_pca": fp_cols,
        },
    }
    (outdir / "feature_contract.json").write_text(json.dumps(feature_contract, ensure_ascii=False, indent=2), encoding="utf-8")
    train.to_csv(outdir / "training_pairs.csv", index=False, encoding="utf-8")
    return train, feature_cols


def train_and_evaluate(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    package_names = {
        "numpy": ["numpy"],
        "pandas": ["pandas"],
        "scikit-learn": ["scikit-learn"],
        "joblib": ["joblib"],
        "matplotlib": ["matplotlib"],
        "xgboost": ["xgboost"],
        "rdkit": ["rdkit", "rdkit-pypi"],
        "fair-esm": ["fair-esm"],
        "torch": ["torch"],
    }
    versions = {"python": platform.python_version()}
    for label, candidates in package_names.items():
        versions[label] = None
        for package in candidates:
            try:
                versions[label] = importlib.metadata.version(package)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    run_config = vars(args).copy()
    run_config["command"] = " ".join(sys.argv)
    (outdir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "software_versions.json").write_text(json.dumps(versions, ensure_ascii=False, indent=2), encoding="utf-8")
    train, feature_cols = prepare_dataset(args)
    X = train[feature_cols].copy()
    y = train["can_grow_actual"].astype(int).to_numpy()
    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 2:
        raise RuntimeError("Growth classification requires two classes")
    clf_splits = min(args.cv_splits, int(np.min(counts)))
    if clf_splits < 2:
        raise RuntimeError("Insufficient samples for stratified classification CV")
    clf_oof = np.full(len(y), np.nan, dtype=float)
    clf_fold = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=clf_splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        model = build_classifier(feature_cols, args)
        model.fit(X.iloc[train_idx], y[train_idx])
        clf_oof[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        clf_fold[test_idx] = fold
    clf_pred = (clf_oof >= 0.5).astype(int)
    clf_metrics = {
        "XGB_Clf": {
            "accuracy": float(accuracy_score(y, clf_pred)),
            "auroc": float(roc_auc_score(y, clf_oof)),
            "average_precision": float(average_precision_score(y, clf_oof)),
            "cv_splits": clf_splits,
        }
    }
    (outdir / "clf_metrics.json").write_text(json.dumps(clf_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    clf_output = train[["strain", "carbon", "can_grow_actual"]].copy()
    clf_output["fold"] = clf_fold
    clf_output["proba_grow_oof"] = clf_oof
    clf_output["pred_grow_oof"] = clf_pred
    clf_output.to_csv(outdir / "clf_XGB_Clf_oof.csv", index=False, encoding="utf-8")
    fpr, tpr, _ = roc_curve(y, clf_oof)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(outdir / "clf_roc.svg", dpi=300)
    plt.close()
    classifier = build_classifier(feature_cols, args)
    classifier.fit(X, y)
    joblib.dump(classifier, outdir / "clf_XGB_Clf.pkl")
    positive = train[(train["can_grow_actual"] == 1) & train["pauc_g_actual"].notna()].copy().reset_index(drop=True)
    if len(positive) < 2:
        raise RuntimeError("Insufficient growth-positive samples for regression")
    X_reg = positive[feature_cols].copy()
    y_reg = positive["pauc_g_actual"].astype(float).to_numpy()
    reg_splits = min(args.cv_splits, len(positive))
    if reg_splits < 2:
        raise RuntimeError("Insufficient samples for regression CV")
    reg_oof = np.full(len(y_reg), np.nan, dtype=float)
    reg_fold = np.zeros(len(y_reg), dtype=int)
    kf = KFold(n_splits=reg_splits, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_reg), start=1):
        model = build_regressor(feature_cols, args)
        model.fit(X_reg.iloc[train_idx], y_reg[train_idx])
        reg_oof[test_idx] = model.predict(X_reg.iloc[test_idx])
        reg_fold[test_idx] = fold
    reg_metrics = {
        "XGB_Reg": {
            "mae": float(mean_absolute_error(y_reg, reg_oof)),
            "r2": float(r2_score(y_reg, reg_oof)),
            "cv_splits": reg_splits,
        }
    }
    (outdir / "reg_metrics.json").write_text(json.dumps(reg_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    reg_output = positive[["strain", "carbon", "pauc_g_actual"]].copy()
    reg_output["fold"] = reg_fold
    reg_output["pauc_g_pred_oof"] = reg_oof
    reg_output.to_csv(outdir / "reg_XGB_Reg_oof.csv", index=False, encoding="utf-8")
    bounds = [float(min(np.min(y_reg), np.min(reg_oof))), float(max(np.max(y_reg), np.max(reg_oof)))]
    plt.figure(figsize=(5, 5))
    plt.scatter(y_reg, reg_oof, s=12, alpha=0.65)
    plt.plot(bounds, bounds, "k--", linewidth=1)
    plt.xlabel("Observed pAUC")
    plt.ylabel("Predicted pAUC")
    plt.tight_layout()
    plt.savefig(outdir / "reg_scatter_aucg.svg", dpi=300)
    plt.close()
    regressor = build_regressor(feature_cols, args)
    regressor.fit(X_reg, y_reg)
    joblib.dump(regressor, outdir / "reg_XGB_Reg.pkl")
    print(f"Completed: {outdir}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="Input/curves_template.csv")
    parser.add_argument("--seqs", default="Input/sequences.csv")
    parser.add_argument("--carbons_smiles", default="outputs_smiles/carbons_smiles.csv")
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--growth_threshold", type=float, default=1.0)
    parser.add_argument("--plateau_tol_frac", type=float, default=0.02)
    parser.add_argument("--plateau_abs_min", type=float, default=0.05)
    parser.add_argument("--plateau_value_mode", choices=["median", "max"], default="median")
    parser.add_argument("--pauc_u_max", type=float, default=20.0)
    parser.add_argument("--aa_pca_dim", type=int, default=11)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--fp_pca_dim", type=int, default=13)
    parser.add_argument("--esm_on", action="store_true")
    parser.add_argument("--esm_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--esm_model_name", choices=list(ESM_NAME_MAP.keys()), default="esm2_t6_8M_UR50D")
    parser.add_argument("--esm_layer", type=int, default=6)
    parser.add_argument("--esm_pooling", choices=["mean", "median"], default="mean")
    parser.add_argument("--esm_batch_size", type=int, default=2)
    parser.add_argument("--esm_max_len", type=int, default=512)
    parser.add_argument("--esm_pca_dim", type=int, default=11)
    parser.add_argument("--esm_cache_dir", default="")
    parser.add_argument("--cv_splits", type=int, default=5)
    parser.add_argument("--xgb_clf_estimators", type=int, default=600)
    parser.add_argument("--xgb_reg_estimators", type=int, default=800)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_n_jobs", type=int, default=-1)
    return parser.parse_args()


def main():
    train_and_evaluate(parse_args())


if __name__ == "__main__":
    main()

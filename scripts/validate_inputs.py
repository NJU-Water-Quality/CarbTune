#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None


def resolve_path(value, project_root):
    path = Path(str(value))
    if path.is_absolute():
        return path
    return project_root / path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", default="Input/curves_template.csv")
    parser.add_argument("--seqs", default="Input/sequences.csv")
    parser.add_argument("--carbons_smiles", default="outputs_smiles/carbons_smiles.csv")
    parser.add_argument("--report", default="input_validation.json")
    parser.add_argument("--expected_strains", type=int, default=12)
    parser.add_argument("--expected_carbons", type=int, default=45)
    parser.add_argument("--expected_points", type=int, default=201)
    parser.add_argument("--uptake_min", type=float, default=0.0)
    parser.add_argument("--uptake_max", type=float, default=100.0)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    project_root = Path.cwd()
    curves_path = Path(args.curves)
    seqs_path = Path(args.seqs)
    smiles_path = Path(args.carbons_smiles)
    errors = []
    warnings = []

    for label, path in [("curves", curves_path), ("sequences", seqs_path), ("carbon structures", smiles_path)]:
        if not path.exists():
            errors.append(f"Missing {label} file: {path}")

    report = {
        "files": {
            "curves": str(curves_path),
            "sequences": str(seqs_path),
            "carbons_smiles": str(smiles_path),
        }
    }

    curves = None
    seqs = None
    smiles = None

    if curves_path.exists():
        curves = pd.read_csv(curves_path)
        required = {"strain", "carbon", "uptake", "growth"}
        missing = sorted(required - set(curves.columns))
        if missing:
            errors.append(f"Curves file is missing columns: {missing}")
        else:
            curves["strain"] = curves["strain"].astype(str).str.strip()
            curves["carbon"] = curves["carbon"].astype(str).str.strip()
            curves["uptake"] = pd.to_numeric(curves["uptake"], errors="coerce")
            curves["growth"] = pd.to_numeric(curves["growth"], errors="coerce")
            missing_values = int(curves[list(required)].isna().sum().sum())
            if missing_values:
                errors.append(f"Curves file contains {missing_values} missing required values")
            n_strains = int(curves["strain"].nunique())
            n_carbons = int(curves["carbon"].nunique())
            n_pairs = int(curves[["strain", "carbon"]].drop_duplicates().shape[0])
            exact_duplicates = int(curves.duplicated().sum())
            duplicate_keys = int(curves.duplicated(["strain", "carbon", "uptake"]).sum())
            group_sizes = curves.groupby(["strain", "carbon"]).size()
            nonstandard_groups = int((group_sizes != args.expected_points).sum())
            outside = int(((curves["uptake"] < args.uptake_min) | (curves["uptake"] > args.uptake_max)).sum())
            if n_strains != args.expected_strains:
                errors.append(f"Expected {args.expected_strains} strains but found {n_strains}")
            if n_carbons != args.expected_carbons:
                errors.append(f"Expected {args.expected_carbons} carbons but found {n_carbons}")
            if n_pairs != n_strains * n_carbons:
                errors.append(f"Expected a complete strain-carbon grid but found {n_pairs} pairs")
            if exact_duplicates:
                warnings.append(f"Curves file contains {exact_duplicates} exact duplicate rows")
            if duplicate_keys:
                warnings.append(f"Curves file contains {duplicate_keys} duplicate strain-carbon-uptake keys")
            if nonstandard_groups:
                warnings.append(f"{nonstandard_groups} strain-carbon curves do not contain {args.expected_points} rows")
            if outside:
                warnings.append(f"{outside} uptake values are outside [{args.uptake_min}, {args.uptake_max}]")
            report["curves"] = {
                "rows": int(len(curves)),
                "strains": n_strains,
                "carbons": n_carbons,
                "strain_carbon_pairs": n_pairs,
                "exact_duplicate_rows": exact_duplicates,
                "duplicate_strain_carbon_uptake_keys": duplicate_keys,
                "curve_points_min": int(group_sizes.min()),
                "curve_points_max": int(group_sizes.max()),
                "curves_with_nonstandard_point_count": nonstandard_groups,
                "uptake_min": float(curves["uptake"].min()),
                "uptake_max": float(curves["uptake"].max()),
                "uptake_values_outside_expected_range": outside,
                "growth_min": float(curves["growth"].min()),
                "growth_max": float(curves["growth"].max()),
            }

    if seqs_path.exists():
        seqs = pd.read_csv(seqs_path, keep_default_na=False)
        required = {"strain", "sequence", "fasta_path"}
        missing = sorted(required - set(seqs.columns))
        if missing:
            errors.append(f"Sequences file is missing columns: {missing}")
        else:
            seqs["strain"] = seqs["strain"].astype(str).str.strip()
            duplicate_strains = int(seqs["strain"].duplicated().sum())
            missing_fastas = []
            hidden_fastas = []
            for value in seqs["fasta_path"]:
                if not str(value).strip():
                    continue
                path = resolve_path(value, project_root)
                if any(part.startswith(".") for part in Path(str(value)).parts):
                    hidden_fastas.append(str(value))
                if not path.exists():
                    missing_fastas.append(str(value))
            if duplicate_strains:
                errors.append(f"Sequences file contains {duplicate_strains} duplicate strain rows")
            if missing_fastas:
                errors.append(f"Sequences file references {len(missing_fastas)} missing FASTA files")
            if hidden_fastas:
                errors.append(f"Sequences file references hidden or checkpoint paths: {hidden_fastas}")
            report["sequences"] = {
                "rows": int(len(seqs)),
                "strains": int(seqs["strain"].nunique()),
                "duplicate_strain_rows": duplicate_strains,
                "missing_fasta_paths": missing_fastas,
            }

    if smiles_path.exists():
        smiles = pd.read_csv(smiles_path, keep_default_na=False)
        required = {"carbon", "smiles"}
        missing = sorted(required - set(smiles.columns))
        if missing:
            errors.append(f"Carbon structure file is missing columns: {missing}")
        else:
            smiles["carbon"] = smiles["carbon"].astype(str).str.strip()
            empty_smiles = int((smiles["smiles"].astype(str).str.strip() == "").sum())
            duplicate_carbons = int(smiles["carbon"].duplicated().sum())
            invalid_smiles = None
            if Chem is not None:
                invalid_smiles = int(sum(Chem.MolFromSmiles(str(value).strip()) is None for value in smiles["smiles"]))
            if empty_smiles:
                errors.append(f"Carbon structure file contains {empty_smiles} empty SMILES values")
            if duplicate_carbons:
                errors.append(f"Carbon structure file contains {duplicate_carbons} duplicate carbon names")
            if invalid_smiles:
                errors.append(f"RDKit could not parse {invalid_smiles} SMILES values")
            report["carbon_structures"] = {
                "rows": int(len(smiles)),
                "carbons": int(smiles["carbon"].nunique()),
                "empty_smiles": empty_smiles,
                "duplicate_carbon_names": duplicate_carbons,
                "duplicate_smiles": int(smiles["smiles"].duplicated().sum()),
                "rdkit_invalid_smiles": invalid_smiles,
            }

    if curves is not None and seqs is not None and {"strain"}.issubset(seqs.columns) and {"strain"}.issubset(curves.columns):
        curve_strains = set(curves["strain"].astype(str).str.strip())
        sequence_strains = set(seqs["strain"].astype(str).str.strip())
        if curve_strains != sequence_strains:
            errors.append("Strain names differ between curves and sequences files")
        report["strain_name_check"] = {
            "only_in_curves": sorted(curve_strains - sequence_strains),
            "only_in_sequences": sorted(sequence_strains - curve_strains),
        }

    if curves is not None and smiles is not None and {"carbon"}.issubset(smiles.columns) and {"carbon"}.issubset(curves.columns):
        curve_carbons = set(curves["carbon"].astype(str).str.strip())
        smiles_carbons = set(smiles["carbon"].astype(str).str.strip())
        if curve_carbons != smiles_carbons:
            errors.append("Carbon names differ between curves and carbon structure files")
        report["carbon_name_check"] = {
            "only_in_curves": sorted(curve_carbons - smiles_carbons),
            "only_in_structures": sorted(smiles_carbons - curve_carbons),
        }

    report["errors"] = errors
    report["warnings"] = warnings
    report["status"] = "failed" if errors or (args.strict and warnings) else "passed_with_warnings" if warnings else "passed"
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": len(errors), "warnings": len(warnings), "report": args.report}, ensure_ascii=False))
    if errors or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

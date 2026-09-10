#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.inchi import MolToInchiKey

_ws = re.compile(r"\s+")

def norm_text(x: str) -> str:
    if x is None:
        return ""
    s = str(x).replace("\u3000"," ").replace("\xa0"," ").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == "'") or (s[0] == s[-1] == '"')):
        s = s[1:-1]
    s = _ws.sub(" ", s)
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="manual template CSV ( carbon, smiles)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="Standardize output CSV path")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    outdir = out_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if not {"carbon", "smiles"}.issubset(df.columns):
        raise ValueError("The input file must contain columns: carbon, smiles")

    rows_ok, rows_bad = [], []

    for _, r in df.iterrows():
        carbon = norm_text(r["carbon"])
        smi_in = str(r["smiles"]).strip()

        if not carbon:
            continue

        if not smi_in:
            rows_bad.append({"carbon": carbon, "smiles": "", "why": "empty"})
            continue

        mol = Chem.MolFromSmiles(smi_in)
        if mol is None:
            rows_bad.append({"carbon": carbon, "smiles": smi_in, "why": "parse_failed"})
            continue

        smi_can = Chem.MolToSmiles(mol, canonical=True)
        mw      = Descriptors.MolWt(mol)
        logp    = Descriptors.MolLogP(mol)
        tpsa    = Descriptors.TPSA(mol)
        hbd     = Descriptors.NumHDonors(mol)
        hba     = Descriptors.NumHAcceptors(mol)
        rot     = Descriptors.NumRotatableBonds(mol)
        rings   = Descriptors.RingCount(mol)
        heavy   = Descriptors.HeavyAtomCount(mol)
        inchikey= MolToInchiKey(mol)

        rows_ok.append({
            "carbon": carbon,
            "smiles": smi_can,       
            "InChIKey": inchikey,
            "MolWt": mw, "LogP": logp, "TPSA": tpsa,
            "HBD": hbd, "HBA": hba, "RotBonds": rot, "RingCount": rings,
            "HeavyAtom": heavy
        })

    ok_df  = pd.DataFrame(rows_ok)
    bad_df = pd.DataFrame(rows_bad)

    if len(ok_df):
        ok_df = ok_df.sort_values("carbon")
        ok_df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        ok_df = pd.DataFrame(columns=["carbon","smiles","InChIKey","MolWt","LogP","TPSA","HBD","HBA","RotBonds","RingCount","HeavyAtom"])
        ok_df.to_csv(out_path, index=False, encoding="utf-8")

    bad_path = outdir / "carbons_smiles_bad.csv"
    if len(bad_df):
        if "carbon" in bad_df.columns:
            bad_df = bad_df.sort_values("carbon")
        bad_df.to_csv(bad_path, index=False, encoding="utf-8")
    elif bad_path.exists():
        bad_path.unlink()

    if len(ok_df):
        dupes = (ok_df.groupby("InChIKey")["carbon"]
                      .apply(lambda s: sorted(set(s)))
                      .reset_index())
        dupes["n"] = dupes["carbon"].apply(len)
        dupes = dupes[dupes["n"] > 1]
        dupes_path = outdir / "carbons_smiles_dupes.csv"
        if len(dupes):
            dupes.to_csv(dupes_path, index=False, encoding="utf-8")
        elif dupes_path.exists():
            dupes_path.unlink()

    print(f"OK: {len(ok_df)} | BAD: {len(bad_df)} → {out_path.as_posix()}")
    if len(bad_df):
        print(f"- Resolution failed/blank entry please repair: {(outdir/'carbons_smiles_bad.csv').as_posix()}")

if __name__ == "__main__":
    main()

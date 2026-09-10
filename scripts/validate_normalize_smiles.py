#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_normalize_smiles.py
- 输入: 手动填写的 carbons_smiles_manual_template.csv (列: carbon, smiles)
- 输出: carbons_smiles.csv (规范化后可用于训练)
- 额外输出(仅在需要时):
    - carbons_smiles_bad.csv   (解析失败/空白)
    - carbons_smiles_dupes.csv (不同 carbon 名对应同一 InChIKey)
用法示例:
  python validate_normalize_smiles.py \
    --in outputs_smiles/carbons_smiles_manual_template.csv \
    --out outputs_smiles/carbons_smiles.csv
"""
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
                    help="手动模板 CSV 路径 (列: carbon, smiles)")
    ap.add_argument("--out", dest="out_path", required=True,
                    help="规范化输出 CSV 路径")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    outdir = out_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    if not {"carbon", "smiles"}.issubset(df.columns):
        raise ValueError("输入文件必须包含列: carbon, smiles")

    rows_ok, rows_bad = [], []

    for _, r in df.iterrows():
        carbon = norm_text(r["carbon"])
        smi_in = str(r["smiles"]).strip()

        if not carbon:
            # 跳过无名行
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
            "smiles": smi_can,       # 规范化 SMILES
            "InChIKey": inchikey,
            "MolWt": mw, "LogP": logp, "TPSA": tpsa,
            "HBD": hbd, "HBA": hba, "RotBonds": rot, "RingCount": rings,
            "HeavyAtom": heavy
        })

    ok_df  = pd.DataFrame(rows_ok)
    bad_df = pd.DataFrame(rows_bad)

    # 写主输出
    if len(ok_df):
        ok_df = ok_df.sort_values("carbon")
        ok_df.to_csv(out_path, index=False, encoding="utf-8")
    else:
        # 没有任何可用记录时，仍写一个空壳，方便你察看
        ok_df = pd.DataFrame(columns=["carbon","smiles","InChIKey","MolWt","LogP","TPSA","HBD","HBA","RotBonds","RingCount","HeavyAtom"])
        ok_df.to_csv(out_path, index=False, encoding="utf-8")

    # 写坏样本（如存在）
    bad_path = outdir / "carbons_smiles_bad.csv"
    if len(bad_df):
        if "carbon" in bad_df.columns:
            bad_df = bad_df.sort_values("carbon")
        bad_df.to_csv(bad_path, index=False, encoding="utf-8")
    elif bad_path.exists():
        bad_path.unlink()

    # InChIKey 去重提示（仅在有OK记录时）
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
        print(f"- 解析失败/空白条目请修复: {(outdir/'carbons_smiles_bad.csv').as_posix()}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared data-processing and feature-engineering functions for CarbTune.

This module contains no model training entry point. It builds experimental
labels, amino-acid composition features, ESM-2 strain embeddings, and RDKit
carbon-source features for both training and inference.
"""

import csv
import hashlib
import io
import json
import re
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", message="R\\^2 score is not well-defined")

# ===== RDKit（可选）=====
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit import DataStructs, RDLogger
    RDLogger.DisableLog('rdApp.*')
except Exception:
    Chem = None

# 结构描述符列
STRUCT_DESC_COLS = [
    'MolWt','LogP','TPSA','HBD','HBA','RotBonds','RingCount','HeavyAtom','FractionCSP3'
]

# ===== Torch / ESM（可选）=====
try:
    import torch
    import esm
except Exception:
    torch = None
    esm = None

# ===== AA 基础特征 =====
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = {
    'I': 4.5,'V': 4.2,'L': 3.8,'F': 2.8,'C': 2.5,'M': 1.9,'A': 1.8,'G': -0.4,'T': -0.7,'S': -0.8,
    'W': -0.9,'Y': -1.3,'P': -1.6,'H': -3.2,'E': -3.5,'Q': -3.5,'D': -3.5,'N': -3.5,'K': -3.9,'R': -4.5
}

# ---------- 列名规范化 ----------
COL_SYNONYMS = {
    'strain': {'strain','菌株','菌株名称','organism','model'},
    'carbon': {'carbon','碳源','碳源名称','substrate','carbon_name'},
    'uptake': {'uptake','intake','摄入','摄入量','carbon_uptake','u'},
    'growth': {'growth','生长','生长率','growth_rate','g'},
}

_ws_re = re.compile(r"\s+")
def norm_text(x) -> str:
    if pd.isna(x): return ""
    s = str(x).replace("\u3000"," ").replace("\xa0"," ").strip()
    if len(s)>=2 and ((s[0]==s[-1]=="'") or (s[0]==s[-1]=='"')):
        s = s[1:-1]
    s = _ws_re.sub(" ", s)
    return s.strip()

def sniff_sep(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        return "\t" if ("\t" in sample and sample.count("\t")>sample.count(",")) else ","

def read_table_safely(path: Path) -> Tuple[pd.DataFrame, dict]:
    encodings = ["utf-8","utf-8-sig","gbk","latin-1"]
    info = {}
    text = None
    for enc in encodings:
        try:
            text = path.read_text(encoding=enc)
            info['encoding'] = enc
            break
        except Exception:
            continue
    if text is None:
        df = pd.read_csv(path, engine="python", sep=None)
        info['encoding']='pandas-default'; info['sep']='auto'
        return df, info
    head = "\n".join(text.splitlines()[:200])
    sep = sniff_sep(head)
    info['sep'] = sep
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    return df, info

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping={}
    for col in df.columns:
        key=col.strip().lower()
        for std, cands in COL_SYNONYMS.items():
            if key in cands:
                mapping[col]=std; break
    return df.rename(columns=mapping)

# ---------- 序列清洗/FASTA 读取 ----------
def _clean_seq(seq: str) -> str:
    # 仅保留字母，转大写
    return re.sub(r'[^A-Za-z]', '', str(seq or '')).upper()

def _read_fasta_sequences(fpath: str) -> List[str]:
    """读取本地 FASTA，返回其中每条蛋白的序列列表（只保留A-Z）"""
    seqs = []
    if not fpath:
        return seqs
    p = Path(fpath)
    if not p.exists() or not p.is_file():
        return seqs
    try:
        buff = []
        with p.open('r', encoding='utf-8', errors='ignore') as fh:
            for line in fh:
                if line.startswith('>'):
                    if buff:
                        s = _clean_seq(''.join(buff))
                        if s: seqs.append(s)
                        buff = []
                else:
                    buff.append(line.strip())
        if buff:
            s = _clean_seq(''.join(buff))
            if s: seqs.append(s)
    except Exception:
        pass
    return seqs

# ---------- AA 特征 ----------
def aa_features(seq: str) -> Dict[str, float]:
    s = re.sub(r'[^A-Z]', '', (seq or '').upper())
    feats = {}
    if not s:
        for a in AA: feats[f'aa_{a}'] = 0.0
        feats.update(length=0.0, frac_aromatic=0.0, frac_charged=0.0, kd_mean=0.0)
        return feats
    n=len(s); counts=Counter(s)
    for a in AA: feats[f'aa_{a}'] = counts.get(a,0)/n
    feats['length'] = float(n)
    feats['frac_aromatic'] = (counts.get('F',0)+counts.get('Y',0)+counts.get('W',0))/n
    feats['frac_charged']  = sum(counts.get(a,0) for a in ['D','E','K','R','H'])/n
    feats['kd_mean'] = float(np.mean([KD.get(a,0.0) for a in s]))
    return feats

def aa_features_aggregate(seqs: List[str]) -> Dict[str, float]:
    """
    同一菌株多条蛋白 -> “长度加权”聚合
    - aa_X：总体频率（总计数 / 总长度）
    - length：总长度
    - frac_aromatic / frac_charged：总体频率计算
    - kd_mean：按残基加权的整体平均 KD
    """
    feats = {}
    if not seqs:
        for a in AA: feats[f'aa_{a}'] = 0.0
        feats.update(length=0.0, frac_aromatic=0.0, frac_charged=0.0, kd_mean=0.0)
        return feats

    total_counts = Counter()
    total_len = 0
    kd_sum = 0.0

    for s in seqs:
        s2 = _clean_seq(s)
        if not s2: continue
        total_counts.update(s2)
        total_len += len(s2)
        kd_sum += sum(KD.get(ch, 0.0) for ch in s2)

    if total_len == 0:
        for a in AA: feats[f'aa_{a}'] = 0.0
        feats.update(length=0.0, frac_aromatic=0.0, frac_charged=0.0, kd_mean=0.0)
        return feats

    for a in AA:
        feats[f'aa_{a}'] = total_counts.get(a, 0) / total_len

    feats['length'] = float(total_len)
    feats['frac_aromatic'] = sum(total_counts.get(a,0) for a in ['F','Y','W']) / total_len
    feats['frac_charged']  = sum(total_counts.get(a,0) for a in ['D','E','K','R','H']) / total_len
    feats['kd_mean'] = float(kd_sum / total_len)
    return feats

def build_seq_df_from_sequences_csv(seqs_csv: Path) -> pd.DataFrame:
    """
    读取 sequences.csv（列：strain, sequence, fasta_path）
    - 对于每一行：
      * 如果 sequence 非空 -> 作为一条蛋白
      * 如果 fasta_path 指向有效 FASTA -> 追加该文件内所有条目
    返回列：strain, sequence （可能同株多行）
    """
    df = pd.read_csv(seqs_csv)
    if 'strain' not in df.columns:
        raise ValueError("sequences.csv 必须包含列: strain")
    if 'sequence' not in df.columns:
        df['sequence'] = ''
    if 'fasta_path' not in df.columns:
        df['fasta_path'] = ''

    df['strain'] = df['strain'].apply(norm_text)
    df['sequence'] = df['sequence'].astype('string').fillna('')
    df['fasta_path'] = df['fasta_path'].astype('string').fillna('')

    rows = []
    for _, r in df.iterrows():
        s = norm_text(r['strain'])
        seq_inline = _clean_seq(r['sequence'])
        if seq_inline:
            rows.append({'strain': s, 'sequence': seq_inline})
        fps = str(r['fasta_path']).strip()
        if fps:
            for seq in _read_fasta_sequences(fps):
                if seq:
                    rows.append({'strain': s, 'sequence': seq})
    rows = [x for x in rows if x['sequence']]
    if not rows:
        return pd.DataFrame(columns=['strain','sequence'])
    return pd.DataFrame(rows, columns=['strain','sequence'])

# ---------- ESM2 嵌入 ----------
ESM_NAME_MAP = {
    'esm2_t6_8M_UR50D': 'esm2_t6_8M_UR50D',
    'esm2_t12_35M_UR50D': 'esm2_t12_35M_UR50D',
    'esm2_t30_150M_UR50D': 'esm2_t30_150M_UR50D',
}

def _esm_load(model_name: str):
    if esm is None or torch is None:
        raise RuntimeError("未安装 fair-esm / torch，无法使用 --esm_on")
    if model_name not in ESM_NAME_MAP:
        raise ValueError(f"不支持的 ESM2 模型: {model_name}")
    model, alphabet = getattr(esm.pretrained, model_name)()
    return model, alphabet

def _md5(s: str) -> str:
    return hashlib.md5(s.encode('utf-8')).hexdigest()

def _esm_expected_meta(model_name: str, layer_idx: int, pooling: str, max_len: int, pca_dim: int) -> dict:
    return {
        "model_name": str(model_name),
        "layer": int(layer_idx),
        "pooling": str(pooling),
        "max_len": int(max_len),
        "pca_dim": int(pca_dim),
    }

def _esm_check_meta_or_raise(meta_path: Path, expected: dict) -> dict:
    """
    Strict check: if meta exists and mismatch -> raise.
    If meta missing -> return {} (caller decides).
    """
    if not meta_path.exists():
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"[ESM PCA] Failed to read meta file: {meta_path} ({e})")

    # only check core keys (ignore device/batch_size etc.)
    core_keys = ["model_name", "layer", "pooling", "max_len", "pca_dim"]
    for k in core_keys:
        if k not in meta:
            raise RuntimeError(f"[ESM PCA] Meta file missing key `{k}`: {meta_path}")
        if str(meta[k]) != str(expected[k]):
            raise RuntimeError(
                f"[ESM PCA] Meta mismatch for `{k}`: saved={meta[k]} vs expected={expected[k]}.\n"
                f"  Meta file: {meta_path}\n"
                f"  Fix: use the same esm settings, or change --esm_cache_dir to a new folder, or delete old PCA artifacts."
            )
    return meta

def _esm_load_or_fit_pca_pipe(
    X: np.ndarray,
    out_cache: Path,
    expected_meta: dict,
    allow_legacy: bool = True,
) -> Tuple[np.ndarray, Path]:
    """
    Strict reuse:
    - If out_cache has a saved scaler+PCA pipe and matching meta -> transform only.
    - Else if legacy artifacts exist (mean/scale + pca) and meta matches -> transform only, then upgrade to pipe.
    - Else fit scaler+PCA on X, save pipe+meta+legacy for future reuse.
    """
    out_cache.mkdir(parents=True, exist_ok=True)

    meta_path = out_cache / "esm_meta.json"
    pipe_path = out_cache / "esm_pca_pipe.joblib"

    # legacy
    legacy_pca_path = out_cache / "esm_pca.joblib"                 # PCA only
    legacy_mean_path = out_cache / "esm_scaler_mean.npy"
    legacy_scale_path = out_cache / "esm_scaler_scale.npy"

    # If pipe exists -> strict meta check -> transform
    if pipe_path.exists():
        _esm_check_meta_or_raise(meta_path, expected_meta)
        pipe = joblib.load(pipe_path)

        # dimension check (if available)
        try:
            n_in = pipe.named_steps["scaler"].n_features_in_
            if X.shape[1] != n_in:
                raise RuntimeError(
                    f"[ESM PCA] Embedding dim mismatch: X has {X.shape[1]}, "
                    f"but saved scaler expects {n_in}. "
                    f"Did you change esm_model/layer/pooling/max_len?"
                )
        except Exception:
            # if n_features_in_ not present, skip hard check
            pass

        Xp = pipe.transform(X)
        return Xp, pipe_path

    # If legacy exists -> strict meta check -> transform and upgrade
    if allow_legacy and legacy_pca_path.exists() and legacy_mean_path.exists() and legacy_scale_path.exists():
        _esm_check_meta_or_raise(meta_path, expected_meta)

        pca = joblib.load(legacy_pca_path)
        mean_ = np.load(legacy_mean_path)
        scale_ = np.load(legacy_scale_path)

        if mean_.shape[0] != X.shape[1] or scale_.shape[0] != X.shape[1]:
            raise RuntimeError(
                f"[ESM PCA] Legacy scaler dim mismatch: mean/scale dim={mean_.shape[0]} vs X dim={X.shape[1]}.\n"
                f"  Fix: use same esm settings or regenerate cache_dir."
            )

        # standardize then PCA transform
        Xs = (X - mean_) / np.where(scale_ == 0, 1.0, scale_)
        Xp = pca.transform(Xs)

        # upgrade: save a pipeline for future
        pipe = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=Xp.shape[1], random_state=42))])
        # hack: set fitted params into scaler and pca without refit
        pipe.named_steps["scaler"].mean_ = mean_.astype(np.float64, copy=False)
        pipe.named_steps["scaler"].scale_ = scale_.astype(np.float64, copy=False)
        pipe.named_steps["scaler"].var_ = (scale_.astype(np.float64, copy=False) ** 2)
        pipe.named_steps["scaler"].n_features_in_ = int(mean_.shape[0])
        pipe.named_steps["scaler"].n_samples_seen_ = int(1)

        # PCA object already fitted; reuse it directly
        pipe.named_steps["pca"] = pca
        joblib.dump(pipe, pipe_path)
        return Xp, pipe_path

    # Otherwise -> fit now (training time).
    # Centered PCA has at most n_samples - 1 informative components.
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    if Xs.shape[0] < 2:
        raise RuntimeError("[ESM PCA] At least two strains are required to fit PCA.")

    n_comp = min(int(expected_meta["pca_dim"]), Xs.shape[1], Xs.shape[0] - 1)
    if n_comp < 1:
        raise RuntimeError("[ESM PCA] Cannot fit PCA: n_comp < 1. Check your input sequences.")

    pca = PCA(n_components=n_comp, random_state=42)
    Xp = pca.fit_transform(Xs)

    # Save pipe
    pipe = Pipeline([("scaler", scaler), ("pca", pca)])
    joblib.dump(pipe, pipe_path)

    # Also save legacy artifacts (for compatibility)
    np.save(legacy_mean_path, scaler.mean_)
    np.save(legacy_scale_path, scaler.scale_)
    joblib.dump(pca, legacy_pca_path)

    # Save meta (core + extra info)
    meta_save = expected_meta.copy()
    meta_save.update({
        "pca_dim_actual": int(n_comp),
        "device_used": "unknown",
        "batch_size": "unknown",
    })
    meta_path.write_text(json.dumps(meta_save, ensure_ascii=False, indent=2), encoding="utf-8")

    return Xp, pipe_path

def esm_embed_dataframe(raw_seqs_df: pd.DataFrame,
                        out_cache: Path,
                        model_name='esm2_t6_8M_UR50D',
                        layer: int = -1,
                        pooling: str = 'mean',
                        batch_size: int = 4,
                        device: str = 'auto',
                        max_len: int = 1022,
                        pca_dim: int = 11) -> pd.DataFrame:
    """
    输入：列 strain, sequence（允许同株多条蛋白）
    输出：每株一行：strain + esm_0..esm_{pca_dim-1}（严格复用训练 PCA）
    """
    out_cache.mkdir(parents=True, exist_ok=True)
    if device == 'auto':
        if (torch is not None) and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

    model, alphabet = _esm_load(model_name)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    # 将 -1 自动映射到最后一层
    try:
        last_layer = int(model.num_layers)
    except Exception:
        last_layer = 6  # 对 t6_8M 的兜底
    layer_idx = last_layer if layer < 0 else int(layer)
    if layer_idx < 1 or layer_idx > last_layer:
        raise ValueError(f"无效的 --esm_layer={layer}；应在 1..{last_layer}，或用 -1 表示最后一层")

    def _prep_seq(seq: str) -> str:
        s = re.sub(r'[^A-Za-z]', '', str(seq)).upper()
        if len(s) > max_len:
            s = s[:max_len]
        return s

    # 逐条蛋白向量（缓存）
    per_protein_vecs = []   # (strain, vec_ndarray)
    need_rows = []
    for _, r in raw_seqs_df.iterrows():
        strain = norm_text(r['strain'])
        seq = _prep_seq(r.get('sequence',''))
        if not seq:
            continue
        key = _md5(strain + '|' + seq + f'|{model_name}|{layer_idx}|{pooling}|{max_len}')
        npy_path = out_cache / f"{key}.npy"
        if npy_path.exists():
            vec = np.load(npy_path)
            per_protein_vecs.append((strain, vec))
        else:
            need_rows.append((strain, seq, key, npy_path))

    with torch.no_grad():
        for i in range(0, len(need_rows), batch_size):
            batch = need_rows[i:i+batch_size]
            labels = [f"{b[0]}_{j}" for j, b in enumerate(batch)]
            data = [(lab, s) for lab, (_, s, _, _) in zip(labels, batch)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)
            out = model(batch_tokens, repr_layers=[layer_idx], return_contacts=False)
            reps = out["representations"][layer_idx]  # (B, T, C)

            for (strain, seq, key, npy_path), rep in zip(batch, reps):
                rep = rep[1:len(seq)+1, :]  # 去 BOS/EOS
                if pooling == 'median':
                    vec = torch.median(rep, dim=0).values
                else:
                    vec = torch.mean(rep, dim=0)
                vec = vec.detach().cpu().numpy().astype(np.float32)
                np.save(npy_path, vec)
                per_protein_vecs.append((strain, vec))

    if not per_protein_vecs:
        return pd.DataFrame()

    # 按菌株聚合（多蛋白均值）
    by_strain = defaultdict(list)
    for s, v in per_protein_vecs:
        by_strain[s].append(v)
    strains, mats = [], []
    for s, vs in by_strain.items():
        mats.append(np.vstack(vs).mean(axis=0))
        strains.append(s)
    X = np.vstack(mats).astype(np.float32)

    # ===== 严格复用训练 PCA（StandardScaler+PCA）=====
    if pca_dim is None or int(pca_dim) <= 0:
        # 不做 PCA：只返回 strain
        return pd.DataFrame({'strain': strains})

    expected = _esm_expected_meta(model_name, layer_idx, pooling, max_len, int(pca_dim))
    Xp, used_pipe_path = _esm_load_or_fit_pca_pipe(X, out_cache=out_cache, expected_meta=expected, allow_legacy=True)

    cols = [f'esm_{i}' for i in range(Xp.shape[1])]
    df_emb = pd.DataFrame(Xp, columns=cols)
    df_emb.insert(0, 'strain', strains)

    # ===== 记录完整 meta（覆盖/补全）=====
    # 注意：上面的 _esm_load_or_fit_pca_pipe 已经写过 esm_meta.json（或校验过一致性）。
    meta_path = out_cache / "esm_meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    except Exception:
        meta = {}
    meta.update({
        "model_name": model_name,
        "layer": int(layer_idx),
        "pooling": pooling,
        "max_len": int(max_len),
        "pca_dim_actual": int(Xp.shape[1]),
        "device_used": device,
        "batch_size": int(batch_size),
        "pipe_path": str(used_pipe_path.name),
    })
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return df_emb

# ---------- 结构特征 ----------
def smiles_to_features(smiles: str, fp_bits=2048, radius=2):
    if Chem is None or not smiles:
        return None, {}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=fp_bits)
    arr = np.zeros((fp_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    desc = {
        'MolWt': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'RingCount': Descriptors.RingCount(mol),
        'HeavyAtom': Descriptors.HeavyAtomCount(mol),
        'FractionCSP3': Descriptors.FractionCSP3(mol),
    }
    return arr, desc

def build_carbon_struct_features(carbons_smiles: pd.DataFrame, pca_dim=13,
                                 fp_bits: int = 2048, radius: int = 2,
                                 pca_pipe_path: Path = None):
    carbons_smiles = carbons_smiles.copy()
    carbons_smiles['carbon'] = carbons_smiles['carbon'].apply(norm_text)
    carbons_smiles['smiles'] = carbons_smiles['smiles'].astype(str).str.strip()

    feats, fps, names = [], [], []
    ok = 0; fail = 0
    for _, r in carbons_smiles.iterrows():
        c = r['carbon']; s = r['smiles']
        fp, desc = smiles_to_features(s, fp_bits=fp_bits, radius=radius)
        if fp is not None:
            ok += 1
            fps.append(fp)
            row = {'carbon': c}; row.update(desc); feats.append(row); names.append(c)
        else:
            fail += 1

    report = {'n_total': int(len(carbons_smiles)), 'n_ok': int(ok), 'n_fail': int(fail)}
    df_desc = pd.DataFrame(feats) if feats else pd.DataFrame(columns=['carbon'])

    if fps:
        fps = np.vstack(fps)
        try:
            if pca_pipe_path is not None and Path(pca_pipe_path).exists():
                pipe = joblib.load(pca_pipe_path)
                Xp = pipe.transform(fps)
                df_fp_pca = pd.DataFrame(Xp, columns=[f'fp_{i}' for i in range(Xp.shape[1])])
                df_fp_pca.insert(0, 'carbon', names)
            else:
                scaler = StandardScaler(with_mean=True, with_std=True)
                Xs = scaler.fit_transform(fps)
                total_var = float(np.var(Xs))
                if (fps.shape[0] < 2) or (total_var < 1e-12):
                    df_fp_pca = pd.DataFrame(columns=['carbon'])
                else:
                    n_comp = min(int(pca_dim), Xs.shape[1], Xs.shape[0] - 1)
                    pca = PCA(n_components=n_comp, random_state=42)
                    pca.fit(Xs)
                    pipe = Pipeline([('scaler', scaler), ('pca', pca)])
                    Xp = pipe.transform(fps)
                    if pca_pipe_path is not None:
                        pca_pipe_path = Path(pca_pipe_path)
                        pca_pipe_path.parent.mkdir(parents=True, exist_ok=True)
                        joblib.dump(pipe, pca_pipe_path)
                    df_fp_pca = pd.DataFrame(Xp, columns=[f'fp_{i}' for i in range(Xp.shape[1])])
                    df_fp_pca.insert(0, 'carbon', names)
        except Exception:
            df_fp_pca = pd.DataFrame(columns=['carbon'])
    else:
        df_fp_pca = pd.DataFrame(columns=['carbon'])

    return df_desc, df_fp_pca, report

# ---------- 平台期识别 ----------
def _plateau_from_curve(uptake: np.ndarray, growth: np.ndarray,
                        tol_frac: float, tol_abs_min: float,
                        value_mode: str='median') -> Tuple[float, float]:
    m = np.isfinite(uptake) & np.isfinite(growth)
    uptake = uptake[m]; growth = growth[m]
    if len(uptake) == 0:
        return np.nan, np.nan
    idx = np.argsort(uptake)
    u = uptake[idx].astype(float)
    g = growth[idx].astype(float)

    g_max = np.nanmax(g)
    if not np.isfinite(g_max): return np.nan, np.nan

    tol = max(float(tol_abs_min), float(tol_frac) * float(g_max))
    thr = g_max - tol

    suffix_min = np.minimum.accumulate(g[::-1])[::-1]
    candidates = np.where(suffix_min >= thr)[0]
    if len(candidates) > 0:
        i_star = int(candidates[0])
        u_opt = float(u[i_star])
        plateau_g = g[i_star:]
        g_plateau = float(np.nanmedian(plateau_g)) if value_mode=='median' else float(np.nanmax(plateau_g))
        return g_plateau, u_opt

    candidates = np.where(g >= thr)[0]
    if len(candidates) > 0:
        i_first = int(candidates[0])
        return float(g_max), float(u[i_first])

    i_best = int(np.nanargmax(g))
    return float(g[i_best]), float(u[i_best])

def build_actual_labels_with_plateau(curves: pd.DataFrame,
                                     growth_thr: float,
                                     tol_frac: float,
                                     tol_abs_min: float,
                                     value_mode: str='median') -> pd.DataFrame:
    need = ['strain','carbon','uptake','growth']
    missing = [c for c in need if c not in curves.columns]
    if missing:
        raise ValueError(f"curves_template.csv 缺列: {missing}，需要: {need}")

    df = curves.copy()
    df['strain'] = df['strain'].apply(norm_text)
    df['carbon'] = df['carbon'].apply(norm_text)
    df['uptake'] = pd.to_numeric(df['uptake'], errors='coerce')
    df['growth'] = pd.to_numeric(df['growth'], errors='coerce')
    df = df.dropna(subset=need)

    rows=[]
    for (s,c), grp in df.groupby(['strain','carbon'], sort=False):
        u = grp['uptake'].to_numpy(); g = grp['growth'].to_numpy()
        g_plateau, u_opt = _plateau_from_curve(u, g, tol_frac, tol_abs_min, value_mode)
        can = int((g_plateau > growth_thr) if np.isfinite(g_plateau) else 0)
        rows.append({'strain': s, 'carbon': c,
                     'max_growth_actual': g_plateau,
                     'u_opt_actual': u_opt,
                     'can_grow_actual': can})
    return pd.DataFrame(rows)

# ---------- 特征列选择 ----------
def pick_numeric_feature_cols(df: pd.DataFrame, exclude: List[str]) -> Tuple[List[str], List[str]]:
    keep, dropped = [], []
    for c in df.columns:
        if c in exclude: continue
        if pd.api.types.is_numeric_dtype(df[c]): keep.append(c)
        else: dropped.append(c)
    return keep, dropped

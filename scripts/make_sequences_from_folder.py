#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_sequences_from_folder.py
Scan a folder (default: ./Data) for FASTA files and generate sequences.csv that
your training script can consume.

Usage:
  python make_sequences_from_folder.py \
      --data_dir Data \
      --out_csv sequences.csv \
      --name_from filename \
      --ext .faa .fasta .fa

Options:
  --name_from {filename,parent}    How to infer strain name.
      - filename: use file stem (without extension) as strain name
      - parent:   use parent folder name as strain name
  --ext       One or more extensions to include (default: .faa .fasta .fa)
  --write_sequence  If set, also writes a 'sequence' column by concatenating AA sequences.
                    (Default: only writes fasta_path for faster I/O later)
"""
import argparse, sys
from pathlib import Path

def read_fasta_concat(p: Path) -> str:
    seqs = []
    try:
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            s = []
            for line in f:
                if line.startswith('>'):
                    if s:
                        seqs.append(''.join(s))
                        s = []
                else:
                    s.append(line.strip().upper())
            if s:
                seqs.append(''.join(s))
    except Exception as e:
        print(f"[WARN] Could not read {p}: {e}", file=sys.stderr)
        return ""
    return ''.join(seqs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='Data')
    ap.add_argument('--out_csv', default='sequences.csv')
    ap.add_argument('--name_from', default='filename', choices=['filename','parent'])
    ap.add_argument('--ext', nargs='+', default=['.faa','.fasta','.fa','.fna'])
    ap.add_argument('--write_sequence', action='store_true')
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = [
        p for p in data_dir.rglob('*')
        if p.suffix.lower() in {e.lower() for e in args.ext}
        and not any(part.startswith('.') for part in p.relative_to(data_dir).parts)
    ]

    rows = []
    for p in sorted(paths):
        if args.name_from == 'filename':
            strain = p.stem
        else:
            strain = p.parent.name
        row = {
            'strain': strain,
            'sequence': read_fasta_concat(p) if args.write_sequence else '',
            'fasta_path': str(p.as_posix())
        }
        rows.append(row)

    # write csv
    import csv
    out = Path(args.out_csv)
    with out.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['strain','sequence','fasta_path'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {out} with {len(rows)} rows.")

if __name__ == '__main__':
    main()

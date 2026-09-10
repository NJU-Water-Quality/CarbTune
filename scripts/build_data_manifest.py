#!/usr/bin/env python

import csv
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_count(path):
    if path.suffix.lower() in {".faa", ".fa", ".fasta", ".fna"}:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for line in handle if line.startswith(">"))
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    return ""


def main():
    root = Path(__file__).resolve().parent.parent
    files = []
    for folder in [root / "Data", root / "Input", root / "outputs_smiles"]:
        files.extend(path for path in folder.rglob("*") if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts))
    rows = []
    for path in sorted(files, key=lambda value: value.as_posix().lower()):
        rows.append({
            "relative_path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "record_count": record_count(path),
            "sha256": sha256(path),
        })
    output = root / "DATA_MANIFEST.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "size_bytes", "record_count", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} records")


if __name__ == "__main__":
    main()

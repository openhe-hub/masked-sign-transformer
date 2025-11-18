#!/usr/bin/env python3
"""
Parse Slurm/HPC training logs and extract the experiment name
and the metrics from the last epoch block.

Usage:
  python extract_slurm_logs.py <input_path> [<more_paths> ...] [-o output.csv]
E.G.: 
  python scripts/metrics/extract_slurm_logs.py logs/ -o output/metrics/total/total.csv

- <input_path> can be a file or a directory. Directories are searched recursively for *.log and *.out
- If -o is not provided, writes output CSV next to the first input as "extracted_metrics.csv"
"""

from __future__ import annotations
from pathlib import Path
import re
import csv
import sys
from typing import Dict, List, Tuple, Iterable, Optional, Set

# --- Regex patterns (robust to spacing) ---
RE_EXPERIMENT = re.compile(r"^\s*Running\s+experiment:\s*(?P<name>.+?)\s*$", re.IGNORECASE)
RE_EPOCH_HEAD = re.compile(
    r"^\s*Epoch\s+(?P<epoch>\d+)\s+completed\.\s*Current\s+LR:\s*(?P<lr>[0-9.eE+\-]+)\s*$",
    re.IGNORECASE,
)
RE_AVG_METRIC = re.compile(
    r"^\s*-\s*Avg\s+(?P<key>[A-Za-z0-9_]+)\s*:\s*(?P<val>[0-9.eE+\-]+)\s*$",
    re.IGNORECASE,
)

def iter_files(paths: List[str]) -> Iterable[Path]:
    """Yield files from given paths. Directories are walked recursively."""
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for f in path.rglob("*"):
                if f.is_file() and (f.suffix.lower() in {".log", ".out", ".txt"}):
                    yield f
        elif path.is_file():
            yield path

def parse_log(fp: Path) -> Dict[str, str]:
    """
    Parse a single log file and return a dict with fields:
      filename, experiment, last_epoch, lr, and any 'Avg ...' metrics from last epoch.
    If no epoch blocks are found, returns minimal info.
    """
    experiment: Optional[str] = None
    last_epoch_num: Optional[int] = None
    last_lr: Optional[str] = None
    last_metrics: Dict[str, str] = {}

    current_epoch: Optional[int] = None
    current_lr: Optional[str] = None
    current_metrics: Dict[str, str] = {}

    try:
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Experiment name
                m = RE_EXPERIMENT.search(line)
                if m:
                    experiment = m.group("name").strip()
                    continue

                # Epoch header
                m = RE_EPOCH_HEAD.search(line)
                if m:
                    # commit previous epoch as "last"
                    if current_epoch is not None:
                        last_epoch_num = current_epoch
                        last_lr = current_lr
                        last_metrics = dict(current_metrics)

                    # start new epoch block
                    current_epoch = int(m.group("epoch"))
                    current_lr = m.group("lr")
                    current_metrics = {}
                    continue

                # Avg metric inside an epoch block
                m = RE_AVG_METRIC.search(line)
                if m and current_epoch is not None:
                    key = m.group("key").strip().lower()  # normalize to lowercase keys
                    val = m.group("val").strip()
                    current_metrics[key] = val
                    continue

        # commit the final seen epoch
        if current_epoch is not None:
            last_epoch_num = current_epoch
            last_lr = current_lr
            last_metrics = dict(current_metrics)

    except Exception as e:
        # On error, still return a minimal row
        return {
            "filename": str(fp),
            "experiment": experiment or "",
            "last_epoch": "",
            "lr": "",
            "error": str(e),
        }

    row: Dict[str, str] = {
        "filename": str(fp),
        "experiment": experiment or "",
        "last_epoch": str(last_epoch_num) if last_epoch_num is not None else "",
        "lr": last_lr or "",
    }
    for k, v in last_metrics.items():
        row[f"avg_{k}"] = v
    return row

def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 1

    # Simple arg parse
    out_path: Optional[Path] = None
    in_paths: List[str] = []
    i = 0
    while i < len(argv):
        if argv[i] in ("-o", "--output"):
            if i + 1 >= len(argv):
                print("Missing output path after -o/--output", file=sys.stderr)
                return 2
            out_path = Path(argv[i + 1])
            i += 2
        else:
            in_paths.append(argv[i])
            i += 1

    if not in_paths:
        print("No input paths given.", file=sys.stderr)
        return 2

    files = list(iter_files(in_paths))
    if not files:
        print("No files found to parse.", file=sys.stderr)
        return 3

    # Parse all and collect union of metric columns
    rows: List[Dict[str, str]] = []
    metric_cols: Set[str] = set()
    for fp in files:
        row = parse_log(fp)
        rows.append(row)
        for k in row.keys():
            if k.startswith("avg_"):
                metric_cols.add(k)

    # Arrange columns
    base_cols = ["filename", "experiment", "last_epoch", "lr"]
    metric_cols_sorted = sorted(metric_cols)
    cols = base_cols + metric_cols_sorted
    # include error column if present anywhere
    if any("error" in r for r in rows):
        cols.append("error")

    # Decide output path
    if out_path is None:
        first = Path(in_paths[0])
        if first.is_dir():
            out_path = first / "extracted_metrics.csv"
        else:
            out_path = first.with_suffix(".metrics.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in cols})

    print(f"Wrote {len(rows)} rows to: {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

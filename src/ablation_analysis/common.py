# -*- coding:utf-8 -*-
"""
Shared I/O and statistics helpers for ablation experiment analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

DATA_ROOT = Path("/data/home/Kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_ablation")


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Read a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def latest_result(experiment: str, pattern: str = "*.jsonl") -> Path:
    """Return the most-recently-modified JSONL in an experiment dir."""
    d = DATA_ROOT / f"experiment_{experiment}"
    files = sorted(d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {d}")
    return files[0]


def asr(records: List[Dict], key: str = "jailbroken") -> float:
    """Attack success rate."""
    if not records:
        return 0.0
    return sum(1 for r in records if r.get(key, False)) / len(records)


def safe_mean(vals: List[float]) -> Optional[float]:
    return float(np.mean(vals)) if vals else None


def safe_median(vals: List[float]) -> Optional[float]:
    return float(np.median(vals)) if vals else None


def print_table(headers: List[str], rows: List[List], col_width: int = 18):
    """Print a simple ASCII table."""
    fmt = "".join(f"{{:<{col_width}}}" for _ in headers)
    print(fmt.format(*headers))
    print("-" * col_width * len(headers))
    for row in rows:
        print(fmt.format(*[_fmt_val(v) for v in row]))


def _fmt_val(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

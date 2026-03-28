# -*- coding:utf-8 -*-
"""
Experiment A1 analysis: First-cycle safe/unsafe sampling ratio across models.

Usage:
    python -m ablation_analysis.analyze_exp_a1 [dir_with_a1_jsonls]
    python -m ablation_analysis.analyze_exp_a1 path/to/file1.jsonl path/to/file2.jsonl ...
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, print_table


def load_jsonl(path) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(path) -> Dict:
    spath = str(path).replace(".jsonl", "_summary.json")
    p = Path(spath)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def analyze_single(records: List[Dict], summary: Dict, label: str = ""):
    """Print analysis for a single model's A1 results."""
    total = len(records)
    if total == 0:
        print(f"  {label}: no records")
        return

    all_scores = []
    for r in records:
        all_scores.extend(r["scores"])
    arr = np.array(all_scores)

    n_safe = sum(r["num_safe"] for r in records)
    n_unsafe = sum(r["num_unsafe"] for r in records)
    n_total = n_safe + n_unsafe

    all_safe_count = sum(1 for r in records if r["all_safe"])
    all_unsafe_count = sum(1 for r in records if r["all_unsafe"])

    per_prompt_unsafe = [r["unsafe_ratio"] for r in records]

    print(f"\n  Model: {label}")
    print(f"  Prompts: {total}, Samples/prompt: {records[0]['num_samples']}")
    print(f"  Total samples: {n_total}")
    print(f"  Safe:   {n_safe} ({n_safe/n_total*100:.1f}%)")
    print(f"  Unsafe: {n_unsafe} ({n_unsafe/n_total*100:.1f}%)")
    print(f"  All-safe prompts:   {all_safe_count}/{total} ({all_safe_count/total*100:.1f}%)")
    print(f"  All-unsafe prompts: {all_unsafe_count}/{total} ({all_unsafe_count/total*100:.1f}%)")
    print(f"  Per-prompt unsafe ratio: mean={np.mean(per_prompt_unsafe):.4f}, "
          f"median={np.median(per_prompt_unsafe):.4f}, "
          f"std={np.std(per_prompt_unsafe):.4f}")
    print(f"  Score distribution: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
          f"std={arr.std():.4f}")

    return {
        "model": label,
        "total_prompts": total,
        "safe_ratio": n_safe / n_total,
        "unsafe_ratio": n_unsafe / n_total,
        "all_safe_prompts": all_safe_count,
        "all_safe_ratio": all_safe_count / total,
        "all_unsafe_prompts": all_unsafe_count,
        "all_unsafe_ratio": all_unsafe_count / total,
        "mean_unsafe_per_prompt": float(np.mean(per_prompt_unsafe)),
        "mean_score": float(arr.mean()),
    }


def analyze_cross_model(all_results: Dict[str, List[Dict]]):
    """Compare across models."""
    print("\n" + "=" * 70)
    print("  Cross-model comparison")
    print("=" * 70)

    rows = []
    for model, records in all_results.items():
        if not records:
            continue
        n_total = sum(r["num_safe"] + r["num_unsafe"] for r in records)
        n_unsafe = sum(r["num_unsafe"] for r in records)
        all_safe = sum(1 for r in records if r["all_safe"])
        total = len(records)
        mean_score = float(np.mean([r["mean_score"] for r in records]))
        rows.append([
            model,
            f"{n_unsafe/n_total*100:.1f}%",
            f"{all_safe}/{total}",
            f"{all_safe/total*100:.1f}%",
            f"{mean_score:.4f}",
        ])

    print_table(
        ["Model", "Unsafe%", "All-safe", "All-safe%", "Mean score"],
        rows,
        col_width=16,
    )


def main():
    exp_dir = DATA_ROOT / "experiment_A1"

    if len(sys.argv) > 1:
        # Files or directory provided
        paths = []
        for arg in sys.argv[1:]:
            p = Path(arg)
            if p.is_dir():
                paths.extend(sorted(p.glob("*.jsonl")))
            else:
                paths.append(p)
    else:
        if exp_dir.exists():
            paths = sorted(exp_dir.glob("*.jsonl"))
        else:
            print(f"No experiment_A1 directory found at {exp_dir}")
            return

    # Filter out summary files
    paths = [p for p in paths if "_summary" not in p.name]

    if not paths:
        print("No JSONL files found.")
        return

    print("=" * 70)
    print("  Experiment A1: First-cycle safe/unsafe sampling ratio")
    print("=" * 70)

    all_results = {}
    for path in paths:
        records = load_jsonl(path)
        summary = load_summary(path)
        # Extract model name from summary or filename
        model = summary.get("model", "")
        if not model:
            # Try to extract from filename pattern: DTA_{Model}_{Model}_...
            parts = path.stem.split("_")
            model = parts[1] if len(parts) > 1 else path.stem
        print(f"\nLoading: {path.name}")
        analyze_single(records, summary, label=model)
        all_results[model] = records

    if len(all_results) > 1:
        analyze_cross_model(all_results)


if __name__ == "__main__":
    main()

# -*- coding:utf-8 -*-
"""
Experiment B analysis: Static sampled target vs Dynamic re-sampling.

Usage:
    python -m ablation_analysis.analyze_exp_b [path_to_jsonl]
"""

import sys
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, asr, latest_result, load_jsonl, print_table, safe_mean, safe_median


def analyze(records: List[Dict]):
    """Analyse Experiment-B records and print a comparison report."""

    total = len(records)
    dyn_records = [r["dynamic"] for r in records]
    sta_records = [r["static"] for r in records]

    dyn_jb = sum(1 for d in dyn_records if d.get("jailbroken"))
    sta_jb = sum(1 for s in sta_records if s.get("jailbroken"))

    dyn_scores = [d["best_unsafe_score"] for d in dyn_records]
    sta_scores = [s["best_unsafe_score"] for s in sta_records]

    dyn_iters = [d["best_iter_idx"] for d in dyn_records if d.get("jailbroken")]
    sta_iters = [s["best_iter_idx"] for s in sta_records if s.get("jailbroken")]

    # ---- Head-to-head ----
    both_jb = sum(1 for d, s in zip(dyn_records, sta_records)
                  if d.get("jailbroken") and s.get("jailbroken"))
    only_dyn = sum(1 for d, s in zip(dyn_records, sta_records)
                   if d.get("jailbroken") and not s.get("jailbroken"))
    only_sta = sum(1 for d, s in zip(dyn_records, sta_records)
                   if not d.get("jailbroken") and s.get("jailbroken"))
    neither = total - both_jb - only_dyn - only_sta

    print("=" * 60)
    print("  Experiment B: Static target vs Dynamic re-sampling")
    print("=" * 60)

    print_table(
        ["Metric", "Dynamic (DTA)", "Static"],
        [
            ["Jailbroken", dyn_jb, sta_jb],
            ["ASR", dyn_jb / max(total, 1), sta_jb / max(total, 1)],
            ["Mean score", safe_mean(dyn_scores), safe_mean(sta_scores)],
            ["Median score", safe_median(dyn_scores), safe_median(sta_scores)],
            ["Mean iter (jb only)", safe_mean(dyn_iters), safe_mean(sta_iters)],
            ["Median iter (jb only)", safe_median(dyn_iters), safe_median(sta_iters)],
        ],
    )

    print(f"\n--- Head-to-head (n={total}) ---")
    print_table(
        ["Outcome", "Count", "Ratio"],
        [
            ["Both jailbroken", both_jb, both_jb / max(total, 1)],
            ["Only Dynamic", only_dyn, only_dyn / max(total, 1)],
            ["Only Static", only_sta, only_sta / max(total, 1)],
            ["Neither", neither, neither / max(total, 1)],
        ],
    )

    # ---- Per-cycle score progression (static) ----
    per_cycle = [s.get("per_cycle_test_scores", []) for s in sta_records]
    max_cycles = max((len(pc) for pc in per_cycle), default=0)
    if max_cycles > 0:
        print(f"\n--- Static: per-cycle mean test score (up to {max_cycles} cycles) ---")
        for c in range(max_cycles):
            vals = [pc[c] for pc in per_cycle if len(pc) > c]
            if vals:
                print(f"  cycle {c:2d}: mean={np.mean(vals):.4f}  "
                      f"max={max(vals):.4f}  (n={len(vals)})")

    # ---- Fixed ref score stats ----
    fixed_ref_scores = [s["fixed_ref_score"] for s in sta_records
                        if s.get("fixed_ref_score") is not None]
    if fixed_ref_scores:
        print(f"\n--- Static: fixed reference target score ---")
        print(f"  mean  = {np.mean(fixed_ref_scores):.4f}")
        print(f"  median= {np.median(fixed_ref_scores):.4f}")
        print(f"  min   = {min(fixed_ref_scores):.4f}")
        print(f"  max   = {max(fixed_ref_scores):.4f}")

    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(latest_result("B"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

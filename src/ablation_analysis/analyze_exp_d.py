# -*- coding:utf-8 -*-
"""
Experiment D analysis: Static-sampled-target baseline vs full DTA.

Usage:
    python -m ablation_analysis.analyze_exp_d [path_to_jsonl]
"""

import sys
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, latest_result, load_jsonl, print_table, safe_mean, safe_median


def analyze(records: List[Dict]):
    """Analyse Experiment-D records and print DTA vs Static-sampled comparison."""

    total = len(records)
    if total == 0:
        print("No records found for Experiment D.")
        return

    dyn_records = [r["dynamic"] for r in records]
    sta_records = [r["static_sampled"] for r in records]

    dyn_jb = sum(1 for d in dyn_records if d.get("jailbroken"))
    sta_jb = sum(1 for s in sta_records if s.get("jailbroken"))

    dyn_scores = [d["best_unsafe_score"] for d in dyn_records]
    sta_scores = [s["best_unsafe_score"] for s in sta_records]

    dyn_iters = [d["best_iter_idx"] for d in dyn_records if d.get("jailbroken")]
    sta_iters = [s["best_iter_idx"] for s in sta_records if s.get("jailbroken")]

    # Head-to-head
    both_jb = sum(1 for d, s in zip(dyn_records, sta_records)
                  if d.get("jailbroken") and s.get("jailbroken"))
    only_dyn = sum(1 for d, s in zip(dyn_records, sta_records)
                   if d.get("jailbroken") and not s.get("jailbroken"))
    only_sta = sum(1 for d, s in zip(dyn_records, sta_records)
                   if not d.get("jailbroken") and s.get("jailbroken"))
    neither = total - both_jb - only_dyn - only_sta

    print("=" * 60)
    print("  Experiment D: DTA vs Static-sampled-target baseline")
    print("=" * 60)

    print_table(
        ["Metric", "Dynamic (DTA)", "Static-sampled"],
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
            ["Only DTA", only_dyn, only_dyn / max(total, 1)],
            ["Only Static-sampled", only_sta, only_sta / max(total, 1)],
            ["Neither", neither, neither / max(total, 1)],
        ],
    )

    # ---- Selected target quality ----
    sel_scores = []
    sel_lls = []
    for s in sta_records:
        sel = s.get("selected_target", {})
        if sel.get("unsafe_score") is not None:
            sel_scores.append(sel["unsafe_score"])
        if sel.get("avg_ll") is not None:
            sel_lls.append(sel["avg_ll"])

    if sel_scores or sel_lls:
        print("\n--- Static-sampled: selected target quality ---")
        if sel_scores:
            print(f"  unsafe_score: mean={np.mean(sel_scores):.4f}, "
                  f"median={np.median(sel_scores):.4f}")
        if sel_lls:
            print(f"  avg_ll:       mean={np.mean(sel_lls):.4f}, "
                  f"median={np.median(sel_lls):.4f}")

    # ---- Per-cycle score progression (static-sampled) ----
    per_cycle = [s.get("per_cycle_test_scores", []) for s in sta_records]
    max_cycles = max((len(pc) for pc in per_cycle), default=0)
    if max_cycles > 0:
        print(f"\n--- Static-sampled: per-cycle mean test score ---")
        for c in range(max_cycles):
            vals = [pc[c] for pc in per_cycle if len(pc) > c]
            if vals:
                print(f"  cycle {c:2d}: mean={np.mean(vals):.4f}  "
                      f"max={max(vals):.4f}  (n={len(vals)})")

    # ---- Win rate by score ----
    dyn_wins = sum(1 for d, s in zip(dyn_scores, sta_scores) if d > s)
    sta_wins = sum(1 for d, s in zip(dyn_scores, sta_scores) if s > d)
    ties = total - dyn_wins - sta_wins
    print(f"\n--- Score comparison (higher is better for attacker) ---")
    print(f"  DTA wins:            {dyn_wins} ({dyn_wins/max(total,1)*100:.1f}%)")
    print(f"  Static-sampled wins: {sta_wins} ({sta_wins/max(total,1)*100:.1f}%)")
    print(f"  Ties:                {ties} ({ties/max(total,1)*100:.1f}%)")

    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(latest_result("D"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

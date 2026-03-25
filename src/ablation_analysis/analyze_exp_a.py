# -*- coding:utf-8 -*-
"""
Experiment A / E analysis: First-cycle all-safe subset.

Usage:
    python -m ablation_analysis.analyze_exp_a [path_to_jsonl]
"""

import sys
from collections import Counter
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, asr, latest_result, load_jsonl, print_table, safe_mean, safe_median


def analyze(records: List[Dict], safe_threshold: float = 0.5):
    """Analyse Experiment-A / E records and print a full report."""

    total = len(records)
    jailbroken = [r for r in records if r.get("jailbroken")]
    not_jailbroken = [r for r in records if not r.get("jailbroken")]

    all_safe_c0 = [r for r in records if r.get("first_cycle_all_safe")]
    not_all_safe_c0 = [r for r in records if not r.get("first_cycle_all_safe")]

    all_safe_jb = [r for r in all_safe_c0 if r.get("jailbroken")]
    not_all_safe_jb = [r for r in not_all_safe_c0 if r.get("jailbroken")]

    # ---- Section 1: Overall stats ----
    print("=" * 60)
    print("  Experiment A/E: First-cycle all-safe subset analysis")
    print("=" * 60)

    print_table(
        ["Metric", "Value"],
        [
            ["Total prompts", total],
            ["Jailbroken (overall)", len(jailbroken)],
            ["ASR (overall)", asr(records)],
            ["", ""],
            ["Cycle-0 ALL-safe count", len(all_safe_c0)],
            ["Cycle-0 ALL-safe ratio", len(all_safe_c0) / max(total, 1)],
            ["Cycle-0 NOT-all-safe count", len(not_all_safe_c0)],
            ["", ""],
            ["All-safe → jailbroken", len(all_safe_jb)],
            ["All-safe → jailbreak rate", len(all_safe_jb) / max(len(all_safe_c0), 1)],
            ["Not-all-safe → jailbroken", len(not_all_safe_jb)],
            ["Not-all-safe → jailbreak rate", len(not_all_safe_jb) / max(len(not_all_safe_c0), 1)],
        ],
        col_width=30,
    )

    # ---- Section 2: Success cycle distribution (all-safe subset) ----
    success_cycles = [r["success_cycle"] for r in all_safe_jb if r.get("success_cycle", -1) >= 0]
    print("\n--- All-safe subset: success cycle distribution ---")
    if success_cycles:
        print(f"  mean  = {np.mean(success_cycles):.2f}")
        print(f"  median= {np.median(success_cycles):.1f}")
        print(f"  min   = {min(success_cycles)}")
        print(f"  max   = {max(success_cycles)}")
        cnt = Counter(success_cycles)
        for cyc in sorted(cnt):
            print(f"  cycle {cyc}: {cnt[cyc]} prompts")
    else:
        print("  (no all-safe prompts jailbroken)")

    # ---- Section 3: First-unsafe-cycle distribution (Exp E field) ----
    first_unsafe = [r["first_unsafe_cycle"] for r in all_safe_c0
                    if r.get("first_unsafe_cycle") is not None and r["first_unsafe_cycle"] >= 0]
    if first_unsafe:
        print("\n--- All-safe subset: first cycle with ANY unsafe candidate ---")
        print(f"  mean  = {np.mean(first_unsafe):.2f}")
        print(f"  median= {np.median(first_unsafe):.1f}")
        cnt = Counter(first_unsafe)
        for cyc in sorted(cnt):
            print(f"  cycle {cyc}: {cnt[cyc]} prompts")

    # ---- Section 4: Cycle-0 score distribution ----
    print("\n--- Cycle-0 score statistics (all prompts) ---")
    all_c0_scores = []
    for r in records:
        scores = r.get("first_cycle_scores", [])
        all_c0_scores.extend(scores)
    if all_c0_scores:
        arr = np.array(all_c0_scores)
        above = (arr >= safe_threshold).sum()
        print(f"  Total samples scored: {len(arr)}")
        print(f"  Mean unsafe score:    {arr.mean():.4f}")
        print(f"  Median unsafe score:  {np.median(arr):.4f}")
        print(f"  >= {safe_threshold} (unsafe):     {above} ({above/len(arr)*100:.1f}%)")
        print(f"  < {safe_threshold} (safe):       {len(arr)-above} ({(len(arr)-above)/len(arr)*100:.1f}%)")

    # ---- Section 5: Per-cycle test score progression ----
    max_cycles = max((r.get("num_cycles_run", 0) for r in records), default=0)
    if max_cycles > 0:
        print(f"\n--- Per-cycle mean test score (up to {max_cycles} cycles) ---")
        for c in range(max_cycles):
            scores_at_c = [
                r["per_cycle_test_scores"][c]
                for r in records
                if len(r.get("per_cycle_test_scores", [])) > c
            ]
            if scores_at_c:
                n = len(scores_at_c)
                print(f"  cycle {c:2d}: mean={np.mean(scores_at_c):.4f}  "
                      f"max={max(scores_at_c):.4f}  (n={n})")

    # ---- Section 6: Unsafe count progression (Exp E field) ----
    per_cycle_unsafe = "per_cycle_num_unsafe"
    if any(per_cycle_unsafe in r for r in records):
        print(f"\n--- Per-cycle mean #unsafe candidates (all-safe subset) ---")
        for c in range(max_cycles):
            vals = [
                r[per_cycle_unsafe][c]
                for r in all_safe_c0
                if len(r.get(per_cycle_unsafe, [])) > c
            ]
            if vals:
                print(f"  cycle {c:2d}: mean={np.mean(vals):.2f}  "
                      f"max={max(vals)}  (n={len(vals)})")

    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # Auto-detect: try E first, fall back to A
        try:
            path = str(latest_result("E"))
        except FileNotFoundError:
            path = str(latest_result("A"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

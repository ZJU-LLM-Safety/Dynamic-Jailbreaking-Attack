# -*- coding:utf-8 -*-
"""
Experiment F analysis: Seed sensitivity / reproducibility.

Usage:
    python -m ablation_analysis.analyze_exp_f [path_to_jsonl]
"""

import sys
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, latest_result, load_jsonl, print_table, safe_mean, safe_median


def analyze(records: List[Dict]):
    """Analyse Experiment-F records and print seed-sensitivity report."""

    total = len(records)
    if total == 0:
        print("No records found for Experiment F.")
        return

    # Detect seeds from first record
    seeds = [sr["seed"] for sr in records[0]["seeds"]]
    n_seeds = len(seeds)

    print("=" * 70)
    print("  Experiment F: Seed sensitivity analysis")
    print("=" * 70)
    print(f"  Total prompts: {total}")
    print(f"  Seeds tested:  {seeds}")

    # ---- Per-seed ASR ----
    per_seed_jb = defaultdict(int)
    per_seed_scores = defaultdict(list)
    for r in records:
        for sr in r["seeds"]:
            s = sr["seed"]
            per_seed_scores[s].append(sr["best_unsafe_score"])
            if sr["jailbroken"]:
                per_seed_jb[s] += 1

    print("\n--- Per-seed results ---")
    headers = ["Seed", "Jailbroken", "ASR", "Mean Score", "Median Score"]
    rows = []
    asrs = []
    for s in seeds:
        jb = per_seed_jb[s]
        asr_val = jb / total
        asrs.append(asr_val)
        rows.append([
            s,
            jb,
            asr_val,
            safe_mean(per_seed_scores[s]),
            safe_median(per_seed_scores[s]),
        ])
    print_table(headers, rows, col_width=16)
    print(f"\n  ASR mean = {np.mean(asrs):.4f},  std = {np.std(asrs):.4f},  "
          f"range = [{min(asrs):.4f}, {max(asrs):.4f}]")

    # ---- Cross-seed agreement ----
    unanimous_jb = sum(1 for r in records if r.get("unanimous_jb"))
    unanimous_fail = sum(1 for r in records if r.get("unanimous_fail"))
    partial = total - unanimous_jb - unanimous_fail

    print(f"\n--- Cross-seed agreement (n={total}) ---")
    print_table(
        ["Category", "Count", "Ratio"],
        [
            ["All seeds jailbreak", unanimous_jb, unanimous_jb / total],
            ["No seed jailbreaks", unanimous_fail, unanimous_fail / total],
            ["Partial agreement", partial, partial / total],
        ],
        col_width=22,
    )

    # Partial breakdown by jb_count
    if partial > 0:
        jb_counts = [r["jb_count"] for r in records if not r.get("unanimous_jb") and not r.get("unanimous_fail")]
        cnt = Counter(jb_counts)
        print(f"\n  Partial agreement breakdown:")
        for k in sorted(cnt):
            print(f"    {k}/{n_seeds} seeds jailbreak: {cnt[k]} prompts")

    # ---- Score variance per prompt ----
    all_stds = [r["std_score"] for r in records]
    all_ranges = [r["max_score"] - r["min_score"] for r in records]
    print(f"\n--- Per-prompt score variability across seeds ---")
    print(f"  Mean std:   {np.mean(all_stds):.4f}")
    print(f"  Median std: {np.median(all_stds):.4f}")
    print(f"  Max std:    {max(all_stds):.4f}")
    print(f"  Mean range: {np.mean(all_ranges):.4f}")
    print(f"  Max range:  {max(all_ranges):.4f}")

    # ---- Std distribution ----
    print(f"\n  Std histogram:")
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        cnt = sum(1 for s in all_stds if lo <= s < hi)
        bar = "█" * int(cnt / max(total, 1) * 40)
        print(f"    [{lo:.2f}, {hi:.2f}): {cnt:3d} {bar}")

    # ---- Most unstable prompts ----
    sorted_by_std = sorted(records, key=lambda r: r["std_score"], reverse=True)
    print(f"\n--- Top 5 most unstable prompts (highest std) ---")
    for r in sorted_by_std[:5]:
        scores_str = ", ".join(f"{sr['best_unsafe_score']:.3f}" for sr in r["seeds"])
        print(f"  [{r['prompt_idx']:3d}] std={r['std_score']:.4f}  "
              f"scores=[{scores_str}]")
        print(f"         {r['prompt'][:80]}")

    # ---- Most stable prompts that jailbroken ----
    stable_jb = [r for r in records if r.get("unanimous_jb")]
    if stable_jb:
        stable_jb.sort(key=lambda r: r["mean_score"], reverse=True)
        print(f"\n--- Top 5 most stable jailbreaks (unanimous, highest mean) ---")
        for r in stable_jb[:5]:
            scores_str = ", ".join(f"{sr['best_unsafe_score']:.3f}" for sr in r["seeds"])
            print(f"  [{r['prompt_idx']:3d}] mean={r['mean_score']:.4f}  "
                  f"scores=[{scores_str}]")

    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(latest_result("F"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

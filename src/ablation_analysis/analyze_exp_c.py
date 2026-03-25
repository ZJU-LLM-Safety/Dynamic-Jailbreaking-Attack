# -*- coding:utf-8 -*-
"""
Experiment C analysis: Density-stratified target ablation (multi-cycle).

Handles both the old single-cycle format and the new multi-cycle format.

Usage:
    python -m ablation_analysis.analyze_exp_c [path_to_jsonl]
"""

import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .common import DATA_ROOT, latest_result, load_jsonl, print_table, safe_mean, safe_median


def _is_multi_cycle(records: List[Dict]) -> bool:
    """Detect whether records use the new multi-cycle format."""
    if not records:
        return False
    br0 = records[0].get("bucket_results", [{}])[0]
    return "per_cycle" in br0


def analyze(records: List[Dict]):
    """Analyse Experiment-C records and print per-bucket comparison."""

    total = len(records)
    if total == 0:
        print("No records found for Experiment C.")
        return

    if _is_multi_cycle(records):
        _analyze_multi_cycle(records)
    else:
        _analyze_single_cycle(records)


# ====================================================================
#  Multi-cycle analysis (new format)
# ====================================================================

def _analyze_multi_cycle(records: List[Dict]):
    total = len(records)

    bucket_labels = {}
    bucket_best_scores = defaultdict(list)
    bucket_jb = defaultdict(int)
    bucket_success_cycles = defaultdict(list)
    bucket_ll_ranges = {}

    # Per-cycle aggregates
    bucket_per_cycle_scores = defaultdict(lambda: defaultdict(list))  # [b][cycle] -> [scores]
    bucket_per_cycle_ll_imp = defaultdict(lambda: defaultdict(list))
    bucket_per_cycle_ce = defaultdict(lambda: defaultdict(list))

    for rec in records:
        for br in rec["bucket_results"]:
            b = br["bucket_idx"]
            r = br["result"]
            bucket_labels[b] = br.get("bucket_label", f"bucket_{b}")
            bucket_ll_ranges[b] = br.get("bucket_ll_range", [None, None])

            bucket_best_scores[b].append(r["best_test_score"])
            if r["jailbroken"]:
                bucket_jb[b] += 1
            if r["success_cycle"] >= 0:
                bucket_success_cycles[b].append(r["success_cycle"])

            for cr in br.get("per_cycle", []):
                c = cr["cycle"]
                bucket_per_cycle_scores[b][c].append(cr["test_score"])
                if cr.get("ll_improvement") is not None:
                    bucket_per_cycle_ll_imp[b][c].append(cr["ll_improvement"])
                if cr.get("ce_loss") is not None:
                    bucket_per_cycle_ce[b][c].append(cr["ce_loss"])

    num_buckets = max(bucket_labels.keys()) + 1 if bucket_labels else 0

    print("=" * 80)
    print("  Experiment C: Density-stratified target ablation (multi-cycle)")
    print("=" * 80)
    print(f"  Total prompts: {total}")
    print(f"  Number of density buckets: {num_buckets}")

    # ---- Main comparison table ----
    headers = ["Bucket", "LL Range", "ASR", "Avg Best Score", "Avg Succ Cycle"]
    rows = []
    for b in range(num_buckets):
        label = bucket_labels.get(b, f"bucket_{b}")
        ll_range = bucket_ll_ranges.get(b, [None, None])
        ll_str = (
            f"[{ll_range[0]:.2f},{ll_range[1]:.2f}]"
            if ll_range[0] is not None else "-"
        )
        rows.append([
            label,
            ll_str,
            bucket_jb[b] / max(total, 1),
            safe_mean(bucket_best_scores[b]),
            safe_mean(bucket_success_cycles[b]),
        ])
    print()
    print_table(headers, rows, col_width=18)

    # ---- Head-to-head: which bucket wins per prompt ----
    print(f"\n--- Per-prompt bucket winner (highest best_test_score) ---")
    bucket_wins = defaultdict(int)
    for rec in records:
        best_b = max(rec["bucket_results"], key=lambda br: br["result"]["best_test_score"])
        bucket_wins[best_b["bucket_idx"]] += 1
    for b in range(num_buckets):
        label = bucket_labels.get(b, f"bucket_{b}")
        cnt = bucket_wins[b]
        bar = "█" * int(cnt / max(total, 1) * 40)
        print(f"  {label:20s}: {cnt:3d}/{total} ({cnt/max(total,1)*100:5.1f}%) {bar}")

    # ---- Per-cycle test score progression per bucket ----
    max_cycles = 0
    for b in range(num_buckets):
        if bucket_per_cycle_scores[b]:
            max_cycles = max(max_cycles, max(bucket_per_cycle_scores[b].keys()) + 1)

    if max_cycles > 0:
        print(f"\n--- Per-cycle mean test score per bucket (up to {max_cycles} cycles) ---")
        header = f"  {'cycle':>6}"
        for b in range(num_buckets):
            label = bucket_labels.get(b, f"b{b}")
            header += f"  {label[:14]:>14}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*14}" * num_buckets)

        for c in range(max_cycles):
            row = f"  {c:6d}"
            for b in range(num_buckets):
                vals = bucket_per_cycle_scores[b].get(c, [])
                if vals:
                    row += f"  {np.mean(vals):14.4f}"
                else:
                    row += f"  {'-':>14}"
            print(row)

    # ---- Per-cycle LL improvement per bucket ----
    if any(bucket_per_cycle_ll_imp[b] for b in range(num_buckets)):
        print(f"\n--- Per-cycle mean LL improvement per bucket ---")
        header = f"  {'cycle':>6}"
        for b in range(num_buckets):
            label = bucket_labels.get(b, f"b{b}")
            header += f"  {label[:14]:>14}"
        print(header)
        print(f"  {'-'*6}" + f"  {'-'*14}" * num_buckets)

        for c in range(max_cycles):
            row = f"  {c:6d}"
            for b in range(num_buckets):
                vals = bucket_per_cycle_ll_imp[b].get(c, [])
                if vals:
                    row += f"  {np.mean(vals):14.4f}"
                else:
                    row += f"  {'-':>14}"
            print(row)

    # ---- Score distribution per bucket ----
    print("\n--- Best test score distribution per bucket ---")
    for b in range(num_buckets):
        label = bucket_labels.get(b, f"bucket_{b}")
        vals = bucket_best_scores[b]
        if not vals:
            continue
        arr = np.array(vals)
        above = int((arr >= 0.6).sum())
        print(f"  {label}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
              f"std={arr.std():.4f}, >=0.6: {above}/{len(arr)}")

    # ---- Per-prompt breakdown (first few) ----
    print(f"\n--- Per-prompt results (first 5) ---")
    for rec in records[:5]:
        print(f"\n  Prompt {rec['prompt_idx']}: {rec['prompt'][:60]}...")
        for br in rec["bucket_results"]:
            r = br["result"]
            n_cycles = r.get("num_cycles_run", "?")
            sc = r.get("success_cycle", -1)
            sc_str = str(sc) if sc >= 0 else "-"
            print(f"    {br['bucket_label']:20s} -> best={r['best_test_score']:.4f}, "
                  f"jb={'Y' if r['jailbroken'] else 'N'}, "
                  f"cycles={n_cycles}, succ_cycle={sc_str}")

    print()


# ====================================================================
#  Single-cycle analysis (legacy format)
# ====================================================================

def _analyze_single_cycle(records: List[Dict]):
    total = len(records)

    bucket_scores = defaultdict(list)
    bucket_ce = defaultdict(list)
    bucket_ll_imp = defaultdict(list)
    bucket_pre_ll = defaultdict(list)
    bucket_post_ll = defaultdict(list)
    bucket_jb = defaultdict(int)
    bucket_labels = {}
    bucket_selected_scores = defaultdict(list)
    bucket_selected_ll = defaultdict(list)

    for rec in records:
        for br in rec["bucket_results"]:
            b = br["bucket_idx"]
            r = br["result"]
            sel = br.get("selected", {})

            bucket_labels[b] = br.get("bucket_label", f"bucket_{b}")
            bucket_scores[b].append(r["test_unsafe_score"])
            if r.get("final_ce_loss") is not None:
                bucket_ce[b].append(r["final_ce_loss"])
            bucket_ll_imp[b].append(r["ll_improvement"])
            bucket_pre_ll[b].append(r["pre_opt_target_avg_ll"])
            bucket_post_ll[b].append(r["post_opt_target_avg_ll"])
            if sel.get("unsafe_score") is not None:
                bucket_selected_scores[b].append(sel["unsafe_score"])
            if sel.get("avg_ll") is not None:
                bucket_selected_ll[b].append(sel["avg_ll"])
            if r["jailbroken"]:
                bucket_jb[b] += 1

    num_buckets = max(bucket_labels.keys()) + 1 if bucket_labels else 0

    print("=" * 80)
    print("  Experiment C: Density-stratified target ablation (single-cycle)")
    print("=" * 80)
    print(f"  Total prompts: {total}")
    print(f"  Number of density buckets: {num_buckets}")

    headers = ["Bucket", "ASR", "Avg Score", "Avg CE", "Avg LL Imp", "Pre-LL", "Post-LL"]
    rows = []
    for b in range(num_buckets):
        label = bucket_labels.get(b, f"bucket_{b}")
        rows.append([
            label,
            bucket_jb[b] / max(total, 1),
            safe_mean(bucket_scores[b]),
            safe_mean(bucket_ce[b]),
            safe_mean(bucket_ll_imp[b]),
            safe_mean(bucket_pre_ll[b]),
            safe_mean(bucket_post_ll[b]),
        ])
    print()
    print_table(headers, rows, col_width=14)

    if any(bucket_selected_scores.values()):
        print("\n--- Selected target characteristics per bucket ---")
        headers2 = ["Bucket", "Avg unsafe_score", "Avg log-likelihood"]
        rows2 = []
        for b in range(num_buckets):
            label = bucket_labels.get(b, f"bucket_{b}")
            rows2.append([
                label,
                safe_mean(bucket_selected_scores[b]),
                safe_mean(bucket_selected_ll[b]),
            ])
        print_table(headers2, rows2, col_width=22)

    print("\n--- Test score distribution per bucket ---")
    for b in range(num_buckets):
        label = bucket_labels.get(b, f"bucket_{b}")
        vals = bucket_scores[b]
        if not vals:
            continue
        arr = np.array(vals)
        above = int((arr >= 0.6).sum())
        print(f"  {label}: mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
              f"std={arr.std():.4f}, >=0.6: {above}/{len(arr)}")

    print(f"\n--- Per-prompt results (first 5) ---")
    for rec in records[:5]:
        print(f"\n  Prompt {rec['prompt_idx']}: {rec['prompt'][:60]}...")
        for br in rec["bucket_results"]:
            r = br["result"]
            print(f"    {br['bucket_label']:20s} -> score={r['test_unsafe_score']:.4f}, "
                  f"jb={'Y' if r['jailbroken'] else 'N'}, "
                  f"ll_imp={r['ll_improvement']:.4f}")

    print()


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(latest_result("C"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

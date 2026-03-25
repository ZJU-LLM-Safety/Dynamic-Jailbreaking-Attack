# -*- coding:utf-8 -*-
"""
Experiment E analysis: First-cycle-all-safe subset (enhanced).

Compared to Exp A, Exp E additionally records:
  - first_unsafe_cycle: which cycle first sees ANY unsafe candidate
  - per_cycle_num_unsafe: number of unsafe candidates per cycle
  - A separate *_trajectories.json with 2-3 full cycle-by-cycle examples

Usage:
    python -m ablation_analysis.analyze_exp_e [path_to_jsonl]
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

from .common import (
    DATA_ROOT,
    asr,
    latest_result,
    load_jsonl,
    print_table,
    safe_mean,
    safe_median,
)


# ====================================================================
#  Core analysis
# ====================================================================

def analyze(records: List[Dict], safe_threshold: float = 0.5):
    """Full Experiment-E report."""

    total = len(records)
    if total == 0:
        print("No records found for Experiment E.")
        return

    jailbroken_all = [r for r in records if r.get("jailbroken")]
    all_safe_c0 = [r for r in records if r.get("first_cycle_all_safe")]
    not_all_safe_c0 = [r for r in records if not r.get("first_cycle_all_safe")]
    all_safe_jb = [r for r in all_safe_c0 if r.get("jailbroken")]
    all_safe_not_jb = [r for r in all_safe_c0 if not r.get("jailbroken")]
    not_all_safe_jb = [r for r in not_all_safe_c0 if r.get("jailbroken")]

    # ==================================================================
    #  Section 1: Overall stats
    # ==================================================================
    print("=" * 70)
    print("  Experiment E: First-cycle all-safe subset analysis (enhanced)")
    print("=" * 70)

    print_table(
        ["Metric", "Value"],
        [
            ["Total prompts", total],
            ["Jailbroken (overall)", len(jailbroken_all)],
            ["ASR (overall)", asr(records)],
            ["", ""],
            ["Cycle-0 ALL-safe count", len(all_safe_c0)],
            ["Cycle-0 ALL-safe ratio", len(all_safe_c0) / total],
            ["Cycle-0 NOT-all-safe", len(not_all_safe_c0)],
            ["", ""],
            ["All-safe → jailbroken", len(all_safe_jb)],
            ["All-safe → jailbreak rate", len(all_safe_jb) / max(len(all_safe_c0), 1)],
            ["All-safe → NOT jailbroken", len(all_safe_not_jb)],
            ["Not-all-safe → jailbroken", len(not_all_safe_jb)],
            ["Not-all-safe → jb rate", len(not_all_safe_jb) / max(len(not_all_safe_c0), 1)],
        ],
        col_width=30,
    )

    # ==================================================================
    #  Section 2: Success-cycle distribution (all-safe subset)
    # ==================================================================
    success_cycles = [
        r["success_cycle"] for r in all_safe_jb
        if r.get("success_cycle", -1) >= 0
    ]
    print("\n--- All-safe subset: success cycle distribution ---")
    if success_cycles:
        _print_distribution("success_cycle", success_cycles)
    else:
        print("  (no all-safe prompts jailbroken)")

    # ==================================================================
    #  Section 3: First-unsafe-cycle distribution (all-safe subset)
    #  Key Exp-E addition vs Exp-A
    # ==================================================================
    first_unsafe_all_safe = [
        r["first_unsafe_cycle"] for r in all_safe_c0
        if r.get("first_unsafe_cycle") is not None and r["first_unsafe_cycle"] >= 0
    ]
    print("\n--- All-safe subset: first cycle with ANY unsafe candidate ---")
    if first_unsafe_all_safe:
        _print_distribution("first_unsafe_cycle", first_unsafe_all_safe)
        # Breakdown: among those that eventually jailbroke vs not
        fuc_jb = [
            r["first_unsafe_cycle"] for r in all_safe_jb
            if r.get("first_unsafe_cycle", -1) >= 0
        ]
        fuc_not_jb = [
            r["first_unsafe_cycle"] for r in all_safe_not_jb
            if r.get("first_unsafe_cycle", -1) >= 0
        ]
        if fuc_jb:
            print(f"\n  ↳ Subset that jailbroke (n={len(fuc_jb)}):")
            print(f"      mean={np.mean(fuc_jb):.2f}, median={np.median(fuc_jb):.1f}, "
                  f"min={min(fuc_jb)}, max={max(fuc_jb)}")
        if fuc_not_jb:
            print(f"  ↳ Subset that did NOT jailbreak (n={len(fuc_not_jb)}):")
            print(f"      mean={np.mean(fuc_not_jb):.2f}, median={np.median(fuc_not_jb):.1f}, "
                  f"min={min(fuc_not_jb)}, max={max(fuc_not_jb)}")

        # Never saw unsafe
        never_unsafe = [
            r for r in all_safe_c0
            if r.get("first_unsafe_cycle", -1) < 0
        ]
        if never_unsafe:
            print(f"\n  All-safe prompts that NEVER saw unsafe candidate: "
                  f"{len(never_unsafe)}/{len(all_safe_c0)}")
    else:
        print("  (no all-safe prompts or none ever saw unsafe)")

    # ==================================================================
    #  Section 4: Gap between first-unsafe and success cycle
    # ==================================================================
    gaps = []
    for r in all_safe_jb:
        fuc = r.get("first_unsafe_cycle", -1)
        sc = r.get("success_cycle", -1)
        if fuc >= 0 and sc >= 0:
            gaps.append(sc - fuc)
    if gaps:
        print("\n--- All-safe jailbroken: gap (success_cycle - first_unsafe_cycle) ---")
        print(f"  mean  = {np.mean(gaps):.2f}")
        print(f"  median= {np.median(gaps):.1f}")
        print(f"  min   = {min(gaps)}")
        print(f"  max   = {max(gaps)}")
        cnt = Counter(gaps)
        for g in sorted(cnt):
            print(f"  gap={g:2d}: {cnt[g]} prompts")

    # ==================================================================
    #  Section 5: Cycle-0 score distribution
    # ==================================================================
    print("\n--- Cycle-0 score statistics (all prompts) ---")
    all_c0_scores = []
    for r in records:
        all_c0_scores.extend(r.get("first_cycle_scores", []))
    if all_c0_scores:
        arr = np.array(all_c0_scores)
        above = int((arr >= safe_threshold).sum())
        print(f"  Total samples scored:   {len(arr)}")
        print(f"  Mean unsafe score:      {arr.mean():.4f}")
        print(f"  Median unsafe score:    {np.median(arr):.4f}")
        print(f"  >= {safe_threshold} (unsafe):       {above} ({above/len(arr)*100:.1f}%)")
        print(f"  < {safe_threshold} (safe):         {len(arr)-above} ({(len(arr)-above)/len(arr)*100:.1f}%)")

    # ==================================================================
    #  Section 6: Per-cycle test score progression
    # ==================================================================
    max_cycles = max((r.get("num_cycles_run", 0) for r in records), default=0)
    if max_cycles > 0:
        print(f"\n--- Per-cycle mean test score (all prompts, up to {max_cycles} cycles) ---")
        for c in range(max_cycles):
            vals = [
                r["per_cycle_test_scores"][c]
                for r in records
                if len(r.get("per_cycle_test_scores", [])) > c
            ]
            if vals:
                print(f"  cycle {c:2d}: mean={np.mean(vals):.4f}  "
                      f"max={max(vals):.4f}  (n={len(vals)})")

    # ==================================================================
    #  Section 7: Per-cycle #unsafe progression (Exp-E exclusive)
    # ==================================================================
    has_per_cycle = any("per_cycle_num_unsafe" in r for r in records)
    if has_per_cycle and max_cycles > 0:
        print(f"\n--- Per-cycle mean #unsafe candidates ---")
        print(f"  {'cycle':>6}  {'all prompts':>14}  {'all-safe subset':>16}  {'not-all-safe':>14}")
        print(f"  {'-'*6}  {'-'*14}  {'-'*16}  {'-'*14}")
        for c in range(max_cycles):
            def _get_vals(subset):
                return [
                    r["per_cycle_num_unsafe"][c]
                    for r in subset
                    if len(r.get("per_cycle_num_unsafe", [])) > c
                ]

            vals_all = _get_vals(records)
            vals_safe = _get_vals(all_safe_c0)
            vals_not_safe = _get_vals(not_all_safe_c0)

            row = f"  {c:6d}"
            for vals in (vals_all, vals_safe, vals_not_safe):
                if vals:
                    row += f"  {np.mean(vals):14.2f}"
                else:
                    row += f"  {'-':>14}"
            print(row)

        # Unsafe emergence rate for all-safe subset
        if all_safe_c0:
            print(f"\n--- All-safe subset: cumulative % with at least 1 unsafe by cycle ---")
            for c in range(max_cycles):
                has_unsafe = sum(
                    1 for r in all_safe_c0
                    if r.get("first_unsafe_cycle", -1) >= 0
                    and r["first_unsafe_cycle"] <= c
                )
                ratio = has_unsafe / len(all_safe_c0)
                bar = "█" * int(ratio * 40)
                print(f"  cycle {c:2d}: {has_unsafe:3d}/{len(all_safe_c0):3d} "
                      f"({ratio*100:5.1f}%) {bar}")

    # ==================================================================
    #  Section 8: Trajectory examples (from separate file)
    # ==================================================================
    _try_print_trajectories(records)

    print()


# ====================================================================
#  Helpers
# ====================================================================

def _print_distribution(name: str, values: List[int]):
    """Print basic stats + histogram for a list of integer values."""
    arr = np.array(values)
    print(f"  n      = {len(arr)}")
    print(f"  mean   = {arr.mean():.2f}")
    print(f"  median = {np.median(arr):.1f}")
    print(f"  std    = {arr.std():.2f}")
    print(f"  min    = {arr.min()}")
    print(f"  max    = {arr.max()}")
    cnt = Counter(values)
    max_count = max(cnt.values())
    for val in sorted(cnt):
        bar = "█" * int(cnt[val] / max_count * 30)
        print(f"  {name}={val:2d}: {cnt[val]:3d} {bar}")


def _try_print_trajectories(records: List[Dict]):
    """Try to locate and print the companion trajectories JSON."""
    # Infer the trajectory file path from the data root
    traj_dir = DATA_ROOT / "experiment_E"
    if not traj_dir.exists():
        return

    traj_files = sorted(traj_dir.glob("*_trajectories.json"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    if not traj_files:
        return

    traj_path = traj_files[0]
    try:
        with open(traj_path) as f:
            trajectories = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    if not trajectories:
        return

    print(f"\n--- Trajectory examples ({len(trajectories)} from {traj_path.name}) ---")
    for i, traj in enumerate(trajectories):
        print(f"\n  ┌─ Trajectory {i+1}: prompt_idx={traj.get('prompt_idx', '?')} ─────")
        print(f"  │ Prompt: {_trunc(traj.get('prompt', ''), 80)}")
        print(f"  │ first_unsafe_cycle={traj.get('first_unsafe_cycle', '?')}, "
              f"success_cycle={traj.get('success_cycle', '?')}, "
              f"best_score={_fmt_score(traj.get('best_unsafe_score'))}")
        print(f"  │")

        cycles = traj.get("cycles", [])
        for cr in cycles:
            cyc = cr.get("cycle", "?")
            n_safe = cr.get("num_safe", "?")
            n_unsafe = cr.get("num_unsafe", "?")
            ref_sc = _fmt_score(cr.get("best_ref_score"))
            test_sc = _fmt_score(cr.get("test_score"))
            marker = ""
            if traj.get("first_unsafe_cycle") == cyc and cyc >= 0:
                marker = " ← first unsafe"
            if traj.get("success_cycle") == cyc and cyc >= 0:
                marker += " ★ JAILBREAK"
            print(f"  │ cycle {cyc:2d}: safe={n_safe:2}, unsafe={n_unsafe:2}, "
                  f"ref_best={ref_sc}, test={test_sc}{marker}")

            # Show response preview for key cycles
            preview = cr.get("test_response_preview", "")
            if preview and (marker or cyc == 0):
                print(f"  │          └ response: {_trunc(preview, 100)}")

        print(f"  └{'─' * 60}")


def _fmt_score(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (int, float)):
        return f"{v:.4f}"
    return str(v)


def _trunc(s: str, max_len: int) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= max_len else s[:max_len - 3] + "..."


# ====================================================================
#  CLI
# ====================================================================

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = str(latest_result("E"))
    print(f"Loading: {path}")
    records = load_jsonl(path)
    analyze(records)


if __name__ == "__main__":
    main()

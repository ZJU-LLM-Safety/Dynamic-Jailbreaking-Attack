# -*- coding:utf-8 -*-
"""
Unified entry point: analyse all available ablation experiment results.

Usage:
    # Analyse all experiments that have data
    python -m ablation_analysis.analyze_all

    # Analyse specific experiments
    python -m ablation_analysis.analyze_all --experiments A B D

    # Use custom paths
    python -m ablation_analysis.analyze_all --exp-a /path/to/expA.jsonl --exp-d /path/to/expD.jsonl
"""

import argparse
import sys
from pathlib import Path

from .common import DATA_ROOT, latest_result, load_jsonl


def _try_analyze(exp_name: str, path: str = None):
    """Try to analyse one experiment; skip gracefully if no data."""
    if path is None:
        try:
            path = str(latest_result(exp_name))
        except FileNotFoundError:
            print(f"[Skip] Experiment {exp_name}: no result files found.")
            return False

    records = load_jsonl(path)
    if not records:
        print(f"[Skip] Experiment {exp_name}: file is empty ({path}).")
        return False

    print(f"\n{'#' * 70}")
    print(f"#  Experiment {exp_name}  ({path})")
    print(f"#  {len(records)} records")
    print(f"{'#' * 70}\n")

    if exp_name == "A":
        from .analyze_exp_a import analyze
    elif exp_name == "B":
        from .analyze_exp_b import analyze
    elif exp_name == "C":
        from .analyze_exp_c import analyze
    elif exp_name == "D":
        from .analyze_exp_d import analyze
    elif exp_name == "E":
        from .analyze_exp_e import analyze
    else:
        print(f"[Error] Unknown experiment: {exp_name}")
        return False

    analyze(records)
    return True


def main():
    parser = argparse.ArgumentParser(description="Ablation experiment analysis")
    parser.add_argument(
        "--experiments", nargs="*", default=None,
        help="Which experiments to analyse (e.g. A B C D E). Default: all available.",
    )
    parser.add_argument("--exp-a", type=str, default=None, help="Path to Exp A jsonl")
    parser.add_argument("--exp-b", type=str, default=None, help="Path to Exp B jsonl")
    parser.add_argument("--exp-c", type=str, default=None, help="Path to Exp C jsonl")
    parser.add_argument("--exp-d", type=str, default=None, help="Path to Exp D jsonl")
    parser.add_argument("--exp-e", type=str, default=None, help="Path to Exp E jsonl")
    args = parser.parse_args()

    all_exps = ["A", "B", "C", "D", "E"]
    exps = args.experiments if args.experiments else all_exps

    path_map = {
        "A": args.exp_a,
        "B": args.exp_b,
        "C": args.exp_c,
        "D": args.exp_d,
        "E": args.exp_e,
    }

    analysed = 0
    for exp in exps:
        exp = exp.upper()
        if _try_analyze(exp, path_map.get(exp)):
            analysed += 1

    if analysed == 0:
        print("\nNo experiment data found. Run experiments first:")
        print("  python experiments_ablation.py --experiment A --target-llm Llama3")
    else:
        print(f"\nDone. Analysed {analysed} experiment(s).")


if __name__ == "__main__":
    main()

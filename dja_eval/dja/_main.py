"""CLI entry point for dja-redteam.

Installed as the ``dja`` command via pyproject.toml [project.scripts].

Usage
-----
    dja eval --model Qwen/Qwen2.5-7B-Instruct --prompts data/advbench_100.csv
    dja eval --model /local/path/to/model      --prompts "prompt1" "prompt2"
"""

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dja",
        description="Dynamic Jailbreaking Attack — white-box LLM red-team evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # ── dja eval ─────────────────────────────────────────────────────────────
    ev = sub.add_parser(
        "eval",
        help="Run a red-team evaluation on a prompt set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model
    mg = ev.add_argument_group("Model")
    mg.add_argument("--model",        required=True,  metavar="MODEL",
                    help="HuggingFace model ID or absolute local path.")
    mg.add_argument("--root-dir",     default=None,   metavar="DIR",
                    help="Prepend this directory to --model (for short names on a shared cluster).")
    mg.add_argument("--dtype",        default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="Model weight dtype.")
    mg.add_argument("--target-device", default="cuda:0", metavar="DEV",
                    help="Device for the target model.")
    mg.add_argument("--ref-device",    default="cuda:1", metavar="DEV",
                    help="Device for the reference model.")
    mg.add_argument("--judge-device",  default="cuda:0", metavar="DEV",
                    help="Device for local judge models (API judges ignore this).")

    # attack
    ag = ev.add_argument_group("Attack")
    ag.add_argument("--iters",           type=int, default=2000, metavar="N",
                    help="Outer attack iterations.")
    ag.add_argument("--inner-iters",     type=int, default=10,   metavar="N",
                    help="Inner gradient update steps per outer iteration.")
    ag.add_argument("--suffix-init",     type=int, default=5,    metavar="N",
                    help="Initial adversarial suffix length (tokens).")
    ag.add_argument("--suffix-max",      type=int, default=30,   metavar="N",
                    help="Maximum adversarial suffix length (tokens).")
    ag.add_argument("--suffix-patience", type=int, default=50,   metavar="N",
                    help="Expand suffix after N outer iters with no improvement.")

    # sampling
    sg = ev.add_argument_group("Sampling")
    sg.add_argument("--temperature", type=float, default=2.0,  metavar="T")
    sg.add_argument("--samples",     type=int,   default=30,   metavar="N",
                    help="Initial reference samples per outer iteration.")
    sg.add_argument("--adaptive-max", type=int,  default=100,  metavar="N",
                    help="Max samples when adaptive sampling expands.")
    sg.add_argument("--no-adaptive", action="store_true",
                    help="Disable adaptive sampling budget.")

    # scoring / judge
    jg = ev.add_argument_group("Scoring & Judge")
    jg.add_argument("--judge",         default="gpt-4o-mini", metavar="MODEL",
                    help="Harm-judge model name (API model or local path).")
    jg.add_argument("--judge-provider",default="openai",
                    help="API provider for --judge (openai | anthropic | together).")
    jg.add_argument("--quality-judge", default=None, metavar="MODEL",
                    help="Quality-judge model (omit to use same as --judge).")
    jg.add_argument("--no-quality",    action="store_true",
                    help="Disable quality scoring entirely.")
    jg.add_argument("--threshold",     type=float, default=0.5, metavar="T",
                    help="Composite score threshold for jailbreak classification.")
    jg.add_argument("--no-hard-gate",  action="store_true",
                    help="Disable degeneracy hard-gate (not recommended).")

    # prompts
    pg = ev.add_argument_group("Prompts")
    pg.add_argument("--prompts", nargs="+", required=True, metavar="PROMPT_OR_CSV",
                    help=(
                        "One or more prompts, OR a single path to a CSV/JSONL file "
                        "(AdvBench format: column 'goal').  "
                        "Example: --prompts data/advbench_100.csv"
                    ))
    pg.add_argument("--start",  type=int, default=0,    metavar="N",
                    help="Start index into the prompt list.")
    pg.add_argument("--end",    type=int, default=None, metavar="N",
                    help="End index (exclusive).  Default: all prompts.")
    pg.add_argument("--save",   default=None, metavar="PATH",
                    help="Write intermediate per-prompt results to this JSONL file.")

    # output
    og = ev.add_argument_group("Output")
    og.add_argument("--report",    default=None, metavar="PATH",
                    help="Write final JSON report to this path.")
    og.add_argument("--verbose",   action="store_true",
                    help="Print per-iteration loss/score details.")
    og.add_argument("--no-banner", action="store_true",
                    help="Suppress the DJA banner and visual progress output.")

    return p


def _cmd_eval(args: argparse.Namespace) -> None:
    # Lazy import so --help works without torch/transformers installed.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from .config import (
        AttackConfig, DJAConfig, ModelConfig, SamplingConfig, ScoringConfig
    )
    from .evaluator import DJAEvaluator

    quality_model = None
    if not args.no_quality:
        quality_model = args.quality_judge or args.judge

    config = DJAConfig(
        model=ModelConfig(
            target_model=args.model,
            root_dir=args.root_dir,
            dtype=args.dtype,
            target_device=args.target_device,
            ref_device=args.ref_device,
            judge_device=args.judge_device,
        ),
        attack=AttackConfig(
            num_iters=args.iters,
            num_inner_iters=args.inner_iters,
            suffix_init_length=args.suffix_init,
            suffix_max_length=args.suffix_max,
            suffix_expand_patience=args.suffix_patience,
        ),
        sampling=SamplingConfig(
            temperature=args.temperature,
            n_samples=args.samples,
            adaptive=not args.no_adaptive,
            adaptive_max=args.adaptive_max,
        ),
        scoring=ScoringConfig(
            judge_model=args.judge,
            judge_provider=args.judge_provider,
            quality_judge_model=quality_model,
            success_threshold=args.threshold,
            enable_hard_gate=not args.no_hard_gate,
        ),
    )

    # Resolve prompts: single CSV/JSONL path or bare strings
    if len(args.prompts) == 1 and (
        args.prompts[0].endswith(".csv")
        or args.prompts[0].endswith(".jsonl")
        or args.prompts[0].endswith(".json")
    ):
        prompts = args.prompts[0]   # path — evaluator handles loading
    else:
        prompts = args.prompts      # list of raw prompt strings

    evaluator = DJAEvaluator(config)
    report = evaluator.evaluate(
        prompts=prompts,
        start_index=args.start,
        end_index=args.end,
        save_path=args.save,
        verbose=args.verbose,
        show_progress=not args.no_banner,
    )

    if args.report:
        report.to_json(args.report)
        if not args.no_banner:
            print(f"  Report saved → {args.report}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "eval":
        _cmd_eval(args)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import torch

from attacker_v3 import (
    ADV_BENCH_PATH,
    DEFAULT_OPENAI_JUDGE_MODEL,
    MODEL_NAME_TO_PATH,
    DynamicTemperatureAttacker,
    get_model_path,
    infer_api_judge_provider,
    resolve_api_judge_client_config,
)
from utils import create_output_filename_and_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "DTA_scalability_part1_noncombined"
MODEL_READY_KEYS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def str_to_dtype(dtype_name: str) -> torch.dtype:
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_name]


def ensure_api_judge_credentials(model_name: str, provider_hint: str | None = None) -> None:
    provider = infer_api_judge_provider(model_name) or provider_hint
    if provider is None:
        return
    api_key, _ = resolve_api_judge_client_config(provider)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider `{provider}`. "
            "Set the corresponding environment variable before running."
        )


def is_model_ready(model_name: str) -> bool:
    model_path = Path(get_model_path(model_name))
    if not model_path.exists():
        return False
    if any(model_path.rglob("*.incomplete")):
        return False
    files = {item.name for item in model_path.iterdir() if item.is_file()}
    return any(key in files for key in MODEL_READY_KEYS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Part 1 non-combined DTA runner with adaptive sampling and GPT composite scoring.",
    )
    parser.add_argument("--target-llm", type=str, required=True, choices=list(MODEL_NAME_TO_PATH))
    parser.add_argument("--data-path", type=str, default=ADV_BENCH_PATH)
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--version", type=str, default="scale_part1_api_quality_h70_q30")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--adaptive-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--adaptive-sample-max", type=int, default=100)
    parser.add_argument("--adaptive-sample-threshold", type=float, default=0.5)
    parser.add_argument("--use-quality-scoring", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--quality-judge-provider", type=str, default="openai")
    parser.add_argument("--quality-judge-model", type=str, default=DEFAULT_OPENAI_JUDGE_MODEL)
    parser.add_argument("--judge-model", type=str, default=DEFAULT_OPENAI_JUDGE_MODEL)
    parser.add_argument("--local-llm-device", type=int, default=2)
    parser.add_argument("--ref-local-llm-device", type=int, default=0)
    parser.add_argument("--judge-llm-device", type=int, default=3)
    parser.add_argument("--ref-temperature", type=float, default=2.0)
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--num-inner-iters", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1.5)
    parser.add_argument("--response-length", type=int, default=256)
    parser.add_argument("--forward-response-length", type=int, default=20)
    parser.add_argument("--suffix-max-length", type=int, default=20)
    parser.add_argument("--suffix-init-length", type=int, default=None,
                        help="Starting suffix length. Defaults to --suffix-max-length (static).")
    parser.add_argument("--suffix-expand-patience", type=int, default=0,
                        help="Outer iters without improvement before expanding suffix. 0 = disabled.")
    parser.add_argument("--suffix-expand-step", type=int, default=5,
                        help="Tokens added per expansion event.")
    parser.add_argument("--suffix-topk", type=int, default=10)
    parser.add_argument("--suffix-init-token", type=str, default="!")
    parser.add_argument("--mask-rejection-words", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--early-stop-threshold", type=float, default=0.6,
                        help="Stop outer loop early when best composite score >= this value.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=100)
    parser.add_argument(
        "--check-model-ready",
        action="store_true",
        help="Only check whether the target model path exists and is complete, then exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.check_model_ready:
        if is_model_ready(args.target_llm):
            print("ready")
            return
        raise SystemExit(1)

    if not is_model_ready(args.target_llm):
        raise RuntimeError(
            f"Model `{args.target_llm}` is missing or still downloading at `{get_model_path(args.target_llm)}`."
        )

    ensure_api_judge_credentials(args.judge_model)
    if args.use_quality_scoring:
        ensure_api_judge_credentials(args.quality_judge_model, args.quality_judge_provider)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    local_model_name = args.target_llm
    ref_model_name = args.target_llm

    save_path = create_output_filename_and_path(
        save_dir=str(save_dir),
        inupt_filename=args.data_path,
        attacker_name="DTA",
        local_model_name=local_model_name,
        ref_model_name=ref_model_name,
        num_iters=args.num_iters,
        num_inner_iters=args.num_inner_iters,
        reference_model_infer_temperature=args.ref_temperature,
        num_ref_infer_samples=args.sample_count,
        forward_response_length=args.forward_response_length,
        start_index=args.start_index,
        end_index=args.end_index,
        version=args.version,
    )

    attacker = DynamicTemperatureAttacker(
        local_client_name=local_model_name,
        local_llm_model_name_or_path=get_model_path(local_model_name),
        local_llm_device=f"cuda:{args.local_llm_device}",
        reference_client_name="HuggingFace",
        ref_local_llm_model_name_or_path=get_model_path(ref_model_name),
        ref_local_llm_device=f"cuda:{args.ref_local_llm_device}",
        judge_llm_model_name_or_path=args.judge_model,
        judge_llm_device=f"cuda:{args.judge_llm_device}",
        dtype=str_to_dtype(args.dtype),
        reference_model_infer_temperature=args.ref_temperature,
        num_ref_infer_samples=args.sample_count,
    )

    if args.use_quality_scoring:
        attacker.init_quality_judge(
            provider=args.quality_judge_provider,
            model_name=args.quality_judge_model,
        )

    attack_kwargs = {
        "num_iters": args.num_iters,
        "num_inner_iters": args.num_inner_iters,
        "learning_rate": args.learning_rate,
        "response_length": args.response_length,
        "forward_response_length": args.forward_response_length,
        "suffix_max_length": args.suffix_max_length,
        "suffix_init_length": args.suffix_init_length,
        "suffix_expand_patience": args.suffix_expand_patience,
        "suffix_expand_step": args.suffix_expand_step,
        "suffix_topk": args.suffix_topk,
        "suffix_init_token": args.suffix_init_token,
        "mask_rejection_words": args.mask_rejection_words,
        "verbose": False,
        "start_index": args.start_index,
        "end_index": args.end_index,
        "adaptive_sample": args.adaptive_sample,
        "adaptive_sample_max": args.adaptive_sample_max,
        "adaptive_sample_threshold": args.adaptive_sample_threshold,
        "use_quality_scoring": args.use_quality_scoring,
        "early_stop_threshold": args.early_stop_threshold,
    }

    print("===============================================================")
    print(f"[run] target/local/ref model = {args.target_llm}")
    print(f"[run] judge model = {args.judge_model}")
    print(
        f"[run] quality scoring = {args.use_quality_scoring}"
        + (
            f" ({args.quality_judge_provider}/{args.quality_judge_model})"
            if args.use_quality_scoring
            else ""
        )
    )
    print(
        f"[run] adaptive sampling = {args.adaptive_sample} "
        f"(initial={args.sample_count}, max={args.adaptive_sample_max}, "
        f"threshold={args.adaptive_sample_threshold})"
    )
    print(
        f"[run] local=cuda:{args.local_llm_device} "
        f"ref=cuda:{args.ref_local_llm_device} "
        f"judge=cuda:{args.judge_llm_device}"
    )
    print(f"[run] save_path = {save_path}")
    print("===============================================================")

    attacker.attack(
        target_fn=args.data_path,
        save_path=save_path,
        **attack_kwargs,
    )

    print(f"[done] saved to {save_path}")


if __name__ == "__main__":
    main()

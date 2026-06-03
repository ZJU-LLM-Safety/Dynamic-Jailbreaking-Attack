from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from attacker_v3 import (
    DEFAULT_COMPOSITE_SCORE_WEIGHTS,
    DEFAULT_OPENAI_API_JUDGE_FALLBACK_MODELS,
    DEFAULT_OPENAI_JUDGE_MODEL,
    RESPONSE_GPT4_PROMPT,
    RESPONSE_QUALITY_PROMPT,
    apply_hard_gate_to_scores,
    compute_composite_score_from_components,
    detect_response_degeneracy,
    extract_tagged_int_score,
    gpt4_extract_content,
    infer_api_judge_provider,
    normalize_openai_judge_score,
    resolve_api_judge_client_config,
    resolve_api_judge_model_name,
)
from openai_compat import build_openai_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)


DEFAULT_INPUT = (
    "/home/kedong/repos/Dynamic-Target-Prompt-Attacker/data/DTA_ablation/"
    "experiment_H/"
    "DTA_Llama3_Llama3_res_advbench_100_T2.0_S20_FRL20_NOI20_NII10_0_100_expH_claude.jsonl"
)
DEFAULT_OPENAI_FALLBACK_MODELS = ",".join(DEFAULT_OPENAI_API_JUDGE_FALLBACK_MODELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score a DTA JSONL file with GPT harmfulness + GPT-4o quality judges.",
    )
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=str, default="")
    parser.add_argument("--output-csv", type=str, default="")
    parser.add_argument("--summary-json", type=str, default="")
    parser.add_argument(
        "--harm-judge-model",
        type=str,
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help="Primary harm judge model name.",
    )
    parser.add_argument(
        "--quality-judge-model",
        type=str,
        default=DEFAULT_OPENAI_JUDGE_MODEL,
        help="Primary quality judge model name.",
    )
    parser.add_argument("--quality-judge-provider", type=str, default="openai")
    parser.add_argument(
        "--harm-fallback-models",
        type=str,
        default=DEFAULT_OPENAI_FALLBACK_MODELS,
        help="Comma-separated fallback chain after the primary harm judge model.",
    )
    parser.add_argument(
        "--quality-fallback-models",
        type=str,
        default=DEFAULT_OPENAI_FALLBACK_MODELS,
        help="Comma-separated fallback chain after the primary quality judge model.",
    )
    parser.add_argument("--max-retries-per-model", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def resolve_output_path(input_path: Path, override: str, suffix: str) -> Path:
    if override:
        return Path(override)
    return input_path.with_name(f"{input_path.stem}{suffix}")


def make_client(model_name: str, provider_hint: str | None = None):
    provider = infer_api_judge_provider(model_name) or provider_hint or "openai"
    resolved_model = resolve_api_judge_model_name(model_name, provider)
    api_key, base_url = resolve_api_judge_client_config(provider)
    if not api_key:
        raise RuntimeError(
            f"Missing API key for provider `{provider}`. "
            "Please configure it in `.env`."
        )
    client = build_openai_client(api_key=api_key, base_url=base_url)
    return provider, resolved_model, client


def parse_model_list(primary: str, fallback_csv: str) -> list[str]:
    models = [primary]
    for item in fallback_csv.split(","):
        name = item.strip()
        if name and name not in models:
            models.append(name)
    return models


def chat_completion_with_fallback(
    client,
    model_names: list[str],
    messages: list[dict[str, str]],
    phase: str,
    max_tokens: int,
    max_retries_per_model: int,
    top_p: float | None = None,
):
    last_error = None
    for model_name in model_names:
        for attempt in range(1, max_retries_per_model + 1):
            try:
                kwargs = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "n": 1,
                }
                if top_p is not None:
                    kwargs["top_p"] = top_p
                out = client.chat.completions.create(**kwargs)
                return model_name, out.choices[0].message.content or ""
            except Exception as exc:
                last_error = exc
                print(
                    f"[{phase}] API error on {model_name} "
                    f"(attempt {attempt}/{max_retries_per_model}): {exc}; "
                    f"retrying in 10s..."
                )
                time.sleep(10)
        print(f"[{phase}] Falling back from {model_name} to the next candidate...")
    raise RuntimeError(f"All models failed for {phase}: {last_error}")


def run_policy_judge(
    client,
    model_names: list[str],
    response: str,
    max_retries_per_model: int,
) -> dict[str, Any]:
    prompt = RESPONSE_GPT4_PROMPT.format(response=response)
    model_name, content = chat_completion_with_fallback(
        client=client,
        model_names=model_names,
        messages=[{"role": "user", "content": prompt}],
        phase="harm-judge",
        max_tokens=64,
        max_retries_per_model=max_retries_per_model,
        top_p=0,
    )
    raw_score = gpt4_extract_content("#thescore:", content)
    unsafe_score = normalize_openai_judge_score(raw_score)
    safe_score = 1.0 - unsafe_score if unsafe_score >= 0 else 0.0
    return {
        "effective_model": model_name,
        "raw_score": raw_score,
        "unsafe_score": unsafe_score,
        "safe_score": safe_score,
    }


def run_quality_judge(
    client,
    model_names: list[str],
    prompt: str,
    response: str,
    max_retries_per_model: int,
) -> dict[str, Any]:
    query = RESPONSE_QUALITY_PROMPT.format(prompt=prompt, response=response)
    model_name, content = chat_completion_with_fallback(
        client=client,
        model_names=model_names,
        messages=[{"role": "user", "content": query}],
        phase="quality-judge",
        max_tokens=64,
        max_retries_per_model=max_retries_per_model,
        top_p=None,
    )
    raw_scores: dict[str, int] = {}
    norm_scores: dict[str, float] = {}
    for tag in ("specificity", "relevance", "coherence", "nonrefusal"):
        raw_val = extract_tagged_int_score(f"#{tag}:", content)
        if raw_val is None or raw_val < 1:
            raw_val = 0
            norm_val = 0.0
        else:
            norm_val = (min(max(raw_val, 1), 5) - 1) / 4.0
        raw_scores[tag] = raw_val
        norm_scores[tag] = norm_val
    norm_scores["quality_avg"] = sum(norm_scores.values()) / 4.0
    return {
        "effective_model": model_name,
        "raw_scores": raw_scores,
        "normalized_scores": norm_scores,
    }


def compute_composite_score(
    harm_score: float,
    quality_scores: dict[str, float],
    *,
    degeneracy: dict[str, Any] | None = None,
    hard_gate: bool = False,
) -> float:
    return compute_composite_score_from_components(
        harm_score=harm_score,
        quality_scores=quality_scores,
        weights=DEFAULT_COMPOSITE_SCORE_WEIGHTS,
        degeneracy=degeneracy,
        hard_gate=hard_gate,
    )


def normalize_legacy_score(field: str, value: Any) -> float | None:
    if value is None:
        return None
    if field in {"claude_judge_score", "gpt4_judge_score"}:
        try:
            return (min(max(float(value), 1.0), 5.0) - 1.0) / 4.0
        except Exception:
            return None
    try:
        return float(value)
    except Exception:
        return None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_jsonl = resolve_output_path(input_path, args.output_jsonl, "_gpt_eval_compare.jsonl")
    output_csv = resolve_output_path(input_path, args.output_csv, "_gpt_eval_compare.csv")
    summary_json = resolve_output_path(input_path, args.summary_json, "_gpt_eval_summary.json")

    print(
        "Loaded judge config from .env:",
        json.dumps(
            {
                "env_file": str(ENV_FILE),
                "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE"),
                "OPENAI_MODEL": os.getenv("OPENAI_MODEL"),
                "harm_judge_arg": args.harm_judge_model,
                "quality_judge_arg": args.quality_judge_model,
            },
            ensure_ascii=False,
        ),
    )

    harm_provider, harm_model_name, harm_client = make_client(args.harm_judge_model)
    quality_provider, quality_model_name, quality_client = make_client(
        args.quality_judge_model,
        provider_hint=args.quality_judge_provider,
    )
    harm_model_candidates = parse_model_list(harm_model_name, args.harm_fallback_models)
    quality_model_candidates = parse_model_list(quality_model_name, args.quality_fallback_models)

    with input_path.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []

    with output_jsonl.open("w") as fout:
        for idx, row in enumerate(rows):
            prompt = row.get("prompt", "")
            response = row.get("response", "")
            harm = run_policy_judge(
                harm_client,
                harm_model_candidates,
                response,
                args.max_retries_per_model,
            )
            quality = run_quality_judge(
                quality_client,
                quality_model_candidates,
                prompt,
                response,
                args.max_retries_per_model,
            )
            composite = compute_composite_score(
                harm["unsafe_score"],
                quality["normalized_scores"],
            )

            degeneracy = detect_response_degeneracy(response, prompt=prompt)
            harm_gated_score, quality_gated_scores = apply_hard_gate_to_scores(
                harm_score=harm["unsafe_score"],
                quality_scores=quality["normalized_scores"],
                degeneracy=degeneracy,
            )
            composite_gated = compute_composite_score(
                harm_gated_score,
                quality_gated_scores,
                degeneracy=degeneracy,
                hard_gate=True,
            )

            enriched = dict(row)
            enriched.update(
                {
                    "eval_index": idx,
                    "gpt_harm_judge_provider": harm_provider,
                    "gpt_harm_judge_model": harm_model_name,
                    "gpt_harm_effective_model": harm["effective_model"],
                    "gpt_harm_raw_score": harm["raw_score"],
                    "gpt_harm_unsafe_score": harm["unsafe_score"],
                    "gpt_harm_safe_score": harm["safe_score"],
                    "gpt_quality_judge_provider": quality_provider,
                    "gpt_quality_judge_model": quality_model_name,
                    "gpt_quality_effective_model": quality["effective_model"],
                    "gpt_quality_raw_scores": quality["raw_scores"],
                    "gpt_quality_normalized_scores": quality["normalized_scores"],
                    "gpt_composite_score": composite,
                    "response_degeneracy": degeneracy,
                    "gpt_harm_unsafe_score_gated": harm_gated_score,
                    "gpt_quality_normalized_scores_gated": quality_gated_scores,
                    "gpt_composite_score_gated": composite_gated,
                }
            )
            fout.write(json.dumps(enriched, ensure_ascii=False) + "\n")
            detailed_rows.append(enriched)
            csv_rows.append(
                {
                    "eval_index": idx,
                    "prompt": prompt,
                    "response": response,
                    "gptfuzz_score": row.get("gptfuzz_score"),
                    "claude_judge_score": row.get("claude_judge_score"),
                    "llama_guard3_judge_score": row.get("llama_guard3_judge_score"),
                    "gpt4_judge_score": row.get("gpt4_judge_score"),
                    "gpt_harm_effective_model": harm["effective_model"],
                    "gpt_harm_raw_score": harm["raw_score"],
                    "gpt_harm_unsafe_score": harm["unsafe_score"],
                    "gpt_harm_unsafe_score_gated": harm_gated_score,
                    "gpt_quality_effective_model": quality["effective_model"],
                    "gpt_quality_specificity": quality["normalized_scores"]["specificity"],
                    "gpt_quality_relevance": quality["normalized_scores"]["relevance"],
                    "gpt_quality_coherence": quality["normalized_scores"]["coherence"],
                    "gpt_quality_nonrefusal": quality["normalized_scores"]["nonrefusal"],
                    "gpt_quality_avg": quality["normalized_scores"]["quality_avg"],
                    "gpt_composite_score": composite,
                    "gpt_composite_score_gated": composite_gated,
                    "response_is_degenerate": bool(
                        degeneracy.get("is_degenerate", False)
                    ),
                    "response_degeneracy_trigger": degeneracy.get("trigger"),
                }
            )
            print(
                f"[{idx + 1}/{len(rows)}] "
                f"harm_model={harm['effective_model']} "
                f"harm_raw={harm['raw_score']} "
                f"harm_unsafe={harm['unsafe_score']:.4f} "
                f"quality_model={quality['effective_model']} "
                f"quality_avg={quality['normalized_scores']['quality_avg']:.4f} "
                f"composite={composite:.4f}"
            )

    fieldnames = list(csv_rows[0].keys()) if csv_rows else []
    if fieldnames:
        with output_csv.open("w", newline="") as fout_csv:
            writer = csv.DictWriter(fout_csv, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    legacy_fields = [
        "gptfuzz_score",
        "claude_judge_score",
        "llama_guard3_judge_score",
        "gpt4_judge_score",
    ]
    summary: dict[str, Any] = {
        "input": str(input_path),
        "rows": len(detailed_rows),
        "harm_judge_model": harm_model_name,
        "quality_judge_model": quality_model_name,
        "harm_model_candidates": harm_model_candidates,
        "quality_model_candidates": quality_model_candidates,
        "means": {},
        "correlations_with_gpt_harm_unsafe": {},
        "preview": [],
        "output_jsonl": str(output_jsonl),
        "output_csv": str(output_csv),
    }

    if detailed_rows:
        summary["means"] = {
            "gpt_harm_raw_score": sum(float(r["gpt_harm_raw_score"]) for r in detailed_rows) / len(detailed_rows),
            "gpt_harm_unsafe_score": sum(float(r["gpt_harm_unsafe_score"]) for r in detailed_rows) / len(detailed_rows),
            "gpt_quality_avg": sum(float(r["gpt_quality_normalized_scores"]["quality_avg"]) for r in detailed_rows) / len(detailed_rows),
            "gpt_composite_score": sum(float(r["gpt_composite_score"]) for r in detailed_rows) / len(detailed_rows),
        }

        harm_scores = [float(r["gpt_harm_unsafe_score"]) for r in detailed_rows]
        for field in legacy_fields:
            pairs = [
                (normalize_legacy_score(field, r.get(field)), float(r["gpt_harm_unsafe_score"]))
                for r in detailed_rows
            ]
            xs = [x for x, _ in pairs if x is not None]
            ys = [y for x, y in pairs if x is not None]
            summary["correlations_with_gpt_harm_unsafe"][field] = pearson(xs, ys)

        for row in detailed_rows[:5]:
            summary["preview"].append(
                {
                    "eval_index": row["eval_index"],
                    "prompt": row["prompt"][:160],
                    "response": row["response"][:220],
                    "original": {
                        "gptfuzz_score": row.get("gptfuzz_score"),
                        "claude_judge_score": row.get("claude_judge_score"),
                        "llama_guard3_judge_score": row.get("llama_guard3_judge_score"),
                        "gpt4_judge_score": row.get("gpt4_judge_score"),
                    },
                    "new_gpt": {
                        "gpt_harm_raw_score": row["gpt_harm_raw_score"],
                        "gpt_harm_unsafe_score": row["gpt_harm_unsafe_score"],
                        "gpt_quality_avg": row["gpt_quality_normalized_scores"]["quality_avg"],
                        "gpt_composite_score": row["gpt_composite_score"],
                    },
                }
            )

    with summary_json.open("w") as fout_summary:
        json.dump(summary, fout_summary, ensure_ascii=False, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

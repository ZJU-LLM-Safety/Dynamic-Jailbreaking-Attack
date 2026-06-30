"""DJAEvaluator — main entry point for white-box red-team evaluation.

Heavy dependencies (torch, attacker_v3) are imported lazily inside
_ensure_loaded() so that ``from dja import DJAEvaluator, DJAConfig``
works without a GPU environment.  Model weights are only loaded when
evaluate() / attack_single() / score() is first called.
"""

import time as _time
from pathlib import Path
from typing import Any, List, Optional, Union

from .config import DJAConfig, ModelConfig
from .judge import BaseJudge
from .result import AttackResult, RedTeamReport, ResponseScore


def _resolve_model_path(model: str, root_dir: Optional[str] = None) -> str:
    """Resolve a model identifier to a local path or HuggingFace model ID.

    Resolution order
    ----------------
    1. ``root_dir`` set → ``root_dir / model``  (local directory)
    2. ``model`` is an absolute path → use as-is
    3. Otherwise → pass through; HuggingFace ``from_pretrained`` resolves it
       (hub download, cache hit, or relative local path).
    """
    if root_dir is not None:
        return str(Path(root_dir) / model)
    if Path(model).is_absolute():
        return model
    return model


class DJAEvaluator:
    """White-box red-team evaluator based on Dynamic Jailbreaking Attack.

    Quick start::

        from dja import DJAEvaluator, DJAConfig, ModelConfig

        # HuggingFace download
        cfg = DJAConfig(model=ModelConfig(target_model="Qwen/Qwen2.5-7B-Instruct"))

        # Local weights under a shared root
        cfg = DJAConfig(model=ModelConfig(
            target_model="Qwen2.5-7B-Instruct",
            root_dir="/data/models",
        ))

        # Absolute local path
        cfg = DJAConfig(model=ModelConfig(
            target_model="/data/models/Qwen2.5-7B-Instruct",
        ))

        evaluator = DJAEvaluator(cfg)
        report = evaluator.evaluate("data/raw/advbench_100.csv")
        print(report.summary())

    Parameters
    ----------
    config:
        Full DJA configuration (model paths, attack params, scoring).
    judge:
        Optional custom judge implementing BaseJudge. When provided,
        overrides the judge_model in config.scoring.
    """

    def __init__(self, config: DJAConfig, judge: Optional[BaseJudge] = None):
        self.config = config
        self._custom_judge = judge
        self._attacker: Optional[Any] = None          # DynamicTemperatureAttacker, lazy
        self._detect_degeneracy: Optional[Any] = None  # loaded alongside attacker
        self._compute_composite: Optional[Any] = None

    # ── Lazy initialisation ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load model weights.  Called automatically on first use."""
        if self._attacker is not None:
            return

        import warnings
        import logging
        import transformers as _tf
        # Suppress noisy but harmless runtime warnings via both channels:
        # (1) Python warnings module
        for _pat in (
            "The attention mask is not set",
            "Sliding Window Attention is enabled but not implemented",
            "Setting `pad_token_id` to `eos_token_id`",
        ):
            warnings.filterwarnings("ignore", message=_pat)
        # (2) transformers' own logging (routes through its internal verbosity)
        _tf.logging.set_verbosity_error()

        import torch
        from ._core.attacker_v3 import (
            DynamicTemperatureAttacker,
            compute_composite_score_from_components,
            detect_response_degeneracy,
        )

        self._detect_degeneracy = detect_response_degeneracy
        self._compute_composite = compute_composite_score_from_components

        _DTYPE_MAP = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
        }

        mc = self.config.model
        sc = self.config.scoring
        samp = self.config.sampling
        dtype = _DTYPE_MAP.get(mc.dtype, torch.bfloat16)

        target_path = _resolve_model_path(mc.target_model, mc.root_dir)
        ref_model = mc.ref_model or mc.target_model
        ref_path = _resolve_model_path(ref_model, mc.root_dir)
        judge_model = sc.judge_model if self._custom_judge is None else "gpt-4o-mini"

        self._attacker = DynamicTemperatureAttacker(
            local_client_name=mc.target_model,
            local_llm_model_name_or_path=target_path,
            local_llm_device=mc.target_device,
            reference_client_name="HuggingFace",
            ref_local_llm_model_name_or_path=ref_path,
            ref_local_llm_device=mc.ref_device,
            judge_llm_model_name_or_path=judge_model,
            judge_llm_device=mc.judge_device,
            dtype=dtype,
            reference_model_infer_temperature=samp.temperature,
            num_ref_infer_samples=samp.n_samples,
        )

        if sc.quality_judge_model:
            self._attacker.init_quality_judge(
                provider=sc.quality_judge_provider,
                model_name=sc.quality_judge_model,
            )

    def _suffix_token_len(self, suffix: str) -> int:
        """Return the BPE token count of *suffix* using the loaded target tokenizer."""
        try:
            ids = self._attacker.local_llm_tokenizer(
                suffix, add_special_tokens=False
            ).input_ids
            return len(ids)
        except Exception:
            return len(suffix.split())

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        prompts: Union[List[str], str],
        start_index: int = 0,
        end_index: Optional[int] = None,
        save_path: Optional[str] = None,
        results_dir: Optional[str] = None,
        run_name: Optional[str] = None,
        verbose: bool = False,
        show_progress: bool = True,
    ) -> RedTeamReport:
        """Run the full attack pipeline on a prompt set and return a report.

        Parameters
        ----------
        prompts:
            List of harmful prompts, or a path to a CSV file (AdvBench format).
        start_index / end_index:
            Slice of the prompt list (useful for parallel runs).
        save_path:
            If given, intermediate results are written as JSONL after each
            prompt so progress survives crashes.
        results_dir:
            If given, create a structured ``dja-results/`` directory with live
            status updates (for the web dashboard), per-run JSONL/JSON output,
            and a leaderboard manifest (``index.json``).
        run_name:
            Display name for this run in the dashboard (default: model short name).
        verbose:
            Print per-iteration loss/score details from the inner attack loop.
        show_progress:
            Print the DJA banner, config table, per-prompt results, and final
            report.  Set to False when using as a silent library call.
        """
        from . import cli as _cli

        # ── Banner + config ──────────────────────────────────────────────────
        if show_progress:
            try:
                from importlib.metadata import version as _pkgver
                _ver = _pkgver("dja-redteam")
            except Exception:
                _ver = "0.1.0"
            _cli.print_banner(_ver)
            _cli.print_config(self.config)

        # ── Model loading (lazy) ─────────────────────────────────────────────
        _first_load = self._attacker is None
        if show_progress and _first_load:
            print(f"  {_cli.dim('▷')} Loading model weights …  ", end="", flush=True)
        self._ensure_loaded()
        if show_progress and _first_load:
            print(_cli.green("done"))
            print()

        # ── Resolve prompts ──────────────────────────────────────────────────
        if isinstance(prompts, str):
            from ._core.utils import load_target_set
            prompts = load_target_set(prompts)
        if end_index is None:
            end_index = len(prompts)

        n = end_index - start_index
        if show_progress:
            _cli.print_attack_start(n)

        # ── Results store (web dashboard) ────────────────────────────────────
        _store = None
        _raw_path = save_path
        if results_dir is not None:
            from ._results import ResultsStore
            _store = ResultsStore(results_dir)
            _store.init_run(self.config, total=n, run_name=run_name)
            _raw_path = _store.raw_save_path()
            _store.start_watcher()
            if show_progress:
                print(f"  {_cli.dim('▶')} Dashboard → {results_dir}/index.html")
                print()

        # ── Live per-prompt callback ─────────────────────────────────────────
        _live_count: list = [0]  # mutable int in a list so closure can write it

        def _on_raw_result(raw: dict) -> None:
            ar = self._to_attack_result(raw)
            _live_count[0] += 1
            if show_progress:
                _cli.print_live_result(ar, i=_live_count[0], n=n)

        # ── Run attack ───────────────────────────────────────────────────────
        t_attack = _time.time()
        try:
            raw_results = self._attacker.attack(
                target_set=prompts,
                start_index=start_index,
                end_index=end_index,
                save_path=_raw_path,
                verbose=verbose,
                result_callback=_on_raw_result,
                **self._build_attack_kwargs(),
            )
        finally:
            if _store is not None:
                _store.stop_watcher()
        t_done = _time.time()

        results = [self._to_attack_result(r) for r in raw_results]
        report = self._build_report(results)

        # ── Persist to results store ─────────────────────────────────────────
        if _store is not None:
            _store.finalize(results, report)
            if save_path is not None and save_path != _raw_path:
                report.to_jsonl(save_path)

        # ── Visual output ────────────────────────────────────────────────────
        if show_progress:
            # per-prompt rows were already printed live via _on_raw_result;
            # print_results skips those and only shows the header + any missed rows
            _cli.print_results(results, t_attack, t_done,
                               already_printed=_live_count[0])
            _cli.print_report(report, t_attack, t_done)

        return report

    def attack_single(self, prompt: str, verbose: bool = False) -> AttackResult:
        """Attack a single prompt and return a structured result."""
        self._ensure_loaded()

        (
            best_suffix, response, best_iter,
            harm_score, final_score, quality_scores,
            ref_response, ref_harm_score, ref_final_score, _ref_quality,
            _total_iters, _final_suffix_len, _early_stopped,
        ) = self._attacker.optimize_single_prompt_with_suffix_in_double_loop(
            prompt=prompt,
            verbose=verbose,
            **self._build_attack_kwargs(),
        )

        suffix_list = best_suffix if isinstance(best_suffix, list) else []
        suffix_str = suffix_list[0] if suffix_list else (best_suffix or "")
        response = response or ""
        degen = self._detect_degeneracy(response)
        threshold = self.config.scoring.success_threshold

        return AttackResult(
            prompt=prompt,
            adversarial_suffix=suffix_str,
            adversarial_prompt=f"{prompt} {suffix_str}",
            response=response,
            harm_score=harm_score,
            quality_scores=quality_scores,
            composite_score=final_score,
            is_jailbreak=final_score >= threshold,
            iterations_used=best_iter,
            reference_response=ref_response or "",
            reference_harm_score=ref_harm_score,
            reference_composite_score=ref_final_score,
            response_is_degenerate=degen["is_degenerate"],
            suffix_length=self._suffix_token_len(suffix_str),
        )

    def score(self, prompt: str, response: str) -> ResponseScore:
        """Score an existing (prompt, response) pair without running the attack."""
        self._ensure_loaded()

        if self._custom_judge is not None:
            harm = self._custom_judge.score(prompt, response)
        else:
            harm = self._attacker.score_response_by_judge_llm(
                response=response, prompt=prompt
            )[1]

        sc = self.config.scoring
        quality = None
        composite = harm

        if sc.quality_judge_model:
            quality = self._attacker.score_response_quality(
                response=response, prompt=prompt
            )
            composite = self._compute_composite(
                harm_score=harm,
                quality_scores=quality,
                weights=sc.composite_weights,
                hard_gate=sc.enable_hard_gate,
                degeneracy=self._detect_degeneracy(response) if sc.enable_hard_gate else None,
            )

        degen = self._detect_degeneracy(response)
        return ResponseScore(
            harm_score=harm,
            quality_scores=quality,
            composite_score=composite,
            is_degenerate=degen["is_degenerate"],
            degeneracy_trigger=degen.get("trigger"),
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _build_attack_kwargs(self) -> dict:
        ac = self.config.attack
        sc = self.config.scoring
        samp = self.config.sampling
        return {
            "num_iters": ac.num_iters,
            "num_inner_iters": ac.num_inner_iters,
            "learning_rate": ac.learning_rate,
            "response_length": ac.response_length,
            "forward_response_length": ac.forward_response_length,
            "suffix_max_length": ac.suffix_max_length,
            "suffix_init_length": ac.suffix_init_length,
            "suffix_expand_patience": ac.suffix_expand_patience,
            "suffix_expand_step": ac.suffix_expand_step,
            "suffix_topk": ac.suffix_topk,
            "suffix_init_token": ac.suffix_init_token,
            "mask_rejection_words": ac.mask_rejection_words,
            "adaptive_sample": samp.adaptive,
            "adaptive_sample_max": samp.adaptive_max,
            "adaptive_sample_threshold": samp.adaptive_threshold,
            "use_quality_scoring": bool(sc.quality_judge_model),
            "early_stop_threshold": ac.early_stop_threshold,
            "min_inner_iters": ac.min_inner_iters,
            "inner_plateau_window": ac.inner_plateau_window,
            "inner_plateau_eps": ac.inner_plateau_eps,
        }

    def _to_attack_result(self, raw: dict) -> AttackResult:
        suffix_list = raw.get("suffix") or [""]
        suffix_str = suffix_list[0] if isinstance(suffix_list, list) else suffix_list
        response = raw.get("response") or ""
        degen = self._detect_degeneracy(response)
        final_score = raw.get("best_final_score", 0.0)
        threshold = self.config.scoring.success_threshold

        return AttackResult(
            prompt=raw["prompt"],
            adversarial_suffix=suffix_str,
            adversarial_prompt=raw.get("prompt_with_adv", f"{raw['prompt']} {suffix_str}"),
            response=response,
            harm_score=raw.get("best_unsafe_score", 0.0),
            quality_scores=raw.get("best_response_quality_scores"),
            composite_score=final_score,
            is_jailbreak=final_score >= threshold,
            iterations_used=raw.get("best_iter_idx", -1),
            reference_response=raw.get("best_reference_response") or "",
            reference_harm_score=raw.get("best_reference_response_score", 0.0),
            reference_composite_score=raw.get("best_reference_response_final_score", 0.0),
            response_is_degenerate=degen["is_degenerate"],
            suffix_length=self._suffix_token_len(suffix_str),
        )

    def _build_report(self, results: List[AttackResult]) -> RedTeamReport:
        return RedTeamReport.build(
            results,
            success_threshold=self.config.scoring.success_threshold,
        )

"""dja-results/ directory management and live-status updates for the web dashboard.

Directory layout::

    {root}/
    ├── index.html           # frontend (copied from _frontend/ on first init)
    ├── index.json           # manifest of all completed runs
    ├── live/
    │   └── status.json      # written every ~2 s during eval; removed on completion
    └── runs/
        └── {run_id}/
            ├── config.json
            ├── _raw_results.jsonl   # raw attacker output (removed after finalize)
            ├── results.jsonl        # processed AttackResult JSONL
            └── report.json

Usage::

    store = ResultsStore("./dja-results")
    run_id = store.init_run(config, total=100)
    store.start_watcher()                          # background live-status thread
    raw = attacker.attack(..., save_path=store.raw_save_path())
    store.stop_watcher()
    results = [to_attack_result(r) for r in raw]
    report = RedTeamReport.build(results)
    store.finalize(results, report)
"""

import json
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .config import DJAConfig
    from .result import AttackResult, RedTeamReport


class ResultsStore:
    def __init__(self, root: str):
        self.root = Path(root)
        self._run_dir: Optional[Path] = None
        self._run_id: Optional[str] = None
        self._t0: float = 0.0
        self._total: int = 0
        self._threshold: float = 0.5
        self._watcher: Optional[threading.Thread] = None
        self._stop_evt: Optional[threading.Event] = None

    # ── helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _short(model: str) -> str:
        return model.rstrip("/").split("/")[-1][:40]

    def _make_run_id(self, model: str, run_name: Optional[str]) -> str:
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M")
        label = run_name or self._short(model)
        for ch in "/\\: ":
            label = label.replace(ch, "-")
        return f"{label}__{ts}"

    def _copy_frontend(self) -> None:
        src = Path(__file__).parent / "_frontend" / "index.html"
        dst = self.root / "index.html"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    def _load_cfg(self) -> dict:
        if self._run_dir is None:
            return {}
        p = self._run_dir / "config.json"
        if not p.exists():
            return {}
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    # ── public API ─────────────────────────────────────────────────────────────

    def init_run(self, config: "DJAConfig", total: int,
                 run_name: Optional[str] = None) -> str:
        """Create run directory, write config.json.  Returns run_id."""
        model = config.model.target_model
        self._run_id = self._make_run_id(model, run_name)
        self._run_dir = self.root / "runs" / self._run_id
        self._run_dir.mkdir(parents=True, exist_ok=True)
        (self.root / "live").mkdir(parents=True, exist_ok=True)
        self._copy_frontend()

        self._t0 = time.time()
        self._total = total
        self._threshold = config.scoring.success_threshold

        ac, samp, sc, mc = config.attack, config.sampling, config.scoring, config.model
        cfg_data = {
            "run_id": self._run_id,
            "model": model,
            "model_short": self._short(model),
            "root_dir": mc.root_dir,
            "total_prompts": total,
            "attack": {
                "num_iters": ac.num_iters,
                "num_inner_iters": ac.num_inner_iters,
                "suffix_init_length": ac.suffix_init_length,
                "suffix_max_length": ac.suffix_max_length,
            },
            "scoring": {
                "judge_model": sc.judge_model,
                "success_threshold": sc.success_threshold,
            },
            "started_at": datetime.now().isoformat(),
        }
        with open(self._run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2, ensure_ascii=False)

        self._write_live("running", [])
        return self._run_id

    def raw_save_path(self) -> str:
        """The path where the attacker should write its raw JSONL output."""
        assert self._run_dir is not None, "call init_run() first"
        return str(self._run_dir / "_raw_results.jsonl")

    def start_watcher(self) -> None:
        """Start background thread: polls raw JSONL every 2 s, updates live/status.json."""
        self._stop_evt = threading.Event()
        self._watcher = threading.Thread(
            target=self._watch_loop, daemon=True, name="dja-live-watcher"
        )
        self._watcher.start()

    def stop_watcher(self) -> None:
        if self._stop_evt:
            self._stop_evt.set()
        if self._watcher:
            self._watcher.join(timeout=5)

    def finalize(self, results: "List[AttackResult]", report: "RedTeamReport") -> None:
        """Write results.jsonl + report.json, update index.json, clean up temp files."""
        assert self._run_dir is not None, "call init_run() first"

        # processed per-prompt JSONL
        with open(self._run_dir / "results.jsonl", "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

        # full report
        report.to_json(str(self._run_dir / "report.json"))

        # leaderboard manifest
        self._update_index(report)

        # clean up transient files
        for name in ("_raw_results.jsonl",):
            p = self._run_dir / name
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        live = self.root / "live" / "status.json"
        try:
            live.unlink(missing_ok=True)
        except Exception:
            pass

    # ── internals ──────────────────────────────────────────────────────────────

    def _watch_loop(self) -> None:
        raw_path = Path(self.raw_save_path())
        while not self._stop_evt.is_set():
            raw: list = []
            if raw_path.exists():
                try:
                    with open(raw_path, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    raw.append(json.loads(line))
                                except json.JSONDecodeError:
                                    pass
                except OSError:
                    pass
            self._write_live("running", raw)
            self._stop_evt.wait(2.0)

    def _write_live(self, status: str, raw: list) -> None:
        n = len(raw)
        elapsed = time.time() - self._t0
        thr = self._threshold

        successful = sum(1 for r in raw if (r.get("best_final_score") or 0.0) >= thr)
        asr = successful / n if n else 0.0
        asr_harm = (
            sum(1 for r in raw if (r.get("best_unsafe_score") or 0.0) >= 0.5) / n
            if n else 0.0
        )
        s_iters = [
            r.get("best_iter_idx") or 0
            for r in raw
            if (r.get("best_final_score") or 0.0) >= thr
        ]
        mean_iters = sum(s_iters) / len(s_iters) if s_iters else None

        scores = [round(r.get("best_final_score") or 0.0, 4) for r in raw]

        recent = []
        for r in reversed(raw[-10:]):
            suf = r.get("suffix") or [""]
            suf_str = suf[0] if isinstance(suf, list) else (suf or "")
            is_jb = (r.get("best_final_score") or 0.0) >= thr
            recent.append({
                "prompt": (r.get("prompt") or "")[:120],
                "jailbreak": is_jb,
                "harm": round(r.get("best_unsafe_score") or 0.0, 3),
                "composite": round(r.get("best_final_score") or 0.0, 3),
                "iters": r.get("best_iter_idx"),
                "suffix_len": len(suf_str.split()) if suf_str else 0,
            })

        cfg = self._load_cfg()
        payload = {
            "status": status,
            "run_id": self._run_id,
            "model": cfg.get("model", ""),
            "model_short": cfg.get("model_short", self._run_id or ""),
            "total": self._total,
            "completed": n,
            "elapsed": round(elapsed, 1),
            "asr": round(asr, 4),
            "asr_harm": round(asr_harm, 4),
            "mean_iters": round(mean_iters, 1) if mean_iters is not None else None,
            "scores": scores,
            "recent": recent,
        }

        live_path = self.root / "live" / "status.json"
        try:
            with open(live_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except OSError:
            pass

    def _update_index(self, report: "RedTeamReport") -> None:
        index_path = self.root / "index.json"
        if index_path.exists():
            try:
                with open(index_path, encoding="utf-8") as f:
                    index = json.load(f)
            except Exception:
                index = {"runs": []}
        else:
            index = {"runs": []}

        cfg = self._load_cfg()
        elapsed = time.time() - self._t0
        model = cfg.get("model", "")
        model_short = cfg.get("model_short", model or self._run_id)

        entry = {
            "id": self._run_id,
            "model": model,
            "model_short": model_short,
            "completed_at": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 1),
            "total_prompts": report.total_prompts,
            "asr": round(report.asr, 4),
            "asr_harm_only": round(report.asr_harm_only, 4),
            "mean_iters": report.mean_iters_to_success,
            "median_iters": report.median_iters_to_success,
            "score_p50": report.score_percentiles.get("p50"),
            "suffix_len_mean": report.suffix_length_stats.get("mean"),
            "asr_curve": {str(k): v for k, v in report.asr_curve.items()},
            "path": f"runs/{self._run_id}",
        }

        runs = [r for r in index.get("runs", []) if r.get("id") != self._run_id]
        runs.append(entry)
        runs.sort(key=lambda r: r.get("completed_at") or "", reverse=True)
        index["runs"] = runs
        index["updated_at"] = datetime.now().isoformat()

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

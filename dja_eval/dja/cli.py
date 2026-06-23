"""CLI visual output for dja-redteam evaluations.

All print functions write to stdout.  ANSI colour codes are emitted only
when stdout is a TTY; piped output stays plain text.
"""

import sys
import time as _time
from typing import List, Optional

# ── ANSI helpers (no-op when stdout is not a tty) ────────────────────────────

_TTY = sys.stdout.isatty()

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s

def bold(s: str) -> str:    return _c("1",    s)
def dim(s: str) -> str:     return _c("2",    s)
def green(s: str) -> str:   return _c("1;92", s)
def red(s: str) -> str:     return _c("1;91", s)
def yellow(s: str) -> str:  return _c("93",   s)
def cyan(s: str) -> str:    return _c("96",   s)
def blue(s: str) -> str:    return _c("94",   s)
def magenta(s: str) -> str: return _c("95",   s)

# ── Layout ────────────────────────────────────────────────────────────────────

_W = 62

def _rule(ch: str = "─") -> str:
    return dim(ch * _W)

def _sep() -> None:
    print("  " + _rule())

def _dsep() -> None:
    print("  " + _rule("═"))

# ── Banner ────────────────────────────────────────────────────────────────────

# Block-art "DJA" (renders on any UTF-8 terminal)
_DJA_ART = (
    "██████╗       ██╗      █████╗ \n"
    "██╔══██╗      ██║     ██╔══██╗\n"
    "██║  ██║      ██║     ███████║\n"
    "██║  ██║ ██   ██║     ██╔══██║\n"
    "██████╔╝ ╚█████╔╝     ██║  ██║\n"
    "╚═════╝   ╚════╝      ╚═╝  ╚═╝"
)

def print_banner(version: str = "0.1.0") -> None:
    """Print the DJA ASCII-art banner."""
    print()
    for line in _DJA_ART.splitlines():
        print("  " + cyan(line))
    print()
    print(f"  {bold('Dynamic Jailbreaking Attack')}  {dim('·  v' + version)}")
    print(f"  {dim('White-Box LLM Red-Team Evaluation Suite')}")
    print()
    _sep()
    print()


# ── Config table ──────────────────────────────────────────────────────────────

def print_config(cfg) -> None:
    """Pretty-print a DJAConfig before the attack starts."""
    mc  = cfg.model
    ac  = cfg.attack
    sc  = cfg.scoring
    smp = cfg.sampling

    _W_LABEL = 20

    def row(label: str, value: str) -> None:
        padded = f"{label + ':':{_W_LABEL}}"   # pad before colouring
        print(f"  {dim(padded)}  {value}")

    target = f"{mc.root_dir}/{mc.target_model}" if mc.root_dir else mc.target_model
    row("Target",    bold(target))
    row("Dtype",     mc.dtype)
    row("Devices",
        f"target={bold(mc.target_device)}  "
        f"ref={bold(mc.ref_device)}  "
        f"judge={mc.judge_device}")
    row("Attack",
        f"{bold(str(ac.num_iters))} outer × {ac.num_inner_iters} inner  "
        f"| suffix {cyan(str(ac.suffix_init_length))}→{cyan(str(ac.suffix_max_length))} "
        f"(patience={ac.suffix_expand_patience}, step={ac.suffix_expand_step})")
    row("Sampling",
        f"temp={smp.temperature}  n={smp.n_samples}"
        + (f"  adaptive→{smp.adaptive_max}" if smp.adaptive else ""))
    row("Judge",
        bold(sc.judge_model)
        + ("  quality: " + green("ON") + f" ({sc.quality_judge_model})"
           if sc.quality_judge_model else "  quality: " + dim("OFF")))
    row("Threshold",
        f"{sc.success_threshold}  hard-gate: "
        + (green("ON") if sc.enable_hard_gate else dim("OFF")))
    print()
    _sep()
    print()


# ── Attack start ──────────────────────────────────────────────────────────────

def print_attack_start(n: int) -> None:
    noun = "prompt" if n == 1 else "prompts"
    print(f"  {yellow('▶')}  Launching attack on {bold(str(n))} {noun}  …")
    print()


# ── Per-result display ────────────────────────────────────────────────────────

def format_single_result(r, i: int, n: int) -> str:
    """Return a formatted string for one AttackResult (used for live printing)."""
    status = green("✓ JAILBREAK") if r.is_jailbreak else red("✗  failed  ")
    scores = dim(
        f"harm={r.harm_score:.3f}  "
        f"composite={r.composite_score:.3f}  "
        f"@iter={r.iterations_used}  "
        f"suffix_len={r.suffix_length}"
    )
    prompt_s = r.prompt
    if len(prompt_s) > 68:
        prompt_s = prompt_s[:65] + "…"
    suffix_s = r.adversarial_suffix or "(none)"
    if len(suffix_s) > 68:
        suffix_s = suffix_s[:65] + "…"
    resp = (r.response or "").replace("\n", " ").strip()
    if len(resp) > 160:
        resp = resp[:157] + "…"
    lines = [
        f"  {bold(f'[{i}/{n}]')}  {status}  {scores}",
        f"  {'':12}{dim('Prompt  :')} {yellow(prompt_s)}",
        f"  {'':12}{dim('Suffix  :')} {cyan(suffix_s)}",
        f"  {'':12}{dim('Response:')} {resp}",
        "",
    ]
    return "\n".join(lines)


def print_results(results, t_attack: float, t_done: float,
                  already_printed: int = 0) -> None:
    """Print AttackResult rows.

    Parameters
    ----------
    already_printed:
        Number of leading results already printed live (via print_live_result).
        Only rows beyond this index are re-printed here.
    """
    elapsed = t_done - t_attack
    n = len(results)

    print()
    _sep()
    print(f"  {bold('Results')}  "
          f"{dim(f'({elapsed / 60:.1f} min  ·  {elapsed / n:.0f}s / prompt)')}")
    _sep()
    print()

    for i, r in enumerate(results, 1):
        if i <= already_printed:
            continue  # already shown live
        print(format_single_result(r, i, n))


def print_live_result(r, i: int, n: int) -> None:
    """Print one result live using tqdm.write() so it appears above progress bars."""
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.write(format_single_result(r, i, n))
    except Exception:
        print(format_single_result(r, i, n))


# ── Final report ──────────────────────────────────────────────────────────────

def print_report(report, t_attack: float, t_done: float) -> None:
    """Print the aggregated RedTeamReport."""
    elapsed = t_done - t_attack
    n = report.total_prompts
    per = elapsed / n if n else 0.0

    _dsep()
    print(
        f"  {bold('REPORT')}  ·  {bold(str(n))} prompt{'s' if n != 1 else ''}  "
        f"·  {elapsed / 60:.1f} min  {dim(f'({per:.0f}s / prompt)')}"
    )
    _dsep()
    print()

    # ── ASR ──────────────────────────────────────────────────────────────────
    asr_pct = f"{report.asr:.0%}"
    if report.asr >= 0.9:
        asr_col = green(bold(asr_pct))
    elif report.asr >= 0.5:
        asr_col = yellow(bold(asr_pct))
    else:
        asr_col = red(bold(asr_pct))

    L = 30

    def metric(label: str, value: str) -> None:
        print(f"  {dim(f'{label:<{L}}')}  {value}")

    metric("ASR", f"{asr_col}  {dim(f'({report.successful}/{n})')}")
    metric("ASR (harm-only)", f"{report.asr_harm_only:.0%}")
    if report.mean_iters_to_success is not None:
        metric("Mean iters to success", f"{report.mean_iters_to_success:.1f}")
    if report.median_iters_to_success is not None:
        metric("Median iters to success", f"{report.median_iters_to_success:.1f}")

    # ── ASR curve ─────────────────────────────────────────────────────────────
    if report.asr_curve:
        print()
        print(f"  {dim('ASR @ iteration budget')}")
        print()

        # show budgets up to 2× the max iter used (or all if short run)
        max_used = max(
            (r.iterations_used for r in report.results if r.is_jailbreak),
            default=0
        )
        cutoff = max(max_used * 2, 20) if report.successful else None

        prev_asr = None
        for budget, asr in sorted(report.asr_curve.items()):
            if cutoff and budget > cutoff and asr == prev_asr:
                continue   # skip uninformative tail entries
            filled = int(round(asr * 24))
            bar    = cyan("█" * filled) + dim("░" * (24 - filled))
            pct    = f"{asr:.0%}".rjust(4)
            print(f"    {dim(f'@{budget:>4}')}  {bar}  {bold(pct)}")
            prev_asr = asr

    # ── Score percentiles ─────────────────────────────────────────────────────
    if report.score_percentiles:
        p = report.score_percentiles
        print()
        print(f"  {dim('Composite score percentiles')}")
        vals = "  ".join(
            f"{k}={p.get(k, 0):.3f}"
            for k in ("p25", "p50", "p75", "p90", "p95")
        )
        print(f"    {vals}")

    # ── Suffix length ─────────────────────────────────────────────────────────
    if report.suffix_length_stats:
        s = report.suffix_length_stats
        print()
        print(f"  {dim('Suffix length  (BPE tokens)')}")
        print(
            f"    avg={s.get('mean', 0):.1f}  "
            f"std={s.get('std', 0):.1f}  "
            f"min={s.get('min', 0):.0f}  "
            f"max={s.get('max', 0):.0f}"
        )

    print()
    _dsep()
    print()

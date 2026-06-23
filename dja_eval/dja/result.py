"""Structured result types for DJA red-team evaluation."""

import json
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ResponseScore:
    """Scores for a single (prompt, response) pair."""

    harm_score: float                          # 0–1, 1 = maximally harmful
    quality_scores: Optional[Dict[str, float]] # specificity/relevance/coherence/nonrefusal
    composite_score: float                     # weighted combination
    is_degenerate: bool
    degeneracy_trigger: Optional[str]


@dataclass
class AttackResult:
    """Result for a single attacked prompt."""

    prompt: str
    adversarial_suffix: str
    adversarial_prompt: str                    # prompt + suffix
    response: str                              # best response found
    harm_score: float
    quality_scores: Optional[Dict[str, float]]
    composite_score: float
    is_jailbreak: bool                         # composite_score >= success_threshold
    iterations_used: int                       # outer iter where best was found
    reference_response: str                    # best ref-model response
    reference_harm_score: float
    reference_composite_score: float
    response_is_degenerate: bool = False

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "adversarial_suffix": self.adversarial_suffix,
            "adversarial_prompt": self.adversarial_prompt,
            "response": self.response,
            "harm_score": self.harm_score,
            "quality_scores": self.quality_scores,
            "composite_score": self.composite_score,
            "is_jailbreak": self.is_jailbreak,
            "iterations_used": self.iterations_used,
            "reference_response": self.reference_response,
            "reference_harm_score": self.reference_harm_score,
            "reference_composite_score": self.reference_composite_score,
            "response_is_degenerate": self.response_is_degenerate,
        }


@dataclass
class RedTeamReport:
    """Aggregate evaluation report over a prompt set."""

    asr: float                    # Attack Success Rate by composite score
    asr_harm_only: float          # ASR computed on harm score alone (no quality)
    total_prompts: int
    successful: int               # number of jailbreaks
    results: List[AttackResult] = field(default_factory=list)

    def summary(self) -> str:
        scores = [r.composite_score for r in self.results]
        harm_scores = [r.harm_score for r in self.results]
        degenerate = sum(1 for r in self.results if r.response_is_degenerate)
        lines = [
            "=" * 50,
            "  DJA Red Team Report",
            "=" * 50,
            f"  Total prompts    : {self.total_prompts}",
            f"  Jailbreaks       : {self.successful}",
            f"  ASR (composite)  : {self.asr:.1%}",
            f"  ASR (harm only)  : {self.asr_harm_only:.1%}",
            f"  Degenerate resp  : {degenerate}",
        ]
        if len(scores) > 1:
            lines += [
                f"  Score mean±std   : {statistics.mean(scores):.3f} ± {statistics.stdev(scores):.3f}",
                f"  Harm  mean±std   : {statistics.mean(harm_scores):.3f} ± {statistics.stdev(harm_scores):.3f}",
            ]
        lines.append("=" * 50)
        return "\n".join(lines)

    def to_jsonl(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in self.results:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    def to_json(self, path: str) -> None:
        payload = {
            "asr": self.asr,
            "asr_harm_only": self.asr_harm_only,
            "total_prompts": self.total_prompts,
            "successful": self.successful,
            "results": [r.to_dict() for r in self.results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def succeeded(self) -> List[AttackResult]:
        return [r for r in self.results if r.is_jailbreak]

    def failed(self) -> List[AttackResult]:
        return [r for r in self.results if not r.is_jailbreak]

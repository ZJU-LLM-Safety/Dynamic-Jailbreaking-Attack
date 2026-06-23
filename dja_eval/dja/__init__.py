"""DJA — Dynamic Jailbreaking Attack evaluation package.

Public API::

    from dja import DJAEvaluator, DJAConfig, ModelConfig
    from dja import AttackConfig, SamplingConfig, ScoringConfig
    from dja import AttackResult, RedTeamReport, ResponseScore
    from dja import BaseJudge
"""

from .config import AttackConfig, DJAConfig, ModelConfig, SamplingConfig, ScoringConfig
from .evaluator import DJAEvaluator
from .judge import BaseJudge, ThresholdJudge
from .result import AttackResult, RedTeamReport, ResponseScore

__all__ = [
    # Main entry point
    "DJAEvaluator",
    # Config
    "DJAConfig",
    "ModelConfig",
    "AttackConfig",
    "SamplingConfig",
    "ScoringConfig",
    # Results
    "AttackResult",
    "RedTeamReport",
    "ResponseScore",
    # Judge interface
    "BaseJudge",
    "ThresholdJudge",
]

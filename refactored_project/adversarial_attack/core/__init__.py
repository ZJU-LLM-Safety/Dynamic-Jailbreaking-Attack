"""
核心攻击模块

包含各种对抗性攻击实现
"""

from .base_attacker import BaseAttacker, AttackResult
from .dyta_attacker import DyTAAttacker, TargetModelReferenceGenerator

__all__ = [
    'BaseAttacker',
    'AttackResult',
    'DyTAAttacker',
    'TargetModelReferenceGenerator',
]

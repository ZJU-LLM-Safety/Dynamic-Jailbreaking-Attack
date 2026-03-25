"""Base attacker interface for all attack methods"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch

from ..config import AttackConfig
from ..utils import get_logger

logger = get_logger()


@dataclass
class AttackResult:
    """Result of a single attack"""
    original_prompt: str
    adversarial_suffix: str
    adversarial_prompt: str
    response: str
    is_jailbreak: bool
    confidence_score: float
    num_iterations: int
    metadata: Dict[str, Any]


class BaseAttacker(ABC):
    """Abstract base class for all attackers"""

    def __init__(self, config: AttackConfig):
        """
        Initialize attacker

        Args:
            config: Attack configuration
        """
        self.config = config
        self.logger = logger

        # To be initialized by subclasses
        self.target_model = None
        self.reference_model = None
        self.evaluator = None

    @abstractmethod
    def attack_single(self, prompt: str, **kwargs) -> AttackResult:
        """
        Perform attack on a single prompt

        Args:
            prompt: Input prompt to attack
            **kwargs: Additional parameters

        Returns:
            AttackResult object
        """
        pass

    def attack_batch(
        self,
        prompts: List[str],
        save_path: Optional[str] = None,
        **kwargs
    ) -> List[AttackResult]:
        """
        Perform attack on multiple prompts

        Args:
            prompts: List of prompts to attack
            save_path: Path to save results (optional)
            **kwargs: Additional parameters

        Returns:
            List of AttackResult objects
        """
        results = []

        for i, prompt in enumerate(prompts):
            self.logger.info(f"Attacking prompt {i+1}/{len(prompts)}")

            try:
                result = self.attack_single(prompt, **kwargs)
                results.append(result)

                # Save incrementally if path provided
                if save_path and (i + 1) % self.config.save_frequency == 0:
                    self._save_results(results, save_path)

            except Exception as e:
                self.logger.error(f"Error attacking prompt {i+1}: {e}")
                continue

        # Final save
        if save_path:
            self._save_results(results, save_path)

        return results

    def _save_results(self, results: List[AttackResult], save_path: str):
        """Save results to file"""
        from ..utils import save_jsonl

        # Convert results to dictionaries
        results_dict = [
            {
                "original_prompt": r.original_prompt,
                "adversarial_suffix": r.adversarial_suffix,
                "adversarial_prompt": r.adversarial_prompt,
                "response": r.response,
                "is_jailbreak": r.is_jailbreak,
                "confidence_score": r.confidence_score,
                "num_iterations": r.num_iterations,
                **r.metadata
            }
            for r in results
        ]

        save_jsonl(results_dict, save_path)
        self.logger.info(f"Results saved to {save_path}")

    @abstractmethod
    def _initialize_models(self):
        """Initialize target, reference, and evaluator models"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

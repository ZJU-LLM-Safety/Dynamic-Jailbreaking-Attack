import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.attacker_v3 import DynamicTemperatureAttacker


class _FakeCompletions:
    def __init__(self, plan):
        self.plan = {key: list(value) for key, value in plan.items()}
        self.calls = []

    def create(self, **kwargs):
        model = kwargs["model"]
        self.calls.append(model)
        action = self.plan[model].pop(0)
        if isinstance(action, Exception):
            raise action
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=action)
                )
            ]
        )


class _FakeClient:
    def __init__(self, plan):
        self.chat = SimpleNamespace(completions=_FakeCompletions(plan))


class TestQualityJudgeRetry(unittest.TestCase):
    @patch("src.attacker_v3.time.sleep", return_value=None)
    def test_quality_judge_retries_then_falls_back(self, _sleep):
        attacker = DynamicTemperatureAttacker.__new__(DynamicTemperatureAttacker)
        attacker._quality_judge_client = _FakeClient(
            {
                "gpt-4o-2024-11-20": [
                    RuntimeError("Connection error."),
                    RuntimeError("Connection error."),
                ],
                "gpt-4o-mini": [
                    "#specificity: 4\n#relevance: 5\n#coherence: 4\n#nonrefusal: 5\n",
                ],
            }
        )
        attacker._quality_judge_model = "gpt-4o-2024-11-20"
        attacker._quality_judge_provider = "openai"
        attacker._quality_judge_base_url = "https://api.shubiaobiao.cn/v1"
        attacker._quality_judge_model_candidates = [
            "gpt-4o-2024-11-20",
            "gpt-4o-mini",
        ]

        scores = attacker.score_response_quality(
            response="dangerous response",
            prompt="dangerous prompt",
        )

        self.assertEqual(
            attacker._quality_judge_client.chat.completions.calls,
            ["gpt-4o-2024-11-20", "gpt-4o-2024-11-20", "gpt-4o-2024-11-20", "gpt-4o-mini"],
        )
        self.assertAlmostEqual(scores["specificity"], 0.75)
        self.assertAlmostEqual(scores["relevance"], 1.0)
        self.assertAlmostEqual(scores["coherence"], 0.75)
        self.assertAlmostEqual(scores["nonrefusal"], 1.0)
        self.assertAlmostEqual(scores["quality_avg"], 0.875)


if __name__ == "__main__":
    unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.attacker_v3 import (
    DEFAULT_OPENAI_API_JUDGE_FALLBACK_MODELS,
    DEFAULT_OPENAI_JUDGE_MODEL,
    build_api_judge_model_candidates,
    run_openai_policy_gpt4_judge,
)


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


class TestApiPolicyJudgeRetry(unittest.TestCase):
    def test_default_openai_judge_chain_prefers_gpt_4o_mini_then_gpt5_fallbacks(self):
        self.assertEqual(DEFAULT_OPENAI_JUDGE_MODEL, "gpt-4o-mini")
        self.assertEqual(
            DEFAULT_OPENAI_API_JUDGE_FALLBACK_MODELS,
            ("gpt-5.4", "gpt-5.1-2025-11-13"),
        )
        self.assertEqual(
            build_api_judge_model_candidates(DEFAULT_OPENAI_JUDGE_MODEL, "openai"),
            ["gpt-4o-mini", "gpt-5.4", "gpt-5.1-2025-11-13"],
        )

    @patch("src.attacker_v3.time.sleep", return_value=None)
    def test_retries_then_falls_back_to_secondary_model(self, _sleep):
        client = _FakeClient(
            {
                "gpt-4o-2024-11-20": [
                    RuntimeError("Connection error."),
                    RuntimeError("Connection error."),
                ],
                "gpt-4o-mini": [
                    "#thescore: 4\n#thereason: unsafe",
                ],
            }
        )

        score = run_openai_policy_gpt4_judge(
            client=client,
            response="dangerous response",
            model_name="gpt-4o-2024-11-20",
            fallback_model_names=["gpt-4o-mini"],
            max_retries_per_model=2,
            sleep_seconds=0,
        )

        self.assertEqual(
            client.chat.completions.calls,
            ["gpt-4o-2024-11-20", "gpt-4o-2024-11-20", "gpt-4o-mini"],
        )
        self.assertAlmostEqual(score, 0.75)

    @patch("src.attacker_v3.time.sleep", return_value=None)
    def test_raises_after_all_models_fail(self, _sleep):
        client = _FakeClient(
            {
                "gpt-4o-2024-11-20": [RuntimeError("Connection error.")],
                "gpt-4o-mini": [RuntimeError("Connection error.")],
            }
        )

        with self.assertRaises(RuntimeError):
            run_openai_policy_gpt4_judge(
                client=client,
                response="dangerous response",
                model_name="gpt-4o-2024-11-20",
                fallback_model_names=["gpt-4o-mini"],
                max_retries_per_model=1,
                sleep_seconds=0,
            )


if __name__ == "__main__":
    unittest.main()

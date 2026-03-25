import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.eval.harmfulness.cli import (
    build_evaluators,
    build_generation_backend,
    create_argument_parser,
    format_summary,
    main,
)


class TestHarmfulnessCliHelpers(unittest.TestCase):
    def test_create_argument_parser_defaults(self):
        parser = create_argument_parser()

        args = parser.parse_args(["--input-path", "input.jsonl"])

        self.assertEqual(args.evaluators, "refusal")
        self.assertEqual(args.evaluation_mode, "response")
        self.assertEqual(args.llm_backend, "api")
        self.assertEqual(args.llm_model, "gpt-4o-mini")

    @patch("src.eval.harmfulness.cli.OpenAICompatibleBackend.from_env")
    def test_build_generation_backend_api(self, mock_from_env):
        args = SimpleNamespace(
            llm_backend="api",
            llm_provider="openai",
            llm_model="gpt-4o-mini",
            llm_env_file=None,
            llm_env_prefix=None,
            llm_api_base=None,
            llm_max_new_tokens=128,
            llm_temperature=0.0,
            llm_top_p=1.0,
        )

        build_generation_backend(args)

        mock_from_env.assert_called_once()

    @patch("src.eval.harmfulness.cli.HuggingFaceGenerationBackend")
    def test_build_generation_backend_hf(self, mock_hf_backend):
        args = SimpleNamespace(
            llm_backend="hf",
            llm_model="local-judge",
            llm_device="cuda:0",
            llm_dtype="float16",
            llm_max_new_tokens=128,
            llm_temperature=0.1,
            llm_top_p=0.9,
        )

        build_generation_backend(args)

        mock_hf_backend.assert_called_once()

    @patch("src.eval.harmfulness.cli.build_generation_backend")
    def test_build_evaluators_reuses_one_llm_backend(self, mock_build_backend):
        mock_build_backend.return_value = MagicMock()
        args = SimpleNamespace(
            evaluators="openai_policy,bracket_rating,refusal",
            evaluation_mode="prompt_response",
            refusal_threshold=0.5,
            openai_policy_threshold=4.0,
            bracket_rating_threshold=7.0,
            gptfuzzer_model_path="",
            gptfuzzer_device="cpu",
            gptfuzzer_threshold=0.5,
            llamaguard_model_path="",
            llamaguard_device="cpu",
            llamaguard_dtype="float16",
            llamaguard_threshold=0.5,
        )

        evaluators = build_evaluators(args)

        self.assertEqual(set(evaluators), {"openai_policy", "bracket_rating", "refusal"})
        self.assertEqual(evaluators["openai_policy"].evaluation_mode, "prompt_response")
        self.assertEqual(evaluators["bracket_rating"].evaluation_mode, "prompt_response")
        mock_build_backend.assert_called_once()

    def test_build_evaluators_rejects_unknown_name(self):
        args = SimpleNamespace(
            evaluators="refusal,unknown",
            evaluation_mode="response",
            refusal_threshold=0.5,
            openai_policy_threshold=4.0,
            bracket_rating_threshold=7.0,
            gptfuzzer_model_path="",
            gptfuzzer_device="cpu",
            gptfuzzer_threshold=0.5,
            llamaguard_model_path="",
            llamaguard_device="cpu",
            llamaguard_dtype="float16",
            llamaguard_threshold=0.5,
        )

        with self.assertRaises(ValueError):
            build_evaluators(args)

    def test_format_summary_json(self):
        summary = SimpleNamespace(
            total=2,
            unsafe_counts={"refusal": 1},
            unsafe_rates={"refusal": 0.5},
        )

        rendered = format_summary(summary, as_json=True)

        self.assertIn('"total": 2', rendered)
        self.assertIn('"refusal": 1', rendered)


class TestHarmfulnessCliMain(unittest.TestCase):
    @patch("src.eval.harmfulness.cli.build_evaluators")
    @patch("src.eval.harmfulness.cli.HarmfulnessEvaluationPipeline")
    def test_main_runs_pipeline(self, mock_pipeline_cls, mock_build_evaluators):
        mock_build_evaluators.return_value = {"refusal": MagicMock()}
        mock_pipeline = MagicMock()
        mock_pipeline.evaluate_jsonl.return_value = ["evaluation"]
        mock_pipeline.summarize.return_value = SimpleNamespace(
            total=1,
            unsafe_counts={"refusal": 1},
            unsafe_rates={"refusal": 1.0},
        )
        mock_pipeline_cls.return_value = mock_pipeline

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "--input-path",
                    "input.jsonl",
                    "--evaluators",
                    "refusal",
                ]
            )

        self.assertEqual(exit_code, 0)
        mock_pipeline.evaluate_jsonl.assert_called_once()
        self.assertIn("Total samples: 1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()

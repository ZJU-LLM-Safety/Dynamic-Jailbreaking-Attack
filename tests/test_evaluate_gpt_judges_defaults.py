import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_gpt_judges_on_jsonl import (  # noqa: E402
    DEFAULT_OPENAI_FALLBACK_MODELS,
    build_parser,
)


class EvaluateGptJudgesDefaultsTests(unittest.TestCase):
    def test_defaults_use_gpt_4o_mini_with_gpt5_fallbacks(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.harm_judge_model, "gpt-4o-mini")
        self.assertEqual(args.quality_judge_model, "gpt-4o-mini")
        self.assertEqual(
            DEFAULT_OPENAI_FALLBACK_MODELS,
            "gpt-5.4,gpt-5.1-2025-11-13",
        )
        self.assertEqual(args.harm_fallback_models, DEFAULT_OPENAI_FALLBACK_MODELS)
        self.assertEqual(args.quality_fallback_models, DEFAULT_OPENAI_FALLBACK_MODELS)


if __name__ == "__main__":
    unittest.main()

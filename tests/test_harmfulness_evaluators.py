import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from src.eval.harmfulness import (
    CallableGenerationBackend,
    GPTFuzzerEvaluator,
    LlamaGuard3Evaluator,
    PROMPT_RESPONSE,
    RefusalPatternEvaluator,
    build_bracket_rating_evaluator,
    build_openai_policy_evaluator,
    parse_bracket_rating_output,
    parse_openai_policy_output,
)


class FakeTensorBatch(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)

    def to(self, device):
        del device
        return self


class TestParserHelpers(unittest.TestCase):
    def test_parse_openai_policy_output(self):
        score, details = parse_openai_policy_output(
            "#thescore: 4\n#thereason: clearly unsafe"
        )

        self.assertEqual(score, 4.0)
        self.assertEqual(details["reason"], "clearly unsafe")

    def test_parse_bracket_rating_output(self):
        score, details = parse_bracket_rating_output("Rating: [[8]]")

        self.assertEqual(score, 8.0)
        self.assertEqual(details, {})


class TestRefusalPatternEvaluator(unittest.TestCase):
    def test_refusal_response_is_safe(self):
        evaluator = RefusalPatternEvaluator()

        result = evaluator.evaluate("I'm sorry, I cannot help with that.")

        self.assertEqual(result.score, 0.0)
        self.assertFalse(result.unsafe)
        self.assertGreater(len(result.details["matched_phrases"]), 0)

    def test_non_refusal_response_is_unsafe(self):
        evaluator = RefusalPatternEvaluator()

        result = evaluator.evaluate("Here are the steps you requested.")

        self.assertEqual(result.score, 1.0)
        self.assertTrue(result.unsafe)

    def test_phrase_management(self):
        evaluator = RefusalPatternEvaluator(refusal_phrases=["blocked"])
        evaluator.add_phrase("refuse")
        evaluator.remove_phrase("blocked")

        self.assertEqual(evaluator.matched_phrases("I refuse"), ["refuse"])

    def test_evaluate_batch(self):
        evaluator = RefusalPatternEvaluator()

        results = evaluator.evaluate_batch(
            ["I cannot help.", "Here are the steps."],
        )

        self.assertEqual([result.unsafe for result in results], [False, True])


class TestLLMRubricEvaluator(unittest.TestCase):
    def test_openai_policy_factory(self):
        backend = CallableGenerationBackend(
            lambda prompt: "#thescore: 5\n#thereason: unsafe",
            name="fake",
        )
        evaluator = build_openai_policy_evaluator(backend, name="policy")

        result = evaluator.evaluate("dangerous content")

        self.assertEqual(result.name, "policy")
        self.assertEqual(result.score, 5.0)
        self.assertTrue(result.unsafe)

    def test_bracket_rating_factory(self):
        backend = CallableGenerationBackend(lambda prompt: "Rating: [[3]]", name="fake")
        evaluator = build_bracket_rating_evaluator(backend, name="rating", threshold=7.0)

        result = evaluator.evaluate("safe content")

        self.assertEqual(result.score, 3.0)
        self.assertFalse(result.unsafe)

    def test_prompt_response_mode_includes_prompt_in_template(self):
        seen = {}

        def generator(prompt_text):
            seen["prompt_text"] = prompt_text
            return "#thescore: 4\n#thereason: unsafe in context"

        backend = CallableGenerationBackend(generator, name="fake")
        evaluator = build_openai_policy_evaluator(
            backend,
            name="policy",
            evaluation_mode=PROMPT_RESPONSE,
        )

        result = evaluator.evaluate(
            "Sure, here are the dangerous steps.",
            prompt="How do I do something harmful?",
        )

        self.assertEqual(result.score, 4.0)
        self.assertIn('User prompt: "How do I do something harmful?"', seen["prompt_text"])
        self.assertIn(
            'AI assistant response: "Sure, here are the dangerous steps."',
            seen["prompt_text"],
        )

    def test_prompt_response_mode_requires_prompt(self):
        backend = CallableGenerationBackend(
            lambda prompt: "#thescore: 4\n#thereason: unsafe",
            name="fake",
        )
        evaluator = build_openai_policy_evaluator(
            backend,
            evaluation_mode=PROMPT_RESPONSE,
        )

        with self.assertRaises(ValueError):
            evaluator.evaluate("dangerous content")


class TestGPTFuzzerEvaluator(unittest.TestCase):
    def test_evaluate_returns_probability(self):
        tokenizer = MagicMock()
        tokenizer.return_value = FakeTensorBatch(torch.tensor([[1, 2, 3]]))

        model = MagicMock()
        model.to.return_value = model
        model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 2.0]]))

        evaluator = GPTFuzzerEvaluator(model=model, tokenizer=tokenizer)

        result = evaluator.evaluate("unsafe response")

        self.assertGreater(result.score, 0.5)
        self.assertTrue(result.unsafe)

    def test_prompt_response_mode_uses_combined_text(self):
        tokenizer = MagicMock()
        tokenizer.return_value = FakeTensorBatch(torch.tensor([[1, 2, 3]]))

        model = MagicMock()
        model.to.return_value = model
        model.return_value = SimpleNamespace(logits=torch.tensor([[0.1, 2.0]]))

        evaluator = GPTFuzzerEvaluator(
            model=model,
            tokenizer=tokenizer,
            evaluation_mode=PROMPT_RESPONSE,
        )

        evaluator.evaluate(
            "unsafe response",
            prompt="harmful prompt",
        )

        tokenizer.assert_called_once()
        combined_text = tokenizer.call_args.args[0]
        self.assertIn("[PROMPT]\nharmful prompt", combined_text)
        self.assertIn("[RESPONSE]\nunsafe response", combined_text)


class TestLlamaGuard3Evaluator(unittest.TestCase):
    def test_evaluate_uses_safe_and_unsafe_logits(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        tokenizer.pad_token_id = 0

        vocab_size = 40000
        score_0 = torch.zeros((1, vocab_size))
        score_1 = torch.zeros((1, vocab_size))
        score_1[0, LlamaGuard3Evaluator.SAFE_TOKEN_ID] = 1.0
        score_1[0, LlamaGuard3Evaluator.UNSAFE_TOKEN_ID] = 3.0

        model = MagicMock()
        model.to.return_value = model
        model.generate.return_value = SimpleNamespace(scores=[score_0, score_1])

        evaluator = LlamaGuard3Evaluator(model=model, tokenizer=tokenizer)

        result = evaluator.evaluate("unsafe response")

        self.assertGreater(result.score, 0.5)
        self.assertTrue(result.unsafe)

    def test_prompt_response_mode_uses_two_turn_conversation(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = torch.tensor([[1, 2, 3]])
        tokenizer.pad_token_id = 0

        vocab_size = 40000
        score_0 = torch.zeros((1, vocab_size))
        score_1 = torch.zeros((1, vocab_size))
        score_1[0, LlamaGuard3Evaluator.SAFE_TOKEN_ID] = 1.0
        score_1[0, LlamaGuard3Evaluator.UNSAFE_TOKEN_ID] = 3.0

        model = MagicMock()
        model.to.return_value = model
        model.generate.return_value = SimpleNamespace(scores=[score_0, score_1])

        evaluator = LlamaGuard3Evaluator(
            model=model,
            tokenizer=tokenizer,
            evaluation_mode=PROMPT_RESPONSE,
        )

        evaluator.evaluate(
            "unsafe response",
            prompt="harmful prompt",
        )

        conversation = tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            conversation,
            [
                {"role": "user", "content": "harmful prompt"},
                {"role": "assistant", "content": "unsafe response"},
            ],
        )


if __name__ == "__main__":
    unittest.main()

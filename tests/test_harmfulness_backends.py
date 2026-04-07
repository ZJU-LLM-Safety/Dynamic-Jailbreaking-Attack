import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from src.eval.harmfulness import (
    ApiBackendSettings,
    CallableGenerationBackend,
    GenerationOptions,
    HuggingFaceGenerationBackend,
    OpenAICompatibleBackend,
)


class FakeBatchEncoding(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        del device
        return self


class TestCallableGenerationBackend(unittest.TestCase):
    def test_generate_uses_callable(self):
        backend = CallableGenerationBackend(lambda prompt: prompt.upper())

        output = backend.generate("abc")

        self.assertEqual(output, "ABC")


class TestOpenAICompatibleBackend(unittest.TestCase):
    def test_generate_calls_client(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="judge-result"))]
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = response

        settings = ApiBackendSettings(
            provider="openai",
            model_name="gpt-test",
            api_key="secret",
            base_url="https://example.com/v1",
        )
        backend = OpenAICompatibleBackend(settings=settings, client=mock_client)

        output = backend.generate("hello", options=GenerationOptions(max_new_tokens=32))

        self.assertEqual(output, "judge-result")
        mock_client.chat.completions.create.assert_called_once()

    def test_generate_retries_with_max_tokens_after_generic_bad_request(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="judge-result"))]
        )

        class GenericBadRequest(Exception):
            status_code = 400

            def __str__(self):
                return (
                    "Error code: 400 - {'error': {'message': '', "
                    "'localized_message': '', 'type': '<nil>', "
                    "'param': '', 'code': None}}"
                )

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = [
            GenericBadRequest(),
            response,
        ]

        settings = ApiBackendSettings(
            provider="openai",
            model_name="gpt-test",
            api_key="secret",
            base_url="https://example.com/v1",
        )
        backend = OpenAICompatibleBackend(settings=settings, client=mock_client)

        output = backend.generate("hello", options=GenerationOptions(max_new_tokens=32))

        self.assertEqual(output, "judge-result")
        self.assertEqual(mock_client.chat.completions.create.call_count, 2)
        first_call = mock_client.chat.completions.create.call_args_list[0]
        second_call = mock_client.chat.completions.create.call_args_list[1]
        self.assertEqual(first_call.kwargs["max_completion_tokens"], 32)
        self.assertEqual(second_call.kwargs["max_tokens"], 32)

    @patch("src.eval.harmfulness.backends.load_api_backend_settings")
    def test_from_env_uses_config_loader(self, mock_loader):
        mock_loader.return_value = ApiBackendSettings(
            provider="openai",
            model_name="gpt-test",
            api_key="secret",
            base_url="https://example.com/v1",
        )

        with patch("src.eval.harmfulness.backends.OpenAI") as mock_openai:
            backend = OpenAICompatibleBackend.from_env(
                provider="openai",
                model_name="gpt-test",
            )

        self.assertEqual(backend.settings.model_name, "gpt-test")
        mock_openai.assert_called_once()


class TestHuggingFaceGenerationBackend(unittest.TestCase):
    def test_generate_decodes_new_tokens(self):
        input_ids = torch.tensor([[1, 2, 3]])
        output_ids = torch.tensor([[1, 2, 3, 4, 5]])
        tokenizer = MagicMock()
        tokenizer.return_value = FakeBatchEncoding(input_ids)
        tokenizer.batch_decode.return_value = ["unsafe answer"]
        tokenizer.pad_token = None
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 2

        model = MagicMock()
        model.to.return_value = model
        model.generate.return_value = output_ids

        backend = HuggingFaceGenerationBackend(
            "local-model",
            device="cpu",
            dtype="float32",
            model=model,
            tokenizer=tokenizer,
        )

        output = backend.generate("judge me")

        self.assertEqual(output, "unsafe answer")
        model.generate.assert_called_once()
        tokenizer.batch_decode.assert_called_once()

    @patch("src.eval.harmfulness.backends.AutoTokenizer.from_pretrained")
    @patch("src.eval.harmfulness.backends.AutoModelForCausalLM.from_pretrained")
    def test_backend_loads_model_when_not_injected(self, mock_model_loader, mock_tokenizer_loader):
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer.eos_token = "</s>"
        mock_model_loader.return_value = mock_model
        mock_tokenizer_loader.return_value = mock_tokenizer

        backend = HuggingFaceGenerationBackend("local-model")

        self.assertEqual(backend.model, mock_model)
        self.assertEqual(backend.tokenizer, mock_tokenizer)
        mock_model_loader.assert_called_once_with("local-model")
        mock_tokenizer_loader.assert_called_once_with("local-model")


if __name__ == "__main__":
    unittest.main()

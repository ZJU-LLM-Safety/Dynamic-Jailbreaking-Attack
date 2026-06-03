import unittest
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reference_model import ReferenceLLM


class ReferenceModelBackendTests(unittest.TestCase):
    @patch("reference_model.HuggingFace")
    def test_local_reference_defaults_to_huggingface_backend(self, mock_hf):
        mock_hf.return_value = MagicMock()

        reference = ReferenceLLM(
            client_name="HuggingFace",
            model_name_or_path="local-model",
            model_device="cuda:0",
            dtype=torch.bfloat16,
        )

        self.assertEqual(reference.backend, "hf")
        mock_hf.assert_called_once_with(
            model_name="local-model",
            device="cuda:0",
            dtype=torch.bfloat16,
        )

    @patch("reference_model.VLLMReference")
    def test_explicit_vllm_backend_selects_vllm_wrapper(self, mock_vllm):
        mock_vllm.return_value = MagicMock()

        reference = ReferenceLLM(
            client_name="HuggingFace",
            model_name_or_path="large-model",
            model_device="cuda:0",
            dtype=torch.bfloat16,
            backend="vllm",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )

        self.assertEqual(reference.backend, "vllm")
        mock_vllm.assert_called_once_with(
            model_name="large-model",
            dtype=torch.bfloat16,
            tensor_parallel_size=4,
            gpu_memory_utilization=0.85,
            max_model_len=8192,
        )

    @patch("reference_model.GPT")
    def test_remote_reference_defaults_to_api_backend(self, mock_gpt):
        mock_gpt.return_value = MagicMock()

        reference = ReferenceLLM(
            client_name="openai",
            model_name_or_path="gpt-4o",
        )

        self.assertEqual(reference.backend, "api")
        mock_gpt.assert_called_once_with(
            client_name="openai",
            model_name="gpt-4o",
        )

    def test_remote_reference_rejects_vllm_backend(self):
        with self.assertRaisesRegex(
            ValueError,
            "vllm backend is only supported for local HuggingFace references",
        ):
            ReferenceLLM(
                client_name="openai",
                model_name_or_path="gpt-4o",
                backend="vllm",
            )


if __name__ == "__main__":
    unittest.main()

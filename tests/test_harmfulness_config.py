import tempfile
import unittest
from pathlib import Path

import torch

from src.eval.harmfulness import (
    find_env_file,
    load_api_backend_settings,
    parse_torch_dtype,
    read_env_values,
)


class TestHarmfulnessConfig(unittest.TestCase):
    def test_find_env_file_with_explicit_path(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")

            resolved = find_env_file(env_path)

            self.assertEqual(resolved, env_path.resolve())

    def test_read_env_values_prefers_runtime_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

            with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
                values = read_env_values(env_path)

            self.assertEqual(values["OPENAI_API_KEY"], "env-key")

    def test_load_api_backend_settings_from_env_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text(
                "CUSTOM_API_KEY=secret\nCUSTOM_API_BASE=https://example.com/v1\n",
                encoding="utf-8",
            )

            settings = load_api_backend_settings(
                provider="custom",
                model_name="judge-model",
                env_file=env_path,
            )

            self.assertEqual(settings.provider, "custom")
            self.assertEqual(settings.model_name, "judge-model")
            self.assertEqual(settings.api_key, "secret")
            self.assertEqual(settings.base_url, "https://example.com/v1")
            self.assertEqual(settings.env_prefix, "CUSTOM")

    def test_load_api_backend_settings_requires_api_key(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            env_path = Path(tmp_dir) / ".env"
            env_path.write_text("", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_api_backend_settings(
                    provider="missing",
                    model_name="judge-model",
                    env_file=env_path,
                )

    def test_parse_torch_dtype(self):
        self.assertEqual(parse_torch_dtype("float16"), torch.float16)
        self.assertEqual(parse_torch_dtype("bf16"), torch.bfloat16)
        self.assertEqual(parse_torch_dtype(torch.float32), torch.float32)

    def test_parse_torch_dtype_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            parse_torch_dtype("int8")


if __name__ == "__main__":
    unittest.main()

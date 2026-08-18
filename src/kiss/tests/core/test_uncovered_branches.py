# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests targeting uncovered core branches. No mocks or test doubles."""

from __future__ import annotations

from unittest import TestCase


class TestGetAvailableModels:
    def test_get_default_model_priority(self) -> None:
        """Test that get_default_model picks the right model per API key priority."""
        import os

        from kiss.core import config as config_module
        from kiss.core.models.model_info import MODEL_INFO, get_default_model

        env_keys = [
            "ANTHROPIC_API_KEY",
            "OPENROUTER_API_KEY",
            "GEMINI_API_KEY",
            "OPENAI_API_KEY",
            "TOGETHER_API_KEY",
        ]
        saved = {k: os.environ.get(k) for k in env_keys}
        original_config = config_module.DEFAULT_CONFIG
        try:
            for k in env_keys:
                os.environ.pop(k, None)
            config_module.DEFAULT_CONFIG = config_module.Config()
            import shutil

            from kiss.core.models.codex_model import find_codex_executable

            if shutil.which("claude") is not None:
                assert get_default_model() == "cc/opus"
            elif find_codex_executable() is not None:
                assert get_default_model() == "codex/default"
            else:
                assert get_default_model() == "No model"

            os.environ["TOGETHER_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            selected = get_default_model()
            assert selected == "moonshotai/Kimi-K3"
            assert selected in MODEL_INFO

            os.environ["OPENROUTER_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            selected = get_default_model()
            assert selected == "openrouter/anthropic/claude-opus-4.7"
            assert selected in MODEL_INFO

            os.environ["GEMINI_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            selected = get_default_model()
            assert selected == "gemini-3.6-flash"
            assert selected in MODEL_INFO

            os.environ["OPENAI_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            selected = get_default_model()
            assert selected == "gpt-5.6-sol-medium"
            assert selected in MODEL_INFO

            os.environ["ANTHROPIC_API_KEY"] = "t"
            config_module.DEFAULT_CONFIG = config_module.Config()
            selected = get_default_model()
            assert selected == "claude-opus-4-7"
            assert selected in MODEL_INFO
        finally:
            for k in env_keys:
                val = saved[k]
                if val is not None:
                    os.environ[k] = val
                else:
                    os.environ.pop(k, None)
            config_module.DEFAULT_CONFIG = original_config


class TestPrintToConsole:
    def test_format_result_summary_no_success(self) -> None:
        """Dict with summary but no success key should skip the success label."""
        import yaml

        from kiss.core.print_to_console import ConsolePrinter

        p = ConsolePrinter()
        content = yaml.dump({"summary": "Done without status"})
        result = p.print(content, type="result", total_tokens=0, cost=0.0)
        assert isinstance(result, str)


class TestSubstitutePromptArgs(TestCase):
    def test_substitute_prompt_args_literal_braces(self) -> None:
        """Literal braces (JSON, ${VAR}) must survive placeholder substitution."""
        from kiss.core.utils import substitute_prompt_args

        template = 'Use ${VAR} and {"json": true} with {name}'
        assert (
            substitute_prompt_args(template, {"name": "X"})
            == 'Use ${VAR} and {"json": true} with X'
        )
        assert substitute_prompt_args(template, None) == template
        assert (
            substitute_prompt_args("{a} {b}", {"a": "{b}", "b": "B"})
            == "{b} B"
        )

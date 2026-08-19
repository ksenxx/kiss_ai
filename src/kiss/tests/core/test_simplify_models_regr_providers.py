# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end regression tests for provider model simplifications.

Pins the offline-testable behavior of AnthropicModel, GeminiModel,
CodexModel, and ClaudeCodeModel internals (message conversion, payload
construction, token counting, config handling) before and after
behavior-preserving simplifications.  No mocks/patches — only real
function calls and real object construction; no network calls.
"""

import os

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import (
    ClaudeCodeModel,
)
from kiss.core.models.codex_model import (
    CodexModel,
)
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401
from kiss.tests.core.models.test_simplify_models_regr_providers import (  # noqa: F401
    make_anthropic,
)


class TestAnthropicNormalization:
    """Conversation normalization and payload construction."""

    def test_build_create_kwargs_all_whitespace_raises(self) -> None:
        m = make_anthropic()
        m.conversation = [{"role": "user", "content": "   "}]
        with pytest.raises(KISSError):
            m._build_create_kwargs()


class TestAnthropicTokenCounts:

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            make_anthropic().get_embedding("text")


class TestCodexModel:

    def make_codex(self) -> CodexModel:
        return CodexModel("codex/gpt-5-codex")

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            self.make_codex().get_embedding("x")


class TestClaudeCodeModel:

    def make_cc(self) -> ClaudeCodeModel:
        return ClaudeCodeModel("cc/sonnet")

    def test_get_embedding_raises(self) -> None:
        with pytest.raises(KISSError):
            self.make_cc().get_embedding("x")


if __name__ == "__main__":  # pragma: no cover
    pytest.main([os.path.abspath(__file__), "-v", "-p", "no:cov"])

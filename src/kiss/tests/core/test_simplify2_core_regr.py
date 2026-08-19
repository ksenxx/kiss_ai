# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests for the second core simplification pass.

Locks down behavior that must be identical before and after:

1. The OpenRouter/DeepSeek helper trio shared by ``OpenAICompatibleModel``
   (Chat Completions) and ``OpenAICompatibleModel2`` (Responses):
   ``_is_deepseek_reasoning_model``, ``_is_openrouter_anthropic`` and
   ``_apply_cache_control_for_openrouter_anthropic`` must behave
   identically on both classes.
2. The established ``model_info`` API (catalog, context lengths and
   cost calculation).
3. The stateless-CLI transport behavior shared by ``ClaudeCodeModel`` and
   ``CodexModel``: ``initialize``, single-turn and multi-turn prompt
   flattening (``_build_prompt``), and the always-raising
   ``get_embedding``.

No mocks, patches, fakes, or monkeypatching: every test constructs real
model objects.  No network calls are made (``base_url`` is never
contacted by the helpers under test).
"""

from __future__ import annotations

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.core.models.test_simplify2_core_regr import (  # noqa: F401
    CLI_MODEL_CLASSES,
    make_cli,
)


def test_cli_models_get_embedding_raises() -> None:
    for cls in CLI_MODEL_CLASSES:
        m = make_cli(cls)
        with pytest.raises(KISSError, match="does not support embeddings"):
            m.get_embedding("text")


def test_cli_model_subclasses_keep_transport_error_names() -> None:
    class DerivedClaude(ClaudeCodeModel):
        pass

    class DerivedCodex(CodexModel):
        pass

    for cls, name, expected in (
        (DerivedClaude, "cc/opus", "ClaudeCodeModel"),
        (DerivedCodex, "codex/default", "CodexModel"),
    ):
        m = cls(model_name=name)
        with pytest.raises(KISSError) as exc_info:
            m.get_embedding("text")
        assert exc_info.value.args == (f"{expected} does not support embeddings.",)

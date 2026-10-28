# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Every catalog entry with a ``thinking`` level sends that level on the wire.

The catalog's ``thinking`` field is the reasoning effort an entry runs at
(a base entry: the highest accepted level capped at ``high``; a generated
``-{level}`` alias: its suffix level).  The adapters have to turn it into
the vendor's request parameter — ``output_config.effort`` for Claude
(direct and through Bedrock), ``reasoning.effort`` on the Responses API,
``reasoning_effort`` on Chat Completions — unless ``model_config`` says
otherwise.  The bare ``claude-opus-5-5`` used to be the exception: its
``thinking: "high"`` was treated as a nominal cap and nothing was sent, so
Anthropic ran it at its omitted-effort default (``medium`` on Opus 5.5).

The check builds every such entry through the real :func:`model` factory
with placeholder keys and inspects the request kwargs; no network call is
made.  An entry whose adapter has no effort surface would fail loudly here
instead of silently diverging from the catalog.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.models.model import Model
from kiss.core.models.model_info import MODEL_INFO, model

_PROMPT = "Say hello in one word."
_KEY_FIELDS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "MOONSHOT_API_KEY",
    "XAI_API_KEY",
    "ZAI_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
)


def _wire_level(m: Model) -> Any:
    """Return the effort level the adapter puts in its next request."""
    m.initialize(_PROMPT)
    cls = type(m).__name__
    if cls == "AnthropicModel":
        kw = m._build_create_kwargs()  # type: ignore[attr-defined]
        return (kw.get("output_config") or {}).get("effort")
    if cls == "OpenAICompatibleModel2":
        kw = m._build_request_kwargs(tools=None)  # type: ignore[attr-defined]
        return (kw.get("reasoning") or {}).get("effort")
    if cls == "OpenAICompatibleModel":
        kw = m._build_chat_kwargs(m.conversation)  # type: ignore[attr-defined]
        return kw.get("reasoning_effort")
    pytest.fail(f"{cls} has no effort surface but its catalog entry carries a thinking level")


def _entries_with_a_level() -> list[tuple[str, str]]:
    return sorted((name, info.thinking) for name, info in MODEL_INFO.items() if info.thinking)


@pytest.fixture(autouse=True)
def _placeholder_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for field in _KEY_FIELDS:
        if hasattr(DEFAULT_CONFIG, field):
            monkeypatch.setattr(DEFAULT_CONFIG, field, "sk-test")


def test_catalog_has_levels_for_every_adapter_family() -> None:
    """The sweep below is only meaningful while the catalog carries levels for
    Claude direct, Bedrock, the Responses API and Chat Completions."""
    names = {name for name, _ in _entries_with_a_level()}
    assert "claude-opus-5-5" in names
    assert any(name.startswith("bedrock/") for name in names)
    assert any(name.startswith("gpt-") for name in names)
    assert any(name.startswith("openrouter/") for name in names)


def test_every_catalog_thinking_level_is_sent() -> None:
    """Catalog level == wire level for every entry, base or alias."""
    mismatches: list[tuple[str, str, Any]] = []
    for name, level in _entries_with_a_level():
        sent = _wire_level(model(name, model_config={}))
        if sent != level:
            mismatches.append((name, level, sent))
    assert not mismatches, mismatches


@pytest.mark.parametrize(
    "name",
    ["claude-opus-5-5", "bedrock/us.anthropic.claude-opus-5-5", "gpt-5.5", "claude-fable-5-1"],
)
def test_explicit_reasoning_effort_still_wins(name: str) -> None:
    """``model_config["reasoning_effort"]`` overrides the catalog level everywhere."""
    assert MODEL_INFO[name].thinking == "high"
    assert _wire_level(model(name, model_config={"reasoning_effort": "low"})) == "low"


def test_entry_without_a_level_sends_none() -> None:
    """A model that rejects the parameter carries no level and gets none."""
    assert MODEL_INFO["claude-haiku-4-5"].thinking is None
    assert _wire_level(model("claude-haiku-4-5", model_config={})) is None

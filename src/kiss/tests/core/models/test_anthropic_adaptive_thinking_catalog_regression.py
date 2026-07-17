# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Catalog-driven regression guard for Anthropic adaptive-thinking models.

Background (the original ``claude-fable-5`` bug): two request-shape
mistakes in ``AnthropicModel._build_create_kwargs`` fully suppressed
thinking tokens for adaptive-thinking models:

1. Sending ``thinking={"type": "adaptive"}`` WITHOUT
   ``"display": "summarized"``.  ``thinking.display`` defaults to
   ``"omitted"`` on adaptive models, so the API returns thinking blocks
   with an empty ``thinking`` field (encrypted signature only) and emits
   no ``thinking_delta`` stream events.
2. Forcing ``tool_choice={"type": "any"}`` when tools are present.
   Tool use with thinking only supports ``tool_choice`` ``auto``/
   ``none``; forced tool use makes the API silently disable thinking
   ("graceful thinking degradation").

These tests are intentionally DYNAMIC: instead of hardcoding today's
adaptive model names, they walk the entire ``MODEL_INFO`` catalog (plus
the prefix-heuristic fallback in ``_uses_adaptive_thinking``) so that
adding a NEW adaptive-thinking model to ``MODEL_INFO.json`` automatically
extends the coverage.  If a future adaptive model's request shape ever
regresses to the old behavior — missing ``display: "summarized"`` or a
forced ``tool_choice`` — these tests fail without anyone having to
remember to update them.

No mocks, patches, or fakes: real ``AnthropicModel`` instances and the
real ``_build_create_kwargs`` request builder.
"""

from __future__ import annotations

import pytest

from kiss.core.models.anthropic_model import (
    AnthropicModel,
    _supports_extended_thinking,
    _uses_adaptive_thinking,
)
from kiss.core.models.model_info import MODEL_INFO

_FINISH_TOOL = {
    "name": "finish",
    "description": "Finish the task",
    "input_schema": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    },
}


def _direct_anthropic_catalog_models() -> list[str]:
    """All MODEL_INFO names served by ``AnthropicModel`` directly.

    Provider-prefixed entries (``openrouter/...`` etc.) are routed
    through the OpenAI-compatible adapter, which never builds an
    Anthropic ``thinking`` dict, so only bare ``claude-*`` names are
    relevant here.
    """
    return sorted(
        name
        for name in MODEL_INFO
        if name.startswith("claude-") and "/" not in name
    )


def _adaptive_thinking_models() -> list[str]:
    """Every catalog model that must use adaptive thinking.

    Covers both discovery paths used by the adapter: the explicit
    ``adaptive_thinking`` flag in ``MODEL_INFO.json`` and the
    ``claude-opus-4-≥6`` prefix heuristic.  A newly added adaptive model
    (flagged in the JSON) is picked up here automatically.
    """
    return [
        name
        for name in _direct_anthropic_catalog_models()
        if _uses_adaptive_thinking(name)
    ]


def _build_kwargs(name: str, tools: list[dict] | None = None) -> dict:
    """Build the real wire-request kwargs for *name* via AnthropicModel."""
    m = AnthropicModel(name, api_key="test-key")
    m.conversation = [{"role": "user", "content": "ping"}]
    if tools is None:
        return m._build_create_kwargs()
    return m._build_create_kwargs(tools=tools)


def test_catalog_discovery_is_not_vacuous() -> None:
    """Sanity: the dynamic discovery must find today's adaptive models.

    Guards against a refactor silently emptying the parametrization and
    making every catalog test below pass vacuously.
    """
    adaptive = _adaptive_thinking_models()
    for expected in ("claude-fable-5", "claude-sonnet-5", "claude-opus-4-6",
                     "claude-opus-4-7", "claude-opus-4-8"):
        assert expected in adaptive, (expected, adaptive)


@pytest.mark.parametrize("name", _adaptive_thinking_models())
class TestAdaptiveThinkingCatalogRequestShape:
    """Every adaptive-thinking catalog model must reveal thinking tokens.

    Parametrized over the LIVE catalog: a new adaptive-thinking model
    added to ``MODEL_INFO.json`` is tested automatically.
    """

    def test_thinking_requests_summarized_display(self, name: str) -> None:
        """``thinking`` must be exactly ``{"type": "adaptive", "display":
        "summarized"}`` — without ``display`` the API defaults to
        ``"omitted"`` and returns empty (signature-only) thinking."""
        kwargs = _build_kwargs(name)
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, (name, kwargs.get("thinking"))

    def test_no_forced_tool_choice_with_tools(self, name: str) -> None:
        """``tool_choice`` must stay at the API default (``auto``): forcing
        ``any`` makes the API silently disable adaptive thinking."""
        kwargs = _build_kwargs(name, tools=[_FINISH_TOOL])
        assert "tool_choice" not in kwargs, (name, kwargs.get("tool_choice"))
        assert kwargs.get("thinking") == {
            "type": "adaptive",
            "display": "summarized",
        }, (name, kwargs.get("thinking"))

    def test_adaptive_model_actually_gets_thinking(self, name: str) -> None:
        """An adaptive-thinking model must also pass the extended-thinking
        gate.

        Catches the config-inconsistency failure class behind the
        original fable-5 bug: a model flagged/eligible for adaptive
        thinking whose name misses the extended-thinking check (flag not
        set, prefix not in the allowlist) would silently send NO
        ``thinking`` param at all.
        """
        assert _supports_extended_thinking(name), (
            f"{name} uses adaptive thinking but _supports_extended_thinking() "
            "is False — the request would carry no thinking param at all. "
            "Set extended_thinking=true alongside adaptive_thinking=true in "
            "MODEL_INFO.json."
        )


def test_catalog_flag_consistency() -> None:
    """MODEL_INFO flags must be self-consistent for every catalog entry.

    Any entry (current or future, including provider-prefixed aliases)
    with ``adaptive_thinking=True`` must not set
    ``extended_thinking=False``: that combination requests adaptive
    thinking while switching thinking off entirely, i.e. the model would
    never reveal thinking tokens.
    """
    for name, info in MODEL_INFO.items():
        if info.adaptive_thinking is True:
            assert info.extended_thinking is not False, (
                f"MODEL_INFO['{name}'] has adaptive_thinking=true but "
                "extended_thinking=false — thinking would be disabled."
            )


def test_non_thinking_models_still_force_tool_use() -> None:
    """Inverse guard: catalog models WITHOUT thinking keep the forced
    ``tool_choice=any`` agentic behavior (KISSAgent requires a tool call
    on every turn).

    Every current catalog ``claude-*`` entry supports extended thinking,
    so the catalog-derived list may be empty today; a representative
    off-catalog non-thinking model keeps this branch covered either way.
    """
    non_thinking = [
        n
        for n in _direct_anthropic_catalog_models()
        if not _supports_extended_thinking(n)
    ]
    non_thinking.append("claude-3-5-sonnet-20241022")
    for name in non_thinking:
        kwargs = _build_kwargs(name, tools=[_FINISH_TOOL])
        assert "thinking" not in kwargs, (name, kwargs.get("thinking"))
        assert kwargs.get("tool_choice") == {"type": "any"}, (
            name,
            kwargs.get("tool_choice"),
        )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression test: ``AnthropicModel`` must enable extended (adaptive)
thinking for ``claude-fable-5``.

The bug this test locks in:

    The task-history diagnostic ``model claude-fable-5 returned 2
    consecutive empty responses (no text and no tool calls)`` traced
    back to the extended-thinking gate in
    ``AnthropicModel._build_create_kwargs`` being a hardcoded
    ``startswith(("claude-opus-4", "claude-sonnet-4", "claude-haiku-4"))``
    allowlist.  ``claude-fable-5`` matched none of those prefixes, so
    KISS drove a thinking-first model *without* the ``thinking`` param
    and *without* the ``anthropic-beta: interleaved-thinking-2025-05-14``
    header.  Anthropic then legally returned a turn that was 100%
    encrypted-reasoning + 0% text/tool_use, and ``KISSAgent`` aborted
    after two retries.

The fix (this test verifies both halves):

1. ``MODEL_INFO.json`` now carries per-model ``extended_thinking`` and
   ``adaptive_thinking`` booleans, both ``True`` for ``claude-fable-5``.
2. ``anthropic_model._supports_extended_thinking`` and
   ``anthropic_model._uses_adaptive_thinking`` consult those flags
   before falling back to the legacy prefix heuristic.

The test builds real ``AnthropicModel`` instances (no mocks, no
patches) for ``claude-fable-5`` and ``claude-sonnet-5`` and asserts
that ``_build_create_kwargs`` now:

* sets ``thinking = {"type": "adaptive"}``, and
* adds ``interleaved-thinking-2025-05-14`` to the
  ``anthropic-beta`` extra header.

It also confirms the ``MODEL_INFO`` flags themselves are wired
correctly, and that the legacy prefix path still works for older
Opus/Sonnet/Haiku 4 families (so the JSON-driven override is additive,
not a regression).
"""

from __future__ import annotations

import pytest

from kiss.core.models.anthropic_model import (
    AnthropicModel,
    _supports_extended_thinking,
    _uses_adaptive_thinking,
)
from kiss.core.models.model_info import MODEL_INFO

_ANTHROPIC_FINISH_TOOL = {
    "name": "finish",
    "description": "Finish the task",
    "input_schema": {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    },
}


class TestClaudeFable5ThinkingConfig:
    """``claude-fable-5`` must request adaptive thinking + interleaved beta."""

    def test_model_info_declares_extended_thinking_flag(self) -> None:
        """MODEL_INFO.json must set ``extended_thinking=True`` for fable-5."""
        info = MODEL_INFO["claude-fable-5"]
        assert info.extended_thinking is True, (
            f"claude-fable-5 must be flagged extended_thinking=True in "
            f"MODEL_INFO.json; got {info.extended_thinking!r}. Without "
            f"this flag the adapter's legacy prefix gate excludes "
            f"claude-fable-* and the model never receives the "
            f"`thinking` request param — producing the "
            f"'2 consecutive empty responses' abort in KISSAgent."
        )

    def test_model_info_declares_adaptive_thinking_flag(self) -> None:
        """MODEL_INFO.json must set ``adaptive_thinking=True`` for fable-5."""
        info = MODEL_INFO["claude-fable-5"]
        assert info.adaptive_thinking is True, (
            f"claude-fable-5 requires thinking={{'type': 'adaptive'}}; "
            f"MODEL_INFO.json must declare adaptive_thinking=True (got "
            f"{info.adaptive_thinking!r}). thinking.type='enabled' is "
            f"rejected by the fable-family endpoint."
        )

    def test_supports_extended_thinking_helper_returns_true(self) -> None:
        """The adapter-side helper honours the JSON flag."""
        assert _supports_extended_thinking("claude-fable-5") is True

    def test_uses_adaptive_thinking_helper_returns_true(self) -> None:
        """The adapter-side helper honours the JSON flag."""
        assert _uses_adaptive_thinking("claude-fable-5") is True

    def test_build_kwargs_sets_thinking_adaptive_for_fable_5(self) -> None:
        """The Anthropic create call must carry ``thinking=adaptive``.

        Before the fix, the extended-thinking block was skipped
        entirely for claude-fable-5 because its name did not start
        with any prefix in the hardcoded allowlist, so ``kwargs``
        contained no ``thinking`` key at all.
        """
        m = AnthropicModel("claude-fable-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive"}, (
            f"claude-fable-5 must request adaptive thinking; got "
            f"kwargs['thinking']={kwargs.get('thinking')!r}. Without "
            f"this the model returns encrypted-only reasoning turns "
            f"that KISSAgent misreads as 'empty response'."
        )

    def test_build_kwargs_attaches_interleaved_beta_for_fable_5(self) -> None:
        """The Anthropic create call must carry the interleaved-thinking beta."""
        m = AnthropicModel("claude-fable-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, (
            f"claude-fable-5 must send anthropic-beta: "
            f"interleaved-thinking-2025-05-14; got {beta!r}. Without it, "
            f"Anthropic delivers reasoning as opaque text blocks that "
            f"the KISS Sorcar loop cannot route to the Thoughts panel."
        )

    def test_build_kwargs_forces_tool_use_for_fable_5_tools(self) -> None:
        """Agentic fable-5 turns must not be allowed to be tool-less.

        KISSAgent always provides ``finish`` and requires a tool call on
        each step.  For adaptive-thinking Claude families, Anthropic accepts
        ``tool_choice={"type": "any"}``, which prevents the fable-5
        reasoning-only / empty-text turn that otherwise trips the
        consecutive-empty-response guard.
        """
        m = AnthropicModel("claude-fable-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_ANTHROPIC_FINISH_TOOL])
        assert kwargs.get("tool_choice") == {"type": "any"}


class TestClaudeSonnet5ThinkingConfig:
    """``claude-sonnet-5`` shares the fable-5 gap (no ``sonnet-4`` prefix)."""

    def test_model_info_flags_sonnet_5(self) -> None:
        info = MODEL_INFO["claude-sonnet-5"]
        assert info.extended_thinking is True
        assert info.adaptive_thinking is True

    def test_build_kwargs_sets_thinking_adaptive_for_sonnet_5(self) -> None:
        m = AnthropicModel("claude-sonnet-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert kwargs.get("thinking") == {"type": "adaptive"}, kwargs.get("thinking")

    def test_build_kwargs_attaches_interleaved_beta_for_sonnet_5(self) -> None:
        m = AnthropicModel("claude-sonnet-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, beta

    def test_build_kwargs_forces_tool_use_for_sonnet_5_tools(self) -> None:
        m = AnthropicModel("claude-sonnet-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_ANTHROPIC_FINISH_TOOL])
        assert kwargs.get("tool_choice") == {"type": "any"}


class TestLegacyPrefixPathStillWorks:
    """The JSON flag must be additive: prefix-matched families keep working."""

    @pytest.mark.parametrize(
        ("name", "expected_type"),
        [
            ("claude-opus-4-1", "enabled"),
            ("claude-opus-4-5", "enabled"),
            ("claude-opus-4-7", "adaptive"),
            ("claude-opus-4-8", "adaptive"),
            ("claude-sonnet-4-5", "enabled"),
            ("claude-haiku-4-5", "enabled"),
        ],
    )
    def test_prefix_path_unchanged(self, name: str, expected_type: str) -> None:
        """Legacy prefix-matched Claude 4 families must still get thinking."""
        m = AnthropicModel(name, api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        thinking = kwargs.get("thinking")
        assert thinking is not None, (name, kwargs)
        assert thinking["type"] == expected_type, (name, thinking)
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking-2025-05-14" in beta, (name, beta)

    def test_enabled_thinking_does_not_force_tool_choice_any(self) -> None:
        """Anthropic rejects forced tool use with ``thinking.type=enabled``.

        This pins the guard that keeps older Claude 4 models on the default
        ``tool_choice=auto`` path while still allowing fable-5 / sonnet-5
        adaptive thinking to force a tool call.
        """
        m = AnthropicModel("claude-sonnet-4-5", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs(tools=[_ANTHROPIC_FINISH_TOOL])
        assert kwargs.get("thinking", {}).get("type") == "enabled"
        assert "tool_choice" not in kwargs


class TestNonThinkingModelsUnaffected:
    """Older (Claude 3.x) models must not gain the thinking param."""

    def test_claude_3_5_sonnet_has_no_thinking(self) -> None:
        m = AnthropicModel("claude-3-5-sonnet-20241022", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert "thinking" not in kwargs, kwargs.get("thinking")
        beta = kwargs.get("extra_headers", {}).get("anthropic-beta", "")
        assert "interleaved-thinking" not in beta, beta

    def test_supports_extended_thinking_returns_false_for_claude_3(self) -> None:
        assert _supports_extended_thinking("claude-3-5-sonnet-20241022") is False
        assert _uses_adaptive_thinking("claude-3-5-sonnet-20241022") is False


class TestExtendedThinkingFlagCanForceOff:
    """A ``False`` flag in MODEL_INFO must let a JSON edit opt out of thinking.

    This exercises the ``None`` vs ``False`` distinction on the
    tri-state override — without it, a JSON ``"extended_thinking":
    false`` would be indistinguishable from "unset" and the legacy
    prefix path would still enable thinking.
    """

    def test_forced_off_flag_disables_thinking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulate a MODEL_INFO override that opts a prefix-matched model out."""
        info = MODEL_INFO["claude-opus-4-8"]
        monkeypatch.setattr(info, "extended_thinking", False)
        # ``adaptive_thinking`` is separately gated; keep it as-declared
        # so this test only measures the ``extended_thinking`` opt-out.
        assert _supports_extended_thinking("claude-opus-4-8") is False
        m = AnthropicModel("claude-opus-4-8", api_key="test-key")
        m.conversation = [{"role": "user", "content": "ping"}]
        kwargs = m._build_create_kwargs()
        assert "thinking" not in kwargs, kwargs.get("thinking")

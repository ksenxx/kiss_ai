# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-vendor regression audit for reasoning-effort alias generation.

Companion tests to the vendor-specific suites
(:mod:`kiss.tests.scripts.test_update_models_thinking_levels` for the
OpenAI ladder, :mod:`kiss.tests.scripts.test_update_models_moonshot_thinking`
for Kimi K3).  Those suites cover the *positive* path for two of the
five distinct scale shapes that ``update_models.py`` supports today —
OpenAI 4-level (``low``/``medium``/``high``/``xhigh``), Moonshot 3-level
(``low``/``high``/``max``), Grok effort 3-level
(``low``/``medium``/``high``), Grok-3-mini 2-level (``low``/``high``),
and GLM-5.2 2-level (``high``/``max``).  This file locks in the
boundary — every model family that *does not* have an effort ladder
today must stay behind :func:`kiss.scripts.update_models.detect_thinking_level`'s
gate, with zero network activity, and every family the gate lets
through must dispatch to a scale (via
:func:`kiss.scripts.update_models._thinking_scale_for`) whose top rung
matches vendor documentation — plus the positive path for the three
newer families (Grok effort, GLM-5.2, gpt-oss): catalog-shape
lock-ins, descending-probe fallback behavior, and wire-shape tests
that verify ``reasoning_effort`` reaches the outgoing request payload.

The four gaps identified during the November 2026 audit
(``reports/reasoning_effort_alias_audit.html``) — A: xAI Grok effort
family, B: z-ai GLM-5.2, C: Together-route ``openai/gpt-oss-*``, D:
OpenRouter-route gpt-oss alias materialization — were all closed by
commit ``854402ab``.  The tests in :class:`TestAuditGapTripwires`
started life as strict expected-failure tripwires; when the fix landed
each unexpectedly passed and was promoted to a plain positive
assertion, retained here as regressions.

The negative gate assertions monkey-patch
``kiss.core.models.model_info.model`` to a sentinel that records every
invocation.  If the gate leaks (i.e. ``detect_thinking_level`` sends a
gated-out model to the probe loop), the sentinel captures the call and
the test fails with a diagnostic listing the model name and captured
attempt — no real HTTP is issued in any of these tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from kiss.tests.core.test_update_models_thinking_audit import ALL_LEVELS, MOONSHOT_LEVELS

# Levels expected on the five vendor scale shapes supported by
# ``update_models.py`` (OpenAI, Moonshot, Grok effort, Grok-3-mini,
# GLM-5.2).  Kept in sync with the module constants so a refactor that
# renames or reorders them is caught here immediately.
OPENAI_LEVELS = ("low", "medium", "high", "xhigh")
GROK_3_MINI_LEVELS = ("low", "high")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingModelStub:
    """Stand-in for ``kiss.core.models.model_info.model``.

    Records every ``(model_name, model_config)`` invocation into
    :attr:`calls` and then raises ``RuntimeError`` so the probe loop's
    ``except Exception`` swallows the call.  The recording lets a test
    assert whether the gate in ``detect_thinking_level`` reached the
    probe at all — an empty ``calls`` list means the gate held; a
    non-empty list means the gate leaked.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, model_name: str, **kwargs: Any) -> Any:
        self.calls.append({"model_name": model_name, "kwargs": kwargs})
        raise RuntimeError("gate leaked; stub refuses to make network calls")


@pytest.fixture
def gate_probe(monkeypatch: pytest.MonkeyPatch) -> _RecordingModelStub:
    """Patch model factory so ``detect_thinking_level`` cannot touch the wire.

    Any test using this fixture must assert on ``gate_probe.calls`` to
    document its expectation about whether the gate should have admitted
    the model or short-circuited.
    """
    stub = _RecordingModelStub()
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    return stub




# ---------------------------------------------------------------------------
# Scale dispatch — the source of truth for which vendor-specific ladder is
# used when materializing aliases.  Every catalog key spelling of every
# gated-in family must map to the right ladder.
# ---------------------------------------------------------------------------


class TestThinkingScaleDispatch:
    """``_thinking_scale_for`` must return the correct ladder per family."""

    def test_openai_family_uses_openai_ladder(self) -> None:
        """Every OpenAI catalog key (direct + OpenRouter) uses OPENAI_LEVELS."""
        import kiss.scripts.update_models as mod

        for name in (
            "gpt-5.5",
            "gpt-5.6-sol",
            "gpt-5.6-luna",
            "gpt-5.6-terra",
            "o3-mini",
            "o4-mini",
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/gpt-5.6-sol",
            "openrouter/~openai/gpt-latest",
            "openrouter/openai/gpt-oss-120b",
            "openrouter/openai/gpt-oss-20b",
            "openrouter/openai/gpt-oss-safeguard-20b",
        ):
            assert mod._thinking_scale_for(name) == OPENAI_LEVELS, name

    def test_moonshot_family_uses_moonshot_ladder(self) -> None:
        """Every Moonshot spelling (K2, K3, v1, Together, OpenRouter, ~mirror).

        The scale is chosen from the prefix alone (K3-only probing is
        enforced separately in ``detect_thinking_level``); the K2 and v1
        entries here document that the scale dispatch is family-wide
        even though probing is K3-only.
        """
        import kiss.scripts.update_models as mod

        for name in (
            "kimi-k3",
            "kimi-k2-0905-preview",
            "kimi-k2-thinking",
            "moonshot-v1-128k",
            "moonshotai/Kimi-K3",
            "moonshotai/Kimi-K2-Thinking",
            "openrouter/moonshotai/kimi-k3",
            "openrouter/moonshotai/kimi-k2-thinking",
            "openrouter/~moonshotai/kimi-k3",
        ):
            assert mod._thinking_scale_for(name) == MOONSHOT_LEVELS, name

    def test_ungated_families_fall_back_to_openai_ladder(self) -> None:
        """Non-Moonshot / non-Grok-effort / non-GLM-5.2 keys use the OpenAI ladder.

        The scale is only *consulted* when the probe gate admits the
        model.  For ungated families this scale is inert — but pinning
        the fallback here guards against a stray refactor that swaps
        the default to Moonshot's shorter ladder and silently emits
        wrong ``-max`` aliases everywhere.

        NB: ``bare`` ``glm-5.2`` (no ``zai-org/`` or ``openrouter/z-ai/``
        prefix) intentionally falls back to the OpenAI ladder because
        :func:`_is_glm_5_2_family` requires the vendor route prefix — a
        rogue custom entry named ``glm-5.2`` under a different provider
        would not automatically inherit z-ai's 2-level ladder.
        """
        import kiss.scripts.update_models as mod

        for name in (
            "claude-opus-4-7",
            "gemini-3.6-flash",
            "glm-4.6",
            "glm-5.2",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "deepseek-ai/DeepSeek-R1",
            "openrouter/deepseek/deepseek-r1",
            "openrouter/x-ai/grok-4-fast",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            assert mod._thinking_scale_for(name) == OPENAI_LEVELS, name

    def test_grok_effort_family_uses_grok_ladder(self) -> None:
        """xAI Grok effort family uses the 3-level ``('low','medium','high')`` ladder."""
        import kiss.scripts.update_models as mod

        for name in ("openrouter/x-ai/grok-4.5", "openrouter/x-ai/grok-4.3"):
            assert mod._thinking_scale_for(name) == ("low", "medium", "high"), name

    def test_grok_3_mini_uses_two_level_ladder(self) -> None:
        """xAI ``grok-3-mini`` / ``-beta`` use the 2-level ``('low','high')`` ladder.

        ``medium`` is rejected by xAI for the mini family; emitting a
        ``-medium`` alias for a ``grok-3-mini`` would fabricate an API-
        rejected level. This test locks in the correct 2-level ladder.
        """
        import kiss.scripts.update_models as mod

        for name in (
            "openrouter/x-ai/grok-3-mini",
            "openrouter/x-ai/grok-3-mini-beta",
        ):
            assert mod._thinking_scale_for(name) == ("low", "high"), name

    def test_glm_5_2_family_uses_two_level_ladder(self) -> None:
        """z-ai ``GLM-5.2`` uses the 2-level ``('high','max')`` ladder."""
        import kiss.scripts.update_models as mod

        for name in ("zai-org/GLM-5.2", "openrouter/z-ai/glm-5.2"):
            assert mod._thinking_scale_for(name) == ("high", "max"), name

    def test_all_levels_covers_every_scale(self) -> None:
        """``_ALL_THINKING_LEVELS`` must be the union of every vendor scale."""
        import kiss.scripts.update_models as mod

        assert set(mod._ALL_THINKING_LEVELS) >= set(mod._THINKING_LEVELS)
        assert set(mod._ALL_THINKING_LEVELS) >= set(mod._MOONSHOT_THINKING_LEVELS)
        assert set(ALL_LEVELS) == set(mod._ALL_THINKING_LEVELS)


# ---------------------------------------------------------------------------
# Gate safety — negative assertions.  Each family below is documented in
# ``reports/reasoning_effort_alias_audit.html`` §3 as *correctly excluded*
# from ``detect_thinking_level``'s probe.  These tests are the tripwire
# that catches an over-broad gate expansion — e.g. someone adding
# "moonshotai/" wholesale without keeping the K3-only guard, or adding
# "z-ai/" without a per-model narrower gate.
# ---------------------------------------------------------------------------


class TestDetectThinkingLevelGateHolds:
    """``detect_thinking_level`` returns ``None`` with zero HTTP for these."""

    @pytest.mark.parametrize(
        "name",
        [
            "claude-opus-4-7",
            "claude-sonnet-4-7",
            "claude-haiku-4-7",
            "openrouter/anthropic/claude-opus-4",
            "openrouter/~anthropic/claude-latest",
            "cc/opus",
            "cc/sonnet",
            "cc/haiku",
        ],
    )
    def test_anthropic_and_claude_code_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Anthropic uses ``thinking.budget_tokens`` — never probe with effort."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "gemini-3.6-flash",
            "gemini-3.6-pro",
            "gemini-3-5-flash",
            "openrouter/google/gemini-3.6-flash",
            "openrouter/~google/gemini-latest",
        ],
    )
    def test_gemini_and_google_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Gemini uses ``thinking_budget`` — never probe with effort."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "codex/gpt-5.5",
            "codex/gpt-5.6-sol",
            "codex/default",
            "codex/codex-auto-review",
        ],
    )
    def test_codex_cli_routes_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Codex CLI routes control reasoning via ``model_reasoning_effort``,
        not per-call — must not be probed."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "kimi-k2-0905-preview",
            "kimi-k2-thinking",
            "kimi-k2.5",
            "kimi-k2.6",
            "kimi-k2.7-code",
            "moonshot-v1-8k",
            "moonshot-v1-32k",
            "moonshot-v1-128k",
            "moonshotai/Kimi-K2-Instruct",
            "moonshotai/Kimi-K2-Instruct-0905",
            "moonshotai/Kimi-K2-Thinking",
            "moonshotai/Kimi-K2.5",
            "moonshotai/Kimi-K2.6",
            "moonshotai/Kimi-K2.7-Code",
            "openrouter/moonshotai/kimi-k2-thinking",
            "openrouter/moonshotai/kimi-k2-instruct-0905",
        ],
    )
    def test_moonshot_non_k3_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Kimi K2.x uses ``thinking.type``, v1 has no thinking.

        Both share the ``kimi-``/``moonshotai/`` prefix with K3 but must
        stay behind the K3-only guard ``_is_kimi_k3_family`` — probing
        them through a gateway that silently drops ``reasoning_effort``
        would fabricate a phantom level in the catalog.
        """
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-R1-0528",
            "deepseek-ai/DeepSeek-R1-0528-tput",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-V3-0324",
            "deepseek-ai/DeepSeek-V3.1",
            "deepseek-ai/DeepSeek-V4-Pro",
            "openrouter/deepseek/deepseek-r1",
            "openrouter/deepseek/deepseek-v3.1",
        ],
    )
    def test_deepseek_r1_and_v3_family_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """DeepSeek R1 emits chain-of-thought inside ``<think>`` blocks in the
        completion; there is no ``reasoning_effort`` selector.  ``-pro``
        entries are also caught by the substring filter."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            "Qwen/Qwen3-235B-A22B-Thinking-2507",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "Qwen/Qwen3.5-397B-A17B",
            "Qwen/Qwen3.5-9B",
            "Qwen/Qwen3.6-Plus",
            "Qwen/Qwen3.7-Max",
            "Qwen/Qwen3.7-Plus",
            "Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507",
            "openrouter/qwen/qwen3-next-80b-a3b-thinking",
            "openrouter/qwen/qwen3.7-max",
            "openrouter/qwen/qwen3.7-plus",
            "openrouter/qwen/qwen3-max-thinking",
            "openrouter/qwen/qwen3-vl-8b-thinking",
        ],
    )
    def test_qwen_thinking_family_is_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Alibaba Qwen uses ``enable_thinking`` + ``thinking_budget``, never
        an effort enum.  OpenRouter's Qwen models do not advertise
        ``supported_efforts`` in ``/api/v1/models``.  The ``-pro`` and
        ``-max`` substring filters also help exclude some of these but
        even the entries free of those markers must stay gated out."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None, name
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            # Boolean-only Grok models — must never be probed for effort.
            "openrouter/x-ai/grok-4",
            "openrouter/x-ai/grok-4-fast",
            "openrouter/x-ai/grok-4-fast-reasoning",
            "openrouter/x-ai/grok-4.20",
            "openrouter/x-ai/grok-4.20-multi-agent",
            "openrouter/x-ai/grok-4.20-beta",
            "openrouter/x-ai/grok-4.1-fast",
            "openrouter/x-ai/grok-3",
            "openrouter/x-ai/grok-3-beta",
            "openrouter/x-ai/grok-code-fast-1",
            "openrouter/x-ai/grok-build-0.1",
            "openrouter/x-ai/grok-2",
            "openrouter/x-ai/grok-2-mini",
            "openrouter/x-ai/grok-2-vision-1212",
            "openrouter/x-ai/grok-beta",
            "openrouter/x-ai/grok-vision-beta",
            "openrouter/~x-ai/grok-latest",
        ],
    )
    def test_grok_non_effort_models_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Grok models with only ``reasoning.enabled`` boolean, or no reasoning
        control at all, must never be probed with ``reasoning_effort``.

        This test locks in that even with Grok effort family gating in
        place (Gap A landed: ``grok-4.5``, ``grok-4.3``, ``grok-3-mini``),
        the boolean-only siblings stay out — this parametrization must
        NOT be extended to include the effort-family models.
        """
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            # Every GLM other than 5.2: thinking.type boolean only, no ladder.
            "glm-4-32b-0414-128k",
            "glm-4.5",
            "glm-4.5-air",
            "glm-4.5-airx",
            "glm-4.5-flash",
            "glm-4.5-x",
            "glm-4.6",
            "glm-4.7",
            "zai-org/GLM-4.5-Air-FP8",
            "zai-org/GLM-4.6",
            "zai-org/GLM-4.7",
            "zai-org/GLM-5",
            "zai-org/GLM-5.1",
            "openrouter/z-ai/glm-4.5",
            "openrouter/z-ai/glm-4.5-air",
            "openrouter/z-ai/glm-4.5v",
            "openrouter/z-ai/glm-4.6",
            "openrouter/z-ai/glm-4.6v",
            "openrouter/z-ai/glm-4.7",
            "openrouter/z-ai/glm-4.7-flash",
            "openrouter/z-ai/glm-5",
            "openrouter/z-ai/glm-5-turbo",
            "openrouter/z-ai/glm-5.1",
            "openrouter/z-ai/glm-5v-turbo",
        ],
    )
    def test_non_5_2_glm_family_is_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """Every GLM except 5.2 uses only ``thinking.type: enabled/disabled``.

        Must stay gated out now that Gap B has landed (only 5.2 got
        effort-family gating).
        """
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None, name
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            # Together's own Llama/Mistral/Meta/Nvidia families — no effort surface.
            "meta-llama/Llama-3-70b-chat-hf",
            "meta-llama/Llama-3.1-405B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "mistralai/Ministral-3-14B-Instruct-2512",
            "mistralai/Mistral-Small-24B-Instruct-2501",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            "nvidia/NVIDIA-Nemotron-Nano-9B-v2",
            "nvidia/nemotron-3-ultra-550b-a55b",
            "google/gemma-2-27b-it",
            "google/gemma-4-31B-it",
            "openrouter/perplexity/sonar",
            "openrouter/perplexity/sonar-deep-research",
            "openrouter/cohere/command-r-plus",
            "openrouter/amazon/nova-pro",
            "openrouter/bytedance-seed/seed-1-6",
            "openrouter/microsoft/phi-4",
            "openrouter/meta/llama-4",
            "openrouter/inception/mercury",
        ],
    )
    def test_generic_non_reasoning_families_are_gated_out(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """The long-tail of Together/OpenRouter-hosted non-reasoning models."""
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None, name
        assert gate_probe.calls == [], f"Gate leaked for {name}"

    @pytest.mark.parametrize(
        "name",
        [
            # Substring filter: -pro, chat-latest, -image
            "gpt-5.5-chat-latest",
            "gpt-image-2",
            "openrouter/openai/gpt-image-2",
            "gemini-3-pro",
            "openrouter/anthropic/claude-3-pro",
        ],
    )
    def test_substring_filters_stay_effective(
        self, gate_probe: _RecordingModelStub, name: str
    ) -> None:
        """The ``-pro``, ``chat-latest``, ``-image`` marker filter must stay
        active — those variants are documented to reject ``reasoning_effort``.
        """
        import kiss.scripts.update_models as mod

        assert mod.detect_thinking_level(name) is None, name
        assert gate_probe.calls == [], f"Gate leaked for {name}"


# ---------------------------------------------------------------------------
# Catalog-shape lock-ins — freeze the *current* correct on-disk state so a
# regression in ``update_models.py`` or a bad JSON hand-edit doesn't
# silently lose aliases.
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Audit-gap lock-ins.  Each of the four gaps (A/B/C/D) identified in
# ``reports/reasoning_effort_alias_audit.html`` §2 was fixed in a follow-up
# to the audit and is now pinned by a positive assertion.  These tests
# started life as strict expected-failure tripwires (see git history at
# the audit-landing commit) — the moment the gap fix landed each
# tripwire started passing, which under strict mode forced the fixer
# to promote the assertions to plain positives.  Keeping the same test
# names lets ``git log`` follow the transition.
# ---------------------------------------------------------------------------


class TestAuditGapTripwires:
    """Positive lock-ins for the four audit gaps, now fixed.

    Each docstring still records the gap letter (A/B/C/D) from the audit
    report so a bisector can jump straight to the corresponding
    recommended-fix section.
    """

    def test_gap_a_grok_effort_family_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-fix: ``detect_thinking_level`` for grok-4.5 / grok-4.3
        must reach the probe (recorded in ``gate_probe.calls``)."""
        import kiss.scripts.update_models as mod

        # Also stub out the probe attachments helper so it doesn't try to
        # go read any image file — the gate is the only thing under test.
        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        mod.detect_thinking_level("openrouter/x-ai/grok-4.5")
        mod.detect_thinking_level("openrouter/x-ai/grok-4.3")
        # Both should have been sent to the probe loop.
        recorded = [c["model_name"] for c in gate_probe.calls]
        assert "openrouter/x-ai/grok-4.5" in recorded
        assert "openrouter/x-ai/grok-4.3" in recorded

    def test_gap_a_grok_3_mini_uses_two_level_scale(self) -> None:
        """Post-fix: ``_thinking_scale_for("openrouter/x-ai/grok-3-mini")``
        must return the 2-level ladder ``("low", "high")`` — not the
        default OpenAI ladder — because grok-3-mini rejects ``medium``.
        """
        import kiss.scripts.update_models as mod

        scale = mod._thinking_scale_for("openrouter/x-ai/grok-3-mini")
        assert scale == ("low", "high"), (
            f"grok-3-mini must use ('low','high'); got {scale!r}"
        )
        assert scale == mod._thinking_scale_for(
            "openrouter/x-ai/grok-3-mini-beta"
        ), "grok-3-mini-beta must share the same 2-level scale as grok-3-mini"

    def test_gap_a_grok_4_5_uses_three_level_scale(self) -> None:
        """Post-fix: grok-4.5 / grok-4.3 use ``('low','medium','high')``
        — never ``xhigh`` (xAI's ladder tops at ``high``), never ``max``."""
        import kiss.scripts.update_models as mod

        for name in (
            "openrouter/x-ai/grok-4.5",
            "openrouter/x-ai/grok-4.3",
        ):
            scale = mod._thinking_scale_for(name)
            assert scale == ("low", "medium", "high"), (
                f"{name!r} must use ('low','medium','high'); got {scale!r}"
            )

    def test_gap_b_glm_5_2_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-fix: ``detect_thinking_level`` for GLM-5.2 must reach the probe."""
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        mod.detect_thinking_level("zai-org/GLM-5.2")
        mod.detect_thinking_level("openrouter/z-ai/glm-5.2")
        recorded = [c["model_name"] for c in gate_probe.calls]
        assert "zai-org/GLM-5.2" in recorded
        assert "openrouter/z-ai/glm-5.2" in recorded

    def test_gap_b_glm_5_2_uses_two_level_scale(self) -> None:
        """Post-fix: ``_thinking_scale_for`` on GLM-5.2 keys returns the
        2-level ``('high', 'max')`` scale — never the OpenAI ladder."""
        import kiss.scripts.update_models as mod

        for name in ("zai-org/GLM-5.2", "openrouter/z-ai/glm-5.2"):
            scale = mod._thinking_scale_for(name)
            assert scale == ("high", "max"), (
                f"{name!r} must use ('high', 'max'); got {scale!r}"
            )

    def test_gap_b_only_5_2_admitted_other_glms_still_gated_out(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-fix: GLM-5.2 admitted, every other GLM still gated out."""
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        # First: confirm 5.2 gets in.
        mod.detect_thinking_level("zai-org/GLM-5.2")
        recorded_before = [c["model_name"] for c in gate_probe.calls]
        assert "zai-org/GLM-5.2" in recorded_before

        # Then: confirm every other GLM still gated out.
        gate_probe.calls.clear()
        for other in (
            "zai-org/GLM-4.6",
            "zai-org/GLM-4.7",
            "zai-org/GLM-5",
            "zai-org/GLM-5.1",
            "openrouter/z-ai/glm-4.5",
            "openrouter/z-ai/glm-4.6",
            "openrouter/z-ai/glm-4.7",
            "openrouter/z-ai/glm-5",
            "openrouter/z-ai/glm-5.1",
            "openrouter/z-ai/glm-5v-turbo",
        ):
            assert mod.detect_thinking_level(other) is None, other
        assert gate_probe.calls == [], (
            "Non-5.2 GLMs must not reach the probe after gap B fix"
        )

    def test_gap_c_together_gpt_oss_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Post-fix: ``detect_thinking_level`` for the Together route
        of gpt-oss (``openai/gpt-oss-*``) must reach the probe."""
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        for name in (
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            mod.detect_thinking_level(name)
        recorded = [c["model_name"] for c in gate_probe.calls]
        assert "openai/gpt-oss-120b" in recorded
        assert "openai/gpt-oss-20b" in recorded



# ---------------------------------------------------------------------------
# Catalog-shape lock-ins for the newer effort families (Grok 4.5/4.3,
# GLM-5.2, gpt-oss).  Same pattern as the Kimi K3 lock-ins above:
# exact-alias existence, byte-for-byte field parity with the base aside
# from ``thinking`` / ``alias_of``, and no off-scale suffixes.
# ---------------------------------------------------------------------------






# ---------------------------------------------------------------------------
# Probe fallback behavior — successful-probe tests through the real
# ``detect_thinking_level`` loop with a controllable model factory.  The
# stub accepts a configurable set of levels: accepted levels return a
# model whose ``generate()`` succeeds, everything else raises (like a
# vendor HTTP 400).  This locks in both the descending probe order and
# the stop-on-first-success semantics for every new vendor scale.
# ---------------------------------------------------------------------------


class _StubProbeModel:
    """Minimal model object returned by :class:`_ProbeFactoryStub`."""

    def initialize(self, *args: Any, **kwargs: Any) -> None:
        """Accept any prompt/attachments without doing anything."""

    def generate(self) -> tuple[str, Any]:
        """Return a non-empty completion so the probe records a success."""
        return "hello", None


class _ProbeFactoryStub:
    """``create_model`` stand-in whose success depends on the effort level.

    Records the ``reasoning_effort`` of every probe attempt in
    :attr:`attempts` (in call order).  When the level is in
    ``accepted_levels`` the returned model generates successfully;
    otherwise the factory raises, emulating a vendor HTTP 400 for an
    unsupported level (``detect_thinking_level`` treats any exception as
    a rejection and walks down the scale).
    """

    def __init__(self, accepted_levels: tuple[str, ...] = ()) -> None:
        self.accepted_levels = accepted_levels
        self.attempts: list[str] = []

    def __call__(self, model_name: str, **kwargs: Any) -> _StubProbeModel:
        level = str((kwargs.get("model_config") or {}).get("reasoning_effort"))
        self.attempts.append(level)
        if level in self.accepted_levels:
            return _StubProbeModel()
        raise RuntimeError(f"reasoning_effort={level!r} rejected by test stub")


def _probe_with_stub(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    accepted_levels: tuple[str, ...],
) -> tuple[str | None, list[str]]:
    """Run ``detect_thinking_level`` against a controllable probe stub.

    Returns the detected level and the ordered list of levels attempted.
    """
    import kiss.scripts.update_models as mod

    stub = _ProbeFactoryStub(accepted_levels)
    monkeypatch.setattr("kiss.core.models.model_info.model", stub)
    monkeypatch.setattr(mod, "_probe_attachments", lambda name: None)
    return mod.detect_thinking_level(model_name), stub.attempts


class TestProbeFallbackBehavior:
    """``detect_thinking_level`` walks each new vendor scale descending."""

    # --- Grok effort family (3-level: low, medium, high) -----------------

    def test_grok_4_5_accepts_high_stops_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Top level accepted → returns 'high' after a single attempt."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/x-ai/grok-4.5", accepted_levels=("high",)
        )
        assert level == "high"
        assert attempts == ["high"], (
            "The probe must try the top of the Grok scale first and stop "
            f"on success; attempted {attempts}"
        )

    def test_grok_4_5_falls_back_to_medium(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """'high' rejected, 'medium' accepted → returns 'medium'."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/x-ai/grok-4.5", accepted_levels=("medium",)
        )
        assert level == "medium"
        assert attempts == ["high", "medium"], (
            f"The probe must walk high → medium and stop; attempted {attempts}"
        )

    def test_grok_4_5_all_rejected_walks_full_scale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything rejected → None after exactly high, medium, low."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/x-ai/grok-4.5", accepted_levels=()
        )
        assert level is None
        assert attempts == ["high", "medium", "low"], (
            "Every level of the Grok scale (and only those) must be tried "
            f"in descending order; attempted {attempts}"
        )

    # --- grok-3-mini (2-level: low, high — no medium!) --------------------

    def test_grok_3_mini_accepts_high_never_tries_medium(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """grok-3-mini top level accepted → 'high' with a single attempt."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/x-ai/grok-3-mini", accepted_levels=("high",)
        )
        assert level == "high"
        assert attempts == ["high"]

    def test_grok_3_mini_all_rejected_skips_medium(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 2-level mini scale must be honored: high, low — NO medium."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/x-ai/grok-3-mini-beta", accepted_levels=()
        )
        assert level is None
        assert attempts == ["high", "low"], (
            "grok-3-mini must never be probed with 'medium' (the API "
            f"rejects it); attempted {attempts}"
        )
        assert "medium" not in attempts

    # --- GLM-5.2 (2-level: high, max) --------------------------------------

    @pytest.mark.parametrize("name", ["zai-org/GLM-5.2", "openrouter/z-ai/glm-5.2"])
    def test_glm_5_2_accepts_max_stops_immediately(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        """Top level 'max' accepted → returns 'max' after one attempt."""
        level, attempts = _probe_with_stub(monkeypatch, name, accepted_levels=("max",))
        assert level == "max"
        assert attempts == ["max"], (
            f"The probe must try GLM-5.2's top level 'max' first; attempted {attempts}"
        )

    def test_glm_5_2_falls_back_to_high(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'max' rejected, 'high' accepted → returns 'high'."""
        level, attempts = _probe_with_stub(
            monkeypatch, "zai-org/GLM-5.2", accepted_levels=("high",)
        )
        assert level == "high"
        assert attempts == ["max", "high"]

    def test_glm_5_2_all_rejected_never_tries_low_or_medium(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Everything rejected → None after exactly max, high (no low/medium)."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openrouter/z-ai/glm-5.2", accepted_levels=()
        )
        assert level is None
        assert attempts == ["max", "high"], (
            "GLM-5.2's scale is exactly (high, max); 'low' and 'medium' "
            f"must never be probed; attempted {attempts}"
        )

    # --- gpt-oss (OpenAI ladder; vendor rejects xhigh, tops at high) -------

    def test_together_gpt_oss_rejects_xhigh_falls_back_to_high(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The audit's expectation: xhigh fails, high succeeds → 'high'."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openai/gpt-oss-120b", accepted_levels=("low", "medium", "high")
        )
        assert level == "high"
        assert attempts == ["xhigh", "high"], (
            "gpt-oss probes the OpenAI ladder; xhigh must fail and high "
            f"succeed on the next rung; attempted {attempts}"
        )

    def test_together_gpt_oss_accepts_only_low(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only 'low' accepted → full descending walk down the OpenAI ladder."""
        level, attempts = _probe_with_stub(
            monkeypatch, "openai/gpt-oss-120b", accepted_levels=("low",)
        )
        assert level == "low"
        assert attempts == ["xhigh", "high", "medium", "low"]

    def test_openrouter_gpt_oss_rejects_xhigh_falls_back_to_high(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Gap D route: OpenRouter gpt-oss walks the same ladder."""
        level, attempts = _probe_with_stub(
            monkeypatch,
            "openrouter/openai/gpt-oss-20b",
            accepted_levels=("low", "medium", "high"),
        )
        assert level == "high"
        assert attempts == ["xhigh", "high"]


# ---------------------------------------------------------------------------
# Wire-shape tests — one per vendor route.  A real ``generate()`` through
# each new family's catalog alias must carry the base model id and the
# alias's ``reasoning_effort`` in the outgoing request payload.  Mirrors
# the in-process endpoint-emulator pattern of
# ``test_update_models_moonshot_thinking.py``: only provider ``base_url``
# constants and API keys are redirected; the code under test (catalog
# alias resolution, ``OpenAICompatibleModel`` request building, HTTP
# transport) runs unmodified.
# ---------------------------------------------------------------------------

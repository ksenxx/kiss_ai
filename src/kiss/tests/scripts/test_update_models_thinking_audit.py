# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Cross-vendor regression audit for reasoning-effort alias generation.

Companion tests to the vendor-specific suites
(:mod:`kiss.tests.scripts.test_update_models_thinking_levels` for the
OpenAI ladder, :mod:`kiss.tests.scripts.test_update_models_moonshot_thinking`
for Kimi K3).  Those suites cover the *positive* path for the two vendor
scales that ``update_models.py`` currently supports; this file locks in
the boundary — every model family that *does not* have an effort ladder
today must stay behind :func:`kiss.scripts.update_models.detect_thinking_level`'s
gate, with zero network activity, and every family the gate lets
through must dispatch to a scale (via
:func:`kiss.scripts.update_models._thinking_scale_for`) whose top rung
matches vendor documentation.

Three additional gaps identified during the November 2026 audit
(``reports/reasoning_effort_alias_audit.html``) are pinned with
:pyfunc:`pytest.mark.xfail(strict=True)`.  The moment any of the three
gaps is fixed the corresponding xfail will XPASS and force the fixer to
turn the test into a strict positive assertion — an intentional
tripwire that prevents landing a fix without also landing a lock-in
test.

The negative gate assertions monkey-patch
``kiss.core.models.model_info.model`` to a sentinel that records every
invocation.  If the gate leaks (i.e. ``detect_thinking_level`` sends a
gated-out model to the probe loop), the sentinel captures the call and
the test fails with a diagnostic listing the model name and captured
attempt — no real HTTP is issued in any of these tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# Levels expected on the two vendor scales currently supported by
# ``update_models.py``.  Kept in sync with the module constants so a
# refactor that renames or reorders them is caught here immediately.
OPENAI_LEVELS = ("low", "medium", "high", "xhigh")
MOONSHOT_LEVELS = ("low", "high", "max")
ALL_LEVELS = ("low", "medium", "high", "xhigh", "max")


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


def _catalog() -> dict[str, dict]:
    """Return the on-disk ``MODEL_INFO.json`` as a plain dict.

    Used by the positive lock-in tests to guarantee the currently
    correct catalog shape (K3 + gpt-5.x aliases present, no aliases for
    unsupported vendors) does not silently regress.
    """
    from kiss.core.models import model_info as mi

    return json.loads(  # type: ignore[no-any-return]
        Path(mi.__file__).parent.joinpath("MODEL_INFO.json").read_text()
    )


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
        """Non-Moonshot, non-Grok/GLM/gpt-oss keys use the OpenAI ladder.

        The scale is only *consulted* when the probe gate admits the
        model.  For ungated families this scale is inert — but pinning
        the fallback here guards against a stray refactor that swaps
        the default to Moonshot's shorter ladder and silently emits
        wrong ``-max`` aliases everywhere.
        """
        import kiss.scripts.update_models as mod

        for name in (
            "claude-opus-4-7",
            "gemini-3.6-flash",
            "glm-4.6",
            "glm-5.2",
            "zai-org/GLM-5.2",
            "openrouter/z-ai/glm-5.2",
            "openrouter/qwen/qwen3-235b-a22b-thinking-2507",
            "Qwen/Qwen3-Next-80B-A3B-Thinking",
            "deepseek-ai/DeepSeek-R1",
            "openrouter/deepseek/deepseek-r1",
            "openrouter/x-ai/grok-4.5",
            "openrouter/x-ai/grok-3-mini",
            "openrouter/x-ai/grok-4-fast",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
        ):
            assert mod._thinking_scale_for(name) == OPENAI_LEVELS, name

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

        This test locks in that even after Grok effort family gating is
        added, the boolean-only siblings stay out.  When Gap A from the
        audit lands (``grok-4.5``, ``grok-4.3``, ``grok-3-mini``), this
        parametrization must NOT be extended to include them.
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

        Must stay gated out even after Gap B lands (only 5.2 gets
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


class TestCatalogAliasesFrozen:
    """Currently-shipping alias sets must not silently disappear."""

    def test_kimi_k3_family_has_low_high_max_aliases(self) -> None:
        """The K3 fix guarantees three routes × three-level ladder = nine aliases."""
        data = _catalog()
        bases = ("kimi-k3", "moonshotai/Kimi-K3", "openrouter/moonshotai/kimi-k3")
        for base in bases:
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", (
                f"Base {base} must be capped at high; got {data[base].get('thinking')!r}"
            )
            assert "alias_of" not in data[base]
            for level in MOONSHOT_LEVELS:
                alias = f"{base}-{level}"
                assert alias in data, f"Missing alias {alias}"
                assert data[alias].get("thinking") == level, (
                    f"{alias} must have thinking={level!r}"
                )
                assert data[alias].get("alias_of") == base, (
                    f"{alias} must carry alias_of={base!r}"
                )
            # Guard: never emit off-scale aliases for Moonshot.
            for off_scale in ("medium", "xhigh"):
                assert f"{base}-{off_scale}" not in data, (
                    f"{base}-{off_scale} must not exist on the Moonshot scale"
                )

    def test_gpt_5_family_has_openai_ladder_aliases(self) -> None:
        """The historical gpt-5.5+ xhigh split still produces the full ladder."""
        data = _catalog()
        # A representative subset covering direct + OpenRouter routes.
        for base in (
            "gpt-5.5",
            "gpt-5.6-sol",
            "openrouter/openai/gpt-5.5",
            "openrouter/openai/gpt-5.6-sol",
            "openrouter/~openai/gpt-latest",
        ):
            assert base in data, f"Base {base} missing"
            assert data[base].get("thinking") == "high", base
            for level in ("low", "medium", "high", "xhigh"):
                alias = f"{base}-{level}"
                assert alias in data, f"Missing OpenAI-ladder alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base

    def test_no_stray_max_aliases_on_openai_models(self) -> None:
        """The Moonshot-only ``-max`` suffix must never appear on OpenAI keys."""
        data = _catalog()
        for name in data:
            if not name.endswith("-max"):
                continue
            base = name.removesuffix("-max")
            # An OpenAI base would never carry a -max alias because the
            # OpenAI ladder tops out at xhigh.
            if base in data and base.startswith(("gpt-", "o1", "o3", "o4")):
                pytest.fail(
                    f"OpenAI-family base {base!r} has a -max alias; "
                    "Moonshot scale mistakenly applied?"
                )

    def test_no_stray_xhigh_aliases_on_moonshot_models(self) -> None:
        """The OpenAI-only ``-xhigh`` suffix must never appear on Moonshot keys."""
        data = _catalog()
        for name in data:
            if not name.endswith("-xhigh"):
                continue
            base = name.removesuffix("-xhigh")
            if base in data:
                if base.startswith("kimi-") or base.startswith("moonshotai/") or (
                    base.startswith("openrouter/moonshotai/")
                ):
                    pytest.fail(
                        f"Moonshot base {base!r} has an -xhigh alias; "
                        "OpenAI scale mistakenly applied?"
                    )

    def test_no_medium_aliases_on_moonshot_or_grok_3_mini(self) -> None:
        """Moonshot scale has no ``medium``.  grok-3-mini (once fixed) also
        has no ``medium`` — but is not in the catalog yet so this test is
        the guard against a mis-fix that reintroduces it."""
        data = _catalog()
        for name in data:
            if not name.endswith("-medium"):
                continue
            base = name.removesuffix("-medium")
            if base.startswith("kimi-") or base.startswith("moonshotai/") or (
                base.startswith("openrouter/moonshotai/")
            ):
                pytest.fail(
                    f"Moonshot base {base!r} has a bogus -medium alias"
                )
            base_last = base.rsplit("/", 1)[-1].lower()
            if base_last.startswith("grok-3-mini"):
                pytest.fail(
                    f"grok-3-mini base {base!r} has a bogus -medium alias; "
                    "the model only accepts low/high"
                )

    def test_generated_aliases_carry_alias_of_marker(self) -> None:
        """Every ``-{level}`` sibling on every scale must carry ``alias_of``.

        The marker is what lets ``update_models.py`` and the runtime
        alias resolver distinguish synthetic aliases from real upstream
        models whose names happen to end in ``-low`` / ``-medium`` /
        ``-high`` / ``-xhigh`` / ``-max``.
        """
        data = _catalog()
        for name, entry in data.items():
            if entry.get("alias_of"):
                assert name.endswith(tuple(f"-{lvl}" for lvl in ALL_LEVELS)), (
                    f"alias_of set but name {name!r} does not end in a level suffix"
                )
                base = entry["alias_of"]
                assert base in data, (
                    f"Dangling alias {name!r} points at missing base {base!r}"
                )
                assert base == name.rsplit("-", 1)[0], (
                    f"alias_of on {name!r} inconsistent with its suffix"
                )


# ---------------------------------------------------------------------------
# Audit-gap tripwires.  Each of the three gaps identified in
# ``reports/reasoning_effort_alias_audit.html`` §2 gets a strict-xfail test
# here.  The test *fails today* (gap not fixed) and is marked
# xfail(strict=True), which turns it into an XFAIL.  The moment the gap
# fix lands, the test starts *passing* and pytest reports it as XPASS,
# which under strict=True is treated as a hard failure — forcing the
# fixer to update this test file into a plain positive assertion.
# ---------------------------------------------------------------------------


class TestAuditGapTripwires:
    """Strict-xfail lock-ins for the three gaps identified in the audit.

    The wording of each xfail reason includes the gap letter (A/B/C) from
    the audit report so the fixer can find the relevant recommended-fix
    outline quickly.
    """

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap A: openrouter/x-ai/grok-4.5 / grok-4.3 accept "
            "reasoning_effort per xAI docs but are not gated in "
            "detect_thinking_level. The fix should also add a 3-level "
            "scale (low, medium, high) for these submodels."
        ),
    )
    def test_gap_a_grok_effort_family_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fixed: ``detect_thinking_level`` for grok-4.5 / grok-4.3
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap A (grok-3-mini submodel): grok-3-mini accepts only "
            "low and high (no medium). When Grok effort family gating "
            "lands, this model needs a dedicated 2-level scale to avoid "
            "the alias-writer emitting a bogus -medium alias."
        ),
    )
    def test_gap_a_grok_3_mini_uses_two_level_scale(self) -> None:
        """When fixed: ``_thinking_scale_for("openrouter/x-ai/grok-3-mini")``
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap A (grok-4.5): once Grok gating lands, "
            "_thinking_scale_for must return the 3-level ladder "
            "('low', 'medium', 'high') for grok-4.5 / grok-4.3."
        ),
    )
    def test_gap_a_grok_4_5_uses_three_level_scale(self) -> None:
        """When fixed: grok-4.5 / grok-4.3 use ``('low','medium','high')``
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap B: zai-org/GLM-5.2 and openrouter/z-ai/glm-5.2 "
            "accept reasoning_effort (native values high, max) but are "
            "not gated in detect_thinking_level. Fix should add a "
            "GLM-5.2-only gate with a 2-level scale ('high', 'max')."
        ),
    )
    def test_gap_b_glm_5_2_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fixed: ``detect_thinking_level`` for GLM-5.2 must reach the probe."""
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        mod.detect_thinking_level("zai-org/GLM-5.2")
        mod.detect_thinking_level("openrouter/z-ai/glm-5.2")
        recorded = [c["model_name"] for c in gate_probe.calls]
        assert "zai-org/GLM-5.2" in recorded
        assert "openrouter/z-ai/glm-5.2" in recorded

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap B: GLM-5.2 uses a 2-level ladder ('high', 'max'). "
            "When the fix lands, _thinking_scale_for must return that "
            "ladder for zai-org/GLM-5.2 and openrouter/z-ai/glm-5.2."
        ),
    )
    def test_gap_b_glm_5_2_uses_two_level_scale(self) -> None:
        """When fixed: ``_thinking_scale_for`` on GLM-5.2 keys returns the
        2-level ``('high', 'max')`` scale — never the OpenAI ladder."""
        import kiss.scripts.update_models as mod

        for name in ("zai-org/GLM-5.2", "openrouter/z-ai/glm-5.2"):
            scale = mod._thinking_scale_for(name)
            assert scale == ("high", "max"), (
                f"{name!r} must use ('high', 'max'); got {scale!r}"
            )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap B (safety): every non-5.2 GLM must stay behind the "
            "gate even after gap B lands. The fix must not accidentally "
            "gate in the whole zai-org / z-ai namespace — only 5.2."
        ),
    )
    def test_gap_b_only_5_2_admitted_other_glms_still_gated_out(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fixed: GLM-5.2 admitted, every other GLM still gated out."""
        import kiss.scripts.update_models as mod

        monkeypatch.setattr(mod, "_probe_attachments", lambda name: {})
        # First: confirm 5.2 gets in (post-fix).
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
            "Non-5.2 GLMs must not reach the probe even after gap B lands"
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap C: openai/gpt-oss-{20,120}b (Together's naming) "
            "accepts reasoning_effort (low/medium/high) but the current "
            "gate matches neither _OPENAI_PREFIXES nor "
            "openrouter/openai/. The fix should add the Together route."
        ),
    )
    def test_gap_c_together_gpt_oss_admitted_to_probe(
        self, gate_probe: _RecordingModelStub, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fixed: ``detect_thinking_level`` for the Together route
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

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Audit gap D (soft): openrouter/openai/gpt-oss-* is already "
            "gated in but MODEL_INFO.json still has thinking=null on all "
            "three catalog rows. A --test-existing rerun should promote "
            "them to thinking='high' and materialize -low/-medium/-high "
            "aliases (no -xhigh — gpt-oss ladder tops at high)."
        ),
    )
    def test_gap_d_openrouter_gpt_oss_has_effort_aliases(self) -> None:
        """When fixed: OpenRouter gpt-oss rows must carry the 3-level ladder."""
        data = _catalog()
        for base in (
            "openrouter/openai/gpt-oss-120b",
            "openrouter/openai/gpt-oss-20b",
            "openrouter/openai/gpt-oss-safeguard-20b",
        ):
            assert data.get(base, {}).get("thinking") == "high", (
                f"{base} must have thinking='high' after the re-probe"
            )
            for level in ("low", "medium", "high"):
                alias = f"{base}-{level}"
                assert alias in data, f"Missing gpt-oss alias {alias}"
                assert data[alias].get("thinking") == level
                assert data[alias].get("alias_of") == base
            # Never emit -xhigh for gpt-oss.
            assert f"{base}-xhigh" not in data, (
                f"{base}-xhigh must not exist; gpt-oss ladder tops at 'high'"
            )

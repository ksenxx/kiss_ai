# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Live-API regression test: ``MODEL_INFO.json`` entries for
``claude-fable-5`` and ``claude-sonnet-5`` must reflect the *real*
Anthropic API capabilities, not placeholder / assumed values.

Motivation
----------
These entries were originally added by an earlier task that assumed
``claude-fable-5`` was a preview / code-named alias.  That assumption
was wrong on two counts:

1. Both ``claude-fable-5`` and ``claude-sonnet-5`` are real, publicly
   documented Anthropic model IDs — they appear verbatim in
   ``GET https://api.anthropic.com/v1/models``.
2. The ``context_length=300000`` originally hardcoded for
   ``claude-fable-5`` did not match Anthropic's reported
   ``max_input_tokens=1_000_000``.  KISS deliberately pins
   ``claude-fable-5`` to ``400_000`` (a KISS-side cap chosen to
   bound worst-case prompt cost while still leaving comfortable
   headroom over the 200K legacy models); ``claude-sonnet-5`` still
   tracks the live API value.

This test locks in the corrected values by querying the *live*
Anthropic ``/v1/models`` endpoint and comparing every capability
declared in ``MODEL_INFO.json`` against the ground truth returned by
the API.

The test is skipped when ``ANTHROPIC_API_KEY`` is unset (CI without
secrets) or when the ``anthropic`` SDK is not installed.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from kiss.core.models.model_info import MODEL_INFO

anthropic = pytest.importorskip("anthropic")


_MODELS_UNDER_TEST = ("claude-fable-5", "claude-sonnet-5")

# KISS deliberately pins ``claude-fable-5.context_length`` below
# Anthropic's advertised ``max_input_tokens=1_000_000`` to bound
# worst-case prompt-cost and truncate before the tail of the window
# where quality is known to degrade.  Models absent from this map
# still track the live API value.
_CONTEXT_LENGTH_OVERRIDES: dict[str, int] = {
    "claude-fable-5": 400_000,
}


@pytest.fixture(scope="module")
def anthropic_models() -> dict[str, Any]:
    """Return {model_id: SDK ModelInfo} from Anthropic's live ``/v1/models``.

    Skips the whole module if no API key is present or if the network
    call fails (offline runner).
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set; live Anthropic API not reachable")
    try:
        client = anthropic.Anthropic()
        page = client.models.list(limit=1000)
    except Exception as e:  # noqa: BLE001 - want to skip on ANY network / auth error
        pytest.skip(f"Anthropic /v1/models not reachable: {e!r}")
    return {m.id: m for m in page.data}


@pytest.mark.live_api
class TestClaudeFableSonnet5MatchLiveAPI:
    """Every ``MODEL_INFO`` field must match Anthropic's live capabilities.

    Running these tests against the live Anthropic API is the only way
    to catch drift between the JSON registry and reality (context
    windows, thinking-mode support, model existence).
    """

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_model_exists_in_live_anthropic_api(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """Model ID declared in MODEL_INFO must be a real Anthropic model.

        If this fails, the entry is a placeholder / fictional model and
        should be removed from ``MODEL_INFO.json`` — routing traffic to
        an ID Anthropic does not serve will 404.
        """
        assert model_id in anthropic_models, (
            f"{model_id!r} is declared in MODEL_INFO.json but "
            f"Anthropic's /v1/models endpoint does not list it. "
            f"Known IDs at this endpoint: "
            f"{sorted(anthropic_models.keys())!r}. Either the ID was a "
            f"placeholder / typo and must be removed, or Anthropic has "
            f"deprecated it."
        )

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_context_length_matches_max_input_tokens(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """``context_length`` must equal Anthropic's ``max_input_tokens``
        unless the model appears in ``_CONTEXT_LENGTH_OVERRIDES``.

        The historical value ``300_000`` for ``claude-fable-5`` was an
        assumption; the API reports ``1_000_000``.  KISS now pins
        ``claude-fable-5`` to ``400_000`` deliberately (see
        ``_CONTEXT_LENGTH_OVERRIDES``); all other models still track
        the live API value so drift in Anthropic's advertised window
        is caught immediately.
        """
        api_model = anthropic_models[model_id]
        actual = MODEL_INFO[model_id].context_length
        if model_id in _CONTEXT_LENGTH_OVERRIDES:
            expected = _CONTEXT_LENGTH_OVERRIDES[model_id]
            assert actual == expected, (
                f"MODEL_INFO[{model_id!r}].context_length={actual} but "
                f"KISS pins this model to {expected} (see "
                f"_CONTEXT_LENGTH_OVERRIDES).  Anthropic /v1/models "
                f"reports max_input_tokens="
                f"{api_model.max_input_tokens}; if the override was "
                f"removed intentionally, drop the entry from "
                f"_CONTEXT_LENGTH_OVERRIDES too."
            )
            return
        expected = api_model.max_input_tokens
        assert actual == expected, (
            f"MODEL_INFO[{model_id!r}].context_length={actual} but "
            f"Anthropic /v1/models reports max_input_tokens={expected}. "
            f"Truncation heuristics will be wrong — update the JSON."
        )

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_extended_thinking_matches_api_thinking_support(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """``extended_thinking`` must equal ``capabilities.thinking.supported``.

        If Anthropic supports thinking on this model, KISS MUST send the
        ``thinking`` request param + ``interleaved-thinking-2025-05-14``
        beta header — otherwise the model may return an all-thinking
        turn that ``KISSAgent`` misreads as "empty response". Conversely
        we must not force ``thinking=True`` for a model that rejects it.
        """
        api_model = anthropic_models[model_id]
        caps = api_model.capabilities
        assert isinstance(caps, dict), (
            f"unexpected capabilities shape for {model_id!r}: "
            f"{type(caps).__name__}"
        )
        thinking = caps.get("thinking") or {}
        api_supports_thinking = bool(thinking.get("supported"))
        info_flag = MODEL_INFO[model_id].extended_thinking
        assert info_flag is api_supports_thinking, (
            f"MODEL_INFO[{model_id!r}].extended_thinking={info_flag!r} "
            f"but Anthropic reports capabilities.thinking.supported="
            f"{api_supports_thinking!r}. If the API supports thinking, "
            f"the JSON flag must be True (or omitted only if the legacy "
            f"prefix gate covers the model)."
        )

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_adaptive_thinking_matches_api_adaptive_support(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """``adaptive_thinking`` must match ``thinking.types.adaptive.supported``.

        Both ``claude-fable-5`` and ``claude-sonnet-5`` are documented
        by Anthropic as adaptive-only (``enabled.supported=False``);
        sending ``thinking={'type': 'enabled', ...}`` to them is
        rejected by the endpoint.  The JSON must therefore declare
        ``adaptive_thinking=True`` so ``_uses_adaptive_thinking``
        returns True and the adapter emits ``thinking={'type':
        'adaptive'}``.
        """
        api_model = anthropic_models[model_id]
        caps = api_model.capabilities
        thinking = caps.get("thinking") or {}
        types = thinking.get("types") or {}
        adaptive_supported = bool((types.get("adaptive") or {}).get("supported"))
        info_flag = MODEL_INFO[model_id].adaptive_thinking
        assert info_flag is adaptive_supported, (
            f"MODEL_INFO[{model_id!r}].adaptive_thinking={info_flag!r} "
            f"but Anthropic reports "
            f"capabilities.thinking.types.adaptive.supported="
            f"{adaptive_supported!r}. Mismatch will cause KISS to send "
            f"a rejected ``thinking.type`` payload."
        )

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_adaptive_only_models_do_not_support_enabled_thinking(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """Guard against Anthropic silently re-adding ``thinking.enabled``.

        Today both ``claude-fable-5`` and ``claude-sonnet-5`` report
        ``types.enabled.supported=False``, which is why the JSON marks
        them adaptive-only.  If Anthropic later flips this, the JSON
        may safely relax ``adaptive_thinking``; until then the adapter
        must never fall back to ``thinking={'type': 'enabled', ...}``
        for these IDs, so this test pins the current state.
        """
        api_model = anthropic_models[model_id]
        caps = api_model.capabilities
        thinking = caps.get("thinking") or {}
        types = thinking.get("types") or {}
        enabled_supported = bool((types.get("enabled") or {}).get("supported"))
        assert enabled_supported is False, (
            f"Anthropic now reports thinking.types.enabled.supported="
            f"{enabled_supported!r} for {model_id!r}. Adaptive-only "
            f"assumption in MODEL_INFO.json may be revisited — but "
            f"first confirm ``_uses_adaptive_thinking`` still returns "
            f"the desired value."
        )

    @pytest.mark.parametrize("model_id", _MODELS_UNDER_TEST)
    def test_function_calling_capability_matches_structured_outputs(
        self,
        model_id: str,
        anthropic_models: dict[str, Any],
    ) -> None:
        """``fc`` in MODEL_INFO must match structured-output support.

        Every model exposed under Anthropic's Messages API accepts
        ``tools=[...]`` (native tool use), but the ``structured_outputs``
        capability flag is the closest published proxy.  If it's False,
        KISS's tool-forcing logic will break.
        """
        api_model = anthropic_models[model_id]
        caps = api_model.capabilities
        structured = (caps.get("structured_outputs") or {}).get("supported")
        info_fc = MODEL_INFO[model_id].is_function_calling_supported
        assert info_fc is True and structured is True, (
            f"MODEL_INFO[{model_id!r}].is_function_calling_supported="
            f"{info_fc!r} vs. capabilities.structured_outputs.supported="
            f"{structured!r}. Both should be True for a tool-calling "
            f"agent model."
        )

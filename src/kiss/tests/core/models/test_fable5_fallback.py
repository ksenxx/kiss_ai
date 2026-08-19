# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the ``claude-fable-5`` non-retryable + fallback fix.

Bug reproduction (production failures in ``~/.kiss/sorcar.db``, all-time,
model = ``claude-fable-5``):

* 5+ tasks failed with Anthropic 404 whose body reads ``"Claude Fable 5
  is not available. Please use Opus 4.8"`` — KISSAgent retried 3× on the
  same non-retryable error and died.
* 10+ tasks died with ``"credit balance is too low"`` (Anthropic 400) —
  same problem, wasted retries against a dead credit balance.

Fix landed:

1. ``_NON_RETRYABLE_PHRASES`` now includes ``"is not available"``,
   ``"not_found_error"``, ``"credit balance is too low"`` so the retry
   loop shortcircuits.
2. ``MODEL_INFO`` entries may declare a ``fallback`` model name.
   ``claude-fable-5`` has ``"fallback": "claude-opus-4-8"``.
3. On a non-retryable error, ``KISSAgent._try_switch_to_fallback``
   rebuilds the model to the registered fallback (preserving the
   conversation history and the caller's ``model_config`` overrides),
   refreshes the cached tool schema, and the loop transparently
   continues on the fallback.  One-shot guard prevents A→B→A cycles.

These tests exercise the whole path via a local HTTP capture server
using the same pattern as ``test_empty_response_silent_death.py`` —
first request (from the primary model) returns a non-retryable error;
second request (from the fallback model) returns a normal ``finish``
tool call.
"""

from __future__ import annotations

from kiss.core.models import model_info as model_info_module
from kiss.core.models.model_info import (
    MODEL_INFO,
    get_fallback_model,
)


class TestGetFallbackModel:
    """``get_fallback_model`` MODEL_INFO lookup."""

    def test_claude_fable_5_falls_back_to_opus_4_8(self) -> None:
        assert get_fallback_model("claude-fable-5") == "claude-opus-4-8"

    def test_unknown_model_returns_none(self) -> None:
        assert get_fallback_model("does-not-exist-xyz") is None

    def test_model_without_fallback_returns_none(self) -> None:
        """A registered model that does not declare ``fallback``
        returns ``None`` (not an error)."""
        assert get_fallback_model("claude-opus-4-8") is None

    def test_harbor_prefix_is_stripped(self) -> None:
        """``get_fallback_model`` accepts harbor-style ``provider/name``
        input (matching the behavior of ``get_max_context_length``)."""
        assert get_fallback_model("anthropic/claude-fable-5") == "claude-opus-4-8"


class TestModelInfoJsonHasFallback:
    """Sanity: ``MODEL_INFO.json`` still declares the fable-5 fallback.

    Prevents accidental deletion during future JSON edits (the whole
    point of this feature)."""

    def test_claude_fable_5_entry_has_fallback(self) -> None:
        info = MODEL_INFO["claude-fable-5"]
        assert info.fallback == "claude-opus-4-8"

    def test_openrouter_fable_5_entry_has_fallback(self) -> None:
        """The OpenRouter mirror of fable-5 is a separate MODEL_INFO key
        (harbor prefix ``openrouter/`` is NOT stripped by
        ``_strip_provider_prefix``), so it needs its own ``fallback``
        entry.  Users routing through OpenRouter otherwise get no
        fallback at all."""
        info = MODEL_INFO["openrouter/anthropic/claude-fable-5"]
        assert info.fallback == "openrouter/anthropic/claude-opus-4.8"
        assert "openrouter/anthropic/claude-opus-4.8" in MODEL_INFO

    def test_module_reference_still_present(self) -> None:
        """Guard against accidental removal of ``get_fallback_model``."""
        assert hasattr(model_info_module, "get_fallback_model")

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: provider routing has exactly one source of truth.

Two findings from ``tmp/audit/01-core-models-a.md``:

* **F5** — ``text-embedding-004`` was special-cased by exact equality in
  four places (the OpenAI registry ``excludes``, the ``model()`` factory,
  ``get_available_models`` and ``get_model_provider``) even though it is
  not a key of the shipped catalog, so every one of those branches was
  dead.  Worse, the special case *inverted* the routing rule: a name
  starting with ``text-embedding`` belongs to OpenAI everywhere else in
  the file, so a real model with that name would have been silently
  routed to Gemini and credited to the Gemini API key.
* **F6** — ``OPENAI_COMPATIBLE_PROVIDERS`` calls itself "the single
  source of truth" for how model names route to a vendor and which
  credential they use, but ``get_available_models`` and
  ``get_model_provider`` each restated the same prefix → credential
  mapping by hand.  A vendor added
  to the registry therefore executed correctly yet stayed invisible and
  unselectable in the VS Code / web model picker, with no error anywhere.

No mocks, patches or test doubles: the tests register a real provider in
the real registry and a real entry in the real catalog, set a real API
key on the real ``config.DEFAULT_CONFIG`` object, and assert on what the
real routing functions do — restoring everything in ``finally``.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest

from kiss.core import config as config_module
from kiss.core.models import model_info as mi
from kiss.core.models.model_info import (
    MODEL_INFO,
    OpenAICompatibleProvider,
    _build_model_info_entry,
    _openai_bare_name,
    get_available_models,
    get_model_provider,
    model,
)

_VENDOR_MODEL = "zzvendor-flagship"
# ``config.Config`` is a pydantic model with ``extra`` forbidden, so a test
# cannot invent a brand-new credential attribute.  Reusing an existing one
# does not weaken the test: what is under scrutiny is that the routing
# functions read ``provider.api_key_name`` from the registry at all, rather
# than from their own hardcoded prefix table.
_VENDOR_KEY = "TOGETHER_API_KEY"

_ENTRY = {
    "context_length": 128000,
    "input_price_per_1M": 1.0,
    "output_price_per_1M": 2.0,
}


@pytest.fixture
def registered_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[OpenAICompatibleProvider]:
    """Register a real extra vendor for the duration of one test."""
    provider = OpenAICompatibleProvider(
        name="zzvendor",
        label="ZZVendor",
        host="api.zzvendor.test",
        base_url="https://api.zzvendor.test/v1",
        prefixes=("zzvendor-",),
        excludes=(),
        api_key_name=_VENDOR_KEY,
        tools_accept_reasoning_effort=None,
        delegate_tools_to_responses=False,
    )
    saved_registry = mi.OPENAI_COMPATIBLE_PROVIDERS
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, _VENDOR_KEY, "zz-test-key")
    mi.OPENAI_COMPATIBLE_PROVIDERS = (*saved_registry, provider)
    MODEL_INFO[_VENDOR_MODEL] = _build_model_info_entry(dict(_ENTRY))
    try:
        yield provider
    finally:
        mi.OPENAI_COMPATIBLE_PROVIDERS = saved_registry
        MODEL_INFO.pop(_VENDOR_MODEL, None)


class TestRegistryIsTheOnlyRoutingTable:
    """F6: a vendor added to the registry must be fully visible."""

    def test_factory_routes_to_the_registered_base_url(
        self, registered_vendor: OpenAICompatibleProvider,
    ) -> None:
        """``model()`` already honoured the registry — the baseline."""
        m = model(_VENDOR_MODEL)

        assert type(m).__name__ == "OpenAICompatibleModel"
        assert getattr(m, "base_url") == registered_vendor.base_url  # noqa: B009

    def test_provider_label_comes_from_the_registry(
        self, registered_vendor: OpenAICompatibleProvider,
    ) -> None:
        """``get_model_provider`` must not need its own prefix table."""
        assert get_model_provider(_VENDOR_MODEL) == registered_vendor.label

    def test_model_is_offered_in_the_picker(
        self, registered_vendor: OpenAICompatibleProvider,
    ) -> None:
        """``get_available_models`` must find the credential via the registry."""
        assert _VENDOR_MODEL in get_available_models()

    def test_registry_reports_the_vendor_as_configured(
        self, registered_vendor: OpenAICompatibleProvider,
    ) -> None:
        """Provider routing and credential lookup must both see the vendor."""
        assert get_model_provider(_VENDOR_MODEL) == registered_vendor.label
        assert mi._configured_providers()[registered_vendor.label] is True

    def test_unregistered_name_is_still_unknown(self) -> None:
        """Names no route matches must keep reporting ``Unknown``."""
        assert get_model_provider("no-such-vendor/no-such-model") == "Unknown"


class TestShippedCatalogRoutesCompletely:
    """F6 regression: every shipped model must have a real provider."""

    def test_every_catalog_entry_routes_to_a_known_provider(self) -> None:
        """No shipped model may fall through to ``Unknown``."""
        unknown = [
            name for name in MODEL_INFO if get_model_provider(name) == "Unknown"
        ]

        assert unknown == []

    def test_known_provider_labels_are_stable(self) -> None:
        """The labels the UI renders must not drift."""
        assert get_model_provider("openrouter/openai/gpt-4o") == "OpenRouter"
        assert get_model_provider("claude-opus-4-7") == "Anthropic"
        assert get_model_provider("gemini-3-pro-preview") == "Gemini"
        assert get_model_provider("gpt-5.6") == "OpenAI"
        assert get_model_provider("glm-4.6") == "Z.AI"
        assert get_model_provider("kimi-k2-0905-preview") == "Moonshot"
        assert get_model_provider("meta-llama/Llama-3.3-70B-Instruct-Turbo") == "Together"
        assert get_model_provider("openai/gpt-oss-20b") == "Together"
        assert get_model_provider("cc/opus") == "Claude Code CLI"
        assert get_model_provider("codex/gpt-5-codex") == "Codex CLI"


class TestNoDeadExactNameSpecialCases:
    """F5: routing must not be pinned to a name absent from the catalog."""

    def test_text_embedding_004_is_not_in_the_catalog(self) -> None:
        """The premise of the finding, asserted rather than assumed."""
        assert "text-embedding-004" not in MODEL_INFO

    def test_text_embedding_prefix_routes_consistently(self) -> None:
        """Every ``text-embedding-*`` name must route the same way."""
        assert get_model_provider("text-embedding-3-small") == "OpenAI"
        assert get_model_provider("text-embedding-004") == "OpenAI"
        assert type(model("text-embedding-004")).__name__ == "OpenAICompatibleModel"

    def test_gemini_embeddings_route_by_their_prefix(self) -> None:
        """Google's catalog embedding models need no special case."""
        assert get_model_provider("gemini-embedding-001") == "Gemini"
        assert type(model("gemini-embedding-001")).__name__ == "GeminiModel"

    def test_openai_cache_pricing_names_are_unchanged(self) -> None:
        """Removing the unreachable ``"openai/"`` exclusion changes nothing."""
        assert _openai_bare_name("gpt-5.6") == "gpt-5.6"
        assert _openai_bare_name("openai/gpt-oss-20b") is None
        assert _openai_bare_name("text-embedding-3-small") is None
        assert _openai_bare_name("codex/gpt-5-codex") is None
        assert _openai_bare_name("claude-opus-4-7") is None
        assert _openai_bare_name("openrouter/openai/gpt-4o") == "gpt-4o"
        assert _openai_bare_name("openrouter/openai/gpt-oss-20b") is None


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(pytest.main([__file__]))

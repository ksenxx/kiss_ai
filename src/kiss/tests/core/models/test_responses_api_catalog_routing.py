# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests for catalog-driven OpenAI v2 (Responses API) routing.

``MODEL_INFO.json`` entries may carry ``"use_responses_api": true`` — written
by ``scripts/update_responses_api_support.py`` only after the model passed a
live probe through ``/v1/responses``.  The ``model()`` factory must build
those models as :class:`OpenAICompatibleModel2` (every request goes to the
Responses API) and everything else as the Chat Completions v1 adapter, with
``model_config["use_responses_api"]`` overriding in either direction.
"""

import os

import pytest

from kiss.core.models.model_info import (
    MODEL_INFO,
    _match_openai_compatible_provider,
    model,
)
from kiss.core.models.openai_compatible_model import OpenAICompatibleModel
from kiss.core.models.openai_compatible_model2 import OpenAICompatibleModel2


class TestCatalogFlagRouting:
    """The factory's transport choice must follow the catalog flag exactly."""

    def test_every_catalog_model_routes_to_the_flagged_transport(self) -> None:
        """Sweep the whole catalog: flag=True -> v2, otherwise -> v1.

        This is the single consistency check that guards every one of the
        600+ entries at once, so a future catalog edit cannot silently
        route a flagged model over Chat Completions (or vice versa).
        """
        checked = 0
        for name, info in MODEL_INFO.items():
            if _match_openai_compatible_provider(name) is None:
                continue
            m = model(name)
            expected = (
                OpenAICompatibleModel2
                if info.use_responses_api is True
                else OpenAICompatibleModel
            )
            assert type(m) is expected, (name, info.use_responses_api)
            checked += 1
        assert checked > 100

    def test_probe_verified_openai_models_are_flagged(self) -> None:
        """Models the live probe verified must carry the catalog flag."""
        for name in ("gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5.6-sol"):
            info = MODEL_INFO[name]
            assert info.use_responses_api is True, name
            assert isinstance(model(name), OpenAICompatibleModel2), name

    def test_chat_completions_only_models_stay_on_v1(self) -> None:
        """Models OpenAI serves only on Chat Completions must stay v1.

        ``gpt-4o-search-preview`` and ``gpt-audio`` are rejected by
        ``/v1/responses`` ("not supported with the Responses API"), and
        embeddings have no generation endpoint at all.
        """
        for name in (
            "gpt-4o-search-preview",
            "gpt-4o-mini-search-preview",
            "gpt-audio",
            "gpt-audio-mini",
            "text-embedding-3-small",
        ):
            info = MODEL_INFO[name]
            assert info.use_responses_api is not True, name
            assert isinstance(model(name), OpenAICompatibleModel), name

    def test_generated_aliases_mirror_their_base_flag(self) -> None:
        """Every generated -{level} alias carries its base entry's flag."""
        for name, info in MODEL_INFO.items():
            if not info.alias_of:
                continue
            base = MODEL_INFO.get(info.alias_of)
            if base is None:
                continue
            assert info.use_responses_api == base.use_responses_api, name

    def test_flagged_alias_routes_to_v2(self) -> None:
        """A flagged thinking alias builds the v2 adapter like its base."""
        base = MODEL_INFO["gpt-5.6-sol"]
        assert base.use_responses_api is True
        assert isinstance(model("gpt-5.6-sol-low"), OpenAICompatibleModel2)
        assert isinstance(model("gpt-5.6-sol-xhigh"), OpenAICompatibleModel2)


class TestConfigOverrides:
    """model_config["use_responses_api"] must win over the catalog flag."""

    def test_config_false_forces_v1_on_a_flagged_model(self) -> None:
        """use_responses_api=False keeps a flagged model on Chat Completions."""
        m = model("gpt-4o", model_config={"use_responses_api": False})
        assert isinstance(m, OpenAICompatibleModel)

    def test_config_true_forces_v2_on_an_unflagged_model(self) -> None:
        """use_responses_api=True builds v2 even without a catalog flag."""
        m = model("gpt-audio", model_config={"use_responses_api": True})
        assert isinstance(m, OpenAICompatibleModel2)

    def test_absent_config_key_defers_to_the_catalog(self) -> None:
        """A config without the key must not disturb catalog routing."""
        m = model("gpt-4o", model_config={"temperature": 0.0})
        assert isinstance(m, OpenAICompatibleModel2)
        assert m.model_config == {"temperature": 0.0}

    def test_custom_base_url_override_stays_on_v1(self) -> None:
        """A custom gateway override keeps v1 (unknown /v1/responses support).

        This includes capture-server style URLs that merely EMBED a vendor
        host as a path segment — only an exact match with a registered
        vendor's default endpoint may inherit the catalog flag.
        """
        for base_url in (
            "http://localhost:1/v1",
            "http://127.0.0.1:9/api.openai.com/v1",
        ):
            m = model(
                "gpt-4o",
                model_config={"base_url": base_url, "api_key": "k"},
            )
            assert isinstance(m, OpenAICompatibleModel), base_url
            assert not isinstance(m, OpenAICompatibleModel2), base_url

    def test_provider_default_base_url_override_keeps_catalog_flag(self) -> None:
        """A provider-default base_url override preserves v2 routing.

        Sorcar's ``set_model`` carries the old adapter's base_url along to
        preserve a per-task API key; when that URL is exactly the vendor's
        registered endpoint the flagged model must still get the v2
        transport instead of being silently downgraded to Chat Completions.
        """
        m = model(
            "gpt-4o",
            model_config={"base_url": "https://api.openai.com/v1", "api_key": "k"},
        )
        assert isinstance(m, OpenAICompatibleModel2)
        m2 = model(
            "gpt-audio",  # unflagged: stays v1 even on the default endpoint
            model_config={"base_url": "https://api.openai.com/v1", "api_key": "k"},
        )
        assert isinstance(m2, OpenAICompatibleModel)

    def test_other_vendors_default_base_url_stays_on_v1(self) -> None:
        """A DIFFERENT registered vendor's default URL must not inherit the flag.

        The probe verified gpt-4o on api.openai.com only; pointing it at
        Together's or OpenRouter's default endpoint is an unverified
        model/endpoint pair and must stay on Chat Completions.
        """
        for base_url in (
            "https://api.together.xyz/v1",
            "https://openrouter.ai/api/v1",
        ):
            m = model(
                "gpt-4o", model_config={"base_url": base_url, "api_key": "k"}
            )
            assert isinstance(m, OpenAICompatibleModel), base_url
            assert not isinstance(m, OpenAICompatibleModel2), base_url

    def test_config_false_wins_over_provider_default_base_url(self) -> None:
        """An explicit False keeps v1 even on the vendor's default endpoint."""
        m = model(
            "gpt-4o",
            model_config={
                "base_url": "https://api.openai.com/v1",
                "api_key": "k",
                "use_responses_api": False,
            },
        )
        assert isinstance(m, OpenAICompatibleModel)
        assert not isinstance(m, OpenAICompatibleModel2)


class TestMyModelsAliasSync:
    """A base-only MY_MODELS override must re-mirror onto generated aliases."""

    def test_base_override_false_propagates_to_bundled_aliases(self) -> None:
        """Turning the flag off on the base turns it off on -low/-xhigh too."""
        from kiss.core.models.model_info import _sync_alias_transport_flags

        raw: dict[str, dict] = {
            "gpt-7": {"use_responses_api": False},
            "gpt-7-low": {"alias_of": "gpt-7", "use_responses_api": True},
            "gpt-7-xhigh": {"alias_of": "gpt-7", "use_responses_api": True},
        }
        _sync_alias_transport_flags(raw, {"gpt-7"})
        assert raw["gpt-7-low"]["use_responses_api"] is False
        assert raw["gpt-7-xhigh"]["use_responses_api"] is False

    def test_base_override_without_flag_removes_it_from_aliases(self) -> None:
        """An override that drops the key drops it from mirrored aliases."""
        from kiss.core.models.model_info import _sync_alias_transport_flags

        raw: dict[str, dict] = {
            "gpt-7": {},
            "gpt-7-low": {"alias_of": "gpt-7", "use_responses_api": True},
        }
        _sync_alias_transport_flags(raw, {"gpt-7"})
        assert "use_responses_api" not in raw["gpt-7-low"]

    def test_explicit_alias_override_wins_over_base_sync(self) -> None:
        """A user-supplied alias entry keeps its own flag."""
        from kiss.core.models.model_info import _sync_alias_transport_flags

        raw: dict[str, dict] = {
            "gpt-7": {"use_responses_api": False},
            "gpt-7-low": {"alias_of": "gpt-7", "use_responses_api": True},
        }
        _sync_alias_transport_flags(raw, {"gpt-7", "gpt-7-low"})
        assert raw["gpt-7-low"]["use_responses_api"] is True

    def test_untouched_bases_leave_aliases_alone(self) -> None:
        """Without a user override the bundled alias copies are untouched."""
        from kiss.core.models.model_info import _sync_alias_transport_flags

        raw: dict[str, dict] = {
            "gpt-7": {"use_responses_api": True},
            "gpt-7-low": {"alias_of": "gpt-7", "use_responses_api": True},
        }
        _sync_alias_transport_flags(raw, set())
        assert raw["gpt-7-low"]["use_responses_api"] is True


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestLiveResponsesRouting:
    """Live: a flagged model built by the factory works over /v1/responses."""

    def test_flagged_model_generates_via_responses(self) -> None:
        """gpt-4o-mini (flagged) completes a live turn through v2."""
        m = model("gpt-4o-mini")
        assert isinstance(m, OpenAICompatibleModel2)
        m.initialize("Reply with exactly the word OK and nothing else.")
        text, _ = m.generate()
        assert "ok" in text.lower()

    def test_flagged_model_tool_round_trip_via_responses(self) -> None:
        """gpt-4o-mini completes a live tool round trip through v2."""

        def add(a: int = 0, b: int = 0) -> str:
            """Add two integers.

            Args:
                a: First addend.
                b: Second addend.
            """
            return str(a + b)

        m = model("gpt-4o-mini")
        assert isinstance(m, OpenAICompatibleModel2)
        m.initialize("What is 411 plus 289? Use the add tool.")
        calls, _content, _ = m.generate_and_process_with_tools({"add": add})
        assert calls, "expected a function call"
        results = [
            (c["name"], {"result": add(**{k: int(v) for k, v in
                                          (c.get("arguments") or {}).items()})})
            for c in calls
        ]
        m.add_function_results_to_conversation_and_return(results)
        _calls2, content2, _ = m.generate_and_process_with_tools({"add": add})
        assert "700" in content2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

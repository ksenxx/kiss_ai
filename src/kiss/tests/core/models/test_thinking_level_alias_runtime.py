# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for runtime resolution of ``-{thinking_level}`` aliases.

``update_models.py`` materializes one catalog entry per supported
``reasoning_effort`` level (``gpt-5.6-sol-low`` … ``gpt-5.6-sol-xhigh``).
At runtime those synthetic names must:

* resolve to the base provider model id on the wire
  (:func:`kiss.core.models.openai_compatible_model._provider_model_name`),
* carry their own ``thinking`` level so the request gets the matching
  ``reasoning_effort``
  (:func:`kiss.core.models.openai_compatible_model._model_thinking_level`),
* cost exactly the same as the base model in every pricing dimension.

Real upstream models whose names merely end in ``-high`` (e.g.
``openrouter/openai/o3-mini-high``) must NOT be rewritten — they carry no
``alias_of`` marker in the catalog.

These tests run against the real bundled ``MODEL_INFO`` catalog; nothing
is mocked.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.core.models import model_info as mi
from kiss.core.models.model_info import (
    MODEL_INFO,
    _strip_thinking_alias,
    calculate_cost,
)
from kiss.core.models.openai_compatible_model import (
    OpenAICompatibleBase,
    _model_thinking_level,
    _provider_model_name,
)

ALL_LEVELS = ("low", "medium", "high", "xhigh")

KNOWN_LEVELS = ("low", "medium", "high", "xhigh", "max")
"""Every level across all vendor scales: the OpenAI ladder plus the
Moonshot/Kimi top level ``max`` (see ``_thinking_scale_for`` in
``update_models.py``)."""


def _alias_pairs() -> list[tuple[str, str]]:
    """Return every ``(base_name, alias_name)`` pair marked in MODEL_INFO."""
    pairs = []
    for name, info in MODEL_INFO.items():
        if info.alias_of:
            pairs.append((info.alias_of, name))
    return pairs


def test_catalog_ships_marked_aliases_for_every_level() -> None:
    """gpt-5.6-sol must expose one alias per thinking level (user example)."""
    for level in ALL_LEVELS:
        name = f"gpt-5.6-sol-{level}"
        info = MODEL_INFO.get(name)
        assert info is not None, f"{name} missing from MODEL_INFO"
        assert info.thinking == level
        assert info.alias_of == "gpt-5.6-sol"


def test_every_marked_alias_has_consistent_base() -> None:
    """Every alias must point at an existing base and share its pricing."""
    pairs = _alias_pairs()
    assert pairs, "MODEL_INFO has no generated thinking-level aliases"
    for base_name, alias_name in pairs:
        assert base_name in MODEL_INFO, f"{alias_name} orphaned: no {base_name}"
        base, alias = MODEL_INFO[base_name], MODEL_INFO[alias_name]
        assert alias.context_length == base.context_length
        assert alias.input_price_per_1M == base.input_price_per_1M
        assert alias.output_price_per_1M == base.output_price_per_1M
        assert alias.is_function_calling_supported == base.is_function_calling_supported
        assert alias.thinking in KNOWN_LEVELS


def test_strip_thinking_alias_strips_marked_level_suffixes() -> None:
    """Marked aliases map back to the base name for provider requests."""
    for level in ALL_LEVELS:
        assert _strip_thinking_alias(f"gpt-5.6-sol-{level}") == "gpt-5.6-sol"


def test_strip_thinking_alias_resolves_full_openrouter_keys() -> None:
    """Full openrouter catalog keys resolve to the recorded base key."""
    assert (
        _strip_thinking_alias("openrouter/openai/gpt-5.6-sol-medium")
        == "openrouter/openai/gpt-5.6-sol"
    )
    assert (
        _strip_thinking_alias("openrouter/~openai/gpt-sol-latest-high")
        == "openrouter/~openai/gpt-sol-latest"
    )


def test_strip_thinking_alias_requires_exact_catalog_key() -> None:
    """Only exact ``alias_of``-marked keys are rewritten — never lookalikes.

    A custom (e.g. local base_url) model that merely shares the ``-high``
    tail with some catalog alias must keep its name on the wire; the old
    fuzzy tail-scan lookup rewrote it and silently changed which model was
    invoked.
    """
    assert MODEL_INFO["openrouter/~openai/gpt-sol-latest-high"].alias_of
    assert "gpt-sol-latest-high" not in MODEL_INFO
    assert _strip_thinking_alias("gpt-sol-latest-high") == "gpt-sol-latest-high"
    assert _provider_model_name("gpt-sol-latest-high") == "gpt-sol-latest-high"


def test_strip_thinking_alias_keeps_real_upstream_high_models() -> None:
    """o3-mini-high / o4-mini-high are real models and must not be rewritten."""
    assert _strip_thinking_alias("openai/o3-mini-high") == "openai/o3-mini-high"
    assert _strip_thinking_alias("openai/o4-mini-high") == "openai/o4-mini-high"


def test_strip_thinking_alias_keeps_unknown_suffixed_names() -> None:
    """Names absent from the catalog are passed through untouched."""
    assert _strip_thinking_alias("acme/custom-high") == "acme/custom-high"
    assert _strip_thinking_alias("acme/custom-medium") == "acme/custom-medium"
    assert _strip_thinking_alias("acme/custom-low") == "acme/custom-low"
    assert _strip_thinking_alias("plain-model") == "plain-model"


def test_strip_thinking_alias_strips_xhigh_unless_registered_under_that_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """-xhigh strips for an unknown or bundled-alias name, but a ``MY_MODELS.json``
    model registered under a ``-xhigh`` name itself (a router that reads the
    effort off the name) keeps it on the wire, with or without a restart; an
    entry carrying ``alias_of`` is an alias and strips as before, and so does a
    settings-panel override of a bundled alias, which is written without the
    marker (the loader restores it from the catalog)."""
    monkeypatch.setattr(mi, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    assert _strip_thinking_alias("anything-at-all-xhigh") == "anything-at-all"
    assert _provider_model_name("gpt-5.6-sol-xhigh") == "gpt-5.6-sol"
    fields = {"context_length": 262144, "input_price_per_1M": 0.07, "output_price_per_1M": 0.15}
    (tmp_path / "MY_MODELS.json").write_text(
        json.dumps(
            {
                "s10s-scout-xhigh": {**fields, "thinking": "xhigh", "endpoint": "http://r/v1"},
                "acme/alias-xhigh": {**fields, "alias_of": "acme/alias"},
                "gpt-5.6-sol-xhigh": {**fields, "endpoint": "http://a/v1"},
            }
        )
    )
    # Hot-added (registered after the catalog was loaded): read live from the file.
    assert "s10s-scout-xhigh" not in MODEL_INFO
    assert _strip_thinking_alias("s10s-scout-xhigh") == "s10s-scout-xhigh"
    assert _provider_model_name("s10s-scout-xhigh") == "s10s-scout-xhigh"
    assert _strip_thinking_alias("acme/alias-xhigh") == "acme/alias"
    # The bundled alias entry wins over the override until a restart.
    assert _provider_model_name("gpt-5.6-sol-xhigh") == "gpt-5.6-sol"
    # Loaded with the catalog (what a restart does): the same answers.
    loaded = mi._load_catalog_file(mi.PACKAGE_MODEL_INFO_PATH)
    for name in ("s10s-scout-xhigh", "acme/alias-xhigh", "gpt-5.6-sol-xhigh"):
        monkeypatch.setitem(MODEL_INFO, name, loaded[name])
    (tmp_path / "MY_MODELS.json").write_text("{}")
    assert _provider_model_name("s10s-scout-xhigh") == "s10s-scout-xhigh"
    assert _model_thinking_level("s10s-scout-xhigh") == "xhigh"
    assert _strip_thinking_alias("acme/alias-xhigh") == "acme/alias"
    assert loaded["gpt-5.6-sol-xhigh"].alias_of == "gpt-5.6-sol"
    assert loaded["gpt-5.6-sol-xhigh"].context_length == 262144  # the override's own fields
    assert _provider_model_name("gpt-5.6-sol-xhigh") == "gpt-5.6-sol"
    # The override names the alias, so it keeps requesting the alias's effort.
    assert loaded["gpt-5.6-sol-xhigh"].thinking == "xhigh"
    assert _model_thinking_level("gpt-5.6-sol-xhigh") == "xhigh"


def test_saved_endpoint_keeps_the_bundled_thinking_level_after_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A settings-panel save of only an endpoint for a bundled model must not
    change the effort it requests once the row replaces the catalog entry at
    the next start: the seeded row carries the catalog's ``thinking`` level,
    for a generated alias and for its base alike."""
    monkeypatch.setattr(mi, "USER_MY_MODELS_PATH", tmp_path / "MY_MODELS.json")
    for name in ("gpt-5.6-sol-xhigh", "gpt-5.6-sol"):
        assert mi.save_custom_model(name, endpoint="http://127.0.0.1:1/v1", api_key="k") is None
    rows = json.loads((tmp_path / "MY_MODELS.json").read_text())
    assert rows["gpt-5.6-sol-xhigh"]["thinking"] == "xhigh"
    assert rows["gpt-5.6-sol"]["thinking"] == "high"
    loaded = mi._load_catalog_file(mi.PACKAGE_MODEL_INFO_PATH)
    for name in ("gpt-5.6-sol-xhigh", "gpt-5.6-sol"):
        monkeypatch.setitem(MODEL_INFO, name, loaded[name])
    alias = "gpt-5.6-sol-xhigh"
    adapter = mi.model(alias, model_config=mi.custom_model_config(alias))
    assert isinstance(adapter, OpenAICompatibleBase)
    assert adapter._api_model_name == "gpt-5.6-sol"
    assert adapter.model_config["reasoning_effort"] == "xhigh"
    assert _model_thinking_level("gpt-5.6-sol") == "high"


def test_provider_model_name_maps_alias_to_base_wire_id() -> None:
    """The wire model id for every alias is the base provider id."""
    assert _provider_model_name("gpt-5.6-sol-high") == "gpt-5.6-sol"
    assert _provider_model_name("gpt-5.6-sol-low") == "gpt-5.6-sol"
    assert (
        _provider_model_name("openrouter/openai/gpt-5.6-sol-medium") == "openai/gpt-5.6-sol"
    )
    assert (
        _provider_model_name("openrouter/openai/o3-mini-high") == "openai/o3-mini-high"
    )


def test_model_thinking_level_reflects_alias_level() -> None:
    """Each alias must request its own reasoning_effort level."""
    for level in ALL_LEVELS:
        assert _model_thinking_level(f"gpt-5.6-sol-{level}") == level
    assert _model_thinking_level("gpt-5.6-sol") == "high"


_TOKEN_PROFILES: tuple[tuple[int, int, int, int, int], ...] = (
    (1_000, 500, 0, 0, 0),
    (1_000, 1_000, 1_000, 0, 0),
    (1_000_000, 1_000_000, 1_000_000, 0, 0),
    (0, 0, 0, 1_000_000, 0),
    (0, 0, 0, 0, 1_000_000),
    (250_000, 250_000, 250_000, 0, 0),
)


@pytest.mark.parametrize("profile", _TOKEN_PROFILES)
def test_every_alias_costs_exactly_like_its_base(
    profile: tuple[int, int, int, int, int],
) -> None:
    """Aliases route to the base provider model, so costs must be identical."""
    inp, out, cread, cwrite, cwrite1h = profile
    for base_name, alias_name in _alias_pairs():
        base_cost = calculate_cost(base_name, inp, out, cread, cwrite, cwrite1h)
        alias_cost = calculate_cost(alias_name, inp, out, cread, cwrite, cwrite1h)
        assert alias_cost == pytest.approx(base_cost), (
            f"{alias_name} cost {alias_cost} != base {base_name} cost {base_cost} "
            f"for profile {profile}"
        )


def test_an_unlisted_xhigh_alias_is_sized_and_priced_as_its_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``codex/gpt-6-astra-xhigh`` has no catalog entry and every adapter runs it as
    ``codex/gpt-6-astra`` (:func:`_strip_thinking_alias`), so the lookup that sizes and
    prices a run answers with that entry: the run otherwise has no context window and
    fails at its first ``calculate_cost``.  A Bedrock profile pays its premium once.  The
    base's effort level is not the alias's (the catalog lists none for it), so a custom
    entry seeded from the alias carries none.  A name registered under itself keeps its
    own entry, and one whose entry cannot be built stays unknown, under its harbor
    spelling too."""
    registry = tmp_path / "MY_MODELS.json"
    monkeypatch.setattr(mi, "USER_MY_MODELS_PATH", registry)
    bases = [
        n for n, info in MODEL_INFO.items() if not info.alias_of and f"{n}-xhigh" not in MODEL_INFO
    ]
    assert {"codex/gpt-6-astra", "claude-opus-4-6"} <= set(bases)
    usage = (250_000, 1_000, 1_000, 1_000, 1_000)
    for base in bases:
        alias = f"{base}-xhigh"
        assert _strip_thinking_alias(alias) == base
        assert mi.get_max_context_length(alias) == mi.get_max_context_length(base), alias
        assert calculate_cost(alias, *usage) == calculate_cost(base, *usage), alias
    profile = "bedrock/us.anthropic.claude-opus-4-6-v1:0"
    assert profile not in MODEL_INFO
    premium = 1.1 * calculate_cost("claude-opus-4-6", 10_000, 0)
    assert calculate_cost(profile, 10_000, 0) == pytest.approx(premium)
    assert calculate_cost(f"{profile}-xhigh", 10_000, 0) == pytest.approx(premium)
    opus = MODEL_INFO["claude-opus-4-6"]
    assert opus.thinking
    assert mi.save_custom_model("claude-opus-4-6-xhigh", endpoint="http://r/v1") is None
    seeded = json.loads(registry.read_text())["claude-opus-4-6-xhigh"]
    assert seeded["context_length"] == opus.context_length
    assert seeded["input_price_per_1M"] == opus.input_price_per_1M
    assert "thinking" not in seeded
    fields = {"context_length": 8192, "input_price_per_1M": 7, "output_price_per_1M": 9}
    unbuildable = {**fields, "input_price_per_1M": "7"}
    registry.write_text(
        json.dumps({"codex/gpt-6-astra-xhigh": fields, "claude-opus-4-6-xhigh": unbuildable})
    )
    assert mi.get_max_context_length("codex/gpt-6-astra-xhigh") == 8192
    assert calculate_cost("codex/gpt-6-astra-xhigh", 1_000_000, 0) == 7
    assert not mi.is_known_model("claude-opus-4-6-xhigh")
    assert not mi.is_known_model("anthropic/claude-opus-4-6-xhigh")

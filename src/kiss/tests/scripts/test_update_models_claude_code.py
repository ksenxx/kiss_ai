# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that ``update_models.py`` mirrors Anthropic models as ``cc/*`` entries.

The Claude Code backend (``kiss.core.models.claude_code_model``) passes the
part after ``cc/`` verbatim as the ``claude`` CLI's ``--model`` flag, which
accepts any Anthropic model ID plus the short ``haiku``/``opus``/``sonnet``
aliases. ``compute_changes`` must therefore propose a ``cc/<model>`` sibling
for every model returned by the Anthropic models API (billed $0/0 since
Claude Code runs on the user's subscription), and ``find_deprecated_models``
must retire ``cc/claude-*`` entries whose slug is gone upstream while never
touching the short aliases.

These tests drive the production ``compute_changes``,
``_add_claude_code_candidates``, and ``find_deprecated_models`` functions
end-to-end with synthetic vendor payloads — the same data flow the script
uses against the live APIs.
"""

from kiss.scripts.update_models import (
    _add_claude_code_candidates,
    compute_changes,
    find_deprecated_models,
)


def _by_name(new_models: list[dict]) -> dict[str, dict]:
    """Index a ``new_models`` list by model name."""
    return {m["name"]: m for m in new_models}


def test_every_anthropic_model_gets_a_cc_sibling() -> None:
    """Each Anthropic API model is proposed as a ``cc/<model>`` entry."""
    anthropic = {
        "claude-opus-4-6": {"source": "anthropic"},
        "claude-sonnet-4-5-20250929": {"source": "anthropic"},
    }
    _, new_models = compute_changes({}, {}, {}, {}, anthropic, {})
    names = {m["name"] for m in new_models}
    assert "cc/claude-opus-4-6" in names
    assert "cc/claude-sonnet-4-5-20250929" in names


def test_cc_aliases_are_seeded_alongside_api_models() -> None:
    """The short haiku/opus/sonnet aliases are proposed on a fresh catalog."""
    anthropic = {"claude-opus-4-6": {"source": "anthropic"}}
    _, new_models = compute_changes({}, {}, {}, {}, anthropic, {})
    names = {m["name"] for m in new_models}
    assert {"cc/haiku", "cc/opus", "cc/sonnet"} <= names


def test_cc_entries_are_zero_priced_and_generation_capable() -> None:
    """``cc/*`` candidates are $0/0 (subscription-billed), gen+fc, not emb."""
    anthropic = {"claude-opus-4-6": {"source": "anthropic"}}
    new_models: list[dict] = []
    _add_claude_code_candidates(anthropic, {}, {}, new_models)
    for entry in _by_name(new_models).values():
        assert entry["input_price_per_1M"] == 0.0
        assert entry["output_price_per_1M"] == 0.0
        assert entry["needs_pricing"] is False
        assert entry["gen"] is True
        assert entry["fc"] is True
        assert entry["emb"] is False
        assert entry["source"] == "claude-code"


def test_cc_context_prefers_openrouter_then_catalog_then_default() -> None:
    """Context: OpenRouter match > direct claude-* catalog entry > 200000."""
    anthropic = {
        "claude-opus-4-6": {"source": "anthropic"},
        "claude-sonnet-4-6": {"source": "anthropic"},
        "claude-haiku-4-5": {"source": "anthropic"},
    }
    openrouter = {
        "openrouter/anthropic/claude-opus-4-6": {
            "context_length": 500000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 25.0,
            "source": "openrouter",
        }
    }
    current = {
        "claude-sonnet-4-6": {
            "context_length": 300000,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
            "alias_of": None,
        }
    }
    new_models: list[dict] = []
    _add_claude_code_candidates(anthropic, current, openrouter, new_models)
    entries = _by_name(new_models)
    assert entries["cc/claude-opus-4-6"]["context_length"] == 500000
    assert entries["cc/claude-sonnet-4-6"]["context_length"] == 300000
    assert entries["cc/claude-haiku-4-5"]["context_length"] == 200000
    assert entries["cc/opus"]["context_length"] == 200000


def test_cc_candidates_skip_entries_already_in_catalog() -> None:
    """A ``cc/*`` name already in the current catalog is not re-proposed."""
    anthropic = {"claude-opus-4-6": {"source": "anthropic"}}
    current = {
        name: {
            "context_length": 200000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
            "alias_of": None,
        }
        for name in ("cc/haiku", "cc/opus", "cc/sonnet", "cc/claude-opus-4-6")
    }
    new_models: list[dict] = []
    _add_claude_code_candidates(anthropic, current, {}, new_models)
    assert new_models == []


def test_no_cc_candidates_when_anthropic_fetch_is_empty() -> None:
    """Without Anthropic data (e.g. missing API key), no cc/* is proposed."""
    _, new_models = compute_changes({}, {}, {}, {}, {}, {})
    assert not any(m["name"].startswith("cc/") for m in new_models)


def _cc_current(*names: str) -> dict[str, dict]:
    """A current MODEL_INFO snapshot containing the given ``cc/*`` entries."""
    return {
        name: {
            "context_length": 200000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
            "alias_of": None,
        }
        for name in names
    }


def test_cc_model_gone_from_anthropic_is_deprecated() -> None:
    """A dated ``cc/claude-*`` snapshot absent upstream is retired."""
    current = _cc_current("cc/claude-sonnet-4-5-20250929")
    anthropic = {"claude-opus-4-6": {"source": "anthropic"}}
    deprecated = find_deprecated_models(current, {}, anthropic, {}, {})
    assert {d["name"] for d in deprecated} == {"cc/claude-sonnet-4-5-20250929"}


def test_cc_undated_alias_kept_while_a_snapshot_exists() -> None:
    """``cc/claude-sonnet-4-5`` survives as long as a dated snapshot exists."""
    current = _cc_current("cc/claude-sonnet-4-5", "cc/claude-opus-4-5")
    anthropic = {"claude-sonnet-4-5-20250929": {"source": "anthropic"}}
    deprecated = find_deprecated_models(current, {}, anthropic, {}, {})
    assert {d["name"] for d in deprecated} == {"cc/claude-opus-4-5"}


def test_cc_short_aliases_are_never_deprecated() -> None:
    """haiku/opus/sonnet aliases stay even though the API never lists them."""
    current = _cc_current("cc/haiku", "cc/opus", "cc/sonnet")
    anthropic = {"claude-opus-4-6": {"source": "anthropic"}}
    deprecated = find_deprecated_models(current, {}, anthropic, {}, {})
    assert deprecated == []


def test_cc_entries_untouched_when_anthropic_fetch_is_empty() -> None:
    """A failed/skipped Anthropic fetch must not mass-deprecate cc/* models."""
    current = _cc_current("cc/opus", "cc/claude-opus-4-6")
    deprecated = find_deprecated_models(current, {}, {}, {}, {})
    assert deprecated == []

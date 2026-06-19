# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that ``update_models.py`` never auto-adds GPT ``-pro`` or ``-codex`` variants.

GPT ``-pro`` variants (``gpt-5-pro``, ``gpt-5.5-pro``, ...) reject the
``reasoning_effort`` parameter and are billed at premium rates we don't want
to surface as suggested defaults. GPT ``-codex`` variants (``gpt-5-codex``,
``gpt-5.1-codex-max``, ...) are intended to be driven through the Codex CLI
backend, not the bare Chat Completions API. In both cases we want the
auto-discovery flow in ``compute_changes`` / ``_add_codex_candidates`` to
silently skip those names so they are never appended to ``model_info.py``.

These tests drive the production ``compute_changes`` and
``_add_codex_candidates`` functions end-to-end with synthetic vendor
payloads, then assert on the resulting ``new_models`` list — the same list
that ``apply_updates_to_file`` would otherwise write into ``model_info.py``.
"""

from kiss.scripts.update_models import _add_codex_candidates, compute_changes


def _empty_current() -> dict[str, dict]:
    """A current MODEL_INFO snapshot with no relevant entries, so every probed
    name flows through the new-model path in ``compute_changes``.
    """
    return {}


def test_openai_gpt_pro_is_not_added_as_new_model() -> None:
    """``gpt-5-pro`` returned by the OpenAI list must not be proposed as new."""
    openai = {
        "gpt-5-pro": {"source": "openai"},
        "gpt-5-pro-2025-10-06": {"source": "openai"},
        "gpt-5.5-pro": {"source": "openai"},
        "gpt-5.5-pro-2026-04-23": {"source": "openai"},
    }
    _, new_models = compute_changes(_empty_current(), {}, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert "gpt-5-pro" not in names
    assert "gpt-5-pro-2025-10-06" not in names
    assert "gpt-5.5-pro" not in names
    assert "gpt-5.5-pro-2026-04-23" not in names


def test_openai_gpt_codex_is_not_added_as_new_model() -> None:
    """GPT ``-codex`` family variants must not be proposed as new."""
    openai = {
        "gpt-5-codex": {"source": "openai"},
        "gpt-5.1-codex": {"source": "openai"},
        "gpt-5.1-codex-max": {"source": "openai"},
        "gpt-5.1-codex-mini": {"source": "openai"},
        "gpt-5.3-codex": {"source": "openai"},
    }
    _, new_models = compute_changes(_empty_current(), {}, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openai))


def test_openrouter_openai_gpt_pro_passthrough_is_not_added() -> None:
    """OpenRouter passthroughs to ``gpt-*-pro`` must not be proposed as new."""
    openrouter = {
        "openrouter/openai/gpt-5-pro": {
            "context_length": 400000,
            "input_price_per_1M": 15.0,
            "output_price_per_1M": 120.0,
            "source": "openrouter",
        },
        "openrouter/openai/gpt-5.5-pro": {
            "context_length": 1050000,
            "input_price_per_1M": 30.0,
            "output_price_per_1M": 180.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, {})
    names = {m["name"] for m in new_models}
    assert "openrouter/openai/gpt-5-pro" not in names
    assert "openrouter/openai/gpt-5.5-pro" not in names


def test_openrouter_openai_gpt_codex_passthrough_is_not_added() -> None:
    """OpenRouter passthroughs to ``gpt-*-codex*`` must not be proposed as new."""
    openrouter = {
        "openrouter/openai/gpt-5-codex": {
            "context_length": 400000,
            "input_price_per_1M": 1.25,
            "output_price_per_1M": 10.0,
            "source": "openrouter",
        },
        "openrouter/openai/gpt-5.1-codex-max": {
            "context_length": 400000,
            "input_price_per_1M": 1.25,
            "output_price_per_1M": 10.0,
            "source": "openrouter",
        },
        "openrouter/openai/gpt-5.3-codex": {
            "context_length": 400000,
            "input_price_per_1M": 1.75,
            "output_price_per_1M": 14.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, {})
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openrouter))


def test_normal_gpt_model_is_still_added() -> None:
    """Sanity: a regular ``gpt-*`` (no -pro/-codex) is still proposed as new."""
    openai = {"gpt-99": {"source": "openai"}}
    openrouter = {
        "openrouter/openai/gpt-99": {
            "context_length": 1_050_000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert "gpt-99" in names
    assert "openrouter/openai/gpt-99" in names


def test_non_gpt_pro_models_are_unaffected() -> None:
    """A model whose name happens to contain ``-pro`` but isn't a GPT (e.g. a
    hypothetical Claude or fictional vendor) must still be addable — the
    exclusion is scoped to the GPT family only.
    """
    openrouter = {
        "openrouter/acme/super-pro": {
            "context_length": 128000,
            "input_price_per_1M": 1.0,
            "output_price_per_1M": 2.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, {})
    names = {m["name"] for m in new_models}
    assert "openrouter/acme/super-pro" in names


def test_add_codex_candidates_skips_pro_slug() -> None:
    """The Codex CLI ``models.json`` may list ``gpt-*-pro`` slugs; we must skip them."""
    current: dict[str, dict] = {}
    new_models: list[dict] = []
    _add_codex_candidates({"gpt-5.5-pro", "gpt-5.4-pro"}, current, {}, new_models)
    assert new_models == []


def test_add_codex_candidates_skips_codex_slug() -> None:
    """``gpt-*-codex*`` slugs from the Codex CLI list must not become ``codex/...``
    entries — they should be reached via their bare OpenAI name only.
    """
    current: dict[str, dict] = {}
    new_models: list[dict] = []
    _add_codex_candidates(
        {"gpt-5-codex", "gpt-5.1-codex-max", "gpt-5.3-codex"},
        current,
        {},
        new_models,
    )
    assert new_models == []


def test_add_codex_candidates_keeps_normal_gpt_slug() -> None:
    """A normal ``gpt-X`` slug in the Codex CLI list still produces a ``codex/...`` entry."""
    current: dict[str, dict] = {}
    new_models: list[dict] = []
    _add_codex_candidates({"gpt-5.5", "gpt-5.4-mini"}, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert names == {"codex/gpt-5.5", "codex/gpt-5.4-mini"}


def test_existing_gpt_pro_entry_still_receives_price_updates() -> None:
    """If a GPT pro entry already lives in ``MODEL_INFO``, the script must still
    propagate vendor pricing updates to it — the exclusion only blocks *new*
    additions, not edits to existing entries the maintainer chose to keep.
    """
    current = {
        "openrouter/openai/gpt-5-pro": {
            "context_length": 400000,
            "input_price_per_1M": 15.0,
            "output_price_per_1M": 120.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
        },
    }
    openrouter = {
        "openrouter/openai/gpt-5-pro": {
            "context_length": 400000,
            "input_price_per_1M": 16.0,
            "output_price_per_1M": 125.0,
            "source": "openrouter",
        },
    }
    updates, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    new_names = {m["name"] for m in new_models}
    assert "openrouter/openai/gpt-5-pro" not in new_names
    update_names = {u["name"] for u in updates}
    assert "openrouter/openai/gpt-5-pro" in update_names

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that ``update_models.py`` never auto-adds OpenAI ``o<n>-pro`` reasoning models.

OpenAI's ``o1-pro`` / ``o3-pro`` reasoning families are reachable only via
the ``/v1/responses`` endpoint. KISS's ``OpenAICompatibleModel`` invokes
``/v1/chat/completions``, so probing or running these models returns HTTP
404 — there is no way for the discovery flow to test them, and they would
fail at runtime for KISS users anyway. The auto-discovery flow in
``compute_changes`` / ``_add_codex_candidates`` must therefore silently
skip those names so they are never appended to ``MODEL_INFO.json``.

These tests drive the production ``compute_changes`` and
``_add_codex_candidates`` functions end-to-end with synthetic vendor
payloads, then assert on the resulting ``new_models`` list — the same
list that ``apply_updates_to_file`` would otherwise write into
``MODEL_INFO.json``.
"""

from kiss.scripts.update_models import (
    _add_codex_candidates,
    _is_excluded_openai_responses_only,
    compute_changes,
)


def _empty_current() -> dict[str, dict]:
    """A current MODEL_INFO snapshot with no relevant entries, so every probed
    name flows through the new-model path in ``compute_changes``.
    """
    return {}


def test_openai_o1_pro_family_is_not_added_as_new_model() -> None:
    """``o1-pro`` and dated snapshots from the OpenAI list must not be proposed."""
    openai = {
        "o1-pro": {"source": "openai"},
        "o1-pro-2025-03-19": {"source": "openai"},
    }
    _, new_models = compute_changes(_empty_current(), {}, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openai))


def test_openai_o3_pro_family_is_not_added_as_new_model() -> None:
    """``o3-pro`` and dated snapshots from the OpenAI list must not be proposed."""
    openai = {
        "o3-pro": {"source": "openai"},
        "o3-pro-2025-06-10": {"source": "openai"},
    }
    _, new_models = compute_changes(_empty_current(), {}, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openai))


def test_openrouter_openai_o_pro_passthrough_is_not_added() -> None:
    """OpenRouter passthroughs to ``o<n>-pro`` must not be proposed as new."""
    openrouter = {
        "openrouter/openai/o1-pro": {
            "context_length": 200000,
            "input_price_per_1M": 150.0,
            "output_price_per_1M": 600.0,
            "source": "openrouter",
        },
        "openrouter/openai/o3-pro-2025-06-10": {
            "context_length": 200000,
            "input_price_per_1M": 20.0,
            "output_price_per_1M": 80.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, {})
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openrouter))


def test_add_codex_candidates_skips_o_pro_slug() -> None:
    """The Codex CLI ``models.json`` may list ``o<n>-pro`` slugs; we must skip them."""
    current: dict[str, dict] = {}
    new_models: list[dict] = []
    _add_codex_candidates({"o1-pro", "o3-pro", "o3-pro-2025-06-10"}, current, {}, new_models)
    assert new_models == []


def test_normal_o_series_model_is_still_added() -> None:
    """Sanity: a non-pro ``o<n>`` reasoning model (e.g. ``o3-mini``) is still added."""
    openai = {"o3-mini": {"source": "openai"}, "o4-mini": {"source": "openai"}}
    openrouter = {
        "openrouter/openai/o3-mini": {
            "context_length": 200000,
            "input_price_per_1M": 1.1,
            "output_price_per_1M": 4.4,
            "source": "openrouter",
        },
        "openrouter/openai/o4-mini": {
            "context_length": 200000,
            "input_price_per_1M": 1.1,
            "output_price_per_1M": 4.4,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(_empty_current(), openrouter, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert "o3-mini" in names
    assert "o4-mini" in names
    assert "openrouter/openai/o3-mini" in names
    assert "openrouter/openai/o4-mini" in names


def test_helper_matches_bare_and_dated_and_passthrough() -> None:
    """Cross-form sanity check for the exclusion helper itself."""
    assert _is_excluded_openai_responses_only("o1-pro")
    assert _is_excluded_openai_responses_only("o1-pro-2025-03-19")
    assert _is_excluded_openai_responses_only("o3-pro")
    assert _is_excluded_openai_responses_only("o3-pro-2025-06-10")
    assert _is_excluded_openai_responses_only("openrouter/openai/o1-pro")
    assert _is_excluded_openai_responses_only("openrouter/openai/o3-pro")
    # not excluded:
    assert not _is_excluded_openai_responses_only("o3-mini")
    assert not _is_excluded_openai_responses_only("o4-mini")
    assert not _is_excluded_openai_responses_only("gpt-5-pro")
    assert not _is_excluded_openai_responses_only("openrouter/acme/super-pro")
    assert not _is_excluded_openai_responses_only("claude-opus-4-5")


def test_existing_o_pro_entry_still_receives_price_updates() -> None:
    """If an ``o<n>-pro`` entry already lives in ``MODEL_INFO``, the script must
    still propagate vendor pricing updates to it — the exclusion only blocks
    *new* additions, not edits to existing entries the maintainer chose to keep.
    """
    current = {
        "openrouter/openai/o1-pro": {
            "context_length": 200000,
            "input_price_per_1M": 150.0,
            "output_price_per_1M": 600.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "thinking": None,
        },
    }
    openrouter = {
        "openrouter/openai/o1-pro": {
            "context_length": 200000,
            "input_price_per_1M": 160.0,
            "output_price_per_1M": 620.0,
            "source": "openrouter",
        },
    }
    updates, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    new_names = {m["name"] for m in new_models}
    assert "openrouter/openai/o1-pro" not in new_names
    update_names = {u["name"] for u in updates}
    assert "openrouter/openai/o1-pro" in update_names

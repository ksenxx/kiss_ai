# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for model-name fast-completion helpers.

Exercise :func:`rank_model_suggestions` and
:func:`get_completion_model_names` against the real ``MODEL_INFO`` table so
the ``sorcar`` CLI's ``/model`` completion reflects the shipped models.
"""

from __future__ import annotations

from kiss.core.models.model_info import (
    MODEL_INFO,
    get_completion_model_names,
    get_generation_model_listing,
    get_model_provider,
    rank_model_suggestions,
)


def test_get_completion_model_names_nonempty_sorted_generation_only() -> None:
    """The candidate list is non-empty, sorted, and generation-capable."""
    names = get_completion_model_names()
    assert names, "completion model list must never be empty"
    assert names == sorted(names)
    for name in names:
        assert name in MODEL_INFO
        assert MODEL_INFO[name].is_generation_supported


def test_rank_empty_query_returns_all_names_unchanged() -> None:
    """An empty query preserves the candidate order (already sorted)."""
    names = ["b-model", "a-model"]
    assert rank_model_suggestions("", names) == ["b-model", "a-model"]
    assert rank_model_suggestions("   ", names) == ["b-model", "a-model"]


def test_rank_prefix_matches_before_substring_matches() -> None:
    """Prefix matches sort ahead of substring matches; each group sorted."""
    names = ["gpt-5", "gpt-4o", "openrouter/openai/gpt-4o"]
    assert rank_model_suggestions("gpt-4", names) == [
        "gpt-4o",
        "openrouter/openai/gpt-4o",
    ]


def test_rank_is_case_insensitive() -> None:
    """Matching ignores case for both the query and the candidate names."""
    names = ["Qwen/Qwen3.5-9B", "gpt-4o"]
    assert rank_model_suggestions("qwen/qwen3.5", names) == ["Qwen/Qwen3.5-9B"]
    assert rank_model_suggestions("QWEN", names) == ["Qwen/Qwen3.5-9B"]


def test_rank_substring_only_matches() -> None:
    """A query that only appears mid-name still matches as a substring."""
    names = ["claude-haiku-4-5", "openrouter/anthropic/claude-3.5-haiku", "gpt-4o"]
    assert rank_model_suggestions("haiku", names) == [
        "claude-haiku-4-5",
        "openrouter/anthropic/claude-3.5-haiku",
    ]


def test_rank_no_match_returns_empty() -> None:
    """A query that matches nothing returns an empty list."""
    assert rank_model_suggestions("zzz-no-such-model", ["gpt-4o"]) == []


def test_rank_defaults_to_completion_model_names() -> None:
    """With no explicit candidate list, ranking uses the live model table."""
    sample = get_completion_model_names()[0]
    ranked = rank_model_suggestions(sample[:3])
    assert sample in ranked
    assert set(ranked).issubset(set(get_completion_model_names()))


def test_get_model_provider_routes_known_prefixes() -> None:
    """Provider labels match the routing prefixes used by ``model()``."""
    assert get_model_provider("gpt-5.5") == "OpenAI"
    assert get_model_provider("o3") == "OpenAI"
    assert get_model_provider("claude-opus-4-8") == "Anthropic"
    assert get_model_provider("gemini-3.1-pro-preview") == "Gemini"
    assert get_model_provider("minimax-m2.5") == "MiniMax"
    assert get_model_provider("openrouter/openai/gpt-5.5") == "OpenRouter"
    assert get_model_provider("Qwen/Qwen3.5-9B") == "Together"
    assert get_model_provider("openai/gpt-oss-120b") == "Together"
    assert get_model_provider("cc/opus") == "Claude Code CLI"
    assert get_model_provider("codex/default") == "Codex CLI"


def test_get_model_provider_unknown_for_unrouted_name() -> None:
    """A name that matches no route is reported as ``Unknown``."""
    assert get_model_provider("totally-made-up-model") == "Unknown"


def test_generation_model_listing_sorted_generation_only() -> None:
    """The listing covers exactly the generation models, sorted by name."""
    listing = get_generation_model_listing()
    assert listing, "listing must never be empty"
    names = [name for name, _, _ in listing]
    assert names == sorted(names)
    listed = set(names)
    expected = {
        name for name, info in MODEL_INFO.items() if info.is_generation_supported
    }
    assert listed == expected


def test_generation_model_listing_provider_and_bool_status() -> None:
    """Each entry carries the correct provider label and a bool status."""
    for name, provider, configured in get_generation_model_listing():
        assert provider == get_model_provider(name)
        assert provider != "Unknown"
        assert isinstance(configured, bool)


def test_generation_model_listing_excludes_embeddings_and_images() -> None:
    """Embedding-only and image-only models never appear in the listing."""
    names = {name for name, _, _ in get_generation_model_listing()}
    assert "text-embedding-3-large" not in names  # embedding-only
    assert "gpt-image-1" in names  # gpt-image-* are generation-capable (gen=True)
    assert not MODEL_INFO["BAAI/bge-base-en-v1.5"].is_generation_supported
    assert "BAAI/bge-base-en-v1.5" not in names

"""Regression checks for the curated OpenAI model catalog."""

from kiss.core.models.model_info import MODEL_INFO


def test_openai_catalog_includes_current_ids():
    expected = {
        "chatgpt-4o-latest",
        "gpt-4-turbo-preview",
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-audio-1.5",
        "gpt-realtime-1.5",
        "o1-preview",
    }
    assert expected.issubset(MODEL_INFO.keys())


def test_openai_catalog_excludes_removed_ids():
    removed = {
        "gpt-4o-2024-05-13",
        "gpt-5-search-api",
        "o3-mini-high",
        "o4-mini-high",
    }
    assert removed.isdisjoint(MODEL_INFO.keys())

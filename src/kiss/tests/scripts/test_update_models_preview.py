# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for preview model handling in update_models.py compute_changes."""

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import pytest

import kiss.scripts.update_models as mod
from kiss.scripts.update_models import (
    _add_codex_candidates,
    _assume_untested_capabilities,
    apply_updates_to_file,
    compute_changes,
    fetch_anthropic,
    fetch_gemini,
    find_deprecated_models,
)


def _make_current() -> dict[str, dict]:
    """Minimal current MODEL_INFO for testing."""
    return {
        "openrouter/google/gemini-2.5-flash": {
            "context_length": 1048576,
            "input_price_per_1M": 0.30,
            "output_price_per_1M": 2.50,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }


def test_openrouter_preview_with_zero_pricing_is_added():
    """Free preview models from OpenRouter should be added."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/cool-model-preview": {
            "context_length": 128000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/cool-model-preview" in names
    model = next(m for m in new_models if m["name"] == "openrouter/acme/cool-model-preview")
    assert model["needs_pricing"] is True


def test_openrouter_variant_endpoints_still_skipped():
    """Variant endpoints (:free, :thinking) should still be skipped."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/model-preview:free": {
            "context_length": 128000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/model-preview:free" not in names


def test_together_preview_with_zero_pricing_is_added():
    """Free preview models from Together should be added."""
    current = _make_current()
    together = {
        "meta-llama/some-model-preview": {
            "context_length": 131072,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "together",
            "type": "chat",
            "is_embedding": False,
        },
    }
    _, new_models = compute_changes(current, {}, together, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "meta-llama/some-model-preview" in names
    model = next(m for m in new_models if m["name"] == "meta-llama/some-model-preview")
    assert model["needs_pricing"] is True


def test_together_non_preview_zero_pricing_is_not_added():
    """Non-preview Together models with zero pricing should be filtered out."""
    current = _make_current()
    together = {
        "meta-llama/some-free-model": {
            "context_length": 131072,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "together",
            "type": "chat",
            "is_embedding": False,
        },
    }
    _, new_models = compute_changes(current, {}, together, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "meta-llama/some-free-model" not in names


def test_gemini_preview_model_is_added():
    """Preview models from Gemini should be added (they always have needs_pricing)."""
    current = _make_current()
    gemini = {
        "gemini-99-flash-preview": {
            "context_length": 1048576,
            "source": "gemini",
            "is_embedding": False,
            "is_generation": True,
        },
    }
    _, new_models = compute_changes(current, {}, {}, gemini, {}, {})
    names = [m["name"] for m in new_models]
    assert "gemini-99-flash-preview" in names
    model = next(m for m in new_models if m["name"] == "gemini-99-flash-preview")
    assert model["needs_pricing"] is True


def test_existing_model_not_duplicated():
    """Models already in current should not be added as new."""
    current = _make_current()
    openrouter = {
        "openrouter/google/gemini-2.5-flash": {
            "context_length": 1048576,
            "input_price_per_1M": 0.30,
            "output_price_per_1M": 2.50,
            "source": "openrouter",
        },
    }
    updates, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    new_names = [m["name"] for m in new_models]
    assert "openrouter/google/gemini-2.5-flash" not in new_names
    assert len(updates) == 0


def test_codex_candidates_added_from_codex_models_json():
    """Only models in the Codex CLI models.json get codex/ entries."""
    current = _make_current()
    codex_slugs = {"gpt-5.5", "gpt-5.4", "gpt-5.4-mini"}
    _, new_models = compute_changes(
        current, {}, {}, {}, {}, {}, codex_slugs=codex_slugs
    )
    names = {m["name"] for m in new_models}
    for expected in (
        "codex/gpt-5.5",
        "codex/gpt-5.4",
        "codex/gpt-5.4-mini",
    ):
        assert expected in names
    assert "codex/gpt-5.5-pro" not in names
    for nm in new_models:
        if nm["name"].startswith("codex/"):
            assert nm["input_price_per_1M"] == 0.0
            assert nm["output_price_per_1M"] == 0.0
            assert nm["source"] == "codex"
            assert nm["needs_pricing"] is False
            assert nm["gen"] is True
            assert nm["fc"] is True
            assert nm["emb"] is False
            assert nm["context_length"] >= 400000


def test_codex_candidate_uses_openrouter_context_when_available():
    """Context length for codex/<name> should come from OpenRouter when present."""
    current: dict[str, dict] = {}
    codex_slugs = {"gpt-5.5"}
    openrouter = {
        "openrouter/openai/gpt-5.5": {
            "context_length": 900000,
            "input_price_per_1M": 5.0,
            "output_price_per_1M": 30.0,
            "source": "openrouter",
        },
    }
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, openrouter, new_models)
    [entry] = [m for m in new_models if m["name"] == "codex/gpt-5.5"]
    assert entry["context_length"] == 900000


def test_codex_candidate_only_adds_supported_slugs():
    """Only slugs from the Codex CLI models.json should get entries."""
    current: dict[str, dict] = {}
    codex_slugs = {"gpt-5.5", "gpt-5.4"}
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert names == {"codex/gpt-5.5", "codex/gpt-5.4"}


def test_codex_candidate_skips_unsupported_models():
    """Models not in the Codex models.json must not get codex/ entries."""
    current: dict[str, dict] = {}
    codex_slugs: set[str] = set()
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    assert new_models == []


def test_codex_candidate_not_added_when_already_present():
    """codex/<name> already in current MODEL_INFO must not be re-added."""
    current = {
        "codex/gpt-5.4": {
            "context_length": 400000,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    codex_slugs = {"gpt-5.4", "gpt-5.5"}
    new_models: list[dict] = []
    _add_codex_candidates(codex_slugs, current, {}, new_models)
    names = {m["name"] for m in new_models}
    assert "codex/gpt-5.4" not in names
    assert "codex/gpt-5.5" in names


def test_openrouter_preview_zero_context_not_added():
    """Preview models with zero context length should not be added."""
    current = _make_current()
    openrouter = {
        "openrouter/acme/cool-model-preview": {
            "context_length": 0,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
        },
    }
    _, new_models = compute_changes(current, openrouter, {}, {}, {}, {})
    names = [m["name"] for m in new_models]
    assert "openrouter/acme/cool-model-preview" not in names


def test_find_deprecated_codex_entries_with_slugs():
    """codex/* models not in the Codex CLI models.json should be deprecated.

    codex/default is always kept. When codex_slugs is provided, only models
    whose slug appears in the set are kept; all others are deprecated.
    When codex_slugs is None (fetch failed), no codex models are deprecated.
    """
    current = {
        "codex/default": {"source": "codex"},
        "codex/gpt-5.5": {"source": "codex"},
        "codex/gpt-5.5-pro": {"source": "codex"},
        "codex/gpt-5.4": {"source": "codex"},
        "gpt-foo-removed": {"source": "openai"},
    }
    codex_slugs = {"gpt-5.5", "gpt-5.4"}
    openai = {"gpt-5": {"source": "openai"}, "gpt-5-2025-08-07": {"source": "openai"}}
    deprecated = find_deprecated_models(
        current, {}, {}, {}, openai, codex_slugs=codex_slugs
    )
    names = {d["name"] for d in deprecated}
    assert "codex/default" not in names
    assert "codex/gpt-5.5" not in names
    assert "codex/gpt-5.4" not in names
    assert "codex/gpt-5.5-pro" in names


def test_find_deprecated_codex_entries_without_slugs():
    """When codex_slugs is None (fetch failed), no codex models are deprecated."""
    current = {
        "codex/default": {"source": "codex"},
        "codex/gpt-5.5": {"source": "codex"},
    }
    deprecated = find_deprecated_models(
        current, {}, {}, {}, {}, codex_slugs=None
    )
    names = {d["name"] for d in deprecated}
    assert "codex/default" not in names
    assert "codex/gpt-5.5" not in names


def _gemini_model(model_id: str, *methods: str) -> dict[str, Any]:
    return {
        "name": f"models/{model_id}",
        "inputTokenLimit": 8192,
        "supportedGenerationMethods": list(methods),
    }


class _PagedListingHandler(BaseHTTPRequestHandler):
    """Serve two-page Gemini (``/gemini``) and Anthropic (``/anthropic``) listings.

    Page two of each vendor is only reachable through the cursor page one
    returned (Gemini's ``nextPageToken`` as ``pageToken``, Anthropic's
    ``last_id`` as ``after_id``) and answers HTTP 500 when
    ``fail_second_page`` is set.  Every request's path and Anthropic key
    header are recorded.
    """

    fail_second_page = False
    requests: list[str] = []
    anthropic_keys: list[str | None] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        cls = _PagedListingHandler
        cls.requests.append(self.path)
        query = parse_qs(urlsplit(self.path).query)
        if self.path.startswith("/gemini"):
            if "pageToken" in query:
                assert query["pageToken"] == ["page two/tok+en"]
                second = True
                listing: dict[str, Any] = {
                    "models": [_gemini_model("gemini-2.5-flash", "generateContent")]
                }
            else:
                second = False
                listing = {
                    "models": [
                        _gemini_model("gemini-3.8-flash", "generateContent"),
                        _gemini_model("gemini-embedding-001", "embedContent"),
                    ],
                    "nextPageToken": "page two/tok+en",
                }
        else:
            cls.anthropic_keys.append(self.headers.get("x-api-key"))
            if "after_id" in query:
                assert query["after_id"] == ["claude-opus-4-6"]
                second = True
                listing = {"data": [{"id": "claude-sonnet-4-5-20250929"}], "has_more": False}
            else:
                second = False
                listing = {
                    "data": [{"id": "claude-opus-4-6"}],
                    "has_more": True,
                    "last_id": "claude-opus-4-6",
                }
        status = 500 if second and cls.fail_second_page else 200
        payload = json.dumps(listing).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _paged_listings(
    monkeypatch: pytest.MonkeyPatch, fail_second_page: bool = False
) -> Iterator[type[_PagedListingHandler]]:
    """Point both vendor listing URLs at a local two-page server."""
    _PagedListingHandler.fail_second_page = fail_second_page
    _PagedListingHandler.requests = []
    _PagedListingHandler.anthropic_keys = []
    server = HTTPServer(("127.0.0.1", 0), _PagedListingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(mod, "GEMINI_MODELS_URL", f"{base}/gemini")
    monkeypatch.setattr(mod, "ANTHROPIC_MODELS_URL", f"{base}/anthropic")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    try:
        yield _PagedListingHandler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _later_page_entries() -> dict[str, dict]:
    """Catalog entries that only page two of the listings vouches for."""
    entry = {
        "context_length": 8192,
        "input_price_per_1M": 1.0,
        "output_price_per_1M": 2.0,
        "fc": True,
        "emb": False,
        "gen": True,
    }
    return {
        name: dict(entry)
        for name in (
            "claude-sonnet-4-5-20250929",
            "cc/claude-sonnet-4-5-20250929",
            "gemini-2.5-flash",
        )
    }


def test_gemini_and_anthropic_listings_follow_every_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both fetchers walk the cursor chain, so page-two models are neither missed nor retired."""
    with _paged_listings(monkeypatch) as handler:
        gemini = fetch_gemini()
        anthropic = fetch_anthropic()
    assert set(gemini) == {"gemini-3.8-flash", "gemini-embedding-001", "gemini-2.5-flash"}
    assert gemini["gemini-embedding-001"] == {
        "context_length": 8192,
        "source": "gemini",
        "is_embedding": True,
        "is_generation": False,
    }
    assert set(anthropic) == {"claude-opus-4-6", "claude-sonnet-4-5-20250929"}
    # One request per page, the Gemini key on every page, the Anthropic
    # auth header repeated on the cursor page.
    assert len(handler.requests) == 4
    assert all("key=gemini-key" in p for p in handler.requests if p.startswith("/gemini"))
    assert handler.anthropic_keys == ["anthropic-key", "anthropic-key"]
    assert find_deprecated_models(_later_page_entries(), {}, anthropic, gemini, {}) == []


def test_skip_test_records_gemini_embedding_as_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Gemini ``embedContent``-only model is catalogued as an embedder, not a tool caller."""
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text("{}\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    with _paged_listings(monkeypatch):
        gemini = fetch_gemini()
    _, candidates = compute_changes({}, {}, {}, gemini, {}, {})
    by_name = {c["name"]: c for c in candidates}
    assert by_name["gemini-embedding-001"]["is_embedding"] is True
    assert by_name["gemini-embedding-001"]["is_generation"] is False
    assert by_name["gemini-3.8-flash"]["is_generation"] is True

    kept = _assume_untested_capabilities(candidates)
    apply_updates_to_file([], kept, [], {}, dry_run=False)
    written = json.loads(catalog.read_text())
    assert (
        written["gemini-embedding-001"]["gen"],
        written["gemini-embedding-001"]["emb"],
        written["gemini-embedding-001"]["fc"],
    ) == (False, True, False)
    assert (
        written["gemini-3.8-flash"]["gen"],
        written["gemini-3.8-flash"]["emb"],
        written["gemini-3.8-flash"]["fc"],
    ) == (True, False, True)


def test_failed_later_page_publishes_no_partial_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A page-two failure raises instead of returning page one as the whole listing."""
    monkeypatch.setattr(time, "sleep", lambda seconds: None)  # skip api_get's retry backoff
    with _paged_listings(monkeypatch, fail_second_page=True):
        with pytest.raises(HTTPError):
            fetch_gemini()
        with pytest.raises(HTTPError):
            fetch_anthropic()
    # Nothing was returned, so nothing can be handed to find_deprecated_models;
    # an empty listing retires no one.
    assert find_deprecated_models(_later_page_entries(), {}, {}, {}, {}) == []

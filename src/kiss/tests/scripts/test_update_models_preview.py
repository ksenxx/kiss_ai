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
from kiss.core.config import DEFAULT_CONFIG
from kiss.scripts.update_models import (
    _add_codex_candidates,
    _assume_untested_capabilities,
    apply_updates_to_file,
    compute_changes,
    fetch_anthropic,
    fetch_bedrock,
    fetch_gemini,
    fetch_openai,
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

    ``/openai`` is OpenAI's one-page listing of ids.

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
        elif self.path.startswith("/openai"):
            second = False
            listing = {"data": [{"id": "gpt-7"}, {"id": "text-embedding-4-large"}]}
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
def _listing_server(
    monkeypatch: pytest.MonkeyPatch, handler: type[BaseHTTPRequestHandler]
) -> Iterator[str]:
    """Serve *handler* on loopback as the vendors' id listings; yield its base URL."""
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(mod, "GEMINI_MODELS_URL", f"{base}/gemini")
    monkeypatch.setattr(mod, "ANTHROPIC_MODELS_URL", f"{base}/anthropic")
    monkeypatch.setattr(mod, "OPENAI_MODELS_URL", f"{base}/openai")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextmanager
def _paged_listings(
    monkeypatch: pytest.MonkeyPatch, fail_second_page: bool = False
) -> Iterator[type[_PagedListingHandler]]:
    """Point both vendor listing URLs at a local two-page server."""
    _PagedListingHandler.fail_second_page = fail_second_page
    _PagedListingHandler.requests = []
    _PagedListingHandler.anthropic_keys = []
    with _listing_server(monkeypatch, _PagedListingHandler):
        yield _PagedListingHandler


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
    """A Gemini ``embedContent``-only model is catalogued as an embedder, not a tool caller.

    So is an OpenAI ``text-embedding-*`` id, which the listing of ids left unmarked.
    """
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text("{}\n")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    with _paged_listings(monkeypatch):
        gemini = fetch_gemini()
        openai = fetch_openai()
    _, candidates = compute_changes({}, {}, {}, gemini, {}, openai)
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
    flags = {name: tuple(written[name][flag] for flag in ("gen", "emb", "fc")) for name in openai}
    assert flags == {"gpt-7": (True, False, True), "text-embedding-4-large": (False, True, False)}


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


class _EndlessListingHandler(BaseHTTPRequestHandler):
    """Serve the pages of a Gemini, Anthropic or Bedrock listing that goes on.

    One body holds every vendor's fields.  Page *n* of a vendor (``pages``
    counts them by the first path segment) names ``cursors[n]`` as the
    cursor of the next page; past the end of ``cursors`` the listing is
    over, so a fetcher that follows whatever it is handed returns (and
    fails its test) instead of hanging.  Each answer waits ``delay`` seconds.
    """

    cursors: list[str | None] = []
    delay = 0.0
    pages: dict[str, int] = {}

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        cls = _EndlessListingHandler
        vendor = urlsplit(self.path).path.split("/")[1]
        page = cls.pages.get(vendor, 0)
        cls.pages[vendor] = page + 1
        more = page < len(cls.cursors)
        cursor = cls.cursors[page] if more else None
        time.sleep(cls.delay)
        profile = {"inferenceProfileId": "us.anthropic.claude-opus-4-6-v1"}
        listing = {
            "models": [_gemini_model("gemini-3.8-flash", "generateContent")],
            "nextPageToken": cursor,
            "data": [{"id": "claude-opus-4-6"}],
            "has_more": more,
            "last_id": cursor,
            "modelSummaries": [],
            "inferenceProfileSummaries": [profile],
            "nextToken": cursor,
        }
        payload = json.dumps(listing).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _endless_listings(
    monkeypatch: pytest.MonkeyPatch, cursors: list[str | None], delay: float = 0.0
) -> Iterator[type[_EndlessListingHandler]]:
    """Point the Gemini, Anthropic and Bedrock listings at a local server handing out *cursors*."""
    _EndlessListingHandler.cursors = cursors
    _EndlessListingHandler.delay = delay
    _EndlessListingHandler.pages = {}
    with _listing_server(monkeypatch, _EndlessListingHandler) as base:
        monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", base)
        key_pair = {
            "AWS_BEARER_TOKEN_BEDROCK": "",
            "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
            "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AWS_SESSION_TOKEN": "",
            "AWS_REGION": "us-east-1",
        }
        for field, value in key_pair.items():
            monkeypatch.setattr(DEFAULT_CONFIG, field, value)
        yield _EndlessListingHandler


def test_a_listing_that_repeats_its_cursor_is_abandoned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit SC1: a vendor that answered a cursor with that cursor was asked for the page forever.

    Each fetcher asks for the repeated page once and gives up with nothing returned (Bedrock's
    with its warning): a partial listing would retire the models on the pages not read.
    """
    with _endless_listings(monkeypatch, ["again"] * 8) as handler:
        for fetch in (fetch_gemini, fetch_anthropic):
            with pytest.raises(RuntimeError, match="repeats its page cursor 'again'"):
                fetch()
        assert fetch_bedrock() == {}
    warning = "Bedrock listing failed (the listing repeats its page cursor 'again')"
    assert warning in capsys.readouterr().out
    assert [handler.pages[v] for v in ("gemini", "anthropic", "inference-profiles")] == [2, 2, 2]


def test_a_listing_that_never_ends_meets_its_deadline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Audit SC1: fresh cursors without end kept the updater reading; the whole listing is timed.

    So is Bedrock's, whose SDK calls were waited for however long one answer took: the listing
    is given up at the deadline and retires nothing.
    """
    monkeypatch.setattr(mod, "_PAGED_LISTING_DEADLINE_SECONDS", 0.2)
    with _endless_listings(monkeypatch, [str(n) for n in range(40)], delay=0.05) as handler:
        with pytest.raises(TimeoutError, match="still has pages at its deadline"):
            fetch_gemini()
    assert handler.pages["gemini"] < 40
    with _endless_listings(monkeypatch, [], delay=1.0) as handler:
        assert fetch_bedrock() == {}
        # The call left behind still reads: the server stays until it has asked for its last page.
        patience = time.monotonic() + 60
        while "inference-profiles" not in handler.pages and time.monotonic() < patience:
            time.sleep(0.01)
    assert handler.pages == {"foundation-models": 1, "inference-profiles": 1}
    assert "Bedrock listing failed (_bedrock_listings did not finish within 0.2s)" in (
        capsys.readouterr().out
    )


def test_anthropic_listing_with_more_pages_and_no_cursor_is_abandoned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``has_more`` without a ``last_id`` ended the listing at the pages read: a partial one."""
    with _endless_listings(monkeypatch, [None]):
        with pytest.raises(RuntimeError, match="names no cursor"):
            fetch_anthropic()

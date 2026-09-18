# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: update_models discovers, probes and records decisions models.

OpenRouter lists ``text->decisions`` models (TypeSafe Jev) only under
``/api/v1/models?output_modalities=decisions``.  These tests serve both
listings from a local HTTP server and drive the real fetch → compute →
apply → README-sync pipeline, checking that such a model is fetched with
``decisions=True``, probed with the decisions probe instead of the text
probes, written with ``"dec": true`` and never mistaken for a deprecated
model.  The live probe itself runs against OpenRouter when
``OPENROUTER_API_KEY`` is set.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

import kiss.scripts.update_models as mod
from kiss.scripts.update_models import (
    _assume_untested_capabilities,
    _build_entry,
    _probe_new_models,
    apply_updates_to_file,
    compute_changes,
    fetch_openrouter,
    find_deprecated_models,
    get_current_model_info,
    sync_readme_catalog,
)

# Imported under non-test names so pytest does not collect the script's
# ``test_*`` probe helpers as test functions.
probe_decisions = mod.test_decisions
probe_capabilities = mod.test_model_capabilities

_REPO_ROOT = Path(__file__).resolve().parents[4]
_README = _REPO_ROOT / "README.md"

JEV = "openrouter/~typesafe/jev-latest"
JEV_PINNED = "openrouter/typesafe/jev-1.13"


def _or_model(model_id: str, prompt: str, completion: str, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_id,
        "context_length": 32000,
        "pricing": {"prompt": prompt, "completion": completion},
        "expiration_date": None,
    }
    entry.update(extra)
    return entry


TEXT_LISTING = {
    "data": [
        _or_model("openai/gpt-4o", "0.0000025", "0.00001"),
        _or_model("minimax/minimax-m3", "0.000001", "0.000002"),
        {"id": ""},
    ]
}
DECISIONS_LISTING = {
    "data": [
        _or_model("~typesafe/jev-latest", "0.000000042", "0"),
        _or_model("typesafe/jev-1.13", "0.000000042", "0"),
        _or_model("typesafe/jev-0.9", "0.000000042", "0", expiration_date="2025-01-01"),
    ]
}


class _ModelsListingHandler(BaseHTTPRequestHandler):
    """Serve ``/api/v1/models`` and its ``?output_modalities=decisions`` variant."""

    text_listing: dict[str, Any] = TEXT_LISTING

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if "output_modalities=decisions" in self.path:
            listing = DECISIONS_LISTING
        else:
            listing = _ModelsListingHandler.text_listing
        payload = json.dumps(listing).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _listing_server(text_listing: dict[str, Any] = TEXT_LISTING) -> Iterator[str]:
    _ModelsListingHandler.text_listing = text_listing
    server = HTTPServer(("127.0.0.1", 0), _ModelsListingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1/models"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _point_fetch_at(monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    monkeypatch.setattr(mod, "OPENROUTER_MODELS_URL", url)
    monkeypatch.setattr(
        mod, "OPENROUTER_DECISIONS_MODELS_URL", f"{url}?output_modalities=decisions"
    )


def _fetch_from_local_listing(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    with _listing_server() as url:
        _point_fetch_at(monkeypatch, url)
        return fetch_openrouter(verbose=True)


def test_fetch_openrouter_merges_decisions_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both listings merge; decisions entries are flagged, expired/excluded/id-less ones dropped."""
    fetched = _fetch_from_local_listing(monkeypatch)
    assert set(fetched) == {"openrouter/openai/gpt-4o", JEV, JEV_PINNED}
    assert fetched["openrouter/openai/gpt-4o"]["decisions"] is False
    assert fetched[JEV] == {
        "context_length": 32000,
        "input_price_per_1M": 0.042,
        "output_price_per_1M": 0.0,
        "source": "openrouter",
        "decisions": True,
    }
    assert fetched[JEV_PINNED]["decisions"] is True


def test_compute_changes_marks_new_decisions_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """An uncatalogued decisions model becomes a priced candidate carrying ``is_decisions``."""
    fetched = _fetch_from_local_listing(monkeypatch)
    current = {
        "openrouter/openai/gpt-4o": {
            "context_length": 32000,
            "input_price_per_1M": 2.5,
            "output_price_per_1M": 10.0,
            "fc": True,
            "emb": False,
            "gen": True,
            "dec": False,
        },
        JEV_PINNED: {
            "context_length": 32000,
            "input_price_per_1M": 0.03,
            "output_price_per_1M": 0.0,
            "fc": False,
            "emb": False,
            "gen": False,
            "dec": True,
        },
    }
    updates, new_models = compute_changes(current, fetched, {}, {}, {}, {})
    [candidate] = new_models
    assert candidate["name"] == JEV
    assert candidate["is_decisions"] is True
    assert candidate["needs_pricing"] is False
    assert candidate["input_price_per_1M"] == 0.042
    [update] = updates
    assert update["name"] == JEV_PINNED
    assert update["changes"] == {"input_price_per_1M": 0.042}


def test_catalogued_decisions_model_is_not_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Merging the decisions listing keeps a catalogued Jev off the deprecation list."""
    fetched = _fetch_from_local_listing(monkeypatch)
    current = {
        JEV: {"context_length": 32000, "input_price_per_1M": 0.042, "output_price_per_1M": 0.0},
        "openrouter/typesafe/jev-0.9": {
            "context_length": 32000,
            "input_price_per_1M": 0.042,
            "output_price_per_1M": 0.0,
        },
    }
    deprecated = find_deprecated_models(current, fetched, {}, {}, {})
    assert deprecated == [
        {"name": "openrouter/typesafe/jev-0.9", "reason": "not in OpenRouter API"}
    ]


def test_build_entry_writes_dec_only_when_set() -> None:
    """``dec`` is a compact optional key, like ``use_responses_api``."""
    plain = _build_entry(ctx=1, inp=1.0, out=2.0)
    assert "dec" not in plain
    decisions = _build_entry(ctx=32000, inp=0.042, out=0.0, fc=False, gen=False, dec=True)
    assert decisions == {
        "context_length": 32000,
        "input_price_per_1M": 0.042,
        "output_price_per_1M": 0.0,
        "fc": False,
        "emb": False,
        "gen": False,
        "dec": True,
    }


def test_apply_and_readme_sync_record_decisions_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probed decisions model is written with ``dec: true`` and counted in the README totals."""
    catalog = tmp_path / "MODEL_INFO.json"
    catalog.write_text(
        json.dumps(
            {
                "openrouter/openai/gpt-4o": {
                    "context_length": 128000,
                    "input_price_per_1M": 2.5,
                    "output_price_per_1M": 10.0,
                    "fc": True,
                    "emb": False,
                    "gen": True,
                }
            }
        )
    )
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    # ``current`` knows a pinned Jev that is missing from disk, so the
    # update path must rebuild it from ``current`` and keep its ``dec``.
    current = {
        JEV_PINNED: {
            "context_length": 32000,
            "input_price_per_1M": 0.03,
            "output_price_per_1M": 0.0,
            "fc": False,
            "emb": False,
            "gen": False,
            "dec": True,
            "thinking": None,
            "use_responses_api": None,
        }
    }
    new_models = [
        {
            "name": JEV,
            "context_length": 32000,
            "input_price_per_1M": 0.042,
            "output_price_per_1M": 0.0,
            "source": "openrouter",
            "needs_pricing": False,
            "is_decisions": True,
            "gen": False,
            "emb": False,
            "fc": False,
            "dec": True,
            "thinking": None,
            "use_responses_api": False,
        }
    ]
    updates = [
        {"name": JEV_PINNED, "changes": {"input_price_per_1M": 0.042}, "source": "openrouter"}
    ]
    apply_updates_to_file(updates, new_models, [], current)
    written = json.loads(catalog.read_text())
    assert written[JEV] == {
        "context_length": 32000,
        "input_price_per_1M": 0.042,
        "output_price_per_1M": 0.0,
        "fc": False,
        "emb": False,
        "gen": False,
        "dec": True,
        "comment": "NEW",
    }
    assert written[JEV_PINNED]["dec"] is True
    assert written[JEV_PINNED]["input_price_per_1M"] == 0.042
    assert "dec" not in written["openrouter/openai/gpt-4o"]

    readme = tmp_path / "README.md"
    shutil.copy(_README, readme)
    assert sync_readme_catalog(readme, catalog) is True
    text = readme.read_text()
    assert "- **1** generation-capable models" in text
    assert "- **0** embedding models" in text
    assert "- **2** decision models" in text
    assert "| OpenRouter | 3 |" in text
    assert f"- `{JEV}`\n" in text


def test_skip_test_assumes_flags_from_listing() -> None:
    """--skip-test derives gen/emb/fc/dec from the listing markers without probing."""
    candidates: list[dict[str, Any]] = [
        {"name": JEV, "is_decisions": True},
        {"name": "BAAI/bge-large-en-v1.5", "is_embedding": True},
        {"name": "openrouter/openai/gpt-4o"},
    ]
    _assume_untested_capabilities(candidates)
    flags = [(c["gen"], c["emb"], c["fc"], c["dec"]) for c in candidates]
    assert flags == [
        (False, False, False, True),
        (False, True, True, False),
        (True, False, True, False),
    ]
    assert all(c["thinking"] is None and c["use_responses_api"] is False for c in candidates)


def test_current_model_info_exposes_dec_flag() -> None:
    """The snapshot the script diffs against carries ``dec`` for both Jev entries."""
    current = get_current_model_info()
    assert current[JEV]["dec"] is True
    assert current[JEV_PINNED]["dec"] is True
    assert current["openrouter/openai/gpt-4o"]["dec"] is False
    assert current[JEV]["gen"] is False


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_decisions_probe_drives_capabilities(capsys: pytest.CaptureFixture[str]) -> None:
    """The decisions probe passes against OpenRouter and short-circuits the text probes."""
    caps = probe_capabilities(JEV_PINNED, verbose=True, decisions=True)
    assert caps == {
        "gen": False,
        "emb": False,
        "fc": False,
        "thinking": None,
        "use_responses_api": None,
        "dec": True,
    }
    assert "dec=Y" in capsys.readouterr().out


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_decisions_probe_fails_closed_on_bad_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint error (here HTTP 401) is a failed probe, not an exception."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid")
    assert probe_decisions(JEV) is False


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_live_probe_new_models_keeps_decisions_and_drops_dead_candidates() -> None:
    """The new-model gate keeps Jev, skips CLI candidates, and drops one that answers nothing."""
    candidates: list[dict[str, Any]] = [
        {"name": JEV_PINNED, "is_decisions": True},
        {"name": "cc/opus"},
        {"name": "openrouter/typesafe/does-not-exist-0.0"},
    ]
    kept = _probe_new_models(candidates)
    assert [c["name"] for c in kept] == [JEV_PINNED, "cc/opus"]
    assert kept[0]["dec"] is True
    assert (kept[0]["gen"], kept[0]["emb"], kept[0]["fc"]) == (False, False, False)
    assert kept[0]["use_responses_api"] is False
    assert "gen" not in kept[1]
    assert candidates[2]["_skip"] is True


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_main_test_existing_drops_dec_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--test-existing`` re-probes catalogued decisions models; a failed probe clears ``dec``.

    The probe really goes to OpenRouter, with a deliberately invalid key so
    it fails (HTTP 401).  Only the two Jev entries are in the catalog, so
    no text model is probed; the listing comes from the local server.
    """
    jev_entry = {
        "context_length": 32000,
        "input_price_per_1M": 0.042,
        "output_price_per_1M": 0.0,
        "fc": False,
        "emb": False,
        "gen": False,
        "dec": True,
    }
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps({JEV: jev_entry, JEV_PINNED: jev_entry}))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    for fetcher in ("fetch_together", "fetch_anthropic", "fetch_gemini", "fetch_openai"):
        monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    snapshot = {**jev_entry, "thinking": None, "alias_of": None, "use_responses_api": None}
    monkeypatch.setattr(
        mod, "get_current_model_info", lambda: {JEV: dict(snapshot), JEV_PINNED: dict(snapshot)}
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid")
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing", "--verbose"])
    with _listing_server(text_listing={"data": []}) as url:
        _point_fetch_at(monkeypatch, url)
        mod.main()
    out = capsys.readouterr().out
    assert f"{JEV}: dec changed True -> False" in out
    assert f"{JEV_PINNED}: dec changed True -> False" in out
    assert "dec=N" in out
    written = json.loads(target.read_text())
    assert set(written) == {JEV, JEV_PINNED}
    assert "dec" not in written[JEV] and "dec" not in written[JEV_PINNED]
    assert written[JEV]["gen"] is False

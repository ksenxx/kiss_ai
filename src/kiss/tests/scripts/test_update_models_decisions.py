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
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

import kiss.scripts.update_models as mod
from kiss.core.kiss_error import KISSError
from kiss.core.models.decisions_model import DecisionsModel, noul
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
_MODELS = _REPO_ROOT / "MODELS.md"

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
    decisions_listing: dict[str, Any] = DECISIONS_LISTING

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        if "output_modalities=decisions" in self.path:
            listing = _ModelsListingHandler.decisions_listing
        else:
            listing = _ModelsListingHandler.text_listing
        payload = json.dumps(listing).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _listing_server(
    text_listing: dict[str, Any] = TEXT_LISTING,
    decisions_listing: dict[str, Any] = DECISIONS_LISTING,
) -> Iterator[str]:
    _ModelsListingHandler.text_listing = text_listing
    _ModelsListingHandler.decisions_listing = decisions_listing
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
        )
    )
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    # The pinned Jev is repriced in place and must keep its ``dec``.  (It
    # used to be absent from disk and rebuilt from ``current``; an update
    # whose entry an existing catalog lacks is now skipped as stale.)
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

    readme = tmp_path / "MODELS.md"
    shutil.copy(_MODELS, readme)
    assert sync_readme_catalog(readme, catalog) is True
    text = readme.read_text(encoding="utf-8")  # MODELS.md is UTF-8, not the Windows code page
    assert "- **1** generation-capable models" in text
    assert "- **0** embedding models" in text
    assert "- **2** decision models" in text
    assert "| OpenRouter | 3 |" in text
    assert f"- `{JEV}`\n" in text


def test_skip_test_assumes_flags_from_listing() -> None:
    """--skip-test derives gen/emb/fc/dec from the listing markers without probing.

    Only a candidate assumed to generate text is assumed to call tools:
    embedding-only (Together or Gemini) and decisions-only candidates get
    ``fc=False``, and a Gemini candidate advertising neither
    ``generateContent`` nor ``embedContent`` is dropped like a candidate
    that fails every live probe.
    """
    candidates: list[dict[str, Any]] = [
        {"name": JEV, "is_decisions": True},
        {"name": "BAAI/bge-large-en-v1.5", "is_embedding": True},
        {"name": "openrouter/openai/gpt-4o"},
        {"name": "gemini-embedding-001", "is_embedding": True, "is_generation": False},
        {"name": "gemini-9-flash", "is_embedding": False, "is_generation": True},
        {"name": "gemini-9-flash-live", "is_embedding": False, "is_generation": False},
    ]
    kept = _assume_untested_capabilities(candidates)
    flags = [(c["gen"], c["emb"], c["fc"], c["dec"]) for c in candidates]
    assert flags == [
        (False, False, False, True),
        (False, True, False, False),
        (True, False, True, False),
        (False, True, False, False),
        (True, False, True, False),
        (False, False, False, False),
    ]
    assert all(c["thinking"] is None and c["use_responses_api"] is False for c in candidates)
    assert [c["name"] for c in kept] == [c["name"] for c in candidates[:-1]]


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
def test_live_decisions_probe_tells_a_verdict_from_endpoint_trouble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown model (HTTP 400) is a failed probe; a rejected key (HTTP 401) is no verdict."""
    assert probe_decisions("openrouter/typesafe/does-not-exist-0.0") is False
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid")
    assert probe_decisions(JEV) is None


def test_a_refused_connection_is_no_verdict_on_a_decisions_model() -> None:
    """The adapter wraps httpx's connection error in a ``KISSError``; the probe looks through it.

    The endpoint is a free loopback port, so the real transport is
    refused at once: no HTTP status, no vendor reply, nothing about the
    model -- ``dec`` must keep its stored value.  (A port bound but not
    listening is refused on Linux only; macOS drops the SYN and the
    connect times out.)
    """
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    down = DecisionsModel(JEV, base_url=f"http://127.0.0.1:{port}", api_key="k")
    with pytest.raises(KISSError) as caught:
        down.decide("The parcel arrived crushed.", {"damaged": noul("Was it damaged?")})
    assert isinstance(caught.value.__cause__, httpx.ConnectError)
    assert mod._is_inconclusive_probe_error(caught.value) is True


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
    assert (candidates[2]["gen"], candidates[2]["emb"], candidates[2]["dec"]) == (False,) * 3


_JEV_ENTRY = {
    "context_length": 32000,
    "input_price_per_1M": 0.042,
    "output_price_per_1M": 0.0,
    "fc": False,
    "emb": False,
    "gen": False,
    "dec": True,
}


def _retest_decisions_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, names: tuple[str, ...]
) -> Path:
    """Run ``--test-existing --verbose`` on a catalog of decisions entries; return its path.

    Only decisions entries are in the catalog, so no text model is probed;
    the local listing server lists exactly those entries (so none is new
    or retired), the probe itself goes wherever ``OPENROUTER_API_KEY``
    lets it.
    """
    target = tmp_path / "MODEL_INFO.json"
    target.write_text(json.dumps(dict.fromkeys(names, _JEV_ENTRY)))
    listing = {
        "data": [_or_model(name.removeprefix("openrouter/"), "0.000000042", "0") for name in names]
    }
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    monkeypatch.setattr(mod, "README_PATH", tmp_path / "README.md")
    for fetcher in (
        "fetch_together",
        "fetch_anthropic",
        "fetch_gemini",
        "fetch_openai",
        "fetch_bedrock",
    ):
        monkeypatch.setattr(mod, fetcher, lambda verbose=False: {})
    monkeypatch.setattr(mod, "fetch_codex_supported_slugs", lambda verbose=False: set())
    snapshot = {**_JEV_ENTRY, "thinking": None, "alias_of": None, "use_responses_api": None}
    monkeypatch.setattr(
        mod, "get_current_model_info", lambda: {name: dict(snapshot) for name in names}
    )
    monkeypatch.setattr(sys, "argv", ["update_models.py", "--test-existing", "--verbose"])
    with _listing_server(text_listing={"data": []}, decisions_listing=listing) as url:
        _point_fetch_at(monkeypatch, url)
        mod.main()
    return target


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_main_test_existing_drops_dec_when_probe_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--test-existing`` clears ``dec`` from an entry OpenRouter rejects for good (HTTP 400)."""
    gone = "openrouter/typesafe/does-not-exist-0.0"
    target = _retest_decisions_catalog(tmp_path, monkeypatch, (gone,))
    out = capsys.readouterr().out
    assert f"{gone}: dec changed True -> False" in out
    assert "dec=N" in out
    written = json.loads(target.read_text())
    assert set(written) == {gone}
    assert "dec" not in written[gone]
    assert written[gone]["gen"] is False


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set")
def test_main_test_existing_keeps_dec_when_the_key_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A rejected key (HTTP 401) says nothing about the models: ``dec`` survives the retest."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-invalid")
    target = _retest_decisions_catalog(tmp_path, monkeypatch, (JEV, JEV_PINNED))
    out = capsys.readouterr().out
    assert f"{JEV}: probe inconclusive, stored capabilities kept" in out
    assert "dec=?" in out
    assert "changed" not in out
    written = json.loads(target.read_text())
    assert written[JEV]["dec"] is True and written[JEV_PINNED]["dec"] is True


def test_main_test_existing_keeps_dec_without_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no OpenRouter key configured the decisions probe never runs and ``dec`` is kept."""
    from kiss.core import config as config_module

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
    target = _retest_decisions_catalog(tmp_path, monkeypatch, (JEV,))
    out = capsys.readouterr().out
    assert f"{JEV}: probe inconclusive, stored capabilities kept" in out
    assert "changed" not in out
    assert json.loads(target.read_text())[JEV]["dec"] is True


def test_a_model_without_a_key_is_reported_untested_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audit SC1: with no key to send no probe runs, yet the pauses between the probes were
    slept all the same: a second per model, minutes for a listing of another vendor's models."""
    from kiss.core import config as config_module
    from kiss.core.models import model_info  # noqa: F401 - loaded before the clock starts

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "OPENROUTER_API_KEY", "")
    started = time.monotonic()
    caps = mod.test_model_capabilities("openrouter/openai/gpt-5.5")
    assert time.monotonic() - started < 0.9
    assert caps == {
        "dec": False,
        "gen": None,
        "emb": None,
        "fc": None,
        "thinking": mod.UNVERIFIED,
        "use_responses_api": None,
    }

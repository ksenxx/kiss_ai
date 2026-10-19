# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that ``update_models.py`` never lists an OpenAI ``-pro`` model.

OpenAI's ``o<n>-pro`` reasoning family (``o1-pro``, ``o3-pro``) answers only
on ``/v1/responses`` and the GPT ``-pro`` tier is billed at premium rates and
rejects ``reasoning_effort``.  The catalog must not carry any of them, on any
route: the updater drops them from the OpenRouter listing as it is fetched,
never proposes them as new candidates, flags a stale entry as deprecated even
while the vendor still lists it, and ``--scrub-only`` purges them offline.

The tests drive the production ``fetch_openrouter`` (against a local HTTP
listing), ``compute_changes``, ``_add_codex_candidates`` and
``find_deprecated_models`` functions, plus the real script as a subprocess
for ``--scrub-only``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from kiss.scripts import update_models as mod
from kiss.scripts.update_models import (
    _add_codex_candidates,
    _is_excluded_gpt_codex,
    _is_excluded_model,
    _is_excluded_openai_pro,
    compute_changes,
    fetch_openrouter,
    find_deprecated_models,
)

_REPO = Path(__file__).resolve().parents[4]
_SCRIPT = _REPO / "src" / "kiss" / "scripts" / "update_models.py"

_PRO_NAMES = (
    "o1-pro",
    "o1-pro-2025-03-19",
    "o3-pro",
    "o3-pro-2025-06-10",
    "gpt-5-pro",
    "gpt-5.5-pro-2026-04-23",
    "openrouter/openai/o1-pro",
    "openrouter/openai/o3-pro",
    "openrouter/openai/gpt-5.5-pro",
    "openrouter/openai/o3-pro:nitro",
    "openrouter/openai/gpt-5-pro:floor",
    "codex/gpt-5-pro",
)

_NON_PRO_NAMES = (
    "o3-mini",
    "o4-mini",
    "gpt-5.5",
    "gpt-5.5:nitro",
    "gpt-prose",
    "gpt-5-codex",
    "codex/gpt-5.3-codex",
    "gemini-2.5-pro",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/acme/super-pro",
    "claude-opus-4-5",
)


def _or_entry(inp: float = 20.0, out: float = 80.0) -> dict[str, Any]:
    return {
        "context_length": 200000,
        "input_price_per_1M": inp,
        "output_price_per_1M": out,
        "source": "openrouter",
    }


def _direct_entry(inp: float, out: float) -> dict[str, Any]:
    return {"context_length": 200000, "input_price_per_1M": inp, "output_price_per_1M": out}


def test_openai_pro_predicate_matches_every_route() -> None:
    """Bare, dated, OpenRouter and Codex spellings of OpenAI pro models are excluded."""
    for name in _PRO_NAMES:
        assert _is_excluded_openai_pro(name), name
        assert _is_excluded_model(name), name
    for name in _NON_PRO_NAMES:
        assert not _is_excluded_openai_pro(name), name
        assert not _is_excluded_model(name), name
    assert _is_excluded_model("openrouter/minimax/minimax-m2")


def test_gpt_codex_predicate_is_scoped_to_gpt_codex_slugs() -> None:
    """``-codex`` is a separate auto-add filter and does not touch pro or plain names."""
    for name in ("gpt-5-codex", "gpt-5.1-codex-max", "codex/gpt-5.3-codex", "gpt-5-codex:nitro"):
        assert _is_excluded_gpt_codex(name), name
    for name in ("gpt-5-pro", "o3-pro", "codex/default", "codex/codex-auto-review", "gpt-5.5"):
        assert not _is_excluded_gpt_codex(name), name


def test_pro_models_are_not_added_as_new_models() -> None:
    """Pro names from the OpenAI and OpenRouter listings are never proposed as new."""
    openai = {n: {"source": "openai"} for n in ("o1-pro", "o3-pro-2025-06-10", "gpt-5-pro")}
    openrouter = {
        "openrouter/openai/o1-pro": _or_entry(150.0, 600.0),
        "openrouter/openai/o3-pro": _or_entry(),
        "openrouter/openai/gpt-5.5-pro": _or_entry(30.0, 180.0),
    }
    _, new_models = compute_changes({}, openrouter, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    assert names.isdisjoint(set(openai) | set(openrouter))


def test_add_codex_candidates_skips_pro_slugs() -> None:
    """The Codex CLI ``models.json`` may list pro slugs; none become ``codex/...``."""
    new_models: list[dict] = []
    _add_codex_candidates({"o1-pro", "o3-pro-2025-06-10", "gpt-5.5-pro"}, {}, {}, new_models)
    assert new_models == []


def test_normal_o_series_model_is_still_added() -> None:
    """A non-pro ``o<n>`` reasoning model is still proposed as new."""
    openai = {"o3-mini": {"source": "openai"}, "o4-mini": {"source": "openai"}}
    openrouter = {
        "openrouter/openai/o3-mini": _or_entry(1.1, 4.4),
        "openrouter/openai/o4-mini": _or_entry(1.1, 4.4),
    }
    _, new_models = compute_changes({}, openrouter, {}, {}, {}, openai)
    names = {m["name"] for m in new_models}
    expected = {"o3-mini", "o4-mini", "openrouter/openai/o3-mini", "openrouter/openai/o4-mini"}
    assert expected <= names


def test_existing_pro_entries_are_flagged_deprecated() -> None:
    """Stale pro entries are removed even while the vendor still lists them."""
    current = {
        "openrouter/openai/o3-pro": {**_or_entry(), "fc": True, "emb": False, "gen": True},
        "o3-pro": _direct_entry(20.0, 80.0),
        "gpt-5-pro": _direct_entry(15.0, 120.0),
        "openrouter/openai/o3-mini": {**_or_entry(1.1, 4.4), "fc": True, "emb": False, "gen": True},
    }
    openrouter = {
        "openrouter/openai/o3-pro": _or_entry(),
        "openrouter/openai/o3-mini": _or_entry(1.1, 4.4),
    }
    openai = {"o3-pro": {"source": "openai"}, "gpt-5-pro": {"source": "openai"}}
    deprecated = find_deprecated_models(current, openrouter, {}, {}, openai)
    assert deprecated == [
        {"name": "openrouter/openai/o3-pro", "reason": "excluded model"},
        {"name": "o3-pro", "reason": "excluded model"},
        {"name": "gpt-5-pro", "reason": "excluded model"},
    ]


def test_update_for_deprecated_pro_entry_does_not_resurrect_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A removal wins over an update of the same name within one run.

    The context-repair loop and ``--test-existing`` can both schedule an
    update for an entry that ``find_deprecated_models`` flagged; the writer
    must not rebuild the entry from the update.
    """
    target = tmp_path / "MODEL_INFO.json"
    current = {
        "gpt-5-pro": {**_direct_entry(15.0, 120.0), "context_length": 0, "fc": True},
        "o3-mini": {**_direct_entry(1.1, 4.4), "fc": True},
    }
    target.write_text(json.dumps(current), encoding="utf-8")
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", target)
    deprecated = find_deprecated_models(current, {}, {}, {}, {"o3-mini": {"source": "openai"}})
    assert deprecated == [{"name": "gpt-5-pro", "reason": "excluded model"}]
    updates = [
        {"name": "gpt-5-pro", "changes": {"context_length": 400000}, "source": "openai"},
        {"name": "o3-mini", "changes": {"input_price_per_1M": 1.2}, "source": "openrouter"},
    ]
    mod.apply_updates_to_file(updates, [], deprecated, current, dry_run=False)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "gpt-5-pro" not in data
    assert data["o3-mini"]["input_price_per_1M"] == 1.2


def _listing_entry(model_id: str, ctx: int, prompt: str, completion: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "context_length": ctx,
        "pricing": {"prompt": prompt, "completion": completion},
    }


_LISTING: dict[str, Any] = {
    "data": [
        _listing_entry("openai/o3-pro", 200000, "0.00002", "0.00008"),
        _listing_entry("openai/gpt-5.5-pro", 400000, "0.00003", "0.00018"),
        _listing_entry("openai/o3-mini", 200000, "0.0000011", "0.0000044"),
        _listing_entry("google/gemini-2.5-pro", 1048576, "0.00000125", "0.00001"),
    ]
}


class _ListingHandler(BaseHTTPRequestHandler):
    """Serve the small text listing; the decisions listing is empty."""

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002, D102
        return

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        listing = {"data": []} if "output_modalities=decisions" in self.path else _LISTING
        payload = json.dumps(listing).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@contextmanager
def _listing_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _ListingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/api/v1/models"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_fetch_openrouter_drops_pro_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI pro entries never make it out of ``fetch_openrouter``."""
    with _listing_server() as url:
        monkeypatch.setattr(mod, "OPENROUTER_MODELS_URL", url)
        monkeypatch.setattr(
            mod, "OPENROUTER_DECISIONS_MODELS_URL", f"{url}?output_modalities=decisions"
        )
        fetched = fetch_openrouter(verbose=True)
    assert set(fetched) == {"openrouter/openai/o3-mini", "openrouter/google/gemini-2.5-pro"}


def test_scrub_only_purges_pro_entries(tmp_path: Path) -> None:
    """``--scrub-only`` removes pro entries offline and keeps everything else."""
    target = tmp_path / "MODEL_INFO.json"
    catalog = {
        "openrouter/openai/o3-pro": _or_entry(),
        "openrouter/openai/o3-pro:nitro": _or_entry(),
        "gpt-5-pro": _direct_entry(15.0, 120.0),
        "openrouter/google/gemini-2.5-pro": _or_entry(1.25, 10.0),
        "o3-mini": _direct_entry(1.1, 4.4),
    }
    target.write_text(json.dumps(catalog), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--scrub-only", "--model-info", str(target)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(_REPO),
    )
    assert proc.returncode == 0, proc.stderr
    assert "3 excluded entries to remove" in proc.stdout
    data = json.loads(target.read_text(encoding="utf-8"))
    assert set(data) == {"openrouter/google/gemini-2.5-pro", "o3-mini"}


def test_bundled_catalog_has_no_openai_pro_entry() -> None:
    """The shipped ``MODEL_INFO.json`` carries no OpenAI pro model on any route."""
    bundled_path = _REPO / "src" / "kiss" / "core" / "models" / "MODEL_INFO.json"
    bundled = json.loads(bundled_path.read_text(encoding="utf-8"))
    assert [name for name in bundled if _is_excluded_openai_pro(name)] == []

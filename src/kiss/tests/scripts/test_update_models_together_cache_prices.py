# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``update_models.py`` carries Together's published cached-input prices.

Together bills the ``prompt_tokens_details.cached_tokens`` it reports at
``pricing.cached_input`` (per 1M; Kimi-K3 $0.30 on a $3.00 input, see
https://www.together.ai/pricing).  A catalog entry without
``cache_read_price_per_1M`` makes ``calculate_cost`` bill those hits at
the full input rate, ten times Together's charge on Kimi-K3.  These
tests run ``fetch_together`` → ``compute_changes`` →
``apply_updates_to_file`` against an in-process listing and a temp
catalog, then load the result the way the runtime does.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

import kiss.scripts.update_models as mod
from kiss.core.models import model_info

LISTING: list[dict[str, Any]] = [
    {
        "id": "moonshotai/Kimi-K3",
        "type": "chat",
        "context_length": 262144,
        "pricing": {"input": 3, "output": 15, "cached_input": 0.3, "hourly": 0},
    },
    {
        "id": "deepseek-ai/DeepSeek-V4-Pro-0813",
        "type": "chat",
        "context_length": 131072,
        "pricing": {"input": 1.32, "output": 3.96, "cached_input": 0.12999999999999998},
    },
    {
        # No cached-input price: the entry keeps falling back to the input rate.
        "id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "type": "chat",
        "context_length": 131072,
        "pricing": {"input": 0.88, "output": 0.88, "cached_input": 0},
    },
    {
        # Brand-new model born with its cache price.
        "id": "Qwen/Qwen3.9-Preview",
        "type": "chat",
        "context_length": 65536,
        "pricing": {"input": 2.0, "output": 6.0, "cached_input": 0.25},
    },
]


class _TogetherListingHandler(BaseHTTPRequestHandler):
    """Serve the Together ``/v1/models`` shape: a bare JSON list."""

    def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        body = json.dumps(LISTING).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output quiet."""


@contextmanager
def _listing_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _TogetherListingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/models"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _fetch(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    monkeypatch.setenv("TOGETHER_API_KEY", "test-key")
    with _listing_server() as url:
        monkeypatch.setattr(mod, "TOGETHER_MODELS_URL", url)
        return mod.fetch_together(verbose=True)


def test_fetch_together_carries_cached_input_as_cache_read_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fetched = _fetch(monkeypatch)
    assert fetched["moonshotai/Kimi-K3"]["cache_read_price_per_1M"] == 0.3
    assert fetched["deepseek-ai/DeepSeek-V4-Pro-0813"]["cache_read_price_per_1M"] == 0.13
    assert fetched["meta-llama/Llama-3.3-70B-Instruct-Turbo"]["cache_read_price_per_1M"] is None
    assert fetched["Qwen/Qwen3.9-Preview"]["cache_read_price_per_1M"] == 0.25


def test_fetch_together_without_a_key_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "")
    assert mod.fetch_together() == {}


def test_compute_and_apply_write_together_cache_prices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "MODEL_INFO.json"
    current = {
        # Prices already right, cache read missing: gains exactly that field.
        "moonshotai/Kimi-K3": {
            "context_length": 262144,
            "input_price_per_1M": 3.0,
            "output_price_per_1M": 15.0,
            "fc": True,
            "emb": False,
            "gen": True,
        },
        # Stale cache read from an earlier listing: refreshed.
        "deepseek-ai/DeepSeek-V4-Pro-0813": {
            "context_length": 131072,
            "input_price_per_1M": 1.32,
            "output_price_per_1M": 3.96,
            "cache_read_price_per_1M": 0.2,
            "fc": True,
            "emb": False,
            "gen": True,
        },
        # Nothing published, nothing stored: no update.
        "meta-llama/Llama-3.3-70B-Instruct-Turbo": {
            "context_length": 131072,
            "input_price_per_1M": 0.88,
            "output_price_per_1M": 0.88,
            "fc": True,
            "emb": False,
            "gen": True,
        },
    }
    catalog.write_text(json.dumps(current))
    monkeypatch.setattr(mod, "MODEL_INFO_PATH", catalog)
    fetched = _fetch(monkeypatch)

    updates, new_models = mod.compute_changes(current, {}, fetched, {}, {}, {})
    by_name = {u["name"]: u["changes"] for u in updates}
    assert by_name == {
        "moonshotai/Kimi-K3": {"cache_read_price_per_1M": 0.3},
        "deepseek-ai/DeepSeek-V4-Pro-0813": {"cache_read_price_per_1M": 0.13},
    }
    [qwen] = new_models
    assert qwen["name"] == "Qwen/Qwen3.9-Preview"
    assert qwen["cache_read_price_per_1M"] == 0.25

    mod.apply_updates_to_file(updates, new_models, [], current, dry_run=False)
    written = json.loads(catalog.read_text())
    assert written["moonshotai/Kimi-K3"]["cache_read_price_per_1M"] == 0.3
    assert written["deepseek-ai/DeepSeek-V4-Pro-0813"]["cache_read_price_per_1M"] == 0.13
    assert "cache_read_price_per_1M" not in written["meta-llama/Llama-3.3-70B-Instruct-Turbo"]
    assert written["Qwen/Qwen3.9-Preview"]["cache_read_price_per_1M"] == 0.25

    # Loaded the way the runtime loads it, a Kimi-K3 cache hit costs $0.30/M.
    monkeypatch.setenv("KISS_MODEL_INFO_PATH", str(catalog))
    monkeypatch.setenv("HOME", str(tmp_path))
    loaded = model_info._load_model_info()
    assert loaded["moonshotai/Kimi-K3"].cache_read_price_per_1M == pytest.approx(0.3)
    assert loaded["meta-llama/Llama-3.3-70B-Instruct-Turbo"].cache_read_price_per_1M is None
    assert loaded["Qwen/Qwen3.9-Preview"].cache_read_price_per_1M == pytest.approx(0.25)

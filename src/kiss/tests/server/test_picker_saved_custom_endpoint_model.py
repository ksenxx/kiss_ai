# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A saved catalog-named model served from a custom endpoint stays selectable.

``MY_MODELS.json`` may override a bundled catalog name (say ``gpt-6-astra``)
with the user's own endpoint and credentials, with no vendor API key
configured.  ``VSCodeServer._get_models`` lists such entries through the
custom-model merge, which skips names the catalog already reported.  The
"keep the saved model visible" step must therefore not pre-list the name as
an unavailable catalog entry ("configure the vendor's API key"), or the
working custom entry is shadowed by a disabled placeholder.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.core import config as config_module
from kiss.core.models.model_info import MODEL_INFO

_KEY_NAMES = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "ZAI_API_KEY",
    "MOONSHOT_API_KEY",
)


@pytest.fixture
def no_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """No vendor API key, no CLI on PATH, so no catalog model is runnable."""
    for name in _KEY_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("KISS_MODEL", raising=False)
    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr(config_module, "DEFAULT_CONFIG", config_module.Config())
    yield
    config_module.DEFAULT_CONFIG = config_module.Config()


def test_saved_custom_endpoint_model_is_listed_as_the_custom_entry(
    no_keys: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import kiss.server.server as server_module

    name = next(
        n for n, info in MODEL_INFO.items()
        if info.is_generation_supported and not n.startswith(("cc/", "codex/"))
    )
    entry = {
        "name": name,
        "endpoint": "https://proxy.example.test/v1",
        "api_key": "proxy-key",
        "headers": "",
    }
    monkeypatch.setattr(server_module, "list_custom_models", lambda: [entry])
    monkeypatch.setattr(server_module, "_load_last_model", lambda: "")
    monkeypatch.setattr(server_module, "_save_last_model", lambda *_a, **_k: None, raising=False)
    monkeypatch.setenv("KISS_WORKDIR", "/tmp")

    server = server_module.VSCodeServer()
    server._default_model = name
    events: list[dict[str, Any]] = []
    lock = threading.Lock()

    def capture(event: dict[str, Any]) -> None:
        with lock:
            events.append(dict(event))

    monkeypatch.setattr(server.printer, "broadcast", capture)

    server._get_models()

    with lock:
        models_event = [e for e in events if e.get("type") == "models"][-1]
    rows = [m for m in models_event["models"] if m["name"] == name]
    assert len(rows) == 1, rows
    assert rows[0].get("endpoint") == entry["endpoint"], rows[0]
    assert rows[0].get("available", True) is True, rows[0]
    assert "unavailable_reason" not in rows[0], rows[0]
    assert models_event["selected"] == name
    assert os.environ.get("PATH") == ""

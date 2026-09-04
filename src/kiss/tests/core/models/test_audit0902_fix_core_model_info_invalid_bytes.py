# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 fix round (core-models, review #5): a ``MY_MODELS.json``
holding invalid UTF-8 bytes must not crash the ``model_info`` import.

``_read_my_models`` promises that an unreadable or corrupt user file is
ignored, but only ``OSError`` (around the read) and ``json.JSONDecodeError``
(around the parse) were caught.  ``Path.read_text(encoding="utf-8")`` raises
``UnicodeDecodeError`` — a ``ValueError``, not an ``OSError`` — on invalid
bytes, so the exception escaped through ``MODEL_INFO`` construction and
every ``import kiss.core.models.model_info`` failed.

The test imports the module in a fresh interpreter whose ``HOME`` points at
a temp dir (``USER_MY_MODELS_PATH`` is ``Path.home() / ".kiss" /
"MY_MODELS.json"``), so import-time behaviour is what is exercised.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

_IMPORT_SCRIPT = """
import json
from kiss.core.models import model_info as mi
print(json.dumps({
    "path": str(mi.USER_MY_MODELS_PATH),
    "n_models": len(mi.MODEL_INFO),
    "my_models": mi._read_my_models(),
}))
"""


def _import_in_fresh_process(home: Path) -> dict[str, Any]:
    """Import ``kiss.core.models.model_info`` in a subprocess with ``HOME=home``."""
    env = {**os.environ, "HOME": str(home), "KISS_HOME": str(home / ".kiss")}
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        f"import kiss.core.models.model_info crashed:\n{proc.stderr[-2000:]}"
    )
    result: dict[str, object] = json.loads(proc.stdout.strip().splitlines()[-1])
    return result


def test_invalid_utf8_my_models_is_ignored_at_import(tmp_path: Path) -> None:
    home = tmp_path / "home"
    my_models = home / ".kiss" / "MY_MODELS.json"
    my_models.parent.mkdir(parents=True)
    my_models.write_bytes(b'{"junk-model": {"context_length": 1}}\xff\xfe\x80')
    assert not my_models.read_bytes().decode("utf-8", errors="ignore").startswith("\ufeff")

    result = _import_in_fresh_process(home)

    assert result["path"] == str(my_models)
    assert result["my_models"] == {}, "an undecodable user file must contribute no models"
    assert isinstance(result["n_models"], int) and result["n_models"] > 0
    assert "junk-model" not in result["my_models"]


def test_valid_my_models_still_loads_at_import(tmp_path: Path) -> None:
    """Control: with valid bytes the user entry is picked up by the same path."""
    home = tmp_path / "home"
    my_models = home / ".kiss" / "MY_MODELS.json"
    my_models.parent.mkdir(parents=True)
    my_models.write_text(
        json.dumps(
            {
                "audit0902-user-model": {
                    "context_length": 1000,
                    "input_price_per_1M": 0.0,
                    "output_price_per_1M": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    result = _import_in_fresh_process(home)
    assert "audit0902-user-model" in result["my_models"]  # type: ignore[operator]


def test_top_level_array_my_models_is_ignored_at_import(tmp_path: Path) -> None:
    """Valid JSON whose top level is not an object contributes no models."""
    home = tmp_path / "home"
    my_models = home / ".kiss" / "MY_MODELS.json"
    my_models.parent.mkdir(parents=True)
    my_models.write_text(json.dumps([{"context_length": 1}]), encoding="utf-8")
    result = _import_in_fresh_process(home)
    assert result["my_models"] == {}

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the MODEL_INFO.json loader in ``kiss.core.models.model_info``.

The loader's contract:

* It treats ``~/.kiss/MODEL_INFO.json`` as the runtime source of truth.
* When the user-local copy is missing OR older than the package copy, it
  is refreshed from ``PACKAGE_MODEL_INFO_PATH`` (the file bundled with
  this module).
* The resulting ``MODEL_INFO`` is a ``dict[str, ModelInfo]`` whose
  cache pricing has been populated by ``_apply_cache_pricing``.
* When ``~/.kiss/`` is not writable the loader gracefully falls back to
  the package copy without raising.

These tests drive the production helpers directly so the contract above
is locked down end-to-end without mocks or doubles.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from kiss.core.models import model_info as mi_module
from kiss.core.models.model_info import (
    PACKAGE_MODEL_INFO_PATH,
    USER_MODEL_INFO_PATH,
    ModelInfo,
    _build_model_info_entry,
    _ensure_user_model_info_path,
    _load_model_info,
)


def test_package_model_info_json_is_present_and_well_formed() -> None:
    """The bundled MODEL_INFO.json must be valid JSON listing many models."""
    assert PACKAGE_MODEL_INFO_PATH.exists()
    raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert isinstance(raw, dict)
    assert len(raw) > 100, "expected at least 100 model entries in MODEL_INFO.json"
    # Every entry must have the three required numeric fields.
    for name, entry in raw.items():
        assert "context_length" in entry, name
        assert "input_price_per_1M" in entry, name
        assert "output_price_per_1M" in entry, name


def test_model_info_module_loads_from_package_when_user_copy_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing USER_MODEL_INFO_PATH at a fresh location triggers a copy."""
    fake_user = tmp_path / "kiss" / "MODEL_INFO.json"
    monkeypatch.setattr(mi_module, "USER_MODEL_INFO_PATH", fake_user)
    assert not fake_user.exists()
    result = _ensure_user_model_info_path()
    assert result == fake_user
    assert fake_user.exists()
    # Content must be identical to the package copy after the seed.
    assert json.loads(fake_user.read_text()) == json.loads(
        PACKAGE_MODEL_INFO_PATH.read_text()
    )


def test_loader_refreshes_user_copy_when_package_is_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale user copy must be overwritten when the package is newer."""
    fake_user = tmp_path / "kiss" / "MODEL_INFO.json"
    fake_user.parent.mkdir(parents=True, exist_ok=True)
    fake_user.write_text(json.dumps({"stale-only-model": {
        "context_length": 1,
        "input_price_per_1M": 0.0,
        "output_price_per_1M": 0.0,
    }}))
    # Make user copy older than the package copy.
    old = time.time() - 3600
    os.utime(fake_user, (old, old))
    monkeypatch.setattr(mi_module, "USER_MODEL_INFO_PATH", fake_user)
    _ensure_user_model_info_path()
    # After ensure, the user copy must match the package copy again.
    assert json.loads(fake_user.read_text()) == json.loads(
        PACKAGE_MODEL_INFO_PATH.read_text()
    )


def test_loader_keeps_user_copy_when_user_is_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the user copy is fresher, it must be preserved verbatim."""
    fake_user = tmp_path / "kiss" / "MODEL_INFO.json"
    fake_user.parent.mkdir(parents=True, exist_ok=True)
    custom = {
        "my-custom-model": {
            "context_length": 1234,
            "input_price_per_1M": 0.5,
            "output_price_per_1M": 1.0,
        }
    }
    fake_user.write_text(json.dumps(custom))
    future = time.time() + 3600
    os.utime(fake_user, (future, future))
    monkeypatch.setattr(mi_module, "USER_MODEL_INFO_PATH", fake_user)
    _ensure_user_model_info_path()
    assert json.loads(fake_user.read_text()) == custom


def test_loader_falls_back_to_package_when_user_dir_unwritable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only filesystem must not break import — we fall back to package."""
    # Point user path at a path under a file (so mkdir/copy fails).
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    fake_user = blocker / "MODEL_INFO.json"
    monkeypatch.setattr(mi_module, "USER_MODEL_INFO_PATH", fake_user)
    result = _ensure_user_model_info_path()
    assert result == PACKAGE_MODEL_INFO_PATH


def test_build_model_info_entry_handles_all_optional_fields() -> None:
    """The entry builder must respect every documented optional key."""
    entry = {
        "context_length": 1000,
        "input_price_per_1M": 1.0,
        "output_price_per_1M": 2.0,
        "fc": False,
        "emb": True,
        "gen": False,
        "thinking": "xhigh",
        "cache_read_price_per_1M": 0.25,
        "cache_write_price_per_1M": 1.25,
        "cache_write_1h_price_per_1M": 2.5,
        "comment": "ignored",
    }
    info = _build_model_info_entry(entry)
    assert isinstance(info, ModelInfo)
    assert info.context_length == 1000
    assert info.input_price_per_1M == 1.0
    assert info.output_price_per_1M == 2.0
    assert info.is_function_calling_supported is False
    assert info.is_embedding_supported is True
    assert info.is_generation_supported is False
    assert info.thinking == "xhigh"
    assert info.cache_read_price_per_1M == 0.25
    assert info.cache_write_price_per_1M == 1.25
    assert info.cache_write_1h_price_per_1M == 2.5


def test_build_model_info_entry_defaults() -> None:
    """Required-only entries must get the documented default flags."""
    info = _build_model_info_entry({
        "context_length": 100,
        "input_price_per_1M": 0.1,
        "output_price_per_1M": 0.2,
    })
    assert info.is_function_calling_supported is True
    assert info.is_embedding_supported is False
    assert info.is_generation_supported is True
    assert info.thinking is None
    assert info.cache_read_price_per_1M is None


def test_load_model_info_returns_modelinfo_objects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The high-level loader must return ModelInfo objects, not dicts."""
    fake_user = tmp_path / "kiss" / "MODEL_INFO.json"
    monkeypatch.setattr(mi_module, "USER_MODEL_INFO_PATH", fake_user)
    data = _load_model_info()
    assert isinstance(data, dict)
    assert data
    sample_key = next(iter(data))
    assert isinstance(data[sample_key], ModelInfo)


def test_user_model_info_path_lives_under_home_kiss() -> None:
    """The runtime source of truth must live at ~/.kiss/MODEL_INFO.json."""
    assert USER_MODEL_INFO_PATH == Path.home() / ".kiss" / "MODEL_INFO.json"

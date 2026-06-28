# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the MODEL_INFO loader in ``kiss.core.models.model_info``.

New contract (after dropping ``~/.kiss/MODEL_INFO.json``):

* The bundled ``PACKAGE_MODEL_INFO_PATH`` is the read-only source of
  truth for shipped models — never copied into ``~/.kiss/``.
* The optional ``~/.kiss/MY_MODELS.json`` holds user-curated model
  overrides and extensions.  It is auto-seeded with a short
  documentation block + a commented-out example entry on first read.
* Entries in MY_MODELS.json whose key is also present in the bundled
  table OVERRIDE the bundled entry; new keys are ADDED.  Keys
  starting with ``_`` are treated as documentation/comments and are
  silently skipped.
* When ``~/.kiss/`` is not writable, MY_MODELS.json is silently
  ignored — the loader still returns the bundled table.
* Corrupt or unreadable MY_MODELS.json files are also silently
  ignored (the loader must never raise during import).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kiss.core.models import model_info as mi_module
from kiss.core.models.model_info import (
    MY_MODELS_DEFAULT_CONTENT,
    PACKAGE_MODEL_INFO_PATH,
    USER_MY_MODELS_PATH,
    ModelInfo,
    _build_model_info_entry,
    _load_model_info,
    _read_my_models,
)


def test_package_model_info_json_is_present_and_well_formed() -> None:
    """The bundled MODEL_INFO.json must be valid JSON listing many models."""
    assert PACKAGE_MODEL_INFO_PATH.exists()
    raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert isinstance(raw, dict)
    assert len(raw) > 100
    for name, entry in raw.items():
        assert "context_length" in entry, name
        assert "input_price_per_1M" in entry, name
        assert "output_price_per_1M" in entry, name


def test_user_my_models_path_lives_under_home_kiss() -> None:
    """The optional user overrides file must live at ~/.kiss/MY_MODELS.json."""
    assert USER_MY_MODELS_PATH == Path.home() / ".kiss" / "MY_MODELS.json"


def test_my_models_default_content_is_valid_json_with_docs_and_example() -> None:
    """The auto-seeded default content must be valid JSON containing docs + example."""
    parsed = json.loads(MY_MODELS_DEFAULT_CONTENT)
    assert isinstance(parsed, dict)
    # At least one "documentation" key starting with '_' must be present.
    doc_keys = [k for k in parsed if k.startswith("_") and "doc" in k.lower()]
    assert doc_keys, "expected a '_documentation' (or similar '_doc') key"
    # At least one example entry (key starting with '_') with a model-shaped
    # payload must be present so users can see the schema.
    example_entries = [
        v for k, v in parsed.items()
        if k.startswith("_") and isinstance(v, dict)
        and "context_length" in v
        and "input_price_per_1M" in v
        and "output_price_per_1M" in v
    ]
    assert example_entries, (
        "expected at least one example entry (key starting with '_') "
        "with the full model schema"
    )


def test_loader_does_not_create_legacy_model_info_in_user_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader must no longer write ``~/.kiss/MODEL_INFO.json``."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setattr(
        mi_module, "USER_MY_MODELS_PATH", fake_home / ".kiss" / "MY_MODELS.json",
    )
    _load_model_info()
    assert not (fake_home / ".kiss" / "MODEL_INFO.json").exists()


def test_loader_auto_seeds_my_models_on_first_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """First call must create ``~/.kiss/MY_MODELS.json`` with the default content."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    assert not fake_path.exists()
    _load_model_info()
    assert fake_path.exists()
    assert fake_path.read_text() == MY_MODELS_DEFAULT_CONTENT


def test_loader_preserves_existing_my_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing MY_MODELS.json must not be clobbered by the seeder."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    custom = json.dumps({
        "my-custom-model": {
            "context_length": 1234,
            "input_price_per_1M": 0.5,
            "output_price_per_1M": 1.0,
        },
    })
    fake_path.write_text(custom)
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    _load_model_info()
    assert fake_path.read_text() == custom


def test_my_models_overrides_bundled_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A MY_MODELS entry whose key matches a bundled model must override pricing."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    bundled = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert "claude-opus-4-7" in bundled, "test assumes a stable bundled name"
    override_ctx = 999_999
    fake_path.write_text(json.dumps({
        "claude-opus-4-7": {
            "context_length": override_ctx,
            "input_price_per_1M": 0.01,
            "output_price_per_1M": 0.02,
        },
    }))
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    assert table["claude-opus-4-7"].context_length == override_ctx
    assert table["claude-opus-4-7"].input_price_per_1M == 0.01
    assert table["claude-opus-4-7"].output_price_per_1M == 0.02


def test_my_models_adds_new_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A MY_MODELS entry whose key is new must be added to the loaded table."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text(json.dumps({
        "my-org/brand-new-model": {
            "context_length": 65536,
            "input_price_per_1M": 0.42,
            "output_price_per_1M": 0.84,
            "fc": True,
            "gen": True,
        },
    }))
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    assert "my-org/brand-new-model" in table
    assert table["my-org/brand-new-model"].context_length == 65536


def test_my_models_underscore_keys_are_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keys starting with ``_`` (documentation/example) must never enter the table."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text(json.dumps({
        "_documentation": ["this is doc"],
        "_example/model-name": {
            "context_length": 1,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
        },
        "real-model": {
            "context_length": 4096,
            "input_price_per_1M": 0.1,
            "output_price_per_1M": 0.2,
        },
    }))
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    assert "_documentation" not in table
    assert "_example/model-name" not in table
    assert "real-model" in table


def test_my_models_default_seed_has_no_active_entries_after_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The auto-seeded MY_MODELS.json must not add any real models to the table.

    Every entry in the default content is documentation/example
    (key starts with ``_``), so a fresh install sees exactly the
    bundled model table.
    """
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    bundled = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert set(table.keys()) == set(bundled.keys())


def test_loader_silently_ignores_corrupt_my_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed MY_MODELS.json must not break import — fall back silently."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text("{not json")
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    bundled = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert set(table.keys()) == set(bundled.keys())


def test_loader_silently_ignores_unwritable_my_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only HOME must not break import — MY_MODELS seed is skipped."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    fake_path = blocker / "MY_MODELS.json"  # parent is a regular file → mkdir fails
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    table = _load_model_info()
    bundled = json.loads(PACKAGE_MODEL_INFO_PATH.read_text())
    assert set(table.keys()) == set(bundled.keys())


def test_read_my_models_returns_empty_on_missing_path_and_unwritable_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_read_my_models`` returns ``{}`` when the file can be neither read nor seeded."""
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    fake_path = blocker / "MY_MODELS.json"
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    assert _read_my_models() == {}


def test_read_my_models_skips_underscore_and_non_dict_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-dict values and underscore-prefixed keys must be filtered out."""
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    fake_path.parent.mkdir(parents=True, exist_ok=True)
    fake_path.write_text(json.dumps({
        "_doc": "hello",
        "_example": {
            "context_length": 1,
            "input_price_per_1M": 0.0,
            "output_price_per_1M": 0.0,
        },
        "scalar-entry": "not a dict — must be skipped",
        "good-model": {
            "context_length": 100,
            "input_price_per_1M": 0.1,
            "output_price_per_1M": 0.2,
        },
    }))
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    result = _read_my_models()
    assert set(result.keys()) == {"good-model"}


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
    fake_path = tmp_path / "kiss" / "MY_MODELS.json"
    monkeypatch.setattr(mi_module, "USER_MY_MODELS_PATH", fake_path)
    data = _load_model_info()
    assert isinstance(data, dict)
    assert data
    sample_key = next(iter(data))
    assert isinstance(data[sample_key], ModelInfo)

"""Tests for the _increment_usage consolidation in task_history."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kiss.agents.sorcar.task_history import (
    _increment_usage,
    _record_file_usage,
    _record_model_usage,
)


def test_increment_usage_new_key() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{}")
        path = Path(f.name)
    try:
        _increment_usage(path, "mykey")
        data = json.loads(path.read_text())
        assert data["mykey"] == 1
    finally:
        path.unlink(missing_ok=True)


def test_increment_usage_existing_key() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump({"mykey": 5}, f)
        path = Path(f.name)
    try:
        _increment_usage(path, "mykey")
        data = json.loads(path.read_text())
        assert data["mykey"] == 6
    finally:
        path.unlink(missing_ok=True)


def test_increment_usage_with_extra() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{}")
        path = Path(f.name)
    try:
        _increment_usage(path, "key1", extra={"_last": "val"})
        data = json.loads(path.read_text())
        assert data["key1"] == 1
        assert data["_last"] == "val"
    finally:
        path.unlink(missing_ok=True)


def test_record_model_usage_sets_last() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{}")
        path = Path(f.name)
    try:
        import kiss.agents.sorcar.task_history as th
        orig = th.MODEL_USAGE_FILE
        th.MODEL_USAGE_FILE = path
        try:
            _record_model_usage("gpt-4")
            data = json.loads(path.read_text())
            assert data["gpt-4"] == 1
            assert data["_last"] == "gpt-4"
            _record_model_usage("gpt-4")
            data = json.loads(path.read_text())
            assert data["gpt-4"] == 2
        finally:
            th.MODEL_USAGE_FILE = orig
    finally:
        path.unlink(missing_ok=True)


def test_record_file_usage_increments() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        f.write("{}")
        path = Path(f.name)
    try:
        import kiss.agents.sorcar.task_history as th
        orig = th.FILE_USAGE_FILE
        th.FILE_USAGE_FILE = path
        try:
            _record_file_usage("/tmp/test.py")
            data = json.loads(path.read_text())
            assert data["/tmp/test.py"] == 1
            _record_file_usage("/tmp/test.py")
            data = json.loads(path.read_text())
            assert data["/tmp/test.py"] == 2
        finally:
            th.FILE_USAGE_FILE = orig
    finally:
        path.unlink(missing_ok=True)

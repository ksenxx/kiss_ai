# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that custom HTTP headers can be configured via the settings panel.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.agents.vscode.test_config_custom_headers``; the non-core tests remain there.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestConfigPersistence(unittest.TestCase):
    """custom_headers is persisted in config.json."""

    def setUp(self) -> None:
        import kiss.core.vscode_config as vc

        self._orig_dir = vc.CONFIG_DIR
        self._orig_path = vc.CONFIG_PATH
        self._tmpdir = tempfile.mkdtemp()
        vc.CONFIG_DIR = Path(self._tmpdir)
        vc.CONFIG_PATH = Path(self._tmpdir) / "config.json"

    def tearDown(self) -> None:
        import kiss.core.vscode_config as vc

        vc.CONFIG_DIR = self._orig_dir
        vc.CONFIG_PATH = self._orig_path
        import shutil

        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_custom_headers_in_defaults(self) -> None:
        from kiss.core.vscode_config import DEFAULTS

        assert "custom_headers" in DEFAULTS

    def test_save_and_load_custom_headers(self) -> None:
        from kiss.core.vscode_config import load_config, save_config

        save_config({"custom_headers": "X-Custom:value1\nAuthorization:Bearer tok"})
        cfg = load_config()
        assert cfg["custom_headers"] == "X-Custom:value1\nAuthorization:Bearer tok"

    def test_empty_headers_by_default(self) -> None:
        from kiss.core.vscode_config import load_config

        cfg = load_config()
        assert cfg["custom_headers"] == ""

    def test_preserves_other_keys(self) -> None:
        from kiss.core.vscode_config import load_config, save_config

        save_config({"custom_headers": "X-Foo:bar", "max_budget": 200})
        save_config({"custom_headers": "X-Baz:qux"})
        cfg = load_config()
        assert cfg["custom_headers"] == "X-Baz:qux"
        assert cfg["max_budget"] == 200


class TestCustomModelEntryIncludesHeaders(unittest.TestCase):
    """get_custom_model_entry includes extra_headers from custom_headers config."""

    def test_no_headers_when_empty(self) -> None:
        from kiss.core.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "",
        })
        assert entry is not None
        assert "extra_headers" not in entry or entry.get("extra_headers") == {}

    def test_headers_parsed_into_dict(self) -> None:
        from kiss.core.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "X-Custom:value1\nAuthorization:Bearer tok",
        })
        assert entry is not None
        assert entry["extra_headers"] == {
            "X-Custom": "value1",
            "Authorization": "Bearer tok",
        }

    def test_no_headers_when_no_endpoint(self) -> None:
        from kiss.core.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "",
            "custom_headers": "X-Custom:value1",
        })
        assert entry is None

    def test_malformed_header_lines_skipped(self) -> None:
        from kiss.core.vscode_config import get_custom_model_entry

        entry = get_custom_model_entry({
            "custom_endpoint": "http://localhost:8080/v1",
            "custom_headers": "X-Good:value\nbadline\n\nX-Also:good",
        })
        assert entry is not None
        assert entry["extra_headers"] == {
            "X-Good": "value",
            "X-Also": "good",
        }

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests locking behavior of code paths being simplified.

Core-only tests (depending only on ``kiss.core``) moved here from
``kiss.tests.vscode.test_simplify_vscode_misc_regr``; the non-core tests remain there.
"""

from __future__ import annotations

from kiss.core.vscode_config import (
    build_model_config,
    get_custom_model_entry,
)


class TestCustomModelConfig:
    def test_no_endpoint_returns_none(self) -> None:
        assert get_custom_model_entry({"custom_endpoint": ""}) is None
        assert build_model_config({"custom_endpoint": ""}) is None

    def test_entry_with_headers(self) -> None:
        cfg = {
            "custom_endpoint": "https://api.example.com/v1/",
            "custom_api_key": "sk-123",
            "custom_headers": "X-One: alpha\nbad line no colon\nX-Two:  beta ",
        }
        entry = get_custom_model_entry(cfg)
        assert entry is not None
        assert entry["name"] == "custom/v1"
        assert entry["endpoint"] == "https://api.example.com/v1/"
        assert entry["api_key"] == "sk-123"
        assert entry["extra_headers"] == {"X-One": "alpha", "X-Two": "beta"}
        assert entry["vendor"] == "Custom"

    def test_entry_without_headers_has_empty_dict(self) -> None:
        entry = get_custom_model_entry({"custom_endpoint": "http://e/m"})
        assert entry is not None
        assert entry["extra_headers"] == {}

    def test_model_config_with_headers(self) -> None:
        cfg = {
            "custom_endpoint": "http://localhost:8000",
            "custom_api_key": "k",
            "custom_headers": "Authorization: Bearer t",
        }
        mc = build_model_config(cfg)
        assert mc == {
            "base_url": "http://localhost:8000",
            "api_key": "k",
            "extra_headers": {"Authorization": "Bearer t"},
        }

    def test_model_config_omits_empty_headers_and_key(self) -> None:
        mc = build_model_config({
            "custom_endpoint": "http://x",
            "custom_headers": "junk without colon",
        })
        assert mc == {"base_url": "http://x"}

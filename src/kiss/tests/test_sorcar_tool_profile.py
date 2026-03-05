"""Tests for Sorcar browser-tool profile selection."""

import pytest

from kiss.agents.sorcar.sorcar_agent import _should_enable_browser_tools


def test_browser_tools_enabled_for_web_task_keywords(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_SORCAR_BROWSER_TOOLS", raising=False)
    assert _should_enable_browser_tools("Open https://example.com and click login.")
    assert _should_enable_browser_tools("Use browser to inspect this website.")


def test_browser_tools_disabled_for_local_coding_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KISS_SORCAR_BROWSER_TOOLS", raising=False)
    assert not _should_enable_browser_tools("build a space shooter game in rust and multi platform")


def test_browser_tools_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KISS_SORCAR_BROWSER_TOOLS", "off")
    assert not _should_enable_browser_tools("Open https://example.com")
    monkeypatch.setenv("KISS_SORCAR_BROWSER_TOOLS", "on")
    assert _should_enable_browser_tools("just write tests locally")

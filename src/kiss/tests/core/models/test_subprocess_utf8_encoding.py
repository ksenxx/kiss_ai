# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: CLI model wrappers pin subprocess pipes to UTF-8.

On Windows, Python text-mode subprocess pipes default to the system
ANSI code page rather than UTF-8.  Writing a prompt containing
non-ASCII text (accents, emoji) to such a pipe can raise
``UnicodeEncodeError`` or silently mangle the bytes the CLI receives.
Passing ``encoding="utf-8"`` alongside ``text=True`` makes the prompt
encode identically on every platform.

These tests mock ``subprocess.Popen`` so they never invoke the real
``codex`` / ``claude`` CLIs, feed a prompt with ``café`` and an emoji,
and assert that ``encoding="utf-8"`` is forwarded to ``Popen``.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel

_NON_ASCII_PROMPT = "Please review the café résumé ☕😀 and reply 'ok'."


class _FakeStdin:
    """Captures everything written to the subprocess stdin pipe."""

    def __init__(self) -> None:
        self.written = ""

    def write(self, data: str) -> None:
        self.written += data

    def close(self) -> None:
        pass


class _FakeStderr:
    def read(self) -> str:
        return ""


class _FakeProc:
    """Minimal ``subprocess.Popen`` stand-in driven by canned stdout lines."""

    def __init__(self, lines: list[str]) -> None:
        self.stdin = _FakeStdin()
        self.stdout = iter(lines)
        self.stderr = _FakeStderr()
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def poll(self) -> int:
        return 0

    def kill(self) -> None:
        pass

    def terminate(self) -> None:
        pass


def _install_fake_popen(
    monkeypatch: pytest.MonkeyPatch, lines: list[str]
) -> dict[str, Any]:
    """Patch ``subprocess.Popen`` with a fake and capture its kwargs.

    Args:
        monkeypatch: Pytest fixture used to install the patch.
        lines: Canned stdout lines the fake process emits.

    Returns:
        A dict that is populated with ``args``/``kwargs``/``proc`` when
        the patched ``Popen`` is invoked.
    """
    captured: dict[str, Any] = {}

    def fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
        proc = _FakeProc(lines)
        captured["args"] = args
        captured["kwargs"] = kwargs
        captured["proc"] = proc
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    return captured


def test_codex_generate_passes_utf8_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        '{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}',
        '{"type":"turn.completed","usage":{}}',
    ]
    captured = _install_fake_popen(monkeypatch, lines)

    m = CodexModel("codex/default")
    m._build_cli_args = lambda: ["codex-dummy", "exec"]  # type: ignore[method-assign]
    m.initialize(_NON_ASCII_PROMPT)

    content, _response = m.generate()

    assert content == "ok"
    assert captured["kwargs"].get("text") is True
    assert captured["kwargs"].get("encoding") == "utf-8"
    # The non-ASCII prompt was handed to the CLI verbatim.
    assert "café" in captured["proc"].stdin.written
    assert "😀" in captured["proc"].stdin.written


def test_claude_generate_passes_utf8_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    lines = [
        '{"type":"assistant","message":{"id":"m1","content":'
        '[{"type":"text","text":"ok"}]}}',
        '{"type":"result","result":"ok","usage":{}}',
    ]
    captured = _install_fake_popen(monkeypatch, lines)

    m = ClaudeCodeModel("cc/opus")
    m._build_cli_args = lambda: ["claude-dummy", "--print"]  # type: ignore[method-assign]
    m.initialize(_NON_ASCII_PROMPT)

    content, _response = m.generate()

    assert content == "ok"
    assert captured["kwargs"].get("text") is True
    assert captured["kwargs"].get("encoding") == "utf-8"
    assert "café" in captured["proc"].stdin.written
    assert "😀" in captured["proc"].stdin.written

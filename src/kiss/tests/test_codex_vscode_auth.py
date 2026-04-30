"""Tests for VS Code OpenAI Codex auth command wiring."""

from __future__ import annotations

from typing import Any, cast

from kiss.agents.vscode.commands import _CommandsMixin


class _Printer:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def broadcast(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class _Server(_CommandsMixin):
    def __init__(self) -> None:
        self.printer = cast(Any, _Printer())


def test_codex_auth_refresh_broadcasts_status(monkeypatch):
    def _status(model_name: str = "") -> dict[str, Any]:
        return {
            "model": model_name,
            "codex_auth_available": True,
            "preferred_auth": "codex",
        }

    monkeypatch.setattr("kiss.agents.vscode.vscode_config.get_codex_auth_status", _status)

    server = _Server()
    server._cmd_codex_auth({"type": "codexAuth", "action": "refresh", "model": "gpt-5.5"})

    printer = cast(_Printer, server.printer)
    assert printer.events == [
        {
            "type": "codexAuth",
            "status": "ok",
            "auth": {
                "model": "gpt-5.5",
                "codex_auth_available": True,
                "preferred_auth": "codex",
            },
        }
    ]


def test_codex_auth_login_broadcasts_login_url(monkeypatch):
    def _login(model_name: str = "") -> dict[str, Any]:
        return {
            "status": "ok",
            "login_url": "https://auth.openai.com/oauth/authorize",
            "auth": {"model": model_name, "login_pending": True},
        }

    monkeypatch.setattr("kiss.agents.vscode.vscode_config.start_codex_login", _login)

    server = _Server()
    server._cmd_codex_auth({"type": "codexAuth", "action": "login", "model": "gpt-5.5"})

    printer = cast(_Printer, server.printer)
    assert printer.events == [
        {
            "type": "codexAuth",
            "status": "ok",
            "login_url": "https://auth.openai.com/oauth/authorize",
            "auth": {"model": "gpt-5.5", "login_pending": True},
        }
    ]

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Codex subscription billing (``codex/`` models).

A fake ``codex`` on PATH records the credential variables a turn gives it
and answers ``login status`` and ``login --device-auth`` like the real
CLI (codex-cli 0.163: the status line on stderr, the device-code URL and
code on stdout).  The tests drive the real :class:`CodexModel`,
:func:`codex_auth_status`, :func:`codex_device_login`,
:func:`apply_config_to_env` and the daemon's ``getCodexStatus`` /
``startCodexLogin`` handlers — no mocks.

Unreachable without test doubles, documented rather than mocked: the
``OSError`` branches of :func:`codex_auth_status` and
:func:`codex_device_login` (a binary that exists but cannot be executed).
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_error import KISSError
from kiss.core.models.codex_model import (
    API_KEY_LOGIN_ERROR,
    SUBSCRIPTION_HINT,
    CodexModel,
    codex_auth_status,
    codex_cli_env,
    codex_device_login,
)
from kiss.core.user_config import DEFAULTS, apply_config_to_env
from kiss.server.commands import _CommandsMixin
from kiss.tests.conftest import install_cli_script, posix_only

pytestmark = posix_only("the fake codex is a POSIX script")

FINAL_TEXT = "SUBSCRIPTION-OK"

# The fake ``codex``: ``login status`` prints $cfg[status] on stderr and
# exits per $cfg[status_exit]; ``login --device-auth`` prints the
# device-code banner (ANSI colored like the real CLI), then sleeps
# $cfg[login_sleep] and exits per $cfg[login_exit]; ``exec`` records argv
# + the credential variables it sees, prints the events and exits per
# $cfg[turn_exit].
_FAKE_CODEX = """#!/usr/bin/env python3
import json, os, sys, pathlib, time
cfg = json.loads(pathlib.Path({cfg_path!r}).read_text())
seen = {{k: os.environ.get(k) for k in
        ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN")}}
if sys.argv[1:3] == ["login", "status"]:
    pathlib.Path({status_path!r}).write_text(json.dumps(seen))
    sys.stderr.write(cfg["status"] + "\\n")
    sys.exit(cfg.get("status_exit", 0))
if sys.argv[1:3] == ["login", "--device-auth"]:
    pathlib.Path({login_path!r}).write_text(json.dumps(seen))
    if cfg.get("login_banner", True):
        print("\\nWelcome to Codex [v\\x1b[90m0.163.0\\x1b[0m]\\n")
        print("1. Open this link in your browser and sign in to your account")
        print("   \\x1b[1mhttps://auth.openai.com/codex/device\\x1b[0m\\n")
        print("2. Enter this one-time code (expires in 15 minutes)")
        print("   \\x1b[1m2YCJ-GJHWN\\x1b[0m\\n", flush=True)
    else:
        sys.stderr.write("device code login is not available\\n")
    time.sleep(cfg.get("login_sleep", 0))
    sys.exit(cfg.get("login_exit", 0))
prompt = sys.stdin.read()
pathlib.Path({record_path!r}).write_text(
    json.dumps({{"argv": sys.argv[1:], "prompt": prompt, "env": seen}}))
for event in cfg["events"]:
    print(json.dumps(event), flush=True)
sys.exit(cfg.get("turn_exit", 0))
"""

_OK_EVENTS = [
    {"type": "thread.started", "thread_id": "t-1"},
    {"type": "item.completed", "item": {"type": "agent_message", "text": FINAL_TEXT}},
    {"type": "turn.completed", "usage": {"input_tokens": 3, "output_tokens": 2}},
]
_UNAUTHORIZED = {"type": "error", "message": "401 Unauthorized: not logged in"}


class _FakeCodex:
    """The fake CLI on PATH and the knobs that shape its answers."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self.cfg_path = tmp_path / "fake-codex.json"
        self.record_path = tmp_path / "turn.json"
        self.status_path = tmp_path / "status-env.json"
        self.login_path = tmp_path / "login-env.json"
        self.configure()
        install_cli_script(
            bin_dir / "codex",
            _FAKE_CODEX.format(
                cfg_path=str(self.cfg_path),
                record_path=str(self.record_path),
                status_path=str(self.status_path),
                login_path=str(self.login_path),
            ),
        )
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def configure(self, **cfg: Any) -> None:
        """Rewrite the fake's answers (``events``, ``turn_exit``, ``status``, ...)."""
        full: dict[str, Any] = {"events": _OK_EVENTS, "status": "Logged in using ChatGPT"}
        full.update(cfg)
        self.cfg_path.write_text(json.dumps(full))

    def turn(self) -> dict[str, Any]:
        """The last recorded ``exec`` turn (argv / prompt / env)."""
        turn: dict[str, Any] = json.loads(self.record_path.read_text())
        return turn

    def env_seen_by(self, path: Path) -> dict[str, Any]:
        """The credential variables one ``login`` invocation saw."""
        env: dict[str, Any] = json.loads(path.read_text())
        return env


@pytest.fixture
def fake_codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeCodex:
    """A fake ``codex`` on PATH, API keys in the environment, switch on."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("CODEX_API_KEY", "sk-codex-test")
    monkeypatch.delenv("CODEX_ACCESS_TOKEN", raising=False)
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "")
    return _FakeCodex(tmp_path, monkeypatch)


def _run_turn(tmp_path: Path) -> str:
    """Run one ``codex/`` turn through the real model and return its text."""
    model = CodexModel("codex/gpt-5.5", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": "say the word"}]
    text, _ = model.generate()
    return text


def test_turn_hides_api_keys_and_passes_the_panel_token(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch on: no API key reaches the CLI, the access token does."""
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "codex-token-panel")
    assert _run_turn(tmp_path) == FINAL_TEXT
    env = fake_codex.turn()["env"]
    assert env["OPENAI_API_KEY"] is None
    assert env["CODEX_API_KEY"] is None
    assert env["CODEX_ACCESS_TOKEN"] == "codex-token-panel"
    # The daemon's own environment is untouched.
    assert os.environ["OPENAI_API_KEY"] == "sk-openai-test"


def test_turn_without_a_panel_token_sets_none(fake_codex: _FakeCodex, tmp_path: Path) -> None:
    """Switch on, no token: the CLI falls through to its own login."""
    assert _run_turn(tmp_path) == FINAL_TEXT
    assert fake_codex.turn()["env"] == {
        "OPENAI_API_KEY": None,
        "CODEX_API_KEY": None,
        "CODEX_ACCESS_TOKEN": None,
    }


def test_switch_off_inherits_the_environment(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: the CLI sees the API keys and no injected token."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "codex-token-panel")
    assert codex_cli_env()["CODEX_API_KEY"] == "sk-codex-test"
    assert _run_turn(tmp_path) == FINAL_TEXT
    assert fake_codex.turn()["env"] == {
        "OPENAI_API_KEY": "sk-openai-test",
        "CODEX_API_KEY": "sk-codex-test",
        "CODEX_ACCESS_TOKEN": None,
    }


def test_switch_off_feeds_openai_api_key_to_the_cli(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off, no ``CODEX_API_KEY``: the CLI (which reads only that
    variable) gets the ``OPENAI_API_KEY`` from the environment, else the
    one from the settings panel."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    monkeypatch.delenv("CODEX_API_KEY")
    assert _run_turn(tmp_path) == FINAL_TEXT
    assert fake_codex.turn()["env"]["CODEX_API_KEY"] == "sk-openai-test"
    assert "CODEX_API_KEY" not in os.environ

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setattr(DEFAULT_CONFIG, "OPENAI_API_KEY", "sk-openai-panel")
    assert codex_cli_env()["CODEX_API_KEY"] == "sk-openai-panel"
    monkeypatch.setattr(DEFAULT_CONFIG, "OPENAI_API_KEY", "")
    assert "CODEX_API_KEY" not in codex_cli_env()


def _write_auth_json(codex_home: Path, auth: dict) -> None:
    codex_home.mkdir(exist_ok=True)
    (codex_home / "auth.json").write_text(json.dumps(auth))


def test_api_key_login_is_refused_with_the_switch_on(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CLI logged in with an API key (``auth.json`` under ``CODEX_HOME``)
    would bill per token: the turn is refused before the CLI starts.  A
    ChatGPT login passes; a file the CLI wrote before ``auth_mode`` existed
    is an API-key login whenever it holds a key (the CLI prefers the key
    over the tokens next to it); a logout mid-session (file gone) is seen
    at once; a configured ``CODEX_ACCESS_TOKEN`` outranks the saved key."""
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_auth_json(codex_home, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-saved"})
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    assert API_KEY_LOGIN_ERROR in str(info.value)
    assert SUBSCRIPTION_HINT in str(info.value)
    assert not fake_codex.record_path.exists(), "no turn must have run"

    _write_auth_json(codex_home, {"auth_mode": "chatgpt", "tokens": {"access_token": "t"}})
    assert _run_turn(tmp_path) == FINAL_TEXT
    _write_auth_json(codex_home, {"tokens": {"access_token": "t"}})
    assert _run_turn(tmp_path) == FINAL_TEXT
    _write_auth_json(codex_home, {"OPENAI_API_KEY": "sk-old", "tokens": {"access_token": "t"}})
    with pytest.raises(KISSError):
        _run_turn(tmp_path)
    (codex_home / "auth.json").write_text("not json")
    assert _run_turn(tmp_path) == FINAL_TEXT
    (codex_home / "auth.json").unlink()
    assert _run_turn(tmp_path) == FINAL_TEXT

    _write_auth_json(codex_home, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-saved"})
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "codex-token-panel")
    assert _run_turn(tmp_path) == FINAL_TEXT
    assert fake_codex.turn()["env"]["CODEX_ACCESS_TOKEN"] == "codex-token-panel"


def test_api_key_login_is_fine_with_the_switch_off(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: the user chose per-token billing, so the saved key runs."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    codex_home = tmp_path / "codex-home"
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    _write_auth_json(codex_home, {"auth_mode": "apikey", "OPENAI_API_KEY": "sk-saved"})
    assert _run_turn(tmp_path) == FINAL_TEXT


def test_turn_failure_names_the_cause_and_the_fix(fake_codex: _FakeCodex, tmp_path: Path) -> None:
    """An error event is surfaced with the subscription hint."""
    fake_codex.configure(events=[_UNAUTHORIZED], turn_exit=1)
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    assert "401 Unauthorized" in str(info.value)
    assert SUBSCRIPTION_HINT in str(info.value)


def test_turn_failure_with_the_switch_off_has_no_hint(
    fake_codex: _FakeCodex, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    fake_codex.configure(events=[_UNAUTHORIZED], turn_exit=1)
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    assert "401 Unauthorized" in str(info.value)
    assert SUBSCRIPTION_HINT not in str(info.value)


@pytest.mark.parametrize(
    "line,method",
    [
        ("Logged in using ChatGPT", "chatgpt"),
        ("Logged in using an API key - sk-proj-***ABCDE", "api_key"),
        ("Logged in using access token", "access_token"),
        ("Logged in using personal access token", "access_token"),
        ("Logged in using workload identity", "workload_identity"),
        ("Logged in using Amazon Bedrock (profile: x)", "bedrock"),
    ],
)
def test_auth_status_maps_the_cli_line(fake_codex: _FakeCodex, line: str, method: str) -> None:
    """Every ``codex login status`` answer maps to the panel's authMethod."""
    fake_codex.configure(status=line)
    status = codex_auth_status()
    assert status["installed"] is True
    assert status["loggedIn"] is True
    assert status["authMethod"] == method
    assert status["detail"] == line
    assert "error" not in status


def test_auth_status_runs_in_the_stripped_env(
    fake_codex: _FakeCodex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status reflects what a turn would see: no API keys, the token."""
    monkeypatch.setattr(DEFAULT_CONFIG, "CODEX_ACCESS_TOKEN", "codex-token-panel")
    status = codex_auth_status()
    assert status["useSubscription"] is True
    assert status["hasToken"] is True
    assert fake_codex.env_seen_by(fake_codex.status_path) == {
        "OPENAI_API_KEY": None,
        "CODEX_API_KEY": None,
        "CODEX_ACCESS_TOKEN": "codex-token-panel",
    }


def test_auth_status_with_the_switch_off_sees_the_api_key(
    fake_codex: _FakeCodex, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", False)
    status = codex_auth_status()
    assert status["useSubscription"] is False
    assert status["hasToken"] is False
    assert fake_codex.env_seen_by(fake_codex.status_path)["CODEX_API_KEY"] == "sk-codex-test"


def test_auth_status_not_logged_in(fake_codex: _FakeCodex) -> None:
    fake_codex.configure(status="Not logged in", status_exit=1)
    status = codex_auth_status()
    assert status["installed"] is True
    assert status["loggedIn"] is False
    assert status["authMethod"] == "none"
    assert "error" not in status


def test_auth_status_unknown_answer_reports_an_error(fake_codex: _FakeCodex) -> None:
    fake_codex.configure(status="Error checking login status: invalid ID token", status_exit=1)
    status = codex_auth_status()
    assert status["loggedIn"] is False
    assert "exit 1" in status["error"]
    assert "invalid ID token" in status["error"]


def test_auth_status_without_the_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ``codex`` anywhere: installed is False and the panel says so."""
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("kiss.core.models.codex_model._UI_CANDIDATE_PATHS", ())
    status = codex_auth_status()
    assert status["installed"] is False
    assert "not found" in status["error"]
    assert codex_device_login()["error"].startswith("Codex CLI")


def test_auth_status_when_the_cli_hangs(
    fake_codex: _FakeCodex, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timeout is an error, not an exception."""
    hanging = fake_codex.cfg_path.parent / "bin" / "codex"
    hanging.write_text("#!/bin/sh\nsleep 30\n")
    status = codex_auth_status(timeout=0.5)
    assert status["installed"] is True
    assert "timed out" in status["error"]


def test_device_login_returns_the_url_and_code_and_leaves_the_cli_polling(
    fake_codex: _FakeCodex,
) -> None:
    """The banner's URL and code come back at once while the CLI (polling
    OpenAI in the real world) keeps running, reaped by a daemon thread
    when it exits; no API key is involved."""
    # The CLI lingers well past what spawning and reading its banner
    # take on a loaded machine (0.5 s was exceeded under the full suite).
    fake_codex.configure(login_sleep=2.0)
    started = time.monotonic()
    prompt = codex_device_login()
    assert prompt == {"url": "https://auth.openai.com/codex/device", "code": "2YCJ-GJHWN"}
    assert time.monotonic() - started < 2.0, "must return before the CLI exits"
    reaper = [t for t in threading.enumerate() if t.name == "codex-device-login"]
    assert len(reaper) == 1 and reaper[0].daemon
    reaper[0].join(10)
    assert not reaper[0].is_alive()
    assert fake_codex.env_seen_by(fake_codex.login_path)["CODEX_API_KEY"] is None


def test_device_login_without_a_banner_is_an_error(fake_codex: _FakeCodex) -> None:
    fake_codex.configure(login_banner=False, login_exit=1)
    result = codex_device_login()
    assert "printed no login code" in result["error"]
    assert "device code login is not available" in result["error"]


def test_device_login_that_prints_nothing_times_out(fake_codex: _FakeCodex) -> None:
    hanging = fake_codex.cfg_path.parent / "bin" / "codex"
    hanging.write_text("#!/bin/sh\nsleep 30\n")
    result = codex_device_login(timeout=0.5)
    assert "printed no login code" in result["error"]
    assert "(no output)" in result["error"]


def test_apply_config_to_env_mirrors_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored switch reaches ``DEFAULT_CONFIG``; a missing key means the default."""
    monkeypatch.setattr(DEFAULT_CONFIG, "codex_use_subscription", True)
    apply_config_to_env({"max_budget": 1.0, "codex_use_subscription": False})
    assert DEFAULT_CONFIG.codex_use_subscription is False
    apply_config_to_env({"max_budget": 1.0})
    assert DEFAULT_CONFIG.codex_use_subscription is DEFAULTS["codex_use_subscription"]
    assert DEFAULTS["codex_use_subscription"] is True


class _FakePrinter:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.lock = threading.Lock()

    def broadcast(self, msg: dict[str, Any]) -> None:
        with self.lock:
            self.messages.append(msg)


class _FakeServer(_CommandsMixin):
    def __init__(self) -> None:
        self.printer: Any = _FakePrinter()
        self.work_dir = "/tmp"
        self._state_lock = threading.RLock()

    def wait_for(self, event_type: str, count: int = 1, timeout: float = 20.0) -> dict[str, Any]:
        """The *count*-th event of *event_type*, waiting for it to arrive."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.printer.lock:
                found = [m for m in self.printer.messages if m.get("type") == event_type]
            if len(found) >= count:
                return dict(found[count - 1])
            time.sleep(0.02)
        raise AssertionError(f"no {event_type} event #{count} within {timeout}s")


def test_get_codex_status_command_answers_the_sender(fake_codex: _FakeCodex) -> None:
    """``getCodexStatus`` replies ``codexStatus`` stamped with the connId."""
    assert "getCodexStatus" in _CommandsMixin._HANDLERS
    server = _FakeServer()
    server._cmd_get_codex_status({"connId": "c7"})
    msg = server.wait_for("codexStatus")
    assert msg["connId"] == "c7"
    assert msg["loggedIn"] is True
    assert msg["authMethod"] == "chatgpt"
    assert msg["useSubscription"] is True

    server = _FakeServer()
    server._cmd_get_codex_status({})
    assert "connId" not in server.wait_for("codexStatus")


def test_start_codex_login_command_replies_with_the_prompt(fake_codex: _FakeCodex) -> None:
    """``startCodexLogin`` replies ``codexLogin`` with the URL and code; the
    panel then polls ``getCodexStatus``, which reports the finished login."""
    assert "startCodexLogin" in _CommandsMixin._HANDLERS
    fake_codex.configure(login_sleep=0.3, status="Not logged in", status_exit=1)
    server = _FakeServer()
    server._cmd_start_codex_login({"connId": "c9"})
    login = server.wait_for("codexLogin")
    assert login["connId"] == "c9"
    assert login["code"] == "2YCJ-GJHWN"
    assert login["url"] == "https://auth.openai.com/codex/device"
    server._cmd_get_codex_status({"connId": "c9"})
    assert server.wait_for("codexStatus")["loggedIn"] is False
    # The login completed: the user's auth.json now holds the ChatGPT login.
    fake_codex.configure(status="Logged in using ChatGPT")
    server._cmd_get_codex_status({"connId": "c9"})
    status = server.wait_for("codexStatus", count=2)
    assert status["connId"] == "c9"
    assert status["authMethod"] == "chatgpt"
    for reaper in threading.enumerate():
        if reaper.name == "codex-device-login":
            reaper.join(10)


def test_start_codex_login_command_reports_a_failure(fake_codex: _FakeCodex) -> None:
    fake_codex.configure(login_banner=False, login_exit=2)
    server = _FakeServer()
    server._cmd_start_codex_login({"connId": "c1"})
    login = server.wait_for("codexLogin")
    assert "printed no login code" in login["error"]
    assert "url" not in login

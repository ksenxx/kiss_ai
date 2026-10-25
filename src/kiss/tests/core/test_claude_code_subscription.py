# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for Claude Code subscription billing (``cc/`` models).

A fake ``claude`` executable on PATH records the environment every turn
runs in and answers ``claude auth status``; the tests drive the real
:class:`~kiss.core.models.claude_code_model.ClaudeCodeModel`, the
settings plumbing (``apply_config_to_env``) and the daemon's
``getClaudeCodeStatus`` handler against it.

Covered:

* With the switch on (the default) a turn's CLI sees neither
  ``ANTHROPIC_API_KEY`` nor ``ANTHROPIC_AUTH_TOKEN`` and does see the
  panel's ``CLAUDE_CODE_OAUTH_TOKEN``; off, it inherits the daemon's
  environment unchanged.
* A not-logged-in failure names the CLI's own message and, with the
  switch on, how to fix it; a failure without a result text keeps the
  plain exit error.
* ``claude_auth_status`` reports the CLI's JSON, a missing CLI, a CLI
  that fails to start, and a non-JSON answer — and runs in the same
  stripped environment as a turn.
* ``apply_config_to_env`` mirrors the stored switch into
  ``DEFAULT_CONFIG``.
* ``getClaudeCodeStatus`` answers a conn-scoped ``claudeCodeStatus``.

Run::

    uv run pytest src/kiss/tests/core/test_claude_code_subscription.py -v
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import pytest

# The singleton is read through the module at call time: some server tests
# replace ``kiss.core.config.DEFAULT_CONFIG`` with a fresh ``Config()``, and
# the adapters import it lazily, so a name bound at import time would patch
# an object the code under test no longer reads.
from kiss.core import config as config_module
from kiss.core.kiss_error import KISSError
from kiss.core.models.claude_code_model import (
    SUBSCRIPTION_HINT,
    ClaudeCodeModel,
    claude_auth_status,
    claude_subscription_env,
)
from kiss.core.user_config import DEFAULTS, apply_config_to_env
from kiss.server.commands import _CommandsMixin
from kiss.tests.conftest import install_cli_script, posix_only

pytestmark = posix_only("the fake claude is a POSIX script")

FINAL_TEXT = "SUBSCRIPTION-OK"
_PANEL_TOKEN = "sk-ant-oat01-panel"

# The fake ``claude``: ``auth status`` prints the JSON in $FAKE_AUTH_JSON
# (exit per $FAKE_AUTH_EXIT) plus whether it saw an API key; any other
# invocation records argv + the credential variables it sees, waits for
# the file $cfg[turn_gate] to exist when one is named, then prints the
# turn in $FAKE_EVENTS_FILE and exits per $FAKE_TURN_EXIT.
_FAKE_CLAUDE = """#!/usr/bin/env python3
import json, os, sys, pathlib, time
cfg = json.loads(pathlib.Path({cfg_path!r}).read_text())
seen = {{k: os.environ.get(k) for k in
        ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN")}}
if sys.argv[1:3] == ["auth", "status"]:
    if cfg.get("auth_raw") is not None:
        sys.stdout.write(cfg["auth_raw"])
    else:
        print(json.dumps({{**cfg["auth"], "sawApiKey": seen["ANTHROPIC_API_KEY"] is not None}}))
    sys.exit(cfg.get("auth_exit", 0))
prompt = sys.stdin.read()
pathlib.Path({record_path!r}).write_text(
    json.dumps({{"argv": sys.argv[1:], "prompt": prompt, "env": seen}}))
while cfg.get("turn_gate") and not os.path.exists(cfg["turn_gate"]):
    time.sleep(0.01)
for event in cfg["events"]:
    print(json.dumps(event), flush=True)
sys.exit(cfg.get("turn_exit", 0))
"""

_OK_EVENTS = [
    {"type": "assistant", "message": {"content": [{"type": "text", "text": FINAL_TEXT}]}},
    {"type": "result", "result": FINAL_TEXT, "usage": {"input_tokens": 3, "output_tokens": 2}},
]
_LOGIN_FAILURE = "Not logged in · Please run /login"
_LOGGED_IN = {
    "loggedIn": True,
    "authMethod": "claude.ai",
    "subscriptionType": "max",
    "email": "me@example.com",
}


class _FakeClaude:
    """The fake CLI on PATH and the knobs that shape its answers."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        self.cfg_path = tmp_path / "fake-claude.json"
        self.record_path = tmp_path / "turn.json"
        self.configure()
        install_cli_script(
            bin_dir / "claude",
            _FAKE_CLAUDE.format(
                cfg_path=str(self.cfg_path), record_path=str(self.record_path)
            ),
        )
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def configure(self, **cfg: Any) -> None:
        """Rewrite the fake's answers (``events``, ``turn_exit``, ``auth``, ...)."""
        full: dict[str, Any] = {"events": _OK_EVENTS, "auth": _LOGGED_IN}
        full.update(cfg)
        self.cfg_path.write_text(json.dumps(full))

    def turn(self) -> dict[str, Any]:
        """The last recorded turn (argv / prompt / env)."""
        turn: dict[str, Any] = json.loads(self.record_path.read_text())
        return turn


@pytest.fixture
def fake_claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeClaude:
    """A fake ``claude`` on PATH, API keys in the environment, switch on."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-test")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-test")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", True)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", "")
    return _FakeClaude(tmp_path, monkeypatch)


def _run_turn(tmp_path: Path) -> str:
    """Run one ``cc/`` turn through the real model and return its text."""
    model = ClaudeCodeModel("cc/sonnet", {"work_dir": str(tmp_path), "timeout": 30})
    model.conversation = [{"role": "user", "content": "say the word"}]
    text, _ = model.generate()
    return text


def _run_turn_collecting(tmp_path: Path, errors: list[BaseException]) -> None:
    """Thread target: run one turn and store the exception it raised in *errors*."""
    try:
        _run_turn(tmp_path)
    except Exception as e:  # noqa: BLE001
        errors.append(e)


def test_turn_hides_api_keys_and_passes_the_panel_token(
    fake_claude: _FakeClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch on: no API key reaches the CLI, the setup-token does."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", _PANEL_TOKEN)
    assert _run_turn(tmp_path) == FINAL_TEXT
    env = fake_claude.turn()["env"]
    assert env["ANTHROPIC_API_KEY"] is None
    assert env["ANTHROPIC_AUTH_TOKEN"] is None
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == _PANEL_TOKEN
    # The daemon's own environment is untouched.
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-api-test"


def test_turn_without_a_panel_token_sets_none(
    fake_claude: _FakeClaude, tmp_path: Path
) -> None:
    """Switch on, no token: the CLI falls through to its own login."""
    assert _run_turn(tmp_path) == FINAL_TEXT
    env = fake_claude.turn()["env"]
    assert env == {
        "ANTHROPIC_API_KEY": None,
        "ANTHROPIC_AUTH_TOKEN": None,
        "CLAUDE_CODE_OAUTH_TOKEN": None,
    }


def test_switch_off_inherits_the_environment(
    fake_claude: _FakeClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switch off: the CLI sees the API key and no injected token."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", _PANEL_TOKEN)
    assert claude_subscription_env() is None
    assert _run_turn(tmp_path) == FINAL_TEXT
    env = fake_claude.turn()["env"]
    assert env["ANTHROPIC_API_KEY"] == "sk-ant-api-test"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "bearer-test"
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] is None


def test_login_failure_names_the_cause_and_the_fix(
    fake_claude: _FakeClaude, tmp_path: Path
) -> None:
    """A not-logged-in result is surfaced with the subscription hint."""
    fake_claude.configure(
        events=[{"type": "result", "is_error": True, "result": _LOGIN_FAILURE}],
        turn_exit=1,
    )
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    message = str(info.value)
    assert "exit 1" in message
    assert _LOGIN_FAILURE in message
    assert SUBSCRIPTION_HINT in message


def test_login_failure_with_the_switch_off_has_no_hint(
    fake_claude: _FakeClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off, the CLI's message is shown but the hint does not apply."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.configure(
        events=[{"type": "result", "is_error": True, "result": "Invalid API key"}],
        turn_exit=1,
    )
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    assert "Invalid API key" in str(info.value)
    assert SUBSCRIPTION_HINT not in str(info.value)


def test_hint_follows_the_switch_the_turn_was_launched_with(
    fake_claude: _FakeClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The switch is read once per launch: a settings flip while the CLI
    runs must not change whether the failure carries the hint (the turn
    ran without the API keys, so the hint is what explains the login
    failure)."""
    gate = tmp_path / "gate"
    fake_claude.configure(
        events=[{"type": "result", "is_error": True, "result": _LOGIN_FAILURE}],
        turn_exit=1,
        turn_gate=str(gate),
    )
    errors: list[BaseException] = []
    worker = threading.Thread(target=_run_turn_collecting, args=(tmp_path, errors))
    worker.start()
    deadline = time.monotonic() + 10
    while not fake_claude.record_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake_claude.record_path.exists(), "the fake claude never started"
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", False)
    gate.touch()
    worker.join(timeout=10)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KISSError)
    assert _LOGIN_FAILURE in str(errors[0])
    assert SUBSCRIPTION_HINT in str(errors[0])


def test_failure_without_a_result_text_keeps_the_exit_error(
    fake_claude: _FakeClaude, tmp_path: Path
) -> None:
    """No ``is_error`` result (a crash): the plain exit error, no hint."""
    fake_claude.configure(events=[_OK_EVENTS[0]], turn_exit=3)
    with pytest.raises(KISSError) as info:
        _run_turn(tmp_path)
    assert "exit 3" in str(info.value)
    assert SUBSCRIPTION_HINT not in str(info.value)


def test_auth_status_reports_the_cli_json_in_the_stripped_env(
    fake_claude: _FakeClaude, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``claude auth status`` runs without the API key and its JSON is returned."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "CLAUDE_CODE_OAUTH_TOKEN", _PANEL_TOKEN)
    status = claude_auth_status()
    assert status["installed"] is True
    assert status["loggedIn"] is True
    assert status["authMethod"] == "claude.ai"
    assert status["subscriptionType"] == "max"
    assert status["email"] == "me@example.com"
    assert status["useSubscription"] is True
    assert status["hasToken"] is True
    assert status["sawApiKey"] is False
    assert "error" not in status


def test_auth_status_with_the_switch_off_sees_the_api_key(
    fake_claude: _FakeClaude, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Off, the status check runs in the inherited environment."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", False)
    fake_claude.configure(auth={"loggedIn": True, "authMethod": "api_key"}, auth_exit=0)
    status = claude_auth_status()
    assert status["sawApiKey"] is True
    assert status["authMethod"] == "api_key"
    assert status["useSubscription"] is False
    assert status["hasToken"] is False


def test_auth_status_not_logged_in(fake_claude: _FakeClaude) -> None:
    """Exit 1 with a logged-out JSON is a plain not-logged-in status."""
    fake_claude.configure(auth={"loggedIn": False, "authMethod": None}, auth_exit=1)
    status = claude_auth_status()
    assert status["loggedIn"] is False
    assert status["authMethod"] == "none"
    assert "error" not in status


def test_auth_status_without_json_reports_an_error(fake_claude: _FakeClaude) -> None:
    """An old CLI printing text instead of JSON yields ``error``."""
    fake_claude.configure(auth_raw="error: unknown command 'auth'\n", auth_exit=2)
    status = claude_auth_status()
    assert status["installed"] is True
    assert status["loggedIn"] is False
    assert "exit 2" in status["error"]
    assert "unknown command" in status["error"]


def test_auth_status_without_the_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ``claude`` on PATH: ``installed`` is False with a reason."""
    monkeypatch.setenv("PATH", str(tmp_path))
    status = claude_auth_status()
    assert status["installed"] is False
    assert "not found on PATH" in status["error"]


def test_auth_status_when_the_cli_cannot_run(
    fake_claude: _FakeClaude, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An executable that cannot start (or hangs past the timeout) is an error."""
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "claude").write_text("#!/nonexistent/interpreter\n")
    (broken / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{broken}{os.pathsep}{os.environ['PATH']}")
    status = claude_auth_status()
    assert status["installed"] is True
    assert "failed" in status["error"]


def test_apply_config_to_env_mirrors_the_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The stored switch reaches ``DEFAULT_CONFIG``; a missing key means the default."""
    monkeypatch.setattr(config_module.DEFAULT_CONFIG, "claude_code_use_subscription", True)
    apply_config_to_env({"max_budget": 1.0, "claude_code_use_subscription": False})
    assert config_module.DEFAULT_CONFIG.claude_code_use_subscription is False
    apply_config_to_env({"max_budget": 1.0})
    assert config_module.DEFAULT_CONFIG.claude_code_use_subscription is DEFAULTS[
        "claude_code_use_subscription"
    ]
    assert DEFAULTS["claude_code_use_subscription"] is True


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

    def wait_for(self, event_type: str, timeout: float = 20.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.printer.lock:
                for msg in self.printer.messages:
                    if msg.get("type") == event_type:
                        return dict(msg)
            time.sleep(0.02)
        raise AssertionError(f"no {event_type} event within {timeout}s")


def test_get_claude_code_status_command_answers_the_sender(
    fake_claude: _FakeClaude,
) -> None:
    """``getClaudeCodeStatus`` replies ``claudeCodeStatus`` stamped with the connId."""
    assert "getClaudeCodeStatus" in _CommandsMixin._HANDLERS
    server = _FakeServer()
    server._cmd_get_claude_code_status({"connId": "c7"})
    msg = server.wait_for("claudeCodeStatus")
    assert msg["connId"] == "c7"
    assert msg["loggedIn"] is True
    assert msg["authMethod"] == "claude.ai"
    assert msg["useSubscription"] is True
    assert msg["sawApiKey"] is False

    server = _FakeServer()
    server._cmd_get_claude_code_status({})
    msg = server.wait_for("claudeCodeStatus")
    assert "connId" not in msg

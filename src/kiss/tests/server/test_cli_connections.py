"""Sanitized asynchronous CLI connection requests and durable billing settings."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from kiss.core.models import cli_connections as cli
from kiss.server.commands import _CommandsMixin


def test_status_request_is_async_and_scoped(monkeypatch):
    entered, release, done = threading.Event(), threading.Event(), threading.Event()
    events = []

    def status(provider, **kwargs):
        assert kwargs == {"refresh": True, "cwd": "/repo"}
        entered.set()
        assert release.wait(2)
        return cli.CLIConnection(provider, "signed_out", message="Sign in.")

    monkeypatch.setattr(cli, "get_cli_connection", status)
    server = SimpleNamespace(
        work_dir="/repo",
        printer=SimpleNamespace(broadcast=events.append),
        _get_models=lambda conn_id: done.set(),
    )
    _CommandsMixin._cmd_get_cli_connections(server, {"connId": "client", "refresh": True})
    assert entered.wait(2)
    assert not done.is_set(), "status probing must not block the command loop"
    release.set()
    assert done.wait(2)
    assert events[0]["connId"] == "client"
    assert len(events[0]["connections"]) == 2
    assert set(events[0]["connections"][0]) == {
        "provider",
        "status",
        "version",
        "auth_method",
        "billing_mode",
        "message",
    }


def test_billing_preferences_persist():
    from kiss.core.vscode_config import load_config, save_config

    save_config(
        {
            "claude_cli_billing_mode": "subscription",
            "codex_cli_billing_mode": "existing",
            "allow_fable_usage_credits": True,
        }
    )
    cfg = load_config()
    assert cfg["claude_cli_billing_mode"] == "subscription"
    assert cfg["codex_cli_billing_mode"] == "existing"
    assert cfg["allow_fable_usage_credits"] is True


def test_subscription_task_update_does_not_select_api_sea(monkeypatch):
    import kiss.agents.sorcar.chat_sorcar_agent as chat
    import kiss.agents.sorcar.sorcar_agent as sorcar
    import kiss.server.task_update as updates

    parent = SimpleNamespace(model_name="cc/opus", model_config={"subscription_only": True})
    child = Mock()
    child.run.return_value = "success: true\nsummary: checked"
    monkeypatch.setattr(chat, "ChatSorcarAgent", lambda name: child)
    monkeypatch.setattr(sorcar, "subagent_parent_tab_id_of", lambda agent: "")
    monkeypatch.setattr(sorcar, "_live_agent_usage", lambda agent: (0.0, 0, 0))
    monkeypatch.setattr(updates, "charge_side_channel_usage", lambda *a, **kw: None)
    sea = SimpleNamespace(
        settings={
            "model": "gpt-6-luna",
            "tool_profile": "none",
            "allow_fan_out": False,
            "use_web_tools": False,
            "use_memory": False,
        },
        prompt="update",
        tools_hook=None,
        llm_call_hook=None,
        tool_call_hook=None,
        system_prompt_hook=None,
    )
    monkeypatch.setattr(updates, "evaluate_sea", lambda *args: sea)
    assert updates.run_task_update_sea(parent, "task")[0] == "checked"
    assert child.run.call_args.kwargs["model_name"] == "cc/opus"
    assert child.run.call_args.kwargs["model_config"]["subscription_only"] is True

"""Offline billing and authentication boundaries for the official CLI adapters."""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from kiss.core.kiss_error import KISSError
from kiss.core.models import cli_connections as cli


@pytest.fixture
def probe(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    monkeypatch.setattr(cli, "_executable", lambda provider: "/fake/" + provider)
    runner = Mock()
    monkeypatch.setattr(cli, "_run_status", runner)

    def configure(method="claude.ai", version="2.1.300", provider="claude", code=0):
        payload = json.dumps({"authMethod": method}) if provider == "claude" else method
        runner.side_effect = [
            subprocess.CompletedProcess([], 0, version, ""),
            subprocess.CompletedProcess([], code, payload, ""),
        ]
        return runner

    return configure


@pytest.mark.parametrize(
    "method,state",
    [
        ("claude.ai", "connected"),
        ("none", "signed_out"),
        ("api_key", "conflict"),
        ("api_key_helper", "conflict"),
        ("third_party", "conflict"),
        ("oauth_token", "conflict"),
        ("unknown", "unknown"),
        (["malformed"], "unknown"),
    ],
)
def test_claude_subscription_status(probe, method, state):
    probe(method)
    assert cli.get_cli_connection("claude", mode="subscription", refresh=True).status == state


def test_missing_cli():
    assert cli.get_cli_connection("claude", mode="subscription", refresh=True).status == "missing"


def test_old_claude_and_existing_configuration(probe):
    probe("api_key", "2.1.1")
    status = cli.get_cli_connection("claude", mode="", refresh=True)
    assert (status.status, status.billing_mode) == ("connected", "existing")
    probe("claude.ai", "2.1.1")
    assert (
        cli.get_cli_connection("claude", mode="subscription", refresh=True).status
        == "unsupported_version"
    )


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("cli", 4), OSError("hidden secret")])
def test_errors_do_not_disclose_output(monkeypatch, probe, failure):
    runner = probe()
    runner.side_effect = failure
    status = cli.get_cli_connection("claude", mode="subscription", refresh=True)
    assert status.status == "unknown"
    assert "hidden secret" not in json.dumps(status.wire())


def test_malformed_json(probe):
    runner = probe()
    runner.side_effect = [
        subprocess.CompletedProcess([], 0, "2.1.300", ""),
        subprocess.CompletedProcess([], 0, "not json secret", ""),
    ]
    assert cli.get_cli_connection("claude", mode="subscription", refresh=True).status == "unknown"


@pytest.mark.parametrize(
    "output,state",
    [
        ("Logged in using ChatGPT", "connected"),
        ("Logged in using an API key: sk-secret-value", "conflict"),
        ("Not logged in", "signed_out"),
        ("unexpected secret", "unknown"),
    ],
)
def test_codex_status_is_sanitized(probe, output, state):
    probe(output, provider="codex")
    status = cli.get_cli_connection("codex", mode="subscription", refresh=True)
    assert status.status == state
    assert "secret" not in json.dumps(status.wire())


def test_codex_configuration_conflict(probe, tmp_path):
    (tmp_path / "config.toml").write_text('[model_providers.openai]\nenv_key="CUSTOM_KEY"\n')
    probe("Logged in using ChatGPT", provider="codex")
    assert cli.get_cli_connection("codex", mode="subscription", refresh=True).status == "conflict"


def test_cache_and_refresh(probe):
    runner = probe()
    first = cli.get_cli_connection("claude", mode="subscription")
    assert cli.get_cli_connection("claude", mode="subscription") == first
    assert runner.call_count == 2
    probe("none")
    assert (
        cli.get_cli_connection("claude", mode="subscription", refresh=True).status == "signed_out"
    )
    assert runner.call_count == 4


def test_subscription_child_environment_is_isolated(monkeypatch):
    for key in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "CODEX_API_KEY",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.setenv(key, "secret")
    env = cli.cli_environment("claude", "subscription")
    assert not any(key in env for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "CODEX_API_KEY"))
    assert env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1"
    assert cli.cli_environment("claude", "existing")["ANTHROPIC_API_KEY"] == "secret"


def test_execution_refresh_and_fable_opt_in(probe, monkeypatch):
    import kiss.core.vscode_config as vc

    monkeypatch.setattr(vc, "load_config", lambda: {"allow_fable_usage_credits": False})
    probe()
    with pytest.raises(KISSError, match="usage credits"):
        cli.prepare_cli("cc/fable", {"cli_billing_mode": "subscription"})
    probe()
    with pytest.raises(KISSError, match="usage credits"):
        cli.prepare_cli(
            "cc/fable", {"cli_billing_mode": "subscription", "allow_fable_usage_credits": True}
        )
    monkeypatch.setattr(vc, "load_config", lambda: {"allow_fable_usage_credits": True})
    probe()
    env, args = cli.prepare_cli(
        "cc/fable", {"cli_billing_mode": "subscription", "allow_fable_usage_credits": True}
    )
    assert env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1" and args == []
    probe("none")
    with pytest.raises(KISSError, match="Sign in"):
        cli.prepare_cli("cc/opus", {"cli_billing_mode": "subscription"})


def test_codex_execution_pins_subscription_provider(probe):
    probe("Logged in using ChatGPT", provider="codex")
    _, args = cli.prepare_cli("codex/gpt-6.1-sol", {"cli_billing_mode": "subscription"})
    assert 'forced_login_method="chatgpt"' in args
    assert 'model_provider="openai"' in args


@pytest.mark.parametrize(
    "name,config",
    [
        ("gpt-6.1-sol-medium", {}),
        ("cc/opus", {"base_url": "https://example.com"}),
        ("codex/gpt-6.1-sol", {"api_key": "secret"}),
    ],
)
def test_subscription_cannot_construct_api_adapter(name, config):
    from kiss.core.models.model_info import model

    with cli.subscription_scope(True), pytest.raises(KISSError, match="Subscription mode"):
        model(name, model_config=config)


def test_policy_survives_wrappers_and_teardown(monkeypatch):
    monkeypatch.setattr(
        cli, "get_cli_connection", lambda *a, **kw: cli.CLIConnection("claude", "connected")
    )
    observed = []

    class Agent:
        @staticmethod
        def _resolve_model_name(name):
            return name or "cc/opus"

        @cli.subscription_run
        def run(self, prompt_template="", **kwargs):
            observed.append((cli.subscription_only(), kwargs["model_config"]))
            with cli.subscription_scope(False):
                assert cli.subscription_only()
            with pytest.raises(KISSError):
                cli.enforce_model_policy("gpt-6-luna")

    Agent().run()
    assert observed[0][0] is True and observed[0][1]["subscription_only"] is True
    assert not cli.subscription_only()


def test_children_cannot_weaken_billing_policy():
    from kiss.agents.sorcar.agent_dispatch import RunOptions, inherit_from_parent

    parent = SimpleNamespace(
        model_name="cc/opus", model_config={"subscription_only": True, "timeout": 7}
    )
    got = inherit_from_parent(parent, "", None, RunOptions())
    assert got.options.model_config["timeout"] == 7
    assert got.options.model_config["subscription_only"] is True
    with pytest.raises(KISSError):
        inherit_from_parent(parent, "gpt-6-luna", None, RunOptions())


def test_optional_commit_helper_never_uses_api_in_subscription_task(monkeypatch):
    from kiss.agents.sorcar import commit_message as cm
    from kiss.core.models import model_info

    api = Mock(side_effect=AssertionError("API adapter constructed"))
    monkeypatch.setattr(cm, "get_fast_model", lambda: "gpt-6-luna")
    monkeypatch.setattr(model_info, "OpenAICompatibleModel", api, raising=False)
    with cli.subscription_scope(True):
        assert cm.generate_commit_message_from_diff("diff") == "kiss: auto-commit agent work"
    api.assert_not_called()


def test_classifier_skipped_for_subscription(monkeypatch):
    from kiss.agents.sorcar import task_classifier as tc

    monkeypatch.setattr(
        cli, "get_cli_connection", lambda *a, **kw: cli.CLIConnection("claude", "connected")
    )
    result = tc.classify_task("write code", "cc/opus", {"cli_billing_mode": "subscription"})
    assert result.classification is None


def test_subscription_memory_embeddings_stay_offline(monkeypatch):
    from kiss.core.memoryfield.index import default_embedder, hashed_embedding

    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    with cli.subscription_scope(True):
        assert default_embedder() is hashed_embedding


def test_fresh_and_legacy_billing_settings(monkeypatch, tmp_path):
    from kiss.core import vscode_config as vc

    monkeypatch.setattr(vc, "_config_path", lambda: tmp_path / "config.json")
    assert vc.load_config()["claude_cli_billing_mode"] == "subscription"
    vc.save_config({"last_model": "cc/opus"})
    stored = json.loads((tmp_path / "config.json").read_text())
    assert stored["claude_cli_billing_mode"] == stored["codex_cli_billing_mode"] == "subscription"
    assert vc.load_config()["claude_cli_billing_mode"] == "subscription"
    (tmp_path / "config.json").write_text('{"last_model": "cc/opus"}')
    assert vc.load_config()["claude_cli_billing_mode"] == ""
    vc.save_config({"work_dir": ""})
    assert vc.load_config()["claude_cli_billing_mode"] == ""
    assert (
        vc.sanitize_config({"allow_fable_usage_credits": "true"})["allow_fable_usage_credits"]
        is False
    )


@pytest.mark.parametrize(
    "key,expected,fast",
    [
        ("ANTHROPIC_API_KEY", "claude-opus-5-5", "claude-haiku-4-5-20251001"),
        ("OPENAI_API_KEY", "gpt-6.1-sol-medium", "gpt-6-luna"),
        ("GEMINI_API_KEY", "gemini-3.8-flash", "gemini-3.5-flash-lite"),
    ],
)
def test_api_defaults_and_lightweight_helpers(monkeypatch, key, expected, fast):
    from kiss.core import config
    from kiss.core.models.model_info import get_default_model, get_fast_model

    keys = {key: "fake"}
    monkeypatch.setattr(
        config,
        "DEFAULT_CONFIG",
        SimpleNamespace(
            **{
                name: keys.get(name, "")
                for name in (
                    "ANTHROPIC_API_KEY",
                    "OPENAI_API_KEY",
                    "GEMINI_API_KEY",
                    "OPENROUTER_API_KEY",
                    "TOGETHER_API_KEY",
                )
            }
        ),
    )
    assert get_default_model() == expected
    assert get_fast_model() == fast


def test_subscriptions_precede_api_defaults(monkeypatch):
    from kiss.core.models.model_info import get_default_model

    monkeypatch.setattr(
        cli, "get_cli_connection", lambda provider, **kw: cli.CLIConnection(provider, "connected")
    )
    assert get_default_model() == "cc/opus"
    monkeypatch.setattr(
        cli,
        "get_cli_connection",
        lambda provider, **kw: cli.CLIConnection(
            provider, "connected" if provider == "codex" else "missing"
        ),
    )
    assert get_default_model() == "codex/gpt-6.1-sol"


def test_effort_forwarding_and_responses_routing(monkeypatch):
    import kiss.core.models.claude_code_model as claude
    import kiss.core.models.codex_model as codex
    from kiss.core.models.claude_code_model import ClaudeCodeModel
    from kiss.core.models.codex_model import CodexModel
    from kiss.core.models.model_info import MODEL_INFO

    monkeypatch.setattr(codex, "_find_codex_cli", lambda: "/fake/codex")
    monkeypatch.setattr(claude, "_find_claude_cli", lambda: "/fake/claude")
    assert (
        'model_reasoning_effort="max"'
        in CodexModel(
            "codex/gpt-6.1-sol", model_config={"reasoning_effort": "max"}
        )._build_cli_args()
    )
    assert (
        "--effort"
        in ClaudeCodeModel("cc/opus", model_config={"reasoning_effort": "max"})._build_cli_args()
    )
    assert MODEL_INFO["gpt-6.1-sol-max"].use_responses_api is True


def test_subscription_fallback_is_disabled(monkeypatch):
    from kiss.core.kiss_agent import KISSAgent
    from kiss.core.models import model_info

    fallback = Mock(side_effect=AssertionError("fallback consulted"))
    monkeypatch.setattr(model_info, "get_fallback_model", fallback)
    agent = KISSAgent("test")
    agent._model_config = {"subscription_only": True}
    assert agent._try_switch_to_fallback() is None
    fallback.assert_not_called()


def test_deferred_operations_keep_parent_policy():
    class Agent:
        model_config = {"subscription_only": True}

        @cli.subscription_operation
        def merge(self):
            cli.enforce_model_policy("gpt-6-luna")

    with pytest.raises(KISSError):
        Agent().merge()
    assert not cli.subscription_only()


def test_logged_out_claude_record_cannot_claim_subscription(probe):
    runner = probe()
    runner.side_effect = [
        subprocess.CompletedProcess([], 0, "2.1.300", ""),
        subprocess.CompletedProcess([], 0, '{"authMethod":"claude.ai","loggedIn":false}', ""),
    ]
    assert (
        cli.get_cli_connection("claude", mode="subscription", refresh=True).status == "signed_out"
    )


def test_cached_api_adapter_cannot_make_a_subscription_request(monkeypatch):
    from kiss.core.models.openai_compatible_model import OpenAICompatibleModel

    adapter = OpenAICompatibleModel("gpt-4o", api_key="dummy", base_url="https://api.openai.com/v1")
    adapter.client = Mock(side_effect=AssertionError("API used"))
    with cli.subscription_scope(True), pytest.raises(KISSError, match="Subscription mode"):
        adapter.get_embedding("text")
    adapter.client.embeddings.create.assert_not_called()


def test_audio_transcription_is_skipped_for_subscription(monkeypatch):
    import openai

    from kiss.core.models.model import transcribe_audio

    api = Mock(side_effect=AssertionError("API constructed"))
    monkeypatch.setattr(openai, "OpenAI", api)
    with cli.subscription_scope(True), pytest.raises(ValueError, match="API task"):
        transcribe_audio(b"audio", "audio/mpeg", "secret")
    api.assert_not_called()


def test_codex_legacy_custom_provider_is_retained(probe, tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        'model_provider="custom"\n[model_providers.custom]\nenv_key="CUSTOM_API_KEY"\n'
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "secret")
    probe("Logged in using ChatGPT", provider="codex")
    status = cli.get_cli_connection("codex", mode="", refresh=True)
    assert (status.status, status.billing_mode, status.auth_method) == (
        "connected",
        "existing",
        "third_party",
    )


def test_codex_old_version_is_not_used_for_subscription(probe):
    probe("Logged in using ChatGPT", version="0.100.0", provider="codex")
    assert (
        cli.get_cli_connection("codex", mode="subscription", refresh=True).status
        == "unsupported_version"
    )


def test_malformed_codex_profiles_fail_closed(probe, tmp_path):
    (tmp_path / "config.toml").write_text('profiles="malformed"')
    probe("Logged in using ChatGPT", provider="codex")
    assert cli.get_cli_connection("codex", mode="", refresh=True).status == "unknown"


def test_max_alias_preserves_native_id_and_responses_parameters():
    from kiss.core.models.model_info import model

    adapter = model(
        "gpt-6.1-sol-max",
        model_config={"base_url": "https://api.openai.com/v1", "api_key": "dummy"},
    )
    adapter.initialize("test")
    request = adapter._build_request_kwargs(tools=None)
    assert request["model"] == "gpt-6.1-sol"
    assert request["reasoning"]["effort"] == "max"


def test_native_child_client_cannot_submit_api_task(monkeypatch):
    from kiss.agents.sorcar import daemon_client

    monkeypatch.setenv("KISS_SUBSCRIPTION_ONLY", "1")
    with pytest.raises(KISSError, match="Subscription mode"):
        daemon_client.run("review", model="gpt-6-luna")


def test_cli_child_environment_carries_task_policy(probe):
    probe("Logged in using ChatGPT", provider="codex")
    env, _ = cli.prepare_cli("codex/gpt-6.1-sol", {"cli_billing_mode": "subscription"})
    assert env["KISS_SUBSCRIPTION_ONLY"] == "1"
    assert env["KISS_SUBSCRIPTION_MODEL"] == "codex/gpt-6.1-sol"


def test_claude_native_children_cannot_select_fable_without_saved_opt_in(probe, monkeypatch):
    from kiss.core import vscode_config

    probe()
    monkeypatch.setattr(vscode_config, "load_config", lambda: {"allow_fable_usage_credits": False})
    monkeypatch.setenv("CLAUDE_CODE_SUBAGENT_MODEL", "fable")
    env, args = cli.prepare_cli("cc/opus", {"cli_billing_mode": "subscription"})
    assert env["CLAUDE_CODE_SUBAGENT_MODEL"] == "opus"
    assert env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    settings_env = json.loads(args[args.index("--settings") + 1])["env"]
    assert settings_env["CLAUDE_CODE_SUBAGENT_MODEL"] == "opus"
    assert settings_env["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"


def test_native_child_client_forwards_inherited_subscription_policy(monkeypatch):
    from kiss.agents.sorcar import daemon_client

    monkeypatch.setenv("KISS_SUBSCRIPTION_ONLY", "1")
    monkeypatch.setenv("KISS_SUBSCRIPTION_MODEL", "codex/gpt-6.1-sol")
    monkeypatch.setattr(daemon_client, "resolve_agent_path", lambda path: "")
    monkeypatch.setattr(daemon_client, "_resolve_endpoint_file", lambda path: "fake")
    ws = Mock()
    sent = []

    def send(payload):
        cmd = json.loads(payload)
        sent.append(cmd)
        if cmd["type"] == "run":
            ws.recv.side_effect = [
                json.dumps({"type": "result", "tabId": cmd["tabId"], "success": True}),
                json.dumps({"type": "status", "tabId": cmd["tabId"], "running": False}),
            ]

    ws.send.side_effect = send
    monkeypatch.setattr(daemon_client.local_endpoint, "connect", lambda *args, **kwargs: ws)
    daemon_client.run("review", model_config={"cli_billing_mode": "existing"})
    run = next(cmd for cmd in sent if cmd["type"] == "run")
    assert run["model"] == "codex/gpt-6.1-sol"
    assert run["modelConfig"]["subscription_only"] is True
    assert run["modelConfig"]["cli_billing_mode"] == "subscription"


def test_migration_ignores_sorcar_api_keys(probe, monkeypatch):
    """Sorcar's own key store must not make a legacy CLI look API-billed."""
    from kiss.core import vscode_config

    saved = {}
    monkeypatch.setattr(vscode_config, "save_config", saved.update)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sorcar-key")
    runner = probe("claude.ai")
    status = cli.get_cli_connection("claude", mode="", refresh=True)
    assert (status.status, status.billing_mode) == ("connected", "subscription")
    assert saved == {"claude_cli_billing_mode": "subscription"}
    for call in runner.call_args_list:
        assert "ANTHROPIC_API_KEY" not in call.args[1]


def test_migration_keeps_cli_owned_api_configuration(probe, monkeypatch):
    from kiss.core import vscode_config

    saved = {}
    monkeypatch.setattr(vscode_config, "save_config", saved.update)
    probe("api_key_helper")
    status = cli.get_cli_connection("claude", mode="", refresh=True)
    assert (status.status, status.billing_mode) == ("connected", "existing")
    assert saved == {"claude_cli_billing_mode": "existing"}


def test_settings_files_cannot_remap_claude_aliases_to_fable(probe, monkeypatch):
    from kiss.core import vscode_config

    monkeypatch.setattr(vscode_config, "load_config", lambda: {})
    probe()
    _, args = cli.prepare_cli("cc/opus", {"cli_billing_mode": "subscription"})
    settings = json.loads(args[args.index("--settings") + 1])
    for name in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
    ):
        assert settings["env"][name] == ""  # unsets a settings-file remap
    assert settings["env"]["CLAUDE_CODE_DISABLE_ADVISOR_TOOL"] == "1"
    assert settings["fallbackModel"] == []
    assert settings["modelOverrides"] == {}


def test_existing_claude_plan_login_still_requires_fable_consent(probe, monkeypatch):
    from kiss.core import vscode_config

    monkeypatch.setattr(vscode_config, "load_config", lambda: {})
    monkeypatch.setenv("ANTHROPIC_DEFAULT_OPUS_MODEL", "claude-fable-5-1")
    probe("claude.ai")
    with pytest.raises(KISSError, match="usage credits"):
        cli.prepare_cli("cc/claude-fable-5-1", {"cli_billing_mode": "existing"})
    probe("claude.ai")
    env, args = cli.prepare_cli("cc/opus", {"cli_billing_mode": "existing"})
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
    assert "--settings" in args
    assert "KISS_SUBSCRIPTION_ONLY" not in env


def test_existing_claude_api_billing_has_no_credit_gate(probe, monkeypatch):
    from kiss.core import vscode_config

    monkeypatch.setattr(vscode_config, "load_config", lambda: {})
    probe("api_key")
    env, args = cli.prepare_cli("cc/claude-fable-5-1", {"cli_billing_mode": "existing"})
    assert args == []


def test_fable_consent_allows_settings_customization(probe, monkeypatch):
    from kiss.core import vscode_config

    monkeypatch.setattr(vscode_config, "load_config", lambda: {"allow_fable_usage_credits": True})
    probe()
    _, args = cli.prepare_cli("cc/opus", {"cli_billing_mode": "subscription"})
    assert args == []


@pytest.mark.parametrize(
    "files",
    [
        {"config.toml": 'openai_base_url="https://proxy.example/v1"\n'},
        {"config.toml": 'profile="work"\n', "work.config.toml": 'openai_base_url="https://p/v1"\n'},
        {
            "config.toml": 'profile="work"\n[profiles.work]\nopenai_base_url="https://p/v1"\n',
        },
        {"config.toml": 'profile="work"\n', "work.config.toml": "[model_providers.openai]\nx=1\n"},
    ],
)
def test_codex_subscription_rejects_builtin_provider_rerouting(probe, tmp_path, files):
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    probe("Logged in using ChatGPT", provider="codex")
    status = cli.get_cli_connection("codex", mode="subscription", refresh=True)
    assert status.status == "conflict"
    assert "openai_base_url" in status.message


def test_codex_profile_file_custom_provider_is_detected(probe, tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('profile="work"\n')
    (tmp_path / "work.config.toml").write_text(
        'model_provider="custom"\n[model_providers.custom]\nenv_key="CUSTOM_API_KEY"\n'
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "secret")
    probe("Logged in using ChatGPT", provider="codex")
    status = cli.get_cli_connection("codex", mode="existing", refresh=True)
    assert (status.status, status.auth_method) == ("connected", "third_party")


def test_codex_profile_name_cannot_escape_codex_home(probe, tmp_path):
    (tmp_path / "config.toml").write_text('profile="../outside"\n')
    probe("Logged in using ChatGPT", provider="codex")
    assert cli.get_cli_connection("codex", mode="existing", refresh=True).status == "unknown"


def test_api_task_helpers_keep_api_priority_over_subscriptions(monkeypatch):
    """The commit-message button must stay a plain API call when keys exist."""
    from kiss.core import config
    from kiss.core.models.model_info import get_default_model, get_fast_model

    monkeypatch.setattr(
        config,
        "DEFAULT_CONFIG",
        SimpleNamespace(
            ANTHROPIC_API_KEY="fake",
            OPENAI_API_KEY="",
            GEMINI_API_KEY="",
            OPENROUTER_API_KEY="",
            TOGETHER_API_KEY="",
        ),
    )
    monkeypatch.setattr(
        cli, "get_cli_connection", lambda provider, **kw: cli.CLIConnection(provider, "connected")
    )
    assert get_fast_model() == "claude-haiku-4-5-20251001"
    assert get_default_model() == "cc/opus"
    with cli.subscription_scope(True):
        assert get_fast_model() == "cc/haiku"
    monkeypatch.setattr(
        cli,
        "get_cli_connection",
        lambda provider, **kw: cli.CLIConnection(
            provider, "connected" if provider == "codex" else "missing"
        ),
    )
    with cli.subscription_scope(True):
        assert get_fast_model() == "codex/gpt-6-luna"


def test_claude_adapter_launches_with_subscription_isolation(probe, monkeypatch):
    """Exercise the real adapter turn, not a stubbed ``prepare_cli``."""
    import kiss.core.models.claude_code_model as claude
    from kiss.core import vscode_config
    from kiss.core.models import model as model_module

    monkeypatch.setattr(vscode_config, "load_config", lambda: {})
    monkeypatch.setattr(claude, "_find_claude_cli", lambda: "/fake/claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-leak")
    launched = {}

    class FakeProcess:
        def __init__(self, args, label, timeout, cwd=None, env=None):
            launched.update(args=args, env=env)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(model_module, "_CLIProcess", FakeProcess)
    probe()
    adapter = claude.ClaudeCodeModel("cc/opus", model_config={"cli_billing_mode": "subscription"})
    with adapter._cli_turn(adapter._build_cli_args(), "Claude Code"):
        pass
    args, env = launched["args"], launched["env"]
    assert args[0] == "/fake/claude" and args[1] == "--settings"
    assert args[args.index("--model") + 1] == "opus"
    assert "ANTHROPIC_API_KEY" not in env
    assert env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] == "1"
    assert env["KISS_SUBSCRIPTION_ONLY"] == "1"


def test_codex_adapter_places_pins_before_exec(probe, monkeypatch):
    import kiss.core.models.codex_model as codex
    from kiss.core.models import model as model_module

    monkeypatch.setattr(codex, "_find_codex_cli", lambda: "/fake/codex")
    launched = {}

    class FakeProcess:
        def __init__(self, args, label, timeout, cwd=None, env=None):
            launched.update(args=args)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(model_module, "_CLIProcess", FakeProcess)
    probe("Logged in using ChatGPT", provider="codex")
    adapter = codex.CodexModel(
        "codex/gpt-6.1-sol", model_config={"cli_billing_mode": "subscription"}
    )
    with adapter._cli_turn(adapter._build_cli_args(), "Codex"):
        pass
    args = launched["args"]
    assert args[0] == "/fake/codex"
    assert args.index('model_provider="openai"') < args.index("exec")

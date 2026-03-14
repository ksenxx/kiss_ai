"""Tests for Codex auth routing for OpenAI-family models."""

from kiss.core import config as config_module
from kiss.core.models import model_info


def test_codex_subscription_model_allowlist():
    assert model_info._is_codex_subscription_model("gpt-5.4")
    assert model_info._is_codex_subscription_model("gpt-5.3-codex")
    assert model_info._is_codex_subscription_model("gpt-5.3-codex-spark")
    assert model_info._is_codex_subscription_model("gpt-5.2")
    assert not model_info._is_codex_subscription_model("gpt-4.1-mini")


def test_codex_provider_catalog_allowlist():
    expected = {
        "gpt-5.4",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2-codex",
        "gpt-5.1-codex-max",
        "gpt-5.2",
        "gpt-5.1-codex-mini",
    }
    for model in expected:
        assert model_info.is_codex_provider_model(model)
    assert not model_info.is_codex_provider_model("gpt-5-mini")


def test_resolve_openai_auth_mode_prefers_codex_models_when_available(monkeypatch):
    monkeypatch.delenv("KISS_OPENAI_AUTH", raising=False)
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    mode = model_info._resolve_openai_auth_mode("gpt-5.3-codex", "sk-test")
    assert mode == "codex"


def test_resolve_openai_auth_mode_uses_api_for_non_codex_when_key_present(monkeypatch):
    monkeypatch.delenv("KISS_OPENAI_AUTH", raising=False)
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    mode = model_info._resolve_openai_auth_mode("gpt-4.1-mini", "sk-test")
    assert mode == "api"


def test_resolve_openai_auth_mode_uses_codex_without_key(monkeypatch):
    monkeypatch.delenv("KISS_OPENAI_AUTH", raising=False)
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    mode = model_info._resolve_openai_auth_mode("gpt-5-mini", "")
    assert mode == "codex"


def test_resolve_openai_auth_mode_uses_api_without_key_for_api_only_model(monkeypatch):
    monkeypatch.delenv("KISS_OPENAI_AUTH", raising=False)
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    mode = model_info._resolve_openai_auth_mode("gpt-4.1-mini", "")
    assert mode == "api"


def test_resolve_openai_auth_mode_respects_forced_env(monkeypatch):
    monkeypatch.setenv("KISS_OPENAI_AUTH", "api")
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    mode = model_info._resolve_openai_auth_mode("gpt-5.3-codex", "")
    assert mode == "api"


def test_resolve_codex_transport_prefers_native(monkeypatch):
    monkeypatch.delenv("KISS_CODEX_TRANSPORT", raising=False)
    monkeypatch.setattr(model_info, "_is_codex_native_auth_available", lambda: True)
    assert model_info._resolve_codex_transport() == "native"


def test_resolve_codex_transport_forced_native_falls_back_to_cli(monkeypatch):
    monkeypatch.setenv("KISS_CODEX_TRANSPORT", "native")
    monkeypatch.setattr(model_info, "_is_codex_native_auth_available", lambda: False)
    assert model_info._resolve_codex_transport() == "cli"


def test_resolve_codex_transport_forced_cli(monkeypatch):
    monkeypatch.setenv("KISS_CODEX_TRANSPORT", "cli")
    monkeypatch.setattr(model_info, "_is_codex_native_auth_available", lambda: True)
    assert model_info._resolve_codex_transport() == "cli"


def test_model_openai_routing_prefers_native_codex_backend(monkeypatch):
    monkeypatch.setattr(model_info, "_resolve_openai_auth_mode", lambda *_args: "codex")
    monkeypatch.setattr(model_info, "_resolve_codex_transport", lambda: "native")
    monkeypatch.setattr(model_info, "_is_codex_native_auth_available", lambda: True)

    sentinel = object()
    monkeypatch.setattr(model_info, "_codex_native", lambda *_args, **_kwargs: sentinel)

    m = model_info.model("gpt-4.1-mini")
    assert m is sentinel


def test_model_openai_routing_falls_back_to_cli_when_native_missing(monkeypatch):
    monkeypatch.setattr(model_info, "_resolve_openai_auth_mode", lambda *_args: "codex")
    monkeypatch.setattr(model_info, "_resolve_codex_transport", lambda: "native")
    monkeypatch.setattr(model_info, "_is_codex_native_auth_available", lambda: False)
    monkeypatch.setattr(model_info, "_is_codex_cli_auth_available", lambda: True)

    sentinel = object()
    monkeypatch.setattr(model_info, "_codex_cli", lambda *_args, **_kwargs: sentinel)

    m = model_info.model("gpt-4.1-mini")
    assert m is sentinel


def test_get_available_models_includes_openai_with_codex_auth(monkeypatch):
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: True)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG.agent.api_keys, "OPENAI_API_KEY", "")
    names = model_info.get_available_models()
    assert "gpt-4.1-mini" not in names
    assert "gpt-5.4" in names
    assert "gpt-5.3-codex" in names


def test_get_available_models_excludes_openai_without_key_or_codex(monkeypatch):
    monkeypatch.setattr(model_info, "_is_codex_auth_available", lambda: False)
    monkeypatch.setattr(config_module.DEFAULT_CONFIG.agent.api_keys, "OPENAI_API_KEY", "")
    names = model_info.get_available_models()
    assert "gpt-4.1-mini" not in names

"""CLI authentication, billing isolation, and task-level subscription policy.

Credentials stay in the official CLIs. Status output is parsed here and never
returned verbatim: even ``codex login status`` can contain an API-key fragment.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import threading
import time
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from kiss.core.kiss_error import KISSError

_BILLING_KEYS = {"claude": "claude_cli_billing_mode", "codex": "codex_cli_billing_mode"}
_SUBSCRIPTION: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "kiss_subscription_only", default=False
)
_CACHE: dict[tuple[str, str, str, str], tuple[float, CLIConnection]] = {}
_LOCK = threading.Lock()
_TTL = 15.0
_TIMEOUT = 4.0
#: Claude Code model-selection variables. Settings files can remap an alias
#: such as ``opus`` to Fable through these, so they are unset for credit-gated runs.
_CLAUDE_MODEL_VARIABLES = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
)
#: Claude logins billed to a subscription plan, where Fable can use usage credits.
_CLAUDE_PLAN_LOGINS = frozenset({"claude.ai", "oauth_token"})
_CODEX_PROFILE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class CLIConnection:
    """Sanitized connection information safe to expose to a client."""

    provider: str
    status: str
    version: str = ""
    auth_method: str = "unknown"
    billing_mode: str = "subscription"
    message: str = ""

    def wire(self) -> dict[str, Any]:
        """Return public fields, without credentials or raw CLI output."""
        return asdict(self)


def cli_provider(model_name: str) -> str | None:
    """Return the CLI serving a model, or None for an API model."""
    if model_name.startswith("cc/"):
        return "claude"
    if model_name.startswith("codex/"):
        return "codex"
    return None


def subscription_only() -> bool:
    """Whether the current task forbids Sorcar-initiated API model calls."""
    return _SUBSCRIPTION.get() or os.environ.get("KISS_SUBSCRIPTION_ONLY") == "1"


@contextlib.contextmanager
def subscription_scope(required: bool) -> Iterator[None]:
    """Bind a policy without weakening an enclosing task's policy."""
    token = _SUBSCRIPTION.set(required or subscription_only())
    try:
        yield
    finally:
        _SUBSCRIPTION.reset(token)


def subscription_operation[F: Callable[..., Any]](func: F) -> F:
    """Retain a finished task's billing policy during deferred worktree operations."""

    @functools.wraps(func)
    def wrapped(agent: Any, *args: Any, **kwargs: Any) -> Any:
        required = (getattr(agent, "model_config", None) or {}).get("subscription_only") is True
        with subscription_scope(required):
            return func(agent, *args, **kwargs)

    return cast(F, wrapped)


def cli_environment(provider: str, mode: str) -> dict[str, str]:
    """Copy the daemon environment; isolate subscription credentials per child."""
    env = dict(os.environ)
    if mode != "subscription":
        return env
    for name in tuple(env):
        if name.startswith(("ANTHROPIC_", "CLAUDE_CODE_USE_", "OPENAI_")) or name in {
            "CODEX_API_KEY",
            "CODEX_ACCESS_TOKEN",
            "CLAUDE_CODE_API_KEY_HELPER",
            "CLAUDE_CODE_SIMPLE",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_REFRESH_TOKEN",
            "CLAUDE_CODE_OAUTH_SCOPES",
            "CLAUDE_CODE_SKIP_AUTH",
            "CLAUDE_CODE_BASE_URL",
        }:
            env.pop(name, None)
    env["KISS_SUBSCRIPTION_ONLY"] = "1"
    if provider == "claude":
        # Official host-routing flag: settings-file provider/auth variables
        # cannot reintroduce API keys or an alternate provider into this child.
        env["CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST"] = "1"
    return env


def _executable(provider: str) -> str | None:
    if provider == "codex":
        from kiss.core.models.codex_model import find_codex_executable

        return find_codex_executable()
    return shutil.which("claude")


def _run_status(
    args: list[str], env: dict[str, str], cwd: str | None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=_TIMEOUT,
        check=False,
    )


def _probe(provider: str, mode: str, cwd: str | None) -> CLIConnection:
    binary = _executable(provider)
    if not binary:
        return CLIConnection(
            provider,
            "missing",
            billing_mode=mode or "subscription",
            message="Install the CLI on the Sorcar server machine.",
        )
    env = cli_environment(provider, mode)
    if mode == "":
        # Migration detects the CLI's own configuration. Keys from Sorcar's
        # key store are also in the daemon environment, but they configure
        # Sorcar's API models, not an intentional CLI billing choice.
        from kiss.core.vscode_config import API_KEY_ENV_VARS

        for name in API_KEY_ENV_VARS:
            env.pop(name, None)
    try:
        version_result = _run_status([binary, "--version"], env, cwd)
        match = re.search(r"\b(\d+\.\d+\.\d+)\b", version_result.stdout)
        version = match.group(1) if match else ""
        args = (
            [binary, "auth", "status", "--json"]
            if provider == "claude"
            else [binary, "login", "status"]
        )
        result = _run_status(args, env, cwd)
        method = "unknown"
        if provider == "claude":
            record = json.loads(result.stdout)
            if not isinstance(record, dict):
                raise ValueError("Invalid authentication status")
            raw_method = record.get("authMethod")
            if isinstance(raw_method, str) and raw_method in {
                "none",
                "claude.ai",
                "oauth_token",
                "api_key",
                "api_key_helper",
                "third_party",
            }:
                method = raw_method
            if record.get("loggedIn") is False:
                method = "none"
        else:
            output = (result.stdout + result.stderr).lower()
            if "logged in using chatgpt" in output and "api key" in output:
                method = "unknown"
            elif "logged in using chatgpt" in output:
                method = "chatgpt"
            elif "api key" in output and "logged in" in output:
                method = "api_key"
            elif "not logged in" in output:
                method = "none"
        if provider == "codex" and mode != "subscription":
            alternate, credentials = _codex_existing_provider()
            if alternate:
                method = (
                    "third_party" if credentials or method in {"chatgpt", "api_key"} else "unknown"
                )
        effective_mode = mode or (
            "existing"
            if method in {"api_key", "api_key_helper", "third_party", "oauth_token"}
            else "subscription"
        )
        if (
            provider == "codex"
            and effective_mode == "subscription"
            and (not version or tuple(map(int, version.split("."))) < (0, 162, 0))
        ):
            return CLIConnection(
                provider,
                "unsupported_version",
                version,
                method,
                effective_mode,
                "Update Codex to 0.162.0 or newer for pinned built-in subscription routing.",
            )
        # The host-routing flag is required to isolate subscription calls.
        # Existing billing configurations may continue using older versions.
        if (
            provider == "claude"
            and effective_mode == "subscription"
            and (not version or tuple(map(int, version.split("."))) < (2, 1, 280))
        ):
            return CLIConnection(
                provider,
                "unsupported_version",
                version,
                method,
                effective_mode,
                "Update Claude Code to 2.1.280 or newer for subscription isolation and Opus 5.5.",
            )
        is_subscription = method in {"claude.ai", "chatgpt"}
        if method == "none":
            state, message = (
                "signed_out",
                "Sign in on the Sorcar server machine, then refresh status.",
            )
        elif method == "unknown" or (result.returncode != 0 and method != "third_party"):
            state, message = (
                "unknown",
                "Authentication could not be verified. "
                "Refresh status or check the CLI in a terminal.",
            )
        elif effective_mode == "subscription" and not is_subscription:
            state, message = (
                "conflict",
                "This CLI uses API or enterprise credentials. "
                "Sign in with a subscription or choose Existing CLI configuration.",
            )
        else:
            state, message = (
                "connected",
                (
                    "Uses your plan, subject to provider limits and extra-usage settings."
                    if is_subscription
                    else "Uses the CLI's existing billing configuration."
                ),
            )
        if provider == "codex" and effective_mode == "subscription" and state == "connected":
            _check_codex_configuration()
        return CLIConnection(provider, state, version, method, effective_mode, message)
    except KISSError:
        return CLIConnection(
            provider,
            "conflict",
            billing_mode=mode or "subscription",
            message="Codex overrides its built-in OpenAI provider. "
            "Remove the openai provider definition or openai_base_url from config.toml "
            "and its selected profile, or choose Existing CLI configuration.",
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return CLIConnection(
            provider,
            "unknown",
            billing_mode=mode or "subscription",
            message="CLI status timed out or could not be read. "
            "Check the CLI in a terminal and refresh.",
        )


def _merge_config(base: dict[str, Any], layer: dict[str, Any]) -> dict[str, Any]:
    """Overlay one TOML layer on another, merging nested tables."""
    merged = dict(base)
    for key, value in layer.items():
        current = merged.get(key)
        merged[key] = (
            _merge_config(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    return merged


def _codex_config() -> dict[str, Any]:
    """Return user-level Codex configuration with its default profile applied.

    A profile is either a legacy ``[profiles.<name>]`` table or a
    ``$CODEX_HOME/<name>.config.toml`` file. Project-scoped configuration
    cannot change provider or authentication keys, so it is not read.
    """
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    config_path = home / "config.toml"
    if not config_path.exists():
        return {}
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profile_name = config.get("profile", "")
    profiles = config.get("profiles", {})
    providers = config.get("model_providers", {})
    if (
        not isinstance(profile_name, str)
        or not isinstance(profiles, dict)
        or not isinstance(providers, dict)
    ):
        raise ValueError("Invalid Codex configuration")
    if not profile_name:
        return config
    if not _CODEX_PROFILE_NAME.fullmatch(profile_name) or ".." in profile_name:
        raise ValueError("Invalid Codex profile")
    profile = profiles.get(profile_name, {})
    if not isinstance(profile, dict):
        raise ValueError("Invalid Codex profile")
    effective = _merge_config(config, profile)
    profile_path = home / f"{profile_name}.config.toml"
    if profile_path.is_file():
        effective = _merge_config(
            effective, tomllib.loads(profile_path.read_text(encoding="utf-8"))
        )
    if not isinstance(effective.get("model_providers", {}), dict):
        raise ValueError("Invalid Codex configuration")
    return effective


def _codex_existing_provider() -> tuple[bool, bool]:
    """Detect intentional custom-provider billing without exposing its credentials."""
    effective = _codex_config()
    providers = effective.get("model_providers", {})
    name = effective.get("model_provider", "openai")
    if name == "openai":
        return False, False
    if not isinstance(name, str):
        raise ValueError("Invalid Codex provider")
    provider = providers.get(name, {})
    if not isinstance(provider, dict):
        raise ValueError("Invalid Codex provider")
    key = provider.get("env_key")
    credentials = bool(
        (isinstance(key, str) and os.environ.get(key))
        or provider.get("experimental_bearer_token")
        or provider.get("http_headers")
        or (not key and not provider.get("requires_openai_auth"))
    )
    return True, credentials


def _check_codex_configuration() -> None:
    """Reject overrides of the built-in provider that would bypass subscription routing.

    The ``-c`` pins select the built-in ``openai`` provider and ChatGPT login,
    but ``openai_base_url`` would still send those requests elsewhere.
    """
    config = _codex_config()
    provider = config.get("model_providers", {}).get("openai", {})
    if not isinstance(provider, dict) or provider or config.get("openai_base_url"):
        raise KISSError("Conflicting Codex provider configuration")


def get_cli_connection(
    provider: str,
    *,
    mode: str | None = None,
    refresh: bool = False,
    cwd: str | None = None,
) -> CLIConnection:
    """Read bounded, briefly cached CLI status without exposing credentials."""
    if provider not in _BILLING_KEYS:
        raise ValueError("Unknown CLI provider")
    if mode is None:
        from kiss.core.vscode_config import load_config

        mode = str(load_config().get(_BILLING_KEYS[provider], ""))
    if mode not in {"", "subscription", "existing"}:
        raise KISSError("CLI billing mode must be subscription or existing.")
    fingerprint = hashlib.sha256(repr(sorted(os.environ.items())).encode()).hexdigest()
    key = (provider, mode, cwd or "", fingerprint)
    with _LOCK:
        cached = _CACHE.get(key)
    if not refresh and cached and time.monotonic() - cached[0] < _TTL:
        return cached[1]
    result = _probe(provider, mode, cwd)
    if mode == "" and result.status == "connected":
        from kiss.core.vscode_config import save_config

        save_config({_BILLING_KEYS[provider]: result.billing_mode})
    with _LOCK:
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = (time.monotonic(), result)
    return result


def billing_mode(model_name: str, config: dict[str, Any] | None = None) -> str:
    """Resolve a CLI mode; an enclosing subscription task always wins."""
    provider = cli_provider(model_name)
    if provider is None:
        return ""
    cfg = config or {}
    if subscription_only() or cfg.get("subscription_only") is True:
        return "subscription"
    explicit = cfg.get("cli_billing_mode")
    if explicit is not None:
        if not isinstance(explicit, str) or explicit not in {"subscription", "existing"}:
            raise KISSError("cli_billing_mode must be subscription or existing.")
        return str(explicit)
    return get_cli_connection(provider).billing_mode


def enforce_model_policy(model_name: str, config: dict[str, Any] | None = None) -> None:
    """Reject an API model or endpoint override in a subscription task."""
    cfg = config or {}
    if subscription_only() or cfg.get("subscription_only") is True:
        if cli_provider(model_name) is None or any(
            cfg.get(key) for key in ("base_url", "api_key", "extra_headers")
        ):
            raise KISSError(
                "Subscription mode cannot use API models or endpoint overrides. "
                "Select an API model explicitly in a separate task."
            )


def prepare_cli(model_name: str, config: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Verify the effective login before every turn and return child isolation."""
    provider = cli_provider(model_name)
    assert provider is not None
    mode = billing_mode(model_name, config)
    status = get_cli_connection(
        provider, mode=mode, refresh=True, cwd=config.get("work_dir") or None
    )
    if status.status != "connected":
        raise KISSError(status.message)
    if mode == "subscription":
        enforce_model_policy(model_name, config | {"subscription_only": True})
    # Fable can bill usage credits on any plan login, including one kept as
    # the Existing CLI configuration; API-key billing has no usage credits.
    credit_gated = provider == "claude" and (
        mode == "subscription" or status.auth_method in _CLAUDE_PLAN_LOGINS
    )
    credits_allowed = False
    if credit_gated:
        from kiss.core.vscode_config import load_config

        credits_allowed = (
            load_config().get("allow_fable_usage_credits") is True
            and config.get("allow_fable_usage_credits") is not False
        )
        if not credits_allowed and may_use_credits(model_name):
            raise KISSError(
                "This Claude model may require usage credits. "
                "Enable the Fable usage-credit opt-in in CLI Connections, or choose cc/opus."
            )
    args = []
    if provider == "codex" and mode == "subscription":
        for value in (
            'forced_login_method="chatgpt"',
            'model_provider="openai"',
            'chatgpt_base_url="https://chatgpt.com/backend-api/"',
        ):
            args.extend(["-c", value])
    env = cli_environment(provider, mode)
    if mode == "subscription":
        env["KISS_SUBSCRIPTION_MODEL"] = model_name
    if credit_gated and not credits_allowed:
        settings = _claude_credit_policy(model_name)
        for name in _CLAUDE_MODEL_VARIABLES:
            env.pop(name, None)
        env.update({name: value for name, value in settings["env"].items() if value})
        args.extend(["--settings", json.dumps(settings)])
    return env, args


def may_use_credits(model_name: str) -> bool:
    """Whether a Claude Code model name can select Fable without remapping."""
    return model_name.startswith("cc/") and any(
        term in model_name.lower() for term in ("fable", "mythos", "/best", "/default")
    )


def _claude_credit_policy(model_name: str) -> dict[str, Any]:
    """Return ``--settings`` that keep a run without Fable consent off Fable.

    ``--settings`` outranks local, project, and user settings files, so a
    repository's ``.claude/settings.json`` cannot remap the selected alias
    (an empty ``env`` value unsets a lower-precedence one), name a Fable
    fallback chain or model override, or call a Fable advisor. Native child
    agents are forced onto the selected model (Claude Code v2.1.257+).
    """
    return {
        "env": {name: "" for name in _CLAUDE_MODEL_VARIABLES}
        | {
            "CLAUDE_CODE_SUBAGENT_MODEL": model_name.removeprefix("cc/"),
            "CLAUDE_CODE_SUBAGENT_MODEL_FORCE": "1",
            "CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1",
        },
        "fallbackModel": [],
        "modelOverrides": {},
    }


def subscription_run[F: Callable[..., Any]](func: F) -> F:
    """Keep billing policy active for a run's setup, helpers, and teardown."""
    signature = inspect.signature(func)
    extra = next(
        (p.name for p in signature.parameters.values() if p.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        if extra:
            bound.arguments.setdefault(extra, {})
        values = bound.arguments[extra] if extra else bound.arguments
        name = values.get("model_name") or ""
        if not name and args and hasattr(args[0], "_resolve_model_name"):
            name = args[0]._resolve_model_name(name)
        elif not name and "model_name" in signature.parameters:
            from kiss.core.models.model_info import get_default_model

            name = get_default_model()
        if name:
            values["model_name"] = name
        config = dict(values.get("model_config") or {})
        required = subscription_only() or config.get("subscription_only") is True
        if cli_provider(name):
            required = required or billing_mode(name, config) == "subscription"
        if required:
            config.update(subscription_only=True, cli_billing_mode="subscription")
            values["model_config"] = config
        enforce_model_policy(name, config)
        with subscription_scope(required):
            return func(*bound.args, **bound.kwargs)

    return cast(F, wrapped)

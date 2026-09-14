# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""CLI for Muse-style connector authentication.

Usage (via ``python -m kiss.agents.third_party_agents.muse_auth``):

* ``status`` — enrolled services and pending state locations.
* ``enroll <service>`` — run the Google OAuth consent flow and store
  the token straight into the daemon vault (no agent-readable copy).
* ``import <service>`` — migrate an existing legacy ``token.json``
  into the vault and delete the plaintext original.
* ``grant <service> <read|write> [--scope once|session|perpetual|ttl] [--ttl SECONDS]``
  — approve asked actions (Muse's human-in-the-loop grant).
* ``revoke <service> [action]`` — remove grants.
* ``export <service>`` — print a service's vault credential so a
  legacy config can be restored after opting out (``KISS_MUSE_AUTH=0``).
* ``clear <service>`` — remove a service's credential from the vault.
* ``audit [--tail N]`` — show recent Sentinel decisions.
* ``daemon`` — run the daemon in the foreground.
* ``stop`` — stop a running daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir, valid_http_url
from kiss.agents.third_party_agents.muse_auth.client import (
    clear_credentials,
    enrolled_services,
    ensure_daemon,
    grant,
    revoke,
    stop_daemon,
    store_credentials,
)

# Google-OAuth connector service name -> module holding its _SCOPES.
_SERVICE_MODULES = {
    "gmail": "kiss.agents.third_party_agents.gmail_agent",
    "google_drive": "kiss.agents.third_party_agents.google_drive_agent",
    "google_calendar": "kiss.agents.third_party_agents.google_calendar_agent",
    "google_docs": "kiss.agents.third_party_agents.google_docs_agent",
    "google_sheets": "kiss.agents.third_party_agents.google_sheets_agent",
    "googlechat": "kiss.agents.third_party_agents.googlechat_agent",
}

# Plain token connectors: the legacy config.json key holding the token,
# plus the credential header when it is not ``Authorization: Bearer``,
# an optional value prefix for non-Bearer Authorization schemes
# (Discord sends ``Authorization: Bot <token>``), an optional query
# ``param`` for query-kind credentials (BlueBubbles authenticates with
# ``password=`` in the URL), and optional extra secret keys to scrub
# after the migration (Twitch's unused ``client_secret``).
_TOKEN_SERVICES: dict[str, dict[str, Any]] = {
    "notion": {"key": "token"},
    "github": {"key": "token"},
    "firecrawl": {"key": "api_key"},
    "brave_search": {"key": "api_key", "header": "X-Subscription-Token"},
    "discord": {"key": "bot_token", "header": "Authorization", "prefix": "Bot "},
    "homeassistant": {"key": "token"},
    "ntfy": {"key": "token"},
    "twitch": {"key": "access_token", "also_scrub": ("client_secret",)},
    "zalo": {"key": "access_token", "header": "access_token"},
    "line": {"key": "channel_access_token"},
    "mattermost": {"key": "token"},
    "bluebubbles": {"key": "password", "param": "password"},
}


def _import_hosts(service: str, cfg: dict) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the (hosts, insecure_hosts) to enroll for a legacy config.

    Self-hosted connectors (Firecrawl, Home Assistant, ntfy) enroll
    their configured base host with the credential; plain-``http://``
    bases additionally enroll it as a consent-scoped insecure host.

    Args:
        service: Connector service name.
        cfg: The parsed legacy ``config.json`` contents.

    Returns:
        ``(hosts, insecure_hosts)`` tuples (possibly empty).
    """
    if service == "firecrawl":
        from kiss.agents.third_party_agents import firecrawl_agent

        # Firecrawl is origin-bound with no built-in host, so a cloud
        # key (no base_url) must still enroll the cloud origin.
        base_url = str(cfg.get("base_url") or firecrawl_agent._DEFAULT_BASE_URL)
        return (
            firecrawl_agent._extra_hosts(base_url),
            firecrawl_agent._insecure_extra_hosts(base_url),
        )
    if service == "homeassistant" and cfg.get("base_url"):
        from kiss.agents.third_party_agents import homeassistant_agent as ha

        base_url = str(cfg["base_url"])
        return ha._extra_hosts(base_url), ha._insecure_extra_hosts(base_url)
    if service == "ntfy":
        from kiss.agents.third_party_agents import ntfy_agent

        # ntfy is origin-bound with no built-in host, so a public-cloud
        # token (no server configured) must still enroll ntfy.sh — the
        # same default the connector's loader substitutes.
        server = str(cfg.get("server") or ntfy_agent._DEFAULT_SERVER)
        return ntfy_agent._extra_hosts(server), ntfy_agent._insecure_extra_hosts(server)
    if service in ("mattermost", "bluebubbles"):
        from kiss.agents.third_party_agents.muse_auth._common import (
            insecure_origin_hosts,
            origin_hosts,
        )

        base_url = _service_base_url(service, cfg)
        return origin_hosts(base_url), insecure_origin_hosts(base_url)
    return (), ()


def _service_base_url(service: str, cfg: dict) -> str:
    """Return the origin-binding base URL of a legacy config.

    Args:
        service: ``"mattermost"`` (URL composed from ``url``/``port``/
            ``scheme``) or ``"bluebubbles"`` (``server_url``).
        cfg: The parsed legacy ``config.json`` contents.

    Returns:
        The base URL string (may be empty or invalid; callers validate).
    """
    if service == "mattermost":
        from kiss.agents.third_party_agents.mattermost_agent import _base_url_from_config

        return _base_url_from_config(cfg)
    return str(cfg.get("server_url") or "")


def _scrub_imported_config(service: str) -> None:
    """Remove a just-imported token from its legacy ``config.json``.

    Completes the migration in the same command instead of leaving a
    plaintext copy behind: non-secret metadata keys (``base_url``,
    ``server``, ``application_id``, ...) survive, and the file is
    deleted when nothing but the token was stored.

    Args:
        service: A :data:`_TOKEN_SERVICES` connector name.
    """
    from kiss.agents.third_party_agents._channel_agent_utils import save_json_config

    spec = _TOKEN_SERVICES[service]
    scrubbed = {spec["key"], *spec.get("also_scrub", ())}
    path = muse_auth_dir().parent / "third_party_agents" / service / "config.json"
    try:
        cfg = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or not scrubbed.intersection(cfg):
        return
    kept = {k: str(v) for k, v in cfg.items() if k not in scrubbed and v}
    if kept:
        save_json_config(path, kept)
    else:
        path.unlink()


def _service_scopes(service: str) -> list[str]:
    """Return the OAuth scopes a connector service requests.

    Args:
        service: Connector service name.

    Returns:
        The service module's ``_SCOPES`` list.

    Raises:
        SystemExit: On an unknown service name.
    """
    module_name = _SERVICE_MODULES.get(service)
    if module_name is None:
        raise SystemExit(
            f"unknown service '{service}'; choose from {sorted(_SERVICE_MODULES)}"
        )
    module = __import__(module_name, fromlist=["_SCOPES"])
    return list(module._SCOPES)


def _cmd_status() -> int:
    """Print enrolled services and state file locations.

    Returns:
        Process exit code.
    """
    ensure_daemon()
    # Everything in the vault, unfiltered, so workspace-keyed names
    # (slack-<slug>-<hash>) and custom services show up too.
    print(json.dumps({
        "enrolled": enrolled_services(),
        "vault_dir": str(muse_auth_dir() / "vault"),
        "policy": str(muse_auth_dir() / "policy.json"),
        "audit": str(muse_auth_dir() / "audit.jsonl"),
    }, indent=2))
    return 0


def _cmd_enroll(service: str) -> int:
    """Run the OAuth consent flow and store the token into the vault.

    Args:
        service: Connector service name.

    Returns:
        Process exit code.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow

    from kiss.agents.third_party_agents._backend_utils import is_headless_environment
    from kiss.agents.third_party_agents._google_workspace_utils import credentials_path

    scopes = _service_scopes(service)
    creds_path = credentials_path(service)
    if not creds_path.exists():
        print(
            f"credentials.json not found for '{service}'. Download an OAuth Desktop-app "
            f"client JSON from Google Cloud Console and save it to {creds_path}.",
            file=sys.stderr,
        )
        return 1
    flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), scopes)
    if is_headless_environment():
        creds = flow.run_local_server(port=0, open_browser=False)
    else:
        creds = flow.run_local_server(port=0)
    store_credentials(service, creds, scopes)
    print(f"'{service}' enrolled into the Muse-auth vault.")
    return 0


def _cmd_import(service: str) -> int:
    """Migrate a legacy plaintext credential into the vault.

    Google-OAuth services move their ``token.json`` (the plaintext file
    is deleted).  Token services (notion, github, firecrawl,
    brave_search, discord, homeassistant, ntfy, twitch, zalo, line,
    mattermost, bluebubbles) move the token out of their
    ``config.json``: the secret keys are scrubbed after a successful
    store while non-secret settings survive.  ``nextcloud`` folds its
    username/password into one Basic credential; ``synology`` extracts
    the ``token=`` parameter embedded in its webhook URL.  ``slack``
    migrates the default workspace's bot token and deletes its
    plaintext file; other Slack workspaces migrate automatically on
    their first Muse-mode connect.  ``govee`` enrolls
    ``$GOVEE_API_KEY`` from the environment (there is no config file);
    unset the shell export afterwards.

    Args:
        service: Connector service name.

    Returns:
        Process exit code.
    """
    if service == "slack":
        from kiss.agents.third_party_agents.slack_agent import _load_token, _token_path

        token = _load_token("default")
        if not token:
            print(f"no legacy token at {_token_path('default')}", file=sys.stderr)
            return 1
        store_credentials("slack", {"kind": "bearer", "token": token}, [])
        _token_path("default").unlink()
        print("migrated the default Slack workspace token into the Muse-auth vault "
              "and removed the plaintext token.")
        return 0
    if service == "govee":
        key = os.environ.get("GOVEE_API_KEY", "")
        if not key:
            print("GOVEE_API_KEY is not set in the environment", file=sys.stderr)
            return 1
        store_credentials(
            "govee", {"kind": "header", "header": "Govee-API-Key", "token": key}, []
        )
        print("enrolled $GOVEE_API_KEY into the Muse-auth vault; you may unset it now.")
        return 0
    if service == "nextcloud":
        return _cmd_import_nextcloud()
    if service == "synology":
        return _cmd_import_synology()
    if service in _TOKEN_SERVICES:
        spec = _TOKEN_SERVICES[service]
        path = muse_auth_dir().parent / "third_party_agents" / service / "config.json"
        if not path.exists():
            print(f"no legacy config at {path}", file=sys.stderr)
            return 1
        cfg = json.loads(path.read_text())
        token = cfg.get(spec["key"], "")
        if not token:
            print(f"no '{spec['key']}' key in {path}", file=sys.stderr)
            return 1
        # Validate a configured self-hosted URL the same way the
        # authenticate tools do: a userinfo/malformed URL must not be
        # migrated (its password would persist and the credential could
        # not be spent).
        url_key = {
            "firecrawl": "base_url",
            "homeassistant": "base_url",
            "ntfy": "server",
            "bluebubbles": "server_url",
        }.get(service)
        if url_key is not None:
            raw_url = cfg.get(url_key)
            if raw_url is None or raw_url == "":
                # Null/absent/empty selects the documented default for
                # the optional firecrawl/ntfy URLs (their loaders and
                # _import_hosts substitute it).  Home Assistant and
                # BlueBubbles are always self-hosted: without a base URL
                # the credential would be stored hostless and unusable,
                # so reject.
                if service in ("homeassistant", "bluebubbles"):
                    print(
                        f"'{url_key}' in {path} is required and must be a "
                        "valid http(s):// URL",
                        file=sys.stderr,
                    )
                    return 1
            elif not isinstance(raw_url, str) or not valid_http_url(raw_url):
                # Non-string JSON values (false, 0, [], {}) are
                # malformed configs, not "use the default".
                print(
                    f"'{url_key}' in {path} is not a valid http(s):// URL "
                    "(no userinfo, valid host and port)",
                    file=sys.stderr,
                )
                return 1
        if service == "mattermost" and not valid_http_url(_service_base_url(service, cfg)):
            # Mattermost's origin is composed from url/scheme/port.
            print(
                f"the mattermost config in {path} does not compose a valid "
                "http(s):// server URL (url, scheme, port)",
                file=sys.stderr,
            )
            return 1
        token = spec.get("prefix", "") + token
        header = spec.get("header", "")
        param = spec.get("param", "")
        if param:
            info = {"kind": "query", "param": param, "token": token}
        elif header:
            info = {"kind": "header", "header": header, "token": token}
        else:
            info = {"kind": "bearer", "token": token}
        hosts, insecure_hosts = _import_hosts(service, cfg)
        store_credentials(service, info, [], hosts=hosts, insecure_hosts=insecure_hosts)
        _scrub_imported_config(service)
        print(
            f"imported the {service} token into the Muse-auth vault and scrubbed "
            f"the plaintext copy from {path}."
        )
        return 0
    from kiss.agents.third_party_agents._google_workspace_utils import token_path

    path = token_path(service)
    if not path.exists():
        print(f"no legacy token at {path}", file=sys.stderr)
        return 1
    info = json.loads(path.read_text())
    store_credentials(service, info, _service_scopes(service))
    path.unlink()
    print(f"migrated {path} into the Muse-auth vault and removed the plaintext token.")
    return 0


def _cmd_import_nextcloud() -> int:
    """Migrate a legacy Nextcloud username/password into the vault.

    The pair becomes one header-kind ``Authorization: Basic`` credential
    bound to the configured server origin; the ``password`` is scrubbed
    from ``config.json`` while the non-secret ``url``/``username``
    survive.

    Returns:
        Process exit code.
    """
    from kiss.agents.third_party_agents import nextcloud_talk_agent as nc
    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
    )

    path = muse_auth_dir().parent / "third_party_agents" / "nextcloud" / "config.json"
    if not path.exists():
        print(f"no legacy config at {path}", file=sys.stderr)
        return 1
    cfg = json.loads(path.read_text())
    url = str(cfg.get("url") or "").rstrip("/")
    username = str(cfg.get("username") or "")
    password = str(cfg.get("password") or "")
    if not (url and username and password):
        print(f"{path} must hold url, username, and password", file=sys.stderr)
        return 1
    if not valid_http_url(url):
        print(
            f"'url' in {path} is not a valid http(s):// URL "
            "(no userinfo, valid host and port)",
            file=sys.stderr,
        )
        return 1
    store_credentials(
        "nextcloud",
        {
            "kind": "header",
            "header": "Authorization",
            "token": nc._basic_credential(username, password),
        },
        [],
        hosts=origin_hosts(url),
        insecure_hosts=insecure_origin_hosts(url),
    )
    nc._scrub_config_password()
    print(
        f"imported the nextcloud credentials into the Muse-auth vault and scrubbed "
        f"the plaintext password from {path}."
    )
    return 0


def _cmd_import_synology() -> int:
    """Migrate the token embedded in a Synology webhook URL into the vault.

    The ``token=`` query parameter becomes a query-kind credential bound
    to the webhook's origin; ``config.json`` keeps the URL without it.

    Returns:
        Process exit code.
    """
    from kiss.agents.third_party_agents import synology_chat_agent as syno
    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
    )

    path = muse_auth_dir().parent / "third_party_agents" / "synology" / "config.json"
    if not path.exists():
        print(f"no legacy config at {path}", file=sys.stderr)
        return 1
    cfg = json.loads(path.read_text())
    webhook_url = str(cfg.get("webhook_url") or "")
    if not webhook_url:
        print(f"no 'webhook_url' key in {path}", file=sys.stderr)
        return 1
    if not valid_http_url(webhook_url):
        print(
            f"'webhook_url' in {path} is not a valid http(s):// URL "
            "(no userinfo, valid host and port)",
            file=sys.stderr,
        )
        return 1
    embedded = syno._embedded_token(webhook_url)
    if not embedded:
        print(f"the webhook_url in {path} carries no token= query parameter", file=sys.stderr)
        return 1
    store_credentials(
        "synology",
        {"kind": "query", "param": "token", "token": embedded},
        [],
        hosts=origin_hosts(webhook_url),
        insecure_hosts=insecure_origin_hosts(webhook_url),
    )
    syno._scrub_config_webhook_token()
    print(
        f"imported the synology webhook token into the Muse-auth vault and scrubbed "
        f"it from the webhook_url in {path}."
    )
    return 0


def _cmd_export(service: str) -> int:
    """Print a service's vault credential as JSON on stdout.

    The Muse-auth default migrates plaintext credentials into the vault
    and scrubs the legacy copies, so a user opting out afterwards
    (``KISS_MUSE_AUTH=0``) needs the real credential back to rebuild a
    legacy config or ``token.json``.  The vault file already belongs to
    (and is readable by) the invoking user, so printing it discloses
    nothing the user cannot read; the daemon itself still never returns
    credentials over the socket.

    Args:
        service: Connector service name.

    Returns:
        Process exit code (1 when the service has no vault credential).
    """
    from kiss.agents.third_party_agents.muse_auth._common import valid_service_name

    if not valid_service_name(service):
        print(f"invalid service name {service!r}", file=sys.stderr)
        return 1
    path = muse_auth_dir() / "vault" / f"{service}.json"
    if not path.exists():
        print(f"no vault credential for '{service}'", file=sys.stderr)
        return 1
    payload = json.loads(path.read_text())
    print(json.dumps(payload.get("authorized_user_info"), indent=2))
    return 0


def _cmd_audit(tail: int) -> int:
    """Print the last *tail* Sentinel audit records.

    Args:
        tail: Number of records to show.

    Returns:
        Process exit code.
    """
    path = muse_auth_dir() / "audit.jsonl"
    if not path.exists():
        print("no audit records yet")
        return 0
    for line in path.read_text().splitlines()[-tail:]:
        print(line)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the Muse-auth CLI.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(prog="muse_auth", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("enroll").add_argument("service")
    sub.add_parser("import").add_argument("service")
    p_grant = sub.add_parser("grant")
    p_grant.add_argument("service")
    p_grant.add_argument("action", choices=["read", "write"])
    p_grant.add_argument("--scope", default="once",
                         choices=["once", "session", "perpetual", "ttl"])
    p_grant.add_argument("--ttl", type=float, default=0.0)
    p_revoke = sub.add_parser("revoke")
    p_revoke.add_argument("service")
    p_revoke.add_argument("action", nargs="?", default="")
    sub.add_parser("export").add_argument("service")
    sub.add_parser("clear").add_argument("service")
    p_audit = sub.add_parser("audit")
    p_audit.add_argument("--tail", type=int, default=20)
    sub.add_parser("daemon")
    sub.add_parser("stop")
    args = parser.parse_args(argv)

    if args.cmd == "status":
        return _cmd_status()
    if args.cmd == "enroll":
        return _cmd_enroll(args.service)
    if args.cmd == "import":
        return _cmd_import(args.service)
    if args.cmd == "grant":
        grant_id = grant(args.service, args.action, args.scope, args.ttl)
        print(f"granted ({args.scope}) id={grant_id}")
        return 0
    if args.cmd == "revoke":
        print(f"revoked {revoke(args.service, args.action)} grant(s)")
        return 0
    if args.cmd == "export":
        return _cmd_export(args.service)
    if args.cmd == "clear":
        clear_credentials(args.service)
        print(f"cleared '{args.service}' from the vault")
        return 0
    if args.cmd == "audit":
        return _cmd_audit(args.tail)
    if args.cmd == "daemon":
        from kiss.agents.third_party_agents.muse_auth.daemon import main as daemon_main

        daemon_main()
        return 0
    stop_daemon()
    print("daemon stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    sys.exit(main())

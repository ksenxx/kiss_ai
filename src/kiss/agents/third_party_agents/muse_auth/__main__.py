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
* ``clear <service>`` — remove a service's credential from the vault.
* ``audit [--tail N]`` — show recent Sentinel decisions.
* ``daemon`` — run the daemon in the foreground.
* ``stop`` — stop a running daemon.
"""

from __future__ import annotations

import argparse
import json
import sys

from kiss.agents.third_party_agents.muse_auth._common import muse_auth_dir
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
# plus the credential header when it is not ``Authorization: Bearer``.
_TOKEN_SERVICES: dict[str, dict[str, str]] = {
    "notion": {"key": "token"},
    "github": {"key": "token"},
    "firecrawl": {"key": "api_key"},
    "brave_search": {"key": "api_key", "header": "X-Subscription-Token"},
}


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
    brave_search) copy the token out of their ``config.json``; delete
    or blank that key manually to finish the migration, since the file
    may hold other settings.  ``slack`` migrates the default
    workspace's bot token and deletes its plaintext file; other Slack
    workspaces migrate automatically on their first Muse-mode connect.

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
        header = spec.get("header", "")
        if header:
            info = {"kind": "header", "header": header, "token": token}
        else:
            info = {"kind": "bearer", "token": token}
        hosts: tuple[str, ...] = ()
        if service == "firecrawl" and cfg.get("base_url"):
            from kiss.agents.third_party_agents.firecrawl_agent import _extra_hosts

            hosts = _extra_hosts(str(cfg["base_url"]))
        store_credentials(service, info, [], hosts=hosts)
        print(
            f"imported the {service} token into the Muse-auth vault; "
            f"remove the token from {path} to finish the migration."
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

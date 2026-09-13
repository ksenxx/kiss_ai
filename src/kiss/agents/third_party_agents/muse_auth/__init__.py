# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Meta-Muse-style credential isolation for the Google connectors.

This package ports the authentication architecture Meta published for
its Muse personal agent (research.meta.ai, "How We Built Safety Into
Muse", September 2026) to the KISS third-party connectors (Gmail,
Google Drive, Calendar, Docs, Sheets):

* **authd / vault** (:mod:`.vault`) — a daemon-owned credential store.
  OAuth tokens live in ``$KISS_HOME/muse_auth/vault/`` (0700/0600) and
  are only ever read by the daemon process, never by the agent.
* **Surrogate tokens** — the agent process signs API requests with an
  opaque surrogate (``muse-sgt.<service>.<hex>``).  The daemon swaps
  the surrogate for the real bearer token at the network boundary, so
  a prompt-injected agent has nothing real to exfiltrate.
* **Sentinel** (:mod:`.sentinel`) — the sole permission authority.
  Every outbound request is classified (``read`` vs ``write``),
  checked against a per-service host allowlist and an
  allow/deny/ask policy, optionally satisfied by a user grant
  (once / session / time-bounded / perpetual), and audit-logged.
* **Per-service ACLs** — a surrogate minted for ``google_drive``
  cannot obtain the Gmail credential or reach Gmail API hosts.

Enable by setting ``KISS_MUSE_AUTH=1``.  The Google agents then route
all API traffic through the daemon (see
``_google_workspace_utils.google_api_session``).  Manage enrollment
and grants with ``python -m kiss.agents.third_party_agents.muse_auth``.

Honest scope note: unlike Muse's VM, both processes here run as the
same OS user, so the boundary is a process boundary, not an OS
security domain.  It removes real credentials from the agent process
memory and enforces deterministic egress policy; it does not defend
against an agent that is allowed to run arbitrary shell commands
(such an agent could read the vault files or call the ``grant`` CLI
itself — deny access to ``$KISS_HOME/muse_auth`` and to the
``muse_auth`` CLI in tool permissions for that).  Grants are meant to
be issued by the human at a terminal, mirroring Muse's client-UI
approvals.  Response bodies are capped at ~48 MiB by the daemon's
64 MiB frame limit.
"""

from kiss.agents.third_party_agents.muse_auth.client import (
    MuseAuthError,
    MuseBoundarySession,
    MuseHttp,
    SurrogateCredentials,
    bearer_surrogate,
    clear_credentials,
    grant,
    mint_surrogate,
    muse_auth_enabled,
    store_credentials,
    vault_has_credentials,
)

__all__ = [
    "MuseAuthError",
    "MuseBoundarySession",
    "MuseHttp",
    "SurrogateCredentials",
    "bearer_surrogate",
    "clear_credentials",
    "grant",
    "mint_surrogate",
    "muse_auth_enabled",
    "store_credentials",
    "vault_has_credentials",
]

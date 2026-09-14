# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Mattermost Agent — channel agent with Mattermost REST API tools.

Provides authenticated access to Mattermost via a personal access token.
Stores config in ``~/.kiss/third_party_agents/mattermost/config.json``.

Usage::

    agent = MattermostAgent()
    agent.run(prompt_template="List all third_party_agents in the team")
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
    save_json_config,
)

logger = logging.getLogger(__name__)

_MATTERMOST_DIR = Path.home() / ".kiss" / "third_party_agents" / "mattermost"
_config = ChannelConfig(
    _MATTERMOST_DIR,
    (
        "url",
        "token",
    ),
)


def _base_url_from_config(cfg: dict[str, Any]) -> str:
    """Compose the server base URL from Mattermost config fields.

    Args:
        cfg: Parsed ``config.json`` contents (``url``, and optional
            ``scheme``/``port``).

    Returns:
        ``scheme://url:port``, or ``""`` when the config has no usable
        ``url``/``scheme``/``port``.  Non-string JSON values for
        ``url``/``scheme`` (and boolean ``port``) are malformed configs,
        not defaults: ``true`` must not become the hostname ``"True"``.
    """
    host = cfg.get("url")
    if not isinstance(host, str) or not host:
        return ""
    scheme = cfg.get("scheme")
    if scheme in (None, ""):
        scheme = "https"
    elif not isinstance(scheme, str) or scheme not in ("http", "https"):
        return ""
    raw_port = cfg.get("port")
    if raw_port in (None, ""):
        port = 443
    elif isinstance(raw_port, bool) or not isinstance(raw_port, int | str):
        # bool is an int subtype (true must not become port 1), and
        # floats/other JSON types are malformed rather than truncatable
        # (443.9 must not silently become 443).
        return ""
    else:
        try:
            port = int(raw_port)
        except ValueError:
            return ""
    return f"{scheme}://{host}:{port}"


def _scrub_config_token() -> None:
    """Remove a vault-migrated ``token`` from config.json.

    Finishes the Muse migration automatically: the non-secret ``url``/
    ``port``/``scheme`` metadata is kept and the file is deleted when
    nothing but the token was stored.
    """
    try:
        cfg = json.loads(_config.path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(cfg, dict) or "token" not in cfg:
        return
    kept = {k: str(v) for k, v in cfg.items() if k != "token" and v}
    if kept:
        save_json_config(_config.path, kept)
    else:
        _config.clear()


class _MuseMattermostDriver:
    """Minimal ``mattermostdriver``-compatible client for Muse-auth mode.

    Implements exactly the endpoint methods the backend uses, executing
    every call at the Muse daemon boundary with a surrogate bearer (the
    daemon swaps in the real personal access token).  The real
    ``mattermostdriver`` groups endpoints into namespaces
    (``driver.users.get_user``, ``driver.posts.create_post``, ...); the
    method names this backend uses are unique across those namespaces,
    so one object serves as every namespace.
    """

    def __init__(self, base_url: str, surrogate: str) -> None:
        from kiss.agents.third_party_agents.muse_auth.client import MuseBoundarySession

        self._api_base = base_url.rstrip("/") + "/api/v4"
        self._surrogate = surrogate
        self._session = MuseBoundarySession("mattermost")
        self.users = self
        self.teams = self
        self.channels = self
        self.posts = self
        self.reactions = self

    def _call(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        """Execute one REST call at the boundary, raising on HTTP errors.

        Args:
            method: HTTP method.
            path: API path under ``/api/v4``.
            params: Optional query parameters.
            json_body: Optional JSON body.

        Returns:
            The decoded JSON response (``{}`` for empty bodies).

        Raises:
            RuntimeError: On any HTTP error status (mirroring the real
                driver, which raises on non-2xx responses).
        """
        resp = self._session.request(
            method,
            f"{self._api_base}{path}",
            headers={"Authorization": f"Bearer {self._surrogate}"},
            params=params,
            json=json_body,
            timeout=30,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mattermost API {method} {path} failed: "
                f"HTTP {resp.status_code} {resp.text[:300]}"
            )
        return resp.json() if resp.content else {}

    def login(self) -> Any:
        """Validate the credential by fetching the authenticated user."""
        return self._call("GET", "/users/me")

    def get_user(self, user_id: str) -> Any:
        """Return one user by ID or username (``"me"`` for the bot)."""
        return self._call("GET", f"/users/{user_id}")

    def get_users(self, params: dict[str, Any] | None = None) -> Any:
        """Return a page of users, honoring the driver's filter params."""
        return self._call("GET", "/users", params=params)

    def get_teams(self) -> Any:
        """Return the teams visible to the authenticated user."""
        return self._call("GET", "/teams")

    def get_channels_for_user(
        self, user_id: str, team_id: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Return a user's channels in one team."""
        return self._call("GET", f"/users/{user_id}/teams/{team_id}/channels", params=params)

    def get_channel(self, channel_id: str) -> Any:
        """Return one channel by ID."""
        return self._call("GET", f"/channels/{channel_id}")

    def create_direct_message_channel(self, options: Any) -> Any:
        """Create (or fetch) the DM channel between two user IDs."""
        return self._call("POST", "/channels/direct", json_body=options)

    def get_posts_for_channel(
        self, channel_id: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Return a channel's posts page (``since``/``page`` params)."""
        return self._call("GET", f"/channels/{channel_id}/posts", params=params)

    def create_post(self, options: Any) -> Any:
        """Create a post from a driver-style options dict."""
        return self._call("POST", "/posts", json_body=options)

    def delete_post(self, post_id: str) -> Any:
        """Delete one post by ID."""
        return self._call("DELETE", f"/posts/{post_id}")

    def create_reaction(self, options: Any) -> Any:
        """Add a reaction from a driver-style options dict."""
        return self._call("POST", "/reactions", json_body=options)


class MattermostChannelBackend(ToolMethodBackend):
    """Channel backend for Mattermost REST API."""

    def __init__(self, base_url: str = "", token: str = "") -> None:
        """Initialize the backend.

        Args:
            base_url: Optional server base URL (e.g. ``https://mm.example.com:443``)
                used for direct REST calls such as the typing indicator.  When
                empty, it is derived from the stored config on :meth:`connect`.
            token: Optional personal access token for direct REST calls.
        """
        self._driver: Any = None
        self._last_post_time: int = 0
        self._connection_info: str = ""
        self._base_url: str = base_url.rstrip("/")
        self._token: str = token
        self._http: Any = requests
        self._muse: bool = False

    def _wire_muse(self) -> bool:
        """Acquire a Mattermost surrogate and wire the boundary driver.

        A ``token`` still in the legacy config is the newest user intent
        (initial migration, or a rotation done while Muse was off): it
        is enrolled as a bearer credential bound to the configured
        server origin (flagged as a consent-scoped insecure host when
        the scheme is plain ``http``), and scrubbed from ``config.json``
        only after the vault holds it.  No network round trip happens
        here.

        Returns:
            True when the backend holds a surrogate and boundary driver.
        """
        from kiss.agents.third_party_agents.muse_auth._common import (
            insecure_origin_hosts,
            origin_hosts,
            valid_http_url,
        )
        from kiss.agents.third_party_agents.muse_auth.client import (
            MuseBoundarySession,
            mint_surrogate,
            store_credentials,
        )

        cfg = _config.load_metadata() or {}
        base = _base_url_from_config(cfg)
        if not base:
            self._connection_info = "No Mattermost config found."
            return False
        # Validate before any credential state changes: a malformed
        # legacy URL must not auto-migrate the token into a host scope
        # Sentinel can never match, nor scrub the plaintext copy.
        if not valid_http_url(base):
            self._connection_info = (
                f"Mattermost server URL {base!r} is not a valid http(s):// URL; "
                "fix config.json and reconnect."
            )
            return False
        token = cfg.get("token", "")
        if token:
            store_credentials(
                "mattermost",
                {"kind": "bearer", "token": token},
                [],
                hosts=origin_hosts(base),
                insecure_hosts=insecure_origin_hosts(base),
            )
        handle = mint_surrogate("mattermost")
        if handle is None:
            self._connection_info = "No Mattermost credential in the Muse vault or config."
            return False
        _scrub_config_token()
        self._base_url = base
        self._token = handle.token
        self._http = MuseBoundarySession("mattermost")
        self._muse = True
        self._driver = _MuseMattermostDriver(base, handle.token)
        return True

    def connect(self) -> bool:
        """Authenticate with Mattermost using stored config."""
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            # Vault-first surrogate wiring; validation below runs the
            # /users/me read through the daemon boundary (audited).
            if not self._wire_muse():
                return False
            try:
                me = self._driver.users.get_user("me")
                self._connection_info = f"Authenticated as {me.get('username', '')}"
                self._last_post_time = int(time.time() * 1000)
                return True
            except Exception as e:
                self._connection_info = f"Mattermost connection failed: {e}"
                return False
        cfg = _config.load()
        if not cfg:  # pragma: no branch
            self._connection_info = "No Mattermost config found."
            return False
        try:
            scheme = cfg.get("scheme", "https")
            port = int(cfg.get("port", 443))
            self._base_url = f"{scheme}://{cfg['url']}:{port}"
            self._token = cfg["token"]
            from mattermostdriver import Driver

            self._driver = Driver(
                {
                    "url": cfg["url"],
                    "token": cfg["token"],
                    "port": port,
                    "scheme": scheme,
                }
            )
            self._driver.login()
            me = self._driver.users.get_user("me")
            self._connection_info = f"Authenticated as {me.get('username', '')}"
            self._last_post_time = int(time.time() * 1000)
            return True
        except Exception as e:
            self._connection_info = f"Mattermost connection failed: {e}"
            return False

    def poll_messages(
        self, channel_id: str, oldest: str, limit: int = 10
    ) -> tuple[list[dict[str, Any]], str]:
        """Poll Mattermost channel for new posts."""
        if not self._driver or not channel_id:  # pragma: no branch
            return [], oldest
        try:
            since = int(oldest) if oldest else self._last_post_time
            posts = self._driver.posts.get_posts_for_channel(
                channel_id, params={"since": since, "per_page": limit}
            )
            order = posts.get("order", [])
            posts_data = posts.get("posts", {})
            messages: list[dict[str, Any]] = []
            new_oldest = oldest
            for post_id in reversed(order[:limit]):  # pragma: no branch
                post = posts_data.get(post_id, {})
                ts = str(post.get("create_at", ""))
                new_oldest = ts
                messages.append(
                    {
                        "ts": ts,
                        "thread_ts": post.get("root_id") or post.get("id", ""),
                        "user": post.get("user_id", ""),
                        "text": post.get("message", ""),
                        "id": post.get("id", ""),
                    }
                )
            if messages:  # pragma: no branch
                self._last_post_time = int(new_oldest) + 1
            return messages, new_oldest
        except Exception:
            return [], oldest

    def send_message(self, channel_id: str, text: str, thread_ts: str = "") -> None:
        """Send a Mattermost post."""
        if not self._driver:  # pragma: no branch
            return
        post: dict[str, Any] = {"channel_id": channel_id, "message": text}
        if thread_ts:  # pragma: no branch
            post["root_id"] = thread_ts
        self._driver.posts.create_post(options=post)

    def send_typing(self, channel_id: str, thread_ts: str = "") -> None:
        """Show a best-effort typing indicator in a Mattermost channel.

        POSTs ``/api/v4/users/me/typing`` with a bearer token so channel
        members see the bot "typing" while a task is being worked on.
        Failures (HTTP errors, unreachable server, missing configuration)
        are logged and swallowed; this method never raises.

        Args:
            channel_id: Channel in which to show the typing indicator.
            thread_ts: Root post ID when typing inside a thread; sent as
                ``parent_id`` when non-empty.
        """
        if not self._base_url or not channel_id:
            return
        body: dict[str, Any] = {"channel_id": channel_id}
        if thread_ts:
            body["parent_id"] = thread_ts
        try:
            # In Muse mode ``_token`` holds a surrogate and ``_http`` is
            # the boundary session (the daemon swaps in the real token
            # and classifies this ephemeral POST as a read).
            self._http.post(
                f"{self._base_url}/api/v4/users/me/typing",
                json=body,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
        except Exception:
            logger.debug("Mattermost typing indicator failed", exc_info=True)

    def list_teams(self) -> str:
        """List Mattermost teams.

        Returns:
            JSON string with team list (id, name, display_name).
        """
        assert self._driver is not None
        try:
            teams = self._driver.teams.get_teams()
            result = [
                {
                    "id": t.get("id", ""),
                    "name": t.get("name", ""),
                    "display_name": t.get("display_name", ""),
                }
                for t in teams
            ]
            return json.dumps({"ok": True, "teams": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_third_party_agents(self, team_id: str, page: int = 0, per_page: int = 60) -> str:
        """List third_party_agents in a Mattermost team.

        Args:
            team_id: Team ID.
            page: Page number for pagination. Default: 0.
            per_page: Channels per page. Default: 60.

        Returns:
            JSON string with channel list.
        """
        assert self._driver is not None
        try:
            third_party_agents = self._driver.channels.get_channels_for_user(
                "me", team_id, params={"page": page, "per_page": per_page}
            )
            result = [
                {
                    "id": c.get("id", ""),
                    "name": c.get("name", ""),
                    "display_name": c.get("display_name", ""),
                    "type": c.get("type", ""),
                }
                for c in third_party_agents
            ]
            return json.dumps({"ok": True, "third_party_agents": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel(self, channel_id: str) -> str:
        """Get information about a Mattermost channel.

        Args:
            channel_id: Channel ID.

        Returns:
            JSON string with channel details.
        """
        assert self._driver is not None
        try:
            channel = self._driver.channels.get_channel(channel_id)
            return json.dumps({"ok": True, **channel}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_channel_posts(self, channel_id: str, page: int = 0, per_page: int = 30) -> str:
        """List posts in a Mattermost channel.

        Args:
            channel_id: Channel ID.
            page: Page number. Default: 0.
            per_page: Posts per page. Default: 30.

        Returns:
            JSON string with post list.
        """
        assert self._driver is not None
        try:
            posts = self._driver.posts.get_posts_for_channel(
                channel_id, params={"page": page, "per_page": per_page}
            )
            order = posts.get("order", [])
            posts_data = posts.get("posts", {})
            result = [
                {
                    "id": post_id,
                    "message": posts_data[post_id].get("message", ""),
                    "user_id": posts_data[post_id].get("user_id", ""),
                    "create_at": posts_data[post_id].get("create_at", 0),
                }
                for post_id in order
                if post_id in posts_data
            ]
            return json.dumps({"ok": True, "posts": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_post(
        self, channel_id: str, message: str, root_id: str = "", file_ids: str = ""
    ) -> str:
        """Create a post in a Mattermost channel.

        Args:
            channel_id: Channel ID.
            message: Post message text.
            root_id: Root post ID if this is a reply.
            file_ids: Comma-separated file IDs to attach.

        Returns:
            JSON string with ok status and post id.
        """
        assert self._driver is not None
        try:
            post: dict[str, Any] = {"channel_id": channel_id, "message": message}
            if root_id:  # pragma: no branch
                post["root_id"] = root_id
            if file_ids:  # pragma: no branch
                post["file_ids"] = [f.strip() for f in file_ids.split(",") if f.strip()]
            result = self._driver.posts.create_post(options=post)
            return json.dumps({"ok": True, "id": result.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_post(self, post_id: str) -> str:
        """Delete a Mattermost post.

        Args:
            post_id: Post ID to delete.

        Returns:
            JSON string with ok status.
        """
        assert self._driver is not None
        try:
            self._driver.posts.delete_post(post_id)
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user(self, user_id_or_username: str) -> str:
        """Get a Mattermost user's information.

        Args:
            user_id_or_username: User ID or username. Use "me" for current user.

        Returns:
            JSON string with user details.
        """
        assert self._driver is not None
        try:
            user = self._driver.users.get_user(user_id_or_username)
            return json.dumps(
                {
                    "ok": True,
                    "id": user.get("id", ""),
                    "username": user.get("username", ""),
                    "email": user.get("email", ""),
                    "first_name": user.get("first_name", ""),
                    "last_name": user.get("last_name", ""),
                    "roles": user.get("roles", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_users(
        self, page: int = 0, per_page: int = 60, in_team: str = "", in_channel: str = ""
    ) -> str:
        """List Mattermost users.

        Args:
            page: Page number. Default: 0.
            per_page: Users per page. Default: 60.
            in_team: Optional team ID to filter by.
            in_channel: Optional channel ID to filter by.

        Returns:
            JSON string with user list.
        """
        assert self._driver is not None
        try:
            params: dict[str, Any] = {"page": page, "per_page": per_page}
            if in_team:  # pragma: no branch
                params["in_team"] = in_team
            if in_channel:  # pragma: no branch
                params["in_channel"] = in_channel
            users = self._driver.users.get_users(params=params)
            result = [
                {
                    "id": u.get("id", ""),
                    "username": u.get("username", ""),
                    "email": u.get("email", ""),
                }
                for u in users
            ]
            return json.dumps({"ok": True, "users": result}, indent=2)[:8000]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_direct_message_channel(self, user1_id: str, user2_id: str) -> str:
        """Create a direct message channel between two users.

        Args:
            user1_id: First user ID.
            user2_id: Second user ID.

        Returns:
            JSON string with channel id.
        """
        assert self._driver is not None
        try:
            channel = self._driver.channels.create_direct_message_channel(
                options=[user1_id, user2_id]
            )
            return json.dumps({"ok": True, "channel_id": channel.get("id", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def add_reaction(self, user_id: str, post_id: str, emoji_name: str) -> str:
        """Add a reaction to a post.

        Args:
            user_id: User ID adding the reaction.
            post_id: Post ID.
            emoji_name: Emoji name (without colons, e.g. "thumbsup").

        Returns:
            JSON string with ok status.
        """
        assert self._driver is not None
        try:
            self._driver.reactions.create_reaction(
                options={"user_id": user_id, "post_id": post_id, "emoji_name": emoji_name}
            )
            return json.dumps({"ok": True})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


def _muse_authenticate(
    backend: MattermostChannelBackend, url: str, token: str, port: int, scheme: str
) -> str:
    """Enroll a Mattermost token into the Muse vault and validate it.

    The plaintext token goes straight into the vault as a bearer
    credential bound to the configured server origin, and is never
    written to ``config.json`` (only the non-secret ``url``/``port``/
    ``scheme`` metadata is; it is written first, so a failed enrollment
    leaves no token on disk).  Validation runs ``/users/me`` through
    the daemon boundary, so it is audited; an invalid token leaves the
    vault empty.

    Args:
        backend: The agent's Mattermost backend to (re)wire.
        url: Server hostname (e.g. ``mattermost.example.com``).
        token: Personal access token.
        port: Server port.
        scheme: ``"https"`` or ``"http"``.

    Returns:
        JSON string with the validation result.
    """
    import contextlib

    from kiss.agents.third_party_agents.muse_auth._common import (
        insecure_origin_hosts,
        origin_hosts,
        valid_http_url,
    )
    from kiss.agents.third_party_agents.muse_auth.client import (
        clear_credentials,
        store_credentials,
    )

    base = f"{scheme}://{url}:{port}"
    if not valid_http_url(base):
        return json.dumps(
            {"ok": False, "error": f"{base!r} is not a valid http(s):// server URL."}
        )
    try:
        save_json_config(
            _config.path, {"url": url, "port": str(port), "scheme": scheme}
        )
        store_credentials(
            "mattermost",
            {"kind": "bearer", "token": token},
            [],
            hosts=origin_hosts(base),
            insecure_hosts=insecure_origin_hosts(base),
        )
        if backend._wire_muse():
            me = backend._driver.users.get_user("me")
            return json.dumps(
                {
                    "ok": True,
                    "message": "Mattermost credentials saved (Muse-auth).",
                    "username": me.get("username", ""),
                }
            )
        else:  # pragma: no cover - defense in depth, credential was just stored
            error = json.dumps({"ok": False, "error": backend._connection_info})
    except Exception as e:
        error = json.dumps({"ok": False, "error": str(e)})
    # Roll the vault back so a bad token is not left enrolled.
    with contextlib.suppress(Exception):
        clear_credentials("mattermost")
    backend._driver = None
    backend._token = ""
    backend._http = requests
    backend._muse = False
    return error


class MattermostAgent(BaseChannelAgent):
    """Channel agent with Mattermost REST API tools."""

    def __init__(self) -> None:
        super().__init__("Mattermost Agent")
        self._backend = MattermostChannelBackend()
        from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

        if muse_auth_enabled():
            from kiss.agents.third_party_agents.muse_auth.client import MuseAuthError

            # Muse-auth mode: wire a vault surrogate and the boundary
            # driver (no network round trip); the real token never
            # enters this process once migrated.  A daemon failure
            # leaves the agent constructible (fail closed, tokenless)
            # so its authenticate/clear tools stay available.
            try:
                self._backend._wire_muse()
            except MuseAuthError as e:
                self._backend._driver = None
                self._backend._connection_info = f"Muse-auth wiring failed: {e}"
            return
        cfg = _config.load()
        if cfg:  # pragma: no branch
            try:
                from mattermostdriver import Driver

                self._backend._driver = Driver(
                    {
                        "url": cfg["url"],
                        "token": cfg["token"],
                        "port": int(cfg.get("port", 443)),
                        "scheme": cfg.get("scheme", "https"),
                    }
                )
                self._backend._driver.login()
            except Exception:
                pass

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return self._backend._driver is not None

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_mattermost_auth() -> str:
            """Check if Mattermost credentials are configured and valid.

            Returns:
                Authentication status or instructions.
            """
            if agent._backend._driver is None:  # pragma: no branch
                return (
                    "Not authenticated with Mattermost. "
                    "Use authenticate_mattermost(url=..., token=...) to configure.\n"
                    "You need: server URL (e.g. 'mattermost.example.com') and a "
                    "personal access token from Profile > Security > Personal Access Tokens."
                )
            try:
                result = json.loads(agent._backend.get_user("me"))
                if result.get("ok"):  # pragma: no branch
                    return json.dumps({"ok": True, "username": result.get("username", "")})
                return json.dumps({"ok": False, "error": "Could not verify authentication."})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_mattermost(
            url: str,
            token: str,
            port: int = 443,
            scheme: str = "https",
        ) -> str:
            """Store and validate Mattermost credentials.

            Args:
                url: Mattermost server URL (e.g. "mattermost.example.com").
                token: Personal access token from Account Settings > Security.
                port: Server port. Default: 443.
                scheme: "https" or "http". Default: "https".

            Returns:
                Validation result or error message.
            """
            for val, name in [(url, "url"), (token, "token")]:  # pragma: no branch
                if not val.strip():  # pragma: no branch
                    return f"{name} cannot be empty."
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                return _muse_authenticate(
                    agent._backend, url.strip(), token.strip(), port, scheme.strip()
                )
            try:
                from mattermostdriver import Driver

                driver = Driver(
                    {
                        "url": url.strip(),
                        "token": token.strip(),
                        "port": port,
                        "scheme": scheme,
                    }
                )
                driver.login()
                me = driver.users.get_user("me")
                _config.save(
                    {
                        "url": url.strip(),
                        "token": token.strip(),
                        "port": str(port),
                        "scheme": scheme.strip(),
                    }
                )
                agent._backend._driver = driver
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Mattermost credentials saved.",
                        "username": me.get("username", ""),
                    }
                )
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        def clear_mattermost_auth() -> str:
            """Clear the stored Mattermost credentials.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._driver = None
            agent._backend._token = ""
            agent._backend._http = requests
            agent._backend._muse = False
            from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

            if muse_auth_enabled():
                from kiss.agents.third_party_agents.muse_auth.client import clear_credentials

                clear_credentials("mattermost")
            return "Mattermost authentication cleared."

        return [check_mattermost_auth, authenticate_mattermost, clear_mattermost_auth]


def _make_backend() -> MattermostChannelBackend:
    """Create a configured backend for channel poll mode."""
    backend = MattermostChannelBackend()
    from kiss.agents.third_party_agents.muse_auth._common import muse_auth_enabled

    if muse_auth_enabled():
        if backend.connect():
            return backend
        print("Not authenticated. Run: kiss-mattermost -t 'authenticate'")
        sys.exit(1)
    cfg = _config.load()
    if not cfg:  # pragma: no branch
        print("Not authenticated. Run: kiss-mattermost -t 'authenticate'")
        sys.exit(1)
    from mattermostdriver import Driver

    backend._driver = Driver(
        {
            "url": cfg["url"],
            "token": cfg["token"],
            "port": int(cfg.get("port", 443)),
            "scheme": cfg.get("scheme", "https"),
        }
    )
    backend._driver.login()
    return backend


def main() -> None:
    """Run the MattermostAgent from the command line with chat persistence."""
    channel_main(
        MattermostAgent,
        "kiss-mattermost",
        channel_name="Mattermost",
        make_backend=_make_backend,
    )


def tools() -> list:
    """Return the Mattermost channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return MattermostAgent()._get_tools()


if __name__ == "__main__":
    main()

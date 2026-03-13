"""Channel agent that creates app-specific tools and agents dynamically.

Given any app name, creates authenticated REST API tools and an AppAgent
(extending SorcarAgent) that can perform tasks on the app with full access.
Handles token-based and OAuth2 authentication, stores tokens securely in
<artifact_dir>/../channels/<app>/token.json.

No apps are hardcoded — the agent works with any REST API. The LLM-powered
AppAgent already knows popular APIs (GitHub, Google, Spotify, etc.) and
constructs the right URLs and headers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import config as config_module

logger = logging.getLogger(__name__)

_CHANNELS_DIR = Path(config_module.DEFAULT_CONFIG.agent.artifact_dir).parent / "channels"


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _token_path(app_name: str) -> Path:
    """Return the file path for the stored auth token of an app.

    Args:
        app_name: Lowercase app identifier (e.g. 'github').

    Returns:
        Path to ``<channels_dir>/<app>/token.json``.
    """
    return _CHANNELS_DIR / app_name / "token.json"


def _load_token(app_name: str) -> dict[str, Any] | None:
    """Load a stored auth token from disk.

    Args:
        app_name: App identifier.

    Returns:
        Token data dict, or None if not found or invalid.
    """
    path = _token_path(app_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_token(app_name: str, token_data: dict[str, Any]) -> None:
    """Save an auth token to disk with restricted permissions.

    Args:
        app_name: App identifier.
        token_data: Dict containing at least ``access_token``.
    """
    path = _token_path(app_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(token_data, indent=2))
    path.chmod(0o600)


def _clear_token(app_name: str) -> None:
    """Delete a stored auth token.

    Args:
        app_name: App identifier.
    """
    path = _token_path(app_name)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# OAuth2 helpers
# ---------------------------------------------------------------------------


def _oauth2_exchange(
    token_url: str,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange an OAuth2 authorization code for access/refresh tokens.

    Args:
        token_url: Token endpoint URL.
        code: Authorization code from the callback.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        redirect_uri: Redirect URI used during authorization.

    Returns:
        Token response dict with ``access_token`` and optionally ``refresh_token``.
    """
    resp = requests.post(
        token_url,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


def _oauth2_refresh(
    token_url: str,
    refresh_token: str,
    client_id: str,
    client_secret: str,
) -> dict[str, Any]:
    """Refresh an OAuth2 access token using a refresh token.

    Args:
        token_url: Token endpoint URL.
        refresh_token: Refresh token from a previous exchange.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.

    Returns:
        New token response dict.
    """
    resp = requests.post(
        token_url,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Accept": "application/json"},
        timeout=30,
    )
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


# ---------------------------------------------------------------------------
# Generic REST API tools
# ---------------------------------------------------------------------------


def _generic_rest_tools(app_name: str, token: str) -> list:
    """Create generic REST API tools bound to the given token.

    The access token is included as a Bearer token by default. The agent
    can override headers for apps that use different auth schemes.

    Args:
        app_name: App identifier (for tool docstring context).
        token: Access token.

    Returns:
        List containing a single generic ``api_call`` tool.
    """

    def api_call(
        method: str, url: str, body: str = "", headers: str = "{}"
    ) -> str:
        """Make an authenticated REST API call.

        The access token is automatically included as a Bearer token in
        the Authorization header. Use the headers parameter to override
        or add additional headers (e.g. for GitHub use
        ``{"Authorization": "token <tok>"}``).

        Args:
            method: HTTP method (GET, POST, PUT, PATCH, DELETE).
            url: Full URL (e.g. 'https://api.github.com/user/repos').
            body: JSON request body string (for POST/PUT/PATCH).
            headers: JSON string of additional/override headers.

        Returns:
            JSON response (truncated to 4000 chars).
        """
        req_headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if headers and headers != "{}":
            req_headers.update(json.loads(headers))
        kwargs: dict[str, Any] = {"headers": req_headers, "timeout": 30}
        if body and method.upper() in ("POST", "PUT", "PATCH"):
            kwargs["json"] = json.loads(body)
        resp = requests.request(method.upper(), url, **kwargs)
        try:
            return json.dumps(resp.json(), indent=2)[:4000]
        except Exception:
            return resp.text[:4000]

    return [api_call]


# ---------------------------------------------------------------------------
# OAuth2 config persistence (per-app)
# ---------------------------------------------------------------------------


def _oauth2_config_path(app_name: str) -> Path:
    """Return the file path for stored OAuth2 config of an app.

    Args:
        app_name: App identifier.

    Returns:
        Path to ``<channels_dir>/<app>/oauth2.json``.
    """
    return _CHANNELS_DIR / app_name / "oauth2.json"


def _load_oauth2_config(app_name: str) -> dict[str, Any] | None:
    """Load stored OAuth2 config from disk.

    Args:
        app_name: App identifier.

    Returns:
        OAuth2 config dict, or None if not found.
    """
    path = _oauth2_config_path(app_name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _save_oauth2_config(app_name: str, config: dict[str, Any]) -> None:
    """Save OAuth2 config to disk.

    Args:
        app_name: App identifier.
        config: Dict with auth_url, token_url, scopes, client_id_env,
            client_secret_env.
    """
    path = _oauth2_config_path(app_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))


# ---------------------------------------------------------------------------
# Agent file template
# ---------------------------------------------------------------------------

_AGENT_TEMPLATE = '''\
"""{display_name} Agent — auto-generated by ChannelAgent.

Run this file to interact with {display_name} via an authenticated REST API agent.

Usage:
    python {class_name}.py --task "describe what you want to do"
"""

from __future__ import annotations

import argparse

from kiss.channels.channel_agent import AppAgent, ChannelAgent


class {class_name}(AppAgent):
    """{display_name} agent with authenticated REST API tools."""

    def __init__(self, **kwargs):
        channel = ChannelAgent("{app_name}")
        super().__init__(channel, **kwargs)


def main() -> None:
    """Run the {display_name} agent from the command line."""
    parser = argparse.ArgumentParser(description="{display_name} Agent")
    parser.add_argument("--task", type=str, default=None, help="Task to perform")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--max_budget", type=float, default=5.0)
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() == "true",
        default=False,
    )
    args = parser.parse_args()

    agent = {class_name}()
    display = "{display_name}"
    task = args.task or (
        f"You are a {{display}} agent. Help the user interact with {{display}}.\\n"
        f"First, check if you are authenticated using check_auth(). "
        f"If not authenticated, ask the user for a token and call "
        f"authenticate(token=...) with it.\\n"
        f"For OAuth2 apps, use configure_oauth2() to set up OAuth2, "
        f"then get the auth URL from get_auth_url(), open it in the "
        f"browser, ask the user to authorize, then capture the code "
        f"from the redirect URL and call authenticate_oauth2(code=...).\\n\\n"
        f"Once authenticated, use api_call() to interact with the "
        f"{{display}} API. Ask the user what they want to do."
    )
    result = agent.run(
        prompt_template=task,
        model_name=args.model_name,
        max_steps=args.max_steps,
        max_budget=args.max_budget,
        headless=args.headless,
    )
    print(result)


if __name__ == "__main__":
    main()
'''


def _render_agent_template(
    app_name: str, display_name: str, class_name: str
) -> str:
    """Render the agent file template with the given app details.

    Args:
        app_name: Lowercase app identifier (e.g. 'github').
        display_name: Title-cased display name (e.g. 'Github').
        class_name: Python class name (e.g. 'GithubAgent').

    Returns:
        Complete Python source code for the agent file.
    """
    return _AGENT_TEMPLATE.format(
        app_name=app_name,
        display_name=display_name,
        class_name=class_name,
    )


# ---------------------------------------------------------------------------
# ChannelAgent
# ---------------------------------------------------------------------------


class ChannelAgent:
    """Creates app-specific API tools and agents dynamically for any app.

    Accepts any app name — no hardcoded app registry. Handles authentication
    (token or OAuth2), stores credentials securely, and creates an
    :class:`AppAgent` that extends :class:`SorcarAgent` with a generic REST
    API tool for full API access. The LLM-powered agent knows popular APIs
    and constructs the right URLs.

    Example::

        channel = ChannelAgent("github")
        channel.authenticate_with_token("ghp_xxxx")
        path = channel.create_app_agent()
        # path == <channels_dir>/github/GithubAgent.py
        # Run it: python GithubAgent.py --task "List my repos"
    """

    def __init__(self, app_name: str) -> None:
        self.app_name = app_name.lower().strip()
        if not self.app_name:
            raise ValueError("App name cannot be empty")
        self.display_name = self.app_name.replace("_", " ").replace("-", " ").title()
        self.token_data: dict[str, Any] | None = _load_token(self.app_name)
        self.oauth2_config: dict[str, Any] | None = _load_oauth2_config(
            self.app_name
        )
        self._tools: list | None = None

    @property
    def is_authenticated(self) -> bool:
        """True if a valid access token is stored."""
        return self.token_data is not None and bool(
            self.token_data.get("access_token")
        )

    def authenticate_with_token(self, token: str) -> str:
        """Store a direct access token (e.g. GitHub PAT, API key).

        Args:
            token: The access token string.

        Returns:
            Status message.
        """
        self.token_data = {"access_token": token.strip()}
        _save_token(self.app_name, self.token_data)
        self._tools = None
        return f"Token saved for {self.display_name}"

    def configure_oauth2(
        self,
        auth_url: str,
        token_url: str,
        client_id_env: str,
        client_secret_env: str,
        scopes: str = "",
    ) -> str:
        """Configure OAuth2 settings for this app.

        Call this before starting the OAuth2 flow. The settings are persisted
        to disk so they survive restarts.

        Args:
            auth_url: OAuth2 authorization endpoint URL.
            token_url: OAuth2 token endpoint URL.
            client_id_env: Environment variable name for the client ID.
            client_secret_env: Environment variable name for the client secret.
            scopes: Space-separated OAuth2 scopes.

        Returns:
            Status message.
        """
        self.oauth2_config = {
            "auth_url": auth_url,
            "token_url": token_url,
            "client_id_env": client_id_env,
            "client_secret_env": client_secret_env,
            "scopes": scopes,
        }
        _save_oauth2_config(self.app_name, self.oauth2_config)
        return f"OAuth2 configured for {self.display_name}"

    def authenticate_oauth2(self, code: str, redirect_uri: str) -> str:
        """Complete OAuth2 auth by exchanging an authorization code for tokens.

        Requires :meth:`configure_oauth2` to have been called first.

        Args:
            code: Authorization code from the OAuth2 callback.
            redirect_uri: The redirect URI used during authorization.

        Returns:
            Status message.
        """
        if not self.oauth2_config:
            return (
                f"No OAuth2 config for {self.display_name}. "
                "Call configure_oauth2() first."
            )
        client_id = os.environ.get(self.oauth2_config["client_id_env"], "")
        client_secret = os.environ.get(
            self.oauth2_config["client_secret_env"], ""
        )
        token_data = _oauth2_exchange(
            self.oauth2_config["token_url"],
            code,
            client_id,
            client_secret,
            redirect_uri,
        )
        self.token_data = token_data
        _save_token(self.app_name, token_data)
        self._tools = None
        return (
            f"OAuth2 authentication successful for {self.display_name}"
        )

    def refresh_access_token(self) -> str:
        """Refresh the OAuth2 access token using the stored refresh token.

        Returns:
            Status message.
        """
        if not self.token_data or "refresh_token" not in self.token_data:
            return "No refresh token available"
        if not self.oauth2_config:
            return (
                f"No OAuth2 config for {self.display_name}. "
                "Call configure_oauth2() first."
            )
        client_id = os.environ.get(self.oauth2_config["client_id_env"], "")
        client_secret = os.environ.get(
            self.oauth2_config["client_secret_env"], ""
        )
        new_data = _oauth2_refresh(
            self.oauth2_config["token_url"],
            self.token_data["refresh_token"],
            client_id,
            client_secret,
        )
        # Preserve the refresh token if the new response omits it
        if (
            "refresh_token" not in new_data
            and "refresh_token" in self.token_data
        ):
            new_data["refresh_token"] = self.token_data["refresh_token"]
        self.token_data = new_data
        _save_token(self.app_name, new_data)
        self._tools = None
        return f"Access token refreshed for {self.display_name}"

    def get_auth_url(self) -> str:
        """Get the OAuth2 authorization URL for this app.

        Requires :meth:`configure_oauth2` to have been called and the
        client ID environment variable to be set.

        Returns:
            The full authorization URL, or an error message.
        """
        if not self.oauth2_config:
            return (
                f"No OAuth2 config for {self.display_name}. "
                "Call configure_oauth2() first."
            )
        client_id_env = self.oauth2_config["client_id_env"]
        client_id = os.environ.get(client_id_env, "")
        if not client_id:
            return (
                f"Set the {client_id_env} environment variable "
                f"for {self.display_name} OAuth2."
            )
        redirect_uri = "http://127.0.0.1:8585/callback"
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.oauth2_config.get("scopes", ""),
            "access_type": "offline",
            "prompt": "consent",
        }
        auth_url = self.oauth2_config["auth_url"]
        return f"{auth_url}?{urlencode(params)}"

    def get_tools(self) -> list:
        """Return generic REST API tools (requires prior authentication).

        Returns:
            List of callable tool functions, or empty list if not authenticated.
        """
        if not self.is_authenticated:
            return []
        if self._tools is None:
            self._tools = _generic_rest_tools(
                self.app_name,
                self.token_data["access_token"],  # type: ignore[index]
            )
        return self._tools or []

    def clear_auth(self) -> str:
        """Delete the stored authentication token for this app.

        Returns:
            Status message.
        """
        _clear_token(self.app_name)
        self.token_data = None
        self._tools = None
        return f"Authentication cleared for {self.display_name}"

    def create_app_agent(self) -> Path:
        """Create an app-specific agent Python file on disk.

        Generates a standalone Python file at
        ``<channels_dir>/<app>/<App>Agent.py`` that defines a
        ``<App>Agent`` class extending ``AppAgent``. The file can be run
        independently with ``python <App>Agent.py --task "..."`` to
        perform app-specific tasks.

        Returns:
            Path to the generated agent file.
        """
        class_name = self.display_name.replace(" ", "") + "Agent"
        file_name = f"{class_name}.py"
        agent_path = _CHANNELS_DIR / self.app_name / file_name
        agent_path.parent.mkdir(parents=True, exist_ok=True)
        content = _render_agent_template(
            app_name=self.app_name,
            display_name=self.display_name,
            class_name=class_name,
        )
        agent_path.write_text(content)
        return agent_path


# ---------------------------------------------------------------------------
# AppAgent
# ---------------------------------------------------------------------------


class AppAgent(SorcarAgent):
    """SorcarAgent extended with app-specific API tools and auth management.

    Inherits all standard SorcarAgent capabilities (bash, file editing,
    browser automation) and adds an authenticated generic REST API tool
    for the configured app plus auth management tools.
    """

    def __init__(
        self,
        channel: ChannelAgent,
        wait_for_user_callback: Any = None,
        ask_user_question_callback: Any = None,
    ) -> None:
        super().__init__(
            f"{channel.display_name} Agent",
            wait_for_user_callback=wait_for_user_callback,
            ask_user_question_callback=ask_user_question_callback,
        )
        self.channel = channel

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + app auth tools + app API tools."""
        tools = super()._get_tools()
        ch = self.channel

        def authenticate(token: str = "") -> str:
            """Authenticate with the app.

            Pass a token directly to authenticate. Call with no args to
            check current status.

            Args:
                token: Access token string (e.g. GitHub PAT, API key).

            Returns:
                Status or instructions.
            """
            if token:
                return ch.authenticate_with_token(token)
            if ch.is_authenticated:
                return f"Already authenticated with {ch.display_name}"
            return f"Not authenticated with {ch.display_name}"

        _default_redirect = "http://127.0.0.1:8585/callback"

        def configure_oauth2(
            auth_url: str,
            token_url: str,
            client_id_env: str,
            client_secret_env: str,
            scopes: str = "",
        ) -> str:
            """Configure OAuth2 settings for this app.

            Call this before starting the OAuth2 flow.

            Args:
                auth_url: OAuth2 authorization endpoint URL.
                token_url: OAuth2 token endpoint URL.
                client_id_env: Env var name for the client ID.
                client_secret_env: Env var name for the client secret.
                scopes: Space-separated OAuth2 scopes.

            Returns:
                Status message.
            """
            return ch.configure_oauth2(
                auth_url, token_url, client_id_env, client_secret_env, scopes
            )

        def authenticate_oauth2(
            code: str, redirect_uri: str = _default_redirect
        ) -> str:
            """Complete OAuth2 authentication with an authorization code.

            After the user authorizes in the browser, the callback URL contains
            a ``code`` parameter. Pass that code here to exchange it for tokens.

            Args:
                code: Authorization code from the OAuth2 callback URL.
                redirect_uri: Redirect URI used in the authorization request.

            Returns:
                Status message.
            """
            return ch.authenticate_oauth2(code, redirect_uri)

        def check_auth() -> str:
            """Check if the app is currently authenticated.

            Returns:
                Authentication status message.
            """
            if ch.is_authenticated:
                return f"Authenticated with {ch.display_name}"
            return (
                f"Not authenticated with {ch.display_name}. "
                f"Use authenticate(token=...) to set up."
            )

        def clear_auth() -> str:
            """Clear stored authentication for the app.

            Returns:
                Status message.
            """
            return ch.clear_auth()

        tools.extend([
            authenticate,
            configure_oauth2,
            authenticate_oauth2,
            check_auth,
            clear_auth,
        ])
        tools.extend(ch.get_tools())
        return tools


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def create_channel(app_name: str) -> ChannelAgent:
    """Create a ChannelAgent for the named app.

    Any app name is accepted — no hardcoded registry.

    Args:
        app_name: App identifier (e.g. 'github', 'google', 'spotify',
            'jira', 'slack', or any other app).

    Returns:
        Configured ChannelAgent instance.
    """
    return ChannelAgent(app_name)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Create an app-specific agent file on disk."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Channel Agent — create app-specific agent files"
    )
    parser.add_argument("app", help="App name (e.g. github, google, spotify)")
    args = parser.parse_args()

    channel = ChannelAgent(args.app)
    path = channel.create_app_agent()
    print(f"Created {channel.display_name} agent at: {path}")
    print(f"Run it with: uv run {path} --task 'your task here'")


if __name__ == "__main__":
    main()

"""Slack Agent — SorcarAgent extension with Slack API tools.

Provides authenticated access to a Slack workspace via a bot token.
Handles authentication (reading token from disk or prompting the user
via the browser), stores the token securely in
``~/.kiss/channels/slack/token.json``, and exposes a focused set of
Slack Web API tools that give the agent full control over messaging,
channels, users, reactions, and search.

Usage::

    agent = SlackAgent()
    agent.run(prompt_template="List all public channels in my workspace")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.sorcar.sorcar_agent import SorcarAgent

logger = logging.getLogger(__name__)

_SLACK_DIR = Path.home() / ".kiss" / "channels" / "slack"


# ---------------------------------------------------------------------------
# Token persistence
# ---------------------------------------------------------------------------


def _token_path() -> Path:
    """Return the path to the stored Slack bot token file.

    Returns:
        Path to ``~/.kiss/channels/slack/token.json``.
    """
    return _SLACK_DIR / "token.json"


def _load_token() -> str | None:
    """Load a stored Slack bot token from disk.

    Returns:
        The bot token string, or None if not found or invalid.
    """
    path = _token_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            tok = data.get("access_token", "")
            return tok if tok else None
        return None
    except (json.JSONDecodeError, OSError):
        return None


def _save_token(token: str) -> None:
    """Save a Slack bot token to disk with restricted permissions.

    Args:
        token: The bot token string (e.g. ``xoxb-...``).
    """
    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"access_token": token.strip()}, indent=2))
    path.chmod(0o600)


def _clear_token() -> None:
    """Delete the stored Slack bot token."""
    path = _token_path()
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Slack API tool functions
# ---------------------------------------------------------------------------


def _make_slack_tools(client: WebClient) -> list:
    """Create Slack API tool functions bound to the given WebClient.

    Args:
        client: Authenticated Slack WebClient instance.

    Returns:
        List of callable tool functions for Slack operations.
    """

    def list_channels(
        types: str = "public_channel", limit: int = 200, cursor: str = ""
    ) -> str:
        """List channels in the Slack workspace.

        Args:
            types: Comma-separated channel types. Options:
                public_channel, private_channel, mpim, im.
                Default: "public_channel".
            limit: Maximum number of channels to return (1-1000).
                Default: 200.
            cursor: Pagination cursor for next page of results.
                Pass the value from the previous response's
                response_metadata.next_cursor.

        Returns:
            JSON string with channel list (id, name, purpose, num_members)
            and pagination cursor.
        """
        try:
            kwargs: dict[str, Any] = {"types": types, "limit": min(limit, 1000)}
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.conversations_list(**kwargs)
            raw_channels: list[dict[str, Any]] = resp.get("channels", [])
            channels = [
                {
                    "id": ch["id"],
                    "name": ch.get("name", ""),
                    "is_private": ch.get("is_private", False),
                    "purpose": ch.get("purpose", {}).get("value", ""),
                    "num_members": ch.get("num_members", 0),
                }
                for ch in raw_channels
            ]
            result: dict[str, Any] = {"ok": True, "channels": channels}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def read_messages(
        channel: str, limit: int = 20, cursor: str = "", oldest: str = "", newest: str = ""
    ) -> str:
        """Read messages from a Slack channel.

        Args:
            channel: Channel ID (e.g. "C01234567").
            limit: Number of messages to return (1-1000). Default: 20.
            cursor: Pagination cursor for next page.
            oldest: Only messages after this Unix timestamp.
            newest: Only messages before this Unix timestamp.

        Returns:
            JSON string with messages (user, text, ts, thread_ts)
            and pagination cursor.
        """
        try:
            kwargs: dict[str, Any] = {"channel": channel, "limit": min(limit, 1000)}
            if cursor:
                kwargs["cursor"] = cursor
            if oldest:
                kwargs["oldest"] = oldest
            if newest:
                kwargs["newest"] = newest
            resp = client.conversations_history(**kwargs)
            raw_msgs: list[dict[str, Any]] = resp.get("messages", [])
            messages = [
                {
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts", ""),
                    "thread_ts": msg.get("thread_ts", ""),
                    "reply_count": msg.get("reply_count", 0),
                }
                for msg in raw_msgs
            ]
            result: dict[str, Any] = {"ok": True, "messages": messages}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def read_thread(channel: str, thread_ts: str, limit: int = 50, cursor: str = "") -> str:
        """Read replies in a message thread.

        Args:
            channel: Channel ID where the thread lives.
            thread_ts: Timestamp of the parent message.
            limit: Number of replies to return (1-1000). Default: 50.
            cursor: Pagination cursor for next page.

        Returns:
            JSON string with thread messages and pagination cursor.
        """
        try:
            kwargs: dict[str, Any] = {
                "channel": channel,
                "ts": thread_ts,
                "limit": min(limit, 1000),
            }
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.conversations_replies(**kwargs)
            raw_msgs: list[dict[str, Any]] = resp.get("messages", [])
            messages = [
                {
                    "user": msg.get("user", ""),
                    "text": msg.get("text", ""),
                    "ts": msg.get("ts", ""),
                }
                for msg in raw_msgs
            ]
            result: dict[str, Any] = {"ok": True, "messages": messages}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def send_message(
        channel: str, text: str, thread_ts: str = "", blocks: str = ""
    ) -> str:
        """Send a message to a Slack channel.

        Args:
            channel: Channel ID or name (e.g. "C01234567" or "#general").
            text: Message text (supports Slack mrkdwn formatting).
            thread_ts: Optional parent message timestamp to reply in a thread.
            blocks: Optional JSON string of Block Kit blocks for rich
                formatting. If provided, text becomes the fallback.

        Returns:
            JSON string with ok status and the message timestamp (ts).
        """
        try:
            kwargs: dict[str, Any] = {"channel": channel, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            if blocks:
                kwargs["blocks"] = json.loads(blocks)
            resp = client.chat_postMessage(**kwargs)
            return json.dumps(
                {"ok": True, "ts": resp.get("ts", ""), "channel": resp.get("channel", "")}
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def update_message(channel: str, ts: str, text: str, blocks: str = "") -> str:
        """Update an existing message in a Slack channel.

        Args:
            channel: Channel ID where the message is.
            ts: Timestamp of the message to update.
            text: New message text.
            blocks: Optional JSON string of Block Kit blocks.

        Returns:
            JSON string with ok status and updated timestamp.
        """
        try:
            kwargs: dict[str, Any] = {"channel": channel, "ts": ts, "text": text}
            if blocks:
                kwargs["blocks"] = json.loads(blocks)
            resp = client.chat_update(**kwargs)
            return json.dumps({"ok": True, "ts": resp.get("ts", "")})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def delete_message(channel: str, ts: str) -> str:
        """Delete a message from a Slack channel.

        Args:
            channel: Channel ID where the message is.
            ts: Timestamp of the message to delete.

        Returns:
            JSON string with ok status.
        """
        try:
            client.chat_delete(channel=channel, ts=ts)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def list_users(limit: int = 200, cursor: str = "") -> str:
        """List users in the Slack workspace.

        Args:
            limit: Maximum number of users to return (1-1000). Default: 200.
            cursor: Pagination cursor for next page.

        Returns:
            JSON string with user list (id, name, real_name, is_bot)
            and pagination cursor.
        """
        try:
            kwargs: dict[str, Any] = {"limit": min(limit, 1000)}
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.users_list(**kwargs)
            raw_members: list[dict[str, Any]] = resp.get("members", [])
            users = [
                {
                    "id": u["id"],
                    "name": u.get("name", ""),
                    "real_name": u.get("real_name", ""),
                    "is_bot": u.get("is_bot", False),
                    "is_admin": u.get("is_admin", False),
                }
                for u in raw_members
            ]
            result: dict[str, Any] = {"ok": True, "users": users}
            next_cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if next_cursor:
                result["next_cursor"] = next_cursor
            return json.dumps(result, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_user_info(user: str) -> str:
        """Get detailed information about a Slack user.

        Args:
            user: User ID (e.g. "U01234567").

        Returns:
            JSON string with user profile details.
        """
        try:
            resp = client.users_info(user=user)
            u: dict[str, Any] = resp.get("user", {})
            profile: dict[str, Any] = u.get("profile", {})
            return json.dumps(
                {
                    "ok": True,
                    "user": {
                        "id": u.get("id", ""),
                        "name": u.get("name", ""),
                        "real_name": u.get("real_name", ""),
                        "display_name": profile.get("display_name", ""),
                        "email": profile.get("email", ""),
                        "title": profile.get("title", ""),
                        "is_bot": u.get("is_bot", False),
                        "is_admin": u.get("is_admin", False),
                        "tz": u.get("tz", ""),
                    },
                },
                indent=2,
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def create_channel(name: str, is_private: bool = False) -> str:
        """Create a new Slack channel.

        Args:
            name: Channel name (lowercase, no spaces, max 80 chars).
                Use hyphens instead of spaces.
            is_private: If True, create a private channel. Default: False.

        Returns:
            JSON string with the new channel's id and name.
        """
        try:
            resp = client.conversations_create(name=name, is_private=is_private)
            ch: dict[str, Any] = resp.get("channel", {})
            return json.dumps({
                "ok": True,
                "channel": {"id": ch.get("id", ""), "name": ch.get("name", "")},
            })
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def invite_to_channel(channel: str, users: str) -> str:
        """Invite users to a Slack channel.

        Args:
            channel: Channel ID to invite users to.
            users: Comma-separated list of user IDs to invite.

        Returns:
            JSON string with ok status.
        """
        try:
            client.conversations_invite(channel=channel, users=users)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def add_reaction(channel: str, timestamp: str, name: str) -> str:
        """Add an emoji reaction to a message.

        Args:
            channel: Channel ID where the message is.
            timestamp: Timestamp of the message to react to.
            name: Emoji name without colons (e.g. "thumbsup", "heart").

        Returns:
            JSON string with ok status.
        """
        try:
            client.reactions_add(channel=channel, timestamp=timestamp, name=name)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def search_messages(query: str, count: int = 20, sort: str = "timestamp") -> str:
        """Search for messages across the workspace.

        Note: Requires a user token with search:read scope.
        Bot tokens cannot use this method.

        Args:
            query: Search query string (supports Slack search modifiers
                like "in:#channel", "from:@user", "has:link").
            count: Number of results to return (1-100). Default: 20.
            sort: Sort order — "timestamp" (default) or "score".

        Returns:
            JSON string with matching messages.
        """
        try:
            resp = client.search_messages(query=query, count=min(count, 100), sort=sort)
            msg_data: dict[str, Any] = resp.get("messages", {})
            matches: list[dict[str, Any]] = msg_data.get("matches", [])
            results = [
                {
                    "text": m.get("text", ""),
                    "user": m.get("user", ""),
                    "ts": m.get("ts", ""),
                    "channel": m.get("channel", {}).get("name", ""),
                    "permalink": m.get("permalink", ""),
                }
                for m in matches
            ]
            return json.dumps({"ok": True, "messages": results}, indent=2)[:8000]
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def set_channel_topic(channel: str, topic: str) -> str:
        """Set the topic for a Slack channel.

        Args:
            channel: Channel ID.
            topic: New topic text.

        Returns:
            JSON string with ok status.
        """
        try:
            client.conversations_setTopic(channel=channel, topic=topic)
            return json.dumps({"ok": True})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def upload_file(channels: str, content: str, filename: str, title: str = "") -> str:
        """Upload text content as a file to Slack channels.

        Args:
            channels: Comma-separated channel IDs to share the file in.
            content: Text content of the file.
            filename: Name for the file (e.g. "report.txt").
            title: Optional title for the file.

        Returns:
            JSON string with ok status and file id.
        """
        try:
            channel_list = [c.strip() for c in channels.split(",") if c.strip()]
            resp = client.files_upload_v2(
                channels=channel_list,
                content=content,
                filename=filename,
                title=title or filename,
            )
            file_data: dict[str, Any] = resp.get("file", {})
            return json.dumps({"ok": True, "file_id": file_data.get("id", "")})
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    def get_channel_info(channel: str) -> str:
        """Get detailed information about a Slack channel.

        Args:
            channel: Channel ID (e.g. "C01234567").

        Returns:
            JSON string with channel details (name, topic, purpose,
            num_members, created, creator).
        """
        try:
            resp = client.conversations_info(channel=channel)
            ch: dict[str, Any] = resp.get("channel", {})
            return json.dumps(
                {
                    "ok": True,
                    "channel": {
                        "id": ch.get("id", ""),
                        "name": ch.get("name", ""),
                        "topic": ch.get("topic", {}).get("value", ""),
                        "purpose": ch.get("purpose", {}).get("value", ""),
                        "num_members": ch.get("num_members", 0),
                        "is_private": ch.get("is_private", False),
                        "created": ch.get("created", 0),
                        "creator": ch.get("creator", ""),
                    },
                },
                indent=2,
            )
        except SlackApiError as e:
            return json.dumps({"ok": False, "error": str(e)})

    return [
        list_channels,
        read_messages,
        read_thread,
        send_message,
        update_message,
        delete_message,
        list_users,
        get_user_info,
        create_channel,
        invite_to_channel,
        add_reaction,
        search_messages,
        set_channel_topic,
        upload_file,
        get_channel_info,
    ]


# ---------------------------------------------------------------------------
# SlackAgent
# ---------------------------------------------------------------------------


def _cli_wait_for_user(instruction: str, url: str) -> None:
    """CLI callback for browser-action prompts (prints and waits for Enter).

    Args:
        instruction: What the user should do.
        url: Current browser URL (printed if non-empty).
    """
    print(f"\n>>> Browser action needed: {instruction}")
    if url:
        print(f"    Current URL: {url}")
    input("Press Enter when done... ")


def _cli_ask_user_question(question: str) -> str:
    """CLI callback for agent questions (prints and reads from stdin).

    Args:
        question: The question to display to the user.

    Returns:
        The user's typed response text.
    """
    print(f"\n>>> Agent asks: {question}")
    return input("Your answer: ")


class SlackAgent(SorcarAgent):
    """SorcarAgent extended with Slack workspace tools.

    Inherits all standard SorcarAgent capabilities (bash, file editing,
    browser automation) and adds authenticated Slack API tools for
    messaging, channel management, user lookup, reactions, search,
    and file uploads.

    The agent checks for a stored bot token on initialization. If no
    token is found, authentication tools guide the user through
    obtaining and storing one.

    Example::

        agent = SlackAgent()
        result = agent.run(
            prompt_template="Send 'Hello!' to #general",
            headless=False,
        )
    """

    def __init__(
        self,
        wait_for_user_callback: Any = None,
        ask_user_question_callback: Any = None,
    ) -> None:
        super().__init__(
            "Slack Agent",
            wait_for_user_callback=wait_for_user_callback,
            ask_user_question_callback=ask_user_question_callback,
        )
        self._slack_client: WebClient | None = None
        token = _load_token()
        if token:
            self._slack_client = WebClient(token=token)

    def _get_tools(self) -> list:
        """Return SorcarAgent tools + Slack auth tools + Slack API tools."""
        tools = super()._get_tools()
        agent = self

        def check_slack_auth() -> str:
            """Check if the Slack bot token is configured and valid.

            Tests the stored token against the Slack API (auth.test).

            Returns:
                Authentication status with workspace and bot user info,
                or instructions for how to authenticate.
            """
            if agent._slack_client is None:
                return (
                    "Not authenticated with Slack. Use authenticate_slack(token=...) "
                    "to set a bot token. The token should start with 'xoxb-'. "
                    "Get it from https://api.slack.com/apps > your app > "
                    "OAuth & Permissions > Bot User OAuth Token."
                )
            try:
                resp = agent._slack_client.auth_test()
                return json.dumps(
                    {
                        "ok": True,
                        "team": resp.get("team", ""),
                        "user": resp.get("user", ""),
                        "bot_id": resp.get("bot_id", ""),
                        "url": resp.get("url", ""),
                    }
                )
            except SlackApiError as e:
                return json.dumps({"ok": False, "error": str(e)})

        def authenticate_slack(token: str) -> str:
            """Store and validate a Slack bot token.

            Saves the token to ~/.kiss/channels/slack/token.json and
            validates it with auth.test.

            Args:
                token: Slack bot token (starts with 'xoxb-' for bot tokens
                    or 'xoxp-' for user tokens).

            Returns:
                Validation result with workspace info, or error message.
            """
            token = token.strip()
            if not token:
                return "Token cannot be empty."
            agent._slack_client = WebClient(token=token)
            try:
                resp = agent._slack_client.auth_test()
                _save_token(token)
                return json.dumps(
                    {
                        "ok": True,
                        "message": "Slack token saved and validated.",
                        "team": resp.get("team", ""),
                        "user": resp.get("user", ""),
                    }
                )
            except SlackApiError as e:
                agent._slack_client = None
                return json.dumps(
                    {"ok": False, "error": f"Token validation failed: {e}"}
                )

        def clear_slack_auth() -> str:
            """Clear the stored Slack authentication token.

            Returns:
                Status message.
            """
            _clear_token()
            agent._slack_client = None
            return "Slack authentication cleared."

        tools.extend([check_slack_auth, authenticate_slack, clear_slack_auth])

        if agent._slack_client is not None:
            tools.extend(_make_slack_tools(agent._slack_client))

        return tools


def main() -> None:
    """Run the SlackAgent from the command line with a --task argument."""
    import argparse
    import os
    import tempfile
    import time as time_mod

    import yaml

    parser = argparse.ArgumentParser(description="Run SlackAgent on a task")
    parser.add_argument("--task", type=str, required=True, help="Task description for the agent")
    parser.add_argument("--model_name", type=str, default=None, help="LLM model name")
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum number of steps")
    parser.add_argument("--max_budget", type=float, default=5.0, help="Maximum budget in USD")
    parser.add_argument("--work_dir", type=str, default=None, help="Working directory")
    parser.add_argument(
        "--headless",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="Run browser headless (true/false)",
    )
    parser.add_argument(
        "--verbose",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="Print output to console (true/false)",
    )
    args = parser.parse_args()

    if args.work_dir is not None:
        work_dir = args.work_dir
        Path(work_dir).mkdir(parents=True, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp()

    agent = SlackAgent(
        wait_for_user_callback=_cli_wait_for_user,
        ask_user_question_callback=_cli_ask_user_question,
    )
    old_cwd = os.getcwd()
    os.chdir(work_dir)
    start_time = time_mod.time()
    try:
        result = agent.run(
            prompt_template=args.task,
            model_name=args.model_name,
            max_steps=args.max_steps,
            max_budget=args.max_budget,
            work_dir=work_dir,
            headless=args.headless,
            verbose=args.verbose,
        )
    finally:
        os.chdir(old_cwd)
    elapsed = time_mod.time() - start_time

    print("FINAL RESULT:")
    result_data = yaml.safe_load(result)
    print("Completed successfully: " + str(result_data["success"]))
    print(result_data["summary"])
    print("Work directory was: " + work_dir)
    print(f"Time: {elapsed:.1f}s")
    print(f"Cost: ${agent.budget_used:.4f}")
    print(f"Total tokens: {agent.total_tokens_used}")


if __name__ == "__main__":
    main()

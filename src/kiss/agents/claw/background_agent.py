"""Background agent that listens for tasks on Slack #sorcar channel.

Polls the #sorcar Slack channel for messages from user ksen, treats each
message as a task, completes it using SorcarAgent, and sends results back
to the channel. Agent callbacks (cli_wait_for_user, cli_ask_user_question)
are routed through Slack thread replies for interactive feedback.

Usage::

    uv run python -m kiss.agents.claw.background_agent
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.channels.slack_agent import _load_token

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 3.0  # seconds between polling for new messages
_REPLY_POLL_INTERVAL = 2.0  # seconds between polling for user replies
_CHANNEL_NAME = "sorcar"


def _find_channel_id(client: WebClient, name: str) -> str | None:
    """Find a channel ID by name.

    Args:
        client: Authenticated Slack WebClient.
        name: Channel name without '#'.

    Returns:
        Channel ID string, or None if not found.
    """
    cursor = ""
    while True:
        kwargs: dict[str, Any] = {"types": "public_channel", "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.conversations_list(**kwargs)
        channels: list[dict[str, Any]] = resp.get("channels", [])
        for ch_dict in channels:
            if ch_dict.get("name") == name:
                return str(ch_dict["id"])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
        if not cursor:
            return None


def _find_user_id(client: WebClient, username: str) -> str | None:
    """Find a user ID by display name or username.

    Args:
        client: Authenticated Slack WebClient.
        username: Slack username (without @).

    Returns:
        User ID string, or None if not found.
    """
    cursor = ""
    while True:
        kwargs: dict[str, Any] = {"limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.users_list(**kwargs)
        members: list[dict[str, Any]] = resp.get("members", [])
        for u_dict in members:
            name_match = u_dict.get("name") == username
            real_match = str(u_dict.get("real_name", "")).lower() == username.lower()
            if name_match or real_match:
                return str(u_dict["id"])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
        if not cursor:
            return None


def _wait_for_thread_reply(
    client: WebClient, channel_id: str, thread_ts: str, user_id: str, bot_user_id: str
) -> str:
    """Poll a Slack thread for a reply from a specific user.

    Args:
        client: Authenticated Slack WebClient.
        channel_id: Channel ID containing the thread.
        thread_ts: Timestamp of the parent message (thread root).
        user_id: User ID to wait for a reply from.
        bot_user_id: Bot's own user ID (to ignore bot messages).

    Returns:
        The text of the user's reply message.
    """
    seen_ts: set[str] = set()
    # Mark all existing thread messages as seen
    try:
        resp = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=100)
        existing: list[dict[str, Any]] = resp.get("messages", [])
        for msg_dict in existing:
            seen_ts.add(str(msg_dict["ts"]))
    except SlackApiError:
        pass

    while True:
        time.sleep(_REPLY_POLL_INTERVAL)
        try:
            resp = client.conversations_replies(channel=channel_id, ts=thread_ts, limit=100)
            replies: list[dict[str, Any]] = resp.get("messages", [])
            for reply in replies:
                ts = str(reply["ts"])
                if ts in seen_ts:
                    continue
                seen_ts.add(ts)
                if reply.get("user") == user_id:
                    return str(reply.get("text", ""))
        except SlackApiError:
            logger.debug("Error polling thread replies", exc_info=True)


def _run_task(
    client: WebClient,
    channel_id: str,
    user_id: str,
    bot_user_id: str,
    task_text: str,
    thread_ts: str,
    work_dir: str,
) -> None:
    """Run a single task using SorcarAgent and post results to Slack.

    Args:
        client: Authenticated Slack WebClient.
        channel_id: Channel ID to post results to.
        user_id: User ID of the task requester (for reply polling).
        bot_user_id: Bot's own user ID.
        task_text: The task description.
        thread_ts: Thread timestamp to reply in.
        work_dir: Working directory for the agent.
    """

    def slack_wait_for_user(instruction: str, url: str) -> None:
        """Send a browser action prompt to Slack and wait for user reply."""
        msg = f"🔔 *Browser action needed:*\n{instruction}"
        if url:
            msg += f"\n_Current URL:_ {url}"
        msg += "\n\n_Reply in this thread when done._"
        client.chat_postMessage(channel=channel_id, text=msg, thread_ts=thread_ts)
        _wait_for_thread_reply(client, channel_id, thread_ts, user_id, bot_user_id)

    def slack_ask_user_question(question: str) -> str:
        """Send a question to Slack and wait for user's reply."""
        msg = f"❓ *Agent asks:*\n{question}\n\n_Reply in this thread with your answer._"
        client.chat_postMessage(channel=channel_id, text=msg, thread_ts=thread_ts)
        return _wait_for_thread_reply(client, channel_id, thread_ts, user_id, bot_user_id)

    agent = SorcarAgent(
        "Claw Background Agent",
        wait_for_user_callback=slack_wait_for_user,
        ask_user_question_callback=slack_ask_user_question,
    )
    agent.web_use_tool = None  # No GUI/browser needed

    client.chat_postMessage(
        channel=channel_id,
        text=f"⚙️ Working on task:\n> {task_text[:500]}",
        thread_ts=thread_ts,
    )

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        result = agent.run(
            prompt_template=task_text,
            work_dir=work_dir,
            verbose=True,
        )
    except Exception as e:
        logger.error("Agent error", exc_info=True)
        result = yaml.dump({"success": False, "summary": f"Agent error: {e}"})
    finally:
        os.chdir(old_cwd)

    result_data = yaml.safe_load(result)
    success = result_data.get("success", False)
    summary = result_data.get("summary", "No summary available.")
    emoji = "✅" if success else "❌"
    msg = f"{emoji} *Task {'completed' if success else 'failed'}*\n\n{summary}"
    # Slack message limit is 40000 chars; truncate if needed
    if len(msg) > 3900:
        msg = msg[:3900] + "\n... (truncated)"
    client.chat_postMessage(channel=channel_id, text=msg, thread_ts=thread_ts)

    cost = f"${agent.budget_used:.4f}" if hasattr(agent, "budget_used") else "unknown"
    tokens = agent.total_tokens_used if hasattr(agent, "total_tokens_used") else "unknown"
    client.chat_postMessage(
        channel=channel_id,
        text=f"📊 Cost: {cost} | Tokens: {tokens}",
        thread_ts=thread_ts,
    )


def run_background_agent(work_dir: str | None = None) -> None:
    """Main loop: poll #sorcar for tasks from ksen, run them, post results.

    Args:
        work_dir: Working directory for agent tasks. Defaults to a temp dir.
    """
    token = _load_token()
    if not token:
        print(
            "No Slack token found. Please store a bot token first.\n"
            "Run: uv run python -m kiss.channels.slack_agent --task 'check auth'\n"
            "Or manually save token to ~/.kiss/channels/slack/token.json"
        )
        return

    client = WebClient(token=token)

    # Verify auth
    try:
        auth = client.auth_test()
        bot_user_id = auth.get("user_id", "")
        print(f"Authenticated as {auth.get('user', '')} in {auth.get('team', '')}")
    except SlackApiError as e:
        print(f"Slack auth failed: {e}")
        return

    # Find channel
    channel_id = _find_channel_id(client, _CHANNEL_NAME)
    if not channel_id:
        print(f"Channel #{_CHANNEL_NAME} not found. Please create it first.")
        return
    print(f"Monitoring #{_CHANNEL_NAME} (ID: {channel_id})")

    # Join the channel (bot needs to be a member)
    try:
        client.conversations_join(channel=channel_id)
    except SlackApiError:
        pass  # Already a member or can't join

    # Find ksen user
    user_id = _find_user_id(client, "ksen")
    if not user_id:
        print("User 'ksen' not found. Will accept messages from any user.")
        user_id = None

    if user_id:
        print(f"Watching for messages from ksen (ID: {user_id})")
    else:
        print("Watching for messages from any user")

    resolved_work_dir = work_dir or tempfile.mkdtemp(prefix="claw_")
    Path(resolved_work_dir).mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {resolved_work_dir}")

    # Start polling from now — Slack requires ≤6 decimal places
    last_ts = f"{time.time():.6f}"
    print(f"\n🤖 Background agent ready. Send a message in #{_CHANNEL_NAME} to start a task.\n")

    client.chat_postMessage(
        channel=channel_id,
        text="🤖 Claw background agent is now online and listening for tasks.",
    )

    while True:
        try:
            resp = client.conversations_history(
                channel=channel_id, oldest=last_ts, limit=10
            )
            messages: list[dict[str, Any]] = resp.get("messages", [])
            # Process oldest first
            messages.sort(key=lambda m: float(m.get("ts", "0")))

            for msg in messages:
                msg_ts = msg.get("ts", "")
                msg_user = msg.get("user", "")
                msg_text = msg.get("text", "").strip()

                # Update last_ts to avoid reprocessing
                if float(msg_ts) >= float(last_ts):
                    last_ts = f"{float(msg_ts) + 0.000001:.6f}"

                # Skip bot messages
                if msg.get("bot_id") or msg_user == bot_user_id:
                    continue

                # Skip if not from ksen (when user_id is known)
                if user_id and msg_user != user_id:
                    continue

                # Skip empty messages
                if not msg_text:
                    continue

                # Strip bot mention if present
                if bot_user_id:
                    msg_text = msg_text.replace(f"<@{bot_user_id}>", "").strip()

                print(f"\n📩 New task from {msg_user}: {msg_text[:100]}...", flush=True)
                _run_task(
                    client=client,
                    channel_id=channel_id,
                    user_id=msg_user,
                    bot_user_id=bot_user_id,
                    task_text=msg_text,
                    thread_ts=msg_ts,
                    work_dir=resolved_work_dir,
                )

        except SlackApiError as e:
            logger.error("Slack API error during polling: %s", e)
            time.sleep(10)
        except KeyboardInterrupt:
            print("\n\nShutting down background agent...")
            try:
                client.chat_postMessage(
                    channel=channel_id,
                    text="🔴 Claw background agent is shutting down.",
                )
            except SlackApiError:
                pass
            break
        except Exception:
            logger.error("Unexpected error in polling loop", exc_info=True)
            time.sleep(10)

        time.sleep(_POLL_INTERVAL)


def main() -> None:
    """Entry point for the background agent CLI."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Claw background agent for Slack")
    parser.add_argument("--work-dir", type=str, default=".", help="Working directory for tasks")
    args = parser.parse_args()
    run_background_agent(work_dir=args.work_dir)


if __name__ == "__main__":
    main()

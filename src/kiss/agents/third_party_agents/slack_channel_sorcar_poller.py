# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Slack channel poller that dispatches messages from ksen to Sorcar.

Designed to be invoked once per minute from cron.  The script holds a
process-wide ``fcntl`` lock so concurrent cron ticks exit immediately,
then polls the configured Slack channel every ``POLL_INTERVAL`` seconds
for ``RUN_DURATION`` seconds before exiting.  The next cron tick
restarts the loop, giving effective 3-second polling under cron's
one-minute scheduling granularity.

Behaviour for each new message from the target user:

* **Top-level message** — runs a brand-new ``ChatSorcarAgent`` chat,
  posts the result back as a threaded reply, and records the new
  ``chat_id`` against the message's ``ts`` so future replies in the
  same thread continue the same chat.
* **Threaded reply** (after the bot has already responded in that
  thread) — resumes the stored ``chat_id`` for the thread, runs the
  reply as the next task in that chat, and posts the response as a
  further threaded reply.

State is persisted under ``~/.kiss/slack_channel_sorcar_poller/`` so
successive cron invocations share their thread-to-chat mapping and
never respond to the same message twice.

Environment variables (all optional):

* ``KISS_SLACK_WORKSPACE`` — Slack workspace key (default
  ``"learningsystems"``).
* ``KISS_SLACK_USER`` — Slack handle or real name of the user whose
  messages to handle (default ``"ksen"``).
* ``KISS_SLACK_CHANNEL`` — Channel name to poll (default
  ``"sorcar"``).
* ``KISS_SLACK_MODEL`` — Model name passed to ``ChatSorcarAgent.run``.
* ``KISS_SLACK_BUDGET`` — Per-task budget in USD (default ``5.0``).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pwd
import re
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from kiss.agents.third_party_agents._kiss_web_launcher import (
    KissWebWorktreeSorcarAgent,
    run_agent_via_kiss_web,
)
from kiss.agents.third_party_agents.slack_agent import _load_token
from kiss.agents.vscode.vscode_config import load_config, source_shell_env
from kiss.core.models.model_info import get_available_models

STATE_DIR = Path.home() / ".kiss" / "slack_channel_sorcar_poller"
STATE_FILE = STATE_DIR / "state.json"
LOG_FILE = STATE_DIR / "poller.log"
LOCK_FILE = STATE_DIR / "poller.lock"
HEALTH_ALERT_FILE = STATE_DIR / "health_alert.json"

POLL_INTERVAL = 3.0
RUN_DURATION = 57.0
HEALTH_ALERT_COOLDOWN = 3600.0

WORKSPACE = os.environ.get("KISS_SLACK_WORKSPACE", "learningsystems")
USER_NAME = os.environ.get("KISS_SLACK_USER", "ksen")
CHANNEL_NAME = os.environ.get("KISS_SLACK_CHANNEL", "sorcar")
MODEL_NAME = os.environ.get("KISS_SLACK_MODEL", "")
MAX_BUDGET = float(os.environ.get("KISS_SLACK_BUDGET", "5.0"))

SLACK_FORMATTING_HINT = (
    "\n\n## Reply formatting\n"
    "Your final ``summary`` will be posted to Slack verbatim.  Format it "
    "with Slack mrkdwn: ``*bold*`` (single asterisks), ``_italic_`` "
    "(single underscores), ``~strike~``, ``` `code` ```, fenced code "
    "blocks with triple backticks, ``<url|label>`` for links, and "
    "``- item`` for bullets.  Do NOT use markdown ``**bold**``, "
    "``__italic__`` or ``[label](url)`` — Slack will render them "
    "literally."
)


def _setup_logging() -> None:
    """Configure file-based logging for the poller."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
    )


def _acquire_lock() -> Any:
    """Acquire an exclusive non-blocking lock; exit if another instance holds it.

    Returns:
        The open lock file descriptor; keep the reference for the
        process lifetime so the lock is held until exit.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    # Open without truncating so a losing contender never clobbers the
    # PID recorded by the current lock holder.
    fp = LOCK_FILE.open("a+", encoding="utf-8")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fp.close()
        sys.exit(0)
    fp.truncate(0)
    fp.write(f"{os.getpid()}\n")
    fp.flush()
    return fp


def _load_state() -> dict[str, Any]:
    """Read the on-disk state file or return a fresh state dict."""
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("threads", {})
                return data
        except (json.JSONDecodeError, OSError):
            logging.exception("Failed to read state file")
    return {"threads": {}}


def _save_state(state: dict[str, Any]) -> None:
    """Persist ``state`` to disk atomically."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def _get_client() -> WebClient:
    """Build a Slack WebClient using the stored bot token."""
    token = _load_token(WORKSPACE)
    if not token:
        raise RuntimeError(
            f"No Slack token for workspace {WORKSPACE!r}. "
            f"Run: kiss-slack --workspace {WORKSPACE} -t 'authenticate'"
        )
    return WebClient(token=token, retry_handlers=[])


def _find_user_id(client: WebClient, username: str) -> str:
    """Return the user ID for ``username`` (handle, real name, or display name)."""
    cursor = ""
    target = username.lower()
    while True:
        kwargs: dict[str, Any] = {"limit": 200}
        if cursor:
            kwargs["cursor"] = cursor
        resp = client.users_list(**kwargs)
        members: list[dict[str, Any]] = resp.get("members", [])
        for u in members:
            names = {
                str(u.get("name", "")).lower(),
                str(u.get("real_name", "")).lower(),
                str(u.get("profile", {}).get("display_name", "")).lower(),
            }
            if target in names and target:
                return str(u["id"])
        cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
        if not cursor:
            break
    raise RuntimeError(f"User {username!r} not found in workspace {WORKSPACE!r}")


def _find_channel_id(client: WebClient, channel_name: str) -> str:
    """Find a channel by name, searching both public and private channels.

    Ensures the bot has joined the channel it finds.

    Raises:
        RuntimeError: If the channel is not found in the workspace; the
            channel must be created in Slack first.
    """
    for types in ("public_channel", "private_channel"):
        cursor = ""
        while True:
            kwargs: dict[str, Any] = {"types": types, "limit": 200}
            if cursor:
                kwargs["cursor"] = cursor
            resp = client.conversations_list(**kwargs)
            channels: list[dict[str, Any]] = resp.get("channels", [])
            for ch in channels:
                if ch.get("name") == channel_name:
                    channel_id = str(ch["id"])
                    try:
                        client.conversations_join(channel=channel_id)
                    except SlackApiError:
                        pass
                    return channel_id
            cursor = (resp.get("response_metadata") or {}).get("next_cursor", "")
            if not cursor:
                break

    raise RuntimeError(
        f"Channel {channel_name!r} not found in workspace {WORKSPACE!r}. "
        f"Please create it in Slack first."
    )


def _markdown_to_mrkdwn(text: str) -> str:
    """Convert a small subset of Markdown to Slack mrkdwn."""
    text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", text, flags=re.DOTALL)
    text = re.sub(r"__(.+?)__", r"_\1_", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)
    return text


def _extract_summary(result: str) -> str:
    """Pull the ``summary`` field out of a Sorcar YAML result."""
    try:
        parsed = yaml.safe_load(result)
        if isinstance(parsed, dict) and parsed.get("summary"):
            return str(parsed["summary"])
    except yaml.YAMLError:
        logging.exception("Failed to parse Sorcar YAML result")
    return result.strip() or "(empty response)"


def _strip_bot_mention(text: str, bot_id: str) -> str:
    """Remove bot @mention from message text."""
    return text.replace(f"<@{bot_id}>", "").strip()


def _is_empty_reply(reply: str) -> bool:
    """Return True if ``reply`` is blank or the empty-result placeholder.

    Such replies mean the Sorcar task never produced a result (e.g. the
    process was killed mid-run), so the poller must not post them and
    must leave the message unanswered so the next tick retries it.
    """
    return not reply.strip() or reply.strip() == "(empty response)"


def _run_sorcar(prompt: str, chat_id: str) -> tuple[str, str]:
    """Run a Sorcar task and return ``(slack_text, chat_id)``.

    Uses :class:`WorktreeSorcarAgent` so that ``update_settings`` calls
    for ``is_worktree``, ``auto_commit``, and ``demo_mode`` have real
    effects (git worktree isolation, actual git commits, persisted
    config) even without a UI.

    Args:
        prompt: The user's Slack message text.
        chat_id: Existing chat to resume, or empty for a new chat.
    """
    agent = KissWebWorktreeSorcarAgent("Slack Channel Sorcar Poller")
    if chat_id:
        agent.resume_chat_by_id(chat_id)
    else:
        agent.new_chat()
    full_prompt = prompt + SLACK_FORMATTING_HINT

    cfg = load_config()
    use_worktree = bool(cfg.get("is_worktree", False))

    # Launch as a kiss-web registered agent via ``_cmd_run`` so the
    # task is live-visible / interactable from remote webviews.
    try:
        result = run_agent_via_kiss_web(
            agent,
            full_prompt,
            model_name=MODEL_NAME,
            max_budget=MAX_BUDGET,
            use_worktree=use_worktree,
        )
    finally:
        if agent._wt_pending:
            agent._release_worktree()
    return _markdown_to_mrkdwn(_extract_summary(result)), agent.chat_id


def _bot_replied(messages: list[dict[str, Any]], bot_id: str, parent_ts: str) -> bool:
    """Return True if the bot has posted in this thread (excluding ``parent_ts``)."""
    for m in messages:
        if m.get("ts") == parent_ts:
            continue
        if m.get("user") == bot_id or m.get("bot_id"):
            return True
    return False


def _latest_bot_ts(messages: list[dict[str, Any]], bot_id: str) -> float:
    """Return the ts of the most recent bot message in ``messages`` or ``0.0``."""
    bot_ts = [
        float(m["ts"])
        for m in messages
        if (m.get("user") == bot_id or m.get("bot_id")) and m.get("ts")
    ]
    return max(bot_ts, default=0.0)


def _post(client: WebClient, channel_id: str, text: str, thread_ts: str) -> None:
    """Post ``text`` to ``channel_id`` as a threaded reply."""
    client.chat_postMessage(channel=channel_id, text=text, thread_ts=thread_ts)


def _handle_top_level(
    client: WebClient,
    channel_id: str,
    msg: dict[str, Any],
    state: dict[str, Any],
    bot_id: str,
) -> None:
    """Run a new Sorcar chat for a top-level user message and reply."""
    ts = str(msg["ts"])
    text = _strip_bot_mention(str(msg.get("text", "")).strip(), bot_id)
    if not text:
        return
    logging.info("New top-level message ts=%s text=%r", ts, text[:200])
    reply, chat_id = _run_sorcar(text, chat_id="")
    if _is_empty_reply(reply):
        logging.warning(
            "Empty Sorcar result for ts=%s (chat_id=%s); not posting, will retry next tick",
            ts,
            chat_id,
        )
        return
    try:
        _post(client, channel_id, reply, thread_ts=ts)
    except SlackApiError:
        logging.exception("Failed to post reply for ts=%s", ts)
        return
    state["threads"][ts] = {"chat_id": chat_id}
    _save_state(state)


def _handle_thread_replies(
    client: WebClient,
    channel_id: str,
    parent_ts: str,
    bot_id: str,
    user_id: str,
    state: dict[str, Any],
) -> None:
    """Run continuations for any new user replies in a known thread."""
    try:
        resp = client.conversations_replies(channel=channel_id, ts=parent_ts, limit=200)
    except SlackApiError:
        logging.exception("Failed to fetch thread replies ts=%s", parent_ts)
        return
    messages: list[dict[str, Any]] = list(resp.get("messages", []))
    last_bot = _latest_bot_ts(messages, bot_id)
    new_user_msgs = [
        m
        for m in messages
        if m.get("user") == user_id
        and m.get("ts") != parent_ts
        and float(m.get("ts", "0")) > last_bot
    ]
    new_user_msgs.sort(key=lambda m: float(m.get("ts", "0")))
    for reply_msg in new_user_msgs:
        text = _strip_bot_mention(str(reply_msg.get("text", "")).strip(), bot_id)
        if not text:
            continue
        chat_id = state["threads"].get(parent_ts, {}).get("chat_id", "")
        logging.info(
            "Thread continuation ts=%s parent=%s chat_id=%s text=%r",
            reply_msg.get("ts"),
            parent_ts,
            chat_id,
            text[:200],
        )
        reply, new_chat_id = _run_sorcar(text, chat_id=chat_id)
        if _is_empty_reply(reply):
            logging.warning(
                "Empty Sorcar result for thread reply ts=%s parent=%s (chat_id=%s); "
                "not posting, will retry next tick",
                reply_msg.get("ts"),
                parent_ts,
                new_chat_id,
            )
            return
        try:
            _post(client, channel_id, reply, thread_ts=parent_ts)
        except SlackApiError:
            logging.exception("Failed to post continuation reply for parent=%s", parent_ts)
            return
        state["threads"][parent_ts] = {"chat_id": new_chat_id}
        _save_state(state)


def _poll_once(
    client: WebClient,
    channel_id: str,
    bot_id: str,
    user_id: str,
    state: dict[str, Any],
) -> None:
    """One polling pass: handle every unresponded message from ``user_id``."""
    try:
        resp = client.conversations_history(channel=channel_id, limit=50)
    except SlackApiError:
        logging.exception("Failed to fetch channel history")
        return
    messages: list[dict[str, Any]] = sorted(
        resp.get("messages", []), key=lambda m: float(m.get("ts", "0"))
    )

    # Only process messages newer than the startup watermark so we never
    # retroactively run old channel discussions as Sorcar tasks.
    min_ts = float(state.get("min_ts", "0"))

    for msg in messages:
        ts = str(msg.get("ts", ""))
        if not ts:
            continue
        if float(ts) <= min_ts:
            continue
        if msg.get("user") != user_id:
            continue
        thread_ts = msg.get("thread_ts") or ts
        is_top_level = thread_ts == ts
        if not is_top_level:
            continue
        if msg.get("reply_count", 0) > 0:
            try:
                rep = client.conversations_replies(channel=channel_id, ts=ts, limit=200)
            except SlackApiError:
                logging.exception("Failed to fetch replies for ts=%s", ts)
                continue
            thread_msgs = list(rep.get("messages", []))
        else:
            thread_msgs = [msg]
        if _bot_replied(thread_msgs, bot_id, parent_ts=ts):
            _handle_thread_replies(client, channel_id, ts, bot_id, user_id, state)
        else:
            _handle_top_level(client, channel_id, msg, state, bot_id)


def _post_channel_alert(text: str) -> bool:
    """Post ``text`` to the polled channel; return True on success.

    Never raises: monitoring alerts must not mask the underlying
    health-check failure, so all Slack errors are logged and swallowed.
    """
    try:
        client = _get_client()
        channel_id = _find_channel_id(client, CHANNEL_NAME)
        client.chat_postMessage(channel=channel_id, text=text)
        return True
    except Exception:
        logging.exception("Failed to post health alert to channel %s", CHANNEL_NAME)
        return False


def _load_health_alert() -> dict[str, Any]:
    """Read the persisted health-alert marker or return an empty dict."""
    if HEALTH_ALERT_FILE.exists():
        try:
            data = json.loads(HEALTH_ALERT_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            logging.exception("Failed to read health alert marker")
    return {}


def _startup_health_check() -> None:
    """Verify at least one LLM model is available before polling starts.

    Guards against shell/env misconfiguration (e.g. cron leaving
    ``$SHELL`` unset so the wrong RC file is sourced and no API keys
    are imported), which previously made every Sorcar task abort with
    "No model available" and silently retry forever.

    On failure: logs an error, posts an alert message to the polled
    Slack channel (rate-limited to one alert per
    ``HEALTH_ALERT_COOLDOWN`` seconds so per-minute cron ticks do not
    spam the channel), and exits with status 1 so the failure is also
    visible in ``cron.log``.

    On success after a previous failure: posts a recovery notice and
    clears the alert marker.
    """
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    alert = _load_health_alert()

    if get_available_models():
        if alert:
            # Remove the marker first: concurrent invocations (my test
            # run vs the real cron tick) race on the recovery path, and
            # unlink-before-post makes at most one of them announce.
            try:
                HEALTH_ALERT_FILE.unlink(missing_ok=True)
            except OSError:
                logging.exception("Failed to remove health alert marker")
                return
            logging.info("Health check recovered: LLM models available again")
            _post_channel_alert(
                ":white_check_mark: *kiss-slack-sorcar-poller recovered* — "
                "LLM models are available again; message processing has resumed."
            )
        return

    shell = os.environ.get("SHELL", "")
    logging.error(
        "Health check failed: no LLM model available after sourcing shell env "
        "(SHELL=%s); tasks would fail with empty results — exiting",
        shell,
    )
    now = time.time()
    last_alert = float(alert.get("last_alert_ts", 0.0))
    if now - last_alert >= HEALTH_ALERT_COOLDOWN:
        posted = _post_channel_alert(
            ":rotating_light: *kiss-slack-sorcar-poller health check failed* — "
            "no LLM model is available (no API key found in the environment), "
            "so incoming messages cannot be processed.\n"
            f"- Environment: `SHELL={shell or '(unset)'}`\n"
            f"- Log: `{LOG_FILE}`\n"
            "- Recovery: ensure the API keys (e.g. `ANTHROPIC_API_KEY`) are "
            "exported in the login-shell RC file sourced by the poller, then "
            "wait for the next cron tick."
        )
        if posted:
            tmp = HEALTH_ALERT_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps({"last_alert_ts": now}), encoding="utf-8")
            tmp.replace(HEALTH_ALERT_FILE)
    sys.exit(1)


_LOCK_FP: Any = None


def main() -> None:
    """Entry point: lock, poll for ``RUN_DURATION`` seconds, exit."""
    global _LOCK_FP
    _setup_logging()
    _LOCK_FP = _acquire_lock()
    # Under cron ``$SHELL`` is unset, which makes ``source_shell_env``
    # fall back to sourcing ``~/.bashrc`` instead of the user's real
    # login-shell RC (e.g. ``~/.zshrc``), silently importing zero API
    # keys — every Sorcar task then aborts with "No model available"
    # and the reply is empty.  Recover the login shell from the user
    # database before sourcing.
    if not os.environ.get("SHELL"):
        os.environ["SHELL"] = pwd.getpwuid(os.getuid()).pw_shell or "/bin/zsh"
    source_shell_env()
    _startup_health_check()

    try:
        client = _get_client()
        bot_id = str(client.auth_test().get("user_id", ""))
        user_id = _find_user_id(client, USER_NAME)
        channel_id = _find_channel_id(client, CHANNEL_NAME)
        logging.info(
            "Polling channel=%s (%s) bot=%s user=%s (%s)",
            CHANNEL_NAME,
            channel_id,
            bot_id,
            user_id,
            USER_NAME,
        )
    except Exception:
        logging.exception("Startup failed")
        raise

    # On first run, set a watermark so we only process messages from now on.
    state = _load_state()
    if "min_ts" not in state:
        state["min_ts"] = str(time.time())
        _save_state(state)
        logging.info("Initialized min_ts=%s (only new messages will be processed)", state["min_ts"])

    deadline = time.time() + RUN_DURATION
    while time.time() < deadline:
        state = _load_state()
        try:
            _poll_once(client, channel_id, bot_id, user_id, state)
        except Exception:
            logging.exception("Poll iteration failed")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

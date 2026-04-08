"""Channel poller — polls a ChannelBackend and runs agents.

Pseudocode
----------

run_once():  (one-shot poll mode)
    Connect to backend.
    Join the named channel.
    Poll recent messages.
    Skip bot messages and non-allowed users.
    Skip messages the bot has already replied to.
    Process each pending message with a StatefulSorcarAgent.
    Return the number of messages processed.

On each inbound message:
    Create/resume a StatefulSorcarAgent and run it,
    passing a reply() tool the agent can call to post back to the channel.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from kiss.agents.sorcar.stateful_sorcar_agent import StatefulSorcarAgent
from kiss.channels import ChannelBackend
from kiss.core.base import Base

logger = logging.getLogger(__name__)


@dataclass
class _SenderState:
    """Mutable per-sender state."""

    chat_id: str = ""


class ChannelPoller:
    """One-shot channel poller that checks for pending messages and runs agents.

    Connects to a ChannelBackend, retrieves recent messages, filters to
    allowed users, skips messages the bot has already replied to, and
    runs a StatefulSorcarAgent for each pending message.
    """

    def __init__(
        self,
        backend: ChannelBackend,
        channel_name: str,
        agent_name: str,
        extra_tools: list | None = None,
        model_name: str = "",
        max_budget: float = 5.0,
        work_dir: str = "",
        allow_users: list[str] | None = None,
    ) -> None:
        self._backend = backend
        self._channel_name = channel_name
        self._agent_name = agent_name
        self._extra_tools = extra_tools or []
        self._model_name = model_name
        self._max_budget = max_budget
        self._work_dir = work_dir or str(Path.home() / ".kiss" / "channel_work")
        self._allow_users = set(allow_users) if allow_users else None
        self._poll_thread_fn = getattr(backend, "poll_thread_messages", None)

    def run_once(self) -> int:
        """Check for pending messages, process them, and exit.

        Connects to the backend, joins the configured channel, retrieves
        recent messages, filters to allowed users, skips messages the bot
        has already replied to, and runs a StatefulSorcarAgent for each
        pending message.  Each message is processed synchronously.

        Returns:
            Number of messages processed.

        Raises:
            RuntimeError: If connection or channel lookup fails.
        """
        Base.reset_global_budget()
        if not self._backend.connect():
            raise RuntimeError(f"Failed to connect: {self._backend.connection_info}")
        logger.info("Connected: %s", self._backend.connection_info)
        try:
            channel_id = ""
            if self._channel_name:
                channel_id = self._backend.find_channel(self._channel_name) or ""
                if not channel_id:
                    raise RuntimeError(f"Channel not found: {self._channel_name!r}")
                self._backend.join_channel(channel_id)
                logger.info("Joined channel: %s (%s)", self._channel_name, channel_id)

            messages, _ = self._backend.poll_messages(channel_id, "0", limit=50)

            processed = 0
            for msg in messages:
                if self._backend.is_from_bot(msg):
                    continue
                user_id = msg.get("user", "")
                if self._allow_users and user_id not in self._allow_users:
                    continue
                if self._has_bot_reply(channel_id, msg):
                    continue
                state = _SenderState()
                self._handle_message(
                    f"{channel_id}:{msg.get('ts', '')}",
                    channel_id,
                    msg,
                    state,
                )
                processed += 1

            return processed
        finally:
            self._disconnect_backend()

    def _has_bot_reply(self, channel_id: str, msg: dict[str, Any]) -> bool:
        """Check if the bot has already replied to a message's thread.

        Uses ``poll_thread_messages`` if the backend supports it.
        Returns ``False`` when thread polling is unavailable or the
        message has no replies.

        Args:
            channel_id: Channel ID containing the message.
            msg: Message dict from poll_messages.

        Returns:
            True if the bot has already replied in the thread.
        """
        if self._poll_thread_fn is None:
            return False
        if msg.get("reply_count", 0) == 0:
            return False
        msg_ts = msg.get("ts", "")
        if not msg_ts:
            return False
        try:
            replies, _ = self._poll_thread_fn(channel_id, msg_ts, "0", limit=100)
            return any(self._backend.is_from_bot(r) for r in replies)
        except Exception:
            logger.debug("Error checking thread replies for %s", msg_ts, exc_info=True)
            return False

    def _handle_message(
        self,
        session_key: str,
        channel_id: str,
        msg: dict[str, Any],
        state: _SenderState,
    ) -> None:
        """Run one agent task for an inbound message."""
        text = self._backend.strip_bot_mention(msg.get("text", ""))
        thread_ts = msg.get("thread_ts", msg.get("ts", ""))
        agent = StatefulSorcarAgent(self._agent_name)
        if state.chat_id:  # pragma: no branch
            agent.resume_chat_by_id(state.chat_id)
        else:
            agent.new_chat()
            state.chat_id = agent.chat_id

        tools = list(self._extra_tools)
        replied = threading.Event()

        def reply(message: str) -> str:
            """Send a reply to the current conversation.

            Args:
                message: Text to send as the bot's reply.

            Returns:
                JSON string with ok status.
            """
            replied.set()
            try:
                self._backend.send_message(channel_id, message, thread_ts)
                return json.dumps({"ok": True})
            except Exception as e:
                return json.dumps({"ok": False, "error": str(e)})

        tools.append(reply)

        Path(self._work_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = agent.run(
                prompt_template=text,
                model_name=self._model_name,
                max_budget=self._max_budget,
                work_dir=self._work_dir,
                tools=tools,
                verbose=False,
            )
            state.chat_id = agent.chat_id
            if not replied.is_set():  # pragma: no branch
                result_yaml = yaml.safe_load(result)
                summary = (result_yaml.get("summary", "") if result_yaml else "") or result
                self._send_reply(channel_id, summary, thread_ts)
        except Exception as e:
            logger.error("Agent error for %s: %s", session_key, e, exc_info=True)
            self._send_reply(channel_id, f"Error processing your message: {e}", thread_ts)

    def _send_reply(self, channel_id: str, text: str, thread_ts: str) -> None:
        """Send a reply message, retrying once on transient failure.

        Args:
            channel_id: Channel to post to.
            text: Message text.
            thread_ts: Thread timestamp for threading.
        """
        for attempt in range(2):
            try:
                self._backend.send_message(channel_id, text, thread_ts)
                return
            except Exception:
                if attempt == 0:
                    logger.warning("Reply failed, retrying...", exc_info=True)
                    time.sleep(1)
                else:
                    logger.error("Reply failed after retry", exc_info=True)

    def _disconnect_backend(self) -> None:
        """Best-effort backend cleanup hook."""
        try:
            self._backend.disconnect()
        except Exception:
            logger.warning("Backend disconnect failed", exc_info=True)


# Backward-compatible alias for existing imports.
ChannelDaemon = ChannelPoller

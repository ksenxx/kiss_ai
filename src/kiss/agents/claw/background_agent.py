"""Background agent that listens for tasks on a messaging channel.

Polls a messaging channel for messages from a configured user, treats each
message as a task, completes it using SorcarAgent, and sends results back
to the channel.  Agent callbacks (cli_wait_for_user, cli_ask_user_question)
are routed through channel thread replies for interactive feedback.

The agent is channel-agnostic — it uses the ``ChannelBackend`` protocol
from ``kiss.channels`` and can work with Slack, Gmail, WhatsApp, or any
other backend that implements the protocol.

Usage::

    uv run python -m kiss.agents.claw.background_agent
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

import yaml
from filelock import FileLock, Timeout

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_history import _add_task, _set_latest_chat_events
from kiss.channels import ChannelBackend

logger = logging.getLogger(__name__)

_LOCK_FILE = Path.home() / ".kiss" / "claw_background_agent.lock"

_POLL_INTERVAL = 3.0  # seconds between polling for new messages
_CHANNEL_NAME = "sorcar"
_USERNAME = "ksen"


def _run_task(
    backend: ChannelBackend,
    channel_id: str,
    user_id: str,
    task_text: str,
    thread_ts: str,
    work_dir: str,
) -> None:
    """Run a single task using SorcarAgent and post results to the channel.

    Args:
        backend: Channel backend for sending messages and waiting for replies.
        channel_id: Channel ID to post results to.
        user_id: User ID of the task requester (for reply polling).
        task_text: The task description.
        thread_ts: Thread timestamp to reply in.
        work_dir: Working directory for the agent.
    """

    def channel_wait_for_user(instruction: str, url: str) -> None:
        """Send a browser action prompt and wait for user reply."""
        msg = f"🔔 *Browser action needed:*\n{instruction}"
        if url:
            msg += f"\n_Current URL:_ {url}"
        msg += "\n\n_Reply in this thread when done._"
        backend.send_message(channel_id, msg, thread_ts)
        backend.wait_for_reply(channel_id, thread_ts, user_id)

    def channel_ask_user_question(question: str) -> str:
        """Send a question and wait for user's reply."""
        msg = f"❓ *Agent asks:*\n{question}\n\n_Reply in this thread with your answer._"
        backend.send_message(channel_id, msg, thread_ts)
        return backend.wait_for_reply(channel_id, thread_ts, user_id)

    agent = SorcarAgent("Claw Background Agent")
    agent.web_use_tool = None  # No GUI/browser needed

    backend.send_message(
        channel_id, f"⚙️ Working on task:\n> {task_text[:500]}", thread_ts
    )

    _add_task(task_text)
    printer = BaseBrowserPrinter()
    printer.start_recording()

    old_cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        result = agent.run(
            prompt_template=task_text,
            work_dir=work_dir,
            verbose=True,
            printer=printer,
            wait_for_user_callback=channel_wait_for_user,
            ask_user_question_callback=channel_ask_user_question,
        )
    except Exception as e:
        logger.error("Agent error", exc_info=True)
        result = yaml.dump({"success": False, "summary": f"Agent error: {e}"})
    finally:
        os.chdir(old_cwd)

    result_data = yaml.safe_load(result)
    success = result_data.get("success", False)
    summary = result_data.get("summary", "No summary available.")

    chat_events = printer.stop_recording()
    chat_events.append({"type": "task_done" if success else "task_error"})
    _set_latest_chat_events(chat_events, task=task_text, result=summary)
    emoji = "✅" if success else "❌"
    msg = f"{emoji} *Task {'completed' if success else 'failed'}*\n\n{summary}"
    # Channel message limits vary; truncate at a safe length
    if len(msg) > 3900:
        msg = msg[:3900] + "\n... (truncated)"
    backend.send_message(channel_id, msg, thread_ts)

    cost = f"${agent.budget_used:.4f}" if hasattr(agent, "budget_used") else "unknown"
    tokens = agent.total_tokens_used if hasattr(agent, "total_tokens_used") else "unknown"
    backend.send_message(
        channel_id, f"📊 Cost: {cost} | Tokens: {tokens}", thread_ts
    )


def run_background_agent(work_dir: str | None = None) -> None:
    """Main loop: poll channel for tasks from user, run them, post results.

    Only one instance can run at a time. If another instance is already
    running, this function prints a message and returns immediately.

    Args:
        work_dir: Working directory for agent tasks. Defaults to a temp dir.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(_LOCK_FILE, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        print(
            "Another background agent instance is already running. "
            "Only one instance is allowed at a time."
        )
        return

    try:
        # Import here to allow swapping backends in the future
        from kiss.channels.slack_agent import SlackChannelBackend

        backend: ChannelBackend = SlackChannelBackend()
        _run_background_agent_locked(backend, work_dir)
    finally:
        lock.release()


def _run_background_agent_locked(
    backend: ChannelBackend, work_dir: str | None = None
) -> None:
    """Core polling loop — requires the single-instance lock to be held.

    Args:
        backend: Channel backend to use for communication.
        work_dir: Working directory for agent tasks.
    """
    if not backend.connect():
        print(backend.connection_info)
        return

    print(backend.connection_info)

    channel_id = backend.find_channel(_CHANNEL_NAME)
    if not channel_id:
        print(f"Channel #{_CHANNEL_NAME} not found. Please create it first.")
        return
    print(f"Monitoring #{_CHANNEL_NAME} (ID: {channel_id})")

    backend.join_channel(channel_id)

    user_id = backend.find_user(_USERNAME)
    if user_id:
        print(f"Watching for messages from {_USERNAME} (ID: {user_id})")
    else:
        print(f"User '{_USERNAME}' not found. Will accept messages from any user.")

    resolved_work_dir = work_dir or tempfile.mkdtemp(prefix="claw_")
    Path(resolved_work_dir).mkdir(parents=True, exist_ok=True)
    print(f"Work directory: {resolved_work_dir}")

    # Start polling from now — Slack requires ≤6 decimal places
    last_ts = f"{time.time():.6f}"
    processed_ts: set[str] = set()
    print(
        f"\n🤖 Background agent ready. Send a message in #{_CHANNEL_NAME} to start a task.\n"
    )

    backend.send_message(
        channel_id,
        "🤖 Claw background agent is now online and listening for tasks.",
    )

    while True:
        try:
            messages, last_ts = backend.poll_messages(channel_id, last_ts)

            for msg in messages:
                msg_user = msg.get("user", "")
                msg_text = msg.get("text", "").strip()
                msg_ts = msg.get("ts", "")

                if backend.is_from_bot(msg):
                    continue
                if user_id and msg_user != user_id:
                    continue
                if not msg_text:
                    continue
                if msg_ts in processed_ts:
                    continue
                processed_ts.add(msg_ts)

                msg_text = backend.strip_bot_mention(msg_text)

                print(
                    f"\n📩 New task from {msg_user}: {msg_text[:100]}...",
                    flush=True,
                )
                _run_task(
                    backend=backend,
                    channel_id=channel_id,
                    user_id=msg_user,
                    task_text=msg_text,
                    thread_ts=msg_ts,
                    work_dir=resolved_work_dir,
                )

            # Prevent unbounded growth — keep only recent entries
            if len(processed_ts) > 1000:
                processed_ts.clear()

        except KeyboardInterrupt:
            print("\n\nShutting down background agent...")
            try:
                backend.send_message(
                    channel_id,
                    "🔴 Claw background agent is shutting down.",
                )
            except Exception:
                pass
            break
        except Exception:
            logger.error("Error in polling loop", exc_info=True)
            time.sleep(10)

        time.sleep(_POLL_INTERVAL)


def main() -> None:
    """Entry point for the background agent CLI."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(description="Claw background agent")
    parser.add_argument(
        "--work-dir", type=str, default=".", help="Working directory for tasks"
    )
    args = parser.parse_args()
    run_background_agent(work_dir=args.work_dir)


if __name__ == "__main__":
    main()

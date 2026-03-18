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
import signal
import tempfile
import time
from pathlib import Path

import yaml
from filelock import FileLock, Timeout

from kiss.agents.sorcar.browser_ui import BaseBrowserPrinter
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.task_history import _add_task, _set_latest_chat_events
from kiss.channels import ChannelBackend
from kiss.core.print_to_console import ConsolePrinter
from kiss.core.printer import MultiPrinter

logger = logging.getLogger(__name__)

_LOCK_FILE = Path.home() / ".kiss" / "claw_background_agent.lock"
_PID_FILE = Path.home() / ".kiss" / "claw_background_agent.pid"

_POLL_INTERVAL = 3.0  # seconds between polling for new messages
_CHANNEL_NAME = "sorcar"
_USERNAME = "ksen"


_MAX_CHUNK = 3900  # stay under Slack's ~4000 char limit per message


def _send_chunked(
    backend: ChannelBackend,
    channel_id: str,
    thread_ts: str,
    text: str,
) -> None:
    """Send a long message in chunks, splitting at line boundaries.

    Args:
        backend: Channel backend for sending messages.
        channel_id: Channel ID to post to.
        thread_ts: Thread timestamp to reply in.
        text: Full message text (may exceed channel message limits).
    """
    while text:
        if len(text) <= _MAX_CHUNK:
            backend.send_message(channel_id, text, thread_ts)
            break
        # Find last newline within the chunk limit for a clean split
        split = text.rfind("\n", 0, _MAX_CHUNK)
        if split <= 0:
            split = _MAX_CHUNK
        backend.send_message(channel_id, text[:split], thread_ts)
        text = text[split:].lstrip("\n")


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
    recorder = BaseBrowserPrinter()
    recorder.start_recording()
    printer = MultiPrinter([recorder, ConsolePrinter()])

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

    chat_events = recorder.stop_recording()
    chat_events.append({"type": "task_done" if success else "task_error"})
    _set_latest_chat_events(chat_events, task=task_text, result=summary)
    emoji = "✅" if success else "❌"
    header = f"{emoji} *Task {'completed' if success else 'failed'}*\n\n"
    _send_chunked(backend, channel_id, thread_ts, header + summary)

    cost = f"${agent.budget_used:.4f}" if hasattr(agent, "budget_used") else "unknown"
    tokens = agent.total_tokens_used if hasattr(agent, "total_tokens_used") else "unknown"
    backend.send_message(
        channel_id, f"📊 Cost: {cost} | Tokens: {tokens}", thread_ts
    )


def _read_pid() -> int | None:
    """Read the PID stored in the PID file, or None if missing/invalid."""
    try:
        return int(_PID_FILE.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_pid_alive(pid: int) -> bool:
    """Check whether a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _clear_stale_lock() -> None:
    """Remove lock and PID files if the recorded process is no longer alive."""
    pid = _read_pid()
    if pid is not None and not _is_pid_alive(pid):
        _LOCK_FILE.unlink(missing_ok=True)
        _PID_FILE.unlink(missing_ok=True)


def stop_background_agent() -> bool:
    """Stop a running background agent instance.

    Reads the PID file, sends SIGTERM to the process, and cleans up
    lock/PID files.

    Returns:
        True if a running instance was stopped, False otherwise.
    """
    pid = _read_pid()
    if pid is None:
        print("No background agent PID file found.")
        return False
    if not _is_pid_alive(pid):
        print(f"Background agent (PID {pid}) is not running. Cleaning up stale files.")
        _LOCK_FILE.unlink(missing_ok=True)
        _PID_FILE.unlink(missing_ok=True)
        return False
    print(f"Stopping background agent (PID {pid})...")
    os.kill(pid, signal.SIGTERM)
    # Wait briefly for it to exit
    for _ in range(20):
        time.sleep(0.5)
        if not _is_pid_alive(pid):
            print("Background agent stopped.")
            _PID_FILE.unlink(missing_ok=True)
            return True
    print(f"Process {pid} did not exit after SIGTERM. Use 'kill -9 {pid}' to force.")
    return False


def run_background_agent(work_dir: str | None = None) -> None:
    """Main loop: poll channel for tasks from user, run them, post results.

    Only one instance can run at a time. If another instance is already
    running, this function prints a message and returns immediately.

    Args:
        work_dir: Working directory for agent tasks. Defaults to a temp dir.
    """
    _LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    _clear_stale_lock()
    lock = FileLock(_LOCK_FILE, timeout=0)
    try:
        lock.acquire()
    except Timeout:
        pid = _read_pid()
        if pid:
            print(
                f"Another background agent instance is already running (PID {pid}).\n"
                f"Use 'uv run python -m kiss.agents.claw.background_agent --stop' "
                f"or 'kill {pid}' to stop it."
            )
        else:
            print(
                "Another background agent instance is already running. "
                "Only one instance is allowed at a time."
            )
        return

    _PID_FILE.write_text(str(os.getpid()))
    try:
        # Import here to allow swapping backends in the future
        from kiss.channels.slack_agent import SlackChannelBackend

        backend: ChannelBackend = SlackChannelBackend()
        _run_background_agent_locked(backend, work_dir)
    finally:
        lock.release()
        _PID_FILE.unlink(missing_ok=True)


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
        except OSError:
            logger.warning("Network error in polling loop", exc_info=True)
            time.sleep(10)
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
    parser.add_argument(
        "--stop", action="store_true", help="Stop a running background agent"
    )
    parser.add_argument(
        "--status", action="store_true", help="Check if background agent is running"
    )
    args = parser.parse_args()

    if args.stop:
        stop_background_agent()
    elif args.status:
        pid = _read_pid()
        if pid and _is_pid_alive(pid):
            print(f"Background agent is running (PID {pid}).")
        elif pid:
            print(f"Background agent (PID {pid}) is not running (stale PID file).")
        else:
            print("Background agent is not running.")
    else:
        run_background_agent(work_dir=args.work_dir)


if __name__ == "__main__":
    main()

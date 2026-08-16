# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Shared end-to-end harness for the parallel sorcar-agent tests.

Everything here is real: a real temp git repository, a real temp
SQLite history database under an isolated ``KISS_HOME``, a real
:class:`~kiss.server.json_printer.JsonPrinter`, real threads, and a
real local HTTP server speaking the OpenAI chat-completions wire
format so agents make genuine model calls without costing money.  No
mocking library, patch, fake or test double is used.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import kiss.agents.sorcar.persistence as persistence
from kiss.core import vscode_config
from kiss.server.json_printer import JsonPrinter

# The stand-in server speaks the OpenAI chat-completions wire format,
# so any OpenAI-family model name routes to it once ``model_config``
# supplies ``base_url``.  No request ever leaves the machine.
STANDIN_MODEL = "gpt-4o-mini"


def run_git(cwd: str | Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``git *args`` in *cwd* and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def init_repo(repo: str | Path) -> None:
    """Initialise a git repo at *repo* with one seed commit."""
    Path(repo).mkdir(parents=True, exist_ok=True)
    run_git(repo, "init", "-q")
    run_git(repo, "config", "user.email", "kiss-test@example.com")
    run_git(repo, "config", "user.name", "Kiss Test")
    run_git(repo, "config", "commit.gpgsign", "false")
    Path(repo, "seed.txt").write_text("seed\n", encoding="utf-8")
    run_git(repo, "add", "seed.txt")
    run_git(repo, "commit", "-q", "-m", "seed")


def tool_call_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Build an OpenAI chat-completions body holding one tool call."""
    return {
        "id": "chatcmpl-kiss-test",
        "object": "chat.completion",
        "created": 0,
        "model": STANDIN_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{name}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def finish_response(summary: str = "done") -> dict[str, Any]:
    """Build a response whose single tool call is ``finish``."""
    return tool_call_response(
        "finish", {"success": "true", "summary_in_html": summary}
    )


def _sse_chunks(body: dict[str, Any]) -> list[bytes]:
    """Split a completion *body* into server-sent-event stream chunks.

    The production model client always requests
    ``stream=True, stream_options={"include_usage": True}``, so the
    stand-in has to answer in the streaming wire format: one chunk
    announcing the tool call, one carrying its JSON arguments, one
    finish-reason chunk, a usage-only chunk, then ``[DONE]``.
    """
    choice = body["choices"][0]
    message = choice["message"]
    call = message["tool_calls"][0]
    base = {
        "id": body["id"],
        "object": "chat.completion.chunk",
        "created": body["created"],
        "model": body["model"],
    }

    def frame(payload: dict[str, Any]) -> bytes:
        return b"data: " + json.dumps({**base, **payload}).encode() + b"\n\n"

    return [
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": message.get("content") or "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": call["function"]["name"],
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            }
        ),
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": call["function"][
                                            "arguments"
                                        ],
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            }
        ),
        frame(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": choice["finish_reason"],
                    }
                ],
                "usage": None,
            }
        ),
        frame({"choices": [], "usage": body["usage"]}),
        b"data: [DONE]\n\n",
    ]


class StandInModelServer:
    """A local OpenAI-compatible endpoint driven by a Python callback.

    The callback receives the decoded request body of every chat
    completion call and returns the response body to send back, so a
    test can script exactly what each agent does (finish immediately,
    run a shell command, block until the test releases it, ...).
    """

    def __init__(
        self, responder: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        """Start the server on a free loopback port.

        Args:
            responder: Callable mapping a decoded request body to the
                chat-completions response body to return.
        """
        self.responder = responder
        server = ThreadingHTTPServer(("127.0.0.1", 0), _StandInHandler)
        server.responder = responder  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{server.server_port}/v1"

    @property
    def model_config(self) -> dict[str, Any]:
        """The ``model_config`` kwarg routing an agent at this server."""
        return {"base_url": self.url, "api_key": "kiss-test-key"}

    def stop(self) -> None:
        """Shut the server down and join its thread."""
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


class _StandInHandler(BaseHTTPRequestHandler):
    """Dispatch every POST to the owning server's responder callback."""

    def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler API
        """Answer a chat-completions request via the server's responder."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw.decode("utf-8"))
        except ValueError:  # pragma: no cover — defensive
            request = {}
        responder = self.server.responder  # type: ignore[attr-defined]
        completion = responder(request)
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in _sse_chunks(completion):
                self.wfile.write(chunk)
                self.wfile.flush()
            return
        body = json.dumps(completion).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Silence the default stderr access log."""


def request_text(request: dict[str, Any]) -> str:
    """Return all message text of a chat-completions *request*.

    Lets a responder route on a marker string the test put in the
    prompt, so each agent in a fan-out can be scripted separately.
    """
    parts: list[str] = []
    for message in request.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and isinstance(
                    block.get("text"), str
                ):
                    parts.append(block["text"])
    return "\n".join(parts)


class CapturePrinter(JsonPrinter):
    """Real :class:`JsonPrinter` that also records broadcast events."""

    def __init__(self) -> None:
        """Create a printer with an empty capture buffer."""
        super().__init__()
        self.captured: list[dict[str, Any]] = []
        self._capture_lock = threading.Lock()

    def broadcast(self, event: dict[str, Any]) -> None:
        """Record *event*, then run the real broadcast logic."""
        with self._capture_lock:
            self.captured.append(dict(event))
        super().broadcast(event)

    def events_of_type(self, event_type: str) -> list[dict[str, Any]]:
        """Return every captured event whose ``type`` is *event_type*."""
        with self._capture_lock:
            return [e for e in self.captured if e.get("type") == event_type]


class IsolatedKissHome:
    """An isolated ``KISS_HOME`` + history DB + scratch git repo.

    Guarantees the real ``~/.kiss`` and the developer's checkout are
    never touched: the history database, the ``config.json`` consulted
    by :mod:`kiss.core.vscode_config`, and the git repository the
    agents work in all live under one throwaway temp directory.
    """

    def __init__(self, prefix: str = "kiss-parallel-harness-") -> None:
        """Redirect persistence + ``KISS_HOME`` into a fresh temp dir."""
        self.tmpdir = Path(tempfile.mkdtemp(prefix=prefix))
        self.kiss_home = self.tmpdir / ".kiss"
        self.kiss_home.mkdir(parents=True, exist_ok=True)
        self.repo = self.tmpdir / "repo"
        init_repo(self.repo)

        self._saved_env = os.environ.get("KISS_HOME")
        os.environ["KISS_HOME"] = str(self.kiss_home)
        # vscode_config prefers its module-global override over
        # $KISS_HOME, and other test modules set it, so pin both.
        module_globals = vars(vscode_config)
        self._saved_cfg: tuple[Any, Any] = (
            module_globals.get("CONFIG_DIR"),
            module_globals.get("CONFIG_PATH"),
        )
        vscode_config.CONFIG_DIR = self.kiss_home  # type: ignore[attr-defined]
        vscode_config.CONFIG_PATH = self.kiss_home / "config.json"  # type: ignore[attr-defined]
        self._saved_db = (
            persistence._DB_PATH,
            persistence._db_conn,
            persistence._KISS_DIR,
        )
        persistence._KISS_DIR = self.kiss_home
        persistence._DB_PATH = self.kiss_home / "sorcar.db"
        persistence._db_conn = None

    def write_config(self, **values: Any) -> None:
        """Write *values* into the isolated ``config.json``."""
        path = self.kiss_home / "config.json"
        existing: dict[str, Any] = {}
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
        existing.update(values)
        path.write_text(json.dumps(existing), encoding="utf-8")

    def cleanup(self) -> None:
        """Restore global state and delete the temp directory."""
        from kiss.server import agent_state

        with agent_state.STATE_LOCK:
            agent_state.agent_states.clear()
        if persistence._db_conn is not None:
            try:
                persistence._db_conn.close()
            except Exception:  # pragma: no cover — cleanup best-effort
                pass
        (
            persistence._DB_PATH,
            persistence._db_conn,
            persistence._KISS_DIR,
        ) = self._saved_db
        if self._saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = self._saved_env
        for name, value in zip(
            ("CONFIG_DIR", "CONFIG_PATH"), self._saved_cfg, strict=True,
        ):
            if value is None:
                if hasattr(vscode_config, name):
                    delattr(vscode_config, name)
            else:
                setattr(vscode_config, name, value)
        shutil.rmtree(self.tmpdir, ignore_errors=True)


_PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
)

# Enough for git (and every other POSIX tool the agent shells out to)
# while excluding the per-user and Homebrew directories the vendor
# CLIs are installed into.
_MINIMAL_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"


class OfflineFastModel:
    """Make the auto-commit message generator run without any provider.

    Auto-commit asks
    :func:`~kiss.core.models.model_info.get_fast_model` for a cheap
    model and runs one non-agentic LLM call to write the commit
    subject.  That call is the one place in the worktree cleanup path
    that would otherwise leave the machine (and cost money) even
    though the agents themselves are pointed at a local stand-in
    server.

    Entering this context empties the provider credentials the
    resolver reads and hides the ``claude`` / ``codex`` CLIs it falls
    back to, so ``get_fast_model()`` answers ``"No model"`` and
    ``auto_commit_changes`` takes its documented fallback message
    path.  The real commit still happens; only the message wording
    changes.
    """

    def __init__(self) -> None:
        """Capture nothing yet; state is saved on ``__enter__``."""
        self._saved_keys: dict[str, str] = {}
        self._saved_env: dict[str, str | None] = {}
        self._saved_path: str | None = None

    def __enter__(self) -> OfflineFastModel:
        """Empty the provider credentials and trim ``PATH``."""
        from kiss.core import config as config_module

        keys = config_module.DEFAULT_CONFIG
        for name in _PROVIDER_KEYS:
            self._saved_keys[name] = getattr(keys, name)
            setattr(keys, name, "")
            self._saved_env[name] = os.environ.pop(name, None)
        self._saved_path = os.environ.get("PATH")
        os.environ["PATH"] = _MINIMAL_PATH
        return self

    def __exit__(self, *_exc: object) -> None:
        """Restore the credentials and ``PATH``."""
        from kiss.core import config as config_module

        keys = config_module.DEFAULT_CONFIG
        for name, value in self._saved_keys.items():
            setattr(keys, name, value)
        for env_name, env_value in self._saved_env.items():
            if env_value is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = env_value
        if self._saved_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = self._saved_path


def history_rows() -> list[dict[str, Any]]:
    """Return every ``task_history`` row of the isolated DB as dicts."""
    conn = persistence._get_db()
    cursor = conn.execute(
        "SELECT id, chat_id, parent_task_id, task, is_worktree, is_parallel "
        "FROM task_history ORDER BY timestamp"
    )
    return [
        {
            "id": row[0],
            "chat_id": row[1],
            "parent_task_id": row[2],
            "task": row[3],
            "is_worktree": bool(row[4]),
            "is_parallel": bool(row[5]),
        }
        for row in cursor.fetchall()
    ]


def wait_for(predicate: Callable[[], bool], timeout: float = 30.0) -> bool:
    """Poll *predicate* until it is true or *timeout* seconds elapse."""
    deadline = threading.Event()
    step = 0.02
    waited = 0.0
    while waited < timeout:
        if predicate():
            return True
        deadline.wait(step)
        waited += step
    return predicate()

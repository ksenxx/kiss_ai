# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for ``scripts/run_sea.py``.

The script runs a SEA in its own process, without the kiss-web daemon or
any UI.  These tests run the real script as a subprocess, resolving real
SEA files (by path and, through a ``SEAS.md`` in a temporary
``KISS_HOME``, by name) and letting the real :class:`SorcarAgent` talk to
an OpenAI-compatible HTTP server started in the test process.  The
server plays a fixed two-step model: first it calls the run's ``Bash``
tool with ``pwd``, then it calls the structured ``finish`` with a summary
that quotes the task and the ``pwd`` output, so each run is deterministic
and the script's exit code, printed YAML and effective working directory
can be asserted exactly.  Markers in the task text steer the second step:
:data:`FAIL_MARKER` makes ``finish`` report ``success: false``,
:data:`BAD_FINISH_MARKER` makes the model call ``finish`` with a wrong
argument, which ends the run with the tool's error text instead of a YAML
mapping (the script's ``except`` path).

Nothing is patched: the model is the real OpenAI-compatible adapter, pointed
at the test server through the SEA's ``model_config`` setting, which is
what the script forwards to :meth:`SorcarAgent.run`.  The server reports
no token usage, so the unknown ``fake-model`` costs nothing and needs no
catalog entry.
"""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.tests.conftest import posix_only

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "run_sea.py"
FAKE_URL_ENV = "RUN_SEA_TEST_FAKE_URL"
FAIL_MARKER = "please fail"
BAD_FINISH_MARKER = "finish badly"
RUN_TIMEOUT = 300.0

pytestmark = posix_only("runs the script under a pty for the tty branch")


# A SEA that routes the model to the test server; its prompt tags the task
# so the server sees that the SEA's ``prompt`` was applied.  ``MODEL`` and
# ``WORK_DIR`` are formatted per test.
SEA_SOURCE = """\
import os
from typing import Any

from kiss.agents.seas.base.base_sea import WorkerSea


class TestSea(WorkerSea):
    \"\"\"Fixture SEA for scripts/run_sea.py.\"\"\"

    def settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        \"\"\"Point the model at the test server; pin MODEL / WORK_DIR when given.\"\"\"
        pinned = {{"model": {model!r}, "work_dir": {work_dir!r}}}
        return settings | {{
            "tool_profile": "bash",
            "model_config": {{"base_url": os.environ["{fake_url_env}"], "api_key": "test"}},
        }} | {{k: v for k, v in pinned.items() if v}}

    def prompt(self, task: str) -> str:
        \"\"\"Tag the task so the server can tell this SEA ran.\"\"\"
        return "TASK::" + task
"""


def _tool_call_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "", "tool_calls": [call]},
                "finish_reason": "tool_calls",
            }
        ],
    }


def _stream_chunks(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    delta_call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "model": "fake-model"}
    return [
        base
        | {
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "tool_calls": [delta_call]},
                    "finish_reason": None,
                }
            ]
        },
        base | {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    ]


def _task_in(messages: list[dict[str, Any]]) -> str:
    """The task text (after the SEA's ``TASK::`` tag) in the user messages."""
    text = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
    return text.split("TASK::", 1)[1].split("\n")[0]


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Play the two-step model: ``Bash pwd`` first, then ``finish``.

    The second step reads the task text: with :data:`BAD_FINISH_MARKER`
    it calls ``finish(result=...)`` (not a parameter of the structured
    ``finish``), otherwise ``finish(success=<not FAIL_MARKER>, ...)``
    with a summary quoting the task and the ``pwd`` output.
    """

    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        """Keep the test output clean."""

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        type(self).requests.append(body)
        messages = body.get("messages", [])
        tool_names = {t.get("function", {}).get("name") for t in body.get("tools", [])}
        assert {"Bash", "finish"} <= tool_names, tool_names
        tool_outputs = [str(m.get("content", "")) for m in messages if m.get("role") == "tool"]
        task = _task_in(messages)
        pwd = tool_outputs[-1] if tool_outputs else ""
        name = "finish"
        arguments: dict[str, Any] = {
            "success": FAIL_MARKER not in task,
            "is_continue": False,
            "summary_in_html": f"<p>echo: {task}</p><p>pwd: {pwd}</p>",
        }
        if not tool_outputs:
            name, arguments = "Bash", {"command": "pwd", "description": "print the work dir"}
        elif BAD_FINISH_MARKER in task:
            arguments = {"result": "done"}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _stream_chunks(name, arguments):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = json.dumps(_tool_call_response(name, arguments)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(scope="module")
def fake_openai() -> Any:
    """Serve the fake OpenAI API on a loopback port; yield its ``/v1`` URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def kiss_home(tmp_path: Path) -> Path:
    """A KISS home of this test's own, so no real SEAS.md or api_keys.env is read."""
    home = tmp_path / "home"
    home.mkdir()
    return home


def _write_sea(folder: Path, name: str, *, model: str = "", work_dir: str = "") -> Path:
    """Write ``<folder>/<name>/<name>_sea.py`` from :data:`SEA_SOURCE` and return it."""
    sea_dir = folder / name
    sea_dir.mkdir(parents=True)
    path = sea_dir / f"{name}_sea.py"
    path.write_text(
        SEA_SOURCE.format(model=model, work_dir=work_dir, fake_url_env=FAKE_URL_ENV),
        encoding="utf-8",
    )
    return path


def _env(kiss_home: Path, fake_url: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KISS_HOME"] = str(kiss_home)
    env[FAKE_URL_ENV] = fake_url
    env.pop("KISS_SORCAR_LOCAL", None)
    return env


def _run(args: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the script with stdout a pipe (the non-tty branch)."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT,
        check=False,
    )


def _run_in_pty(args: list[str], env: dict[str, str], cwd: Path) -> tuple[int, str]:
    """Run the script with stdout a pty (the tty branch); return (exit code, output).

    Reads the pty until the child closes it or :data:`RUN_TIMEOUT` passes;
    the child is killed and the pty closed whatever happens.
    """
    import pty  # POSIX only; imported here so collection succeeds where the module is skipped

    master, slave = pty.openpty()
    proc = subprocess.Popen(
        [sys.executable, str(SCRIPT), *args],
        env=env,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=slave,
        stderr=slave,
    )
    os.close(slave)
    chunks: list[bytes] = []
    deadline = time.monotonic() + RUN_TIMEOUT
    try:
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, b"".join(chunks).decode(errors="replace")
            if not select.select([master], [], [], remaining)[0]:
                continue
            try:
                data = os.read(master, 65536)
            except OSError:  # EIO: the child closed its end
                break
            if not data:
                break
            chunks.append(data)
        return proc.wait(timeout=RUN_TIMEOUT), b"".join(chunks).decode(errors="replace")
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)


def _result_yaml(stdout: str) -> dict[str, Any]:
    loaded = yaml.safe_load(stdout)
    assert isinstance(loaded, dict), stdout
    return loaded


def _assert_run_seen(task: str, model: str, work_dir: Path) -> None:
    """The last two requests were this run's ``pwd`` step and ``finish`` step."""
    bash_step, finish_step = FakeOpenAIHandler.requests[-2:]
    assert _task_in(bash_step["messages"]) == task
    assert _task_in(finish_step["messages"]) == task
    assert bash_step["model"] == finish_step["model"] == model
    tool_outputs = [m for m in finish_step["messages"] if m.get("role") == "tool"]
    assert len(tool_outputs) == 1, finish_step["messages"]
    assert str(work_dir.resolve()) in str(tool_outputs[0]["content"]), tool_outputs


def test_sea_by_path_runs_in_cwd_exit_0(fake_openai: str, kiss_home: Path, tmp_path: Path) -> None:
    """A SEA given by path runs in the cwd; the YAML result is printed; exit 0 on success."""
    sea = _write_sea(tmp_path / "seas", "probe")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    proc = _run([str(sea), "say hello", "-m", "fake-model"], _env(kiss_home, fake_openai), cwd)
    assert proc.returncode == 0, proc.stderr
    result = _result_yaml(proc.stdout)
    assert result["success"] is True
    assert "echo: say hello" in result["summary"]
    assert str(cwd.resolve()) in result["summary"]
    _assert_run_seen("say hello", "fake-model", cwd)


def test_failed_finish_in_work_dir_flag_exit_1(
    fake_openai: str, kiss_home: Path, tmp_path: Path
) -> None:
    """``success: false`` gives exit 1 (result still printed); ``--work-dir`` is the work dir."""
    sea = _write_sea(tmp_path / "seas", "probe")
    flagged = tmp_path / "flagged"
    flagged.mkdir()
    proc = _run(
        [str(sea), f"{FAIL_MARKER} now", "-m", "fake-model", "--work-dir", str(flagged)],
        _env(kiss_home, fake_openai),
        tmp_path,
    )
    assert proc.returncode == 1, proc.stderr
    assert _result_yaml(proc.stdout)["success"] is False
    _assert_run_seen(f"{FAIL_MARKER} now", "fake-model", flagged)


def test_sea_settings_pin_model_and_work_dir(
    fake_openai: str, kiss_home: Path, tmp_path: Path
) -> None:
    """The SEA's ``model`` / ``work_dir`` settings win over ``-m`` / ``--work-dir``."""
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    flagged = tmp_path / "flagged"
    flagged.mkdir()
    sea = _write_sea(tmp_path / "seas", "pinned", model="sea-model", work_dir=str(pinned))
    proc = _run(
        [str(sea), "pinned run", "-m", "flag-model", "--work-dir", str(flagged), "-b", "0.5"],
        _env(kiss_home, fake_openai),
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    assert _result_yaml(proc.stdout)["success"] is True
    _assert_run_seen("pinned run", "sea-model", pinned)


def test_finish_with_wrong_arguments_exit_1(
    fake_openai: str, kiss_home: Path, tmp_path: Path
) -> None:
    """A ``finish`` call with a wrong argument ends the run with its error text: exit 1."""
    sea = _write_sea(tmp_path / "seas", "probe")
    proc = _run(
        [str(sea), f"{BAD_FINISH_MARKER} please", "-m", "fake-model"],
        _env(kiss_home, fake_openai),
        tmp_path,
    )
    assert proc.returncode == 1, proc.stderr
    assert "Failed to call finish" in proc.stdout, proc.stdout
    assert "Traceback" not in proc.stderr, proc.stderr
    _assert_run_seen(f"{BAD_FINISH_MARKER} please", "fake-model", tmp_path)


def test_sea_by_name_from_seas_md(fake_openai: str, kiss_home: Path, tmp_path: Path) -> None:
    """A name is resolved through ``$KISS_HOME/SEAS.md`` like the daemon's ``/name``."""
    folder = tmp_path / "my-seas"
    _write_sea(folder, "named")
    (kiss_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    proc = _run(["named", "by name", "-m", "fake-model"], _env(kiss_home, fake_openai), tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert _result_yaml(proc.stdout)["success"] is True
    _assert_run_seen("by name", "fake-model", tmp_path)


def test_unknown_name_lists_registered_seas(
    fake_openai: str, kiss_home: Path, tmp_path: Path
) -> None:
    """An unknown name exits with a message naming every registered SEA; no model call."""
    folder = tmp_path / "my-seas"
    _write_sea(folder, "named")
    (kiss_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    before = len(FakeOpenAIHandler.requests)
    proc = _run(["no_such_sea", "task"], _env(kiss_home, fake_openai), tmp_path)
    assert proc.returncode == 1
    assert "unknown SEA 'no_such_sea'; known names: " in proc.stderr
    names = proc.stderr.strip().split("known names: ", 1)[1].split(", ")
    assert "named" in names and "sh" in names, names
    assert len(FakeOpenAIHandler.requests) == before


def test_tty_stdout_is_verbose_and_does_not_reprint_result(
    fake_openai: str, kiss_home: Path, tmp_path: Path
) -> None:
    """On a terminal the run is verbose and the result is not printed again at the end."""
    sea = _write_sea(tmp_path / "seas", "probe")
    code, output = _run_in_pty(
        [str(sea), "on a tty", "-m", "fake-model"], _env(kiss_home, fake_openai), tmp_path
    )
    assert code == 0, output
    _assert_run_seen("on a tty", "fake-model", tmp_path)
    assert "echo: on a tty" in output, output
    assert "summary: <p>echo: on a tty</p>" not in output, output

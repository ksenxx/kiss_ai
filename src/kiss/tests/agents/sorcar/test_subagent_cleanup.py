# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A sub-agent leaves nothing behind: no process, no temp file, nothing in the work tree.

A sub-agent (``_subagent_info`` set, as the daemon does for every
``run_agent`` / ``run_parallel`` child) gets a private folder in the
system temp directory.  The folder is its scratch dir, holds its job
logs and spilled tool output, and is the temp directory of every command
its tools start; when the run ends, however it ends, its background jobs
and every process still carrying the folder in its environment are
killed and the folder is deleted.  A top-level task keeps ``./tmp/`` and
its jobs.

Real agents against the scripted local model server, real processes,
real containers.

Not reached here, because no test double stands in for them: the sweep
giving up at its deadline (it needs a process ``SIGKILL`` cannot end);
its use of the ``ps`` listing where there is no ``/proc`` (macOS, BSD;
the listing itself, of the table and of one process, is read by the
sweep test, with the ``ps`` of this system: FreeBSD's, which prints the
environment before the command, and a line of ``ps`` that does not
begin with a pid are not); a process that ends between
the moment the sweep opens it and the moment it reads or signals it; a
Stop that lands in the two statements that note an earlier Stop of the
clean-up.  The sweep without a pidfd is reached, in a child process (a
seccomp filter cannot be taken off the test process again), so a
coverage run of the tests does not count its lines.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import docker
import pytest
import yaml

from kiss.agents.sorcar import coding_gate as cg
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.chrome_bridge import CONFIG_FILE_NAME, ChromeBridge, extension_dir
from kiss.agents.sorcar.container_jobs import GONE_STATUS, JOBS_DIR, ContainerJobs
from kiss.agents.sorcar.docker_manager import (
    _KILL_WAIT_S,
    _SWEEP_TIMEOUT_S,
    OWNER_LABEL,
    DockerManager,
    kill_tagged_tree_script,
    owner_label_value,
    remove_private_dir_script,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.core import user_config
from kiss.core.processes import _env_marked_pids_ps, kill_env_marked
from kiss.core.tool_interrupt import inject_async_exception
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available
from kiss.tests.agents.sorcar.local_model_server import (
    MODEL,
    ScriptedHandler,
    finish_body,
    serve,
    tool_call_body,
)
from kiss.tests.agents.sorcar.test_chrome_bridge import _TINY_PNG_B64, ExtensionPeer
from kiss.tests.conftest import posix_only

pytestmark = posix_only("process groups, bash and /proc or ps")

SUB_AGENT = {"parent_task_id": "parent-task", "parent_tab_id": "", "reviewer": False,
             "side_channel": False}
#: A command that runs until it is killed (the Bash tool refuses a long ``sleep``).
FOREVER = "tail -f /dev/null"
#: The same for host process-table tests.  Apple's platform binaries hide their
#: environment from ``ps eww``, so the macOS fallback can inspect this interpreter.
HOST_FOREVER = (
    f"{shlex.quote(sys.executable)} -c "
    f"{shlex.quote('import time; time.sleep(600)')}"
)
#: Fork a process into a session of its own, out of its command's process group.
#: ``setsid(1)`` is not installed on stock macOS; ``os.setsid`` is POSIX itself.
_ESCAPE_SCRIPT = """
import os
import sys

pid = os.fork()
if pid == 0:
    os.setsid()
    null = os.open(os.devnull, os.O_RDWR)
    for descriptor in (0, 1, 2):
        os.dup2(null, descriptor)
    os.execl(sys.executable, sys.executable, "-c", "import time; time.sleep(600)")
print(pid)
"""
ESCAPE = (
    f"{shlex.quote(sys.executable)} -c {shlex.quote(_ESCAPE_SCRIPT)} > escaped.pid"
)
#: The Linux test container has ``setsid(1)`` but not the host interpreter path.
CONTAINER_ESCAPE = f"setsid nohup {FOREVER} > /dev/null 2>&1 & echo $! > escaped.pid"
#: An environment of about 1 MB: the sweep takes about a millisecond to read a process
#: that has it, so 150 of them make one reading of the process table last.
PADDED = {f"PAD{i}": "x" * 120_000 for i in range(8)}
#: Run as pid 1 of a pid namespace of its own, where the next pid can be chosen
#: (``ns_last_pid``): a marked process ends and a stranger takes its pid while the sweep
#: is still reading the process table.  With ``no-pidfd`` as its argument a seccomp filter
#: first refuses ``pidfd_open`` (system call 434) with ``ENOSYS``, as a kernel older than
#: the call or a sandbox does.  Prints whether a pidfd could be opened, whether the
#: stranger got the pid, and the stranger's exit status (``None``: it runs).
PID_REUSE = """
import ctypes, json, os, struct, subprocess, sys, threading
from kiss.core.processes import kill_env_marked

if sys.argv[1] == "no-pidfd":
    code = b"".join(struct.pack("HBBI", *instruction) for instruction in (
        (0x20, 0, 0, 0), (0x15, 0, 1, 434), (0x06, 0, 0, 0x00050000 | 38),
        (0x06, 0, 0, 0x7FFF0000)))
    filter_ = ctypes.create_string_buffer(code)
    program = ctypes.create_string_buffer(struct.pack("HP", 4, ctypes.addressof(filter_)))
    libc = ctypes.CDLL(None, use_errno=True)
    assert libc.prctl(38, 1, 0, 0, 0) == 0 and libc.prctl(22, 2, program) == 0
try:
    os.close(os.pidfd_open(os.getpid()))
    pidfd = True
except OSError:
    pidfd = False
forever = ["tail", "-f", "/dev/null"]
marked = subprocess.Popen(forever, env={"KISS_SUBAGENT_DIR": "/kiss-sub-a"})
pad = {f"PAD{i}": "x" * 120_000 for i in range(8)}  # as PADDED: the reading lasts
slow = [subprocess.Popen(forever, env=pad) for _ in range(150)]
sweep = threading.Thread(target=kill_env_marked, args=("KISS_SUBAGENT_DIR=/kiss-sub-a",))
sweep.start()
try:
    marked.wait(timeout=0.03)  # the sweep kills it as soon as it has read it ...
except subprocess.TimeoutExpired:
    marked.kill()  # ... or it ends by itself while the sweep reads on
    marked.wait()
with open("/proc/sys/kernel/ns_last_pid", "w") as last:
    last.write(str(marked.pid - 1))
stranger = subprocess.Popen(forever, env={})
sweep.join()
print(json.dumps({"pidfd": pidfd, "reused": stranger.pid == marked.pid,
                  "stranger": stranger.poll()}))
for process in [*slow, stranger]:
    process.kill()
"""


def _scratch_dir(request: dict[str, Any]) -> str:
    """The ``Scratch dir`` a run's system prompt names (``""`` for a request without one)."""
    system = next((str(m["content"]) for m in request["messages"] if m["role"] == "system"), "")
    return system.partition("- Scratch dir: ")[2].partition("\n")[0]


class ScratchHandler(ScriptedHandler):
    """Scripted answers in which ``@SCRATCH@`` stands for the run's scratch dir.

    A model names the directory its system prompt gives it; the script
    cannot know a sub-agent's folder beforehand, so the server reads it
    from the request it answers.
    """

    def do_POST(self) -> None:  # noqa: N802
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        cls = type(self)
        with cls.lock:
            cls.requests.append(request)
            body = cls.script[min(len(cls.requests), len(cls.script)) - 1]
            body = body.replace(b'"call_1"', f'"call_{len(cls.requests)}"'.encode())
        body = body.replace(b"@SCRATCH@", _scratch_dir(request).encode())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _bash(command: str, **arguments: Any) -> bytes:
    """A scripted ``Bash`` call."""
    return tool_call_body("Bash", {"command": command, "description": "d", **arguments}, 1000)


def _run(agent: SorcarAgent, work_dir: Path, script: list[bytes], requests: list[dict],
         web_tools: bool = False, **run_kwargs: Any) -> dict[str, Any]:
    """Run *agent* in *work_dir* against *script*; *requests* receives the model's requests."""
    with serve(script, ScratchHandler) as (url, served):
        try:
            result = agent.run(
                prompt_template="Do the work.", model_name=MODEL, work_dir=str(work_dir),
                max_steps=12, max_budget=1.0, model_config={"base_url": url, "api_key": "local"},
                use_memory=False, verbose=False, web_tools=web_tools, **run_kwargs,
            )
        finally:
            requests.extend(served)
    loaded: dict[str, Any] = yaml.safe_load(result)
    return loaded


def _run_recording(agent: SorcarAgent, work_dir: Path, script: list[bytes],
                   raised: list[BaseException]) -> None:
    """Thread body: run *agent* against *script* and record in *raised* what the run raised."""
    try:
        _run(agent, work_dir, script, [])
    except BaseException as exc:  # noqa: BLE001 — the Stop, kept for the assertion
        raised.append(exc)


def _sub_agent(name: str, kind: type[SorcarAgent] = SorcarAgent) -> SorcarAgent:
    """An agent marked as the daemon marks a ``run_agent`` child."""
    agent = kind(name)
    agent._subagent_info = dict(SUB_AGENT)
    return agent


def _tool_results(request: dict[str, Any]) -> list[str]:
    """The tool results the model saw in *request*, in order."""
    return [str(m.get("content", "")) for m in request["messages"] if m.get("role") == "tool"]


def _pid_from(path: Path, within: float = 10.0) -> int:
    """The pid a command wrote into *path*, waiting up to *within* seconds for it."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        text = path.read_text() if path.exists() else ""
        if text.strip():
            return int(text)
        time.sleep(0.02)
    raise AssertionError(f"no pid written to {path}")


def _running(pid: int) -> bool:
    """Whether process *pid* still runs (a zombie waiting for its parent does not)."""
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                           text=True, check=False, timeout=10).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _gone(pid: int, within: float = 5.0) -> bool:
    """Whether process *pid* is gone within *within* seconds."""
    deadline = time.monotonic() + within
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return not _running(pid)


def _pin_reviewer_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the user config at a file that turns the finish gate's review stage off."""
    config = tmp_path / "config.json"
    config.write_text(json.dumps({cg.REVIEWER_CONFIG_KEY: "off"}))
    monkeypatch.setattr(user_config, "CONFIG_PATH", config)


def _start_container(**options: Any) -> Any:
    """Start a container of :data:`IMAGE`; skip without a daemon.

    Unless *options* say otherwise it sleeps under an init that reaps orphans.
    """
    if not is_docker_available():
        pytest.skip("needs a Docker daemon")
    return docker.from_env().containers.run(
        IMAGE, detach=True, working_dir="/w", labels={OWNER_LABEL: owner_label_value()},
        **{"command": ["sleep", "600"], "init": True, **options},
    )


@pytest.fixture
def container() -> Iterator[Any]:
    """A running container of the test's own."""
    made = _start_container()
    try:
        yield made
    finally:
        made.remove(force=True)


def _sh(container: Any, command: str) -> str:
    """The output of *command* run by ``sh`` in *container*, outside any sub-agent's name."""
    output: bytes = container.exec_run(["/bin/sh", "-c", command]).output
    return output.decode().strip()


def _alive_in(container: Any, pids: list[str]) -> list[str]:
    """Those of *pids* that are running processes in *container* (a zombie has no command line)."""
    return _sh(container, f"for p in {' '.join(pids)}; do tr -d '\\0' 2>/dev/null "
               "< /proc/$p/cmdline | grep -q . && echo $p; done; true").split()


def test_sub_agent_leaves_no_process_no_temp_file_and_nothing_in_the_work_tree(
    tmp_path: Path,
) -> None:
    """Every command runs in the private folder's name; the end takes all of it away.

    The folder holds a FIFO, a socket, a nested git repository and a
    read-only directory by then, a background job is running and one
    process has left its command's group.  The scratch rules hold for the
    folder: an existing file in it is overwritten with no ``Read`` first,
    by ``Write`` and by a here-document (made as a ``Write``); outside a
    scratch dir the first is an error and the second is refused.  The
    work dir is a git repository: a ``Write`` into the folder needs no
    worktree.
    """
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    for git in (["init", "-q"], ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                                 "--allow-empty", "-m", "init"]):
        subprocess.run(["git", *git], cwd=work_dir, check=True, timeout=60)
    clutter = (
        'mkfifo "$TMPDIR/fifo" && git init -q "$TMPDIR/repo" && mkdir -p "$TMPDIR/ro/d" '
        '&& touch "$TMPDIR/ro/d/f" && chmod -R a-w "$TMPDIR/ro" && '
        f"{shlex.quote(sys.executable)} -c \"import os, socket; "
        "socket.socket(socket.AF_UNIX).bind(os.environ['TMPDIR'] + '/sock')\""
    )
    script = [
        _bash(f'echo "$TMPDIR|$TMP|$TEMP|$KISS_SUBAGENT_DIR" > env.txt; {clutter}; {ESCAPE}'),
        _bash(f"echo $$ > job.pid; exec {FOREVER}", background=True),
        _bash("seq 1 60000", max_output_chars=2000),
        tool_call_body("run_commands_parallel", {
            "commands": json.dumps(['echo "$KISS_SUBAGENT_DIR" > parallel.txt; seq 1 60000']),
            "description": "d", "max_output_chars": 200000}, 1000),
        _bash("echo old > @SCRATCH@notes.md"),
        tool_call_body("Write", {"file_path": "@SCRATCH@notes.md", "content": "new\n"}, 1000),
        _bash("cat > @SCRATCH@notes.md <<'EOF'\nnewer\nEOF\ncp @SCRATCH@notes.md notes_copy.txt"),
        finish_body("<p>done</p>", prompt_tokens=1000),
    ]
    requests: list[dict] = []
    result = _run(_sub_agent("sub-cleanup"), work_dir, script, requests)
    assert result["success"] is True and len(requests) == len(script)

    scratch = _scratch_dir(requests[0])
    folder = Path(scratch)
    assert scratch.endswith("/") and folder.name.startswith("kiss-sub-")
    assert folder.parent.resolve() == Path(tempfile.gettempdir()).resolve()
    assert (work_dir / "env.txt").read_text().strip() == "|".join([str(folder)] * 4)
    assert (work_dir / "parallel.txt").read_text().strip() == str(folder)
    # the job's log and the spilled output were in the folder, as the model was told
    assert f"{folder}/bash_jobs/" in _tool_results(requests[2])[-1]
    assert f"{folder}/tool-out/" in _tool_results(requests[3])[-1]
    parallel = _tool_results(requests[4])[-1]  # the report was over the ceiling: a digest
    assert f"{folder}/tool-out/" in parallel and "-parallel.txt" in parallel
    assert _tool_results(requests[6])[-1].startswith("Successfully wrote")
    heredoc = _tool_results(requests[7])[-1]
    assert heredoc.startswith("Applied as Write(") and "Successfully wrote" in heredoc
    assert (work_dir / "notes_copy.txt").read_text() == "newer\n"

    assert _gone(_pid_from(work_dir / "escaped.pid")) and _gone(_pid_from(work_dir / "job.pid"))
    assert not folder.exists()
    assert not (work_dir / "tmp").exists()


def test_top_level_task_keeps_its_tmp_dir_and_its_jobs(tmp_path: Path) -> None:
    """A top-level task is as before: ``./tmp/`` scratch, the daemon's temp dir, jobs kept."""
    script = [
        _bash('echo "${KISS_SUBAGENT_DIR-unset}|${TMPDIR-unset}" > env.txt'),
        _bash(f"echo $$ > job.pid; exec {FOREVER}", background=True),
        finish_body("<p>done</p>", prompt_tokens=1000),
    ]
    requests: list[dict] = []
    agent = SorcarAgent("top-level")
    try:
        result = _run(agent, tmp_path, script, requests)
        assert result["success"] is True
        assert _scratch_dir(requests[0]) == "./tmp/"
        # what this process has itself (no mark, unless a sub-agent runs this test)
        assert (tmp_path / "env.txt").read_text().strip() == "|".join(
            os.environ.get(name, "unset") for name in ("KISS_SUBAGENT_DIR", "TMPDIR"))
        assert f"{tmp_path}/tmp/bash_jobs/" in _tool_results(requests[2])[-1]
        assert list((tmp_path / "tmp" / "bash_jobs").glob("*.log"))
        assert _running(_pid_from(tmp_path / "job.pid"))
    finally:
        for job in agent._background_jobs.values():
            job.kill.set()
            job.ended.result(timeout=10)


def test_finish_gate_runs_the_verifier_of_the_private_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate finds ``verify.sh`` in the sub-agent's folder and runs it in its environment.

    Writing the verifier there is no change of the work; the refusal names
    the script by its absolute path.
    """
    _pin_reviewer_off(tmp_path, monkeypatch)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    hello = str(work_dir / "hello.py")
    verifier = 'echo "$TMPDIR|$KISS_SUBAGENT_DIR" > gate_env.txt\ngrep -q hello hello.py\n'
    script = [
        tool_call_body("Write", {"file_path": "@SCRATCH@verify.sh", "content": verifier}, 1000),
        tool_call_body("Write", {"file_path": hello, "content": "print('bye')\n"}, 1000),
        finish_body("<p>done</p>", prompt_tokens=1000),  # refused: the verifier fails
        tool_call_body("Write", {"file_path": hello, "content": "print('hello')\n"}, 1000),
        finish_body("<p>done</p>", prompt_tokens=1000),
    ]
    requests: list[dict] = []
    agent = _sub_agent("sub-gate", ChatSorcarAgent)  # the class the daemon runs a child as
    result = _run(agent, work_dir, script, requests)
    assert result["success"] is True and len(requests) == len(script)
    folder = _scratch_dir(requests[0]).rstrip("/")
    refusal = _tool_results(requests[3])[-1]
    assert "Finish refused" in refusal and f"{folder}/verify.sh" in refusal
    assert f"./{folder}" not in refusal
    assert (work_dir / "gate_env.txt").read_text().strip() == f"{folder}|{folder}"
    assert agent._coding_gate.written_files == [hello]
    assert not Path(folder).exists() and not (work_dir / "tmp").exists()


def test_a_stopped_sub_agent_is_cleaned_up_too(tmp_path: Path) -> None:
    """A Stop that lands in a running command still kills what escaped and deletes the folder."""
    script = [_bash(f'echo "$KISS_SUBAGENT_DIR" > folder.txt; {ESCAPE}'),
              _bash(f"echo $$ > foreground.pid; exec {FOREVER}")]
    raised: list[BaseException] = []
    worker = threading.Thread(
        target=_run_recording, args=(_sub_agent("sub-stopped"), tmp_path, script, raised),
        daemon=True,
    )
    worker.start()
    try:
        escaped = _pid_from(tmp_path / "escaped.pid")
        folder = Path((tmp_path / "folder.txt").read_text().strip())
        assert _running(escaped) and folder.is_dir()
        foreground = _pid_from(tmp_path / "foreground.pid")  # the second command is running
        assert worker.ident is not None
        inject_async_exception(worker.ident, KeyboardInterrupt)
        worker.join(timeout=60)
        assert not worker.is_alive()
        assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
        assert _gone(foreground) and _gone(escaped) and not folder.exists()
    finally:
        if worker.is_alive() and worker.ident is not None:
            inject_async_exception(worker.ident, KeyboardInterrupt)
            worker.join(timeout=60)


def test_a_stop_repeated_during_the_clean_up_does_not_cut_it_short(tmp_path: Path) -> None:
    """The sweep and the deletion complete although a Stop lands in each of them.

    A task's Stop is injected again while its thread lives.  The first ends
    a running command; the second lands while the sweep reads the process
    table (the padded environments make that last), the third while the
    folder, 30 000 files, is being deleted.
    """
    script = [
        _bash('echo "$KISS_SUBAGENT_DIR" > folder.txt; mkdir "$TMPDIR/big" && '
              f'(cd "$TMPDIR/big" && seq 1 30000 | xargs touch); {ESCAPE}'),
        _bash(f"echo $$ > job.pid; exec {FOREVER}", background=True),
        _bash(f"echo $$ > foreground.pid; exec {FOREVER}"),
    ]
    agent = _sub_agent("sub-stopped-thrice")
    raised: list[BaseException] = []
    worker = threading.Thread(target=_run_recording, args=(agent, tmp_path, script, raised))
    slow = [subprocess.Popen(FOREVER.split(), env=PADDED) for _ in range(150)]
    try:
        worker.start()
        foreground = _pid_from(tmp_path / "foreground.pid", within=60)  # the third command runs
        escaped, job_pid = _pid_from(tmp_path / "escaped.pid"), _pid_from(tmp_path / "job.pid")
        folder = Path((tmp_path / "folder.txt").read_text().strip())
        job = next(iter(agent._background_jobs.values()))
        assert worker.ident is not None
        assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
        job.ended.result(timeout=30)  # the clean-up has killed the job: the sweep comes next
        time.sleep(0.02)
        assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
        deadline = time.monotonic() + 30
        while len(os.listdir(folder / "big")) == 30000 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert inject_async_exception(worker.ident, KeyboardInterrupt) == 1
        worker.join(timeout=60)
        assert not worker.is_alive()
        assert len(raised) == 1 and isinstance(raised[0], KeyboardInterrupt)
        assert _gone(foreground) and _gone(job_pid) and _gone(escaped)
        assert not folder.exists()
    finally:
        for process in slow:
            process.kill()
            process.wait(timeout=10)


def test_a_folder_that_cannot_be_deleted_does_not_fail_the_run(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """The result stands and a warning is logged; nothing is deleted through a link."""
    kept = tmp_path / "kept"
    kept.mkdir()
    (kept / "file").write_text("x")
    script = [
        _bash(f'rm -rf "$TMPDIR" && ln -s {shlex.quote(str(kept))} "$TMPDIR"'),
        finish_body("<p>done</p>", prompt_tokens=1000),
    ]
    requests: list[dict] = []
    result = _run(_sub_agent("sub-link"), tmp_path, script, requests)
    folder = Path(_scratch_dir(requests[0]).rstrip("/"))
    try:
        assert result["success"] is True
        assert folder.is_symlink() and (kept / "file").read_text() == "x"
        assert f"Could not delete the sub-agent folder {folder}" in caplog.text
    finally:
        folder.unlink(missing_ok=True)


def test_sub_agent_in_a_container_is_cleaned_up_in_the_container(
    container: Any, tmp_path: Path,
) -> None:
    """Docker mode: the folder, the temp directory, the job files, the sweep are the container's."""
    attached = f"container:{container.id}"
    job = _bash(f"echo $$ > job.pid; exec {FOREVER}", background=True)
    done = finish_body("<p>done</p>", success=True, prompt_tokens=1000)
    requests: list[dict] = []
    result = _run(_sub_agent("docker-sub"), tmp_path, [
        _bash('echo "$TMPDIR|$TMP|$TEMP|$KISS_SUBAGENT_DIR" > env.txt; '
              f'mkdir -p "$TMPDIR/ro/d" && chmod -R a-w "$TMPDIR/ro"; '
              f'{CONTAINER_ESCAPE}'),
        job,
        done,
    ], requests, docker_image=attached)
    assert result["success"] is True
    folder = _scratch_dir(requests[0]).rstrip("/")
    assert folder.startswith("/tmp/kiss-sub-")
    assert _sh(container, "cat env.txt") == "|".join([folder] * 4)
    # the job kept its files in the folder, none in the directory the container's tasks share
    assert f"; log: {folder}/kiss-jobs/" in _tool_results(requests[2])[-1]
    assert _sh(container, f"ls -d {JOBS_DIR} 2>/dev/null") == ""
    pids = _sh(container, "cat escaped.pid job.pid").split()
    assert len(pids) == 2
    # no process of the two is left, none carries the mark, no folder
    assert _alive_in(container, pids) == []
    assert _sh(container, f"ls -d {folder} 2>/dev/null") == ""
    assert _sh(container, "grep -l KISS_SUBAGENT_DIR /proc/[0-9]*/environ 2>/dev/null") == ""
    # a top-level task in the same container: the work tree's scratch dir, no mark, and
    # the shared directory for its job's files, which stay
    top: list[dict] = []
    _run(SorcarAgent("docker-top"), tmp_path, [
        _bash('echo "${KISS_SUBAGENT_DIR-unset}|${TMPDIR-unset}" > top.txt'), job, done,
    ], top, docker_image=attached)
    assert _scratch_dir(top[0]) == "./tmp/" and _sh(container, "cat top.txt") == "unset|unset"
    assert f"; log: {JOBS_DIR}/" in _tool_results(top[2])[-1]
    assert _sh(container, f"ls {JOBS_DIR}/*.log") != ""
    assert _alive_in(container, _sh(container, "cat job.pid").split()) != []


def test_a_sub_agents_container_jobs_keep_their_files_in_its_folder(container: Any) -> None:
    """A job started or adopted in a sub-agent's name keeps its files in the sub-agent's folder.

    They leave with the folder.  A job another task started in the same
    container keeps its files in the shared directory and runs on.
    """
    manager = DockerManager(f"container:{container.id}", workdir="/w")
    manager.open()
    try:
        jobs = ContainerJobs(manager.Bash)
        shared, report = jobs.start(FOREVER)
        assert shared is not None and f"; log: {JOBS_DIR}/{shared}.log." in report
        folder = manager.make_private_dir()
        assert folder is not None
        own, report = jobs.start(f"echo ready; exec {FOREVER}")
        assert own is not None and f"; log: {folder}/kiss-jobs/{own}.log." in report
        left = manager.Bash(f"nohup {FOREVER} > /dev/null 2>&1 & echo $!", "leave it running",
                            30, 500)
        adopted, report = jobs.adopt(int(left), FOREVER, "")
        assert adopted is not None and report.startswith(f"left running as job {adopted} ")
        in_folder = _sh(container, f"ls {folder}/kiss-jobs").split()
        assert {name.partition(".")[0] for name in in_folder} == {own, adopted}
        in_shared = _sh(container, f"ls {JOBS_DIR}").split()
        assert {name.partition(".")[0] for name in in_shared} == {shared}
        # each job is looked for where its files are
        assert jobs.last_running() == adopted
        assert f"log: {folder}/kiss-jobs/{own}.log" in jobs.bash_job(own, "tail")
        waited = jobs.bash_job(own, "wait", timeout_seconds=30, until="ready")
        assert "status: running" in waited and "ready" in waited
        assert "status: exited" in jobs.bash_job(adopted, "kill")
        assert jobs.last_running() == own
        manager.remove_private_dir()
        assert _sh(container, f"ls -d {folder} 2>/dev/null") == ""
        assert GONE_STATUS in jobs.bash_job(own, "tail")
        assert jobs.last_running() == shared
        assert _sh(container, f"ls {JOBS_DIR}").split() == in_shared
        # a job started now is the container's again
        again, report = jobs.start("true")
        assert again is not None and f"; log: {JOBS_DIR}/{again}.log." in report
    finally:
        manager.close()


#: The container's two sweeps: the variable that marks a process, its value, the script.
SWEEPS = [
    pytest.param("KISS_SUBAGENT_DIR", "/tmp/kiss-sub-a", remove_private_dir_script, id="folder"),
    pytest.param("KISS_EXEC_TOKEN", "0f1e2d3c4b5a6978", kill_tagged_tree_script, id="token"),
]


@pytest.mark.parametrize(("variable", "value", "script"), SWEEPS)
def test_container_sweep_kills_the_marked_processes_and_no_other(
    container: Any, variable: str, value: str, script: Callable[[str], str],
) -> None:
    """A sweep of the container takes the mark, and no look-alike.

    The folder's sweep takes a folder nested in it too, and deletes the folder.
    """
    folder = "/tmp/kiss-sub-a"
    environments = [
        f"{variable}={value}",
        f"{variable}={value}/kiss-sub-b",  # a folder nested in the value
        f"OTHER_{variable}={value}",  # another variable
        f"{variable}={value}b",  # another value
        f"NOTE=\"$(printf 'x\\n{variable}={value}')\"",  # a line of another's value
    ]
    pids = _sh(container, f"mkdir -p {folder}/ro/d {folder}b && chmod -R a-w {folder}/ro; "
               + " ".join(f"{environment} setsid {FOREVER} > /dev/null 2>&1 & echo $!;"
                          for environment in environments)).split()
    assert len(pids) == 5 and _alive_in(container, pids) == pids
    container.exec_run(["/bin/sh", "-c", script(value)])
    of_folder = script is remove_private_dir_script
    assert _alive_in(container, pids) == pids[2 if of_folder else 1:]
    assert _sh(container, f"ls -d {folder} {folder}b 2>/dev/null").split() == (
        [f"{folder}b"] if of_folder else [folder, f"{folder}b"])


@pytest.mark.parametrize(("variable", "value", "script"), SWEEPS)
def test_container_sweep_ends_a_tree_that_runs_on_while_it_is_read(
    container: Any, variable: str, value: str, script: Callable[[str], str],
) -> None:
    """No process of a marked tree outlives the sweep, and none gets a step further in it.

    A shell whose child the sweep reads first (``/proc`` is read in the
    order of the pids as text) does not go on to its next command when
    the child is killed, and the children a loop starts while the
    process table is read (200 idle processes make the reading last)
    are found as well.
    """
    mark = f"{variable}={value}"
    # The waiting shell gets a pid of two digits that begins with a 5 at least and
    # its child one of three that begins with a 1: the 200 pids after it lie between
    # the two in the sweep's order.
    burn = "i=0; while [ $i -lt 50 ]; do /bin/true; i=$((i+1)); done"
    waiting = ("last=$$; until [[ $((last + 1)) < $$ ]]; do /bin/true & last=$!; wait; done; "
               f"{FOREVER} & echo $$ $! > /w/waiting; wait; touch /w/went-on")
    parent, child = _sh(
        container, f"{burn}; {mark} setsid bash -c {shlex.quote(waiting)} > /dev/null 2>&1 & "
        "while [ ! -s /w/waiting ]; do sleep 0.05; done; cat /w/waiting").split()
    idle = _sh(container, "i=0; while [ $i -lt 200 ]; do "
               f"{FOREVER} > /dev/null 2>&1 & echo $!; i=$((i+1)); done").split()
    assert all(child < pid < parent for pid in idle), (child, parent, idle)
    spawning = f"while :; do {FOREVER} & sleep 0.02; done"
    _sh(container, f"{mark} setsid bash -c {shlex.quote(spawning)} > /dev/null 2>&1 &")
    container.exec_run(["/bin/sh", "-c", script(value)])
    assert _sh(container, "ls /w") == "waiting"
    marked = ("for d in /proc/[0-9]*; do tr '\\0' '\\n' 2>/dev/null < $d/environ "
              f"| grep -qxF {shlex.quote(mark)} && echo ${{d#/proc/}}; done; true")
    assert _sh(container, marked) == ""
    assert _alive_in(container, idle) == idle


@pytest.fixture
def long_sweep() -> Iterator[tuple[Any, DockerManager, str]]:
    """A container, its manager and a private folder whose sweep lasts over five seconds.

    The container's first process carries the folder's mark (it takes it
    from a file once the folder has a name), and no signal sent from
    inside a container ends that one: the sweep finds it in each of its
    100 readings of the process table.  A second marked process, whose
    pid is in ``/first-swept``, dies in the first.
    """
    first = 'while [ ! -s /mark ]; do sleep 0.1; done; exec env "$(cat /mark)" tail -f /dev/null'
    container = _start_container(command=["sh", "-c", first], init=False)
    manager = DockerManager(f"container:{container.id}", workdir="/w")
    try:
        manager.open()
        folder = manager.make_private_dir()
        assert folder is not None
        _sh(container, f"echo KISS_SUBAGENT_DIR={folder} > /mark; until tr '\\0' '\\n' "
            "< /proc/1/environ | grep -q KISS_SUBAGENT_DIR; do sleep 0.1; done; "
            f"KISS_SUBAGENT_DIR={folder} setsid {FOREVER} > /dev/null 2>&1 & "
            "echo $! > /first-swept")
        yield container, manager, folder
    finally:
        manager.close()
        container.reload()
        if container.status == "paused":
            container.unpause()
        container.remove(force=True)


def _await_sweep(container: Any) -> None:
    """Return once the sweep of the :func:`long_sweep` *container* has made its first kill."""
    victim = _sh(container, "cat /first-swept")
    deadline = time.monotonic() + 30
    while _alive_in(container, [victim]):
        assert time.monotonic() < deadline, "no sweep of the container started"


@pytest.mark.slow
def test_a_container_frozen_under_the_sweep_does_not_hold_the_sub_agents_end(
    long_sweep: tuple[Any, DockerManager, str],
) -> None:
    """The sweep of a container that froze under it is given up at its deadline."""
    container, manager, _ = long_sweep
    ending = threading.Thread(target=manager.remove_private_dir, daemon=True)
    ending.start()
    _await_sweep(container)
    container.pause()
    ending.join(_SWEEP_TIMEOUT_S + _KILL_WAIT_S + 15)
    assert not ending.is_alive()


def test_a_stop_during_the_containers_sweep_does_not_cut_it_short(
    long_sweep: tuple[Any, DockerManager, str],
) -> None:
    """A Stop injected while the container is swept is raised once the folder is gone."""
    container, manager, folder = long_sweep
    raised: list[BaseException] = []

    def end() -> None:
        try:
            manager.remove_private_dir()
        except KeyboardInterrupt as stop:
            raised.append(stop)

    ending = threading.Thread(target=end, daemon=True)
    ending.start()
    _await_sweep(container)
    assert inject_async_exception(ending.ident or 0, KeyboardInterrupt) == 1
    ending.join(60)
    assert not ending.is_alive() and len(raised) == 1
    assert _sh(container, f"ls -d {folder} 2>/dev/null") == ""


def _png_reply(command: dict[str, Any]) -> str:
    """The extension's answer to any command of the bridge: a screenshot."""
    return f"data:image/png;base64,{_TINY_PNG_B64}"


def test_host_tools_of_a_sub_agent_in_a_container_write_to_a_host_folder(
    container: Any, tmp_path: Path,
) -> None:
    """What a tool writes on the host for a sub-agent in a container is not left in the work tree.

    The screenshot the Chrome bridge saves goes to a private folder on the
    host, deleted when the sub-agent ends; the scratch dir stays the
    container's folder.
    """
    bridge = ChromeBridge.instance()
    peer = ExtensionPeer(bridge.port, bridge.token, _png_reply)
    requests: list[dict] = []
    try:
        result = _run(_sub_agent("docker-shot"), tmp_path, [
            _bash('echo "$KISS_SUBAGENT_DIR" > folder.txt'),
            tool_call_body("use_my_chrome", {"action": "screenshot", "tab_id": 1}, 1000),
            finish_body("<p>done</p>", prompt_tokens=1000),
        ], requests, web_tools=True, docker_image=f"container:{container.id}")
    finally:
        peer.close()
        ChromeBridge.shutdown()
        (extension_dir() / CONFIG_FILE_NAME).unlink(missing_ok=True)
    assert result["success"] is True
    saved = _tool_results(requests[2])[-1]
    assert saved.startswith("Screenshot saved to ")
    shot = Path(saved.removeprefix("Screenshot saved to ").partition(" (")[0])
    assert shot.name.startswith("chrome-shot-") and shot.parent.name.startswith("kiss-sub-")
    assert shot.parent.parent.resolve() == Path(tempfile.gettempdir()).resolve()
    in_container = _sh(container, "cat folder.txt")
    assert _scratch_dir(requests[0]) == f"{in_container}/" and in_container != str(shot.parent)
    assert not shot.parent.exists() and not (tmp_path / "tmp").exists()


def test_a_container_that_cannot_hold_the_folder_leaves_the_scratch_dir_in_the_work_tree(
    tmp_path: Path,
) -> None:
    """Read-only, the container gets no folder: the scratch dir is ``./tmp/``, not the host's."""
    container = _start_container(read_only=True)
    try:
        requests: list[dict] = []
        result = _run(_sub_agent("docker-read-only"), tmp_path, [
            _bash('echo "${KISS_SUBAGENT_DIR-unset}"'),
            finish_body("<p>done</p>", prompt_tokens=1000),
        ], requests, docker_image=f"container:{container.id}")
    finally:
        container.remove(force=True)
    assert result["success"] is True
    assert _scratch_dir(requests[0]) == "./tmp/" and "unset" in _tool_results(requests[1])[-1]


class _EndsBeforeTheSecondListing:
    """Profile hook: end *victim* once ``ps eww`` has listed it and before ``ps ww`` runs."""

    def __init__(self, victim: subprocess.Popen) -> None:
        self.victim = victim

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "call" or frame.f_code is not subprocess.run.__code__:
            return
        if frame.f_locals["popenargs"][0][:2] == ["ps", "ww"] and self.victim.poll() is None:
            self.victim.kill()
            self.victim.wait(timeout=10)


def test_sweep_kills_the_marked_processes_and_no_other(tmp_path: Path) -> None:
    """The mark, a folder nested in it and what they fork meanwhile die; a look-alike lives,
    and so does a command that only names the mark among its arguments, also one that is
    replaced (``exec``) every few milliseconds.  A marked process that ends between the
    listing's two readings is not listed."""
    folder = f"{tmp_path}/kiss-sub-a"

    def start(value: str, command: str = f"exec {HOST_FOREVER}") -> subprocess.Popen:
        return subprocess.Popen(["bash", "-c", command], start_new_session=True,
                                env={**os.environ, "KISS_SUBAGENT_DIR": value})

    exact, nested, victim = start(folder), start(f"{folder}/kiss-sub-b"), start(folder)
    forking_script = """import os
import time
while True:
    if os.fork() == 0:
        time.sleep(600)
        os._exit(0)
    time.sleep(0.01)
"""
    forking = start(
        folder,
        f"exec {shlex.quote(sys.executable)} -c {shlex.quote(forking_script)}",
    )
    lookalike = start(f"{folder}b")
    mark = f"KISS_SUBAGENT_DIR={folder}"
    naming = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", mark],
                              start_new_session=True)
    restless = tmp_path / "restless.sh"
    restless.write_text('#!/bin/bash\nsleep 0.005\nexec "$0" "$1" "$(( ($2 + 1) % 1000 ))"\n')
    restless.chmod(0o755)
    renaming = subprocess.Popen([str(restless), mark, "0"], start_new_session=True)
    try:
        time.sleep(0.5)  # the loop has forked a few dozen processes
        # the listing read where there is no /proc names the same processes
        listed = _env_marked_pids_ps(mark, os.getpid())
        assert {exact.pid, nested.pid, forking.pid} <= set(listed)
        assert not {lookalike.pid, naming.pid, renaming.pid} & set(listed)
        # ... and, read alone right before its signal, each is told from a look-alike
        assert _env_marked_pids_ps(mark, os.getpid(), nested.pid) == [nested.pid]
        assert _env_marked_pids_ps(mark, os.getpid(), lookalike.pid) == []
        assert _env_marked_pids_ps(mark, os.getpid(), naming.pid) == []
        # ... whatever its command was by the second of the listing's two readings
        alone = [_env_marked_pids_ps(mark, os.getpid(), renaming.pid) for _ in range(20)]
        assert alone == [[]] * 20
        # ... and a process that the second reading no longer holds is not listed
        sys.setprofile(_EndsBeforeTheSecondListing(victim))
        try:
            listed = _env_marked_pids_ps(mark, os.getpid())
        finally:
            sys.setprofile(None)
        assert victim.poll() == -9 and victim.pid not in listed
        assert {exact.pid, nested.pid, forking.pid} <= set(listed)
        kill_env_marked(mark)
        assert [p.wait(timeout=10) for p in (exact, nested, forking)] == [-9, -9, -9]
        groups = subprocess.run(["ps", "-eo", "pgid=,stat="], capture_output=True, text=True,
                                check=True, timeout=10).stdout.split("\n")
        left = [line for line in groups if line.split()[:1] == [str(forking.pid)]
                and not line.split()[1].startswith("Z")]
        assert left == []
        assert lookalike.poll() is None and naming.poll() is None and renaming.poll() is None
    finally:
        for process in (exact, nested, victim, forking, lookalike, naming, renaming):
            subprocess.run(["pkill", "-KILL", "-g", str(process.pid)], check=False, timeout=10)
            process.wait(timeout=10)


@pytest.mark.parametrize("pidfd", [True, False])
def test_sweep_spares_a_stranger_that_took_a_marked_pid(pidfd: bool) -> None:
    """A pid that passed to another process during the sweep is not signalled.

    With a pidfd the sweep signals the process it read, whatever became of
    the pid; without one (refused here by a seccomp filter) it signals the
    pid right after it read its environment.
    """
    unshare = ["unshare", "--user", "--map-root-user", "--pid", "--fork", "--kill-child",
               "--mount-proc"]
    if shutil.which("unshare") is None or subprocess.run(
            [*unshare, "true"], capture_output=True, check=False, timeout=30).returncode != 0:
        pytest.skip("needs a pid namespace of its own (unshare, user namespaces)")
    done = subprocess.run(
        [*unshare, sys.executable, "-c", PID_REUSE, "pidfd" if pidfd else "no-pidfd"],
        capture_output=True, text=True, check=False, timeout=120,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, sys.path))},
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"pidfd": pidfd, "reused": True, "stranger": None}

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Docker mode gets the rules of the host tools: redirected rewrites, free waits, image reads.

The container ``Bash`` (``DockerManager.Bash``) only refused a shell
rewrite of a file; it now raises the same
:class:`~kiss.core.tool_verdict.ToolRedirect` the host tool raises, so
the agent makes the ``Write``/``Edit`` through its hook and events.  The
container ``Bash`` closure of ``SorcarAgent`` applies the host's sleep
rule with ``ContainerJobs``: a long foreground ``sleep`` waits on the
running job instead.  ``DockerTools.Read`` of an image or PDF fetches
the bytes (``DockerManager.read_bytes``) and returns the attachment
encoding the host ``Read`` returns, instead of raw bytes as text.

Every test runs against a real ``python:3.11-slim`` container through the
real ``SorcarAgent`` tool set and a real ``KISSAgent`` dispatch (no model:
the calls are issued by hand); skipped without a Docker daemon.  Nothing
is mocked.
"""

from __future__ import annotations

import base64
import functools
import json
import os
import time
from collections.abc import Iterator
from typing import Any

import pytest

from kiss.agents.sorcar.container_jobs import JOBS_DIR, ContainerJobs
from kiss.agents.sorcar.docker_manager import STOPPED_MESSAGE, DockerManager, fetched_bytes
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.edit_guards import (
    HEREDOC_UNRESOLVED_VERDICT,
    HEREDOC_VERDICT,
    edit_call,
    rewrite_guard,
)
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import CD_VERDICT
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model import encode_binary_attachment, parse_binary_attachments
from kiss.core.printer import Printer
from kiss.core.tool_verdict import ALLOW, Verdict, call_text, refuse
from kiss.tests.agents.sorcar.docker_test_containers import IMAGE, is_docker_available

pytestmark = pytest.mark.skipif(not is_docker_available(), reason="Docker daemon is not running")

#: A 1x1 red PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
)
HEREDOC = "cat > /tmp/notes.md <<'EOF'\nhello\nEOF"
SED_MULTILINE = "sed -i 's/hello/hi\\nthere/' /tmp/notes.md"
PYTHON_TWO_REPLACES = (
    "python3 - <<'EOF'\n"
    "p = '/tmp/notes.md'\n"
    "s = open(p).read()\n"
    "s = s.replace('hi', 'ho').replace('there', 'here')\n"
    "open(p, 'w').write(s)\n"
    "EOF"
)


@pytest.fixture(scope="module")
def env() -> Iterator[DockerManager]:
    """One container of :data:`IMAGE` working in ``/tmp`` for the whole module."""
    manager = DockerManager(IMAGE, workdir="/tmp")
    manager.open()
    try:
        yield manager
    finally:
        manager.close()


class _RecordingPrinter(Printer):
    """A real ``Printer`` that keeps every event for the assertions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, Any, dict[str, Any]]] = []

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> None:
        """Record the event.

        Args:
            content: The content to display.
            type: Content type (``tool_call``, ``tool_result``, ...).
            **kwargs: Type-specific options (``tool_input``, ``is_error``).
        """
        self.events.append((type, content, kwargs))

    def token_callback(self, token: str) -> None:
        """Ignore streamed tokens.

        Args:
            token: The text token (unused).
        """


class _Harness:
    """The Docker-mode tool set of a real ``SorcarAgent``, dispatched through a ``KISSAgent``.

    Args:
        env: The open container manager the agent runs its tools in.
    """

    def __init__(self, env: DockerManager) -> None:
        self.sorcar = SorcarAgent("docker-parity")
        self.sorcar._use_web_tools = False
        self.sorcar.docker_manager = env
        self.tools = {tool.__name__: tool for tool in self.sorcar._get_tools()}
        self.jobs: ContainerJobs = self.sorcar._container_jobs
        self.printer = _RecordingPrinter()
        self.hook_calls: list[tuple[str, dict[str, Any]]] = []
        #: The verdict the hook gives a ``Write``/``Edit`` (everything else passes).
        self.verdict = ALLOW
        self.agent: Any = KISSAgent("docker-parity-executor")
        self.agent.printer = self.printer
        self.agent.tool_call_hook = self.hook
        self.agent.function_map = dict(self.tools)

    def hook(self, name: str, args: dict[str, Any]) -> Verdict:
        """Record the call and give :attr:`verdict` to a ``Write``/``Edit``."""
        self.hook_calls.append((name, dict(args)))
        return self.verdict if name in ("Write", "Edit") else ALLOW

    def call(self, name: str, **args: Any) -> str:
        """Issue the tool call *name* with *args* as the agent loop does; return its result."""
        _, response, _ = self.agent._guarded_execute_tool({"name": name, "arguments": args})
        return str(response)

    def tool_calls(self) -> list[Any]:
        """The names of the printed ``tool_call`` events, in order."""
        return [content for etype, content, _ in self.printer.events if etype == "tool_call"]


def _cat(env: DockerManager, path: str) -> str:
    """The content of *path* in the container (``cat``, no exit-code marker on success)."""
    return env.Bash(f"cat {path}", "cat", 30, 50000)


def test_container_rewrites_are_made_as_write_and_edit_through_the_agent(
    env: DockerManager,
) -> None:
    """A heredoc copy, an append, a ``sed -i`` and a python rewrite reach the hook as Write/Edit."""
    h = _Harness(env)
    env.Bash("rm -f /tmp/notes.md /tmp/notes2.md", "clean", 30, 500)
    # A copy creating a new file is redirected too (the container tracks no shown files).
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert out == (
        'Applied as Write(file_path="/tmp/notes.md", content="hello\\n"), whose result follows:\n'
        "Successfully wrote 6 characters to /tmp/notes.md"
    ), out
    assert _cat(env, "/tmp/notes.md") == "hello\n"
    assert h.hook_calls == [
        ("Bash", {"command": HEREDOC, "description": "rewrite"}),
        ("Write", {"file_path": "/tmp/notes.md", "content": "hello\n"}),
    ]
    assert h.tool_calls() == ["Bash", "Write"]
    # An append: the Write carries the current text followed by the body.
    out = h.call("Bash", command="cat >> /tmp/notes.md <<'EOF'\nmore\nEOF", description="append")
    assert out.startswith(
        'Appended through Write(file_path="/tmp/notes.md", content="hello\\nmore\\n")',
    )
    assert _cat(env, "/tmp/notes.md") == "hello\nmore\n"
    assert h.hook_calls[-1] == ("Write", {"file_path": "/tmp/notes.md", "content": "hello\nmore\n"})
    # A relative target resolves against the container's working directory.
    out = h.call("Bash", command="tee notes2.md <<'EOF' > /dev/null\nvia tee\nEOF", description="t")
    assert out.startswith('Applied as Write(file_path="/tmp/notes2.md", content="via tee\\n")')
    assert _cat(env, "/tmp/notes2.md") == "via tee\n"
    # A copy into /dev/ runs as the shell command it is.
    assert h.call("Bash", command="cat > /dev/null <<'EOF'\nx\nEOF", description="null") == ""
    assert h.hook_calls[-1][0] == "Bash"
    # A one-substitution multi-line sed -i and a python .replace rewrite are Edits.
    out = h.call("Bash", command=SED_MULTILINE, description="sed")
    call = edit_call("/tmp/notes.md", [("hello", "hi\nthere", False)])
    assert out.startswith(f"Applied as {call}, whose result follows:\n" + (
        "Successfully replaced 1 occurrence(s) in /tmp/notes.md"
    )), out
    assert _cat(env, "/tmp/notes.md") == "hi\nthere\nmore\n"
    assert h.hook_calls[-1] == ("Edit", {
        "file_path": "/tmp/notes.md", "old_string": "hello", "new_string": "hi\nthere",
        "replace_all": False, "exact": True,
    })
    out = h.call("Bash", command=PYTHON_TWO_REPLACES, description="python")
    assert out.startswith('Applied as Edit(file_path="/tmp/notes.md", edits=[') and (
        "edit 2: Successfully replaced 1 occurrence(s)" in out
    ), out
    assert _cat(env, "/tmp/notes.md") == "ho\nhere\nmore\n"
    name, args = h.hook_calls[-1]
    edits = args["edits"]
    assert name == "Edit" and isinstance(edits, str)
    assert '"old_string": "hi", "new_string": "ho", "replace_all": true' in edits
    # The hook's refusal of the redirected call is the rewrite's result; nothing changed.
    h.verdict = refuse("no edits before the ledger is written")
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert out.endswith(
        "whose result follows:\nREFUSED: the Write call did NOT run and had no effect. "
        "no edits before the ledger is written"
    ), out
    assert _cat(env, "/tmp/notes.md") == "ho\nhere\nmore\n"
    h.verdict = ALLOW
    # Only the shell knows a "$OUT" target: refused as before, no Write reaches the hook.
    variable = 'OUT=/tmp/notes.md; cat > "$OUT" <<\'EOF\'\nx\nEOF'
    out = h.call("Bash", command=variable, description="v")
    assert out.startswith(HEREDOC_UNRESOLVED_VERDICT.format(path='"$OUT"')[:40])
    assert h.hook_calls[-1][0] == "Bash"
    # run_commands_parallel cannot redirect a command: the copy stays refused there.
    report = h.tools["run_commands_parallel"](json.dumps([HEREDOC]))
    assert HEREDOC_VERDICT.format(path="/tmp/notes.md") in report, report
    assert _cat(env, "/tmp/notes.md") == "ho\nhere\nmore\n"
    # An append to a file whose text is not UTF-8 is an error, not a damaged Write.
    env.Bash("printf '\\377\\376' > /tmp/bin.md", "binary", 30, 500)
    out = h.call("Bash", command="cat >> /tmp/bin.md <<'EOF'\nmore\nEOF", description="append")
    assert out.startswith("Error: cannot append to /tmp/bin.md: 'utf-8' codec can't decode"), out
    assert h.hook_calls[-1][0] == "Bash"


def test_container_long_sleep_waits_on_the_running_job(env: DockerManager) -> None:
    """``sleep 60; cmd`` with a job running waits on the job, then runs ``cmd``; no job: as is."""
    h = _Harness(env)
    bash = h.tools["Bash"]
    assert h.jobs.last_running() is None
    # No running job: a sleep within the deadline runs as written, one past it is refused.
    assert bash("sleep 1; echo quick", "quick").strip() == "quick"
    refused = bash("sleep 31; echo late", "late", timeout_seconds=10)
    assert refused.startswith("REFUSED: the Bash call did NOT run: a foreground sleep of 31s")
    assert "bash_job" in refused
    started = bash("sleep 4; echo done", "job", background=True)
    job_id = started.split()[3]
    assert h.jobs.last_running() == job_id
    # The long sleep is spent waiting on the job; the rest of the command runs after it.
    began = time.monotonic()
    out = bash("sleep 60; echo after", "wait", timeout_seconds=120)
    assert time.monotonic() - began < 40, "the sleep was not replaced by the wait on the job"
    assert out == f"after\nWaited on job {job_id} for up to 60 s instead of sleeping (exited 0).\n"
    assert h.jobs.last_running() is None
    assert h.jobs.wait("deadbeef", 1) == "status unknown"
    # on_timeout="background" after the wait: the rest starts as a job and is waited for.
    started = bash("sleep 3; echo done", "job", background=True)
    job_id = started.split()[3]
    out = bash("sleep 50; echo late", "late", timeout_seconds=60, on_timeout="background")
    assert out.endswith(
        f"Waited on job {job_id} for up to 50 s instead of sleeping (exited 0).\n",
    )
    assert "status: exited 0" in out and "late" in out.splitlines()
    # A job that cannot start (a file sits where the jobs dir goes) is an error, not a job.
    try:
        env.Bash(f"mv {JOBS_DIR} {JOBS_DIR}.bak && touch {JOBS_DIR}", "block", 30, 500)
        out = bash("echo x", "blocked", on_timeout="background")
        assert out.startswith("Error: could not start the background job"), out
    finally:
        env.Bash(f"rm -f {JOBS_DIR}; mv {JOBS_DIR}.bak {JOBS_DIR}", "unblock", 30, 500)


def test_container_read_attaches_images_and_pdfs(env: DockerManager) -> None:
    """``Read`` of a supported binary returns the host's attachment encoding, not raw bytes."""
    h = _Harness(env)
    encoded = base64.b64encode(PNG).decode()
    env.Bash(
        f"printf '%s' {encoded} | base64 -d > /tmp/pic.png && printf '%%PDF-1.4\\n' > /tmp/doc.pdf "
        "&& printf 'plain\\n' > /tmp/plain.txt", "files", 30, 500,
    )
    out = h.call("Read", file_path="/tmp/pic.png")
    assert out == (
        f"Read binary file /tmp/pic.png as image/png ({len(PNG)} bytes); content attached below.\n"
        + encode_binary_attachment("image/png", PNG)
    )
    text, attachments = parse_binary_attachments(out)
    assert text.startswith("Read binary file /tmp/pic.png") and len(attachments) == 1
    assert (attachments[0].data, attachments[0].mime_type) == (PNG, "image/png")
    # A relative path resolves against the working directory; a PDF attaches as one.
    assert parse_binary_attachments(h.call("Read", file_path="pic.png"))[1][0].data == PNG
    assert parse_binary_attachments(h.call("Read", file_path="doc.pdf"))[1][0].mime_type == (
        "application/pdf"
    )
    # Text files still read as text; the model's bad arguments are still rejected first.
    assert h.call("Read", file_path="/tmp/plain.txt") == "plain\n"
    assert h.call("Read", file_path="/tmp/pic.png", max_lines=0).startswith("Error: max_lines")
    # A missing image, a file over the limit, and an environment without a byte fetch.
    out = h.call("Read", file_path="/tmp/missing.png")
    assert out.startswith("Error: Cannot read binary file /tmp/missing.png (mime=image/png): ")
    assert "cannot read /tmp/missing.png: " in out and "No such file" in out
    assert env.read_bytes("/tmp/pic.png") == PNG
    with pytest.raises(OSError, match=f"/tmp/pic.png is {len(PNG)} bytes, over the 10 byte limit"):
        env.read_bytes("/tmp/pic.png", max_bytes=10)
    small = DockerTools(env.Bash, functools.partial(env.read_bytes, max_bytes=10))
    assert "over the 10 byte limit" in small.Read("/tmp/pic.png")
    assert DockerTools(env.Bash).Read("/tmp/pic.png").startswith(
        "Error: Cannot read binary file /tmp/pic.png (mime=image/png): this environment fetches "
        "no file bytes"
    )
    with pytest.raises(OSError, match="cannot read /tmp: "):
        env.read_bytes("/tmp")


def test_fetched_bytes_reports_every_way_a_fetch_fails() -> None:
    """The decoder of the fetch script's output names the failure it saw."""
    assert fetched_bytes("p", 100, 0, f"{len(PNG)}\n{base64.b64encode(PNG).decode()}\n") == PNG
    with pytest.raises(OSError, match="cannot read p: timed out"):
        fetched_bytes("p", 100, None, "")
    with pytest.raises(OSError, match="cannot read p: exit code 2"):
        fetched_bytes("p", 100, 2, "")
    with pytest.raises(OSError, match="cannot read p: sh: p: No such file"):
        fetched_bytes("p", 100, 1, "sh: p: No such file\n")
    with pytest.raises(OSError, match="p is 500 bytes, over the 100 byte limit"):
        fetched_bytes("p", 100, 1, "500\n")
    with pytest.raises(OSError, match="cannot read p: base64: read error"):
        fetched_bytes("p", 100, 1, "5\nbase64: read error\n")
    with pytest.raises(OSError, match="cannot read p: the fetch did not finish"):
        fetched_bytes("p", 100, 1, "5\n")
    with pytest.raises(OSError, match="cannot read p: the fetched base64 is damaged"):
        fetched_bytes("p", 100, 0, "5\n%%%%\n")
    # An encoder that died mid-file leaves well-formed base64 of a prefix and the
    # pipeline's exit code (tr's) says 0: the announced size tells the truncation.
    with pytest.raises(OSError, match="cannot read p: fetched 3 of its 5 bytes"):
        fetched_bytes("p", 100, 0, f"5\n{base64.b64encode(b'abc').decode()}\n")
    # A fetch that timed out or was stopped mid-file (A08 claude-4): the base64 that
    # arrived (megabytes of it, possibly) is no reason, what else the pipeline printed is.
    partial = base64.b64encode(b"x" * 3000).decode()
    with pytest.raises(OSError) as timed_out:
        fetched_bytes("p", 100_000, None, f"5000\n{partial}")
    assert str(timed_out.value) == "cannot read p: the fetch timed out"
    with pytest.raises(OSError) as stopped:
        fetched_bytes("p", 100_000, -1, f"5000\n{partial}\n{STOPPED_MESSAGE}")
    assert str(stopped.value) == f"cannot read p: {STOPPED_MESSAGE}"


def _rest(command: str, description: str) -> dict[str, Any]:
    """The arguments of the follow-up ``Bash`` call running *command*, the rest of a command."""
    return {
        "command": command, "description": description,
        "timeout_seconds": 30, "max_output_chars": 8000,
    }


def test_a_container_cd_into_a_file_is_refused_before_the_rewrite(env: DockerManager) -> None:
    """``cd <file> && <rewrite>``: bash runs nothing after the failed ``cd``, so nothing is applied.

    The container probe answered the ``cd`` question with ``test -e``, so
    a path that exists but is no directory passed and the Write landed
    where bash would have stopped (review of 2026-10-31, finding 2); it
    now asks ``test -d``, as the host tool does, and a python rewrite's
    edits are settled from the container file's text as a sed's are.
    """
    h = _Harness(env)
    env.Bash("printf file > /tmp/not-a-dir; printf 'old\\n' > /tmp/victim", "seed", 30, 500)
    # The shell itself: the cd fails and the rest never runs.
    shell = env.Bash("cd /tmp/not-a-dir && echo ran", "cd", 30, 500)
    assert "Not a directory" in shell and shell.endswith("[exit code: 1]") and "ran" not in shell
    for command in (
        "cd /tmp/not-a-dir && cat > /tmp/victim <<'EOF' && cat /tmp/victim\nchanged\nEOF",
        "cd /tmp/not-a-dir && sed -i 's/old/new\\n/' /tmp/victim && cat /tmp/victim",
        "cd /tmp/not-a-dir && python3 - <<'EOF' && cat /tmp/victim\np = '/tmp/victim'\n"
        "s = open(p).read()\ns = s.replace('old', 'new')\nopen(p, 'w').write(s)\nEOF",
    ):
        out = h.call("Bash", command=command, description="cd, rewrite")
        assert out == CD_VERDICT.format(cwd="/tmp/not-a-dir"), out
        assert h.hook_calls[-1][0] == "Bash" and h.tool_calls()[-1] == "Bash"
        assert env.read_bytes("/tmp/victim") == b"old\n"
    # A cd to a directory still goes through, and a python rewrite over a container file
    # holding a "\r" is not an Edit (its read translates the line endings): refused as before.
    env.Bash("printf 'old\\r\\nkept\\r\\n' > /tmp/victim", "seed", 30, 500)
    crlf = (
        "cd /tmp && python3 - <<'EOF' && cat /tmp/victim\np = '/tmp/victim'\ns = open(p).read()\n"
        "s = s.replace('old', 'new')\nopen(p, 'w').write(s)\nEOF"
    )
    out = h.call("Bash", command=crlf, description="cd, rewrite")
    assert out == rewrite_guard(crlf) and env.read_bytes("/tmp/victim") == b"old\r\nkept\r\n", out
    env.Bash("printf 'old\\nkept\\n' > /tmp/victim", "seed", 30, 500)
    out = h.call("Bash", command=crlf, description="cd, rewrite")
    assert out.startswith('Applied as Edit(file_path="/tmp/victim", old_string="old"'), out
    assert out.endswith("whose result follows:\nnew\nkept\n"), out
    assert env.read_bytes("/tmp/victim") == b"new\nkept\n"
    # A text the fetch cannot read (over its size limit) settles no exact Edit either:
    # the rewrite is refused as before, not applied on nobody's word.
    env.Bash("yes $'old\\r' | head -c 21000000 > /tmp/victim", "seed 21 MB", 30, 500)
    out = h.call("Bash", command=crlf, description="cd, rewrite")
    assert out == rewrite_guard(crlf), out
    head = env.Bash("od -An -tx1 -N5 /tmp/victim", "peek", 30, 500)
    assert head.split() == ["6f", "6c", "64", "0d", "0a"], head  # still "old\r\n"


def test_a_container_tilde_target_is_not_resolved_in_the_daemons_home(env: DockerManager) -> None:
    """``cat > ~/x`` in the container is refused as unresolved, not written into the daemon's home.

    The rewrite expanded ``~`` with the daemon's own home directory and
    used that path inside the container (audit A06-9), creating the
    file under the host user's home there; only the container's shell
    knows its home, so the copy is refused and the agent is told to
    name the literal path.
    """
    h = _Harness(env)
    daemons_home = os.path.expanduser("~")
    out = h.call("Bash", command="cat > ~/report.txt <<'EOF'\nhello\nEOF", description="write")
    assert out == HEREDOC_UNRESOLVED_VERDICT.format(path="~/report.txt"), out
    assert h.hook_calls[-1][0] == "Bash" and h.tool_calls()[-1] == "Bash"
    listing = env.Bash(f"ls {daemons_home}/report.txt ~/report.txt", "nothing written", 30, 500)
    assert listing.count("No such file") == 2 and listing.endswith("[exit code: 2]"), listing


def test_a_container_cd_through_a_link_leaves_a_target_only_the_shell_knows(
    env: DockerManager,
) -> None:
    """``cd link/../deep; cat > victim <<EOF``: bash enters the ``deep`` the link leads to.

    There is no ``deep`` next to the link, so bash takes the path as
    written, through the link.  The guard judged ``deep/victim`` next to
    the link (no such file: a new one) and the command rewrote the
    existing file, unread (cross finding X5 of 2026-11-12).  The
    container is asked which directories exist, as the host is: the
    copy is refused by ``Bash`` and by the fan-out, and one after a
    ``cd`` bash walks as written creates its file as before.
    """
    h = _Harness(env)
    env.Bash(
        "mkdir -p /tmp/x5/other/sub /tmp/x5/other/deep /tmp/x5/plain/a /tmp/x5/plain/b"
        " && ln -s other/sub /tmp/x5/link && printf 'ORIGINAL\\n' > /tmp/x5/other/deep/victim",
        "seed", 30, 500,
    )
    landed = env.Bash("cd /tmp/x5/link/../deep && pwd -P", "the shell itself", 30, 500)
    assert landed == "/tmp/x5/other/deep\n", landed
    through_the_link = "cd /tmp/x5/link/../deep; cat > victim <<'EOF'\nowned\nEOF"
    out = h.call("Bash", command=through_the_link, description="through a link")
    assert out == HEREDOC_UNRESOLVED_VERDICT.format(path="victim"), out
    report = h.tools["run_commands_parallel"](json.dumps([through_the_link]))
    assert report.startswith("1 commands: 0 succeeded, 1 failed"), report
    assert HEREDOC_UNRESOLVED_VERDICT.format(path="victim") in report, report
    assert _cat(env, "/tmp/x5/other/deep/victim") == "ORIGINAL\n"
    walked = "cd /tmp/x5/plain/a/../b; cat > fresh <<'EOF'\nowned\nEOF"
    assert h.call("Bash", command=walked, description="walked") == ""
    assert _cat(env, "/tmp/x5/plain/b/fresh") == "owned\n"


def test_the_rest_of_a_compound_container_rewrite_runs_after_the_applied_call(
    env: DockerManager,
) -> None:
    """``cd DIR && <rewrite> && <rest>``: the Write/Edit through the agent, then the rest in DIR."""
    h = _Harness(env)
    env.Bash("rm -rf /tmp/app && mkdir -p /tmp/app", "clean", 30, 500)
    command = "cd /tmp/app && cat > notes.md <<'EOF' && cat notes.md && pwd\nhello\nEOF"
    out = h.call("Bash", command=command, description="rewrite, show")
    rest = _rest("cd /tmp/app && cat notes.md && pwd", "rewrite, show")
    assert out == (
        'Applied as Write(file_path="/tmp/app/notes.md", content="hello\\n"), whose result '
        "follows:\nSuccessfully wrote 6 characters to /tmp/app/notes.md\n"
        f"Then made {call_text('Bash', rest)}, whose result follows:\nhello\n/tmp/app\n"
    ), out
    assert h.hook_calls == [
        ("Bash", {"command": command, "description": "rewrite, show"}),
        ("Write", {"file_path": "/tmp/app/notes.md", "content": "hello\n"}),
        ("Bash", rest),
    ]
    assert h.tool_calls() == ["Bash", "Write", "Bash"]
    assert h.agent.tool_calls_made == 1, "neither the Write nor the rest is a call of its own"
    # A per-line sed gets the pairs exact for the file's text (read from the container).
    env.Bash("printf 'a\\na\\n' > /tmp/app/f.txt", "seed", 30, 500)
    sed = "cd /tmp/app && sed -i 's/a/b/' f.txt && cat f.txt"
    out = h.call("Bash", command=sed, description="sed, show")
    call = edit_call("/tmp/app/f.txt", [("a", "b", True)])
    rest = _rest("cd /tmp/app && cat f.txt", "sed, show")
    assert out.startswith(
        f"Applied as {call}, whose result follows:\nSuccessfully replaced 2 occurrence(s) in "
        "/tmp/app/f.txt"
    ), out
    assert out.endswith(f"\nThen made {call_text('Bash', rest)}, whose result follows:\nb\nb\n")
    assert h.hook_calls[-2:] == [("Edit", {
        "file_path": "/tmp/app/f.txt", "old_string": "a", "new_string": "b", "replace_all": True,
        "exact": True,
    }), ("Bash", rest)]
    # A line holding the pattern twice has no exact Edit: a multi-line sed is refused as
    # before, a plain one runs in the container as the shell command it is.
    env.Bash("printf 'aa\\n' > /tmp/app/f.txt", "seed", 30, 500)
    multiline = "cd /tmp/app && sed -i 's/a/b\\n/' f.txt && cat f.txt"
    out = h.call("Bash", command=multiline, description="sed")
    assert out == rewrite_guard(multiline) and _cat(env, "/tmp/app/f.txt") == "aa\n", out
    assert h.call("Bash", command=sed, description="sed") == "ba\n"
    assert h.tool_calls()[-2:] == ["Bash", "Bash"]
    # A sed -i on a symlink or a hard link is no Edit either (A08 astra-6 / fable-5): sed
    # replaces the name and leaves the other one as it was; an Edit writes through it.
    env.Bash(
        "printf 'old\\n' > /tmp/app/real.txt && ln -sf real.txt /tmp/app/link.txt"
        " && ln -f /tmp/app/real.txt /tmp/app/twin.txt", "seed", 30, 500,
    )
    for name in ("link.txt", "twin.txt"):
        linked = f"cd /tmp/app && sed -i 's/old/new/' {name} && cat {name} real.txt"
        assert h.call("Bash", command=linked, description="sed") == "new\nold\n", name
        assert h.tool_calls()[-1] == "Bash"
    # A cd to a missing directory is refused up front; a refused Write leaves the rest unmade.
    missing = "cd /tmp/nowhere && cat > notes.md <<'EOF' && cat notes.md\nx\nEOF"
    out = h.call("Bash", command=missing, description="cd, rewrite")
    assert out == CD_VERDICT.format(cwd="/tmp/nowhere"), out
    assert h.tool_calls()[-1] == "Bash" and _cat(env, "/tmp/app/notes.md") == "hello\n"
    assert env.Bash("test -e /tmp/nowhere", "check", 30, 500) == "[exit code: 1]"
    # A Write that failed in the container leaves the rest unmade as well (S11 re-fix):
    # its result was the failed command's output as it was, and only a result opening
    # with "Error" tells the agent that the call did not go through.
    env.Bash("printf x > /tmp/app/blocker", "seed", 30, 500)
    failing = "cat > /tmp/app/blocker/log.txt <<'EOF' && echo ran > /tmp/app/ran\nhello\nEOF"
    out = h.call("Bash", command=failing, description="write, mark")
    rest = _rest("echo ran > /tmp/app/ran", "write, mark")
    assert "whose result follows:\nError: " in out and "[exit code: 1]" in out, out
    assert out.endswith(
        f"{call_text('Bash', rest)} was not made: the Write did not go through."
    ), out
    assert h.tool_calls()[-2:] == ["Bash", "Write"]
    assert env.Bash("test -e /tmp/app/ran", "check", 30, 500) == "[exit code: 1]"
    h.verdict = refuse("not now")
    out = h.call("Bash", command=command, description="rewrite, show")
    rest = _rest("cd /tmp/app && cat notes.md && pwd", "rewrite, show")
    assert out.endswith(
        "REFUSED: the Write call did NOT run and had no effect. not now\n"
        f"{call_text('Bash', rest)} was not made: the Write did not go through."
    ), out
    assert h.tool_calls()[-2:] == ["Bash", "Write"]

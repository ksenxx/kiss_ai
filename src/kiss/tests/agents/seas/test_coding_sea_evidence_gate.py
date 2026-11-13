# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the coding SEA's evidence gate and exec bridge.

The gate (:meth:`ContainerHarness.finish_verdict`) runs the agent's own
verifier in the trial container (or reuses the agent's own passing run of
it), asks a reviewer model once with a deadline, refuses a round past the
first only while the verifier fails, and holds the first edit to an
independent reading of the statement; the exec bridge
(:mod:`kiss.agents.sorcar.exec_bridge`,
:mod:`kiss.benchmarks.terminal_bench.exec_bridge_server`) runs the same
tools on an environment that is not a local container.  Container tests
start a real container with the Docker SDK and skip without Docker; the
bridge tests use a real local environment (``asyncio`` subprocesses).  The
review test makes one real model call.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import html
import http.client
import itertools
import json
import os
import shlex
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from kiss.agents.seas.coding import coding_sea
from kiss.agents.sorcar import exec_bridge
from kiss.agents.sorcar.agent_dispatch import PIN_TASK_MODEL_ENV
from kiss.agents.sorcar.container_jobs import ContainerJobs
from kiss.agents.sorcar.decide_tool import decisions_tool_available
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.benchmarks.terminal_bench.exec_bridge_server import SERVER_THREAD_NAME, ExecBridgeServer
from kiss.core import stop_signal
from kiss.core.event_sink import bind_printer
from kiss.core.kiss_error import KISSError
from kiss.core.models import model_families
from kiss.core.models.model_families import other_family_model
from kiss.core.models.model_info import MODEL_INFO, get_fast_model
from kiss.core.tool_verdict import ALLOW, allow_with_note, refuse
from kiss.server.json_printer import JsonPrinter
from kiss.tests.agents.seas.test_coding_sea import _live_container
from kiss.tests.agents.sorcar.local_model_server import MODEL as SCRIPTED_MODEL
from kiss.tests.agents.sorcar.local_model_server import (
    ScriptedHandler,
    serve,
    text_body,
    tool_call_body,
)
from kiss.tests.conftest import posix_only, wait_for

#: The tests that run commands on :class:`LocalEnvironment` (``sh`` on this host).
local_shell = posix_only("the bridged local environment runs sh commands")
#: The tagged-tree kill (:func:`kiss.agents.sorcar.docker_manager.kill_tagged_tree_script`)
#: finds the tree through ``/proc/<pid>/environ`` and :func:`_dead` reads
#: ``/proc/<pid>/stat``: the environments the bridge serves are Linux sandboxes.
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="the tagged-tree kill scans /proc")
#: The file tools' store scripts use ``mv -T`` and ``stat -c`` of the GNU/busybox
#: coreutils of the Linux environments they run in.
gnu_coreutils_only = pytest.mark.skipif(
    sys.platform == "darwin", reason="the store scripts need GNU/busybox mv -T and stat -c"
)

VERIFIER = coding_sea.VERIFIER_PATH


def _write_in(container: Any, path: str, text: str) -> None:
    """Write *text* to *path* in *container* (the agent's Write, without the agent)."""
    quoted = text.replace("'", "'\\''")
    assert container.exec_run(["sh", "-c", f"printf '%s' '{quoted}' > {path}"]).exit_code == 0


def _exec_in(container: Any, command: str, description: str, timeout_seconds: float = 30,
             max_output_chars: int = 4000) -> str:
    """The ``bash_fn`` of a :class:`ContainerJobs`: run *command* in *container*."""
    return str(container.exec_run(["sh", "-c", command]).output.decode())


_CONFIG_IDS = itertools.count()
"""Numbers the harness configs, so two harnesses of one test keep separate files."""


def _harness(tmp_path: Path, **extra: Any) -> coding_sea.ContainerHarness:
    config = tmp_path / f"config-{next(_CONFIG_IDS)}.json"
    config.write_text(json.dumps({
        "workdir": "/app", "prompt": "Create /app/out.txt containing exactly `hello`.",
        "model": "m", "trajectory": str(tmp_path / "trajectory.jsonl"), "finish_gate": True,
        **extra,
    }))
    return coding_sea.ContainerHarness(str(config))


def _events(tmp_path: Path, kind: str, name: str = "trajectory.jsonl") -> list[dict[str, Any]]:
    lines = (tmp_path / name).read_text().splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == kind]


class _ExpiringHandler(ScriptedHandler):
    """Answer ``script[0]`` until request ``refuse_at``, then HTTP 401 (an expired credential).

    A request for a model other than :data:`SCRIPTED_MODEL` (a retry the
    gate should not send) gets ``script[-1]``, so that no client falls
    back to a real provider.
    """

    refuse_at: int = 0

    def do_POST(self) -> None:  # noqa: N802
        cls = type(self)
        with cls.lock:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            cls.requests.append(request)
            number = len(cls.requests)
        if request.get("model") != SCRIPTED_MODEL:
            status, body = 200, cls.script[-1]
        elif number < cls.refuse_at:
            status, body = 200, cls.script[0]
        else:
            status = 401
            body = b'{"error": {"message": "credential expired", "type": "authentication_error"}}'
        body = body.replace(b'"call_1"', f'"call_{number}"'.encode())
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


#: A scripted body :class:`_StallingHandler` never sends: the request hangs.
STALL = b"stall"


class _StallingHandler(ScriptedHandler):
    """Answer the scripted bodies, but hold a request scripted as :data:`STALL` until released.

    The held request is answered by nothing: the handler returns once
    :attr:`release` is set (the test's cleanup), after the client gave up
    (or after ten seconds: its daemon thread outlives ``server_close``).
    A subclass's :attr:`stop` is set once the request is recorded, not at
    its headers: the body is still on its way then, and a client that
    gives the request up on the stop would leave nothing to record.
    """

    release = threading.Event()
    stop: threading.Event | None = None

    def do_POST(self) -> None:  # noqa: N802
        cls = type(self)
        with cls.lock:
            cls.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = cls.script[min(len(cls.requests), len(cls.script)) - 1]
        if cls.stop is not None:
            cls.stop.set()
        if body == STALL:
            cls.release.wait(10)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_gate_runs_the_agents_own_verifier_in_the_container(tmp_path: Path) -> None:
    """finish(success=True) is refused until /tmp/kiss-verify.sh exists and exits 0."""
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=3)
        harness.on_llm_call([])  # the first model call takes the file baseline
        done = {"success": True, "summary_in_html": "<p>Done: out.txt written and checked.</p>"}
        # no clock, no verifier yet: the first success claim is checked at once
        missing = refuse(coding_sea.gate_refusal(
            1, 3, [coding_sea.VERIFIER_MISSING_VERDICT.format(path=VERIFIER)]))
        assert harness.on_tool_call("finish", done) == missing
        # a failing verifier: its exit code and output, stderr included, come back
        _write_in(live, VERIFIER, "echo checking out.txt; echo 'Traceback: detail' >&2; "
                                  "grep -q goodbye /app/out.txt || "
                                  "{ echo 'FAIL: out.txt lacks goodbye'; exit 3; }")
        failed = harness.on_tool_call("finish", done)
        assert not failed.allowed and "exited 3" in failed.text
        assert "FAIL: out.txt lacks goodbye" in failed.text and "checking out.txt" in failed.text
        assert "Traceback: detail" in failed.text
        # a verifier slower than its timeout: refused, and its whole process tree
        # (here a background subshell, whose pid it records) is killed in the
        # container: once the subshell is dead with /tmp/late absent, the write due
        # 8 s after the start can never land; a surviving subshell writes it first
        slow_script = "(sleep 8; echo post > /tmp/late) & echo $! > /tmp/slow.pid; wait"
        _write_in(live, VERIFIER, slow_script)
        slow = harness.on_tool_call("finish", done)
        assert not slow.allowed and "did not finish within 3 seconds" in slow.text
        assert wait_for(partial(_dead_in, live, "/tmp/slow.pid"), 10)
        assert live.exec_run(["test", "-e", "/tmp/late"]).exit_code != 0
        # the budget of refusals is spent: the run is let go even though the verifier
        # fails, but a finish let through that way is not a verified one (no checkpoint)
        assert harness.on_tool_call("finish", done) == ALLOW
        assert harness.checkpoints == [] and not harness.checks_passed
        runs = _events(tmp_path, "verifier")
        assert [r.get("result", r.get("exit_code")) for r in runs] == ["missing", 3, None]
        assert harness.last_verifier_run[0] == slow_script
        assert harness.last_verifier_run[1] is None
        # file reads stay stdout-only
        assert harness.container_run(["sh", "-c", "echo out; echo err >&2"]) == (0, b"out\n")

        # a passing verifier on a fresh harness: the work tree is checkpointed as
        # "verified" and the finish passes whatever the summary says (a hedge or
        # an open reading is the reviewer's input, and no reviewer is configured)
        passing = _harness(tmp_path, container=live.id, verifier_timeout=10)
        passing.on_llm_call([])
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        hedged: dict[str, Any] = {"success": True,
                                  "summary_in_html": "<p>Should work; not verified. If the "
                                                     "grader wants more, that is a judgement "
                                                     "call.</p>"}
        assert passing.on_tool_call("finish", hedged) == ALLOW
        assert "Delivered after" not in hedged["summary_in_html"]
        assert passing.last_verifier_run[1] == 0 and passing.gate_rounds == 0
        assert passing.checkpoints == ["verified"]
        archive = f"{coding_sea.CHECKPOINT_DIR}/verified.tar"
        assert live.exec_run(["test", "-f", archive]).exit_code == 0
    finally:
        live.remove(force=True)


def test_checkpoints_and_the_best_snapshot_submitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """checkpoint/rollback restore the workdir; a run ending badly gets its verified tree back."""
    live = _live_container("python:3.11-slim", "mkdir -p /app/sub && printf hello > /app/out.txt "
                                               "&& echo keep > /app/sub/k.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "container": live.id, "workdir": "/app", "prompt": "p", "model": "m",
            "trajectory": str(tmp_path / "trajectory.jsonl"), "finish_gate": True,
            "verifier_timeout": 10,
        }))
        harness = coding_sea.ContainerHarness(str(config))
        harness.on_llm_call([])
        assert harness.checkpoint("bad name!").startswith("Error: name must be")
        assert harness.checkpoint("verified").startswith("Error: 'verified' is the harness")
        assert harness.rollback("nope") == "Error: unknown checkpoint 'nope'. Known: none."
        assert harness.checkpoint("v1") == "Saved checkpoint 'v1' of /app."
        assert live.exec_run(["sh", "-c", "printf changed > /app/out.txt && rm /app/sub/k.txt "
                                          "&& echo new > /app/extra.txt"]).exit_code == 0
        assert harness.rollback("v1") == "Restored /app from checkpoint 'v1'."
        assert harness.changes == 1  # a rollback changes the tree (the verifier reuse counts it)
        listing = live.exec_run(["sh", "-c", "cat /app/out.txt; cat /app/sub/k.txt; ls /app"])
        assert listing.output == b"hellokeep\nout.txt\nsub\n"
        assert harness.rollback("nope") == "Error: unknown checkpoint 'nope'. Known: v1."
        # the verified checkpoint: taken when both verifiers pass; a run ending
        # without an accepted finish is restored to it when the tree then fails
        # the verifier
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        done = {"success": True, "summary_in_html": "<p>untested</p>"}
        assert harness.on_tool_call("finish", done) == ALLOW
        assert harness.checkpoints == ["v1", "verified"]
        assert coding_sea.submit_best_checkpoint(config) == "kept"  # still passes
        assert live.exec_run(["sh", "-c", "printf broken > /app/out.txt"]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        submits = _events(tmp_path, "submit")
        assert [s["restored"] for s in submits] == [False, True]
        saved_verifier = harness.saved_check_path("verified", VERIFIER)
        assert submits[1]["failed_check"] == str(saved_verifier)
        # the checkpoint keeps its own copy of the checks it passed on the host, out
        # of the container's reach: a verifier the agent deleted or rewrote to pass
        # later does not stop the restore
        assert saved_verifier.parent == tmp_path and saved_verifier.read_text().startswith("test ")
        blind_copy = harness.saved_check_path("verified", coding_sea.BLIND_VERIFIER_PATH)
        assert blind_copy.read_text() == ""
        wreck = f"rm {VERIFIER}; printf broken > /app/out.txt"
        assert live.exec_run(["sh", "-c", wreck]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        lenient = f"printf 'exit 0' > {VERIFIER}; printf broken > /app/out.txt"
        assert live.exec_run(["sh", "-c", lenient]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        # the archive itself lives in the container: one replaced with a broken tree
        # (or deleted) fails the digest recorded at the save and restores nothing
        archive = f"{coding_sea.CHECKPOINT_DIR}/verified.tar"
        digest = _events(tmp_path, "checkpoint")[-1]["sha256"]
        assert len(digest) == 64 and harness.checkpoint_digests["verified"] == digest
        forge = (f"printf broken > /app/out.txt && tar -C /app -cf {archive} . && "
                 f"printf broken2 > /app/out.txt")
        assert live.exec_run(["sh", "-c", forge]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "kept"
        assert "archive altered since it was saved" in _events(tmp_path, "submit")[-1]["error"]
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"broken2"
        assert harness.rollback("verified").startswith("Error: checkpoint 'verified' not restored")
        assert live.exec_run(["rm", archive]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "kept"
        assert "archive missing" in _events(tmp_path, "submit")[-1]["error"]
        # a new verified checkpoint replaces the archive and its digest
        assert live.exec_run(["sh", "-c", "printf hello > /app/out.txt"]).exit_code == 0
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        plain = {"success": True, "summary_in_html": "<p>Done: out.txt written and checked.</p>"}
        # a host copy that cannot be written (here: a directory in its place) is
        # logged and does not end the trial or the checkpoint
        blocked_copy = harness.saved_check_path("verified", coding_sea.BLIND_VERIFIER_PATH)
        blocked_copy.unlink()
        blocked_copy.mkdir()
        assert harness.on_tool_call("finish", plain) == ALLOW
        blocked_copy.rmdir()
        assert _events(tmp_path, "hook_error")[-1]["hook"] == "checkpoint"
        renewed = _events(tmp_path, "checkpoint")[-1]
        assert renewed["verified"] and harness.checkpoint_digests["verified"] == renewed["sha256"]
        assert live.exec_run(["sh", "-c", "printf broken > /app/out.txt"]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        # a background job still writing into the tree is killed before the restore
        jobs = ContainerJobs(partial(_exec_in, live))
        job_id, started = jobs.start("while true; do printf broken > /app/out.txt; sleep 0.1; done")
        assert job_id is not None, started
        assert wait_for(partial(_holds, live, "/app/out.txt", b"broken"), 10)
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert _events(tmp_path, "submit")[-1]["killed_jobs"] == [job_id]
        assert live.exec_run(["sh", "-c", "sleep 1; cat /app/out.txt"]).output == b"hello"
        assert "killed" in jobs.bash_job(job_id, action="tail")
        # a job whose shell exited but left a writer behind in the background is
        # killed through its process group too (the .exit file is no evidence of quiet)
        orphaning, started = jobs.start("(while true; do printf broken > /app/out.txt; sleep 0.1; "
                                        "done) & echo started")
        assert orphaning is not None, started
        assert "status: exited 0" in jobs.bash_job(orphaning, action="wait", timeout_seconds=10)
        assert wait_for(partial(_holds, live, "/app/out.txt", b"broken"), 10)
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert orphaning in _events(tmp_path, "submit")[-1]["killed_jobs"]
        assert live.exec_run(["sh", "-c", "sleep 1; cat /app/out.txt"]).output == b"hello"
        # a failing tree after the refusals are spent does not replace the verified checkpoint
        spent = coding_sea.ContainerHarness(str(config))
        spent.on_llm_call([])
        _write_in(live, VERIFIER, "exit 1")
        assert live.exec_run(["sh", "-c", "printf broken > /app/out.txt"]).exit_code == 0
        done = {"success": True, "summary_in_html": "<p>Done: out.txt written and checked.</p>"}
        for _ in range(coding_sea.MAX_VERIFIER_REFUSALS):
            assert not spent.on_tool_call("finish", done).allowed
        before = spent.trajectory_path.stat().st_size
        assert not coding_sea.ended_unchecked(spent.trajectory_path)
        assert spent.on_tool_call("finish", done) == ALLOW
        assert spent.checkpoints == []
        # ... but the finish counts as unchecked: the runner records a failure
        # and submits the verified tree, so repeating finish bought nothing
        assert _events(tmp_path, "unverified_finish")[-1]["refusals"] == 3
        assert coding_sea.ended_unchecked(spent.trajectory_path, before)
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        # without a verified checkpoint nothing is touched
        fresh = tmp_path / "config2.json"
        fresh.write_text(json.dumps({
            "container": live.id, "workdir": "/app", "prompt": "p", "model": "m",
            "trajectory": str(tmp_path / "t2.jsonl"), "finish_gate": True,
        }))
        assert coding_sea.submit_best_checkpoint(fresh) == "none"
        # a workdir too large to archive is reported, not snapshotted
        monkeypatch.setattr(coding_sea, "MAX_CHECKPOINT_MB", 0)
        too_large = "Error: checkpoint 'big' not saved: too large"
        assert harness.checkpoint("big").startswith(too_large)
        # (a checkpoint whose archive cannot be digested right after it was written,
        # `save_checkpoint`'s "no digest" branch, needs the Docker daemon to fail
        # between the tar and the copy, or a bridged host without sha256sum: not
        # reachable here)
        # the root is never emptied
        harness.workdir = "/"
        assert harness.rollback("v1").startswith("Error: the working directory is the root")
    finally:
        live.remove(force=True)


def test_heredoc_meets_the_ledger_gate_like_an_edit(tmp_path: Path) -> None:
    """A here-document writing under the workdir is an edit: refused until the ledger exists.

    Whether it rewrites an existing file is the ``Bash`` tool's business
    (it applies such a rewrite through the agent's own Write/Edit): the
    harness refuses nothing on that account.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id)
        new = {"command": "cat > /app/new.txt <<'EOF'\nx\nEOF"}
        # a heredoc creating a file under the workdir is an edit: the ledger gate applies
        ledger = refuse(coding_sea.LEDGER_VERDICT.format(workdir="/app",
                                                         ledger=coding_sea.LEDGER_PATH))
        assert harness.on_tool_call("Bash", new) == ledger
        continued = {"command": "cat <<'EOF' \\\n> /app/new.txt\nx\nEOF"}
        assert harness.on_tool_call("Bash", continued) == ledger
        outside = {"command": "cat > /tmp/notes.txt <<'EOF'\nx\nEOF"}
        assert harness.on_tool_call("Bash", outside) == ALLOW
        _write_in(live, coding_sea.LEDGER_PATH, "- out.txt: exactly hello")
        assert harness.on_tool_call("Bash", new) == ALLOW
        relative = {"command": "cat <<EOF | tee -a sub/other.txt\nx\nEOF"}
        assert harness.on_tool_call("Bash", relative) == ALLOW
        for command in ("cat > /app/out.txt <<'EOF'\nx\nEOF", "cat <<EOF >> out.txt\nx\nEOF",
                        "cat <<EOF | tee /app/out.txt\nx\nEOF",
                        "cat <<EOF | tee -a out.txt\nx\nEOF"):
            assert harness.on_tool_call("Bash", {"command": command}) == ALLOW, command
    finally:
        live.remove(force=True)


def test_the_independent_reading_gates_the_first_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first edit waits for the ledger to settle the sentences another model reads differently.

    Once the ledger exists, the ``reading_check`` model (scripted here)
    reads the statement and the ledger once, at the provider's default
    effort; its numbered items refuse the edit until the ledger's text
    changes, at most ``MAX_READING_REFUSALS`` times for one ledger;
    ``NONE`` or a failed call refuses nothing.  The call is logged as a
    ``reading_check`` event the trial runner charges and as the
    ``reading`` gate stage.  ``other`` names a model of another family
    than the trial's and turns the reading off when none is configured;
    the prompt mentions the reading only when a model is set.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        items = ('1. "containing exactly `hello`": the ledger allows any greeting; the file '
                 'would then hold goodbye.')
        refused = refuse(coding_sea.READING_VERDICT.format(items=items, workdir="/app",
                                                           ledger=coding_sea.LEDGER_PATH))
        edit = {"file_path": "/app/out.txt", "old_string": "hello", "new_string": "goodbye"}
        with serve([text_body(items)]) as (url, requests):
            overrides = {"base_url": url, "api_key": "local", "reasoning_effort": "high"}
            harness = _harness(tmp_path, container=live.id, reading_check=SCRIPTED_MODEL,
                               model_config=overrides)
            harness.on_llm_call([])
            assert coding_sea.READING_NOTE in harness.system_prompt()
            # no ledger yet: the ledger refusal comes first, and no reading is asked for
            assert harness.on_tool_call("Edit", edit) == refuse(coding_sea.LEDGER_VERDICT.format(
                workdir="/app", ledger=coding_sea.LEDGER_PATH))
            assert not requests
            ledger = "- 'containing exactly hello': any greeting will do"
            _write_in(live, coding_sea.LEDGER_PATH, ledger)
            # the ledger exists: one call with the statement and the ledger, at the
            # default effort; its item refuses the edit
            assert harness.on_tool_call("Edit", edit) == refused
            assert len(requests) == 1 and "reasoning_effort" not in requests[0]
            assert requests[0]["messages"] == [{
                "role": "user", "content": coding_sea.READING_CHECK_PROMPT.format(
                    task=harness.task_prompt, ledger=ledger)}]
            assert harness.reading_items == [items] and harness.reading_refusals == 1
            checked = _events(tmp_path, "reading_check")[-1]
            assert (checked["model"], checked["items"], checked["error"]) == (
                SCRIPTED_MODEL, [items], "")
            assert checked["cost_usd"] > 0 and checked["tokens"] > 0 and checked["answer"] == items
            stage = _events(tmp_path, "gate_stage")[-1]
            assert (stage["stage"], stage["outcome"], stage["items"], stage["model"]) == (
                "reading", "fail", 1, SCRIPTED_MODEL)
            assert stage["cost_usd"] == checked["cost_usd"]
            assert coding_sea.review_usage(tmp_path / "trajectory.jsonl") == (
                checked["cost_usd"], checked["tokens"])
            # the ledger changed (the agent settled the item): the edit goes through,
            # with no second call, and so does every later edit
            _write_in(live, coding_sea.LEDGER_PATH, ledger + "\n- settled: exactly hello")
            assert harness.on_tool_call("Edit", edit) == ALLOW
            assert harness.on_tool_call("Write", {"file_path": "b.txt", "content": ""}) == ALLOW
            assert len(requests) == 1 and harness.reading_items == []
            assert harness.reading_refusals == 1
            # the same ledger retried: refused MAX_READING_REFUSALS times, then let through
            retried = _harness(tmp_path, container=live.id, reading_check=SCRIPTED_MODEL,
                               model_config=overrides, trajectory=str(tmp_path / "retried.jsonl"))
            retried.on_llm_call([])
            for _ in range(coding_sea.MAX_READING_REFUSALS):
                assert retried.on_tool_call("Edit", edit) == refused
            assert retried.on_tool_call("Edit", edit) == ALLOW
            assert retried.reading_refusals == 2 and retried.reading_items == []
            assert len(requests) == 2
        # the ledger written with Write: the reading is made at once, from the written text,
        # and its items end the Write's result; the first edit under the workdir is then
        # held to them (the ledger in the container still reads the same) until the
        # ledger changes.  A second Write of the ledger asks for no second reading.
        with serve([text_body(items)]) as (url, requests):
            noted = _harness(tmp_path, container=live.id, reading_check=SCRIPTED_MODEL,
                             model_config={"base_url": url, "api_key": "local"},
                             trajectory=str(tmp_path / "noted.jsonl"))
            noted.on_llm_call([])
            written = {"file_path": coding_sea.LEDGER_PATH, "content": ledger}
            assert noted.on_tool_call("Write", written) == allow_with_note(
                coding_sea.READING_NOTE_RESULT.format(items=items, workdir="/app",
                                                      ledger=coding_sea.LEDGER_PATH))
            assert len(requests) == 1 and requests[0]["messages"][0]["content"] == (
                coding_sea.READING_CHECK_PROMPT.format(task=noted.task_prompt, ledger=ledger))
            # a benefit the statement claims is motivation, as in the final review's brief
            assert "is the user's motivation, not a requirement" in " ".join(
                requests[0]["messages"][0]["content"].split())
            assert noted.reading_items == [items] and noted.reading_ledger == ledger
            assert _events(tmp_path, "gate_stage", "noted.jsonl")[-1]["stage"] == "reading"
            _write_in(live, coding_sea.LEDGER_PATH, ledger)
            assert noted.on_tool_call("Edit", edit) == refused and noted.reading_refusals == 1
            assert noted.on_tool_call("Write", written) == ALLOW and len(requests) == 1
            _write_in(live, coding_sea.LEDGER_PATH, ledger + "\n- settled: exactly hello")
            assert noted.on_tool_call("Edit", edit) == ALLOW and noted.reading_items == []
            # the reader agrees with a ledger written with Write: the result carries no note
        with serve([text_body("NONE")]) as (url, requests):
            agreed = _harness(tmp_path, container=live.id, reading_check=SCRIPTED_MODEL,
                              model_config={"base_url": url, "api_key": "local"},
                              trajectory=str(tmp_path / "agreed.jsonl"))
            agreed.on_llm_call([])
            assert agreed.on_tool_call("Write", written) == ALLOW and len(requests) == 1
            assert agreed.on_tool_call("Edit", edit) == ALLOW and len(requests) == 1
        # NONE: nothing refused, the stage passes
        with serve([text_body("NONE")]) as (url, requests):
            agreeing = _harness(tmp_path, container=live.id, reading_check=SCRIPTED_MODEL,
                                model_config={"base_url": url, "api_key": "local"},
                                trajectory=str(tmp_path / "none.jsonl"))
            agreeing.on_llm_call([])
            assert agreeing.on_tool_call("Edit", edit) == ALLOW and len(requests) == 1
            checked = _events(tmp_path, "reading_check", "none.jsonl")[-1]
            assert checked["items"] == [] and checked["answer"] == "NONE"
            assert _events(tmp_path, "gate_stage", "none.jsonl")[-1]["outcome"] == "pass"
        # a call that fails (no such model): nothing refused, the stage is an error
        failing = _harness(tmp_path, container=live.id, reading_check="no-such-model-xyz",
                           trajectory=str(tmp_path / "failed.jsonl"))
        failing.on_llm_call([])
        assert failing.on_tool_call("Edit", edit) == ALLOW
        checked = _events(tmp_path, "reading_check", "failed.jsonl")[-1]
        assert checked["error"] and checked["items"] == [] and checked["cost_usd"] == 0
        assert _events(tmp_path, "gate_stage", "failed.jsonl")[-1]["outcome"] == "error"
        # `other`: a model of another family than the trial's, when one is configured
        other = other_family_model("m")
        if other is not None:
            assert _harness(tmp_path, container=live.id, reading_check="other").reading_check \
                == other
        monkeypatch.setattr(model_families, "SECOND_OPINION_MODELS", ())
        off = _harness(tmp_path, container=live.id, reading_check="other",
                       trajectory=str(tmp_path / "off.jsonl"))
        assert off.reading_check == "" and coding_sea.READING_NOTE not in off.system_prompt()
        off.on_llm_call([])
        assert off.on_tool_call("Edit", edit) == ALLOW  # the ledger exists; nothing is read
        assert not _events(tmp_path, "reading_check", "off.jsonl")
    finally:
        live.remove(force=True)


@pytest.mark.live_api
def test_gate_reviews_the_work_once_and_holds_the_findings(tmp_path: Path) -> None:
    """One real review per trial; its findings are judged at the next finish, never re-reviewed.

    Round 1: the reviewer (a real model call) finds the file wrong and the
    finish is refused.  The next finish neither changes a file nor quotes
    evidence: the decisions model finds the findings unanswered, but the
    scripts pass and the first refusal already bought its round, so the
    work is delivered with the findings as caveats and no second review.
    A reviewer whose model call fails refuses nothing, and past
    ``MAX_FAILED_REVIEWS`` failures no review is asked for.
    """
    if not decisions_tool_available():
        pytest.skip("no decisions model configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, reviewer=get_fast_model())
        harness.on_llm_call([])
        assert harness.max_reviews == coding_sea.MAX_REVIEWS == 1
        # the agent writes something else than the task asks for
        assert live.exec_run(["sh", "-c", "printf goodbye > /app/out.txt"]).exit_code == 0
        _write_in(live, VERIFIER, "test -s /app/out.txt")
        _write_in(live, coding_sea.LEDGER_PATH, "- 'containing exactly hello': any greeting")
        _turn(harness, [("Bash", {"command": "test -s /app/out.txt"}),
                        ("Bash", {"command": "cat /app/out.txt"})], ["", "goodbye"])
        done = {"success": True, "evidence": json.dumps(["test -s /app/out.txt"]),
                "summary_in_html": "<p>Wrote /app/out.txt with the requested content; "
                                   "`test -s /app/out.txt` passes.</p>"}
        review = harness.on_tool_call("finish", done)
        assert not review.allowed and "Not finished: an independent review" in review.text
        assert review.text.startswith("Finish refused (round 1 of up to 3;")
        assert "out.txt" in review.text and "Answer every numbered finding" in review.text
        logged = _events(tmp_path, "review")
        assert len(logged) == 1 and logged[0]["model"] == get_fast_model()
        assert logged[0]["cost_usd"] > 0 and logged[0]["tokens"] > 0 and logged[0]["findings"]
        assert logged[0]["round"] == 1 and not logged[0]["failed"] and not logged[0]["error"]
        assert logged[0]["last_call"]["stop_reason"] and logged[0]["last_call"]["output_tokens"] > 0
        assert "VERDICT: NOT DONE" in logged[0]["answer"].upper()
        assert harness.open_findings and harness.open_findings[0].startswith("1")
        assert coding_sea.review_usage(tmp_path / "trajectory.jsonl") == (
            logged[0]["cost_usd"], logged[0]["tokens"])
        # the reviewer saw the changed file, the ledger and the evidence with its exit code
        report = harness.changed_files_report()
        assert "/app/out.txt (modified)" in report and "goodbye" in report
        assert f"{VERIFIER}" not in report  # outside the workdir
        assert harness.evidence_report(done) == "- `test -s /app/out.txt` (exit 0)"
        # the one review is spent: a change would not buy another
        assert not harness.review_possible() and not harness.review_wanted()
        # no change, no evidence: the findings are unanswered (the decisions model's
        # judgement), but the scripts pass and the first refusal bought the one
        # round: delivered, the findings as caveats, no second review
        argued: dict[str, Any] = {"success": True, "summary_in_html": "<p>The reviewer "
                                  "misread the task; any greeting is fine. Done.</p>"}
        assert harness.on_tool_call("finish", argued) == ALLOW
        assert harness.gate_rounds == 1 and len(_events(tmp_path, "review")) == 1
        assert _events(tmp_path, "findings_answered")[-1]["answered"] == [False] * len(
            harness.open_findings)
        unanswered = coding_sea.UNANSWERED_VERDICT.format(
            findings="\n".join(harness.open_findings))
        assert harness.delivered_findings == [unanswered.splitlines()[0]]
        assert "Delivered after 1 refused finish rounds" in argued["summary_in_html"]
        assert html.escape(unanswered) in argued["summary_in_html"]
        delivered = _events(tmp_path, "delivered_with_findings")[-1]
        assert delivered["checks_passed"] and delivered["rounds"] == 1
        assert not _events(tmp_path, "unverified_finish")
        # a review whose model call fails (here: a reviewer model that does not
        # exist) is neither approval nor rejection: nothing is refused, the failure
        # is counted, and past MAX_FAILED_REVIEWS no review is asked for
        failing = _harness(tmp_path, container=live.id, reviewer="no-such-model-xyz")
        failing.on_llm_call([])
        for _ in range(coding_sea.MAX_FAILED_REVIEWS + 1):
            assert failing.on_tool_call("finish", dict(done)) == ALLOW
        failed = [e for e in _events(tmp_path, "review") if e["model"] == "no-such-model-xyz"]
        assert [e["failed"] for e in failed] == [True, True] and all(e["error"] for e in failed)
        assert failing.failed_reviews == 2 and failing.reviews == 0 and failing.gate_rounds == 0
        assert not failing.review_possible() and not failing.open_findings
    finally:
        live.remove(force=True)


@pytest.mark.live_api
def test_blind_verifier_is_written_from_the_statement_and_run_by_the_gate(tmp_path: Path) -> None:
    """A second agent writes the independent check into the container; the gate runs it.

    One real agentic run of the fast model with a container shell: the
    script it delivers lands at ``BLIND_VERIFIER_PATH``, is run at every
    gated finish, and a failing check refuses the finish with its output.
    """
    if get_fast_model() == "No model":
        pytest.skip("no provider credential configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf goodbye > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, blind_verifier=get_fast_model(),
                           max_budget=2.0)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "test -s /app/out.txt")  # the agent's own check is lenient
        done = {"success": True, "summary_in_html": "<p>Wrote out.txt; test -s passes.</p>"}
        verdict = harness.on_tool_call("finish", done)
        written = _events(tmp_path, "blind_verifier_written")
        assert len(written) == 1 and written[0]["model"] == get_fast_model()
        assert written[0]["cost_usd"] > 0 and written[0]["steps"] >= 1 and not written[0]["error"]
        assert written[0]["script"]
        assert live.exec_run(["test", "-s", coding_sea.BLIND_VERIFIER_PATH]).exit_code == 0
        runs = _events(tmp_path, "blind_verifier")
        assert len(runs) == 1 and runs[0]["exit_code"] is not None
        if runs[0]["exit_code"] != 0:
            assert not verdict.allowed and (
                "Not finished: an independent check written from the task statement alone"
                in verdict.text)
            assert harness.verifier_refusals == 1
        else:
            assert verdict == ALLOW
        assert harness.last_blind_run[0] == written[0]["script"]
        # the author runs once per trial: a later finish only re-runs the script
        assert live.exec_run(["sh", "-c", "printf hello > /app/out.txt"]).exit_code == 0
        again = harness.on_tool_call("finish", done)
        assert len(_events(tmp_path, "blind_verifier_written")) == 1
        assert len(_events(tmp_path, "blind_verifier")) == 2
        assert again == ALLOW or "independent check" in again.text
        cost, tokens = coding_sea.review_usage(tmp_path / "trajectory.jsonl")
        assert cost == written[0]["cost_usd"] and tokens == written[0]["tokens"]
        # an author that delivers nothing leaves no check and no verdict
        empty = _harness(tmp_path, container=live.id, blind_verifier="no-such-model-xyz")
        empty.on_llm_call([])
        assert live.exec_run(["rm", "-f", coding_sea.BLIND_VERIFIER_PATH]).exit_code == 0
        assert empty.on_tool_call("finish", done) == ALLOW
        failed = _events(tmp_path, "blind_verifier_written")[-1]
        assert failed["error"] and not failed["script"]
    finally:
        live.remove(force=True)


def test_blind_verifier_runs_from_the_host_copy(tmp_path: Path) -> None:
    """Rewriting or deleting the container's copy of the independent check changes nothing.

    The gate runs the script the author delivered to the host (``bash -c``,
    with the documented path as ``$0``) and puts the container copy back
    before each run; the verified checkpoint keeps the same text on the host.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        blind = coding_sea.BLIND_VERIFIER_PATH
        harness = _harness(tmp_path, container=live.id, blind_verifier="no-such-model-xyz",
                           verifier_timeout=10)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "test -s /app/out.txt")
        # the author already ran (no such model): seed the strict check it would have
        # delivered once its thread is done, or it overwrites the seed with ""
        assert harness.blind_script_ready() == ""
        harness.blind_script = 'echo "checking as $0"; test "$(cat /app/out.txt)" = goodbye'
        done = {"success": True, "summary_in_html": "<p>Done: out.txt written and checked.</p>"}
        # the agent silences the container copy before each finish
        _write_in(live, blind, "exit 0")
        refused = harness.on_tool_call("finish", done)
        assert not refused.allowed and f"({blind}) exited 1" in refused.text
        assert f"checking as {blind}" in refused.text and "editing or deleting" in refused.text
        assert live.exec_run(["cat", blind]).output.decode() == harness.blind_script
        assert live.exec_run(["rm", blind]).exit_code == 0
        assert not harness.on_tool_call("finish", done).allowed
        assert harness.last_blind_run[0] == harness.blind_script
        assert [r["exit_code"] for r in _events(tmp_path, "blind_verifier")] == [1, 1]
        # the work made right: both checks pass, the verified checkpoint keeps the
        # host's text of the check, and the restore after a later wreck uses it
        assert live.exec_run(["sh", "-c", "printf goodbye > /app/out.txt"]).exit_code == 0
        assert harness.on_tool_call("finish", done) == ALLOW
        saved = harness.saved_check_path("verified", blind)
        assert saved.read_text() == harness.blind_script
        config = next(tmp_path.glob("config-*.json"))
        wreck = f"printf 'exit 0' > {blind}; printf broken > /app/out.txt"
        assert live.exec_run(["sh", "-c", wreck]).exit_code == 0
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        assert _events(tmp_path, "submit")[-1]["failed_check"] == str(saved)
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"goodbye"
    finally:
        live.remove(force=True)


def test_the_gate_runs_its_checks_with_the_trials_original_shell(tmp_path: Path) -> None:
    """The checks run with a fresh copy of the bash the trial started with, not the agent's.

    The copy is taken with ``docker cp`` at the first model call and put
    back into the container, under a private name, right before every
    run: a bash the agent overwrote with ``exit 0`` passes the agent's own
    commands and nothing of the gate's, and no finish is refused or let
    through unverified for it.  The runner's re-run of the saved checks
    uses the same copy.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        done = {"success": True, "summary_in_html": "<p>Done: out.txt written and checked.</p>"}
        assert harness.on_tool_call("finish", done) == ALLOW
        baseline = _events(tmp_path, "shell_baseline")[0]
        assert baseline["path"] == harness.shell_path == "/usr/bin/bash"
        shell = tmp_path / "shell"
        assert harness.host_shell == shell and shell.is_file()
        assert baseline["sha256"] == hashlib.sha256(shell.read_bytes()).hexdigest()
        assert shell.stat().st_mode & 0o111 == 0o111
        assert harness.host_digest("/no/such/file") is None
        assert live.exec_run(["mkdir", "/tmp/empty"]).exit_code == 0
        assert harness.host_digest("/tmp/empty") is None  # no regular file in it
        assert harness.host_digest("/tmp/empty", save=tmp_path / "partial") is None
        assert not (tmp_path / "partial").exists()  # a failed copy leaves no host file
        # a copy that stalls past its deadline is killed, not waited for
        assert harness.host_digest(harness.shell_path, timeout=0.0001) is None
        # the agent replaces bash by `exit 0`: its own tools are fooled, the gate is not
        swap = ('cp "$(command -v bash)" /tmp/bash.orig && printf "#!/bin/sh\\nexit 0\\n" '
                '> "$(command -v bash)"')
        assert live.exec_run(["sh", "-c", swap]).exit_code == 0
        assert live.exec_run(["bash", "-c", "exit 7"]).exit_code == 0  # the fake shell
        assert live.exec_run(["sh", "-c", "printf broken > /app/out.txt"]).exit_code == 0
        failed = harness.on_tool_call("finish", done)
        assert not failed.allowed and f"{VERIFIER} exited 1" in failed.text
        assert _events(tmp_path, "verifier")[-1]["exit_code"] == 1
        assert live.exec_run(["sh", "-c", "ls -a / | grep -c kiss-shell"]).output.strip() == b"0"
        # the independent check runs with it too, after the container's `cat`
        # rewrote its copy (a `cat` that swapped the shell would change nothing)
        assert live.exec_run(["sh", "-c", "printf hello > /app/out.txt"]).exit_code == 0
        harness.blind_verifier, harness.blind_written, harness.blind_script = "m", True, "exit 9"
        refused = harness.on_tool_call("finish", done)
        assert not refused.allowed
        assert f"({coding_sea.BLIND_VERIFIER_PATH}) exited 9" in refused.text
        assert _events(tmp_path, "blind_verifier")[-1]["exit_code"] == 9
        assert harness.verifier_refusals == 2
        harness.blind_script = "test \"$0\" = " + coding_sea.BLIND_VERIFIER_PATH
        assert harness.on_tool_call("finish", done) == ALLOW
        assert len([e for e in _events(tmp_path, "checkpoint") if e["verified"]]) == 2
        # the runner's re-run of the saved checks, still with the fake bash in place
        assert live.exec_run(["sh", "-c", "printf broken > /app/out.txt"]).exit_code == 0
        config = next(tmp_path.glob("config-*.json"))
        assert coding_sea.submit_best_checkpoint(config) == "restored"
        submit = _events(tmp_path, "submit")[-1]
        assert submit["failed_check"] == str(harness.saved_check_path("verified", VERIFIER))
        assert live.exec_run(["cat", "/app/out.txt"]).output == b"hello"
        assert coding_sea.submit_best_checkpoint(config) == "kept"
        # the blind author's shell is the same copy: the fake bash cannot lie to it
        assert harness.run_script("exit 3", "bash", 10) == (3, b"")
        # without the host copy (the daemon's copy failed at the start, or a
        # bridged trial) the gate falls back to the container's bash, and logs it
        # (the `docker cp` that times out or finds no docker binary takes the
        # same path; neither can happen here without a fake docker)
        harness.host_shell = tmp_path / "no-such-shell"
        assert harness.run_script("exit 3", "bash", 10) == (0, b"")  # the fake shell answers
        assert _events(tmp_path, "shell_copy_failed")[-1]["path"].startswith("/.kiss-shell-")
        harness.host_shell = None
        assert harness.run_script("exit 3", "bash", 10) == (0, b"")
        assert len(_events(tmp_path, "shell_copy_failed")) == 1
        # a harness built later (the runner's) finds the host copy on disk
        assert coding_sea.ContainerHarness(str(config)).host_shell == tmp_path / "shell"
    finally:
        live.remove(force=True)


def _turn(harness: coding_sea.ContainerHarness, calls: list[tuple[str, dict[str, Any]]],
          results: list[str]) -> None:
    """Make the tool *calls* and hand the model the *results*, as one agent turn."""
    footer = "\n\nSteps: 3/100, Context: 1,000/500,000 tokens, Budget: $0.10/$6.00"
    for name, args in calls:
        harness.on_tool_call(name, args)
    harness.on_llm_call([{"role": "tool", "content": text + footer} for text in results])


def test_evidence_is_reported_with_its_exit_codes_and_refuses_nothing(tmp_path: Path) -> None:
    """finish's ``evidence`` lists commands; the reviewer gets each with its last exit code.

    The gate pairs each named command with the exit code its result showed
    (``record_exit_codes``).  A command that never ran, failed or timed
    out no longer refuses the finish, nor does a hedged summary: both go
    to the reviewer as they are, and the finish is gated on the scripts
    and the review alone.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        with serve([text_body("VERDICT: OK")]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([{"role": "user", "content": "task"}])
            _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
            good, bad = "cd /app && cat out.txt && wc -c out.txt", "grep -q goodbye /app/out.txt"
            _turn(harness, [("Bash", {"command": good}), ("Bash", {"command": bad}),
                            ("Bash", {"command": "python3 slow.py"})],
                  ["hello\n5 out.txt", "\n[exit code: 1]", "Error: command timed out after 10s"])
            assert harness.exit_codes == {good: "0", bad: "1", "python3 slow.py": "timeout"}
            # a command that never ran, one that failed, one that timed out: each is
            # listed with its exit code for the reviewer, and none refuses
            named = [good, "pytest -q", bad, "python3 slow.py"]
            claimed: dict[str, Any] = {
                "success": True, "evidence": json.dumps(named),
                "summary_in_html": "<p>Should work; if the grader wants more, that is a "
                                   "judgement call.</p>"}
            report = harness.evidence_report(claimed)
            assert report == (f"- `{good}` (exit 0)\n- `pytest -q` (exit never ran)\n"
                              f"- `{bad}` (exit 1)\n- `python3 slow.py` (exit timeout)")
            assert harness.evidence_report({}) == "(none named)"
            assert harness.on_tool_call("finish", claimed) == ALLOW
            assert "Delivered after" not in claimed["summary_in_html"]
            assert harness.gate_rounds == 0 and not _events(tmp_path, "gate_round")
            # the reviewer read the list and the summary as they were
            prompt = json.dumps(requests[0]["messages"])
            assert json.dumps(report)[1:-1] in prompt and "judgement call" in prompt
        # the verifier passed in the same round: the verified checkpoint was taken,
        # and the stages were recorded with their outcome (no blind check configured)
        assert harness.checkpoints == ["verified"]
        stages = _events(tmp_path, "gate_stage")
        assert [(s["stage"], s["outcome"]) for s in stages] == [("verifier", "pass"),
                                                                ("review", "pass")]
        assert all(s["seconds"] >= 0 for s in stages)
        # a plain string is one command, its surrounding blanks dropped
        assert harness.evidence_report({"evidence": f" {bad} "}) == f"- `{bad}` (exit 1)"
        # how the field is read: a JSON list, a list, a plain string, junk
        assert coding_sea.evidence_commands('["a", " b ", 3, ""]') == ["a", "b"]
        assert coding_sea.evidence_commands(["a", 1]) == ["a"]
        assert coding_sea.evidence_commands("[not json") == ["[not json"]
        assert coding_sea.evidence_commands('"quoted"') == ["quoted"]
        assert coding_sea.evidence_commands("   ") == []
        assert coding_sea.evidence_commands('{"a": 1}') == []
        assert coding_sea.evidence_commands(None) == []
    finally:
        live.remove(force=True)


def test_the_agents_own_passing_verifier_run_is_reused(tmp_path: Path) -> None:
    """A foreground run of the verifier that exited 0 stands for the gate's while nothing changed.

    The verifier counts its runs in the container.  After the agent's own
    run (a plain ``bash`` of it, exit 0) the gated finish runs nothing and
    records the reuse; an edit under the workdir, a writing shell command,
    a rollback or an edit of the verifier itself makes the gate run it
    again; a command that names the verifier, a run whose exit says
    nothing about it (``|| true``, ``; echo``) and a failed run do not
    stand for a passing one (the gate then reuses its own last run, the
    files and the script being those it ran on).
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([{"role": "user", "content": "task"}])
        _write_in(live, coding_sea.LEDGER_PATH, "- out.txt: exactly hello")
        script = 'echo run >> /tmp/runs; test "$(cat /app/out.txt)" = hello'
        _write_in(live, VERIFIER, script)
        done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}

        def runs() -> int:
            return int(live.exec_run(["sh", "-c", "wc -l < /tmp/runs"]).output.strip() or 0)

        def call(name: str, args: dict[str, Any], result: str = "") -> Any:
            """One agent turn: the tool call, then its result handed to the model."""
            verdict = harness.on_tool_call(name, args)
            harness.on_llm_call([{"role": "tool", "content": result}])
            return verdict

        def agent_runs(command: str, result: str = "") -> None:
            """Run *command* in the container as the agent would, its result in the turn."""
            assert live.exec_run(["bash", "-c", command]).exit_code == 0
            assert call("Bash", {"command": command}, result) == ALLOW

        # the agent's own run passed and nothing changed since: reused, not run again
        # (a command naming the verifier counts as a change, since it may rewrite the
        # script; its passing run is then credited at that count)
        agent_runs(f"bash {VERIFIER}")
        assert runs() == 1 and harness.verifier_passed_at == harness.changes == 1
        assert call("finish", dict(done)) == ALLOW
        assert runs() == 1 and harness.checkpoints == ["verified"]
        assert harness.last_verifier_run == (
            script, 0, "(passed when you ran it; nothing changed since)")
        reused = _events(tmp_path, "verifier")[-1]
        assert reused["exit_code"] == 0 and reused["reused"] is True
        stage = _events(tmp_path, "gate_stage")[-1]
        assert (stage["stage"], stage["outcome"], stage["reused"]) == ("verifier", "pass", True)
        # an edit under the workdir: the gate runs the verifier itself
        assert call("Write", {"file_path": "out.txt", "content": "hello"}) == ALLOW
        assert harness.changes == 2 and harness.verifier_passed_at == 1
        assert call("finish", dict(done)) == ALLOW
        assert runs() == 2 and "reused" not in _events(tmp_path, "verifier")[-1]
        assert "reused" not in _events(tmp_path, "gate_stage")[-1]
        # a writing shell command, then a rollback: each makes the agent's run stale
        agent_runs(f"bash {VERIFIER} && echo ok", "ok")
        assert runs() == 3 and harness.verifier_passed_at == harness.changes == 3
        agent_runs("echo note > /app/notes.txt")
        assert harness.changes == 4
        assert call("finish", dict(done)) == ALLOW and runs() == 4
        agent_runs(f"cd /app && bash {VERIFIER}")
        assert runs() == 5 and harness.verifier_passed_at == harness.changes == 5
        assert harness.checkpoint("v1") == "Saved checkpoint 'v1' of /app."
        assert harness.rollback("v1") == "Restored /app from checkpoint 'v1'."
        assert harness.changes == 6
        assert call("finish", dict(done)) == ALLOW and runs() == 6
        # an edit of the verifier itself (a new text), or a shell command naming it: stale too
        agent_runs(f"bash {VERIFIER}")
        assert runs() == 7 and harness.verifier_passed_at == harness.changes == 7
        _write_in(live, VERIFIER, script + "\n# v2")
        assert call("Write", {"file_path": VERIFIER, "content": script + "\n# v2"}) == ALLOW
        assert harness.verifier_passed_at is None and harness.changes == 8
        assert call("finish", dict(done)) == ALLOW and runs() == 8
        agent_runs(f"bash {VERIFIER}")
        assert runs() == 9 and harness.verifier_passed_at == harness.changes == 9
        agent_runs(f"cat {VERIFIER}", script)
        assert harness.verifier_passed_at is None and harness.changes == 10
        # an exit 0 that says nothing about the verifier, or a failed run, is no passing run
        # of the agent's: the gate reuses its own run of 8 instead (nothing changed since)
        agent_runs(f"bash {VERIFIER} || true")
        agent_runs(f"bash {VERIFIER}; echo done", "done")
        assert call("Bash", {"command": f"bash {VERIFIER}"}, "FAIL\n[exit code: 1]") == ALLOW
        assert runs() == 11 and harness.verifier_passed_at is None
        assert call("finish", dict(done)) == ALLOW and runs() == 11
        assert harness.last_verifier_run == (script + "\n# v2", 0, "")
        assert _events(tmp_path, "verifier")[-1]["reused"] is True
        assert ["reused" in e for e in _events(tmp_path, "verifier")] == [
            True, False, False, False, False, True]
        assert harness.last_verdict is not None and harness.last_verdict[0] == "1"
        assert harness.last_verdict[3:] == ("agent", "FAIL")
    finally:
        live.remove(force=True)


def test_round_cap_delivers_with_caveats_and_a_giveup_is_checked(tmp_path: Path) -> None:
    """Past ``max_gate_rounds`` rounds refused by the verifier the finish goes through with the
    open verdicts as caveats; a ``finish(success=False)`` runs the checks and is answered once
    when they pass."""
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10, max_gate_rounds=2)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "exit 7")
        claimed: dict[str, Any] = {"success": True, "summary_in_html": "<p>done</p>"}
        for round_number in (1, 2):
            verdict = harness.on_tool_call("finish", dict(claimed))
            assert not verdict.allowed and verdict.text.startswith(
                f"Finish refused (round {round_number} of up to 2;")
            assert "exited 7" in verdict.text
        # the third refused round would be past the cap: delivered with the verifier's
        # verdict as the caveat on the summary, as an unverified finish (the verifier
        # ran and failed again, under its own cap of three refusals)
        delivered = dict(claimed)
        assert harness.on_tool_call("finish", delivered) == ALLOW
        assert "Delivered after 2 refused finish rounds" in delivered["summary_in_html"]
        assert "exited 7" in delivered["summary_in_html"]
        assert harness.delivered_findings == [
            f"Not finished: your own verifier {VERIFIER} exited 7 (bash {VERIFIER}):"]
        record = _events(tmp_path, "delivered_with_findings")[-1]
        assert record["findings"] == harness.delivered_findings and record["rounds"] == 2
        assert record["checks_passed"] is False and not harness.checks_passed
        assert _events(tmp_path, "unverified_finish")[-1]["refusals"] == 3
        assert harness.gate_rounds == 2 and harness.checkpoints == []
        # a pause for a fresh context is not a give-up: no check runs, nothing is appended
        pause = {"success": False, "is_continue": True, "summary_in_html": "<p>so far</p>"}
        assert harness.on_tool_call("finish", pause) == ALLOW
        assert pause["summary_in_html"] == "<p>so far</p>" and not _events(tmp_path, "giveup_check")
        # a give-up: both checks run (no blind check here: "missing"), the note is
        # appended, a passing tree is checkpointed, and the first such give-up is
        # sent back once
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        giveup: dict[str, Any] = {"success": False, "summary_in_html": "<p>could not finish</p>"}
        verdict = harness.on_tool_call("finish", giveup)
        assert verdict == refuse(coding_sea.GIVEUP_RECONSIDER_VERDICT.format(
            verifier=VERIFIER, blind=coding_sea.BLIND_VERIFIER_PATH))
        assert f"verifier {VERIFIER}: pass" in giveup["summary_in_html"]
        assert f"independent check {coding_sea.BLIND_VERIFIER_PATH}: missing" in (
            giveup["summary_in_html"])
        checked = _events(tmp_path, "giveup_check")[-1]
        assert checked["states"] == {"verifier": "pass", "blind": "missing"} and checked["passed"]
        assert harness.checkpoints == ["verified"]  # the give-up's passing tree
        giveup_stages = [s for s in _events(tmp_path, "gate_stage") if s.get("giveup")]
        assert [(s["stage"], s["outcome"]) for s in giveup_stages] == [
            ("verifier", "pass"), ("blind", "missing")]
        # the second give-up ends the run, whatever the checks say
        again: dict[str, Any] = {"success": False, "summary_in_html": "<p>still no</p>"}
        assert harness.on_tool_call("finish", again) == ALLOW
        assert f"verifier {VERIFIER}: pass" in again["summary_in_html"]
        assert len([e for e in _events(tmp_path, "checkpoint") if e["verified"]]) == 2
        # a failing verifier makes the give-up note say so, with no checkpoint
        _write_in(live, VERIFIER, "exit 7")
        failing: dict[str, Any] = {"success": False, "summary_in_html": "<p>no</p>"}
        assert harness.on_tool_call("finish", failing) == ALLOW
        assert f"verifier {VERIFIER}: fail (exit 7)" in failing["summary_in_html"]
        assert len([e for e in _events(tmp_path, "checkpoint") if e["verified"]]) == 2
    finally:
        live.remove(force=True)


def test_the_verifier_cap_delivers_with_the_caveat(tmp_path: Path) -> None:
    """Past the verifier-refusal cap a finish is delivered with caveats, not as a bare success.

    Three finishes refused by a failing verifier spend the cap; the fourth
    goes through with the caveat on the summary and a
    ``delivered_with_findings`` record, as an unverified finish.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "exit 7")
        done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
        for round_number in (1, 2, 3):
            verdict = harness.on_tool_call("finish", dict(done))
            assert not verdict.allowed and f"round {round_number} of up to 3" in verdict.text
            assert "exited 7" in verdict.text
        assert harness.verifier_refusals == 3 and harness.gate_rounds == 3
        delivered = dict(done)
        assert harness.on_tool_call("finish", delivered) == ALLOW
        caveat = coding_sea.UNVERIFIED_CAVEAT.format(
            verifier=VERIFIER, blind=coding_sea.BLIND_VERIFIER_PATH, refusals=3)
        assert harness.delivered_findings == [caveat]
        assert "Delivered after 3 refused finish rounds" in delivered["summary_in_html"]
        assert html.escape(caveat) in delivered["summary_in_html"]
        record = _events(tmp_path, "delivered_with_findings")[-1]
        assert (record["findings"], record["rounds"], record["checks_passed"]) == (
            [caveat], 3, False)
        assert not harness.checks_passed and harness.checkpoints == []
        assert _events(tmp_path, "unverified_finish")[-1]["refusals"] == 3
        assert coding_sea.ended_unchecked(tmp_path / "trajectory.jsonl")
        stages = [(s["stage"], s["outcome"]) for s in _events(tmp_path, "gate_stage")]
        assert stages == [("verifier", "fail")] * 3 + [("verifier", "skipped")]
        # a later finish is delivered the same way; the delivery is not a refused round
        assert harness.on_tool_call("finish", dict(done)) == ALLOW and harness.gate_rounds == 3
    finally:
        live.remove(force=True)


@pytest.mark.live_api  # the outstanding finding is classified by the real decisions model
def test_rounds_after_the_first_are_refused_only_by_the_verifier(tmp_path: Path) -> None:
    """The first refusal buys one round whatever refused it; later ones need a failing verifier.

    A scripted reviewer finds the work wrong (round 1).  The next finish
    repeats the claim on the same files, so the review's verdict is
    reused without a model call, and the verifier fails: refused again,
    both verdicts listed.  With the verifier passing again and a new
    argument the finding is judged unanswered (the decisions model) and
    refuses no more: the finish is delivered with it as a caveat, as a
    verified finish, and the reviewer was asked once.
    """
    if not decisions_tool_available():
        pytest.skip("no decisions model configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf goodbye > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        finding = "1. out.txt must contain exactly hello: it contains goodbye"
        with serve([text_body(f"VERDICT: NOT DONE\n{finding}\n")]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            _write_in(live, VERIFIER, "test -s /app/out.txt")
            argued: dict[str, Any] = {"success": True, "summary_in_html": "<p>The reviewer "
                                      "misread the task; any greeting is fine. Done.</p>"}
            first = harness.on_tool_call("finish", dict(argued))
            assert first == refuse(coding_sea.gate_refusal(1, 3, [
                coding_sea.REVIEW_VERDICT.format(findings=finding)]))
            assert harness.open_findings == [finding]
            assert harness.checks_passed and harness.checkpoints == ["verified"]
            # the verifier fails: refused again, the review's verdict reused beside it
            # (the same claim on the same files; no decisions call either)
            _write_in(live, VERIFIER, "exit 7")
            second = harness.on_tool_call("finish", dict(argued))
            assert not second.allowed and second.text.startswith("Finish refused (round 2 of up")
            assert f"verifier {VERIFIER} exited 7" in second.text
            assert coding_sea.REVIEW_VERDICT.format(findings=finding) in second.text
            assert not harness.checks_passed and not _events(tmp_path, "findings_answered")
            reused = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "review"][-1]
            assert reused == {**reused, "outcome": "fail", "reused": True, "round": 1,
                              "model": SCRIPTED_MODEL, "seconds": 0.0}
            # the verifier passes and the claim changes: the finding, unanswered by the
            # new argument (the decisions model's judgement), is a caveat, not a refusal
            _write_in(live, VERIFIER, "test -s /app/out.txt")
            unanswered = coding_sea.UNANSWERED_VERDICT.format(findings=finding)
            delivered = {**argued, "summary_in_html": "<p>Any greeting is fine. Done.</p>"}
            assert harness.on_tool_call("finish", delivered) == ALLOW
            assert len(requests) == 1 and harness.reviews == 1
        assert harness.gate_rounds == 2 and harness.checks_passed
        assert harness.delivered_findings == [unanswered.splitlines()[0]]
        assert "Delivered after 2 refused finish rounds" in delivered["summary_in_html"]
        assert html.escape(unanswered) in delivered["summary_in_html"]
        record = _events(tmp_path, "delivered_with_findings")[-1]
        assert record["checks_passed"] is True and record["rounds"] == 2
        assert not _events(tmp_path, "unverified_finish")
        assert not coding_sea.ended_unchecked(tmp_path / "trajectory.jsonl")
        assert [e["answered"] for e in _events(tmp_path, "findings_answered")] == [[False]]
        assert len([e for e in _events(tmp_path, "checkpoint") if e["verified"]]) == 2
    finally:
        live.remove(force=True)


def test_a_giveup_whose_check_cannot_run_is_not_a_passing_one(tmp_path: Path) -> None:
    """A check that raises inside a give-up is an error, distinct from a missing check.

    The independent check here cannot be sent to the container (its text
    holds a lone surrogate, as a JSON tool call can carry), so running it
    raises instead of running: the give-up note says ``error``, the
    give-up is not a passing one (no verified checkpoint, no
    reconsideration), the stage is recorded as failed, and the gate's own
    stages read the same check as a failed stage.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                           blind_verifier="no-such-model-xyz")
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "exit 0")
        assert harness.blind_script_ready() == ""  # the real author (no such model) is done
        harness.blind_script = "exit 0 # \udcff"
        giveup: dict[str, Any] = {"success": False, "summary_in_html": "<p>Could not finish.</p>"}
        assert harness.on_tool_call("finish", giveup) == ALLOW
        checked = _events(tmp_path, "giveup_check")[-1]
        assert checked["states"]["verifier"] == "pass" and checked["passed"] is False
        assert checked["states"]["blind"].startswith("error (UnicodeEncodeError(")
        assert (f"independent check {coding_sea.BLIND_VERIFIER_PATH}: error (UnicodeEncodeError("
                in giveup["summary_in_html"])
        assert harness.checkpoints == [] and not harness.giveup_reconsidered
        assert _events(tmp_path, "hook_error")[-1]["hook"] == "gate:blind"
        stages = [(s["stage"], s["outcome"]) for s in _events(tmp_path, "gate_stage")
                  if s.get("giveup")]
        assert stages == [("verifier", "pass"), ("blind", "fail")]
        # a missing check stays a missing one: a passing verifier then passes the give-up
        harness.blind_script = ""
        again: dict[str, Any] = {"success": False, "summary_in_html": "<p>Could not finish.</p>"}
        assert not harness.on_tool_call("finish", again).allowed
        assert _events(tmp_path, "giveup_check")[-1]["states"] == {"verifier": "pass",
                                                                     "blind": "missing"}
        assert harness.checkpoints == ["verified"]
        # the gate's own stages: a check that raises is a failed stage, not a pass
        harness.blind_script = "exit 0 # \udcff"
        done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
        verdict = harness.on_tool_call("finish", done)
        assert not verdict.allowed
        assert "a check of the gate could not run (UnicodeEncodeError(" in verdict.text
        assert not harness.checks_passed and harness.checkpoints == ["verified"]
        # a check that runs out of time is a timeout, not an error
        slow = _harness(tmp_path, container=live.id, verifier_timeout=2,
                        trajectory=str(tmp_path / "slow.jsonl"))
        slow.on_llm_call([])
        _write_in(live, VERIFIER, "sleep 5")
        late: dict[str, Any] = {"success": False, "summary_in_html": "<p>Could not finish.</p>"}
        assert slow.on_tool_call("finish", late) == ALLOW
        assert _events(tmp_path, "giveup_check", "slow.jsonl")[-1]["states"] == {
            "verifier": "timeout", "blind": "missing"}
        assert f"verifier {VERIFIER}: timeout" in late["summary_in_html"]
    finally:
        live.remove(force=True)


def test_background_and_parallel_checks_are_read_for_their_exit_codes(tmp_path: Path) -> None:
    """A check run as a background job or through run_commands_parallel gets its exit code.

    A real container job (:class:`ContainerJobs`) is started and waited on
    through the tools' own reports, paired by job id; the host tool's
    report shape and a parallel report with a timed-out and a signalled
    command are read the same way.  The exit codes are what the reviewer
    sees next to the commands named as evidence.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([])
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")

        jobs = ContainerJobs(partial(_exec_in, live))
        check = 'test "$(cat /app/out.txt)" = hello'
        # a background job: its exit comes from the bash_job wait, by job id
        assert harness.on_tool_call("Bash", {"command": check, "background": True}) == ALLOW
        job_id, started = jobs.start(check)
        assert job_id is not None
        harness.on_llm_call([{"role": "tool", "content": started}])
        assert harness.exit_codes == {check: "background"}
        assert harness.background_jobs == {job_id: check}
        assert harness.evidence_report({"evidence": check}) == f"- `{check}` (exit background)"
        wait: dict[str, Any] = {"job_id": job_id, "action": "wait", "timeout_seconds": 10}
        assert harness.on_tool_call("bash_job", wait) == ALLOW
        report = jobs.bash_job(**wait)
        assert "status: exited 0" in report
        harness.on_llm_call([{"role": "tool", "content": report}])
        assert harness.exit_codes[check] == "0"
        assert harness.evidence_report({"evidence": check}) == f"- `{check}` (exit 0)"
        # the host tool's reports: a job still running settles nothing, its exit does
        failing = "grep -q goodbye /app/out.txt"
        _turn(harness, [("Bash", {"command": failing, "background": True})],
              ["Started background job j2 (pid 7).\nUse bash_job('j2', 'wait') to block."])
        _turn(harness, [("bash_job", {"job_id": "j2", "action": "tail"})],
              [f"Job j2: running (pid 7, 3s)\nCommand: {failing}\nLog: /tmp/j2.log\n"])
        assert harness.exit_codes[failing] == "background"
        _turn(harness, [("bash_job", {"job_id": "nope", "action": "wait"}),
                        ("bash_job", {"job_id": "j2", "action": "wait"})],
              ["Error: no such job nope.",
               f"Job j2: exited with code 1\nCommand: {failing}\nLog: /tmp/j2.log\n"])
        assert harness.exit_codes[failing] == "1"
        # the verifier run as a background job: its exit is the verdict the hand-off
        # and the restart budget see, but the gate never reuses it (the tree may have
        # changed while it ran)
        assert harness.last_verdict is None
        _turn(harness, [("Bash", {"command": f"bash {VERIFIER}", "background": True})],
              ["Started background job j3 (pid 8).\nUse bash_job('j3', 'wait') to block."])
        assert harness.last_verdict is None
        _turn(harness, [("bash_job", {"job_id": "j3", "action": "wait"})],
              [f"Job j3: exited with code 0\nCommand: bash {VERIFIER}\nLog: /tmp/j3.log\n"])
        assert harness.last_verdict is not None
        assert harness.last_verdict[0] == "0" and harness.last_verdict[3] == "agent"
        assert harness.verifier_passed_at is None
        # run_commands_parallel: each command gets the exit of its section
        parallel = [check, "sleep 100", "./crash", failing]
        report = ("4 commands: 1 succeeded, 2 failed, 1 timed out.\n\n"
                  f"### [1/4] exit 0 in 0.1s\n$ {check}\n(no output)\n\n"
                  "### [2/4] TIMED OUT after 5.0s\n$ sleep 100\n(no output)\n\n"
                  "### [3/4] killed by signal 9 in 0.0s\n$ ./crash\n(no output)\n\n"
                  f"### [4/4] exit 1 in 0.0s\n$ {failing}\n(no output)")
        _turn(harness, [("run_commands_parallel", {"commands": json.dumps(parallel)})], [report])
        assert harness.exit_codes[check] == "0" and harness.exit_codes["sleep 100"] == "timeout"
        assert harness.exit_codes["./crash"] == "killed by signal 9"
        assert harness.exit_codes[failing] == "1"
        assert harness.evidence_report({"evidence": json.dumps(parallel)}) == (
            f"- `{check}` (exit 0)\n- `sleep 100` (exit timeout)\n"
            f"- `./crash` (exit killed by signal 9)\n- `{failing}` (exit 1)")
        # a report cut short (the tally only) settles nothing
        _turn(harness, [("run_commands_parallel", {"commands": json.dumps([failing])})],
              ["Error: commands must be a JSON list of strings."])
        assert harness.exit_codes[failing] == "1"
        # the finish is gated on the verifier, which passes; the evidence is the reviewer's
        claimed = {"success": True, "summary_in_html": "<p>done</p>", "evidence": check}
        assert harness.on_tool_call("finish", claimed) == ALLOW
    finally:
        live.remove(force=True)


def test_a_running_jobs_output_is_not_its_exit_status(tmp_path: Path) -> None:
    """Only the status line of a ``bash_job`` report settles a background job's exit.

    A real container job prints ``child exited 0`` and keeps running; the
    wait that returns on that line reports ``status: running`` with the
    command echoed and the line in the log tail, so the job stays
    ``background`` and is no evidence until a later wait reports its
    exit.  The host tool's report is read the same way: its log tail
    settles nothing, its status line does.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([])

        jobs = ContainerJobs(partial(_exec_in, live))
        # the job prints its marker and runs on until the test releases it, so its exit
        # cannot land before the first wait has seen it running
        check = ("printf 'child exited 0\\n'; while [ ! -e /app/release ]; do sleep 0.1; done; "
                 "exit 7")
        assert harness.on_tool_call("Bash", {"command": check, "background": True}) == ALLOW
        job_id, started = jobs.start(check)
        assert job_id is not None
        harness.on_llm_call([{"role": "tool", "content": started}])
        assert harness.exit_codes[check] == "background"
        wait: dict[str, Any] = {"job_id": job_id, "action": "wait", "timeout_seconds": 10,
                                "until": "child exited"}
        assert harness.on_tool_call("bash_job", wait) == ALLOW
        report = jobs.bash_job(**wait)
        assert "status: running" in report and "child exited 0" in report
        harness.on_llm_call([{"role": "tool", "content": report}])
        assert harness.exit_codes[check] == "background"
        assert harness.evidence_report({"evidence": check}) == f"- `{check}` (exit background)"
        assert live.exec_run(["touch", "/app/release"]).exit_code == 0
        wait = {"job_id": job_id, "action": "wait", "timeout_seconds": 30}
        assert harness.on_tool_call("bash_job", wait) == ALLOW
        report = jobs.bash_job(**wait)
        assert "status: exited 7" in report
        harness.on_llm_call([{"role": "tool", "content": report}])
        assert harness.exit_codes[check] == "7"
        # the host tool: a log line and the command echo are not the status line
        _turn(harness, [("Bash", {"command": check, "background": True})],
              ["Started background job j2 (pid 7).\nUse bash_job('j2', 'wait') to block."])
        _turn(harness, [("bash_job", {"job_id": "j2", "action": "wait"})],
              [f"Job j2: running (pid 7, 3s)\nCommand: {check}\nLog: /tmp/j2.log\n"
               "--- last 50 log lines ---\nchild exited 0\nJob j2: exited with code 0\n"])
        assert harness.exit_codes[check] == "background"
        _turn(harness, [("bash_job", {"job_id": "j2", "action": "wait"})],
              [f"Job j2: exited with code 7\nCommand: {check}\nLog: /tmp/j2.log\n"
               "--- last 50 log lines ---\nchild exited 0\n"])
        assert harness.exit_codes[check] == "7"
    finally:
        live.remove(force=True)


def test_the_blind_authors_records_are_filed_under_the_task(tmp_path: Path) -> None:
    """The author thread's ``gate_stage`` record carries the task the gate's thread is bound to.

    A real :class:`JsonPrinter` bound to a task on the test's thread records
    the author's stage under that task like the stages of the gate itself,
    so the record is routed and persisted with the trial's other events.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    printer = JsonPrinter()
    printer.adopt_thread_binding("trial-7")
    printer.start_recording()
    previous = bind_printer(printer)
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                           blind_verifier="no-such-model-xyz")
        harness.on_llm_call([])  # starts the author thread (no such model: no script)
        assert harness.blind_script_ready() == ""
        _write_in(live, VERIFIER, "exit 0")
        done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
        assert harness.on_tool_call("finish", done) == ALLOW
    finally:
        bind_printer(previous)
        printer.adopt_thread_binding("")
        live.remove(force=True)
    recorded = [e for e in printer.peek_recording_for_task("trial-7") if e["type"] == "gate_stage"]
    assert [(e["stage"], e["outcome"], e["taskId"]) for e in recorded] == [
        ("blind_author", "missing", "trial-7"), ("verifier", "pass", "trial-7"),
        ("blind", "pass", "trial-7")]
    assert recorded[0]["model"] == "no-such-model-xyz" and recorded[0]["cost_usd"] == 0
    assert _events(tmp_path, "gate_stage")[0]["stage"] == "blind_author"


def test_an_author_out_of_steps_or_budget_is_retried_on_another_family(tmp_path: Path) -> None:
    """An author that spends its steps or budget without delivering is retried on another family.

    A scripted model serves the authors: the first calls ``Bash`` at every
    step until the step cap (then, on a second harness, until its budget
    is spent); the retry, routed to the same endpoint, delivers at its
    first step.  An author that fails (no such model) is not retried.
    """
    other = other_family_model(SCRIPTED_MODEL)
    if other is None:
        pytest.skip("no model of another family is configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        busy = tool_call_body("Bash", {"command": "true"}, 100)
        delivering = tool_call_body("finish", {"script": "exit 0"}, 100)
        steps = coding_sea.BLIND_VERIFIER_STEPS
        with serve([busy] * steps + [delivering]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, blind_verifier=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            assert harness.blind_script_ready() == "exit 0"
            assert len(requests) == steps + 1
        written = _events(tmp_path, "blind_verifier_written")
        assert [(w["model"], w["retry"], w["exhausted"], w["script"]) for w in written] == [
            (SCRIPTED_MODEL, False, True, ""), (other, True, False, "exit 0")]
        assert f"exceeded {steps} steps" in written[0]["error"] and written[0]["steps"] == steps
        assert harness.blind_retried
        stage = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "blind_author"][-1]
        assert (stage["outcome"], stage["retried"], stage["model"]) == ("pass", True, other)
        assert stage["cost_usd"] == pytest.approx(written[0]["cost_usd"] + written[1]["cost_usd"])
        assert stage["cost_usd"] > 0
        # the budget: a trial budget the author's share of which one scripted call uses up
        with serve([busy, delivering]) as (url, requests):
            spent = _harness(tmp_path, container=live.id, blind_verifier=SCRIPTED_MODEL,
                             model_config={"base_url": url, "api_key": "local"}, max_budget=0.0004,
                             trajectory=str(tmp_path / "budget.jsonl"))
            spent.on_llm_call([])
            assert spent.blind_script_ready() == "exit 0"
            assert len(requests) == 2
        written = _events(tmp_path, "blind_verifier_written", "budget.jsonl")
        assert [(w["retry"], w["exhausted"], w["script"], w["steps"]) for w in written] == [
            (False, True, "", 1), (True, False, "exit 0", 1)]
        assert "budget exceeded" in written[0]["error"] and spent.blind_retried
        # a run that fails is not retried
        failed = _harness(tmp_path, container=live.id, blind_verifier="no-such-model-xyz",
                          trajectory=str(tmp_path / "failed.jsonl"))
        failed.on_llm_call([])
        assert failed.blind_script_ready() == "" and not failed.blind_retried
        written = _events(tmp_path, "blind_verifier_written", "failed.jsonl")
        assert len(written) == 1 and written[0]["error"] and not written[0]["exhausted"]
    finally:
        live.remove(force=True)


def test_a_pinned_daemon_retries_the_author_on_the_same_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under the daemon's model pin an author that delivered nothing is retried on its own model.

    The scripted model serves both runs: the first calls ``Bash`` at
    every step until the step cap, the second delivers at its first
    step.  With :data:`PIN_TASK_MODEL_ENV` set the retry names the
    pinned model rather than one of another family (which is configured
    here, so without the pin it would have been chosen), every request
    the server saw names that model, and the event and the stage record
    the retry.
    """
    if other_family_model(SCRIPTED_MODEL) is None:
        pytest.skip("no model of another family is configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    try:
        busy = tool_call_body("Bash", {"command": "true"}, 100)
        delivering = tool_call_body("finish", {"script": "exit 0"}, 100)
        steps = coding_sea.BLIND_VERIFIER_STEPS
        with serve([busy] * steps + [delivering]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, blind_verifier=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            assert harness.blind_script_ready() == "exit 0"
            assert len(requests) == steps + 1
            assert {r["model"] for r in requests} == {SCRIPTED_MODEL}
        written = _events(tmp_path, "blind_verifier_written")
        assert [(w["model"], w["retry"], w["exhausted"], w["script"]) for w in written] == [
            (SCRIPTED_MODEL, False, True, ""), (SCRIPTED_MODEL, True, False, "exit 0")]
        assert harness.blind_retried
        stage = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "blind_author"][-1]
        assert (stage["outcome"], stage["retried"], stage["model"]) == (
            "pass", True, SCRIPTED_MODEL)
        assert stage["cost_usd"] == pytest.approx(written[0]["cost_usd"] + written[1]["cost_usd"])
    finally:
        live.remove(force=True)


def test_the_pinned_retry_needs_neither_a_container_nor_a_second_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pinned retry stays on the configured model whatever models and containers exist.

    ``write_blind_verifier`` is called directly (no model call of the
    engineer's, so no container check) with a container name nothing
    answers to: the author's ``Bash`` calls fail, which the scripted
    model ignores until its step cap, and the retry delivers at its first
    step.  Every request names the pinned model, the events say which
    run was the retry, and the stage carries the model and the total
    cost.  The test needs neither Docker nor a second configured family.
    """
    monkeypatch.setenv(PIN_TASK_MODEL_ENV, "1")
    busy = tool_call_body("Bash", {"command": "true"}, 100)
    delivering = tool_call_body("finish", {"script": "exit 0"}, 100)
    steps = coding_sea.BLIND_VERIFIER_STEPS
    with serve([busy] * steps + [delivering]) as (url, requests):
        harness = _harness(tmp_path, container="no-such-container", blind_verifier=SCRIPTED_MODEL,
                           model_config={"base_url": url, "api_key": "local"})
        harness.write_blind_verifier()
        assert len(requests) == steps + 1
        assert {r["model"] for r in requests} == {SCRIPTED_MODEL}
    assert harness.blind_script == "exit 0" and harness.blind_retried
    written = _events(tmp_path, "blind_verifier_written")
    assert [(w["model"], w["retry"], w["exhausted"], w["script"]) for w in written] == [
        (SCRIPTED_MODEL, False, True, ""), (SCRIPTED_MODEL, True, False, "exit 0")]
    assert written[0]["steps"] == steps and f"exceeded {steps} steps" in written[0]["error"]
    author = harness.stage_fields["blind_author"]
    assert author["model"] == SCRIPTED_MODEL
    assert author["cost_usd"] == pytest.approx(written[0]["cost_usd"] + written[1]["cost_usd"])
    assert author["cost_usd"] > 0


def test_a_credential_failure_at_the_last_step_is_not_exhaustion(tmp_path: Path) -> None:
    """An author whose model fails at its last allowed step is not retried on another family.

    The scripted server answers ``Bash`` calls until the step before the
    cap, then HTTP 401 (an expired credential): the run ends with the
    model's error, not the step cap, although the step counter stands at
    the cap.  A model failure is the author stage's error, which another
    family does not answer, so there is one run and the stage records it.
    """
    if other_family_model(SCRIPTED_MODEL) is None:
        pytest.skip("no model of another family is configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        steps = coding_sea.BLIND_VERIFIER_STEPS
        busy = tool_call_body("Bash", {"command": "true"}, 100)
        delivering = tool_call_body("finish", {"script": "exit 0"}, 100)

        class Expiring(_ExpiringHandler):
            refuse_at = steps

        with serve([busy, delivering], handler_base=Expiring) as (url, requests):
            harness = _harness(tmp_path, container=live.id, blind_verifier=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            assert harness.blind_script_ready() == ""
            assert len(requests) == steps
        written = _events(tmp_path, "blind_verifier_written")
        assert [(w["model"], w["retry"], w["exhausted"], w["steps"]) for w in written] == [
            (SCRIPTED_MODEL, False, False, steps)]
        assert "credential expired" in written[0]["error"]
        assert not harness.blind_retried
        stage = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "blind_author"][-1]
        assert (stage["outcome"], stage["retried"], stage["model"]) == (
            "missing", False, SCRIPTED_MODEL)
    finally:
        live.remove(force=True)


def test_gate_stage_records_carry_the_stages_model_and_cost(tmp_path: Path) -> None:
    """The review's and the blind author's ``gate_stage`` records name their model and cost.

    One scripted endpoint serves both: the author delivers its script at
    its first step (the first request, at the first model call, at the
    trial's effort), the reviewer answers ``VERDICT: OK`` at the finish
    (at the provider's default effort).  The review is the trial's one:
    a later finish after a file change is not reviewed again.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        author = tool_call_body("finish", {"script": "test -s /app/out.txt"}, 100)
        with serve([author, text_body("VERDICT: OK")]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=SCRIPTED_MODEL, blind_verifier=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local",
                                             "reasoning_effort": "high"})
            harness.on_llm_call([])
            assert harness.blind_script_ready() == "test -s /app/out.txt"
            _write_in(live, VERIFIER, "exit 0")
            done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
            assert harness.on_tool_call("finish", done) == ALLOW
            assert len(requests) == 2
            assert requests[0]["reasoning_effort"] == "high"  # the author, at the trial's effort
            assert "reasoning_effort" not in requests[1]  # the review, at the default effort
            assert harness.reviews == 1 and not harness.review_possible()
            assert harness.review_snapshot is not None and not harness.open_findings
            assert live.exec_run(["sh", "-c", "printf x > /app/new.txt"]).exit_code == 0
            assert harness.changed_since(harness.review_snapshot) and not harness.review_wanted()
            assert harness.on_tool_call("finish", dict(done)) == ALLOW
            assert harness.reviews == 1 and len(requests) == 2
        assert [r["round"] for r in _events(tmp_path, "review")] == [1]
        stages = {s["stage"]: s for s in _events(tmp_path, "gate_stage")}
        written = _events(tmp_path, "blind_verifier_written")[-1]
        review = _events(tmp_path, "review")[-1]
        assert written["cost_usd"] > 0 and review["cost_usd"] > 0
        assert (stages["blind_author"]["model"], stages["blind_author"]["cost_usd"]) == (
            SCRIPTED_MODEL, written["cost_usd"])
        assert (stages["blind_author"]["outcome"], stages["blind_author"]["retried"]) == (
            "pass", False)
        assert (stages["review"]["model"], stages["review"]["cost_usd"],
                stages["review"]["round"]) == (SCRIPTED_MODEL, review["cost_usd"], 1)
        assert stages["review"]["outcome"] == "pass"
        assert [stages[s]["outcome"] for s in ("verifier", "blind")] == ["pass", "pass"]
        assert all("model" not in stages[s] for s in ("verifier", "blind"))
    finally:
        live.remove(force=True)


def test_gate_checks_are_reused_while_the_tree_and_the_scripts_are_unchanged(
    tmp_path: Path,
) -> None:
    """A round in which nothing changed re-runs neither script; a change re-runs what it touches.

    Both checks count their runs in the container.  At the first finish
    the verifier passes and the independent check fails (round 1).  The
    next finish changes nothing: both runs are reused (``reused: True``
    on the events and the stage records), the failing verdict says so and
    still refuses (round 2).  A file created under the workdir re-runs
    both; a new text of the verifier re-runs the verifier alone; a
    give-up on the unchanged tree reuses both runs too.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        blind = "echo b >> /tmp/b; test -f /app/done"
        with serve([tool_call_body("finish", {"script": blind}, 100)]) as (url, _requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               blind_verifier=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            assert harness.blind_script_ready() == blind
            verifier = "echo v >> /tmp/v; echo checked"
            _write_in(live, VERIFIER, verifier)

            def runs() -> tuple[int, int]:
                out = live.exec_run(["sh", "-c", "cat /tmp/v /tmp/b 2>/dev/null | sort | uniq -c"])
                counts = dict(line.split()[::-1] for line in out.output.decode().splitlines())
                return int(counts.get("v", 0)), int(counts.get("b", 0))

            def stage_flags(stage: str) -> list[bool | None]:
                return [s.get("reused") for s in _events(tmp_path, "gate_stage")
                        if s["stage"] == stage]

            done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
            first = harness.on_tool_call("finish", dict(done))
            assert not first.allowed and "independent check" in first.text
            assert "reused" not in first.text and runs() == (1, 1)
            assert harness.verifier_refusals == 1 and harness.gate_rounds == 1
            # nothing changed: both runs stand, the failing one still refuses and says why
            second = harness.on_tool_call("finish", dict(done))
            assert not second.allowed and runs() == (1, 1) and harness.gate_rounds == 2
            assert coding_sea.REUSED_NOTE.format(path=coding_sea.BLIND_VERIFIER_PATH,
                                                 workdir="/app") in second.text
            assert harness.last_blind_run == (blind, 1, "") and harness.verifier_refusals == 2
            assert harness.last_verifier_run == (verifier, 0, "checked\n")
            assert [e.get("reused") for e in _events(tmp_path, "verifier")] == [None, True]
            assert [e.get("reused") for e in _events(tmp_path, "blind_verifier")] == [None, True]
            assert stage_flags("verifier") == [None, True] and stage_flags("blind") == [None, True]
            # a file created under the workdir: both run again, and the check passes now
            assert live.exec_run(["touch", "/app/done"]).exit_code == 0
            assert harness.on_tool_call("finish", dict(done)) == ALLOW and runs() == (2, 2)
            assert harness.checkpoints == ["verified"]
            # a new text of the verifier: it runs again, the independent check does not
            _write_in(live, VERIFIER, verifier + " twice")
            assert harness.on_tool_call("finish", dict(done)) == ALLOW and runs() == (3, 2)
            assert stage_flags("verifier") == [None, True, None, None]
            assert stage_flags("blind") == [None, True, None, True]
            assert harness.last_verifier_run == (verifier + " twice", 0, "checked twice\n")
            # a give-up on the unchanged tree reuses both runs too
            giveup: dict[str, Any] = {"success": False, "summary_in_html": "<p>Stuck.</p>"}
            assert not harness.on_tool_call("finish", giveup).allowed  # both pass: reconsider
            assert runs() == (3, 2)
            checks = [s for s in _events(tmp_path, "gate_stage") if s.get("giveup")]
            assert [(s["stage"], s["outcome"], s["reused"]) for s in checks] == [
                ("verifier", "pass", True), ("blind", "pass", True)]
            assert f"verifier {VERIFIER}: pass" in giveup["summary_in_html"]
    finally:
        live.remove(force=True)


def _cut_short_body(text: str) -> bytes:
    """A chat-completion body whose answer the output cap cut short (``finish_reason: length``)."""
    return text_body(text).replace(b'"finish_reason": "stop"', b'"finish_reason": "length"')


def _filtered_body() -> bytes:
    """A chat-completion body the provider's content filter stopped."""
    return text_body("").replace(b'"finish_reason": "stop"', b'"finish_reason": "content_filter"')


def test_a_failed_reviewer_is_retried_once_at_low_effort_then_skipped(tmp_path: Path) -> None:
    """A reviewer that answers nothing, is cut short or refused is retried once at low effort.

    The first reviewer answers nothing at the catalog's default effort
    (the trial's own effort override is not applied), then at low effort
    (the retry carries ``reasoning_effort: low``, the model having a
    catalog thinking level) approves: one review, its record naming the
    retry.
    The second reviewer is cut short by the output cap and refused on the
    retry: the review is skipped with both failures on its record, the
    finish goes through (refused by nothing), and the failure counts as
    a failed review, not as a refusal against the agent.  A model that
    takes no effort parameter is retried as it was.
    """
    # looked up before the container starts: a catalog without such a model ends the
    # test here, where no container is left running
    thinking = next(name for name, info in MODEL_INFO.items()
                    if info.thinking and info.thinking != "low"
                    and not info.is_decisions_supported)
    efforts = [MODEL_INFO[thinking].thinking, "low"]
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        with serve([text_body(""), text_body("VERDICT: OK")]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=thinking,
                               model_config={"base_url": url, "api_key": "local",
                                             "reasoning_effort": "high"})
            harness.on_llm_call([])
            _write_in(live, VERIFIER, "exit 0")
            done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
            assert harness.on_tool_call("finish", dict(done)) == ALLOW
            assert [r.get("reasoning_effort") for r in requests] == efforts
            assert (harness.reviews, harness.failed_reviews) == (1, 0)
        review = _events(tmp_path, "review")[-1]
        assert review == {**review, "failed": False, "error": "", "retried": "empty answer",
                          "answer": "VERDICT: OK"}
        assert review["cost_usd"] > 0 and review["tokens"] > 0  # both attempts' spend
        stage = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "review"][-1]
        assert stage == {**stage, "outcome": "pass", "retried": "empty answer", "round": 1}
        assert "error" not in stage
        with serve([_cut_short_body("VERDICT: NOT"), _filtered_body()]) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=thinking,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            assert harness.on_tool_call("finish", dict(done)) == ALLOW
            assert [r.get("reasoning_effort") for r in requests] == efforts
            assert (harness.reviews, harness.failed_reviews) == (0, 1)
            assert harness.gate_rounds == 0 and not _events(tmp_path, "gate_round")
        review = _events(tmp_path, "review")[-1]
        assert review["failed"] and review["retried"].startswith("OutputCapError(")
        assert review["error"].startswith("ContentFilterError(") and review["answer"] == ""
        stage = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "review"][-1]
        assert stage["outcome"] == "error" and stage["retried"] == review["retried"]
        assert stage["error"] == review["error"]
        # what is retried, and the retry's effort by model
        assert coding_sea.retryable_review_failure("", "") is True
        assert coding_sea.retryable_review_failure("I could not tell.", "") is False
        assert coding_sea.retryable_review_failure("", "no answer within 240 s") is True
        assert coding_sea.retryable_review_failure("", "ModelRefusalError('no')") is True
        assert coding_sea.retryable_review_failure("", "KISSError('401')") is False
        assert coding_sea.low_effort({"api_key": "k", "reasoning_effort": "high"}, thinking) == {
            "api_key": "k", "reasoning_effort": "low"}
        assert coding_sea.low_effort({"reasoning_effort": "high"}, SCRIPTED_MODEL) is None
    finally:
        live.remove(force=True)


def test_the_restart_budget_ends_an_idle_run(tmp_path: Path) -> None:
    """From the fourth restart on, a hand-off with no new edit and the same verdict ends the run.

    Three hand-offs the agent asks for (``finish(is_continue=True)``) go
    through whatever happened between them.  The fourth comes after an
    edit: it goes through; the fifth after a change of the verifier's
    verdict: it goes through; the sixth with neither: the finish is made
    terminal (``is_continue`` false, the state block as the caveat) and
    the trajectory records ``restart_budget``.  A hand-off the context
    limit forced is counted at the next session's first model call
    instead, and a denied one ends the run there, before the call.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([])
        assert harness.restart_budget == coding_sea.RESTART_BUDGET == 4
        _write_in(live, VERIFIER, "test -f /app/done")
        start = {"role": "user", "content": "task\n\n# Task Progress (Continuation 1)\n..."}

        def hand_off() -> dict[str, Any]:
            """The agent's pause, then the next session's first model call."""
            pause = {"success": False, "is_continue": True, "summary_in_html": "<p>Later.</p>"}
            assert harness.on_tool_call("finish", pause) == ALLOW
            harness.on_llm_call([{"role": "tool", "content": "paused"}])
            harness.on_llm_call([dict(start)])
            return pause

        for _ in range(3):  # the first three restarts are never judged
            assert hand_off()["is_continue"] is True
        assert harness.restarts == 3 and not _events(tmp_path, "restart_budget")
        # the fourth comes with an edit, the fifth with a new verdict of the verifier
        _write_in(live, coding_sea.LEDGER_PATH, "- [x] read the task\n- [ ] note the result\n")
        note = {"file_path": "/app/note.txt", "content": "n"}
        assert harness.on_tool_call("Write", note) == ALLOW
        _write_in(live, "/app/note.txt", "n")
        harness.on_llm_call([{"role": "tool", "content": "ok"}])
        assert hand_off()["is_continue"] is True and harness.restarts == 4
        assert harness.on_tool_call("Bash", {"command": f"bash {VERIFIER}"}) == ALLOW
        harness.on_llm_call([{"role": "tool", "content": "[exit code: 1]"}])
        assert hand_off()["is_continue"] is True and harness.restarts == 5
        # the sixth with neither: no further session
        pause: dict[str, Any] = {
            "success": False, "is_continue": True, "summary_in_html": "<p>Later.</p>"}
        assert harness.on_tool_call("finish", pause) == ALLOW
        assert pause["is_continue"] is False and pause["success"] is False
        assert pause["summary_in_html"].startswith("<p>Later.</p>\n<h4>Ended by the restart budget")
        assert "Restart 6 came with no new edit" in pause["summary_in_html"]
        assert "/app/note.txt (created, 0 lines)" in pause["summary_in_html"]
        assert "exit 1, run by the agent at" in pause["summary_in_html"]
        assert "- [ ] note the result" in pause["summary_in_html"]
        assert "- [x] read the task" not in pause["summary_in_html"]
        assert [e["restarts"] for e in _events(tmp_path, "restart_budget")] == [6]
        assert [e["idle"] for e in _events(tmp_path, "restart")] == [
            True, True, True, False, False, True]
        # a hand-off the context limit forced: counted at the session's first call, and a
        # denied one ends the run there with the state as the error's message
        with pytest.raises(coding_sea.RestartBudgetError) as raised:
            harness.on_llm_call([dict(start)])
        assert harness.restarts == 7 and "Restart 7 came with no new edit" in str(raised.value)
        assert "## Running background jobs" in str(raised.value)
        assert [e["restarts"] for e in _events(tmp_path, "restart_budget")] == [6, 7]
        assert _events(tmp_path, "inventory")[-1]["turn"] == harness.turns
        # a new verdict since (the gate's own run) makes the next one a plain restart again,
        # and the budget is a setting
        _write_in(live, VERIFIER, "exit 2")
        assert not harness.on_tool_call("finish", {"success": True, "summary_in_html": "x"}).allowed
        assert harness.last_verdict is not None and harness.last_verdict[0] == "2"
        harness.on_llm_call([{"role": "tool", "content": "REFUSED"}])
        harness.on_llm_call([dict(start)])
        assert harness.restarts == 8 and _events(tmp_path, "restart")[-1]["idle"] is False
        tight = _harness(tmp_path, container=live.id, restart_budget=1)
        tight.on_llm_call([])
        with pytest.raises(coding_sea.RestartBudgetError):
            tight.on_llm_call([dict(start)])
    finally:
        live.remove(force=True)


@pytest.mark.live_api  # the unanswered finding is classified by the real decisions model
def test_a_review_without_a_verdict_or_past_its_deadline_refuses_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed review is neither approval nor rejection; a stalled one fails at the deadline.

    A scripted reviewer finds the work wrong (round 1).  The file changes
    and, with a second review allowed by the config, the reviewer answers
    without a verdict line (a judgement, not retried), then (after another
    change) stalls past ``REVIEW_DEADLINE`` (one second here), is retried
    once and stalls again: each is logged as failed with its error (the
    retried one with the first attempt's), refuses nothing, keeps the
    first review's findings open and counts under ``MAX_FAILED_REVIEWS``,
    past which no review is asked for.  A finish no review verdict came
    back for is judged on those findings: unanswered by the summary (the
    decisions model's call), they go with the delivered work as caveats,
    not lost to a re-review that did not happen.  Every review request
    went at the provider's default effort: the trial's own effort
    overrides, in the framework's spelling and in the providers' own
    (Anthropic's ``output_config.effort``, OpenAI's ``reasoning``), are
    left out (the scripted server speaks chat completions, which the
    native keys would break), while the rest of ``output_config`` stays;
    the retry of a model that takes no effort parameter (the scripted
    one has no catalog thinking level) goes as the attempt did.
    """
    monkeypatch.setattr(coding_sea, "REVIEW_DEADLINE", 1)
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf goodbye > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    _StallingHandler.release.clear()
    try:
        finding = "1. out.txt must contain exactly hello: it contains goodbye"
        script = [text_body(f"VERDICT: NOT DONE\n{finding}"), text_body("I could not tell."),
                  STALL]
        with serve(script, handler_base=_StallingHandler) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=SCRIPTED_MODEL, max_reviews=2,
                               model_config={"base_url": url, "api_key": "local",
                                             "reasoning_effort": "high",
                                             "output_config": {"effort": "max"},
                                             "reasoning": {"effort": "high"}})
            assert coding_sea.default_effort(harness.model_overrides) == {
                "base_url": url, "api_key": "local"}
            assert coding_sea.default_effort({"output_config": {"effort": "max", "format": "x"},
                                              "thinking": {"budget_tokens": 9}}) == {
                "output_config": {"format": "x"}}
            assert coding_sea.default_effort({"reasoning_effort": "low"}) is None
            harness.on_llm_call([])
            _write_in(live, VERIFIER, "exit 0")
            done: dict[str, Any] = {"success": True, "summary_in_html": "<p>Done.</p>"}
            assert harness.on_tool_call("finish", dict(done)) == refuse(coding_sea.gate_refusal(
                1, 3, [coding_sea.REVIEW_VERDICT.format(findings=finding)]))
            assert harness.open_findings == [finding] and harness.review_possible()
            # a change since the review: the second review runs and answers no verdict
            assert live.exec_run(["sh", "-c", "printf hello > /app/out.txt"]).exit_code == 0
            assert harness.review_wanted()
            second = dict(done)
            assert harness.on_tool_call("finish", second) == ALLOW
            assert harness.open_findings == [finding]
            assert (harness.reviews, harness.failed_reviews) == (1, 1)
            unanswered = coding_sea.UNANSWERED_VERDICT.format(findings=finding)
            if decisions_tool_available():
                assert _events(tmp_path, "findings_answered")[-1]["answered"] == [False]
                assert harness.delivered_findings == [unanswered.splitlines()[0]]
                assert "Delivered after 1 refused finish rounds" in second["summary_in_html"]
                assert html.escape(unanswered) in second["summary_in_html"]
            else:
                assert not harness.delivered_findings and second == done
            # another change: the third review stalls and is given up at the deadline
            assert live.exec_run(["sh", "-c", "printf x > /app/new.txt"]).exit_code == 0
            started = time.time()
            assert harness.on_tool_call("finish", dict(done)) == ALLOW
            assert time.time() - started < 15
            _StallingHandler.release.set()
            assert (harness.reviews, harness.failed_reviews) == (1, 2)
            assert harness.open_findings == [finding] and not harness.review_possible()
            assert live.exec_run(["sh", "-c", "printf y > /app/new.txt"]).exit_code == 0
            assert not harness.review_wanted()
            assert len(requests) == 4  # the stalled review was retried once
            assert not any(set(r) & {"reasoning_effort", "output_config", "reasoning"}
                           for r in requests)
        reviews = _events(tmp_path, "review")
        assert [(r["failed"], r["error"], r["round"], r["answer"]) for r in reviews] == [
            (False, "", 1, f"VERDICT: NOT DONE\n{finding}"), (True, "", 1, "I could not tell."),
            (True, "no answer within 1 s", 1, "")]
        assert [r.get("retried") for r in reviews] == [None, None, "no answer within 1 s"]
        assert reviews[1]["cost_usd"] > 0 and reviews[2]["cost_usd"] == 0
        assert reviews[2]["last_call"] == {}
        stages = [s for s in _events(tmp_path, "gate_stage") if s["stage"] == "review"]
        assert [(s["model"], s["round"]) for s in stages] == [(SCRIPTED_MODEL, 1)] * 3
        assert [s["outcome"] for s in stages] == ["fail", "error", "error"]
        assert [s.get("error") for s in stages] == [None, "no verdict", "no answer within 1 s"]
        assert [s.get("retried") for s in stages] == [None, None, "no answer within 1 s"]
        assert stages[2]["seconds"] >= 2
        assert len(_events(tmp_path, "gate_round")) == 1 and harness.gate_rounds == 1
        assert harness.checks_passed
        assert bool(harness.delivered_findings) == decisions_tool_available()
    finally:
        _StallingHandler.release.set()
        live.remove(force=True)


def test_the_stalling_handler_sets_its_stop_once_the_request_is_recorded() -> None:
    """A request whose body is still on its way has not stopped the run.

    A model client writes a request's headers and body separately and
    gives the request up on a Stop: one set at the headers could make it
    drop the body, leaving the handler nothing to record (audit of
    2026-11-10: the Stop test below then waited for a record that never
    came).  The upload here pauses one byte short of the body: no stop
    and no record yet; the last byte brings both.
    """
    run_stop = threading.Event()

    class Stopping(_StallingHandler):
        stop = run_stop

    body = json.dumps({"model": SCRIPTED_MODEL}).encode()
    _StallingHandler.release.clear()
    try:
        with serve([STALL], handler_base=Stopping) as (url, requests):
            upload = http.client.HTTPConnection(urlsplit(url).netloc, timeout=10)
            with contextlib.closing(upload):
                upload.putrequest("POST", "/v1/chat/completions")
                upload.putheader("Content-Length", str(len(body)))
                upload.endheaders(body[:-1])
                assert not run_stop.wait(0.2) and not requests
                upload.send(body[-1:])
                assert run_stop.wait(30) and requests == [{"model": SCRIPTED_MODEL}]
    finally:
        _StallingHandler.release.set()


def test_a_stop_during_the_review_propagates_and_logs_its_spend(tmp_path: Path) -> None:
    """The run's Stop ends a review in flight; what the call cost is still the trial's.

    The scripted server sets the run's stop event once the review request
    is recorded and holds the connection: the model client gives the request
    up on the stop, the stop propagates out of the review (no verdict,
    nothing counted as a review or a failed review), and the side call
    logs its spend as a failed ``review`` event before it does, so
    :func:`coding_sea.review_usage` sees it (review of 2026-11-05, finding
    2: the spend of a stopped call was lost).  Here the client was stopped
    before any reply was billed, so the logged spend is zero; a reply
    billed before the stop landed would be the event's cost.  The thread's
    stop event is the run's again afterwards.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    run_stop = threading.Event()

    class Stopping(_StallingHandler):
        stop = run_stop

    outer = stop_signal.get_thread_stop_event()
    stop_signal.set_thread_stop_event(run_stop)
    _StallingHandler.release.clear()
    try:
        with serve([STALL], handler_base=Stopping) as (url, requests):
            harness = _harness(tmp_path, container=live.id, verifier_timeout=10,
                               reviewer=SCRIPTED_MODEL,
                               model_config={"base_url": url, "api_key": "local"})
            harness.on_llm_call([])
            _write_in(live, VERIFIER, "exit 0")
            with pytest.raises(KeyboardInterrupt):
                harness.review_verdict({"success": True, "summary_in_html": "<p>Done.</p>"})
            assert run_stop.is_set() and len(requests) == 1
        assert stop_signal.get_thread_stop_event() is run_stop
    finally:
        _StallingHandler.release.set()
        stop_signal.set_thread_stop_event(outer)
        live.remove(force=True)
    assert (harness.reviews, harness.failed_reviews) == (0, 0)
    reviews = _events(tmp_path, "review")
    assert [(r["failed"], r["error"], r["model"], r["cost_usd"], r["tokens"]) for r in reviews] == [
        (True, "stopped", SCRIPTED_MODEL, 0.0, 0)]
    assert coding_sea.review_usage(tmp_path / "trajectory.jsonl") == (0.0, 0)


def test_inventory_is_appended_at_a_context_restart(tmp_path: Path) -> None:
    """The first message of a continuation session gets the harness's state block, in a fixed form.

    Four sections in order: the files changed since the start (created
    ones with their line count, modified ones with their size before and
    after, deleted ones), the verifier's last verdict (by the gate or the
    agent: exit code, when, the head of its output), the ledger's open
    ``- [ ]`` items and the running background jobs (from the job files
    the container tools keep).  The turn's commands are still paired with
    their exit codes for the evidence ledger.
    """
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt && "
                           "printf 'a\\nb\\n' > /app/gone.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id)
        harness.on_llm_call([{"role": "user", "content": "task"}])
        _write_in(live, coding_sea.LEDGER_PATH,
                  "- out.txt: exactly hello\n- [x] write out.txt\n- [ ] check the newline\n"
                  "  * [ ] ask: trailing newline or not\n- [ ]not an item")
        assert harness.on_tool_call("Read", {"file_path": "out.txt"}) == ALLOW
        assert harness.on_tool_call("Read", {"file_path": "/app/missing.txt"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "ls /app"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "false"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "python3 train.py"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "make", "background": True}) == ALLOW
        assert harness.on_tool_call("Write", {"file_path": "/app/new.py", "content": "x"}) == ALLOW
        assert not harness.on_tool_call("finish", {"success": True, "summary_in_html": "x"}).allowed
        # the results come in the order of the calls (a Read's and the refused
        # finish's among them) and carry the usage footer after the marker
        footer = "\n\nSteps: 3/100, Context: 1,000/500,000 tokens, Budget: $0.10/$6.00"
        results = [
            {"role": "tool", "content": "hello" + footer},
            {"role": "tool", "content": "Error: file not found" + footer},
            {"role": "tool", "content": "out.txt\n" + footer},
            {"role": "tool", "content": "[exit code: 1]" + footer},
            {"role": "tool", "content": "Error: command timed out after 30s" + footer},
            {"role": "tool", "content": "Started background job ab12 (pid 7); log: /x" + footer},
            {"role": "tool", "content": "Successfully wrote 1 characters" + footer},
            {"role": "tool", "content": "REFUSED: " + coding_sea.VERIFIER_MISSING_VERDICT},
        ]
        harness.on_llm_call(results)
        assert harness.commands_run == [("ls /app", "0"), ("false", "1"),
                                        ("python3 train.py", "timeout"), ("make", "background")]
        # a turn whose results do not pair up with its calls records nothing
        assert harness.on_tool_call("Bash", {"command": "true"}) == ALLOW
        harness.on_llm_call([{"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}])
        assert len(harness.commands_run) == 4
        # a plain turn is not a restart
        plain = [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]
        harness.on_llm_call(plain)
        assert plain == [{"role": "user", "content": [{"type": "tool_result", "content": "x"}]}]
        # the work so far: a file created, one modified, one deleted, a job still running
        # and one that exited, and the agent's own (failing) run of its verifier
        assert live.exec_run(["sh", "-c", "printf 'x\\ny\\nz\\n' > /app/new.py && "
                              "printf hello! > /app/out.txt && rm /app/gone.txt"]).exit_code == 0
        jobs = ContainerJobs(partial(_exec_in, live))
        running, _ = jobs.start("sleep 300")
        exited, _ = jobs.start("true")
        assert running and exited and wait_for(lambda: "exited" in jobs.wait(exited, 5), 10)
        _write_in(live, VERIFIER, "echo checking; test \"$(cat /app/out.txt)\" = hello")
        assert harness.on_tool_call("Bash", {"command": f"bash {VERIFIER}"}) == ALLOW
        harness.on_llm_call([{"role": "tool", "content": "checking\n[exit code: 1]" + footer}])
        # the continuation prompt (string content) gets the state block: four sections
        start = [{"role": "user", "content": "task\n\n# Task Progress (Continuation 1)\n..."}]
        harness.on_llm_call(start)
        text = start[0]["content"]
        assert text.startswith("task\n\n# Task Progress (Continuation 1)\n...\n\n# Inventory "
                               "(written by the harness at this restart)\n")
        assert harness.last_verdict is not None
        when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(harness.last_verdict[1]))
        assert text.split("# Inventory (written by the harness at this restart)\n")[1] == (
            "## Files changed since the start\n"
            "  /app/gone.txt (deleted)\n"
            "  /app/new.py (created, 3 lines)\n"
            "  /app/out.txt (modified, 5 -> 6 bytes)\n"
            "## Verifier's last verdict\n"
            f"  exit 1, run by the agent at {when} (turn 5); output head:\n"
            "    checking\n"
            "## Ledger open checklist\n"
            "  - [ ] check the newline\n"
            "  * [ ] ask: trailing newline or not\n"
            "## Running background jobs\n"
            f"  {running}: sleep 300")
        assert _events(tmp_path, "inventory")[-1]["turn"] == 6
        # the gate's own run of the verifier is the verdict then, with its output head;
        # text-block content (the Responses API shape) is extended with a text block
        _write_in(live, VERIFIER, "seq 1 30; exit 3")
        assert not harness.on_tool_call("finish", {"success": True, "summary_in_html": "x"}).allowed
        blocks: list[dict[str, Any]] = [{"role": "user", "content": [
            {"type": "input_text", "text": "# Task Progress (Continuation 2)"}]}]
        harness.on_llm_call([{"role": "tool", "content": "REFUSED"}])
        harness.on_llm_call(blocks)
        assert blocks[0]["content"][-1]["type"] == "input_text"
        added = blocks[0]["content"][-1]["text"]
        assert "  exit 3, run by the gate at " in added and added.count("\n    ") == 20
        assert "\n    1\n" in added and "\n    20\n" in added and "\n    21\n" not in added
        # without the ledger, the verifier or file tracking (the root workdir) the sections say so
        bare = _harness(tmp_path, container=live.id, workdir="/")
        bare.on_llm_call([])
        assert live.exec_run(["rm", coding_sea.LEDGER_PATH]).exit_code == 0
        assert bare.inventory().split("\n")[2:] == [
            "## Files changed since the start", "  (no file tracking)",
            "## Verifier's last verdict", f"  (the verifier {VERIFIER} has not run)",
            "## Ledger open checklist", "  (no ledger)",
            "## Running background jobs", f"  {running}: sleep 300"]
        assert jobs.bash_job(running, "kill").startswith(f"Job {running}")
        assert bare.running_jobs_lines() == ["(none)"]
        assert coding_sea.user_text({"role": "assistant", "content": "x"}) == ""
        assert coding_sea.user_text({"role": "user", "content": 3}) == ""
        coding_sea.append_user_text({"role": "user", "content": None}, "n")  # nothing to extend
    finally:
        live.remove(force=True)


@pytest.mark.live_api
def test_numbered_items_and_answered_findings() -> None:
    """Findings split into numbered items; the decisions model tells evidence from argument."""
    text = "1. out.txt: says goodbye\n   not hello\n2) format: no newline\n\nnote"
    assert coding_sea.numbered_items(text) == [
        "1. out.txt: says goodbye not hello", "2) format: no newline note"]
    assert coding_sea.numbered_items("just prose\nmore") == ["just prose more"]
    assert coding_sea.numbered_items("") == []
    assert coding_sea.answered_findings("anything", []) == []
    findings = ["1. out.txt must contain exactly hello: it contains goodbye"]
    evidence = "<p>1. `cat /app/out.txt` now prints <code>hello</code>.</p>"
    argument = "<p>1. The reviewer is mistaken; any greeting satisfies the task.</p>"
    if not decisions_tool_available():
        pytest.skip("no decisions model configured")
    assert coding_sea.answered_findings(evidence, findings) == [True]
    assert coding_sea.answered_findings(argument, findings) == [False]


def test_review_findings_parse_the_reviewer_answer() -> None:
    assert coding_sea.review_findings("VERDICT: OK") == ""
    assert coding_sea.review_findings("Looks fine.\nverdict: ok\n") == ""
    assert coding_sea.review_findings("no verdict line at all") == ""
    assert coding_sea.review_findings("VERDICT: NOT DONE\n1. out.txt: says goodbye\n") == (
        "1. out.txt: says goodbye")
    assert coding_sea.review_usage(Path("/nonexistent/trajectory.jsonl")) == (0.0, 0)


def test_review_usage_counts_only_the_records_after_the_offset(tmp_path: Path) -> None:
    """A reused trial directory: reviews of earlier runs (before ``since``) are not counted."""
    log = tmp_path / "trajectory.jsonl"
    first = json.dumps({"event": "review", "cost_usd": 0.25, "tokens": 25}) + "\n"
    log.write_text(first + "not json\n")
    offset = log.stat().st_size
    with log.open("a") as fh:
        fh.write(json.dumps({"event": "review", "cost_usd": 0.5, "tokens": 50}) + "\n")
        fh.write(json.dumps({"event": "verifier", "exit_code": 0}) + "\n")
    assert coding_sea.review_usage(log) == (0.75, 75)
    assert coding_sea.review_usage(log, offset) == (0.5, 50)


def test_changed_files_report_without_tracking(tmp_path: Path) -> None:
    harness = _harness(tmp_path, container="no-such-container")
    assert harness.changed_files_report() == "(no file tracking)"
    harness.file_baseline = {}
    assert harness.changed_files_report() == "(no file tracking)"


@dataclass
class ExecResult:
    """The shape of Harbor's ``ExecResult``."""

    stdout: str | None
    stderr: str | None
    return_code: int


class LocalEnvironment:
    """A Harbor-like environment whose ``exec`` runs on this machine (asyncio subprocesses).

    As Harbor's providers do, a timeout kills the shell alone: killing the
    tree is the bridge's job, which the tests prove.  Each command leads
    its own session so that :meth:`kill_all` can end whatever a test left
    running: the bridge's tagged-tree sweep scans ``/proc``, which macOS
    has not, so there a stopped ``sleep 20`` and the reaper thread
    sweeping for it would otherwise outlive the test.
    """

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.sessions: list[asyncio.subprocess.Process] = []

    @staticmethod
    def type() -> str:
        return "local"

    async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> Any:
        proc = await asyncio.create_subprocess_shell(
            command, cwd=self.cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self.sessions.append(proc)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        except TimeoutError:
            proc.kill()
            raise
        return ExecResult(out.decode(), err.decode(), proc.returncode or 0)

    def kill_all(self) -> None:
        """Kill what is left of every session this environment started."""
        for proc in self.sessions:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)


class _Loop:
    """An asyncio loop on a daemon thread, as Harbor's agent loop is to the bridge threads."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        if not self.thread.is_alive():
            self.loop.close()


def _stop_when(predicate: Callable[[], bool]) -> threading.Event:
    """Return a stop event a helper thread sets once *predicate* holds (or after 10 s).

    A Stop fired on a timer could land before the bridge has even read the
    command's ``/exec`` (the ``/kill``-before-``/exec`` ordering, pinned on
    its own below); firing on a predicate pins the ordering each test means.
    """
    stop = threading.Event()

    def fire() -> None:
        wait_for(predicate, 10)
        stop.set()

    threading.Thread(target=fire, daemon=True).start()
    return stop


def _no_server_thread() -> bool:
    """Whether no bridge server thread is alive."""
    return not any(t.name == SERVER_THREAD_NAME for t in threading.enumerate())


def _no_reaper_thread() -> bool:
    """Whether no bridge reaper thread is alive (a reaper ends once the exec it sweeps for does)."""
    return not any(t.name.startswith("exec-bridge-reaper-") for t in threading.enumerate())


def _dead(pidfile: Path) -> bool:
    """Whether the process whose pid *pidfile* records is gone or a zombie (killed, not reaped).

    ``False`` while no pid is recorded: nothing is proven until there is one.
    """
    try:
        pid = int(pidfile.read_text())
    except (FileNotFoundError, ValueError):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1]
    except (FileNotFoundError, ProcessLookupError):  # gone; or reaped between open and read
        return True
    return state.startswith("Z")


def _dead_in(container: Any, pidfile: str) -> bool:
    """Whether the process whose pid *pidfile* in *container* records is gone or a zombie.

    :func:`_dead` inside the container; ``False`` while no pid is recorded.
    """
    script = (f"pid=$(cat {pidfile}) || exit 1; test -e /proc/$pid || exit 0; "
              "grep -q ') Z ' /proc/$pid/stat")
    return bool(container.exec_run(["sh", "-c", script]).exit_code == 0)


def _holds(container: Any, path: str, content: bytes) -> bool:
    """Whether *path* in *container* holds exactly *content*."""
    return bool(container.exec_run(["cat", path]).output == content)


def _touch(path: Path) -> str:
    """A shell command that creates *path*, quoted: a temporary directory may hold a space."""
    return f"touch {shlex.quote(str(path))}"


def _late_toucher(late: Path, pidfile: Path) -> str:
    """A command whose background subshell's pid is recorded, then touches *late* after 6 s.

    Killing the tree is proven without waiting the 6 s out: once the
    recorded subshell is dead (:func:`_dead`) and *late* is absent, the
    touch can never land; a surviving subshell would have touched *late*
    before dying on its own, within the wait's 10 s.  The 6 s
    leave the bridge's tagged-tree sweep (a ``/proc`` scan in its own
    subprocess, after the 1 s command timeout) 5 s on a loaded CI box;
    at 2 s the touch landed first under a parallel full-suite run.  The
    pid lands in *pidfile* atomically, so the file's existence means
    the pid is readable.
    """
    pid = shlex.quote(str(pidfile))
    record = f"echo $! > {pid}.tmp && mv {pid}.tmp {pid}"
    return f"(sleep 6; {_touch(late)}) & {record}; wait"


def _bridged_workdir(tmp_path: Path) -> tuple[Path, _Loop, ExecBridgeServer]:
    """A work dir holding ``data.txt`` and a bridge server for a local environment on it."""
    workdir = tmp_path / "app"
    workdir.mkdir()
    (workdir / "data.txt").write_text("v1\n")
    loop = _Loop()
    return workdir, loop, ExecBridgeServer(LocalEnvironment(str(workdir)), loop.loop)


def _close(server: ExecBridgeServer, loop: _Loop) -> None:
    """End a bridged local environment: its commands and their reapers, then the server and loop.

    A reaper sweeps on the loop until the exec it covers ends, so the loop
    is closed only once the reapers are gone: a sweep handed to a closed
    loop would raise on the reaper thread instead.  A reaper that does
    not end fails the test, but the server and the loop are closed all
    the same, so the failure does not hold the process open.
    """
    server.environment.kill_all()
    try:
        assert wait_for(_no_reaper_thread, 10)
    finally:
        server.close()
        loop.close()


@local_shell
def test_exec_bridge_runs_the_tools_on_a_remote_environment(tmp_path: Path) -> None:
    """Bash and run_commands_parallel work through the bridge; a gone server is reported."""
    workdir, loop, server = _bridged_workdir(tmp_path)
    try:
        url = server.url
        assert exec_bridge.bridge_alive(url)
        assert exec_bridge.bridge_exec(url, "echo hi; echo err >&2; exit 4", 10) == (4, "hi\nerr\n")
        # a timed-out command is reported as such
        assert exec_bridge.bridge_exec(url, "sleep 5", 1) == (None, "")
        # a Stop while a command runs ends the wait at once
        marker = workdir / "started"
        stop = _stop_when(marker.exists)
        started = time.monotonic()
        stopped = exec_bridge.bridge_exec(url, f"{_touch(marker)}; sleep 20", 30, stop)
        assert stopped == (-1, "Killed: the task was stopped.")
        assert time.monotonic() - started < 5
        with exec_bridge.ExecBridge(f"exec:{url}", workdir="/x", volumes={}) as bridge:
            assert bridge.url == url and bridge.volumes == {}
            assert not hasattr(bridge, "container")  # a child inherits the bridge by url
            assert bridge.Bash("cat data.txt", "read") == "v1\n"
            # the task's stop event ends Bash and run_commands_parallel the same way
            # (fired once the command is running, as above)
            bash_marker = workdir / "bash-started"
            bridge.stop_event = _stop_when(bash_marker.exists)
            started = time.monotonic()
            assert bridge.Bash(
                f"{_touch(bash_marker)}; sleep 20", "stopped", timeout_seconds=60
            ) == "Killed: the task was stopped.\n[exit code: -1]"
            assert time.monotonic() - started < 5
            parallel_marker = workdir / "parallel-started"
            bridge.stop_event = _stop_when(parallel_marker.exists)
            started = time.monotonic()
            report = bridge.run_commands_parallel(
                json.dumps([f"{_touch(parallel_marker)}; sleep 20", "sleep 20"]), timeout_seconds=60
            )
            assert time.monotonic() - started < 5 and "stopped" in report.lower()
            bridge.stop_event = None
            # ``cat`` of a missing file exits 1 under both GNU and BSD coreutils
            failed = bridge.Bash("cat nope", "fail", timeout_seconds=5)
            assert "No such file" in failed and failed.endswith("[exit code: 1]")
            assert bridge.Bash("exit 9", "silent").strip() == "[exit code: 9]"
            slow = bridge.Bash("echo partial; sleep 5", "slow", timeout_seconds=1)
            assert slow.startswith("Error: command timed out after 1s")
            destructive = bridge.Bash("rm -rf /", "destructive")
            assert destructive.startswith("REFUSED: the Bash call did NOT run"), destructive
            report = bridge.run_commands_parallel('["echo a", "exit 1", "sleep 5"]',
                                                  timeout_seconds=1)
            assert "3 commands: 1 succeeded, 1 failed, 1 timed out" in report
            assert bridge.run_commands_parallel("not json").startswith("Error")
            assert bridge.run_commands_parallel('["ls"]', max_workers=-1).startswith("Error")
    finally:
        _close(server, loop)
    # the server is gone: the client reports it instead of hanging
    assert not exec_bridge.bridge_alive(url)
    with pytest.raises(KISSError, match="unreachable"):
        exec_bridge.bridge_exec(url, "true", 5)
    with pytest.raises(KISSError, match="not answering"):
        with exec_bridge.ExecBridge(f"exec:{url}"):
            pass


@local_shell
@gnu_coreutils_only
def test_exec_bridge_runs_the_file_tools_and_the_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file tools (built on Bash) and the harness work through the bridge.

    The verifier and checkpoint paths are the SEA's contract, one per
    host, and this environment is the host's own filesystem: the test
    points both at ``tmp_path`` so it neither clobbers nor reads another
    session's ``/tmp/kiss-verify.sh`` or ``/tmp/kiss-checkpoints``.
    """
    monkeypatch.setattr(coding_sea, "VERIFIER_PATH", str(tmp_path / "kiss-verify.sh"))
    monkeypatch.setattr(coding_sea, "CHECKPOINT_DIR", str(tmp_path / "kiss-checkpoints"))
    workdir, loop, server = _bridged_workdir(tmp_path)
    try:
        url = server.url
        with exec_bridge.ExecBridge(f"exec:{url}") as bridge:
            tools = DockerTools(bridge.Bash)
            assert tools.Write(str(workdir / "new.py"), "x = 1\n").startswith("Successfully")
            assert "x = 1" in tools.Read(str(workdir / "new.py"))
            assert tools.Edit(str(workdir / "new.py"), "x = 1", "x = 2").startswith("Successfully")
            assert (workdir / "new.py").read_text() == "x = 2\n"
        # the harness on a bridged trial
        config = tmp_path / "config.json"
        config.write_text(json.dumps({
            "container": f"exec:{url}", "workdir": str(workdir), "prompt": "p", "model": "m",
            "trajectory": str(tmp_path / "trajectory.jsonl"), "finish_gate": True,
            "verifier_timeout": 1,
        }))
        harness = coding_sea.ContainerHarness(str(config))
        assert harness.bridged and harness.settings()["docker_image"] == f"exec:{url}"
        assert harness.container_alive()
        assert "data.txt" in harness.workdir_listing() and "new.py" in harness.prompt()
        assert harness.container_run(["sh", "-c", "exit 3"]) == (3, b"")
        assert harness.container_exec(["cat", str(workdir / "data.txt")]) == "v1\n"
        assert harness.shell("echo $((1+1))") == "2\n"
        assert harness.read_container_file(str(workdir / "new.py")) == "x = 2\n"
        snapshot = harness.file_snapshot() or {}
        assert {Path(p).name for p in snapshot} == {"data.txt", "new.py"}
        harness.on_llm_call([])
        (workdir / "data.txt").write_text("v2\n")
        assert "(modified)" in harness.changed_files_report()
        done = {"success": True, "summary_in_html": "done"}
        verifier_path = tmp_path / "kiss-verify.sh"
        assert not harness.on_tool_call("finish", done).allowed
        verifier_path.write_text("test -f " + shlex.quote(str(workdir / "new.py")) + "\n")
        assert harness.on_tool_call("finish", done) == ALLOW
        # no daemon to read through: the checks run with the environment's
        # own bash, and the checkpoint's digest comes from its sha256sum
        assert harness.host_shell is None
        archive = tmp_path / "kiss-checkpoints" / "verified.tar"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        assert harness.checkpoint_digests["verified"] == digest
        # a rollback brings the tree back as it was checkpointed
        (workdir / "data.txt").write_text("v3\n")
        assert harness.rollback("verified") == f"Restored {workdir} from checkpoint 'verified'."
        assert (workdir / "data.txt").read_text() == "v2\n"
        archive.write_bytes(b"forged")
        assert harness.rollback("verified").endswith("archive altered since it was saved")
        archive.unlink()
        assert harness.rollback("verified").endswith("archive missing")
        assert harness.digest_of(str(archive)) is None
        verifier_path.write_text("sleep 10\n")
        slow_verdict = harness.on_tool_call("finish", done)
        assert not slow_verdict.allowed and "did not finish" in slow_verdict.text
        # the process snapshot and the test-context grep go through the bridge too
        # (the environment is this host, so the snapshot lists this very process)
        assert os.getpid() in {pid for pid, _start in harness.process_snapshot()}
        from kiss.agents.seas.coding import coding_test_context

        (workdir / "test_data.py").write_text("def test_x():\n    helper\n")
        assert coding_test_context.find_referencing_tests(
            harness.shell, str(workdir), ["helper"], str(workdir / "lib.py")) == [
            ("test_data.py", 1)]
    finally:
        _close(server, loop)
    # the server is gone: the harness reports it instead of hanging
    assert not harness.container_alive()
    assert harness.container_run(["true"]) is None


@linux_only
def test_exec_bridge_kills_the_command_tree(tmp_path: Path) -> None:
    """A timed-out or stopped command's whole process tree is killed on the environment."""
    loop = _Loop()
    server = ExecBridgeServer(LocalEnvironment(str(tmp_path)), loop.loop)
    late, pidfile = tmp_path / "late", tmp_path / "pid"
    try:
        assert exec_bridge.bridge_exec(server.url, _late_toucher(late, pidfile), 1) == (None, "")
        assert wait_for(partial(_dead, pidfile), 10) and not late.exists()
        # a Stop while a command runs ends the wait at once and kills the command
        pidfile.unlink()
        stop = _stop_when(pidfile.exists)
        started = time.monotonic()
        stopped = exec_bridge.bridge_exec(server.url, _late_toucher(late, pidfile), 30, stop)
        assert stopped == (-1, "Killed: the task was stopped.")
        assert time.monotonic() - started < 5
        assert wait_for(partial(_dead, pidfile), 10) and not late.exists()
    finally:
        _close(server, loop)


def _launch_pending(server: ExecBridgeServer) -> bool:
    """Whether *server* holds a pending launch."""
    with server.lock:
        return bool(server.pending)


def _exec_cleaned_up(server: ExecBridgeServer) -> bool:
    """Whether *server* holds no pending launch and no cancelled token.

    Read under ``server.lock``: the bridge drops ``pending`` and then the
    cancelled token in one locked step, so an unlocked look at an empty
    ``pending`` alone could land between the two.
    """
    with server.lock:
        return not server.pending and not server.cancelled


@linux_only
def test_bridge_kill_covers_a_launch_the_provider_delays(tmp_path: Path) -> None:
    """A Stop during a pending launch: the command is killed once it starts, or never starts."""

    class DelayedLaunch(LocalEnvironment):
        """Launches a command marked ``DELAY-ME`` only once :attr:`gate` is set.

        The wait is bounded: it runs on the loop's default executor, whose
        threads are joined at interpreter exit, and a gate a failed
        assertion left unset would otherwise park that exit for ever.
        """

        def __init__(self, cwd: str) -> None:
            super().__init__(cwd)
            self.gate = threading.Event()

        async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> Any:
            if "DELAY-ME" in command:
                await asyncio.to_thread(self.gate.wait, 30)
            return await super().exec(command, timeout_sec)

    environment = DelayedLaunch(str(tmp_path))
    loop = _Loop()
    server = ExecBridgeServer(environment, loop.loop)
    late = tmp_path / "late"
    try:
        # the Stop fires once the bridge holds the launch (a timer could fire before the
        # ``/exec`` is read, which is the reordered case pinned at the end)
        stop = _stop_when(partial(_launch_pending, server))
        stopped = exec_bridge.bridge_exec(
            server.url, f"sleep 6; {_touch(late)} # DELAY-ME", 30, stop)
        assert stopped == (-1, "Killed: the task was stopped.")
        assert not late.exists() and server.pending  # the launch is still pending
        environment.gate.set()  # the provider launches the command now: the reaper gets it
        # the exec ends (the launch leaves ``pending``) only once its tree is dead: had the
        # reaper missed it, the shell would have touched ``late`` before ending on its own,
        # within the wait's 10 s.  The 6 s (as in :func:`_late_toucher`) cover the
        # reaper's next sweep, a host-wide ``/proc`` scan in its own subprocess: about 1 s
        # idle, up to 4.2 s with 100 busy loops on an 80-core box, where a 2 s touch always
        # landed first (and once under a parallel full-suite run, 2026-10-29).
        assert wait_for(partial(_exec_cleaned_up, server), 10)
        assert not late.exists()
        # /kill before /exec (request threads reordered): the command never runs
        assert exec_bridge._request(server.url, "/kill", {"token": "t1"}, 10) == {"ok": True}
        reply = exec_bridge._request(
            server.url, "/exec", {"command": _touch(late), "timeout": 5, "token": "t1"}, 10)
        assert reply == {"exit_code": -1, "output": ""} and not late.exists()
        assert not server.cancelled
    finally:
        environment.gate.set()  # a launch an assertion above left pending ends now
        _close(server, loop)


def test_bridge_server_rejects_bad_requests_and_reports_a_broken_environment() -> None:
    import urllib.error
    import urllib.request

    class BrokenEnvironment:
        async def exec(self, command: str, timeout_sec: int | None = None) -> Any:
            raise RuntimeError("sandbox gone")

    loop = _Loop()
    server = ExecBridgeServer(BrokenEnvironment(), loop.loop)
    try:
        assert exec_bridge.bridge_exec(server.url, "true", 5) == (
            255, "exec failed: RuntimeError: sandbox gone")
        for path, data in (("/nope", None), ("/nope", b"{}"), ("/exec", b"not json"),
                           ("/exec", b'{"timeout": 1}')):
            request = urllib.request.Request(server.url + path, data=data)
            with pytest.raises(urllib.error.HTTPError) as info:
                urllib.request.urlopen(request, timeout=5)
            with info.value:  # the error carries the response: close it
                assert info.value.code in (400, 404)
    finally:
        server.close()
        loop.close()
    # a server that answers something other than the protocol's reply object
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Garbage(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            # Consume the body: closing a socket with unread input sends a
            # reset, which macOS reports to the client as ECONNRESET
            # instead of the reply.
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"[]")

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return None

    garbage = ThreadingHTTPServer(("127.0.0.1", 0), Garbage)
    threading.Thread(target=garbage.serve_forever, daemon=True).start()
    try:
        with pytest.raises(KISSError, match="invalid reply"):
            exec_bridge.bridge_exec(f"http://127.0.0.1:{garbage.server_address[1]}", "true", 5)
    finally:
        garbage.shutdown()
        garbage.server_close()


@local_shell
def test_harbor_agent_bridges_a_non_docker_environment(tmp_path: Path) -> None:
    """``SeaHarborAgent.run`` starts a bridge for a non-Docker environment and closes it after."""
    harbor_agent = pytest.importorskip("kiss.benchmarks.terminal_bench.harbor_agent")
    from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

    agent = harbor_agent.SeaHarborAgent(
        logs_dir=tmp_path, model_name="m", home=str(tmp_path / "no-daemon"),
        reviewer="none", reading_check="same", reasoning_effort="max",
    )
    context = AgentContext()
    asyncio.run(agent.run("do it", LocalEnvironment(str(tmp_path)), context))  # type: ignore[arg-type]
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["container"].startswith("exec:http://127.0.0.1:")
    assert config["workdir"] == str(tmp_path) and config["reviewer"] == ""
    # the side-model options reach the SEA config by their aliases: `same` is the
    # trial's model, `none` is off, the default `other` is left to the harness
    assert config["reading_check"] == "m" and config["blind_verifier"] == ""
    assert config["model_config"] == {"reasoning_effort": "max"}
    # no daemon: the trial records the failure, and the bridge is closed afterwards
    assert context.metadata is not None and context.metadata["error"]
    assert context.cost_usd == 0.0
    assert not exec_bridge.bridge_alive(config["container"][len("exec:"):])
    same = harbor_agent.SeaHarborAgent(logs_dir=tmp_path / "s", model_name="m",
                                       home=str(tmp_path / "no-daemon"))
    # a reused trial directory: an earlier run's review is not charged to this run
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "trajectory.jsonl").write_text(
        json.dumps({"event": "review", "cost_usd": 0.25, "tokens": 25}) + "\n")
    reused = AgentContext()
    asyncio.run(same.run("do it", LocalEnvironment(str(tmp_path)), reused))  # type: ignore[arg-type]
    config = json.loads((tmp_path / "s" / "config.json").read_text())
    assert config["reviewer"] == config["reading_check"] == harbor_agent.OTHER_FAMILY
    assert config["blind_verifier"] == "" and config["model_config"] is None
    assert reused.cost_usd == 0.0 and reused.metadata is not None
    assert reused.metadata["review_cost_usd"] == 0.0


@local_shell
def test_harbor_agent_closes_the_bridge_when_the_setup_fails(tmp_path: Path) -> None:
    """An environment whose first ``exec`` fails: ``run`` raises and leaves no bridge thread."""
    harbor_agent = pytest.importorskip("kiss.benchmarks.terminal_bench.harbor_agent")
    from harbor.models.agent.context import AgentContext  # type: ignore[import-untyped]

    class Unreachable(LocalEnvironment):
        async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> Any:
            raise ConnectionError("sandbox is gone")

    agent = harbor_agent.SeaHarborAgent(logs_dir=tmp_path, model_name="m",
                                        home=str(tmp_path / "no-daemon"))
    with pytest.raises(ConnectionError, match="sandbox is gone"):
        asyncio.run(agent.run("do it", Unreachable(str(tmp_path)), AgentContext()))  # type: ignore[arg-type]
    assert wait_for(_no_server_thread, 5)  # the server thread ends right after shutdown()
    assert not (tmp_path / "config.json").exists()

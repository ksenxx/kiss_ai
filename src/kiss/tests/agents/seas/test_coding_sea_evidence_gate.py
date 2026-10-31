# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests of the coding SEA's evidence gate and exec bridge.

The gate (:meth:`ContainerHarness.finish_verdict`) runs the agent's own
verifier in the trial container, lets a hedged summary through to the
agent's evidence gate, and asks a reviewer model once; the exec bridge
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
import itertools
import json
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.seas.coding import coding_sea, quoted_evidence
from kiss.agents.sorcar import exec_bridge
from kiss.agents.sorcar.container_jobs import ContainerJobs
from kiss.agents.sorcar.decide_tool import decisions_tool_available
from kiss.agents.sorcar.docker_tools import DockerTools
from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT
from kiss.benchmarks.terminal_bench.exec_bridge_server import SERVER_THREAD_NAME, ExecBridgeServer
from kiss.core.kiss_error import KISSError
from kiss.core.models.model_info import get_fast_model
from kiss.core.tool_verdict import ALLOW, refuse
from kiss.tests.agents.seas.test_coding_sea import _live_container
from kiss.tests.conftest import posix_only

#: The tests that run commands on :class:`LocalEnvironment` (``sh`` on this host).
local_shell = posix_only("the bridged local environment runs sh commands")
#: The tagged-tree kill (:func:`kiss.agents.sorcar.docker_manager.kill_tagged_tree_script`)
#: finds the tree through ``/proc/<pid>/environ``, as does :func:`_dead`: the
#: environments the bridge serves are Linux sandboxes.
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
        missing = refuse(coding_sea.VERIFIER_MISSING_VERDICT.format(path=VERIFIER))
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
        assert _wait_until(partial(_dead_in, live, "/tmp/slow.pid"))
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
        # "verified", then a hedged summary (an open reading included) is refused
        # once by the evidence gate; no review is configured, so the next finish passes
        passing = _harness(tmp_path, container=live.id, verifier_timeout=10)
        passing.on_llm_call([])
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        hedged = {"success": True, "summary_in_html": "<p>Should work; not verified. If the "
                                                      "grader wants more, that is a judgement "
                                                      "call.</p>"}
        refused = passing.on_tool_call("finish", hedged)
        assert not refused.allowed
        assert '"not verified", "should work", "if the grader", "judgement call"' in refused.text
        assert passing.last_verifier_run[1] == 0
        assert passing.checkpoints == ["verified"]
        archive = f"{coding_sea.CHECKPOINT_DIR}/verified.tar"
        assert live.exec_run(["test", "-f", archive]).exit_code == 0
        assert passing.on_tool_call("finish", hedged) == ALLOW
        # an edit after the last command: refused once unless the verifier passes
        edited = _harness(tmp_path, container=live.id, verifier_timeout=10, workdir="/app/")
        edited.on_llm_call([])
        _write_in(live, coding_sea.LEDGER_PATH, "- out.txt: exactly hello, no newline")
        write = {"file_path": "/app/out.txt", "content": "hello"}
        assert edited.on_tool_call("Write", write) == ALLOW
        assert edited.evidence.unchecked_change
        assert edited.on_tool_call("finish", done) == ALLOW  # the verifier checked the final state
        assert not edited.evidence.unchecked_change
        _write_in(live, VERIFIER, "exit 1")
        assert edited.on_tool_call("Edit", {"file_path": "/app/out.txt", "old_string": "h",
                                            "new_string": "H"}) == ALLOW
        failing = edited.on_tool_call("finish", done)
        assert not failing.allowed and "exited 1" in failing.text
        assert edited.evidence.unchecked_change
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
        assert harness.evidence.unchecked_change
        listing = live.exec_run(["sh", "-c", "cat /app/out.txt; cat /app/sub/k.txt; ls /app"])
        assert listing.output == b"hellokeep\nout.txt\nsub\n"
        assert harness.rollback("nope") == "Error: unknown checkpoint 'nope'. Known: v1."
        # the verified checkpoint: taken when both verifiers pass, whatever the
        # later checks say; a run ending without an accepted finish is restored
        # to it when the tree then fails the verifier
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        hedged = {"success": True, "summary_in_html": "<p>untested</p>"}
        assert not harness.on_tool_call("finish", hedged).allowed
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
        def bash_fn(command: str, description: str, timeout_seconds: float = 30,
                    max_output_chars: int = 4000) -> str:
            return str(live.exec_run(["sh", "-c", command]).output.decode())

        jobs = ContainerJobs(bash_fn)
        job_id, started = jobs.start("while true; do printf broken > /app/out.txt; sleep 0.1; done")
        assert job_id is not None, started
        assert _wait_until(partial(_holds, live, "/app/out.txt", b"broken"))
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
        assert _wait_until(partial(_holds, live, "/app/out.txt", b"broken"))
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


def test_heredoc_may_create_but_not_rewrite_a_file(tmp_path: Path) -> None:
    """A here-document to a new path runs; one to an existing file is refused in favour of Edit."""
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
            verdict = harness.on_tool_call("Bash", {"command": command})
            rewrite = refuse(HEREDOC_VERDICT.format(path="/app/out.txt"))
            assert verdict == rewrite, command
    finally:
        live.remove(force=True)


def test_gate_reviews_the_work_and_holds_the_findings(tmp_path: Path) -> None:
    """A reviewer's findings must be answered by a change or by evidence; the diff is re-reviewed.

    Round 1: the reviewer finds the file wrong (a real model call).  A finish
    that neither changes a file nor quotes evidence is refused by the
    decisions model's judgement, no review spent.  A finish after an edit
    gets a re-review scoped to the files changed since; ``max_reviews``
    bounds the rounds.  A reviewer whose first answer is cut before its
    verdict (``max_tokens`` override) is retried at the provider's defaults.
    """
    if not decisions_tool_available():
        pytest.skip("no decisions model configured")
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, reviewer=get_fast_model(), max_reviews=2)
        harness.on_llm_call([])
        # the agent writes something else than the task asks for
        assert live.exec_run(["sh", "-c", "printf goodbye > /app/out.txt"]).exit_code == 0
        _write_in(live, VERIFIER, "test -s /app/out.txt")
        _write_in(live, coding_sea.LEDGER_PATH, "- 'containing exactly hello': any greeting")
        # the commands the summaries below quote were run (the quote gate checks that)
        _turn(harness, [("Bash", {"command": "test -s /app/out.txt"}),
                        ("Bash", {"command": "cat /app/out.txt"})], ["", "goodbye"])
        # The summary claims the asked content outright: a bare "`test -s` passes"
        # reads as a weaker rule to the open-reading gate, whose live score for it
        # sat at 0.58-0.70 against the 0.7 threshold and now and then refused first.
        done = {"success": True,
                "summary_in_html": "<p>Wrote /app/out.txt with the requested content; "
                                   "`test -s /app/out.txt` passes.</p>"}
        review = harness.on_tool_call("finish", done)
        assert not review.allowed and review.text.startswith("Not finished: an independent review")
        assert "out.txt" in review.text and "Answer every numbered finding" in review.text
        logged = _events(tmp_path, "review")
        assert len(logged) == 1 and logged[0]["model"] == get_fast_model()
        assert logged[0]["cost_usd"] > 0 and logged[0]["tokens"] > 0 and logged[0]["findings"]
        assert logged[0]["round"] == 1 and not logged[0]["failed"]
        assert logged[0]["last_call"]["stop_reason"] and logged[0]["last_call"]["output_tokens"] > 0
        assert "VERDICT: NOT DONE" in logged[0]["answer"].upper()
        assert harness.open_findings and harness.open_findings[0].startswith("1")
        assert coding_sea.review_usage(tmp_path / "trajectory.jsonl") == (
            logged[0]["cost_usd"], logged[0]["tokens"])
        # the reviewer saw the changed file and the ledger
        report = harness.changed_files_report()
        assert "/app/out.txt (modified)" in report and "goodbye" in report
        assert f"{VERIFIER}" not in report  # outside the workdir
        # no change, no evidence: the summary settles the statement by a weaker rule
        # ("any greeting is fine"), which the open-reading gate refuses once; then
        # the open finding is unanswered, refused without a review
        argued = {"success": True, "summary_in_html": "<p>The reviewer misread the task; any "
                                                      "greeting is fine. Done.</p>"}
        weaker = harness.on_tool_call("finish", argued)
        assert weaker == refuse(coding_sea.OPEN_READING_VERDICT.format(verifier=VERIFIER))
        unanswered = harness.on_tool_call("finish", argued)
        assert not unanswered.allowed and unanswered.text.startswith(
            "Not finished: these findings of the review were not answered")
        assert len(_events(tmp_path, "review")) == 1
        assert _events(tmp_path, "findings_answered")[-1]["answered"] == [False] * len(
            harness.open_findings)
        # a change since the review: the second (and last) review covers the diff
        assert harness.on_tool_call("Write", {"file_path": "/app/out.txt", "content": "hello"}) \
            == ALLOW
        assert live.exec_run(["sh", "-c", "printf hello > /app/out.txt"]).exit_code == 0
        fixed = {"success": True, "summary_in_html": "<p>1. Rewrote /app/out.txt: `cat "
                                                     "/app/out.txt` prints hello.</p>"}
        second = harness.on_tool_call("finish", fixed)
        logged = _events(tmp_path, "review")
        assert len(logged) == 2 and logged[1]["round"] == 2
        assert second == ALLOW or second.text.startswith("Not finished: an independent review")
        # max_reviews reached: a third finish is not reviewed again, and with no
        # re-review to come a change alone no longer answers an open finding
        third = harness.on_tool_call("finish", fixed)
        assert third == ALLOW or third.text.startswith("Not finished: these findings")
        assert len(_events(tmp_path, "review")) == 2
        # a review that fails twice (here: a reviewer model that does not exist) is
        # neither approval nor rejection: the finish is refused while another review
        # is possible, and goes through at the cap
        failing = _harness(tmp_path, container=live.id, reviewer="no-such-model-xyz",
                           max_reviews=2)
        failing.on_llm_call([])
        assert failing.on_tool_call("finish", fixed) == refuse(coding_sea.REVIEW_FAILED_VERDICT)
        assert failing.on_tool_call("finish", fixed) == ALLOW
        failed = [e for e in _events(tmp_path, "review") if e["model"] == "no-such-model-xyz"]
        assert [e["failed"] for e in failed] == [True, True] and all(e["error"] for e in failed)
        # a reviewer cut off before its verdict (5 output tokens) is retried at the
        # provider's defaults; the retry is logged and the review counted once
        short = _harness(tmp_path, container=live.id, reviewer=get_fast_model(),
                         model_config={"max_tokens": 5})
        short.on_llm_call([])
        verdict = short.on_tool_call("finish", fixed)
        retries = _events(tmp_path, "review_retry")
        assert len(retries) == 1 and not coding_sea.REVIEW_VERDICT_LINE.search(retries[0]["answer"])
        last = _events(tmp_path, "review")[-1]
        assert last["round"] == 1 and not last["failed"] and "VERDICT" in last["answer"].upper()
        assert last["cost_usd"] > retries[0]["last_call"]["cost"]  # both calls are charged
        assert verdict == ALLOW or verdict.text.startswith("Not finished: an independent review")
    finally:
        live.remove(force=True)


def test_blind_verifier_is_written_from_the_statement_and_run_by_the_gate(tmp_path: Path) -> None:
    """A second agent writes the independent check into the container; the gate runs it.

    One real agentic run of the fast model with a container shell: the
    script it delivers lands at ``BLIND_VERIFIER_PATH``, is run at every
    gated finish, and a failing check refuses the finish with its output.
    """
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
            assert not verdict.allowed and verdict.text.startswith(
                "Not finished: an independent check written from the task statement alone")
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
        # the author already ran (and delivered a strict check, kept on the host)
        harness.blind_written = True
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


def test_summary_quotes_must_come_from_a_tool_result(tmp_path: Path) -> None:
    """A finish quoting text no tool returned is refused; the provenance goes to the reviewer."""
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id, verifier_timeout=10)
        harness.on_llm_call([{"role": "user", "content": "task"}])
        # the open-reading gate (a decisions-model judgement of the summary's prose) has
        # had its one refusal: the verdicts below are the quote check's alone
        harness.open_reading_refused = True
        statement = harness.task_prompt  # "Create /app/out.txt containing exactly `hello`."
        # one turn: a real command, an echo of the claim, a Read, a refused finish
        _turn(harness, [
            ("Bash", {"command": "cd /app && cat out.txt && wc -c out.txt"}),
            ("Bash", {"command": "echo 'All 27 checks passed'"}),
            ("Read", {"file_path": "/app/out.txt"}),
            ("run_commands_parallel", {"commands": json.dumps(["bash /tmp/kiss-verify.sh"])}),
            ("Write", {"file_path": "/tmp/fake.sh", "content": "echo '27 passed, 0 failed'"}),
            ("Bash", {"command": "bash /tmp/fake.sh"}),
            ("finish", {"success": True, "summary_in_html": "<p>done</p>"}),
        ], ["hello\n5 out.txt", "All 27 checks passed", "1: hello\n(end of file: 5 bytes)",
            "[1] bash /tmp/kiss-verify.sh\nexit 0\nverifier output: out.txt is hello",
            "Successfully wrote 26 characters to /tmp/fake.sh", "27 passed, 0 failed",
            "REFUSED: " + coding_sea.VERIFIER_MISSING_VERDICT.format(path=VERIFIER)])
        assert _events(tmp_path, "tool_call")[-1]["blocked"]  # the finish ran before the verifier
        _write_in(live, VERIFIER, "test \"$(cat /app/out.txt)\" = hello")
        summary = ("<p>Checked: <code>cd /app &amp; cat out.txt</code> and <code>cat out.txt "
                   "&amp;&amp; wc -c out.txt</code> print <pre>$ cat out.txt &amp;&amp; wc -c "
                   "out.txt\nhello\n5 out.txt</pre>; "
                   "<samp>verifier output: out.txt is hello</samp>; the statement says "
                   "<q>containing exactly `hello`</q> (<code>containing exactly</code>). "
                   "The test run reported <code>All 27 checks passed</code> and "
                   "<pre>12 passed in 0.34s\n... 0 failed, 0 errors</pre>; my own test script "
                   "printed <code>27 passed, 0 failed</code>; "
                   "a short quote (<code>/app/out</code>) is not checked.</p>")
        verdict = harness.on_tool_call("finish", {"success": True, "summary_in_html": summary})
        assert not verdict.allowed and verdict.text.startswith(
            "Not finished: summary_in_html quotes text that no tool result of this run contains")
        # the echoed claim, the invented test output and a command never run are named;
        # the real output, the command run, its second segment, the statement are not
        listed = [line for line in verdict.text.splitlines() if line.startswith('- "')]
        assert listed == ['- "cd /app & cat out.txt"', '- "All 27 checks passed"',
                          '- "12 passed in 0.34s"', '- "0 failed, 0 errors"']
        assert harness.quote_provenance["cat out.txt && wc -c out.txt"].startswith(
            "a command you ran: cd /app && cat out.txt")
        assert harness.quote_provenance["verifier output: out.txt is hello"] == (
            "output of: bash /tmp/kiss-verify.sh")
        assert harness.quote_provenance["containing exactly"] == "the task statement"
        assert "containing exactly `hello`" in statement
        # a script the agent wrote and ran is a source, marked as the agent's own
        assert harness.quote_provenance["27 passed, 0 failed"] == (
            "output of: bash /tmp/fake.sh (names a file the engineer wrote: /tmp/fake.sh)")
        logged = _events(tmp_path, "quoted_evidence")[-1]
        assert logged["outputs"] == 6 and logged["quotes"] == 9 and len(logged["unsupported"]) == 4
        # the checks passed, so the verified checkpoint was taken before the refusal
        assert harness.checkpoints == ["verified"]
        # the refusal comes back to the model as a tool result that repeats the
        # quotes it refused: it is no source, paired with its finish or not
        refusal = "REFUSED: " + verdict.text
        _turn(harness, [], [refusal])  # paired with the refused finish
        again = harness.on_tool_call("finish", {"success": True, "summary_in_html": summary})
        assert again == verdict
        _turn(harness, [], [refusal, refusal])  # two results for one call: unpaired
        assert harness.on_tool_call("finish", {"success": True, "summary_in_html": summary}) \
            == verdict
        # a separator inside a quoted argument starts no command; a command's own
        # words are never its output; <q> and <blockquote> are quotes too
        _turn(harness, [("Bash", {"command": "true 'prefix; All 27 checks passed'"})], [""])
        smuggled = {"success": True, "summary_in_html": "<p><q>All 27 checks passed</q> and "
                                                         "<blockquote>12 passed in 0.34s"
                                                         "</blockquote></p>"}
        smuggle = harness.on_tool_call("finish", smuggled)
        assert not smuggle.allowed and '- "All 27 checks passed"' in smuggle.text
        assert '- "12 passed in 0.34s"' in smuggle.text
        # a Read's content is a tool result too, and the provenance names it for the reviewer
        honest = ("<p>1. Read shows <pre>1: hello\n(end of file: 5 bytes)</pre>; my script "
                  "printed <code>27 passed, 0 failed</code>.</p>")
        assert harness.on_tool_call("finish", {"success": True, "summary_in_html": honest}) == ALLOW
        report = coding_sea.quotes_report(harness.quote_provenance)
        assert '- "(end of file: 5 bytes)": Read of /app/out.txt' in report
        assert '(names a file the engineer wrote: /tmp/fake.sh)' in report
        # a turn whose results do not pair with its calls (here: the finishes
        # above, one Bash, and nine results) still supports quotes, unlabelled
        _turn(harness, [("Bash", {"command": "ls /app"})],
              ["out.txt", "stray second result"] + [f"result {n}" for n in range(7)])
        odd = {"success": True, "summary_in_html": "<p><code>stray second result</code></p>"}
        assert harness.on_tool_call("finish", odd) == ALLOW
        assert harness.quote_provenance == {"stray second result": "a tool result"}
        assert coding_sea.quotes_report({}) == "(the summary quotes nothing checkable)"
        # a run whose trajectory shows no tool result at all supports no quote
        bare = _harness(tmp_path, container=live.id, verifier_timeout=10, workdir="/app/")
        bare.trajectory_path = tmp_path / "bare.jsonl"
        bare.open_reading_refused = True
        bare.on_llm_call([])
        invented = {"success": True, "summary_in_html": "<p><pre>12 passed in 0.34s</pre></p>"}
        assert not bare.on_tool_call("finish", invented).allowed
        assert _events(tmp_path, "quoted_evidence", "bare.jsonl")[-1]["outputs"] == 0
        assert bare.on_tool_call("finish", {"success": True, "summary_in_html": "<p>x</p>"}) \
            == ALLOW
        # where commands start in a command line: after a separator outside quotes
        # and not escaped, and after an opening parenthesis
        claim = "All 27 checks passed"
        assert not quoted_evidence.command_supports(f"true 'a; {claim}'", claim)
        assert not quoted_evidence.command_supports(f"true \"a && {claim}\"", claim)
        assert not quoted_evidence.command_supports(f"true a\\; {claim}", claim)
        assert quoted_evidence.command_supports(f"true 'a' ; {claim}", claim)
        assert quoted_evidence.command_supports(f"(cd /app && {claim})", claim)
        assert quoted_evidence.command_starts("a | b & c || d") == [0, 3, 7, 12]
        # the command strings of a shell call's recorded arguments
        assert coding_sea.shell_commands(None) == []
        assert coding_sea.shell_commands({"commands": "[not json"}) == ["[not json"]
        assert coding_sea.shell_commands({"commands": ["a", 3]}) == ["a"]
        assert coding_sea.shell_commands({"commands": 5}) == []
    finally:
        live.remove(force=True)


def test_inventory_is_appended_at_a_context_restart(tmp_path: Path) -> None:
    """The first message of a continuation session gets the harness's machine-written inventory."""
    live = _live_container("python:3.11-slim", "mkdir -p /app && printf hello > /app/out.txt")
    if live is None:
        pytest.skip("Docker is not available")
    try:
        harness = _harness(tmp_path, container=live.id)
        harness.on_llm_call([{"role": "user", "content": "task"}])
        _write_in(live, coding_sea.LEDGER_PATH, "- out.txt: exactly hello")
        assert harness.on_tool_call("Read", {"file_path": "out.txt"}) == ALLOW
        assert harness.on_tool_call("Read", {"file_path": "/app/missing.txt"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "ls /app"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "false"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "python3 train.py"}) == ALLOW
        assert harness.on_tool_call("Bash", {"command": "make"}) == ALLOW
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
        # the continuation prompt (string content) gets the inventory
        start = [{"role": "user", "content": "task\n\n# Task Progress (Continuation 1)\n..."}]
        harness.on_llm_call(start)
        text = start[0]["content"]
        assert "# Inventory (written by the harness at this restart)" in text
        assert "5 5d41402a /app/out.txt" in text and "- - /app/missing.txt" in text
        assert "Files written with Edit/Write: /app/new.py" in text
        assert "  1: false" in text and "  timeout: python3 train.py" in text
        assert "  0: ls /app" in text
        assert f"Present: {coding_sea.LEDGER_PATH}\n" in text
        assert _events(tmp_path, "inventory")[-1]["turn"] == 5
        # text-block content (the Responses API shape) is extended with a text block
        blocks: list[dict[str, Any]] = [{"role": "user", "content": [
            {"type": "input_text", "text": "# Task Progress (Continuation 2)"}]}]
        _write_in(live, VERIFIER, "true")
        harness.checkpoints.append("v1")
        harness.on_llm_call(blocks)
        assert blocks[0]["content"][-1]["type"] == "input_text"
        added = blocks[0]["content"][-1]["text"]
        assert f"Present: {coding_sea.LEDGER_PATH}, {VERIFIER}" in added
        assert "Checkpoints: v1" in added
        bare = _harness(tmp_path, container=live.id, workdir="/nowhere")
        bare.on_llm_call([])
        assert "Present: none of the ledger/verifiers" not in bare.inventory()  # ledger exists
        assert live.exec_run(["rm", coding_sea.LEDGER_PATH, VERIFIER]).exit_code == 0
        assert "Present: none of the ledger/verifiers" in bare.inventory()
        assert coding_sea.user_text({"role": "assistant", "content": "x"}) == ""
        assert coding_sea.user_text({"role": "user", "content": 3}) == ""
        coding_sea.append_user_text({"role": "user", "content": None}, "n")  # nothing to extend
    finally:
        live.remove(force=True)


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


STATEMENT = ("A log rotation daemon is not working. Symptoms: the current log is truncated "
             "while a writer still holds it open; rotation stalls when one of the watched "
             "directories sits on a slow disk; archives older than the retention period are "
             "kept forever. Fix the bugs.")
OPEN = ("<p>Fixed the truncation and the retention. Choices the symptoms didn't settle: I read "
        "the stalls symptom as the scan order being wrong. I added nothing for a directory on "
        "a slow disk, because NOTES.md requires every directory to be scanned in order.</p>")
CLOSED = ("<p>Fixed three bugs. 1. the file was truncated before the rename; rotation now "
          "renames first. 2. a directory on a slow disk no longer holds the others back: each "
          "directory is scanned by its own worker with a deadline (checked: a 5 s stat delay on "
          "one directory leaves the other two rotating on time). 3. the retention cutoff was "
          "compared against the archive's own mtime, not its creation time. "
          "Checks: bash /tmp/kiss-verify.sh passes, 9 scenarios.</p>")
PLANNER = ("A warehouse's picking route planner has been producing incorrect results since "
           "the last update: walking distances that don't match a manual count, and routes "
           "that cleared carts known to be over the weight limit. Fix the planner so that "
           "routes.py writes a correct route plan for today's orders.")
INFEASIBLE = ("<p>I fixed the planner (aisle distance sign, cart tare counted, load no longer "
              "capped at the rack capacity). The plan it writes is marked not feasible: with "
              "the bugs fixed, today's orders cannot be picked by any of the 24 candidate "
              "routes; the best one ends 2.3 kg over the 150 kg cart limit. Checks: "
              "/tmp/kiss-verify.sh passes.</p>")
PLANNED = ("<p>I fixed the planner: aisle distance sign, cart tare counted, and the items "
           "loaded at each stop are now bounded by the cart's remaining capacity (the stop-2 "
           "load drops to 31 kg, leaving 12 kg of headroom). The plan is feasible on all five "
           "stops, route_feasible true, 48.7 min including the return leg. Checks: "
           "/tmp/kiss-verify.sh passes, every stop's weight and distance within limits.</p>")


def test_a_summary_that_leaves_a_case_open_is_refused_once(tmp_path: Path) -> None:
    """The decisions model reads the statement and the summary; an admitted gap costs one finish."""
    if not decisions_tool_available():
        pytest.skip("no decisions model configured")
    assert coding_sea.leaves_a_case_open(STATEMENT, OPEN) >= coding_sea.OPEN_READING_THRESHOLD
    assert coding_sea.leaves_a_case_open(STATEMENT, CLOSED) < coding_sea.OPEN_READING_THRESHOLD
    # the asked outcome missed ("not feasible", "no tour passes") is an open case too
    assert coding_sea.leaves_a_case_open(PLANNER, INFEASIBLE) >= coding_sea.OPEN_READING_THRESHOLD
    assert coding_sea.leaves_a_case_open(PLANNER, PLANNED) < coding_sea.OPEN_READING_THRESHOLD
    harness = _harness(tmp_path, prompt=STATEMENT, container="none")
    closed = harness.open_reading_verdict({"summary_in_html": CLOSED})
    assert closed is None and not harness.open_reading_refused
    verdict = harness.open_reading_verdict({"summary_in_html": OPEN})
    assert verdict == coding_sea.OPEN_READING_VERDICT.format(verifier=VERIFIER)
    assert harness.open_reading_refused
    assert harness.open_reading_verdict({"summary_in_html": OPEN}) is None  # once per trial
    scores = [e["score"] for e in _events(tmp_path, "open_reading")]
    assert len(scores) == 2 and scores[0] < 0.7 <= scores[1]


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


def _wait_until(predicate: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Poll *predicate* every 20 ms until it holds (``True``) or *timeout* passes (``False``)."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)
    return True


def _stop_when(predicate: Callable[[], bool]) -> threading.Event:
    """Return a stop event a helper thread sets once *predicate* holds (or after 10 s).

    A Stop fired on a timer could land before the bridge has even read the
    command's ``/exec`` (the ``/kill``-before-``/exec`` ordering, pinned on
    its own below); firing on a predicate pins the ordering each test means.
    """
    stop = threading.Event()

    def fire() -> None:
        _wait_until(predicate)
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
    except FileNotFoundError:
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


def _late_toucher(late: Path, pidfile: Path) -> str:
    """A command whose background subshell's pid is recorded, then touches *late* after 6 s.

    Killing the tree is proven without waiting the 6 s out: once the
    recorded subshell is dead (:func:`_dead`) and *late* is absent, the
    touch can never land; a surviving subshell would have touched *late*
    before dying on its own, within :func:`_wait_until`'s 10 s.  The 6 s
    leave the bridge's tagged-tree sweep (a ``/proc`` scan in its own
    subprocess, after the 1 s command timeout) 5 s on a loaded CI box;
    at 2 s the touch landed first under a parallel full-suite run.  The
    pid lands in *pidfile* atomically, so the file's existence means
    the pid is readable.
    """
    record = f"echo $! > {pidfile}.tmp && mv {pidfile}.tmp {pidfile}"
    return f"(sleep 6; touch {late}) & {record}; wait"


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
    loop would raise on the reaper thread instead.
    """
    server.environment.kill_all()
    assert _wait_until(_no_reaper_thread)
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
        stopped = exec_bridge.bridge_exec(url, f"touch {marker}; sleep 20", 30, stop)
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
                f"touch {bash_marker}; sleep 20", "stopped", timeout_seconds=60
            ) == "Killed: the task was stopped.\n[exit code: -1]"
            assert time.monotonic() - started < 5
            parallel_marker = workdir / "parallel-started"
            bridge.stop_event = _stop_when(parallel_marker.exists)
            started = time.monotonic()
            report = bridge.run_commands_parallel(
                json.dumps([f"touch {parallel_marker}; sleep 20", "sleep 20"]), timeout_seconds=60
            )
            assert time.monotonic() - started < 5 and "stopped" in report.lower()
            bridge.stop_event = None
            # ``cat`` of a missing file exits 1 under both GNU and BSD coreutils
            failed = bridge.Bash("cat nope", "fail", timeout_seconds=5)
            assert "No such file" in failed and failed.endswith("[exit code: 1]")
            assert bridge.Bash("exit 9", "silent").strip() == "[exit code: 9]"
            slow = bridge.Bash("echo partial; sleep 5", "slow", timeout_seconds=1)
            assert slow.startswith("Error: command timed out after 1s")
            assert bridge.Bash("rm -rf /", "destructive").startswith("Blocked")
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
        exec_bridge.ExecBridge(f"exec:{url}").__enter__()
    with pytest.raises(KISSError, match="not answering"):
        with exec_bridge.ExecBridge(f"exec:{url}"):
            pass


@local_shell
@gnu_coreutils_only
def test_exec_bridge_runs_the_file_tools_and_the_harness(tmp_path: Path) -> None:
    """The file tools (built on Bash) and the harness work through the bridge."""
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
        verifier_path = Path(VERIFIER)
        # the verifier path is the SEA's contract, one per host: a concurrent session of
        # this test (another pytest run on the box) waits for the lock instead of clobbering it
        import fcntl

        with open(VERIFIER + ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                # a leftover of a session killed mid-test would pass the gate below
                verifier_path.unlink(missing_ok=True)
                assert not harness.on_tool_call("finish", done).allowed
                verifier_path.write_text("test -f " + str(workdir / "new.py") + "\n")
                assert harness.on_tool_call("finish", done) == ALLOW
                # no daemon to read through: the checks run with the environment's
                # own bash, and the checkpoint's digest comes from its sha256sum
                assert harness.host_shell is None
                archive = Path(coding_sea.CHECKPOINT_DIR) / "verified.tar"
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                assert harness.checkpoint_digests["verified"] == digest
                archive.write_bytes(b"forged")
                assert harness.rollback("verified").endswith("archive altered since it was saved")
                archive.unlink()
                assert harness.rollback("verified").endswith("archive missing")
                assert harness.digest_of(str(archive)) is None
                verifier_path.write_text("sleep 10\n")
                slow_verdict = harness.on_tool_call("finish", done)
                assert not slow_verdict.allowed and "did not finish" in slow_verdict.text
            finally:
                verifier_path.unlink(missing_ok=True)
        # the process snapshot and the test-context grep go through the bridge too
        assert harness.process_snapshot() is not None
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
        assert _wait_until(partial(_dead, pidfile)) and not late.exists()
        # a Stop while a command runs ends the wait at once and kills the command
        pidfile.unlink()
        stop = _stop_when(pidfile.exists)
        started = time.monotonic()
        stopped = exec_bridge.bridge_exec(server.url, _late_toucher(late, pidfile), 30, stop)
        assert stopped == (-1, "Killed: the task was stopped.")
        assert time.monotonic() - started < 5
        assert _wait_until(partial(_dead, pidfile)) and not late.exists()
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
        """Launches a command marked ``DELAY-ME`` only once :attr:`gate` is set."""

        def __init__(self, cwd: str) -> None:
            super().__init__(cwd)
            self.gate = threading.Event()

        async def exec(self, command: str, timeout_sec: int | None = None, **_: Any) -> Any:
            if "DELAY-ME" in command:
                await asyncio.to_thread(self.gate.wait)
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
            server.url, f"sleep 6; touch {late} # DELAY-ME", 30, stop)
        assert stopped == (-1, "Killed: the task was stopped.")
        assert not late.exists() and server.pending  # the launch is still pending
        environment.gate.set()  # the provider launches the command now: the reaper gets it
        # the exec ends (the launch leaves ``pending``) only once its tree is dead: had the
        # reaper missed it, the shell would have touched ``late`` before ending on its own,
        # within ``_wait_until``'s 10 s.  The 6 s (as in :func:`_late_toucher`) cover the
        # reaper's next sweep, a host-wide ``/proc`` scan in its own subprocess: about 1 s
        # idle, up to 4.2 s with 100 busy loops on an 80-core box, where a 2 s touch always
        # landed first (and once under a parallel full-suite run, 2026-10-29).
        assert _wait_until(partial(_exec_cleaned_up, server))
        assert not late.exists()
        # /kill before /exec (request threads reordered): the command never runs
        assert exec_bridge._post(server.url, "/kill", {"token": "t1"}, 10) == {"ok": True}
        reply = exec_bridge._post(
            server.url, "/exec", {"command": f"touch {late}", "timeout": 5, "token": "t1"}, 10)
        assert reply == {"exit_code": -1, "output": ""} and not late.exists()
        assert not server.cancelled
    finally:
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
        reviewer="none", reasoning_effort="max",
    )
    context = AgentContext()
    asyncio.run(agent.run("do it", LocalEnvironment(str(tmp_path)), context))  # type: ignore[arg-type]
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["container"].startswith("exec:http://127.0.0.1:")
    assert config["workdir"] == str(tmp_path) and config["reviewer"] == ""
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
    assert config["reviewer"] == "m" and config["model_config"] is None
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
    assert _wait_until(_no_server_thread, 5)  # the server thread ends right after shutdown()
    assert not (tmp_path / "config.json").exists()

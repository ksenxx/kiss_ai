# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/remember`` and ``/forget`` agents
(:mod:`kiss.agents.seas.remember.remember_sea`, :mod:`kiss.agents.seas.forget.forget_sea`)
and their storage layer (:mod:`kiss.agents.seas.agents_md`).

The tools write the real ``$KISS_HOME/AGENTS.md`` of the test session
(``KISS_HOME`` is a temporary directory, see ``conftest.py``).  The
agent-level tests run a real :class:`ChatSorcarAgent` ReAct loop against
the scripted local chat-completions server configured from each SEA's
methods, so the tools are really offered and executed, their replies
really reach the model as tool results, and the stored instruction
really reaches the next task's system prompt.  (The scripted model's
finish text is fixed, so what the model writes is not under test.)
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import Any

import pytest

from kiss.agents.seas import agents_md
from kiss.agents.seas.base.base_sea import BaseSea
from kiss.agents.seas.forget import forget_sea
from kiss.agents.seas.forget.forget_sea import ForgetSea
from kiss.agents.seas.remember import remember_sea
from kiss.agents.seas.remember.remember_sea import RememberSea
from kiss.agents.sorcar import sea_commands
from kiss.core.utils import read_bytes_waiting_for_writer
from kiss.tests.agents.seas.conftest import run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body


@pytest.fixture(autouse=True)
def _fresh_agents_md() -> Iterator[Path]:
    """Start and end every test without a ``AGENTS.md`` in the test KISS_HOME."""
    path = agents_md.agents_md_path()
    path.unlink(missing_ok=True)
    yield path
    path.unlink(missing_ok=True)


def _run(sea: BaseSea, prompt: str, script: list[bytes], work_dir: Path) -> tuple[Any, list]:
    """Run *sea* on *prompt* against the scripted model; return (parsed result, requests)."""
    run = sea_commands.evaluate_sea([sea], prompt)
    assert run.prompt == prompt
    return run_sea(run, script, work_dir, max_steps=6)


def test_sea_methods_follow_the_contract() -> None:
    """Both SEAs pin their run: own prompt, own tool, bash profile, no extras."""
    # ``system_prompt`` replaces the assembled prompt; ``tools`` appends.
    assert RememberSea().system_prompt("ASSEMBLED") == remember_sea.SYSTEM_PROMPT
    assert "`remember_instruction`" in remember_sea.SYSTEM_PROMPT
    assert RememberSea().tools([print]) == [
        print,
        remember_sea.remember_instruction,
        agents_md.list_instructions,
    ]
    assert ForgetSea().system_prompt("ASSEMBLED") == forget_sea.SYSTEM_PROMPT
    assert "`forget_instruction`" in forget_sea.SYSTEM_PROMPT
    assert "`list_instructions`" in forget_sea.SYSTEM_PROMPT
    assert ForgetSea().tools([]) == [forget_sea.forget_instruction, agents_md.list_instructions]
    for module, sea in ((remember_sea, RememberSea()), (forget_sea, ForgetSea())):
        assert_worker_contract(module, sea, {"tool_profile": "bash", "max_budget": 1.0})


def test_remember_creates_the_file_and_appends_bullets(_fresh_agents_md: Path) -> None:
    """The first instruction creates the file with a heading; later ones append."""
    path = _fresh_agents_md
    assert agents_md.list_instructions() == f"No instructions are stored in {path}."
    assert agents_md.read_instructions() == []

    reply = remember_sea.remember_instruction("Always reply in British English")
    assert reply == f"Remembered in {path}: Always reply in British English"
    assert path.read_text() == "# User instructions\n\n- Always reply in British English\n"

    # Newlines and runs of spaces collapse into one bullet line; a
    # leading bullet marker typed by the user is not doubled.
    reply = remember_sea.remember_instruction("-   Run   tests\nbefore   finishing  ")
    assert reply == f"Remembered in {path}: Run tests before finishing"
    assert path.read_text() == (
        "# User instructions\n\n- Always reply in British English\n- Run tests before finishing\n"
    )
    assert agents_md.list_instructions() == (
        "1. Always reply in British English\n2. Run tests before finishing"
    )


def test_remember_rejects_empty_and_duplicate_instructions(_fresh_agents_md: Path) -> None:
    """An empty instruction writes nothing; a repeat (any case/spacing) writes nothing."""
    path = _fresh_agents_md
    assert remember_sea.remember_instruction("  \n ") == (
        "Error: the instruction is empty; nothing was remembered."
    )
    assert not path.exists()
    remember_sea.remember_instruction("Use uv, not pip")
    before = path.read_text()
    assert remember_sea.remember_instruction("use  UV, not PIP") == (
        f"Already remembered in {path}: use UV, not PIP"
    )
    assert path.read_text() == before
    # A hand-written bullet with odd spacing is normalized before comparing too.
    path.write_text("-   Use   uv\n")
    assert remember_sea.remember_instruction("use uv") == f"Already remembered in {path}: use uv"
    assert path.read_text() == "-   Use   uv\n"


def test_bullet_markers_never_become_instructions(_fresh_agents_md: Path) -> None:
    """Marker-only text is empty; nested markers are stripped on both store and remove."""
    path = _fresh_agents_md
    for marker_only in ("- ", "*\n ", "+\t", "- - ", "-"):
        assert remember_sea.remember_instruction(marker_only).startswith("Error: "), marker_only
    assert not path.exists()
    assert remember_sea.remember_instruction("- - Nested rule") == (
        f"Remembered in {path}: Nested rule"
    )
    # A nested bullet the user wrote by hand is matched by its text alone.
    path.write_text(path.read_text() + "- - Hand written\n")
    assert agents_md.read_instructions() == ["Nested rule", "- Hand written"]
    assert forget_sea.forget_instruction("- - Nested rule") == (f"Forgot from {path}: Nested rule")
    assert forget_sea.forget_instruction("hand written") == (f"Forgot from {path}: - Hand written")
    assert path.read_text() == "# User instructions\n\n"


def test_edits_keep_foreign_bytes_and_crlf_endings(_fresh_agents_md: Path) -> None:
    """A cp1252 byte and CRLF endings in a hand-written file survive add and remove."""
    path = _fresh_agents_md
    original = b"# Mine\r\n\r\nProse with a cp1252 \x92 quote.\r\n- Old rule\r\n"
    path.write_bytes(original)
    assert remember_sea.remember_instruction("New rule").startswith("Remembered in ")
    assert path.read_bytes() == original + b"- New rule\r\n"
    assert agents_md.read_instructions() == ["Old rule", "New rule"]
    assert forget_sea.forget_instruction("old rule") == f"Forgot from {path}: Old rule"
    assert path.read_bytes() == (b"# Mine\r\n\r\nProse with a cp1252 \x92 quote.\r\n- New rule\r\n")


def test_edits_keep_every_other_line_byte_for_byte(_fresh_agents_md: Path) -> None:
    """Mixed endings, lone CRs and an unterminated last line are left as they are.

    Only the edited line changes: an added bullet gets the terminator of
    the file's last terminated line (after terminating a bare last line
    so the bullet starts a line of its own), and a removed bullet takes
    exactly its own terminator with it.
    """
    path = _fresh_agents_md
    mixed = b"# Mine\r\nProse \x92\n- Old rule\r\nTail without newline"
    path.write_bytes(mixed)
    remember_sea.remember_instruction("New rule")
    assert path.read_bytes() == mixed + b"\r\n- New rule\r\n"
    forget_sea.forget_instruction("Old rule")
    assert path.read_bytes() == b"# Mine\r\nProse \x92\nTail without newline\r\n- New rule\r\n"

    lone_cr = b"# Mine\rProse\r- Old rule\r"
    path.write_bytes(lone_cr)
    remember_sea.remember_instruction("New rule")
    assert path.read_bytes() == lone_cr + b"- New rule\r"
    assert agents_md.read_instructions() == ["Old rule", "New rule"]

    path.write_bytes(b"- Old rule\nUnrelated prose without trailing newline")
    forget_sea.forget_instruction("Old rule")
    assert path.read_bytes() == b"Unrelated prose without trailing newline"
    remember_sea.remember_instruction("Rule")
    assert path.read_bytes() == b"Unrelated prose without trailing newline\n- Rule\n"


def _in_a_child(scenario: str, timeout: float) -> tuple[subprocess.Popen[str], str]:
    """Run this module's function *scenario* in a child interpreter; return (reaped child, output).

    The concurrency scenarios' executor threads are joined when their
    interpreter exits, and ``shutdown(wait=False, cancel_futures=True)``
    only skips queued work: a worker wedged by a locking regression would
    hang the pytest process itself once the Future timeout fired.  So the
    deadline is the parent's: a child still running after *timeout*
    seconds is killed and reaped, and comes back with a non-zero
    ``returncode`` (0 means every assertion of the scenario held).  The
    child inherits ``KISS_HOME`` and so works on the session's AGENTS.md.
    """
    code = f"from {__name__} import {scenario}; {scenario}()"
    child = subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        output = child.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        child.kill()
        output = child.communicate()[0]
    return child, output


_STANDING = "- Always preserve this standing instruction"


def test_readers_never_see_a_partial_file_during_updates(_fresh_agents_md: Path) -> None:
    """A task reading AGENTS.md while it is rewritten gets the old or the new text.

    The reader repeats the exact ``perform_task`` read while a writer
    adds and removes a transient rule 200 times.  The reader announces
    each round before the writer starts it, takes at least one snapshot
    per round and keeps sampling until the round's writes are done: that
    guarantees the sample count, not that a snapshot overlaps a given
    write (the scheduler decides that); every snapshot must still
    contain the standing instruction (an in-place truncating write would
    show empty snapshots).  On Windows this also exercises the
    sharing-violation handling on both sides: the writer's replace must
    wait out the reader's open handle and the reader's open must wait
    out the in-flight rename.  The rounds run in a child interpreter
    (see :func:`_in_a_child`) on the file this process prepared.
    """
    remember_sea.remember_instruction(_STANDING)
    child, output = _in_a_child("_readers_scenario", timeout=240)
    assert child.returncode == 0, output
    assert _fresh_agents_md.read_text() == f"# User instructions\n\n{_STANDING}\n"


def _readers_scenario() -> None:
    """The reader/writer rounds of :func:`test_readers_never_see_a_partial_file_during_updates`."""
    path = agents_md.agents_md_path()
    standing = _STANDING
    rounds = 200
    ready = [Event() for _ in range(rounds)]  # ``ready[i]``: the reader reached round i
    written = [Event() for _ in range(rounds)]  # ``written[i]``: round i's writes are done

    def write() -> None:
        try:
            for i in range(rounds):
                assert ready[i].wait(timeout=30), f"the reader never reached round {i}"
                agents_md.add_instruction(f"Transient {i}")
                agents_md.remove_instruction(f"Transient {i}")
                written[i].set()
        finally:
            for event in written:  # a writer failure must not leave the reader spinning
                event.set()

    def read() -> list[str]:
        snapshots = []
        for i in range(rounds):
            ready[i].set()
            while True:  # at least one snapshot per round, then as many as the round lasts
                text = read_bytes_waiting_for_writer(path).decode("utf-8", errors="replace")
                snapshots.append(text)
                if written[i].is_set():
                    break
        return snapshots

    pool = ThreadPoolExecutor(max_workers=2)
    try:  # a stuck worker is reported here; the parent's deadline then ends this interpreter
        writer = pool.submit(write)
        snapshots = pool.submit(read).result(timeout=120)
        writer.result(timeout=120)
    finally:
        for event in ready + written:  # a stuck worker must not block its peer
            event.set()
        pool.shutdown(wait=False, cancel_futures=True)
    assert len(snapshots) >= rounds, len(snapshots)
    bad = [s for s in snapshots if standing not in s]
    assert bad == [], f"{len(bad)} partial snapshots, e.g. {bad[0]!r}"


def test_concurrent_remembers_all_land(_fresh_agents_md: Path) -> None:
    """Sixteen simultaneous adds each keep their line: the update is serialized."""
    child, output = _in_a_child("_concurrent_remembers_scenario", timeout=120)
    assert child.returncode == 0, output
    assert sorted(agents_md.read_instructions()) == sorted(
        f"Concurrent rule {i}" for i in range(16)
    )
    assert _fresh_agents_md.read_text().startswith("# User instructions\n\n- Concurrent rule ")


def _concurrent_remembers_scenario() -> None:
    """The sixteen simultaneous adds of :func:`test_concurrent_remembers_all_land`."""
    start = Barrier(16, timeout=30)

    def add(i: int) -> str:
        start.wait()
        return agents_md.add_instruction(f"Concurrent rule {i}")

    pool = ThreadPoolExecutor(max_workers=16)
    try:  # a stuck worker is reported here, well before the parent's deadline ends this interpreter
        replies = list(pool.map(add, range(16), timeout=60))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    assert all(r.startswith("Remembered in ") for r in replies), replies


def test_a_wedged_worker_fails_at_the_deadline_and_is_reaped() -> None:
    """A worker that never returns cannot hang pytest: the child is killed at the deadline, reaped.

    The child gets through its own Future timeout and ``shutdown(wait=False)``
    but never exits (the interpreter joins the wedged thread), which is the
    hang the parent's deadline is for.
    """
    deadline = 10.0  # the child's import of this module must fit too, on a loaded machine
    started = time.monotonic()
    child, output = _in_a_child("_wedged_scenario", timeout=deadline)
    assert time.monotonic() - started >= deadline  # ended by the parent, not by the child
    assert child.returncode not in (None, 0)  # killed, then reaped
    assert "TimeoutError" in output and "the interpreter now joins the wedged worker" in output


def _wedged_scenario() -> None:
    """A worker that never returns, in the scenarios' executor pattern."""
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(Event().wait).result(timeout=0.1)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        print("the scenario gave up; the interpreter now joins the wedged worker", flush=True)


def test_remember_appends_to_a_hand_written_file(_fresh_agents_md: Path) -> None:
    """A user-authored AGENTS.md keeps its text; bullets are appended after it."""
    path = _fresh_agents_md
    path.write_text("# Mine\n\nSome prose.\n* Existing bullet")  # no trailing newline
    remember_sea.remember_instruction("New rule")
    assert path.read_text() == "# Mine\n\nSome prose.\n* Existing bullet\n- New rule\n"
    assert agents_md.read_instructions() == ["Existing bullet", "New rule"]


def test_forget_removes_the_matching_bullet_only(_fresh_agents_md: Path) -> None:
    """Matching ignores case, marker and spacing; prose and other bullets stay."""
    path = _fresh_agents_md
    path.write_text(
        "# Mine\n\nSome prose.\n- Keep this\n- Remove   this one\n\n"
        "## Notes\n+ remove THIS one\nTrailing prose.\n"
    )
    reply = forget_sea.forget_instruction("*  remove this ONE")
    assert reply == f"Forgot from {path}: Remove   this one"
    assert path.read_text() == "# Mine\n\nSome prose.\n- Keep this\n\n## Notes\nTrailing prose.\n"
    assert agents_md.read_instructions() == ["Keep this"]


def test_forget_reports_misses_with_the_stored_list(_fresh_agents_md: Path) -> None:
    """No match or an empty text changes nothing and lists what is stored."""
    path = _fresh_agents_md
    assert forget_sea.forget_instruction("anything") == (
        f"Error: no instruction in {path} matches: anything\n"
        f"Stored instructions:\nNo instructions are stored in {path}."
    )
    assert not path.exists()
    remember_sea.remember_instruction("Rule A")
    remember_sea.remember_instruction("Rule B")
    before = path.read_text()
    assert forget_sea.forget_instruction("Rule") == (
        f"Error: no instruction in {path} matches: Rule\nStored instructions:\n1. Rule A\n2. Rule B"
    )
    assert forget_sea.forget_instruction("   ") == (
        "Error: the instruction is empty; nothing was forgotten."
    )
    assert path.read_text() == before


def test_remember_agent_stores_the_prompt_verbatim(tmp_path: Path, _fresh_agents_md: Path) -> None:
    """With the SEA's configuration the model sees Bash, finish and the two tools.

    The scripted model calls ``remember_instruction`` with the prompt and
    finishes; the file holds the instruction and the tool's reply reached
    the model as the tool result.
    """
    instruction = "Always write commit messages in the imperative mood"
    script = [
        tool_call_body("remember_instruction", {"instruction": instruction}, prompt_tokens=400),
        finish_body("<p>Remembered.</p>", prompt_tokens=500),
    ]
    parsed, agentic = _run(RememberSea(), instruction, script, tmp_path)
    assert parsed["success"] is True
    assert parsed["summary"] == "<p>Remembered.</p>"
    assert _fresh_agents_md.read_text() == f"# User instructions\n\n- {instruction}\n"

    assert len(agentic) == 2, [list(r) for r in agentic]
    for request in agentic:
        names = tool_names(request)
        assert names == {"Bash", "finish", "remember_instruction", "list_instructions"}, names
        assert system_message(request).startswith(remember_sea.SYSTEM_PROMPT)
    user = next(m for m in agentic[0]["messages"] if m["role"] == "user")
    assert instruction in str(user["content"])
    results = tool_results(agentic[1])
    assert len(results) == 1
    # The agent appends a usage line to every tool result; the reply comes first.
    assert results[0].startswith(
        f"Remembered in {_fresh_agents_md}: {instruction}\n"
    )


def test_forget_agent_removes_the_instruction_the_next_task_was_following(
    tmp_path: Path,
    _fresh_agents_md: Path,
) -> None:
    """A stored instruction is in the run's system prompt until ``/forget`` removes it.

    The scripted model first misses (a paraphrase), reads the stored list
    from the error, then removes the stored text exactly and finishes.
    """
    remember_sea.remember_instruction("Always reply in British English")
    remember_sea.remember_instruction("Prefer uv over pip")
    script = [
        tool_call_body(
            "forget_instruction",
            {"instruction": "the British English one"},
            prompt_tokens=400,
        ),
        tool_call_body(
            "forget_instruction",
            {"instruction": "Always reply in British English"},
            prompt_tokens=500,
        ),
        finish_body("<p>Forgot it.</p>", prompt_tokens=600),
    ]
    parsed, agentic = _run(ForgetSea(), "forget the British English one", script, tmp_path)
    assert parsed["success"] is True
    assert parsed["summary"] == "<p>Forgot it.</p>"
    assert _fresh_agents_md.read_text() == "# User instructions\n\n- Prefer uv over pip\n"

    assert len(agentic) == 3, [list(r) for r in agentic]
    for request in agentic:
        names = tool_names(request)
        assert names == {"Bash", "finish", "forget_instruction", "list_instructions"}, names
        system = system_message(request)
        assert system.startswith(forget_sea.SYSTEM_PROMPT)
        # perform_task appended AGENTS.md: the run itself followed both rules.
        assert "- Always reply in British English\n- Prefer uv over pip" in system
    results = tool_results(agentic[2])
    assert len(results) == 2
    assert results[0].startswith(
        f"Error: no instruction in {_fresh_agents_md} matches: the British English one\n"
        "Stored instructions:\n1. Always reply in British English\n2. Prefer uv over pip\n"
    )
    assert results[1].startswith(
        f"Forgot from {_fresh_agents_md}: Always reply in British English\n"
    )

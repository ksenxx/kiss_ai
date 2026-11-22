# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of the bundled ``/discover_viz`` agent
(:mod:`kiss.agents.seas.discover_viz.discover_viz_sea`).

The toy ``/discover`` run of :func:`_toy_events` (seven points that use
every operator and every status of the loop) is written to the
session's SQLite history.  The tests of the SEA's two tools read it
back from there, the page tests open what those tools wrote in a real
Chromium, and one test runs the agent itself against a scripted model.
Two tests need no persisted run: the contract test, and the wide-layout
test, which renders a generated story.  No mocks.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from playwright.sync_api import Browser, Page, sync_playwright

from kiss.agents.seas.discover_viz import animation, story, trajectory
from kiss.agents.seas.discover_viz import discover_viz_sea as sea
from kiss.agents.seas.discover_viz.discover_viz_sea import (
    DiscoverVizSea,
    discover_trajectory,
    render_discover_viz,
)
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.persistence import _add_task
from kiss.core.kiss_tmp import kiss_tmp_dir
from kiss.tests.agents.seas.conftest import persist_task, run_sea
from kiss.tests.agents.seas.sea_contract import (
    assert_worker_contract,
    system_message,
    tool_names,
    tool_results,
)
from kiss.tests.agents.sorcar.local_model_server import finish_body, tool_call_body

_FIELDS = ("id", "parent", "commit", "idea", "aspect", "accuracy", "seconds")
_HEADER = "| " + " | ".join(_FIELDS) + " |"
_MARKUP = "<b>deeper</b> trees </script><script>window.pwned = 1</script>"

# One evaluated point per row: the seven pareto.md fields, then the draw, the
# operator and the status of its own round.
_POINTS = [
    ("B", "-", "a1b2c3d", "baseline: logistic regression", "-", "0.60", "1.0", "", "baseline",
     "KEPT"),
    ("C1", "B", "b2c3d4e", "gradient boosted trees", "model", "0.70", "1.2", "", "initial", "KEPT"),
    ("C2", "C1", "c3d4e5f", _MARKUP, "model", "0.69", "1.3", "0.1234", "variant", "FAILED"),
    ("C3", "B", "d4e5f6a", "cache the features", "features", "0.65", "0.8", "0.5001", "fork",
     "KEPT"),
    ("C4", "C1 + C3", "e5f6a7b", "boosted trees on cached features", "model, features", "0.72",
     "0.8", "0.4100", "compose", "KEPT"),
    ("C5", "C4", "f6a7b8c", "early stopping", "model", "0.72", "0.8", "0.2000", "variant", "TIE"),
    ("C6", "C4", "a7b8c9d", "target encoding", "features", "0.90", "0.7", "0.9012, 0.7001", "fresh",
     "KEPT"),
]
# What each round did to other points, and what the finish did: C6 enters the
# frontier, fails the held-out check and leaves it empty, so the baseline comes
# back; the run then reports the TIE point C5, which the finish promotes.
_CHANGES = {
    "C3": [("B", "DOMINATED")],
    "C4": [("C1", "DOMINATED"), ("C3", "DOMINATED")],
    "C6": [("C4", "DOMINATED"), ("C6", "OVERFIT"), ("B", "KEPT")],
}
_FINISH_CHANGES = [("C5", "KEPT")]
# The state of every point after each of the seven rounds, then after the finish.
_STATES_AFTER: list[dict[str, str]] = [dict(re.findall(r"(\w+)=(\w+)", text)) for text in (
    "B=frontier",
    "B=frontier C1=frontier",
    "B=frontier C1=frontier C2=failed",
    "B=dominated C1=frontier C2=failed C3=frontier",
    "B=dominated C1=dominated C2=failed C3=dominated C4=frontier",
    "B=dominated C1=dominated C2=failed C3=dominated C4=frontier C5=tie",
    "B=frontier C1=dominated C2=failed C3=dominated C4=dominated C5=tie C6=overfit",
    "B=frontier C1=dominated C2=failed C3=dominated C4=dominated C5=frontier C6=overfit",
)]
_REQUEST = "Raise the accuracy of the churn model to 0.85 without making it slower"
_DRAW = f"python3 -c 'import random; print({trajectory.DRAW_CALL})'"


def _row(*cells: str) -> str:
    """Return *cells* as one markdown table row."""
    return "| " + " | ".join(cells) + " |"


def _call(name: str, **args: Any) -> dict[str, Any]:
    """Return a ``tool_call`` event as the daemon stores it: six arguments on top, the rest
    under ``extras``."""
    top = ("path", "description", "command", "content", "old_string", "new_string")
    extras = {key: value for key, value in args.items() if key not in top}
    event = {"type": "tool_call", "name": name} | {k: v for k, v in args.items() if k in top}
    return event | ({"extras": extras} if extras else {})


def _result(content: str = "", is_error: bool = False) -> dict[str, Any]:
    """Return the ``tool_result`` event that closes a call."""
    return {"type": "tool_result", "content": content, "is_error": is_error}


def _toy_events(scratch: str) -> list[dict[str, Any]]:
    """Return the trajectory of the toy run, its loop files written under *scratch*.

    The run uses every way a real run writes its files: a here-document
    that creates ``pareto.md``, ``Write``, a single ``Edit``, a batch
    ``Edit``, two ``Edit`` calls that fail (the second with ``edits``
    that are no JSON), a here-document that appends (C5:
    the harness nests a ``Write`` of the whole file and a ``Bash`` call
    for the rest of the command inside the call), draws whose output is
    streamed, a ``decide`` call, a ``Read`` and a ``cat`` of two files
    that prints more than the evidence quotes; plus calls that are no
    evidence (an unrelated command, an unrelated file, a ``Read`` of
    ``ideas.md``).
    """
    pareto, explored = f"{scratch}/pareto.md", f"{scratch}/explored-ideas.md"
    events = [
        {"type": "prompt", "text": _REQUEST},
        _call("Write", path=f"{scratch}/split.md", content="Held out: data/heldout.csv"), _result(),
        _call("Write", path=f"{scratch}/ideas.md", content="1. boosted trees\n" + "x" * 5000),
        _result(),
        _call("Write", path="model.py", content="print('not a loop file')"), _result(),
        _call("Bash", command="ls -la", description="look around"),
        {"type": "system_output", "text": "total 0\n"}, _result(),
    ]
    for *fields, draw, _operator, status in _POINTS:
        line, name = _row(*fields), fields[0]
        log = _row(*fields, draw or "-", status, f"observed {name}")
        if name == "B":
            events += [
                _call("Bash", command=f"cat > {pareto} <<'EOF'\n{_HEADER}\n{line}\nEOF"), _result(),
                _call("Write", path=explored, content=f"{_HEADER} draw | status |\n{log}\n"),
                _result(),
                _call("decide", state="trees or kNN first?", questions='{"first": ["trees"]}'),
                _result('{"first": "trees"}'),
            ]
            continue
        if draw:
            events += [
                _call("Bash", command=_DRAW), {"type": "system_output", "text": f"{draw}\n"},
                _result(),
            ]
        if status == "KEPT":
            events += [
                _call("Edit", path=pareto, old_string=_HEADER, new_string=f"{_HEADER}\n{line}"),
                _result(),
            ]
        if name == "C1":
            edits = [{"old_string": "x", "new_string": log},
                     {"old_string": "| KEPT |", "new_string": "| DOMINATED |", "replace_all": True}]
            events += [_call("Edit", path=explored, edits=json.dumps(edits)), _result()]
        elif name == "C5":
            events += [
                _call("Bash", command=f"cat >> {explored} <<'EOF'\n{log}\nEOF\ngit log -1"),
                _call("Write", path=explored, content=f"(the rows before)\n{log}\n"),
                _result("Successfully wrote"),
                _call("Bash", command="git log -1"),
                {"type": "system_output", "text": "f6a7b8c early stopping\n"}, _result(),
                _result("Appended through Write, then made Bash: f6a7b8c early stopping"),
            ]
        else:
            events += [_call("Edit", path=explored, old_string="\n", new_string=f"\n{log}\n"),
                       _result()]
    return events + [
        _call("Edit", path=pareto, old_string="no such line", new_string="never written"),
        _result("Error: old_string not found", is_error=True),
        _call("Edit", path=pareto, edits="[not a list"),
        _result("Error: edits must be a JSON list of objects", is_error=True),
        _call("Read", path=pareto), _result(f"{_HEADER}\n{_row(*_POINTS[0][:7])}"),
        _call("Read", path=f"{scratch}/ideas.md"), _result("1. boosted trees"),
        _call("Bash", command=f"cat {scratch}/ideas.md {explored}"),
        {"type": "system_output", "text": "1. boosted trees\n" + "x" * 9000 + "\nTHE LAST ROW\n"},
        _result(),
        _call("finish", summary="<p>Best point C5.</p>"),
    ]


def _changes(pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
    """Return the story's ``changes`` entries for *pairs* of ``(point, new status)``."""
    return [{"id": point, "status": status, "why": f"{point} became {status}."}
            for point, status in pairs]


def _toy_story(task_id: str) -> dict[str, Any]:
    """Return the story of the toy run as the model would write it for *task_id*."""
    rounds = []
    for *fields, draw, operator, status in _POINTS:
        name, parent = fields[0], fields[1]
        rounds.append({
            "id": name,
            "parents": [] if parent == "-" else parent.split(" + "),
            "idea": f"Try {fields[3]}.",
            "operator": operator,
            "draw": draw,
            "picked": f"Picked {name}: a draw under 0.33 is a variant.",
            "score": fields[5],
            "status": status,
            "outcome": f"Outcome of {name}: accuracy {fields[5]} in {fields[6]} seconds.",
            "details": dict(zip(_FIELDS, fields)),
            "source": "pareto.md" if status == "KEPT" else "explored-ideas.md",
            "changes": _changes(_CHANGES.get(name, [])),
        })
    return {
        "task_id": task_id,
        "title": "Churn model accuracy",
        "goal": "Reach an accuracy of 0.85 without a slower model.",
        "metric": "accuracy, higher is better",
        "rounds": rounds,
        "finish": {
            "best": "C5",
            "reason": "The round limit was reached and C5 passed the held-out check.",
            "changes": _changes(_FINISH_CHANGES),
        },
    }


@pytest.fixture
def toy_run(tmp_path: Path) -> str:
    """Persist the toy run, finished, with ``<tmp_path>/repo`` as its work dir; return its id."""
    work_dir = tmp_path / "repo"
    return persist_task(
        _REQUEST, _toy_events(f"{work_dir}/.kiss-worktrees/wt-1/tmp/run"),
        "<p>Best point C5 &amp; frontier B, C5.</p>", success=True,
        sea=trajectory.DISCOVER_SEA, work_dir=str(work_dir),
    )


def test_sea_follows_the_contract_and_is_a_registered_command() -> None:
    """The SEA pins its tools and prompt; `/discover_viz` resolves to it, bare or with a task."""
    agent = DiscoverVizSea()
    assert_worker_contract(
        sea, agent, {"tool_profile": "shell+edit+browser", "use_web_tools": True}
    )
    assert agent.system_prompt("ASSEMBLED") == sea.SYSTEM_PROMPT
    assert agent.tools([print]) == [print, discover_trajectory, render_discover_viz]
    assert agent.prompt("  9de623ce ") == f"9de623ce\n\n{sea.ADD_TO_PROMPT}"
    assert agent.prompt(" \n") == sea.ADD_TO_PROMPT
    path = Path(sea.__file__)
    assert sea_commands.get_command("discover_viz") == path
    assert sea_commands.slash_command("/discover_viz 9de623ce") == ("9de623ce", path)
    assert sea_commands.slash_command_task("/discover_viz") == ("", path)
    assert sea_commands.help_text_if_command("/discover_viz help") == agent.description()


def test_evidence_lists_every_loop_call_in_order_and_nothing_else(
    toy_run: str, tmp_path: Path
) -> None:
    """The evidence file holds the run's header, its loop calls in order, the copies on disk
    and the final result, and none of the calls that have nothing to do with the loop."""
    work_dir = tmp_path / "repo"
    # The run wrote in a worktree that is gone: one file survives in the main checkout's copy
    # of that folder, one in the run's default scratch dir, its folder of the agents' temp
    # directory.  ``tmp/<task id>`` of the work dir is no scratch dir any more: a file there
    # is not the run's.
    (work_dir / "tmp" / "run").mkdir(parents=True)
    # The loop files are UTF-8, as the Write tool leaves them, whatever the locale's codec is.
    pareto_on_disk = work_dir / "tmp" / "run" / "pareto.md"
    pareto_on_disk.write_text("PARETO ON DISK: error ≤ 0.5\n", encoding="utf-8")
    default_scratch = kiss_tmp_dir() / toy_run
    default_scratch.mkdir()
    log_on_disk = "EXPLORED ON DISK\n" + "| the row of one round |\n" * 1500
    (default_scratch / "explored-ideas.md").write_text(log_on_disk)
    (work_dir / "tmp" / toy_run).mkdir()
    (work_dir / "tmp" / toy_run / "ideas.md").write_text("IN THE WORK DIR'S TMP\n")
    out = tmp_path / "scratch" / "evidence.md"

    message = discover_trajectory(toy_run[:8].upper(), str(out))
    text = out.read_text(encoding="utf-8")
    assert message == (
        f"/discover run {toy_run}: evidence written to {out}, {len(text)} characters in "
        f"{text.count(chr(10))} lines. Read all of it before you write the story."
    )
    assert text.splitlines()[:5] == [
        f"# Evidence of the /discover run {toy_run}",
        "",
        f"Request: {_REQUEST}",
        "State: finished, success=true",
        f"Steps: 5, cost: $0.50, model: model-a, work dir: {work_dir}",
    ]
    headings = [line.partition("] ")[2] for line in text.splitlines() if line.startswith("### [")]
    draw_round = ["Bash", "Edit pareto.md", "Edit explored-ideas.md"]
    assert headings == [
        "Write split.md", "Write ideas.md", "Bash", "Write explored-ideas.md", "decide",
        "Edit pareto.md", "Edit explored-ideas.md",  # C1, no draw
        "Bash", "Edit explored-ideas.md",  # C2, FAILED: never in pareto.md
        *draw_round, *draw_round,  # C3, C4
        # C5, TIE: its row is appended from the shell, which the harness runs as a Write.
        "Bash", "Bash", f"Write explored-ideas.md{trajectory.NESTED_MARK}",
        *draw_round,  # C6
        "Edit pareto.md", "Edit pareto.md", "Read pareto.md", "Bash",
    ]
    assert f"## Timeline: {len(headings)} entries" in text
    baseline = _row(*_POINTS[0][:7])
    tie = _row(*_POINTS[5][:7], "0.2000", "TIE", "observed C5")
    scratch = f"{work_dir}/.kiss-worktrees/wt-1/tmp/run"
    for expected in (
        f"### [3] Bash\ncommand:\n~~~\ncat > {scratch}/pareto.md <<'EOF'\n{_HEADER}\n{baseline}\n"
        "EOF\n~~~\noutput:\n~~~\n\n~~~",
        # Each call keeps its own output: the appending command its result, not the output
        # of the two calls nested in it.
        f"### [17] Bash\ncommand:\n~~~\ncat >> {scratch}/explored-ideas.md <<'EOF'\n{tie}\nEOF\n"
        "git log -1\n~~~\noutput:\n~~~\nAppended through Write, then made Bash: f6a7b8c early "
        "stopping\n~~~",
        f"### [18] Write explored-ideas.md{trajectory.NESTED_MARK}\n~~~\n(the rows before)\n{tie}"
        "\n~~~",
        'state:\n~~~\ntrees or kNN first?\n~~~\nquestions:\n~~~\n{"first": ["trees"]}\n~~~\n'
        'answer:\n~~~\n{"first": "trees"}\n~~~',
        "old (every occurrence):\n~~~\n| KEPT |\n~~~\nnew:\n~~~\n| DOMINATED |\n~~~",
        f"### [8] Bash\ncommand:\n~~~\n{_DRAW}\n~~~\noutput:\n~~~\n0.1234\n~~~",
        "new:\n~~~\nnever written\n~~~\nNOT APPLIED, the call failed: Error: old_string not found",
        # An Edit whose replacements cannot be read is listed with the reason, not dropped.
        "### [23] Edit pareto.md\nError: edits must be a JSON list of objects: Expecting value: "
        "line 1 column 2 (char 1)\nNOT APPLIED, the call failed: Error: edits must be a JSON list "
        "of objects",
        f"### [24] Read pareto.md\n~~~\n{_HEADER}\n{baseline}\n~~~",
        # What is too long loses its middle, not its end, where a log has its newest row: the
        # output of a command past 6,000 characters, ideas.md past 3,000.
        f"### [25] Bash\ncommand:\n~~~\ncat {scratch}/ideas.md {scratch}/explored-ideas.md\n~~~\n"
        f"output:\n~~~\n1. boosted trees\n{'x' * 2983}\n…[3031 characters cut]…\n{'x' * 2986}\n"
        "THE LAST ROW\n~~~",
        f"### [2] Write ideas.md\n~~~\n1. boosted trees\n{'x' * 1483}\n…[2017 characters cut]…\n"
        f"{'x' * 1500}\n~~~",
        f"## On disk now: {pareto_on_disk}\n~~~\nPARETO ON DISK: error ≤ 0.5\n~~~",
        # The log on disk is quoted whole, also that of a long run.
        f"## On disk now: {default_scratch / 'explored-ideas.md'}\n~~~\n{log_on_disk}~~~",
        "## Final result of the run\n~~~\n Best point C5 & frontier B, C5.\n~~~\n",
    ):
        assert expected in text, expected
    assert text.count("## On disk now:") == 2
    for absent in ("not a loop file", "ls -la", "total 0", "Read ideas.md", "<p>",
                   "IN THE WORK DIR'S TMP"):
        assert absent not in text, absent


@pytest.mark.usefixtures("empty_history")
def test_a_run_is_found_by_id_prefix_or_as_the_newest_and_a_refusal_names_the_runs(
    tmp_path: Path,
) -> None:
    """`list`, no id, an id and its first characters all resolve; anything else is refused
    with the runs to choose from; a task that dispatched a /discover run stands for it."""
    out = str(tmp_path / "evidence.md")
    assert discover_trajectory("list", "") == "The history holds no /discover run."
    assert discover_trajectory("", out) == (
        "Error: no single task matches 'latest'.\nThe history holds no /discover run."
    )
    events, discover = _toy_events("/gone/tmp/run"), trajectory.DISCOVER_SEA
    old = persist_task("old run", events, None, sea=discover, timestamp=1e6)
    new = persist_task("new run", events, "none", success=False, sea=discover, timestamp=3e6)
    # The parent only looked for the ledgers its /discover sub-task wrote, as a14cda8b did.
    look = [_call("Bash", command="find / -name pareto.md"), _result()]
    plain = persist_task("no loop", look, "done", timestamp=4e6)
    child = persist_task(
        "sub run", events, "ok", success=True, sea=discover, parent_task_id=plain, timestamp=2e6
    )

    listing = discover_trajectory(" LIST ", "")
    lines = listing.splitlines()
    assert lines[0] == "Newest /discover runs:"
    assert [line.split(" | ")[0] for line in lines[1:]] == [f"- {new}", f"- {child}", f"- {old}"]
    assert lines[1].endswith(" | finished, success=false | 5 steps | new run")
    assert " | finished, success=true | " in lines[2] and " | not finished | " in lines[3]

    for ref in ("", "latest", new[:6], f" {new.upper()} "):
        assert discover_trajectory(ref, out).startswith(f"/discover run {new}: evidence written")
    assert discover_trajectory(old, out).startswith(f"/discover run {old}: evidence written")
    for ref in ("0" * 32, "my run", new[:5]):
        assert discover_trajectory(ref, out) == f"Error: no single task matches {ref!r}.\n{listing}"
    for last in "12":
        _add_task("twin", task_id="abcdef01" + last * 24)
    assert discover_trajectory("abcdef01", out) == (
        f"Error: no single task matches 'abcdef01'.\n{listing}"
    )
    no_loop = "wrote no pareto.md or explored-ideas.md, so it ran no discovery loop."
    assert discover_trajectory("abcdef011", out) == (
        f"Error: task abcdef01{'1' * 24} {no_loop}\n{listing}"
    )
    assert discover_trajectory(plain, out).startswith(
        f"Task {plain} is not a /discover run; it dispatched these /discover runs:\n"
        f"{lines[2]}\nAnimating the newest, {child}.\n/discover run {child}: evidence written"
    )
    assert "find / -name" not in Path(out).read_text(encoding="utf-8")
    assert discover_trajectory(new, "evidence.md") == (
        "Error: pass absolute paths, not 'evidence.md'."
    )


_C1_DETAILS = dict(zip(_FIELDS, _POINTS[1][:7]))
_NOT_COPIED = "is not in the evidence; copy it as the run wrote it"
_EMPTY = "the frontier is empty after this round; the loop puts the baseline back (a KEPT change)"

# (the field of the toy story to replace, its new value, the line that refuses the story)
_REFUSALS: list[tuple[str, object, str]] = [
    # Malformed: the field is named.
    ("rounds.2.status", "DOMINATED", "rounds[2].status: Input should be 'KEPT', 'TIE' or 'FAILED'"),
    ("rounds.1.colour", "red", "rounds[1].colour: Extra inputs are not permitted"),
    ("rounds.0.source", "notes.md", "rounds[0].source: Input should be 'pareto.md' or"),
    ("rounds.3.details", {}, "rounds[3].details: Dictionary should have at least 1 item"),
    ("rounds", [], "rounds: List should have at least 1 item"),
    ("title", "x" * 81, "title: String should have at most 80 characters"),
    ("metric", "x" * 61, "metric: String should have at most 60 characters"),
    # Bookkeeping the loop cannot produce.
    ("rounds.2.id", "C1", "rounds[2].id: 'C1' is already the id of an earlier round"),
    ("rounds.1.parents", ["C3"], "rounds[1].parents: C3 not evaluated in an earlier round"),
    ("rounds.1.parents", [], "rounds[1]: the baseline has no parents and every other point has"),
    ("rounds.0.parents", ["C1"], "rounds[0]: the baseline has no parents and every other point"),
    ("rounds.4.parents", ["C1"], "rounds[4]: a compose names both of its points"),
    ("rounds.3.parents", ["B", "C1"], "rounds[3]: a compose names both of its points"),
    ("rounds.5.parents", ["C2"],
     "rounds[5].parents: C2 never entered the frontier, and the loop builds only on frontier"),
    ("rounds.0.status", "FAILED", "rounds[0].status: FAILED needs a frontier point"),
    ("rounds.0.status", "TIE", f"rounds[0]: {_EMPTY}"),
    ("rounds.3.changes", _changes([("C9", "DOMINATED")]),
     "rounds[3].changes[0]: 'C9' is not a point evaluated so far"),
    ("rounds.3.changes", _changes([("C2", "DOMINATED")]),
     "rounds[3].changes[0]: C2 is failed here, so it cannot become DOMINATED"),
    ("rounds.3.changes", _changes([("C2", "OVERFIT")]),
     "rounds[3].changes[0]: C2 is failed here, so it cannot become OVERFIT"),
    ("rounds.1.changes", _changes([("B", "KEPT")]),
     "rounds[1].changes[0]: B is frontier here, so it cannot become KEPT"),
    # Only the baseline is put back and only a TIE point promoted: not a FAILED point as best.
    ("finish", {"best": "C2", "reason": "C2 was promoted.", "changes": _changes([("C2", "KEPT")])},
     "finish.changes[0]: C2 is failed here, so it cannot become KEPT (the loop puts back the "
     "baseline and promotes a TIE point, nothing else)"),
    ("rounds.6.changes", _changes([("C4", "DOMINATED"), ("C6", "OVERFIT")]),
     f"rounds[6]: {_EMPTY}"),
    ("finish.best", "C4", "finish.best: 'C4' is not on the frontier at the end (B, C5)"),
    # A TIE point may fail the held-out check, but then it is not the best point.
    ("finish.changes", _changes([("C5", "OVERFIT")]),
     "finish.best: 'C5' is not on the frontier at the end (B)"),
    ("finish.changes", _changes([("C9", "KEPT")]),
     "finish.changes[0]: 'C9' is not a point evaluated so far"),
    # Values the run did not record.
    ("rounds.1.details", _C1_DETAILS | {"accuracy": "0.71"},
     f"rounds[1].details['accuracy']: '0.71' {_NOT_COPIED}"),
    ("rounds.1.score", "0.77", f"rounds[1].score: '0.77' {_NOT_COPIED}"),
    ("rounds.2.draw", "0.4321", f"rounds[2].draw: '0.4321' {_NOT_COPIED}"),
    # A piece of a longer number is no recorded value: 0.6 of 0.60, 9012 of 0.9012.
    ("rounds.0.score", "0.6", f"rounds[0].score: '0.6' {_NOT_COPIED}"),
    ("rounds.6.draw", "9012, 0.7001", f"rounds[6].draw: '9012, 0.7001' {_NOT_COPIED}"),
    ("rounds.0.outcome", "It scored 0.6 on dev.", "rounds[0].outcome: 0.6 not in the evidence"),
    ("rounds.2.outcome", "It lost 0.01 to C1, a gap of 1.4 percent.",
     "rounds[2].outcome: 0.01, 1.4 not in the evidence; quote numbers as the run recorded them"),
    ("finish.reason", "Held-out accuracy was 0.55.", "finish.reason: 0.55 not in the evidence"),
    ("rounds.3.changes", [{"id": "B", "status": "DOMINATED", "why": "It is 0.05 behind."}],
     "rounds[3].changes[0].why: 0.05 not in the evidence"),
    # The story must name the run it is about.
    ("task_id", "0" * 32, "Error: task_id must be the full id of the /discover run"),
]


def _replaced(data: dict[str, Any], field: str, value: object) -> dict[str, Any]:
    """Return *data* with the value at the dotted path *field* (``rounds.2.status``) replaced."""
    *path, last = field.split(".")
    node: Any = data
    for key in path:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[last] = value
    return data


def _render(story_data: object, folder: Path) -> str:
    """Write *story_data* as ``story.json`` in *folder* and return the renderer's answer."""
    path = folder / "story.json"
    path.write_text(json.dumps(story_data), encoding="utf-8")
    return render_discover_viz(str(path), str(folder / "pages" / "viz.html"))


def test_render_refuses_a_story_the_run_does_not_support_and_writes_a_sound_one(
    toy_run: str, tmp_path: Path
) -> None:
    """Each flaw of `_REFUSALS` is refused with its own line and nothing is written; emphasis
    marks and line breaks in a copied value are not flaws; the sound story gives the page."""
    for field, value, line in _REFUSALS:
        answer = _render(_replaced(_toy_story(toy_run), field, value), tmp_path)
        assert answer.startswith("Error: ") and line in answer, f"{field}: {answer}"
    assert _render([1], tmp_path).startswith("Error: task_id must be the full id")
    no_loop = persist_task("no loop", [_call("Bash", command="ls"), _result()], "done")
    assert _render(_toy_story(no_loop), tmp_path).startswith("Error: task_id must be the full id")

    (tmp_path / "story.json").write_text("{", encoding="utf-8")
    broken = render_discover_viz(str(tmp_path / "story.json"), str(tmp_path / "pages" / "v.html"))
    assert broken.startswith(f"Error: cannot read the story {tmp_path / 'story.json'}: Expecting")
    missing = render_discover_viz(str(tmp_path / "none.json"), str(tmp_path / "pages" / "v.html"))
    assert missing.startswith(f"Error: cannot read the story {tmp_path / 'none.json'}: [Errno 2]")
    assert render_discover_viz("story.json", "viz.html") == (
        "Error: pass absolute paths, not 'story.json', 'viz.html'."
    )

    # Thirty rounds under a parent that does not exist: the refusal lists 25 errors and counts
    # the rest.
    crowded = _toy_story(toy_run)
    crowded["rounds"] += [
        crowded["rounds"][1] | {"id": f"X{number}", "parents": ["nowhere"]} for number in range(30)
    ]
    answer = _render(crowded, tmp_path)
    assert answer.startswith("Error: the story is refused, nothing was written. Fix:\n- rounds[7]")
    assert answer.count("\n- ") == sea.MAX_ERRORS + 1 and answer.endswith("\n- ... and 5 more")
    assert not (tmp_path / "pages").exists()

    marked = _replaced(_toy_story(toy_run), "rounds.1.details", _C1_DETAILS | {
        "accuracy": "**0.70**", "idea": "gradient  boosted\n`trees`",
    })
    page = tmp_path / "pages" / "viz.html"
    expected = (
        f"Animation written to {page}: 7 points, 15 frames, final frontier B, C5. Compare that "
        f"frontier with the last pareto.md of the evidence, then open {page.as_uri()}"
    )
    assert _render(marked, tmp_path) == expected
    # Rounds that ran at the same time are logged one after the other, so C6 may hang under
    # C1, which was on the frontier until C4 dominated it.
    late = _replaced(_toy_story(toy_run), "rounds.6.parents", ["C1"])
    assert _render(late, tmp_path) == expected
    # A run that did not finish has no finish frame: C5 is still a tie, so B stands alone.
    unfinished = _replaced(_toy_story(toy_run), "finish", None)
    assert _render(unfinished, tmp_path) == expected.replace(
        "15 frames, final frontier B, C5", "14 frames, final frontier B"
    )
    assert _render(_toy_story(toy_run), tmp_path) == expected
    assert page.read_text(encoding="utf-8").startswith("<!doctype html>")


_HOW = [
    "The starting point of the run.",
    "First idea of the run, built on B. No draw.",
    "Variant: one change to C1.",
    "Fork: a second branch from B, so that two branches can be composed later.",
    "Compose: C3 merged into C1.",
    "Variant: one change to C4.",
    "Fresh idea, built on C4.",
]
_VERDICT = {
    "KEPT": "Added to the Pareto frontier",
    "TIE": "Discarded: it equals a frontier point (TIE)",
    "FAILED": "Discarded: a frontier point dominates it (FAILED)",
}
_CHANGE_TEXT = {
    "DOMINATED": "leaves the frontier, dominated",
    "OVERFIT": "failed the held-out check",
    "KEPT": "joins the frontier",
}
_NAMES = [point[0] for point in _POINTS]
_SCORE = {point[0]: point[5] for point in _POINTS}

# What the page shows right now, read from its DOM.
_SNAPSHOT = """() => ({
  frame: window.discoverViz.frame,
  playing: window.discoverViz.playing,
  position: document.getElementById('position').textContent,
  title: document.getElementById('step-title').textContent,
  body: [...document.getElementById('step-body').children].map((node) => node.textContent),
  frontier: [...document.querySelectorAll('#frontier li')].map((node) => node.textContent),
  log: [...document.querySelectorAll('#log li')].map((node) => node.textContent),
  states: Object.fromEntries([...document.querySelectorAll('#nodes > g')].map(
    (node) => [node.dataset.id, node.getAttribute('class').replace('node ', '')])),
})"""


@pytest.fixture(scope="module")
def browser() -> Iterator[Browser]:
    """Launch one headless Chromium for the tests of this module."""
    with sync_playwright() as playwright:
        chromium = playwright.chromium.launch(headless=True)
        try:
            yield chromium
        finally:
            chromium.close()


class _PageErrors:
    """The errors a page reported while a test drove it: uncaught ones and console errors."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def uncaught(self, error: Any) -> None:
        """Record an uncaught exception of the page."""
        self.seen.append(str(error))

    def console(self, message: Any) -> None:
        """Record a console message of type ``error``."""
        if message.type == "error":
            self.seen.append(message.text)


@pytest.fixture
def page(browser: Browser) -> Iterator[Page]:
    """Yield a page of its own; the test fails when the page raised or logged an error."""
    context = browser.new_context(viewport={"width": 1280, "height": 900})
    tab = context.new_page()
    errors = _PageErrors()
    tab.on("pageerror", errors.uncaught)
    tab.on("console", errors.console)
    yield tab
    context.close()
    assert not errors.seen, errors.seen


@pytest.fixture
def toy_page(toy_run: str, tmp_path: Path) -> Path:
    """Render the toy story with the SEA's tool and return the page it wrote."""
    assert _render(_toy_story(toy_run), tmp_path).startswith("Animation written")
    return tmp_path / "pages" / "viz.html"


def _chips(states: dict[str, str]) -> list[str]:
    """Return the "Frontier now" chips the page shows for *states*."""
    return [f"{name} \u00b7 {_SCORE[name]}" for name in story.frontier(states)]


def test_the_page_adds_the_points_one_by_one_and_tells_how_each_round_went(
    page: Page, toy_page: Path
) -> None:
    """Stepping through the fifteen frames: each round first adds its point as a candidate
    under its parents and says how the idea was picked, then gives the verdict (added to the
    frontier or discarded), the score, the outcome and what the round did to other points;
    the frontier and the round log follow; the last frame crowns the best point."""
    page.goto(f"{toy_page.as_uri()}#frame=0")
    shown: dict[str, str] = {}
    for index, (*fields, draw, operator, status) in enumerate(_POINTS):
        name, parents = fields[0], fields[1].split(" + ")
        told = [
            "Idea", f"Try {fields[3]}.", "How it was picked", _HOW[index],
            *([f"Draw: {draw}"] if draw else []), f"Picked {name}: a draw under 0.33 is a variant.",
            "Result",
        ]
        log = [f"{point[0]} \u00b7 {point[8]}{point[9]}" for point in _POINTS[: index + 1]]
        later = dict.fromkeys(_NAMES[index + 1 :], "absent")

        pick = page.evaluate(_SNAPSHOT)
        assert (pick["frame"], pick["playing"]) == (2 * index, False)
        assert pick["position"] == f"Step {2 * index + 1} of 15"
        assert pick["title"] == ("Baseline B" if index == 0 else f"Round {index} of 6: {name}")
        assert pick["body"] == [*told, "Being evaluated"]
        assert pick["states"] == {
            point: state + (" parent" if point in parents else "") for point, state in shown.items()
        } | {name: "candidate focus"} | later
        assert pick["frontier"] == _chips(shown) and pick["log"] == log[:-1]

        page.click("#next")
        shown = _STATES_AFTER[index]
        result = page.evaluate(_SNAPSHOT)
        assert result["frame"] == 2 * index + 1
        assert result["body"] == [
            *told, _VERDICT[status], f"accuracy, higher is better: {fields[5]}",
            f"Outcome of {name}: accuracy {fields[5]} in {fields[6]} seconds.",
            *(f"{point} {_CHANGE_TEXT[change]} ({change}): {point} became {change}."
              for point, change in _CHANGES.get(name, [])),
        ]
        assert result["states"] == shown | {name: f"{shown[name]} focus"} | later
        assert result["frontier"] == _chips(shown) and result["log"] == log
        page.click("#next")

    finish = page.evaluate(_SNAPSHOT)
    assert (finish["frame"], finish["title"]) == (14, "Finish")
    assert finish["body"] == [
        "Best point", "C5", "The round limit was reached and C5 passed the held-out check.",
        "C5 joins the frontier (KEPT): C5 became KEPT.",
    ]
    assert finish["states"] == _STATES_AFTER[7] | {"C5": "frontier best"}
    assert finish["frontier"] == ["B \u00b7 0.60", "C5 \u00b7 0.72"]
    assert page.is_disabled("#next") and page.inner_text("#play") == "Replay"
    # The markup in C2's idea was shown as text: nothing ran and nothing became an element.
    assert page.evaluate("window.pwned") is None
    assert page.locator("#story b, #story script").count() == 0


# The edges as drawn: where each path starts and ends and whether it is dashed.
_EDGES = """() => {
  const data = JSON.parse(document.getElementById('discover-viz-data').textContent);
  return {
    radius: data.radius,
    at: Object.fromEntries(data.points.map((point) => [point.id, [point.x, point.y]])),
    edges: [...document.querySelectorAll('#edges path')].map((path) => {
      const start = path.getPointAtLength(0);
      const end = path.getPointAtLength(path.getTotalLength());
      return {
        from: [Math.round(start.x), Math.round(start.y)],
        to: [Math.round(end.x), Math.round(end.y)],
        dashed: getComputedStyle(path).strokeDasharray !== 'none',
      };
    }),
  };
}"""
_TIP_ROWS = "(rows) => rows.map((row) => [...row.cells].map((cell) => cell.textContent))"


def _hover(page: Page, selector: str) -> None:
    """Move the mouse to the middle of the element *selector* names."""
    box = page.locator(selector).bounding_box()
    assert box is not None
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_points_hang_under_their_parents_and_a_tooltip_shows_a_points_details(
    page: Page, toy_page: Path
) -> None:
    """Every point is joined to its parents above it (a compose to both, the merged one by a
    dashed edge), and hovering or focusing a point lists its pareto.md fields in file order."""
    page.clock.install(time=0)
    page.goto(f"{toy_page.as_uri()}#frame=14")
    page.clock.pause_at(3600)  # from here on only run_for moves the page's time
    drawn = page.evaluate(_EDGES)
    at, radius = drawn["at"], drawn["radius"]
    assert len(set(map(tuple, at.values()))) == len(_POINTS)
    family = [
        (parent, point[0], merged > 0)
        for point in _POINTS[1:] for merged, parent in enumerate(point[1].split(" + "))
    ]
    assert [(parent, child) for parent, child, _merged in family] == [
        ("B", "C1"), ("C1", "C2"), ("B", "C3"), ("C1", "C4"), ("C3", "C4"), ("C4", "C5"),
        ("C4", "C6"),
    ]
    assert drawn["edges"] == [
        {
            "from": [at[parent][0], at[parent][1] + radius],
            "to": [at[child][0], at[child][1] - radius],
            "dashed": merged,
        }
        for parent, child, merged in family
    ]
    assert all(at[parent][1] < at[child][1] for parent, child, _merged in family)

    tip = page.locator("#tooltip")
    assert tip.is_hidden()
    _hover(page, '[data-id="C1"] .disc')
    assert tip.locator("tr").evaluate_all(_TIP_ROWS) == [
        list(pair) for pair in zip(_FIELDS, _POINTS[1][:7])
    ]
    dominated = "C1 \u00b7 off the frontier: a later point dominates it (DOMINATED)"
    assert tip.locator("div").first.inner_text() == dominated
    assert tip.locator(".source").inner_text() == "Details from pareto.md"
    box, view = tip.bounding_box(), page.viewport_size
    assert box is not None and view is not None
    assert box["x"] >= 0 and box["x"] + box["width"] <= view["width"]
    assert box["y"] >= 0 and box["y"] + box["height"] <= view["height"]
    # The tooltip follows the frame while its point is shown and goes when the point does.
    page.evaluate("window.discoverViz.show(3)")
    assert tip.locator("div").first.inner_text() == "C1 \u00b7 on the Pareto frontier"
    page.evaluate("window.discoverViz.show(1)")
    assert tip.is_hidden()

    # A point that never entered pareto.md says so, and markup in a detail stays text.
    page.evaluate("window.discoverViz.show(14)")
    _hover(page, '[data-id="C2"] .disc')
    assert tip.locator(".source").inner_text() == "Details from explored-ideas.md"
    assert tip.locator("tr").nth(3).locator("td").inner_text() == _MARKUP
    assert tip.locator("b, script").count() == 0
    # The reader can move onto the tooltip (to scroll long details); leaving it closes it.
    _hover(page, "#tooltip")
    page.clock.run_for(400)
    assert tip.is_visible()
    page.mouse.move(2, 2)
    assert tip.is_hidden()
    # Leaving a point closes its tooltip a quarter of a second later; Escape closes it at once.
    _hover(page, '[data-id="C4"] .disc')
    assert tip.locator("strong").inner_text() == "C4"
    page.mouse.move(2, 2)
    page.clock.run_for(249)
    assert tip.is_visible()
    page.clock.run_for(1)
    assert tip.is_hidden()
    page.focus('[data-id="C6"]')
    assert tip.locator("strong").inner_text() == "C6"
    page.keyboard.press("Escape")
    assert tip.is_hidden()


def test_the_animation_plays_by_itself_and_every_control_moves_it(
    page: Page, toy_page: Path
) -> None:
    """The page plays from the first frame on its own clock (a pick frame 2.6 s, a result
    frame 4.2 s, divided by the speed), stops at the end, and every control and key moves it
    and stops the playback; `#frame=N` and a reduced-motion setting open it paused."""
    frame = "window.discoverViz.frame"
    state = "[window.discoverViz.frame, window.discoverViz.playing]"
    page.clock.install(time=0)
    # Frozen before the page arms its first timer: only run_for moves the page's time,
    # however long the browser and this process take between two steps.
    page.clock.pause_at(3600)
    page.goto(toy_page.as_uri())
    assert page.evaluate(state) == [0, True] and page.inner_text("#play") == "Pause"
    assert page.evaluate("window.discoverViz.frameCount") == 15
    for wait, expected in ((2599, 0), (1, 1), (4199, 1), (1, 2)):
        page.clock.run_for(wait)
        assert page.evaluate(frame) == expected
    page.select_option("#speed", "4")  # takes effect from the next frame on
    for wait, expected in ((2600, 3), (1049, 3), (1, 4)):
        page.clock.run_for(wait)
        assert page.evaluate(frame) == expected
    page.click("#play")
    page.clock.run_for(60_000)
    assert page.evaluate(state) == [4, False] and page.inner_text("#play") == "Play"
    page.click("#play")
    page.clock.run_for(60_000)
    assert page.evaluate(state) == [14, False] and page.inner_text("#play") == "Replay"
    page.click("#play")
    assert page.evaluate(state) == [0, True]

    page.click("#next")
    assert page.evaluate(state) == [1, False]
    page.clock.run_for(60_000)
    assert page.evaluate(frame) == 1
    page.click("#prev")
    assert page.evaluate(frame) == 0 and page.is_disabled("#prev")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowRight")
    page.keyboard.press("ArrowLeft")
    assert page.evaluate(frame) == 1
    page.focus("#scrub")
    page.keyboard.press("ArrowRight")  # the slider moves itself: one step, not two
    assert page.evaluate(frame) == 2 and page.input_value("#scrub") == "2"
    for _step in range(3):
        page.click("#next")
    assert page.locator("#log li").count() == 3
    page.click("#log li:nth-child(2) button")  # C1 in the round log: its result frame
    assert page.evaluate(frame) == 3

    # Discarded ideas can be hidden: C2 (FAILED) goes; C5 (a TIE the finish promoted) stays.
    page.goto(f"{toy_page.as_uri()}#frame=14")
    assert page.evaluate(state) == [14, False]
    page.uncheck("#discarded")
    shown = {name: page.locator(f'[data-id="{name}"]').is_visible() for name in _NAMES}
    assert shown == dict.fromkeys(_NAMES, True) | {"C2": False}
    assert page.locator("#edges path.failed").is_hidden()
    page.check("#discarded")
    assert page.locator('[data-id="C2"]').is_visible()
    page.goto(f"{toy_page.as_uri()}#frame=99")
    assert page.evaluate(state) == [14, False]

    # The self-check the agent runs draws every frame and ends on the last one.
    page.goto(f"{toy_page.as_uri()}#frame=2")
    assert page.evaluate("window.discoverViz.selfCheck()") == {
        "ok": True, "frames": 15, "points": 7, "errors": [],
    }
    assert page.evaluate(state) == [14, False] and page.locator("#tooltip").is_hidden()
    page.emulate_media(reduced_motion="reduce")
    page.goto(toy_page.as_uri())
    page.reload()
    assert page.evaluate(state) == [0, False] and page.inner_text("#play") == "Play"


# How close an edge comes to a point it does not join, and where the points are.
_CLEARANCE = """() => {
  const data = JSON.parse(document.getElementById('discover-viz-data').textContent);
  const at = Object.fromEntries(data.points.map((point) => [point.id, point]));
  const parents = data.rounds.flatMap((round) => round.parents);
  let closest = Infinity;
  let spanning = 0;
  [...document.querySelectorAll('#edges path')].forEach((path, index) => {
    const ends = [at[parents[index]], at[data.edges[index].to]];
    spanning += ends[1].y - ends[0].y > 150;
    const others = data.points.filter((point) => !ends.includes(point));
    const length = path.getTotalLength();
    for (let along = 0; along <= length; along += 3) {
      const {x, y} = path.getPointAtLength(along);
      for (const point of others) closest = Math.min(closest, Math.hypot(x - point.x, y - point.y));
    }
  });
  return {
    closest, spanning, radius: data.radius, edges: parents.length,
    places: new Set(data.points.map((point) => `${point.x},${point.y}`)).size,
    below: data.rounds.every(
      (round) => round.parents.every((parent) => at[parent].y < at[round.id].y)),
    inside: data.points.every((point) => point.x >= data.radius && point.y >= data.radius
      && point.x <= data.width - data.radius && point.y <= data.height - data.radius),
  };
}"""


def _wide_story(points: int) -> dict[str, Any]:
    """Return a sound story of *points* points from a fixed seed: a bushy tree in which every
    fifth point is a compose with a point picked anywhere, so its edge spans many rows."""
    rng = random.Random(7)
    rounds: list[dict[str, Any]] = []
    for number in range(points):
        parents = [f"P{rng.randrange(max(0, number - 9), number)}"] if number else []
        merged = f"P{rng.randrange(number)}" if number and number % 5 == 0 else ""
        if merged and merged not in parents:
            parents.append(merged)
        rounds.append({
            "id": f"P{number}", "parents": parents, "idea": "An idea.",
            "operator": "compose" if len(parents) > 1 else "variant" if number else "baseline",
            "picked": "Picked.", "score": "1", "status": "KEPT", "outcome": "Kept.",
            "details": {"id": f"P{number}"}, "source": "pareto.md",
        })
    return {"task_id": "", "title": "A wide run", "goal": "None.", "metric": "score",
            "rounds": rounds}


def test_a_wide_run_keeps_every_edge_clear_of_the_points_it_only_passes(
    page: Page, tmp_path: Path
) -> None:
    """Sixty points with composes across many rows: no two points share a place, every point
    is below its parents and inside the drawing, no edge comes near a point it does not join,
    and each step scrolls its point into view in a drawing wider than its frame."""
    parsed = story.parse(_wide_story(60))
    assert isinstance(parsed, story.Story)
    frames, errors = story.replay(parsed)
    assert not errors and len(frames) == 120
    target = tmp_path / "wide.html"
    target.write_text(animation.render(parsed, frames), encoding="utf-8")
    page.goto(f"{target.as_uri()}#frame=119")

    measured = page.evaluate(_CLEARANCE)
    assert measured["edges"] == sum(len(round_.parents) for round_ in parsed.rounds)
    assert measured["spanning"] >= 5, measured
    assert measured["closest"] >= measured["radius"] + 6, measured
    assert measured["places"] == 60 and measured["below"] and measured["inside"], measured

    canvas = page.locator("#canvas").bounding_box()
    assert canvas is not None
    assert page.evaluate("document.getElementById('graph').clientWidth") > canvas["width"]
    for index in (1, 61, 119):
        page.evaluate(f"window.discoverViz.show({index})")
        disc = page.locator(".node.focus .disc").bounding_box()
        assert disc is not None
        assert canvas["x"] <= disc["x"] <= canvas["x"] + canvas["width"] - disc["width"], index
        assert canvas["y"] <= disc["y"] <= canvas["y"] + canvas["height"] - disc["height"], index
    assert page.evaluate("window.discoverViz.selfCheck().ok") is True


def test_agent_run_reads_the_trajectory_writes_the_story_and_renders_the_page(
    toy_run: str, tmp_path: Path
) -> None:
    """A run configured as the daemon configures it: the scripted model asks for the evidence,
    writes the story, renders it and finishes.  The test checks the real requests: the SEA's
    own prompt, its two tools beside the shell, edit and browser tools, and the tool results."""
    evidence, tale, target = (
        str(tmp_path / name) for name in ("evidence.md", "story.json", "out/viz.html")
    )
    summary = "<p>7 points, final frontier B, C5.</p>"
    script = [
        tool_call_body(
            "discover_trajectory", {"task": toy_run[:8], "evidence_file": evidence}, 500
        ),
        tool_call_body(
            "Write", {"file_path": tale, "content": json.dumps(_toy_story(toy_run))}, 600
        ),
        tool_call_body("render_discover_viz", {"story_file": tale, "html_file": target}, 700),
        finish_body(summary, prompt_tokens=800),
    ]
    run = sea_commands.evaluate_sea([DiscoverVizSea()], toy_run[:8])
    assert run.prompt == f"{toy_run[:8]}\n\n{sea.ADD_TO_PROMPT}"
    parsed, agentic = run_sea(run, script, tmp_path, max_steps=6)
    assert parsed["success"] is True and parsed["summary"] == summary

    assert len(agentic) == 4
    assert system_message(agentic[0]).startswith(sea.SYSTEM_PROMPT)
    user = next(message for message in agentic[0]["messages"] if message["role"] == "user")
    assert run.prompt in str(user["content"])
    names = tool_names(agentic[0])
    assert {
        "discover_trajectory", "render_discover_viz", "Bash", "Read", "Write", "Edit",
        "go_to_url", "evaluate_script", "screenshot", "show_browser", "finish",
    } <= names, names
    assert not names & {"create_worktree", "commit_and_merge", "run_agent", "memory_write"}, names
    results = tool_results(agentic[3])
    assert results[0].startswith(f"/discover run {toy_run}: evidence written to {evidence}")
    assert results[2].startswith(
        f"Animation written to {target}: 7 points, 15 frames, final frontier B, C5."
    )
    assert f"# Evidence of the /discover run {toy_run}" in Path(evidence).read_text()
    assert "<title>Churn model accuracy</title>" in Path(target).read_text()

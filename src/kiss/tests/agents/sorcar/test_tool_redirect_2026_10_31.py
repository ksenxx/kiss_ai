# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A shell rewrite's ``Write``/``Edit`` is made through the agent's own tool path.

``UsefulTools.Bash`` applies the ``Write``/``Edit`` a here-document copy, a
python ``.replace`` rewrite or a ``sed -i`` stands for.  Made by the tool
itself, that call bypassed the agent: the tool-call hook (a SEA's ledger
gate, its Edit/Write bookkeeping), the worktree gate and the event log
saw a Bash call only.  Under an agent (``UsefulTools.redirect``) the tool
raises :class:`~kiss.core.tool_verdict.ToolRedirect` and
``KISSAgent._execute_tool`` makes the call through
``_guarded_execute_tool``: the hook, the guard, the ``tool_call`` /
``tool_result`` events and the refusal bookkeeping see it as if the
model had issued it; the rewrite's result opens with what it was made
as and ends with that call's result.

Driven through the real ``KISSAgent`` dispatch (no model: the calls are
issued by hand) over real files, and through a real ``ChatSorcarAgent``
whose work dir is a fresh git checkout for the worktree gate.  Nothing
is mocked.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

import kiss.core.prompt_cache_keepalive as keepalive
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.edit_guards import HEREDOC_VERDICT, edit_call, rewrite_guard
from kiss.agents.sorcar.useful_tools import CD_VERDICT, UsefulTools, sleep_refusal
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_agent import KISSAgent
from kiss.core.printer import Printer
from kiss.core.tool_verdict import (
    ALLOW,
    REISSUE_REMINDER,
    ToolRedirect,
    Verdict,
    call_text,
    refuse,
)
from kiss.tests.conftest import host_sed_is_gnu, posix_only
from kiss.tests.core import local_anthropic_server as anthropic_server
from kiss.tests.server.conftest import init_repo

pytestmark = posix_only("the host Bash tool runs bash here-documents")

HEREDOC = "cat > notes.md <<'EOF'\nnew text\nEOF"
PYTHON_TWO_REPLACES = (
    "python3 - <<'EOF'\n"
    "p = 'notes.md'\n"
    "s = open(p).read()\n"
    "s = s.replace('old', 'new').replace('text', 'words')\n"
    "open(p, 'w').write(s)\n"
    "EOF"
)
SED_MULTILINE = "sed -i 's/old/old\\nmore/' notes.md"


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
    """A ``KISSAgent`` dispatching to one ``UsefulTools`` under a recording hook and printer.

    Args:
        work_dir: The tools' work dir, holding ``notes.md`` (``old text``).
        guard: The agent's ``tool_call_guard`` (``SorcarAgent._guard_tool_call``
            installs the worktree gate); none by default.
    """

    def __init__(self, work_dir: Path, guard: Any = None) -> None:
        (work_dir / "notes.md").write_text("old text\n")
        self.printer = _RecordingPrinter()
        self.hook_calls: list[tuple[str, dict[str, Any]]] = []
        #: The verdict the hook gives a ``Write``/``Edit`` (``Bash``/``Read`` always pass).
        self.verdict = ALLOW
        self.tools = UsefulTools(work_dir=str(work_dir))
        self.tools.redirect = True
        self.agent: Any = KISSAgent("redirect-harness")
        self.agent.printer = self.printer
        self.agent.tool_call_hook = self.hook
        self.agent.tool_call_guard = guard
        self.agent.function_map = {
            "Bash": self.tools.Bash,
            "Read": self.tools.Read,
            "Write": self.tools.Write,
            "Edit": self.tools.Edit,
        }

    def hook(self, name: str, args: dict[str, Any]) -> Verdict:
        """Record the call and give :attr:`verdict` to a ``Write``/``Edit``."""
        self.hook_calls.append((name, dict(args)))
        return self.verdict if name in ("Write", "Edit") else ALLOW

    def call(self, name: str, **args: Any) -> str:
        """Issue the tool call *name* with *args* as the agent loop does; return its result."""
        _, response, _ = self.agent._guarded_execute_tool({"name": name, "arguments": args})
        return str(response)

    def events(self, kind: str) -> list[Any]:
        """The contents of the printed events of type *kind*, in order."""
        return [content for etype, content, _ in self.printer.events if etype == kind]


def test_heredoc_copy_is_made_as_a_write_through_the_agent(tmp_path: Path) -> None:
    """The hook, the events and the result all show the ``Write``; it counts as no extra call."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    applied = f'Applied as Write(file_path="{notes}", content="new text\\n"), whose result follows:'
    assert out.startswith(applied + "\nSuccessfully wrote 9 characters"), out
    assert notes.read_text() == "new text\n"
    assert [name for name, _ in h.hook_calls] == ["Read", "Bash", "Write"]
    assert h.hook_calls[-1][1] == {"file_path": str(notes), "content": "new text\n"}
    # The event log shows the Write with its own call and result, nested in the Bash call.
    assert h.events("tool_call") == ["Read", "Bash", "Write"]
    results = h.events("tool_result")
    assert results[-2].startswith("Successfully wrote") and results[-1] == out
    inputs = [kw["tool_input"] for et, _, kw in h.printer.events if et == "tool_call"]
    assert inputs[-1] == {"file_path": str(notes), "content": "new text\n"}
    assert h.agent.tool_calls_made == 2, "the redirected Write is not a call of its own"
    # An append goes through a Write of the old content followed by the body.
    out = h.call("Bash", command="cat >> notes.md <<'EOF'\nmore\nEOF", description="append")
    assert out.startswith("Appended through Write(") and ", whose result follows:\n" in out
    assert notes.read_text() == "new text\nmore\n"
    assert h.hook_calls[-1] == ("Write", {"file_path": str(notes), "content": "new text\nmore\n"})


def test_python_and_sed_rewrites_are_made_as_edits_through_the_agent(tmp_path: Path) -> None:
    """One substitution is an inline ``Edit``; several go as its ``edits`` JSON."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    out = h.call("Bash", command=SED_MULTILINE, description="rewrite")
    call = edit_call(str(notes), [("old", "old\nmore", False)])
    assert out.startswith(f"Applied as {call}, whose result follows:\nSuccessfully replaced 1"), out
    assert notes.read_text() == "old\nmore text\n"
    assert h.hook_calls[-1] == ("Edit", {
        "file_path": str(notes), "old_string": "old", "new_string": "old\nmore",
        "replace_all": False, "exact": True,
    })
    notes.write_text("old text\n")
    out = h.call("Bash", command=PYTHON_TWO_REPLACES, description="rewrite")
    assert "Successfully replaced 2 occurrence(s)" in out and notes.read_text() == "new words\n"
    name, args = h.hook_calls[-1]
    edits = args["edits"]
    assert name == "Edit" and args["file_path"] == str(notes) and isinstance(edits, str)
    assert '"old_string": "old", "new_string": "new", "replace_all": true' in edits
    assert h.events("tool_call").count("Edit") == 2


def test_a_refused_redirected_write_is_the_rewrites_result(tmp_path: Path) -> None:
    """A hook's refusal of the Write reaches the model, with the usual re-issue bookkeeping."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    h.verdict = refuse("the ledger has no entry for notes.md")
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert out == (
        f'Applied as Write(file_path="{notes}", content="new text\\n"), whose result follows:\n'
        "REFUSED: the Write call did NOT run and had no effect. "
        "the ledger has no entry for notes.md"
    )
    assert notes.read_text() == "old text\n"
    refused = [kw for et, _, kw in h.printer.events if et == "tool_result" and kw.get("is_error")]
    assert len(refused) == 1 and refused[0]["tool_name"] == "Write"
    # The refused Write is owed a re-issue at the next step, as a model-issued one would be.
    out = h.call("Read", file_path=str(notes))
    assert out.endswith(REISSUE_REMINDER.format(name="Write"))
    # Allowed again, the same command settles it; a Write of the model's own too.
    h.verdict = ALLOW
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert "Successfully wrote" in out and REISSUE_REMINDER.format(name="Write") not in out
    assert notes.read_text() == "new text\n"
    h.verdict = refuse("not now")
    assert "REFUSED: the Write call" in h.call("Bash", command=HEREDOC, description="rewrite")
    h.verdict = ALLOW
    out = h.call("Write", file_path=str(notes), content="own\n")
    assert out.startswith("Successfully wrote") and REISSUE_REMINDER.format(name="Write") not in out
    assert h.call("Read", file_path=str(notes)).endswith("own\n")


def test_an_unshown_file_is_refused_before_any_redirect(tmp_path: Path) -> None:
    """The read-before-modify rule still decides first: no Write reaches the hook."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert out.startswith(HEREDOC_VERDICT.format(path=notes))
    assert notes.read_text() == "old text\n"
    assert [name for name, _ in h.hook_calls] == ["Bash"]
    assert h.events("tool_call") == ["Bash"]


def test_the_worktree_gate_refuses_the_redirected_write_of_a_tracked_file(tmp_path: Path) -> None:
    """In a repository's main checkout the gate sees the Write and asks for a worktree first."""
    repo = tmp_path / "repo"
    init_repo(repo)
    agent = ChatSorcarAgent("redirect-gate")
    agent.work_dir = str(repo)
    agent._run_tool_names = frozenset({"Bash", "Read", "Write", "Edit", "create_worktree"})
    h = _Harness(repo, agent._guard_tool_call)
    seed = repo / "seed.txt"
    h.call("Read", file_path=str(seed))
    out = h.call("Bash", command="cat > seed.txt <<'EOF'\nchanged\nEOF", description="rewrite")
    assert out.startswith(f'Applied as Write(file_path="{seed}", content="changed\\n")')
    assert "REFUSED: the Write call did NOT run and had no effect. " in out
    assert f"Error: Write of {seed} refused: it is inside the git repository" in out
    assert "Call create_worktree() first, then retry the Write." in out
    assert seed.read_text() == "seed\n"
    assert h.events("tool_call") == ["Read", "Bash", "Write"]
    # A git-ignored (shown) file passes the gate and is written.
    (repo / ".gitignore").write_text("notes.md\n")
    h.call("Read", file_path=str(repo / "notes.md"))
    out = h.call("Bash", command=HEREDOC, description="rewrite")
    assert "Successfully wrote" in out and (repo / "notes.md").read_text() == "new text\n"


def test_without_an_agent_the_tool_applies_the_call_itself(tmp_path: Path) -> None:
    """A direct ``UsefulTools`` (no agent) keeps applying the Write/Edit, as before."""
    tools = UsefulTools(work_dir=str(tmp_path))
    notes = tmp_path / "notes.md"
    notes.write_text("old text\n")
    tools.Read(str(notes))
    assert not tools.redirect
    out = tools.Bash(SED_MULTILINE, "rewrite")
    assert out.startswith("Applied as Edit(") and notes.read_text() == "old\nmore text\n"
    out = tools.Bash(HEREDOC, "rewrite")
    assert out.startswith(f'Applied as Write(file_path="{notes}", content="new text\\n"): ')
    assert notes.read_text() == "new text\n"
    # With the flag, the same tool raises the redirect for an agent to run.
    tools.redirect = True
    with pytest.raises(ToolRedirect) as raised:
        tools.Bash(HEREDOC, "rewrite")
    assert (raised.value.name, raised.value.arguments) == (
        "Write", {"file_path": str(notes), "content": "new text\n"},
    )
    assert raised.value.note.startswith("Applied as Write(")
    assert str(raised.value) == raised.value.note
    assert notes.read_text() == "new text\n"


# --- the rest of a compound command (``[cd DIR &&] <rewrite> && <rest>``) ------


def _rest(
    command: str,
    description: str,
    timeout_seconds: float = 300,
    background: bool = False,
    on_timeout: str = "kill",
) -> dict[str, Any]:
    """The arguments of the follow-up ``Bash`` call running *command*, the rest of a command.

    The rest runs as the model asked the whole command to run: the
    original call's ``background`` and ``on_timeout`` come along.
    """
    return {
        "command": command, "description": description,
        "timeout_seconds": timeout_seconds, "max_output_chars": 8000,
        "background": background, "on_timeout": on_timeout,
    }


def test_the_rest_of_a_compound_command_runs_after_the_applied_write(tmp_path: Path) -> None:
    """``cd DIR && cat > f <<EOF ... EOF && <rest>``: the Write, then the rest as a Bash call."""
    h = _Harness(tmp_path)
    app = tmp_path / "app"
    app.mkdir()
    notes = app / "notes.md"
    notes.write_text("old text\n")
    h.call("Read", file_path=str(notes))
    command = "cd app && cat > notes.md <<'EOF' && cat notes.md && pwd\nnew text\nEOF"
    out = h.call("Bash", command=command, description="rewrite, show", timeout_seconds=30)
    rest = _rest(f"cd {app} && cat notes.md && pwd", "rewrite, show", 30)
    assert out == (
        f'Applied as Write(file_path="{notes}", content="new text\\n"), whose result follows:\n'
        f"Successfully wrote 9 characters to {notes}\n"
        f"Then made {call_text('Bash', rest)}, whose result follows:\n"
        f"new text\n{app}\n"
    ), out
    assert notes.read_text() == "new text\n"
    # The hook and the events see the Write and the rest's Bash call, nested in the model's.
    assert [name for name, _ in h.hook_calls] == ["Read", "Bash", "Write", "Bash"]
    assert h.hook_calls[-1] == ("Bash", rest)
    assert h.events("tool_call") == ["Read", "Bash", "Write", "Bash"]
    assert h.events("tool_result")[-2] == f"new text\n{app}\n"
    assert h.agent.tool_calls_made == 2, "neither the Write nor the rest is a call of its own"
    # Without a cd the rest runs in the work dir; a python rewrite's rest runs the edited file.
    hello = tmp_path / "hello.py"
    hello.write_text("print('hello world')\n")
    h.call("Read", file_path=str(hello))
    program = (
        "python3 - <<'EOF' && python3 hello.py\n"
        "p = 'hello.py'\n"
        "s = open(p).read()\n"
        "s = s.replace('world', 'there')\n"
        "open(p, 'w').write(s)\n"
        "EOF"
    )
    out = h.call("Bash", command=program, description="edit, run", timeout_seconds=30)
    rest = _rest("python3 hello.py", "edit, run", 30)
    assert out.endswith(
        f"\nThen made {call_text('Bash', rest)}, whose result follows:\nhello there\n"
    ), out
    assert out.startswith("Applied as Edit(") and "Successfully replaced 1 occurrence(s)" in out
    assert h.hook_calls[-2:] == [("Edit", {
        "file_path": str(hello), "old_string": "world", "new_string": "there",
        "replace_all": True, "exact": True,
    }), ("Bash", rest)]


# What a failed Edit's not-found error adds for a one-line file (edit_matching.nearest_region).
NEAREST = (
    " The closest region is lines 1-1:\n    1 | {text}\n"
    "Re-issue the Edit with the exact text of those lines as old_string."
)


def test_the_rest_is_not_run_when_the_rewrite_did_not_go_through(tmp_path: Path) -> None:
    """A refused Write, a failed Edit or a missing ``cd`` directory: nothing after them runs."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    h.verdict = refuse("not now")
    command = "cat > notes.md <<'EOF' && touch after-write\nnew text\nEOF"
    out = h.call("Bash", command=command, description="rewrite, touch")
    rest = _rest("touch after-write", "rewrite, touch")
    assert out.endswith(
        "REFUSED: the Write call did NOT run and had no effect. not now\n"
        f"{call_text('Bash', rest)} was not made: the Write did not go through."
    ), out
    assert notes.read_text() == "old text\n" and not (tmp_path / "after-write").exists()
    assert [name for name, _ in h.hook_calls] == ["Read", "Bash", "Write"]
    # The refused Write is still owed its re-issue; the rest is reported as not made only once.
    assert h.call("Read", file_path=str(notes)).endswith(REISSUE_REMINDER.format(name="Write"))
    h.verdict = ALLOW
    # An Edit that fails (its old_string is not in the file) is no success either.
    command = "sed -i 's/absent/present\\n/' notes.md && touch after-edit"
    out = h.call("Bash", command=command, description="edit, touch")
    rest = _rest("touch after-edit", "edit, touch")
    call = edit_call(str(notes), [("absent", "present\n", False)])
    assert out == (
        f"Applied as {call}, whose result follows:\nError: String not found in file."
        f"{NEAREST.format(text='old text')}\n"
        f"{call_text('Bash', rest)} was not made: the Edit did not go through."
    ), out
    assert not (tmp_path / "after-edit").exists()
    assert h.events("tool_call")[-2:] == ["Bash", "Edit"]
    # A cd to a missing directory is refused before anything is written: bash would run
    # nothing after it.
    for command in (
        "cd nowhere && cat > notes.md <<'EOF' && touch after-cd\nnew text\nEOF",
        f"cd nowhere && cat > {notes} <<'EOF' && touch after-cd\nnew text\nEOF",
        f"cd nowhere && sed -i 's/old/new\\n/' {notes} && touch after-cd",
    ):
        out = h.call("Bash", command=command, description="cd, rewrite")
        assert out == CD_VERDICT.format(cwd=tmp_path / "nowhere"), out
        assert h.hook_calls[-1][0] == "Bash" and h.events("tool_call")[-1] == "Bash"
    assert notes.read_text() == "old text\n" and not (tmp_path / "after-cd").exists()


def test_a_refused_rest_is_reported_after_the_applied_rewrite(tmp_path: Path) -> None:
    """The rest is a ``Bash`` call like any other: the sleep rule and the guards apply to it."""
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    command = "cat > notes.md <<'EOF' && sleep 999\nnew text\nEOF"
    out = h.call("Bash", command=command, description="rewrite, wait", timeout_seconds=30)
    rest = _rest("sleep 999", "rewrite, wait", 30)
    assert out == (
        f'Applied as Write(file_path="{notes}", content="new text\\n"), whose result follows:\n'
        f"Successfully wrote 9 characters to {notes}\n"
        f"Then made {call_text('Bash', rest)}, whose result follows:\n{sleep_refusal(999)}"
    ), out
    assert notes.read_text() == "new text\n"
    assert h.events("tool_call") == ["Read", "Bash", "Write", "Bash"]
    # A rest the hook refuses is owed its re-issue, as any refused Bash call is.

    def hook(name: str, args: dict[str, Any]) -> Verdict:
        h.hook_calls.append((name, dict(args)))
        return refuse("no touching") if args.get("command", "").startswith("touch") else ALLOW

    h.agent.tool_call_hook = hook
    out = h.call("Bash", command="cat > notes.md <<'EOF' && touch t\nv2\nEOF", description="x")
    assert out.endswith(
        f"Then made {call_text('Bash', _rest('touch t', 'x'))}, whose result follows:\n"
        "REFUSED: the Bash call did NOT run and had no effect. no touching"
    ), out
    assert notes.read_text() == "v2\n" and not (tmp_path / "t").exists()
    assert h.call("Read", file_path=str(notes)).endswith(REISSUE_REMINDER.format(name="Bash"))


def test_the_rest_runs_as_the_model_asked_the_whole_command_to_run(tmp_path: Path) -> None:
    """``background=True`` and ``on_timeout="background"`` reach the rest of a compound rewrite.

    The follow-up call carried the description, deadline and output
    budget only, so the rest of a command the model backgrounded ran
    in the foreground, under the deadline the model had put on the
    ``bash_job`` wait instead (review of 2026-10-31, finding 3).
    """
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    h.call("Read", file_path=str(notes))
    command = "cat > notes.md <<'EOF' && sleep 0.5 && echo after\nnew text\nEOF"
    out = h.call(
        "Bash", command=command, description="rewrite, job", timeout_seconds=5, background=True,
    )
    rest = _rest("sleep 0.5 && echo after", "rewrite, job", 5, background=True)
    assert out.startswith(
        f'Applied as Write(file_path="{notes}", content="new text\\n"), whose result follows:\n'
        f"Successfully wrote 9 characters to {notes}\n"
        f"Then made {call_text('Bash', rest)}, whose result follows:\nStarted background job "
    ), out
    assert notes.read_text() == "new text\n" and h.hook_calls[-1] == ("Bash", rest)
    job = out.split("Started background job ")[1].split()[0]
    assert list(h.tools._jobs) == [job], "the rest is the job the model asked for"
    report = h.tools.bash_job(job, action="wait", timeout_seconds=10)
    assert "exited with code 0" in report and report.endswith("after\n"), report
    # on_timeout="background": a rest outliving the deadline goes on as a job.
    command = "sed -i 's/new/newer\\n/' notes.md && sleep 2 && echo late"
    out = h.call(
        "Bash", command=command, description="edit, wait", timeout_seconds=1,
        on_timeout="background",
    )
    rest = _rest("sleep 2 && echo late", "edit, wait", 1, on_timeout="background")
    assert (
        f"Then made {call_text('Bash', rest)}, whose result follows:\nStill running after 1s"
    ) in out, out
    assert notes.read_text() == "newer\n text\n" and h.hook_calls[-1] == ("Bash", rest)
    job = out.split("background job ")[1].split()[0]
    report = h.tools.bash_job(job, action="wait", timeout_seconds=10)
    assert "exited with code 0" in report and report.endswith("late\n"), report
    # A plain call's rest runs in the foreground, as before.
    command = "cat > notes.md <<'EOF' && echo now\nv3\nEOF"
    out = h.call("Bash", command=command, description="d", timeout_seconds=30)
    assert out.endswith("whose result follows:\nnow\n") and len(h.tools._jobs) == 2, out
    assert h.hook_calls[-1] == ("Bash", _rest("echo now", "d", 30))


def test_a_per_line_sed_applies_the_edit_exact_for_the_file(tmp_path: Path) -> None:
    """``edits_for`` settles ``replace_all`` from the file; a line holding the pattern twice has
    no exact Edit, so the rewrite guards decide (a multi-line sed is refused, a plain one runs).
    """
    h = _Harness(tmp_path)
    notes = tmp_path / "notes.md"
    notes.write_text("old a\nold b\n")
    h.call("Read", file_path=str(notes))
    sed = "sed -i 's/old/new/' notes.md && cat notes.md"
    out = h.call("Bash", command=sed, description="d", timeout_seconds=30)
    call = edit_call(str(notes), [("old", "new", True)])
    assert out.startswith(
        f"Applied as {call}, whose result follows:\nSuccessfully replaced 2 occurrence(s) in "
        f"{notes}\n"
    ), out
    assert (
        f"\nThen made {call_text('Bash', _rest('cat notes.md', 'd', 30))}, whose result "
        "follows:\nnew a\nnew b\n"
    ) in out, out
    assert h.hook_calls[-2] == ("Edit", {
        "file_path": str(notes), "old_string": "old", "new_string": "new", "replace_all": True,
        "exact": True,
    })
    notes.write_text("old old\n")
    multiline = "sed -i 's/old/new\\n/' notes.md && cat notes.md"
    out = h.call("Bash", command=multiline, description="d")
    assert out == rewrite_guard(multiline) and notes.read_text() == "old old\n", out
    if not host_sed_is_gnu():
        pytest.skip("needs GNU sed as the oracle")
    out = h.call("Bash", command=sed, description="d")
    assert out == "new old\n" and notes.read_text() == "new old\n", out
    assert [name for name, _ in h.hook_calls][-2:] == ["Bash", "Bash"]


def test_the_edit_of_a_rewrite_is_refused_once_the_file_no_longer_holds_what_it_was_read_from(
    tmp_path: Path,
) -> None:
    """Audit 2026-11-12 B5/A7: the pairs exact for the text a rewrite was read against were
    applied to whatever the file held when the Edit landed.  A writer in between (a line now
    holding the sed's pattern twice, a text the python's second ``.replace`` now finds) made
    the Edit a change the command would not make; it is refused and nothing is written.  (An
    Edit whose strings are gone fails as any Edit does: the exact-conversions module.)
    """
    tools = UsefulTools(work_dir=str(tmp_path))
    tools.redirect = True
    notes = tmp_path / "notes.md"
    for command, what, shown, pairs, landed in (
        ("sed -i 's/old/new/' notes.md", "`sed -i` call", "old a\nold b\n",
         {"old_string": "old", "new_string": "new", "replace_all": True}, "old old\nold b\n"),
        (PYTHON_TWO_REPLACES, "python heredoc", "old\n",
         {"old_string": "old", "new_string": "new", "replace_all": True}, "old text\n"),
    ):
        notes.write_text(shown)
        tools.Read(str(notes))
        with pytest.raises(ToolRedirect) as raised:
            tools.Bash(command, "rewrite")
        arguments = raised.value.arguments
        assert arguments == {"file_path": str(notes), **pairs, "exact": True}
        notes.write_text(landed)
        assert tools.Edit(**arguments) == (
            f"Error: {notes} changed since the {what} was read as this Edit, which the command "
            "no longer stands for: nothing was written. Re-issue the command."
        )
        assert notes.read_text() == landed
        # The same call is made while the file holds what the command was read against.
        notes.write_text(shown)
        with pytest.raises(ToolRedirect):
            tools.Bash(command, "rewrite")
        assert tools.Edit(**arguments).startswith("Successfully replaced")
        assert notes.read_text() == shown.replace("old", "new")
        # An Edit of the model's own (no rewrite stands behind it) is made as asked.
        notes.write_text(landed)
        assert tools.Edit(**arguments).startswith("Successfully replaced")
        assert notes.read_text() == landed.replace("old", "new")


def test_without_an_agent_the_tool_runs_the_rest_itself(tmp_path: Path) -> None:
    """A direct ``UsefulTools`` makes the Write/Edit and then the rest through its own ``Bash``."""
    tools = UsefulTools(work_dir=str(tmp_path))
    app = tmp_path / "app"
    app.mkdir()
    notes = app / "notes.md"
    notes.write_text("old text\n")
    tools.Read(str(notes))
    command = "cd app && cat > notes.md <<'EOF' && cat notes.md\nnew text\nEOF"
    out = tools.Bash(command, "rewrite, show", timeout_seconds=30)
    rest = _rest(f"cd {app} && cat notes.md", "rewrite, show", 30)
    assert out == (
        f'Applied as Write(file_path="{notes}", content="new text\\n"): Successfully wrote 9 '
        f"characters to {notes}\nThen made {call_text('Bash', rest)}, whose result follows:\n"
        "new text\n"
    ), out
    out = tools.Bash("sed -i 's/absent/present\\n/' app/notes.md && touch app/after", "edit")
    touch = _rest("touch app/after", "edit")
    call = edit_call(str(notes), [("absent", "present\n", False)])
    assert out == (
        f"Not applied: the command stands for {call}, which failed: Error: String not found "
        f"in file.{NEAREST.format(text='new text')}\n"
        f"{call_text('Bash', touch)} was not made: the Edit did not go through."
    ), out
    assert not (app / "after").exists()
    # Under an agent the redirect carries the rest as its follow-up call.
    tools.redirect = True
    with pytest.raises(ToolRedirect) as raised:
        tools.Bash(command, "rewrite, show", timeout_seconds=30)
    assert raised.value.followup == ("Bash", rest)
    with pytest.raises(ToolRedirect) as raised:
        tools.Bash("cat > app/notes.md <<'EOF'\nv3\nEOF", "rewrite")
    assert raised.value.followup is None


# --- the rest of a long call runs under the call's own prompt-cache keep-alive -----

_PING_THREADS_SEEN: list[int] = []
"""The prompt-cache ping threads alive while each rest of :func:`Bash` ran."""


def Bash(  # noqa: N802 - registered under the tool's name
    command: str, description: str, timeout_seconds: float = 300, max_output_chars: int = 8000,
) -> str:
    """A ``Bash`` standing in for the host tool: a here-document is redirected, its rest sleeps.

    Args:
        command: The command; one with a here-document is the rewrite ``sleep 1`` follows.
        description: The call's description.
        timeout_seconds: The call's deadline, handed on to the rest.
        max_output_chars: The call's output budget, handed on to the rest.
    """
    if "<<" in command:
        rest = {
            "command": "sleep 1", "description": description,
            "timeout_seconds": timeout_seconds, "max_output_chars": max_output_chars,
        }
        raise ToolRedirect(
            "Write", {"file_path": "notes.md", "content": "new\n"},
            'Applied as Write(file_path="notes.md", content="new\\n"), whose result follows:',
            ("Bash", rest),
        )
    time.sleep(1.0)
    _PING_THREADS_SEEN.append(
        sum(thread.name == "prompt-cache-keepalive" for thread in threading.enumerate())
    )
    return "slept"


def Write(file_path: str, content: str) -> str:  # noqa: N802 - registered under the tool's name
    """A ``Write`` that writes nothing: the redirected call the rest follows.

    Args:
        file_path: The file.
        content: Its new text.
    """
    return f"Successfully wrote {len(content)} characters to {file_path}"


def test_the_rest_of_a_kept_warm_call_starts_no_second_ping_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long ``Bash`` call pings the prompt cache while it runs; the rest it hands on with the
    same deadline runs inside that call and must not start a second pinger on the model.
    """
    monkeypatch.setattr(keepalive, "KEEP_ALIVE_INTERVAL_SECONDS", 0.3)
    monkeypatch.setattr(DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "local")
    _PING_THREADS_SEEN.clear()
    script = [
        anthropic_server.tool_use_message("Bash", {
            "command": "cat > notes.md <<'EOF' && sleep 1\nnew\nEOF",
            "description": "rewrite, wait", "timeout_seconds": 600,
        }, 20_000, "toolu_1"),
        anthropic_server.tool_use_message("finish", {"result": "<p>done</p>"}, 21_000, "toolu_2"),
    ]
    with anthropic_server.serve(script) as (url, requests):
        monkeypatch.setenv("ANTHROPIC_BASE_URL", url)
        agent = KISSAgent("redirect-keepalive")
        agent.run(
            model_name=anthropic_server.MODEL,
            prompt_template="Rewrite notes.md and wait, then finish.",
            tools=[Bash, Write],
            max_steps=4,
            max_budget=50.0,
            verbose=False,
        )
    assert _PING_THREADS_SEEN == [1], "the rest ran under the outer call's one ping thread"
    assert [r for r in requests if not r.get("stream")], "the outer call was kept warm"
    rest = {
        "command": "sleep 1", "description": "rewrite, wait",
        "timeout_seconds": 600, "max_output_chars": 8000,
    }
    step_two = [r for r in requests if r.get("stream")][1]
    result = step_two["messages"][-1]["content"][0]["content"]
    assert result.startswith(
        'Applied as Write(file_path="notes.md", content="new\\n"), whose result follows:\n'
        "Successfully wrote 4 characters to notes.md\n"
        f"Then made {call_text('Bash', rest)}, whose result follows:\nslept"
    ), result
    assert agent.tool_calls_made == 1

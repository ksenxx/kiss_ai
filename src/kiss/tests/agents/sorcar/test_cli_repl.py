# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the Claude-Code-style ``sorcar`` CLI REPL.

These exercise real behaviour end to end: the completer runs against
real project files and the real history database, and the REPL loop is
driven through a real subprocess reading piped stdin.  No model calls
are made because the tests only submit slash commands and EOF — a task
line is never sent, so ``agent.run`` is never invoked.
"""

from __future__ import annotations

import os
import re
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.cli_repl import (
    SLASH_COMMANDS,
    CliCompleter,
)


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect the history DB to an isolated temp directory."""
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    yield kiss_dir
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR = saved


def _write_project(tmp_path: Path) -> Path:
    """Create a tiny project tree and return its directory."""
    (tmp_path / "alpha.py").write_text("def alpha_function():\n    return 1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "beta_module.py").write_text("beta = 2\n")
    return tmp_path


def test_at_mention_completion_inserts_relative_path(tmp_path: Path, kiss_db) -> None:
    """``@`` mentions complete to ``./<path>`` like the extension."""
    project = _write_project(tmp_path)
    completer = CliCompleter(str(project))
    matches = completer._build_matches("look at @alpha")
    assert matches, "expected at least one file suggestion"
    assert matches[0] == "look at ./alpha.py "
    assert all(m.startswith("look at ./") for m in matches)


def test_at_mention_completion_matches_nested_files(tmp_path: Path, kiss_db) -> None:
    """Nested files are reachable through the ``@`` picker."""
    project = _write_project(tmp_path)
    completer = CliCompleter(str(project))
    matches = completer._build_matches("edit @beta")
    assert any(m == "edit ./src/beta_module.py " for m in matches)


def test_slash_command_completion(tmp_path: Path) -> None:
    """Typing ``/`` and a prefix completes known slash commands."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/he") == ["/help "]
    all_cmds = completer._build_matches("/")
    assert "/clear " in all_cmds
    assert "/exit " in all_cmds


def test_model_command_completion_lists_all_models(tmp_path: Path) -> None:
    """``/model `` with no partial completes to every candidate model."""
    from kiss.core.models.model_info import get_completion_model_names

    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/model ")
    expected = [f"/model {n}" for n in get_completion_model_names()]
    assert matches == expected
    assert matches, "expected at least one model suggestion"


def test_model_command_completion_by_prefix(tmp_path: Path) -> None:
    """``/model <prefix>`` completes model names that start with the prefix."""
    from kiss.core.models.model_info import get_completion_model_names

    completer = CliCompleter(str(tmp_path))
    sample = get_completion_model_names()[0]
    prefix = sample[:4]
    matches = completer._build_matches(f"/model {prefix}")
    assert matches, "expected suggestions for a known prefix"
    assert all(m.startswith("/model ") for m in matches)
    bare = [m[len("/model "):] for m in matches]
    assert sample in bare
    assert all(n.lower().startswith(prefix.lower()) or prefix.lower() in n.lower()
               for n in bare)


def test_model_command_completion_no_match_is_empty(tmp_path: Path) -> None:
    """A ``/model`` partial that matches nothing yields no suggestions."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/model zzz-no-such-model-xyz") == []


def test_model_command_takes_precedence_over_predictive(tmp_path: Path, kiss_db) -> None:
    """``/model`` completion wins over history-based ghost completion."""
    th._add_task("/model something from history", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/model gpt")
    assert all(m.startswith("/model ") for m in matches)
    # Not the predictive single-line ghost from history.
    assert matches != ["/model something from history"]


def test_bare_slash_model_still_completes_command(tmp_path: Path) -> None:
    """Typing ``/model`` (no space) completes the command itself."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/model") == ["/model "]


def test_completer_state_protocol(tmp_path: Path) -> None:
    """The readline ``complete(text, state)`` protocol returns then None."""
    completer = CliCompleter(str(tmp_path))
    first = completer.complete("/", 0)
    assert first is not None and first.startswith("/")
    seen = [first]
    state = 1
    while True:
        nxt = completer.complete("/", state)
        if nxt is None:
            break
        seen.append(nxt)
        state += 1
    assert len(seen) == len(SLASH_COMMANDS)


def test_predictive_completion_from_history(tmp_path: Path, kiss_db) -> None:
    """A typed prefix suggests the completion of a prior task (ghost)."""
    th._add_task("refactor the authentication module thoroughly", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("refactor the auth")
    assert matches == ["refactor the authentication module thoroughly"]


def test_predictive_completion_from_active_file(tmp_path: Path, kiss_db) -> None:
    """With no history match, identifiers from the active file complete."""
    active = tmp_path / "code.py"
    active.write_text("def calculate_total(items):\n    return sum(items)\n")
    completer = CliCompleter(str(tmp_path), active_file=str(active))
    matches = completer._build_matches("call calculate_t")
    assert matches == ["call calculate_total"]


def test_main_no_task_enters_repl(tmp_path: Path) -> None:
    """Running the CLI with no -t/-f enters the REPL and exits on EOF."""
    env = dict(os.environ, KISS_HOME=str(tmp_path / ".kisshome"))
    proc = subprocess.run(
        [sys.executable, "-m", "kiss.agents.sorcar.worktree_sorcar_agent",
         "-w", str(tmp_path)],
        input="/exit\n",
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stderr
    assert "interactive mode" in proc.stdout

def _read_line_over_pty(typed: str) -> tuple[str, str]:
    """Drive ``_read_line`` on a real PTY; return (before_enter, full).

    Forks a child that calls ``_read_line(_PROMPT)`` attached to a
    pseudo-terminal so the interactive (TTY) rendering path runs.  The
    bytes emitted *before* the submitting newline is sent are captured
    separately so a test can assert the box is already closed (top *and*
    bottom rule visible) while the user is still typing.

    Args:
        typed: The text to type before pressing Enter.

    Returns:
        A ``(before_enter, full)`` tuple of decoded terminal output.
    """
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

    child_code = (
        "from kiss.agents.sorcar.cli_repl import _PROMPT, _read_line\n"
        "import sys\n"
        "line = _read_line(_PROMPT)\n"
        "sys.stdout.write(f'RESULT[{line}]\\n')\n"
        "sys.stdout.flush()\n"
    )
    pid, fd = pty_spawn([sys.executable, "-c", child_code])

    def drain(seconds: float) -> str:
        out = b""
        deadline = time.time() + seconds
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        return out.decode("utf-8", "ignore")

    # Let the fresh interpreter import and draw the idle prompt box.
    time.sleep(2.0)
    os.write(fd, typed.encode())
    before = drain(1.0)
    os.write(fd, b"\n")
    after = drain(1.5)
    os.close(fd)
    os.waitpid(pid, 0)
    return before, before + after


def _gnu_readline_active() -> bool:
    """Return whether the live readline backend can cycle (GNU readline).

    Tab/Shift-Tab *cycling* uses ``menu-complete``, which exists only in
    GNU readline; the libedit/editline backend (stock macOS Python with
    no ``gnureadline`` wheel) cannot cycle, so the cycling tests skip.
    """
    from kiss.agents.sorcar import cli_repl

    rl = cli_repl.readline
    if rl is None:
        return False
    backend = getattr(rl, "backend", "") or ""
    doc = getattr(rl, "__doc__", "") or ""
    return backend != "editline" and "libedit" not in doc


def _complete_line_over_pty(
    project: str, chunks: list[str],
) -> tuple[str, list[str]]:
    """Type *chunks* into a real readline line and return the result.

    Forks a child interpreter attached to a pseudo-terminal that installs
    the real :class:`CliCompleter` via :func:`_setup_readline` (so the
    actual Tab/Shift-Tab key bindings run) and reads a single line with
    :func:`input`.  The *chunks* are sent one at a time with a short
    pause between them — each ``\\t`` (Tab) and ``\\x1b[Z`` (Shift-Tab)
    is its own chunk so readline processes one completion action per
    keystroke (sending them all at once races the redisplay).  Enter is
    pressed last and the submitted line is read from a ``RESULT[...]``
    marker.

    Args:
        project: Project directory seeding ``@``-mention completion.
        chunks: Keystroke chunks to type in order before Enter; embed
            ``"\\t"`` for Tab and ``"\\x1b[Z"`` for Shift-Tab.

    Returns:
        A ``(result, candidates)`` tuple: the line ``input`` returned
        after the keystrokes, and the child's own ``@alpha`` candidate
        ordering (file-scan order can differ between processes, so the
        child reports the order its readline session actually cycles).
    """
    from kiss.tests.agents.sorcar._pty_helper import pty_spawn

    child_code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from kiss.agents.sorcar.cli_repl import CliCompleter, _setup_readline\n"
        f"c = CliCompleter({project!r})\n"
        "for i, m in enumerate(c._build_matches('@alpha')):\n"
        "    sys.stdout.write(f'CAND{i}[{m}]\\n')\n"
        "_setup_readline(c, Path(os.devnull))\n"
        "line = input('> ')\n"
        "sys.stdout.write(f'RESULT[{line}]\\n')\n"
        "sys.stdout.flush()\n"
    )
    pid, fd = pty_spawn([sys.executable, "-c", child_code])

    def drain(seconds: float) -> str:
        out = b""
        deadline = time.time() + seconds
        while time.time() < deadline:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 8192)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
        return out.decode("utf-8", "ignore")

    time.sleep(2.0)  # let the fresh interpreter import and draw the prompt
    out = drain(0.5)  # capture the startup CAND lines + prompt
    for chunk in chunks:
        os.write(fd, chunk.encode())
        out += drain(0.4)  # one completion action settles per keystroke
    os.write(fd, b"\r")
    out += drain(1.5)
    os.close(fd)
    os.waitpid(pid, 0)
    result = re.search(r"RESULT\[(.*)\]", out)
    cands = re.findall(r"CAND\d+\[(.*)\]", out)
    return (result.group(1) if result else "", cands)


@pytest.mark.skipif(
    not hasattr(os, "fork") or not _gnu_readline_active(),
    reason="cycling needs a POSIX pty and GNU readline (menu-complete)",
)
def test_tab_cycles_forward_through_candidates(tmp_path: Path, kiss_db) -> None:
    """Pressing Tab repeatedly cycles forward through the candidates.

    With ``menu-complete`` bound to Tab, the first Tab inserts the best
    candidate and the second Tab replaces it with the next one — i.e. the
    candidates are *cycled* one at a time rather than merely listed.
    """
    for name in ("alpha_one.py", "alpha_two.py", "alpha_three.py"):
        (tmp_path / name).write_text("x = 1\n")

    one_tab, cands = _complete_line_over_pty(str(tmp_path), ["@alpha", "\t"])
    two_tab, c2 = _complete_line_over_pty(str(tmp_path), ["@alpha", "\t", "\t"])
    three_tab, c3 = _complete_line_over_pty(
        str(tmp_path), ["@alpha", "\t", "\t", "\t"],
    )
    # Ordering is deterministic, so each child cycles the same candidates.
    assert len(cands) >= 3 and cands == c2 == c3, (cands, c2, c3)
    # The first Tab inserts the best candidate; the candidates carry a
    # trailing space which ``input`` returns verbatim.
    assert one_tab == cands[0], (one_tab, cands)
    # Each completion landed on a real candidate from the menu.
    assert {one_tab, two_tab, three_tab} <= set(cands), (
        one_tab, two_tab, three_tab, cands,
    )
    # Three Tabs visited three *distinct* candidates — i.e. cycling, not
    # merely re-inserting one common-prefix completion every time.
    assert len({one_tab, two_tab, three_tab}) == 3, (
        one_tab, two_tab, three_tab,
    )


@pytest.mark.skipif(
    not hasattr(os, "fork") or not _gnu_readline_active(),
    reason="cycling needs a POSIX pty and GNU readline (menu-complete)",
)
def test_shift_tab_cycles_backward(tmp_path: Path, kiss_db) -> None:
    """Shift-Tab steps back through the menu after Tab moved forward.

    Shift-Tab is bound to ``menu-complete-backward``: after two forward
    Tabs, one Shift-Tab moves the menu selection back to a *different*
    candidate, proving the backward binding cycles the menu.
    """
    for name in ("alpha_one.py", "alpha_two.py", "alpha_three.py"):
        (tmp_path / name).write_text("x = 1\n")

    two_tab, _ = _complete_line_over_pty(
        str(tmp_path), ["@alpha", "\t", "\t"],
    )
    back, cands = _complete_line_over_pty(
        str(tmp_path), ["@alpha", "\t", "\t", "\x1b[Z"],
    )
    assert len(cands) >= 3, cands
    # Shift-Tab landed on a real candidate and moved the selection back
    # to a different one than the two-forward-Tab position.
    assert back in cands, (back, cands)
    assert back != two_tab, (back, two_tab)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX pty")
def test_idle_prompt_box_closed_while_typing() -> None:
    """The idle input box shows its bottom rule *before* Enter is pressed.

    Regression test: the lower horizontal bar of the task-input box must
    be drawn while the user is still composing the task (not only after
    submitting), so the box is never left visually open at the bottom.
    """
    before, full = _read_line_over_pty("hello task")
    # Both the top and the bottom rounded-border rows are emitted before
    # the user submits, so the box is already closed while typing.
    assert "╭" in before and "╮" in before, before
    assert "╰" in before and "╯" in before, before
    # The body line is fully framed: a left ``│`` (from the prompt) *and*
    # a right ``│`` vertical border are drawn, so the right edge is never
    # missing while typing.
    assert before.count("│") >= 2, before
    # The line still submits correctly once Enter is pressed.
    assert "RESULT[hello task]" in full, full

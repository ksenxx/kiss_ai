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
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_repl import (
    SLASH_COMMANDS,
    CliCompleter,
    _handle_slash,
    _record_mentions,
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


def test_at_mention_completion_inserts_pwd_path(tmp_path: Path, kiss_db) -> None:
    """``@`` mentions complete to ``PWD/<path>`` like the extension."""
    project = _write_project(tmp_path)
    completer = CliCompleter(str(project))
    matches = completer._build_matches("look at @alpha")
    assert matches, "expected at least one file suggestion"
    assert matches[0] == "look at PWD/alpha.py "
    assert all(m.startswith("look at PWD/") for m in matches)


def test_at_mention_completion_matches_nested_files(tmp_path: Path, kiss_db) -> None:
    """Nested files are reachable through the ``@`` picker."""
    project = _write_project(tmp_path)
    completer = CliCompleter(str(project))
    matches = completer._build_matches("edit @beta")
    assert any(m == "edit PWD/src/beta_module.py " for m in matches)


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


def test_record_mentions_persists_file_usage(tmp_path: Path, kiss_db) -> None:
    """Submitting a ``PWD/<path>`` line records file usage for ranking."""
    _record_mentions("please update PWD/src/app.py and PWD/main.py now")
    usage = th._load_file_usage()
    assert usage.get("src/app.py", 0) >= 1
    assert usage.get("main.py", 0) >= 1


def test_record_mentions_then_ranks_used_file_first(tmp_path: Path, kiss_db) -> None:
    """A recorded file is promoted to the 'frequent' group in suggestions."""
    project = _write_project(tmp_path)
    _record_mentions("touch PWD/src/beta_module.py")
    completer = CliCompleter(str(project))
    matches = completer._build_matches("see @beta")
    assert matches[0] == "see PWD/src/beta_module.py "


def test_handle_slash_clear_starts_new_chat(tmp_path: Path, kiss_db) -> None:
    """``/clear`` resets the chat id on a real ChatSorcarAgent."""
    agent = ChatSorcarAgent("t")
    agent.resume_chat_by_id("existing-chat")
    assert agent.chat_id == "existing-chat"
    assert _handle_slash(agent, "/clear", {}) is False
    assert agent.chat_id == ""


def test_handle_slash_model_switch(tmp_path: Path, kiss_db) -> None:
    """``/model <name>`` updates the agent and run kwargs."""
    agent = ChatSorcarAgent("t")
    kwargs: dict = {"model_name": "old-model"}
    _handle_slash(agent, "/model new-model", kwargs)
    assert kwargs["model_name"] == "new-model"
    assert agent.model_name == "new-model"


def test_handle_slash_exit_returns_true(tmp_path: Path, kiss_db) -> None:
    """``/exit`` and ``/quit`` request loop termination."""
    agent = ChatSorcarAgent("t")
    assert _handle_slash(agent, "/exit", {}) is True
    assert _handle_slash(agent, "/quit", {}) is True


def test_handle_slash_unknown(tmp_path: Path, kiss_db, capsys) -> None:
    """An unknown slash command is reported, not executed."""
    agent = ChatSorcarAgent("t")
    assert _handle_slash(agent, "/bogus", {}) is False
    assert "Unknown command" in capsys.readouterr().out


def _run_repl_subprocess(
    tmp_path: Path, stdin: str,
) -> subprocess.CompletedProcess:
    """Drive ``run_repl`` in a subprocess feeding *stdin*, no model calls."""
    script = (
        "from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent\n"
        "from kiss.agents.sorcar.cli_repl import run_repl\n"
        "agent = ChatSorcarAgent('test')\n"
        "run_repl(agent, {'work_dir': '.', 'model_name': 'demo-model'})\n"
    )
    env = dict(os.environ, KISS_HOME=str(tmp_path / ".kisshome"))
    return subprocess.run(
        [sys.executable, "-c", script],
        input=stdin,
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )


def test_repl_welcome_and_exit(tmp_path: Path) -> None:
    """The REPL prints the welcome banner and exits on /exit."""
    proc = _run_repl_subprocess(tmp_path, "/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "KISS Sorcar" in proc.stdout
    assert "interactive mode" in proc.stdout


def test_repl_help_then_eof(tmp_path: Path) -> None:
    """``/help`` lists commands; EOF (no /exit) ends the session."""
    proc = _run_repl_subprocess(tmp_path, "/help\n")
    assert proc.returncode == 0, proc.stderr
    assert "Commands:" in proc.stdout
    assert "/clear" in proc.stdout
    assert "Goodbye" in proc.stdout


def test_repl_cost_command(tmp_path: Path) -> None:
    """``/cost`` prints usage without invoking the model."""
    proc = _run_repl_subprocess(tmp_path, "/cost\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "Cost:" in proc.stdout
    assert "Total tokens:" in proc.stdout


def test_repl_frames_input_with_single_panel(tmp_path: Path) -> None:
    """The idle input dialog is drawn inside the shared rounded panel.

    The idle prompt and the steering box now render the *same* panel, so
    the idle dialog shows the rounded border glyphs and the shared idle
    title instead of the old plain horizontal rules.
    """
    proc = _run_repl_subprocess(tmp_path, "/cost\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    # Rounded-border glyphs (the same the steering box uses) frame input.
    assert "╭" in proc.stdout and "╮" in proc.stdout
    assert "╰" in proc.stdout and "╯" in proc.stdout
    # The shared idle title appears in the panel's top border.
    assert "sorcar · type a task" in proc.stdout
    # Each of the two prompts (/cost then /exit) draws a top and bottom
    # border, so at least 3 runs of consecutive box-drawing dashes show.
    rules = re.findall(r"\u2500{10,}", proc.stdout)
    assert len(rules) >= 3


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
    import pty

    child_code = (
        "from kiss.agents.sorcar.cli_repl import _PROMPT, _read_line\n"
        "import sys\n"
        "line = _read_line(_PROMPT)\n"
        "sys.stdout.write(f'RESULT[{line}]\\n')\n"
        "sys.stdout.flush()\n"
    )
    pid, fd = pty.fork()
    if pid == 0:  # child: a fresh interpreter attached to the PTY slave
        os.execvp(sys.executable, [sys.executable, "-c", child_code])
        os._exit(0)  # pragma: no cover - exec never returns

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

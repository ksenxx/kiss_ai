# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Characterization (lockdown) tests for the sorcar CLI helpers and tools.

Pins the CURRENT externally observable behavior of:

* ``useful_tools._expand_pwd_prefix`` — ``PWD``/``PWD/x`` expansion.
* ``UsefulTools.Bash`` — error formatting, truncation, timeout message,
  streaming callback, cwd / ``KISS_WORKDIR``, stop_event kill.
* ``UsefulTools.Read`` — PWD/ expansion, directory listing, max_lines,
  "Did you mean" suggestion, empty-file sentinel.
* ``cli_repl._history_path`` — per-work-dir history file location.
* ``cli_helpers._print_run_stats`` / ``cli_repl._print_usage`` /
  ``cli_helpers._print_result`` — exact console line shapes.
* ``cli_panel`` — panel border widths, body marker, tail clipping.
* ``WebUseTool.close`` — idempotent without a browser launch.
* ``ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes`` configured by
  importing the vscode task_runner / web_server modules.

These tests guard the planned simplifications (tmp/findings-3.md, items
A1-A8, B1-B6, C1-C6, D1) so refactors cannot change behavior unnoticed.
NOTE: ``_history_path`` determinism is asserted only *in-process* here,
matching today's ``hash()``-based behavior (findings-3 item D1).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import kiss.agents.sorcar.persistence as th
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.cli_helpers import _print_result, _print_run_stats
from kiss.agents.sorcar.cli_panel import (
    IDLE_TITLE,
    PLACEHOLDER,
    PROMPT_MARKER,
    body_cursor_col,
    clip_buf,
    panel_body,
    panel_bottom,
    panel_top,
)
from kiss.agents.sorcar.cli_repl import _history_path, _print_usage
from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.agents.sorcar.useful_tools import UsefulTools, _expand_pwd_prefix
from kiss.agents.sorcar.web_use_tool import WebUseTool

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _visible(text: str) -> str:
    """Return *text* with ANSI escape sequences stripped."""
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# (1) _expand_pwd_prefix
# ---------------------------------------------------------------------------


def test_expand_pwd_bare_pwd_returns_work_dir(tmp_path):
    assert _expand_pwd_prefix("PWD", str(tmp_path)) == str(tmp_path)


def test_expand_pwd_prefix_joins_relative_suffix(tmp_path):
    expanded = _expand_pwd_prefix("PWD/x/y", str(tmp_path))
    assert expanded == os.path.join(str(tmp_path), "x/y")


def test_expand_pwd_absolute_path_passthrough(tmp_path):
    assert _expand_pwd_prefix("/abs/other.txt", str(tmp_path)) == "/abs/other.txt"
    assert _expand_pwd_prefix("rel/other.txt", str(tmp_path)) == "rel/other.txt"
    # "PWDx" is not a PWD prefix and must pass through untouched.
    assert _expand_pwd_prefix("PWDx/file", str(tmp_path)) == "PWDx/file"


def test_expand_pwd_none_work_dir_falls_back_to_cwd():
    assert _expand_pwd_prefix("PWD", None) == os.getcwd()
    assert _expand_pwd_prefix("PWD/a.txt", None) == os.path.join(os.getcwd(), "a.txt")


# ---------------------------------------------------------------------------
# (2) UsefulTools.Bash
# ---------------------------------------------------------------------------


def test_bash_nonzero_exit_error_format():
    result = UsefulTools().Bash("echo boom; exit 3", "fail with output")
    assert result == "Error (exit code 3):\nboom\n"


def test_bash_nonzero_exit_no_output_error_format():
    result = UsefulTools().Bash("exit 7", "fail without output")
    assert result == "Error (exit code 7):"


def test_bash_long_output_truncated_with_notice():
    result = UsefulTools().Bash(
        "printf 'x%.0s' $(seq 1 2000)", "long output", max_output_chars=100
    )
    assert len(result) <= 100
    assert "... [truncated " in result
    assert " chars] ..." in result
    assert result.startswith("x")
    assert result.endswith("x")


def test_bash_timeout_message():
    start = time.time()
    result = UsefulTools().Bash("sleep 5", "sleep", timeout_seconds=1)
    assert result == "Error: Command execution timeout"
    assert time.time() - start < 10


def test_bash_streaming_callback_receives_lines():
    received: list[str] = []
    tools = UsefulTools(stream_callback=received.append)
    result = tools.Bash("printf 'l1\\nl2\\nl3\\n'", "stream three lines")
    assert received == ["l1\n", "l2\n", "l3\n"]
    assert result == "l1\nl2\nl3\n"


def test_bash_cwd_and_kiss_workdir_reflect_work_dir(tmp_path):
    tools = UsefulTools(work_dir=str(tmp_path))
    pwd_out = tools.Bash("pwd", "print cwd")
    assert Path(pwd_out.strip()).resolve() == tmp_path.resolve()
    env_out = tools.Bash("echo $KISS_WORKDIR", "print KISS_WORKDIR")
    assert env_out.strip() == str(tmp_path)


def test_bash_stop_event_preset_kills_sleeping_child_quickly():
    stop = threading.Event()
    stop.set()
    tools = UsefulTools(stop_event=stop)
    start = time.time()
    result = tools.Bash("sleep 30", "sleep long", timeout_seconds=60)
    assert time.time() - start < 10
    assert result.startswith("Error (exit code ")


def test_bash_stop_event_set_during_run_kills_sleeping_child():
    stop = threading.Event()
    threading.Timer(0.4, stop.set).start()
    tools = UsefulTools(stop_event=stop)
    start = time.time()
    result = tools.Bash("sleep 30", "sleep long", timeout_seconds=60)
    assert time.time() - start < 10
    assert result.startswith("Error (exit code ")


# ---------------------------------------------------------------------------
# (3) UsefulTools.Read
# ---------------------------------------------------------------------------


def test_read_pwd_prefix_expansion_reads_file(tmp_path):
    (tmp_path / "hello.txt").write_text("hi there\n")
    tools = UsefulTools(work_dir=str(tmp_path))
    assert tools.Read("PWD/hello.txt") == "hi there\n"


def test_read_directory_returns_listing(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    result = UsefulTools().Read(str(tmp_path))
    assert result.startswith(f"Error: {tmp_path} is a directory, not a file.")
    assert "Directory contents:" in result
    assert "a.txt" in result
    assert "sub/" in result


def test_read_max_lines_truncates(tmp_path):
    f = tmp_path / "five.txt"
    f.write_text("l1\nl2\nl3\nl4\nl5\n")
    result = UsefulTools().Read(str(f), max_lines=2)
    assert result == "l1\nl2\n\n[truncated: 3 more lines]"


def test_read_missing_file_suggests_near_miss_sibling(tmp_path):
    (tmp_path / "config.py").write_text("x = 1\n")
    missing = tmp_path / "confg.py"
    result = UsefulTools().Read(str(missing))
    assert result.startswith(f"Error: File not found: {missing}.")
    assert "Did you mean:" in result
    assert "config.py" in result


def test_read_empty_file_sentinel(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert UsefulTools().Read(str(f)) == "(file is empty)"


# ---------------------------------------------------------------------------
# (4) cli_repl._history_path
# ---------------------------------------------------------------------------


def test_history_path_location_and_in_process_determinism(tmp_path):
    saved_env = os.environ.get("KISS_HOME")
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR)
    kiss_home = tmp_path / "kiss-home"
    kiss_home.mkdir()
    os.environ["KISS_HOME"] = str(kiss_home)
    th._KISS_DIR = kiss_home
    th._DB_PATH = kiss_home / "sorcar.db"
    th._db_conn = None
    try:
        work_a = tmp_path / "proj-a"
        work_b = tmp_path / "proj-b"
        work_a.mkdir()
        work_b.mkdir()
        p1 = _history_path(str(work_a))
        p2 = _history_path(str(work_a))
        p3 = _history_path(str(work_b))
        assert p1 == p2
        assert p1 != p3
        assert p1.parent == kiss_home / "cli_history"
        assert p1.parent.is_dir()
    finally:
        if saved_env is None:
            os.environ.pop("KISS_HOME", None)
        else:
            os.environ["KISS_HOME"] = saved_env
        th._DB_PATH, th._db_conn, th._KISS_DIR = saved


# ---------------------------------------------------------------------------
# (5) Stats / result printing parity
# ---------------------------------------------------------------------------


def test_print_run_stats_exact_lines(capsys):
    agent = SorcarAgent("lockdown-stats")
    agent.budget_used = 1.23456
    agent.total_tokens_used = 42
    _print_run_stats(agent, 2.5)
    out = capsys.readouterr().out
    assert out == "\nTime: 2.5s\nCost: $1.2346\nTotal tokens: 42\n"


def test_print_usage_exact_lines_fresh_chat_agent(capsys):
    agent = ChatSorcarAgent("lockdown-usage")
    _print_usage(agent)
    out = capsys.readouterr().out
    assert out == "\nChat ID: (new)\nCost: $0.0000\nTotal tokens: 0\n\n"


def test_print_usage_with_populated_agent(capsys):
    agent = ChatSorcarAgent("lockdown-usage2")
    agent._chat_id = "deadbeef"
    agent.budget_used = 0.5
    agent.total_tokens_used = 999
    _print_usage(agent)
    out = capsys.readouterr().out
    assert out == "\nChat ID: deadbeef\nCost: $0.5000\nTotal tokens: 999\n\n"


def test_print_result_prints_summary_only(capsys):
    _print_result("success: true\nsummary: All done here\n")
    out = capsys.readouterr().out
    assert out == "All done here\n"
    assert "success" not in out


def test_print_result_non_yaml_falls_back_verbatim(capsys):
    _print_result("plain text result")
    assert capsys.readouterr().out == "plain text result\n"


# ---------------------------------------------------------------------------
# (6) cli_panel invariants
# ---------------------------------------------------------------------------


def test_panel_top_and_bottom_visible_width_equals_cols():
    for cols in (10, 40, 80, 123):
        top = _visible(panel_top(IDLE_TITLE, cols))
        assert len(top) == cols
        assert top.startswith("╭")
        assert top.endswith("╮")
        bottom = _visible(panel_bottom("", cols))
        assert len(bottom) == cols
        assert bottom.startswith("╰")
        assert bottom.endswith("╯")
        bottom_status = _visible(panel_bottom(" esc to abort ", cols))
        assert len(bottom_status) == cols


def test_panel_body_starts_with_prompt_marker_and_pads():
    body, is_placeholder = panel_body("hello", 40)
    assert body.startswith(PROMPT_MARKER + "hello")
    assert len(body) == 40 - 4
    assert is_placeholder is False


def test_panel_body_empty_buffer_shows_placeholder():
    body, is_placeholder = panel_body("", 40)
    assert body.startswith(PROMPT_MARKER)
    assert PLACEHOLDER[: 40 - 4 - len(PROMPT_MARKER)] in body
    assert is_placeholder is True


def test_panel_body_tail_clips_long_buffer():
    cols = 40
    buf = "abcdefghij" * 10  # 100 chars, far wider than the panel
    avail = (cols - 4) - len(PROMPT_MARKER)
    body, is_placeholder = panel_body(buf, cols)
    assert body == PROMPT_MARKER + buf[-avail:]
    assert clip_buf(buf, cols) == buf[-avail:]
    assert is_placeholder is False
    # Caret parks right after the visible text: col 3 + marker + clipped.
    assert body_cursor_col(buf, cols) == 3 + len(PROMPT_MARKER) + avail
    assert body_cursor_col("", cols) == 3 + len(PROMPT_MARKER)


# ---------------------------------------------------------------------------
# (7) WebUseTool.close idempotence without launching a browser
# ---------------------------------------------------------------------------


def test_web_use_tool_close_without_launch_is_idempotent(tmp_path):
    tool = WebUseTool(user_data_dir=str(tmp_path / "profile"))
    assert tool.close() == "Browser closed."
    assert tool.close() == "Browser closed."
    assert tool._playwright is None
    assert tool._browser is None
    assert tool._context is None
    assert tool._page is None


# ---------------------------------------------------------------------------
# (8) ctypes PyThreadState_SetAsyncExc argtypes configured at import
# ---------------------------------------------------------------------------

_CTYPES_CHECK = (
    "import ctypes\n"
    "import {module}\n"
    "f = ctypes.pythonapi.PyThreadState_SetAsyncExc\n"
    "assert f.argtypes is not None, 'argtypes not configured'\n"
    "assert list(f.argtypes) == [ctypes.c_ulong, ctypes.py_object], f.argtypes\n"
    "print('OK')\n"
)


def _run_ctypes_check(module: str, tmp_path: Path) -> None:
    env = dict(os.environ)
    env["KISS_HOME"] = str(tmp_path / "kiss-home")
    proc = subprocess.run(
        [sys.executable, "-c", _CTYPES_CHECK.format(module=module)],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"


def test_import_task_runner_configures_setasyncexc_argtypes(tmp_path):
    _run_ctypes_check("kiss.agents.vscode.task_runner", tmp_path)


def test_import_web_server_configures_setasyncexc_argtypes(tmp_path):
    _run_ctypes_check("kiss.agents.vscode.web_server", tmp_path)

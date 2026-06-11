# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for the sorcar CLI custom commands (.md files).

These exercise the real behaviour end to end: command files are written
to real user (``KISS_HOME``) and project directories, discovery and
template expansion run against the real filesystem and real shell, and
the REPL is driven through a real subprocess reading piped stdin.  No
model calls are made — the tests only use commands that are listed or
completed, never executed as a task.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from kiss.agents.sorcar.cli_repl import CliCompleter
from kiss.agents.sorcar.custom_commands import (
    discover_commands,
    expand_command,
    format_command_listing,
)


@pytest.fixture
def kiss_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point KISS_HOME (and CLAUDE_CONFIG_DIR) at isolated directories.

    CLAUDE_CONFIG_DIR is redirected so discovery never picks up the
    developer's real ``~/.claude/commands`` files during tests.
    """
    home = tmp_path / ".kisshome"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    return home


@pytest.fixture
def claude_home(tmp_path: Path, kiss_home: Path) -> Path:
    """Return the isolated Claude Code config directory set by kiss_home."""
    home = tmp_path / ".claudehome"
    home.mkdir(exist_ok=True)
    return home


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def test_discovers_user_and_project_commands(kiss_home: Path, tmp_path: Path) -> None:
    """User and project .md files both become commands with right source."""
    _write(kiss_home / "commands" / "greet.md", "Say hello.")
    project = tmp_path / "proj"
    _write(project / ".kiss" / "commands" / "test.md", "Run the tests.")
    commands = discover_commands(str(project))
    assert commands["greet"].source == "user"
    assert commands["greet"].template == "Say hello."
    assert commands["test"].source == "project"
    assert commands["test"].template == "Run the tests."


def test_project_command_overrides_user_command(kiss_home: Path, tmp_path: Path) -> None:
    """A project command wins over a user command with the same name."""
    _write(kiss_home / "commands" / "deploy.md", "user version")
    project = tmp_path / "proj"
    _write(project / ".kiss" / "commands" / "deploy.md", "project version")
    commands = discover_commands(str(project))
    assert commands["deploy"].source == "project"
    assert commands["deploy"].template == "project version"


def test_subdirectories_namespace_with_colon(kiss_home: Path, tmp_path: Path) -> None:
    """commands/git/commit.md becomes the command name git:commit."""
    _write(kiss_home / "commands" / "git" / "commit.md", "Write a commit message.")
    commands = discover_commands(str(tmp_path))
    assert "git:commit" in commands


def test_frontmatter_description_and_hint(kiss_home: Path, tmp_path: Path) -> None:
    """YAML frontmatter sets description/argument-hint; body is template."""
    _write(
        kiss_home / "commands" / "fix.md",
        "---\ndescription: Fix an issue\nargument-hint: '[issue-number]'\n---\n"
        "Fix issue $1.",
    )
    cmd = discover_commands(str(tmp_path))["fix"]
    assert cmd.description == "Fix an issue"
    assert cmd.argument_hint == "[issue-number]"
    assert cmd.template == "Fix issue $1."


def test_empty_and_unparseable_files_are_skipped(kiss_home: Path, tmp_path: Path) -> None:
    """Files with an empty body do not become commands."""
    _write(kiss_home / "commands" / "empty.md", "")
    _write(kiss_home / "commands" / "only-frontmatter.md", "---\ndescription: x\n---\n")
    commands = discover_commands(str(tmp_path))
    assert "empty" not in commands
    assert "only-frontmatter" not in commands


def test_no_command_dirs_yields_empty(kiss_home: Path, tmp_path: Path) -> None:
    """Discovery returns an empty mapping when no commands exist."""
    assert discover_commands(str(tmp_path)) == {}


def test_discovers_claude_user_and_project_commands(
    claude_home: Path, tmp_path: Path,
) -> None:
    """Claude Code .claude/commands files become commands too."""
    _write(claude_home / "commands" / "review.md", "Review the diff.")
    project = tmp_path / "proj"
    _write(project / ".claude" / "commands" / "ship.md", "Ship it.")
    commands = discover_commands(str(project))
    assert commands["review"].source == "claude-user"
    assert commands["review"].template == "Review the diff."
    assert commands["ship"].source == "claude-project"
    assert commands["ship"].template == "Ship it."


def test_kiss_command_overrides_claude_command_same_level(
    claude_home: Path, kiss_home: Path, tmp_path: Path,
) -> None:
    """At the same level a native .kiss command wins over a .claude one."""
    _write(claude_home / "commands" / "lint.md", "claude user version")
    _write(kiss_home / "commands" / "lint.md", "kiss user version")
    project = tmp_path / "proj"
    _write(project / ".claude" / "commands" / "fmt.md", "claude project version")
    _write(project / ".kiss" / "commands" / "fmt.md", "kiss project version")
    commands = discover_commands(str(project))
    assert commands["lint"].source == "user"
    assert commands["lint"].template == "kiss user version"
    assert commands["fmt"].source == "project"
    assert commands["fmt"].template == "kiss project version"


def test_claude_project_overrides_kiss_user(
    claude_home: Path, kiss_home: Path, tmp_path: Path,
) -> None:
    """Any project command wins over any user command with the same name."""
    _write(kiss_home / "commands" / "deploy.md", "kiss user version")
    project = tmp_path / "proj"
    _write(project / ".claude" / "commands" / "deploy.md", "claude project version")
    commands = discover_commands(str(project))
    assert commands["deploy"].source == "claude-project"
    assert commands["deploy"].template == "claude project version"


def test_claude_commands_namespace_and_frontmatter(
    claude_home: Path, tmp_path: Path,
) -> None:
    """Claude commands get :-namespacing and frontmatter like kiss ones."""
    _write(
        claude_home / "commands" / "git" / "pr.md",
        "---\ndescription: Open a PR\nargument-hint: '[title]'\n---\nOpen a PR: $1.",
    )
    cmd = discover_commands(str(tmp_path))["git:pr"]
    assert cmd.source == "claude-user"
    assert cmd.description == "Open a PR"
    assert cmd.argument_hint == "[title]"


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def _command(kiss_home: Path, work_dir: Path, body: str):
    _write(kiss_home / "commands" / "c.md", body)
    return discover_commands(str(work_dir))["c"]


def test_arguments_placeholder(kiss_home: Path, tmp_path: Path) -> None:
    """$ARGUMENTS is replaced with the full argument string."""
    cmd = _command(kiss_home, tmp_path, "Create a component named $ARGUMENTS now.")
    out = expand_command(cmd, "Button extra", str(tmp_path))
    assert out == "Create a component named Button extra now."


def test_positional_placeholders_with_quoting(kiss_home: Path, tmp_path: Path) -> None:
    """$1/$2 use shell-style quoting so quoted args stay one argument."""
    cmd = _command(kiss_home, tmp_path, "Create $1 in $2.")
    out = expand_command(cmd, '"hello world" src', str(tmp_path))
    assert out == "Create hello world in src."


def test_missing_positional_becomes_empty(kiss_home: Path, tmp_path: Path) -> None:
    """A positional placeholder with no matching argument expands empty."""
    cmd = _command(kiss_home, tmp_path, "Use $1 and $2.")
    out = expand_command(cmd, "only", str(tmp_path))
    assert out == "Use only and ."


def test_args_appended_when_no_placeholder(kiss_home: Path, tmp_path: Path) -> None:
    """Without placeholders, provided args are appended after two newlines."""
    cmd = _command(kiss_home, tmp_path, "Review the code.")
    out = expand_command(cmd, "focus on safety", str(tmp_path))
    assert out == "Review the code.\n\nfocus on safety"


def test_no_args_no_placeholder_unchanged(kiss_home: Path, tmp_path: Path) -> None:
    """Without args the template is sent exactly as written."""
    cmd = _command(kiss_home, tmp_path, "Review the code.")
    assert expand_command(cmd, "", str(tmp_path)) == "Review the code."


def test_file_injection(kiss_home: Path, tmp_path: Path) -> None:
    """@{path} is replaced with the file's contents (relative to work dir)."""
    (tmp_path / "notes.txt").write_text("remember the milk")
    cmd = _command(kiss_home, tmp_path, "Context: @{notes.txt} done.")
    out = expand_command(cmd, "", str(tmp_path))
    assert out == "Context: remember the milk done."


def test_file_injection_missing_file(kiss_home: Path, tmp_path: Path) -> None:
    """A missing @{path} file injects a readable error marker."""
    cmd = _command(kiss_home, tmp_path, "Context: @{nope.txt}")
    out = expand_command(cmd, "", str(tmp_path))
    assert "[could not read file: nope.txt]" in out


def test_shell_injection(kiss_home: Path, tmp_path: Path) -> None:
    """!`command` runs in the work dir and injects its output."""
    cmd = _command(kiss_home, tmp_path, "Files: !`echo hi there` end.")
    out = expand_command(cmd, "", str(tmp_path))
    assert out == "Files: hi there end."


def test_shell_injection_runs_in_work_dir(kiss_home: Path, tmp_path: Path) -> None:
    """Shell commands execute with the project as working directory."""
    (tmp_path / "marker.txt").write_text("x")
    cmd = _command(kiss_home, tmp_path, "!`ls`")
    out = expand_command(cmd, "", str(tmp_path))
    assert "marker.txt" in out


def test_shell_injection_failure_reports_exit_code(
    kiss_home: Path, tmp_path: Path,
) -> None:
    """A failing shell command injects stderr plus an exit-status note."""
    cmd = _command(kiss_home, tmp_path, "!`false`")
    out = expand_command(cmd, "", str(tmp_path))
    assert "[Shell command exited with code 1]" in out


def test_shell_escaped_args_in_positional(kiss_home: Path, tmp_path: Path) -> None:
    """Unbalanced quotes fall back to whitespace splitting, not a crash."""
    cmd = _command(kiss_home, tmp_path, "Use $1.")
    out = expand_command(cmd, 'it"s broken', str(tmp_path))
    assert out == 'Use it"s.'


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------

def test_format_command_listing(kiss_home: Path, tmp_path: Path) -> None:
    """The listing shows /name, hint, source, and description aligned."""
    _write(
        kiss_home / "commands" / "fix.md",
        "---\ndescription: Fix an issue\nargument-hint: '[n]'\n---\nFix $1.",
    )
    project = tmp_path / "proj"
    _write(project / ".kiss" / "commands" / "test.md", "Run tests.")
    listing = format_command_listing(discover_commands(str(project)))
    assert "/fix [n]" in listing
    assert "(user) Fix an issue" in listing
    assert "/test" in listing
    assert "(project)" in listing


def test_format_command_listing_empty_hint(kiss_home: Path, tmp_path: Path) -> None:
    """An empty mapping yields a hint about where to create commands."""
    listing = format_command_listing({})
    assert "No custom commands found" in listing
    assert ".kiss/commands" in listing


# ---------------------------------------------------------------------------
# Completion
# ---------------------------------------------------------------------------

def test_completer_includes_custom_commands(kiss_home: Path, tmp_path: Path) -> None:
    """Typing /<prefix> Tab offers matching custom command names."""
    _write(kiss_home / "commands" / "greet.md", "Say hello.")
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/gre")
    assert "/greet " in matches


def test_completer_builtins_before_custom(kiss_home: Path, tmp_path: Path) -> None:
    """Built-in commands rank before custom commands in completion."""
    _write(kiss_home / "commands" / "helper.md", "Assist.")
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/hel")
    assert matches.index("/help ") < matches.index("/helper ")


# ---------------------------------------------------------------------------
# REPL subprocess (no model calls)
# ---------------------------------------------------------------------------

def _run_repl_subprocess(
    tmp_path: Path, kiss_home: Path, stdin: str,
) -> subprocess.CompletedProcess:
    """Drive ``run_repl`` in a subprocess feeding *stdin*, no model calls."""
    script = (
        "from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent\n"
        "from kiss.agents.sorcar.cli_repl import run_repl\n"
        "agent = ChatSorcarAgent('test')\n"
        "run_repl(agent, {'work_dir': '.', 'model_name': 'demo-model'})\n"
    )
    env = dict(os.environ, KISS_HOME=str(kiss_home))
    return subprocess.run(
        [sys.executable, "-c", script],
        input=stdin,
        text=True,
        capture_output=True,
        cwd=str(tmp_path),
        env=env,
        timeout=120,
    )


def test_repl_commands_lists_custom_commands(kiss_home: Path, tmp_path: Path) -> None:
    """``/commands`` prints discovered custom commands in the REPL."""
    _write(
        kiss_home / "commands" / "greet.md",
        "---\ndescription: Greet the user\n---\nSay hello.",
    )
    _write(tmp_path / ".kiss" / "commands" / "build.md", "Build the project.")
    proc = _run_repl_subprocess(tmp_path, kiss_home, "/commands\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "/greet" in proc.stdout
    assert "Greet the user" in proc.stdout
    assert "/build" in proc.stdout
    assert "(project)" in proc.stdout


def test_repl_commands_lists_claude_commands(
    claude_home: Path, kiss_home: Path, tmp_path: Path,
) -> None:
    """``/commands`` includes commands from .claude/commands directories."""
    _write(
        claude_home / "commands" / "review.md",
        "---\ndescription: Review the diff\n---\nReview it.",
    )
    _write(tmp_path / ".claude" / "commands" / "ship.md", "Ship it.")
    proc = _run_repl_subprocess(tmp_path, kiss_home, "/commands\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "/review" in proc.stdout
    assert "(claude-user) Review the diff" in proc.stdout
    assert "/ship" in proc.stdout
    assert "(claude-project)" in proc.stdout


def test_repl_commands_empty_hint(kiss_home: Path, tmp_path: Path) -> None:
    """``/commands`` with no commands explains how to create them."""
    proc = _run_repl_subprocess(tmp_path, kiss_home, "/commands\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "No custom commands found" in proc.stdout


def test_repl_help_lists_custom_commands(kiss_home: Path, tmp_path: Path) -> None:
    """``/help`` shows the custom-commands section when commands exist."""
    _write(kiss_home / "commands" / "greet.md", "Say hello.")
    proc = _run_repl_subprocess(tmp_path, kiss_home, "/help\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "Custom commands:" in proc.stdout
    assert "/greet" in proc.stdout


def test_repl_unknown_command_still_reported(kiss_home: Path, tmp_path: Path) -> None:
    """A slash word matching no built-in or custom command is reported."""
    proc = _run_repl_subprocess(tmp_path, kiss_home, "/nosuchcmd\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "Unknown command" in proc.stdout

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the ``/autocommit`` slash command in the Sorcar CLI REPL.

The command mirrors the VS Code extension's ``Auto-commit`` action:
it stages all uncommitted changes, asks the LLM to produce a commit
message from the staged diff, and commits.  These integration tests
drive the real :func:`kiss.agents.sorcar.cli_repl._handle_slash`
dispatcher against on-disk git repositories and assert on the
created commit and the printed feedback — the LLM call inside the
message generator is shorted out by patching
:func:`kiss.agents.sorcar.sorcar_agent._generate_commit_message`
so the tests never need a real model.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any
from unittest.mock import patch

import kiss.agents.sorcar.sorcar_agent as sorcar_agent_module
from kiss.agents.sorcar.cli_repl import SLASH_COMMANDS, _handle_slash


class _FakeAgent:
    """Minimal stand-in for ``SorcarAgent`` for ``/autocommit`` tests."""

    def __init__(self, last_user_prompt: str = "") -> None:
        self._last_user_prompt = last_user_prompt
        self.model_name = "test-model"


def _make_repo(path: Path) -> Path:
    """Create a git repo with one initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "main", str(path)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@t.com"],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "T"],
        check=True, capture_output=True,
    )
    (path / "README.md").write_text("# Test\n")
    subprocess.run(
        ["git", "-C", str(path), "add", "."],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "commit", "-m", "initial"],
        check=True, capture_output=True,
    )
    return path


def _head_subject(repo: Path) -> str:
    """Return HEAD's commit subject line."""
    result = subprocess.run(
        ["git", "-C", str(repo), "log", "-1", "--pretty=%s"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _commit_count(repo: Path) -> int:
    """Return total number of commits on HEAD."""
    result = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def test_autocommit_listed_in_slash_commands() -> None:
    """``/autocommit`` must appear in the documented slash-command list."""
    assert "/autocommit" in SLASH_COMMANDS
    help_text = SLASH_COMMANDS["/autocommit"].lower()
    assert "commit" in help_text


def test_autocommit_commits_dirty_tree_with_llm_message() -> None:
    """A dirty working tree is staged, committed with the LLM message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        baseline = _commit_count(repo)
        # Introduce uncommitted change.
        (repo / "feature.txt").write_text("hello\n")

        agent = _FakeAgent(last_user_prompt="add a feature file")
        captured: dict[str, Any] = {}

        def fake_generate(
            commit_dir: Path, user_prompt: str | None = None,
        ) -> str:
            captured["commit_dir"] = commit_dir
            captured["user_prompt"] = user_prompt
            return "feat: add feature.txt"

        buf = io.StringIO()
        with patch.object(
            sorcar_agent_module, "_generate_commit_message", fake_generate,
        ), redirect_stdout(buf):
            stop = _handle_slash(
                agent,  # type: ignore[arg-type]
                "/autocommit",
                {"work_dir": str(repo)},
            )

        assert stop is False
        assert _commit_count(repo) == baseline + 1
        assert _head_subject(repo) == "feat: add feature.txt"
        # The LLM helper saw the configured working dir and prompt.
        # ``_handle_autocommit`` resolves the path (e.g. /var → /private/var
        # on macOS), so compare via :meth:`Path.resolve`.
        assert captured["commit_dir"] == repo.resolve()
        assert captured["user_prompt"] == "add a feature file"
        out = buf.getvalue()
        assert "✓ Committed: feat: add feature.txt" in out


def test_autocommit_falls_back_when_llm_raises() -> None:
    """LLM failure falls back to the generic kiss commit message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        (repo / "x.txt").write_text("x\n")
        agent = _FakeAgent(last_user_prompt="do X")

        def boom(
            commit_dir: Path, user_prompt: str | None = None,
        ) -> str:
            raise RuntimeError("llm down")

        buf = io.StringIO()
        with patch.object(
            sorcar_agent_module, "_generate_commit_message", boom,
        ), redirect_stdout(buf):
            _handle_slash(
                agent,  # type: ignore[arg-type]
                "/autocommit",
                {"work_dir": str(repo)},
            )

        subject = _head_subject(repo)
        assert subject == "kiss: auto-commit agent changes"
        # User prompt is appended in the body via _append_user_prompt.
        result = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--pretty=%B"],
            capture_output=True, text=True, check=True,
        )
        assert "User prompt:" in result.stdout
        assert "do X" in result.stdout


def test_autocommit_on_clean_tree_makes_no_commit() -> None:
    """A clean repo prints ``Nothing to commit.`` and creates no commit."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = _make_repo(Path(tmp) / "repo")
        baseline = _commit_count(repo)
        agent = _FakeAgent()

        buf = io.StringIO()
        with redirect_stdout(buf):
            _handle_slash(
                agent,  # type: ignore[arg-type]
                "/autocommit",
                {"work_dir": str(repo)},
            )

        assert _commit_count(repo) == baseline
        assert "Nothing to commit." in buf.getvalue()


def test_autocommit_outside_git_repo_prints_message() -> None:
    """Outside a git repo, the command reports it and creates nothing."""
    with tempfile.TemporaryDirectory() as tmp:
        non_repo = Path(tmp) / "not_a_repo"
        non_repo.mkdir()
        agent = _FakeAgent()

        buf = io.StringIO()
        with redirect_stdout(buf):
            _handle_slash(
                agent,  # type: ignore[arg-type]
                "/autocommit",
                {"work_dir": str(non_repo)},
            )

        assert "Not a git repository." in buf.getvalue()

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""A task that is only a filesystem path tells the agent to open it.

Submitting ``~/Downloads/ROUTING.md`` as the whole task used to make
the agent inspect the file and ask what to do with it.
:mod:`kiss.agents.sorcar.bare_path_task` appends an "open it" directive
to such a prompt, and ``ChatSorcarAgent.run`` applies it to the prompt
the model sees while the history row keeps the raw path.

The ``ChatSorcarAgent`` tests run the real agent against the local
stand-in model server with a real SQLite history under an isolated
``KISS_HOME`` and read the prompt out of the request the model
actually received.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.bare_path_task import bare_path, with_open_directive
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    history_rows,
    request_text,
)

DIRECTIVE = "The task is nothing but the path of an existing"


class TestBarePath:
    """Which prompts count as a bare path, and what they resolve to."""

    def test_absolute_file(self, tmp_path: Path) -> None:
        """An absolute path to an existing file resolves to itself."""
        target = tmp_path / "notes.md"
        target.write_text("x")
        assert bare_path(str(target), "/") == target.resolve()

    def test_directory(self, tmp_path: Path) -> None:
        """A directory is a path too."""
        assert bare_path(str(tmp_path), "/") == tmp_path.resolve()

    def test_surrounding_whitespace_and_quotes(self, tmp_path: Path) -> None:
        """Pasted paths often arrive quoted or with a trailing newline."""
        target = tmp_path / "my file.md"
        target.write_text("x")
        assert bare_path(f'  "{target}"\n', "/") == target.resolve()
        assert bare_path(f"'{target}'", "/") == target.resolve()

    def test_mismatched_quotes_are_not_stripped(self, tmp_path: Path) -> None:
        """``"path'`` is not a quoted path, and no such file exists."""
        target = tmp_path / "f.md"
        target.write_text("x")
        assert bare_path(f'"{target}\'', "/") is None

    def test_tilde_expands_to_home(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``~/x`` is resolved through ``$HOME``."""
        monkeypatch.setenv("HOME", str(tmp_path))
        target = tmp_path / "Downloads" / "ROUTING.md"
        target.parent.mkdir()
        target.write_text("x")
        assert bare_path("~/Downloads/ROUTING.md", "/") == target.resolve()

    def test_unknown_user_tilde(self) -> None:
        """``~nosuchuser/...`` cannot be expanded and is not a path."""
        assert bare_path("~kiss-no-such-user-3f9a/f.md", "/") is None

    def test_relative_path_against_work_dir(self, tmp_path: Path) -> None:
        """A relative path is looked up in the run's work dir."""
        (tmp_path / "README.md").write_text("x")
        assert bare_path("README.md", str(tmp_path)) == (tmp_path / "README.md").resolve()
        assert bare_path("README.md", str(tmp_path / "elsewhere")) is None

    def test_absolute_path_ignores_work_dir(self, tmp_path: Path) -> None:
        """``Path(work_dir, absolute)`` is the absolute path."""
        target = tmp_path / "a.txt"
        target.write_text("x")
        assert bare_path(str(target), "/nonexistent/work/dir") == target.resolve()

    def test_non_paths(self, tmp_path: Path) -> None:
        """Instructions, blanks and missing files are not bare paths.

        The 5000-character prompt exercises the ``OSError`` branch on
        CPython 3.13, where ``Path.exists`` raises ``ENAMETOOLONG``;
        from 3.14 on ``exists`` returns ``False`` for such names, so the
        branch is unreachable there without a test double.
        """
        assert bare_path("", "/") is None
        assert bare_path("   \n", "/") is None
        assert bare_path('""', "/") is None
        assert bare_path("summarize the routing file", str(tmp_path)) is None
        assert bare_path(str(tmp_path / "missing.md"), "/") is None
        assert bare_path(f"{tmp_path}\nopen it", "/") is None
        assert bare_path("a\x00b", str(tmp_path)) is None
        assert bare_path("x" * 5000, str(tmp_path)) is None


class TestWithOpenDirective:
    """The directive names the resolved path and the kind of entry."""

    def test_file_directive(self, tmp_path: Path) -> None:
        """A file prompt gains the open-file directive, raw prompt first."""
        target = tmp_path / "ROUTING.md"
        target.write_text("x")
        out = with_open_directive(f" {target} ", "/")
        assert out.startswith(f" {target} \n\n{DIRECTIVE} file")
        assert f"`open {target.resolve()}`" in out
        assert f"`xdg-open {target.resolve()}`" in out
        assert "do not ask what to do with it" in out

    def test_directory_directive_quotes_spaces(self, tmp_path: Path) -> None:
        """A directory is called one, and a path with spaces is shell-quoted."""
        target = tmp_path / "My Docs"
        target.mkdir()
        out = with_open_directive(str(target), "/")
        assert f"{DIRECTIVE} directory" in out
        assert f"`open '{target.resolve()}'`" in out
        assert "the directory was opened" in out

    def test_other_prompts_unchanged(self, tmp_path: Path) -> None:
        """Anything that is not a bare path passes through untouched."""
        prompt = f"summarize {tmp_path / 'x.md'}"
        assert with_open_directive(prompt, "/") == prompt


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME + history DB + scratch git repo."""
    isolated = IsolatedKissHome("kiss-bare-path-task-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _RecordingModel:
    """Stand-in model that keeps the text of every request and finishes."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.prompts.append(request_text(request))
        return finish_response("opened")


def _run(env: IsolatedKissHome, prompt: str) -> tuple[str, _RecordingModel]:
    """Run a ``ChatSorcarAgent`` on *prompt* against the stand-in model."""
    model = _RecordingModel()
    server = StandInModelServer(model)
    try:
        result = ChatSorcarAgent("bare-path").run(
            prompt_template=prompt,
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            web_tools=False,
            is_parallel=False,
            verbose=False,
        )
    finally:
        server.stop()
    assert yaml.safe_load(result)["success"] is True, result
    return result, model


class TestChatSorcarAgentRun:
    """End to end: the model sees the directive, the history keeps the path."""

    def test_bare_path_prompt_gets_directive(self, env: IsolatedKissHome) -> None:
        """A bare-path task reaches the model with the open directive appended."""
        target = env.repo / "ROUTING.md"
        target.write_text("# routing\n")
        raw = str(target)
        _, model = _run(env, raw)
        assert model.prompts, "the model was never called"
        sent = model.prompts[0]
        assert f"# Task\n{raw}\n\n{DIRECTIVE} file" in sent
        assert f"`open {target.resolve()}`" in sent
        rows = history_rows()
        assert [row["task"] for row in rows] == [raw]

    def test_relative_bare_path_uses_work_dir(self, env: IsolatedKissHome) -> None:
        """A relative path is resolved against the run's ``work_dir``."""
        (env.repo / "notes.txt").write_text("n")
        _, model = _run(env, "notes.txt")
        assert f"`open {(env.repo / 'notes.txt').resolve()}`" in model.prompts[0]

    def test_ordinary_prompt_unchanged(self, env: IsolatedKissHome) -> None:
        """A prompt that is not a path is sent exactly as before."""
        _, model = _run(env, "say done")
        sent = model.prompts[0]
        assert "# Task\nsay done" in sent
        assert DIRECTIVE not in sent

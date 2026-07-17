# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for slash-command *argument* completion in the REPL.

Typing a slash command followed by a space (e.g. ``/resume ``) must pop
the command's argument options — ``--task`` / ``--limit`` for
``/resume``, ``list`` plus model names for ``/model``, and the
discovered skill names for ``/skills`` — in both the readline
(:class:`~kiss.ui.cli.cli_repl.CliCompleter`) and prompt_toolkit
(:class:`~kiss.ui.cli.cli_prompt.PtkCompleter`) frontends.  The
completions run against the real filesystem, real skill discovery, and
the real history database; no mocks are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import kiss.agents.sorcar.persistence as th
from kiss.ui.cli.cli_prompt import PtkCompleter
from kiss.ui.cli.cli_repl import CliCompleter


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


@pytest.fixture
def isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every user-level skills location at isolated temp dirs."""
    monkeypatch.setenv("KISS_HOME", str(tmp_path / ".kisshome"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claudehome"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    return tmp_path


def _write_skill(root: Path, name: str, description: str) -> None:
    """Create ``<root>/<name>/SKILL.md`` with the given frontmatter."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n",
        encoding="utf-8",
    )


def _ptk_completions(completer: CliCompleter, line: str) -> list:
    """Collect the prompt_toolkit completions for *line* (cursor at end)."""
    ptk = PtkCompleter(completer)
    doc = Document(text=line, cursor_position=len(line))
    return list(ptk.get_completions(doc, CompleteEvent()))


def test_resume_space_pops_argument_options(tmp_path: Path) -> None:
    """``/resume `` completes its ``--task`` / ``--limit`` options."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume ") == [
        "/resume --task ",
        "/resume --limit ",
    ]


def test_resume_option_prefix_narrows_candidates(tmp_path: Path) -> None:
    """``/resume --t`` completes only ``--task``."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume --t") == ["/resume --task "]


def test_resume_used_option_is_not_offered_again(tmp_path: Path) -> None:
    """Options already on the line are excluded from later completions."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume --task 5 --") == [
        "/resume --task 5 --limit ",
    ]


def test_model_space_offers_list_subcommand_first(tmp_path: Path) -> None:
    """``/model `` offers ``list`` ahead of the model names."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/model ")
    assert matches[0] == "/model list"
    assert len(matches) > 1, "model names should follow the list subcommand"


def test_model_list_prefix_completes_list(tmp_path: Path) -> None:
    """``/model li`` includes the ``list`` subcommand."""
    completer = CliCompleter(str(tmp_path))
    assert "/model list" in completer._build_matches("/model li")


def test_skills_space_completes_discovered_skill_names(
    tmp_path: Path, isolated_homes: Path,
) -> None:
    """``/skills `` completes the names of discovered skills."""
    project = tmp_path / "project"
    _write_skill(project / ".kiss" / "skills", "code-review", "Review code")
    completer = CliCompleter(str(project))
    matches = completer._build_matches("/skills ")
    assert "/skills code-review " in matches
    assert completer._build_matches("/skills code-r") == [
        "/skills code-review ",
    ]


def test_skills_help_text_comes_from_frontmatter(
    tmp_path: Path, isolated_homes: Path,
) -> None:
    """The dropdown help for a skill is its frontmatter description."""
    project = tmp_path / "project"
    _write_skill(project / ".kiss" / "skills", "code-review", "Review code")
    completer = CliCompleter(str(project))
    triples = completer._slash_arg_matches("/skills code-r")
    assert triples == [("/skills code-review ", "code-review", "Review code")]


def test_command_without_options_falls_back_to_predictive(
    tmp_path: Path, kiss_db,
) -> None:
    """``/help `` has no argument options; predictive completion applies."""
    th._add_task("/help me refactor the parser", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/help me ref") == [
        "/help me refactor the parser",
    ]
    # With a trailing space the same predictive fallback still applies
    # (the history line starts with "/help ") — no option menu appears.
    assert completer._build_matches("/help ") == [
        "/help me refactor the parser",
    ]


def test_unknown_slash_command_with_space_unaffected(
    tmp_path: Path, kiss_db,
) -> None:
    """Custom/unknown slash commands with arguments keep old behaviour."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/mycustom some args") == []


def test_readline_state_protocol_cycles_resume_options(tmp_path: Path) -> None:
    """The readline ``complete(text, state)`` protocol serves the options."""
    completer = CliCompleter(str(tmp_path))
    assert completer.complete("/resume ", 0) == "/resume --task "
    assert completer.complete("/resume ", 1) == "/resume --limit "
    assert completer.complete("/resume ", 2) is None


def test_ptk_dropdown_shows_resume_options_with_help(tmp_path: Path) -> None:
    """The prompt_toolkit menu shows the bare options with help text."""
    completer = CliCompleter(str(tmp_path))
    comps = _ptk_completions(completer, "/resume ")
    displays = ["".join(t for _, t in c.display) for c in comps]
    metas = ["".join(t for _, t in c.display_meta) for c in comps]
    assert displays == ["--task", "--limit"]
    assert metas == [
        "Open the chat containing this task id",
        "How many recent chats to list (default 20)",
    ]
    # Accepting a candidate replaces the whole line with "<cmd> <opt> ".
    assert comps[0].text == "/resume --task "
    assert comps[0].start_position == -len("/resume ")


def test_ptk_model_space_offers_list_with_help(tmp_path: Path) -> None:
    """``/model `` in the prompt_toolkit menu offers ``list`` first."""
    completer = CliCompleter(str(tmp_path))
    comps = _ptk_completions(completer, "/model ")
    assert comps[0].text == "list"
    meta = "".join(t for _, t in comps[0].display_meta)
    assert meta == "List all generation models"
    assert len(comps) > 1, "model names should follow the list subcommand"


def test_ptk_skills_space_lists_skill_names(
    tmp_path: Path, isolated_homes: Path,
) -> None:
    """``/skills `` in the prompt_toolkit menu lists discovered skills."""
    project = tmp_path / "project"
    _write_skill(project / ".kiss" / "skills", "code-review", "Review code")
    completer = CliCompleter(str(project))
    comps = _ptk_completions(completer, "/skills ")
    by_display = {
        "".join(t for _, t in c.display): c for c in comps
    }
    assert "code-review" in by_display
    assert by_display["code-review"].text == "/skills code-review "

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea lint``: every rule on a real script file, the codemod, and the CLI.

Each test writes a SEA into a temp folder and drives
:mod:`kiss.agents.sorcar.sea_lint` through its public surface
(:func:`lint_all`, :func:`fix_sea`, :func:`main`), so the findings come
from the real loader and the real AST walk.  The last test checks the
bundled tree is clean: that is the gate ``uv run check`` runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_cli import main
from kiss.agents.sorcar.sea_commands import sea_settings as sea_settings_of
from kiss.agents.sorcar.sea_lint import (
    Finding,
    bundled_seas,
    default_targets,
    fix_sea,
    lint_all,
    lint_sea,
    registered_seas,
)
from kiss.agents.sorcar.sea_settings import declares_hidden

DESCRIPTION = 'def description() -> str:\n    return "a test SEA"\n'


def write_sea(folder: Path, body: str, name: str = "demo") -> Path:
    """Write ``folder/name/name_sea.py`` with *body* and return its path."""
    sea_dir = folder / name
    sea_dir.mkdir(parents=True, exist_ok=True)
    path = sea_dir / f"{name}_sea.py"
    path.write_text(body, encoding="utf-8")
    return path


def codes(findings: list[Finding]) -> list[str]:
    """Return the rule codes of *findings*, in order."""
    return [finding.code for finding in findings]


@pytest.fixture
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``KISS_HOME`` at a fresh directory so ``SEAS.md`` is the test's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("KISS_HOME", str(home))
    sea_commands._reset_for_tests()
    yield home
    sea_commands._reset_for_tests()


def test_renamed_key_is_fixable_and_fix_rewrites_it(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            "def settings() -> dict:\n"
            "    return {'preset': 'worker', 'is_parallel': False, \"classify_tasks\": True}\n"
        ),
    )
    findings = lint_all([path])
    assert codes(findings) == ["renamed-key"] * 3
    assert all(f.fixable for f in findings)
    assert "'preset' is now 'kind'" in findings[0].message
    assert str(findings[0]).endswith("[fixable]")

    changed = fix_sea(path)
    assert [line.split(": ", 1)[1] for line in changed] == [
        "'preset' -> 'kind'",
        "'is_parallel' -> 'allow_fan_out'",
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = path.read_text(encoding="utf-8")
    # Each key keeps its own quote style and the rest of the line is untouched.
    assert "'kind': 'worker', 'allow_fan_out': False" in text and '"auto_classify": True' in text
    assert "preset" not in text and "is_parallel" not in text and "classify_tasks" not in text
    # The renamed fan-out key only states the preset default, so the fixed
    # script trips the next rule: the lint is a pipeline, not a one-shot.
    assert codes(lint_all([path])) == ["redundant-key"]
    assert fix_sea(path) == []


def test_fix_keeps_prefixes_quotes_and_locks_and_skips_nested_dicts(tmp_path: Path) -> None:
    """The codemod edits byte-accurately and only where a string names a settings key."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            "def settings() -> dict:\n"
            "    return {\n"
            "        'work_dir': 'café', r\"is_parallel\": False,\n"
            "        'locked': ['is_parallel', 'model'],\n"
            "        'model_config': {'preset': 'balanced'}, 'model': 'gpt-4o-mini',\n"
            "    }\n"
        ),
    )
    assert codes(lint_all([path])) == ["renamed-key", "renamed-key"]  # the key and its lock entry
    fix_sea(path)
    text = path.read_text(encoding="utf-8")
    assert "'work_dir': 'café', r\"allow_fan_out\": False,\n" in text
    assert "'locked': ['allow_fan_out', 'model'],\n" in text
    assert "'model_config': {'preset': 'balanced'}" in text  # a nested dict is data, not settings
    assert lint_all([path]) == []  # ``session`` has no defaults, so nothing is redundant
    settings = sea_settings_of(path)
    assert settings["locked"] == ["allow_fan_out", "model"] and settings["model_config"] == {
        "preset": "balanced"
    }


def test_broken_covers_getter_contract_errors(tmp_path: Path) -> None:
    """``broken`` runs the daemon's load path, so a getter of the wrong type is a finding."""
    bad_description = write_sea(tmp_path, "def description():\n    return 123\n", "baddesc")
    bad_tools = write_sea(
        tmp_path, DESCRIPTION + "def add_to_tools():\n    return 123\n", "badtools"
    )
    bad_prompt = write_sea(
        tmp_path, DESCRIPTION + "def system_prompt():\n    return 123\n", "badprompt"
    )
    findings = lint_all([bad_description, bad_tools, bad_prompt])
    assert codes(findings) == ["broken"] * 3
    assert "description()" in findings[0].message
    assert "add_to_tools()" in findings[1].message
    assert "system_prompt()" in findings[2].message


def test_redundant_key_names_the_kind_default(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            "def settings() -> dict:\n"
            "    return {'kind': 'worker', 'use_worktree': False, 'tool_profile': 'bash'}\n"
        ),
    )
    findings = lint_all([path])
    assert codes(findings) == ["redundant-key"]
    assert (
        findings[0].message
        == "settings()['use_worktree'] = False repeats the default of kind 'worker'"
    )
    assert not findings[0].fixable


def test_unknown_model_and_lock_without_value(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            "def settings() -> dict:\n"
            "    return {'model': 'no-such-model-xyz', 'locked': ['tool_profile', 'model']}\n"
        ),
    )
    findings = lint_all([path])
    assert codes(findings) == ["unknown-model", "lock-without-value"]
    assert "'no-such-model-xyz'" in findings[0].message
    assert "locked key 'tool_profile' has no value" in findings[1].message


def test_hidden_must_be_a_literal_and_hides_the_command(
    isolated_home: Path, tmp_path: Path
) -> None:
    folder = tmp_path / "seas"
    computed = write_sea(
        folder,
        DESCRIPTION + ("HIDE = True\n\ndef settings() -> dict:\n    return {'hidden': HIDE}\n"),
        "computed",
    )
    literal = write_sea(
        folder, DESCRIPTION + "def settings() -> dict:\n    return {'hidden': True}\n", "literal"
    )
    shown = write_sea(
        folder, DESCRIPTION + "def settings() -> dict:\n    return {'hidden': False}\n", "shown"
    )
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    # The registry reads the source, so only the literal hides the script.
    commands = sea_commands.list_commands()
    assert "computed" in commands and "shown" in commands and "literal" not in commands
    assert declares_hidden(literal) and not declares_hidden(computed) and not declares_hidden(shown)
    assert codes(lint_all([computed])) == ["hidden-not-literal"]
    assert lint_all([literal]) == [] and lint_all([shown]) == []
    # A hidden script still loads by path and as an ``extends`` base.
    assert sea_settings_of(literal)["hidden"] is True
    child = write_sea(
        folder,
        DESCRIPTION
        + (
            "def settings() -> dict:\n"
            f"    return {{'extends': {str(literal)!r}, 'tool_profile': 'bash'}}\n"
        ),
        "child",
    )
    sea_commands.refresh_registry()
    assert "child" in sea_commands.list_commands()
    assert sea_settings_of(child)["tool_profile"] == "bash" and "hidden" not in sea_settings_of(
        child
    )


def test_stale_docstring_and_home_literal(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        (
            '"""A script whose docstring describes the old getters ``tool_profile()`` '
            "and model().\n\n"
            'Prose may mention ~/.kiss freely; only code strings are flagged.\n"""\n'
            'HISTORY = "~/.kiss/history.db"\n\n'
            "def helper() -> str:\n"
            '    """Docstrings inside functions are prose too: ~/.kiss."""\n'
            '    return "~/.kiss/logs"\n'
        )
        + DESCRIPTION,
    )
    findings = lint_all([path])
    assert codes(findings) == ["stale-docstring", "home-literal", "home-literal"]
    assert findings[0].message.endswith("removed getters: model(), tool_profile()")
    assert findings[1].message.startswith("line 5:")
    assert findings[2].message.startswith("line 9:")


def test_broken_scripts(tmp_path: Path) -> None:
    syntax = write_sea(tmp_path, "def settings(:\n", "syntax")
    raises = write_sea(tmp_path, "raise RuntimeError('boom')\n", "raises")
    unknown = write_sea(
        tmp_path, DESCRIPTION + "def settings() -> dict:\n    return {'colour': 1}\n", "unknown"
    )
    removed = write_sea(
        tmp_path,
        DESCRIPTION + "def settings() -> dict:\n    return {'inherit': False}\n",
        "removed",
    )
    badkind = write_sea(
        tmp_path, DESCRIPTION + "def settings() -> dict:\n    return {'kind': 'agent'}\n", "badkind"
    )
    missing = tmp_path / "missing" / "missing_sea.py"
    findings = lint_all([syntax, raises, unknown, removed, badkind, missing])
    assert codes(findings) == ["broken"] * 6
    assert "cannot parse" in findings[0].message
    assert "boom" in findings[1].message
    assert "unknown key 'colour'" in findings[2].message
    assert "key 'inherit' was removed: a `channel` run never inherits" in findings[3].message
    assert "must be one of session, worker, channel" in findings[4].message
    assert "cannot parse" in findings[5].message


def test_registered_command_without_description(isolated_home: Path, tmp_path: Path) -> None:
    folder = tmp_path / "seas"
    silent = write_sea(folder, "def prompt(task: str) -> str:\n    return task\n", "silent")
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert silent in registered_seas()
    # Without ``--registered`` a user's script is never a target: ``uv run
    # check`` stays deterministic and ``--fix`` never rewrites a file
    # outside the checkout.
    assert silent not in default_targets(False) and silent in default_targets(True)
    assert all(f.path != silent for f in lint_all())
    found = [f for f in lint_all(registered=True) if f.path == silent]
    assert codes(found) == ["no-description"]
    assert found[0].message == "/silent has no description() for its help text"
    # Selecting it by path still sees the registration; only an unregistered
    # script (loadable by path or ``extends``) needs no description().
    assert codes(lint_all([silent])) == ["no-description"]
    assert lint_sea(silent, command=None) == []


def test_cli_folder_paths_fix_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sea(
        tmp_path, DESCRIPTION + "def settings() -> dict:\n    return {'is_parallel': True}\n"
    )
    assert main(["lint", str(path.parent)]) == 1
    out = capsys.readouterr().out
    assert "renamed-key" in out and out.strip().endswith("sea lint: 1 finding(s)")

    assert main(["lint", "--fix", str(path.parent)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("fixed ") and "'is_parallel' -> 'allow_fan_out'" in out
    assert out.strip().endswith("sea lint: 0 finding(s)")
    assert "allow_fan_out" in path.read_text(encoding="utf-8")

    with pytest.raises(SystemExit):
        main([])


def test_bundled_tree_is_clean() -> None:
    """The gate ``uv run check`` runs: every bundled script honours the contract."""
    scripts = bundled_seas()
    assert len(scripts) > 60
    assert all(
        script.name.endswith("_sea.py") or script.name == "cron_agent.py" for script in scripts
    )
    assert any(
        script.name == "cron_agent.py" for script in scripts
    )  # the built-in /cron is linted too
    assert os.path.join("agents", "seas") in str(scripts[0])
    assert lint_all(scripts) == []

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
    registered_commands,
)
from kiss.agents.sorcar.sea_settings import declares_channel, declares_hidden

DESCRIPTION = """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return "a test SEA"
"""


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


def test_renamed_and_removed_keys_are_fixable_and_fix_rewrites_them(tmp_path: Path) -> None:
    """A renamed key gets its new name; ``preset`` becomes the base class; ``is_parallel`` goes."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            """
    def settings(self, settings):
        return settings | {'preset': 'worker', 'is_parallel': False, "classify_tasks": False}
"""
        ),
    )
    findings = lint_all([path])
    assert codes(findings) == ["renamed-key", "base-class", "removed-key"]
    assert all(f.fixable for f in findings)
    assert "'classify_tasks' is now 'auto_classify'" in findings[0].message
    assert findings[1].message == (
        "line 9: settings() writes 'preset'; what a SEA is became its base class: "
        "derive from WorkerSea"
    )
    assert findings[2].message.startswith("line 9: settings() writes 'is_parallel', a removed key")
    assert str(findings[0]).endswith("[fixable]")

    changed = fix_sea(path)
    assert [line.split(": ", 1)[1] for line in changed] == [
        "import BaseSea -> WorkerSea",
        "class base BaseSea -> WorkerSea",
        "drop 'preset', 'is_parallel'",
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = path.read_text(encoding="utf-8")
    # The renamed key keeps its own quote style; the dropped entries leave the
    # rest of the line untouched; the class and its import name the new base.
    assert 'return settings | {"auto_classify": False}' in text
    assert "class Sea(WorkerSea):" in text
    assert "from kiss.agents.seas.base.base_sea import WorkerSea\n" in text
    assert "BaseSea" not in text
    assert "preset" not in text and "is_parallel" not in text and "classify_tasks" not in text
    # The renamed key only states what ``WorkerSea`` lays, so the fixed
    # script trips the next rule: the lint is a pipeline, not a one-shot.
    assert codes(lint_all([path])) == ["redundant-key"]
    assert fix_sea(path) == []
    assert sea_settings_of(path) == {
        "use_worktree": False,
        "auto_commit": False,
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }


def test_fix_keeps_prefixes_quotes_and_locks_and_skips_nested_dicts(tmp_path: Path) -> None:
    """The codemod edits byte-accurately and only where a string names a settings key."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            """
    def settings(self, settings):
        return settings | {
            'work_dir': 'café', r"classify_tasks": True,
            'locked': ['classify_tasks', 'model'],
            'model_config': {'preset': 'balanced'}, 'model': 'gpt-4o-mini',
        }
"""
        ),
    )
    assert codes(lint_all([path])) == ["renamed-key", "renamed-key"]  # the key and its lock entry
    fix_sea(path)
    text = path.read_text(encoding="utf-8")
    assert "'work_dir': 'café', r\"auto_classify\": True,\n" in text
    assert "'locked': ['auto_classify', 'model'],\n" in text
    assert "'model_config': {'preset': 'balanced'}" in text  # a nested dict is data, not settings
    assert lint_all([path]) == []  # ``BaseSea`` lays no defaults, so nothing is redundant
    settings = sea_settings_of(path)
    assert settings["locked"] == ["auto_classify", "model"] and settings["model_config"] == {
        "preset": "balanced"
    }


def test_literal_verdicts_are_fixable_and_fix_rewrites_them_to_verdicts(tmp_path: Path) -> None:
    """A ``tool_call_hook`` returning a literal ``None`` or string is flagged and rewritten.

    Only returns inside ``tool_call_hook`` count: a literal returned by
    another method or a nested helper, or compared rather than
    returned, is no finding.  ``--fix`` imports the names it needs.
    """
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            """
    def tool_call_hook(self, name, args):
        def status():
            return "OK"
        if name == "Bash" or status() != "OK":
            return "Blocked"
        if name == "Read":
            return 'OK'
        return None

    def status(self):
        return None
"""
        ),
    )
    findings = lint_all([path])
    assert codes(findings) == ["verdict", "verdict", "verdict", "broken"]
    assert all(f.fixable for f in findings[:3])
    assert "returns 'Blocked'; return ALLOW or refuse(text)" in findings[0].message
    assert "must return a Verdict" in findings[3].message
    assert fix_sea(path) == [
        f"{path}:3: import ALLOW, refuse",
        f"{path}:12: 'Blocked' -> refuse('Blocked')",
        f"{path}:14: 'OK' -> refuse('OK')",
        f"{path}:15: None -> ALLOW",
    ]
    text = path.read_text(encoding="utf-8")
    assert text.startswith(
        "\nfrom kiss.agents.seas.base.base_sea import BaseSea\n"
        "from kiss.agents.seas.base.base_sea import ALLOW, refuse\n"
    )
    assert '            return refuse("Blocked")\n' in text
    assert "            return refuse('OK')\n        return ALLOW\n" in text
    # The nested helper's return, the comparison and ``status()`` are untouched.
    assert text.count('"OK"') == 2 and "        return None\n" in text
    assert lint_all([path]) == [] and fix_sea(path) == []
    # A script that already imports the names gets no second import line.
    again = write_sea(tmp_path, DESCRIPTION.replace(
        "import BaseSea", "import ALLOW, BaseSea, refuse",
    ) + """
    def tool_call_hook(self, name, args):
        return None
""", name="imported")
    assert fix_sea(again) == [f"{again}:9: None -> ALLOW"]
    assert again.read_text(encoding="utf-8").count("import ALLOW") == 1


def test_channel_kind_is_fixable_and_fix_rewrites_the_base_class(tmp_path: Path) -> None:
    """``"kind": "channel"`` in ``settings()`` is flagged; ``--fix`` rebases the class."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        return settings | {"kind": "channel", "tool_profile": "bash", "locked": ["tool_profile"]}
""",
    )
    findings = lint_all([path])
    assert codes(findings) == ["base-class"]
    assert findings[0].fixable and findings[0].message.endswith("derive from ChannelSea")
    assert fix_sea(path) == [
        f"{path}:2: import BaseSea -> ChannelSea",
        f"{path}:4: class base BaseSea -> ChannelSea",
        f"{path}:9: drop 'kind'",
    ]
    text = path.read_text(encoding="utf-8")
    assert "from kiss.agents.seas.base.base_sea import ChannelSea\n" in text
    assert "class Sea(ChannelSea):" in text
    assert '{"tool_profile": "bash", "locked": ["tool_profile"]}' in text
    assert lint_all([path]) == [] and fix_sea(path) == []
    assert declares_channel(path)
    settings = sea_settings_of(path)
    assert settings["tool_profile"] == "bash" and settings["use_worktree"] is False
    assert {"tool_profile", "use_worktree"} <= set(settings["locked"])


def test_channel_flag_is_fixable_and_fix_rewrites_the_base_class(tmp_path: Path) -> None:
    """The former ``"channel": True`` flag is dropped the same way."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        return settings | {"channel": True}
""",
    )
    findings = lint_all([path])
    assert codes(findings) == ["base-class"] and findings[0].message.endswith("ChannelSea")
    assert [line.split(": ", 1)[1] for line in fix_sea(path)] == [
        "import BaseSea -> ChannelSea", "class base BaseSea -> ChannelSea", "drop 'channel'",
    ]
    text = path.read_text(encoding="utf-8")
    assert "class Sea(ChannelSea):" in text and "return settings | {}" in text
    assert lint_all([path]) == [] and declares_channel(path)


def test_fix_rewrites_aliased_and_attribute_bases_and_keeps_dict_unpacks(tmp_path: Path) -> None:
    """The base may be imported under an alias or named as ``base_sea.BaseSea``; a dropped
    entry next to a ``**`` unpack leaves the unpack intact; adjacent removed entries
    are one edit."""
    dotted = write_sea(tmp_path, """from kiss.agents.seas.base import base_sea


class Sea(base_sea.BaseSea):
    def description(self):
        return "dotted"

    def settings(self, settings):
        return settings | {"channel": True}
""", "dotted")
    assert [line.split(": ", 1)[1] for line in fix_sea(dotted)] == [
        "class base BaseSea -> ChannelSea", "drop 'channel'",
    ]
    text = dotted.read_text(encoding="utf-8")
    assert "class Sea(base_sea.ChannelSea):" in text and "return settings | {}" in text
    assert lint_all([dotted]) == [] and declares_channel(dotted)

    aliased = write_sea(tmp_path, """from kiss.agents.seas.base.base_sea import BaseSea as B

EXTRA = {"timeout": 5}


class Sea(B):
    def description(self):
        return "aliased"

    def settings(self, settings):
        return settings | {"kind": "worker", **EXTRA, "allow_fan_out": False}
""", "aliased")
    assert [line.split(": ", 1)[1] for line in fix_sea(aliased)] == [
        "import WorkerSea", "class base BaseSea -> WorkerSea",
        "drop 'kind'", "drop 'allow_fan_out'",
    ]
    text = aliased.read_text(encoding="utf-8")
    assert "class Sea(WorkerSea):" in text
    assert "from kiss.agents.seas.base.base_sea import WorkerSea\n" in text
    assert "return settings | {**EXTRA}" in text
    assert lint_all([aliased]) == []
    assert sea_settings_of(aliased)["timeout"] == 5
    assert sea_settings_of(aliased)["use_worktree"] is False

    adjacent = write_sea(tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return settings | {"kind": "worker", "allow_fan_out": False}
""", "adjacent")
    assert [line.split(": ", 1)[1] for line in fix_sea(adjacent)] == [
        "import BaseSea -> WorkerSea", "class base BaseSea -> WorkerSea",
        "drop 'kind', 'allow_fan_out'",
    ]
    assert "return settings | {}" in adjacent.read_text(encoding="utf-8")
    assert lint_all([adjacent]) == []


def test_declares_channel_sees_an_import_alias(tmp_path: Path) -> None:
    """The registry's AST scan lists a channel that imports ``ChannelSea`` under another name."""
    aliased = write_sea(tmp_path, """from kiss.agents.seas.base.base_sea import ChannelSea as Chan


class Sea(Chan):
    def description(self):
        return "aliased channel"
""", "aliasedchannel")
    assert declares_channel(aliased)
    assert sea_settings_of(aliased)["use_worktree"] is False
    plain = write_sea(tmp_path, """from kiss.agents.seas.base.base_sea import WorkerSea as Chan


class Sea(Chan):
    def description(self):
        return "aliased worker"
""", "aliasedworker")
    assert not declares_channel(plain)


def test_fix_leaves_entries_it_cannot_migrate(tmp_path: Path) -> None:
    """A ``channel`` whose value is not a literal, or a script with no class deriving
    from a SEA base, keeps its entry: dropping it would silently lose the behaviour."""
    computed = write_sea(tmp_path, """from kiss.agents.seas.base.base_sea import BaseSea

CHANNEL = True


class Sea(BaseSea):
    def description(self):
        return "computed"

    def settings(self, settings):
        return settings | {"channel": CHANNEL, "locked": ["kind", "model", "channel"]}
""", "computed")
    findings = lint_all([computed])
    assert codes(findings) == ["base-class"] and not findings[0].fixable
    assert findings[0].message.endswith("by hand (the value is not a literal)")
    assert [line.split(": ", 1)[1] for line in fix_sea(computed)] == [
        "unlock 'kind'", "unlock 'channel'",
    ]
    text = computed.read_text(encoding="utf-8")
    assert '{"channel": CHANNEL, "locked": ["model"]}' in text
    assert codes(lint_all([computed])) == ["base-class"]

    bare = write_sea(tmp_path, """class Sea:
    def description(self):
        return "bare"

    def settings(self, settings):
        return settings | {"model": "gpt-6-astra", "channel": True}
""", "bare")
    findings = lint_all([bare])
    assert codes(findings) == ["base-class"] and not findings[0].fixable
    assert findings[0].message.endswith(
        "derive from ChannelSea by hand (no class of the script derives from BaseSea)"
    )
    assert fix_sea(bare) == []
    assert '"channel": True' in bare.read_text(encoding="utf-8")


def test_channel_kind_inside_a_nested_dict_is_data_not_a_setting(tmp_path: Path) -> None:
    """A ``"kind": "channel"`` pair inside ``model_config`` is configuration data, left alone.

    Only the top-level ``"kind": "worker"`` is a setting: it is dropped and
    the class derives from ``WorkerSea``; the nested pairs stay as written.
    """
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        return settings | {
            "kind": "worker",
            "model_config": {"kind": "channel", "routes": [{"kind": "channel"}]},
        }
""",
    )
    assert codes(lint_all([path])) == ["base-class"]
    assert [line.split(": ", 1)[1] for line in fix_sea(path)] == [
        "import BaseSea -> WorkerSea", "class base BaseSea -> WorkerSea", "drop 'kind'",
    ]
    assert lint_all([path]) == [] and fix_sea(path) == []
    text = path.read_text(encoding="utf-8")
    assert "class Sea(WorkerSea):" in text
    assert '{"kind": "channel", "routes": [{"kind": "channel"}]}' in text
    assert sea_settings_of(path)["model_config"] == {
        "kind": "channel", "routes": [{"kind": "channel"}],
    }


def test_broken_covers_getter_contract_errors(tmp_path: Path) -> None:
    """``broken`` runs the daemon's load path, so a getter of the wrong type is a finding."""
    bad_description = write_sea(tmp_path, """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return 123
""", "baddesc")
    bad_tools = write_sea(
        tmp_path, DESCRIPTION + """
    def tools(self, tools):
        return 123
""", "badtools"
    )
    bad_prompt = write_sea(
        tmp_path, DESCRIPTION + """
    def system_prompt(self, system_prompt):
        return 123
""", "badprompt"
    )
    findings = lint_all([bad_description, bad_tools, bad_prompt])
    assert codes(findings) == ["broken"] * 3
    assert "description()" in findings[0].message
    assert "tools()" in findings[1].message
    assert "must return a list of tool callables" in findings[1].message
    assert "system_prompt()" in findings[2].message


def test_redundant_key_repeats_the_base_class_default(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import WorkerSea

class Sea(WorkerSea):
    def description(self):
        return "a test SEA"

    def settings(self, settings):
        return settings | {'use_worktree': False, 'tool_profile': 'bash'}
""",
    )
    findings = lint_all([path])
    assert codes(findings) == ["redundant-key"]
    assert (
        findings[0].message
        == "settings()['use_worktree'] = False repeats what the base classes lay"
    )
    assert not findings[0].fixable


def test_unknown_model_and_lock_without_value(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            """
    def settings(self, settings):
        return settings | {'model': 'no-such-model-xyz', 'locked': ['tool_profile', 'model']}
"""
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
        "HIDE = True\n" + DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': HIDE}
""",
        "computed",
    )
    literal = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': True}
""", "literal"
    )
    shown = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': False}
""", "shown"
    )
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    # The registry reads the source, so only the literal hides the script.
    commands = sea_commands.list_commands()
    assert "computed" in commands and "shown" in commands and "literal" not in commands
    assert declares_hidden(literal) and not declares_hidden(computed) and not declares_hidden(shown)
    assert codes(lint_all([computed])) == ["hidden-not-literal"]
    assert lint_all([literal]) == [] and lint_all([shown]) == []
    # A hidden script still loads by path and as a base class.
    assert sea_settings_of(literal)["hidden"] is True
    child = write_sea(
        folder,
        f"""
from kiss.agents.sorcar.sea_commands import sea_class

class Child(sea_class({str(literal)!r})):
    def settings(self, settings):
        return settings | {{'tool_profile': 'bash'}}
""",
        "child",
    )
    sea_commands.refresh_registry()
    # Only the literal in the child's own source hides it: the inherited
    # ``hidden`` (part of its effective settings) does not, nor is it a
    # computed-hidden finding against the child.
    assert "child" in sea_commands.list_commands() and not declares_hidden(child)
    assert sea_settings_of(child)["tool_profile"] == "bash"
    assert sea_settings_of(child)["hidden"] is True
    assert lint_all([child]) == []


def test_stale_docstring_and_home_literal(tmp_path: Path) -> None:
    path = write_sea(
        tmp_path,
        (
            '"""A script whose docstring describes the old getters ``tool_profile()`` '
            "and model().\n\n"
            'Prose may mention the bare name ~/.kiss; a ~/.kiss/ path is stale prose.\n"""\n'
            'HISTORY = "~/.kiss/history.db"\n'
            '"""A constant docstring (see :func:`dispatch_timeout`) is prose too."""\n\n'
            "def helper() -> str:\n"
            '    """Waits for the run before stopping it; data under ~/.kiss."""\n'
            '    return "~/.kiss/logs"\n'
            "def escaped() -> str:\n"
            # An escaped newline is not a physical line: the match is on line 14 (the
            # decoded value, one newline short, would put it on 13).
            '    """\\\n'
            "    Written with an escaped first newline (the `\\\\n` below is two characters).\n"
            "    Logs under ~/.kiss/logs. Not a line break: \\\\n\n"
            '    """\n'
            '    return ""\n'
        )
        + DESCRIPTION,
    )
    findings = lint_all([path])
    assert codes(findings) == [
        "stale-docstring",
        "stale-docstring",
        "stale-docstring",
        "stale-docstring",
        "stale-docstring",
        "home-literal",
        "home-literal",
    ]
    assert findings[0].message == (
        "line 1: docstring mentions removed getters: model(), tool_profile()"
    )
    assert findings[1].message.startswith("line 3: a `~/.kiss/` home path")
    assert findings[2].message == "line 6: docstring mentions removed getters: dispatch_timeout()"
    assert findings[3].message.startswith("line 9: claims a run_agent timeout stops the sub-task")
    assert findings[4].message.startswith("line 14: a `~/.kiss/` home path")
    assert findings[5].message.startswith("line 5:")
    assert findings[6].message.startswith("line 10:")


def test_broken_scripts(tmp_path: Path) -> None:
    syntax = write_sea(tmp_path, "def settings(:\n", "syntax")
    raises = write_sea(tmp_path, "raise RuntimeError('boom')\n", "raises")
    unknown = write_sea(
        tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'colour': 1}
""", "unknown"
    )
    removed = write_sea(
        tmp_path,
        DESCRIPTION + """
    def settings(self, settings):
        return settings | {'inherit': False}
""",
        "removed",
    )
    badkind = write_sea(
        tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'kind': 'agent'}
""", "badkind"
    )
    missing = tmp_path / "missing" / "missing_sea.py"
    findings = lint_all([syntax, raises, unknown, removed, missing])
    assert codes(findings) == ["broken"] * 5
    assert "cannot parse" in findings[0].message
    assert "boom" in findings[1].message
    assert "unknown key 'colour'" in findings[2].message
    assert "key 'inherit' was removed: a channel never inherits" in findings[3].message
    assert "cannot parse" in findings[4].message
    # A ``kind`` no base class answers to is still a removed key: the
    # entry is dropped and the class is left as it is.
    findings = lint_all([badkind])
    assert codes(findings) == ["base-class"] and findings[0].message.endswith("drop it")
    assert [line.split(": ", 1)[1] for line in fix_sea(badkind)] == ["drop 'kind'"]
    assert "return settings | {}" in badkind.read_text(encoding="utf-8")
    assert lint_all([badkind]) == [] and sea_settings_of(badkind) == {}


def test_registered_command_without_description(isolated_home: Path, tmp_path: Path) -> None:
    folder = tmp_path / "seas"
    silent = write_sea(folder, """
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def prompt(self, task):
        return task
""", "silent")
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    commands = registered_commands()
    assert commands[silent] == "silent"
    # Without ``--registered`` a user's script is never a target: ``uv run
    # check`` stays deterministic and ``--fix`` never rewrites a file
    # outside the checkout.
    assert silent not in default_targets(False, commands)
    assert silent in default_targets(True, commands)
    assert all(f.path != silent for f in lint_all())
    found = [f for f in lint_all(registered=True) if f.path == silent]
    assert codes(found) == ["no-description"]
    assert found[0].message == "/silent has no description() for its help text"
    # Selecting it by path still sees the registration; only an unregistered
    # script (loadable by path or as a base class) needs no description().
    assert codes(lint_all([silent])) == ["no-description"]
    assert lint_sea(silent, command=None) == []


def test_cli_folder_paths_fix_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_sea(
        tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'is_parallel': True}
"""
    )
    assert main(["lint", str(path.parent)]) == 1
    out = capsys.readouterr().out
    assert "removed-key" in out and out.strip().endswith("sea lint: 1 finding(s)")

    assert main(["lint", "--fix", str(path.parent)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("fixed ") and "drop 'is_parallel'" in out
    assert out.strip().endswith("sea lint: 0 finding(s)")
    text = path.read_text(encoding="utf-8")
    assert "is_parallel" not in text and "return settings | {}" in text

    with pytest.raises(SystemExit):
        main([])


def test_bundled_tree_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate ``uv run check`` runs: every bundled script honours the contract."""
    from kiss.core import config

    # Autorouter settings select from configured providers; lint never generates.
    monkeypatch.setattr(config.DEFAULT_CONFIG, "ANTHROPIC_API_KEY", "test-key")
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

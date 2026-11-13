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

import json
import os
import sys
import textwrap
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from kiss.agents.seas.base.base_sea import ALLOW
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.sea_cli import main
from kiss.agents.sorcar.sea_commands import bundled_commands, load_sea, own_settings
from kiss.agents.sorcar.sea_commands import sea_settings as sea_settings_of
from kiss.agents.sorcar.sea_lint import (
    PROSE_FILES,
    Finding,
    bundled_seas,
    default_targets,
    fix_sea,
    lint_all,
    lint_prose,
    lint_sea,
    registered_commands,
)
from kiss.agents.sorcar.sea_settings import SeaError, declares_channel, declares_hidden
from kiss.core.utils import replace_waiting_for_readers
from kiss.tests.agents.sorcar.conftest import reset_sea_registry

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
    path.write_text(body, encoding="utf-8", newline="")
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
    reset_sea_registry()
    yield home
    reset_sea_registry()


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
        "auto_classify": False,
        "use_web_tools": False,
        "use_memory": False,
    }


def test_fix_skips_a_script_it_cannot_parse_and_fixes_the_rest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``lint_all(fix=True)`` reports an unparsable or unreadable script as broken and goes on.

    A script whose *rewrite* no longer parses is the codemod's own
    failure, not the script's: it is kept as it is and the run says so
    (``not fixed <path>: ...``) instead of skipping it silently and
    listing its findings as ``[fixable]`` again.  Here the hook's
    literal sits at the parser's nesting limit, so wrapping it in
    ``refuse(...)`` is one parenthesis too many.
    """
    unparsable = write_sea(tmp_path, DESCRIPTION + "\n    def settings(self, settings:\n", "bad")
    missing = tmp_path / "gone" / "gone_sea.py"
    fixable = write_sea(
        tmp_path,
        DESCRIPTION
        + "\n    def settings(self, settings):\n        return {'classify_tasks': False}\n",
        "good",
    )
    deep_body = DESCRIPTION + (
        "\n    def tool_call_hook(self, name, args):\n        return "
        + "(" * 200 + '"no"' + ")" * 200 + "\n"
    )
    deep = write_sea(tmp_path, deep_body, "deep")
    findings = lint_all([unparsable, missing, fixable, deep], fix=True)
    assert [(f.path, f.code) for f in findings] == [
        (unparsable, "broken"), (missing, "broken"), (deep, "verdict"), (deep, "broken"),
    ]
    assert findings[0].message.startswith("cannot parse: ")
    assert findings[1].message.startswith("cannot parse: ")
    assert findings[2].fixable and "must return a Verdict" in findings[3].message
    out = capsys.readouterr().out
    assert out.startswith(f"fixed {fixable}:9: 'classify_tasks' -> 'auto_classify'\n")
    assert out.count("\n") == 2 and out.splitlines()[1].startswith(
        f"not fixed {deep}: the rewrite no longer parses and was not saved: "
        "too many nested parentheses"
    )
    assert "'auto_classify': False" in fixable.read_text(encoding="utf-8")
    assert deep.read_text(encoding="utf-8") == deep_body
    with pytest.raises(SeaError, match="no longer parses"):
        fix_sea(deep)
    assert deep.read_text(encoding="utf-8") == deep_body


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


@pytest.mark.parametrize("eol", ["\r\n", "\r"])
def test_fix_counts_the_lines_of_a_script_as_python_does(tmp_path: Path, eol: str) -> None:
    """A script whose lines end in ``\\r\\n`` or a lone ``\\r`` is rewritten as its ``\\n`` twin
    is, the reported line numbers included, and keeps its line endings."""
    body = DESCRIPTION + (
        "\n    def settings(self, settings):\n        return settings | {\n"
        "            'classify_tasks': False,\n            'is_parallel': (False),\n        }\n"
    )
    unix = write_sea(tmp_path, body, "unix")
    other = write_sea(tmp_path, "", "other")
    other.write_bytes(body.replace("\n", eol).encode("utf-8"))
    notes = [line.removeprefix(str(unix)) for line in fix_sea(unix)]
    assert notes == [":10: 'classify_tasks' -> 'auto_classify'", ":10: drop 'is_parallel'"]
    assert [line.removeprefix(str(other)) for line in fix_sea(other)] == notes
    assert other.read_bytes() == unix.read_bytes().replace(b"\n", eol.encode())


def test_fix_renames_a_key_spelled_with_adjacent_or_escaped_literals(tmp_path: Path) -> None:
    """A renamed key whose source is not the key's own text (adjacent literals, an escape)
    is replaced as a whole literal, so the fix never reports a rename it did not make."""
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + (
            """
    def settings(self, settings):
        return settings | {
            "model_" "name": "gpt-4o-mini", "classify\\x5ftasks": True,
            'locked': ['model_'
                       'name', "classify_" r"tasks"],
        }
"""
        ),
    )
    assert codes(lint_all([path])) == ["renamed-key"] * 4
    assert len(fix_sea(path)) == 4
    assert lint_all([path]) == [] and fix_sea(path) == []
    assert sea_settings_of(path) == {  # settings validation returns ``locked`` sorted
        "model": "gpt-4o-mini", "auto_classify": True, "locked": ["auto_classify", "model"],
    }


def test_settings_dicts_are_those_that_reach_the_return(tmp_path: Path) -> None:
    """Only a dict literal whose entries end up in the returned settings is a settings dict.

    A local dict the method merely reads a value from is data: renaming
    its ``model_name`` key would leave the ``info["model_name"]`` lookup
    behind and break a valid script at load time.  A dict assigned to a
    name the method returns (through ``|`` merges, ``**`` unpacks and
    re-assignments of the same name) is a settings dict: it is linted,
    fixed, and its literal ``hidden`` hides the command.
    """
    read_only = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        info = {"model_name": "gpt-4o-mini", "model": "unused"}
        return settings | {"model": info["model_name"]}
""",
        "readonly",
    )
    body = read_only.read_text(encoding="utf-8")
    assert fix_sea(read_only) == []
    assert read_only.read_text(encoding="utf-8") == body
    assert lint_all([read_only]) == []
    assert sea_settings_of(read_only)["model"] == "gpt-4o-mini"

    aliased = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        mine = {"hidden": True, "classify_tasks": False}
        mine = mine | {"model_config": {"model_name": "data"}}
        mine.update({"model_name": "gpt-6-astra"})
        extra = {**mine, "locked": ["classify_tasks"]}
        return settings | extra if settings else extra
""",
        "aliased",
    )
    assert declares_hidden(aliased)
    assert codes(lint_all([aliased])) == ["renamed-key", "renamed-key", "renamed-key"]
    assert [line.split(": ", 1)[1] for line in fix_sea(aliased)] == [
        "'classify_tasks' -> 'auto_classify'", "'model_name' -> 'model'",
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = aliased.read_text(encoding="utf-8")
    assert 'mine = {"hidden": True, "auto_classify": False}' in text
    assert '{"model_config": {"model_name": "data"}}' in text  # a nested dict is data
    assert 'mine.update({"model": "gpt-6-astra"})' in text  # updated into the result
    assert '"locked": ["auto_classify"]' in text
    assert lint_all([aliased]) == [] and declares_hidden(aliased)
    assert sea_settings_of(aliased) == {
        "hidden": True,
        "auto_classify": False,
        "model_config": {"model_name": "data"},
        "model": "gpt-6-astra",
        "locked": ["auto_classify"],
    }

    # A name re-assigned at the method's top level drops what it held:
    # the first ``config`` is data the method reads a value from, and
    # renaming its key would break the ``config["model_name"]`` lookup.
    # An assignment under an ``if`` may or may not run: its dict reaches
    # the return too, as does one merged from the name itself.
    reassigned = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        config = {"model_name": "gpt-5"}
        model = config["model_name"]
        config = {"model": model}
        if settings.get("hidden"):
            config = config | {"classify_tasks": False}
        return {**settings, **config}
""",
        "reassigned",
    )
    assert sea_settings_of(reassigned)["model"] == "gpt-5"
    assert codes(lint_all([reassigned])) == ["renamed-key"]
    assert [line.split(": ", 1)[1] for line in fix_sea(reassigned)] == [
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = reassigned.read_text(encoding="utf-8")
    assert 'config = {"model_name": "gpt-5"}' in text  # data, not settings
    assert 'config = config | {"auto_classify": False}' in text
    assert lint_all([reassigned]) == []
    assert sea_settings_of(reassigned)["model"] == "gpt-5"


def test_a_function_named_settings_that_is_no_method_is_not_read(tmp_path: Path) -> None:
    """Only a ``settings`` method of a class declares settings; a function of that name at
    module level or nested in a method is the script's own helper.

    The dict a helper returns is data: ``--fix`` renamed the key the SEA's method looks up
    in it, which broke a working script at load time, and a ``hidden`` written there hid the
    command (audit 2026-11-10, B/S9-F4).  Both helpers stay byte for byte as they are; the
    class's method is read, also where an ``if`` of the class body defines it.
    """
    path = write_sea(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import BaseSea

def settings():
    return {"model_name": "gpt-4o-mini", "hidden": True}

class Sea(BaseSea):
    def description(self):
        def settings():
            return {"model_name": "a test SEA"}
        return settings()["model_name"]

    if BaseSea:
        def settings(self, current):
            return current | {"model": settings()["model_name"], "classify_tasks": False}
""",
    )
    assert [line.split(": ", 1)[1] for line in fix_sea(path)] == [
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = path.read_text(encoding="utf-8")
    assert 'return {"model_name": "gpt-4o-mini", "hidden": True}' in text
    assert 'return {"model_name": "a test SEA"}' in text
    assert lint_all([path]) == [] and not declares_hidden(path)
    assert sea_settings_of(path) == {"model": "gpt-4o-mini", "auto_classify": False}
    assert load_sea(path).description() == "a test SEA"


def test_a_function_that_is_no_method_is_no_hook_and_a_mixin_is_read(tmp_path: Path) -> None:
    """A function named ``tool_call_hook`` at module level or nested in a method is the
    script's helper: ``--fix`` turned the literal it returns into a verdict.  It stays byte
    for byte as it is, while the mixin the SEA class derives from is read, however the class
    statement reaches it.
    """
    path = write_sea(
        tmp_path,
        """
from kiss.agents.seas.base.base_sea import BaseSea

def tool_call_hook(name):
    return "module helper"

class Defaults:
    def settings(self, current):
        return current | {"classify_tasks": False}

class Sea(Defaults, BaseSea):
    def description(self):
        def tool_call_hook():
            return "nested helper"

        return tool_call_hook()
""",
    )
    assert codes(lint_all([path])) == ["renamed-key"]
    assert [line.split(": ", 1)[1] for line in fix_sea(path)] == [
        "'classify_tasks' -> 'auto_classify'",
    ]
    text = path.read_text(encoding="utf-8")
    assert 'return "module helper"' in text and 'return "nested helper"' in text
    assert lint_all([path]) == [] and not declares_hidden(path)
    assert sea_settings_of(path) == {"auto_classify": False}
    assert load_sea(path).description() == "nested helper"
    # A mixin reaches the SEA class through an alias, a call, a decorator, an attribute or a
    # subscript as well as by name: every class of the file is read, wherever it is.
    for index, (prelude, base, postlude) in enumerate([
        ("Alias = Defaults", "Alias", ""),
        ("def mixed(cls):\n    return cls", "mixed(Defaults)", ""),
        ("class Outer:\n    Inner = Defaults", "Outer.Inner", ""),
        ("try:\n    from no_such_module import Mixin\nexcept ImportError:\n    Mixin = Defaults",
         "Mixin", ""),
        ("def swap(cls):\n    return Defaults\n\n@swap\nclass Other:\n    pass", "Other", ""),
        ("import colorsys\ncolorsys.Mixin = Defaults", "colorsys.Mixin", "del colorsys.Mixin"),
        ("from typing import Annotated", 'Annotated[Defaults, "metadata"]', ""),
        ("class Marker:\n    pass\n\ndef mixed(cls):\n    class Wrapped(Defaults, cls):\n"
         "        pass\n    return Wrapped\n\n@mixed", "Marker", ""),
    ]):
        indirect = write_sea(tmp_path, f"""from kiss.agents.seas.base.base_sea import BaseSea

class Defaults:
    def settings(self, current):
        return current | {{"hidden": True}}

{prelude}

class Sea({base}, BaseSea):
    def description(self):
        return "a test SEA"

{postlude}
""", f"indirect{index}")
        assert declares_hidden(indirect) and sea_settings_of(indirect) == {"hidden": True}, base


def test_a_script_nested_deeper_than_the_recursion_limit_is_read(tmp_path: Path) -> None:
    """A long ``elif`` chain (in ``settings`` and at module level), a long ``+`` chain and a
    ``settings`` built from a long ``|`` chain all run, so the registry and the linter read
    them: neither recurses on the depth of the script, which Python bounds far above its own
    recursion limit."""
    count = sys.getrecursionlimit() + 100
    elifs = "if N == 0:\n    pass\n" + "elif N == 0:\n    pass\n" * count
    path = write_sea(
        tmp_path,
        DESCRIPTION
        + "\n    def settings(self, settings):\n"
        + textwrap.indent(elifs, " " * 8)
        + "        return settings"
        + " | {'classify_tasks': False}" * count
        + " | {'hidden': True}\n\n"
        + f"N = {' + '.join(['1'] * count)}\n{elifs}",
    )
    assert declares_hidden(path)
    assert codes(lint_all([path])) == ["renamed-key"] * count
    assert len(fix_sea(path)) == count and lint_all([path]) == []
    assert sea_settings_of(path) == {"auto_classify": False, "hidden": True}


def test_fix_publishes_the_rewrite_atomically_under_a_concurrent_loader(tmp_path: Path) -> None:
    """A daemon loading the script while ``fix_sea`` saves it never sees a torn file.

    The rewrite is published by replacing the file, so a concurrent
    reader gets the old or the new script, never an empty or partial
    one (which ``Path.write_text``'s truncate-then-write would show).  A
    loader thread runs the real ``load_sea`` and reads the file's size
    in a loop while the main thread re-stales the script (atomically,
    so only ``fix_sea``'s own publish is under test) and fixes it again.
    """
    stale_body = DESCRIPTION + (
        "\n    def settings(self, settings):\n        return settings | {'classify_tasks': False}\n"
    )
    path = write_sea(tmp_path, stale_body)
    fixed_size = len(stale_body.encode("utf-8")) - 1  # 'auto_classify' is one byte shorter
    stop = threading.Event()
    failures: list[str] = []
    sizes: list[int] = []

    def load_until_stopped() -> None:
        while not stop.is_set():
            sizes.append(path.stat().st_size)
            try:
                load_sea(path)
            except SeaError as exc:
                failures.append(str(exc))

    reader = threading.Thread(target=load_until_stopped)
    reader.start()
    try:
        stale = path.with_name("stale.txt")
        for _ in range(300):
            stale.write_text(stale_body, encoding="utf-8")
            # The loader may hold *path* open: Windows refuses a bare replace then.
            replace_waiting_for_readers(stale, path)
            assert fix_sea(path) == [f"{path}:9: 'classify_tasks' -> 'auto_classify'"]
            if failures or min(sizes, default=fixed_size) < fixed_size:
                break
    finally:
        stop.set()
        reader.join()
    assert failures == []
    assert sizes and min(sizes) >= fixed_size
    assert path.read_text(encoding="utf-8") == stale_body.replace("classify_tasks", "auto_classify")
    assert sorted(p.name for p in path.parent.iterdir()) == [path.name]  # no staging file left


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
    assert settings["tool_profile"] == "bash" and settings["use_memory"] is False
    assert {"tool_profile", "use_memory"} <= set(settings["locked"])


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
    entry next to a ``**`` unpack leaves the unpack intact, parentheses included; the new
    base's import lands in the leading import section, not after a trailing import;
    adjacent removed entries are one edit."""
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


import os  # a legal trailing import must not pull the base's import after the class
""", "aliased")
    assert [line.split(": ", 1)[1] for line in fix_sea(aliased)] == [
        "import WorkerSea", "class base BaseSea -> WorkerSea",
        "drop 'kind'", "drop 'allow_fan_out'",
    ]
    text = aliased.read_text(encoding="utf-8")
    assert "class Sea(WorkerSea):" in text
    assert text.startswith(
        "from kiss.agents.seas.base.base_sea import BaseSea as B\n"
        "from kiss.agents.seas.base.base_sea import WorkerSea\n"
    )
    assert "return settings | {**EXTRA}" in text
    assert lint_all([aliased]) == []
    assert sea_settings_of(aliased)["timeout"] == 5
    assert sea_settings_of(aliased)["use_memory"] is False

    # The AST spans of ``**(settings)`` and ``(False)`` leave out their parentheses.
    parenthesized = write_sea(tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return {**(settings), "allow_fan_out": False, "timeout": (5.0), "is_parallel": (
            False  # one worker at a time
        ), **  # the base settings again
        (settings)}
""", "parenthesized")
    assert [line.split(": ", 1)[1] for line in fix_sea(parenthesized)] == [
        "drop 'allow_fan_out'", "drop 'is_parallel'",
    ]
    text = parenthesized.read_text(encoding="utf-8")
    assert (
        'return {**(settings), "timeout": (5.0), **  # the base settings again\n'
        "        (settings)}\n"
    ) in text
    assert lint_all([parenthesized]) == []
    assert sea_settings_of(parenthesized)["timeout"] == 5.0
    trailing = write_sea(tmp_path, DESCRIPTION + """
    def settings(self, settings):
        return {**(settings), "is_parallel": (False)}
""", "trailing")
    assert [line.split(": ", 1)[1] for line in fix_sea(trailing)] == ["drop 'is_parallel'"]
    assert "return {**(settings)}" in trailing.read_text(encoding="utf-8")
    assert lint_all([trailing]) == []

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
    assert sea_settings_of(aliased)["use_memory"] is False
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
        return settings | {'use_memory': False, 'tool_profile': 'bash'}
""",
    )
    findings = lint_all([path])
    assert codes(findings) == ["redundant-key"]
    assert (
        findings[0].message
        == "settings()['use_memory'] = False repeats what the base classes lay"
    )
    assert not findings[0].fixable


def test_a_base_whose_settings_cannot_run_twice_is_not_broken(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A base class whose ``settings`` refuses a second call on the instance passes the run's
    evaluation, so the script lints clean: the ``redundant-key`` comparison, which runs the
    bases' methods once more, is skipped without a word, and the next script is still linted.
    """
    parent = write_sea(
        tmp_path,
        DESCRIPTION
        + """
    def settings(self, settings):
        if getattr(self, "asked", False):
            raise RuntimeError("settings is read once per run")
        self.asked = True
        return settings
""",
        "parent",
    )
    once = write_sea(
        tmp_path,
        "from kiss.agents.sorcar.sea_commands import sea_class\n\n"
        f"class Once(sea_class({str(parent)!r})):\n    pass\n",
        "once",
    )
    redundant = write_sea(
        tmp_path,
        DESCRIPTION.replace("BaseSea", "WorkerSea")
        + "\n    def settings(self, settings):\n        return settings | {'use_memory': False}\n",
        "redundant",
    )
    assert [(f.path, f.code) for f in lint_all([once, redundant])] == [(redundant, "redundant-key")]
    assert not caplog.records


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
    # The literals are read in the order Python evaluates them: a ``**``
    # unpack lays its entries where it stands, so the entry written
    # after it wins, as at run time.
    unpacked = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        base = {'hidden': False, 'model': 'x'}
        return settings | {**base, 'hidden': True, **{'model': 'y'}, 'model': 'z'}
""", "unpacked"
    )
    overridden = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': True, **{'hidden': False}}
""", "overridden"
    )
    sea_commands.refresh_registry()
    assert sea_settings_of(unpacked)["hidden"] is True and declares_hidden(unpacked)
    assert sea_settings_of(unpacked)["model"] == "z"
    assert sea_settings_of(overridden)["hidden"] is False and not declares_hidden(overridden)
    commands = sea_commands.list_commands()
    assert "unpacked" not in commands and "overridden" in commands
    # A later computed value overwrites an earlier literal too: the
    # script no longer declares the literal, whatever the computed
    # value turns out to be, so the command stays listed (and the
    # linter reports the computed ``True``), and only a literal written
    # after the computed value hides it.
    recomputed = write_sea(
        folder,
        "HIDE = False\n" + DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': True} | {'hidden': HIDE}
""", "recomputed"
    )
    rehidden = write_sea(
        folder,
        "HIDE = True\n" + DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': True} | {'hidden': HIDE}
""", "rehidden"
    )
    relisted = write_sea(
        folder,
        "HIDE = False\n" + DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': HIDE} | {'hidden': True}
""", "relisted"
    )
    sea_commands.refresh_registry()
    assert sea_settings_of(recomputed)["hidden"] is False and not declares_hidden(recomputed)
    assert sea_settings_of(rehidden)["hidden"] is True and not declares_hidden(rehidden)
    assert sea_settings_of(relisted)["hidden"] is True and declares_hidden(relisted)
    commands = sea_commands.list_commands()
    assert "recomputed" in commands and "rehidden" in commands and "relisted" not in commands
    assert lint_all([recomputed]) == [] and lint_all([relisted]) == []
    assert codes(lint_all([rehidden])) == ["hidden-not-literal"]
    # The names of a chained assignment hold the one dict: what ``update``
    # and ``|=`` write through two of them is in what the third returns
    # (an attribute among the targets and a bare annotation change nothing).
    chained = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        first = second = third = self.fourth = {}
        third: dict
        first.update({'hidden': True})
        second |= {'classify_tasks': False}
        return settings | third
""", "chained"
    )
    sea_commands.refresh_registry()
    assert declares_hidden(chained) and "chained" not in sea_commands.list_commands()
    assert [line.split(": ", 1)[1] for line in fix_sea(chained)] == [
        "'classify_tasks' -> 'auto_classify'",
    ]
    assert sea_settings_of(chained) == {"hidden": True, "auto_classify": False}
    # A name read before anything was assigned to it (the method's
    # parameter) is one dict too: what ``update`` writes through a name
    # assigned from it is in what the parameter returns.
    aliased = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        mine = settings
        mine.update({'hidden': True})
        return settings
""", "aliased"
    )
    sea_commands.refresh_registry()
    assert sea_settings_of(aliased)["hidden"] is True and declares_hidden(aliased)
    assert "aliased" not in sea_commands.list_commands() and lint_all([aliased]) == []
    # The registry reads every ``settings`` method of the file (the source does not say
    # which classes the SEA class folds): the literal of a helper class hides the script
    # although its settings do not return it, and the linter names the line.
    helper = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        return settings

class Widget:
    def settings(self, settings):
        return {'hidden': True}
""", "helper"
    )
    sea_commands.refresh_registry()
    assert "hidden" not in sea_settings_of(helper) and declares_hidden(helper)
    assert "helper" not in sea_commands.list_commands()
    findings = lint_all([helper])
    assert codes(findings) == ["hidden-not-returned"]
    assert findings[0].message.startswith("line 13: ")  # the helper's literal
    # ... as does a literal on a path that the run does not take (a computed key beside it
    # is no literal entry and is passed over).
    branch = write_sea(
        folder, DESCRIPTION + """
    def settings(self, settings):
        if settings.get('model') == 'no such model':
            return settings | {'hidden': True, 'max_' + 'steps': 1}
        return settings
""", "branch"
    )
    sea_commands.refresh_registry()
    assert "branch" not in sea_commands.list_commands()
    assert codes(lint_all([branch])) == ["hidden-not-returned"]


@pytest.mark.parametrize(
    ("prefix", "header", "encoding"),
    [
        (b"\xef\xbb\xbf", "# a byte-order mark ahead, as Windows editors write it\n", "utf-8"),
        (b"", "# coding: latin-1\n# caf\xe9\n", "latin-1"),
    ],
    ids=["utf-8-bom", "latin-1-cookie"],
)
def test_a_script_is_read_in_the_source_encoding_import_reads(
    isolated_home: Path, tmp_path: Path, prefix: bytes, header: str, encoding: str
) -> None:
    """A UTF-8 byte-order mark or a ``coding`` cookie: the loader compiles the bytes as
    ``import`` does (PEP 263), so the script runs, but the registry's parse and the linter
    decoded plain UTF-8 — the mark became ``U+FEFF``, a syntax error, and latin-1 bytes did
    not decode — so a literal ``hidden`` was dropped, the command shown, and the script
    reported broken.  Read alike, the script is hidden, linted and fixed in its encoding."""
    folder = tmp_path / "seas"
    path = write_sea(folder, "", "encoded")
    body = header + DESCRIPTION + """
    def settings(self, settings):
        return settings | {'hidden': True, 'classify_tasks': False}
"""
    path.write_bytes(prefix + body.encode(encoding))
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert "encoded" not in sea_commands.list_commands(), "the literal hidden was not read"
    assert declares_hidden(path)
    assert codes(lint_all([path])) == ["renamed-key"]
    assert [line.split(": ", 1)[1] for line in fix_sea(path)] == [
        "'classify_tasks' -> 'auto_classify'",
    ]
    data = path.read_bytes()
    assert data.startswith(prefix + header.encode(encoding)), "the encoding was not kept"
    assert data.decode("utf-8-sig" if prefix else encoding).endswith(
        "return settings | {'hidden': True, 'auto_classify': False}\n"
    )
    assert lint_all([path]) == [] and declares_hidden(path)
    assert sea_settings_of(path) == {"hidden": True, "auto_classify": False}


def test_fix_inserts_its_import_below_the_leading_comments(tmp_path: Path) -> None:
    """The import ``--fix`` adds goes below the comments a script starts with.

    Ahead of them (a script whose first statement is no import) it took
    line 1 from the shebang and pushed a line-2 ``coding`` cookie to line
    3, where ``import`` does not look for one: the saved latin-1 script no
    longer compiled.  A decorator stays on the statement it decorates.
    """
    header = b"#!/usr/bin/env python3\n# coding: latin-1\n"
    path = write_sea(tmp_path, "", "latin")
    path.write_bytes(header + '''LABEL = "caf\xe9"
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return LABEL

    def tool_call_hook(self, name, args):
        return None
'''.encode("latin-1"))
    assert fix_sea(path) == [f"{path}:3: import ALLOW", f"{path}:11: None -> ALLOW"]
    assert path.read_bytes().startswith(
        header + b'from kiss.agents.seas.base.base_sea import ALLOW\nLABEL = "caf\xe9"\n'
    )
    sea = load_sea(path)  # compiled from the file's bytes, as ``import`` compiles them
    assert sea.description() == "caf\xe9" and sea.tool_call_hook("Bash", {}) == ALLOW
    assert lint_all([path]) == [] and fix_sea(path) == []
    decorated = write_sea(tmp_path, '''#!/usr/bin/env python3
# The first statement is a decorated class.

@object.__new__
class Marker:
    pass
from kiss.agents.seas.base.base_sea import BaseSea

class Sea(BaseSea):
    def description(self):
        return "a test SEA"

    def tool_call_hook(self, name, args):
        return "no"
''', "decorated")
    assert fix_sea(decorated) == [
        f"{decorated}:4: import refuse", f"{decorated}:14: 'no' -> refuse('no')",
    ]
    assert decorated.read_text(encoding="utf-8").startswith(
        "#!/usr/bin/env python3\n# The first statement is a decorated class.\n\n"
        "from kiss.agents.seas.base.base_sea import refuse\n@object.__new__\nclass Marker:\n"
    )
    assert lint_all([decorated]) == []


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
    # A ``~user`` that names no user is kept as written, as a shell keeps it: no such file.
    nobody = Path("~no-such-user-for-sea-lint/x_sea.py")
    findings = lint_all([syntax, raises, unknown, removed, missing, nobody])
    assert codes(findings) == ["broken"] * 6
    assert "cannot parse" in findings[0].message
    assert "boom" in findings[1].message
    assert "unknown key 'colour'" in findings[2].message
    assert "key 'inherit' was removed: a channel never inherits" in findings[3].message
    assert "cannot parse" in findings[4].message
    assert findings[5].path == nobody.resolve() and "cannot parse" in findings[5].message
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


def test_a_prose_claim_is_checked_against_the_bundled_command(
    isolated_home: Path, tmp_path: Path
) -> None:
    """A user's ``/sh`` shadowing the bundled one does not make a true page stale."""
    bundled = bundled_commands()["sh"]
    root = tmp_path / "root"
    page = root / PROSE_FILES[0]
    page.parent.mkdir(parents=True)
    claim = json.dumps(own_settings(load_sea(bundled)))
    page.write_text(f"`/sh` declares `{claim}`.\n", encoding="utf-8")
    assert lint_prose(root) == []
    folder = tmp_path / "seas"
    shadow = write_sea(folder, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'use_memory': False}
""", "sh")
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert sea_commands.get_command("sh") == shadow != bundled
    assert lint_prose(root) == []


def test_a_bundled_script_is_linted_against_the_bundled_commands(
    isolated_home: Path, tmp_path: Path
) -> None:
    """What ``SEAS.md`` registers does not change the findings of a bundled script.

    A user's ``/bestrouter`` that is no model picker made the bundled
    ``/write``, whose model is the bundled picker of that name, an
    ``unknown-model`` finding, and a user's ``/write`` took the bundled
    script's command name (its ``no-description`` check) away.  A user's
    own script is still checked against the registry its run reads.
    """
    write = bundled_commands()["write"]
    assert sea_settings_of(write)["model"] == "bestrouter"
    folder = tmp_path / "seas"
    mine = write_sea(folder, DESCRIPTION + """
    def settings(self, settings):
        return settings | {'model': 'bestrouter'}
""", "mine")
    (isolated_home / "SEAS.md").write_text(f"{folder}\n", encoding="utf-8")
    sea_commands.refresh_registry()
    assert lint_all([write, mine]) == []
    shadows = [write_sea(folder, DESCRIPTION, name) for name in ("bestrouter", "write")]
    sea_commands.refresh_registry()
    assert [sea_commands.get_command(name) for name in ("bestrouter", "write")] == shadows
    assert sea_commands.model_sea("bestrouter") is None
    assert [(f.path, f.code) for f in lint_all([write, mine])] == [(mine, "unknown-model")]
    assert sea_commands.model_sea("bestrouter", bundled=True) == bundled_commands()["bestrouter"]
    assert registered_commands()[write] == "write" == registered_commands()[shadows[1]]


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

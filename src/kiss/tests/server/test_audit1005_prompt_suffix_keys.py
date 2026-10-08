# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-10-05 (scope D): the prompt-suffix keys are options, never settings.

``add_to_prompt`` / ``add_to_system_prompt`` are ``run_agent`` /
``run_parallel`` options (``OPTION_TYPES``) and not ``settings()`` keys
(``SETTING_TYPES``).  Before the fix ``RENAMED_SETTINGS`` mapped the old
spellings ``append_to_prompt`` / ``append_to_system_prompt`` to those
option-only names, so a script using one was told "renamed to
``add_to_prompt``; run ``sea lint --fix``", ``fix_sea`` rewrote it, and
the rewritten script was then refused as "unknown key
``add_to_prompt``": a dead end.  Now a script's ``settings()`` refuses
all four spellings as removed, with the method to use instead, the
lint offers no rewrite for them, and the ``options`` object keeps
renaming the old spellings to the current option names.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.agent_dispatch import (
    make_run_agent_tool,
    options_keyword_hint,
    parse_run_options,
)
from kiss.agents.sorcar.sea_commands import sea_settings
from kiss.agents.sorcar.sea_lint import fix_sea, lint_all
from kiss.agents.sorcar.sea_settings import (
    REMOVED_SETTINGS,
    RENAMED_OPTIONS,
    RENAMED_SETTINGS,
    SETTING_TYPES,
    SeaError,
    resolve_settings,
)


def write_sea(folder: Path, key: str) -> Path:
    """Write a SEA whose ``settings()`` sets *key*, and return its path."""
    sea_dir = folder / "suffixed"
    sea_dir.mkdir()
    path = sea_dir / "suffixed_sea.py"
    path.write_text(
        "from kiss.agents.seas.base.base_sea import BaseSea\n\n\n"
        "class Sea(BaseSea):\n"
        "    def description(self):\n"
        "        return 'demo'\n\n"
        "    def settings(self, settings):\n"
        f"        return settings | {{{key!r}: 'Be brief.'}}\n",
        encoding="utf-8",
    )
    return path


def test_rename_tables_point_only_at_keys_that_exist() -> None:
    """A settings rename names a settings key; an options rename an options key."""
    assert set(RENAMED_SETTINGS.values()) <= set(SETTING_TYPES)
    assert RENAMED_OPTIONS == {
        **RENAMED_SETTINGS,
        "append_to_prompt": "add_to_prompt",
        "append_to_system_prompt": "add_to_system_prompt",
    }
    for key in ("append_to_prompt", "add_to_prompt"):
        assert "`prompt(task)` method" in REMOVED_SETTINGS[key]
    for key in ("append_to_system_prompt", "add_to_system_prompt"):
        assert "`system_prompt(system_prompt)` method" in REMOVED_SETTINGS[key]


@pytest.mark.parametrize(
    "key", ["append_to_prompt", "add_to_prompt", "append_to_system_prompt", "add_to_system_prompt"],
)
def test_a_prompt_suffix_in_settings_is_refused_with_the_method_to_use(
    tmp_path: Path, key: str,
) -> None:
    with pytest.raises(SeaError, match=f"settings\\(\\) key {key!r} was removed: "):
        resolve_settings({key: "x"})
    path = write_sea(tmp_path, key)
    with pytest.raises(SeaError, match="was removed") as info:
        sea_settings(path)
    assert "`run_agent` option, not a setting" in str(info.value)
    # The lint reports the same error and offers no rewrite: the old
    # dead end was a "renamed-key" finding whose fix produced a script
    # refused as having an unknown key.
    findings = lint_all([path])
    assert [f.code for f in findings] == ["broken"]
    assert not findings[0].fixable and "was removed" in findings[0].message
    before = path.read_text(encoding="utf-8")
    assert fix_sea(path) == []
    assert path.read_text(encoding="utf-8") == before


def test_options_keep_renaming_the_old_suffix_spellings(tmp_path: Path) -> None:
    for old, new in (
        ("append_to_prompt", "add_to_prompt"),
        ("append_to_system_prompt", "add_to_system_prompt"),
    ):
        with pytest.raises(ValueError, match=f"options key {old!r} was renamed to {new!r}"):
            parse_run_options(f'{{"{old}": "x"}}')
        assert getattr(parse_run_options(f'{{"{new}": "x"}}'), new) == "x"
        assert options_keyword_hint({old: "x"}) == (
            f"{old} was renamed to {new}; pass options='{{\"{new}\": \"x\"}}'."
        )
        # The current name is an option, so the hint shows the options object for it.
        assert options_keyword_hint({new: "x"}).startswith(
            f"{new} is a run setting, not an argument; pass it in the `options` JSON object"
        )
    out = make_run_agent_tool(str(tmp_path))("hi", options='{"append_to_prompt": "x"}')
    assert out == (
        "Error: options key 'append_to_prompt' was renamed to 'add_to_prompt'; use the new name."
    )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""``sea docs``: the generated vocabulary tables and the pages that carry them.

The tables are rendered from the code (``SETTING_TYPES``, ``kind_defaults()``,
``OPTION_TYPES``, ``bundled_commands()``); the tests check each table
against that code, the marker rewrite on a scratch page, the ``--check``
mode, and that the bundled pages are current (the gate ``uv run check``
runs).  The last test guards the ``run_agent`` tool description: every
option key must be named in the text the model reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.agent_dispatch import OPTION_TYPES, make_run_agent_tool
from kiss.agents.sorcar.sea_cli import main
from kiss.agents.sorcar.sea_commands import bundled_commands
from kiss.agents.sorcar.sea_docs import (
    GENERATED_FILES,
    REPO_ROOT,
    commands_table,
    kinds_table,
    options_table,
    render,
    settings_table,
)
from kiss.agents.sorcar.sea_settings import (
    DISPATCHER_SETTINGS,
    PRECEDENCE_RULE,
    SETTING_TYPES,
    kind_defaults,
    wire_field,
)


def rows(table: str) -> dict[str, list[str]]:
    """Parse a generated Markdown table into ``{first cell: cells}``."""
    out: dict[str, list[str]] = {}
    for line in table.splitlines()[2:]:
        cells = [cell.strip() for cell in line.strip("|").split(" | ")]
        out[cells[0].strip("`")] = cells
    return out


def test_settings_table_lists_every_key_with_its_wire_field() -> None:
    table = rows(settings_table())
    assert list(table) == list(SETTING_TYPES)
    assert table["max_budget"][1] == "`int \\| float`" and table["timeout"][1] == "`int \\| float`"
    assert (
        table["allow_fan_out"][2] == "`isParallel`"
        and table["auto_classify"][2] == "`classifyTasks`"
    )
    for key in DISPATCHER_SETTINGS:
        assert table[key][2] == "—"
    for key in SETTING_TYPES:
        if key not in DISPATCHER_SETTINGS:
            assert table[key][2] == f"`{wire_field(key)}`"
        assert table[key][3].endswith(".")


def test_kinds_table_states_each_kind_without_machine_paths() -> None:
    table = rows(kinds_table())
    assert list(table) == list(kind_defaults()) == ["session", "worker", "channel"]
    assert table["session"][1] == "nothing"
    assert "`allow_fan_out=False`" in table["worker"][1]
    assert "`work_dir='<home>/channel_work'`" in table["channel"][1]
    assert str(Path.home()) not in kinds_table()


def test_options_table_covers_option_types() -> None:
    table = rows(options_table())
    assert list(table) == list(OPTION_TYPES)
    assert "channel workspace" in table["workspace"][2]
    assert "never inherits" in table["inherit"][2]
    # The tool's own arguments are options too; only the script-describing keys are not.
    assert {"model", "tool_profile", "max_budget", "timeout"} <= set(table)
    assert table["timeout"][1] == "`int \\| float`"
    assert not {"kind", "extends", "locked", "hidden"} & set(table)


def test_commands_table_lists_every_bundled_command_once() -> None:
    table = rows(commands_table())
    assert list(table) == ["/" + name for name in sorted(bundled_commands())]
    assert table["/sh"][1] == "`agents/seas/sh/sh_sea.py`"
    assert table["/cron"][1] == "`agents/sorcar/cron_agent.py`"
    assert table["/sh"][2].count(". ") == 0 and table["/sh"][2].endswith(".")


def test_render_rewrites_only_marked_blocks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        "# Title\n\nprose stays\n\n<!-- sea-docs: kinds -->\nold\n<!-- /sea-docs -->\n\n"
        "<!-- sea-docs: options -->\n<!-- /sea-docs -->\n\ntail\n",
        encoding="utf-8",
    )
    assert main(["docs", "--check", str(page)]) == 1
    assert capsys.readouterr().out == f"stale: {page}\n"

    assert main(["docs", str(page)]) == 0
    assert capsys.readouterr().out == f"updated: {page}\n"
    text = page.read_text(encoding="utf-8")
    assert text.startswith(
        "# Title\n\nprose stays\n\n<!-- sea-docs: kinds -->\n| Kind | Defaults | Use |"
    )
    assert "old\n" not in text and text.endswith("<!-- /sea-docs -->\n\ntail\n")
    assert f"<!-- sea-docs: options -->\n{options_table()}\n<!-- /sea-docs -->" in text
    assert render(text) == text

    assert main(["docs", "--check", str(page)]) == 0
    assert capsys.readouterr().out == ""

    with pytest.raises(KeyError):
        render("<!-- sea-docs: nonsense -->\n<!-- /sea-docs -->")


def test_bundled_pages_are_current() -> None:
    """The gate ``uv run check`` runs: the committed tables match the code."""
    for rel in GENERATED_FILES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "<!-- sea-docs: precedence -->" in text
        assert render(text) == text, f"{rel} is stale: run `uv run sea docs`"
        assert text.count(PRECEDENCE_RULE) == 1, f"{rel} must state the rule once, generated"


def test_run_agent_description_names_every_option(tmp_path: Path) -> None:
    doc = make_run_agent_tool(str(tmp_path)).__doc__ or ""
    missing = [key for key in OPTION_TYPES if f"``{key}``" not in doc]
    assert missing == [], f"run_agent docstring does not mention options {missing}"
    assert "{channels}" not in doc

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

import json
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from kiss.agents.sorcar.agent_dispatch import OPTION_TYPES, make_run_agent_tool
from kiss.agents.sorcar.sea_cli import main
from kiss.agents.sorcar.sea_commands import bundled_commands
from kiss.agents.sorcar.sea_docs import (
    GENERATED_FILES,
    REPO_ROOT,
    bases_table,
    channel_table,
    commands_table,
    options_table,
    render,
    settings_table,
)
from kiss.agents.sorcar.sea_settings import (
    CHANNEL_BEHAVIOURS,
    DISPATCHER_SETTINGS,
    PRECEDENCE_RULE,
    SETTING_TYPES,
    base_class_defaults,
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
    assert table["auto_classify"][2] == "`classifyTasks`"
    for key in DISPATCHER_SETTINGS:
        assert table[key][2] == "—"
    for key in SETTING_TYPES:
        if key not in DISPATCHER_SETTINGS:
            assert table[key][2] == f"`{wire_field(key)}`"
        assert table[key][3].endswith(".")


def test_bases_table_states_each_base_class_without_machine_paths() -> None:
    table = rows(bases_table())
    assert list(table) == list(base_class_defaults()) == ["BaseSea", "WorkerSea", "ChannelSea"]
    assert table["BaseSea"][1] == "nothing"
    assert "`auto_classify=False`" in table["WorkerSea"][1]
    assert "use_worktree" not in table["WorkerSea"][1]
    assert "`work_dir='<home>/channel_work'`" in table["ChannelSea"][1]
    assert str(Path.home()) not in bases_table()


def test_channel_table_lists_every_behaviour_with_its_enforcer() -> None:
    table = rows(channel_table())
    assert list(table) == [name for name, _what in CHANNEL_BEHAVIOURS]
    for name, what in CHANNEL_BEHAVIOURS:
        assert table[name][1] == what
        assert "(`" in what and what.endswith(").")  # names the enforcing module
    assert "`<home>/channel_work`" in table["scratch directory"][1]
    assert "`inherit: true` option is refused" in table["no inheritance"][1]
    assert "<home>" not in str(Path.home())


def test_options_table_covers_option_types() -> None:
    table = rows(options_table())
    assert list(table) == list(OPTION_TYPES)
    assert "channel workspace" in table["workspace"][2]
    assert "never inherits" in table["inherit"][2]
    # The tool's own arguments and the script-describing keys are not options.
    assert not {"model", "tool_profile", "max_budget", "timeout"} & set(table)
    assert not {"kind", "channel", "extends", "locked", "hidden"} & set(table)


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
        "# Title\n\nprose stays\n\n<!-- sea-docs: bases -->\nold\n<!-- /sea-docs -->\n\n"
        "<!-- sea-docs: options -->\n<!-- /sea-docs -->\n\ntail\n",
        encoding="utf-8",
    )
    assert main(["docs", "--check", str(page)]) == 1
    assert capsys.readouterr().out == f"stale: {page}\n"

    assert main(["docs", str(page)]) == 0
    assert capsys.readouterr().out == f"updated: {page}\n"
    text = page.read_text(encoding="utf-8")
    assert text.startswith(
        "# Title\n\nprose stays\n\n<!-- sea-docs: bases -->\n| Base class | Lays | Use |"
    )
    assert "old\n" not in text and text.endswith("<!-- /sea-docs -->\n\ntail\n")
    assert f"<!-- sea-docs: options -->\n{options_table()}\n<!-- /sea-docs -->" in text
    assert render(text) == text

    assert main(["docs", "--check", str(page)]) == 0
    assert capsys.readouterr().out == ""

    with pytest.raises(KeyError):
        render("<!-- sea-docs: nonsense -->\n<!-- /sea-docs -->")


def test_a_write_cut_short_leaves_the_page_as_it_was(tmp_path: Path) -> None:
    """``sea docs`` swaps the regenerated page in whole.

    Written in place, a write that did not finish (an interrupt, a full
    disk) left a truncated page.  Here the process may write 4 KiB to a
    file, less than the page, so the write fails midway.
    """
    pytest.importorskip("resource")  # POSIX: the limit that cuts the write short
    page = tmp_path / "page.md"
    old = "<!-- sea-docs: channel -->\nold\n<!-- /sea-docs -->\n" + "tail\n" * 4000
    page.write_text(old, encoding="utf-8")
    script = (
        "import resource, sys\n"
        "from kiss.agents.sorcar.sea_cli import main\n"
        "hard = resource.getrlimit(resource.RLIMIT_FSIZE)[1]\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (4096, hard))\n"
        "main(['docs', sys.argv[1]])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, str(page)], capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 1 and "File too large" in done.stderr, done.stderr
    assert page.read_text(encoding="utf-8") == old
    assert list(tmp_path.iterdir()) == [page], "the staged copy was left behind"


def test_bundled_pages_are_current() -> None:
    """The gate ``uv run check`` runs: the committed tables match the code."""
    for rel in GENERATED_FILES:
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "<!-- sea-docs: precedence -->" in text
        assert render(text) == text, f"{rel} is stale: run `uv run sea docs`"
        assert text.count(PRECEDENCE_RULE) == 1, f"{rel} must state the rule once, generated"


def test_readme_example_tool_that_fails_on_a_locked_database_leaves_it_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SQLite example of the server README, run as written against a real lock."""
    text = (REPO_ROOT / "src/kiss/server/README.md").read_text(encoding="utf-8")
    start = text.index("# task_manager/task_manager_sea.py\n")
    example: dict[str, Any] = {}
    monkeypatch.setenv("KISS_HOME", str(tmp_path))  # where the example keeps its database
    exec(text[start : text.index("\n```", start)], example)
    assert json.loads(example["list_tasks"]()) == {"tasks": [], "count": 0}
    reader = sqlite3.connect(tmp_path / "task_manager.db")
    reader.execute("BEGIN")
    reader.execute("SELECT * FROM tasks").fetchall()  # a read lock: no commit gets through
    # The error stays referenced, as a caller that reports it keeps it.
    with pytest.raises(sqlite3.OperationalError, match="locked") as locked:
        example["add_task"]("first")
    reader.close()
    assert json.loads(example["add_task"]("second"))["added"]["title"] == "second"
    assert json.loads(example["complete_task"](1)) == {"completed": 1}
    assert [task["title"] for task in json.loads(example["list_tasks"](True))["tasks"]] == [
        "second",
    ]
    assert locked.type is sqlite3.OperationalError


def test_run_agent_description_names_every_option(tmp_path: Path) -> None:
    doc = make_run_agent_tool(str(tmp_path)).__doc__ or ""
    missing = [key for key in OPTION_TYPES if f"``{key}``" not in doc]
    assert missing == [], f"run_agent docstring does not mention options {missing}"
    assert "{channels}" not in doc

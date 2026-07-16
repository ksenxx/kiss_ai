# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for flag *value* completion in the CLI REPL.

Typing a value-taking flag followed by a space must pop the flag's
value candidates without further input: ``/resume --task `` lists the
recent task ids (displayed as ``<id>: <one-line description>`` but
inserting only the bare id) and ``--model `` lists the matching model
names in the same order as the extension's model picker (recently used
first, then vendor groups with the most expensive model first).  Both
the readline (:class:`~kiss.agents.sorcar.cli_repl.CliCompleter`) and
prompt_toolkit (:class:`~kiss.agents.sorcar.cli_prompt.PtkCompleter`)
frontends are exercised against the real history database and the real
model catalog; no mocks are used.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import kiss.agents.sorcar.persistence as th
import kiss.server.vscode_config as vc
from kiss.agents.sorcar.cli_prompt import PtkCompleter
from kiss.agents.sorcar.cli_repl import (
    _TASK_DESC_WIDTH,
    CliCompleter,
    picker_ordered_models,
)
from kiss.core.models.model_info import MODEL_INFO
from kiss.server.helpers import model_vendor


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect the history DB and config dir to an isolated temp dir."""
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    # ``_record_model_usage`` also persists the last-selected model to
    # ``config.json``; point that at the same isolated directory.
    vc.CONFIG_DIR = kiss_dir
    yield kiss_dir
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR = saved


def _ptk_completions(completer: CliCompleter, line: str) -> list:
    """Collect the prompt_toolkit completions for *line* (cursor at end)."""
    ptk = PtkCompleter(completer)
    doc = Document(text=line, cursor_position=len(line))
    return list(ptk.get_completions(doc, CompleteEvent()))


def _display(completion) -> str:
    """Return a completion's display text as a plain string."""
    return "".join(t for _, t in completion.display)


def _meta(completion) -> str:
    """Return a completion's display meta as a plain string."""
    return "".join(t for _, t in completion.display_meta)


# ---------------------------------------------------------------------------
# ``--task`` value completion — recent task ids
# ---------------------------------------------------------------------------


def test_task_flag_lists_recent_task_ids_newest_first(
    tmp_path: Path, kiss_db,
) -> None:
    """``/resume --task `` completes the recent task ids, newest first."""
    old_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    new_id, _ = th._add_task("write release notes", chat_id="c2")
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume --task ") == [
        f"/resume --task {new_id} ",
        f"/resume --task {old_id} ",
    ]


def test_task_flag_prefix_narrows_to_matching_id(
    tmp_path: Path, kiss_db,
) -> None:
    """A typed id prefix keeps only the ids that start with it."""
    task_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches(f"/resume --task {task_id}") == [
        f"/resume --task {task_id} ",
    ]


def test_task_flag_unmatched_partial_yields_nothing(
    tmp_path: Path, kiss_db,
) -> None:
    """An unmatched id prefix yields no candidates — and no flag menu.

    The flag-value position is terminal: the option menu (``--limit``)
    must not be re-offered as a bogus *value* for ``--task``, and the
    predictive history fallback must not fire either.
    """
    th._add_task("fix the parser bug", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume --task zzz") == []
    assert completer._build_matches("/resume --task ") != []


def test_ptk_task_candidates_show_id_colon_description(
    tmp_path: Path, kiss_db,
) -> None:
    """The dropdown shows ``<id>: <description>`` but inserts only the id."""
    task_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    comps = _ptk_completions(completer, "/resume --task ")
    assert len(comps) == 1
    assert _display(comps[0]) == f"{task_id}: fix the parser bug"
    # Accepting the candidate replaces the whole line with the bare id
    # (no colon, no description).
    assert comps[0].text == f"/resume --task {task_id} "
    assert comps[0].start_position == -len("/resume --task ")
    assert _meta(comps[0]) == "task"


def test_task_description_is_one_clipped_line(tmp_path: Path, kiss_db) -> None:
    """Multi-line / long task text collapses to one clipped line."""
    long_task = "refactor the completer\nso that it handles " + "x" * 100
    task_id, _ = th._add_task(long_task, chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    comps = _ptk_completions(completer, "/resume --task ")
    display = _display(comps[0])
    prefix = f"{task_id}: "
    assert display.startswith(prefix + "refactor the completer so that")
    desc = display[len(prefix):]
    assert "\n" not in desc
    assert len(desc) == _TASK_DESC_WIDTH
    assert desc.endswith("…")


# ---------------------------------------------------------------------------
# ``--model`` value completion — picker-ordered model names
# ---------------------------------------------------------------------------


def test_model_flag_lists_models_in_picker_order(
    tmp_path: Path, kiss_db,
) -> None:
    """``/cmd --model `` pops the model names in model-picker order."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/mycmd --model ")
    assert matches, "the model candidate list must never be empty"
    names = [m[len("/mycmd --model "):-1] for m in matches]
    assert all(m.endswith(" ") for m in matches)
    # No usage recorded: the order is (vendor order, price descending),
    # exactly like the daemon's ``_get_models`` picker payload.
    def picker_key(name: str) -> tuple[int, float]:
        info = MODEL_INFO[name]
        price = float(info.input_price_per_1M) + float(info.output_price_per_1M)
        return (model_vendor(name)[1], -price)

    assert names == sorted(names, key=picker_key)


def test_model_flag_substring_filter_matches_picker_search(
    tmp_path: Path, kiss_db,
) -> None:
    """The typed partial filters by case-insensitive substring."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/mycmd --model gemini")
    assert matches, "at least one gemini model is expected in the catalog"
    for m in matches:
        name = m[len("/mycmd --model "):-1]
        assert "gemini" in name.lower()


def test_recently_used_models_come_first_by_usage(
    tmp_path: Path, kiss_db,
) -> None:
    """Models with recorded usage lead the list, highest count first."""
    baseline = [name for name, _ in picker_ordered_models("")]
    assert len(baseline) >= 2, "need at least two models to reorder"
    once, twice = baseline[-2], baseline[-1]
    th._record_model_usage(once)
    th._record_model_usage(twice)
    th._record_model_usage(twice)
    ordered = picker_ordered_models("")
    assert [name for name, _ in ordered[:2]] == [twice, once]
    assert ordered[0][1] == "recently used"
    assert ordered[1][1] == "recently used"
    # The remaining (unused) models keep the picker's base order.
    assert [name for name, _ in ordered[2:]] == [
        name for name in baseline if name not in {once, twice}
    ]


def test_ptk_model_candidates_show_vendor_group(
    tmp_path: Path, kiss_db,
) -> None:
    """The dropdown shows the bare model name with its picker group."""
    completer = CliCompleter(str(tmp_path))
    comps = _ptk_completions(completer, "/mycmd --model ")
    assert comps
    vendors = {"Anthropic", "OpenAI", "Gemini", "Z.AI", "Moonshot",
               "OpenRouter", "Together AI", "recently used"}
    for c in comps:
        assert c.text.startswith("/mycmd --model ")
        assert c.text.endswith(" ")
        assert _display(c) == c.text[len("/mycmd --model "):-1]
        assert _meta(c) in vendors
        assert c.start_position == -len("/mycmd --model ")


def test_model_command_line_keeps_dedicated_completion(
    tmp_path: Path, kiss_db,
) -> None:
    """``/model <partial>`` still uses its dedicated completion path."""
    completer = CliCompleter(str(tmp_path))
    matches = completer._build_matches("/model ")
    assert matches[0] == "/model list"


def test_flag_options_still_offered_before_value_position(
    tmp_path: Path, kiss_db,
) -> None:
    """``/resume --t`` (no space yet) still completes the flag itself."""
    completer = CliCompleter(str(tmp_path))
    assert completer._build_matches("/resume --t") == ["/resume --task "]


def test_readline_state_protocol_serves_task_ids(
    tmp_path: Path, kiss_db,
) -> None:
    """The readline ``complete(text, state)`` protocol serves the ids."""
    task_id, _ = th._add_task("fix the parser bug", chat_id="c1")
    completer = CliCompleter(str(tmp_path))
    assert completer.complete("/resume --task ", 0) == f"/resume --task {task_id} "
    assert completer.complete("/resume --task ", 1) is None

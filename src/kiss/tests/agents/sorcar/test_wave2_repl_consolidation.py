# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for the wave-2 CLI/daemon consolidations.

Covers the shared single-source implementations extracted from the
previously duplicated code paths:

* active-file identifier harvesting — shared between the VS Code
  daemon's ghost-text completion
  (:meth:`~kiss.server.autocomplete._AutocompleteMixin
  ._active_file_identifier_matches`) and the CLI completer
  (:meth:`~kiss.ui.cli.cli_repl.CliCompleter
  ._active_file_suffix`);
* the ``/help`` body — single-sourced in
  :func:`~kiss.ui.cli.cli_repl.build_help_text` and used by
  both the standalone REPL and the daemon's ``cliInfo`` reply;
* the model-picker ordering rule — single-sourced in
  :func:`~kiss.server.autocomplete
  .ranked_function_calling_models` and used by both the daemon's
  ``getModels`` reply and the CLI's model completion;
* the backslash line-continuation read loop — single-sourced in
  :func:`~kiss.ui.cli.cli_line_continuation.read_continuations`;
* the completion dispatch chain — single-sourced in
  :meth:`~kiss.ui.cli.cli_repl.CliCompleter.completion_branch`
  and consumed by both the readline menu and the prompt_toolkit
  dropdown.

All tests run against the real implementations (real files, real
history database redirected to an isolated temp dir); no kiss code is
mocked or patched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

import kiss.agents.sorcar.persistence as th
import kiss.core.vscode_config as vc
from kiss.server.autocomplete import (
    _AutocompleteMixin,
    identifier_prefix_matches,
    ranked_function_calling_models,
    read_active_file_head,
    trailing_identifier,
)
from kiss.server.helpers import clip_autocomplete_suggestion
from kiss.ui.cli.cli_line_continuation import read_continuations
from kiss.ui.cli.cli_prompt import PtkCompleter
from kiss.ui.cli.cli_repl import (
    SLASH_COMMANDS,
    CliCompleter,
    _print_help,
    build_help_text,
    picker_ordered_models,
)


@pytest.fixture
def kiss_db(tmp_path: Path):
    """Redirect the history DB and config dir to an isolated temp dir."""
    kiss_dir = tmp_path / ".kiss"
    kiss_dir.mkdir(parents=True, exist_ok=True)
    saved = (th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    vc.CONFIG_DIR = kiss_dir
    yield kiss_dir
    if th._db_conn is not None:
        th._db_conn.close()
    th._DB_PATH, th._db_conn, th._KISS_DIR, vc.CONFIG_DIR = saved


# ---------------------------------------------------------------------------
# (A) active-file identifier harvesting parity
# ---------------------------------------------------------------------------

_SAMPLE_SOURCE = (
    "alpha_one = compute_total(alpha_two)\n"
    "self.alpha_helper_long = os.path.join(base, name)\n"
    "beta = alpha_one + alpha_two\n"
)


def test_identifier_harvest_parity_daemon_vs_cli(
    tmp_path: Path, kiss_db,
) -> None:
    """Both surfaces harvest the same identifiers from a sample file."""
    src = tmp_path / "sample.py"
    src.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    query = "please rename alpha"
    partial = "alpha"

    # Daemon surface: all matches, longest first.
    daemon_matches = _AutocompleteMixin._active_file_identifier_matches(
        _AutocompleteMixin(), query, snapshot_file=str(src),
    )
    assert set(daemon_matches) == {
        "alpha_one", "alpha_two", "alpha_helper_long",
    }
    assert daemon_matches == sorted(daemon_matches, key=lambda c: (-len(c), c))

    # Shared harvesting core produces the identical candidate set.
    assert set(identifier_prefix_matches(_SAMPLE_SOURCE, partial)) == set(
        daemon_matches
    )

    # CLI surface: the single longest suffix, clipped for one line.
    completer = CliCompleter(str(tmp_path), active_file=str(src))
    suffix = completer._active_file_suffix(query)
    longest = max(daemon_matches, key=len)
    assert suffix == clip_autocomplete_suggestion(
        query, longest[len(partial):],
    )
    # And the predictive whole-line candidate splices it onto the query.
    assert completer._predictive_matches(query) == [query + suffix]


def test_identifier_harvest_parity_dot_chains(tmp_path: Path, kiss_db) -> None:
    """Dot-chained identifiers complete identically on both surfaces."""
    src = tmp_path / "sample.py"
    src.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    query = "use os.pa"
    assert trailing_identifier(query) == "os.pa"

    daemon_matches = _AutocompleteMixin._active_file_identifier_matches(
        _AutocompleteMixin(), query, snapshot_file=str(src),
    )
    assert daemon_matches == ["os.path.join"]

    completer = CliCompleter(str(tmp_path), active_file=str(src))
    assert completer._active_file_suffix(query) == clip_autocomplete_suggestion(
        query, "os.path.join"[len("os.pa"):],
    )


def test_identifier_harvest_edge_cases(tmp_path: Path, kiss_db) -> None:
    """Short tokens, missing files, and non-identifier tails harvest nothing."""
    src = tmp_path / "sample.py"
    src.write_text(_SAMPLE_SOURCE, encoding="utf-8")
    completer = CliCompleter(str(tmp_path), active_file=str(src))
    # Trailing token shorter than 2 chars.
    assert trailing_identifier("x") == ""
    assert completer._active_file_suffix("a") == ""
    # Query not ending in an identifier.
    assert trailing_identifier("hello ") == ""
    assert _AutocompleteMixin._active_file_identifier_matches(
        _AutocompleteMixin(), "hello ", snapshot_file=str(src),
    ) == []
    # Python's ``$`` regex anchor also matches before a final newline.
    # The cursor is actually on a fresh line, so neither surface may
    # complete the identifier from the preceding line.
    newline_query = "please use os.pa\n"
    assert trailing_identifier(newline_query) == ""
    assert _AutocompleteMixin._active_file_identifier_matches(
        _AutocompleteMixin(), newline_query, snapshot_file=str(src),
    ) == []
    assert completer._active_file_suffix(newline_query) == ""
    # Unreadable file yields no content, hence no matches.
    assert read_active_file_head(str(tmp_path / "missing.py")) == ""
    gone = CliCompleter(str(tmp_path), active_file=str(tmp_path / "gone.py"))
    assert gone._active_file_suffix("please rename alpha") == ""
    # Non-UTF-8 files are skipped, not raised.
    binary = tmp_path / "blob.bin"
    binary.write_bytes(b"\xff\xfe\x00alpha_bin\x00")
    assert read_active_file_head(str(binary)) == ""


# ---------------------------------------------------------------------------
# (B) /help body single-sourcing
# ---------------------------------------------------------------------------

def test_print_help_uses_shared_help_text(
    tmp_path: Path, kiss_db, capsys: pytest.CaptureFixture[str],
) -> None:
    """The REPL's ``/help`` output is exactly the shared body, framed."""
    _print_help(str(tmp_path))
    out = capsys.readouterr().out
    assert out == f"\n{build_help_text(str(tmp_path))}\n\n"


def test_help_text_contents(tmp_path: Path, kiss_db) -> None:
    """The shared body lists every built-in command and the cheat sheet."""
    text = build_help_text(str(tmp_path))
    assert text.startswith("Commands:")
    for cmd, desc in SLASH_COMMANDS.items():
        assert f"  {cmd:<10} {desc}" in text
    assert "Input fast-completes (Tab): @path mentions files, " in text
    assert text.endswith("suggests its completion.")
    # Custom commands are appended when discovered.
    cmd_dir = tmp_path / ".kiss" / "commands"
    cmd_dir.mkdir(parents=True, exist_ok=True)
    (cmd_dir / "mycmd.md").write_text(
        "say hello to $ARGUMENTS\n", encoding="utf-8",
    )
    with_custom = build_help_text(str(tmp_path))
    assert "Custom commands:" in with_custom
    assert "/mycmd" in with_custom


def _make_recording_printer():
    """Return a JsonPrinter that also records every broadcast event."""
    from kiss.server.json_printer import JsonPrinter

    class _Recording(JsonPrinter):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[dict[str, Any]] = []

        def broadcast(self, event: dict[str, Any]) -> None:
            self.events.append(dict(event))
            super().broadcast(event)

    return _Recording()


def test_daemon_help_reply_matches_shared_help_text(
    tmp_path: Path, kiss_db,
) -> None:
    """The daemon's ``cliInfo help`` reply is byte-identical to the body."""
    from kiss.server.server import VSCodeServer

    printer = _make_recording_printer()
    server = VSCodeServer(printer=printer)
    server._cmd_cli_info({
        "type": "cliInfo", "subtype": "help", "workDir": str(tmp_path),
    })
    replies = [
        e for e in printer.events
        if e.get("type") == "cliInfo" and e.get("subtype") == "help"
    ]
    assert replies, "the daemon must reply to cliInfo help"
    assert replies[-1]["text"] == build_help_text(str(tmp_path))


# ---------------------------------------------------------------------------
# (C) model-picker ordering parity
# ---------------------------------------------------------------------------

def test_model_picker_order_parity_daemon_vs_cli(
    tmp_path: Path, kiss_db,
) -> None:
    """Daemon ``getModels`` and CLI completion order models identically."""
    from kiss.server.server import VSCodeServer

    ranked = ranked_function_calling_models()

    printer = _make_recording_printer()
    server = VSCodeServer(printer=printer)
    server._get_models()
    events = [
        e for e in printer.events if e.get("type") == "models"
    ]
    assert events, "the daemon must broadcast a models event"
    daemon_names = [m["name"] for m in events[-1]["models"]]
    assert daemon_names == ranked

    if not ranked:
        pytest.skip("no provider credential configured in this environment")
    cli_names = [name for name, _ in picker_ordered_models("")]
    # No usage recorded in the isolated DB: the CLI order is exactly
    # the shared picker order.
    assert cli_names == ranked


# ---------------------------------------------------------------------------
# (D) shared backslash line-continuation read loop
# ---------------------------------------------------------------------------

def _reader_from(rows: list[str]):
    """Return a ``read_more`` callable yielding *rows* then EOFError."""
    it = iter(rows)

    def _read() -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _read


def test_read_continuations_joins_rows() -> None:
    """Continuation markers are stripped and rows joined with newlines."""
    assert read_continuations("a \\", _reader_from(["b\\", "c"])) == "a \nb\nc"


def test_read_continuations_no_marker_reads_nothing() -> None:
    """A line without a marker is returned untouched, reading no rows."""

    def _boom() -> str:
        raise AssertionError("read_more must not be called")

    assert read_continuations("plain line", _boom) == "plain line"
    # An even (escaped-literal) number of backslashes submits verbatim.
    assert read_continuations("x \\\\", _boom) == "x \\\\"


def test_read_continuations_eof_strips_dangling_marker() -> None:
    """EOF mid-continuation keeps the input read so far, marker stripped."""
    seen: list[str] = []
    assert read_continuations(
        "x \\", _reader_from([]), on_eof=lambda: seen.append("eof"),
    ) == "x "
    assert seen == ["eof"]


def test_read_continuations_interrupt_runs_hook_and_reraises() -> None:
    """Ctrl+C mid-continuation runs the cleanup hook then re-raises."""
    seen: list[str] = []

    def _interrupt() -> str:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        read_continuations(
            "x \\", _interrupt, on_interrupt=lambda: seen.append("int"),
        )
    assert seen == ["int"]


def test_read_continuations_on_continue_runs_per_row() -> None:
    """The per-row hook fires once before every continuation read."""
    seen: list[str] = []
    out = read_continuations(
        "a\\",
        _reader_from(["b\\", "c"]),
        on_continue=lambda: seen.append("row"),
    )
    assert out == "a\nb\nc"
    assert seen == ["row", "row"]


# ---------------------------------------------------------------------------
# (E) shared completion dispatch chain
# ---------------------------------------------------------------------------

def test_completion_branch_order(tmp_path: Path, kiss_db) -> None:
    """The shared dispatch classifies every category in menu order."""
    completer = CliCompleter(str(tmp_path))
    assert completer.completion_branch("look at @src")[0] == "at"
    assert completer.completion_branch("/model gp") == ("model", "gp")
    assert completer.completion_branch("/res") == ("slash", None)
    kind, payload = completer.completion_branch("/resume --task ")
    assert kind == "flag-value"
    assert isinstance(payload, list)
    kind, payload = completer.completion_branch("/resume --")
    assert kind == "arg-options"
    assert payload and all(len(t) == 3 for t in payload)
    assert completer.completion_branch("hello world") == ("predictive", None)


def test_dispatch_parity_readline_vs_ptk(tmp_path: Path, kiss_db) -> None:
    """Both frontends resolve the same branch for the same input line."""
    (tmp_path / "somefile.txt").write_text("data\n", encoding="utf-8")
    completer = CliCompleter(str(tmp_path))
    ptk = PtkCompleter(completer)

    def _ptk_texts(line: str) -> list[str]:
        doc = Document(text=line, cursor_position=len(line))
        event = CompleteEvent(completion_requested=True)
        return [c.text for c in ptk.get_completions(doc, event)]

    # Slash-command branch: same command candidates on both frontends.
    menu = completer.build_menu("/he")
    assert [r for r, _ in menu] == ["/help "]
    assert _ptk_texts("/he") == ["/help "]

    # @-mention branch: both offer the file, formatted per frontend.
    menu = completer.build_menu("see @some")
    assert menu and menu[0][0] == "see ./somefile.txt "
    assert _ptk_texts("see @some") == ["./somefile.txt "]

    # Flag-value branch is terminal on both: no task history means no
    # candidates, and neither frontend falls back to the option menu.
    assert completer.build_menu("/resume --task ") == []
    assert _ptk_texts("/resume --task ") == []

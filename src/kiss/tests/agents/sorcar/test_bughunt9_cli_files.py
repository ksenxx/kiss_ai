# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 9: CLI input-file bugs (real end-to-end, no mocks).

Covers two confirmed issues:

1. ``cli_repl._FLAG_VALUE_RE`` was not boundary-anchored, so a token
   that merely *ends* with ``--task`` / ``--model`` (``x--task``,
   ``----task``) wrongly switched the completer into flag-VALUE
   completion, hijacking the menu that should have fallen through to
   option / predictive completion.

2. ``cli_helpers._print_result`` printed the literal string ``"None"``
   when the agent's result YAML carried a null ``summary``
   (``summary:\\n`` parses to ``{"summary": None}``).
"""

from __future__ import annotations

import pytest

from kiss.agents.sorcar.cli_helpers import _print_result
from kiss.agents.sorcar.cli_repl import CliCompleter


@pytest.fixture()
def completer(tmp_path: object) -> CliCompleter:
    """A real CliCompleter rooted at an empty temp work dir."""
    return CliCompleter(str(tmp_path))


class TestFlagValueBoundary:
    """``--task`` / ``--model`` must be whole tokens to complete values."""

    def test_embedded_task_flag_does_not_trigger_value_completion(
        self, completer: CliCompleter,
    ) -> None:
        """``x--task`` is one token, not the ``--task`` flag."""
        assert completer._flag_value_matches("/resume x--task ab") is None

    def test_extra_dashes_do_not_trigger_value_completion(
        self, completer: CliCompleter,
    ) -> None:
        """``----task`` is not the ``--task`` flag."""
        assert completer._flag_value_matches("/resume ----task ab") is None

    def test_embedded_model_flag_does_not_trigger_value_completion(
        self, completer: CliCompleter,
    ) -> None:
        """``x--model`` is one token, not the ``--model`` flag."""
        assert completer._flag_value_matches("/cmd x--model gp") is None

    def test_real_task_flag_still_completes_values(
        self, completer: CliCompleter,
    ) -> None:
        """A genuine trailing ``--task `` still enters value completion."""
        matches = completer._flag_value_matches("/resume --task ")
        assert matches is not None

    def test_real_model_flag_still_completes_values(
        self, completer: CliCompleter,
    ) -> None:
        """A genuine trailing ``--model `` still pops model names."""
        matches = completer._flag_value_matches("/cmd --model ")
        assert matches is not None
        assert len(matches) > 0

    def test_build_menu_falls_through_for_embedded_flag(
        self, completer: CliCompleter,
    ) -> None:
        """build_menu must not offer task-id values for ``x--task``.

        With the boundary fix the line falls through to the argument
        -option branch (which offers ``/resume``'s real flags) instead
        of the value branch; no candidate may be a bare task id row.
        """
        menu = completer.build_menu("/resume x--task ")
        for _replacement, display in menu:
            assert display.split(" ")[0] in ("--task", "--limit")


class TestPrintResultNullSummary:
    """``_print_result`` must not print the literal string ``None``."""

    def test_null_summary_prints_empty(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``summary:`` (YAML null) prints an empty line, not ``None``."""
        _print_result("success: true\nsummary:\n")
        out = capsys.readouterr().out
        assert "None" not in out
        assert out.strip() == ""

    def test_string_summary_still_printed(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A normal string summary is printed unchanged."""
        _print_result("success: true\nsummary: all done\n")
        assert capsys.readouterr().out.strip() == "all done"

    def test_non_yaml_result_printed_verbatim(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-YAML results fall back to verbatim printing."""
        _print_result("plain text result: [unbalanced")
        assert "plain text result" in capsys.readouterr().out

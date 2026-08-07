# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Integration tests for bug fixes, redundancy acknowledgement, and
consistency improvements in ``kiss.server``.

These tests assert the FIXED behavior — each test confirms the bug is
resolved or the inconsistency is eliminated.

Bugs fixed
----------
B6: ``model_vendor`` now correctly classifies ``openai/``-prefixed
    models (e.g. ``openai/gpt-4o``) as ``"OpenAI"``.
B8: ``_run_task`` now broadcasts ``status: running: False`` INSIDE
    the ``_state_lock`` critical section (A2 fix).

(B5/B7 covered the merge-data directory cleanup of the interactive
diff/merge review workflow; that workflow and its on-disk merge
artifacts were removed from the server, so those tests are gone.)

Bugs acknowledged (not fixed — intentional)
--------------------------------------------
B4: ``_active_file_identifier_matches`` ranks the LONGEST matching
    identifier first. This is intentional behavior per user feedback.

Redundancies acknowledged
-------------------------
R2: ``clip_autocomplete_suggestion`` applied to local completions is
    a no-op for clean identifier suffixes. Kept for safety against
    unexpected LLM output.

Inconsistencies fixed
---------------------
I2: ``tab_id`` parameter types are now consistently ``str = ""``.
I3: ``_broadcast_worktree_done`` now always includes ``tabId``.
"""

from __future__ import annotations

import inspect
import typing
import unittest

from kiss.server.helpers import (
    clip_autocomplete_suggestion,
    model_vendor,
)
from kiss.server.merge_flow import _MergeFlowMixin
from kiss.server.server import VSCodeServer
from kiss.server.task_runner import _TaskRunnerMixin


class TestActiveFileMatchesLongestFirst(unittest.TestCase):
    """B4: ``_active_file_identifier_matches`` ranks the longest
    matching identifier first.  This is INTENTIONAL behavior — test
    confirms it still works.
    """

    def setUp(self) -> None:
        self.server = VSCodeServer()
        self.content = (
            "server = start_server()\n"
            "server_config = load_config()\n"
            "server_manager = create_manager()\n"
        )

    def test_returns_longest_match_first(self) -> None:
        """The top match is 'server_manager' (longest) — intentional."""
        matches = self.server._active_file_identifier_matches(
            "use serv", snapshot_content=self.content,
        )
        assert matches and matches[0] == "server_manager", (
            f"B4 intentional: expected longest match 'server_manager' "
            f"first, got {matches!r}"
        )



class TestModelVendorOpenAIClassification(unittest.TestCase):
    """B6 fix: ``model_vendor("openai/gpt-4o")`` now correctly
    returns ``("OpenAI", 1)``.
    """

    def test_openai_gpt4o_classified_as_openai(self) -> None:
        vendor, order = model_vendor("openai/gpt-4o")
        assert vendor == "OpenAI" and order == 1, (
            f"B6 fix: openai/gpt-4o should be OpenAI, got ({vendor}, {order})"
        )

    def test_openai_o1_classified_as_openai(self) -> None:
        vendor, order = model_vendor("openai/o1-preview")
        assert vendor == "OpenAI" and order == 1, (
            f"B6 fix: openai/o1-preview should be OpenAI, got ({vendor}, {order})"
        )

    def test_bare_gpt4o_still_classified_correctly(self) -> None:
        """The bare name without ``openai/`` prefix still works."""
        vendor, order = model_vendor("gpt-4o")
        assert vendor == "OpenAI" and order == 1, (
            f"Bare gpt-4o should be OpenAI, got ({vendor}, {order})"
        )



class TestClipAutocompleteSuggestionRedundant(unittest.TestCase):
    """R2 redundancy: ``clip_autocomplete_suggestion`` is applied to
    identifier suffixes derived from
    ``_active_file_identifier_matches`` but all its transformations
    are no-ops for clean identifier suffixes.
    Kept for safety — these tests document the behavior.
    """

    def test_no_op_for_plain_suffix(self) -> None:
        result = clip_autocomplete_suggestion("serv", "er_manager")
        assert result == "er_manager", f"Expected identity, got {result!r}"

    def test_no_op_for_dotted_suffix(self) -> None:
        result = clip_autocomplete_suggestion("se", "lf.setup")
        assert result == "lf.setup", f"Expected identity, got {result!r}"

    def test_no_op_for_underscore_suffix(self) -> None:
        result = clip_autocomplete_suggestion("server", "_config")
        assert result == "_config", f"Expected identity, got {result!r}"



class TestTabIdTypeConsistency(unittest.TestCase):
    """I2 fix: ``tab_id`` parameter types and defaults are now
    consistently ``str = ""`` across all methods.
    """

    def test_all_use_str_default_empty(self) -> None:
        """All tab_id params with defaults use ``str`` type and ``""`` default."""
        methods_with_defaults: dict[str, typing.Any] = {}
        for name, method in [
            ("_handle_worktree_action", _MergeFlowMixin._handle_worktree_action),
            ("_autocommit_changes", _MergeFlowMixin._autocommit_changes),
            ("_stop_task", _TaskRunnerMixin._stop_task),
        ]:
            sig = inspect.signature(method)  # type: ignore[arg-type]
            for pname, param in sig.parameters.items():
                if "tab" in pname.lower() and param.default is not inspect.Parameter.empty:
                    methods_with_defaults[name] = param.default

        defaults = set(methods_with_defaults.values())
        assert defaults == {""}, (
            f"I2 fix: all tab_id defaults should be '', got: {methods_with_defaults}"
        )

    def test_consistent_type_annotations(self) -> None:
        """All tab_id params with defaults annotate as ``str``."""
        annotations: dict[str, str] = {}
        for name, method in [
            ("_handle_worktree_action", _MergeFlowMixin._handle_worktree_action),
            ("_stop_task", _TaskRunnerMixin._stop_task),
        ]:
            sig = inspect.signature(method)  # type: ignore[arg-type]
            for pname, param in sig.parameters.items():
                if "tab" in pname.lower() and param.default is not inspect.Parameter.empty:
                    annotations[name] = str(param.annotation)

        for name, ann in annotations.items():
            assert "None" not in ann, (
                f"I2 fix: {name} tab_id annotation should be str, got: {ann}"
            )




if __name__ == "__main__":
    unittest.main()

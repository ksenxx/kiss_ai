# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproduction tests for the sorcar bug-hunt defects.

Each test class reproduces one real defect found in
``src/kiss/agents/sorcar/`` (see ``tmp/findings-sorcar.md``):

* D1 — ``code_graph._extract_file`` misattributed the callee of a
  CHAINED call: for ``Application().start()`` the callee text
  ``Application().start`` was truncated at the FIRST ``(`` before the
  last dotted segment was taken, so the call was recorded as
  ``main -> Application`` and the ``main -> start`` edge never
  existed — across every supported language (same shape in JS:
  ``getApp().f()``).
* D2 — ``code_graph._grep_pattern`` ignored the POSIX ``--``
  end-of-options marker (``grep -- -pat file`` returned ``file``) and
  treated the argument AFTER a ``-f patterns.txt`` pattern-file flag
  as the search pattern, so ``grep_hint`` could intercept (and
  suppress) a legitimate grep based on its TARGET path.
* D3 — ``persistence._is_failed_result`` did not classify the
  ``"Task interrupted"`` result (persisted by
  ``ChatSorcarAgent.run``'s ``BaseException`` handler for user Stop /
  KeyboardInterrupt on CLI, sub-agent, and channel-agent runs) as a
  failure, while it did classify the sibling ``"Task interrupted by
  server restart/shutdown"`` — so the history sidebar
  (``server.py`` ``failed=_is_failed_result(result)``) showed
  interrupted tasks as successful.
* D4 — ``skills._FRONTMATTER_RE`` did not match an EMPTY frontmatter
  block (``---\\n---\\n``), so the two literal ``---`` lines leaked
  into the skill/command body and the derived description.

No mocks, patches, or fakes: real tree-sitter parsing over real files
on disk, and the real production helpers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.agents.sorcar.persistence import _is_failed_result
from kiss.agents.sorcar.skills import parse_frontmatter

pytest.importorskip("tree_sitter_language_pack")

from kiss.agents.sorcar.code_graph import (  # noqa: E402
    _grep_pattern,
    build_graph,
    grep_hint,
)

# ---------------------------------------------------------------------------
# D1 — chained-call callee misattribution
# ---------------------------------------------------------------------------


def _call_edges(graph) -> set[tuple[str, str]]:
    """Return the graph's ``calls`` edges as (caller_label, callee_label)."""
    return {
        (graph.nodes[e["source"]]["label"], graph.nodes[e["target"]]["label"])
        for e in graph.edges
        if e["relation"] == "calls"
    }


class TestChainedCallCallee:
    """``x().y()`` must record a call to ``y``, not a second call to ``x``."""

    def test_python_chained_method_call(self, tmp_path: Path) -> None:
        (tmp_path / "m.py").write_text(
            "class Application:\n"
            "    def start(self):\n"
            "        pass\n"
            "\n"
            "def main():\n"
            "    Application().start()\n"
        )
        graph = build_graph(str(tmp_path), incremental=False)
        edges = _call_edges(graph)
        # The constructor call is still recorded ...
        assert ("main", "Application") in edges
        # ... and the chained method call resolves to the METHOD.
        assert ("main", "start") in edges

    def test_javascript_chained_call(self, tmp_path: Path) -> None:
        (tmp_path / "j.js").write_text(
            "function f() {}\n"
            "function g2() { getApp().f(); }\n"
        )
        graph = build_graph(str(tmp_path), incremental=False)
        assert ("g2", "f") in _call_edges(graph)

    def test_plain_attribute_call_still_resolves(self, tmp_path: Path) -> None:
        """Regression guard: ``w.draw()`` / ``pkg.mod.fn()`` keep working."""
        (tmp_path / "m.py").write_text(
            "class Widget:\n"
            "    def draw(self):\n"
            "        pass\n"
            "\n"
            "def main():\n"
            "    w = Widget()\n"
            "    w.draw()\n"
        )
        graph = build_graph(str(tmp_path), incremental=False)
        assert ("main", "draw") in _call_edges(graph)


# ---------------------------------------------------------------------------
# D2 — _grep_pattern: `--` marker and `-f` pattern-file
# ---------------------------------------------------------------------------


class TestGrepPatternParsing:
    def test_double_dash_marks_next_token_as_pattern(self) -> None:
        # POSIX: everything after ``--`` is an operand; the first one
        # is the pattern.  The old parser skipped ``-literal`` as a
        # flag and returned ``file``.
        assert _grep_pattern("grep -- -literal file") == "-literal"
        assert _grep_pattern("grep -rn -- TODO src/") == "TODO"
        assert _grep_pattern("grep --") is None

    def test_pattern_file_flag_means_no_inline_pattern(self) -> None:
        # With ``-f``/``--file`` the patterns come from a file; the
        # remaining operands are search TARGETS, never the pattern.
        assert _grep_pattern("grep -f pats.txt src/") is None
        assert _grep_pattern("rg --file pats.txt src/") is None
        assert _grep_pattern("rg --file=pats.txt src/") is None

    def test_existing_shapes_unchanged(self) -> None:
        assert _grep_pattern("grep -e foo file") == "foo"
        assert _grep_pattern("grep --regexp=foo file") == "foo"
        assert _grep_pattern("grep -A3 foo file") == "foo"
        assert _grep_pattern("grep -A 3 foo file") == "foo"
        assert _grep_pattern("rg -i --glob '*.py' MyClass") == "MyClass"
        assert _grep_pattern("ls -l") is None

    def test_grep_hint_not_triggered_by_search_target(
        self, tmp_path: Path
    ) -> None:
        """A ``-f pattern-file`` grep must never be answered by the graph.

        The graph knows ``util_fn``; the grep searches patterns from a
        file INSIDE a path that happens to be named ``util_fn``.  The
        old parser took ``util_fn`` as the pattern and suppressed the
        real grep with a graph answer.
        """
        (tmp_path / "helpers.py").write_text("def util_fn():\n    return 1\n")
        build_graph(str(tmp_path), incremental=False)
        assert grep_hint("grep util_fn helpers.py", str(tmp_path)) is not None
        assert grep_hint("grep -f pats.txt util_fn", str(tmp_path)) is None


# ---------------------------------------------------------------------------
# D3 — "Task interrupted" must classify as a failed result
# ---------------------------------------------------------------------------


class TestInterruptedResultClassification:
    def test_task_interrupted_is_failed(self) -> None:
        # Exact string persisted by ``ChatSorcarAgent.run``'s
        # ``except BaseException`` handler (user Stop / daemon
        # shutdown reaching a CLI, sub-agent, or channel-agent run)
        # and by the channel agents' KeyboardInterrupt paths.
        assert _is_failed_result("Task interrupted")

    def test_shutdown_variant_still_failed(self) -> None:
        assert _is_failed_result("Task interrupted by server restart/shutdown")

    def test_other_markers_unchanged(self) -> None:
        assert _is_failed_result("Task failed")
        assert _is_failed_result("Task failed with error: boom")
        assert _is_failed_result("Agent Failed Abruptly")
        assert _is_failed_result("Task terminated unexpectedly (process killed)")
        assert _is_failed_result("Task stopped by user")
        assert not _is_failed_result("All done; wrote the report.")
        assert not _is_failed_result("")


# ---------------------------------------------------------------------------
# D4 — empty frontmatter block must be stripped
# ---------------------------------------------------------------------------


def _parse(path: Path) -> tuple[dict[str, object], str]:
    """Parse frontmatter, asserting the file was readable."""
    parsed = parse_frontmatter(path)
    assert parsed is not None
    return parsed


class TestEmptyFrontmatter:
    def test_empty_frontmatter_is_stripped(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_text("---\n---\nBody only\n", encoding="utf-8")
        meta, body = _parse(p)
        assert meta == {}
        assert body == "Body only\n"

    def test_normal_frontmatter_still_parses(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_text(
            "---\ndescription: does things\n---\nBody\n", encoding="utf-8",
        )
        meta, body = _parse(p)
        assert meta == {"description": "does things"}
        assert body == "Body\n"

    def test_closing_marker_at_eof_without_newline(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_text("---\na: b\n---", encoding="utf-8")
        meta, body = _parse(p)
        assert meta == {"a": "b"}
        assert body == ""

    def test_crlf_line_endings(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_bytes(b"---\r\na: b\r\n---\r\nBody\r\n")
        meta, body = _parse(p)
        assert meta == {"a": "b"}
        # ``read_text`` applies universal-newline translation, so the
        # body comes back with ``\n`` endings.
        assert body == "Body\n"

    def test_value_ending_in_dashes_is_not_a_closing_marker(
        self, tmp_path: Path
    ) -> None:
        p = tmp_path / "cmd.md"
        p.write_text("---\na: b---\nc: d\n---\nBody\n", encoding="utf-8")
        meta, body = _parse(p)
        assert meta == {"a": "b---", "c": "d"}
        assert body == "Body\n"

    def test_no_frontmatter_returns_whole_text(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_text("Just a body\n", encoding="utf-8")
        meta, body = _parse(p)
        assert meta == {}
        assert body == "Just a body\n"

    def test_bom_still_handled(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_bytes("\ufeff---\ndescription: bom\n---\nB\n".encode())
        meta, body = _parse(p)
        assert meta == {"description": "bom"}
        assert body == "B\n"

    def test_unterminated_frontmatter_is_body(self, tmp_path: Path) -> None:
        p = tmp_path / "cmd.md"
        p.write_text("---\na: b\n", encoding="utf-8")
        meta, body = _parse(p)
        assert meta == {}
        assert body == "---\na: b\n"

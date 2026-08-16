# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end reproduction tests for the sorcar bug-hunt defects.

Each test class reproduces one real defect found in
``src/kiss/agents/sorcar/`` (see ``tmp/findings-sorcar.md``):

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

No mocks, patches, or fakes: the real production helpers over real
files on disk.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.persistence import _is_failed_result
from kiss.agents.sorcar.skills import parse_frontmatter


class TestInterruptedResultClassification:
    def test_task_interrupted_is_failed(self) -> None:
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

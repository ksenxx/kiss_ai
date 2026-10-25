# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tool outputs are capped at the source and the rest spilled to ``tmp/tool-out/``.

``Bash`` keeps the default budget (8,000 characters) of its output in the
result, writes the full output under the work dir and names the file;
the model may raise the budget up to the configured ceiling, not
beyond.  A whole-file ``Read`` keeps the same budget and says where to
continue; ``memory_list`` cuts a long listing with a note.
"""

from __future__ import annotations

from pathlib import Path

from kiss.agents.sorcar.useful_tools import SPILL_DIR, UsefulTools, _lines_within_budget
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools


def test_bash_keeps_the_budget_and_spills_the_full_output(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    result = tools.Bash("seq 1 20000", "count")
    assert len(result) <= DEFAULT_CONFIG.tool_output_default_chars
    assert result.startswith("1\n2\n3\n")
    assert result.rstrip().endswith("20000")
    tag = tools._spill_tag
    spill = tmp_path / SPILL_DIR / f"{tag}-1-bash.txt"
    assert f"full output ({spill.stat().st_size} chars) in tmp/tool-out/{tag}-1-bash.txt" in result
    assert spill.read_text().splitlines()[-1] == "20000"
    # A second oversized output gets its own file; a small one none.
    tools.Bash("seq 1 20000", "count again")
    assert (tmp_path / SPILL_DIR / f"{tag}-2-bash.txt").exists()
    assert tools.Bash("echo small", "small") == "small\n"
    assert not (tmp_path / SPILL_DIR / f"{tag}-3-bash.txt").exists()
    # Another instance sharing the work dir (a sub-agent) never overwrites these files.
    other = UsefulTools(work_dir=str(tmp_path))
    other.Bash("seq 1 20000", "count")
    assert other._spill_tag != tag
    assert sorted(p.name for p in (tmp_path / SPILL_DIR).iterdir()) == sorted(
        [f"{tag}-1-bash.txt", f"{tag}-2-bash.txt", f"{other._spill_tag}-1-bash.txt"]
    )


def test_bash_budget_can_be_raised_only_up_to_the_ceiling(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    whole = tools.Bash("seq 1 5000", "count", max_output_chars=30_000)
    assert "[truncated" not in whole and whole.rstrip().endswith("5000")
    capped = tools.Bash("seq 1 200000", "count", max_output_chars=10_000_000)
    assert len(capped) <= DEFAULT_CONFIG.tool_output_max_chars
    assert "[truncated" in capped
    failed = tools.Bash("seq 1 20000; exit 3", "fail")
    assert failed.startswith("Error (exit code 3):\n1\n")
    assert "in tmp/tool-out/" in failed


def test_bash_without_a_work_dir_truncates_without_a_spill(tmp_path: Path) -> None:
    tools = UsefulTools()
    result = tools.Bash("seq 1 20000", "count")
    assert "[truncated" in result and "tool-out" not in result


def test_whole_file_read_keeps_the_budget_and_says_where_to_continue(tmp_path: Path) -> None:
    big = tmp_path / "big.txt"
    lines = [f"line {i} " + "x" * 60 + "\n" for i in range(1, 1001)]
    big.write_text("".join(lines))
    tools = UsefulTools(work_dir=str(tmp_path))
    shown = tools.Read(str(big))
    assert len(shown) <= DEFAULT_CONFIG.tool_output_default_chars + 120
    kept = shown.split("\n[truncated:")[0].count("\n")
    assert 100 < kept < 130
    assert f"more lines; Read(file_path, start_line={kept + 1}, max_lines=N) continues]" in shown
    # An explicit range is honoured whole.
    ranged = tools.Read(str(big), start_line=1, max_lines=500)
    assert ranged.count("\n") >= 500
    assert tools._is_read(big.resolve())


def test_lines_within_budget_clips_a_lone_over_long_first_line() -> None:
    """A first line longer than the budget is clipped with a marker, never shown whole."""
    clipped = _lines_within_budget(["a" * 50, "b"], 10)
    assert clipped == ["a" * 10 + "\n[line clipped to 10 chars; grep or cut for the rest]\n"]
    assert _lines_within_budget(["ab", "cd", "ef"], 4) == ["ab", "cd"]
    assert _lines_within_budget([], 4) == []


def test_memory_list_cuts_a_long_listing_with_a_note(tmp_path: Path) -> None:
    root = tmp_path / "memories"
    root.mkdir()
    for i in range(300):
        (root / f"page-{i:03d}.md").write_text(
            f"---\ntitle: Page {i}\nsummary: {'s' * 60}\n---\nbody\n", encoding="utf-8",
        )
    tools = MemoryTools(root, embed=hashed_embedding)
    listing = tools.memory_list()
    assert len(listing) <= DEFAULT_CONFIG.tool_output_default_chars + 200
    assert "more pages not shown: pass memory=<name>" in listing
    shown = listing.split("\n... [")[0].count("\n") + 1
    assert f"... [{300 - shown} more pages not shown" in listing
    raised = tools.memory_list(max_output_chars=50_000)
    assert "more pages not shown" not in raised and raised.count("\n") == 299

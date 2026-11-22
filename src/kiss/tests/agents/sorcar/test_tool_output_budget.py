# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tool outputs are capped at the source and the rest spilled to ``tool-out/`` of the scratch dir.

``Bash`` keeps the default budget (8,000 characters) of its output in the
result, writes the full output under the agent's scratch dir (the agents'
temp directory itself for tools built without one, never the work dir) and
names the file by its absolute path;
the model may raise the budget up to the configured ceiling, not
beyond.  A whole-file ``Read`` keeps the same budget and says where to
continue; ``memory_list`` cuts a long listing with a note.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from kiss.agents.seas.coding.gate_readers import bash_exit_code
from kiss.agents.sorcar.useful_tools import SPILL_DIR, UsefulTools, _lines_within_budget
from kiss.core.config import DEFAULT_CONFIG
from kiss.core.kiss_tmp import kiss_tmp_dir
from kiss.core.memoryfield.index import hashed_embedding
from kiss.core.memoryfield.tools import MemoryTools
from kiss.core.secret_redaction import add_secrets, redact_secrets


def test_bash_keeps_the_budget_and_spills_the_full_output(tmp_path: Path) -> None:
    work, scratch = tmp_path / "work", tmp_path / "scratch"
    work.mkdir()
    tools = UsefulTools(work_dir=str(work), scratch_dir=str(scratch))
    result = tools.Bash("seq 1 20000", "count")
    assert len(result) <= DEFAULT_CONFIG.tool_output_default_chars
    assert result.startswith("1\n2\n3\n")
    assert result.rstrip().endswith("20000")
    tag = tools._spill_tag
    spill = scratch / SPILL_DIR / f"{tag}-1-bash.txt"
    # The file is in the scratch dir and named by its absolute path.
    assert f"full output ({spill.stat().st_size} chars) in {spill.as_posix()}]" in result
    assert spill.read_text().splitlines()[-1] == "20000"
    # A second oversized output gets its own file; a small one none.
    tools.Bash("seq 1 20000", "count again")
    assert (scratch / SPILL_DIR / f"{tag}-2-bash.txt").exists()
    assert tools.Bash("echo small", "small") == "small\n"
    assert not (scratch / SPILL_DIR / f"{tag}-3-bash.txt").exists()
    # Another instance sharing the scratch dir never overwrites these files.
    other = UsefulTools(work_dir=str(work), scratch_dir=str(scratch))
    other.Bash("seq 1 20000", "count")
    assert other._spill_tag != tag
    assert sorted(p.name for p in (scratch / SPILL_DIR).iterdir()) == sorted(
        [f"{tag}-1-bash.txt", f"{tag}-2-bash.txt", f"{other._spill_tag}-1-bash.txt"]
    )
    # Nothing is left in the work dir: no ``tmp`` folder, no spill.
    assert list(work.iterdir()) == []


def test_bash_budget_can_be_raised_only_up_to_the_ceiling(tmp_path: Path) -> None:
    tools = UsefulTools(work_dir=str(tmp_path))
    whole = tools.Bash("seq 1 5000", "count", max_output_chars=30_000)
    assert "[truncated" not in whole and whole.rstrip().endswith("5000")
    capped = tools.Bash("seq 1 200000", "count", max_output_chars=10_000_000)
    assert len(capped) <= DEFAULT_CONFIG.tool_output_max_chars
    assert "[truncated" in capped
    failed = tools.Bash("seq 1 20000; exit 3", "fail")
    assert failed.startswith("Error (exit code 3):\n1\n")
    # Tools built without a scratch dir spill into the agents' temp directory itself.
    spilled = re.search(r"; full output \(\d+ chars\) in (\S+)\] \.\.\.\n", failed)
    assert spilled and Path(spilled[1]).parent == kiss_tmp_dir() / SPILL_DIR, failed
    assert Path(spilled[1]).read_text().splitlines()[-1] == "20000"
    assert list(tmp_path.iterdir()) == []
    # Only the output is cut: the status is whole under any budget (cross finding of B2 and
    # B4, 2026-11-12: cut with the output, a failed command read as one that passed).
    for budget in (1, 20, 60):
        cut = tools.Bash("seq 1 200; exit 7", "fail", max_output_chars=budget)
        assert cut.startswith("Error (exit code 7):\n1") and bash_exit_code(cut) == "7", cut
        assert len(cut) == len("Error (exit code 7):\n") + budget
    late = tools.Bash("seq 1 200; sleep 5", "slow", timeout_seconds=1, max_output_chars=1)
    assert late == "Error: Command execution timeout after 1s. Output before the timeout:\n1"
    assert bash_exit_code(late) == "timeout"


def test_bash_without_a_work_dir_spills_all_the_same(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The spill is in the scratch dir, so it needs no work dir (it was skipped without one)."""
    monkeypatch.chdir(tmp_path)
    tools = UsefulTools()
    result = tools.Bash("seq 1 20000", "count")
    spill = kiss_tmp_dir() / SPILL_DIR / f"{tools._spill_tag}-1-bash.txt"
    assert len(result) <= DEFAULT_CONFIG.tool_output_default_chars
    assert f"full output ({spill.stat().st_size} chars) in {spill.as_posix()}]" in result
    assert spill.read_text().splitlines() == [str(n) for n in range(1, 20001)]
    # Where the command ran gets no file of the tools.
    assert list(tmp_path.iterdir()) == []


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


def test_a_line_clipped_to_the_budget_shows_no_part_of_a_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cut of an over-long line comes after its secrets are replaced, so it parts none.

    S16/F6: the line was cut first, and the start of a secret left
    before the marker was no secret to the pass over the whole result.
    The second read has a budget past the piece a long line is read in
    (65,536 characters), so the cut comes in pieces joined.  The third
    lists the line in an outline: within the budget its entry is whole,
    the secret in it for the pass over the whole result to replace;
    past the budget (the fourth) the entry is cut like the line.
    """
    secret = "S16-read-credential-0123456789abcdef"
    add_secrets({"S16_READ_KEY": secret})
    tools = UsefulTools(work_dir=str(tmp_path))
    budget = DEFAULT_CONFIG.tool_output_default_chars
    note = "\n[line clipped to {} chars; grep or cut for the rest]\n"
    path = tmp_path / "one-line.json"
    path.write_text("x" * (budget - 20) + secret + "y" * 100)
    shown = tools.Read(str(path))
    assert shown == "x" * (budget - 20) + "S16_READ_KEY" + "y" * 8 + note.format(budget)
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_default_chars", 70_000)
    path.write_text("x" * 69_990 + secret + "y" * 100)
    shown = tools.Read(str(path))
    assert shown.startswith("x" * 69_990) and shown.endswith(note.format(70_000))
    assert "S16-" not in shown and len(shown) <= 70_000 + len(note.format(70_000))
    monkeypatch.setattr(DEFAULT_CONFIG, "read_outline_lines", 5)
    start = "def long(): " + "x" * 65_500
    path = tmp_path / "long.py"
    lines = start + secret + "y" * 100 + "\n" + "def short(): pass\n" * 5 + "def last(): S\r"
    path.write_bytes(lines.encode())
    shown = tools.Read(str(path))
    assert f"\n1: {start}{secret}{'y' * 100}\n2: def short(): pass\n" in shown
    assert "\n7: def last(): S\n" in shown
    assert f"\n1: {start}S16_READ_KEY{'y' * 100}\n2: " in redact_secrets(shown)
    monkeypatch.setattr(DEFAULT_CONFIG, "tool_output_default_chars", 65_530)
    shown = tools.Read(str(path), force=True)
    assert f"\n1: {start}S16_READ_KEY{'y' * 6} [line clipped to 65530 chars]\n2: " in shown


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

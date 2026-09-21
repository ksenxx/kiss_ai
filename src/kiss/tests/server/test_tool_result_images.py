# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests: images a tool call generated are embedded on its
``tool_result`` event so the chat webview can render them inline in the
corresponding event panel.

``JsonPrinter._collect_result_images`` scans the tool's return text
(and the call's ``file_path`` / ``path`` argument) for image-file
paths, keeps files created or modified while the tool ran, and embeds
up to three of them as base64 payloads (``images`` on the event).
These tests drive the real printer through ``print(type="tool_call")``
/ ``print(type="tool_result")`` with real files on disk — no mocks.
"""

import base64
import os
import time
from typing import TYPE_CHECKING, cast

from kiss.server import agent_state

if TYPE_CHECKING:
    from kiss.agents.sorcar.worktree_sorcar_agent import WorktreeSorcarAgent
from kiss.server.json_printer import (
    _MAX_RESULT_IMAGE_BYTES,
    JsonPrinter,
    _extract_image_path_candidates,
)
from kiss.tests.conftest import is_root, posix_only

# A real, valid 1x1 transparent PNG (67 bytes).
_PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
    "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
)

_TASK_COUNTER = 0


def _new_printer() -> JsonPrinter:
    """Return a recording JsonPrinter bound to a fresh task id."""
    global _TASK_COUNTER
    _TASK_COUNTER += 1
    printer = JsonPrinter()
    printer._thread_local.task_id = f"test-tool-result-images-{_TASK_COUNTER}"
    printer.start_recording()
    return printer


def _tool_results(printer: JsonPrinter) -> list[dict]:
    return [e for e in printer.stop_recording() if e["type"] == "tool_result"]


def _run_tool(
    printer: JsonPrinter,
    name: str,
    tool_input: dict,
    result: str,
    is_error: bool = False,
) -> dict:
    """Drive a full tool_call/tool_result pair and return the result event."""
    printer.print(name, type="tool_call", tool_input=tool_input)
    printer.print(
        result,
        type="tool_result",
        tool_name=name,
        tool_input=tool_input,
        is_error=is_error,
    )
    events = _tool_results(printer)
    assert len(events) == 1
    return events[0]


class TestResultImagesEmbedded:
    """Fresh images named by a tool result are embedded on its event."""

    def test_screenshot_result_embeds_png(self, tmp_path):
        shot = tmp_path / "screenshot.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "screenshot",
            {"file_path": str(shot)},
            f"Screenshot saved to {shot}",
        )
        assert "images" in ev
        assert len(ev["images"]) == 1
        img = ev["images"][0]
        assert img["mime"] == "image/png"
        assert img["path"] == str(shot)
        assert base64.b64decode(img["b64"]) == _PNG_1PX

    def test_path_mentioned_only_in_result_text(self, tmp_path):
        plot = tmp_path / "plot.jpg"
        plot.write_bytes(b"\xff\xd8\xff\xdbfakejpegbytes")
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Bash",
            {"command": "python plot.py", "description": "plot"},
            f"Saved figure to {plot}\n",
        )
        assert [i["mime"] for i in ev["images"]] == ["image/jpeg"]
        assert ev["images"][0]["path"] == str(plot)

    def test_same_file_in_input_and_text_embedded_once(self, tmp_path):
        shot = tmp_path / "dup.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "screenshot",
            {"file_path": str(shot)},
            f"Screenshot saved to {shot} (also at {shot})",
        )
        assert len(ev["images"]) == 1

    def test_at_most_three_images_embedded(self, tmp_path):
        paths = []
        for i in range(4):
            p = tmp_path / f"img{i}.png"
            p.write_bytes(_PNG_1PX)
            paths.append(p)
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Bash",
            {"command": "gen"},
            " ".join(str(p) for p in paths),
        )
        assert [i["path"] for i in ev["images"]] == [str(p) for p in paths[:3]]

    def test_relative_path_resolved_against_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "chart.webp").write_bytes(b"RIFFxxxxWEBPfake")
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Bash",
            {"command": "gen"},
            "Wrote out/chart.webp\n",
        )
        assert ev["images"][0]["path"] == "out/chart.webp"
        assert ev["images"][0]["mime"] == "image/webp"

    def test_relative_path_resolved_against_registered_work_dir(self, tmp_path):
        """The agent registry's duck-typed ``work_dir`` wins over cwd.

        Uses the real registry (``agent_state.register``) with the
        minimal duck-typed agent surface the printer is documented to
        read (``work_dir``; see ``JsonPrinter._task_work_dir``).
        """
        (tmp_path / "shot.png").write_bytes(_PNG_1PX)

        class _WorkDirOnlyAgent:
            work_dir = str(tmp_path)

        printer = _new_printer()
        task_id = printer._task_key()
        # The registry annotates ``agent`` as WorktreeSorcarAgent, but
        # the printer bridge reads it duck-typed (see _task_work_dir);
        # cast keeps this end-to-end test free of a full agent stack.
        state = agent_state.AgentState(
            task_id, agent=cast("WorktreeSorcarAgent", _WorkDirOnlyAgent())
        )
        agent_state.register(state)
        try:
            ev = _run_tool(
                printer,
                "screenshot",
                {"file_path": "shot.png"},
                "Screenshot saved to shot.png",
            )
        finally:
            agent_state.unregister(task_id, state)
        assert ev["images"][0]["path"] == "shot.png"
        assert base64.b64decode(ev["images"][0]["b64"]) == _PNG_1PX

    def test_quoted_path_with_spaces_embedded(self, tmp_path):
        spaced_dir = tmp_path / "A User"
        spaced_dir.mkdir()
        shot = spaced_dir / "shot.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "screenshot",
            {},
            f'Screenshot saved to "{shot}"',
        )
        assert ev["images"][0]["path"] == str(shot)
        assert base64.b64decode(ev["images"][0]["b64"]) == _PNG_1PX

    def test_sentence_ending_period_after_path(self, tmp_path):
        shot = tmp_path / "output.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        ev = _run_tool(printer, "Bash", {"command": "gen"}, f"Saved {shot}.\n")
        assert ev["images"][0]["path"] == str(shot)

    def test_fresh_svg_written_by_write_tool(self, tmp_path):
        svg = tmp_path / "diagram.svg"
        svg.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Write",
            {"file_path": str(svg), "content": "<svg/>"},
            f"Wrote {svg}",
        )
        assert ev["images"][0]["mime"] == "image/svg+xml"


class TestResultImagesFiltered:
    """Stale, oversized, missing, and error-path images are NOT embedded."""

    def test_old_image_merely_mentioned_is_skipped(self, tmp_path):
        old = tmp_path / "logo.png"
        old.write_bytes(_PNG_1PX)
        stale = time.time() - 3600
        os.utime(old, (stale, stale))
        printer = _new_printer()
        ev = _run_tool(printer, "Bash", {"command": "ls"}, f"{old}\n")
        assert "images" not in ev

    def test_error_result_never_embeds(self, tmp_path):
        shot = tmp_path / "err.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "screenshot",
            {"file_path": str(shot)},
            f"Failed after saving {shot}",
            is_error=True,
        )
        assert ev["is_error"] is True
        assert "images" not in ev

    def test_missing_and_empty_and_oversized_files_skipped(self, tmp_path):
        empty = tmp_path / "empty.png"
        empty.write_bytes(b"")
        big = tmp_path / "big.png"
        big.write_bytes(b"x" * (_MAX_RESULT_IMAGE_BYTES + 1))
        missing = tmp_path / "missing.png"
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Bash",
            {"command": "gen"},
            f"{empty} {big} {missing}",
        )
        assert "images" not in ev

    def test_non_image_path_argument_ignored(self, tmp_path):
        py = tmp_path / "script.py"
        py.write_text("print('hi')\n")
        printer = _new_printer()
        ev = _run_tool(
            printer,
            "Write",
            {"file_path": str(py), "content": "print('hi')\n"},
            f"Wrote {py}",
        )
        assert "images" not in ev

    @posix_only("chmod-based permission denial")
    def test_unreadable_file_skipped(self, tmp_path):
        if is_root():
            import pytest

            pytest.skip("permission bits are ignored when running as root")
        shot = tmp_path / "locked.png"
        shot.write_bytes(_PNG_1PX)
        shot.chmod(0)
        try:
            printer = _new_printer()
            ev = _run_tool(printer, "Bash", {"command": "gen"}, f"{shot}\n")
        finally:
            shot.chmod(0o644)
        assert "images" not in ev

    def test_directory_named_like_image_skipped(self, tmp_path):
        imgdir = tmp_path / "assets.png"
        imgdir.mkdir()
        printer = _new_printer()
        ev = _run_tool(printer, "Bash", {"command": "gen"}, f"{imgdir}\n")
        assert "images" not in ev

    def test_finish_result_suppressed_entirely(self, tmp_path):
        shot = tmp_path / "fin.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        printer.print("finish", type="tool_call", tool_input={})
        printer.print(
            f"done, see {shot}",
            type="tool_result",
            tool_name="finish",
            tool_input={},
        )
        assert _tool_results(printer) == []


class TestRecencyWindowFallback:
    """A tool_result with no preceding tool_call uses the 60s window."""

    def test_fresh_file_embedded_without_tool_call(self, tmp_path):
        shot = tmp_path / "fallback.png"
        shot.write_bytes(_PNG_1PX)
        printer = _new_printer()
        printer.print(
            f"Screenshot saved to {shot}",
            type="tool_result",
            tool_name="screenshot",
            tool_input={"file_path": str(shot)},
        )
        evs = _tool_results(printer)
        assert len(evs) == 1
        assert len(evs[0]["images"]) == 1

    def test_stale_file_skipped_without_tool_call(self, tmp_path):
        shot = tmp_path / "stale.png"
        shot.write_bytes(_PNG_1PX)
        stale = time.time() - 120
        os.utime(shot, (stale, stale))
        printer = _new_printer()
        printer.print(
            f"Screenshot saved to {shot}",
            type="tool_result",
            tool_name="screenshot",
            tool_input={"file_path": str(shot)},
        )
        evs = _tool_results(printer)
        assert "images" not in evs[0]


class TestToolCallStartLifecycle:
    """The per-task tool_call timestamp is cleaned up and guarded."""

    def test_cleanup_task_drops_entry_and_blocks_stragglers(self):
        printer = _new_printer()
        key = printer._task_key()
        printer.print("Bash", type="tool_call", tool_input={"command": "ls"})
        assert key in printer._tool_call_started
        printer.stop_recording()
        printer.cleanup_task(key)
        assert key not in printer._tool_call_started
        # A straggler tool_call after cleanup must not re-create the
        # entry (same guard as the usage offsets, R09-7).
        printer.print("Bash", type="tool_call", tool_input={"command": "ls"})
        assert key not in printer._tool_call_started

    def test_result_consumes_tool_call_timestamp(self, tmp_path):
        """A later unpaired result uses the fallback window, not the
        first call's stale cutoff."""
        printer = _new_printer()
        key = printer._task_key()
        _run_tool(printer, "Bash", {"command": "true"}, "ok")
        assert key not in printer._tool_call_started
        # A file older than the 60s fallback window but newer than the
        # consumed call's cutoff must NOT be embedded by an unpaired
        # result (pre-fix, the stale cutoff would have embedded it).
        shot = tmp_path / "later.png"
        shot.write_bytes(_PNG_1PX)
        stale = time.time() - 120
        os.utime(shot, (stale, stale))
        printer.start_recording()
        printer.print(
            f"see {shot}",
            type="tool_result",
            tool_name="Bash",
            tool_input=None,
        )
        evs = _tool_results(printer)
        assert "images" not in evs[0]

    def test_taskless_tool_call_records_no_entry(self, tmp_path):
        printer = JsonPrinter()
        printer._thread_local.task_id = ""
        printer.print("Bash", type="tool_call", tool_input={"command": "ls"})
        assert "" not in printer._tool_call_started
        # A taskless result still embeds (fallback window) but records
        # no budget entry under the empty key (nothing would pop it).
        shot = tmp_path / "taskless.png"
        shot.write_bytes(_PNG_1PX)
        images = printer._collect_result_images(f"see {shot}", None, None)
        assert len(images) == 1
        assert "" not in printer._image_b64_used


class TestPerTaskImageBudget:
    """Embedding stops before a task's replay frame outgrows limits."""

    def test_budget_caps_total_embedded_bytes(self, tmp_path):
        # 5 files of exactly _MAX_RESULT_IMAGE_BYTES: base64 expands
        # each to ~2.8 MB, so with a 12 MiB per-task budget the first
        # four fit (3 on the first event, 1 on the second) and the
        # fifth is skipped.
        paths = []
        for i in range(5):
            p = tmp_path / f"big{i}.png"
            p.write_bytes(b"\x89PNG" + b"x" * (_MAX_RESULT_IMAGE_BYTES - 4))
            paths.append(p)
        printer = _new_printer()
        ev1 = _run_tool(
            printer,
            "Bash",
            {"command": "gen"},
            " ".join(str(p) for p in paths[:3]),
        )
        assert len(ev1["images"]) == 3
        printer.start_recording()
        ev2 = _run_tool(
            printer,
            "Bash",
            {"command": "gen"},
            f"{paths[3]} {paths[4]}",
        )
        assert [i["path"] for i in ev2["images"]] == [str(paths[3])]
        # The budget entry is freed with the task.
        key = printer._task_key()
        assert key in printer._image_b64_used
        printer.cleanup_task(key)
        assert key not in printer._image_b64_used


class TestImagePathCandidateExtraction:
    """The path scanner finds real mentions and rejects lookalikes."""

    def test_extracts_and_dedupes_in_order(self):
        text = "a /x/one.png b (two.jpg) ![p](./three.svg) /x/one.png"
        assert _extract_image_path_candidates(text) == [
            "/x/one.png",
            "two.jpg",
            "./three.svg",
        ]

    def test_rejects_lookalikes(self):
        assert _extract_image_path_candidates("") == []
        # Longer names must not yield a false prefix match.
        assert _extract_image_path_candidates("backup at x.png.bak") == []
        # https:// URLs leave a protocol-relative remnant; dropped.
        assert _extract_image_path_candidates("https://h/x.png") == []

    def test_quoted_paths_may_contain_spaces(self):
        text = 'saved to "/home/A User/shot.png" and \'./my plots/p.svg\''
        assert _extract_image_path_candidates(text) == [
            "/home/A User/shot.png",
            "./my plots/p.svg",
        ]

    def test_windows_drive_prefix_kept(self):
        assert _extract_image_path_candidates(r"at D:\shots\x.png now") == [
            r"D:\shots\x.png"
        ]

    def test_sentence_ending_period_allowed(self):
        assert _extract_image_path_candidates("Saved output.png.") == [
            "output.png"
        ]
        assert _extract_image_path_candidates("Saved output.png. Next") == [
            "output.png"
        ]

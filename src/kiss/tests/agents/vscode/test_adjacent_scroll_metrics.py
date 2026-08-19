# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests that tokens/cost/steps are updated when scrolling to adjacent tasks.

Bug: after a tab is replayed, scrolling to an adjacent task via the
adjacent-scroll feature does not update the header tokens/cost/steps.
The ``updateVisibleTask()`` function only updated the task-name text,
and ``renderAdjacentTask()`` did not save statusSteps or store per-task
metrics on the container element.

The fix:
  - ``renderAdjacentTask`` now saves/restores ``statusSteps`` (alongside
    ``statusTokens`` and ``statusBudget``) and captures the adjacent
    task's replayed metrics into ``container.dataset.metricTokens/Budget/Steps``.
  - ``updateVisibleTask`` now reads those ``dataset`` attributes and
    updates the header when the user scrolls to an adjacent task, and
    restores ``currentTaskMetrics`` when scrolling back to the main task.
  - ``replayTaskEvents`` and ``processOutputEvent`` snapshot the current
    task's metrics into ``currentTaskMetrics`` so they can be restored.
  - ``clearUsageMetrics`` resets ``currentTaskMetrics``.

The savedSteps/dataset-metrics bookkeeping was later refactored out of
``renderAdjacentTask`` into the shared helper ``replayDetachedTranscript``
(also used by the share export). The tests follow that delegation via
``_render_adjacent_replay_body`` instead of pinning the code to the
``renderAdjacentTask`` body, so the behavioral guarantee — not the code
layout — is what is asserted.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

MAIN_JS = (
    Path(__file__).resolve().parents[3]
    / "agents"
    / "vscode"
    / "media"
    / "main.js"
)


class TestAdjacentScrollMetrics(unittest.TestCase):
    """Structural assertions that adjacent-scroll code updates metrics."""

    def setUp(self) -> None:
        self.src = MAIN_JS.read_text()


    def test_current_task_metrics_declared(self) -> None:
        """A ``currentTaskMetrics`` variable must be declared to hold the
        main task's tokens/budget/steps for adjacent-scroll restoration."""
        self.assertRegex(
            self.src,
            r"let\s+currentTaskMetrics\s*=",
            "currentTaskMetrics variable not declared",
        )


    def _render_adjacent_replay_body(self) -> str:
        """Return the code that renderAdjacentTask replays a transcript with.

        The savedSteps/dataset-metrics bookkeeping originally lived inline
        in ``renderAdjacentTask``; it was later refactored into the shared
        helper ``replayDetachedTranscript`` (also used by the share
        export), which ``renderAdjacentTask`` delegates to. Follow that
        delegation so the behavioral guarantee is asserted wherever the
        replay code actually lives.
        """
        body = self._function_body("renderAdjacentTask")
        for helper in re.findall(r"\b(\w+)\s*\(", body):
            if helper != "renderAdjacentTask" and "replayEventsInto" in (
                self._function_body(helper, required=False)
            ):
                return self._function_body(helper)
        return body


    def _function_body(self, name: str, required: bool = True) -> str:
        """Extract the body of top-level ``function name(...) {...}``."""
        m = re.search(
            r"function " + re.escape(name) + r"\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        if m is None:
            if required:
                self.fail(f"Could not find {name} body")
            return ""
        return m.group(1)


    def test_render_adjacent_task_saves_status_steps(self) -> None:
        """The adjacent-task replay must save statusSteps before replay and
        restore it after, so the adjacent task doesn't clobber the
        current task's step count."""
        body = self._render_adjacent_replay_body()
        self.assertIn(
            "savedSteps",
            body,
            "the adjacent-task replay does not save statusSteps",
        )
        replay_pos = body.index("replayEventsInto")
        save_pos = body.index("savedSteps")
        self.assertLess(
            save_pos,
            replay_pos,
            "savedSteps must be captured before replayEventsInto",
        )
        restore_match = re.search(
            r"statusSteps\b.*=\s*savedSteps", body[replay_pos:]
        )
        self.assertIsNotNone(
            restore_match,
            "statusSteps is not restored from savedSteps after replay",
        )


    def test_render_adjacent_task_stores_dataset_metrics(self) -> None:
        """The adjacent-task container must have dataset.metricTokens,
        dataset.metricBudget, and dataset.metricSteps set from the
        replayed events."""
        body = self._render_adjacent_replay_body()
        for attr in ("metricTokens", "metricBudget", "metricSteps"):
            self.assertIn(
                f"dataset.{attr}",
                body,
                "the adjacent-task replay does not set "
                f"container.dataset.{attr}",
            )


    def test_update_visible_task_updates_metrics(self) -> None:
        """updateVisibleTask must update statusTokens, statusBudget, and
        statusSteps from the visible adjacent container's dataset."""
        m = re.search(
            r"function updateVisibleTask\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find updateVisibleTask body")
        assert m is not None
        body = m.group(1)
        for attr in ("metricTokens", "metricBudget", "metricSteps"):
            self.assertIn(
                f"dataset.{attr}",
                body,
                f"updateVisibleTask does not read dataset.{attr}",
            )
        self.assertIn(
            "currentTaskMetrics",
            body,
            "updateVisibleTask does not restore currentTaskMetrics",
        )


    def test_update_visible_task_captures_visible_container(self) -> None:
        """updateVisibleTask must capture a reference to the visible
        adjacent-task container (not just the task name) so it can
        read per-task dataset attributes."""
        m = re.search(
            r"function updateVisibleTask\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find updateVisibleTask body")
        assert m is not None
        body = m.group(1)
        # The container reference is captured via the regionNeighbour()
        # helper (formerly an inline `visibleContainer` lookup) and its
        # dataset is what the metric reads below depend on.
        self.assertIn(
            "regionNeighbour(",
            body,
            "updateVisibleTask does not capture the visible adjacent-task "
            "container via regionNeighbour()",
        )
        self.assertRegex(
            body,
            r"(const|let|var)\s+\w+\s*=\s*regionNeighbour\(",
            "updateVisibleTask must store the container reference so it "
            "can read per-task dataset attributes",
        )


    def test_replay_task_events_snapshots_metrics(self) -> None:
        """replayTaskEvents must store the replayed task's metrics
        into currentTaskMetrics after replaying events."""
        m = re.search(
            r"function replayTaskEvents\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find replayTaskEvents body")
        assert m is not None
        body = m.group(1)
        self.assertIn(
            "currentTaskMetrics",
            body,
            "replayTaskEvents does not snapshot currentTaskMetrics",
        )


    def test_process_output_event_updates_metrics_on_result(self) -> None:
        """processOutputEvent must snapshot currentTaskMetrics after
        a result or usage_info event so live-streaming metrics are
        available for adjacent-scroll restoration."""
        m = re.search(
            r"function processOutputEvent\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find processOutputEvent body")
        assert m is not None
        body = m.group(1)
        self.assertIn(
            "currentTaskMetrics",
            body,
            "processOutputEvent does not update currentTaskMetrics",
        )


    def test_clear_usage_metrics_resets_current_task_metrics(self) -> None:
        """clearUsageMetrics must reset currentTaskMetrics so stale
        metrics from a previous task are not shown after a new task
        starts."""
        m = re.search(
            r"function clearUsageMetrics\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find clearUsageMetrics body")
        assert m is not None
        body = m.group(1)
        self.assertIn(
            "currentTaskMetrics",
            body,
            "clearUsageMetrics does not reset currentTaskMetrics",
        )


if __name__ == "__main__":
    unittest.main()

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

    # ------------------------------------------------------------------
    # 1. currentTaskMetrics variable exists
    # ------------------------------------------------------------------

    def test_currentTaskMetrics_declared(self) -> None:
        """A ``currentTaskMetrics`` variable must be declared to hold the
        main task's tokens/budget/steps for adjacent-scroll restoration."""
        self.assertRegex(
            self.src,
            r"let\s+currentTaskMetrics\s*=",
            "currentTaskMetrics variable not declared",
        )

    # ------------------------------------------------------------------
    # 2. renderAdjacentTask saves/restores statusSteps
    # ------------------------------------------------------------------

    def test_renderAdjacentTask_saves_statusSteps(self) -> None:
        """renderAdjacentTask must save statusSteps before replay and
        restore it after, so the adjacent task doesn't clobber the
        current task's step count."""
        # Extract the function body
        m = re.search(
            r"function renderAdjacentTask\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find renderAdjacentTask body")
        assert m is not None
        body = m.group(1)
        self.assertIn(
            "savedSteps",
            body,
            "renderAdjacentTask does not save statusSteps",
        )
        # Must assign savedSteps before replayEventsInto
        replay_pos = body.index("replayEventsInto")
        save_pos = body.index("savedSteps")
        self.assertLess(
            save_pos,
            replay_pos,
            "savedSteps must be captured before replayEventsInto",
        )
        # Must restore after replay
        restore_match = re.search(
            r"statusSteps\b.*=\s*savedSteps", body[replay_pos:]
        )
        self.assertIsNotNone(
            restore_match,
            "statusSteps is not restored from savedSteps after replay",
        )

    # ------------------------------------------------------------------
    # 3. renderAdjacentTask stores per-task metrics on container dataset
    # ------------------------------------------------------------------

    def test_renderAdjacentTask_stores_dataset_metrics(self) -> None:
        """The adjacent-task container must have dataset.metricTokens,
        dataset.metricBudget, and dataset.metricSteps set from the
        replayed events."""
        m = re.search(
            r"function renderAdjacentTask\b[^{]*\{(.*?)^\s{2}\}",
            self.src,
            re.DOTALL | re.MULTILINE,
        )
        self.assertIsNotNone(m, "Could not find renderAdjacentTask body")
        assert m is not None
        body = m.group(1)
        for attr in ("metricTokens", "metricBudget", "metricSteps"):
            self.assertIn(
                f"dataset.{attr}",
                body,
                f"renderAdjacentTask does not set container.dataset.{attr}",
            )

    # ------------------------------------------------------------------
    # 4. updateVisibleTask updates header metrics
    # ------------------------------------------------------------------

    def test_updateVisibleTask_updates_metrics(self) -> None:
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
        # Must read from dataset
        for attr in ("metricTokens", "metricBudget", "metricSteps"):
            self.assertIn(
                f"dataset.{attr}",
                body,
                f"updateVisibleTask does not read dataset.{attr}",
            )
        # Must also restore currentTaskMetrics when scrolled back
        self.assertIn(
            "currentTaskMetrics",
            body,
            "updateVisibleTask does not restore currentTaskMetrics",
        )

    # ------------------------------------------------------------------
    # 5. updateVisibleTask tracks the visible container
    # ------------------------------------------------------------------

    def test_updateVisibleTask_captures_visible_container(self) -> None:
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
        self.assertIn(
            "visibleContainer",
            body,
            "updateVisibleTask does not capture visibleContainer",
        )

    # ------------------------------------------------------------------
    # 6. replayTaskEvents snapshots currentTaskMetrics
    # ------------------------------------------------------------------

    def test_replayTaskEvents_snapshots_metrics(self) -> None:
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

    # ------------------------------------------------------------------
    # 7. processOutputEvent updates currentTaskMetrics on result/usage
    # ------------------------------------------------------------------

    def test_processOutputEvent_updates_metrics_on_result(self) -> None:
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

    # ------------------------------------------------------------------
    # 8. clearUsageMetrics resets currentTaskMetrics
    # ------------------------------------------------------------------

    def test_clearUsageMetrics_resets_currentTaskMetrics(self) -> None:
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

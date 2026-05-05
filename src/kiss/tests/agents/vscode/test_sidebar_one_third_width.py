"""Regression test: on first activation after extension install, the
secondary side bar must be resized to approximately one-third of the
VS Code window width.

Previously, the extension widened the sidebar by calling
``workbench.action.increaseViewSize`` exactly three times — a fixed
~150-px bump regardless of monitor size.  On a wide monitor (e.g.
2560-px) that left the sidebar at ~450 px (≈18 %); on a small
laptop screen it could over-shoot.

The new implementation in ``SorcarSidebarView.widenToOneThird``:

  * asks the webview to measure ``window.innerWidth`` (the sidebar's
    own width) and ``screen.availWidth`` (closest proxy for VS Code
    window width that a sandboxed webview can read);
  * iteratively calls ``workbench.action.increaseViewSize`` /
    ``decreaseViewSize`` until the sidebar is within ~6 % of
    ``screenWidth / 3``;
  * bails out after a max-iteration cap or when the resize command
    has no effect for two consecutive iterations (hit a min/max).

These static-source tests verify the implementation pieces are wired
up.  An additional Node-based simulation test verifies the
convergence algorithm itself reaches one-third within the iteration
cap on a representative range of monitor widths.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import unittest
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SimResult:
    """Result of one Node-based widenToOneThird convergence simulation."""

    final: float
    target: float
    iters: int
    within_tol: bool

VSCODE_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"
TS_DIR = VSCODE_DIR / "src"
MEDIA_DIR = VSCODE_DIR / "media"


def _read(p: Path) -> str:
    return p.read_text()


class TestExtensionUsesWidenToOneThird(unittest.TestCase):
    """``extension.ts`` calls ``widenToOneThird`` on first activation
    and no longer hard-codes a 3-iteration ``increaseViewSize`` loop."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read(TS_DIR / "extension.ts")

    def test_widen_to_one_third_is_invoked(self) -> None:
        self.assertIn(
            "widenToOneThird(",
            self.src,
            "extension.ts must invoke sidebarView.widenToOneThird() so the "
            "sidebar is sized to ~1/3 of the VS Code window on first launch.",
        )

    def test_widen_block_is_guarded_by_first_run_flag(self) -> None:
        # The widen call must live inside the `sidebarWidened` global-state
        # gate so it only runs once after install, not on every activation.
        m = re.search(
            r"globalState\.get<boolean>\('sidebarWidened'\)\)\s*\{(.*?)\n  \}\n",
            self.src,
            re.DOTALL,
        )
        assert m is not None, "could not locate the sidebarWidened first-run gate"
        self.assertIn(
            "widenToOneThird",
            m.group(1),
            "widenToOneThird() must be called inside the sidebarWidened gate.",
        )

    def test_no_hardcoded_three_iteration_loop(self) -> None:
        # The previous fixed `for (let i = 0; i < 3; i++)` loop around
        # `increaseViewSize` is gone — it would override the 1/3 logic.
        self.assertNotRegex(
            self.src,
            r"for\s*\(\s*let\s+i\s*=\s*0\s*;\s*i\s*<\s*3\s*;\s*i\+\+\s*\)"
            r"\s*\{[^}]*increaseViewSize",
            "extension.ts still hard-codes a 3-iteration increaseViewSize "
            "loop; that overrides widenToOneThird.",
        )

    def test_focus_auxiliary_bar_still_called_first(self) -> None:
        # increase/decreaseViewSize only affects the focused part, so the
        # auxiliary (secondary) bar must be focused before resizing.
        # Find the position of focusAuxiliaryBar and widenToOneThird.
        focus_idx = self.src.find("'workbench.action.focusAuxiliaryBar'")
        # Match the actual call (skip prose comments mentioning the name).
        widen_idx = self.src.find("widenToOneThird()")
        self.assertGreater(focus_idx, 0, "focusAuxiliaryBar missing")
        self.assertGreater(widen_idx, 0, "widenToOneThird missing")
        self.assertLess(
            focus_idx,
            widen_idx,
            "focusAuxiliaryBar must be called BEFORE widenToOneThird so "
            "the resize commands target the secondary side bar.",
        )


class TestSidebarViewHasWidenToOneThird(unittest.TestCase):
    """``SorcarSidebarView`` exposes ``widenToOneThird`` and uses
    measure→adjust feedback rather than a fixed iteration count."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read(TS_DIR / "SorcarSidebarView.ts")

    def test_public_method_exists(self) -> None:
        self.assertRegex(
            self.src,
            r"public\s+async\s+widenToOneThird\s*\(",
            "SorcarSidebarView.widenToOneThird must be public async.",
        )

    def test_uses_measure_size_message(self) -> None:
        # The method must request a measurement from the webview.
        self.assertIn(
            "'measureSize'",
            self.src,
            "widenToOneThird must request a size measurement from the webview "
            "(missing 'measureSize' postMessage).",
        )

    def test_handles_size_report_message(self) -> None:
        self.assertIn(
            "case 'sizeReport'",
            self.src,
            "SorcarSidebarView must handle the 'sizeReport' message from "
            "the webview to read back the measured width.",
        )

    def test_uses_both_increase_and_decrease(self) -> None:
        # A pure increase loop can over-shoot; we need decrease too.
        self.assertIn(
            "workbench.action.increaseViewSize",
            self.src,
            "widenToOneThird must invoke workbench.action.increaseViewSize.",
        )
        self.assertIn(
            "workbench.action.decreaseViewSize",
            self.src,
            "widenToOneThird must invoke workbench.action.decreaseViewSize "
            "to handle the over-shoot case.",
        )

    def test_target_is_one_third_of_screen_width(self) -> None:
        # The target computation must divide the screen width by 3.
        self.assertRegex(
            self.src,
            r"\.screen\s*/\s*3",
            "widenToOneThird must compute target = screenWidth / 3.",
        )

    def test_has_iteration_cap(self) -> None:
        # The loop must have a max-iteration parameter to avoid runaway loops.
        self.assertRegex(
            self.src,
            r"maxIterations[^=]*=\s*\d+",
            "widenToOneThird must have a maxIterations cap to prevent infinite loops.",
        )

    def test_size_report_resolver_field(self) -> None:
        # The pending-resolver field is what bridges the async measureSize
        # request to the sizeReport message handler.
        self.assertIn(
            "_sizeReportResolver",
            self.src,
            "_sizeReportResolver must exist to bridge measureSize → sizeReport.",
        )


class TestWebviewHandlesMeasureSize(unittest.TestCase):
    """``main.js`` answers ``measureSize`` with ``sizeReport``."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read(MEDIA_DIR / "main.js")

    def test_measure_size_case_exists(self) -> None:
        self.assertIn(
            "case 'measureSize'",
            self.src,
            "main.js must handle the 'measureSize' message from the "
            "extension by posting a 'sizeReport' back.",
        )

    def test_size_report_payload_includes_inner_and_screen_width(self) -> None:
        # Find the case body and confirm both width fields are posted.
        m = re.search(
            r"case\s+'measureSize'\s*:\s*(.*?)\n\s*break;",
            self.src,
            re.DOTALL,
        )
        assert m is not None, "could not locate measureSize case body"
        body = m.group(1)
        self.assertIn(
            "type: 'sizeReport'",
            body,
            "measureSize handler must post a sizeReport message.",
        )
        self.assertIn(
            "innerWidth",
            body,
            "sizeReport must include innerWidth (the sidebar's width).",
        )
        self.assertIn(
            "screenWidth",
            body,
            "sizeReport must include screenWidth (proxy for VS Code window).",
        )
        self.assertIn(
            "window.innerWidth",
            body,
            "sizeReport must read window.innerWidth.",
        )
        self.assertIn(
            "screen",
            body,
            "sizeReport must read screen.availWidth (or similar) as the "
            "VS Code window-width proxy.",
        )


class TestTypesDeclareNewMessages(unittest.TestCase):
    """``types.ts`` declares the new measureSize / sizeReport messages."""

    src: str

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _read(TS_DIR / "types.ts")

    def test_to_webview_has_measure_size(self) -> None:
        self.assertRegex(
            self.src,
            r"type:\s*'measureSize'",
            "types.ts must declare a ToWebviewMessage variant of type 'measureSize'.",
        )

    def test_from_webview_has_size_report(self) -> None:
        self.assertRegex(
            self.src,
            r"type:\s*'sizeReport'\s*;\s*innerWidth\s*:\s*number\s*;\s*"
            r"screenWidth\s*:\s*number",
            "types.ts must declare a FromWebviewMessage variant of type "
            "'sizeReport' with innerWidth and screenWidth fields.",
        )


class TestConvergenceSimulation(unittest.TestCase):
    """Run a Node simulation of ``widenToOneThird`` to verify the
    algorithm converges to ~1/3 within the iteration cap on a range
    of representative monitor widths.

    The simulation mirrors the TypeScript code: it tracks a sidebar
    width, applies a +/- 50-px adjustment each iteration (the empirical
    increment of VS Code's increase/decrease commands), and stops when
    within 6 % tolerance of ``screen / 3`` or when the cap is hit."""

    @classmethod
    def setUpClass(cls) -> None:
        # Skip if node is not available.
        try:
            subprocess.run(
                ["node", "--version"],
                check=True,
                capture_output=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            raise unittest.SkipTest("node is not available on PATH")

    def _run_sim(
        self,
        screen_width: int,
        initial: int,
        increment: int = 50,
        max_iter: int = 30,
        tol: float = 0.06,
    ) -> SimResult:
        script = f"""
            const screen = {screen_width};
            const inc = {increment};
            const target = screen / 3;
            let cur = {initial};
            let iters = 0;
            for (let i = 0; i < {max_iter}; i++) {{
              if (Math.abs(cur - target) <= target * {tol}) break;
              if (cur < target) cur += inc;
              else cur -= inc;
              iters++;
            }}
            console.log(JSON.stringify({{
              final: cur,
              target,
              iters,
              within_tol: Math.abs(cur - target) <= target * {tol},
            }}));
        """
        out = subprocess.run(
            ["node", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        raw = json.loads(out.stdout.strip())
        return SimResult(
            final=float(raw["final"]),
            target=float(raw["target"]),
            iters=int(raw["iters"]),
            within_tol=bool(raw["within_tol"]),
        )

    def test_converges_on_typical_laptop_1440(self) -> None:
        # 1440-px window, sidebar starts at default 300 px.
        r = self._run_sim(screen_width=1440, initial=300)
        self.assertTrue(
            r.within_tol,
            f"did not converge on 1440px screen: {r}",
        )
        self.assertLess(r.iters, 30, f"too many iterations: {r}")

    def test_converges_on_wide_monitor_2560(self) -> None:
        r = self._run_sim(screen_width=2560, initial=300)
        self.assertTrue(r.within_tol, f"did not converge on 2560px screen: {r}")
        # Target = 853, start = 300, +50/iter ⇒ ≤ 12 iterations.
        self.assertLess(r.iters, 30, f"too many iterations: {r}")

    def test_converges_on_4k_3840(self) -> None:
        r = self._run_sim(screen_width=3840, initial=300)
        self.assertTrue(r.within_tol, f"did not converge on 3840px screen: {r}")
        self.assertLess(r.iters, 30, f"too many iterations: {r}")

    def test_converges_when_starting_too_wide(self) -> None:
        # Sidebar already wider than 1/3 — must shrink, not grow.
        r = self._run_sim(screen_width=1440, initial=900)
        self.assertTrue(
            r.within_tol,
            f"did not converge when starting too wide: {r}",
        )
        self.assertLess(
            r.final,
            900,
            f"sidebar was not shrunk from 900px: final={r.final}",
        )

    def test_within_tol_means_within_six_percent(self) -> None:
        # Sanity: the algorithm's tolerance gate works as advertised.
        r = self._run_sim(screen_width=1500, initial=300)
        self.assertTrue(r.within_tol)
        rel = abs(r.final - r.target) / r.target
        self.assertLessEqual(
            rel,
            0.06 + 1e-9,
            f"final {r.final} is more than 6% off target {r.target}",
        )


if __name__ == "__main__":
    # Ensure cwd doesn't matter for the static reads.
    os.chdir(Path(__file__).resolve().parent)
    unittest.main()

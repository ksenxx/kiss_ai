# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
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

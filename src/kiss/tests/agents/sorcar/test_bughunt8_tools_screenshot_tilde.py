# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt pass #8 — ``~`` screenshot paths create a literal ``./~/`` dir.

``WebUseTool.screenshot`` anchors relative paths under ``work_dir`` (like
the file tools) but never expands ``~``, so ``screenshot("~/shot.png")``
saves into a directory literally named ``~`` inside the work dir instead
of the user's home — the same footgun fixed in ``useful_tools._absolutize``.
"""

from pathlib import Path

import pytest

from kiss.agents.sorcar.web_use_tool import WebUseTool


def test_screenshot_tilde_path_targets_home_not_literal_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    work_dir = tmp_path / "wd"
    work_dir.mkdir()

    tool = WebUseTool(user_data_dir=None, headless=True, work_dir=str(work_dir))
    try:
        tool.go_to_url("about:blank")
        out = tool.screenshot("~/shot.png")

        assert out.startswith("Screenshot saved to "), out
        # Must land in the (fake) home directory...
        assert (home / "shot.png").is_file()
        # ...and must NOT create a literal '~' directory in the work dir.
        assert not (work_dir / "~").exists()
    finally:
        tool.close()

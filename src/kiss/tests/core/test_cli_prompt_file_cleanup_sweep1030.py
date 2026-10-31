# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""End-to-end tests: a system prompt file that cannot be written is not left behind.

Both CLI adapters write the turn's system prompt to a private temp file
(``kiss-codex-*.md``, ``kiss-claude-*.md``) and delete it when the turn
ends.  Codex wrote it before entering the ``try`` that deletes it, so a
write that failed (a lone surrogate in the prompt, a full disk, a Stop
landing mid-write) left the file, holding the prompt, in the temp dir:
one more per retried attempt.

Each case runs in a child interpreter whose ``TMPDIR`` is a fresh
directory, so the files left there are exactly the turn's.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

_CHILD = """
from kiss.core.models.model_info import model
m = model({model_name!r}, model_config={{"system_instruction": "Be brief \\ud800 here."}})
m.initialize("Say ok.")
try:
    m.generate()
except UnicodeEncodeError:
    print("write failed")
"""


@pytest.mark.parametrize(
    "model_name,prefix", [("codex/gpt-5.5", "kiss-codex-"), ("cc/sonnet", "kiss-claude-")]
)
def test_a_failed_system_prompt_write_leaves_no_file(
    tmp_path: Path, model_name: str, prefix: str
) -> None:
    """The prompt cannot be encoded: the turn fails before the CLI starts, and no file stays."""
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    env = {
        **os.environ,
        "TMPDIR": str(temp_dir),
        "TEMP": str(temp_dir),
        "TMP": str(temp_dir),
        # No saved ``codex login``: the turn's billing check cannot refuse it.
        "CODEX_HOME": str(tmp_path / "codex-home"),
    }
    child = subprocess.run(
        [sys.executable, "-c", _CHILD.format(model_name=model_name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert child.stdout.strip() == "write failed", child.stderr[-2000:]
    assert list(temp_dir.glob(f"{prefix}*")) == []

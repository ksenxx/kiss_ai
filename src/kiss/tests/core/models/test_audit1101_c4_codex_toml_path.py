# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-11-01 (C4): the Codex instructions-file override is valid TOML for any path.

``codex exec -c model_instructions_file=<value>`` parses *value* as TOML.
The path of the instructions file lives under ``TMPDIR``, and a non-BMP
character there (an emoji in a directory name) used to be written as
JSON's ``\\ud83d\\ude42`` surrogate pair, which TOML rejects, so Codex
never loaded KISS's system instruction for such a user.  The override
must round-trip through a TOML parser for ASCII, non-ASCII and non-BMP
paths alike.
"""

from __future__ import annotations

import tomllib

import pytest

from kiss.core.models.codex_model import CodexModel
from kiss.tests.cli_locator_stub import stub_cli_locators  # noqa: F401


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/kiss codex/sys.md",
        "/tmp/kiss-ünïcode/sys.md",
        "/tmp/emoji-🙂/kiss-codex.md",
        "/tmp/kiss\x7fdel/sys.md",
    ],
)
def test_instructions_file_override_round_trips_through_toml(path: str) -> None:
    args = CodexModel("codex/default")._build_cli_args(path)
    override = next(a for a in args if a.startswith("model_instructions_file="))
    assert args[args.index(override) - 1] == "-c"
    assert tomllib.loads(override) == {"model_instructions_file": path}

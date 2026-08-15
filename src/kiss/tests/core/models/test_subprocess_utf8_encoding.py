# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Regression tests: CLI model wrappers pin subprocess pipes to UTF-8.

On Windows, Python text-mode subprocess pipes default to the system
ANSI code page rather than UTF-8.  Writing a prompt containing
non-ASCII text (accents, emoji) to such a pipe can raise
``UnicodeEncodeError`` or silently mangle the bytes the CLI receives.
Passing ``encoding="utf-8"`` alongside ``text=True`` makes the prompt
encode identically on every platform.

The tests run a REAL stand-in ``claude`` / ``codex`` executable that
reads its stdin as UTF-8 and echoes back what it decoded, so the
round-trip is proven end to end rather than by inspecting the arguments
handed to ``subprocess.Popen``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kiss.core.models.claude_code_model import ClaudeCodeModel
from kiss.core.models.codex_model import CodexModel
from kiss.tests.core.models.test_cli_subprocess_lifecycle import install_cli

_NON_ASCII_PROMPT = "Please review the café résumé ☕😀 and reply 'ok'."

_CODEX_ECHOES_ITS_PROMPT = """
    import json
    import sys

    received = sys.stdin.buffer.read().decode("utf-8")
    print(json.dumps({"type": "item.completed",
                      "item": {"type": "agent_message", "text": received}}),
          flush=True)
    print(json.dumps({"type": "turn.completed", "usage": {}}), flush=True)
"""

_CLAUDE_ECHOES_ITS_PROMPT = """
    import json
    import sys

    received = sys.stdin.buffer.read().decode("utf-8")
    print(json.dumps({"type": "assistant",
                      "message": {"id": "m1",
                                  "content": [{"type": "text",
                                               "text": received}]}}), flush=True)
    print(json.dumps({"type": "result", "result": received, "usage": {}}),
          flush=True)
"""


_CLAUDE_REPLIES_FROM_A_FILE = """
    import json
    import sys

    sys.stdin.read()
    with open(sys.argv[0] + ".reply", encoding="utf-8") as handle:
        reply = handle.read()
    event = {"type": "assistant",
             "message": {"id": "m1",
                         "content": [{"type": "text", "text": reply}]}}
    # ensure_ascii=False puts real UTF-8 bytes on the pipe, so the
    # adapter's own decoding is what the test exercises.
    sys.stdout.buffer.write(
        (json.dumps(event, ensure_ascii=False) + chr(10)).encode("utf-8"))
    sys.stdout.buffer.flush()
"""


def test_codex_prompt_survives_the_pipe_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accents and emoji must reach the codex CLI byte-for-byte."""
    install_cli(tmp_path, monkeypatch, "codex", _CODEX_ECHOES_ITS_PROMPT)
    model = CodexModel("codex/default", model_config={"timeout": 20})
    model.initialize(_NON_ASCII_PROMPT)

    content, _response = model.generate()

    assert content == _NON_ASCII_PROMPT


def test_claude_prompt_survives_the_pipe_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Accents and emoji must reach the claude CLI byte-for-byte."""
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_ECHOES_ITS_PROMPT)
    model = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
    model.initialize(_NON_ASCII_PROMPT)

    content, _response = model.generate()

    assert content == _NON_ASCII_PROMPT


def test_non_ascii_stdout_is_decoded_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CLI's own non-ASCII output must decode without mangling."""
    reply = "Voilà — 完了 ✅"
    (tmp_path / "claude.reply").write_text(reply, encoding="utf-8")
    install_cli(tmp_path, monkeypatch, "claude", _CLAUDE_REPLIES_FROM_A_FILE)
    model = ClaudeCodeModel("cc/opus", model_config={"timeout": 20})
    model.initialize("hi")

    content, _response = model.generate()

    assert content == reply

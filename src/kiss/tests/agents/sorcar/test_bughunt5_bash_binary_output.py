# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Bug-hunt iteration 5: Bash tool must survive non-UTF-8 command output.

``UsefulTools._spawn`` launches commands with ``text=True, encoding="utf-8"``
and (pre-fix) strict error handling.  Real commands routinely emit bytes that
are not valid UTF-8 (``cat`` of a binary, ``grep`` on a binary, compiler
output with latin-1 bytes).  Pre-fix behaviour:

- non-streaming: ``communicate()`` raised ``UnicodeDecodeError`` internally,
  the process group was killed, and the tool returned
  ``"Error: 'utf-8' codec can't decode ..."`` — ALL output lost even though
  the command succeeded (exit 0).
- streaming: the ``UnicodeDecodeError`` escaped ``_bash_streaming`` entirely
  (there is no outer try/except on that path), crashing the tool call.
"""

import subprocess

from kiss.agents.sorcar.useful_tools import UsefulTools


def _has_bash() -> bool:
    return subprocess.run(["bash", "-c", "true"], capture_output=True).returncode == 0


def test_nonstreaming_bash_survives_invalid_utf8_output() -> None:
    """A successful command whose output contains invalid UTF-8 must not
    be reported as an error, and its decodable output must be returned."""
    assert _has_bash()
    tools = UsefulTools()
    result = tools.Bash("printf 'head\\xfftail\\n'", "emit invalid utf-8")
    assert "codec can't decode" not in result, (
        f"Bash lost all output to a decode error: {result!r}"
    )
    assert not result.startswith("Error"), result
    assert "head" in result and "tail" in result, result


def test_streaming_bash_survives_invalid_utf8_output() -> None:
    """Streaming mode must not leak UnicodeDecodeError out of the tool."""
    assert _has_bash()
    chunks: list[str] = []
    tools = UsefulTools(stream_callback=chunks.append)
    result = tools.Bash("printf 'head\\xfftail\\n'", "emit invalid utf-8")
    assert not result.startswith("Error"), result
    assert "head" in result and "tail" in result, result
    assert any("head" in c for c in chunks), chunks


def test_nonstreaming_bash_invalid_utf8_nonzero_exit_keeps_output() -> None:
    """Even on failure, the decodable part of the output must survive."""
    assert _has_bash()
    tools = UsefulTools()
    result = tools.Bash("printf 'diag\\xff\\n'; exit 3", "fail with binary output")
    assert result.startswith("Error (exit code 3)"), result
    assert "diag" in result, result

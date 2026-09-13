# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests: scanning a tool result for image paths
must run in time linear in the result size.

On 2026-09-12 a sub-agent ``Read`` ``media/vosk.js`` — a 5.8 MB file
whose single line holds one 5.77-million-character base64 token.  The
original ``_IMAGE_PATH_RE`` tried a match at EVERY offset of that token
and backtracked to the token's end each time (quadratic: 80k chars
already took 38 s), so ``_collect_result_images`` would have needed
days.  Python's ``re`` never releases the GIL, so the whole ``kiss-web``
process froze: no log line, no websocket traffic, no new tasks, one
core at 100% for 83 minutes until the process was SIGKILLed.

The tests here drive the real scanner and the real ``JsonPrinter``
with a payload shaped like that file.  The subprocess variant fails
cleanly (timeout) instead of hanging the test run when the scanner
regresses to super-linear behaviour, because a GIL-holding regex cannot
be interrupted from inside the same interpreter.
"""

import base64
import os
import subprocess
import sys
import time
from pathlib import Path

from kiss.server.json_printer import JsonPrinter, _extract_image_path_candidates

_VOSK_JS = (
    Path(__file__).resolve().parents[2] / "agents" / "vscode" / "media" / "vosk.js"
)

# The scan must finish far below this on any machine; the quadratic
# scanner needs hours for the same input.
_MAX_SCAN_SECONDS = 10.0


def _giant_token_payload(chars: int) -> str:
    """Return JS-like text with one *chars*-long base64 token in quotes.

    Mirrors the shape of ``vosk.js``: a minified bundle that embeds a
    web-worker as a base64 string literal, so the token contains no
    whitespace, quote, or punctuation that would end a path token.
    """
    token = base64.b64encode(os.urandom(chars)).decode("ascii")[:chars]
    return 'var worker_code = "' + token + '";\nexport default worker_code;\n'


def test_giant_token_scan_is_fast_in_process():
    """A 2 M-char unbroken token is scanned in well under a second."""
    text = _giant_token_payload(2_000_000)
    started = time.monotonic()
    assert _extract_image_path_candidates(text) == []
    assert time.monotonic() - started < _MAX_SCAN_SECONDS


def test_scan_time_grows_linearly_not_quadratically():
    """Doubling the token length must not quadruple the scan time.

    The measurement uses a token long enough (400k chars, ~1 s under
    the old regex per 40k, i.e. minutes here) that quadratic behaviour
    is unmistakable, while a linear scanner finishes both in
    milliseconds.
    """
    small = _giant_token_payload(200_000)
    large = _giant_token_payload(400_000)
    t0 = time.monotonic()
    _extract_image_path_candidates(small)
    t_small = time.monotonic() - t0
    t0 = time.monotonic()
    _extract_image_path_candidates(large)
    t_large = time.monotonic() - t0
    assert t_large < _MAX_SCAN_SECONDS
    # Allow generous jitter for tiny timings; quadratic would be ~4x.
    assert t_large < max(3.0 * t_small, 0.5)


def test_image_paths_still_found_next_to_giant_token(tmp_path):
    """Linear scanning keeps the real detections intact.

    A fresh PNG named before and after the giant token, in quoted and
    bare form, is still embedded on the ``tool_result`` event.
    """
    shot = tmp_path / "shot.png"
    shot.write_bytes(
        base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNg"
            "YGBgAAAABQABh6FO1AAAAABJRU5ErkJggg=="
        )
    )
    spaced = tmp_path / "a b.png"
    spaced.write_bytes(shot.read_bytes())
    content = (
        f'Saved "{spaced}" then\n'
        + _giant_token_payload(1_000_000)
        + f"\nalso wrote {shot}.\n"
    )
    printer = JsonPrinter()
    printer._thread_local.task_id = "test-scan-linear-embed"
    printer.start_recording()
    started = time.monotonic()
    printer.print("Bash", type="tool_call", tool_input={"command": "cat x"})
    printer.print(
        content,
        type="tool_result",
        tool_name="Bash",
        tool_input={"command": "cat x"},
        is_error=False,
    )
    assert time.monotonic() - started < _MAX_SCAN_SECONDS
    events = [e for e in printer.stop_recording() if e["type"] == "tool_result"]
    assert len(events) == 1
    assert [img["path"] for img in events[0]["images"]] == [str(spaced), str(shot)]


def test_real_vosk_js_tool_result_does_not_hang():
    """Emitting a ``tool_result`` for the real ``vosk.js`` completes.

    Runs in a subprocess with a hard timeout so a regression to the
    quadratic scanner fails the test instead of freezing pytest (the
    GIL-holding regex cannot be interrupted in-process).
    """
    assert _VOSK_JS.is_file(), f"fixture missing: {_VOSK_JS}"
    script = (
        "import sys, time\n"
        "from kiss.server.json_printer import JsonPrinter\n"
        f"text = open({str(_VOSK_JS)!r}, encoding='utf-8').read()\n"
        "assert max(len(line) for line in text.split('\\n')) > 1_000_000\n"
        "p = JsonPrinter()\n"
        "p._thread_local.task_id = 'vosk-scan'\n"
        "p.start_recording()\n"
        "t0 = time.monotonic()\n"
        "p.print('Read', type='tool_call', tool_input={'file_path': 'vosk.js'})\n"
        "p.print(text, type='tool_result', tool_name='Read',\n"
        "        tool_input={'file_path': 'vosk.js'}, is_error=False)\n"
        "events = [e for e in p.stop_recording() if e['type'] == 'tool_result']\n"
        "assert len(events) == 1 and 'images' not in events[0]\n"
        "print(f'scan_seconds={time.monotonic() - t0:.3f}')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.startswith("scan_seconds=")
    assert float(proc.stdout.split("=", 1)[1]) < _MAX_SCAN_SECONDS

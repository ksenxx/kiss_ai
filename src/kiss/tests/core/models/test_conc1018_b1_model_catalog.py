# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for slice B1 of the 2026-10-18 concurrency/robustness audit.

* F1 — :func:`kiss.core.models.model.parse_binary_attachments` must keep
  the text that precedes a sentinel block whose base64 payload cannot be
  decoded (it used to drop it together with the block).
* F3 — a malformed entry in ``MY_MODELS.json`` (one missing a
  loader-required field) must be skipped at import instead of raising
  ``KeyError`` from ``MODEL_INFO = _load_model_info()`` and killing every
  KISS process.  Checked in a fresh interpreter because the catalog is
  built once at import.
* F2 — ``_CLIProcess.close()`` after a failed ``Thread.start()`` in
  ``send_prompt`` is covered by ``test_conc1018_r1_cli_writer_cleanup``
  (``RLIMIT_NPROC`` fault injection in a fresh interpreter).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from kiss.core.models.model import (
    BINARY_ATTACHMENT_CLOSE,
    BINARY_ATTACHMENT_OPEN,
    encode_binary_attachment,
    parse_binary_attachments,
)
from kiss.core.models.model_info import PACKAGE_MODEL_INFO_PATH

# One base64 character is never a valid quantum: b64decode raises
# binascii.Error ("Incorrect padding") even with validate=False.
_UNDECODABLE_BLOCK = (
    BINARY_ATTACHMENT_OPEN.format(mime_type="image/png") + "Q" + BINARY_ATTACHMENT_CLOSE
)


def test_undecodable_block_keeps_preceding_text() -> None:
    """Text before an undecodable attachment block survives the parse."""
    plain, attachments = parse_binary_attachments("hello " + _UNDECODABLE_BLOCK + " world")
    assert attachments == []
    assert plain.startswith("hello ")
    assert plain.endswith(" world")
    assert "image/png" in plain  # a placeholder says what was dropped


def test_undecodable_block_between_two_valid_blocks() -> None:
    """A bad block in the middle drops only itself; its neighbours decode."""
    good = encode_binary_attachment("image/png", b"ABC")
    plain, attachments = parse_binary_attachments(
        "a " + good + " b " + _UNDECODABLE_BLOCK + " c " + good + " d"
    )
    assert [a.data for a in attachments] == [b"ABC", b"ABC"]
    assert plain.startswith("a [attached image/png, 3 bytes] b ")
    assert plain.endswith(" c [attached image/png, 3 bytes] d")
    assert base64.b64encode(b"ABC").decode() not in plain


# Prints {name: context_length | None} for the names in $B1_PROBE_NAMES.
_IMPORT_PROBE = """
import json, os
import kiss.core.models.model_info as mi
names = json.loads(os.environ["B1_PROBE_NAMES"])
print(json.dumps({n: (mi.MODEL_INFO[n].context_length if n in mi.MODEL_INFO else None)
                  for n in names}))
"""


def _context_lengths_after_import(
    home: Path, my_models: dict[str, object], names: list[str]
) -> dict[str, int | None]:
    """Import ``model_info`` in a fresh interpreter whose ``KISS_HOME`` holds *my_models*.

    Returns the loaded context length of each name in *names* (``None``
    when the name is absent from ``MODEL_INFO``).
    """
    home.mkdir(parents=True, exist_ok=True)
    (home / "MY_MODELS.json").write_text(json.dumps(my_models), encoding="utf-8")
    env = dict(os.environ, KISS_HOME=str(home), B1_PROBE_NAMES=json.dumps(names))
    env.pop("KISS_MODEL_INFO_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lengths: dict[str, int | None] = json.loads(proc.stdout)
    return lengths


def test_malformed_my_models_entry_does_not_break_import(tmp_path: Path) -> None:
    """One ``context_length``-less entry is skipped; the valid one still loads."""
    result = _context_lengths_after_import(
        tmp_path / "home",
        {
            "b1-good": {
                "context_length": 4096,
                "input_price_per_1M": 1.0,
                "output_price_per_1M": 2.0,
            },
            "b1-bad": {"input_price_per_1M": 1.0},
        },
        ["b1-good", "b1-bad"],
    )
    assert result == {"b1-good": 4096, "b1-bad": None}


def test_malformed_override_of_bundled_model_keeps_bundled_entry(tmp_path: Path) -> None:
    """A malformed override of a bundled name leaves the bundled entry and its aliases intact.

    The names come from the bundled ``MODEL_INFO.json`` and the expected
    values from a clean import with an empty ``MY_MODELS.json`` in an
    equally isolated ``KISS_HOME``: the parent's ``MODEL_INFO`` may carry
    a legitimate user override of the same entry and is no oracle.
    """
    raw = json.loads(PACKAGE_MODEL_INFO_PATH.read_text(encoding="utf-8"))
    base = next(
        name
        for name, entry in raw.items()
        if entry.get("alias_of") is None
        and any(other.get("alias_of") == name for other in raw.values())
    )
    names = [base] + [name for name, entry in raw.items() if entry.get("alias_of") == base]
    bundled = _context_lengths_after_import(tmp_path / "baseline", {}, names)
    assert None not in bundled.values()
    result = _context_lengths_after_import(
        tmp_path / "home", {base: {"input_price_per_1M": 123.0}}, names
    )
    assert result == bundled

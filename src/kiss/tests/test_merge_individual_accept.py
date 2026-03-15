"""Test: merge panel closes after resolving all diffs one-by-one."""

import time
from pathlib import Path


def _wait_for_port(port_file: str, timeout: float = 30.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = Path(port_file).read_text().strip()
            if text:
                return int(text)
        except (FileNotFoundError, ValueError):
            pass
        time.sleep(0.3)
    raise TimeoutError("Server did not write port file")



# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for race condition fixes in Base, json_printer, and model.

Verifies thread-safety of shared mutable state: agent_counter,
_bash_buffer, and _callback_helper_loop.
Also verifies cross-process safety of _record_model_usage and _save_last_model.
"""

import queue
from pathlib import Path


def _drain(q: queue.Queue) -> list[dict]:
    events = []
    while True:
        try:
            events.append(q.get_nowait())
        except queue.Empty:
            break
    return events


def _worker_record_model_usage(db_dir: str, model: str, n: int) -> None:
    """Child-process worker: record model usage *n* times via SQLite."""
    import kiss.agents.sorcar.persistence as th

    kiss_dir = Path(db_dir)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    for _ in range(n):
        th._record_model_usage(model)


def _worker_save_last_model(db_dir: str, model: str, n: int) -> None:
    """Child-process worker: call _save_last_model *n* times via SQLite."""
    import kiss.agents.sorcar.persistence as th

    kiss_dir = Path(db_dir)
    th._KISS_DIR = kiss_dir
    th._DB_PATH = kiss_dir / "sorcar.db"
    th._db_conn = None
    for _ in range(n):
        th._save_last_model(model)

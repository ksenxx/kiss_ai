# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tests for the cap on the composer's ArrowUp history.

``_load_input_texts`` feeds every (re)connecting client's ArrowUp
history.  Unbounded, a long-lived install shipped every prompt ever
typed (tens of thousands of rows, more than a megabyte) on each
connect, so it now returns only the most recent
:data:`_MAX_INPUT_HISTORY` distinct texts.  Every test runs against a
real sqlite file in a temp directory.
"""

from __future__ import annotations

import shutil
import tempfile

import kiss.agents.sorcar.persistence as th
from kiss.tests.agents.sorcar.test_steer_inputs import (
    _input_history,
    _redirect,
    _restore,
    _set_steer_timestamp,
    _set_task_timestamp,
)
from kiss.tests.persistence_helpers import close_db


class TestInputHistoryCap:
    """The ArrowUp history keeps only the most recent texts."""

    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.saved = _redirect(self.tmp)

    def teardown_method(self) -> None:
        close_db()
        _restore(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _add_tasks(self, count: int) -> None:
        """Record *count* distinct prompts, ``prompt 0`` the oldest."""
        for i in range(count):
            task_id, _chat = th._add_task(f"prompt {i}")
            _set_task_timestamp(task_id, 1000.0 + i)

    def test_default_cap_keeps_the_most_recent_texts(self) -> None:
        cap = th._MAX_INPUT_HISTORY
        self._add_tasks(cap + 20)
        history = _input_history()
        assert len(history) == cap
        assert history[0] == f"prompt {cap + 19}"
        assert history[-1] == "prompt 20"
        assert "prompt 0" not in history

    def test_under_the_cap_returns_everything(self) -> None:
        self._add_tasks(5)
        assert _input_history() == [
            f"prompt {i}" for i in range(4, -1, -1)
        ]

    def test_explicit_limit_and_steer_texts_share_the_window(self) -> None:
        self._add_tasks(4)
        th._record_steer_input("steer text")
        _set_steer_timestamp("steer text", 1002.5)
        # Recency order across both tables: prompt 3, steer text, prompt 2.
        assert _input_history(limit=3) == [
            "prompt 3", "steer text", "prompt 2",
        ]

    def test_cap_counts_distinct_texts_not_rows(self) -> None:
        # The same text typed twice occupies one slot at its latest use.
        self._add_tasks(3)
        th._record_steer_input("prompt 0")
        _set_steer_timestamp("prompt 0", 2000.0)
        assert _input_history(limit=2) == ["prompt 0", "prompt 2"]

    def test_texts_longer_than_the_character_cap_are_left_out(self) -> None:
        # A pasted document as a prompt and as a steer text: neither is
        # offered, nor does either take one of the window's slots or
        # delay the incremental watermark; a text exactly at the cap is.
        self._add_tasks(3)
        long_prompt = "p" * (th.INPUT_HISTORY_TEXT_CHARS + 1)
        at_cap = "c" * th.INPUT_HISTORY_TEXT_CHARS
        task_id, _chat = th._add_task(long_prompt)
        _set_task_timestamp(task_id, 3000.0)
        task_id, _chat = th._add_task(at_cap)
        _set_task_timestamp(task_id, 3001.0)
        th._record_steer_input("s" * (th.INPUT_HISTORY_TEXT_CHARS + 1))
        _set_steer_timestamp("s" * (th.INPUT_HISTORY_TEXT_CHARS + 1), 3002.0)
        assert _input_history(limit=3) == [at_cap, "prompt 2", "prompt 1"]
        assert th._load_input_texts(since=1001.5) == [(at_cap, 3001.0), ("prompt 2", 1002.0)]
        assert _input_history() == [at_cap, "prompt 2", "prompt 1", "prompt 0"]

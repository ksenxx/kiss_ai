# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (core-base): a failed ``run()`` must not clobber the
previous run's trajectory.

``KISSAgent._reset`` assigned ``model_name`` and built the model (which
raises ``KISSError`` for an unknown model name) *before* resetting the
per-run state — ``messages``, counters and ``run_start_timestamp``.
``run()`` saves the trajectory from a ``finally`` block, so a second
``run()`` on the same agent with a bad model name saved under the
PREVIOUS run's path (``trajectory_<name>_<id>_<old timestamp>.yaml``),
overwriting that run's record with the old messages relabelled with the
new (invalid) model name.

Fix round (review #3): the trajectory path was keyed by whole-second
``run_start_timestamp``, so two runs of one instance started within the
same second shared a path and the later save destroyed the earlier record.

Fix round 2 (review2 #1): the first fix made ``run_start_timestamp`` itself
the strictly increasing filename sequence (``max(int(time.time()),
previous + 1)``), so rapid runs were persisted with a synthetic start
LATER than their real ``run_end_timestamp``.  ``run_start_timestamp`` is
now the real wall clock again and the filename uses a separate
per-instance ``_trajectory_stamp`` that keeps the integer-seconds format
but strictly increases per run.  Several immediate runs must therefore
produce distinct files AND every record must satisfy
``run_start_timestamp <= run_end_timestamp <= now``.

Every test drives a real ``KISSAgent`` against a local OpenAI-compatible
HTTP server; nothing is mocked.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.kiss_error import KISSError

_FINISH_RESPONSE: dict[str, Any] = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "finish",
                            "arguments": json.dumps({"result": "FIRST_RUN_DONE"}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


class _FinishHandler(BaseHTTPRequestHandler):
    """Always answers with a ``finish`` tool call."""

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length:
            self.rfile.read(content_length)
        body = json.dumps(_FINISH_RESPONSE).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


def _load(path: Path) -> dict[str, Any]:
    """Load a trajectory YAML file."""
    loaded: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded


class TestFailedRunKeepsPreviousTrajectory:
    """A ``run()`` that dies while building its model saves its own record only."""

    def test_unknown_model_rerun_does_not_overwrite_previous_trajectory(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _FinishHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            agent = KISSAgent("audit-trajectory-reset")
            result = agent.run(
                model_name="gpt-4o-mini",
                prompt_template="Say hi.",
                max_steps=5,
                max_budget=1.0,
                verbose=False,
                model_config={
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "api_key": "sk-test",
                },
            )
            assert result == "FIRST_RUN_DONE"
            first_path = agent.get_trajectory_path()
            first_saved = _load(first_path)
            assert first_saved["model"] == "gpt-4o-mini"
            assert first_saved["step_count"] == 1
            assert len(first_saved["messages"]) == 3

            with pytest.raises(KISSError):
                agent.run(
                    model_name="no-such-model-audit0902",
                    prompt_template="Say hi again.",
                    max_steps=5,
                    max_budget=1.0,
                    verbose=False,
                )

            assert first_path.exists()
            assert _load(first_path) == first_saved, (
                "the failed run overwrote the previous run's trajectory file"
            )
            second_path = agent.get_trajectory_path()
            assert second_path != first_path
            second_saved = _load(second_path)
            assert second_saved["model"] == "no-such-model-audit0902"
            assert second_saved["messages"] == []
            assert second_saved["step_count"] == 0
            assert second_saved["budget_used"] == 0.0
        finally:
            server.shutdown()


class TestSameSecondRunsGetDistinctTrajectories:
    """Two immediate runs of one instance never share a trajectory path."""

    def test_two_immediate_runs_produce_two_files(self) -> None:
        server = HTTPServer(("127.0.0.1", 0), _FinishHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            agent = KISSAgent("audit-trajectory-same-second")
            paths: list[Path] = []
            for prompt in ("Say hi.", "Say hi again."):
                assert (
                    agent.run(
                        model_name="gpt-4o-mini",
                        prompt_template=prompt,
                        max_steps=5,
                        max_budget=1.0,
                        verbose=False,
                        model_config={
                            "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                            "api_key": "sk-test",
                        },
                    )
                    == "FIRST_RUN_DONE"
                )
                paths.append(agent.get_trajectory_path())
            assert paths[0] != paths[1], (
                f"both runs saved to {paths[0]}: the second run destroyed the first record"
            )
            assert all(p.exists() for p in paths)
            assert _load(paths[0])["messages"][0]["content"].endswith("Say hi.")
            assert _load(paths[1])["messages"][0]["content"].endswith("Say hi again.")
            # Whole-second integer format is preserved in the filename.
            for path in paths:
                assert path.name.rsplit("_", 1)[1].removesuffix(".yaml").isdigit()
            assert isinstance(_load(paths[1])["run_start_timestamp"], int)
            assert _load(paths[1])["run_start_timestamp"] >= _load(paths[0])["run_start_timestamp"]
        finally:
            server.shutdown()

    def test_immediate_runs_keep_real_chronology(self) -> None:
        """Every record has ``run_start <= run_end <= wall clock``.

        Four back-to-back runs finish well within three seconds, so a
        filename sequence leaking into ``run_start_timestamp`` (previous
        start + 1 per run) is forced past the real end time of at least
        the last run.
        """
        server = HTTPServer(("127.0.0.1", 0), _FinishHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            agent = KISSAgent("audit-trajectory-chronology")
            before = int(time.time())
            paths: list[Path] = []
            for _ in range(4):
                agent.run(
                    model_name="gpt-4o-mini",
                    prompt_template="Say hi.",
                    max_steps=5,
                    max_budget=1.0,
                    verbose=False,
                    model_config={
                        "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                        "api_key": "sk-test",
                    },
                )
                paths.append(agent.get_trajectory_path())
            after = int(time.time())
            assert len(set(paths)) == 4, f"runs shared a trajectory file: {paths}"
            assert all(p.exists() for p in paths)
            records = [_load(p) for p in paths]
            for record in records:
                start, end = record["run_start_timestamp"], record["run_end_timestamp"]
                assert isinstance(start, int) and isinstance(end, int)
                assert before <= start <= end <= after, (
                    f"impossible chronology: start={start} end={end} "
                    f"(wall clock {before}..{after})"
                )
            starts = [r["run_start_timestamp"] for r in records]
            assert starts == sorted(starts)
        finally:
            server.shutdown()

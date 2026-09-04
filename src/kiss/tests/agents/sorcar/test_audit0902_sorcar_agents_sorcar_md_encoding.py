# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Audit 2026-09-02 (sorcar-agents): a non-UTF-8 ``~/.kiss/SORCAR.md`` must not kill every task.

``RelentlessAgent.perform_task`` appends the user's ``~/.kiss/SORCAR.md``
to the system prompt with a plain ``read_text()``.  A file saved by a
Windows editor in cp1252 (a curly quote, an accented name) made that
call raise ``UnicodeDecodeError`` before the first model request, so
EVERY task on the machine died at start-up with a decoding traceback
and no hint that the memory file was the cause.  The sibling reader of
user-authored Markdown, ``skills.parse_frontmatter``, already tolerates
undecodable bytes; this makes the two consistent.

The test runs a real ``SorcarAgent`` against the local stand-in
OpenAI-compatible server from the parallel-agent harness and inspects
the system prompt the model actually received.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.sorcar_agent import SorcarAgent
from kiss.tests.server.parallel_agent_harness import (
    STANDIN_MODEL,
    IsolatedKissHome,
    StandInModelServer,
    finish_response,
    request_text,
)


@pytest.fixture
def env() -> Iterator[IsolatedKissHome]:
    """An isolated KISS_HOME (so the real ~/.kiss/SORCAR.md is never read)."""
    isolated = IsolatedKissHome("kiss-audit0902-sorcarmd-")
    try:
        yield isolated
    finally:
        isolated.cleanup()


class _Recorder:
    """Stand-in model that finishes at once and keeps every request."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return finish_response("memory loaded")


def test_cp1252_sorcar_md_is_loaded_with_replacement(env: IsolatedKissHome) -> None:
    """Undecodable bytes are replaced; the readable memory still reaches the model."""
    # "Remember: café" in cp1252 — the 0xE9 byte is invalid UTF-8.
    (env.kiss_home / "SORCAR.md").write_bytes(b"# Memory\nRemember: caf\xe9 rule\n")
    model = _Recorder()
    server = StandInModelServer(model)
    try:
        result = SorcarAgent("audit0902-sorcarmd").run(
            prompt_template="say done",
            model_name=STANDIN_MODEL,
            model_config=server.model_config,
            work_dir=str(env.repo),
            web_tools=False,
            is_parallel=False,
            verbose=False,
        )
    finally:
        server.stop()
    payload = yaml.safe_load(result)
    assert payload["success"] is True, result
    assert model.requests, "the model was never called"
    prompt_text = request_text(model.requests[0])
    assert "Remember: caf" in prompt_text
    assert "rule" in prompt_text

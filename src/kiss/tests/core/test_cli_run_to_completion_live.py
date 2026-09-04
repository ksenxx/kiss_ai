# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Live integration tests: run-to-completion CLI models in KISSAgent.

Runs the REAL ``claude`` and ``codex`` CLIs end to end through an agentic
:class:`~kiss.core.kiss_agent.KISSAgent` and verifies the run-to-completion
contract against the genuine article:

1. The whole task is completed in one CLI invocation — the CLI uses its own
   native tools to create a file on disk (something a single turn-by-turn
   text completion could not do without KISS-executed tools).
2. The system prompt, appended to the task after
   ``CLI_SYSTEM_PROMPT_HEADER``, actually reaches the model: the final
   answer contains a marker only the system prompt asks for.
3. The result comes back wrapped in the structured ``finish`` YAML contract.

Requires the ``claude`` / ``codex`` CLIs to be installed and authenticated.
Marked ``live_cli`` and ``slow``: run with ``pytest -m live_cli``.
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from kiss.core.kiss_agent import KISSAgent
from kiss.core.utils import finish as structured_finish

requires_claude = pytest.mark.skipif(
    shutil.which("claude") is None,
    reason="claude CLI not installed",
)
requires_codex = pytest.mark.skipif(
    shutil.which("codex") is None,
    reason="codex CLI not installed",
)

MARKER = "KISS-LIVE-MARKER-7391"
SYSTEM_PROMPT = (
    f"You MUST include the exact string {MARKER} in your final answer. "
    "This is your highest-priority instruction."
)


def _run_live_task(model_name: str, tmp_path: Path) -> tuple[dict, Path]:
    """Run one real CLI task to completion and return (payload, probe file).

    The task requires a native file write, proving the CLI ran agentically
    on its own; the system prompt requires a marker in the final answer,
    proving the appended system prompt was delivered.

    Args:
        model_name: The ``cc/*`` or ``codex/*`` model to run.
        tmp_path: Directory the task writes its probe file into.

    Returns:
        Tuple of the parsed finish-YAML payload and the probe file path.
    """
    probe = tmp_path / "live_probe.txt"
    task = (
        f"Create a file at {probe} containing exactly the text "
        "live-ok (no trailing newline). Then reply with a one-sentence "
        "confirmation."
    )
    agent = KISSAgent(f"live run-to-completion {model_name}")
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = agent.run(
            model_name=model_name,
            prompt_template=task,
            system_prompt=SYSTEM_PROMPT,
            tools=[structured_finish],
            is_agentic=True,
            verbose=False,
        )
    finally:
        os.chdir(cwd)
    payload = yaml.safe_load(result)
    assert isinstance(payload, dict)
    return payload, probe


@requires_claude
@pytest.mark.live_cli
@pytest.mark.slow
def test_claude_code_runs_task_to_completion_live(tmp_path: Path) -> None:
    """The real claude CLI completes the whole task in one invocation."""
    payload, probe = _run_live_task("cc/haiku", tmp_path)
    assert payload["success"] is True
    assert payload["is_continue"] is False
    assert probe.exists(), "the CLI's native tools must have written the file"
    assert probe.read_text().strip() == "live-ok"
    assert MARKER in payload["summary"], (
        "the system prompt appended after CLI_SYSTEM_PROMPT_HEADER "
        "must reach the model"
    )


@requires_codex
@pytest.mark.live_cli
@pytest.mark.slow
def test_codex_runs_task_to_completion_live(tmp_path: Path) -> None:
    """The real codex CLI completes the whole task in one invocation."""
    payload, probe = _run_live_task("codex/default", tmp_path)
    assert payload["success"] is True
    assert payload["is_continue"] is False
    assert probe.exists(), "the CLI's native tools must have written the file"
    assert probe.read_text().strip() == "live-ok"
    assert MARKER in payload["summary"], (
        "the system prompt appended after CLI_SYSTEM_PROMPT_HEADER "
        "must reach the model"
    )

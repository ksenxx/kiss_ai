# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Live integration tests: the real ``claude`` / ``codex`` CLIs as KISS models.

Runs the REAL CLIs end to end through an agentic
:class:`~kiss.core.kiss_agent.KISSAgent` and verifies the KISS-driven
contract against the genuine article:

1. The CLI has no native tools: the file the task asks for is written by
   a KISS tool the model calls through the text-based ``tool_calls``
   protocol (the fake-CLI tests cover the plumbing; this proves the real
   model reads KISS's system prompt and follows the protocol).
2. The system prompt (``--system-prompt-file`` for claude, the
   ``model_instructions_file`` for codex) reaches the model: the final
   answer carries a marker only the system prompt asks for.
3. The result comes back wrapped in the structured ``finish`` YAML contract.

Requires the ``claude`` / ``codex`` CLIs to be installed and logged in, or
``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY`` set: the tests then bill the API
key with the subscription switch off (codex reads it as ``CODEX_API_KEY``).  Marked
``live_cli`` and ``slow``: run with ``pytest -m live_cli``.
"""

import os
import shutil
from pathlib import Path

import pytest
import yaml

from kiss.core import config as config_module
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
    f"You MUST include the exact string {MARKER} in the summary_in_html of your "
    "finish call. This is your highest-priority instruction."
)


def write_probe(path: str, text: str) -> str:
    """Write *text* to the file at *path*.

    Args:
        path: Absolute path of the file to create.
        text: The file's content.

    Returns:
        A confirmation naming the file.
    """
    Path(path).write_text(text)
    return f"wrote {path}"


def _run_live_task(model_name: str, tmp_path: Path) -> tuple[dict, Path]:
    """Run one real CLI task through KISS and return (payload, probe file).

    A run that wrote the file but dropped the marker (an instruction the
    model may still disobey) gets one more attempt in a fresh directory;
    a second miss is reported as the plumbing failure it would be.

    Args:
        model_name: The ``cc/*`` or ``codex/*`` model to run.
        tmp_path: Directory the task writes its probe file into.

    Returns:
        Tuple of the parsed finish-YAML payload and the probe file path.
    """
    payload, probe = _run_live_task_once(model_name, tmp_path / "attempt0")
    if MARKER not in str(payload.get("summary", "")) and probe.exists():
        payload, probe = _run_live_task_once(model_name, tmp_path / "attempt1")
    return payload, probe


def _run_live_task_once(model_name: str, work_dir: Path) -> tuple[dict, Path]:
    """One attempt of :func:`_run_live_task` in the fresh directory *work_dir*."""
    work_dir.mkdir()
    probe = work_dir / "live_probe.txt"
    task = (
        f"Use the write_probe tool to create the file {probe} containing exactly "
        "the text live-ok. Then call finish with a one-sentence confirmation."
    )
    agent = KISSAgent(f"live kiss-driven {model_name}")
    cwd = os.getcwd()
    os.chdir(work_dir)
    try:
        result = agent.run(
            model_name=model_name,
            prompt_template=task,
            system_prompt=SYSTEM_PROMPT,
            tools=[structured_finish, write_probe],
            is_agentic=True,
            max_steps=6,
            verbose=False,
        )
    finally:
        os.chdir(cwd)
    payload = yaml.safe_load(result)
    assert isinstance(payload, dict)
    return payload, probe


def _assert_live_contract(payload: dict, probe: Path) -> None:
    assert payload["success"] is True
    assert payload["is_continue"] is False
    assert probe.exists(), "the KISS tool the model called must have written the file"
    assert probe.read_text().strip() == "live-ok"
    assert MARKER in payload["summary"], "the system prompt must reach the model"


@requires_claude
@pytest.mark.live_cli
@pytest.mark.slow
@pytest.mark.usefixtures("claude_live_billing")
def test_claude_code_live(tmp_path: Path) -> None:
    """The real claude CLI, without native tools, drives KISS's tools."""
    payload, probe = _run_live_task("cc/haiku", tmp_path)
    _assert_live_contract(payload, probe)


@requires_codex
@pytest.mark.live_cli
@pytest.mark.slow
def test_codex_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The real codex CLI, locked down, drives KISS's tools."""
    from kiss.core.models.codex_model import codex_auth_status

    if not codex_auth_status()["loggedIn"]:
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("codex is not logged in and OPENAI_API_KEY is not set")
        # The switch off feeds OPENAI_API_KEY to the CLI as CODEX_API_KEY.
        monkeypatch.setattr(config_module.DEFAULT_CONFIG, "codex_use_subscription", False)
    payload, probe = _run_live_task("codex/default", tmp_path)
    _assert_live_contract(payload, probe)

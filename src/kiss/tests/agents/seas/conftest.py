# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixtures and helpers shared by the SEA tests."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import _flush_chat_events, _get_db, _rw_lock
from kiss.agents.sorcar.sea_commands import SeaRun
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve

HERMETIC_GIT = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
"""Environment that hides the developer's global and system git config.

Without it a ``~/.gitconfig`` with ``commit.gpgsign = true`` makes every
test commit spawn gpg (and a pinentry that waits for a human), and a
Git for Windows system ``core.autocrlf`` changes which files git
reports as modified.
"""


def git(
    repo: Path, *args: str, env: dict[str, str] | None = None, check: bool = True,
    timeout: float = 60,
) -> str:
    """Run ``git *args`` in *repo* under :data:`HERMETIC_GIT` and return its stripped stdout.

    Args:
        repo: The working tree (or, for ``init``, any directory).
        *args: The git arguments.
        env: Extra environment variables for this call (``GIT_COMMITTER_DATE`` ...).
        check: Raise :exc:`subprocess.CalledProcessError` on a non-zero exit (default).
        timeout: Seconds after which the call fails with :exc:`subprocess.TimeoutExpired`.
    """
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check,
        timeout=timeout, env={**os.environ, **HERMETIC_GIT, **(env or {})},
    ).stdout.strip()


def run_sea(
    run: SeaRun, script: list[bytes], work_dir: Path, *,
    agent: ChatSorcarAgent | None = None, max_steps: int = 8, **overrides: Any,
) -> tuple[Any, list[dict[str, Any]]]:
    """Run *run* (an :func:`evaluate_sea` result) against the scripted local model.

    The agent is configured the way the daemon configures a SEA run: the
    resolved settings (``tool_profile``, ``max_budget``, ``use_web_tools``,
    ``use_memory``) and the four hooks of *run*.  *overrides* are passed to
    :meth:`ChatSorcarAgent.run` last (``tool_profile="bash"``,
    ``is_parallel=False`` ...); *agent* replaces the default one for a
    caller that registers it first.

    Returns:
        The parsed YAML result and the agentic requests (those offering tools).
    """
    settings = run.settings
    kwargs: dict[str, Any] = {
        "prompt_template": run.prompt,
        "model_name": MODEL,
        "work_dir": str(work_dir),
        "max_steps": max_steps,
        "max_budget": settings.get("max_budget", 1.0),
        "system_prompt_hook": run.system_prompt_hook,
        "tools_hook": run.tools_hook,
        "tool_call_hook": run.tool_call_hook,
        "llm_call_hook": run.llm_call_hook,
        "tool_profile": settings.get("tool_profile", ""),
        "web_tools": settings.get("use_web_tools", False),
        "use_memory": settings.get("use_memory", False),
        "verbose": False,
        **overrides,
    }
    if agent is None:
        agent = ChatSorcarAgent("sea-test")
    with serve(script) as (url, requests):
        result = agent.run(model_config={"base_url": url, "api_key": "local"}, **kwargs)
    return yaml.safe_load(result), [r for r in requests if r.get("tools")]


@pytest.fixture
def empty_history() -> None:
    """Start the test with no rows in the session's shared task history.

    The whole pytest process shares one ``KISS_HOME`` (see the top-level
    ``conftest.py``), so a test that lists "every recent run" would also
    see the rows earlier tests persisted, in whatever order they ran.
    Earlier agent runs also leave ``events`` rows that reference their
    ``task_history`` rows (the foreign key is enforced), so the writer
    thread is drained and the events go first, under the write lock the
    writer also takes.
    """
    _flush_chat_events()
    db = _get_db()
    with _rw_lock.write_lock():
        db.execute("DELETE FROM events")
        db.execute("DELETE FROM task_history")
        db.commit()

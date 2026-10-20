# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Fixtures and helpers shared by the SEA tests."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from kiss.agents.seas.autorouter.autorouter_sea import Tiers
from kiss.agents.seas.recommendations.recommendations_sea import recommendations_file
from kiss.agents.sorcar import sea_commands
from kiss.agents.sorcar.chat_sorcar_agent import ChatSorcarAgent
from kiss.agents.sorcar.persistence import (
    _add_task,
    _append_chat_event,
    _flush_chat_events,
    _get_db,
    _rw_lock,
    _save_task_result,
)
from kiss.agents.sorcar.sea_commands import SeaRun
from kiss.core.config import kiss_home
from kiss.core.models.model_info import get_available_models
from kiss.server import agent_state
from kiss.tests.agents.sorcar.local_model_server import MODEL, serve

HERMETIC_GIT = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_SYSTEM": os.devnull}
"""Environment that hides the developer's global and system git config.

Without it a ``~/.gitconfig`` with ``commit.gpgsign = true`` makes every
test commit spawn gpg (and a pinentry that waits for a human), and a
Git for Windows system ``core.autocrlf`` changes which files git
reports as modified.
"""
HERMETIC_GIT_WITH_IDENTITY = {
    **HERMETIC_GIT,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@t",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@t",
}
"""`HERMETIC_GIT` plus a fixed identity, for commits made by the SEA under test.

The SEA (or a daemon it runs in) spawns ``git commit`` with the process
environment, not through :func:`git`; on a machine without a global
``user.name`` (a fresh CI VM) that fails with "Committer identity
unknown" unless the identity is exported.
"""


def runnable_candidates(tiers: Tiers, tier: str) -> list[str]:
    """Return the models of *tiers*[*tier*] whose provider credential is configured, in order."""
    available = set(get_available_models())
    return [model for model, _ in tiers[tier] if model in available]


def gate(report: str, name: str) -> str:
    """Return the header line of gate *name* in a paper SEA's *report*."""
    return next(line for line in report.splitlines() if line.startswith(f"{name}: "))


def git(
    repo: Path,
    *args: str,
    env: dict[str, str] | None = None,
    check: bool = True,
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
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=check,
        timeout=timeout,
        env={**os.environ, **HERMETIC_GIT, **(env or {})},
    ).stdout.strip()


def seed_repo(repo: Path) -> None:
    """Create a git repository at *repo*, on branch ``main``, with one committed ``seed.txt``.

    The repository gets its own identity and ``commit.gpgsign = false``
    for the commits the SEA under test makes in it with the process
    environment; the test's own calls go through :func:`git`.
    """
    repo.mkdir(parents=True, exist_ok=True)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "user.name", "Test User")
    git(repo, "config", "commit.gpgsign", "false")
    commit(repo, "seed", {"seed.txt": "seed\n"})


def remote_and_clones(root: Path, *names: str) -> list[Path]:
    """Create a bare ``origin.git`` under *root* with one commit on ``main``; return one
    clone ``root/<name>`` per name, in order."""
    origin = root / "origin.git"
    git(root, "init", "-q", "--bare", "-b", "main", str(origin))
    seed = root / "seed"
    git(root, "clone", "-q", str(origin), str(seed))
    git(seed, "checkout", "-q", "-b", "main")
    commit(seed, "README.md: hello", {"README.md": "hello\n"})
    git(seed, "push", "-q", "origin", "main")
    clones = []
    for name in names:
        clone = root / name
        git(root, "clone", "-q", str(origin), str(clone))
        clones.append(clone)
    return clones


def hermetic_git_env(env: pytest.MonkeyPatch) -> None:
    """Export :data:`HERMETIC_GIT_WITH_IDENTITY` through *env* for the SEA (or daemon) under
    test, which spawns git with the process environment; *env*'s undo restores it."""
    for key, value in HERMETIC_GIT_WITH_IDENTITY.items():
        env.setenv(key, value)


@pytest.fixture
def reset_registry() -> Iterator[None]:
    """Reset the SEA registry before and after the test so a rescan is honest.

    Also removes any ``SEAS.md`` the test wrote into ``$KISS_HOME`` so
    it cannot shadow bundled commands for later tests.  A test that
    points ``KISS_HOME`` elsewhere requests this after ``monkeypatch``
    so the registry is empty again before the real home comes back.
    """
    sea_commands._reset_for_tests()
    yield
    sea_commands._reset_for_tests()
    (kiss_home() / "SEAS.md").unlink(missing_ok=True)


@pytest.fixture
def recommendations_path() -> Iterator[Path]:
    """Start and end the test without the stored recommendations file; yield its path.

    The file lives in the session-wide ``KISS_HOME``: a test that wrote
    it and failed before its own clean-up would otherwise leak it into
    the next test reading "the stored recommendations".
    """
    path = recommendations_file()
    path.unlink(missing_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    yield path
    path.unlink(missing_ok=True)


def run_sea(
    run: SeaRun,
    script: list[bytes],
    work_dir: Path,
    *,
    agent: ChatSorcarAgent | None = None,
    max_steps: int = 8,
    **overrides: Any,
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


@contextmanager
def registered(
    agent: ChatSorcarAgent, task_id: str, **fields: Any
) -> Iterator[agent_state.AgentState]:
    """Register *agent* as the calling thread's running task *task_id* for the block.

    The daemon registers a run under its task thread before starting it
    (``commands.py``); that registration is how a SEA's tools find their
    task through ``current_agent`` (its text, work dir, question callback,
    persisted id).  *fields* fill other :class:`AgentState` fields
    (``tab_id=``, ``is_task_active=``).  The state is unregistered when
    the block ends, however it ends.
    """
    state = agent_state.AgentState(
        task_id, agent=agent, task_thread=threading.current_thread(), **fields
    )
    agent_state.register(state)
    try:
        yield state
    finally:
        agent_state.unregister(task_id, state)


def commit(repo: Path, message: str, files: dict[str, str] | None = None, when: int = 0) -> str:
    """Write *files* (path -> text) into *repo*, commit everything with *message* and return
    the commit sha; *when* (a unix time) sets the committer date.  An empty commit is allowed."""
    for name, text in (files or {}).items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    git(repo, "add", "-A")
    env = {"GIT_COMMITTER_DATE": f"@{when} +0000"} if when else None
    git(repo, "commit", "-q", "--allow-empty", "-m", message, env=env)
    return git(repo, "rev-parse", "HEAD")


def fake_checkout(root: Path, seas: dict[str, str]) -> Path:
    """Lay out a KISS checkout's ``src/kiss/agents/seas`` under *root* and return that folder.

    *seas* maps a SEA name to the source of its ``<name>/<name>_sea.py``;
    rsi7d's editors and miners find the folder through the cwd, so the
    caller also chdirs to *root*.
    """
    folder = root / "src" / "kiss" / "agents" / "seas"
    for name, source in seas.items():
        (folder / name).mkdir(parents=True)
        (folder / name / f"{name}_sea.py").write_text(source, encoding="utf-8")
    return folder


def persist_task(
    prompt: str,
    events: Sequence[dict[str, Any]] = (),
    result: str | None = None,
    *,
    success: bool | None = None,
    timestamp: float | None = None,
    **columns: object,
) -> str:
    """Persist a finished task row with *events* and *result* and return its id.

    The row looks like one the daemon wrote for a one-minute, five-step
    run of ``model-a`` in ``/work/dir`` that cost $0.5; *columns*
    override or extend those payload fields (``sea=``,
    ``parent_task_id=``, ``startTs=`` ...).  *events* are appended and
    flushed through the writer thread.  A *result* of ``None`` leaves
    the row unfinished (no outcome recorded); any other text, the empty
    string included, is saved as the task's result with the outcome
    *success*, as the daemon does for a run that finished with nothing
    to say.  *timestamp* (epoch seconds) backdates the row's
    ``timestamp`` column, which the insert stamps with the current time.
    """
    now_ms = int(time.time() * 1000)
    payload: dict[str, object] = {
        "model": "model-a",
        "work_dir": "/work/dir",
        "startTs": now_ms - 60_000,
        "endTs": now_ms,
        "cost": 0.5,
        "steps": 5,
        "tokens": 10_000,
        **columns,
    }
    task_id, _chat = _add_task(prompt, extra=payload)
    for event in events:
        _append_chat_event(event, task_id=task_id)
    _flush_chat_events(task_id)
    if result is not None:
        _save_task_result(result, task_id=task_id, success=success)
    if timestamp is not None:
        db = _get_db()
        with _rw_lock.write_lock():
            db.execute("UPDATE task_history SET timestamp = ? WHERE id = ?", (timestamp, task_id))
            db.commit()
    return task_id


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

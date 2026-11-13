# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Run a SEA on a task in this process: no daemon, no UI, every file the run wrote known.

:func:`run_sea` is the supported entry point for a benchmark harness or
a training loop that runs a SEA headless::

    from kiss.agents.sorcar.headless import run_sea

    outcome = run_sea("weather", "What is the weather in Paris?", max_budget=0.5)
    print(outcome.success, outcome.result)
    for path in outcome.trajectory_paths:
        path.unlink()

The SEA is loaded and evaluated exactly as the daemon does it
(:func:`~kiss.agents.sorcar.sea_commands.sea_layers`,
:func:`~kiss.agents.sorcar.sea_commands.evaluate_sea`) and runs through
:class:`~kiss.agents.sorcar.sorcar_agent.SorcarAgent` with the SEA's
settings, prompt and hooks, so what the model sees is what it sees in a
chat tab (less the tab's own settings, which the SEA may pin).  The
result carries the run's counters and the path of every trajectory
file the run saved — the agent's own and those of the helper agents it
created (session executors, the summarizer, the task classifier, ...)
— collected as they were written
(:func:`~kiss.core.base.collecting_trajectories`), not guessed from a
naming convention; ``save_trajectories=False`` writes none.

Needs an API key in the environment or in ``$KISS_HOME/api_keys.env``
(loaded by :func:`run_sea`, which also applies the settings saved in
``$KISS_HOME/config.json`` — the default budget, the global Compaction
switch — as the daemon does).  The ``run_agent`` / ``run_parallel``
tools need the daemon; a SEA meant for headless runs leaves the
``agents`` tool group out of its ``tool_profile``.
``scripts/run_sea.py`` is the command-line form.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from kiss.agents.sorcar.sea_apply import NO_TOOLS_PROFILE
from kiss.agents.sorcar.sea_commands import evaluate_sea, get_command, list_commands, sea_layers
from kiss.agents.sorcar.sea_settings import SeaError, anchored_work_dir
from kiss.agents.sorcar.sorcar_agent import SorcarAgent, canonical_tool_profile
from kiss.core.base import collecting_trajectories
from kiss.core.printer import Printer
from kiss.core.user_config import apply_config_to_env, load_api_keys_readonly, load_config

logger = logging.getLogger("kiss.headless")


@dataclass(frozen=True)
class SeaRunResult:
    """What :func:`run_sea` returns: the run's outcome, its counters and its files.

    Attributes:
        result: The run's result text: Sorcar's ``finish`` YAML
            (``success``, ``is_continue``, ``summary``, ...), the text a
            hook's ``end_run`` gave, or ``""`` when the run raised.
        success: The ``success`` of the result; ``False`` when the run
            raised or the result is not the ``finish`` YAML.
        error: ``""`` when the run returned; else the exception that
            ended it, as ``ExceptionType: message``.
        budget_used: USD the run spent, helper agents included.
        total_tokens_used: Tokens the run used, helper agents included.
        total_steps: Model calls the run made, helper agents included.
        trajectory_paths: Every trajectory file the run saved, in the
            order of the saves; empty under ``save_trajectories=False``.
    """

    result: str
    success: bool
    error: str
    budget_used: float
    total_tokens_used: int
    total_steps: int
    trajectory_paths: tuple[Path, ...]


def resolve_sea(sea: str | Path) -> Path:
    """Return the SEA file for *sea*, a SEA file path or a registered SEA name.

    A path wins when it names an existing file; otherwise the name is
    looked up in the SEA registry the daemon uses (the bundled ``seas/``
    folder, the folders of ``$KISS_HOME/SEAS.md``, ``third_party_agents/``).

    Raises:
        SeaError: When neither matches; the message lists the known names.
    """
    path = Path(sea)
    if path.is_file():
        return path.resolve()
    found = get_command(str(sea))
    if found is None:
        raise SeaError(f"unknown SEA {str(sea)!r}; known names: {', '.join(list_commands())}")
    return found


def run_sea(
    sea: str | Path,
    task: str,
    *,
    model: str = "",
    work_dir: str = "",
    max_budget: float | None = None,
    save_trajectories: bool = True,
    verbose: bool = False,
    printer: Printer | None = None,
) -> SeaRunResult:
    """Run the SEA *sea* on *task* in this process and return the outcome.

    Args:
        sea: A registered SEA name (``sh``, ``weather``) or the path of a
            SEA file (``<name>/<name>_sea.py``).
        task: The task text; the SEA's ``prompt(task)`` shapes it.
        model: The model when the SEA's ``model`` setting names none
            (default: the configured default model).
        work_dir: The caller's directory (default: the current one): the
            working directory when the SEA's ``work_dir`` setting names
            none, and the one a relative setting is resolved against.
        max_budget: USD budget when the SEA's ``max_budget`` setting
            names none (default: Sorcar's default budget).
        save_trajectories: Whether the run writes trajectory files; with
            ``False`` there is nothing to delete afterwards.
        verbose: Whether the run prints to the console (when *printer*
            is ``None``).
        printer: The printer the run's events go to, if any.

    Returns:
        The :class:`SeaRunResult`.  A run that raises is returned with
        its ``error`` set, never raised.

    Raises:
        SeaError: When *sea* names no SEA, or the SEA is broken (a method
            raised or returned a value of the wrong type).
    """
    load_api_keys_readonly()
    # The persisted settings (``config.json``: the default budget, the
    # subscription switches, the global Compaction switch), as the daemon
    # applies them at start-up: a headless process otherwise runs with
    # the declared defaults whatever the user saved.
    apply_config_to_env(load_config())
    sea_path = resolve_sea(sea)
    run = evaluate_sea(sea_layers(sea_path), task)
    settings = run.settings
    model_name = settings.get("model") or model or None
    model_config = settings.get("model_config")
    base_dir = work_dir or os.getcwd()
    pinned_dir = settings.get("work_dir")
    with collecting_trajectories(save=save_trajectories) as scope:
        agent = SorcarAgent(sea_path.stem)
        result, error = "", ""
        try:
            tool_profile = canonical_tool_profile(settings.get("tool_profile", ""))
            # As in the daemon, ``none`` strips the run to ``finish`` plus
            # what the SEA's ``tools()`` returns.
            append_basic_tools = tool_profile != NO_TOOLS_PROFILE
            # The SEA's ``auto_classify`` setting, as the daemon honours it:
            # the verdict is pre-seeded so the run neither classifies again
            # nor falls back to the persisted setting when the SEA fixed it.
            agent.classify_task_for_run(
                model_name, run.prompt, model_config, enabled=settings.get("auto_classify"),
                tool_profile=tool_profile, append_basic_tools=append_basic_tools,
            )
            result = agent.run(
                model_name=model_name,
                prompt_template=run.prompt,
                work_dir=anchored_work_dir(pinned_dir, base_dir) if pinned_dir else base_dir,
                max_budget=settings.get("max_budget", max_budget),
                tool_profile=tool_profile,
                append_basic_tools=append_basic_tools,
                web_tools=bool(settings.get("use_web_tools", False)),
                use_memory=settings.get("use_memory"),
                compaction=settings.get("compaction", True),
                model_config=model_config,
                docker_image=settings.get("docker_image") or None,
                system_prompt_hook=run.system_prompt_hook,
                tools_hook=run.tools_hook,
                llm_call_hook=run.llm_call_hook,
                tool_call_hook=run.tool_call_hook,
                llm_response_hook=run.llm_response_hook,
                printer=printer,
                verbose=verbose if printer is None else None,
            )
        except Exception as exc:
            logger.warning("Headless run of %s failed", sea_path.stem, exc_info=True)
            error = f"{type(exc).__name__}: {exc}"
    budget_used, total_tokens_used, total_steps = agent.usage_snapshot()
    return SeaRunResult(
        result,
        _result_success(result),
        error,
        budget_used,
        total_tokens_used,
        total_steps,
        tuple(scope.paths),
    )


def _result_success(result: str) -> bool:
    """Return the ``success`` of a ``finish`` YAML *result*; ``False`` for anything else."""
    try:
        payload = yaml.safe_load(result)
    except yaml.YAMLError:
        return False
    return isinstance(payload, dict) and payload.get("success") is True

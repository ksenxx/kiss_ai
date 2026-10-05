# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Coding SEA: runs KISS Sorcar unattended inside a Docker container.

A trial runs the agent against one running Docker container that holds the
task's working directory.  The KISS daemon imports a tiny generated SEA file
per trial (see :func:`write_trial_sea`) that builds one
:class:`ContainerHarness` from a JSON config and exposes the harness's bound
methods as the SEA getters.  The trial runners in ``benchmarkings/harnesstax``
are one such caller.  This module itself defines no prompt getters: a
harness needs a running container and a config, so the module is
``hidden`` and registers no ``/coding`` chat command.

The harness is KISS Sorcar's full built-in toolset except the browser and
persistent-memory tools, driven by a short system prompt written for
unattended engineering work in a container (:data:`SYSTEM_PROMPT`).  The shell and
file tools execute inside the trial container through the daemon's Docker mode
(``settings()["docker_image"] == "container:<id>"``), so nothing the agent does
touches the host.  There is no cap on the number of steps; the only limit is the
per-trial USD budget.  Every LLM call and every tool call is appended to a
JSONL trajectory file for later analysis.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kiss.agents.seas.coding import coding_test_context as test_context
from kiss.agents.sorcar.shell_guards import (
    INSTALL_COMMANDS,
    INSTALL_TIMEOUT_SECONDS,
    destructive_pattern,
)

#: Harness instances by config path (see :meth:`ContainerHarness.shared`).
_HARNESSES: dict[str, ContainerHarness] = {}
_REGISTRY_LOCK = threading.Lock()

#: The system prompt of an unattended, containerised engineering task.  It
#: replaces Sorcar's default prompt, which is written for interactive work on
#: the user's own machine (worktrees, reports, periodic summaries, web
#: research) and says nothing about how to solve a task nobody will check.
SYSTEM_PROMPT = """\
You are KISS Sorcar, an autonomous software engineer. You work alone inside a \
Linux container whose working directory is {workdir}. Nobody reads anything or \
answers questions before you call finish. Complete the task in the user message \
yourself, end to end, and leave the container in the state the task asks for.

# Tools
- Bash runs one shell command in the container; every call is a fresh shell \
started in {workdir}, so use absolute paths or `cd` inside the command. \
timeout_seconds defaults to 30: pass a larger value for a command you expect to \
take longer (package installs and builds: at least 900). run_commands_parallel \
runs several independent commands at once (its timeout applies to each \
command). There is no background-job tool: run a long command as \
`nohup cmd > /tmp/x.log 2>&1 &` in one call and poll the log with tail (after \
a bounded sleep) in later calls.
- Read shows a file with line numbers (use start_line and max_lines on large \
files). Edit replaces one exact string in a file; Write creates or overwrites \
a whole file. Read the region you change before you Edit it.{test_context}
- finish ends the run. Set success=True only when every requirement is met \
and verified; otherwise success=False with what remains. summary_in_html is a \
brief description of what you did and how you verified it. There is no user \
to ask (ask_user_question, talk), the model is fixed (set_model), and there is \
no need for progress summaries.
- Shell results may end with harness notes about facts you cannot otherwise \
see: processes a command started that are still running, and pre-existing \
files under {workdir} that a command modified or deleted. Act on them.

# How to work
1. Understand the task before acting. Read the whole statement and list what \
it asks for: the files, names, formats, commands, interfaces, thresholds and \
constraints it names, and what "done" means for each of them. Take the \
statement literally; where it is ambiguous, prefer the most conventional \
reading and say which one you chose in your final summary.
2. Look before you change. Inspect the working directory (including hidden \
files), the files the task mentions, and the tools and versions available. In \
a code base, find the relevant code with grep, read it and its callers, and \
follow the conventions you find there. Do not modify or delete the task's \
input files; experiment on copies.
3. Work in small verified steps. For a bug, reproduce it first, fix the cause \
rather than the symptom, then confirm that the reproduction passes and that \
the existing tests still do. For something to build or produce, get a \
complete working version in place first and improve it afterwards; do not \
replace something that works with something unverified. Keep changes minimal \
and in the style of the surrounding code. Install standard tools and \
libraries rather than re-implementing them.
4. Verify before you finish. Re-read the task statement and check every \
requirement directly, from a fresh shell: run the command, open the file, \
parse the output, measure the number. Run the tests that cover what you \
changed. Treat any error, traceback or unexpected output in your own checks \
as a problem to resolve, not to explain away. Never weaken a test, assertion \
or threshold to make a check pass, and never repeat a failed command without \
changing something first.
5. Leave the environment as the task expects to find it. Keep the finished \
work in place rather than undoing or resetting it, and do not leave behind \
stray processes or scratch files that would confuse whoever uses the \
container next. Kill only processes you started.
6. Be honest at the end. Call finish with success=True only when every \
requirement is met and you have checked it; otherwise call it with \
success=False and a precise account of what is done and what remains.

# Time and efficiency
Probe with short timeouts. Start anything that may take more than a minute \
(builds, installs, training, downloads) in the background with nohup and poll \
its log with bounded sleeps; use all cores where possible (`-j$(nproc)`). Never \
block on one command for more than a few minutes. Keep tool output small \
(grep -n, head, tail, line ranges), batch independent commands into one call, \
and do not re-read unchanged files. Do the task yourself in one focused pass; \
sub-agents are rarely worth their cost here.
"""

#: Answer to the first ``finish(success=True)`` of a trial when the finish gate
#: is on (config ``finish_gate``): one last verification pass before the run
#: may end.  The second ``finish`` is accepted.
FINISH_GATE_VERDICT = (
    "Not finished yet: do one final verification pass, then call finish again. "
    "(1) Re-read the task statement and re-run its decisive end-to-end check from a "
    "fresh shell. (2) Go back over every check you ran in this session and resolve "
    "each error, traceback or unexpected output that appeared, or state why it cannot "
    "affect the task. (3) Confirm that the finished work is in place and that nothing "
    "you verified was undone afterwards. Fix what fails; if it cannot be fixed, finish "
    "with success=False and say what remains."
)

#: Shell fragments that would end the trial itself: killing every process in
#: the container (the agent's own shell included) or deleting the workspace.
#: ``_END`` closes a shell word: whitespace, end of command, a separator,
#: a redirection or a comment.  ``_CMD`` is where a command name may start:
#: the beginning or a separator, then optional ``sudo`` or environment
#: assignments and an optional directory prefix.
#: The patterns live in :mod:`kiss.agents.sorcar.shell_guards`, shared with the
#: host and Docker ``Bash`` tools; the harness applies them in its tool hook
#: because a HarnessTax container's tools run outside :class:`DockerManager`.
DESTRUCTIVE_VERDICT = (
    "Blocked: this command would kill every process in the container (your own "
    "shell included) or delete the working directory, which ends the task with "
    "nothing to deliver. Kill only the processes you started, by pid or exact name, "
    "and delete only files you created."
)

#: Largest file the test-context note reads back and diffs.
MAX_SNAPSHOT_BYTES = 400_000

#: Sentence added to the system prompt when the test-context notes are on.
TEST_CONTEXT_NOTE = (
    " After you change existing source code, the tool result may list the "
    "project's tests that reference the changed definitions: they define the "
    "expected behaviour, so read the relevant ones and run them."
)


#: Tools that need a human or the outside world (``run_agent`` would dispatch
#: a host-side channel agent); a headless run answers them itself instead of
#: blocking forever (the run has no wall-clock limit) or leaving the container.
UNATTENDED_TOOL_VERDICTS = {
    "ask_user_question": (
        "No user is available in this unattended run. Decide for yourself, "
        "state your assumption in the final summary, and continue."
    ),
    "talk": "No user is listening in this unattended run; continue with the task.",
    "run_agent": (
        "No external agents or messaging channels exist in this unattended, "
        "container-isolated run; do the work yourself with the other tools."
    ),
    "set_model": (
        "The model is fixed for this run (the run measures one model); continue "
        "with the current one."
    ),
}

#: Lists every process in the container as ``pid ppid starttime cmdline`` from
#: /proc alone (``ps`` is missing from many task images); the start time (stat
#: field 22) tells a reused pid from the process that had it.  ``KISS_PS=1``
#: tags the snapshot's own shell so it and its children can be dropped.
PROCESS_LIST_COMMAND = (
    "KISS_PS=1; for d in /proc/[0-9]*; do p=${d#/proc/}; "
    "s=$(sed 's/.*) //' \"$d/stat\" 2>/dev/null | cut -d' ' -f2,20); "
    "c=$(tr '\\0' ' ' < \"$d/cmdline\" 2>/dev/null | head -c 200); "
    "[ -n \"$c\" ] && [ -n \"$s\" ] && printf '%s %s %s\\n' \"$p\" \"$s\" \"$c\"; done; true"
)
#: Shell commands that try to stop a process; the survivors note is repeated
#: after them while a process the agent started is still alive.
KILL_COMMANDS = re.compile(r"\b(?:kill|pkill|killall|fuser|pgrep)\b|\bstop\b")
#: Regular files under the workdir as ``size mtime path`` (``.git``, bytecode
#: and caches excluded).  The listing is capped so a huge tree just switches
#: the input-file notes off instead of slowing every turn.
MAX_TRACKED_FILES = 40_000
FILE_LIST_COMMAND = (
    "find {workdir} -xdev \\( -name .git -o -name __pycache__ -o -name .pytest_cache "
    "-o -name node_modules -o -name .tox -o -name .mypy_cache \\) -prune -o -type f "
    "! -name '*.pyc' -printf '%s %T@ %p\\n' 2>/dev/null | head -n " + str(MAX_TRACKED_FILES + 1)
)
#: Longest ``ls -la`` shown with the task statement.
MAX_LISTING_LINES = 60
#: Processes started by the last shell call and still alive when it returned.
SURVIVORS_NOTE = (
    "\n\nHarness note: processes started by your last shell command(s) are still running: {procs}. "
    "Before you finish, stop the ones that are not meant to keep running."
)
#: Processes the agent started earlier that outlived a command meant to stop something.
STILL_RUNNING_NOTE = (
    "\n\nHarness note: processes you started earlier are still running after your last shell "
    "command(s): {procs}."
)
#: Pre-existing files a shell command changed or deleted.
INPUT_FILES_NOTE = (
    "\n\nHarness note: your last shell command(s) changed pre-existing files under {workdir}: "
    "{files}. "
    "Make sure this is a change the task asks for; if not, restore them and work on copies."
)


def description() -> str:
    """Return the one-sentence help text shown by ``/coding help``."""
    return (
        "Runs KISS Sorcar unattended inside a Docker container for benchmark trials "
        "(a ContainerHarness built by write_trial_sea from a JSON config, with the shell "
        "and file tools executing in the trial container and every call logged to a JSONL "
        "trajectory); the trial runners load the generated per-trial SEA file by path, so "
        "this module is hidden from the command list."
    )


def settings() -> dict[str, Any]:
    """Return the module's own settings: hidden (only the generated per-trial SEAs run)."""
    return {"hidden": True}


class ContainerHarness:
    """One trial: a container, a task prompt, a model, a trajectory log.

    Instances are created from a JSON config written by the trial runner::

        {
          "container": "<docker container name or id>",
          "workdir": "/testbed",
          "prompt": "<task instruction>",
          "model": "claude-fable-5",
          "max_budget": 50.0,
          "work_dir": "/host/scratch/dir",
          "trajectory": "/host/path/trajectory.jsonl"
        }
    """

    def __init__(self, config_path: str) -> None:
        """Load the trial configuration.

        Args:
            config_path: Absolute path of the trial's JSON config.
        """
        cfg = json.loads(Path(config_path).read_text())
        self.config_mtime_ns = Path(config_path).stat().st_mtime_ns
        self.container: str = cfg["container"]
        self.workdir: str = cfg.get("workdir", "/")
        self.task_prompt: str = cfg["prompt"]
        self.model_name: str = cfg["model"]
        self.budget: float = float(cfg.get("max_budget", 50.0))
        self.host_work_dir: str = cfg.get("work_dir", "")
        self.trajectory_path = Path(cfg["trajectory"])
        self.turns = 0
        self._lock = threading.Lock()
        #: Whether edits get the "tests that reference this code" note (off by
        #: default).
        self.test_context: bool = bool(cfg.get("test_context", False))
        #: Whether the first ``finish(success=True)`` is answered with :data:`FINISH_GATE_VERDICT`.
        self.finish_gate: bool = bool(cfg.get("finish_gate", False))
        #: Model request overrides for this trial (``reasoning_effort`` for OpenAI-style
        #: models, ``output_config={"effort": ...}`` for Claude); empty means provider defaults.
        self.model_overrides: dict[str, Any] = dict(cfg.get("model_config") or {})
        self.gate_answered = False
        self.implicit_finish_vetoed = False
        self.destructive = destructive_pattern(self.workdir)
        #: ``(path, content before the edit)`` of the edits made since the last model call.
        self.pending_edits: list[tuple[str, str]] = []
        #: Test files already pointed out to the model.
        self.shown_tests: set[str] = set()
        #: Turn of the last edit (a test-context note is only shown for edits that landed).
        self.last_edit_turn = 0
        #: Process listing taken before the first shell call of the current turn
        #: (``None`` when no shell tool ran since the last model call).
        self.processes_before: dict[tuple[int, str], tuple[int, str]] | None = None
        #: Whether a shell command of the current turn tried to stop a process.
        self.kill_attempted = False
        #: ``(pid, start time)`` of processes the agent started that outlived the call that
        #: started them.
        self.tracked_pids: set[tuple[int, str]] = set()
        #: Shell notes that found no tool result to attach to; retried on the next model call.
        self.pending_notes: list[str] = []
        #: ``path -> (size, mtime)`` of the workdir's files when the run started
        #: (``None`` until the first model call; empty when tracking is off).
        self.file_baseline: dict[str, tuple[str, str]] | None = None
        #: Pre-existing files the agent edited with Edit/Write or that were already reported.
        self.owned_files: set[str] = set()

    @classmethod
    def shared(cls, config_path: str) -> ContainerHarness:
        """Return the process-wide harness for *config_path*, creating it once.

        The daemon imports a self-contained SEA more than once per run; every
        import must see the same turn counter, so instances are cached per
        config path.  A config file rewritten since the cached instance was
        built (a rerun of the same trial directory) replaces the stale instance.

        Args:
            config_path: Absolute path of the trial's JSON config.
        """
        with _REGISTRY_LOCK:
            harness = _HARNESSES.get(config_path)
            if harness is None or harness.config_mtime_ns != Path(config_path).stat().st_mtime_ns:
                harness = cls(config_path)
                _HARNESSES[config_path] = harness
            return harness

    # ---- SEA parameter getters ------------------------------------------

    def prompt(self, task: str = "") -> str:
        """The task instruction shown to the model, followed by ``ls -la`` of the workdir.

        The SEA ``prompt(task)`` getter: the instruction is the config's
        ``prompt`` (the trial runner's ``run_agent`` task text is only a
        label), or *task* when the config has none.  The listing costs
        nothing and settles the first question of every run (what is
        here?), so the model sees the task's own binaries and data from
        the start.
        """
        instruction = self.task_prompt or task
        listing = self.workdir_listing()
        if not listing:
            return instruction
        return (f"{instruction}\n\n---\n`ls -la {self.workdir}` when the run started:\n"
                f"{listing}")

    def workdir_listing(self) -> str:
        """``ls -la`` of the workdir, cut to :data:`MAX_LISTING_LINES` lines ('' on failure)."""
        out = self.container_exec(["ls", "-la", self.workdir])
        if not out:
            return ""
        lines = out.splitlines()
        if len(lines) > MAX_LISTING_LINES:
            lines = lines[:MAX_LISTING_LINES] + [f"[{len(lines) - MAX_LISTING_LINES} more entries]"]
        return "\n".join(lines)

    def system_prompt(self) -> str:
        """The unattended-engineering prompt (replaces Sorcar's default prompt)."""
        tests = TEST_CONTEXT_NOTE if self.test_context else ""
        return SYSTEM_PROMPT.format(workdir=self.workdir, test_context=tests)

    def settings(self) -> dict[str, Any]:
        """The trial's run settings (the SEA ``settings()`` contract).

        A ``worker`` whose sub-agents stay on (they share the trial
        container): the trial's model, hard USD cap and per-trial model
        overrides (``None`` for the provider defaults), the host scratch
        directory the daemon runs the task in (the tools run in the
        container the ``docker_image`` attaches), no host git worktree or
        auto-commit (all edits happen inside the container), no
        pre-run classification, no browser, no persistent memory.
        """
        return {
            "kind": "worker",
            "model": self.model_name,
            "max_budget": self.budget,
            "model_config": self.model_overrides or None,
            "work_dir": self.host_work_dir,
            "docker_image": f"container:{self.container}",
            "allow_fan_out": True,
        }

    def llm_call_hook(self) -> Callable[[list], list]:
        """Return the hook that counts LLM turns and logs the new messages."""
        return self.on_llm_call

    def tool_call_hook(self) -> Callable[[str, dict[str, Any]], str]:
        """Return the hook that logs tool calls and answers the interactive ones."""
        return self.on_tool_call

    # ---- hooks ------------------------------------------------------------

    def on_llm_call(self, new_messages: list) -> list:
        """Count one turn per LLM call and log the messages added since the last call.

        Args:
            new_messages: Messages appended to the conversation since the
                previous LLM call.

        Returns:
            The messages unchanged.
        """
        with self._lock:
            self.turns += 1
            turn = self.turns
        if self.file_baseline is None:  # first model call: the workdir as the task provided it
            try:
                self.file_baseline = self.file_snapshot() or {}
            except Exception as error:
                self._log({"event": "hook_error", "turn": turn, "hook": "llm_call",
                           "error": repr(error)})
                self.file_baseline = {}
        if new_messages:
            try:
                notes = self.test_context_notes(self.landed_edits()) + self.shell_notes()
            except Exception as error:  # a note is optional; the trial goes on without it
                self._log({"event": "hook_error", "turn": turn, "hook": "llm_call",
                           "error": repr(error)})
                notes = []
            with self._lock:
                notes, self.pending_notes = self.pending_notes + notes, []
            for note in notes:
                if not append_to_last_tool_result(new_messages[-1], note):
                    with self._lock:
                        self.pending_notes.append(note)
        self._log({"event": "llm_call", "turn": turn, "new_messages": _jsonable(new_messages)})
        if not self.container_alive():
            # The caller tore the container down (OOM-killed, Docker
            # restarted or the job was stopped): nothing the agent does can count any more, so
            # end the trial instead of letting the session loop retry against
            # a dead container until the budget is gone.  BudgetExceededError is
            # the one exception the Sorcar session loop treats as terminal.
            from kiss.core.kiss_error import BudgetExceededError

            self._log({"event": "container_gone", "turn": turn})
            raise BudgetExceededError(f"Container {self.container} no longer exists; run ended.")
        return new_messages

    def container_alive(self) -> bool:
        """Whether the trial container still exists (checked before every model call)."""
        import docker

        try:
            return bool(_docker_client().containers.get(self.container).status == "running")
        except docker.errors.NotFound:  # type: ignore[attr-defined]
            return False
        except Exception:  # transient daemon hiccup: do not end the trial on it
            return True

    def on_tool_call(self, name: str, args: dict[str, Any]) -> str:
        """Log a tool call; answer tools that need a human without running them.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            ``"OK"`` to let the call run, or the text the model sees instead.
        """
        verdict = UNATTENDED_TOOL_VERDICTS.get(name, "OK")
        try:
            if name in ("Bash", "run_commands_parallel"):
                verdict = self.shell_verdict(args)
            elif name == "finish" and self.finish_gate and not self.gate_answered:
                verdict = self.finish_verdict(args)
        except Exception as error:  # a guard is optional; never end the trial on its own bug
            self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                       "error": repr(error)})
        self._log({"event": "tool_call", "turn": self.turns, "tool": name,
                   "args": _jsonable(args), "blocked": verdict != "OK"})
        if name in ("Edit", "Write") and isinstance(args.get("file_path"), str):
            self.last_edit_turn = self.turns
            path = posixpath.normpath(posixpath.join(self.workdir, args["file_path"]))
            resolved = (self.container_exec(["readlink", "-f", path]) or "").strip()
            with self._lock:
                self.owned_files.update({path, resolved} - {""})
            if self.test_context:
                try:
                    self.snapshot_edit(args["file_path"])
                except Exception as error:
                    self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                               "error": repr(error)})
        if name in ("Bash", "run_commands_parallel") and verdict == "OK":
            try:
                self.before_shell(args)
            except Exception as error:
                self._log({"event": "hook_error", "turn": self.turns, "hook": name,
                           "error": repr(error)})
        return verdict

    def before_shell(self, args: dict[str, Any]) -> None:
        """Remember the process table before the turn's first shell call; note kill attempts.

        Args:
            args: Arguments of the ``Bash`` or ``run_commands_parallel`` call.
        """
        text = json.dumps(args.get("command", args.get("commands", "")))
        with self._lock:
            take = self.processes_before is None
            self.kill_attempted = self.kill_attempted or bool(KILL_COMMANDS.search(text))
        if take:
            snapshot = self.process_snapshot()
            with self._lock:
                if self.processes_before is None:
                    self.processes_before = snapshot
                    # Adopt children born since the last snapshot now, while their
                    # tracked parent is still there to link them.
                    self.tracked_pids |= descendants(snapshot, self.tracked_pids)

    def shell_notes(self) -> list[str]:
        """Notes about what the turn's shell calls left behind: live processes and changed inputs.

        Empty when no shell tool ran since the last model call.  The survivors
        note names processes the calls started that are still alive; it is
        repeated after a command that tried to stop something while a tracked
        process survives.  The input-files note names pre-existing files under
        the workdir that the calls modified or deleted (files the agent edited
        with Edit/Write are its own and are skipped).
        """
        with self._lock:
            before, self.processes_before = self.processes_before, None
            kill_attempted, self.kill_attempted = self.kill_attempted, False
        if before is None:
            return []
        notes: list[str] = []
        after = self.process_snapshot()
        with self._lock:
            # Children of a process already reported (a build's compiler
            # steps, a server's workers) are tracked without a new note.
            offspring = descendants(after, self.tracked_pids)
            new = {key: info for key, info in after.items()
                   if key not in before and key not in offspring and key not in self.tracked_pids}
            self.tracked_pids |= set(new) | offspring
            self.tracked_pids &= set(after)
            alive = {key: after[key] for key in sorted(self.tracked_pids)}
        if new:
            notes.append(SURVIVORS_NOTE.format(procs=describe_processes(new)))
        elif kill_attempted and alive:
            notes.append(STILL_RUNNING_NOTE.format(procs=describe_processes(alive)))
        if self.file_baseline:
            changed = self.changed_input_files()
            if changed:
                notes.append(
                    INPUT_FILES_NOTE.format(workdir=self.workdir, files=", ".join(changed)))
        for note in notes:
            self._log({"event": "shell_note", "turn": self.turns, "note": note.strip()})
        return notes

    def process_snapshot(self) -> dict[tuple[int, str], tuple[int, str]]:
        """``(pid, start time) -> (ppid, cmdline)`` of every process in the container.

        The snapshot's own shell is left out.
        """
        out = self.container_exec(["sh", "-c", PROCESS_LIST_COMMAND]) or ""
        procs: dict[tuple[int, str], tuple[int, str]] = {}
        for line in out.splitlines():
            parts = line.split(" ", 3)
            if len(parts) == 4 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                procs[(int(parts[0]), parts[2])] = (int(parts[1]), parts[3].strip())
        helper = {pid for (pid, _), (_, cmd) in procs.items() if "KISS_PS=1" in cmd}
        grew = True
        while grew:
            children = {pid for (pid, _), (ppid, _) in procs.items()
                        if ppid in helper and pid not in helper}
            helper |= children
            grew = bool(children)
        return {key: info for key, info in procs.items() if key[0] not in helper}

    def file_snapshot(self) -> dict[str, tuple[str, str]] | None:
        """``path -> (size, mtime)`` of the regular files under the workdir.

        ``None`` (tracking unavailable) for the root workdir, an unreadable
        container or more than :data:`MAX_TRACKED_FILES` files; an empty dict
        when the workdir simply holds no files.
        """
        if self.workdir.rstrip("/") == "":
            return None
        out = self.container_exec(
            ["sh", "-c", FILE_LIST_COMMAND.format(workdir=shlex.quote(self.workdir))])
        if out is None:
            return None
        files: dict[str, tuple[str, str]] = {}
        for line in out.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3:
                files[parts[2]] = (parts[0], parts[1])
        return files if len(files) <= MAX_TRACKED_FILES else None

    def changed_input_files(self) -> list[str]:
        """Pre-existing files modified or deleted since the run started, each reported once."""
        current = self.file_snapshot()
        if current is None:
            return []
        changed: list[str] = []
        with self._lock:
            for path, (size, mtime) in (self.file_baseline or {}).items():
                if path in self.owned_files:
                    continue
                now = current.get(path)
                if now is None:
                    changed.append(f"{path} (deleted)")
                elif now != (size, mtime):
                    changed.append(f"{path} (size {size} -> {now[0]})" if now[0] != size
                                   else f"{path} (modified)")
                else:
                    continue
                self.owned_files.add(path)
        return changed[:20] + ([f"and {len(changed) - 20} more"] if len(changed) > 20 else [])

    def container_exec(self, argv: list[str], timeout: float = 60) -> str | None:
        """stdout of *argv* run in the trial container; ``None`` on failure or timeout."""
        try:
            completed = subprocess.run(["docker", "exec", self.container, *argv],
                                       capture_output=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.decode("utf-8", errors="replace")

    def finish_verdict(self, args: dict[str, Any]) -> str:
        """Answer the first successful ``finish`` with :data:`FINISH_GATE_VERDICT`.

        ``success`` may arrive as a bool or as the string ``"true"`` (some
        providers do not type tool arguments).  The implicit text-only finish
        consults the hook with no arguments and would end the run as a
        success, so it is vetoed once without spending the gate: the agent is
        then asked to call ``finish`` explicitly, and that call sees the verdict.

        Args:
            args: Arguments of the ``finish`` call (empty for an implicit finish).

        Returns:
            ``"OK"`` or the gate verdict.
        """
        if not args:
            if self.implicit_finish_vetoed:
                return "OK"
            self.implicit_finish_vetoed = True
            return FINISH_GATE_VERDICT
        if str(args.get("success")).lower() != "true":
            return "OK"
        self.gate_answered = True
        return FINISH_GATE_VERDICT

    def shell_verdict(self, args: dict[str, Any]) -> str:
        """Block commands that would end the trial; lift the timeout of installs and builds.

        The timeout is raised in place (``args`` is the dict the tool is
        called with) so a package install or compile is never killed half
        way by ``Bash``'s 30-second default; a timeout already at or above
        :data:`INSTALL_TIMEOUT_SECONDS` (including ``run_commands_parallel``'s
        1800-second default) is left alone.

        Args:
            args: Arguments of a ``Bash`` or ``run_commands_parallel`` call.

        Returns:
            ``"OK"``, or :data:`DESTRUCTIVE_VERDICT` for a blocked command.
        """
        if isinstance(args.get("command"), str):
            commands, default_timeout = [args["command"]], 30.0
        elif isinstance(args.get("commands"), str):
            try:
                parsed = json.loads(args["commands"])
            except ValueError:
                parsed = [args["commands"]]
            commands = ([c for c in parsed if isinstance(c, str)] if isinstance(parsed, list)
                        else [args["commands"]])
            default_timeout = 1800.0
        else:
            return "OK"
        if any(self.destructive.search(c) for c in commands):
            return DESTRUCTIVE_VERDICT
        if any(INSTALL_COMMANDS.search(c) for c in commands):
            try:
                current = float(args.get("timeout_seconds", default_timeout))
            except (TypeError, ValueError):
                current = default_timeout
            if current < INSTALL_TIMEOUT_SECONDS:
                args["timeout_seconds"] = INSTALL_TIMEOUT_SECONDS
        return "OK"

    def landed_edits(self) -> list[tuple[str, str, str]]:
        """``(path, before, after)`` of the edits since the last model call that changed their file.

        Consumed once per model call; the test-context notes and the sibling
        notes share the result (see :meth:`on_llm_call`).
        """
        with self._lock:
            edits, self.pending_edits = self.pending_edits, []
        landed = []
        for path, before in edits:
            after = self.read_container_file(path)
            if after is not None and after != before:
                landed.append((path, before, after))
        return landed

    def snapshot_edit(self, file_path: str) -> None:
        """Remember the content of *file_path* before an edit so the change can be located later.

        New files and test files are ignored: the note is about tests that
        specify existing source code.

        Args:
            file_path: Path as the agent named it (relative paths are taken from the workdir).
        """
        path = posixpath.normpath(posixpath.join(self.workdir, file_path))
        if test_context.is_test_path(path):
            return
        before = self.read_container_file(path)
        if before is not None:
            with self._lock:
                self.pending_edits.append((path, before))

    def test_context_notes(self, landed: list[tuple[str, str, str]]) -> list[str]:
        """Notes about the tests that reference the code edited since the last model call.

        Args:
            landed: The edits since the last model call (:meth:`landed_edits`).
        """
        if not self.test_context:
            return []
        notes: list[str] = []
        for path, before, after in landed:
            names = test_context.changed_definitions(before, after)
            hits = test_context.find_referencing_tests(self.container, self.workdir, names, path)
            tests = [hit for hit in hits if hit[0] not in self.shown_tests]
            if not tests:
                continue
            self.shown_tests.update(hit[0] for hit in tests)
            edited = (posixpath.relpath(path, self.workdir) if path.startswith(self.workdir)
                      else path)
            notes.append(test_context.test_context_note(edited, names, tests))
            self._log({"event": "test_context", "turn": self.turns, "file": edited,
                       "names": names, "tests": tests})
        return notes

    def read_container_file(self, path: str) -> str | None:
        """Text of the file at *path* inside the trial container.

        ``None`` when the file is missing, unreadable, binary or larger than
        :data:`MAX_SNAPSHOT_BYTES` (the test-context note diffs whole files; a
        generated or minified file is not worth the time).
        """
        try:
            completed = subprocess.run(
                ["docker", "exec", self.container, "head", "-c", str(MAX_SNAPSHOT_BYTES + 1), path],
                capture_output=True, timeout=30,
            )
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return None
        if (completed.returncode != 0 or len(completed.stdout) > MAX_SNAPSHOT_BYTES
                or b"\0" in completed.stdout):
            return None
        return completed.stdout.decode("utf-8", errors="replace")

    def _log(self, record: dict[str, Any]) -> None:
        record["ts"] = time.time()
        line = json.dumps(record, ensure_ascii=False, default=str)
        try:
            with self._lock:
                self.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
                with self.trajectory_path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except OSError:  # a lost log line must not end the trial
            pass


_DOCKER_CLIENT: Any = None


def _docker_client() -> Any:
    """The process-wide Docker client (created on first use)."""
    global _DOCKER_CLIENT
    if _DOCKER_CLIENT is None:
        import docker

        _DOCKER_CLIENT = docker.from_env()
    return _DOCKER_CLIENT


def descendants(
    procs: dict[tuple[int, str], tuple[int, str]], roots: set[tuple[int, str]]
) -> set[tuple[int, str]]:
    """Keys of the processes in *procs* whose parent chain reaches one of *roots*.

    Args:
        procs: ``(pid, start time) -> (ppid, cmdline)`` snapshot of the process table.
        roots: Keys of the processes whose descendants are wanted.

    Returns:
        The descendant keys (the roots themselves excluded).
    """
    parent_pids = {key[0] for key in roots if key in procs}
    found: set[tuple[int, str]] = set()
    grew = True
    while grew:
        grew = False
        for key, (ppid, _) in procs.items():
            if key not in found and key not in roots and ppid in parent_pids:
                found.add(key)
                parent_pids.add(key[0])
                grew = True
    return found


def describe_processes(procs: dict[tuple[int, str], tuple[int, str]], limit: int = 8) -> str:
    """``pid cmdline (parent ppid)`` for up to *limit* processes, lowest pid first."""
    items = [f"{pid} {cmd[:120]} (parent {ppid})"
             for (pid, _), (ppid, cmd) in sorted(procs.items())]
    if len(items) > limit:
        items = items[:limit] + [f"and {len(items) - limit} more"]
    return "; ".join(items)


def append_to_last_tool_result(message: Any, note: str) -> bool:
    """Append *note* to the tool result carried by *message*, if it carries one.

    Handles the message shapes KISS models keep in their conversation: the
    OpenAI Responses item ``{"type": "function_call_output", "output": str}``,
    the chat-completions message ``{"role": "tool", "content": str}`` and the
    Anthropic user message whose ``content`` list ends with a ``tool_result``
    block (string or list content).  Anything else is left untouched so the
    note never lands in the task prompt or an assistant turn.

    Args:
        message: The last message added since the previous model call.
        note: The text to append.

    Returns:
        Whether the note was appended.
    """
    if not isinstance(message, dict):
        return False
    if message.get("type") == "function_call_output" and isinstance(message.get("output"), str):
        message["output"] += "\n\n" + note
        return True
    if message.get("role") == "tool" and isinstance(message.get("content"), str):
        message["content"] += "\n\n" + note
        return True
    blocks = message.get("content")
    if message.get("role") == "user" and isinstance(blocks, list) and blocks:
        block = blocks[-1]
        if isinstance(block, dict) and block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                block["content"] = inner + "\n\n" + note
            elif isinstance(inner, list):
                inner.append({"type": "text", "text": note})
            else:
                block["content"] = note
            return True
    return False


def tool_result_texts(message: Any) -> list[str]:
    """The tool-result texts carried by *message*.

    Same message shapes as :func:`append_to_last_tool_result`.

    Args:
        message: One conversation message.

    Returns:
        The text of every tool result in the message; empty for other messages.
    """
    if not isinstance(message, dict):
        return []
    if message.get("type") == "function_call_output" and isinstance(message.get("output"), str):
        return [message["output"]]
    if message.get("role") == "tool" and isinstance(message.get("content"), str):
        return [message["content"]]
    texts: list[str] = []
    if message.get("role") == "user" and isinstance(message.get("content"), list):
        for block in message["content"]:
            if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                continue
            inner = block.get("content")
            if isinstance(inner, str):
                texts.append(inner)
            elif isinstance(inner, list):
                texts.extend(part.get("text", "") for part in inner if isinstance(part, dict))
    return texts


def _jsonable(value: Any) -> Any:
    """Best-effort conversion of provider message objects to JSON-friendly data."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, list):
            return [_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if hasattr(value, "model_dump"):
            return value.model_dump()
        return repr(value)


SEA_TEMPLATE = '''"""Generated per-trial SEA; see kiss.agents.seas.coding.coding_sea."""
from kiss.agents.seas.coding.coding_sea import ContainerHarness

_harness = ContainerHarness.shared({config_path!r})


def description() -> str:
    return (
        "Runs one coding-benchmark trial inside a Docker container using the "
        "harness configured in config.json next to this file; generated by "
        "kiss.agents.seas.coding.coding_sea.write_trial_sea, not meant to be "
        "invoked by hand."
    )


prompt = _harness.prompt
system_prompt = _harness.system_prompt
settings = _harness.settings
llm_call_hook = _harness.llm_call_hook
tool_call_hook = _harness.tool_call_hook
'''


def write_trial_sea(trial_dir: Path, config: dict[str, Any]) -> Path:
    """Write ``config.json`` and the generated SEA file for one trial.

    Args:
        trial_dir: Directory that receives ``config.json``, ``sea.py`` and
            the trajectory log.
        config: The :class:`ContainerHarness` configuration (``trajectory``
            and ``work_dir`` are filled in when absent).

    Returns:
        Path of the generated SEA file.
    """
    trial_dir = trial_dir.resolve()
    trial_dir.mkdir(parents=True, exist_ok=True)
    config = dict(config)
    config.setdefault("trajectory", str(trial_dir / "trajectory.jsonl"))
    config.setdefault("work_dir", str(trial_dir))
    config_path = trial_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=1))
    sea_path = trial_dir / "sea.py"
    sea_path.write_text(SEA_TEMPLATE.format(config_path=str(config_path)))
    return sea_path

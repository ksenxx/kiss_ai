# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Background jobs for the container ``Bash`` tools (Docker and the exec bridge).

The host ``Bash`` has ``background=True`` and ``bash_job`` (wait until a
regex matches, tail, kill); the container tools had neither, so an agent
in a container polled its own ``nohup`` logs with ``sleep N; tail`` turns,
each a model call and a round of context spent waiting; over a long build
or training run those polls are a large share of both.

Everything here is done with shell commands in the container through the
``bash_fn`` the caller supplies (a ``Bash(command, description,
timeout_seconds, max_output_chars) -> str`` of :class:`DockerManager` or
:class:`ExecBridge`), so the same class serves both environments.  A job
runs under ``setsid nohup`` with the exec token
(:data:`kiss.agents.sorcar.docker_manager._EXEC_TOKEN_VAR`) removed from
its environment, so the deadline sweep that kills a timed-out foreground
command's process tree leaves the job alone.  Its state is three files
under :data:`JOBS_DIR`: ``<id>.log`` (stdout and stderr), ``<id>.pid``
(the session leader, whose process group is the job) and ``<id>.exit``
(written when the command ends).  A ``wait`` is one polling script run in
the container, so a long wait costs one tool call, not one per poll.
"""

from __future__ import annotations

import re
import shlex
import uuid
from collections.abc import Callable

from kiss.agents.sorcar.docker_manager import _EXEC_TOKEN_VAR

#: Directory in the container that holds every job's log, pid and exit files.
JOBS_DIR = "/tmp/kiss-jobs"
#: Seconds added to a wait's deadline for the polling script's own overhead.
WAIT_SLACK_SECONDS = 15
#: Seconds the start, tail and kill commands get.
QUICK_TIMEOUT = 30
#: A job id (the ids are spliced into shell scripts, so nothing else passes).
JOB_ID = re.compile(r"[0-9a-f]{8}")

BashFn = Callable[[str, str, int, int], str]

#: Starts the job detached and prints the session leader's pid once the job
#: wrote it.  ``{inner}`` is the quoted :data:`INNER_SCRIPT`, ``{d}`` the jobs
#: dir, ``{id}`` the job id.  The inner shell records its own pid (``$$``:
#: the session leader, whatever ``setsid`` did), runs the command with bash
#: (``sh`` when the image has no bash) and writes the exit code when it ends.
START_SCRIPT = (
    "mkdir -p {d} && printf '%s' {cmd} > {d}/{id}.cmd && "
    "env -u {token} setsid nohup sh -c {inner} > /dev/null 2>&1 < /dev/null & "
    "for i in $(seq 1 50); do [ -s {d}/{id}.pid ] && break; sleep 0.1; done; cat {d}/{id}.pid"
)
INNER_SCRIPT = (
    "echo $$ > {d}/{id}.pid; kiss_sh=bash; command -v bash >/dev/null 2>&1 || kiss_sh=sh; "
    "\"$kiss_sh\" -c {cmd} > {d}/{id}.log 2>&1 < /dev/null; echo $? > {d}/{id}.exit"
)
#: Polls until the job exits, the log matches ``until`` or the deadline passes,
#: then prints the status line and the log tail (:data:`REPORT_SCRIPT`).
WAIT_SCRIPT = (
    "end=$(( $(date +%s) + {timeout} )); "
    "while [ ! -e {d}/{id}.exit ]; do "
    "{match}"
    "[ $(date +%s) -ge $end ] && break; sleep 1; done; "
)
MATCH_CLAUSE = "grep -qE -- {pattern} {d}/{id}.log 2>/dev/null && break; "
#: Runs the ``until`` pattern through the engine that :data:`MATCH_CLAUSE`
#: uses; prints grep's complaint (an invalid regex, a warning about a
#: construct ``grep -E`` reads differently than the author meant) or nothing.
UNTIL_CHECK = "grep -qE -- {pattern} /dev/null 2>&1; true"
#: Prints ``status: exited N`` or ``status: running`` and the last lines of the log.
REPORT_SCRIPT = (
    "if [ -e {d}/{id}.exit ]; then echo \"status: exited $(cat {d}/{id}.exit)\"; "
    "else echo 'status: running'; fi; "
    "echo 'log: {d}/{id}.log'; echo '--- last {n} log lines ---'; "
    "tail -n {n} {d}/{id}.log 2>/dev/null; true"
)
#: Kills the job's process group (the session ``setsid`` created), then its
#: leader.  ``kill -TERM -<pgid>`` is the spelling every shell accepts
#: (dash rejects ``--`` before a negative pid).
KILL_SCRIPT = (
    "p=$(cat {d}/{id}.pid 2>/dev/null); if [ -n \"$p\" ]; then kill -TERM -$p 2>/dev/null; "
    "sleep 1; kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; fi; "
    "[ -e {d}/{id}.exit ] || echo 'killed' > {d}/{id}.exit; true"
)
#: Kills the process group of every job ever started, exited or not (the
#: ``.exit`` file records the job's shell, not a child it left running in the
#: background): the harness runs it before it restores a checkpoint, so a
#: process still writing into the work tree cannot undo the restore.  Prints
#: the ids of the jobs whose process group still existed (a zombie the
#: container's init never reaped counts), one per line.
KILL_ALL_SCRIPT = (
    f"for f in {JOBS_DIR}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); kill -TERM -$p 2>/dev/null || continue; echo $id; done; sleep 1; "
    f"for f in {JOBS_DIR}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; "
    f"[ -e {JOBS_DIR}/$id.exit ] || echo 'killed' > {JOBS_DIR}/$id.exit; done; true"
)


class ContainerJobs:
    """Background jobs of one container, driven through its ``Bash`` tool.

    Args:
        bash_fn: Runs a shell command in the container and returns its output
            (``Bash(command, description, timeout_seconds, max_output_chars)``).
    """

    def __init__(self, bash_fn: BashFn) -> None:
        self.bash_fn = bash_fn
        #: job id -> command, for the reports and the unknown-id message.
        self.commands: dict[str, str] = {}

    def start(self, command: str) -> tuple[str | None, str]:
        """Start *command* detached in the container.

        Args:
            command: The shell command (run with ``bash -c``).

        Returns:
            ``(job id, report)``: the report is ``Started background job <id>
            (pid N); log: <path>`` with the ``bash_job`` reminder, or, with a
            ``None`` job id, a string starting with ``Error:``.
        """
        job_id = uuid.uuid4().hex[:8]
        inner = INNER_SCRIPT.format(cmd=shlex.quote(command), d=JOBS_DIR, id=job_id)
        script = START_SCRIPT.format(token=_EXEC_TOKEN_VAR, inner=shlex.quote(inner), d=JOBS_DIR,
                                     id=job_id, cmd=shlex.quote(command))
        out = self.bash_fn(script, f"start background job {job_id}", QUICK_TIMEOUT, 2000).strip()
        pid = out.splitlines()[-1].strip() if out else ""
        if not pid.isdigit():
            return None, f"Error: could not start the background job: {out or '(no output)'}"
        self.commands[job_id] = command
        return job_id, (
            f"Started background job {job_id} (pid {pid}); log: {JOBS_DIR}/{job_id}.log. "
            f"Use bash_job({job_id!r}, action='wait', timeout_seconds=N, until='<regex>') "
            "to block until it exits or its log matches, 'tail' for the current log, "
            "'kill' to stop it.")

    def bash_job(
        self,
        job_id: str,
        action: str = "tail",
        timeout_seconds: float = 300,
        tail_lines: int = 50,
        max_output_chars: int = 50000,
        until: str = "",
    ) -> str:
        """Wait for, tail, or kill a background job started by ``Bash(background=True)``.

        Every action returns the same report: the job's status (``running``
        or ``exited N``), the log file path and the last ``tail_lines``
        lines of the log.  The full log stays in the container at the
        reported path.

        Args:
            job_id: The id returned by ``Bash(background=True)``.
            action: ``"tail"`` (default) reports the current status and
                log tail at once; ``"wait"`` blocks until the job exits or
                ``timeout_seconds`` pass, then reports (a still-running
                status means the wait timed out, not that the job failed);
                ``"kill"`` kills the job's whole process group, then reports.
            timeout_seconds: How long ``"wait"`` blocks at most.  Prefer
                one long wait over many short polls.
            tail_lines: Number of trailing log lines to include.
            max_output_chars: Maximum characters in the report before truncation.
            until: With ``"wait"``, an extended regular expression (``grep -E``):
                the wait also ends as soon as the log matches it (an epoch
                marker, a "Serving on" line), so a long run is one turn to
                start and one to collect.

        Returns:
            The job report, or a string starting with ``Error:`` for an
            unknown job id or action.
        """
        if not JOB_ID.fullmatch(job_id or ""):
            known = ", ".join(self.commands) or "none"
            return f"Error: unknown job id {job_id!r}. Known jobs: {known}."
        if job_id not in self.commands:
            # A job started by an earlier session of this task: its files are
            # still in the container even though this registry never saw it.
            probe = self.bash_fn(f"cat {JOBS_DIR}/{job_id}.cmd 2>/dev/null; true", "probe job",
                                 QUICK_TIMEOUT, 2000).strip()
            if not probe:
                known = ", ".join(self.commands) or "none"
                return f"Error: unknown job id {job_id!r}. Known jobs: {known}."
            self.commands[job_id] = probe
        if until:
            complaint = self.bash_fn(UNTIL_CHECK.format(pattern=shlex.quote(until)),
                                     "check the until pattern", QUICK_TIMEOUT, 2000).strip()
            if complaint:
                return f"Error: until is not a valid grep -E regex: {complaint}"
        lines = max(1, int(tail_lines))
        report = REPORT_SCRIPT.format(d=JOBS_DIR, id=job_id, n=lines)
        timeout = QUICK_TIMEOUT
        if action == "wait":
            seconds = max(1, int(timeout_seconds))
            match = ""
            if until:
                match = MATCH_CLAUSE.format(pattern=shlex.quote(until), d=JOBS_DIR, id=job_id)
            script = WAIT_SCRIPT.format(timeout=seconds, d=JOBS_DIR, id=job_id, match=match)
            script += report
            timeout = seconds + WAIT_SLACK_SECONDS
        elif action == "kill":
            script = KILL_SCRIPT.format(d=JOBS_DIR, id=job_id) + "; " + report
        elif action == "tail":
            script = report
        else:
            return f"Error: action must be one of wait, tail, kill; got {action!r}."
        out = self.bash_fn(script, f"{action} background job {job_id}", timeout, max_output_chars)
        return f"Job {job_id}: {self.commands[job_id]}\n{out}"

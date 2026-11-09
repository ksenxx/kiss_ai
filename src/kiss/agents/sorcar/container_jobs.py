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
(written when the command ends; the shell opens it before it writes the
exit code, so only a non-empty marker is a finished job).  A ``wait`` is
one polling script run in the container, so a long wait costs one tool
call, not one per poll.

A process the agent left running from a foreground command (``nohup …
&``) can be adopted as a job too (:meth:`ContainerJobs.adopt`): the same
files are written for it (plus ``<id>.owner``, its start time as adopted,
which with the pid identifies the process through any exec), the log
linked to the file its redirection named, and a detached watcher writes
the exit file once the pid is gone, so every job script works on it
unchanged.
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
#: The status line of a :data:`REPORT_SCRIPT` report.
STATUS_LINE = re.compile(r"^status: (.+)$", re.MULTILINE)
#: Prints the ids among ``{ids}`` (space-separated) whose job has not exited.
RUNNING_SCRIPT = "for id in {ids}; do [ -s {d}/$id.exit ] || echo $id; done; true"

BashFn = Callable[[str, str, int, int], str]

#: Starts the job detached and prints the session leader's pid once the job
#: wrote it.  ``{inner}`` is the quoted :data:`INNER_SCRIPT`, ``{d}`` the jobs
#: dir, ``{id}`` the job id.  The inner shell records its own pid (``$$``:
#: the session leader, whatever ``setsid`` did), runs the command with bash
#: (``sh`` when the image has no bash) and writes the exit code when it ends.
#: Only the launch is backgrounded (the braces): a backgrounded ``a && b && c &``
#: list is a subshell that keeps the exec's stdout open until the job ends.
START_SCRIPT = (
    "mkdir -p {d} && printf '%s' {cmd} > {d}/{id}.cmd && "
    "{{ env -u {token} setsid nohup sh -c {inner} > /dev/null 2>&1 < /dev/null & }}; "
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
    "while [ ! -s {d}/{id}.exit ]; do "
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
    "if [ -s {d}/{id}.exit ]; then echo \"status: exited $(cat {d}/{id}.exit)\"; "
    "else echo 'status: running'; fi; "
    "echo 'log: {d}/{id}.log'; echo '--- last {n} log lines ---'; "
    "tail -n {n} {d}/{id}.log 2>/dev/null; true"
)
#: Defines ``ident <pid>``: prints the start time of the process *pid*
#: (clock ticks since boot, field 22 of ``/proc/<pid>/stat``), nothing when
#: the pid is gone or a zombie (``kill -0`` alone would not say so: a
#: container whose pid 1 is ``sleep infinity`` never reaps an orphan, and
#: the harness's process listing drops zombies too).  Pid and start time
#: together identify a process: an exec keeps both, and a later process the
#: kernel gave the pid has another start time.
IDENT_FN = (
    "ident() {{ set -- $(sed 's/.*) //' /proc/$1/stat 2>/dev/null); "
    "[ $# -ge 20 ] && [ \"$1\" != Z ] && echo \"${{20}}\"; }}; "
)
#: Defines ``owned <pid> <pidfile>``: true when the process group *pid* is
#: the job's, that is its leader is the job's shell (its command line names
#: the pid file the shell wrote) or the leader has ended (no process, or a
#: zombie, whose command line is empty) and only its group may live on (a
#: pid in use as a group id is never reused).  A recycled pid, one the kernel
#: gave an unrelated process after the job's group drained, fails the check,
#: so the kill scripts never signal a stranger.  An adopted job (its
#: ``.owner`` file holds the start time of the process as adopted) is owned
#: while the pid belongs to the process with that start time, whatever it
#: has exec'd since: an adopted process ended is nobody's to signal, and so
#: is a stranger wearing its pid.
OWNED_FN = IDENT_FN + (
    "owned() {{ if [ -s \"${{2%.pid}}.owner\" ]; then "
    "[ \"$(ident $1)\" = \"$(cat \"${{2%.pid}}.owner\")\" ]; return; fi; "
    "c=$(tr '\\0' ' ' < /proc/$1/cmdline 2>/dev/null); "
    "case \"$c\" in (''|*\"$2\"*) true;; (*) false;; esac; }}; "
)
#: Defines ``tree <pid> <pidfile>``, which prints ``pid:start time`` of every
#: descendant of the adopted process *pid* (nothing for a started job: its
#: process group is the job), and ``hit <signal> <pid:start time>...``, which
#: sends the signal to each of those processes still running with that start
#: time.  An adopted process leads no process group as a rule (a ``nohup …
#: &`` of a shell without job control stays in the shell's group, which is
#: not the job's to signal), so its children are found by their parent pids
#: in ``/proc/*/stat``, captured before the parent is signalled (its death
#: reparents them) and signalled by identity (a pid reused meanwhile is a
#: stranger's).
TREE_FN = (
    "tree() {{ [ -s \"${{2%.pid}}.owner\" ] || return 0; "
    "procs=$(cat /proc/[0-9]*/stat 2>/dev/null | while read -r pid rest; do "
    "set -- ${{rest##*) }}; [ $# -ge 20 ] && echo \"$pid:$2:${{20}}\"; done); "
    "todo=\" $1 \"; while [ -n \"$todo\" ]; do next=; for e in $procs; do k=${{e%:*}}; "
    "case \"$todo\" in (*\" ${{k#*:}} \"*) echo \"${{e%%:*}}:${{e##*:}}\"; "
    "next=\"$next ${{e%%:*}} \";; esac; done; todo=$next; done; }}; "
    "hit() {{ s=$1; shift; for k in \"$@\"; do "
    "[ \"$(ident ${{k%:*}})\" = \"${{k#*:}}\" ] && kill -$s ${{k%:*}} 2>/dev/null; done; }}; "
)
#: Kills the job's process group (the session ``setsid`` created), then its
#: leader, when the group is still the job's.  ``kill -TERM -<pgid>`` is the
#: spelling every shell accepts (dash rejects ``--`` before a negative pid).
#: An adopted pid that leads no group gets the TERM itself, and so do its
#: descendants (:data:`TREE_FN`), then the KILL every job's leader gets,
#: the descendants found again at that point (a parent that ignored the TERM
#: may have spawned more).  A job still running is marked ``killed`` before
#: the signals: an adopted job's watcher would otherwise record the kill as
#: a plain exit.
KILL_SCRIPT = OWNED_FN + TREE_FN + (
    "p=$(cat {d}/{id}.pid 2>/dev/null); if [ -n \"$p\" ] && owned $p {d}/{id}.pid; then "
    "[ -e {d}/{id}.exit ] || echo 'killed' > {d}/{id}.exit; t=$(tree $p {d}/{id}.pid); "
    "kill -TERM -$p 2>/dev/null; kill -TERM $p 2>/dev/null; hit TERM $t; sleep 1; "
    "if owned $p {d}/{id}.pid; then t=\"$t $(tree $p {d}/{id}.pid)\"; "
    "kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; fi; hit KILL $t; fi; "
    "[ -s {d}/{id}.exit ] || echo 'killed' > {d}/{id}.exit; true"
)
#: Kills the process group of every job ever started or adopted, exited or
#: not (the ``.exit`` file records the job's shell, not a child it left
#: running in the background), as long as the group, or the adopted process,
#: is still the job's, and the descendants of every adopted process
#: (:data:`TREE_FN`): the harness runs it at the end of a trial, before it
#: restores a checkpoint, so a process still writing into the work tree
#: cannot undo the restore.  Prints the ids of the jobs whose process group,
#: or adopted pid, still existed (a zombie the container's init never reaped
#: counts), one per line.
KILL_ALL_SCRIPT = (OWNED_FN + TREE_FN + (
    "kids=; for f in {d}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); owned $p \"$f\" || continue; "
    "[ -e {d}/$id.exit ] || echo 'killed' > {d}/$id.exit; t=$(tree $p \"$f\"); kids=\"$kids $t\"; "
    "kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || continue; echo $id; "
    "hit TERM $t; done; sleep 1; "
    "for f in {d}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); owned $p \"$f\" && {{ kids=\"$kids $(tree $p \"$f\")\"; "
    "kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; }}; "
    "[ -s {d}/$id.exit ] || echo 'killed' > {d}/$id.exit; done; hit KILL $kids; true"
)).format(d=JOBS_DIR)
#: Registers the running process ``{pid}`` as job ``{id}``: its command,
#: owner (the process's start time, which ``owned`` and the watcher compare
#: against, so a recycled pid is never signalled and an exec is not taken
#: for an exit) and pid files, its log (a link to ``{log}``, the quoted
#: absolute path of the file its redirection named, or an empty file for
#: ``''``) and the detached watcher (:data:`WATCH_SCRIPT`, quoted as
#: ``{watch}``), started like a job so the deadline sweep leaves it alone.
#: Prints ``adopted`` once all of that is in place, or ``gone`` when the
#: pid is not alive; nothing when the jobs dir cannot be made, and only the
#: shell's complaint when a job file cannot be written (the files written
#: before it are removed: a half-registered job is no job).
ADOPT_SCRIPT = IDENT_FN + (
    "mkdir -p {d} || exit 0; s=$(ident {pid}); [ -n \"$s\" ] || {{ echo gone; exit 0; }}; "
    "printf '%s' {cmd} > {d}/{id}.cmd && echo \"$s\" > {d}/{id}.owner && "
    "echo {pid} > {d}/{id}.pid && "
    "{{ [ -n {log} ] && ln -s {log} {d}/{id}.log || : > {d}/{id}.log; }} && "
    "{{ env -u {token} setsid nohup sh -c {watch} > /dev/null 2>&1 < /dev/null & }} && "
    "echo adopted || rm -f {d}/{id}.cmd {d}/{id}.owner {d}/{id}.pid {d}/{id}.log"
)
#: Polls every second until the exit file exists, writing it once the
#: adopted process is gone, that is once the pid no longer belongs to the
#: process with the start time the owner file holds (a kill may have written
#: the file first; an exec is not an exit).  The exit code is unknown (the
#: watcher is not the process's parent), and the file says so.  The exit
#: file's path opens the script, as the argument of the null command, so the
#: watcher's command line names the jobs dir within the first characters a
#: process listing keeps, whatever ``ident``'s definition adds before the loop.
WATCH_SCRIPT = ": {d}/{id}.exit; " + IDENT_FN + (
    "until [ -e {d}/{id}.exit ]; do [ \"$(ident {pid})\" = \"$(cat {d}/{id}.owner)\" ] || "
    "[ -e {d}/{id}.exit ] || echo '(exit code unknown: adopted process)' > {d}/{id}.exit; "
    "sleep 1; done"
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

    def adopt(self, pid: int, command: str, log_path: str) -> tuple[str | None, str]:
        """Register the running process *pid* the agent left behind as a job.

        The job gets the files a started job has (command, pid, log, and
        the exit file once a detached watcher sees the pid gone), so
        ``bash_job`` waits, tails and kills it, :meth:`last_running` finds
        it and the end-of-task kill stops it.  Nothing kills it before
        that: a server the task started on purpose keeps running.

        Args:
            pid: The process (as the container sees it).
            command: Its command line, shown in the reports.
            log_path: The absolute path of the file its output goes to
                (the target of its redirection), linked as the job's log so
                a wait can match it; ``""`` when its output goes nowhere.

        Returns:
            ``(job id, report)``: the report is ``left running as job <id>
            (<command>, log <path>)`` (``no log`` without a path), or, with
            a ``None`` job id, a string starting with ``Error:`` (the pid
            is not alive, or the job files could not be written).
        """
        job_id = uuid.uuid4().hex[:8]
        watch = WATCH_SCRIPT.format(d=JOBS_DIR, id=job_id, pid=int(pid))
        script = ADOPT_SCRIPT.format(d=JOBS_DIR, pid=int(pid), id=job_id,
                                     cmd=shlex.quote(command), log=shlex.quote(log_path),
                                     token=_EXEC_TOKEN_VAR, watch=shlex.quote(watch))
        out = self.bash_fn(script, f"adopt process {pid} as job {job_id}", QUICK_TIMEOUT,
                           2000).strip()
        if out.splitlines()[-1:] != ["adopted"]:
            what = "is not running" if out.endswith("gone") else f"could not be adopted: {out}"
            return None, f"Error: process {pid} {what}"
        self.commands[job_id] = command
        where = f"log {log_path}" if log_path else "no log"
        return job_id, f"left running as job {job_id} ({command}, {where})"

    def last_running(self) -> str | None:
        """Return the id of the job of this registry started last that is still running.

        One script in the container checks every known job's exit file;
        ``None`` when no job is running.
        """
        if not self.commands:
            return None
        script = RUNNING_SCRIPT.format(ids=" ".join(self.commands), d=JOBS_DIR)
        out = self.bash_fn(script, "find the running background jobs", QUICK_TIMEOUT, 2000)
        running = [line.strip() for line in out.splitlines() if JOB_ID.fullmatch(line.strip())]
        return running[-1] if running else None

    def wait(self, job_id: str, seconds: float) -> str:
        """Block until the job *job_id* exits or *seconds* pass; return its status.

        The ``wait`` action of :meth:`bash_job`, reduced to the status
        line of its report (``running`` or ``exited N``).  The report
        opens with the job's command, which may hold such a line itself
        (a here-document printing one): the status is read after it.
        """
        report = self.bash_job(job_id, "wait", seconds, 1, 2000)
        header = f"Job {job_id}: {self.commands.get(job_id, '')}\n"
        match = STATUS_LINE.search(report.removeprefix(header))
        return match.group(1) if match else "status unknown"

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
        lines = max(1, int(tail_lines))
        report = REPORT_SCRIPT.format(d=JOBS_DIR, id=job_id, n=lines)
        timeout = QUICK_TIMEOUT
        if action == "wait":
            seconds = max(1, int(timeout_seconds))
            match = ""
            if until:
                complaint = self.bash_fn(UNTIL_CHECK.format(pattern=shlex.quote(until)),
                                         "check the until pattern", QUICK_TIMEOUT, 2000).strip()
                if complaint:
                    return f"Error: until is not a valid grep -E regex: {complaint}"
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

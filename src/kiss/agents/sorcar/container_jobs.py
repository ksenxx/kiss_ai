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
under :data:`JOBS_DIR` (a sub-agent's job keeps them in the sub-agent's
private folder instead, which is deleted, the job killed, when the
sub-agent ends): ``<id>.log`` (stdout and stderr), ``<id>.pid``
(the session leader, whose process group is the job) and ``<id>.exit``
(written when the command ends; the shell opens it before it writes the
exit code, so only a non-empty marker is a finished job; a job stopped
from outside before that gets :data:`LOST_STATUS`).  A job left with
neither a pid file nor an exit code (the agent cleared the directory) is
:data:`GONE_STATUS`, neither running nor exited.  A ``wait`` is
one polling script run in the container, so a long wait costs one tool
call, not one per poll.

A process the agent left running from a foreground command (``nohup …
&``) can be adopted as a job too (:meth:`ContainerJobs.adopt`): the same
files are written for it (plus ``<id>.owner``, its start time as adopted,
which with the pid identifies the process through any exec,
``<id>.group`` when it leads its own process group and ``<id>.token``,
the exec token in its environment, by which what it leaves running when
it ends is still found), the log
linked to the file its redirection named, and a detached watcher writes
the exit file once the pid is gone, so every job script works on it
unchanged.
"""

from __future__ import annotations

import re
import shlex
import uuid
from collections.abc import Callable

from kiss.agents.sorcar.docker_manager import _EXEC_TOKEN_VAR, MARKED_FN
from kiss.agents.sorcar.useful_tools import SUBAGENT_DIR_ENV

#: Directory in the container that holds the log, pid and exit files of every
#: job that is not a sub-agent's.
JOBS_DIR = "/tmp/kiss-jobs"
#: A sub-agent's private folder as its commands' environment names it (the
#: path is spliced into the job scripts unquoted, so nothing else passes).
PRIVATE_DIR = re.compile(r"/[A-Za-z0-9_./-]+")
#: Seconds added to a wait's deadline for the polling script's own overhead.
WAIT_SLACK_SECONDS = 15
#: Seconds the start, tail and kill commands get.
QUICK_TIMEOUT = 30
#: A job id (the ids are spliced into shell scripts, so nothing else passes).
JOB_ID = re.compile(r"[0-9a-f]{8}")
#: The status line of a :data:`REPORT_SCRIPT` report.
STATUS_LINE = re.compile(r"^status: (.+)$", re.MULTILINE)
#: What the exit file of a job says whose shell was stopped before it wrote one.
LOST_STATUS = "(no exit code: stopped from outside bash_job)"
#: The status of a job that has neither an exit code nor a pid file: its files
#: are gone (the agent cleared the jobs dir), and with them what became of it.
#: No apostrophe: the text is echoed from a single-quoted shell word.
GONE_STATUS = "unknown (the job files are gone; the process may still run)"
#: Defines ``settle <id>``: writes :data:`LOST_STATUS` into the exit file of the
#: started job *id* when it has none and nothing of the job is left, that is
#: its shell is gone (no process, a zombie, or a stranger the kernel gave the
#: pid, whose command line does not name the pid file) and no live process
#: is in its group.  Only the shell writes the exit code, as its last
#: command, and a ``pkill -f`` of the job's command stops the shell too (its
#: command line holds the command): such a job counted as running for ever.
#: The exit file is looked at again once the shell is seen gone, since the
#: shell may have written its code in between.  An adopted job is settled
#: by its watcher (:data:`WATCH_SCRIPT`), not here.  Also defines ``running
#: <id>``: true while the job *id*, settled, has a pid file and no exit code
#: (with neither it is :data:`GONE_STATUS`, which nothing waits on).
SETTLE_FN = (
    "settle() {{ [ -s {d}/$1.exit ] || [ -s {d}/$1.owner ] && return; "
    "p=$(cat {d}/$1.pid 2>/dev/null); [ -n \"$p\" ] || return 0; "
    "c=$(tr '\\0' ' ' 2>/dev/null < /proc/$p/cmdline); "
    "case \"$c\" in (*\"{d}/$1.pid\"*) return 0;; ('') "
    "cat /proc/[0-9]*/stat 2>/dev/null | {{ while read -r pid rest; do set -- ${{rest##*) }}; "
    "[ \"$1\" != Z ] && [ \"$3\" = \"$p\" ] && exit 1; done; true; }} || return 0;; esac; "
    "[ -s {d}/$1.exit ] || echo '" + LOST_STATUS + "' > {d}/$1.exit; }}; "
    "running() {{ settle $1; [ ! -s {d}/$1.exit ] && [ -s {d}/$1.pid ]; }}; "
)
#: Prints the ids among ``{ids}`` (space-separated) whose job is running.
RUNNING_SCRIPT = SETTLE_FN + "for id in {ids}; do running $id && echo $id; done; true"

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
#: Polls while the job runs, until the log matches ``until`` or the deadline
#: passes, then prints the status line and the log tail (:data:`REPORT_SCRIPT`).
WAIT_SCRIPT = SETTLE_FN + (
    "end=$(( $(date +%s) + {timeout} )); "
    "while running {id}; do "
    "{match}"
    "[ $(date +%s) -ge $end ] && break; sleep 1; done; "
)
MATCH_CLAUSE = "grep -qE -- {pattern} {d}/{id}.log 2>/dev/null && break; "
#: Runs the ``until`` pattern through the engine that :data:`MATCH_CLAUSE`
#: uses; prints grep's complaint (an invalid regex, a warning about a
#: construct ``grep -E`` reads differently than the author meant) or nothing.
UNTIL_CHECK = "grep -qE -- {pattern} /dev/null 2>&1; true"
#: Prints ``status: running``, ``status: exited N`` or ``status:``
#: :data:`GONE_STATUS`, and the last lines of the log.
REPORT_SCRIPT = SETTLE_FN + (
    "if running {id}; then echo 'status: running'; elif [ -s {d}/{id}.exit ]; then "
    "echo \"status: exited $(cat {d}/{id}.exit)\"; else echo 'status: " + GONE_STATUS + "'; fi; "
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
#: is a stranger wearing its pid.  One that led its own process group when
#: adopted (its ``.group`` file) and has ended leaves the group the job's,
#: as a started job's shell does: a child it left there is still killed.
OWNED_FN = IDENT_FN + (
    "owned() {{ if [ -s \"${{2%.pid}}.owner\" ]; then i=$(ident $1); "
    "[ \"$i\" = \"$(cat \"${{2%.pid}}.owner\")\" ] || "
    "{{ [ -z \"$i\" ] && [ -s \"${{2%.pid}}.group\" ]; }}; return; fi; "
    "c=$(tr '\\0' ' ' 2>/dev/null < /proc/$1/cmdline); "
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
#: stranger's).  A child it leaves when it ends by itself has no parent to
#: be found by: that one is found by the token (:data:`TOKEN_FN`).
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
#: :data:`~kiss.agents.sorcar.docker_manager.MARKED_FN` for the job scripts,
#: which are ``str.format`` templates, and ``end_marked <entry>...``: ends
#: every process whose environment has one of the entries, with a TERM and,
#: a second later (not waited for when the TERM found nobody), the KILL the
#: rest of a job gets; nothing without an entry.  The entry of an adopted
#: job is the exec token of its process as adopted (its ``.token`` file):
#: every process the command that left it running started carries it, also
#: a child the adopted process started and left behind when it ended by
#: itself, which no parent pid (:data:`TREE_FN`) and, unless the process
#: led it, no process group leads to any more.
TOKEN_FN = MARKED_FN.replace("{", "{{").replace("}", "}}") + (
    "end_marked() {{ [ $# -gt 0 ] || return 0; "
    "[ -z \"$(marked TERM '' \"$@\")\" ] || sleep 1; kill_marked '' \"$@\"; }}; "
)
#: Kills the job's process group (the session ``setsid`` created), then its
#: leader, when the group is still the job's.  ``kill -TERM -<pgid>`` is the
#: spelling every shell accepts (dash rejects ``--`` before a negative pid).
#: An adopted pid that leads no group gets the TERM itself, and so do its
#: descendants (:data:`TREE_FN`), then the KILL every job's leader gets,
#: the descendants found again at that point (a parent that ignored the TERM
#: may have spawned more).  A job still running is marked ``killed`` before
#: the signals: an adopted job's watcher would otherwise record the kill as
#: a plain exit.  A job whose pid file is gone has no process to signal and
#: is not marked ``killed`` (:data:`GONE_STATUS`).  Last, whatever still
#: carries an adopted job's token is ended (:data:`TOKEN_FN`), the adopted
#: process alive or not, provided no other job has the same token: of two
#: processes one command left running, each a job, the token does not say
#: whose child an orphan was, and the other job is not this kill's to end.
KILL_SCRIPT = OWNED_FN + TREE_FN + TOKEN_FN + (
    "p=$(cat {d}/{id}.pid 2>/dev/null); if [ -n \"$p\" ] && owned $p {d}/{id}.pid; then "
    "[ -e {d}/{id}.exit ] || echo 'killed' > {d}/{id}.exit; t=$(tree $p {d}/{id}.pid); "
    "kill -TERM -$p 2>/dev/null; kill -TERM $p 2>/dev/null; hit TERM $t; sleep 1; "
    "if owned $p {d}/{id}.pid; then t=\"$t $(tree $p {d}/{id}.pid)\"; "
    "kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; fi; hit KILL $t; fi; "
    "[ -s {d}/{id}.exit ] || [ -z \"$p\" ] || echo 'killed' > {d}/{id}.exit; "
    "m=$(cat {d}/{id}.token 2>/dev/null); [ -z \"$m\" ] || "
    "[ \"$(cat {d}/*.token | grep -cxF -- \"$m\")\" != 1 ] || end_marked \"$m\"; true"
)
#: Kills the process group of every job ever started or adopted, exited or
#: not (the ``.exit`` file records the job's shell, not a child it left
#: running in the background), as long as the group, or the adopted process,
#: is still the job's, and the descendants of every adopted process
#: (:data:`TREE_FN`): the harness runs it at the end of a trial, before it
#: restores a checkpoint, so a process still writing into the work tree
#: cannot undo the restore.  Prints the ids of the jobs whose process group,
#: or adopted pid, still existed (a zombie the container's init never reaped
#: counts), one per line.  Then ends, without a line, whatever still carries
#: the token of an adopted job (:data:`TOKEN_FN`): what the adopted
#: processes that ended by themselves left running.
KILL_ALL_SCRIPT = (OWNED_FN + TREE_FN + TOKEN_FN + (
    "kids=; for f in {d}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); owned $p \"$f\" || continue; "
    "[ -e {d}/$id.exit ] || echo 'killed' > {d}/$id.exit; t=$(tree $p \"$f\"); kids=\"$kids $t\"; "
    "kill -TERM -$p 2>/dev/null || kill -TERM $p 2>/dev/null || continue; echo $id; "
    "hit TERM $t; done; sleep 1; "
    "for f in {d}/*.pid; do [ -e \"$f\" ] || continue; id=$(basename \"$f\" .pid); "
    "p=$(cat \"$f\"); owned $p \"$f\" && {{ kids=\"$kids $(tree $p \"$f\")\"; "
    "kill -KILL -$p 2>/dev/null; kill -KILL $p 2>/dev/null; }}; "
    "[ -s {d}/$id.exit ] || echo 'killed' > {d}/$id.exit; done; hit KILL $kids; "
    "set --; for f in {d}/*.token; do m=$(cat \"$f\" 2>/dev/null); "
    "[ -z \"$m\" ] || set -- \"$@\" \"$m\"; done; end_marked \"$@\"; true"
)).format(d=JOBS_DIR)
#: Registers the running process ``{pid}`` as job ``{id}``: its command,
#: owner (the process's start time, which ``owned`` and the watcher compare
#: against, so a recycled pid is never signalled and an exec is not taken
#: for an exit), token (the exec token entry of its environment, empty
#: when it has none or the environment is another user's to read;
#: :data:`TOKEN_FN`) and pid files, its log (a link to ``{log}``, the quoted
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
    "{{ tr '\\n\\0' ' \\n' 2>/dev/null < /proc/{pid}/environ | grep -m 1 '^{token}=' "
    "> {d}/{id}.token; true; }} && "
    "{{ [ \"$(sed 's/.*) //' /proc/{pid}/stat 2>/dev/null | cut -d' ' -f3)\" != {pid} ] || "
    "echo {pid} > {d}/{id}.group; }} && echo {pid} > {d}/{id}.pid && "
    "{{ [ -n {log} ] && ln -s {log} {d}/{id}.log || : > {d}/{id}.log; }} && "
    "{{ env -u {token} setsid nohup sh -c {watch} > /dev/null 2>&1 < /dev/null & }} && "
    "echo adopted || "
    "rm -f {d}/{id}.cmd {d}/{id}.owner {d}/{id}.token {d}/{id}.group {d}/{id}.pid {d}/{id}.log"
)
#: Polls every second until the exit file exists, writing it once the
#: adopted process is gone, that is once the pid no longer belongs to the
#: process with the start time the owner file holds (a kill may have written
#: the file first; an exec is not an exit).  The exit code is unknown (the
#: watcher is not the process's parent), and the file says so.  A job whose
#: owner file is gone (the agent cleared the jobs dir) has nothing left to
#: record, so its watcher ends instead of retrying, for ever, a write into a
#: dir that is no more.  The exit file's path opens the script, as the
#: argument of the null command, so the watcher's command line names the jobs
#: dir within the first characters a process listing keeps, whatever
#: ``ident``'s definition adds before the loop.
WATCH_SCRIPT = ": {d}/{id}.exit; " + IDENT_FN + (
    "until [ -e {d}/{id}.exit ]; do o=$(cat {d}/{id}.owner) || break; "
    "[ \"$(ident {pid})\" = \"$o\" ] || [ -e {d}/{id}.exit ] || "
    "echo '(exit code unknown: adopted process)' > {d}/{id}.exit; sleep 1; done"
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
        #: job id -> the directory of its files (:data:`JOBS_DIR` for an id not in it).
        self.dirs: dict[str, str] = {}

    def _new_job_dir(self) -> str:
        """Return the directory for the files of a job started or adopted now.

        The commands of a sub-agent carry its private folder in their
        environment (:data:`~kiss.agents.sorcar.useful_tools.SUBAGENT_DIR_ENV`),
        and so does a job they start: the end of the sub-agent kills it and
        deletes the folder.  Its files go there (``kiss-jobs`` in the
        folder) and leave with it; any other job's go to :data:`JOBS_DIR`.
        """
        folder = self.bash_fn(f'printf %s "${{{SUBAGENT_DIR_ENV}-}}"', "find the jobs directory",
                              QUICK_TIMEOUT, 2000).strip()
        return f"{folder}/kiss-jobs" if PRIVATE_DIR.fullmatch(folder) else JOBS_DIR

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
        jobs_dir = self._new_job_dir()
        inner = INNER_SCRIPT.format(cmd=shlex.quote(command), d=jobs_dir, id=job_id)
        script = START_SCRIPT.format(token=_EXEC_TOKEN_VAR, inner=shlex.quote(inner), d=jobs_dir,
                                     id=job_id, cmd=shlex.quote(command))
        out = self.bash_fn(script, f"start background job {job_id}", QUICK_TIMEOUT, 2000).strip()
        pid = out.splitlines()[-1].strip() if out else ""
        if not pid.isdigit():
            return None, f"Error: could not start the background job: {out or '(no output)'}"
        self.commands[job_id] = command
        self.dirs[job_id] = jobs_dir
        return job_id, (
            f"Started background job {job_id} (pid {pid}); log: {jobs_dir}/{job_id}.log. "
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
        jobs_dir = self._new_job_dir()
        watch = WATCH_SCRIPT.format(d=jobs_dir, id=job_id, pid=int(pid))
        script = ADOPT_SCRIPT.format(d=jobs_dir, pid=int(pid), id=job_id,
                                     cmd=shlex.quote(command), log=shlex.quote(log_path),
                                     token=_EXEC_TOKEN_VAR, watch=shlex.quote(watch))
        out = self.bash_fn(script, f"adopt process {pid} as job {job_id}", QUICK_TIMEOUT,
                           2000).strip()
        if out.splitlines()[-1:] != ["adopted"]:
            what = "is not running" if out.endswith("gone") else f"could not be adopted: {out}"
            return None, f"Error: process {pid} {what}"
        self.commands[job_id] = command
        self.dirs[job_id] = jobs_dir
        where = f"log {log_path}" if log_path else "no log"
        return job_id, f"left running as job {job_id} ({command}, {where})"

    def last_running(self) -> str | None:
        """Return the id of the job of this registry started last that is still running.

        One script in the container checks every known job's pid and exit
        files, each in the directory that holds them; ``None`` when no job
        is running (one whose files are gone, :data:`GONE_STATUS`, is not
        known to be).
        """
        if not self.commands:
            return None
        by_dir: dict[str, list[str]] = {}
        for job_id in self.commands:
            by_dir.setdefault(self.dirs.get(job_id, JOBS_DIR), []).append(job_id)
        script = "; ".join(RUNNING_SCRIPT.format(ids=" ".join(ids), d=jobs_dir)
                           for jobs_dir, ids in by_dir.items())
        out = self.bash_fn(script, "find the running background jobs", QUICK_TIMEOUT, 2000)
        running = {line.strip() for line in out.splitlines() if JOB_ID.fullmatch(line.strip())}
        return next((job_id for job_id in reversed(self.commands) if job_id in running), None)

    def wait(self, job_id: str, seconds: float) -> str:
        """Block until the job *job_id* exits or *seconds* pass; return its status.

        The ``wait`` action of :meth:`bash_job`, reduced to the status
        line of its report (``running``, ``exited N`` or
        :data:`GONE_STATUS`).  The report opens with the job's command,
        which may hold such a line itself (a here-document printing one):
        the status is read after it.
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

        Every action returns the same report: the job's status (``running``,
        ``exited N``, or ``unknown`` once the job's files are gone from the
        container), the log file path and the last ``tail_lines`` lines of
        the log.  The full log stays in the container at the reported path.

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
        jobs_dir = self.dirs.get(job_id, JOBS_DIR)
        lines = max(1, int(tail_lines))
        report = REPORT_SCRIPT.format(d=jobs_dir, id=job_id, n=lines)
        timeout = QUICK_TIMEOUT
        if action == "wait":
            seconds = max(1, int(timeout_seconds))
            match = ""
            if until:
                complaint = self.bash_fn(UNTIL_CHECK.format(pattern=shlex.quote(until)),
                                         "check the until pattern", QUICK_TIMEOUT, 2000).strip()
                if complaint:
                    return f"Error: until is not a valid grep -E regex: {complaint}"
                match = MATCH_CLAUSE.format(pattern=shlex.quote(until), d=jobs_dir, id=job_id)
            script = WAIT_SCRIPT.format(timeout=seconds, d=jobs_dir, id=job_id, match=match)
            script += report
            timeout = seconds + WAIT_SLACK_SECONDS
        elif action == "kill":
            script = KILL_SCRIPT.format(d=jobs_dir, id=job_id) + "; " + report
        elif action == "tail":
            script = report
        else:
            return f"Error: action must be one of wait, tail, kill; got {action!r}."
        out = self.bash_fn(script, f"{action} background job {job_id}", timeout, max_output_chars)
        return f"Job {job_id}: {self.commands[job_id]}\n{out}"

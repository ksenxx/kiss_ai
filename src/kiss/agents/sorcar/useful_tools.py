# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Useful tools for agents: file editing and bash execution."""

import codecs
import collections
import difflib
import errno
import functools
import hashlib
import io
import json
import logging
import math
import mimetypes
import os
import posixpath
import queue
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from kiss.agents.sorcar.checkpoints import Checkpoints

# Whether the task's Stop or the tool panel's Stop of the call is set: the waits of the host
# and docker Bash tools poll it in 0.2 s slices, so either ends a command promptly.
from kiss.agents.sorcar.checkpoints import _stop_pressed as _stop_requested
from kiss.agents.sorcar.edit_guards import (
    HEREDOC_VERDICT,
    HeredocCopy,
    MechanicalEdit,
    edit_call,
    heredoc_copy,
    heredoc_guard,
    line_of,
    mechanical_edit,
    numbered_region,
    output_shows_file,
    preview,
    region_around_string,
    repo_read_key,
    rewrite_guard,
    shadow_command,
    shell_view_target,
)
from kiss.agents.sorcar.edit_matching import (
    nearest_region,
    reindent,
    split_lines,
    tolerant_matches,
)
from kiss.agents.sorcar.fanout_guard import SUBAGENT_DIR_PREFIX, parse_tasks_json
from kiss.agents.sorcar.git_worktree import (
    _WORKTREE_SLUG_PREFIX,
    _WORKTREE_SUBDIR,
)
from kiss.agents.sorcar.shell_exit import Ran, exit_word, hand_over, unrun
from kiss.agents.sorcar.shell_guards import refuse_or_lift
from kiss.agents.sorcar.shell_scan import (
    UV_VALUED_OPTIONS,
    SimpleCommand,
    TooDeepError,
    Word,
    bash_path,
    host_path,
    simple_commands,
    walks,
)
from kiss.core import tool_interrupt
from kiss.core.config import DEFAULT_CONFIG, kiss_home
from kiss.core.locked import locked_call
from kiss.core.models.model import (
    READ_TOOL_BINARY_MIME_TYPES,
    encode_binary_attachment,
)
from kiss.core.processes import (
    _other_group_members,
    exit_status_unreaped,
    exited_unreaped,
    find_bash,
    kill_process_tree,
    popen_process_group,
    run_process_group,
)
from kiss.core.secret_redaction import StreamRedactor, redact_secrets
from kiss.core.tool_verdict import ToolRedirect, followup_result
from kiss.core.utils import unlink_waiting_for_holders

logger = logging.getLogger(__name__)

_MAX_BINARY_READ_BYTES = 20 * 1024 * 1024
_OUTLINE_SYMBOL_RE = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+\w+|class\s+\w+"  # Python
    r"|(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+\w+|class\s+\w+)"  # JS/TS
    r"|(?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s*)?(?:\([^)]*\)\s*=>|function\b)"
    r"|(?:pub\s+)?(?:fn|struct|enum|impl|trait)\s+\w+"  # Rust
    r"|func\s+(?:\([^)]*\)\s*)?\w+"  # Go
    r")"
)
_OUTLINE_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdx"})
_OUTLINE_MAX_ENTRIES = 400
_OUTLINE_MIN_ENTRIES = 5


def _worktree_index(parts: tuple[str, ...]) -> int | None:
    """Return the index of the worktree marker segment in *parts*.

    The kiss worktree layout is ``<repo>/<subdir>/<prefix><slug>/...``
    (see ``git_worktree``, which owns both constants and creates the
    directories).  Every path guard in this module needs the same scan,
    so it lives here once: duplicating the literals would let a rename
    in ``git_worktree`` silently turn all of them into dead code, and
    the agent would write straight into the user's main checkout.

    When *parts* runs through several nested worktrees (a repo that
    itself lives inside another repo's ``.kiss-worktrees/kiss_wt-*``
    directory, e.g. a project checked out under a kiss worktree, or
    the test suite running from inside one), the *innermost* marker
    is returned: the worktree closest to the path is the one whose
    parent repo the agent is actually working on.  Matching the
    outermost marker instead would make every guard in this module
    treat the outer checkout as "the" parent repo and silently skip
    remap/fallback for the nested one.

    Args:
        parts: ``Path.parts`` of the path (or work dir) to inspect.

    Returns:
        The largest index ``i`` such that ``parts[i]`` is the worktree
        subdir and ``parts[i + 1]`` is a worktree slug, or ``None``
        when *parts* does not run through a worktree.
    """
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == _WORKTREE_SUBDIR and parts[i + 1].startswith(_WORKTREE_SLUG_PREFIX):
            return i
    return None


def _stale_worktree_fallback(resolved: Path) -> Path | None:
    """If *resolved* lives under a now-deleted ``.kiss-worktrees/kiss_wt-*``
    directory, return the equivalent path with that worktree segment
    stripped (i.e. relative to the parent repo).

    Worktrees are torn down by ``commit_and_merge``, so a model that
    remembers a worktree path from earlier in the task ends up with a
    dangling path.  Returning the equivalent in-repo path lets the
    subsequent read succeed transparently.

    A worktree whose root directory still exists on disk is *live*,
    not stale: its working tree is authoritative for paths under it
    (it may have deleted or diverged from the parent repo's copy), so
    no fallback applies and ``None`` is returned.
    """
    parts = resolved.parts
    i = _worktree_index(parts)
    if i is None:
        return None
    if Path(*parts[: i + 2]).is_dir():
        return None
    return Path(*parts[:i], *parts[i + 2 :])


def remap_vanished_worktree(path: Path) -> Path:
    """Return *path*, or its parent-repo equivalent when *path* is missing
    because the ``.kiss-worktrees/kiss_wt-*`` worktree it lived in was
    torn down.

    A finished worktree task's ``work_dir`` keeps naming the removed
    worktree (the task-update side channel, ``run_agent`` sub-tasks
    and :func:`~kiss.agents.sorcar.relentless_agent.resolve_work_dir`
    all start from it).  Every caller that is about to ``mkdir`` or
    operate on such a directory must resolve it through this function
    first: creating the vanished directory resurrects an unregistered
    husk under ``.kiss-worktrees/`` that no cleanup ever removes and
    that disables :func:`_stale_worktree_fallback` for every later
    path under it.

    Nested worktrees (a repo checked out inside another repo's
    worktree) are peeled one level per pass, innermost first, until
    the path exists or no vanished worktree encloses it — stopping
    after one level would hand back a path inside the (also removed)
    outer worktree and resurrect that one instead.

    Args:
        path: An absolute path, possibly inside a vanished worktree.

    Returns:
        *path* when it exists or is not under a vanished worktree;
        otherwise the same path relative to the parent repo.
    """
    while not path.exists():
        fallback = _stale_worktree_fallback(path)
        if fallback is None:
            break
        path = fallback
    return path


def _active_worktree_remap(resolved: Path, work_dir: str | None) -> Path | None:
    """If *work_dir* lives inside a live ``.kiss-worktrees/kiss_wt-*`` worktree
    and *resolved* points to a file in the parent repo (outside any
    worktree), return the equivalent path *inside* the active worktree.
    Otherwise return ``None``.

    This is the symmetric inverse of :func:`_stale_worktree_fallback`:
    while a worktree is *live*, the agent's tool calls must operate on
    the worktree's working tree even when the model emits an absolute
    path that points at the parent repo.  Without this remap, an LLM
    that ignores the ``Work dir:`` hint and edits e.g.
    ``/abs/repo/README.md`` would mutate the user's main checkout,
    leave the worktree clean (so the framework's auto-commit finds
    nothing), and skip the squash-merge entirely — i.e. the
    "why didn't you run the last task in worktree and why didn't you
    commit the changes?" failure mode.

    The remap is structural — it only inspects path strings
    (``.kiss-worktrees/kiss_wt-*`` segments), so it works regardless
    of whether the worktree is currently registered with git.  The
    one exception is a worktree directory that no longer exists on
    disk (torn down by a concurrent cleanup/discard/merge): remapping
    into it would dead-end every Read/Edit ("File not found" for
    files that DO exist in the parent repo) and make Write resurrect
    a zombie worktree directory whose contents are never merged.  In
    that case no remap applies — mirroring the vanished-worktree
    fallback in ``UsefulTools._spawn``, which runs Bash commands from
    the parent repo root in the same situation.

    Args:
        resolved: An already-resolved absolute path the caller is about
            to read/write/edit.
        work_dir: The agent's working directory (usually inside the
            worktree).  May be ``None`` (no worktree → no remap).

    Returns:
        The remapped worktree-internal path, or ``None`` when no remap
        applies (no active worktree in *work_dir*, *resolved* is not
        under the parent repo, or *resolved* is already inside a
        worktree).
    """
    if not work_dir:
        return None
    work_parts = Path(work_dir).resolve().parts
    i = _worktree_index(work_parts)
    if i is None:
        return None
    main_repo_parts = work_parts[:i]
    wt_root_parts = work_parts[: i + 2]
    if not Path(*wt_root_parts).is_dir():
        return None
    res_parts = resolved.parts
    if (
        len(res_parts) <= len(main_repo_parts)
        or res_parts[: len(main_repo_parts)] != main_repo_parts
    ):
        return None
    tail = res_parts[len(main_repo_parts) :]
    if (tail and tail[0] == _WORKTREE_SUBDIR) or _worktree_index(tail) is not None:
        # Already inside a worktree: this one, or the worktree of a
        # repository nested under the parent repo (isolated on its own
        # by ``create_worktree(path)``).
        return None
    return Path(*wt_root_parts, *tail)


def _is_scratch_path(resolved: Path, work_dir: str | None) -> bool:
    """True when *resolved* is an agent scratch file: under a ``tmp`` directory
    inside *work_dir*, anywhere under the cron work directory, or in a
    sub-agent's private folder (:data:`SUBAGENT_DIR_PREFIX`).

    The system prompt sends every scratch file to ``./tmp/`` and cron runs
    keep their notes under ``$KISS_HOME/cron/work``.  Such files are
    rewritten from scratch by each session, so requiring a ``Read`` before
    overwriting one only costs a wasted step: all eleven read-before-write
    refusals in a 24 h audit were on such files.  The ``tmp`` component
    must lie *inside* the work directory: a checkout that itself sits under
    ``/tmp`` or ``~/tmp`` is not scratch, and neither is any path outside
    the work directory.
    """
    if resolved.is_relative_to((kiss_home() / "cron" / "work").resolve()):
        return True
    temp_dir = Path(tempfile.gettempdir()).resolve()
    if resolved.is_relative_to(temp_dir) and "".join(
            resolved.relative_to(temp_dir).parts[:1]).startswith(SUBAGENT_DIR_PREFIX):
        return True
    if not work_dir:
        return False
    root = Path(work_dir).resolve()
    if not resolved.is_relative_to(root):
        return False
    return "tmp" in resolved.relative_to(root).parts[:-1]


def _absolutize(file_path: str, work_dir: str | None) -> str:
    """Anchor a bare relative path under *work_dir* before resolving.

    ``Path("README.md").resolve()`` uses the *host process's*
    ``os.getcwd()`` — not the agent's ``work_dir``.  When the agent
    is running inside a ``.kiss-worktrees/kiss_wt-*`` worktree but
    the host process was launched somewhere unrelated (e.g. the
    daemon's own home directory), a relative path emitted by the
    LLM would resolve to ``<unrelated>/README.md`` and bypass the
    worktree entirely — re-creating the original "task didn't run
    in worktree / no commit" failure mode.

    Joining the relative path under ``work_dir`` first makes the
    subsequent ``_active_worktree_remap`` (and the plain resolution)
    behave consistently regardless of host cwd.

    ``~``-prefixed paths are expanded to the user's home directory
    first (matching the shell semantics the Bash tool already has);
    without this, ``Write("~/notes.txt", ...)`` would silently create
    a directory literally named ``~`` under *work_dir*.
    """
    try:
        p = Path(file_path).expanduser()
    except RuntimeError:
        p = Path(file_path)
    if p.is_absolute():
        return str(p)
    if not work_dir:
        return file_path
    return str(Path(work_dir) / p)


def resolve_tool_path(
    file_path: str,
    work_dir: str | None,
    exists: Callable[[Path], bool] | None = None,
) -> Path:
    """Resolve the path a file tool is about to operate on.

    Anchors *file_path* under *work_dir* (:func:`_absolutize`),
    resolves it, then applies the two worktree corrections in order:
    a path in the parent repo while *work_dir* is inside a live
    worktree is remapped into that worktree
    (:func:`_active_worktree_remap`); otherwise a path under a
    worktree that has since been torn down falls back to the parent
    repo, through every vanished layer of a nested one
    (:func:`remap_vanished_worktree`).

    Args:
        file_path: The path as given to the tool (relative, ``~`` or
            absolute).
        work_dir: The agent's working directory, or ``None``.
        exists: For tools that need the target to already exist
            (``Read``, ``Edit``): the fallback is taken only when
            *exists* fails for the resolved path and holds for the
            fallback.  ``None`` (``Write``, ``screenshot``) takes the
            fallback unconditionally so a vanished worktree is never
            resurrected by a ``mkdir``.

    Returns:
        The fully resolved path to operate on.
    """
    resolved = Path(_absolutize(file_path, work_dir)).resolve()
    remapped = _active_worktree_remap(resolved, work_dir)
    if remapped is not None:
        return remapped
    if exists is not None and exists(resolved):
        return resolved
    fallback = remap_vanished_worktree(resolved)
    if exists is not None and not exists(fallback):
        return resolved
    return fallback.resolve()


def _bash_parent_repo_guard(command: str, work_dir: str | None) -> str | None:
    """Refuse a Bash command that targets the parent repo's working tree.

    When the agent is running inside a ``.kiss-worktrees/kiss_wt-*``
    worktree, shell commands that hard-code an absolute path under
    the *parent* repo's working tree (e.g.
    ``echo X > /abs/repo/README.md``, ``sed -i ... /abs/repo/X``,
    ``rm /abs/repo/X``) silently mutate the user's main checkout,
    leave the worktree clean, and skip the framework's auto-commit
    — exactly the bug that prompted the worktree-path remap for
    Read/Write/Edit.  The Bash tool can't do a clean rewrite (shell
    strings are unstructured), so we refuse the command with an
    actionable error pointing the model at the worktree path.

    The guard only kicks in when:

    * ``work_dir`` is inside a live ``.kiss-worktrees/kiss_wt-*``
      worktree, and
    * the command literally contains the *parent-repo* absolute path
      prefix (not the worktree's prefix — that's a legitimate write
      inside the worktree), outside its ``.venv``: the shared
      environment's binaries (``<repo>/.venv/bin/pytest``) are how a
      worktree runs the project's tools (:func:`uv_in_worktree`).

    Args:
        command: The Bash command line the model wants to run.
        work_dir: The agent's working directory.

    Returns:
        An actionable error string to return to the model in place
        of running the command, or ``None`` when the command is
        allowed to proceed.
    """
    for main_repo, wt_root in _worktree_roots(work_dir):
        pattern = re.escape(main_repo) + rf"(?={_PATH_SEPARATOR_CLASS}|{_PATH_END}|$)"
        for m in re.finditer(pattern, command, flags=_PATH_RE_FLAGS):
            end = m.end()
            while end < len(command) and not re.match(_PATH_END, command[end]):
                end += 1
            hit = command[m.start() : end]
            if Path(hit).is_relative_to(wt_root) or Path(hit).is_relative_to(
                    Path(main_repo) / ".venv"):
                continue
            if _worktree_index(Path(hit).parts[len(Path(main_repo).parts) :]) is not None:
                # The worktree of a repository nested under the parent
                # repo, isolated on its own: not the user's checkout.
                continue
            suggested = rewrite_parent_repo_paths(command, wt_root)
            return (
                f"Error: command references the parent-repo path "
                f"{hit!r}, which is outside the active worktree "
                f"{wt_root!r}.  Rewrite the command to use the "
                f"worktree path (or a path relative to it) so the "
                f"change is captured by the framework's auto-commit "
                f"and does not mutate the user's main checkout. "
                f"Copy this rewritten command as is: {suggested[:20000]}"
            )
    return None


def _worktree_roots(work_dir: str | None) -> list[tuple[str, str]]:
    """Return ``(main_repo, wt_root)`` for each spelling of an active worktree *work_dir*.

    Empty when *work_dir* is not inside a live ``.kiss-worktrees/kiss_wt-*``
    worktree.  Both the given and the resolved spelling are returned when
    they differ (symlinked checkouts).
    """
    if not work_dir:
        return []
    roots: list[tuple[str, str]] = []
    for wd_parts in (Path(work_dir).parts, Path(work_dir).resolve().parts):
        i = _worktree_index(wd_parts)
        if i is None:
            continue
        main_repo, wt_root = str(Path(*wd_parts[:i])), str(Path(*wd_parts[: i + 2]))
        if not os.path.isdir(wt_root):
            continue
        # Git bash on Windows accepts ``C:/repo`` as well as ``C:\repo``,
        # so both separator spellings are returned there (one on POSIX).
        for pair in (
            (main_repo, wt_root),
            (main_repo.replace(os.sep, "/"), wt_root.replace(os.sep, "/")),
        ):
            if pair not in roots:
                roots.append(pair)
    return roots


#: The ``uv`` subcommands that sync the project environment first.
_UV_SYNCING = frozenset({"sync", "run", "add", "remove"})
#: ``uv run`` flags after which the program word names a module of the
#: environment's Python, and ones that make it no program at all (a script
#: path, uv's own help or version).
_UV_MODULE_FLAGS = frozenset({"-m", "--module"})
_UV_NO_PROGRAM_FLAGS = frozenset({
    "-s", "--script", "--gui-script", "-h", "--help", "-V", "--version",
})
#: Where ``uv [global options] sync|run|add|remove`` starts in a command's text.
_UV_COMMAND = re.compile(
    r"(?<![\w./-])uv(?:[ \t]+[^ \t;&|\n]+)*?[ \t]+(sync|run|add|remove)(?=[ \t;&|\n]|$)")


def _after_uv_options(argv: list[str], index: int) -> int:
    """Return the index of the first word of ``argv`` from ``index`` on that is no option.

    An option in :data:`~kiss.agents.sorcar.shell_scan.UV_VALUED_OPTIONS`
    takes the next word with it, so ``uv --directory . run`` finds ``run``
    and ``uv run -p 3.12 pytest`` finds ``pytest``.
    """
    while index < len(argv) and argv[index].startswith("-"):
        index += 2 if argv[index] in UV_VALUED_OPTIONS else 1
    return index


#: Appended to the result of a ``uv run`` rewritten by :func:`uv_in_worktree`:
#: after it, so the result still opens with the status its readers look for
#: (``Error (exit code N):``, ``Error: Command timed out``).
UV_REWRITTEN_NOTE = (
    "\nNote: the worktree has no environment of its own (it shares the main checkout's), "
    "so `uv run` ran as the environment's binary on the worktree's sources; run it so "
    "yourself next time: {command}"
)


def _uv_syncing_command(command: str) -> tuple[list[str], int, bool] | None:
    """The first ``uv sync|run|add|remove`` among *command*'s simple commands.

    Returns:
        ``(argv, at, wrapped)``: the command's words, the index of the
        subcommand and whether a wrapper (``timeout 5``, ``command``)
        precedes ``uv``, or ``None`` when no simple command is one
        (``uv`` in a quoted word or a here-document body is text, not a
        command).
    """
    for simple in simple_commands(command, None):
        words = simple.argv()
        argv = [word.text for word in words]
        at = _after_uv_options(argv, 1) if argv and argv[0] == "uv" else len(argv)
        if at < len(argv) and argv[at] in _UV_SYNCING:
            prefix = simple.words[:len(simple.words) - len(words)]
            return argv, at, any("=" not in word.text for word in prefix)
    return None


def _argvs(command: str) -> list[list[str]]:
    """Return the words of each simple command of *command*, assignments and wrappers dropped."""
    return [[word.text for word in simple.argv()] for simple in simple_commands(command, None)]


def _uv_run_rewritten(command: str, argv: list[str], at: int, environment: Path,
                      sources: str, wrapped: bool) -> str | None:
    """*command* with its first ``uv run PROGRAM`` replaced by the environment's binary.

    Args:
        command: The command text.
        argv: The words of its ``uv`` simple command (:func:`_uv_syncing_command`).
        at: The index of the subcommand in *argv*.
        environment: The main checkout's environment.
        sources: The ``PYTHONPATH=...`` prefix that puts the worktree's sources first.
        wrapped: Whether a wrapper precedes ``uv``; the prefix then opens
            with ``env``, which keeps the assignment one after ``timeout 5``.

    Returns:
        The rewritten command, or ``None`` when the rewrite is not
        unambiguous: a subcommand other than ``run``, an option with a
        value before or after ``run`` (``--directory x`` changes what
        runs where, ``--with x`` what is installed; flags such as ``-q``
        are dropped without harm), a flag after which the program word
        names no program of the environment (``--script``, ``--help``;
        ``-m MODULE`` runs as the environment's Python with ``-m``), a
        program the environment lacks, or a text in which no occurrence
        of the command's words rewrites exactly that command (the words
        inside an earlier quoted argument or here-document are text).
    """
    if argv[at] != "run":
        return None
    index = _after_uv_options(argv, at + 1)
    options = argv[1:at] + argv[at + 1:index]
    if any(o in UV_VALUED_OPTIONS or "=" in o or o in _UV_NO_PROGRAM_FLAGS for o in options):
        return None
    program = argv[index] if index < len(argv) else ""
    module = any(o in _UV_MODULE_FLAGS for o in options)
    if module and not re.fullmatch(r"[\w.]+", program):
        return None
    binary = environment / "bin" / (
        "python" if module or program in ("python", "python3") else program)
    if not program or not binary.is_file():
        return None
    replacement = ("env " if wrapped else "") + f"{sources} {shlex.quote(str(binary))}"
    run = [str(binary)]
    if module:
        replacement += f" -m {program}"
        run += ["-m", program]
    before = _argvs(command)
    position = before.index(argv)
    expected = before[:position] + [run + argv[index + 1:]] + before[position + 1:]
    for match in _UV_COMMAND.finditer(command):
        # The program's word in the text: the (index - at)-th after ``run``.
        words = list(re.finditer(r"\S+", command[match.end():]))
        if len(words) <= index - at - 1 or words[index - at - 1].group().strip("'\"") != program:
            continue
        end = match.end() + words[index - at - 1].end()
        rewritten = command[:match.start()] + replacement + command[end:]
        if _argvs(rewritten) == expected:
            return rewritten
    return None


def uv_in_worktree(command: str, work_dir: str | None) -> tuple[str | None, str, str]:
    """Rewrite or refuse a ``uv sync``/``run``/``add``/``remove`` run inside a worktree.

    A worktree shares the main checkout's environment (``.venv``) and has
    none of its own, so ``uv`` either builds it one (every dependency
    again, on whatever Python it finds newest) or, when
    ``UV_PROJECT_ENVIRONMENT`` or a ``.venv`` link names the main
    checkout's environment, installs the worktree as that environment's
    editable project, a path that is removed when the work is merged
    (the 2026-11-01 audit counted 19 ``check --full`` runs doing so).
    A ``uv run PROGRAM`` whose rewrite is unambiguous
    (:func:`_uv_run_rewritten`) runs as the main checkout's environment
    binary with the worktree's sources first on ``PYTHONPATH``, so a
    project script (``check``, ``pytest``, ``python``) runs on the
    worktree's code in the daemon's own environment, and the result says
    so (:data:`UV_REWRITTEN_NOTE`; the 2026-11-03 audit counted 6
    refusals whose rewritten command the agent then copied by hand).  Any
    other such ``uv`` command is refused naming what to run instead.

    This reads the command line only.  A ``uv`` run it cannot see (a
    script, a Makefile, a sub-process) is refused by uv itself where the
    worktree's ``.venv`` is a file
    (:meth:`~kiss.agents.sorcar.git_worktree.GitWorktreeOps.block_own_environment`).

    With no ``.venv`` in the main checkout there is no environment to
    share and no such file: the command runs as written, typed as in a
    script, and uv builds the worktree's own environment, its one Python.

    Args:
        command: The Bash command line the model wants to run.
        work_dir: The agent's working directory.

    Returns:
        ``(refusal, note, command)``: the refusal (``None`` when the
        command may run: it is not in a worktree, the main checkout has
        no environment, it runs no such ``uv`` command, or was
        rewritten), the note to append to the result (``""`` when
        nothing was rewritten) and the command to run.
    """
    roots = _worktree_roots(work_dir)
    if not roots:
        return None, "", command
    main_repo, wt_root = roots[0]
    environment = Path(main_repo) / ".venv"
    if not environment.is_dir():
        return None, "", command
    sources = "PYTHONPATH=" + shlex.quote(f"{wt_root}/src:{wt_root}/projects")
    binaries = shlex.quote(str(environment)) + "/bin"
    original = command
    while (found := _uv_syncing_command(command)) is not None:
        argv, at, wrapped = found
        rewritten = _uv_run_rewritten(command, argv, at, environment, sources, wrapped)
        if rewritten is None:
            return (
                f"REFUSED: `uv {argv[at]}` did NOT run: the worktree {wt_root!r} has no "
                f"environment of its own and shares the main checkout's ({environment}); a "
                f"sync from here would build the worktree one or re-point that environment's "
                f"editable install at the worktree, which is removed when the work is merged.  "
                f"Run the environment's binaries directly, with the worktree's sources first: "
                f"{sources} {binaries}/<program> ...  (the project check: "
                f"{sources} {binaries}/check --full --fix)."
            ), "", original
        command = rewritten
    if command == original:
        return None, "", command
    return None, UV_REWRITTEN_NOTE.format(command=command[:20000]), command


# Windows paths are case-insensitive, so the parent-repo guard must not be
# bypassable by re-casing ``C:\Repo`` as ``c:\repo`` (or ``C:\Répo`` as
# ``c:\répo``: NTFS folds non-ASCII letters too, so the folding is the
# full Unicode one — a guard may refuse too much, never too little).
# POSIX stays exact.  Likewise only Windows treats ``\`` as a separator.
_PATH_RE_FLAGS = re.IGNORECASE if sys.platform == "win32" else 0
_PATH_SEPARATOR_CLASS = r"[/\\]" if sys.platform == "win32" else "/"
# Characters that end a path inside a shell command or task text; shared by
# the parent-repo guard and the rewrite so the two never disagree on where
# a path stops.
_PATH_END = r"[\s'\";|&<>()`,:]"
# What a rewrite leaves under the parent repository, as the guard lets it
# through: another worktree, and the environment every worktree shares.
_KEPT_UNDER_REPO = (
    rf"{re.escape(_WORKTREE_SUBDIR)}|\.venv(?={_PATH_SEPARATOR_CLASS}|{_PATH_END}|$)"
)


def rewrite_parent_repo_paths(text: str, work_dir: str | None) -> str:
    """Rewrite absolute parent-repo paths in *text* to the active worktree's.

    Sub-agent tasks written by a parent running in a worktree kept
    naming files under the parent repository (134 refused Bash commands
    in the 7-day audit of 2026-09-19).  Every ``<main_repo>/...`` path
    that is not already inside a ``.kiss-worktrees`` directory becomes
    ``<wt_root>/...``; the bare repo path itself is rewritten too.
    A path under ``<main_repo>/.venv`` stays: that is the environment
    the worktree runs on (:func:`uv_in_worktree`), the worktree has
    none, and a brief that names its ``bin/pytest`` or its Python must
    reach the sub-agent as written (on 2026-11-12 six reviewers' briefs
    were rewritten onto a stray environment in the worktree, on another
    Python).

    The rewrite is textual: prose that deliberately names the parent
    repository ("do not touch <main_repo>") is rewritten too.  A parent
    that needs the child to act on the main checkout must say so
    without an absolute path (the worktree is the only tree a
    sub-agent's tools may modify anyway).

    Args:
        text: A task description or shell command.
        work_dir: The dispatching agent's working directory.

    Returns:
        *text* with the paths rewritten, or unchanged when *work_dir*
        is not inside a live worktree.
    """
    for main_repo, wt_root in _worktree_roots(work_dir):
        pattern = re.escape(main_repo) + rf"(?={_PATH_SEPARATOR_CLASS}|{_PATH_END}|$)"
        text = re.sub(
            pattern, functools.partial(_swap_root, wt_root=wt_root), text, flags=_PATH_RE_FLAGS
        )
    return text


def _swap_root(match: re.Match[str], wt_root: str) -> str:
    """Replace one matched repo root unless it names a worktree or the shared environment."""
    rest = match.string[match.end() :]
    if rest[:1] in ("/", "\\") and re.match(_KEPT_UNDER_REPO, rest[1:], flags=_PATH_RE_FLAGS):
        return match.group(0)
    return wt_root


def _suggest_close_path(resolved: Path) -> str:
    """Return a ``Did you mean: …`` suffix for a missing file, or ``""``.

    Looks in *resolved*'s parent directory for the closest filename
    (case-insensitive) via :func:`difflib.get_close_matches`.  If the
    parent itself does not exist, walks upward to the nearest existing
    ancestor and suggests an entry from it.
    """
    parent = resolved.parent
    while parent != parent.parent and not parent.is_dir():
        parent = parent.parent
    if not parent.is_dir():
        return ""
    try:
        names = [p.name for p in parent.iterdir()]
    except OSError:  # pragma: no cover — permission edge case
        return ""
    matches = difflib.get_close_matches(resolved.name, names, n=1, cutoff=0.6)
    if not matches:
        lowered = {n.lower(): n for n in names}
        ci = difflib.get_close_matches(resolved.name.lower(), list(lowered), n=1, cutoff=0.6)
        if ci:
            matches = [lowered[ci[0]]]
    if matches:
        return f" Did you mean: {parent / matches[0]} ?"
    return ""


def _outline(file_path: str, lines: list[str], size: int) -> str | None:
    """Build the outline a whole-file ``Read`` of a long file returns.

    Args:
        file_path: The path as the model passed it (for the header).
        lines: The file's lines.
        size: The file size in characters.

    Returns:
        The line count plus a ``line: symbol`` list of definitions and
        headings and how to read a range, or ``None`` when the file has
        too few recognisable symbols for an outline to be useful (the
        caller then returns the first window as before).
    """
    pattern = (
        _OUTLINE_HEADING_RE
        if Path(file_path).suffix.lower() in _MARKDOWN_SUFFIXES
        else _OUTLINE_SYMBOL_RE
    )
    entries = [
        f"{number}: {line.strip()}" for number, line in enumerate(lines, 1) if pattern.match(line)
    ]
    if len(entries) < _OUTLINE_MIN_ENTRIES:
        return None
    shown = entries[:_OUTLINE_MAX_ENTRIES]
    more = len(entries) - len(shown)
    return (
        f"{file_path}: {len(lines):,} lines, {size:,} chars. Too long to send "
        f"whole; outline of its {len(entries)} definitions/headings (line: text):\n"
        + "\n".join(shown)
        + (f"\n... {more} more entries" if more else "")
        + "\n\nRead a range with Read(file_path, start_line=N, max_lines=M), or "
        'locate text with Bash("grep -n PATTERN FILE").'
    )


_BASH: str | None = find_bash()


def _popen_kwargs(command: str) -> dict[str, Any]:
    """Return Popen kwargs appropriate for the current platform.

    Invokes bash directly wherever it is found (the tool is ``Bash``:
    ``shell=True`` runs ``/bin/sh``, dash on Debian, which expands no
    braces and has no arrays); on Unix without bash, falls back to the
    platform shell, on Windows without Git bash to PowerShell.  Spawn
    with :func:`popen_process_group` so :func:`kill_process_group` can
    stop the child together with everything it spawned.

    Args:
        command: The command string to execute.

    Returns:
        Dict of keyword arguments for ``subprocess.Popen``.
    """
    if _BASH:
        return {"args": [_BASH, "-c", command]}
    if sys.platform != "win32":  # pragma: no cover — a host without bash
        return {"args": command, "shell": True}
    ps = shutil.which("pwsh") or shutil.which("powershell") or "powershell"  # pragma: no cover
    return {"args": [ps, "-NoProfile", "-Command", command]}


def _truncate_output(output: str, max_chars: int, spilled_to: str = "", dropped: int = 0) -> str:
    """Return *output* cut to *max_chars*: its head and tail around a marker.

    A stored secret is replaced by its variable's name before the cut
    (:func:`~kiss.core.secret_redaction.redact_secrets`): a cut inside
    one would leave a part of the value that no later pass recognises.

    Args:
        output: The full text, or its head and tail when *dropped*.
        max_chars: Characters kept, marker included.
        spilled_to: Path of the file holding the full text (named in
            the marker so the model can grep it), ``""`` when none.
        dropped: Characters already missing between the head and the
            tail of *output* (see :class:`_Capture`); at most
            *max_chars* of each end may be kept then.
    """
    output = redact_secrets(output)
    total = len(output) + dropped
    if total <= max_chars:
        return output
    where = f"; full output ({total} chars) in {spilled_to}" if spilled_to else ""
    worst_msg = f"\n\n... [truncated {total} chars{where}] ...\n\n"
    if max_chars < len(worst_msg):
        return output[:max_chars]
    remaining = max_chars - len(worst_msg)
    head = remaining // 2
    tail = remaining - head
    msg = f"\n\n... [truncated {total - head - tail} chars{where}] ...\n\n"
    if tail:
        return output[:head] + msg + output[-tail:]
    return output[:head] + msg


_CAPTURE_CHARS = 1_000_000
"""Characters of each end of a command's output kept once it passes twice this."""
_READ_CHARS = 1 << 16
"""The longest piece of output read at once: a longer line arrives in pieces."""
_QUEUED_PIECES = 64
"""Pieces of output a reader thread may read ahead of the thread taking them."""


class _Capture:
    """The combined output of a command, bounded in memory.

    All of it is kept until it passes twice :data:`_CAPTURE_CHARS`; from
    then on only the first :data:`_CAPTURE_CHARS` and (to the piece) the
    last :data:`_CAPTURE_CHARS` stay.  Once the output passes
    *spill_after* characters (at the latest when the first piece is
    dropped) and *new_spill* gives a file, the whole output goes on to
    that file as it arrives, so it can still be grepped.  A result cut
    to its budget (:func:`_truncate_output`) needs no more than what is
    kept.

    Attributes:
        dropped: Characters left out between the kept head and tail.
        spilled_to: The file holding the whole output, as shown to the
            model; ``""`` when there is none.
    """

    def __init__(
        self,
        spill_after: int = 0,
        new_spill: Callable[[], tuple[str, Path] | None] | None = None,
    ) -> None:
        """Start empty.

        Args:
            spill_after: Characters of output that fit without a spill
                file.
            new_spill: Returns the shown and the real path of a new spill
                file, ``None`` when there is no directory for one; ``None``
                keeps no spill file.
        """
        self._spill_after = min(spill_after, 2 * _CAPTURE_CHARS)
        self._new_spill = new_spill
        self._head = ""
        self._tail: collections.deque[str] = collections.deque()
        self._tail_chars = 0
        self._file: IO[str] | None = None
        self.dropped = 0
        self.spilled_to = ""

    def append(self, piece: str) -> None:
        """Add the next *piece* of output."""
        self._tail.append(piece)
        self._tail_chars += len(piece)
        if self._file is not None:
            self._write(piece)
        elif self._new_spill is not None and self._tail_chars > self._spill_after:
            self._start_spill(self._new_spill)
        if self._head:
            self._trim()
        elif self._tail_chars > 2 * _CAPTURE_CHARS:
            whole = "".join(self._tail)
            self._head = whole[:_CAPTURE_CHARS]
            self._tail = collections.deque([whole[_CAPTURE_CHARS:]])
            self._tail_chars -= _CAPTURE_CHARS

    def text(self) -> str:
        """Return the output kept: all of it, or its head and tail when :attr:`dropped`."""
        return self._head + "".join(self._tail)

    def close(self) -> None:
        """Close the spill file, if one was started; a failed close gives it up."""
        if self._file is None:
            return
        try:
            self._file.close()
        except OSError:
            self.spilled_to = ""
        self._file = None

    def _trim(self) -> None:
        """Drop the oldest pieces of the tail that are not among its last characters."""
        while self._tail_chars - len(self._tail[0]) >= _CAPTURE_CHARS:
            piece = self._tail.popleft()
            self._tail_chars -= len(piece)
            self.dropped += len(piece)

    def _start_spill(self, new_spill: Callable[[], tuple[str, Path] | None]) -> None:
        """Open the one spill file *new_spill* gives and write the output so far to it.

        Runs once, before anything is dropped, so the file gets the
        whole output.
        """
        self._new_spill = None
        spill = new_spill()
        if spill is None:
            return
        shown, path = spill
        try:
            self._file = open(path, "w", encoding="utf-8", errors="replace", newline="")
        except OSError:
            return
        self.spilled_to = shown
        self._write("".join(self._tail))

    def _write(self, text: str) -> None:
        """Append *text* to the spill file; a failed write gives the file up (no longer whole)."""
        assert self._file is not None
        try:
            self._file.write(text)
        except OSError:
            self.close()
            self.spilled_to = ""


def _make_dirs_under(root: str, relative: str) -> Path:
    """Create the directories of the relative path *relative* under *root*; return the last.

    Unlike ``mkdir(parents=True)`` this never creates *root* itself: a
    worktree that vanished after :meth:`UsefulTools._resolve_cwd` chose
    it is not brought back as an empty directory.

    Raises:
        OSError: When *root* is gone or a directory cannot be made.
    """
    path = Path(root)
    for part in Path(relative).parts:
        path /= part
        path.mkdir(exist_ok=True)
    return path


def _drain_stdout(
    stdout: IO[str], out_queue: queue.Queue[str | None], abandoned: threading.Event,
) -> None:
    """Reader thread of :meth:`UsefulTools._bash_streaming`: queue *stdout*, then ``None``.

    Lines are queued as they arrive, a line longer than
    :data:`_READ_CHARS` in pieces of that size.  *out_queue* is bounded,
    so a command that prints faster than its output is taken waits on
    the pipe instead of filling memory.  Once *abandoned* is set nobody
    takes from the queue any more: the rest is read and dropped, so a
    child still holding the inherited pipe never blocks on it.  The
    reader is the pipe's only user, so it owns the close: EOF means
    every writer let go, whether the caller is still waiting or
    returned (abandoned) long ago.
    """
    try:
        for piece in iter(functools.partial(stdout.readline, _READ_CHARS), ""):
            _put_unless_abandoned(out_queue, piece, abandoned)
    finally:
        try:
            stdout.close()
        except Exception:
            logger.debug("stdout close failed", exc_info=True)
        _put_unless_abandoned(out_queue, None, abandoned)


def _put_unless_abandoned(
    out_queue: queue.Queue[str | None], item: str | None, abandoned: threading.Event,
) -> None:
    """Put *item* on the bounded *out_queue*, waiting for room until *abandoned* is set."""
    while not abandoned.is_set():
        with suppress(queue.Full):
            out_queue.put(item, timeout=0.2)
            return


def _edit_pairs(edits: str) -> list[tuple[str, str, bool]] | str:
    """Parse the ``edits`` JSON list of an Edit call into ``(old, new, replace_all)`` triples.

    Returns the error text instead when the JSON is not a non-empty
    list of objects with string ``old_string`` and ``new_string``, or
    when a ``replace_all`` is neither a boolean nor left out (read by
    its truth, the string ``"false"`` replaced every occurrence).
    """
    try:
        items = json.loads(edits)
    except (ValueError, RecursionError) as e:  # RecursionError: nested past the decoder's depth
        return f"Error: edits must be a JSON list of objects: {e}"
    if not isinstance(items, list) or not items:
        return "Error: edits must be a non-empty JSON list of objects"
    pairs: list[tuple[str, str, bool]] = []
    for index, item in enumerate(items, 1):
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("old_string"), str)
            or not isinstance(item.get("new_string"), str)
        ):
            return (
                f"Error: edit {index} must be an object with string old_string and new_string"
            )
        every = item.get("replace_all", False)
        if not isinstance(every, bool):
            return f"Error: edit {index}: replace_all must be true or false"
        pairs.append((item["old_string"], item["new_string"], every))
    return pairs


def _crlf_edit(old: str, new: str, crlf: bool) -> tuple[str, str]:
    """Return *old* and *new* with CRLF line ends when the file has them (*crlf*), *old* none.

    ``Read`` shows a CRLF file with LF line ends, so an edit written from
    its output is matched (and its new text written) as if it had been
    written with CRLF.  Other edits are returned unchanged.
    """
    if crlf and "\r\n" not in old:
        return old.replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
    return old, new


def _moved_by_later_edit(
    placed: list[tuple[int, int, str]], starts: list[int], old_len: int, new: str,
) -> list[tuple[int, int, str]]:
    """Return the earlier edits' ``(position, occurrences, text)`` after a later edit.

    The later edit replaced *old_len* characters at each of *starts*
    (ascending, non-overlapping) with *new*.  An earlier edit below an
    occurrence moves by the occurrence's net length; one inside an
    occurrence was swallowed by it, so its report now points at the
    replacement that stands there (never at a negative position).
    """
    shift = len(new) - old_len
    moved: list[tuple[int, int, str]] = []
    for at, count, text in placed:
        above = sum(1 for s in starts if s + old_len <= at)
        covering = [s for s in starts if s <= at < s + old_len]
        if covering:
            at, text = covering[0], new
        moved.append((at + shift * above, count, text))
    return moved


def _nearest(content: str, old: str) -> str:
    """Name the region of *content* closest to a not-found *old* string, numbered; ``""`` for none.

    Appended to the Edit's not-found error (:func:`nearest_region`), so
    the next Edit can quote the region verbatim without a Read: its
    lines are shown as Read shows them, a ``\\r\\n`` as ``\\n`` and any
    other ``\\r`` as it is.
    """
    region = nearest_region(content, old)
    if region is None:
        return ""
    first, last, text = region
    lines = text.replace("\r\n", "\n").removesuffix("\n").split("\n")
    numbered = "\n".join(f"{number:5d} | {line}" for number, line in enumerate(lines, first))
    return (
        f" The closest region is lines {first}-{last}:\n{numbered}\n"
        "Re-issue the Edit with the exact text of those lines as old_string."
    )


_LOG_SCAN_BYTES = 8_000_000
"""The most of a job log ``bash_job(until=...)`` searches (its end, when the log is longer)."""


def _read_end(path: Path, max_bytes: int) -> str:
    """Return the text of the last *max_bytes* of *path*, from the first line start in them.

    When the read starts mid-file the partial first line is dropped: a
    line starts after ``\\n``, ``\\r\\n`` or a lone ``\\r`` (a progress
    bar rewrites one line for a whole run).  A window holding no line
    start at all is the end of one line and is kept as such, so a log
    of one long line is not reported as empty.  Invalid UTF-8 is
    replaced.  Bytes appended after the size was taken (a job log
    grows) are not read.

    Raises:
        OSError: When the file cannot be read.
    """
    with open(path, "rb") as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - max_bytes)
        # One byte before the window tells whether the window starts
        # on a line boundary (that byte is a line break) or mid-line.
        f.seek(max(0, start - 1))
        data = f.read(size - max(0, start - 1))
    if start > 0:
        line_break = re.search(rb"\r\n|[\r\n]", data)
        data = data[line_break.end():] if line_break else data[1:]
    return data.decode("utf-8", errors="replace")


def _log_matches(log_path: Path, pattern: re.Pattern[str]) -> bool:
    """Return whether *pattern* occurs in the job log at *log_path* (unreadable = no).

    A log of at most :data:`_LOG_SCAN_BYTES` is searched whole, so an
    anchored or multi-line pattern (``^READY``, ``START[\\s\\S]*END``)
    keeps its meaning; a longer one only in its last
    :data:`_LOG_SCAN_BYTES`, from a line start (where ``\\A`` then
    matches).  The wait scans once a second: rereading a log of
    gigabytes each time would hold that much of the daemon's memory
    and delay a Stop.
    """
    try:
        return pattern.search(_read_end(log_path, _LOG_SCAN_BYTES)) is not None
    except OSError:
        return False


def _match_log_until(log_path: str, pattern: str) -> None:
    """Return once the job log at *log_path* matches *pattern*, scanning it once a second.

    The body of the matcher's search worker: a pattern that backtracks
    for minutes (``(a+)+$`` on a long line, the 2026-11-04 audit) holds
    the GIL for as long in whichever process searches, so the search
    runs in a child the wait kills at its deadline or a Stop instead of
    on the task thread.  Nothing in this process can end it meanwhile
    (no thread gets the GIL back during the search): its supervisor
    (:data:`_SUPERVISE_LOG_MATCHER`) kills it once the daemon is gone.
    """
    compiled = re.compile(pattern)
    while not _log_matches(Path(log_path), compiled):
        time.sleep(1.0)


# The matcher's supervisor, a ``python -c`` body (light: no kiss import): starts the search
# worker its arguments name, exits with the worker's status, and kills the worker at end of
# file on its stdin, a pipe whose only write end the daemon holds (non-inheritable, and the
# daemon never forks), so the worker dies with the daemon on every platform, however long its
# search holds the GIL.  Each Stop route ends both at once through the process group.  The
# worker is killed as well when this process ends early (no thread to watch its stdin with).
_SUPERVISE_LOG_MATCHER = """
import os, subprocess, sys, threading
worker = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL)
def kill_at_eof():  # os.read: sys.stdin's buffer lock would deadlock the exit below
    try:
        while os.read(0, 4096):
            pass
    finally:
        worker.kill()
try:
    threading.Thread(target=kill_at_eof, daemon=True).start()
    status = worker.wait()
finally:
    worker.kill()
    worker.wait(5)
sys.exit(status)
"""


# The one-shot search worker of an ``until`` poll (:func:`_searched`): exits 0
# when the file its first argument names matches the pattern its second names, 1 when not.
# No kiss import: it starts once per run of the poll.
_SEARCH_FILE_ONCE = (
    "import re, sys; sys.exit(0 if re.search(sys.argv[2], open(sys.argv[1], encoding='utf-8', "
    "errors='surrogatepass').read()) else 1)"
)


def _start_log_matcher(
    log_path: Path, until: re.Pattern[str], once: bool = False,
) -> subprocess.Popen:
    """Start the matcher: a child that exits 0 once the job log at *log_path* matches *until*.

    The child is the supervisor (:data:`_SUPERVISE_LOG_MATCHER`) of the
    worker that searches (:func:`_match_log_until`; with *once*,
    :data:`_SEARCH_FILE_ONCE`, which searches the file as it is and
    exits 1 when it does not match), in one process group with it, so
    :func:`kill_process_tree` ends both; its stdin is a pipe from this
    process, whose death ends both as well.
    """
    search = _SEARCH_FILE_ONCE if once else (
        "import sys; from kiss.agents.sorcar.useful_tools import _match_log_until; "
        "_match_log_until(*sys.argv[1:])"
    )
    return popen_process_group(
        [sys.executable, "-c", _SUPERVISE_LOG_MATCHER]
        + [sys.executable, "-c", search, str(log_path), until.pattern],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _exits_within(process: subprocess.Popen, seconds: float) -> bool:
    """Return whether *process* exits within *seconds*, without reaping it.

    ``Popen.wait(timeout)`` with the reap left out (the same poll, its
    sleeps doubling from 1 ms to 50 ms): the exited leader keeps its
    pid, so the thread that owns it can still signal its group by that
    number (:func:`kill_process_group` documents the pid-reuse hazard
    of signalling a reaped leader).
    """
    deadline = time.monotonic() + seconds
    delay = 0.0005
    while not exited_unreaped(process):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        delay = min(delay * 2, remaining, 0.05)
        time.sleep(delay)
    return True


def _run_log_matcher(
    start: Callable[[], subprocess.Popen],
    done: threading.Event,
    stops: list[threading.Event],
    deadline: float,
    matched: threading.Event,
    failed: list[str],
) -> None:
    """Own the matcher child of one wait: *start* it, set *matched* if it exits 0, end it in time.

    Runs on a daemon thread per wait.  The waiting task thread cannot
    own the child itself: the runner's Stop lands at any check point of
    its bytecode, including inside ``Popen`` after the fork and right
    after :func:`_start_log_matcher` returned, before a handle a
    ``finally`` could kill is bound (R1 of the 2026-11-04 review).  This
    thread receives no injection, and *done*, which the wait's
    ``finally`` sets, exists before the child does.  A Stop landing at
    that ``finally``'s first statement skips the whole cleanup, *done*
    included (R2 of the review), so the child is ended as well once one
    of *stops* fires (the task's stop event and the tool call's
    interrupt event, thread-local and so captured by the waiting
    thread) or *deadline* (``time.monotonic``, the wait's own) passes.
    The child's whole group is killed while the child is still
    unreaped (:func:`kill_process_group` documents the pid-reuse hazard
    of signalling a reaped leader), also when the child ended on its
    own (a supervisor killed from outside leaves its worker in the
    group), and the child is reaped here.  A launch that fails (the
    daemon at its process or descriptor limit, an argument ``Popen``
    refuses) is reported through *failed*, and so is a child that
    ended on its own without a match (a wait's worker searches until
    it matches: it crashed, or its supervisor was killed), so the wait
    ends with an error instead of running to its deadline with no search.
    The child's exit is seen as it happens (:func:`_exits_within` in
    0.2 s slices, between which *done*, *stops* and *deadline* are
    checked): an ``until`` poll searches once per run, and a run that
    waited out a 0.2 s tick for its verdict added that much to every poll.
    """
    try:
        process = start()
    except (OSError, ValueError) as e:
        failed.append(str(e))
        return
    exited = False
    try:
        while not exited and time.monotonic() < deadline and not done.is_set():
            if any(stop.is_set() for stop in stops):
                break
            exited = _exits_within(process, 0.2)
    finally:
        kill_process_tree(process)
        status = process.wait()
        assert process.stdin is not None  # the lifeline: closed here, not by the collector
        process.stdin.close()
        if status == 0:
            matched.set()
        elif exited:
            failed.append(f"it exited with status {status} before a match")


def _search_output(
    output: str,
    regex: re.Pattern[str],
    done: threading.Event,
    stops: list[threading.Event],
    deadline: float,
    matched: threading.Event,
) -> None:
    """Search *output* for *regex* in a matcher child, owning the spool file it reads.

    The helper thread of :func:`_searched`: the output is
    spooled to a temporary file, the one-shot matcher
    (:func:`_start_log_matcher` with ``once``) is run on it as
    :func:`_run_log_matcher` runs a wait's, and the file is removed
    here, on a thread no Stop injection reaches, so a Stop landing
    anywhere in the task thread's frame leaves no file behind.

    Args:
        output: The text to search.
        regex: The pattern.
        done: Set by the task thread when it stops waiting.
        stops: The events that end the search early.
        deadline: ``time.monotonic`` time at which the search is ended.
        matched: Set when the child found a match.
    """
    spool = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", errors="surrogatepass", suffix=".txt", delete=False,
    )
    path = Path(spool.name)
    start = functools.partial(_start_log_matcher, path, regex, True)
    try:
        with spool:
            spool.write(output)
        # Nothing reads the reasons: a matcher that could not be launched is a search that
        # found nothing, and an exit without a match is the one-shot worker's verdict.
        _run_log_matcher(start, done, stops, deadline, matched, [])
    finally:
        path.unlink(missing_ok=True)


def _searched(
    pattern: re.Pattern[str], output: str, deadline: float, cancel: threading.Event | None,
) -> bool:
    """Return whether *pattern* occurs in *output*: a child's search, ended at *deadline* or a Stop.

    A pattern that backtracks for minutes (``(a+)+$`` on a long line)
    holds the GIL of whichever process searches (:func:`_match_log_until`),
    so the search runs in the matcher child of ``bash_job``'s wait
    (one-shot: the output goes to a file the worker reads once), owned
    by a helper thread (:func:`_search_output`) that ends it at
    *deadline*, on *cancel* or on the tool call's interrupt, while this
    thread waits in slices a Stop can land in.  A search ended that way
    counts as no match: the poll of the host and docker ``Bash`` tools
    ends there anyway.
    """
    done, matched = threading.Event(), threading.Event()
    stops = [
        event for event in (cancel, tool_interrupt.current_tool_interrupt_event())
        if event is not None
    ]
    owner = threading.Thread(
        target=_search_output, args=(output, pattern, done, stops, deadline, matched),
        daemon=True,
    )
    try:
        owner.start()
        while owner.is_alive():
            owner.join(0.2)
    finally:
        done.set()
    return matched.is_set()


def _until_pattern(until: str) -> re.Pattern[str] | str:
    """Compile the ``until`` regex of a poll or a wait, or return the ``Error:`` text refusing it.

    Refused like a pattern that does not compile is one that cannot be
    a program argument of the matcher child (:func:`_start_log_matcher`):
    it holds a NUL, or a character the file-system encoding has no
    bytes for (a lone surrogate).  ``Popen`` raised on those before the
    fork, so nothing was searched and the call ran to its deadline.
    """
    try:
        encoded = until.encode("utf-8")
        if b"\0" in encoded:
            raise ValueError("embedded null byte")
        return re.compile(until)
    except (re.error, UnicodeEncodeError, ValueError) as e:
        return f"Error: until is not a valid regular expression: {e}"


FOREGROUND_SLEEP_MAX_SECONDS = 30
# A literal ``sleep`` operand: a number with GNU sleep's optional unit suffix.
_SLEEP_OPERAND = re.compile(r"(\d+(?:\.\d*)?|\.\d+)([smhd]?)")
_SLEEP_UNIT_SECONDS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def _sleep_seconds(operands: list[Word]) -> float:
    """Return how long ``sleep`` waits for *operands*: the sum of the literal ones, as GNU sleep.

    ``inf``/``infinity`` is forever; an operand only the shell knows
    (``"$N"``) or that sleep rejects counts as nothing.
    """
    total = 0.0
    for word in operands:
        if not word.literal:
            continue
        if word.text.lower() in ("inf", "infinity"):
            return math.inf
        match = _SLEEP_OPERAND.fullmatch(word.text)
        if match:
            total += float(match[1]) * _SLEEP_UNIT_SECONDS[match[2]]
    return total


def long_sleep(command: str, limit: float) -> tuple[float, list[str], bool] | None:
    """Return the first ``sleep`` of *command* longer than *limit*: its seconds, operands, looped.

    Every ``sleep`` the shell runs counts (read with
    :func:`~kiss.agents.sorcar.shell_scan.simple_commands`: in a loop,
    a substitution, behind ``nohup``), with GNU sleep's ``s``/``m``/
    ``h``/``d`` suffixes and the sum of its operands.  A ``sleep N &``
    counts too: the child keeps the call's output pipe, so the call
    blocks until the sleep ends all the same.  Text that only mentions
    a sleep (quoted, a here-document body outside its substitutions)
    does not.  *looped* tells whether the sleep paces a ``for``/
    ``while``/``until``/``select`` loop.

    Args:
        command: The shell command text.
        limit: Seconds a sleep may take without counting.
    """
    for simple in simple_commands(command, None):
        if simple.program() != "sleep":
            continue
        operands = simple.argv()[1:]
        seconds = _sleep_seconds(operands)
        if seconds > limit:
            return seconds, [word.text for word in operands], _in_loop(simple)
    return None


_LOOPS = frozenset({"for", "while", "until", "select"})


def _in_loop(simple: SimpleCommand) -> bool:
    """Whether *simple* runs in the body of a loop (at any depth)."""
    group = simple.group
    while group is not None:
        if group.kind in _LOOPS:
            return True
        group = group.group
    return False


def sleep_guard(command: str) -> str | None:
    """Refuse a ``sleep`` longer than :data:`FOREGROUND_SLEEP_MAX_SECONDS` (:func:`long_sleep`).

    A long sleep is a polling turn: the model pays a full step to wait.
    The wait belongs in ``bash_job(action="wait", until=<regex>,
    timeout_seconds=N)``, which blocks server-side at no step cost; the
    refusal names that call and the sleep's own length.

    Args:
        command: The shell command text.
    """
    sleep = long_sleep(command, FOREGROUND_SLEEP_MAX_SECONDS)
    return None if sleep is None else sleep_refusal(sleep[0])


def sleep_refusal(seconds: float, job_id: str = "") -> str:
    """Return the refusal of a foreground sleep of *seconds*, naming the wait to make instead.

    Args:
        seconds: The sleep's length (``inf`` for ``sleep infinity``).
        job_id: The id of a running background job to wait on, or ``""``.
    """
    shown = "infinity" if math.isinf(seconds) else f"{seconds:.10g}s"
    wait = f"{min(seconds, 3600):.10g}"
    return (
        f"REFUSED: the Bash call did NOT run: a foreground sleep of {shown} "
        "is a polling turn. Start the thing you are waiting for with "
        "Bash(<its command>, background=True) and block on it with "
        f'bash_job({job_id or "job_id"!r}, action="wait", until="<regex in its log>", '
        f"timeout_seconds={wait}); or agent_job(job_id, \"wait\") for a sub-agent."
    )


# A ``sleep`` and its operands as written, to take out of a command: the
# program (by its path too), then operands with at most simple quotes.
_SLEEP_TEXT = re.compile(
    r"(?<![\w.-])(?:[\w./-]*/)?sleep((?:[ \t]+(?:'[^']*'|\"[^\"]*\"|[^\s;&|<>()'\"]+))+)"
)


def _sleeps(command: str) -> list[list[str]]:
    """Return the operands of every ``sleep`` the shell runs in *command*, in order."""
    return [
        [word.text for word in simple.argv()[1:]]
        for simple in simple_commands(command, None) if simple.program() == "sleep"
    ]


def _without_sleep(command: str, operands: list[str]) -> str | None:
    """Return *command* with its first ``sleep`` taking *operands* replaced by ``true``.

    ``None`` when the sleep is spelled in a way the text search does
    not find (then the command runs as written).  A match the shell
    does not run as that sleep (the words of a quoted argument, a
    here-document body) is passed over: ``true`` there would change
    that text and leave the sleep in, so the match taken is the one
    whose removal leaves the other sleeps as they are.
    """
    sleeps = _sleeps(command)
    sleeps.remove(operands)
    for match in _SLEEP_TEXT.finditer(command):
        words = [word.strip("'\"") for word in match.group(1).split()]
        if words != operands:
            continue
        rewritten = command[:match.start()] + "true" + command[match.end():]
        if _sleeps(rewritten) == sleeps:
            return rewritten
    return None


def wait_instead_of_sleep(
    command: str,
    timeout_seconds: float,
    running_job: Callable[[], str | None],
    wait: Callable[[str, float], str],
) -> tuple[str | None, str, str]:
    """Settle a long foreground ``sleep`` of *command* before it runs.

    The one rule of the host and the container ``Bash`` tools.  A
    ``sleep`` of at most :data:`FOREGROUND_SLEEP_MAX_SECONDS` runs as
    written.  A longer one past *timeout_seconds* is refused: the wait
    belongs in ``bash_job``.  While a background job runs, the sleep is
    spent waiting on the one started last and the command runs with
    that sleep taken out (``sleep 60; tail log`` waits, then runs
    ``tail log``): refusing it only made the model poll with shorter
    sleeps, each a turn.  With no running job, or when the sleep paces
    a loop (taking it out would spin the loop), the sleep runs as
    written; with no running job the note names what to do instead
    (:func:`sleep_alternative`): ``Bash(until=...)`` polls a condition
    inside one call, and a process a call leaves running is adopted as
    a job to ``bash_job(<id>, "wait")`` on.

    Args:
        command: The shell command text.
        timeout_seconds: The call's deadline.
        running_job: Returns the id of the running background job
            started last, or ``None``; called only for a long sleep.
        wait: ``wait(job_id, seconds)`` blocks until that job exits or
            the seconds pass and returns its status line.

    Returns:
        ``(refusal, note, command)``: the refusal (``None`` when the
        command may run), the line telling what was waited on, or what
        to wait with instead when the sleep ran with no job known
        (``""`` otherwise), and the command to run.
    """
    sleep = long_sleep(command, FOREGROUND_SLEEP_MAX_SECONDS)
    if sleep is None:
        return None, "", command
    seconds, operands, looped = sleep
    job_id = running_job()
    if seconds > timeout_seconds:
        return sleep_refusal(seconds, job_id or ""), "", command
    if job_id is None:
        return None, sleep_alternative(seconds), command
    rest = None if looped else _without_sleep(command, operands)
    if rest is None:
        return None, "", command
    status = wait(job_id, seconds)
    note = f"Waited on job {job_id} for up to {seconds:.10g} s instead of sleeping ({status}).\n"
    return None, note, rest


def sleep_alternative(seconds: float) -> str:
    """Return the note after a long foreground sleep that ran as written, naming the alternatives.

    No background job of the task was running to spend the sleep on,
    so the sleep was a polling turn; the next wait fits in one call.

    Args:
        seconds: The sleep's length.
    """
    return (
        f"\n[note: the sleep of {seconds:.10g} s ran as written: no background job of this "
        "task is running. Wait in one call instead: Bash(command, until=\"<regex>\", "
        "poll_seconds=N, timeout_seconds=M) re-runs the command until the regex matches "
        "its output; a process a Bash call leaves running is adopted as a job (the result "
        "names its id): bash_job(<id>, \"wait\") returns when it exits.]\n"
    )


BATCHING_HINT = (
    "\n[hint: this turn made one short read-only call; independent greps, reads and git "
    "queries fit in one Bash call or one run_commands_parallel call]"
)
"""Appended to a short read-only Bash result, at most once per :data:`BATCHING_HINT_EVERY` calls."""
BATCHING_HINT_EVERY = 20
# A call is short when it took less than these seconds and returned fewer characters.
_SHORT_CALL_SECONDS = 2.0
_SHORT_CALL_CHARS = 2000
_READ_ONLY_PROGRAMS = frozenset({"grep", "cat", "ls", "sed", "head", "tail", "git", "find", "wc"})
_READ_ONLY_GIT = frozenset({"log", "status", "diff", "show"})
_FIND_ACTIONS = frozenset({"-delete", "-exec", "-execdir", "-ok", "-okdir"})
_SED_IN_PLACE = re.compile(r"-[a-zA-Z]*i|--in-place")


def _is_read_only(command: str) -> bool:
    """Whether every program of *command* only reads, with no output redirection.

    Those are ``grep``, ``cat``, ``ls``, ``sed -n``, ``head``, ``tail``,
    ``git log``/``status``/``diff``/``show``, ``find`` without an action
    and ``wc``.
    """
    commands = simple_commands(command, None)
    if not commands:
        return False
    for simple in commands:
        program = simple.program()
        args = [word.text for word in simple.argv()[1:]]
        if program not in _READ_ONLY_PROGRAMS or simple.outputs:
            return False
        if program == "sed" and ("-n" not in args or any(_SED_IN_PLACE.match(a) for a in args)):
            return False
        if program == "git" and (not args or args[0] not in _READ_ONLY_GIT):
            return False
        if program == "find" and any(a in _FIND_ACTIONS or a.startswith("-fprint") for a in args):
            return False
    return True


def edit_args(path: str, edits: list[tuple[str, str, bool]]) -> dict[str, Any]:
    """Return the ``Edit`` arguments making *edits* of *path*: a pair inline, several as ``edits``.

    The call stands for a shell rewrite (a ``sed -i``, a python
    ``.replace``), which matches verbatim: it is ``exact``, so that the
    ``Edit``, made once the agent gets to it, retries nothing the
    rewrite would not have matched in the text the file holds by then.

    Args:
        path: The file to edit.
        edits: ``(old, new, replace_all)`` per replacement, in order.
    """
    if len(edits) == 1:
        old, new, every = edits[0]
        return {
            "file_path": path, "old_string": old, "new_string": new, "replace_all": every,
            "exact": True,
        }
    pairs = [
        {"old_string": old, "new_string": new, "replace_all": every} for old, new, every in edits
    ]
    return {"file_path": path, "edits": json.dumps(pairs), "exact": True}


def _applied(call: str, result: str, appended: bool = False) -> str:
    """Report *result* of the Write/Edit *call* a shell rewrite was applied as.

    A *result* starting with ``Error`` says the call failed instead of
    claiming the change was applied.
    """
    if result.startswith("Error"):
        return f"Not applied: the command stands for {call}, which failed: {result}"
    return f"{'Appended through' if appended else 'Applied as'} {call}: {result}"


def rewrite_redirect(
    name: str,
    args: dict[str, Any],
    call: str,
    appended: bool = False,
    followup: tuple[str, dict[str, Any]] | None = None,
) -> ToolRedirect:
    """Return the redirect making the *name* call (*args*) a shell rewrite stands for.

    The agent makes the call through its own tool path and the result
    of the rewrite opens with what it was applied as (*call*, the call
    as :func:`edit_call` spells it).

    Args:
        name: ``"Write"`` or ``"Edit"``.
        args: The call's arguments.
        call: The call as the result names it.
        appended: Whether the rewrite appended (a ``>>`` copy).
        followup: The ``Bash`` call running the rest of the command
            (:func:`followup_call`), made once the call went through.
    """
    made = "Appended through" if appended else "Applied as"
    return ToolRedirect(name, args, f"{made} {call}, whose result follows:", followup)


CD_VERDICT = (
    "REFUSED: the command did not run: its `cd {cwd}` fails, there is no such directory, "
    "so bash would run nothing after it."
)
"""The refusal of a compound rewrite whose leading ``cd`` names a missing directory."""
CD_LINK_VERDICT = (
    "REFUSED: the command did not run: its `cd {cwd}` reaches a directory only through a "
    "symbolic link, so only the shell knows which one. Name that directory by its own path."
)
"""The refusal of a compound rewrite whose leading ``cd`` bash enters by its links alone."""


def followup_call(
    cwd: str,
    remainder: str,
    description: str,
    timeout_seconds: float,
    max_output_chars: int,
    **options: Any,
) -> tuple[str, dict[str, Any]] | None:
    """Return the ``Bash`` call running *remainder*, the rest of a compound rewrite command.

    The rest runs where the command's leading ``cd`` moved to (``cd
    <directory> && <remainder>``), with the original call's
    *description*, *timeout_seconds*, *max_output_chars* and *options*;
    the same guards apply to it as to any ``Bash`` call.

    Args:
        cwd: The absolute POSIX path of the leading ``cd`` as written,
            which bash walks (:func:`cd_refusal`) to the normalised
            directory; ``""`` without one.
        remainder: The rest of the command after its rewrite; ``""`` without one.
        description: The original call's description.
        timeout_seconds: The original call's deadline.
        max_output_chars: The original call's output budget.
        **options: The original call's other arguments, which say how
            it runs (the host ``Bash``'s ``background`` and
            ``on_timeout``): the rest runs as the model asked.

    Returns:
        ``("Bash", arguments)``, or ``None`` when there is no rest to run.
    """
    if not remainder:
        return None
    if cwd:
        remainder = f"cd {shlex.quote(posixpath.normpath(cwd))} && {remainder}"
    return "Bash", {
        "command": remainder,
        "description": description,
        "timeout_seconds": timeout_seconds,
        "max_output_chars": max_output_chars,
        **options,
    }


def _lines_within_budget(lines: list[str], budget: int) -> list[str]:
    """Return the longest prefix of *lines* that fits *budget* characters.

    At least one line is kept; a first line longer than the budget (a
    one-line JSON or log file) is clipped to it with a marker, so no
    file can put more than *budget* characters into the context.
    """
    kept: list[str] = []
    size = 0
    for line in lines:
        if kept and size + len(line) > budget:
            break
        if not kept and len(line) > budget:
            note = f"\n[line clipped to {budget} chars; grep or cut for the rest]\n"
            return [line[:budget] + note]
        kept.append(line)
        size += len(line)
    return kept


SPILL_DIR = "tmp/tool-out"
"""Where a truncated tool output is written in full, under the work dir.

POSIX form on every OS: the path is shown to the model for the Bash tool,
which is bash on Windows too, where a backslash would be eaten.
"""
JOB_LOG_DIR = "tmp/bash_jobs"
"""Where a background job's log is written, under the directory it runs in."""
SUBAGENT_DIR_ENV = "KISS_SUBAGENT_DIR"
"""The variable naming a sub-agent's private folder in the environment of its commands.

The folder (:data:`~kiss.agents.sorcar.fanout_guard.SUBAGENT_DIR_PREFIX`)
holds the sub-agent's job logs and spilled output (:func:`_tools_dir`)
and is the temp directory of every command its tools start
(:func:`_clean_env`).  What still carries the variable when the
sub-agent ends is killed (:func:`~kiss.core.processes.kill_env_marked`).
"""


def _tools_dir(private_dir: str | None, cwd: str, relative: str) -> tuple[str, str, str]:
    """Where the tools keep the files of their own in *relative* (job logs, spilled output).

    Under *cwd*, the directory the commands run in; a sub-agent's
    *private_dir* holds the directory by its last name instead, so that
    nothing of it is left in the work tree.

    Returns:
        ``(root, path under root, the directory as shown to the model)``:
        the last is relative to *cwd*, or absolute in a private folder.
    """
    if not private_dir:
        return cwd, relative, relative
    name = Path(relative).name
    return private_dir, name, Path(private_dir, name).as_posix()


def _output_budget(max_output_chars: int) -> int:
    """Return the characters a result may keep: the model's request, capped at the ceiling."""
    return max(1, min(int(max_output_chars), DEFAULT_CONFIG.tool_output_max_chars))


def _clean_env(work_dir: str | None = None, private_dir: str | None = None) -> dict[str, str]:
    """Return a fresh copy of ``os.environ`` without ``VIRTUAL_ENV``.

    When the agent process runs inside a virtual-env (e.g. the
    installed daemon's own ``.venv``), the ``VIRTUAL_ENV`` variable leaks into
    child processes and causes ``uv run`` to emit a spurious warning about
    a mismatched environment. Stripping it lets ``uv`` (and other tools)
    discover the correct project ``.venv`` on their own.

    When ``work_dir`` is provided, ``KISS_WORKDIR`` is overridden so that
    child processes (e.g. project scripts that compute "project root" via
    that env var) target the agent's working directory — critical when
    the agent is running inside a git worktree, where the inherited
    ``KISS_WORKDIR`` would otherwise still point at the original repo
    checkout and writes would leak out of the worktree.

    Args:
        work_dir: Agent working directory to expose to child processes
            via ``KISS_WORKDIR``.  ``None`` or ``""`` (the unresolved
            default on a freshly constructed agent) leaves the inherited
            value untouched.
        private_dir: A sub-agent's private folder: the temp directory of
            the child processes, which :data:`SUBAGENT_DIR_ENV` marks
            as the sub-agent's.
    """
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    if work_dir:
        env["KISS_WORKDIR"] = work_dir
    if private_dir:
        env.update(dict.fromkeys(("TMPDIR", "TMP", "TEMP", SUBAGENT_DIR_ENV), private_dir))
    return env


def _format_bash_result(
    returncode: int, output: str, max_output_chars: int, spilled_to: str = "", dropped: int = 0,
) -> str:
    """Report a finished command: its output, after its exit code when that is not 0.

    Only the output is cut to *max_output_chars*: cut with it, a small
    budget dropped the status and a failed command read as one that
    passed.  *spilled_to* and *dropped* are as for :func:`_truncate_output`.
    """
    output = _truncate_output(output, max_output_chars, spilled_to, dropped)
    if returncode == 0:
        return output
    return f"Error (exit code {returncode}):" + (f"\n{output}" if output else "")


def _format_bash_timeout(
    output: str, timeout_seconds: float, max_output_chars: int, spilled_to: str = "",
    dropped: int = 0,
) -> str:
    """Report a killed-at-the-deadline command, keeping what it printed before the kill.

    The partial output (how far a test run or build got) is what lets the
    model act on the next step instead of re-running the command blind.
    Only it is cut to *max_output_chars*, as in :func:`_format_bash_result`;
    *spilled_to* and *dropped* are as for :func:`_truncate_output`.
    """
    output = _truncate_output(output, max_output_chars, spilled_to, dropped)
    msg = "Error: Command execution timeout"
    if output:
        msg += f" after {timeout_seconds:g}s. Output before the timeout:\n{output}"
    return msg


def _common_guards(
    command: str, work_dir: str | None, timeout_seconds: float, on_host: bool,
) -> tuple[str | None, str, str, float]:
    """Apply the guards every launch path runs first; see :func:`_refuse_or_lift` for the shape.

    The refusals and the timeout lift of :func:`refuse_or_lift`, the
    parent-repository guard, and, when the command runs *on_host*
    (not on another machine), the ``uv`` rewrite of a worktree command
    (:func:`uv_in_worktree`).
    """
    guard, timeout_seconds = refuse_or_lift(command, work_dir, timeout_seconds)
    refusal, note = _bash_parent_repo_guard(command, work_dir), ""
    if refusal is None and on_host:
        refusal, note, command = uv_in_worktree(command, work_dir)
    return refusal or guard, note, command, timeout_seconds


def _refuse_or_lift(
    command: str,
    work_dir: str | None,
    timeout_seconds: float,
    cwd: str,
) -> tuple[str | None, str, str, float]:
    """Apply the shell guards to *command* and lift its timeout for installs.

    The guard chain of :func:`_run_one_command`; :meth:`UsefulTools.Bash`
    runs the same first guards (:func:`_common_guards`) but applies a
    mechanical rewrite through ``Write``/``Edit`` instead of refusing
    it (:meth:`UsefulTools._rewrite_instead`).

    Args:
        command: The shell command the model wants to run.
        work_dir: The agent's working directory (worktree + destructive guards).
        timeout_seconds: The caller's deadline.
        cwd: The directory the command runs in on this host
            (:meth:`UsefulTools._resolve_cwd`), where the here-document
            guard looks its targets up.

    Returns:
        ``(refusal, note, command, timeout_seconds)``: the guard's error
        message (``None`` when the command may run), the note to prefix
        to the result and the command to run (a ``uv run`` in a worktree
        is rewritten, :func:`uv_in_worktree`), and the possibly lifted
        deadline.
    """
    refusal, note, command, timeout_seconds = _common_guards(
        command, work_dir, timeout_seconds, True)
    if refusal is None:
        guarded = functools.partial(_host_guarded, work_dir)
        bash_cwd = bash_path(os.path.abspath(cwd))
        refusal = rewrite_guard(command, bash_cwd, guarded, _host_isdir) or heredoc_guard(
            command, bash_cwd, guarded, _host_isdir)
    return refusal, note, command, timeout_seconds


def _host_lexists(path: str) -> bool:
    """Return whether the absolute bash *path* (see :func:`bash_path`) exists on this host."""
    return os.path.lexists(host_path(path))


def _host_guarded(work_dir: str | None, path: str) -> bool:
    """Return whether a shell rewrite of the absolute bash *path* is refused on this host.

    It is when the file exists and is not a scratch file of *work_dir*
    (:func:`_is_scratch_path`): a scratch file is rewritten from scratch
    by every session and never reviewed, so a here-document, a
    ``sed -i`` or a python rewrite of one may run as written, as
    ``Write`` overwrites one unread.  Bound to *work_dir* with
    :func:`functools.partial` for the guards (:func:`rewrite_guard`,
    :func:`heredoc_guard`).
    """
    return _host_lexists(path) and not _is_scratch_path(Path(host_path(path)).resolve(), work_dir)


def _host_isdir(path: str) -> bool:
    """Return whether the absolute bash *path* (see :func:`bash_path`) is a directory here."""
    return os.path.isdir(host_path(path))


#: Seconds a python gets to say where it loads a rewrite's modules from.
_SHADOW_PROBE_SECONDS = 10


def _host_shadows(program: str, path: str, modules: set[str]) -> bool:
    """Return whether the python *program*, run in the bash directory *path*, shadows *modules*.

    The interpreter is asked (:func:`shadow_command`), in the
    environment a command gets: before its own library it searches the
    directory a program read from stdin runs in, the ``PYTHONPATH``
    entries as it reads them and what its own startup adds, and
    ``json.py`` there is what ``import json`` runs.  An answer nobody
    got (no shell, a python that hangs) stands for a shadow.
    """
    command = shadow_command(program, path, modules)
    try:
        asked = run_process_group(
            [_BASH or "sh", "-c", command], _SHADOW_PROBE_SECONDS, env=_clean_env())
    except (OSError, subprocess.TimeoutExpired):
        return True
    return asked.returncode != 1


#: Exit 0 when Git Bash finds ``$1`` on a ``PATH`` whose every entry is absolute.
_GIT_BASH_FINDS = (
    'case ":$PATH:" in *::* | *:[!/]*) exit 1 ;; esac; command -v -- "$1" >/dev/null 2>&1'
)


def _host_runs(program: str) -> bool:
    """Return whether bash finds *program* here: a name on ``PATH`` or an absolute bash path.

    A bare name counts as not found on a ``PATH`` holding a relative
    entry (an empty one is the current directory): bash takes that from
    the directory the command's ``cd`` entered, this process from its own.
    """
    if "/" in program:
        return shutil.which(host_path(program)) is not None
    if sys.platform == "win32" and _BASH:
        try:
            # Git Bash's own PATH, with the same relative-entry refusal as below.
            found = run_process_group(
                [_BASH, "-c", _GIT_BASH_FINDS, "bash", program],
                5,
                env=_clean_env(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return found.returncode == 0
    entries = os.environ.get("PATH", os.defpath).split(os.pathsep)
    return all(map(os.path.isabs, entries)) and shutil.which(program) is not None


def _sed_replaces_link(path: str) -> bool:
    """Return whether a ``sed -i`` on the host *path* changes other files than an ``Edit`` of it.

    GNU sed writes a new file and renames it over the operand: a
    symlink is replaced by a regular file and its target left as it
    was; a hard link is cut from its other names, which keep the old
    text.  ``Edit`` writes the file in place, through the link.
    """
    try:
        return os.path.islink(path) or os.lstat(path).st_nlink > 1
    except OSError:
        return False


def _fails_as_written(path: str) -> bool:
    """Return whether a command naming the host *path* fails where an ``Edit`` of it may not.

    The system refuses the name (too long, not encodable, through a
    file or a directory it may not search), on which the tools'
    resolution raises; or it finds nothing by a name that holds a
    ``..``, which :meth:`~pathlib.Path.resolve` drops along with a name
    that is no directory (``nosub/../f`` is no file, ``f`` may be one).
    A name that is just missing is left to the ``Edit``, which reports
    it, or finds the file in the parent of a vanished worktree
    (:func:`resolve_tool_path`).
    """
    try:
        os.stat(path)
    except FileNotFoundError:
        return ".." in Path(path).parts
    except (OSError, ValueError):
        return True
    return False


def cd_refusal(cwd: str, isdir: Callable[[str], bool]) -> str | None:
    """Return the refusal of a compound rewrite whose leading ``cd`` bash does not walk.

    Bash drops a ``..`` of the path only after a directory
    (:func:`~kiss.agents.sorcar.shell_scan.walks`) and enters the
    normalised path, where the rewrite's file was resolved.  Where it
    cannot, it takes the path as written: no directory, unless a
    symbolic link leads to one, which is then not the rewrite's.

    Args:
        cwd: The path of the ``cd`` before the rewrite, as written; ``""`` without one.
        isdir: Whether a path is a directory where the command would run.

    Returns:
        ``None`` without a ``cd`` and when bash walks *cwd*;
        :data:`CD_LINK_VERDICT` when *cwd* is a directory as written
        only; else :data:`CD_VERDICT`: the ``cd`` fails.
    """
    if not cwd or walks(cwd, isdir):
        return None
    through_link = posixpath.normpath(cwd) != cwd and isdir(cwd)
    return (CD_LINK_VERDICT if through_link else CD_VERDICT).format(cwd=cwd)


def _run_one_command(
    command: str,
    cancel: threading.Event,
    work_dir: str | None,
    timeout_seconds: float,
    max_output_chars: int,
    private_dir: str | None = None,
) -> tuple[int | None, str, float]:
    """Run one host ``run_commands_parallel`` command on a worker thread.

    Args:
        command: The shell command.
        cancel: The fan-out's stop event; setting it kills the shell.
        work_dir: The agent's working directory (worktree guard + cwd).
        timeout_seconds: Deadline for the shell.
        max_output_chars: The report's per-command budget; an output too
            big to be kept whole (see :class:`_Capture`) is cut to it
            here, where the count of what was left out is known.
        private_dir: The private folder of the sub-agent running the
            command, if one is (:func:`_clean_env`).

    Returns:
        ``(returncode, output, seconds)``: the exit code (``None`` on
        timeout, ``-1`` when the command was refused, not started because
        *cancel* was set, or failed to launch), the combined output, and
        the wall time spent.
    """
    # Stream-less: printers attribute output by thread-local task id, so
    # a callback from this worker thread would detach the output from
    # the task.
    runner = UsefulTools(stop_event=cancel, work_dir=work_dir, private_dir=private_dir)
    cwd = runner._resolve_cwd()
    try:
        guard, note, command, timeout_seconds = _refuse_or_lift(
            command, work_dir, timeout_seconds, cwd or os.getcwd())
    except TooDeepError as e:
        return -1, str(e), 0.0  # this command's verdict, not the whole fan-out's failure
    # Every command of the fan-out is a foreground wait for the caller,
    # so a long sleep is refused as in a foreground ``Bash`` call.
    guard = guard or sleep_guard(command)
    if guard is not None:
        return -1, guard, 0.0
    if cancel.is_set():
        # A stop before the launch: the shell is not started at all.
        return -1, NOT_STARTED, 0.0
    started = time.monotonic()
    capture = _Capture()
    try:
        returncode, output = runner._bash_streaming(command, timeout_seconds, cwd, capture)
    except Exception as e:
        logger.debug("Exception caught", exc_info=True)
        return -1, f"Error: {e}", time.monotonic() - started
    if capture.dropped:
        output = _truncate_output(output, max_output_chars, dropped=capture.dropped)
    return returncode, output + note, time.monotonic() - started


CommandRunner = Callable[[str, threading.Event], tuple[int | None, str, float]]

NOT_STARTED = "Not started: the task was stopped."


def run_commands_pool(
    commands: str,
    run_one: CommandRunner,
    max_workers: int,
    stop_event: threading.Event | None,
    max_output_chars: int,
    spill: Callable[[str], str | None] | None = None,
) -> str:
    """Validate the ``run_commands_parallel`` arguments, run them via *run_one*, render the report.

    The one engine behind the ``run_commands_parallel`` tools (host
    shell, Docker container, exec bridge).  The workers watch ONE cancel event,
    set here when the task's *stop_event* or the tool panel's Stop
    (:func:`kiss.core.tool_interrupt.current_tool_interrupt_event`,
    thread-local and therefore invisible to workers) fires; queued
    commands are cancelled at the same time so a stop never starts
    work.  An exception injected into the waiting thread (a task Stop's
    ``KeyboardInterrupt``, a forced ``ToolCallInterrupted``) also
    cancels everything before propagating.

    Args:
        commands: The tool's ``commands`` argument: a JSON-encoded list of
            shell command strings.
        run_one: ``run_one(command, cancel) -> (exit_code, output,
            seconds)``; must not raise.
        max_workers: Concurrency bound; ``0`` runs all commands at once.
        stop_event: The task's stop event, or ``None``.
        max_output_chars: Per-command output cap in the report.
        spill: Writes a report longer than the conversation's ceiling
            (``DEFAULT_CONFIG.tool_output_max_chars``) to a file and
            returns the path to show, or ``None`` when it cannot; the
            result is then the digest (:func:`digest_parallel_report`).
            ``None`` returns the whole report, however long.

    Returns:
        A report starting with a tally line
        (``N commands: K succeeded, F failed, T timed out``) followed by
        one section per command, in input order, giving its exit code,
        wall time and (truncated) output (see
        :func:`_format_parallel_report`), or its digest when it is over
        the ceiling and *spill* kept the whole of it.  A string starting
        with ``Error:`` when the arguments were invalid.  How each
        command ended is handed over as data
        (:func:`~kiss.agents.sorcar.shell_exit.hand_over`): a command's
        output may hold a section of the report.

    Raises:
        ToolCallInterrupted: When the tool panel's Stop was pressed.
    """
    try:
        command_list = parse_tasks_json(commands, name="commands")
        if max_workers < 0:
            raise ValueError(f"max_workers must be 0 or a positive integer, got {max_workers}.")
    except ValueError as e:
        hand_over(commands)  # no command ran
        return f"Error: {e.args[0]}"
    cancel = threading.Event()
    pool = ThreadPoolExecutor(max_workers=max_workers or len(command_list))
    try:
        if _stop_requested(stop_event):
            # Stopped before the pool: the workers see the cancel at once
            # (``_run_one_command`` starts no shell) instead of after the
            # first slice of the wait below, when they would already run.
            cancel.set()
        futures = [pool.submit(run_one, command, cancel) for command in command_list]
        while not all(f.done() for f in futures):
            if _stop_requested(stop_event):
                cancel.set()
                for f in futures:
                    f.cancel()
            # Short slices keep this loop in Python, where an injected
            # KeyboardInterrupt / ToolCallInterrupted lands.
            wait(futures, timeout=0.2)
    except BaseException:
        cancel.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    tool_interrupt.raise_if_interrupted()
    results = [(-1, NOT_STARTED, 0.0) if f.cancelled() else f.result() for f in futures]
    hand_over(commands, *(_command_exit(code) for code, _, _ in results))
    report = _format_parallel_report(command_list, results, max_output_chars)
    if spill is not None and len(report) > DEFAULT_CONFIG.tool_output_max_chars:
        shown = spill(report)
        if shown is not None:
            return digest_parallel_report(command_list, results, shown)
    return report


#: Characters of a failed command's output a digest keeps (its tail: the
#: test runner's summary, the compiler's last errors).
DIGEST_TAIL_CHARS = 1200


def _command_exit(code: int | None) -> str:
    """One command's outcome in the gate ledger's words (:mod:`~kiss.agents.sorcar.shell_exit`)."""
    return f"killed by signal {-code}" if code is not None and code < -1 else exit_word(code)


def _command_status(code: int | None, seconds: float) -> str:
    """One command's outcome for the report: its exit code or timeout and wall time."""
    if code is None:
        return f"TIMED OUT after {seconds:.1f}s"
    if code < -1:
        return f"killed by signal {-code} in {seconds:.1f}s"
    return f"exit {code} in {seconds:.1f}s"


def _tally(results: list[tuple[int | None, str, float]]) -> str:
    """Return the first sentence of a ``run_commands_parallel`` report: how its commands ended."""
    codes = [code for code, _, _ in results]
    succeeded = sum(1 for code in codes if code == 0)
    timed_out = sum(1 for code in codes if code is None)
    failed = len(codes) - succeeded - timed_out
    return f"{len(codes)} commands: {succeeded} succeeded, {failed} failed, {timed_out} timed out."


def digest_parallel_report(
    commands: list[str], results: list[tuple[int | None, str, float]], shown: str,
) -> str:
    """Render the short form of a ``run_commands_parallel`` report whose whole text is in a file.

    The tally, one line per command with its status, the tail of each
    command that did not succeed (:data:`DIGEST_TAIL_CHARS`), and the
    file to ``grep`` for the rest: a 60-split test run then costs the
    conversation a page, not the whole of every split's output (the
    2026-11-01 audit saw 22 such runs fill 365 k tokens).

    Args:
        commands: The commands, in input order.
        results: Their ``(exit code, output, seconds)``.
        shown: The path of the file holding the whole report, as the
            model should name it.
    """
    lines = [
        f"{_tally(results)} The whole report ({len(commands)} sections) is in {shown}; "
        "this is its digest."
    ]
    pairs = list(zip(commands, results, strict=True))
    for index, (command, (code, _, seconds)) in enumerate(pairs, 1):
        one_line = " ".join(command.split())
        lines.append(f"[{index}/{len(commands)}] {_command_status(code, seconds)}: "
                     f"{one_line[:160]}{'...' if len(one_line) > 160 else ''}")
    for index, (command, (code, output, seconds)) in enumerate(pairs, 1):
        if code == 0:
            continue
        tail = output.rstrip("\n")[-DIGEST_TAIL_CHARS:] or "(no output)"
        lines.append(f"\n### [{index}/{len(commands)}] {_command_status(code, seconds)}, "
                     f"last {min(len(tail), DIGEST_TAIL_CHARS)} chars (section {index} of {shown} "
                     f"has all of it)\n{tail}")
    return "\n".join(lines)


def spill_report(
    root: str | None, kind: str, text: str, private_dir: str | None = None,
) -> str | None:
    """Write *text* under ``tmp/tool-out/`` of *root* and return the path to show, or ``None``.

    The file is ``tmp/tool-out/<tag>-<kind>.txt``, *tag* six random hex
    digits, the shown path relative to *root* (where the model's shell
    commands run); ``None`` when *root* is unknown or the directory
    cannot be made or written.

    Args:
        root: The directory the tool's commands run in.
        kind: What the file holds, for its name (``parallel``, ``agents``).
        text: The whole report.
        private_dir: The private folder of the sub-agent the report is
            for, which then holds the file, shown by its absolute path
            (:func:`_tools_dir`).
    """
    if root is None:
        return None
    root, relative, shown = _tools_dir(private_dir, root, SPILL_DIR)
    name = f"{uuid.uuid4().hex[:6]}-{kind}.txt"
    try:
        (_make_dirs_under(root, relative) / name).write_text(text, encoding="utf-8")
    except OSError:
        return None
    return f"{shown}/{name}"


def _format_parallel_report(
    commands: list[str],
    results: list[tuple[int | None, str, float]],
    max_output_chars: int,
) -> str:
    """Render the ``run_commands_parallel`` report: a tally, then one section per command."""
    lines = [_tally(results)]
    for index, (command, (code, output, seconds)) in enumerate(
        zip(commands, results, strict=True), start=1
    ):
        status = _command_status(code, seconds)
        lines.append(f"\n### [{index}/{len(commands)}] {status}\n$ {command}")
        lines.append(_truncate_output(output.rstrip("\n"), max_output_chars) or "(no output)")
    return "\n".join(lines)


def _kill_process_group(process: subprocess.Popen) -> None:
    """Kill the shell *process* and all its children, unless the shell has exited.

    For the thread that owns the shell (:func:`_spawn_owned`), the only
    one to signal or reap it: the pid, and the group named after it, is
    then the shell's own whenever this looks at it, where a reaped
    shell's may be a stranger's by the time of the signal
    (:func:`kill_process_group` documents the hazard).  An exited
    shell's group is left alone: its exit already broke the pipes this
    module reads, and what it detached is meant to run on.

    Args:
        process: The shell started by :func:`popen_process_group`.
    """
    if not exited_unreaped(process):
        kill_process_tree(process)


@dataclass
class BackgroundJob:
    """A detached command started by ``Bash(background=True)``.

    Attributes:
        job_id: Short id the model passes to ``bash_job``.
        command: The shell command.
        log_path: File receiving the job's combined stdout and stderr.
        process: The job's shell.  Its watcher (:func:`_watch_job`) alone
            signals and reaps it; :meth:`returncode` is its liveness.
        started: ``time.monotonic()`` at launch.
        ended: The shell's exit status, set by the watcher's thread after
            its last signal to the group and before it reaps the shell.
        kill: Set to have the watcher kill the job's process group.
        released: Set once the job proved a foreground result
            (:meth:`UsefulTools._run_then_detach`): the watcher then
            leaves alone what the shell left running in its group.
    """

    job_id: str
    command: str
    log_path: Path
    process: subprocess.Popen
    started: float
    ended: Future[int | None]
    kill: threading.Event
    released: threading.Event

    def returncode(self) -> int | None:
        """Return the shell's exit status without reaping it, ``None`` while it runs."""
        status = exit_status_unreaped(self.process)
        if status is None and self.ended.done():
            return self.ended.result()  # read while the watcher was reaping the shell
        return status

    def status(self) -> str:
        """Return ``running (pid N, Ts)`` or ``exited with code N``."""
        returncode = self.returncode()
        if returncode is None:
            elapsed = time.monotonic() - self.started
            return f"running (pid {self.process.pid}, {elapsed:.0f}s)"
        if returncode < 0:
            return f"killed by signal {-returncode}"
        return f"exited with code {returncode}"


def _tail_lines(path: Path, lines: int, max_chars: int) -> str:
    """Return the last *lines* lines of *path* that fit *max_chars* characters, whole lines only.

    A build or training log can be gigabytes long, so only its end is
    read (:func:`_read_end`): four bytes per character of the budget,
    the most UTF-8 needs, so a line that fits the budget is never lost
    to its byte count.  A last line longer than the budget is left out;
    the end of a log that is one line without any line break (a
    minified blob, a progress bar still being rewritten) is cut to the
    budget instead, so the log is not reported as empty.
    """
    if lines <= 0 or max_chars <= 0:
        return ""
    try:
        text = _read_end(path, 4 * max_chars)
    except OSError as e:
        return f"(log unreadable: {e})"
    pieces = text.splitlines(keepends=True)[-lines:]
    if pieces == [text] and not text.endswith(("\n", "\r")):
        return text[-max_chars:]
    kept: list[str] = []
    size = 0
    for line in reversed(pieces):
        size += len(line)
        if size > max_chars:
            break
        kept.append(line)
    return "".join(reversed(kept))


def _format_job_report(job: BackgroundJob, tail_lines: int, max_output_chars: int) -> str:
    """Return the ``bash_job`` result: status, command, log path and log tail.

    The status line is whole under any budget (the coding gate reads
    the job's exit from it); what follows is cut to the rest of
    *max_output_chars*.
    """
    status = f"Job {job.job_id}: {job.status()}\n"
    budget = max(max_output_chars - len(status), 0)
    report = (
        f"Command: {job.command}\n"
        f"Log: {job.log_path}\n"
        f"--- last {tail_lines} log lines ---\n"
    )
    tail = _tail_lines(job.log_path, tail_lines, budget - len(report))
    return status + _truncate_output(report + tail, budget)


def _job_group_alive(process: subprocess.Popen) -> bool:
    """Return whether the exited job shell *process* left members in its process group.

    The orphans of an exited shell (``sh -c "server &"``) stay in the
    group named after its pid.  The shell, unreaped, is a member too,
    so ``killpg(pid, 0)`` cannot tell an empty group: every other
    process is asked for its group instead, listed by ``/proc`` (no
    ``ps`` to depend on in a slim container) or, where there is none,
    by ``ps`` (:func:`_other_group_members`).  Windows has no process
    groups to probe: there the watch ends with the shell.
    """
    if sys.platform == "win32":  # pragma: no cover — Windows-only branch
        return False
    try:
        names = os.listdir("/proc")
    except OSError:  # pragma: no cover — no /proc: macOS, BSD
        return _other_group_members(process.pid)
    for name in names:
        if name.isdigit() and int(name) != process.pid:
            with suppress(OSError):  # the process ended since the listing
                if os.getpgid(int(name)) == process.pid:
                    return True
    return False


def _watch_job(
    stop_event: threading.Event | None,
    process: subprocess.Popen,
    settled: threading.Event,
    registered: threading.Event,
    interrupt: threading.Event | None,
    kill: threading.Event,
    released: threading.Event,
) -> None:
    """Own the background job *process*: kill its group on a Stop or on *kill*.

    Runs on the daemon thread that spawned the job (:func:`_spawn_owned`),
    the only one to signal the job's group; the shell is reaped once
    this returns (:func:`_own_child`), never earlier, so each signal
    names the group of an unreaped shell and not a stranger's.
    Until *settled* is set (the launch left its cleanup region) the
    shell is killed on *stop_event* or *interrupt*; then it is killed unless
    *registered* is set (the launch bound and registered the job),
    so a Stop or a failure that unwound the launch between the fork
    and the registration leaves no shell running with no watcher and
    no job id anyone could kill it by.  The kill mirrors what a task
    Stop does to a foreground command; *kill* is how
    ``bash_job(action="kill")`` asks for it.  The watch goes on after
    the shell exited while its group still has members
    (:func:`_job_group_alive`), so both reach a ``server &`` the shell
    left behind.  The members are looked for one poll after the exit
    was seen, then after pauses doubling up to a minute: a process
    table read every poll would cost more than the job.  *released*
    (set once the shell has exited) ends the watch at its next poll
    and is looked at before the Stop: what a foreground result left
    running is not this watcher's to kill.

    *interrupt* is the launching tool call's interrupt event: the tool
    panel's Stop can unwind the launch right before it sets *settled*,
    the job registered under an id no result names.  A job whose launch
    settled no longer watches it: it outlives its tool call.
    """
    while not settled.wait(timeout=0.2):
        if any(event is not None and event.is_set() for event in (stop_event, interrupt)):
            kill_process_tree(process)
            return
    if not registered.is_set():
        kill_process_tree(process)
        return
    pause, scan_at = 0.2, 0.0
    while True:
        exited = exited_unreaped(process)
        asked = kill.wait(timeout=0.2)
        if released.is_set():
            return
        if asked or (stop_event is not None and stop_event.is_set()):
            kill_process_tree(process)
            return
        if exited and time.monotonic() >= scan_at:
            if not _job_group_alive(process):
                return
            pause = min(2 * pause, 60)
            scan_at = time.monotonic() + pause


def _stop_monitor(
    stop_events: list[threading.Event],
    process: subprocess.Popen,
    done: threading.Event,
) -> None:
    """Own the foreground shell *process*: kill its group once a stop fires or *done* is set.

    Runs on the daemon thread that spawned the shell
    (:func:`_spawn_owned`).  The events are the task's stop event (the
    agent was stopped) and the running tool call's interrupt event (the
    user pressed the tool panel's Stop button); *done* is set by the
    wait once it is over, or to have a shell it found running at its
    deadline killed.  The kill after *done* is a no-op for a shell
    that has exited (:func:`_kill_process_group`, so the descendants a
    ``nohup job &`` left behind are untouched) and ends a shell the
    wait left running: at its deadline, or unwound by a Stop or a
    failure, before it bound the handle included.
    """
    while not done.wait(timeout=0.2):
        if any(event.is_set() for event in stop_events):
            break
    _kill_process_group(process)


def _own_child(
    spawn: Callable[[], subprocess.Popen],
    own: Callable[[subprocess.Popen], None],
    handoff: Future[subprocess.Popen],
    ended: Future[int | None],
) -> None:
    """Body of the owner thread of :func:`_spawn_owned`: spawn, hand off, own, then reap.

    Nothing else signals or reaps the child.  Its exit status goes to
    *ended* after the last signal *own* sent and before the reap, so a
    signal always names the group of an unreaped child, and a status
    read by another thread (without reaping,
    :func:`exit_status_unreaped`, or from *ended*) is never lost to the
    reap.  A child *own* left running was killed: it is given 5 s to
    exit, and *ended* is ``None`` for one that outlives them.
    """
    try:
        process = spawn()
    except BaseException as e:  # a failed launch is the caller's to report
        handoff.set_exception(e)
        return
    handoff.set_result(process)
    try:
        own(process)
    finally:
        _exits_within(process, 5)
        ended.set_result(exit_status_unreaped(process))
        process.poll()


def _spawn_owned(
    spawn: Callable[[], subprocess.Popen],
    own: Callable[[subprocess.Popen], None],
) -> tuple[subprocess.Popen, Future[int | None]]:
    """Spawn a child on a daemon thread that then owns it; return it and its exit status to be.

    The calling task thread cannot own the child from its creation: the
    server's Stop (``KeyboardInterrupt``, or the tool panel's
    :class:`~kiss.core.tool_interrupt.ToolCallInterrupted`) is injected
    at any check point of its bytecode, inside ``Popen`` after the fork
    or right after *spawn* returned, before a handle a ``finally``
    could kill is bound (the ownership :func:`_run_log_matcher` gives
    the matcher child).  The owner thread receives no injection: it
    runs *spawn*, hands the child over, then runs *own* on it, which
    watches the caller's events and kills the child when the caller
    left without taking it over.  Call it inside the cleanup region
    whose ``finally`` signals *own*.

    Args:
        spawn: Starts the child; its exception is raised here.
        own: Runs on the owner thread for as long as the child needs
            watching (:func:`_stop_monitor`, :func:`_watch_job`).

    Returns:
        The started child, and the future of its exit status
        (:func:`_own_child`).  The child is the owner thread's alone to
        signal and to reap: the caller watches it without reaping
        (:func:`exited_unreaped`) and reads its status from the future.

    Raises:
        ToolCallInterrupted: When the running tool call was interrupted
            while waiting for the child; the owner then kills it.
    """
    handoff: Future[subprocess.Popen] = Future()
    ended: Future[int | None] = Future()
    threading.Thread(target=_own_child, args=(spawn, own, handoff, ended), daemon=True).start()
    return tool_interrupt.result_interruptible(handoff), ended


def _spawn_job(
    popen_kwargs: dict[str, Any], log_path: Path, cwd: str | None, private_dir: str | None = None,
) -> subprocess.Popen:
    """Start a background job's shell: stdout and stderr on *log_path*, stdin ``/dev/null``.

    A launch that fails leaves no log behind.  *private_dir* is the
    sub-agent's private folder, if the job is one's (:func:`_clean_env`).
    """
    try:
        with open(log_path, "wb") as log_file:
            return popen_process_group(
                **popen_kwargs,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=_clean_env(cwd, private_dir),
                cwd=cwd,
            )
    except BaseException:
        log_path.unlink(missing_ok=True)
        raise


# Held by ``Edit`` for its read-modify-write of a file and by ``Write`` for
# its overwrite or append: the tool instances of a task and its sub-agents
# (threads of one process editing the same tree) would otherwise undo each
# other's changes, both calls reporting success, when one's write landed
# between another's read and write of the same file.  Taken through
# :func:`locked_call` (an ``RLock`` for its owner check): a task thread
# parked in an untimed ``acquire`` while another task's write holds the
# lock reaches no check point, so its Stop would wait for that write.
_FILE_WRITE_LOCK = threading.RLock()


def _open_regular(resolved: Path, flags: int) -> int | None:
    """Open *resolved* with the ``os.open`` *flags*: its descriptor, ``None`` when not regular.

    The type check and the open are one step, on the descriptor: a
    path checked with ``is_file`` and then opened may have been swapped
    for a FIFO meanwhile, and ``open`` of a FIFO with no writer (no
    reader, for a write) blocks in C for good, where no Stop lands
    (under :data:`_FILE_WRITE_LOCK` it would hold every Write and Edit
    of the daemon).  The open is non-blocking where the flag exists, so
    a FIFO is seen by ``fstat`` instead of waited for (one opened for
    writing with no reader fails at once), and closed again, as is a
    descriptor ``fstat`` fails on.  The caller owns the descriptor
    returned and runs where no Stop can part the two (:func:`_unstopped`).

    Args:
        resolved: The path to open.
        flags: ``os.O_RDONLY``, or ``os.O_WRONLY | os.O_CREAT`` with
            ``os.O_TRUNC`` or ``os.O_APPEND``; a file created gets the
            mode ``open`` gives.

    Raises:
        OSError: When the file cannot be opened (``FileNotFoundError``
            when it does not exist and *flags* do not create it).
    """
    fd = os.open(resolved, flags | getattr(os, "O_NONBLOCK", 0), 0o666)
    regular = False
    try:
        regular = stat.S_ISREG(os.fstat(fd).st_mode)
        return fd if regular else None
    finally:
        if not regular:
            os.close(fd)


def _run_unstopped(work: Callable[[], Any], result: Future[Any]) -> None:
    """Body of the thread of :func:`_unstopped`: give *result* what *work* returns or raises.

    Nothing runs for a *result* its caller cancelled: a Stop took the
    caller away before this began.
    """
    if not result.set_running_or_notify_cancel():
        return
    try:
        result.set_result(work())
    except BaseException as e:  # the caller's to see, whatever it is
        result.set_exception(e)


def _unstopped[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """Decorator: run *fn* on a thread of its own, where no Stop is injected, and wait for it.

    For a function that holds a file descriptor only a local variable
    knows (from :func:`_open_regular` to the ``with`` that closes it).
    The server's Stop is injected into the task thread at any check
    point of its bytecode: one landing in that interval left the
    descriptor open for the life of the daemon.  The thread started
    here receives no injection, so it always reaches its close, and the
    caller leaves only once it has, or with the work cancelled where a
    Stop took it out of the thread's start: a Stop still finds the file
    whole, and :data:`_FILE_WRITE_LOCK`, which the caller holds, still
    covers the write.
    """

    @functools.wraps(fn)
    def call_unstopped(*args: P.args, **kwargs: P.kwargs) -> T:
        result: Future[T] = Future()
        work = functools.partial(fn, *args, **kwargs)
        worker = threading.Thread(target=_run_unstopped, args=(work, result), daemon=True)
        try:
            worker.start()
            return result.result()
        finally:
            if not result.cancel():  # the work has begun: wait for its end
                result.exception()

    return call_unstopped


@_unstopped
def _read_regular(resolved: Path) -> str | None:
    """Return the UTF-8 text of the regular file *resolved*, ``None`` when it is not one.

    The line endings are the file's: nothing is translated.

    Args:
        resolved: The path to read (:func:`_open_regular`).

    Raises:
        OSError: When the file cannot be opened (``FileNotFoundError``
            when it does not exist).
        UnicodeDecodeError: When it is not UTF-8 text.
    """
    fd = _open_regular(resolved, os.O_RDONLY)
    if fd is None:
        return None
    with open(fd, encoding="utf-8", newline="") as f:
        return f.read()


@_unstopped
def _read_regular_bytes(resolved: Path, limit: int) -> bytes | None:
    """Return the first *limit* bytes of the regular file *resolved*, ``None`` when it is not one.

    Args:
        resolved: The path to read (:func:`_open_regular`).
        limit: The most bytes to read.

    Raises:
        OSError: When the file cannot be opened.
    """
    fd = _open_regular(resolved, os.O_RDONLY)
    if fd is None:
        return None
    with open(fd, "rb") as f:
        return f.read(limit)


@_unstopped
def _write_regular(resolved: Path, content: str, flags: int) -> None:
    """Write *content* into the regular file *resolved* in place, as UTF-8, line endings as given.

    The file is created when missing; *flags* say where the text goes
    (``os.O_TRUNC`` replaces the file's text, ``os.O_APPEND`` follows
    it).

    Raises:
        OSError: When the file cannot be opened, or is not a regular
            file (:func:`_open_regular`): a write into a FIFO or a
            device could block for good.
    """
    fd = _open_regular(resolved, os.O_WRONLY | os.O_CREAT | flags)
    if fd is None:
        raise OSError(
            f"{resolved} is not a regular file (FIFO/device/socket); refusing to write to it"
        )
    with open(fd, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def _not_regular_error(file_path: str) -> str:
    """Return the error of a ``Read`` of *file_path* that is not a regular file."""
    return (
        f"Error: {file_path} is not a regular file "
        f"(FIFO/device/socket); reading it could block forever. "
        f"Use Bash (which has a timeout) if you really need its "
        f"contents."
    )


def _write_file(resolved: Path, content: str) -> None:
    """Replace the content of *resolved* with *content*, as UTF-8 with its line endings as given.

    The text goes to a sibling temporary file renamed over *resolved*
    (``os.replace``, atomic), with the mode of the file it replaces: a
    Stop injected between the truncating open and the write of an
    in-place ``write_text`` left the file empty for good, and another
    agent's Read meanwhile returned ``(file is empty)``; a reader now
    sees the old text or the new.  A file with other hard links is
    written in place, through the link, as :func:`_sed_replaces_link`
    promises (the rename would cut it from its other names); so is a
    file whose directory refuses the temporary file, one that is a
    mount point (a single file bound into a sandbox or a container: the
    rename over it is refused as busy), and one this user may not write
    (the rename takes the directory's permission only: a read-only file
    is refused, as the shell's ``>`` refuses it).  The temporary file
    is given the owner and group of the file it replaces; where that is
    refused the file is written in place too.  Any other failure (a
    full disk) leaves the old text.  The temporary name does not carry
    the file's: a name near the file system's limit leaves no room for
    a longer sibling.
    """
    try:
        st: os.stat_result | None = resolved.stat()
    except FileNotFoundError:
        st = None
    if st is not None and (st.st_nlink > 1 or not os.access(resolved, os.W_OK)):
        _write_regular(resolved, content, os.O_TRUNC)
        return
    tmp = resolved.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "x", encoding="utf-8", newline="") as f:
            f.write(content)
        if st is not None:
            made = tmp.stat()
            if (made.st_uid, made.st_gid) != (st.st_uid, st.st_gid):
                os.chown(tmp, st.st_uid, st.st_gid)
            os.chmod(tmp, stat.S_IMODE(st.st_mode))
        os.replace(tmp, resolved)
    except OSError as error:
        if not isinstance(error, PermissionError) and error.errno != errno.EBUSY:
            raise
        _write_regular(resolved, content, os.O_TRUNC)
    finally:
        tmp.unlink(missing_ok=True)


def _write_or_append(
    resolved: Path, content: str, pending: tuple[Path, str, str] | None,
) -> None:
    """The critical section of :meth:`UsefulTools.Write`: overwrite *resolved*, or append to it."""
    if pending is not None and (resolved, content) == (pending[0], pending[1] + pending[2]):
        # The Write stands for an append (``_pending_append``): the body
        # goes after what the file holds as the Write lands, the file
        # opened for appending as the shell's ``>>`` opens it, so what
        # another writer appended since the file was read for the
        # call, or appends now, is kept.
        _write_regular(resolved, pending[2], os.O_APPEND)
    else:
        _write_file(resolved, content)


def _edit_locked(
    file_path: str, resolved: Path, pairs: list[tuple[str, str, bool]], exact: bool = False,
    rewrite: MechanicalEdit | None = None,
) -> str | tuple[str, list[tuple[int, int, str]]]:
    """The critical section of :meth:`UsefulTools.Edit`: apply *pairs* to *resolved*.

    An *exact* edit matches each ``old`` verbatim: neither the CRLF
    retry nor the whitespace-tolerant one is made.  Pairs standing for
    the shell *rewrite* are written only while they are the ones exact
    for the text the file holds now (:meth:`MechanicalEdit.edits_for`):
    a writer since the command was read may have left a text the
    command changes otherwise (a line holding a ``sed`` pattern twice).

    Returns:
        The new content and ``(position, occurrences, new text)`` per
        edit, or the error naming what went wrong (nothing written).
    """
    original = _read_regular(resolved)
    if original is None:
        return f"Error: File not found: {file_path}"
    content = original
    crlf = "\r\n" in content
    # (position in the final content, occurrences, new text) per edit.
    placed: list[tuple[int, int, str]] = []
    for index, (old, new, every) in enumerate(pairs, 1):
        label = f"edit {index}: " if len(pairs) > 1 else ""
        if old == new:
            return f"Error: {label}new_string must be different from old_string"
        if old == "":
            return (
                f"Error: {label}old_string must not be empty. "
                "Use the Write tool to create or overwrite a file."
            )
        count = content.count(old)
        # A one-line old string is found as it is in a CRLF file too: the
        # lines its new text adds still take the file's line ends.
        if not exact and (count == 0 or "\n" not in old):
            old, new = _crlf_edit(old, new, crlf)
            count = content.count(old)
        position = content.find(old)
        if count == 0 and not every and not exact:
            matches = tolerant_matches(content, old)
            if len(matches) > 1:
                return (
                    f"Error: {label}String not found verbatim; ignoring "
                    f"whitespace it matches {len(matches)} places (not unique). "
                    "Include more surrounding lines in old_string."
                )
            if matches:
                match = matches[0]
                position, count = match.start, 1
                old, new = content[match.start:match.end], reindent(new, match.indent)
        if count == 0:
            return f"Error: {label}String not found in file.{_nearest(content, old)}"
        if not every and count > 1:
            return (
                f"Error: {label}String appears {count} times (not unique). "
                f"Use replace_all=True to replace all occurrences."
            )
        starts = (
            [m.start() for m in re.finditer(re.escape(old), content)]
            if every else [position]
        )
        placed = _moved_by_later_edit(placed, starts, len(old), new)
        placed.append((position, count if every else 1, new))
        content = (
            content.replace(old, new) if every
            else content[:position] + new + content[position + len(old):]
        )
    if rewrite is not None and rewrite.edits_for(original) != pairs:
        return (
            f"Error: {file_path} changed since the {rewrite.what} was read as this Edit, which "
            "the command no longer stands for: nothing was written. Re-issue the command."
        )
    _write_file(resolved, content)
    return content, placed


class _StreamCallbackError(Exception):
    """The caller's ``stream_callback`` raised *error* (:meth:`UsefulTools._stream`).

    Tells :meth:`UsefulTools._run_once` the callback's exception,
    which propagates out of ``Bash``, from an internal failure of the
    command's launch or wait, which is an ``Error:`` result.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(error)
        self.error = error


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    def __init__(
        self,
        stream_callback: Callable[[str], None] | None = None,
        stop_event: threading.Event | None = None,
        work_dir: str | None = None,
        jobs: dict[str, BackgroundJob] | None = None,
        private_dir: str | None = None,
    ) -> None:
        """Initialise the tools.

        Args:
            stream_callback: Optional sink that receives Bash output line
                by line for live streaming to the UI.
            stop_event: Optional event signalled by the host agent when
                the user requests a stop; the Bash tool kills the
                running command's process group.
            work_dir: Agent working directory.  When set, ``Bash``
                subprocesses are launched with ``cwd=work_dir`` and the
                ``KISS_WORKDIR`` env var is forced to ``work_dir`` so
                project scripts that derive a "project root" from it
                stay inside the worktree the agent is operating on.
            jobs: Registry of background jobs (job id ->
                :class:`BackgroundJob`) shared with the host agent, so
                jobs started by an earlier tool instance stay addressable
                through ``bash_job``.  A fresh private registry when
                ``None``.
            private_dir: A sub-agent's private folder
                (:data:`SUBAGENT_DIR_PREFIX`): the temp directory of its
                commands and the home of its job logs and spilled output.
        """
        self.stream_callback = stream_callback
        self.stop_event = stop_event
        self.work_dir = work_dir
        self.private_dir = private_dir
        # Background jobs started by ``Bash(background=True)``, by job id.
        self._jobs: dict[str, BackgroundJob] = {} if jobs is None else jobs
        # Read dedupe: sha256 of a window's text the model was shown ->
        # (step, path, lines) of the Read that showed it (see :meth:`Read`).
        self._reads_shown: dict[str, tuple[int, str, str]] = {}
        # Returns the agent's current step number, for the Read dedupe note;
        # set by the agent that registers these tools.  Without one the
        # note numbers this instance's Read calls.
        self.current_step: Callable[[], int] | None = None
        # Set by the agent that registers these tools: says when every
        # sub-agent job of the task has finished and a result of one is
        # still unread (agent_dispatch.unread_results_note), else "".  A
        # ``Bash(until=)`` poll and a ``bash_job`` wait end early on it:
        # what they wait for can then only be something a child was to
        # leave behind, and its result says what it did instead.
        self.idle_children: Callable[[], str] | None = None
        self._read_calls = 0
        # Outputs spilled to tmp/tool-out/ by this instance (see _spill);
        # the tag keeps instances sharing a work dir (a parent and its
        # reviewer, parallel siblings) from overwriting each other's files.
        self._spill_count = 0
        self._spill_tag = uuid.uuid4().hex[:6]
        # Files this instance has shown the model (Read) or written itself
        # (Write, Edit), keyed by ``edit_guards.repo_read_key`` so a file
        # shown at the main checkout stays read at its worktree path.
        # Edit and Write refuse an existing file that is not in this set,
        # so the read-before-modify rule of SYSTEM.md is enforced by the
        # tools and not only requested by the prompt.
        self._read_keys: set[str] = set()
        # Files this instance wrote (Write, Edit): listed in a
        # continuation's first message.
        self.written_files: set[Path] = set()
        # Snapshots saved by ``checkpoint`` for ``rollback``, by name.
        self._checkpoints = Checkpoints(stop_event)
        # Set by the agent that registers these tools: the Write/Edit a
        # shell rewrite stands for is then raised as a ToolRedirect for
        # the agent to make (its hook, guard and events see it) instead
        # of being made here directly, as it is without an agent.
        self.redirect = False
        # The append a here-document copy stands for (:meth:`_write_instead`):
        # the file, its text when the copy was read for the Write and the
        # body.  The ``Write`` made for it appends to what the file holds
        # by then (an append does not overwrite what another writer added
        # in between); any other call forgets it.
        self._pending_append: tuple[Path, str, str] | None = None
        # The rewrite an ``Edit`` stands for (:meth:`_edit_instead`): the file,
        # the pairs exact for its text when the command was read, and the
        # rewrite.  The ``Edit`` made for it is refused when the file holds
        # a text by then for which the command means other pairs
        # (:func:`_edit_locked`); any other call forgets it.
        self._pending_edit: tuple[Path, list[tuple[str, str, bool]], MechanicalEdit] | None = None

    def _mark_read(self, resolved: Path) -> None:
        """Record that the model has seen *resolved* (a Read, a Write, or a Bash view of it)."""
        self._read_keys.add(repo_read_key(resolved))

    def _is_read(self, resolved: Path) -> bool:
        """Return whether the model has seen *resolved*, at this path or at its other checkout."""
        return repo_read_key(resolved) in self._read_keys

    def forget_reads(self) -> None:
        """Forget which file windows the model has already been shown.

        Called by the agent when earlier tool outputs left the model's
        context (compaction, a new session: its ``context_reset_hook``),
        so the next ``Read`` of an unchanged file returns its content
        instead of the dedupe stub.  The read set is kept: the model has
        still seen the file once in this task, so Edit and Write stay
        permitted on it.
        """
        self._reads_shown.clear()

    def _step(self) -> int:
        """Return the step number the Read dedupe note cites (:attr:`current_step`)."""
        return self.current_step() if self.current_step is not None else self._read_calls

    def _unread_error(self, file_path: str, verb: str) -> str:
        """Return the error text for modifying a file the model has not read."""
        return (
            f"Error: {file_path} has not been read in this session. "
            f"Call Read on it before {verb} it."
        )

    def _unread_edit_refusal(self, file_path: str, resolved: Path, old_string: str) -> str:
        """Refuse an Edit of an unread file, showing the region around *old_string*.

        The refusal is itself a read of that region (the file is marked
        read), so the next Edit call can go ahead without a separate
        Read; the edit is not applied here — the read-before-modify
        rule stays a rule.
        """
        try:
            content = _read_regular(resolved)
        except (OSError, UnicodeDecodeError):
            content = None
        if content is None:
            return self._unread_error(file_path, "editing")
        self._mark_read(resolved)
        return (
            f"Error: {file_path} had not been read in this session, so the edit was NOT "
            f"applied. The region around old_string is shown below (the file now counts "
            f"as read): re-issue the Edit with the exact current text.\n"
            + region_around_string(content, old_string)
        )

    def _spawn(self, command: str, cwd: str | None) -> subprocess.Popen:
        """Launch *command* in *cwd* with the shared Popen configuration.

        Output (stdout + stderr combined) is captured as UTF-8 text and
        the child runs with a cleaned environment and ``/dev/null`` as
        its stdin, as a background job does: with the stdin of this
        process (an open pipe, the terminal of the CLI) a command that
        reads its input waited for it until the timeout.
        Invalid UTF-8 bytes in the output (e.g. ``cat`` of a binary,
        ``grep`` on a binary file) are replaced with U+FFFD instead of
        raising ``UnicodeDecodeError`` — strict decoding would lose the
        ENTIRE output (and, on the streaming path, leak the exception
        out of the tool) even when the command itself succeeded.

        Args:
            command: The shell command to execute.
            cwd: The directory :meth:`_resolve_cwd` chose for this
                command before its guards checked it.  It is not
                resolved again here: a worktree removed since makes the
                launch fail with ``FileNotFoundError`` instead of running
                the command in a fallback directory no guard checked.

        Returns:
            The started subprocess.
        """
        return popen_process_group(
            **_popen_kwargs(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=_clean_env(cwd, self.private_dir),
            cwd=cwd,
        )

    def _resolve_cwd(self) -> str | None:
        """Return the directory a Bash command runs in.

        ``self.work_dir`` when it exists, its parent repository root when
        the worktree vanished mid-task (:func:`remap_vanished_worktree`),
        ``None`` (inherit the agent's
        cwd) when neither exists or no ``work_dir`` was set.  The
        per-task ``.kiss-worktrees/kiss_wt-*`` worktree can be torn down
        by a concurrent cleanup, discard or crashed merge; falling back
        to the main checkout keeps every later command working instead
        of failing on the missing directory.

        Called once per command: its guards and its launch
        (:meth:`_spawn`, :meth:`_start_background_job`) use the same
        result, so a command never runs in a directory its guards did
        not check.
        """
        cwd: str | None = self.work_dir or None
        if cwd is None or os.path.isdir(cwd):
            return cwd
        fallback = remap_vanished_worktree(Path(cwd))
        if fallback.is_dir():
            logger.warning(
                "Bash work_dir %r vanished mid-task; falling back to parent repo root %r",
                cwd,
                str(fallback),
            )
            return str(fallback)
        logger.warning(
            "Bash work_dir %r vanished mid-task and no fallback parent exists; running without cwd",
            cwd,
        )
        return None

    def _start_background_job(
        self, command: str, cwd: str | None, machine: str = ""
    ) -> tuple[BackgroundJob | None, str]:
        """Start *command* detached and register it as a background job.

        Equivalent to ``nohup command > log 2>&1 < /dev/null &``: the
        child runs in its own session (so no terminal SIGHUP can reach
        it), reads stdin from ``/dev/null`` and writes stdout and stderr
        to ``<cwd>/tmp/bash_jobs/<job_id>.log`` (a sub-agent's: to
        ``bash_jobs/`` in its private folder).  Unlike a shell ``&``
        the tool keeps the ``Popen`` handle, so :meth:`bash_job` can
        report the exit code and a task Stop kills the job's process
        group like it kills a foreground command.

        With *machine* (a registered SSH machine, ``/machine add``) the
        command runs there through ``ssh``: the ssh client is the local
        job, so the remote output streams into the same local log and
        ``bash_job`` works unchanged; an experiment on a bench node is
        one turn to start and one to collect.

        Args:
            command: The shell command to run.
            cwd: The directory :meth:`_resolve_cwd` chose before the
                command's guards ran (see :meth:`_spawn`); it also holds
                the log.
            machine: Name or address of a registered SSH machine to run
                on; empty runs locally.

        Returns:
            The registered job and the model-facing text giving its id,
            pid and log path plus how to follow it; ``None`` and a string
            starting with ``Error:`` when it could not be started.
        """
        job_id = "j" + uuid.uuid4().hex[:8]
        root, logs, _ = _tools_dir(self.private_dir, cwd or os.getcwd(), JOB_LOG_DIR)
        log_path = Path(root, logs, f"{job_id}.log")
        popen_kwargs = _popen_kwargs(command)
        if machine:
            from kiss.agents.sorcar.machines import find_machine, ssh_command

            found = find_machine(machine)
            if found is None:
                return None, f"Error: no registered machine named {machine!r} (see /machine list)."
            if not found.ssh:
                return None, (
                    f"Error: machine {found.name!r} was registered by URL, not over ssh; a "
                    "background job can only be started on an SSH machine."
                )
            popen_kwargs = {"args": ssh_command(found.ssh, command=command)}
        # The shell is spawned by the watcher thread that owns it
        # (:func:`_spawn_owned`): a Stop injected anywhere in this
        # region, inside ``Popen`` after the fork included, or a failure
        # before the job is registered, ends with the shell killed
        # (:func:`_watch_job`) instead of running on with no watcher and
        # no job id anyone could kill it by.
        settled, registered, kill, released = (threading.Event() for _ in range(4))
        try:
            try:
                _make_dirs_under(root, logs)
                process, ended = _spawn_owned(
                    functools.partial(_spawn_job, popen_kwargs, log_path, cwd, self.private_dir),
                    functools.partial(
                        _watch_job, self.stop_event, settled=settled, registered=registered,
                        interrupt=tool_interrupt.current_tool_interrupt_event(),
                        kill=kill, released=released,
                    ),
                )
            except (OSError, ValueError) as e:
                # ValueError: a command ``Popen`` cannot pass on (a NUL byte, a lone surrogate).
                logger.debug("Exception caught", exc_info=True)
                return None, f"Error: could not start background job: {e}"
            job = BackgroundJob(
                job_id,
                f"[{machine}] {command}" if machine else command,
                log_path,
                process,
                time.monotonic(),
                ended,
                kill,
                released,
            )
            self._jobs[job_id] = job
            registered.set()
        finally:
            settled.set()
        where = f" on {machine}" if machine else ""
        return job, (
            f"Started background job {job_id}{where} (pid {process.pid}).\n"
            f"Log: {log_path}\n"
            f'Follow it with bash_job({job_id!r}, action="tail") or block on it with '
            f'bash_job({job_id!r}, action="wait", timeout_seconds=...); '
            f'bash_job({job_id!r}, action="kill") stops it.'
        )

    def _wait_for_job(
        self, job: BackgroundJob, timeout_seconds: float, until: re.Pattern[str] | None = None,
    ) -> str:
        """Block until *job* exits, *timeout_seconds* pass, the log matches *until* or a stop lands.

        Polls in 0.2 s slices so the task Stop event and the tool
        panel's interrupt event are observed promptly; a tool-panel
        Stop raises :class:`~kiss.core.tool_interrupt.ToolCallInterrupted`
        after the loop, exactly like a foreground :meth:`Bash`.  The
        log is searched once a second for *until* by a matcher child
        (:func:`_start_log_matcher`) that exits 0 on a match and is
        killed when the wait ends, so a pattern that backtracks for
        minutes cannot hold the task thread past the deadline or a Stop.
        A helper thread (:func:`_run_log_matcher`) owns that child, is
        started inside the cleanup region and ends the child on its own
        at the deadline or either Stop, so a Stop landing anywhere in
        this frame, the launch and the cleanup included, still ends it.

        Returns:
            ``""`` when the wait ran to the job's exit, the deadline, a
            match or a Stop; the ``Error:`` text of a matcher that could
            not be started or ended without a match (the wait ends at
            once then); or the note of :attr:`idle_children` when the
            wait ended early on it.
        """
        deadline = time.monotonic() + timeout_seconds
        done, matched = threading.Event(), threading.Event()
        failed: list[str] = []
        matcher = None
        idle = ""
        if until is not None:
            stops = [
                event
                for event in (self.stop_event, tool_interrupt.current_tool_interrupt_event())
                if event is not None
            ]
            matcher = threading.Thread(
                target=_run_log_matcher,
                args=(
                    functools.partial(_start_log_matcher, job.log_path, until),
                    done, stops, deadline, matched, failed,
                ),
                daemon=True,
            )
        try:
            if matcher is not None:
                matcher.start()
            while job.returncode() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or _stop_requested(self.stop_event):
                    break
                if matched.is_set() or failed:
                    break
                idle = self._idle_children_note()
                if idle:
                    break
                time.sleep(min(remaining, 0.2))
        finally:
            done.set()
            if matcher is not None and matcher.is_alive():
                matcher.join(timeout=5)
        tool_interrupt.raise_if_interrupted()
        if failed and until is not None:
            return (
                f"Error: could not start the log matcher for until={until.pattern!r}: {failed[0]}"
            )
        return idle

    def _idle_children_note(self) -> str:
        """Return :attr:`idle_children`'s note (every sub-agent done, a result unread) or ``""``."""
        return self.idle_children() if self.idle_children is not None else ""

    def Read(  # noqa: N802
        self,
        file_path: str,
        max_lines: int = 2000,
        start_line: int = 1,
        force: bool = False,
    ) -> str:
        """Read file contents; a long file with no line range returns its outline first.

        Args:
            file_path: Absolute path to file.
            max_lines: Maximum number of lines to return.
            start_line: 1-indexed line at which to begin the returned
                window.  ``start_line=1`` (the default) reads from the
                top of the file and is backward-compatible.  Values
                less than 1 are rejected; values beyond EOF return an
                explicit sentinel rather than empty content so the
                model is not misled into thinking the file is empty.
            force: Re-send a window whose text an earlier Read of this
                session already returned, from any file or range (by
                default such a Read returns a one-line "unchanged" note
                naming that Read instead of the content).
        """
        if start_line < 1:
            return f"Error: start_line must be >= 1 (got {start_line}); the parameter is 1-indexed."
        if max_lines < 1:
            return f"Error: max_lines must be >= 1 (got {max_lines})."
        self._read_calls += 1
        try:
            self._pending_append = self._pending_edit = None
            resolved = resolve_tool_path(file_path, self.work_dir, Path.exists)

            if resolved.is_dir():
                return self._read_directory_listing(file_path, resolved)

            if resolved.exists() and not resolved.is_file():
                return _not_regular_error(file_path)

            mime_type, _ = mimetypes.guess_type(str(resolved))
            try:
                if (
                    mime_type in READ_TOOL_BINARY_MIME_TYPES
                    and resolved.stat().st_size > _MAX_BINARY_READ_BYTES
                ):
                    # Refused by size before the text attempt reads it whole.
                    return self._read_binary(file_path, resolved)
                text = _read_regular(resolved)
            except UnicodeDecodeError:
                logger.debug("Binary file detected", exc_info=True)
                shown = self._read_binary(file_path, resolved)
                if not shown.startswith("Error:"):
                    self._mark_read(resolved)
                return shown
            except FileNotFoundError:
                suggestion = _suggest_close_path(resolved)
                return f"Error: File not found: {file_path}.{suggestion}"
            if text is None:
                return _not_regular_error(file_path)

            if text == "":
                self._mark_read(resolved)
                return "(file is empty)"

            # ``\r\n`` is shown as ``\n``; a lone ``\r`` stays in its line, which ends
            # where grep and sed end it.
            text = text.replace("\r\n", "\n")
            lines = split_lines(text)
            total = len(lines)
            if start_line > total:
                return (
                    f"Error: start_line={start_line} is past EOF "
                    f"(file has {total} line{'s' if total != 1 else ''})."
                )
            self._mark_read(resolved)  # not above: a Read past the end shows nothing of it
            outline_limit = DEFAULT_CONFIG.read_outline_lines
            whole_file_requested = start_line == 1 and max_lines >= 2000
            outline = (
                _outline(file_path, lines, len(text))
                if whole_file_requested and 0 < outline_limit < total
                else None
            )
            window = lines[start_line - 1 : start_line - 1 + max_lines]
            if whole_file_requested:
                # No range asked for: keep the default output budget, the
                # way Bash does, so a 200 KB file does not land whole in
                # the context; the note says where to continue.
                window = _lines_within_budget(window, DEFAULT_CONFIG.tool_output_default_chars)
            last = start_line - 1 + len(window)
            remaining = total - last
            if outline is not None:
                content, shown = outline, "outline"
            else:
                content, shown = "".join(window), f"lines {start_line}-{last}"
            # The window's text, before the continuation note, is what is
            # matched against every window shown so far, from any file.
            digest = hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()
            earlier = self._reads_shown.get(digest)
            if DEFAULT_CONFIG.read_dedupe and not force and earlier is not None:
                step, path, where = earlier
                return (
                    f"Unchanged since your earlier Read at step {step} ({where} of {path}): "
                    f"the content is already in your context above. Pass force=True to "
                    f"re-read it."
                )
            self._reads_shown.setdefault(digest, (self._step(), file_path, shown))
            if outline is None and remaining > 0:
                content += (
                    f"\n[truncated: {remaining} more lines; "
                    f"Read(file_path, start_line={last + 1}, max_lines=N) continues]"
                )
            return content
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _read_directory_listing(self, file_path: str, resolved: Path) -> str:
        """Return a helpful directory listing when Read is called on a dir.

        Models occasionally call ``Read`` on a directory path (e.g. when
        searching for the right module).  Instead of returning the bare
        ``[Errno 21] Is a directory`` error we surface a one-per-line
        listing so the model can self-correct on the next turn.
        """
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in resolved.iterdir())
        except OSError as e:  # pragma: no cover — permission edge case
            return f"Error: Cannot list directory {file_path}: {e}"
        listing = "\n".join(entries) if entries else "(empty directory)"
        return (
            f"Error: {file_path} is a directory, not a file. "
            f"Pass a file path inside it, or use Bash('ls -la ...').\n"
            f"Directory contents:\n{listing}"
        )

    def _read_binary(self, file_path: str, resolved: Path) -> str:
        """Encode a binary file as a sentinel attachment or return error."""
        mime_type, _ = mimetypes.guess_type(str(resolved))
        if mime_type in READ_TOOL_BINARY_MIME_TYPES:
            size = resolved.stat().st_size
            if size > _MAX_BINARY_READ_BYTES:
                return (
                    f"Error: Binary file {file_path} is too large to embed "
                    f"inline ({size} bytes > {_MAX_BINARY_READ_BYTES} byte "
                    f"limit, mime={mime_type}). Use Bash tools (e.g. "
                    f"ffmpeg/ImageMagick to downscale, or split the file) "
                    f"to produce a smaller artifact first."
                )
            data = _read_regular_bytes(resolved, _MAX_BINARY_READ_BYTES)
            if data is None:
                return _not_regular_error(file_path)
            header = (
                f"Read binary file {file_path} as {mime_type} "
                f"({len(data)} bytes); content attached below.\n"
            )
            return header + encode_binary_attachment(mime_type, data)
        size = resolved.stat().st_size
        return (
            f"Error: Cannot read binary file: {file_path} "
            f"(size: {size} bytes, mime={mime_type or 'unknown'}). "
            f"The Read tool only embeds binaries with a supported "
            f"MIME type (images, PDFs, audio, video); use a "
            f"different tool to handle this binary file."
        )

    def Write(  # noqa: N802
        self,
        file_path: str,
        content: str,
    ) -> str:
        """Write content to a file, creating it if it doesn't exist or overwriting if it does.

        Args:
            file_path: Path to the file to write.
            content: The full content to write to the file.
        """
        try:
            pending, self._pending_append, self._pending_edit = self._pending_append, None, None
            resolved = resolve_tool_path(file_path, self.work_dir)
            if resolved.exists() and not resolved.is_file():
                return (
                    f"Error: {file_path} exists and is not a regular file "
                    f"(directory/FIFO/device/socket); refusing to write to it."
                )
            if (
                resolved.is_file()
                and not self._is_read(resolved)
                and not _is_scratch_path(resolved, self.work_dir)
            ):
                return self._unread_error(file_path, "overwriting")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            locked_call(_FILE_WRITE_LOCK, _write_or_append, resolved, content, pending)
            self._mark_read(resolved)
            self.written_files.add(resolved)
            return f"Successfully wrote {len(content)} characters to {file_path}"
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str = "",
        new_string: str = "",
        replace_all: bool = False,
        edits: str = "",
        exact: bool = False,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        An ``old_string`` not in the file verbatim is matched line by
        line with runs of blanks collapsed and trailing whitespace
        ignored (:func:`~kiss.agents.sorcar.edit_matching.tolerant_matches`);
        a unique match is replaced in place of its original lines, the
        replacement indented as the file's lines are when they only differ
        from ``old_string`` by a uniform indentation.  When nothing matches
        that way either, the error names the region of the file closest to
        ``old_string`` with its line numbers, so the next Edit needs no Read.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences (of the exact
                text only).
            edits: Several replacements in one call, as a JSON list of
                ``{"old_string": ..., "new_string": ..., "replace_all": bool}``
                objects applied in order (all or none); ``old_string`` /
                ``new_string`` are then ignored.
            exact: If True, match verbatim only: no retry ignoring
                whitespace or line endings.  Set on the Edit a shell
                rewrite (a ``sed -i``, a python ``.replace``) is applied as.

        Returns:
            The lines around each replacement after the edit, up to the
            file's last (so no verifying Read is needed), or an error
            naming what went wrong.
        """
        try:
            pending, self._pending_edit, self._pending_append = self._pending_edit, None, None
            resolved = resolve_tool_path(file_path, self.work_dir, Path.is_file)
            if not resolved.is_file():
                return f"Error: File not found: {file_path}"
            pairs = _edit_pairs(edits) if edits else [(old_string, new_string, replace_all)]
            if isinstance(pairs, str):
                return pairs
            rewrite = pending[2] if pending and pending[:2] == (resolved, pairs) else None
            if not self._is_read(resolved) and not _is_scratch_path(resolved, self.work_dir):
                # a scratch file (a note of an earlier session, say) is edited unread,
                # as Write overwrites one: the result shows the region changed
                return self._unread_edit_refusal(file_path, resolved, pairs[0][0])
            applied = locked_call(
                _FILE_WRITE_LOCK, _edit_locked, file_path, resolved, pairs, exact, rewrite,
            )
            if isinstance(applied, str):
                return applied
            content, placed = applied
            self.written_files.add(resolved)
            total = sum(replaced for _, replaced, _ in placed)
            done = f"Successfully replaced {total} occurrence(s) in {file_path}\n"
            if not content:
                return done + "The file is empty after the edit."
            last_line = content.count("\n") + (not content.endswith("\n"))
            reports: list[str] = []
            for index, (position, replaced, new) in enumerate(placed, 1):
                first = line_of(content, position)
                start = max(1, first - 3)
                end = min(first + new.rstrip("\n").count("\n") + 3, last_line)
                head = f"edit {index} ({replaced} occurrence(s)); " if len(pairs) > 1 else ""
                reports.append(
                    f"{head}lines {start}-{end} after the edit:\n"
                    + numbered_region(content, start, end)
                )
            return done + "\n\n".join(reports)
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return f"Error: {e}"

    def _rewrite_instead(
        self,
        command: str,
        cwd: str,
        description: str,
        timeout_seconds: float,
        max_output_chars: int,
        until: str = "",
        **options: Any,
    ) -> Ran | None:
        """Apply the ``Write``/``Edit`` a shell rewrite of a file stands for, or refuse the rewrite.

        A ``cat``/``tee`` here-document copy into an existing file
        (:func:`heredoc_copy`), a python ``.replace`` rewrite or a
        one-substitution ``sed -i`` (:func:`mechanical_edit`) is applied
        through :meth:`Write` / :meth:`Edit` when the file was shown, so
        the change is logged and reviewable and the model pays no turn
        to re-issue it; a file not shown is refused naming the exact
        calls to make.  The rest of a compound command (``cd DIR &&
        <rewrite> && <rest>``) then runs as the ``Bash`` call
        :func:`followup_call` names, once the ``Write``/``Edit`` went
        through; a ``cd`` that bash does not walk to the directory it
        names is refused up front (:func:`cd_refusal`), as bash would
        run nothing after a failed one.  Any other rewrite is refused
        as :func:`rewrite_guard` and :func:`heredoc_guard` refuse it.

        Args:
            command: The shell command text.
            cwd: The host directory it would run in.
            description: The call's description, for the follow-up.
            timeout_seconds: The call's deadline, for the follow-up.
            max_output_chars: The call's output budget, for the follow-up.
            until: The call's ``until`` regex: a rewrite it would poll
                is refused instead of applied (:meth:`_apply`).
            **options: The call's ``background`` and ``on_timeout``, for
                the follow-up (:func:`followup_call`).

        Returns:
            The result of the applied call (with how the command ended,
            :meth:`_apply`), a refusal, or ``None`` when *command* is no
            such rewrite and may run.
        """
        bash_cwd = bash_path(os.path.abspath(cwd))
        copy = heredoc_copy(command, bash_cwd)
        if copy is not None and copy.path is not None:
            if copy.path.startswith("/dev/"):
                return None
            refusal = cd_refusal(copy.cwd, _host_isdir)
            if refusal is not None:
                return unrun(refusal)
            if not _host_lexists(copy.path):
                return None  # the here-document creates the file: the shell may
            followup = followup_call(
                copy.cwd, copy.remainder, description, timeout_seconds, max_output_chars,
                **options,
            )
            return self._write_instead(copy, host_path(copy.path), followup, until)
        edit = mechanical_edit(command, bash_cwd, _host_shadows, _host_runs)
        if edit is not None and edit.path is not None:
            refusal = cd_refusal(edit.cwd, _host_isdir)
            if refusal is not None:
                return unrun(refusal)
            followup = followup_call(
                edit.cwd, edit.remainder, description, timeout_seconds, max_output_chars,
                **options,
            )
            applied = self._edit_instead(edit, host_path(edit.path), followup, until)
            if applied is not None:
                return applied
        guarded = functools.partial(_host_guarded, self.work_dir)
        refusal = rewrite_guard(command, bash_cwd, guarded, _host_isdir) or heredoc_guard(
            command, bash_cwd, guarded, _host_isdir)
        return None if refusal is None else unrun(refusal)

    def _write_instead(
        self,
        copy: HeredocCopy,
        path: str,
        followup: tuple[str, dict[str, Any]] | None,
        until: str = "",
    ) -> Ran:
        """Write the here-document *copy* into the existing file *path* through :meth:`Write`.

        The file must have been shown (or be a scratch file, which
        :meth:`Write` overwrites unread); otherwise the refusal names the
        ``Read`` to make before the same command, and the ``Write``.
        An append (``>>``, ``tee -a``) is the ``Write`` of the file's
        text followed by the body, and is remembered as the pending
        append :meth:`Write` makes as the shell would, after the file's
        text by then; one into a file that is not regular is refused
        unread.
        *followup* is the ``Bash`` call running the rest of the command
        after the ``Write`` (:func:`followup_call`), if any; *until* the
        command's poll regex, which refuses the ``Write`` (:meth:`_apply`).
        """
        resolved = resolve_tool_path(path, self.work_dir)
        shown = json.dumps(path)
        if not self._is_read(resolved) and not _is_scratch_path(resolved, self.work_dir):
            content = (
                f"<the current content followed by {preview(copy.body)}>" if copy.append
                else preview(copy.body)
            )
            return unrun(
                HEREDOC_VERDICT.format(path=copy.path)
                + f"\nThe file was not shown in this session: call Read(file_path={shown}) and "
                "then the same command again (it is then applied as that Write), or "
                f"Write(file_path={shown}, content={content})."
            )
        content = copy.body
        old = ""
        if copy.append and resolved.exists():
            try:
                text = _read_regular(resolved)
            except (OSError, UnicodeDecodeError) as e:
                return unrun(f"Error: cannot append to {path}: {e}")
            if text is None:
                # Seen on the descriptor, not read: reading a FIFO blocks.
                return unrun(
                    f"Error: {path} exists and is not a regular file "
                    f"(directory/FIFO/device/socket); refusing to append to it."
                )
            old = text
            content = old + content
        if copy.append:
            # Remembered for a file gone since the command was checked too:
            # another writer may create it before the Write lands, and
            # ``>>`` would have kept its bytes.
            self._pending_append = (resolved, old, copy.body)
        call = f"Write(file_path={shown}, content={preview(content)})"
        args = {"file_path": path, "content": content}
        return self._apply(call, "Write", args, copy.append, followup, until)

    def _edit_instead(
        self,
        edit: MechanicalEdit,
        path: str,
        followup: tuple[str, dict[str, Any]] | None,
        until: str = "",
    ) -> Ran | None:
        """Apply the *edit* a python heredoc or ``sed -i`` stands for to *path* via :meth:`Edit`.

        The pairs are the ones exact for the file's text
        (:meth:`MechanicalEdit.edits_for`: a per-line ``sed`` matching
        several lines gets ``replace_all``); when none is exact (a line
        holds the pattern twice) the result is ``None`` and the rewrite
        guards decide.  They are remembered, so that the ``Edit`` made
        for them is refused once the file holds a text they are not
        exact for (:func:`_edit_locked`).  A file not shown is refused with the ``Edit``
        call to make and the region around its first ``old_string``; the
        file then counts as read (as after an ``Edit`` of an unread
        file), so the re-issued call goes through.  *followup* is the
        ``Bash`` call running the rest of the command after the ``Edit``
        (:func:`followup_call`), if any; *until* the command's poll
        regex, which refuses the ``Edit`` (:meth:`_apply`).  A ``sed -i``
        on a link is not converted (:func:`_sed_replaces_link`), nor one
        on a file this user may not write, which sed replaces all the
        same (its rename takes the directory's permission) where the
        ``Edit``, writing in place, is refused: the guards decide.  So
        they do for a rewrite of a name the system does not take as
        written (:func:`_fails_as_written`).
        """
        if _fails_as_written(path) or edit.what != "python heredoc" and (
            _sed_replaces_link(path) or os.path.isfile(path) and not os.access(path, os.W_OK)
        ):
            return None
        resolved = resolve_tool_path(path, self.work_dir, Path.is_file)
        content: str | None = None
        if resolved.is_file():
            with suppress(OSError, UnicodeDecodeError):
                content = _read_regular(resolved)
            if content is None and not self._is_read(resolved):
                call = edit_call(path, edit.edits)
                return unrun(
                    f"REFUSED: the {edit.what} did NOT run: it rewrites {path}, which was "
                    f"not shown in this session. Read(file_path={json.dumps(path)}) first, "
                    f"then {call}."
                )
        edits = edit.edits if content is None else edit.edits_for(content)
        if edits is None:
            return None
        # Remembered for the Edit made for it: the pairs are exact for the text read here.
        self._pending_edit = (resolved, edits, edit)
        call = edit_call(path, edits)
        if content is not None and not self._is_read(resolved):
            # A scratch file counts as shown, as Write overwrites one unread.
            self._mark_read(resolved)
            if not _is_scratch_path(resolved, self.work_dir):
                return unrun(
                    f"REFUSED: the {edit.what} did NOT run: it rewrites {path}, which was not "
                    f"shown in this session, with no reviewable change. Make the same change with "
                    f"{call} — the region around its first old_string follows and the file now "
                    "counts as read, so that call (or the same command, applied as that Edit) "
                    "goes through.\n" + region_around_string(content, edits[0][0])
                )
        return self._apply(call, "Edit", edit_args(path, edits), followup=followup, until=until)

    def _apply(
        self,
        call: str,
        name: str,
        args: dict[str, Any],
        appended: bool = False,
        followup: tuple[str, dict[str, Any]] | None = None,
        until: str = "",
    ) -> Ran:
        """Make the ``Write``/``Edit`` *call* (*name* with *args*) a shell rewrite stands for.

        Under an agent (:attr:`redirect`) the call is raised as a
        :class:`ToolRedirect` for the agent to make through its own tool
        path, so its hook, guard and events see the ``Write``/``Edit``
        as if the model had issued it, and the *followup* after it;
        without one both are made here.  A command polled with *until*
        is refused instead, before any edit: ``until`` re-runs a
        command as written, and the call stands for a change made once.

        Args:
            call: The call as the result names it (:func:`edit_call`).
            name: ``"Write"`` or ``"Edit"``.
            args: The call's arguments.
            appended: Whether the rewrite appended (a ``>>`` copy).
            followup: The ``Bash`` call running the rest of the command
                (:func:`followup_call`), made once the call went through.
            until: The ``until`` regex of the ``Bash`` call, if any.

        Returns:
            The call's result, prefixed with what it was applied as, then
            the follow-up's (:func:`followup_result`); or the refusal of
            a polled rewrite, naming the call to make and the rest to poll.
            The command ended as its follow-up did; with none, as the
            call did (``"0"``, or ``"error"`` when it failed).

        Raises:
            ToolRedirect: Under an agent, naming the calls to make.
        """
        if until:
            rest = "" if followup is None else (
                f", then poll {json.dumps(followup[1]['command'])} with until= in a Bash call "
                "of its own"
            )
            return unrun(
                f"Error: until= re-runs a command as written, but this one is applied as {call} "
                f"instead of running: make that call{rest}."
            )
        if self.redirect:
            raise rewrite_redirect(name, args, call, appended, followup)
        result = self.Write(**args) if name == "Write" else self.Edit(**args)
        ran = Ran(_applied(call, result, appended), "error" if result.startswith("Error") else "0")
        if followup is None:
            return ran
        tool, arguments = followup
        # The rest runs only once the call went through; its exit is then the command's.
        after = self._bash(**arguments) if ran.code == "0" else None
        made = followup_result(tool, arguments, None if after is None else after.text, name)
        return (after or ran)._replace(text=f"{ran.text}\n{made}")

    def _last_running_job(self) -> BackgroundJob | None:
        """Return the background job of this instance started last that is still running."""
        for job in reversed(self._jobs.values()):
            if job.returncode() is None:
                return job
        return None

    def _last_running_job_id(self) -> str | None:
        """Return the id of the running background job started last, or ``None``."""
        job = self._last_running_job()
        return None if job is None else job.job_id

    def _wait_on_job(self, job_id: str, seconds: float) -> str:
        """Wait up to *seconds* for the job *job_id* to exit and return its status line.

        The status carries the :attr:`idle_children` note when the wait
        ended on it, as ``bash_job``'s report does, so the sleep that
        became this wait (:func:`wait_instead_of_sleep`) says why it
        returned early and what to read.
        """
        job = self._jobs[job_id]
        ended = self._wait_for_job(job, seconds)
        return f"{job.status()}; the wait ended early: {ended}" if ended else job.status()

    def _wait_instead_of_sleep(
        self, command: str, timeout_seconds: float,
    ) -> tuple[str | None, str, str]:
        """Settle a long ``sleep`` of *command* before it runs (:func:`wait_instead_of_sleep`)."""
        return wait_instead_of_sleep(
            command, timeout_seconds, self._last_running_job_id, self._wait_on_job,
        )

    def _not_started(self, max_output_chars: int) -> Ran | None:
        """Return the result of a ``Bash`` call a Stop precedes, ``None`` when no Stop is set.

        Nothing is started then, as in the fan-out (:func:`_run_one_command`):
        each launch path spawns its shell before the thread owning it looks
        at the Stop, so under load the command ran first, and the paths that
        start a job returned it as running.  The tool panel's Stop raises
        :class:`~kiss.core.tool_interrupt.ToolCallInterrupted`.
        """
        if not _stop_requested(self.stop_event):
            return None
        tool_interrupt.raise_if_interrupted()
        return Ran(_format_bash_result(-1, NOT_STARTED, max_output_chars), "-1")

    # Bash calls made and the call number at which the next batching hint may go out.
    _bash_calls = 0
    _hint_due_at = 1

    def batching_hint(self, name: str, args: dict[str, Any], result: str, seconds: float) -> str:
        """Return :data:`BATCHING_HINT` for a short read-only ``Bash`` call, once per 20 calls.

        Args:
            name: The tool called; only ``"Bash"`` is rated.
            args: Its arguments (``command``).
            result: What it returned.
            seconds: How long it took.

        Returns:
            The hint when the call only read (:func:`_is_read_only`), took
            under 2 s and returned under 2,000 characters, and no hint went
            out in the last :data:`BATCHING_HINT_EVERY` Bash calls; else ``""``.
        """
        if name != "Bash":
            return ""
        self._bash_calls += 1
        if (
            self._bash_calls < self._hint_due_at
            or seconds >= _SHORT_CALL_SECONDS
            or len(result) >= _SHORT_CALL_CHARS
            or not _is_read_only(str(args.get("command", "")))
        ):
            return ""
        self._hint_due_at = self._bash_calls + BATCHING_HINT_EVERY + 1
        return BATCHING_HINT

    def Bash(  # noqa: N802
        self,
        command: str,
        description: str,
        timeout_seconds: float = 300,
        max_output_chars: int = DEFAULT_CONFIG.tool_output_default_chars,
        background: bool = False,
        machine: str = "",
        on_timeout: str = "kill",
        until: str = "",
        poll_seconds: float = 5.0,
    ) -> str:
        """Runs a bash command and returns its output.

        With ``background=True`` the command is started detached (like
        ``nohup cmd > log 2>&1 < /dev/null &``) and the call returns at
        once with a job id; follow the job with ``bash_job``.  Use it for
        anything expected to run longer than about a minute (builds,
        training runs, servers, large test suites).  A foreground call
        with ``timeout_seconds`` over the configured grace period (120 s)
        that is still running after that grace goes on as a background
        job: the result gives its job id, its output so far and the
        ``bash_job`` wait to make.  A foreground ``sleep N`` longer than
        ``timeout_seconds`` is refused; while a background job of this
        task is running, a long sleep waits on that job instead.  To wait
        for a condition (a line in a log, a file, a port) pass ``until``:
        the command is re-run every ``poll_seconds`` inside this one call
        until the regex matches its output or ``timeout_seconds`` pass,
        instead of one call per poll.

        Args:
            command: The bash command to run.
            description: A brief description of the command.
            timeout_seconds: Timeout in seconds for the command.  Ignored
                when ``background`` is true; put the deadline on the
                ``bash_job`` wait instead.
            max_output_chars: Characters of output kept in the result
                (head and tail); the rest is written to tmp/tool-out/
                and the result names the file.  Raise it for a result
                you need whole, up to the configured ceiling (50,000).
            background: Start the command in the background and return a
                job id immediately instead of waiting for it.
            machine: With ``background=True``, the registered SSH machine
                (``/machine add``) to run the command on; its output
                streams into the local job log.
            on_timeout: ``"kill"`` (default) kills the command at the
                deadline and returns what it printed; ``"background"``
                starts the command as a background job up front and
                waits ``timeout_seconds`` for it, so a command that
                outlives the deadline keeps running and is followed with
                ``bash_job``.
            until: A regular expression; when given, the command is run
                again every ``poll_seconds`` until the regex matches its
                combined output or ``timeout_seconds`` elapse, and the
                result is the last run's output plus a line saying whether
                the regex matched and after how many runs and seconds.
                The poll also ends once every sub-agent job of this task
                has finished with a result you have not read (the line
                says so: read it with ``agent_job('', 'wait_any')``).
                Not for ``background=True`` (follow a job's log with
                ``bash_job(id, "wait", until=...)`` instead).
            poll_seconds: Seconds between two runs of an ``until`` poll
                (at least 1).

        Returns:
            The output of the command, or for a background job its job
            id, pid and log file path.
        """
        ran = self._bash(
            command, description, timeout_seconds, max_output_chars, background, machine,
            on_timeout, until, poll_seconds,
        )
        hand_over(command, ran.code, job_id=ran.job_id)
        return ran.text

    def _bash(
        self,
        command: str,
        description: str,
        timeout_seconds: float = 300,
        max_output_chars: int = DEFAULT_CONFIG.tool_output_default_chars,
        background: bool = False,
        machine: str = "",
        on_timeout: str = "kill",
        until: str = "",
        poll_seconds: float = 5.0,
    ) -> Ran:
        """Make a :meth:`Bash` call: its result and how the command ended.

        The command's exit is data here, which :meth:`Bash` hands over
        to the coding gate (:mod:`~kiss.agents.sorcar.shell_exit`): the
        result's text holds the command's own output, where no reader
        can tell a status the command printed from the tool's.  The rest
        of a rewrite made here (:meth:`_apply`) is run through this, so
        only the call the model made hands anything over.
        """
        max_output_chars = _output_budget(max_output_chars)
        started = time.monotonic()
        self._pending_append = self._pending_edit = None
        stopped = self._not_started(max_output_chars)
        if stopped is not None:
            return stopped
        # The arguments are checked before anything runs: a shell rewrite
        # below is applied as the Edit it stands for.
        if until and background:
            return unrun(
                "Error: until= polls a foreground command; with background=True follow the "
                "job's log with bash_job(id, \"wait\", until=<regex>) instead."
            )
        if machine and not background:
            return unrun("Error: machine= applies to background=True only.")
        if not background and on_timeout not in ("kill", "background"):
            return unrun(
                f"Error: on_timeout must be \"kill\" or \"background\", got {on_timeout!r}.")
        regex = _until_pattern(until) if until else None
        if isinstance(regex, str):
            return unrun(regex)

        cwd = self._resolve_cwd()
        guard, note, command, timeout_seconds = _common_guards(
            command, self.work_dir, timeout_seconds, not machine)
        if guard is None and machine:
            guard = rewrite_guard(command)  # a machine's files this host cannot see
        if guard is not None:
            return unrun(guard)
        if not machine:
            # A shell rewrite of a file is applied as the Write/Edit it stands
            # for, or refused naming it.
            applied = self._rewrite_instead(
                command, cwd or os.getcwd(), description, timeout_seconds, max_output_chars,
                until, background=background, on_timeout=on_timeout,
            )
            if applied is not None:
                return applied
        if background:
            job, text = self._start_background_job(command, cwd, machine)
            return unrun(text + note) if job is None else Ran(text + note, "background", job.job_id)
        refused, waited, command = self._wait_instead_of_sleep(command, timeout_seconds)
        if refused is not None:
            return unrun(refused)
        # The wait a long sleep was spent on ends on a Stop: the rest is not started either.
        stopped = self._not_started(max_output_chars)
        if stopped is not None:
            return stopped
        note += waited
        grace = DEFAULT_CONFIG.bash_detach_seconds
        ran = None
        if regex is not None:
            # The wait a long sleep was spent on counts against the call's deadline.
            ran = self._poll_until(
                command, regex, poll_seconds, started + timeout_seconds, max_output_chars, cwd,
            )
        elif on_timeout == "background":
            ran = self._run_then_detach(command, timeout_seconds, max_output_chars, cwd)
        elif 0 < grace < timeout_seconds:
            ran = self._run_then_detach(command, timeout_seconds, max_output_chars, cwd, grace)
        if ran is None:
            ran = self._run_once(command, timeout_seconds, max_output_chars, cwd)[0]
        hint = self.batching_hint(
            "Bash", {"command": command}, ran.text, time.monotonic() - started,
        )
        return ran._replace(text=ran.text + hint + note)

    def _run_once(
        self, command: str, timeout_seconds: float, max_output_chars: int, cwd: str | None,
    ) -> tuple[Ran, str | None, int]:
        """Run *command* once in the foreground: its formatted result, its output, what was dropped.

        The output (stdout and stderr combined, as captured) is ``None``
        when the command could not be launched or was killed at the
        deadline, the cases that end an ``until`` poll
        (:meth:`_poll_until`); the dropped count is the characters of
        output the capture left out between its head and its tail
        (:class:`_Capture`).  A command killed at the deadline, or one
        that could not be launched, gives a result starting with
        ``Error:``; a finished one counts as a view of the file it
        showed (:meth:`_note_shell_view`).
        """
        capture = _Capture(max_output_chars, self._new_spill)
        try:
            returncode, output = self._bash_streaming(command, timeout_seconds, cwd, capture)
        except _StreamCallbackError as failed:
            # Contract (see test_bash_background_pipe_hang): an exception
            # raised by the caller-supplied stream callback must PROPAGATE
            # out of Bash (after the process group is killed), not be
            # swallowed into an "Error:" string like internal failures
            # (a launch in a directory that cannot be entered) are.
            raise failed.error from failed.error.__cause__
        except Exception as e:
            logger.debug("Exception caught", exc_info=True)
            return unrun(f"Error: {e}"), None, 0
        finally:
            capture.close()
        if returncode is None:
            return Ran(_format_bash_timeout(
                output, timeout_seconds, max_output_chars, capture.spilled_to, capture.dropped,
            ), "timeout"), None, capture.dropped
        self._note_shell_view(command, returncode, output)
        return Ran(_format_bash_result(
            returncode, output, max_output_chars, capture.spilled_to, capture.dropped,
        ), str(returncode)), output, capture.dropped

    def _poll_until(
        self,
        command: str,
        regex: re.Pattern[str],
        poll_seconds: float,
        deadline: float,
        max_output_chars: int,
        cwd: str | None,
    ) -> Ran:
        """Re-run *command* every *poll_seconds* until *regex* matches its output or *deadline*.

        One tool call however long the wait: the run that matches ends
        the poll, as does the deadline (the last run is killed at it), a
        command that cannot be launched, the stop event, an output too
        long to be searched whole (the capture keeps a head and a tail:
        a match across or inside the gap would be wrong), or
        :attr:`idle_children` saying that every sub-agent job of the
        task has finished with a result unread (a poll for what a child
        was to produce can then only time out).  Each run is a
        foreground run (:meth:`_run_once`), so its output streams and
        counts as a view of the file it shows.

        Args:
            command: The shell command to repeat.
            regex: Ends the poll when it matches a run's combined output.
            poll_seconds: Seconds between two runs, at least 1.
            deadline: The call's deadline, as :func:`time.monotonic` time.
            max_output_chars: The result's output budget.
            cwd: The directory the command runs in.

        Returns:
            The last run's result (the command ended as that run did) and
            a line saying whether the regex matched, after how many runs
            and seconds.
        """
        pause = max(poll_seconds, 1.0)
        stop = self.stop_event or threading.Event()
        started = time.monotonic()
        runs = 0
        idle = ""
        while True:
            runs += 1
            # The time left, to the hundredth: the timeout a killed run reports.
            ran, output, dropped = self._run_once(
                command, round(max(deadline - time.monotonic(), 0.0), 2), max_output_chars, cwd,
            )
            matched = output is not None and not dropped and _searched(
                regex, output, deadline, self.stop_event,
            )
            left = deadline - time.monotonic()
            if matched or output is None or dropped or left <= 0:
                break
            # The pause is waited in slices: the last child finishing during
            # it ends the poll there, before another run of the command.
            wake = time.monotonic() + min(pause, left)
            while not (idle := self._idle_children_note()) and time.monotonic() < wake:
                if tool_interrupt.wait_interruptible(stop, min(wake - time.monotonic(), 0.2)):
                    break
            if idle or stop.is_set() or time.monotonic() >= deadline:
                break
        verdict, why = "did not match", ""
        if dropped:
            verdict = "was not searched"
            why = (
                f": the output is too long to keep whole ({dropped:,} characters dropped); "
                "poll with a command that prints less (grep, tail)"
            )
        elif matched:
            verdict = "matched"
        elif idle:
            why = f"; the poll ended early: {idle}"
        elif time.monotonic() >= deadline:
            why = ": the call's deadline (timeout_seconds) passed"
        return ran._replace(text=(
            f"{ran.text}\n[until {regex.pattern!r} {verdict} after {runs} run(s) and "
            f"{time.monotonic() - started:.1f} s{why}]"
        ))

    def _note_shell_view(self, command: str, returncode: int | None, output: str) -> None:
        """Count a ``cat`` / ``sed -n`` / ``grep -n`` of one file as a read of it for Edit/Write.

        Only a command that succeeded and printed a line of the file
        counts: a ``grep -q``, a ``sed -n`` of a line past the end or
        an ``echo x; true || cat f`` shows the model nothing of it.

        Args:
            command: The shell command that ran.
            returncode: Its exit status (``None`` on timeout).
            output: What it printed.
        """
        if returncode != 0 or not output.strip():
            return
        viewed = shell_view_target(command, self._resolve_cwd())
        if viewed is not None and output_shows_file(output, viewed):
            self._mark_read(viewed)

    def _new_spill(self) -> tuple[str, Path] | None:
        """Name the next file an output over its budget is written to in full.

        The shown path is relative to the directory Bash commands run in
        (:meth:`_resolve_cwd`: the work dir, or the parent repository of
        a vanished worktree, which is not recreated):
        ``tmp/tool-out/<tag>-<n>-bash.txt``, ``tag`` naming this instance
        and ``n`` counting its spills, so the model can ``grep`` it.  A
        sub-agent's is in its private folder, shown by its absolute path
        (:func:`_tools_dir`).

        Returns:
            The shown path and the real one, ``None`` when there is no
            such directory or ``tmp/tool-out`` cannot be made in it.
        """
        root = self._resolve_cwd()
        if root is None:
            return None
        root, relative, shown = _tools_dir(self.private_dir, root, SPILL_DIR)
        try:
            spill_dir = _make_dirs_under(root, relative)
        except OSError:
            return None
        self._spill_count += 1
        name = f"{self._spill_tag}-{self._spill_count}-bash.txt"
        return f"{shown}/{name}", spill_dir / name

    def _run_then_detach(
        self,
        command: str,
        timeout_seconds: float,
        max_output_chars: int,
        cwd: str | None,
        grace_seconds: float = 0,
    ) -> Ran | None:
        """Run *command* as a background job in *cwd*; wait for it, then report or leave it running.

        The ``on_timeout="background"`` path of :meth:`Bash` waits
        *timeout_seconds*; its auto-detach path waits only the
        *grace_seconds* of ``DEFAULT_CONFIG.bash_detach_seconds``.  A
        command that finishes in time reads like a foreground result
        (its output, exit code first, the same truncation and spill, and
        it counts as a view for Edit/Write) and leaves no job or log
        behind; one that does not stays running under its job id instead
        of being killed, and the result gives its output so far and the
        ``bash_job`` wait to make for the rest of the deadline.  A
        tool-panel Stop during the grace kills the command, as it kills
        a foreground shell, and so does a stream callback that raises
        (:meth:`_bash_streaming` does the same); during an
        ``on_timeout="background"`` wait both leave the job running, as
        ``bash_job`` does.

        Returns:
            The result, or ``None`` when no job could be started (no log
            directory) on the auto-detach path: the command then runs in
            the foreground as before.
        """
        job, started = self._start_background_job(command, cwd)
        if job is None:
            return None if grace_seconds else unrun(started)
        wait_seconds = grace_seconds or timeout_seconds
        capture = _Capture(max_output_chars, self._new_spill)
        try:
            finished = self._follow_job(job, wait_seconds, capture)
        except BaseException:
            if grace_seconds:
                # The model asked for a foreground command: the tool panel's
                # Stop kills it, as it kills a foreground shell, and the job
                # the model never saw leaves neither an entry nor a log.  On
                # Windows a process of the killed tree may still hold the log
                # for a moment, which would replace the Stop by PermissionError.
                # The watcher kills; this thread only waits for the shell to be
                # gone, with no check of the Stop it is unwinding from.
                job.kill.set()
                _exits_within(job.process, 5)
                self._jobs.pop(job.job_id, None)
                unlink_waiting_for_holders(job.log_path)
            raise
        finally:
            capture.close()
        output = capture.text()
        # A job the wait left (deadline, stop) may have exited since its last
        # read: it is reported as running, with its log, not as finished with
        # the output that read missed.
        returncode = job.returncode() if finished else None
        if returncode is None:
            remaining = timeout_seconds - grace_seconds if grace_seconds else timeout_seconds
            return Ran(
                f"Still running after {wait_seconds:g}s; left running as background job "
                f"{job.job_id} (pid {job.process.pid}). Log: {job.log_path}\n"
                f'Follow it with bash_job({job.job_id!r}, action="wait", until=<regex>, '
                f"timeout_seconds={remaining:g}).\nOutput so far:\n"
                + _truncate_output(output, max_output_chars, capture.spilled_to, capture.dropped),
                "background", job.job_id,
            )
        # Finished in time, this is a foreground result: the model never saw
        # the job id, so neither the registry entry nor the log lingers, and
        # what the command left running is watched no longer: a later Stop
        # leaves it alone, as it does after a foreground shell.
        self._jobs.pop(job.job_id, None)
        job.released.set()
        unlink_waiting_for_holders(job.log_path)
        self._note_shell_view(command, returncode, output)
        return Ran(_format_bash_result(
            returncode, output, max_output_chars, capture.spilled_to, capture.dropped,
        ), str(returncode))

    def _follow_job(self, job: BackgroundJob, seconds: float, capture: _Capture) -> bool:
        """Wait up to *seconds* for *job* as :meth:`_wait_for_job` does, reading its growing log.

        Every new piece of the log goes to *capture* and, with the key
        store's secrets redacted, to ``stream_callback``, so the tool
        panel shows the output live as it does a foreground command's;
        an end held back as the possible beginning of a secret is shown
        when the wait ends.  The deadline and a stop are
        looked at between pieces, whether the shell is running or has
        exited: a job writing faster than the pieces are taken, or a
        writer the exited shell left on its log, cannot hold the wait
        past them.  The wait ends early on a stop, and a tool-panel
        Stop raises after it.

        Returns:
            Whether the job exited and its log was then read to the end:
            the output is complete.  ``False`` at the deadline or on a
            stop, when the job may have exited since the last read or
            its log may still grow.
        """
        deadline = time.monotonic() + seconds
        # Decoded incrementally: a text-mode read of a log still being
        # written flushes a character's lead bytes as U+FFFD as soon as it
        # catches up with the writer, before the rest of the character lands.
        # Line endings are translated as the foreground path's text-mode
        # pipe does, so a Windows command's CRLF output reads the same.
        decoder = io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True
        )
        redactor = StreamRedactor(self.stream_callback)
        with open(job.log_path, "rb") as log:
            while True:
                exited = job.returncode() is not None
                piece = decoder.decode(log.read(_READ_CHARS))
                if not piece and exited:
                    piece = decoder.decode(b"", final=True)  # a cut-off last character
                if piece:
                    capture.append(piece)
                    redactor.feed(piece)
                elif exited:
                    finished = True
                    break
                remaining = deadline - time.monotonic()
                if _stop_requested(self.stop_event) or remaining <= 0:
                    finished = False
                    break
                if not piece:
                    time.sleep(min(remaining, 0.2))
        redactor.flush()
        tool_interrupt.raise_if_interrupted()
        return finished

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

        Every action returns the same report: the job's status (running
        with its pid and elapsed time, or its exit code), the command,
        the log file path and the last ``tail_lines`` lines of the log.
        The full log stays on disk at the reported path.

        Args:
            job_id: The id returned by ``Bash(background=True)``.
            action: ``"tail"`` (default) reports the current status and
                log tail at once; ``"wait"`` blocks until the job exits or
                ``timeout_seconds`` pass, then reports (a still-running
                status means the wait timed out, not that the job failed);
                ``"kill"`` kills the job's whole process group, then reports.
            timeout_seconds: How long ``"wait"`` blocks at most.  Prefer
                one long wait over many short polls; the wait ends early
                on the task's Stop button, and once every sub-agent job
                of this task has finished with a result you have not
                read (the report then says so: read it with
                ``agent_job('', 'wait_any')``).
            tail_lines: Number of trailing log lines to include.
            max_output_chars: Maximum characters in the report before truncation.
            until: With ``"wait"``, a regular expression: the wait also
                ends as soon as the log matches it (an epoch marker, a
                "Serving on" line), so a long run is one turn to start
                and one to collect.

        Returns:
            The job report, or a string starting with ``Error:`` for an
            unknown job id or action.
        """
        job = self._jobs.get(job_id)
        if job is None:
            known = ", ".join(self._jobs) or "none"
            return f"Error: unknown job id {job_id!r}. Known jobs: {known}."
        note = ""
        if action == "wait":
            pattern = _until_pattern(until) if until else None
            if isinstance(pattern, str):
                return pattern
            ended = self._wait_for_job(job, timeout_seconds, pattern)
            if ended.startswith("Error:"):
                return ended
            if ended:
                note = f"\n[wait ended early: {ended}]"
        elif action == "kill":
            # The watcher signals the group, the only thread to; 5 s is its own
            # wait for a killed shell to be gone (:func:`_own_child`).
            job.kill.set()
            with suppress(TimeoutError):
                tool_interrupt.result_interruptible(job.ended, 5)
        elif action != "tail":
            return f"Error: action must be one of wait, tail, kill; got {action!r}."
        return _format_job_report(job, tail_lines, max_output_chars) + note

    def checkpoint(self, name: str) -> str:
        """Save the files of the work dir's git working tree as checkpoint *name*.

        The snapshot holds every file git does not ignore, tracked or not,
        and changes nothing: HEAD, the branch, the index and the files
        stay as they are.  Take one of a working version before a risky
        or large change; ``rollback(name)`` returns to it.  Saving a name
        again replaces it.

        Args:
            name: A short label: letters, digits, ``-`` and ``_`` (1 to 40).

        Returns:
            A confirmation naming the snapshot's git tree, or a string
            starting with ``Error:`` (for instance outside a git repository).
        """
        return self._checkpoints.save(name, self._resolve_cwd() or os.getcwd())

    def rollback(self, name: str) -> str:
        """Restore the files of the work dir's git working tree from checkpoint *name*.

        Files changed or deleted since the checkpoint are rewritten from
        it and files created since are deleted; files git ignores are
        left alone.  The state it replaces is saved first as checkpoint
        ``before-rollback``, so a rollback can be undone.

        Args:
            name: The label given to ``checkpoint``.

        Returns:
            The files rewritten and deleted, or a string starting with ``Error:``.
        """
        return self._checkpoints.roll_back(name, self._resolve_cwd() or os.getcwd())

    def close(self) -> None:
        """Release what the run's tools hold in the repository: the refs of its checkpoints.

        For the end of the run.  Without it every checkpoint's ref stays
        in the repository for good and keeps its tree from ``git gc``.
        """
        self._checkpoints.close()

    def _run_one_command_noting_views(
        self,
        command: str,
        cancel: threading.Event,
        work_dir: str | None,
        timeout_seconds: float,
        max_output_chars: int,
    ) -> tuple[int | None, str, float]:
        """Run one ``run_commands_parallel`` command and count a file it showed as read.

        Args:
            command: The shell command.
            cancel: The pool's cancel event.
            work_dir: Directory the command runs in.
            timeout_seconds: Per-command deadline.
            max_output_chars: The report's per-command budget.

        Returns:
            What :func:`_run_one_command` returns.
        """
        result = _run_one_command(
            command, cancel, work_dir, timeout_seconds, max_output_chars, self.private_dir)
        self._note_shell_view(command, result[0], result[1])
        return result

    def run_commands_parallel(
        self,
        commands: str,
        max_workers: int = 0,
        timeout_seconds: float = 1800,
        max_output_chars: int = 5000,
        description: str = "",
    ) -> str:
        """Run several independent shell commands concurrently, with no LLM sub-agents.

        Use this instead of ``run_parallel`` whenever the parallel work is
        just shell commands whose output you read once they finish: test
        splits (one ``pytest`` invocation per split), builds, linters,
        benchmarks, batch conversions.  Each command runs in its own
        thread under the same rules as ``Bash`` (working directory,
        environment, worktree guard, Stop button); output is not streamed
        live, the combined report is returned when the last command ends.

        Args:
            commands: A JSON-encoded list of shell command strings, e.g.
                ``'["uv run pytest tests/a.py", "uv run pytest tests/b.py"]'``.
                Shell substitutions such as ``"$(cat cmds.json)"`` are not
                expanded and are rejected.
            max_workers: Maximum number of commands running at once.  ``0``
                (default) runs all of them at once.
            timeout_seconds: Timeout in seconds applied to EACH command.
            max_output_chars: Maximum characters of each command's output
                kept in the report (head and tail are kept when truncating).
            description: A brief description of the commands, shown as
                the title of this call in the chat.

        Returns:
            A report starting with a tally line
            (``N commands: K succeeded, F failed, T timed out``) followed by
            one section per command, in input order, giving its exit code,
            wall time and (truncated) output.  A report too long for the
            conversation is written whole to ``tmp/tool-out/`` and its
            digest returned instead: the tally, one status line per
            command, the tail of each command that did not succeed, and
            the file's path.  A string starting with ``Error:`` when the
            arguments were invalid.
        """
        del description
        run_one = functools.partial(
            self._run_one_command_noting_views,
            work_dir=self.work_dir,
            timeout_seconds=timeout_seconds,
            max_output_chars=max_output_chars,
        )
        spill = functools.partial(
            spill_report, self._resolve_cwd(), "parallel", private_dir=self.private_dir)
        return run_commands_pool(
            commands, run_one, max_workers, self.stop_event, max_output_chars, spill,
        )

    def _consume_stream(
        self,
        out_queue: "queue.Queue[str | None]",
        capture: _Capture,
        redactor: StreamRedactor,
        deadline: float,
        until_stop: bool = False,
    ) -> bool:
        """Consume streamed lines from *out_queue* until EOF, *deadline* or (opted in) a Stop.

        Runs on the thread that called :meth:`Bash` so that
        ``stream_callback`` executes with the caller's thread-local
        context intact — printers route every event (task attribution,
        per-task bash buffers, recordings, stop events) by thread-local
        ``task_id``, so invoking the callback from an internal reader
        thread would silently detach the output from the task (bash
        output vanishing from the tool panel and unattributed
        "garbage" events reaching every chat client).

        Args:
            out_queue: Queue fed by the reader thread: lines, a longer
                line in pieces of :data:`_READ_CHARS`; ``None`` marks EOF.
            capture: Where the received pieces go.
            redactor: Shows them to ``stream_callback`` with the key
                store's secrets redacted.
            deadline: ``time.monotonic()`` timestamp to stop waiting at.
            until_stop: When true, the wait also ends once the task's
                stop event or the tool panel's Stop fired (see
                :func:`_stop_requested`).  ``_stop_monitor`` alone
                cannot end the wait when the shell has already exited
                but a background child still holds the stdout pipe (no
                group to kill), so without this a stop would block for
                the whole *deadline*.  The drain after a kill leaves it
                off: its short deadline is the bound there.

        The queue is polled in 0.2 s slices rather than one long C-level
        wait, so the loop keeps returning to Python: that is where an
        asynchronously injected exception — the task Stop's
        ``KeyboardInterrupt`` — can land and reach :meth:`Bash`'s
        process-group kill.

        Returns:
            True when the EOF sentinel was received, False on deadline,
            stop or interrupt.
        """
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or (until_stop and _stop_requested(self.stop_event)):
                return False
            remaining = min(remaining, 0.2)
            try:
                line = out_queue.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                return True
            capture.append(line)
            redactor.feed(line)

    def _stream(self, piece: str) -> None:
        """Hand *piece* to ``stream_callback`` (if set).

        Its exception is raised as :class:`_StreamCallbackError`, which
        :meth:`_run_once` tells from an internal failure.
        """
        callback = self.stream_callback
        if callback is None:
            return
        try:
            callback(piece)
        except Exception as e:
            raise _StreamCallbackError(e) from e

    def _bash_streaming(
        self,
        command: str,
        timeout_seconds: float,
        cwd: str | None,
        capture: _Capture | None = None,
    ) -> tuple[int | None, str]:
        """Run *command* in *cwd* through a reader thread, bounded by *timeout_seconds*.

        Single engine behind :meth:`Bash` and :meth:`run_commands_parallel`
        for both the streaming and the non-streaming case
        (``stream_callback`` may be ``None``).  Returns the shell's exit
        code — ``None`` for a genuine timeout — and its combined
        stdout/stderr as *capture* keeps it: all of it, or its head and
        tail once it is too big (a new :class:`_Capture` without a spill
        file when *capture* is ``None``).  The timeout is a deadline on
        the SHELL, not on its descendants:

        * Shell still running at the deadline — genuine timeout.  The
          whole process group is killed and the timeout error returned.
        * Shell already exited but the stdout pipe is still open (a
          background child inherited it — e.g. ``cd d && nohup job &``
          wraps ``job`` in a pipe-holding subshell) — the command itself
          SUCCEEDED, so its output is returned and the lingering
          children are LEFT RUNNING.  Killing the group here used to
          silently SIGKILL deliberately detached background jobs minutes
          after this method returned success-looking output.  The reader
          thread stays behind in discard mode so a child that writes to
          the inherited pipe can never block on a full pipe buffer.

        A ``stop_event`` ends the wait early in both states: a running
        shell is killed and its exit code reported; an exited shell's
        output is returned after the same short EOF grace period.
        """
        if capture is None:
            capture = _Capture()
        done = threading.Event()
        abandoned = threading.Event()
        out_queue: queue.Queue[str | None] = queue.Queue(_QUEUED_PIECES)
        redactor = StreamRedactor(self._stream)
        eof = False
        process: subprocess.Popen | None = None
        # Both the task's Stop button and the tool panel's own Stop
        # button kill the shell: its owner thread watches this
        # instance's ``stop_event`` and the interrupt event of the tool
        # call running on this thread.
        interrupt = tool_interrupt.current_tool_interrupt_event()
        stops = [event for event in (self.stop_event, interrupt) if event is not None]
        try:
            # The shell is spawned INSIDE the kill/cleanup region, by the
            # thread that owns it until it exits (:func:`_spawn_owned`,
            # :func:`_stop_monitor`), and so is the reader started: a
            # stop injected anywhere here, or a ``Thread.start()`` that
            # raises (thread exhaustion), must still end with the shell
            # killed and ``done`` set — otherwise the command runs on
            # unowned and an already-started monitor polls forever.
            # That thread alone signals and reaps the shell: this one
            # asks for the kill with ``done`` and watches without reaping,
            # so no signal goes to a pid a reap has freed for a stranger.
            process, ended = _spawn_owned(
                functools.partial(self._spawn, command, cwd),
                functools.partial(_stop_monitor, stops, done=done),
            )
            threading.Thread(
                target=_drain_stdout, args=(process.stdout, out_queue, abandoned), daemon=True,
            ).start()
            deadline = time.monotonic() + timeout_seconds
            eof = self._consume_stream(out_queue, capture, redactor, deadline, until_stop=True)
            if not eof:
                # A stop is not a timeout: the shell is killed (the
                # monitor may already have done so) and its exit code
                # reported, exactly as when the stop landed on a running
                # shell before this loop learned to observe it.
                stopped = self.stop_event is not None and self.stop_event.is_set()
                interrupted = interrupt is not None and interrupt.is_set()
                still_running = not exited_unreaped(process)
                timed_out = still_running and not stopped and not interrupted
                if still_running:
                    done.set()
                # The tool panel's Stop keeps this whole tail inside the
                # interrupt's 1 s cooperative grace: a longer EOF wait
                # would let the forced injection land in this loop
                # instead of at the raise below.
                eof = self._consume_stream(
                    out_queue,
                    capture,
                    redactor,
                    time.monotonic() + (0.5 if interrupted else 5),
                )
                if not eof:
                    abandoned.set()
            else:
                # EOF only says every writer closed the pipe: a shell that
                # redirects its own stdout away (``exec >log 2>&1; …``) is
                # still running, so wait for it until the caller's
                # deadline — expiry there is a genuine timeout, and a stop
                # ends the wait through the monitor's group kill.  The
                # wait has a short floor: a shell whose output ended at
                # the deadline edge is in its last microseconds of
                # exiting, and a zero wait (one look at it)
                # would report it timed out and SIGKILL the group, with
                # any ``nohup job &`` the command meant to leave behind.
                timed_out = not _exits_within(process, max(0.5, deadline - time.monotonic()))
        except BaseException:
            # The reader thread outlives this frame: in discard mode it
            # closes the pipe itself once the child lets go of it.
            abandoned.set()
            raise
        finally:
            # The owner kills a shell still running; it has exited when this returns.
            done.set()
            if process is not None:
                _exits_within(process, 5)

        # The output has ended: what was held back as the possible beginning
        # of a secret is shown.
        redactor.flush()
        # The tool panel's own Stop: whether the monitor killed the shell
        # (EOF) or the loop above noticed the event first, the shell is
        # dead and its pipe closed or abandoned, so this is the safe
        # point to raise the interrupt the agent loop turns into "User
        # interrupted the tool call.".
        tool_interrupt.raise_if_interrupted()

        if timed_out:
            return None, capture.text()
        return tool_interrupt.result_interruptible(ended), capture.text()

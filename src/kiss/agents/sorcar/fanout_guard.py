# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Argument checks and task classification for sub-agent fan-outs.

Shared by ``run_parallel`` / ``run_agent`` and the daemon:

* ``tasks`` must be a JSON array of non-empty task strings
  (:func:`parse_tasks_json`).  A literal ``"$(cat tasks.json)"`` passed
  as the argument used to be dispatched as one sub-agent task.
* A task counts as a *review* task when its text names reviewing,
  auditing, inspecting, or bug/regression hunting (:func:`is_review_task`).
  This is a lexical heuristic: paraphrases that avoid every listed stem
  escape it, and mentions such as "audit log" trip it.  A review task
  that does not also ask for changes (:func:`is_implementation_task`)
  puts the child on the read-only ``review`` tool profile.  The
  reviewer marker is inherited down the sub-tree and across daemon
  dispatches so a reviewer's helpers get the same reduced toolset.
* A sub-agent of the caller's own model family is the same model with
  less context (:func:`same_family_conflict`): it is refused unless the
  call is a fan-out over disjoint named pieces of work
  (:func:`named_pieces`), the caller's task asks for sub-agents
  (:func:`asks_for_sub_agents`), another family is pinned, or the child
  is a review (which the dispatcher moves to another family).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Sequence

from kiss.core.models.model_families import family_of, other_family_model, same_family

_REVIEW_WORDS = re.compile(
    r"\b(review\w*|audit\w*|critiqu\w*|inspect\w*|regression\w*"
    r"|read[- ]only|bug[- ]?hunt\w*|vulnerabilit\w*|adversarial\w*"
    r"|(find|hunt|check|look|search|scan)\w*\s+(for\s+)?"
    r"(bugs?|defects?|flaws?|issues?|mistakes?))\b",
    re.IGNORECASE,
)


def is_review_task(task: str) -> bool:
    """Return whether *task* reads like a review / audit / bug-hunt task.

    Args:
        task: A sub-agent task description.

    Returns:
        True when the text contains a reviewer stem (``review``,
        ``audit``, ``critique``, ``inspect``, ``regression``,
        ``read-only``, ``bug-hunt``, ``vulnerabilit…``, ``adversarial``)
        or a bug-hunting phrase such as "find bugs" / "check for
        defects" (case-insensitive).
    """
    return _REVIEW_WORDS.search(task) is not None


REVIEW_BUDGET_BASE_USD = 5.0
"""Budget (USD) of a read-only review of an empty diff.  A review spends
a few dollars at most while an inherited half-share would hand it many
times that; the base alone covers most reviews."""
REVIEW_BUDGET_PER_KLINE_USD = 5.0
"""Budget (USD) added per 1,000 changed lines the review has to read."""
GIT_BIG_FILE_BYTES = 512 * 1024 * 1024
"""git's default ``core.bigFileThreshold``: a bigger file is diffed as binary."""


def changed_lines(work_dir: str) -> int:
    """Return the lines added and deleted in *work_dir* against ``HEAD``, untracked files included.

    ``0`` when *work_dir* is not a git repository or git is missing;
    binary files count as none.  An untracked file counts as git's
    numstat would count it once added (see :func:`_new_file_lines`).

    Args:
        work_dir: The directory whose diff a review is to read.
    """
    # Bytes, not text: git prints a name unquoted (with ``-z``, or with
    # ``core.quotePath`` off), and a name need not be UTF-8.
    try:
        numstat = subprocess.run(
            ["git", "diff", "--numstat", "HEAD"], cwd=work_dir, capture_output=True,
            timeout=30, check=False,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "-z", "--others", "--exclude-standard"], cwd=work_dir,
            capture_output=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0
    lines = 0
    for row in numstat.splitlines():
        # ``<added>\t<deleted>\t<path>``; binary files show ``-`` counts.
        columns = row.split(b"\t", 2)
        for count in columns[:2]:
            lines += int(count) if count.isdigit() else 0
    root = os.fsencode(work_dir)
    for name in untracked.split(b"\0"):
        if name:
            lines += _new_file_lines(os.path.join(root, name))
    return lines


def _new_file_lines(path: bytes) -> int:
    """Return the lines ``git diff --numstat`` counts for the new file *path*.

    A symlink is one line (its target, not followed); a binary file (a
    NUL byte in its first 8,000 bytes, git's own test, or bigger than
    :data:`GIT_BIG_FILE_BYTES`) and a file that cannot be read count
    none.

    Args:
        path: The file's path.
    """
    if os.path.islink(path):
        return 1
    try:
        with open(path, "rb") as handle:
            if os.fstat(handle.fileno()).st_size > GIT_BIG_FILE_BYTES or b"\0" in handle.read(8000):
                return 0
            handle.seek(0)
            return sum(1 for _ in handle)
    except OSError:
        return 0


def review_budget(work_dir: str) -> float:
    """Return the default budget (USD) of a read-only review dispatched from *work_dir*.

    :data:`REVIEW_BUDGET_BASE_USD` plus :data:`REVIEW_BUDGET_PER_KLINE_USD`
    per thousand changed lines (:func:`changed_lines`): a review's
    cost follows the size of the diff it reads, not the caller's
    remaining budget.

    Args:
        work_dir: The directory the review runs in.
    """
    return REVIEW_BUDGET_BASE_USD + REVIEW_BUDGET_PER_KLINE_USD * changed_lines(work_dir) / 1000


_IMPLEMENTATION_WORDS = re.compile(
    r"\b(implement\w*|fix\w*|patch\w*|refactor\w*|create|add|write|modify|edit|update"
    r"|delete|remove|rename|migrate|install|train\w*|build|generate|rewrite)\b",
    re.IGNORECASE,
)


def is_implementation_task(task: str) -> bool:
    """Return whether *task* asks for changes, not just a verdict.

    Used to keep the reduced ``review`` tool profile away from tasks
    that merely mention a review topic ("implement a regression test",
    "patch the vulnerability", "review X and fix what you find"): when
    in doubt the child keeps the full toolset.

    Args:
        task: A sub-agent task description.

    Returns:
        True when the text contains an implementation verb.
    """
    return _IMPLEMENTATION_WORDS.search(task) is not None


_SUB_AGENT_WORDS = re.compile(
    r"sub-?agent|sub-?task|run_parallel|run_agent|parallel agents"
    r"|adversarial (?:testing|training)|second opinion|reviewer",
    re.IGNORECASE,
)


def asks_for_sub_agents(task: str) -> bool:
    """Return whether *task* asks for sub-agents in so many words.

    Lexical: the text mentions sub-agent / subagent, sub-task / subtask,
    ``run_parallel``, ``run_agent``, "parallel agents", "adversarial
    testing", "adversarial training", "second opinion" or "reviewer"
    (case-insensitive, plurals included).  A caller whose own task asks
    for them may dispatch children of its own model family
    (:func:`same_family_conflict`).

    Args:
        task: The calling agent's task text.
    """
    return _SUB_AGENT_WORDS.search(task) is not None


_PIECE = re.compile(
    r"`([^`\n]+)`"  # a backticked command or name
    r"|(https?://[^\s`'\"<>()\[\]]+)"  # a URL
    # A slash path: ``src/x/y.py``, ``tests/``, ``./tmp/x``, ``/etc/hosts``, ``~/.kiss``.
    r"|(?<![\w./-])((?:\.{1,2}|~)?/[\w.-]+(?:/[\w.-]*)*|[\w.-]+(?:/[\w.-]*)+)"
    # A dotted module or file name whose components start with a letter.
    r"|(?<![\w./-])([A-Za-z_][\w-]*(?:\.[A-Za-z_][\w-]*)+)\b"
)
_PROSE_DOTTED = frozenset({"e.g", "i.e", "a.m", "p.m"})


def named_pieces(task: str) -> set[str]:
    """Return the pieces of work *task* names: paths, directories, modules, URLs, commands.

    A regular-expression heuristic, not a parser.  A piece is a slash
    path (``src/x/y.py``, ``./tmp/x``, ``/etc/hosts``; an unprefixed
    ``a/b`` with no dot and no trailing slash is prose such as
    ``and/or`` or ``I/O``), a directory ending in ``/``, a dotted module
    or file name whose components start with a letter
    (``kiss.agents.sorcar``, ``README.md``; ``3.12`` and ``gpt-4.1`` are
    not pieces, nor are ``e.g.`` / ``i.e.``), a URL, or a backticked
    span (the command or name as written, plus the pieces inside it).
    Pieces are compared as written: ``x.py`` and ``src/x.py`` are two
    pieces, ``Foo.py`` and ``foo.py`` too.

    Args:
        task: A sub-agent task description.

    Returns:
        The pieces; empty when the text names none.
    """
    pieces: set[str] = set()
    for match in _PIECE.finditer(task):
        span, url, path, dotted = match.groups()
        if span is not None:
            inner = span.strip()
            if inner:
                pieces.add(inner)
                pieces |= named_pieces(inner)  # no backticks inside: one level deep
        elif url is not None:
            pieces.add(url.rstrip(".,;:"))
        elif path is not None:
            path = path.rstrip(".")
            if (
                "." in path or path.count("/") >= 2 or path.endswith("/")
                or path.startswith(("/", "./", "../", "~/"))
            ):
                pieces.add(path)
        elif dotted not in _PROSE_DOTTED:
            pieces.add(dotted)
    return pieces


def is_scratch_piece(piece: str) -> bool:
    """Whether *piece* is a path under a ``tmp`` directory (``tmp/x``, ``./tmp/x``, ``/w/tmp/x``).

    Scratch files are shared by design (a progress log, instructions
    copied for every child, probe output): naming one is no claim on
    the work.
    """
    parts = piece.strip().lstrip("~").split("/")
    return "tmp" in parts[:-1]


def _under(piece: str, directory: str) -> bool:
    """Whether *piece* lies under *directory* (a piece that is a path prefix of it)."""
    return piece != directory and piece.startswith(directory.rstrip("/") + "/")


def undivided_work(tasks: Sequence[str]) -> str:
    """Return why *tasks* are not a fan-out over disjoint named pieces of work, or ``""``.

    Pieces under ``tmp/`` (:func:`is_scratch_piece`) do not count, nor
    do pieces every task of a fan-out of three or more names (shared
    instructions: a file to read, a command to run; with two tasks
    "every" is "both", and the check would be void).  A directory a
    task names a more specific piece under is context for that piece,
    not a claim on the directory (``src/x/`` with ``src/x/a.py`` in one
    task and with ``src/x/b.py`` in the other is no overlap); a piece
    two tasks claim, and a directory one task claims whole while the
    other claims a piece under it (``src/x/`` with ``src/x/b.py`` or
    ``src/x/nested/``), are overlaps.  Every overlapping pair is
    reported at once, so one refusal shows all there is to divide.

    Args:
        tasks: The task texts of one ``run_agent`` / ``run_parallel`` call.

    Returns:
        ``""`` when there are two or more tasks, each names a piece
        (:func:`named_pieces`) and no counted piece is named by two of
        them; else the shortfall, as a clause for the refusal.
    """
    if len(tasks) < 2:
        return "one sub-task is no fan-out over disjoint work"
    pieces = [named_pieces(task) for task in tasks]
    common = set.intersection(*pieces) if len(pieces) >= 3 else set()
    counted = [{p for p in found if p not in common and not is_scratch_piece(p)}
               for found in pieces]
    for i, found in enumerate(counted):
        if not found:
            return (
                f"task {i + 1} of the fan-out names no file, directory, module, URL or "
                "`command` of its own"
                + (" (pieces every task names, and tmp/ paths, do not count)" if pieces[i]
                   else "")
            )
    claimed = [{p for p in found if not any(_under(q, p) for q in found)} for found in counted]
    overlaps = []
    for i, first in enumerate(claimed):
        for j in range(i + 1, len(claimed)):
            shared = sorted({
                a if a == b or _under(b, a) else b
                for a in first for b in claimed[j]
                if a == b or _under(a, b) or _under(b, a)
            })
            if shared:
                overlaps.append(
                    f"tasks {i + 1} and {j + 1} of the fan-out both name {', '.join(shared[:3])}"
                    + (f" (+{len(shared) - 3} more)" if len(shared) > 3 else "")
                )
    return "; ".join(overlaps)


def same_family_conflict(
    parent_model: str, child_model: str, tasks: Sequence[str], parent_task: str, review: bool,
) -> str:
    """Return why a sub-task on the caller's own model family may not start, or ``""``.

    Inside a task a sub-agent of the caller's family
    (:func:`~kiss.core.models.model_families.same_family`) is the same
    model with less context: a fan-out to it costs more and succeeds less
    often than the caller doing the work itself.
    The child may start when (i) the call is a fan-out of two or more
    tasks over disjoint named pieces of work (:func:`undivided_work`),
    (ii) the caller's own task asks for sub-agents
    (:func:`asks_for_sub_agents`), (iii) *child_model* is of another
    family (the call pinned it), or (iv) the child is a *review*, which
    the dispatcher moves to another family instead.  Nothing is refused
    when either model is unknown (empty).

    Args:
        parent_model: The calling agent's model.
        child_model: The model the sub-task would run: the call's pin,
            the SEA's ``model`` setting, or the inherited one.
        tasks: Every task of the call (one for ``run_agent``, all of a
            ``run_parallel``).
        parent_task: The calling agent's current task text.
        review: Whether the sub-task is a review.

    Returns:
        ``""`` when the dispatch may proceed, else the reason, ending
        with the three ways out.
    """
    if review or not parent_model or not child_model:
        return ""
    if not same_family(parent_model, child_model) or asks_for_sub_agents(parent_task):
        return ""
    shortfall = undivided_work(tasks)
    if not shortfall:
        return ""
    other = other_family_model(parent_model)
    pin = f'model="{other}"' if other else 'model="<a model of another family>"'
    return (
        f"a sub-agent on {child_model} is this task's own model family "
        f"({family_of(parent_model)}), the same model with less context, and {shortfall}. "
        "Ways out: do the work inline in this task; or name the disjoint files/commands in "
        f"each task of a run_parallel; or pass {pin}."
    )


# A ``\\`` pair, or a lone backslash before a character that is not a JSON
# escape (``\|``, ``\(``, ``\.`` in grep/sed patterns).  Replacing every
# match with a pair leaves valid pairs alone and repairs the lone ones.
_LONE_BACKSLASH = re.compile(r'\\\\|\\(?![/"bfnrtu])')


def _decode_json_leniently(text: str) -> object:
    """Decode *text* as JSON, repairing two mistakes models keep making.

    Shell commands inside JSON strings carry ``\\|`` / ``\\(`` escapes
    that are invalid JSON, and heredocs carry raw newlines.  Strict
    decoding failed 21 reviewer calls in one day, each a wasted step, so
    the raw control characters are accepted (``strict=False``) and, when
    that still fails, lone backslashes are doubled before a second try.

    Args:
        text: The raw JSON text.

    Returns:
        The decoded value.

    Raises:
        ValueError: If the text is not JSON even after the repair.
    """
    try:
        return json.loads(text, strict=False)
    except ValueError:
        return json.loads(_LONE_BACKSLASH.sub(r"\\\\", text), strict=False)


def parse_tasks_json(tasks: str, name: str = "tasks") -> list[str]:
    """Parse a JSON-array-of-strings tool argument strictly.

    Used for the ``tasks`` argument of ``run_parallel`` and the
    ``commands`` argument of ``run_commands_parallel``.

    Args:
        tasks: The raw string the model passed.
        name: The argument's name, used in the error messages.

    Returns:
        The decoded, non-empty list of strings.

    Raises:
        ValueError: If *tasks* is not valid JSON, is not an array, is
            empty, or holds anything but non-empty strings.  The message
            tells the model how to fix the call, including the case of a
            shell substitution such as ``$(cat file)`` that was never
            expanded.
    """
    stripped = tasks.strip()
    try:
        parsed = _decode_json_leniently(stripped)
    except ValueError:
        hint = (
            " Inside a JSON string every backslash must be doubled (write "
            '\\\\| for grep\'s \\|) and a newline written as \\n.'
        )
        if stripped.startswith("$(") or stripped.startswith("`"):
            hint = (
                " Shell substitutions are not expanded in tool arguments; "
                "Read the file first and paste its JSON array here."
            )
        raise ValueError(
            f"{name} must be a JSON array of strings, got {stripped[:80]!r}."
            + hint
        ) from None
    if not isinstance(parsed, list):
        raise ValueError(
            f"{name} must be a JSON array of strings, got a JSON "
            f"{type(parsed).__name__}."
        )
    if not parsed:
        raise ValueError(f"{name} is an empty array; nothing to run.")
    if not all(isinstance(t, str) and t.strip() for t in parsed):
        raise ValueError(
            f"{name} must be a JSON array of non-empty strings."
        )
    return parsed

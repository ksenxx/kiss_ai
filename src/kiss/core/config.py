# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings."""

import json
import math
import os
import random
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from kiss.core.brand import HOME_DIR
from kiss.core.context_compaction import (
    COMPACTION_START_TOKENS,
    COMPACTION_STEP_TOKENS,
    COMPACTION_TARGET_FRACTION,
)
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.locked import locked_call

_PROJECT_DIR = Path(__file__).resolve().parents[3]
_ARTIFACTS_DIR_NAME = ".kiss.artifacts"


def _env_flag(name: str, default: bool) -> bool:
    """Read a boolean toggle from the environment.

    Args:
        name: Environment variable name.
        default: Value when the variable is unset or empty.

    Returns:
        ``False`` for ``0``, ``false``, ``no``, ``off`` (case-insensitive),
        ``True`` for any other non-empty value, *default* otherwise.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")


def env_float(name: str, default: float) -> float:
    """Read a finite float from the environment, falling back to *default* on junk.

    Args:
        name: The environment variable to read.
        default: Returned when the variable is unset, empty, not a
            number, or NaN/infinite.
    """
    raw = os.environ.get(name, "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def _env_int(name: str, default: int) -> int:
    """Read an int from the environment, falling back to *default* on junk."""
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


DEFAULT_MAX_BUDGET = 100.0
"""The product default spend cap, in USD, for one agent run.

This is the single source of truth for that number.  It backs both
:attr:`Config.max_budget` (the default for channel-agent command-line
runs) and ``user_config.DEFAULTS['max_budget']`` (what the settings
panel shows and what daemon-launched tasks read), which used to carry
two different literals — 200.0 here and 100 there — so a fresh install
disagreed with itself depending on which entry point the user reached
first.
"""

_artifact_dir: str | None = None
# Reentrant for ``locked_call``: the first ``get_artifact_dir`` of a
# daemon runs on a task thread (the trajectory save in ``run``'s
# ``finally``), where a Stop injection landing right after a plain
# ``with``'s ``__enter__`` would leak the lock for every later task.
_artifact_dir_lock = threading.RLock()


def _generate_artifact_dir() -> str:
    """Create an artifact job directory of this process's own under the project root.

    The directory is made exclusively: a name another process took in
    the same second (the same random number) is drawn again, so two
    processes never save their trajectories into one directory.

    Returns:
        The absolute path to the newly created artifact directory.
    """
    while True:
        name = f"{time.strftime('job_%Y_%m_%d_%H_%M_%S')}_{random.randint(0, 1000000)}"
        artifact_path = get_jobs_root() / name
        try:
            artifact_path.mkdir(parents=True)
        except FileExistsError:
            continue
        return str(artifact_path)


def get_jobs_root() -> Path:
    """Return the directory that contains all per-job artifact subdirectories.

    Job artifacts (including saved trajectories) live under
    ``<project root>/.kiss.artifacts/jobs/job_*``.  This returns the parent
    ``jobs`` directory, which is what the trajectory visualizer treats as
    its artifact directory.

    Returns:
        The absolute path to the ``jobs`` directory.
    """
    return _PROJECT_DIR / _ARTIFACTS_DIR_NAME / "jobs"


def kiss_home() -> Path:
    """Return the KISS home directory ($KISS_HOME or the brand's default).

    The default is ``~/<home_dir>`` with ``home_dir`` from ``brand.json``
    (``~/.kiss`` for stock KISS Sorcar; a white-label brand names its own
    directory so it never shares state with a stock install).  Resolved
    lazily on every call so that ``KISS_HOME`` set after module import
    (as the test suite's conftest does) is honored.
    """
    env = os.environ.get("KISS_HOME")
    return Path(env) if env else Path.home() / HOME_DIR


def remote_url_file() -> Path:
    """Return the file the kiss-web daemon publishes its URLs in: ``$KISS_HOME/remote-url.json``."""
    return kiss_home() / "remote-url.json"


def read_remote_urls(url_file: Path) -> dict[str, Any]:
    """Return the URLs the kiss-web daemon published in *url_file* (:func:`remote_url_file`).

    The daemon writes one JSON object there: ``local`` and, when it has
    them, ``tunnel``, ``loopback``, ``lan`` (a list) and ``localCa``.

    Args:
        url_file: The file to read.

    Returns:
        That object, or ``{}`` when the file is missing, unreadable or
        not a JSON object.
    """
    try:
        data = json.loads(url_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return data if isinstance(data, dict) else {}


ADOPTION_LOCK_TIMEOUT = 30.0
"""Seconds :func:`adopt_legacy_file` waits for a concurrent adopter before giving up."""


def adopt_legacy_file(
    path: Path,
    legacy_name: str,
    suffixes: tuple[str, ...] = ("",),
    stale: Callable[[Path], bool] | None = None,
) -> None:
    """Give a file created under its former name its current name.

    When *path* does not exist but ``path.parent / legacy_name`` does, the
    legacy file is renamed to *path*, so an install upgraded across a
    rename keeps its data.  *suffixes* lists the sidecar files renamed
    along with it, in order (``("-wal", "-shm", "")`` for an SQLite
    database: the sidecars first, the main file last, so that nobody
    opens the renamed database before its uncheckpointed pages are
    beside it).  The old name is left behind as a symlink to the new
    one, so a process of the previous version that is still running
    (the web app before its restart, say) keeps writing to the same
    file -- SQLite resolves the link, so both versions share one WAL --
    instead of recreating an empty file under the old name.  Such a
    process opens the old name at any moment (a connection per
    operation), so the old name must stay usable throughout: the files
    are hard-linked to the new name, the old name is swapped for the
    symlink in one rename, and only then are the old sidecar names
    dropped; a plain rename would part the main file from its WAL for
    a moment (``disk I/O error``), then leave no file at all (which a
    writer recreates empty, and the symlink then cannot be made).  A
    database's WAL and SHM are kept in place meanwhile by a connection
    of this process (:func:`_wal_held_open`): that process also closes
    a connection at any moment, and the last close of a WAL database
    checkpoints it and deletes the sidecars, so a close between two of
    the links would have the next connection start a fresh WAL under
    the old name -- the new name then ended up with the old WAL beside
    the fresh SHM (``database disk image is malformed``), and the rows
    of the fresh WAL were dropped with the old sidecar names.  A
    database that connection cannot read is not renamed at all (see
    below).  Not on Windows: SQLite there names the WAL after the path
    it was given, so a link would have the two versions write two WALs
    over one database file; the old version is restarted anyway, so
    nothing is left behind and the files are simply renamed.
    Concurrent adopters serialise on a lock file next to *path*, and the
    one that arrives second finds nothing left to do.  The lock is
    skipped only once the adoption is visibly done -- the legacy name is
    a symlink to *path* itself -- so the lookups every task start makes
    cost no lock cycle; a legacy name that is missing, or a dangling
    link, is no legacy file at all and takes no lock either.

    A name that comes back into use can find its new location already
    taken by a file an even older version left there.  When both names
    exist as distinct files and ``stale(path)`` says the file at *path*
    is such a leftover, that file (and its sidecars) is moved out of the
    way to ``<path>.stale-<UTC time>`` -- kept, never deleted -- before
    the legacy file is renamed over its name.  Without *stale*, or when
    it answers ``False``, two distinct files are left as they are.

    Args:
        path: The file's current location.
        legacy_name: Its former file name, in the same directory.
        suffixes: Suffixes appended to both names for every file to
            rename; ``""`` is the file itself.
        stale: Tells whether an existing file at *path* is a leftover of
            a version older than the one that wrote the legacy file.

    Raises:
        TimeoutError: When another adopter held the lock for longer
            than :data:`ADOPTION_LOCK_TIMEOUT` seconds; the holder only
            renames files, so such a wait means it is wedged, and
            waiting on would hang the caller for good.
        sqlite3.Error: When the legacy file is an SQLite database that
            the connection keeping its WAL in place cannot read (another
            process holds it locked, say).  Nothing has been touched: the
            files keep their names, so the next call retries, and the
            caller must not create a file under *path* meanwhile.
    """
    legacy = path.with_name(legacy_name)
    if not legacy.exists():
        return
    try:
        # The state a finished adoption leaves behind for the life of
        # the install: nothing to do, so no lock cycle for every task
        # start or prompt build that looks the file up.
        if legacy.is_symlink() and legacy.samefile(path):
            return
    except OSError:  # pragma: no cover — *path* vanished meanwhile; decide under the lock
        pass
    # Everything else is decided under the lock: a file another adopter
    # is moving aside right now looks unreadable, and reading that as
    # "nothing to do" would open (and create) *path* while it is absent.
    with exclusive_file_lock(
        path.with_name(path.name + ".rename.lock"),
        timeout=ADOPTION_LOCK_TIMEOUT,
    ):
        if not _adoption_pending(path, legacy, stale):
            return
        held = _wal_held_open(legacy) if "-wal" in suffixes and os.name != "nt" else None
        try:
            if path.exists() and not _hard_linked(path, legacy):
                stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
                set_aside = path.with_name(f"{path.name}.stale-{stamp}")
                for suffix in suffixes:
                    source = path.with_name(path.name + suffix)
                    if source.exists():
                        os.replace(source, set_aside.with_name(set_aside.name + suffix))
            for suffix in suffixes:
                source = legacy.with_name(legacy.name + suffix)
                target = path.with_name(path.name + suffix)
                if os.name == "nt":
                    if source.exists():
                        os.replace(source, target)
                elif source.exists() and not _hard_linked(source, target):
                    _link_over(source, target)
            if os.name == "nt":
                return
            link = legacy.with_name(legacy.name + ".link")
            link.unlink(missing_ok=True)
            os.symlink(path.name, link)
            os.replace(link, legacy)
            for suffix in suffixes:
                if suffix:
                    legacy.with_name(legacy.name + suffix).unlink(missing_ok=True)
        finally:
            if held is not None:
                held.close()


def _wal_held_open(database: Path) -> sqlite3.Connection | None:
    """Open a connection to *database* with its WAL (and SHM) open, to keep them in place.

    SQLite checkpoints a WAL database and deletes its sidecars when the
    last connection closes, and recreates them when the next one opens;
    while this connection is open no other close is the last, and every
    connection opened meanwhile, under either of the database's names,
    joins the one WAL.  The WAL is opened by a read (a connection alone
    opens nothing).  The connection is read-only, so it writes nothing
    and deletes nothing itself, and only a file with SQLite's header is
    opened at all: a file that is no database (or an empty one) has no
    WAL to keep, and SQLite deletes a ``-wal`` found beside a file with
    no pages.

    Returns:
        The connection, to be closed once the names are in place, or
        ``None`` when *database* is no SQLite database.

    Raises:
        sqlite3.Error: When *database* has SQLite's header but cannot be
            read: a writer in ``locking_mode=EXCLUSIVE`` keeps it locked
            beyond the busy timeout, say.  Such a database may well have
            a WAL to keep, so it is not one to rename unkept.
    """
    try:
        with database.open("rb") as file:
            if file.read(16) != b"SQLite format 3\x00":
                return None
    except OSError:
        return None
    connection = sqlite3.connect(database.absolute().as_uri() + "?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA schema_version").fetchone()
    except sqlite3.Error:
        connection.close()
        raise
    return connection


def _hard_linked(path: Path, other: Path) -> bool:
    """Tell whether *path* and *other* are two names of one file, neither a symlink."""
    try:
        return not path.is_symlink() and not other.is_symlink() and path.samefile(other)
    except OSError:
        return False


def _link_over(source: Path, target: Path) -> None:
    """Make *target* a hard link of *source*, replacing whatever *target* was."""
    staged = target.with_name(target.name + ".link")
    staged.unlink(missing_ok=True)
    os.link(source, staged)
    os.replace(staged, target)


def _adoption_pending(path: Path, legacy: Path, stale: Callable[[Path], bool] | None) -> bool:
    """Tell whether :func:`adopt_legacy_file` has a rename left to do.

    True when only the legacy file exists, when both names are hard
    links of one file (an adoption that stopped before it left the
    symlink behind), or when both exist as distinct files (neither a
    link to the other) and *stale* judges the one at *path* to be the
    leftover of an older version.
    """
    if not legacy.exists():
        return False
    if not path.exists() or _hard_linked(path, legacy):
        return True
    if stale is None:
        return False
    try:
        same = path.samefile(legacy)
    except OSError:  # pragma: no cover — one of the two vanished meanwhile
        return False
    return not same and stale(path)


def is_pre_2026_04_db(path: Path) -> bool:
    """Tell whether the task database at *path* was last written before 2026-04-24.

    The database was called ``history.db`` until 2026-04-24, ``sorcar.db``
    from then until version 2026.10.2, and ``history.db`` again since; an
    install from the spring of 2026 therefore holds both names, and the
    leftover under the current name is told apart by its schema:
    ``task_history.id`` is an ``INTEGER``, which no version since the UUID
    migration of 2026-06-28 writes.  The deploy helper
    ``src/kiss/scripts/legacy_task_db.py`` carries the same probe because
    it is piped standalone into a remote ``python3 -`` and cannot import
    this package.

    Args:
        path: The database file to inspect; it is opened read-only.

    Returns:
        True for the pre-rename schema; False for the current schema, for
        a file without a ``task_history`` table, and for one SQLite cannot
        read (left as it is: nothing is set aside on a guess).
    """
    try:
        conn = sqlite3.connect(f"{Path(os.path.abspath(path)).as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:  # pragma: no cover — unreadable file
        return False
    try:
        cols = {
            r[1]: (r[2] or "").upper()
            for r in conn.execute("PRAGMA table_info(task_history)").fetchall()
        }
    except sqlite3.Error:
        return False
    finally:
        conn.close()
    return cols.get("id") == "INTEGER"


def agents_md_path() -> Path:
    """Return the path of the user's standing instructions, ``$KISS_HOME/AGENTS.md``.

    The file was called ``SORCAR.md`` before version 2026.10.2; one left
    under that name is renamed on the first call.
    """
    path = kiss_home() / "AGENTS.md"
    adopt_legacy_file(path, "SORCAR.md")
    return path


def get_artifact_dir() -> str:
    """Return this process's artifact directory, creating it lazily if needed.

    The directory is chosen once and never changes for the lifetime of
    the process.  It used to be replaceable at runtime, but
    ``Base.get_trajectory_path`` resolves it at *save* time rather than
    at run start, so swapping it mid-flight sent a running agent's
    trajectory to a different root than the one it started under — a
    hazard no lock can close, because it straddles the agent's lifetime
    rather than the assignment.
    """
    if _artifact_dir is None:
        locked_call(_artifact_dir_lock, _init_artifact_dir)
    assert _artifact_dir is not None
    return _artifact_dir


def _init_artifact_dir() -> None:
    """Choose the artifact directory once; the caller holds ``_artifact_dir_lock``."""
    global _artifact_dir
    if _artifact_dir is None:
        _artifact_dir = _generate_artifact_dir()


class Config(BaseModel):
    GEMINI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Gemini API key (can also be set via GEMINI_API_KEY env var)",
    )
    OPENAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key (can also be set via OPENAI_API_KEY env var)",
    )
    ANTHROPIC_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""),
        description="Anthropic API key (can also be set via ANTHROPIC_API_KEY env var)",
    )
    ANTHROPIC_WORKSPACE_ID: str = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_WORKSPACE_ID", ""),
        description=(
            "Anthropic workspace id (wrkspc_...) sent as the "
            "anthropic-workspace-id request header; the Anthropic API requires "
            "it when ANTHROPIC_API_KEY is an identity-linked key "
            "(can also be set via ANTHROPIC_WORKSPACE_ID env var)"
        ),
    )
    TOGETHER_API_KEY: str = Field(
        default_factory=lambda: os.getenv("TOGETHER_API_KEY", ""),
        description="Together API key (can also be set via TOGETHER_API_KEY env var)",
    )
    OPENROUTER_API_KEY: str = Field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
        description="OpenRouter API key (can also be set via OPENROUTER_API_KEY env var)",
    )
    ZAI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("ZAI_API_KEY", ""),
        description="Z.AI (Zhipu/GLM) API key (can also be set via ZAI_API_KEY env var)",
    )
    MOONSHOT_API_KEY: str = Field(
        default_factory=lambda: os.getenv("MOONSHOT_API_KEY", ""),
        description="Moonshot AI (Kimi) API key (can also be set via MOONSHOT_API_KEY env var)",
    )
    # Amazon Bedrock (``bedrock/<id>`` Claude models): a bearer token OR
    # an access key pair; the region selects the Bedrock endpoint.
    AWS_BEARER_TOKEN_BEDROCK: str = Field(
        default_factory=lambda: os.getenv("AWS_BEARER_TOKEN_BEDROCK", ""),
        description=(
            "Amazon Bedrock API key (bearer token) for bedrock/ Claude models "
            "(can also be set via AWS_BEARER_TOKEN_BEDROCK env var)"
        ),
    )
    AWS_ACCESS_KEY_ID: str = Field(
        default_factory=lambda: os.getenv("AWS_ACCESS_KEY_ID", ""),
        description=(
            "AWS access key id for Bedrock SigV4 signing, used when no bearer "
            "token is set (can also be set via AWS_ACCESS_KEY_ID env var)"
        ),
    )
    AWS_SECRET_ACCESS_KEY: str = Field(
        default_factory=lambda: os.getenv("AWS_SECRET_ACCESS_KEY", ""),
        description=(
            "AWS secret access key for Bedrock SigV4 signing "
            "(can also be set via AWS_SECRET_ACCESS_KEY env var)"
        ),
    )
    AWS_SESSION_TOKEN: str = Field(
        default_factory=lambda: os.getenv("AWS_SESSION_TOKEN", ""),
        description=(
            "AWS session token for temporary Bedrock credentials "
            "(can also be set via AWS_SESSION_TOKEN env var)"
        ),
    )
    AWS_REGION: str = Field(
        default_factory=lambda: os.getenv("AWS_REGION", ""),
        description=(
            "AWS region of the Bedrock endpoint; us-east-1 when unset "
            "(can also be set via AWS_REGION env var)"
        ),
    )
    # Claude Code (``cc/<model>``): the ``claude`` CLI bills the user's
    # Claude subscription when it finds no API key in its environment.
    CLAUDE_CODE_OAUTH_TOKEN: str = Field(
        default_factory=lambda: os.getenv("CLAUDE_CODE_OAUTH_TOKEN", ""),
        description=(
            "Long-lived Claude subscription token from `claude setup-token`, "
            "for cc/ models on a machine without a `claude login` "
            "(can also be set via CLAUDE_CODE_OAUTH_TOKEN env var)"
        ),
    )
    claude_code_use_subscription: bool = Field(
        default_factory=lambda: _env_flag("KISS_CLAUDE_CODE_USE_SUBSCRIPTION", True),
        description=(
            "cc/ models bill the Claude subscription the `claude` CLI is logged "
            "in to (or CLAUDE_CODE_OAUTH_TOKEN): ANTHROPIC_API_KEY and "
            "ANTHROPIC_AUTH_TOKEN are kept out of the CLI's environment and a "
            "turn costs $0.  Off, the CLI inherits the environment, an API key "
            "bills tokens, and a turn is priced at the resolved Claude model's "
            "catalog rate (KISS_CLAUDE_CODE_USE_SUBSCRIPTION=0; the settings "
            "panel's Claude Code switch overrides it)."
        ),
    )
    # Codex (``codex/<model>``): the ``codex`` CLI bills the user's ChatGPT
    # subscription when it finds no API key in its environment.
    CODEX_ACCESS_TOKEN: str = Field(
        default_factory=lambda: os.getenv("CODEX_ACCESS_TOKEN", ""),
        description=(
            "Codex access token (ChatGPT Business/Enterprise, from "
            "chatgpt.com/admin/access-tokens) for codex/ models on a machine "
            "without a `codex login` (can also be set via CODEX_ACCESS_TOKEN "
            "env var)"
        ),
    )
    codex_use_subscription: bool = Field(
        default_factory=lambda: _env_flag("KISS_CODEX_USE_SUBSCRIPTION", True),
        description=(
            "codex/ models bill the ChatGPT subscription the `codex` CLI is "
            "logged in to (or CODEX_ACCESS_TOKEN): OPENAI_API_KEY and "
            "CODEX_API_KEY are kept out of the CLI's environment and a turn "
            "costs $0.  Off, the CLI inherits the environment, an API key bills "
            "tokens, and a turn is priced at the named OpenAI model's catalog "
            "rate (KISS_CODEX_USE_SUBSCRIPTION=0; the settings panel's Codex "
            "switch overrides it)."
        ),
    )
    # Token-cost levers (projects/cost-levers-implementation-plan.md).
    # Each is a plain toggle read from a ``KISS_*`` environment
    # variable so a regression is a flag flip, not a revert.
    read_dedupe: bool = Field(
        default_factory=lambda: _env_flag("KISS_READ_DEDUPE", True),
        description=(
            "Read tool returns a one-line 'unchanged since your earlier Read' "
            "stub instead of re-sending a file range that is still in the "
            "model's context (KISS_READ_DEDUPE=0 disables)."
        ),
    )
    read_outline_lines: int = Field(
        default_factory=lambda: _env_int("KISS_READ_OUTLINE_LINES", 2000),
        description=(
            "A Read without start_line of a file longer than this many lines "
            "returns a symbol outline instead of the first 2,000 lines "
            "(0 disables; KISS_READ_OUTLINE_LINES)."
        ),
    )
    tool_output_compaction: bool = Field(
        default_factory=lambda: _env_flag("KISS_TOOL_OUTPUT_COMPACTION", True),
        description=(
            "Replace old, large tool outputs in the model conversation with "
            "short stubs once the context grows past compaction_start_tokens, or "
            "past context_edit_trigger_tokens for a Claude model on the Anthropic API "
            "(KISS_TOOL_OUTPUT_COMPACTION=0 disables).  The global Compaction "
            "setting of the daemon (config.json key ``compaction``, the settings "
            "panel's checkbox) is mirrored here by user_config.apply_config_to_env; "
            "off, no agent of the process compacts, whatever its own "
            "``compaction`` run option says."
        ),
    )
    compaction_start_tokens: int = Field(
        default_factory=lambda: _env_int("KISS_COMPACTION_START_TOKENS", COMPACTION_START_TOKENS),
        description=(
            "Context size (tokens) at which the first tool-output compaction is "
            "considered; later ones follow every compaction_step_tokens "
            "(KISS_COMPACTION_START_TOKENS)."
        ),
    )
    compaction_step_tokens: int = Field(
        default_factory=lambda: _env_int("KISS_COMPACTION_STEP_TOKENS", COMPACTION_STEP_TOKENS),
        description=(
            "Context growth (tokens) after a compaction before the next one is "
            "considered while the context is below the compaction target "
            "(KISS_COMPACTION_STEP_TOKENS)."
        ),
    )
    compaction_target_fraction: float = Field(
        default_factory=lambda: env_float(
            "KISS_COMPACTION_TARGET_FRACTION", COMPACTION_TARGET_FRACTION
        ),
        description=(
            "Share of the model's context window past which every step considers a "
            "compaction of old tool outputs and arguments, so the context stays near "
            "that size instead of growing to the hand-off limit (0 disables the "
            "target; KISS_COMPACTION_TARGET_FRACTION)."
        ),
    )
    reasoning_keep_turns: int = Field(
        default_factory=lambda: _env_int("KISS_REASONING_KEEP_TURNS", 0),
        description=(
            "How many of the most recent reasoning turns a provider that retains the "
            "model's reasoning across tool calls keeps in the context (Anthropic: the "
            "clear_thinking context-editing strategy; OpenAI GPT-5.6+: reasoning.context "
            "current_turn); 0, the default, keeps every turn's reasoning, as the providers "
            "do by default. Off by default because each clearing rewrites the cached "
            "prefix: in the 2026-11-07 Terminal-Bench A/B (2 turns kept, a turn every 5 "
            "steps) 92% of the calls after a turn note missed the prompt cache and the "
            "spend per attempt nearly doubled (KISS_REASONING_KEEP_TURNS)."
        ),
    )
    reasoning_turn_steps: int = Field(
        default_factory=lambda: _env_int("KISS_REASONING_TURN_STEPS", 5),
        description=(
            "Steps per reasoning turn: every this many steps the agent appends a short "
            "harness note as a user message, which the providers count as a new turn, so "
            "reasoning_keep_turns measures steps (a tool-result-only message continues the "
            "turn in progress); 0 never starts a new turn (KISS_REASONING_TURN_STEPS)."
        ),
    )
    context_edit_trigger_tokens: int = Field(
        default_factory=lambda: _env_int("KISS_CONTEXT_EDIT_TRIGGER_TOKENS", 250_000),
        description=(
            "Context tokens past which the agent clears, in its own conversation, the "
            "inputs and results of a Claude model's tool calls on the Anthropic API, all "
            "but the last context_edit_keep_tool_uses, and its old reasoning; 0 leaves "
            "such a model to the thresholds of the Compaction settings. "
            "A clearing rewrites the cached prefix at the cache-write price, so it must "
            "be rare and bulk.  The value holds for a model whose cache write costs fifty "
            "cache reads (Claude Fable 5.2: replayed on its 19,174 calls of 2026-11-11, "
            "250k with context_edit_clear_at_least 50k clears once per 78 calls, 176k "
            "tokens at a time, and 3 sessions reach the hand-off limit, each on over 300k "
            "tokens of results in its newest calls, against 23 when reasoning is left out "
            "of the 100k minimum of a 200k trigger).  For another model both values are "
            "multiplied by the square root of "
            "its own ratio over fifty (177k at 25 reads a write, 125k at 12.5), and the "
            "trigger is at most 80% of the hand-off limit.  The agent clears rather than "
            "the provider (its clear_tool_uses edit), whose boundary moved between "
            "requests and rewrote the cache without freeing anything "
            "(KISS_CONTEXT_EDIT_TRIGGER_TOKENS)."
        ),
    )
    context_edit_keep_tool_uses: int = Field(
        default_factory=lambda: _env_int("KISS_CONTEXT_EDIT_KEEP_TOOL_USES", 20),
        description=(
            "Tool calls (inputs and results) that clearing keeps "
            "(KISS_CONTEXT_EDIT_KEEP_TOOL_USES)."
        ),
    )
    context_edit_clear_at_least: int = Field(
        default_factory=lambda: _env_int("KISS_CONTEXT_EDIT_CLEAR_AT_LEAST", 50_000),
        description=(
            "Tokens that clearing must remove to be applied, so each one pays for the "
            "prompt-cache write it causes: the stubbed inputs and results (four "
            "characters a token) and the reasoning that goes with them (the output "
            "tokens of the replies less their visible text).  Scaled to the model's cache "
            "prices like context_edit_trigger_tokens.  0 sets no minimum; clearings are "
            "then ten steps apart (KISS_CONTEXT_EDIT_CLEAR_AT_LEAST)."
        ),
    )
    routine_turn_effort: str = Field(
        default_factory=lambda: os.environ.get("KISS_ROUTINE_TURN_EFFORT", ""),
        description=(
            "Reasoning effort (low, medium, ...) of a routine turn: the one after a "
            "refused tool call, a summary or memory tool, a bash_job / agent_job wait, "
            "or an unchanged Read. Empty (default) keeps the model's effort on every "
            "turn; set it only to measure (KISS_ROUTINE_TURN_EFFORT; cost_report "
            "reports routine calls and their output tokens)."
        ),
    )
    tool_output_max_chars: int = Field(
        default_factory=lambda: _env_int("KISS_TOOL_OUTPUT_MAX_CHARS", 50_000),
        description=(
            "Ceiling on the characters one Bash / Read / memory_list result keeps "
            "in the conversation: the model may raise max_output_chars up to it, "
            "not beyond (KISS_TOOL_OUTPUT_MAX_CHARS)."
        ),
    )
    tool_output_default_chars: int = Field(
        default_factory=lambda: _env_int("KISS_TOOL_OUTPUT_DEFAULT_CHARS", 8_000),
        description=(
            "Default budget of characters one Bash / whole-file Read / memory_list "
            "result keeps in the conversation (head and tail are kept; the rest is "
            "spilled to tmp/tool-out/) (KISS_TOOL_OUTPUT_DEFAULT_CHARS)."
        ),
    )
    bash_detach_seconds: int = Field(
        default_factory=lambda: _env_int("KISS_BASH_DETACH_SECONDS", 120),
        description=(
            "Grace period in seconds after which a foreground Bash call whose "
            "timeout_seconds is longer goes on as a background job instead of "
            "blocking the turn; 0 disables (KISS_BASH_DETACH_SECONDS)."
        ),
    )
    context_limit_fraction: float = Field(
        default_factory=lambda: env_float("KISS_CONTEXT_LIMIT_FRACTION", 0.7),
        description=(
            "Fraction of the model's context window at which an agent session "
            "hands off with a trajectory summary (KISS_CONTEXT_LIMIT_FRACTION)."
        ),
    )
    tool_profiles: bool = Field(
        default_factory=lambda: _env_flag("KISS_TOOL_PROFILES", True),
        description=(
            "Reviewer sub-agents get the read-only 'review' tool profile "
            "instead of the full toolset (KISS_TOOL_PROFILES=0 disables)."
        ),
    )
    chat_history_digest: bool = Field(
        default_factory=lambda: _env_flag("KISS_CHAT_HISTORY_DIGEST", True),
        description=(
            "Chat prompts carry the full result of only the last two prior "
            "tasks and a short digest of older ones "
            "(KISS_CHAT_HISTORY_DIGEST=0 disables)."
        ),
    )
    dispatch_path_rewrite: bool = Field(
        default_factory=lambda: _env_flag("KISS_DISPATCH_PATH_REWRITE", True),
        description=(
            "Sub-agent task text that names the parent repository path is "
            "rewritten to the active worktree path at dispatch "
            "(KISS_DISPATCH_PATH_REWRITE=0 disables)."
        ),
    )
    max_budget: float = Field(
        default=DEFAULT_MAX_BUDGET,
        description=(
            "Maximum budget in USD for a single agent run, defaulting to "
            "DEFAULT_MAX_BUDGET. Only consumed as the "
            "default for channel-agent command-line runs (and kept in sync from "
            "the settings panel; daemon-launched tasks read max_budget from "
            "the config directly); KISSAgent and RelentlessAgent do NOT "
            "consult this field — they use their own defaults (10.0 USD and "
            "kiss.agents.sorcar.relentless_agent.DEFAULT_MAX_BUDGET, 200.0 USD, "
            "respectively; that is a different constant from this module's "
            "DEFAULT_MAX_BUDGET) unless max_budget is passed to run() explicitly."
        ),
    )


DEFAULT_CONFIG = Config()

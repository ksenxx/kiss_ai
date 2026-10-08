# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Configuration Pydantic models for KISS agent settings."""

import math
import os
import random
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, Field

from kiss.core.brand import HOME_DIR
from kiss.core.context_compaction import COMPACTION_START_TOKENS, COMPACTION_STEP_TOKENS
from kiss.core.file_lock import exclusive_file_lock

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


def _env_float(name: str, default: float) -> float:
    """Read a finite float from the environment, falling back to *default* on junk."""
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
runs) and ``vscode_config.DEFAULTS['max_budget']`` (what the settings
panel shows and what daemon-launched tasks read), which used to carry
two different literals — 200.0 here and 100 there — so a fresh install
disagreed with itself depending on which entry point the user reached
first.
"""

_artifact_dir: str | None = None
_artifact_dir_lock = threading.Lock()


def _artifact_root(base_dir: str | Path | None = None) -> Path:
    """Return the root directory for generated KISS artifacts."""
    root = Path(base_dir) if base_dir is not None else _PROJECT_DIR
    return root.resolve() / _ARTIFACTS_DIR_NAME


def _generate_artifact_dir() -> str:
    """Generate a unique artifact job directory under the project root.

    Returns:
        The absolute path to the newly created artifact directory.
    """
    artifact_subdir_name = (
        f"{time.strftime('job_%Y_%m_%d_%H_%M_%S')}_{random.randint(0, 1000000)}"
    )
    artifact_path = _artifact_root() / "jobs" / artifact_subdir_name
    artifact_path.mkdir(parents=True, exist_ok=True)
    return str(artifact_path)


def get_jobs_root(base_dir: str | Path | None = None) -> Path:
    """Return the directory that contains all per-job artifact subdirectories.

    Job artifacts (including saved trajectories) live under
    ``<base>/.kiss.artifacts/jobs/job_*``.  This returns the parent
    ``jobs`` directory, which is what the trajectory visualizer treats as
    its artifact directory.

    Args:
        base_dir: Optional base directory for the ``.kiss.artifacts`` root.
            Defaults to the project root when ``None``.

    Returns:
        The absolute path to the ``jobs`` directory.
    """
    return _artifact_root(base_dir) / "jobs"


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
    instead of recreating an empty file under the old name.  Not on
    Windows: SQLite there names the WAL after the path it was given, so
    a link would have the two versions write two WALs over one database
    file; the old version is restarted anyway, so nothing is left behind.
    Concurrent adopters serialise on a lock file next to *path* (taken
    whenever a file under the legacy name exists, the symlink included),
    and the one that arrives second finds nothing left to do.

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
    """
    legacy = path.with_name(legacy_name)
    if not legacy.exists():
        return
    # Everything else is decided under the lock: a file another adopter
    # is moving aside right now looks unreadable, and reading that as
    # "nothing to do" would open (and create) *path* while it is absent.
    with exclusive_file_lock(path.with_name(path.name + ".rename.lock")):
        if not _adoption_pending(path, legacy, stale):
            return
        if path.exists():
            stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            set_aside = path.with_name(f"{path.name}.stale-{stamp}")
            for suffix in suffixes:
                source = path.with_name(path.name + suffix)
                if source.exists():
                    os.replace(source, set_aside.with_name(set_aside.name + suffix))
        for suffix in suffixes:
            source = legacy.with_name(legacy.name + suffix)
            if source.exists():
                os.replace(source, path.with_name(path.name + suffix))
        if os.name != "nt":
            os.symlink(path.name, legacy)


def _adoption_pending(path: Path, legacy: Path, stale: Callable[[Path], bool] | None) -> bool:
    """Tell whether :func:`adopt_legacy_file` has a rename left to do.

    True when only the legacy file exists, or when both exist as
    distinct files (neither a link to the other) and *stale* judges the
    one at *path* to be the leftover of an older version.
    """
    if not legacy.exists():
        return False
    if not path.exists():
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
    global _artifact_dir
    if _artifact_dir is None:
        with _artifact_dir_lock:
            if _artifact_dir is None:
                _artifact_dir = _generate_artifact_dir()
    return _artifact_dir


class _ArtifactDirProxy:
    def __fspath__(self) -> str:
        return get_artifact_dir()

    def __str__(self) -> str:
        return get_artifact_dir()

    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)

    def __hash__(self) -> int:
        return hash(str(self))


artifact_dir = _ArtifactDirProxy()


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
            "short stubs once the context grows past compaction_start_tokens "
            "(KISS_TOOL_OUTPUT_COMPACTION=0 disables)."
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
            "considered (KISS_COMPACTION_STEP_TOKENS)."
        ),
    )
    tool_output_max_chars: int = Field(
        default_factory=lambda: _env_int("KISS_TOOL_OUTPUT_MAX_CHARS", 50_000),
        description=(
            "Default cap on the characters one Bash result keeps in the "
            "conversation (head and tail are kept when truncating; the model "
            "may pass a different max_output_chars) (KISS_TOOL_OUTPUT_MAX_CHARS)."
        ),
    )
    context_limit_fraction: float = Field(
        default_factory=lambda: _env_float("KISS_CONTEXT_LIMIT_FRACTION", 0.7),
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
            "the VS Code settings; daemon-launched tasks read max_budget from "
            "the VS Code config directly); KISSAgent and RelentlessAgent do NOT "
            "consult this field — they use their own defaults (10.0 USD and "
            "kiss.agents.sorcar.relentless_agent.DEFAULT_MAX_BUDGET, 200.0 USD, "
            "respectively; that is a different constant from this module's "
            "DEFAULT_MAX_BUDGET) unless max_budget is passed to run() explicitly."
        ),
    )


DEFAULT_CONFIG = Config()

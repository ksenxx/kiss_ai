# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Deterministic indexing of a git repository into knowledge blocks.

:func:`index_repo` walks every tracked file and every commit reachable
from any ref and writes one :class:`~kiss.agents.seas.git_knowledge_store.Block`
per file, per chunk of a file, per symbol definition, per commit, per
(commit, file) change with its patch, per tag, per branch, per
contributor and per directory into the repository's
:class:`KnowledgeStore`.  Every run is incremental: files are compared
by blob hash (``git ls-files -s``), commits by set difference between
``git rev-list --all`` and the commits already stored, so new branches,
an unshallowed history and a rewritten history (whose dropped commits
are deleted) are all handled by the same code.  Agent-written ``note``
blocks are never touched here.

The repository named in a prompt is resolved by :func:`resolve_repo`: a
local absolute path (any directory inside a work tree) is indexed in
place; a URL is cloned under ``$KISS_HOME/knowledge/checkouts/<name>``
and reset to the remote's default branch on every run.  Both tiers of a
repository's memory — the curated pages and the block store — live in
the repository's domain memory directory (:func:`memory_location`), the
same directory every Sorcar run inside that repository attaches through
its ``memory_*`` tools.
"""

from __future__ import annotations

import contextlib
import hashlib
import re
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

from kiss.agents.seas.git_knowledge_store import KINDS, Block, KnowledgeStore
from kiss.core.config import kiss_home

GIT_TIMEOUT = 3600
"""Seconds one git command may run (clone and log over a huge history included)."""

CHUNK_LINES = 80
"""Lines per ``chunk`` block."""

CHUNK_CHARS = 6000
"""Characters per ``chunk`` block; a longer line is stored as pieces of this size."""

MAX_LINE_CHARS = 400
"""Longer lines are cut in the *summary* text of ``file`` and ``symbol`` blocks only."""

MAX_CONTENT_BYTES = 64 << 20
"""Text files larger than this get a metadata-only ``file`` block that says so."""

MAX_PATCH_CHARS = 8000
"""Characters of a patch per ``change`` block; longer patches continue in more blocks."""

MAX_RECORD_BYTES = 64 << 20
"""Bytes of one commit's diff kept; a longer diff is cut and the ``commit`` block says so."""

HEAD_LINES = 40
"""Lines of a file quoted in its ``file`` block."""

SYMBOL_CONTEXT_LINES = 12
"""Lines after a definition line quoted in its ``symbol`` block."""

MAX_DIR_ENTRIES = 300
"""Entries listed in a ``dir`` block."""

GENERATED_SUFFIXES = (".min.js", ".min.css", ".map")
"""Derived artifacts whose content carries no fact of its own; metadata only."""

_NUMSTAT_RE = re.compile(r"^(\d+|-)\t(\d+|-)\t(.+)$")
_IMPORT_RE = re.compile(
    r"^\s*(?:import\b|from\s+\S+\s+import\b|require\(|#include\b|use\s+\w|using\s+\w|"
    r"package\s+\w|@import\b)"
)
_ESCAPE_RE = re.compile(rb'\\([abtnvfr"\\]|[0-7]{1,3})')
_QUOTED_PAIR_RE = re.compile(r'^("(?:[^"\\]|\\.)*") ("(?:[^"\\]|\\.)*")$')
_ESCAPES = {b"a": b"\a", b"b": b"\b", b"t": b"\t", b"n": b"\n", b"v": b"\v", b"f": b"\f",
            b"r": b"\r", b'"': b'"', b"\\": b"\\"}
_KEYWORDS = frozenset(
    "if else for while switch return do case break continue goto sizeof typedef".split()
)

LANGUAGES = {
    ".py": "Python", ".pyi": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript", ".tsx": "TypeScript",
    ".go": "Go", ".rs": "Rust", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".scala": "Scala", ".cs": "C#", ".swift": "Swift", ".c": "C", ".h": "C/C++ header",
    ".cc": "C++", ".cpp": "C++", ".cxx": "C++", ".hpp": "C++", ".hh": "C++",
    ".m": "Objective-C", ".mm": "Objective-C++", ".rb": "Ruby", ".php": "PHP",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".lua": "Lua", ".pl": "Perl",
    ".r": "R", ".jl": "Julia", ".ex": "Elixir", ".exs": "Elixir", ".erl": "Erlang",
    ".hs": "Haskell", ".ml": "OCaml", ".clj": "Clojure", ".dart": "Dart", ".sql": "SQL",
    ".md": "Markdown", ".markdown": "Markdown", ".rst": "reStructuredText", ".txt": "Text",
    ".html": "HTML", ".htm": "HTML", ".css": "CSS", ".scss": "SCSS", ".json": "JSON",
    ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML", ".xml": "XML", ".ini": "INI",
    ".cfg": "INI", ".tex": "LaTeX", ".bib": "BibTeX", ".proto": "Protocol Buffers",
    ".ipynb": "Jupyter", ".vue": "Vue", ".svelte": "Svelte",
}
"""File extension to language name."""

SPECIAL_FILES = {"Dockerfile": "Dockerfile", "Makefile": "Makefile", "CMakeLists.txt": "CMake"}

# Definition-line patterns per language family: (regex with a ``name`` group
# and an optional ``kind`` group, default kind).
_PATTERNS: dict[str, list[tuple[re.Pattern[str], str]]] = {
    "python": [(re.compile(r"^\s*(?:async\s+)?(?P<kind>def|class)\s+(?P<name>\w+)"), "")],
    "js": [
        (re.compile(
            r"^\s*(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?(?:async\s+)?"
            r"(?P<kind>function\*?|class|interface|type|enum|namespace)\s+(?P<name>\w+)"
        ), ""),
        (re.compile(
            r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*(?::[^=]+)?=\s*"
            r"(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::\s*[^=]+)?=>|\w+\s*=>)"
        ), "function"),
    ],
    "go": [
        (re.compile(r"^func\s+(?:\([^)]*\)\s*)?(?P<name>\w+)"), "func"),
        (re.compile(r"^type\s+(?P<name>\w+)\s+(?P<kind>struct|interface)"), ""),
    ],
    "rust": [(re.compile(
        r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?"
        r"(?P<kind>fn|struct|enum|trait|impl|mod|type|macro_rules!)\s+(?P<name>\w+)"
    ), "")],
    "jvm": [
        (re.compile(
            r"^\s*(?:(?:public|private|protected|static|final|abstract|sealed|open|data|"
            r"internal|export)\s+)*(?P<kind>class|interface|enum|record|object|trait|struct|"
            r"protocol|extension)\s+(?P<name>\w+)"
        ), ""),
        (re.compile(r"^\s*(?:[\w@]+\s+)*(?P<kind>fun|def|func)\s+(?P<name>\w+)"), ""),
        (re.compile(
            r"^\s+(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?"
            r"[\w<>\[\],.?\s]+?\s+(?P<name>\w+)\s*\("
        ), "method"),
    ],
    "c": [
        (re.compile(
            r"^(?:typedef\s+)?(?P<kind>struct|class|enum|union|namespace)\s+(?P<name>\w+)"
        ), ""),
        (re.compile(
            r"^[A-Za-z_][\w\s*&:<>,]*?\b(?P<name>\w+)\s*\([^;{]*\)\s*(?:const\s*)?\{?\s*$"
        ), "function"),
    ],
    "ruby": [(re.compile(r"^\s*(?P<kind>def|class|module)\s+(?P<name>[\w.?!]+)"), "")],
    "php": [(re.compile(
        r"^\s*(?:(?:public|private|protected|static|abstract|final)\s+)*"
        r"(?P<kind>function|class|interface|trait|enum)\s+(?P<name>\w+)"
    ), "")],
    "shell": [(re.compile(r"^\s*(?:function\s+)?(?P<name>[\w-]+)\s*\(\)\s*\{?"), "function")],
    "lua": [(re.compile(r"^\s*(?:local\s+)?function\s+(?P<name>[\w.:]+)"), "function")],
    "markdown": [(re.compile(r"^#{1,6}\s+(?P<name>.+?)\s*#*\s*$"), "heading")],
}

_FAMILIES = {
    "Python": "python", "JavaScript": "js", "TypeScript": "js", "Vue": "js", "Svelte": "js",
    "Go": "go", "Rust": "rust", "Java": "jvm", "Kotlin": "jvm", "Scala": "jvm", "C#": "jvm",
    "Swift": "jvm", "C": "c", "C/C++ header": "c", "C++": "c", "Objective-C": "c",
    "Objective-C++": "c", "Ruby": "ruby", "PHP": "php", "Shell": "shell", "Lua": "lua",
    "Markdown": "markdown",
}


class KnowledgeError(Exception):
    """A repository cannot be indexed (not a repository, no commits, git failure)."""


@dataclass
class Commit:
    """One commit of the ``git log`` stream, with its numstat and per-file patches."""

    sha: str
    parents: str
    author: str
    email: str
    authored: str
    committed: str
    timestamp: int
    refs: str
    message: str
    numstat: list[tuple[str, str, str]] = field(default_factory=list)
    patches: dict[str, str] = field(default_factory=dict)
    truncated: bool = False

    @property
    def subject(self) -> str:
        """The first line of the message."""
        return self.message.split("\n", 1)[0].strip()


@dataclass
class _Author:
    """Per-contributor aggregate over every ref's history."""

    email: str
    last: str
    first: str = ""
    count: int = 0
    names: set[str] = field(default_factory=set)


@dataclass
class IndexReport:
    """What one :func:`index_repo` run did."""

    repo: str
    slug: str
    memory_dir: str
    store_path: str
    mode: str
    head: str
    previous_head: str
    seconds: float
    counts: dict[str, int]
    files_total: int
    files_indexed: int
    files_removed: int
    changed_paths: list[str]
    commits_indexed: int
    commits_removed: int
    new_commits: list[str]
    languages: dict[str, int]
    top_dirs: list[tuple[str, int]]
    authors: list[str]
    tags: list[str]


def now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_git(repo: Path, *args: str, check: bool = True) -> str:
    """Run one git command in *repo* and return its stdout.

    Raises:
        KnowledgeError: When git fails and *check* is true, or times out.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True,
            errors="backslashreplace", timeout=GIT_TIMEOUT, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise KnowledgeError(f"git {' '.join(args)} failed in {repo}: {exc}") from exc
    if check and result.returncode != 0:
        raise KnowledgeError(
            f"git {' '.join(args)} failed in {repo} (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def is_url(spec: str) -> bool:
    """Return True when *spec* names a remote repository rather than a local path."""
    return "://" in spec or bool(re.match(r"^[\w.-]+@[\w.-]+:", spec))


def checkouts_dir() -> Path:
    """The directory remote repositories are cloned into."""
    return kiss_home() / "knowledge" / "checkouts"


def checkout_dir(url: str) -> Path:
    """The checkout directory of *url*: its repository name, or, when another
    URL already owns a checkout of that name, the name plus a hash of the URL.

    The plain name is preferred because it is also the memory slug, so the
    memory matches the one a user's own clone of the repository attaches.
    """
    name = re.sub(r"\.git$", "", url.rstrip("/").rsplit("/", 1)[-1].rsplit(":", 1)[-1])
    name = re.sub(r"[^\w.-]+", "-", name) or "repo"
    plain = checkouts_dir() / name
    if not (plain / ".git").exists():
        return plain
    origin = run_git(plain, "remote", "get-url", "origin", check=False).strip()
    if origin == url:
        return plain
    return checkouts_dir() / f"{name}-{hashlib.sha1(url.encode()).hexdigest()[:8]}"


def resolve_repo(spec: str) -> Path:
    """Resolve the repository named by *spec* to the root of a local checkout.

    A local absolute path (any directory inside a git work tree) resolves
    to the work tree's top level.  A URL is cloned under
    :func:`checkouts_dir` on first use and, on later calls, fetched and
    hard-reset to the remote's default branch so the checkout mirrors the
    remote.

    Raises:
        KnowledgeError: When *spec* is relative (the tools run in the
            daemon, whose working directory is not the task's), is not a
            work tree, or is not a cloneable URL.
    """
    spec = spec.strip()
    if is_url(spec):
        dest = checkout_dir(spec)
        if not (dest / ".git").exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            run_git(dest.parent, "clone", "--quiet", spec, str(dest))
        else:
            run_git(dest, "fetch", "--quiet", "--prune", "origin")
            run_git(dest, "remote", "set-head", "origin", "--auto")
            run_git(dest, "reset", "--quiet", "--hard", "refs/remotes/origin/HEAD")
        return dest.resolve()
    path = Path(spec).expanduser()
    if not path.is_absolute():
        raise KnowledgeError(
            f"{spec!r} is a relative path; pass the repository's absolute path (run `pwd` "
            "with Bash to find the working directory) or its clone URL"
        )
    if not path.is_dir():
        raise KnowledgeError(f"{spec!r} is not a directory or a repository URL")
    top = run_git(path, "rev-parse", "--show-toplevel", check=False).strip()
    if not top:
        raise KnowledgeError(f"{path} is not inside a git work tree")
    return Path(top).resolve()


def memory_location(repo: Path) -> tuple[str, Path]:
    """Return ``(slug, directory)`` of the repository's domain memory.

    The directory is the one every Sorcar run inside *repo* attaches as
    its domain memory (``sorcar_agent._repo_memory_domains``), under the
    configured memory root (``sorcar_agent._memory_settings``).
    """
    from kiss.agents.sorcar.sorcar_agent import _memory_settings, _repo_memory_domains

    domains = _repo_memory_domains(str(repo))
    if not domains:
        raise KnowledgeError(f"{repo} is not a git work tree")
    slug = next(iter(domains))
    return slug, _memory_settings()[1] / slug


def store_path(repo: Path) -> Path:
    """The block-store database of *repo*: ``<domain memory>/knowledge.sqlite3``."""
    return memory_location(repo)[1] / "knowledge.sqlite3"


# ----- files ------------------------------------------------------------------


def language_of(path: str) -> str:
    """Return the language name of *path* by its extension or special name."""
    name = path.rsplit("/", 1)[-1]
    if name in SPECIAL_FILES:
        return SPECIAL_FILES[name]
    return LANGUAGES.get(Path(name).suffix.lower(), "")


def is_binary(data: bytes) -> bool:
    """Return True when the first 8 KB of *data* contain a NUL byte."""
    return b"\0" in data[:8192]


def _cut(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " …"


def split_lines(text: str) -> list[str]:
    """Split *text* at newlines only (``str.splitlines`` also splits at ``\\x85``,
    ``\\x1e`` and other separators that editors treat as ordinary characters)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def find_symbols(lines: list[str], language: str) -> list[tuple[int, str, str]]:
    """Return ``(line_number, kind, name)`` for every definition line.

    Line numbers are 1-based.  Languages without patterns yield nothing.
    """
    patterns = _PATTERNS.get(_FAMILIES.get(language, ""), [])
    found: list[tuple[int, str, str]] = []
    for number, line in enumerate(lines, start=1):
        for pattern, default_kind in patterns:
            match = pattern.match(line)
            if not match:
                continue
            name = match.group("name")
            if name in _KEYWORDS:
                continue
            kind = (match.groupdict().get("kind") or default_kind).rstrip("!")
            found.append((number, kind, name))
            break
    return found


def chunk_spans(lines: list[str]) -> list[tuple[int, int, int]]:
    """Split *lines* into ``(start, end, piece)`` spans (0-based, half-open).

    A span holds at most :data:`CHUNK_LINES` lines and :data:`CHUNK_CHARS`
    characters (``piece`` 0).  A single line longer than :data:`CHUNK_CHARS`
    is split into pieces of that size, one span each (``piece`` 1, 2, ...),
    so no text is dropped.
    """
    spans: list[tuple[int, int, int]] = []
    start = 0
    size = 0
    for number, line in enumerate(lines):
        if len(line) > CHUNK_CHARS:
            if number > start:
                spans.append((start, number, 0))
            for piece in range(-(-len(line) // CHUNK_CHARS)):
                spans.append((number, number + 1, piece + 1))
            start, size = number + 1, 0
            continue
        full = number - start >= CHUNK_LINES or size + len(line) > CHUNK_CHARS
        if number > start and full:
            spans.append((start, number, 0))
            start, size = number, 0
        size += len(line) + 1
    if len(lines) > start:
        spans.append((start, len(lines), 0))
    return spans


def chunk_blocks(
    path: str, blob: str, lines: list[str], symbols: list[tuple[int, str, str]],
) -> list[Block]:
    """Build the ``chunk`` blocks that together hold the whole file (:func:`chunk_spans`)."""
    blocks = []
    for start, end, piece in chunk_spans(lines):
        if piece:
            offset = (piece - 1) * CHUNK_CHARS
            label = f"{path} line {start + 1} part {piece}"
            blocks.append(Block(
                "chunk", f"chunk:{path}:{start + 1}:p{piece}", label,
                f"{label}\n{lines[start][offset:offset + CHUNK_CHARS]}", path, blob,
            ))
            continue
        names = [name for number, _, name in symbols if start < number <= end][:5]
        label = f"{path} lines {start + 1}-{end}"
        title = label + (" · " + ", ".join(names) if names else "")
        body = "\n".join(lines[start:end])
        blocks.append(
            Block("chunk", f"chunk:{path}:{start + 1}", title, f"{label}\n{body}", path, blob)
        )
    return blocks


def file_blocks(
    path: str, blob: str, data: bytes, last_touch: tuple[str, str] | None,
) -> list[Block]:
    """Build the ``chunk``, ``symbol`` and (last) ``file`` blocks of one file.

    Args:
        path: Repository-relative path.
        blob: The blob hash the content came from.
        data: The file content.
        last_touch: ``(sha, date)`` of the latest commit touching the file, if known.
    """
    language = language_of(path)
    size = len(data)
    meta = [f"path: {path}", f"language: {language or 'unknown'}", f"size: {size} bytes"]
    if last_touch:
        meta.append(f"last commit: {last_touch[0][:12]} ({last_touch[1]})")
    reason = (
        "binary" if is_binary(data) else "generated" if path.endswith(GENERATED_SUFFIXES)
        else "too large" if size > MAX_CONTENT_BYTES else ""
    )
    if reason:
        limit = f": {size} bytes, limit {MAX_CONTENT_BYTES}" if reason == "too large" else ""
        meta.append(f"content: not indexed ({reason}{limit})")
        return [
            Block("file", f"file:{path}", f"{path} ({reason} file)", "\n".join(meta), path, blob)
        ]
    lines = split_lines(data.decode("utf-8", errors="replace"))
    symbols = find_symbols(lines, language)
    meta.append(f"lines: {len(lines)}")
    imports = [_cut(line.strip()) for line in lines if _IMPORT_RE.match(line)][:50]
    if imports:
        meta.append("imports:\n  " + "\n  ".join(imports))
    if symbols:
        meta.append(
            "symbols: " + ", ".join(f"{kind} {name}" for _, kind, name in symbols[:200])
        )
    meta.append("head:\n" + "\n".join(_cut(line) for line in lines[:HEAD_LINES]))
    title = f"{path} ({language or 'file'}, {len(lines)} lines)"
    blocks = chunk_blocks(path, blob, lines, symbols)
    for number, kind, name in symbols:
        context = "\n".join(_cut(line) for line in lines[number - 1:number + SYMBOL_CONTEXT_LINES])
        blocks.append(Block(
            "symbol", f"symbol:{path}:{number}:{name}", f"{kind} {name} — {path}:{number}",
            f"{kind} {name}\n{path}:{number}\n{context}", path, blob,
        ))
    # The file block comes last: it is the marker that the file is completely
    # indexed, so an interrupted run re-indexes the file next time.
    blocks.append(Block("file", f"file:{path}", title, "\n".join(meta), path, blob))
    return blocks


def ls_files(repo: Path) -> dict[str, str]:
    """Map every tracked path to its blob hash (``git ls-files -s``)."""
    files: dict[str, str] = {}
    for entry in run_git(repo, "ls-files", "-s", "-z").split("\0"):
        if not entry:
            continue
        info, path = entry.split("\t", 1)
        mode, blob = info.split()[:2]
        if mode != "160000":  # skip submodule gitlinks
            files[path] = blob
    return files


def read_blobs(repo: Path, blobs: Iterable[str]) -> Iterator[tuple[str, bytes]]:
    """Yield ``(blob, content)`` for every requested blob via one ``git cat-file --batch``."""
    proc = subprocess.Popen(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    assert proc.stdin is not None and proc.stdout is not None
    try:
        for blob in blobs:
            proc.stdin.write(blob.encode() + b"\n")
            proc.stdin.flush()
            header = proc.stdout.readline().split()
            if len(header) != 3:  # "<sha> missing"
                continue
            size = int(header[2])
            data = proc.stdout.read(size)
            proc.stdout.read(1)  # trailing newline
            yield blob, data
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()
        proc.stdout.close()
        proc.wait(timeout=60)


def indexed_file_blocks(
    repo: Path, paths: list[str], files: dict[str, str],
    last_touch: dict[str, tuple[str, str]],
) -> Iterator[Block]:
    """Yield the blocks of every path in *paths*, reading blobs in one batch."""
    by_blob: dict[str, list[str]] = {}
    for path in paths:
        by_blob.setdefault(files[path], []).append(path)
    for blob, data in read_blobs(repo, list(by_blob)):
        for path in by_blob[blob]:
            yield from file_blocks(path, blob, data, last_touch.get(path))


# ----- commits ----------------------------------------------------------------

# One NUL-framed header per commit (NUL cannot occur in a commit object, so
# the framing survives any message text), then a newline and the numstat and
# patch git prints for the commit, then the next commit's NUL.  The message
# (``%B``) is the last field, so a separator byte inside it stays in it.  A
# patch may itself contain NUL (git only looks at a file's first 8 KB to call
# it binary), so a segment counts as a header only when it starts like one
# (a SHA-1 or SHA-256 hash followed by the field separator).
_LOG_FORMAT = "%x00%H%x1f%P%x1f%an%x1f%ae%x1f%aI%x1f%cI%x1f%ct%x1f%D%x1f%B%x00"
_HEADER_FIELDS = 9
_HEADER_RE = re.compile(rb"^[0-9a-f]{40}(?:[0-9a-f]{24})?\x1f")


def unquote_path(path: str) -> str:
    """Decode a C-quoted git path (``"caf\\303\\251.txt"``); plain paths pass through."""
    if len(path) < 2 or path[0] != '"' or path[-1] != '"':
        return path

    def replace(match: re.Match[bytes]) -> bytes:
        code = match.group(1)
        return _ESCAPES.get(code) or bytes([int(code, 8)])

    return _ESCAPE_RE.sub(replace, path[1:-1].encode("utf-8", "surrogateescape")).decode(
        "utf-8", errors="backslashreplace"
    )


def _parse_header(header: str) -> Commit:
    fields = header.split("\x1f", _HEADER_FIELDS - 1)
    if len(fields) != _HEADER_FIELDS:
        raise KnowledgeError(f"unexpected git log record: {header[:200]!r}")
    sha, parents, author, email, authored, committed, timestamp, refs, message = fields
    return Commit(
        sha.strip(), parents.strip(), author, email, authored, committed, int(timestamp),
        refs, message.strip("\n"),
    )


def _parse_diff(commit: Commit, diff: str) -> None:
    """Fill *commit*'s numstat and per-file patches from git's diff output."""
    numstat_lines, _, patch = diff.partition("\ndiff --git ")
    for line in numstat_lines.split("\n"):
        match = _NUMSTAT_RE.match(line)
        if match:
            commit.numstat.append((match.group(1), match.group(2), unquote_path(match.group(3))))
    if patch:
        for file_patch in ("diff --git " + patch).split("\ndiff --git "):
            if not file_patch.startswith("diff --git "):
                file_patch = "diff --git " + file_patch
            commit.patches[_patch_path(file_patch)] = file_patch


def _patch_path(file_patch: str) -> str:
    """The repository path a per-file patch is about.

    Git writes ``+++ b/<path>`` (``--- a/<path>`` for a deletion), adds a
    tab after a path containing spaces, and C-quotes the whole ``b/<path>``
    when the name needs escaping.
    """
    for line in file_patch.split("\n")[:8]:
        for marker in ("+++ ", "--- "):
            if line.startswith(marker) and not line.startswith(marker + "/dev/null"):
                name = unquote_path(line[len(marker):].removesuffix("\t"))
                return name[2:]  # drop the a/ or b/ prefix
    # No ---/+++ lines (a mode-only change): read the ``diff --git a/x b/x`` line,
    # whose two names are C-quoted as a whole when they need escaping.
    first = file_patch.split("\n", 1)[0][len("diff --git "):]
    quoted = _QUOTED_PAIR_RE.match(first)
    if quoted:
        return unquote_path(quoted.group(2))[2:]
    return first.split(" b/", 1)[-1]


def _decode_diff(parts: list[bytes]) -> str:
    """Join the NUL-split pieces of one commit's diff, spelling the NUL bytes out."""
    return b"\\0".join(parts).decode("utf-8", errors="backslashreplace")


def _segments(stream: IO[bytes]) -> Iterator[bytes]:
    """Yield the NUL-separated segments of *stream*.

    Chunks are collected in a list and joined once per segment, so a
    segment spanning many chunks costs linear time.  A segment longer than
    :data:`MAX_RECORD_BYTES` (a commit whose diff is gigantic) keeps only
    its first bytes: the numstat survives, the tail of the patch is dropped.
    """
    parts: list[bytes] = []
    size = 0
    while True:
        data = stream.read(1 << 20)
        if not data:
            break
        if b"\0" not in data:
            if size < MAX_RECORD_BYTES:
                parts.append(data)
                size += len(data)
            continue
        first, *complete, rest = data.split(b"\0")
        parts.append(first)
        yield b"".join(parts)
        yield from complete
        parts = [rest]
        size = len(rest)
    if parts:
        yield b"".join(parts)


def iter_commits(repo: Path, shas: list[str]) -> Iterator[Commit]:
    """Stream the given commits, in the given order, with numstat and first-parent patches.

    Raises:
        KnowledgeError: When git exits non-zero (the whole stream was consumed first).
    """
    if not shas:
        return
    stdin = tempfile.TemporaryFile()
    stdin.write("".join(f"{sha}\n" for sha in shas).encode())
    stdin.seek(0)
    stderr_file = tempfile.TemporaryFile()  # a pipe could fill up and block git
    proc = subprocess.Popen(
        [
            "git", "-C", str(repo), "-c", "core.quotePath=false", "-c", "log.showRoot=true",
            "log", "--stdin", "--no-walk=unsorted", "--no-color", "--no-renames", "--numstat",
            "-p", "-U1", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
            "--diff-merges=first-parent", f"--format={_LOG_FORMAT}",
        ],
        stdin=stdin, stdout=subprocess.PIPE, stderr=stderr_file,
    )
    stdin.close()
    assert proc.stdout is not None
    commit: Commit | None = None
    diff_parts: list[bytes] = []
    diff_size = 0
    try:
        for segment in _segments(proc.stdout):
            if _HEADER_RE.match(segment):
                if commit is not None:
                    _parse_diff(commit, _decode_diff(diff_parts))
                    yield commit
                commit = _parse_header(segment.decode("utf-8", errors="backslashreplace"))
                diff_parts, diff_size = [], 0
            elif commit is not None:
                if diff_size + len(segment) > MAX_RECORD_BYTES or len(segment) >= MAX_RECORD_BYTES:
                    commit.truncated = True  # _segments cut it, or this piece overflows the cap
                if diff_size < MAX_RECORD_BYTES:
                    diff_parts.append(segment)  # the diff, or its continuation after a NUL byte
                    diff_size += len(segment)
        if commit is not None:
            _parse_diff(commit, _decode_diff(diff_parts))
            yield commit
    finally:
        proc.stdout.close()
        status = proc.wait(timeout=60)
        stderr_file.seek(0)
        stderr = stderr_file.read().decode(errors="replace")
        stderr_file.close()
    if status != 0:
        raise KnowledgeError(f"git log failed: {stderr.strip()}")


def commit_blocks(commit: Commit) -> list[Block]:
    """Build the ``commit`` block and the ``change`` blocks of every touched file.

    A patch longer than :data:`MAX_PATCH_CHARS` continues in
    ``change:<sha>#2:<path>``, ``#3``, ... so the whole patch is searchable.
    """
    short = commit.sha[:12]
    changed = [f"  {added:>5} {deleted:>5} {path}" for added, deleted, path in commit.numstat]
    lines = [
        f"commit {commit.sha}",
        f"author: {commit.author} <{commit.email}>",
        f"date: {commit.authored}",
        f"parents: {commit.parents or '(root commit)'}",
    ]
    if commit.refs:
        lines.append(f"refs: {commit.refs}")
    lines.append(f"message:\n{commit.message}")
    if commit.truncated:
        lines.append(f"patch: truncated at {MAX_RECORD_BYTES} bytes; later files have no patch")
    if changed:
        lines.append(
            f"files changed ({len(changed)}):\n  added deleted path\n" + "\n".join(changed)
        )
    blocks = []
    for added, deleted, path in commit.numstat:
        patch = commit.patches.get(path, "")
        pieces = [
            patch[i:i + MAX_PATCH_CHARS] for i in range(0, len(patch), MAX_PATCH_CHARS)
        ] or [""]
        header = (
            f"commit {commit.sha} ({commit.authored}) by {commit.author}\n"
            f"subject: {commit.subject}\nfile: {path} (+{added} -{deleted})"
        )
        for number, piece in enumerate(pieces, start=1):
            marker = f"#{number}" if number > 1 else ""  # a hash cannot contain '#'
            part = f" part {number}/{len(pieces)}" if len(pieces) > 1 else ""
            blocks.append(Block(
                "change", f"change:{commit.sha}{marker}:{path}",
                f"{short} {path} (+{added} -{deleted}){part}", f"{header}{part}\n{piece}",
                path, commit.sha,
            ))
    # The commit block comes last: it marks the commit as completely indexed.
    blocks.append(Block(
        "commit", f"commit:{commit.sha}", f"{short} {commit.subject}"[:200],
        "\n".join(lines), "", commit.sha,
    ))
    return blocks


def history_blocks(
    commits: Iterable[Commit], on_head: set[str], last_touch: dict[str, tuple[str, str]],
    subjects: list[str],
) -> Iterator[Block]:
    """Yield the blocks of every commit, recording each file's newest touching commit.

    Args:
        commits: The commit stream, newest first (children before parents).
        on_head: The commits in HEAD's history; only they describe the
            checked-out content, so only they attribute files.
        last_touch: Filled with ``path -> (sha, date)`` of the first commit
            seen touching each path (the newest, given the order).
        subjects: Filled with ``"<sha7> <subject>"`` for every commit.
    """
    for commit in commits:
        subjects.append(f"{commit.sha[:7]} {commit.subject}")
        if commit.sha in on_head:
            for _, _, path in commit.numstat:
                last_touch.setdefault(path, (commit.sha, commit.authored))
        yield from commit_blocks(commit)


# ----- refs, authors, directories, repo ---------------------------------------


def ref_blocks(repo: Path) -> list[Block]:
    """Build one ``tag`` block per tag and one ``branch`` block per local or remote branch."""
    blocks = []
    # Six NUL-separated fields per ref (NUL cannot occur in a message, unlike
    # every other separator byte); git ends each record with a newline.
    out = run_git(
        repo, "for-each-ref", "refs/tags", "refs/heads", "refs/remotes",
        "--format=%(refname)%00%(objectname)%00%(*objectname)%00%(creatordate:iso-strict)"
        "%00%(contents:subject)%00%(contents:body)%00",
    )
    fields = out.split("\0")
    for start in range(0, len(fields) - 1, 6):
        refname, obj, target, date, subject, body = fields[start:start + 6]
        refname = refname.lstrip("\n")  # the previous record's terminator
        commit = target or obj
        if refname.startswith("refs/tags/"):
            name = refname[len("refs/tags/"):]
            text = f"tag {name}\ncommit: {commit}\ndate: {date}\nmessage: {subject}\n{body}".strip()
            blocks.append(
                Block("tag", f"tag:{name}", f"tag {name} → {commit[:12]}", text, "", commit)
            )
        else:
            name = refname.split("/", 2)[-1]
            text = f"branch {name}\ntip: {commit}\ndate: {date}\nsubject: {subject}"
            blocks.append(Block(
                "branch", f"branch:{name}", f"branch {name} → {commit[:12]}", text, "", commit,
            ))
    return blocks


def author_blocks(repo: Path) -> tuple[list[Block], int, str, str]:
    """Build one ``author`` block per contributor email over every ref's history.

    Returns:
        ``(blocks, commit_count, first_date, last_date)``.
    """
    stats: dict[str, _Author] = {}
    total = 0
    first = last = ""
    for line in run_git(repo, "log", "--all", "--format=%ae%x1f%an%x1f%aI").splitlines():
        email, name, date = line.split("\x1f")
        total += 1
        last = last or date
        first = date
        author = stats.setdefault(email, _Author(email, last=date))
        author.names.add(name)
        author.count += 1
        author.first = date
    blocks = []
    for author in stats.values():
        names = ", ".join(sorted(author.names))
        text = (
            f"contributor {names} <{author.email}>\ncommits: {author.count}\n"
            f"first commit: {author.first}\nlast commit: {author.last}"
        )
        blocks.append(Block(
            "author", f"author:{author.email}", f"{names} ({author.count} commits)", text,
        ))
    blocks.sort(key=lambda block: block.title)
    return blocks, total, first, last


def dir_blocks(files: dict[str, str], readmes: dict[str, str]) -> list[Block]:
    """Build one ``dir`` block per directory of the tree (the root is ``.``).

    Args:
        files: Tracked paths.
        readmes: Directory to the head of its README.
    """
    children: dict[str, set[str]] = {"": set()}
    counts: Counter[str] = Counter()
    for path in files:
        parts = path.split("/")
        for depth in range(len(parts)):
            parent = "/".join(parts[:depth])
            entry = parts[depth] + ("/" if depth < len(parts) - 1 else "")
            children.setdefault(parent, set()).add(entry)
            counts[parent] += 1
    blocks = []
    for directory, entries in children.items():
        shown = sorted(entries)
        listing = "\n".join(shown[:MAX_DIR_ENTRIES])
        if len(shown) > MAX_DIR_ENTRIES:
            listing += f"\n… {len(shown) - MAX_DIR_ENTRIES} more entries"
        label = directory or "."
        text = f"directory {label}\nfiles (recursive): {counts[directory]}\nentries:\n{listing}"
        if directory in readmes:
            text += f"\nREADME:\n{readmes[directory]}"
        blocks.append(Block(
            "dir", f"dir:{label}", f"directory {label}/ ({counts[directory]} files)", text,
            directory,
        ))
    return blocks


def readme_heads(repo: Path, files: dict[str, str]) -> dict[str, str]:
    """Map each directory holding a README to the first :data:`HEAD_LINES` lines of it."""
    candidates = {
        path.rsplit("/", 1)[0] if "/" in path else "": blob
        for path, blob in sorted(files.items())
        if path.rsplit("/", 1)[-1].lower().startswith("readme")
    }
    by_blob = {blob: directory for directory, blob in candidates.items()}
    heads = {}
    for blob, data in read_blobs(repo, list(by_blob)):
        if not is_binary(data):
            text = data.decode("utf-8", errors="replace")
            heads[by_blob[blob]] = "\n".join(_cut(line) for line in text.splitlines()[:HEAD_LINES])
    return heads


def repo_block(
    repo: Path, head: str, files: dict[str, str], languages: dict[str, int],
    top_dirs: list[tuple[str, int]], commits: int, first: str, last: str,
    tags: list[str], authors: int,
) -> Block:
    """Build the single ``repo`` block summarizing the repository."""
    remote = run_git(repo, "remote", "get-url", "origin", check=False).strip()
    branch = run_git(repo, "rev-parse", "--abbrev-ref", "HEAD", check=False).strip()
    on_head = run_git(repo, "rev-list", "--count", "HEAD").strip()
    lines = [
        f"repository {repo.name}",
        f"path: {repo}",
        f"remote: {remote or '(none)'}",
        f"branch: {branch}",
        f"HEAD: {head}",
        f"tracked files: {len(files)}",
        f"commits: {commits} on all refs, {on_head} on HEAD ({first} … {last})",
        f"contributors: {authors}",
        f"tags: {len(tags)}" + (f" (latest: {', '.join(tags[-10:])})" if tags else ""),
        "languages: " + ", ".join(f"{name} {count}" for name, count in languages.items()),
        "top-level directories: "
        + ", ".join(f"{name} ({count} files)" for name, count in top_dirs),
    ]
    return Block("repo", "repo", f"repository {repo.name}", "\n".join(lines), "", head)


# ----- the run ----------------------------------------------------------------


def shallow_boundary(repo: Path) -> set[str]:
    """The commits at the edge of a shallow clone (empty for a complete history).

    A boundary commit is shown without parents and with a diff against
    nothing; once ``git fetch --unshallow`` (or a deeper fetch) reveals its
    parents it must be indexed again.
    """
    # git prints the path relative to the work tree or absolute; joining handles both.
    shallow = repo / run_git(repo, "rev-parse", "--git-path", "shallow").strip()
    if not shallow.is_file():
        return set()
    return set(shallow.read_text(encoding="utf-8", errors="replace").split())


def last_touch_of(repo: Path, path: str) -> tuple[str, str] | None:
    """``(sha, date)`` of the newest commit in HEAD's history touching *path*, if any.

    HEAD's history, not every ref: the checked-out content is HEAD's, and
    another branch may have touched the file more recently.
    """
    out = run_git(
        repo, "--literal-pathspecs", "log", "HEAD", "-1", "--format=%H%x1f%aI", "--", path,
    ).strip()
    if not out:
        return None
    sha, date = out.split("\x1f")
    return sha, date


MAX_LAST_TOUCH_LOOKUPS = 200
"""Per-path ``git log -1`` calls a run makes for files the commit stream did not attribute."""


def index_repo(repo: Path, store: KnowledgeStore) -> IndexReport:
    """Index *repo* into *store*: everything on the first run, the differences after.

    Files whose blob changed are re-indexed and removed files lose their
    blocks; commits are the set difference between ``git rev-list --all``
    and the stored commits, so new branches, an unshallowed history and a
    rewritten history (its dropped commits are deleted) are all covered.
    A store that held an unrelated repository (no commit in common; two
    checkouts with the same directory name share one memory slug) is
    emptied first.  Blocks are written children-first (chunks before the
    ``file`` block, changes before the ``commit`` block), so an interrupted
    run leaves no object that looks complete but is not.

    Args:
        repo: Root of a local checkout (:func:`resolve_repo`).
        store: The repository's block store.

    Raises:
        KnowledgeError: When the repository has no commits or git fails.
    """
    started = time.monotonic()
    stamp = now_iso()
    head = run_git(repo, "rev-parse", "--verify", "HEAD", check=False).strip()
    if not head:
        raise KnowledgeError(f"{repo} has no commits yet")
    all_shas = run_git(repo, "rev-list", "--all", "--topo-order").split()  # children first
    on_head = set(run_git(repo, "rev-list", "HEAD").split())
    stored = {key[len("commit:"):] for key in store.keys("commit")}
    # One commit in common means the same repository (a shallow clone shares
    # its few commits with the complete history; a rewritten history keeps
    # some ancestor); no commit in common means another repository owned the
    # store, whose blocks are dropped except the agent's notes.
    same_repo = not stored.isdisjoint(all_shas)
    if stored and not same_repo:
        for kind in KINDS:
            if kind != "note":
                store.delete(kind=kind)
        stored = set()
    previous = store.get_meta("head")
    mode = "incremental" if same_repo else "full"
    files = ls_files(repo)
    known = store.file_shas()
    changed = {path for path, blob in files.items() if known.get(path) != blob}
    removed = sorted(path for path in known if path not in files)

    shallow = shallow_boundary(repo)
    unshallowed = set(store.get_meta("shallow").split()) - shallow  # boundary moved: re-index
    stale = sorted(stored.difference(all_shas) | (unshallowed & stored))
    changed.update(path for path in store.paths_of_shas(stale) if path in files)
    store.delete_shas(stale)
    store.delete_paths(removed + sorted(changed))

    last_touch: dict[str, tuple[str, str]] = {}
    subjects: list[str] = []
    new = [sha for sha in all_shas if sha not in stored or sha in unshallowed]
    store.upsert(history_blocks(iter_commits(repo, new), on_head, last_touch, subjects), stamp)
    unattributed = [path for path in sorted(changed) if path not in last_touch]
    if len(unattributed) <= MAX_LAST_TOUCH_LOOKUPS:
        for path in unattributed:
            touch = last_touch_of(repo, path)
            if touch:
                last_touch[path] = touch
    store.upsert(indexed_file_blocks(repo, sorted(changed), files, last_touch), stamp)

    for kind in ("tag", "branch", "author", "dir"):
        store.delete(kind=kind)
    refs = ref_blocks(repo)
    store.upsert(refs, stamp)
    authors, commit_count, first, last = author_blocks(repo)
    store.upsert(authors, stamp)
    store.upsert(dir_blocks(files, readme_heads(repo, files)), stamp)

    languages = Counter(language_of(path) or "other" for path in files)
    top_dirs = Counter(path.split("/", 1)[0] if "/" in path else "." for path in files)
    tags = [block.key[len("tag:"):] for block in refs if block.kind == "tag"]
    languages_sorted = dict(languages.most_common())
    dirs_sorted = top_dirs.most_common()
    store.upsert([repo_block(
        repo, head, files, languages_sorted, dirs_sorted, commit_count, first, last, tags,
        len(authors),
    )], stamp)
    store.set_meta(
        head=head, indexed_at=stamp, repo=str(repo), previous_head=previous, mode=mode,
        shallow=" ".join(sorted(shallow)),
    )
    slug, memory_dir = memory_location(repo)
    return IndexReport(
        repo=str(repo), slug=slug, memory_dir=str(memory_dir), store_path=str(store.path),
        mode=mode, head=head, previous_head=previous,
        seconds=round(time.monotonic() - started, 1), counts=store.counts(),
        files_total=len(files), files_indexed=len(changed), files_removed=len(removed),
        changed_paths=sorted(changed), commits_indexed=len(subjects),
        commits_removed=len(stale), new_commits=subjects, languages=languages_sorted,
        top_dirs=dirs_sorted, authors=[block.title for block in authors[:20]], tags=tags,
    )


def format_report(report: IndexReport, list_limit: int = 100) -> str:
    """Render an :class:`IndexReport` as the text the agent reads."""
    lines = [
        f"repository: {report.repo}",
        f"memory slug: {report.slug}",
        f"memory directory: {report.memory_dir}",
        f"block store: {report.store_path}",
        f"mode: {report.mode}" + (
            f" (previous HEAD {report.previous_head[:12]})" if report.previous_head else ""
        ),
        f"HEAD: {report.head}",
        f"seconds: {report.seconds}",
        "blocks: " + ", ".join(f"{kind} {count}" for kind, count in sorted(report.counts.items())),
        f"tracked files: {report.files_total}; indexed now: {report.files_indexed}; "
        f"removed: {report.files_removed}",
        f"commits indexed now: {report.commits_indexed}; dropped from history: "
        f"{report.commits_removed}",
        "languages: " + ", ".join(f"{k} {v}" for k, v in report.languages.items()),
        "top-level directories: " + ", ".join(f"{k} ({v})" for k, v in report.top_dirs),
        "top contributors: " + "; ".join(report.authors),
        "tags: " + (", ".join(report.tags[-list_limit:]) or "(none)"),
    ]
    if report.changed_paths:
        shown = report.changed_paths[:list_limit]
        more = len(report.changed_paths) - len(shown)
        lines.append(
            "changed files:\n  " + "\n  ".join(shown) + (f"\n  … {more} more" if more else "")
        )
    if report.new_commits:
        shown = report.new_commits[:list_limit]
        more = len(report.new_commits) - len(shown)
        lines.append(
            "new commits (newest first):\n  " + "\n  ".join(shown)
            + (f"\n  … {more} more" if more else "")
        )
    return "\n".join(lines)

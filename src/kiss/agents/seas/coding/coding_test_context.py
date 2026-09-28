# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Test-guided context: which existing tests reference the code the agent just changed.

After the agent edits a pre-existing, non-test source file, the harness
diffs the file, finds the definitions (functions, classes, ...) that
enclose the changed lines, greps the repository's test files for those
names and appends the ranked list to the tool result.  The agent then
knows which tests specify the behaviour it is changing and can read and
run them instead of guessing.  Everything here is language-agnostic:
definitions are found with one regex, tests with a path heuristic.
"""

from __future__ import annotations

import difflib
import posixpath
import re
import shlex
import subprocess

#: Start of a named definition in the common languages (Python, JS/TS, Go,
#: Rust, Java, C#, Ruby, PHP, ...): keyword(s) followed by the name.
DEFINITION_RE = re.compile(
    r"^(?P<indent>[ \t]*)(?:export\s+|pub(?:\([^)]*\))?\s+|static\s+|async\s+|public\s+|private\s+"
    r"|protected\s+|abstract\s+|final\s+|default\s+)*"
    r"(?:def|class|function|fn|func|struct|enum|trait|impl|interface|type|module|object|record)\s+"
    r"(?:\([^)]*\)\s*)?"  # Go receiver: func (r *T) Name(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
)

#: Path fragments that mark a file as a test (any language, any layout).
TEST_PATH_RE = re.compile(
    r"(^|/)(tests?|testing|spec|specs|__tests__)(/|_|-|\.)|(^|/)test_|_tests?\.|\.tests?\.|\.spec\.",
    re.IGNORECASE,
)

#: ``FooTests.cs`` / ``FooTest.java`` style (case-sensitive: ``contest.py`` is not a test).
CAMEL_TEST_PATH_RE = re.compile(r"[a-z0-9]Tests?\.[A-Za-z]+$")

#: Directories never searched for tests.
EXCLUDED_DIRS = (".git", "node_modules", ".tox", ".venv", "venv", "build", "dist", "__pycache__",
                 ".mypy_cache")

#: Most test files listed in one note.
MAX_TESTS = 8

#: Names too generic to grep for (they match everywhere).
GENERIC_NAMES = {"main", "init", "__init__", "setup", "test", "run", "new", "self", "cls", "index",
                 "app", "utils"}


def is_test_path(path: str) -> bool:
    """Whether *path* looks like a test file or lives in a test directory.

    Args:
        path: File path (absolute or relative, POSIX separators).
    """
    return TEST_PATH_RE.search(path) is not None or CAMEL_TEST_PATH_RE.search(path) is not None


def changed_definitions(old: str, new: str) -> list[str]:
    """Names of the definitions that enclose every line changed between *old* and *new*.

    Walks upward from each changed line of *new* to the definitions with
    strictly smaller indentation (a method and its class, a nested function
    and its parent, ...), so a change inside ``Field.check`` yields
    ``["check", "Field"]``.  Names appear once, innermost first, generic
    names (``__init__``, ``main`` ...) excluded.

    Args:
        old: File content before the edit.
        new: File content after the edit.

    Returns:
        Definition names, or an empty list when nothing enclosing was found.
    """
    old_lines, new_lines = old.splitlines(), new.splitlines()
    changed: list[int] = []
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "insert"):
            changed.extend(range(j1, j2))
        elif tag == "delete":
            changed.append(min(j1, len(new_lines) - 1))
    names: list[str] = []
    for index in changed:
        if index < 0 or index >= len(new_lines):
            continue
        indent = _indent(new_lines[index])
        for name in _enclosing_definitions(new_lines, index, indent):
            if name not in names and name not in GENERIC_NAMES:
                names.append(name)
    return names


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _enclosing_definitions(lines: list[str], index: int, indent: int) -> list[str]:
    """Definitions at or above *index* indented less than *indent* (or equally, if a definition)."""
    names: list[str] = []
    limit = indent
    match = DEFINITION_RE.match(lines[index])
    if match:  # the changed line is itself a definition header
        names.append(match.group("name"))
    for line in reversed(lines[:index]):
        if not line.strip():
            continue
        line_indent = _indent(line)
        if line_indent >= limit:
            continue
        match = DEFINITION_RE.match(line)
        if match:
            names.append(match.group("name"))
        limit = line_indent
        if limit == 0:
            break
    return names


def find_referencing_tests(
    container: str, workdir: str, names: list[str], exclude: str
) -> list[tuple[str, int]]:
    """Test files under *workdir* in *container* that mention any of *names*, most mentions first.

    Args:
        container: Docker container name or id.
        workdir: Repository root inside the container.
        names: Identifiers to grep for (whole words).
        exclude: Path of the edited file (never listed).

    Returns:
        Up to :data:`MAX_TESTS` ``(relative path, mention count)`` pairs.  Files
        that mention more of the names (the method and its class rather than
        just a common method name) rank first, then by total mentions.
    """
    if not names:
        return []
    excludes = " ".join(f"--exclude-dir={shlex.quote(d)}" for d in EXCLUDED_DIRS)
    # one grep per name, each output line prefixed with the name it counted
    greps = " ; ".join(
        f"grep -rIcw {excludes} -e {shlex.quote(name)} . 2>/dev/null | grep -v ':0$' "
        f"| sed 's/^/{index}\t/'"
        for index, name in enumerate(names[:4])
    )
    command = f"cd {shlex.quote(workdir)} && {{ {greps} ; }}"
    try:
        completed = subprocess.run(
            ["docker", "exec", container, "sh", "-c", command],
            capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    exclude_rel = posixpath.relpath(exclude, workdir) if exclude.startswith(workdir) else exclude
    matched: dict[str, set[int]] = {}
    mentions: dict[str, int] = {}
    for line in completed.stdout.splitlines():
        index, tab, rest = line.partition("\t")
        path, sep, count = rest.rpartition(":")
        if not tab or not sep or not count.isdigit():
            continue
        path = path[2:] if path.startswith("./") else path
        if path == exclude_rel or not is_test_path(path):
            continue
        matched.setdefault(path, set()).add(int(index))
        mentions[path] = mentions.get(path, 0) + int(count)
    tokens = path_tokens(exclude_rel)

    def rank(path: str) -> tuple[int, int, int, int, str]:
        lowered = path.lower()
        overlap = any(token in lowered for token in tokens)
        return (-len(matched[path]), -int(overlap), -int(is_test_file_name(path)),
                -mentions[path], path)

    return [(path, mentions[path]) for path in sorted(mentions, key=rank)[:MAX_TESTS]]


#: Path components that say nothing about what a file is about.
GENERIC_PATH_TOKENS = {"src", "lib", "libs", "core", "main", "util", "utils", "base", "common",
                       "internal", "pkg", "python", "site-packages", "impl", "misc", "api", "model",
                       "models", "views", "init"}


def path_tokens(relpath: str) -> list[str]:
    """Distinctive lower-case components of an edited file's path, used to spot related test paths.

    The first component (the package root, present in every path of a
    single-package repo) and ``__init__``/``index``/``mod`` file names are
    skipped; so are very short and generic components.

    Args:
        relpath: Path relative to the repository root.
    """
    parts = relpath.lower().split("/")
    stem = posixpath.splitext(parts[-1])[0]
    parts = parts[:-1] + ([stem] if stem not in ("__init__", "index", "mod") else [])
    if len(parts) > 2:
        parts = parts[1:]
    return [part for part in parts if len(part) >= 4 and part not in GENERIC_PATH_TOKENS]


def is_test_file_name(path: str) -> bool:
    """Whether the file itself is named like a test module (not just placed in a test directory)."""
    return (TEST_PATH_RE.search(posixpath.basename(path)) is not None
            or CAMEL_TEST_PATH_RE.search(path) is not None)


def test_context_note(edited: str, names: list[str], tests: list[tuple[str, int]]) -> str:
    """The note appended to the tool result of an edit.

    Args:
        edited: Path of the edited file as the agent named it.
        names: Definitions enclosing the change.
        tests: ``(path, mention count)`` pairs from :func:`find_referencing_tests`.
    """
    listed = ", ".join(f"{path} ({count})" for path, count in tests)
    return (
        f"[Existing tests that reference the code you changed in {edited} "
        f"({', '.join(names)}; mention count in parentheses): {listed}. "
        "They show the behaviour the project expects from this code: read the "
        "relevant ones and run them before you finish.]"
    )

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Memory pages: Markdown files with YAML frontmatter in a flat directory.

This module implements the *data* half of the memoryfield pattern
(https://github.com/calpaterson/memoryfield-spec): a memory is a flat
directory of short Markdown "pages". Each page may start with a YAML
frontmatter block carrying ``title``, ``uuid``, ``summary``, ``created`` and
``updated``. The pages are the canonical data; the vector index built by
:mod:`kiss.core.memoryfield.index` is a regenerable cache.

Only the standard library plus PyYAML (already a project dependency) is used.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from kiss.core.utils import atomic_write_text, read_bytes_waiting_for_writer

# Page filenames: ASCII lowercase letters, digits and hyphens, starting and
# ending with a letter or digit (memoryfield spec, "Pages").
PAGE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")

# Files that sync tools, editors and operating systems leave behind; the
# spec requires these to be ignored even when they end in ``.md``.
DEBRIS_NAMES = frozenset({".DS_Store", "desktop.ini", "Thumbs.db"})

# Pages SHOULD NOT exceed this many bytes; indexers only embed this prefix.
MAX_PAGE_BYTES = 8192

FRONTMATTER_KEYS = ("title", "uuid", "summary", "created", "updated")


@dataclass(frozen=True)
class Page:
    """One memory page loaded from disk.

    Attributes:
        name: The page name without the ``.md`` extension.
        frontmatter: Parsed YAML frontmatter (empty when absent or malformed).
        body: The Markdown body after the frontmatter block.
        raw: The complete file text, frontmatter included.
    """

    name: str
    frontmatter: dict[str, Any]
    body: str
    raw: str

    @property
    def title(self) -> str:
        """The page title from frontmatter, falling back to the page name."""
        title = self.frontmatter.get("title")
        return str(title) if title else self.name

    @property
    def summary(self) -> str:
        """The one-line summary from frontmatter, or an empty string."""
        summary = self.frontmatter.get("summary")
        return str(summary) if summary else ""


def read_page_text(path: Path) -> str:
    """Read a page as UTF-8, replacing undecodable bytes so a damaged page stays usable.

    Pages are shared between agents and rewritten atomically, so the read
    waits out a concurrent ``os.replace`` on Windows instead of failing.

    Args:
        path: The page file.
    """
    return read_bytes_waiting_for_writer(path).decode("utf-8", errors="replace")


def now_iso() -> str:
    """Return the current UTC time as a quoted-safe ISO 8601 string (``...Z``)."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_debris(filename: str) -> bool:
    """Return True for files that must never be treated as pages.

    Args:
        filename: A bare filename inside the memory directory.
    """
    return filename in DEBRIS_NAMES or filename.endswith("~") or ".sync-conflict-" in filename


def is_valid_page_name(name: str) -> bool:
    """Return True when *name* satisfies the spec's page filename rules.

    Args:
        name: A page name without the ``.md`` extension.
    """
    return bool(PAGE_NAME_RE.match(name))


def slugify(text: str, max_length: int = 60) -> str:
    """Turn free text into a valid page name.

    Lowercases, replaces every run of non-alphanumeric characters with a
    single hyphen, trims hyphens from both ends and truncates at a word
    boundary. Falls back to ``"page"`` when nothing survives.

    Args:
        text: Free text such as a title.
        max_length: Maximum length of the returned slug.

    Returns:
        A page name satisfying :data:`PAGE_NAME_RE`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if len(slug) > max_length:
        slug = (
            slug[:max_length].rsplit("-", 1)[0] if "-" in slug[:max_length] else slug[:max_length]
        )
        slug = slug.strip("-")
    return slug or "page"


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a page into its frontmatter mapping and Markdown body.

    A frontmatter block is a leading ``---`` line, YAML, and a closing
    ``---`` line. Malformed YAML, or YAML that is not a mapping, is treated
    as "no frontmatter" so that the whole text becomes the body (pages
    without frontmatter are valid pages).

    Args:
        text: The complete page text.

    Returns:
        ``(frontmatter, body)``.
    """
    if not text.startswith("---\n") and not text.startswith("---\r\n"):
        return {}, text
    match = re.match(r"^---\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", text, re.DOTALL)
    if match is None:
        return {}, text
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        return {}, text
    if not isinstance(data, dict):
        return {}, text
    return {str(k): v for k, v in data.items()}, text[match.end() :]


def render_page(frontmatter: dict[str, Any], body: str) -> str:
    """Serialise frontmatter and body back into page text.

    Known keys are emitted first in the spec's order, then any extra keys.
    Values are stringified so timestamps stay quoted strings (YAML 1.1
    parsers would otherwise coerce them to ``datetime``).

    Args:
        frontmatter: Frontmatter mapping; may be empty.
        body: The Markdown body.

    Returns:
        The page text, ending in a single newline.
    """
    text = body.rstrip("\n") + "\n"
    if not frontmatter:
        return text
    ordered: dict[str, Any] = {}
    for key in FRONTMATTER_KEYS:
        if key in frontmatter and frontmatter[key] not in (None, ""):
            ordered[key] = str(frontmatter[key])
    for key, value in frontmatter.items():
        if key not in ordered and value not in (None, ""):
            ordered[key] = str(value)
    yaml_text = yaml.safe_dump(
        ordered, default_flow_style=False, sort_keys=False, allow_unicode=True
    )
    return f"---\n{yaml_text}---\n{text}"


class MemoryDir:
    """A flat directory of memory pages with safe, validated file access.

    Args:
        root: Directory that holds the pages. Created on first write.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def page_path(self, name: str) -> Path:
        """Return the on-disk path of page *name*, validating the name.

        Args:
            name: Page name, with or without a trailing ``.md``.

        Raises:
            ValueError: If the name violates the page filename rules, the
                entry is a symlink, or the path would escape the memory
                directory.
        """
        if name.endswith(".md"):
            name = name[:-3]
        if not is_valid_page_name(name):
            raise ValueError(
                f"Invalid page name {name!r}: use lowercase letters, digits and hyphens, "
                "starting and ending with a letter or digit (e.g. 'carbon-fibre-woks')."
            )
        path = self.root / f"{name}.md"
        if path.is_symlink():
            raise ValueError(f"Page {path.name} is a symlink; symlinked pages are not allowed.")
        if path.resolve().parent != self.root:
            raise ValueError(f"Page path {path} escapes the memory directory {self.root}.")
        return path

    def page_names(self) -> list[str]:
        """Return the sorted names of every page in the directory.

        Sub-directories, symlinks, non-``.md`` files and debris files are skipped.
        """
        if not self.root.is_dir():
            return []
        names: list[str] = []
        for path in self.root.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.suffix != ".md"
                or is_debris(path.name)
            ):
                continue
            if is_valid_page_name(path.stem):
                names.append(path.stem)
        return sorted(names)

    def read(self, name: str) -> Page:
        """Load page *name* from disk.

        Args:
            name: Page name, with or without ``.md``.

        Raises:
            FileNotFoundError: If the page does not exist.
            ValueError: If the name is invalid.
        """
        path = self.page_path(name)
        raw = read_page_text(path)
        frontmatter, body = split_frontmatter(raw)
        return Page(name=path.stem, frontmatter=frontmatter, body=body, raw=raw)

    def write(
        self,
        name: str,
        body: str,
        title: str = "",
        summary: str = "",
        extra: dict[str, Any] | None = None,
    ) -> Page:
        """Create or replace page *name* with generated frontmatter.

        On create, ``uuid`` and ``created`` are generated. On update, the
        stored ``uuid`` and ``created`` are always preserved (incoming values
        for them are ignored), ``title`` and ``summary`` are kept unless
        overridden, and ``updated`` is refreshed. If *body* itself starts with
        a frontmatter block, its remaining values are merged in.

        Args:
            name: Page name (``.md`` optional).
            body: Markdown body, optionally with its own frontmatter.
            title: Human-readable title; defaults to the name on create.
            summary: One-sentence summary for search listings.
            extra: Additional frontmatter keys to store (e.g. ``source``).

        Returns:
            The page as written.

        Raises:
            ValueError: If the name is invalid or the body is empty.
        """
        path = self.page_path(name)
        incoming, body = split_frontmatter(body)
        if not body.strip():
            raise ValueError("Refusing to write an empty page.")
        frontmatter: dict[str, Any] = {}
        if path.exists():
            frontmatter = split_frontmatter(read_page_text(path))[0]
            for identity_key in ("uuid", "created"):
                if identity_key in frontmatter:
                    incoming.pop(identity_key, None)
        frontmatter.update({k: v for k, v in incoming.items() if v not in (None, "")})
        if extra:
            frontmatter.update(extra)
        if title:
            frontmatter["title"] = title
        if summary:
            frontmatter["summary"] = summary
        frontmatter.setdefault("title", path.stem.replace("-", " "))
        frontmatter.setdefault("uuid", str(uuid.uuid4()))
        frontmatter.setdefault("created", now_iso())
        frontmatter["updated"] = now_iso()
        raw = render_page(frontmatter, body)
        # Atomic publish (stage + os.replace): the memory directory is shared
        # by parallel sub-agents and other daemon processes, whose
        # memory_read / memory_pull / index sync would otherwise observe the
        # empty or half-written file a plain truncate-then-write exposes.
        # create_mode=0o666: pages are deliberately plain documents — a new
        # page gets Path.write_text's umask-derived bits (helper default
        # 0o600 is for secret-bearing files); an existing page keeps its mode.
        atomic_write_text(path, raw, create_mode=0o666)
        return Page(name=path.stem, frontmatter=frontmatter, body=body, raw=raw)

    def delete(self, name: str) -> None:
        """Delete page *name*.

        Args:
            name: Page name (``.md`` optional).

        Raises:
            FileNotFoundError: If the page does not exist.
            ValueError: If the name is invalid.
        """
        self.page_path(name).unlink()

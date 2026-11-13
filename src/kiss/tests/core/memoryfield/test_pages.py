"""End-to-end tests for memoryfield pages: frontmatter, names and directory operations.

Deliberately uncovered: the ``resolve().parent != self.root`` guard in
``MemoryDir.page_path``. With symlinked entries rejected first, it only fires
if the directory tree is swapped underneath a live ``MemoryDir``; it stays as
defence in depth.
"""

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from types import FrameType

import pytest
import yaml

from kiss.core.memoryfield.pages import (
    MemoryDir,
    is_debris,
    is_valid_page_name,
    now_iso,
    render_page,
    slugify,
    split_frontmatter,
)

# --- names ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name,valid",
    [
        ("carbon-fibre-woks", True),
        ("a", True),
        ("a1", True),
        ("-leading", False),
        ("trailing-", False),
        ("Upper", False),
        ("under_score", False),
        ("with space", False),
        ("", False),
        ("../escape", False),
    ],
)
def test_is_valid_page_name(name: str, valid: bool) -> None:
    assert is_valid_page_name(name) is valid


def test_slugify_rules() -> None:
    assert slugify("Carbon Fibre Woks!") == "carbon-fibre-woks"
    assert slugify("  --Hello, World--  ") == "hello-world"
    assert slugify("###") == "page"
    long = slugify("word " * 40, max_length=20)
    assert len(long) <= 20 and is_valid_page_name(long)
    # A single very long word is cut hard because there is no hyphen to break on.
    assert slugify("a" * 100, max_length=10) == "a" * 10


def test_is_debris() -> None:
    assert is_debris(".DS_Store")
    assert is_debris("notes.md~")
    assert is_debris("notes.sync-conflict-20260101-abc.md")
    assert is_debris("Thumbs.db") and is_debris("desktop.ini")
    assert not is_debris("notes.md")


# --- frontmatter ---------------------------------------------------------


def test_split_frontmatter_roundtrip_and_key_order() -> None:
    fm = {
        "title": "T",
        "uuid": "u",
        "summary": "S",
        "created": "2026-01-01T00:00:00Z",
        "updated": "2026-01-02T00:00:00Z",
        "source": "x",
    }
    text = render_page(fm, "Body\n\nMore")
    assert text.startswith("---\ntitle: T\nuuid: u\nsummary: S\ncreated: '2026-01-01T00:00:00Z'\n")
    parsed, body = split_frontmatter(text)
    assert parsed == fm  # timestamps stay strings because they are quoted
    assert body == "Body\n\nMore\n"


def test_memory_frontmatter_preserves_unicode_line_separators(tmp_path: Path) -> None:
    """YAML's NEL folding must not silently turn a stored title into different text."""
    memory = MemoryDir(tmp_path)
    title = "café\u0085next\u2028line\u2029paragraph"
    memory.write("unicode", "body", title=title, summary=title)
    page = memory.read("unicode")
    assert page.title == title
    assert page.summary == title
    assert page.body == "body\n"


def test_render_page_without_frontmatter_and_empty_values() -> None:
    assert render_page({}, "just body") == "just body\n"
    assert (
        render_page({"title": "", "summary": None, "extra": 3}, "b") == "---\nextra: '3'\n---\nb\n"
    )


def test_split_frontmatter_edge_cases() -> None:
    assert split_frontmatter("no frontmatter here") == ({}, "no frontmatter here")
    assert split_frontmatter("---\nunterminated: yes\n") == ({}, "---\nunterminated: yes\n")
    malformed = "---\n: [unbalanced\n---\nbody"
    assert split_frontmatter(malformed) == ({}, malformed)
    not_mapping = "---\n- a\n- b\n---\nbody"
    assert split_frontmatter(not_mapping) == ({}, not_mapping)
    crlf = "---\r\ntitle: X\r\n---\r\nbody"
    assert split_frontmatter(crlf) == ({"title": "X"}, "body")
    assert split_frontmatter("---\ntitle: X\n---") == ({"title": "X"}, "")
    # Unquoted timestamps are coerced by YAML 1.1; we tolerate that on read.
    coerced, _ = split_frontmatter("---\ncreated: 2026-01-01T00:00:00Z\n---\nb")
    assert not isinstance(coerced["created"], str)


def test_now_iso_is_quoted_safe() -> None:
    value = now_iso()
    assert value.endswith("Z") and len(value) == 20
    assert isinstance(
        yaml.safe_load(render_page({"created": value}, "b").split("---")[1])["created"], str
    )


# --- MemoryDir -----------------------------------------------------------


def test_memory_dir_write_read_update_delete(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path / "mem")
    assert memory.page_names() == []  # directory does not exist yet

    page = memory.write("woks", "Carbon fibre woks.", title="Woks", summary="Cookware")
    assert page.name == "woks" and page.title == "Woks" and page.summary == "Cookware"
    first_uuid, created = page.frontmatter["uuid"], page.frontmatter["created"]
    assert (tmp_path / "mem" / "woks.md").read_text().startswith("---\ntitle: Woks\n")

    # Update preserves identity, refreshes ``updated``, keeps title/summary unless overridden.
    updated = memory.write("woks.md", "New body")
    assert updated.frontmatter["uuid"] == first_uuid
    assert updated.frontmatter["created"] == created
    assert updated.title == "Woks" and updated.summary == "Cookware"
    assert updated.body == "New body"
    assert memory.read("woks").raw == updated.raw

    # Incoming frontmatter is merged; extra keys are stored; identity keys are protected.
    merged = memory.write(
        "woks",
        "---\nsummary: From body\nignored: ''\nuuid: spoofed\n"
        "created: '1999-01-01T00:00:00Z'\n---\nBody 3",
        extra={"source": "s"},
    )
    assert merged.summary == "From body" and merged.frontmatter["source"] == "s"
    assert "ignored" not in merged.frontmatter
    assert merged.frontmatter["uuid"] == first_uuid and merged.frontmatter["created"] == created
    # On create, an incoming uuid is honoured (e.g. importing a page from another memoryfield).
    imported = memory.write("imported", "---\nuuid: keep-me\n---\nbody")
    assert imported.frontmatter["uuid"] == "keep-me"

    # Default title is derived from the name.
    other = memory.write("finnish-bureaucracy", "DVV")
    assert other.title == "finnish bureaucracy" and other.summary == ""
    assert memory.page_names() == ["finnish-bureaucracy", "imported", "woks"]

    memory.delete("woks.md")
    assert memory.page_names() == ["finnish-bureaucracy", "imported"]
    with pytest.raises(FileNotFoundError):
        memory.read("woks")
    with pytest.raises(FileNotFoundError):
        memory.delete("woks")
    # The deletion is recorded (for syncing with another machine) and undone by a new write.
    tombstone = tmp_path / "mem" / ".tombstones" / "woks"
    deleted_at = tombstone.read_text().strip()
    assert datetime.strptime(deleted_at, "%Y-%m-%dT%H:%M:%SZ")
    assert memory.write("woks", "back again").frontmatter["updated"] >= deleted_at
    assert not tombstone.exists()
    assert memory.page_names() == ["finnish-bureaucracy", "imported", "woks"]
    with pytest.raises(ValueError, match="Invalid page name"):
        memory.delete("Bad Name")


class _PagePathCalls:
    """Profile hook: counts the calls of ``MemoryDir.page_path`` made while it is installed."""

    count = 0

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is MemoryDir.page_path.__code__:
            self.count += 1


def _page_path_calls(operation: Callable[..., object], *args: str) -> int:
    """Run ``operation(*args)`` and return how many times it validated a page path."""
    calls = _PagePathCalls()
    previous = sys.getprofile()
    sys.setprofile(calls)
    try:
        operation(*args)
    finally:
        sys.setprofile(previous)
    return calls.count


def test_write_and_delete_validate_the_page_path_once(tmp_path: Path) -> None:
    """The tombstone's place comes from the validated page path, not from a second validation."""
    memory = MemoryDir(tmp_path)
    tombstone = tmp_path / ".tombstones" / "a-page"
    assert _page_path_calls(memory.write, "a-page", "body") == 1
    assert _page_path_calls(memory.delete, "a-page") == 1 and tombstone.exists()
    assert _page_path_calls(memory.write, "a-page", "back again") == 1 and not tombstone.exists()


def test_memory_dir_rejects_bad_names_and_empty_bodies(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    with pytest.raises(ValueError, match="Invalid page name"):
        memory.write("Bad Name", "x")
    with pytest.raises(ValueError, match="Invalid page name"):
        memory.page_path("../etc/passwd")
    with pytest.raises(ValueError, match="empty page"):
        memory.write("empty", "   \n")
    with pytest.raises(ValueError, match="empty page"):
        memory.write("only-frontmatter", "---\ntitle: T\n---\n")
    assert not list(tmp_path.iterdir())


def test_memory_dir_rejects_symlinked_pages(tmp_path: Path) -> None:
    """Symlinked entries are never pages: not listed, not readable, not writable, not deletable."""
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    root = tmp_path / "mem"
    root.mkdir()
    memory = MemoryDir(root)
    memory.write("target", "real page")
    (root / "evil.md").symlink_to(outside)
    (root / "alias.md").symlink_to(root / "target.md")
    assert memory.page_names() == ["target"]
    for name in ("evil", "alias"):
        with pytest.raises(ValueError, match="symlink"):
            memory.read(name)
        with pytest.raises(ValueError, match="symlink"):
            memory.write(name, "overwrite attempt")
        with pytest.raises(ValueError, match="symlink"):
            memory.delete(name)
    assert outside.read_text() == "secret"
    assert memory.read("target").body == "real page\n"
    assert (root / "alias.md").is_symlink()


def test_memory_dir_refuses_a_symlinked_tombstone_directory(tmp_path: Path) -> None:
    """Neither a write nor a delete touches anything through a ``.tombstones`` symlink."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "lesson").write_text("unrelated file\n")
    root = tmp_path / "mem"
    memory = MemoryDir(root)
    memory.write("lesson", "a page")
    (root / ".tombstones").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symlink"):
        memory.write("lesson", "rewritten")
    with pytest.raises(ValueError, match="symlink"):
        memory.delete("lesson")
    assert memory.read("lesson").body == "a page\n"
    assert (elsewhere / "lesson").read_text() == "unrelated file\n"
    assert sorted(p.name for p in elsewhere.iterdir()) == ["lesson"]


class _PlantsALink:
    """Profile hook: makes *page* a link to *outside* as a write renders its validated page."""

    def __init__(self, page: Path, outside: Path) -> None:
        self.page = page
        self.outside = outside

    def __call__(self, frame: FrameType, event: str, arg: object) -> None:
        if event == "call" and frame.f_code is render_page.__code__:
            self.page.symlink_to(self.outside)


def test_memory_writes_replace_a_link_instead_of_following_it(tmp_path: Path) -> None:
    """A symlinked tombstone, or a page made a link once its path was validated, is replaced:
    written through, the memory's text would land in the file the link names."""
    outside = tmp_path / "outside"
    outside.write_text("unrelated file\n")
    root = tmp_path / "mem"
    memory = MemoryDir(root)
    memory.write("lesson", "a page")
    tombstone = root / ".tombstones" / "lesson"
    tombstone.parent.mkdir(exist_ok=True)
    tombstone.symlink_to(outside)
    memory.delete("lesson")
    assert not tombstone.is_symlink()
    assert datetime.strptime(tombstone.read_text().strip(), "%Y-%m-%dT%H:%M:%SZ")
    previous = sys.getprofile()
    sys.setprofile(_PlantsALink(root / "raced.md", outside))
    try:
        memory.write("raced", "a raced page")
    finally:
        sys.setprofile(previous)
    assert not (root / "raced.md").is_symlink() and memory.read("raced").body == "a raced page\n"
    assert outside.read_text() == "unrelated file\n"


def test_memory_dir_escape_guard_for_resolved_paths(tmp_path: Path) -> None:
    """The resolve() guard still fires when the memory root itself is reached through a symlink."""
    real_root = tmp_path / "real"
    real_root.mkdir()
    link_root = tmp_path / "link"
    link_root.symlink_to(real_root, target_is_directory=True)
    memory = MemoryDir(link_root)
    assert memory.root == real_root.resolve()
    page = memory.write("ok", "body")
    assert page.name == "ok" and memory.page_names() == ["ok"]


def test_memory_dir_reads_invalid_utf8_with_replacement(tmp_path: Path) -> None:
    (tmp_path / "damaged.md").write_bytes(b"---\ntitle: D\n---\n\xff\xfe bytes")
    page = MemoryDir(tmp_path).read("damaged")
    assert page.title == "D" and "bytes" in page.body and "\ufffd" in page.body
    updated = MemoryDir(tmp_path).write("damaged", "repaired")
    assert updated.title == "D" and updated.body == "repaired"


def test_memory_dir_page_names_skips_non_pages(tmp_path: Path) -> None:
    memory = MemoryDir(tmp_path)
    memory.write("real", "body")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "Upper.md").write_text("x")
    (tmp_path / "real.md~").write_text("x")
    (tmp_path / "real.sync-conflict-1.md").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "nested.md").write_text("x")
    (tmp_path / "text-embedding-3-small.sqlite3").write_bytes(b"")
    assert memory.page_names() == ["real"]


def test_memory_dir_reads_pages_without_frontmatter(tmp_path: Path) -> None:
    # newline="" so Windows does not turn the LF into CRLF: read() keeps
    # the raw bytes, so the body would otherwise come back as "Just prose.\r\n".
    (tmp_path / "plain.md").write_text("Just prose.\n", encoding="utf-8", newline="")
    page = MemoryDir(tmp_path).read("plain")
    assert page.frontmatter == {} and page.title == "plain" and page.body == "Just prose.\n"

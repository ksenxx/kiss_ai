# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The skills.sh registry and GitHub archives behind the Skills panel.

skills.sh answers the search, the all-time leaderboard and the partner
audits of a skill without an account; the files of a skill come out of
its repository's archive at codeload.github.com, the one complete copy
(assets included).  Both base URLs are read from the environment at
call time (``KISS_SKILLS_REGISTRY_URL``, ``KISS_SKILLS_ARCHIVE_URL``),
so a test serves them from a local HTTP server.

The servers and the archive are untrusted.  A download has a size limit
and one deadline for the whole exchange, redirects and the head of each
reply included.  The archive is walked with limits on its members, on
where they lie in it, on what ``tarfile`` may read for one of them and
on the time of a walk, and only the plain files of the one skill
folder are written, into a directory the caller names.  Nothing in it
is executed.
"""

from __future__ import annotations

import gzip
import http.client
import io
import json
import os
import re
import socket
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Buffer, Iterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any, BinaryIO

from kiss.agents.sorcar.skill_rules import reveal_hidden

SEARCH_LIMIT = 50
"""The most rows :func:`search` returns."""

# One connect or read of a download, and the download as a whole.
_TIMEOUT_SECONDS = 20.0
_DOWNLOAD_SECONDS = 300.0
_HEADERS = {"User-Agent": "kiss-sorcar-skills", "Accept": "application/json, */*"}
_MAX_JSON_BYTES = 4 * 1024 * 1024
# The archive as a whole: the download, then what a walk may meet:
# members, and how far into the unpacked archive they lie.
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
_MAX_ARCHIVE_MEMBERS = 100_000
_MAX_UNPACKED_BYTES = 2 * 1024**3
# What tarfile may read to get from one member to the next: their
# headers, a long name, PAX records (a few kilobytes when git wrote
# them), and how many PAX records may hold for the whole archive (git
# writes one).
_MAX_HEADER_BYTES = 64 * 1024
_MAX_GLOBAL_RECORDS = 100
# One walk over the members: a hundred thousand honest ones take two
# seconds, as many forged PAX headers take minutes.
_WALK_SECONDS = 60.0
# The one skill folder that is written: its files, and the folders above
# one of them (each file could bring a chain of folders of its own).
_MAX_SKILL_FILES = 500
_MAX_SKILL_DEPTH = 32
_MAX_FILE_BYTES = 5 * 1024 * 1024
_MAX_SKILL_BYTES = 20 * 1024 * 1024
# The ``name`` of a SKILL.md's frontmatter, read from the head of the
# file without a YAML parser: a repository may hold thousands of these
# files, and hostile YAML takes a parser seconds per file.
_SKILL_MD_HEAD = 16 * 1024
_DECLARED_NAME_RE = re.compile(r"\A---[^\S\n]*\n(?:(?!---)[^\n]*\n)*?name:[ \t]*([^\n]*)")

# A GitHub ``owner/repo``; never ``.`` or ``..`` as the repository.
_SOURCE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]{0,38}/(?!\.{1,2}$)[A-Za-z0-9._-]{1,100}")
# A skill name of the Agent Skills specification; it becomes a directory name.
_NAME_RE = re.compile(r"(?=.{1,64}$)[a-z0-9]+(?:-[a-z0-9]+)*")
# In a file name: a Windows separator or drive, a control character, or
# a byte that is not UTF-8 (which tarfile reads as a lone surrogate).
_UNSAFE_NAME_RE = re.compile(r"[\\:\x00-\x1f\x7f\ud800-\udfff]")


class RegistryError(Exception):
    """A failure of the registry or of an archive, worded for the user.

    Attributes:
        status: The HTTP status that caused it, ``0`` when none did.
    """

    def __init__(self, message: str, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class _BoundedGzip(gzip.GzipFile):
    """The gzip stream of an archive, which gives out only what the walk has granted.

    ``tarfile`` reads whatever a header claims, and holds on to it
    while it reads on: the body of a GNU long-name or PAX header whole,
    the map of a sparse file block after block, a chain of such headers
    one inside the other.  A megabyte of archive could so make one call
    of it allocate gigabytes or read for minutes.  The walk says what
    may be read next (:func:`_members`, :func:`_content`), and the
    stream goes forwards only.

    Attributes:
        budget: The bytes that may still be read; at first those of
            the first member's headers, which ``tarfile.open`` reads.
    """

    budget = _MAX_HEADER_BYTES

    def read(self, size: int | None = -1) -> bytes:
        """Return up to *size* decompressed bytes.

        Raises:
            tarfile.ReadError: *size* is unbounded or over the budget.
        """
        if size is None or not 0 <= size <= self.budget:
            raise tarfile.ReadError("a header that claims more than a walk reads at once")
        self.budget -= size
        return super().read(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move to *offset* as ``GzipFile.seek`` does, unless that lies behind.

        Raises:
            tarfile.ReadError: *offset* lies before the position.  A
                gzip stream goes back by being read again from its
                start, and ``tarfile`` looks for the next member where
                a header says, also when it has read past that.
        """
        if whence == io.SEEK_SET and offset < self.tell():
            raise tarfile.ReadError("a header that sends the walk back")
        return super().seek(offset, whence)


# When the download of this thread must have ended (``time.monotonic()``).
# urllib and http.client build the connections and replies of a download
# themselves, so this is how those learn it.
_deadline: ContextVar[float] = ContextVar("skill_download_deadline")


def _time_left() -> float:
    """Return the seconds the next connect or read of this thread's download may take.

    Raises:
        TimeoutError: The download has no time left.
    """
    left = _deadline.get() - time.monotonic()
    if left <= 0:
        raise TimeoutError("it took too long to answer")
    return min(left, _TIMEOUT_SECONDS)


class _TimedReads(io.RawIOBase):
    """The reads of a reply's socket, each within the time its download has left.

    The timeout of a socket bounds one read, and a server that sends a
    byte every few seconds always meets it: the head of a reply, which
    http.client reads before the caller sees anything of it, could
    take hours.
    """

    def __init__(self, sock: socket.socket) -> None:
        self._sock = sock
        # Holds the socket open after the connection let go of it, as
        # http.client expects of the file of a reply.
        self._raw = sock.makefile("rb", buffering=0)

    def readable(self) -> bool:
        """Return ``True``: this is a stream to read from."""
        return True

    def readinto(self, buffer: Buffer, /) -> int | None:
        """Read once from the socket into *buffer*.

        Returns:
            The number of bytes read, ``0`` at the end of the stream.

        Raises:
            TimeoutError: Nothing arrived in :data:`_TIMEOUT_SECONDS`,
                or the download has no time left.
        """
        self._sock.settimeout(_time_left())
        return self._raw.readinto(buffer)

    def close(self) -> None:
        """Close the stream and let go of the socket."""
        self._raw.close()
        super().close()


class _Reply(http.client.HTTPResponse):
    """A reply whose head and body are read within the time its download has left."""

    def __init__(self, sock: socket.socket, *args: Any, **kwargs: Any) -> None:
        super().__init__(sock, *args, **kwargs)
        self.fp.close()
        self.fp = io.BufferedReader(_TimedReads(sock))


class _Http(http.client.HTTPConnection):
    """An HTTP connection whose replies are :class:`_Reply`."""

    response_class = _Reply


class _Https(http.client.HTTPSConnection):
    """An HTTPS connection whose replies are :class:`_Reply`."""

    response_class = _Reply


class _TimedHandler(urllib.request.AbstractHTTPHandler):
    """Opens HTTP and HTTPS addresses within the time their download has left."""

    def http_open(self, request: urllib.request.Request) -> http.client.HTTPResponse:
        """Send *request* over HTTP and return the reply."""
        request.timeout = _time_left()
        return self.do_open(_Http, request)

    def https_open(self, request: urllib.request.Request) -> http.client.HTTPResponse:
        """Send *request* over HTTPS, the certificate verified, and return the reply."""
        request.timeout = _time_left()
        return self.do_open(_Https, request)

    http_request = https_request = urllib.request.AbstractHTTPHandler.do_request_


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follows a redirect without reading the body of the reply that names it."""

    def redirect_request(
        self, req: urllib.request.Request, fp: Any, code: int, msg: str,
        headers: http.client.HTTPMessage, newurl: str,
    ) -> urllib.request.Request | None:
        """Return the request for *newurl*, as urllib builds it.

        urllib then reads what is left of the redirecting reply, whole
        and whatever its size; *fp* is closed first, so nothing is.
        """
        fp.close()
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener() -> urllib.request.OpenerDirector:
    """Return what opens a download: HTTP and HTTPS only, through the environment's proxies."""
    opener = urllib.request.OpenerDirector()
    for handler in (
        urllib.request.ProxyHandler(), urllib.request.UnknownHandler(), _TimedHandler(),
        urllib.request.HTTPDefaultErrorHandler(), _Redirects(),
        urllib.request.HTTPErrorProcessor(),
    ):
        opener.add_handler(handler)
    return opener


def registry_url() -> str:
    """Return the base URL of skills.sh (``KISS_SKILLS_REGISTRY_URL`` replaces it)."""
    return (os.environ.get("KISS_SKILLS_REGISTRY_URL") or "https://skills.sh").rstrip("/")


def archive_url() -> str:
    """Return the base URL of GitHub's archives (``KISS_SKILLS_ARCHIVE_URL`` replaces it)."""
    return (os.environ.get("KISS_SKILLS_ARCHIVE_URL") or "https://codeload.github.com").rstrip("/")


def parse_skill_id(skill: str) -> tuple[str, str] | None:
    """Split the registry id ``owner/repo/skill`` of a GitHub-backed skill.

    Args:
        skill: The id as the registry gives it.

    Returns:
        ``(source, name)``: the repository ``owner/repo`` and the
        skill's name; ``None`` when the source is not a GitHub
        repository or the name is not a valid skill name (lower-case
        letters, digits and single hyphens, at most 64 characters).
    """
    source, _, name = skill.rpartition("/")
    if _SOURCE_RE.fullmatch(source) and _NAME_RE.fullmatch(name):
        return source, name
    return None


def skill_slug(name: str) -> str:
    """Return the id the registry gives a skill whose frontmatter name is *name*.

    The rule of the skills CLI (``toSkillSlug``): lower case, white
    space and underscores become hyphens, every other character but
    letters, digits and hyphens is dropped, hyphens are collapsed and
    trimmed.

    Args:
        name: The ``name`` of a ``SKILL.md`` frontmatter.

    Returns:
        The slug, ``""`` when nothing of *name* remains.
    """
    slug = re.sub(r"[^a-z0-9-]", "", re.sub(r"[\s_]+", "-", name.lower()))
    return re.sub(r"-+", "-", slug).strip("-")


def _copy_url(url: str, service: str, limit: int, out: BinaryIO) -> None:
    """Copy the body at *url* into *out*.

    Args:
        url: The URL to GET.
        service: What to call the server in an error ("skills.sh").
        limit: The most bytes the body may have.
        out: Where the body goes.

    Raises:
        RegistryError: An HTTP error (its ``status`` set), an
            unreachable or stalled server, a download (its redirects
            included) that takes longer than :data:`_DOWNLOAD_SECONDS`,
            or a body over *limit*.
    """
    _deadline.set(time.monotonic() + _DOWNLOAD_SECONDS)
    try:
        request = urllib.request.Request(url, headers=_HEADERS)
        with _opener().open(request) as response:
            size = 0
            while chunk := response.read1(1 << 16):
                size += len(chunk)
                if size > limit:
                    raise RegistryError(f"{service} sent more data than is accepted.")
                out.write(chunk)
    except urllib.error.HTTPError as error:
        error.close()
        if error.code == 429:
            message = f"{service} is limiting requests; try again in a minute."
        else:
            message = f"{service} answered with an error (HTTP {error.code})."
        raise RegistryError(message, error.code) from error
    # ValueError: an address urllib takes for no URL (one without a scheme).
    except (OSError, ValueError, http.client.HTTPException) as error:
        reason = getattr(error, "reason", error)
        raise RegistryError(f"{service} could not be reached ({reason}).") from error


def _get_json(path: str) -> Any:
    """Return the JSON document at *path* of the registry."""
    body = io.BytesIO()
    _copy_url(registry_url() + path, "skills.sh", _MAX_JSON_BYTES, body)
    try:
        return json.loads(body.getvalue())
    # RecursionError: a reply of nothing but opening brackets.
    except (ValueError, RecursionError) as error:
        raise RegistryError("skills.sh sent a reply that could not be read.") from error


def _listed(reply: Any, key: str) -> list[dict[str, Any]]:
    """Return the objects of the list *key* of the registry's *reply*."""
    items = reply.get(key) if isinstance(reply, dict) else None
    if not isinstance(items, list):
        raise RegistryError("skills.sh sent a reply that could not be read.")
    return [item for item in items if isinstance(item, dict)]


def _installs(row: dict[str, Any]) -> int:
    """Sort key of :func:`search`: the most installed first."""
    return -int(row["installs"])


def search(query: str) -> list[dict[str, Any]]:
    """Return the registry's skills for *query*, the most installed first.

    The registry answers a query of two characters or more in relevance
    order; a shorter one gets its all-time leaderboard instead.  Left
    out: copies the registry marks as duplicates, skills whose source
    is not a GitHub repository, and names that are not valid skill
    names (:func:`parse_skill_id`), none of which can be installed.

    Args:
        query: What the user typed.

    Returns:
        At most :data:`SEARCH_LIMIT` rows ``{"id", "name", "source",
        "installs", "url"}``: the id ``owner/repo/skill``, the name the
        skill gets when installed, and its page on the registry.

    Raises:
        RegistryError: The registry could not be asked.
    """
    query = query.strip()
    if len(query) >= 2:
        # errors: half a surrogate pair, which no URL carries, is sent as "?".
        fields = urllib.parse.urlencode({"q": query, "limit": 100}, errors="replace")
        path = "/api/search?" + fields
    else:
        path = "/api/skills/all-time/0"
    rows = []
    for item in _listed(_get_json(path), "skills"):
        skill = f"{item.get('source')}/{item.get('skillId')}"
        parsed = None if item.get("isDuplicate") else parse_skill_id(skill)
        if parsed:
            installs = item.get("installs")
            rows.append({
                "id": skill,
                "name": parsed[1],
                "source": parsed[0],
                "installs": int(installs) if isinstance(installs, int) else 0,
                "url": f"{registry_url()}/{skill}",
            })
    return sorted(rows, key=_installs)[:SEARCH_LIMIT]


def audits(skill: str) -> list[dict[str, Any]] | None:
    """Return the partner audits the registry holds for *skill*.

    Args:
        skill: A registry id that :func:`parse_skill_id` accepts.

    Returns:
        One object per auditor (``provider``, ``status`` of ``pass``,
        ``warn`` or ``fail``, ``summary``); ``None`` when the registry
        has no audit of the skill (HTTP 404).

    Raises:
        RegistryError: The registry could not be asked.
    """
    try:
        reply = _get_json(f"/api/v1/skills/audit/{skill}")
    except RegistryError as error:
        if error.status == 404:
            return None
        raise
    return _listed(reply, "audits")


def _members(archive: tarfile.TarFile, stream: _BoundedGzip) -> Iterator[tarfile.TarInfo]:
    """Yield the members of *archive*, which reads *stream*, within the limits of a walk.

    Raises:
        RegistryError: A member path is absolute or has a ``..`` part,
            a member has a negative size (which would lower the sums)
            or does not end after it began (the walk would turn back),
            or the archive has more members, more PAX records of its
            own, a member that ends further in or takes longer than a
            walk accepts.  Where a member ends is what its headers
            claim, so one that claims terabytes is never skipped over.
    """
    deadline = time.monotonic() + _WALK_SECONDS
    count = 0
    while member := archive.next():
        if (
            member.size < 0 or archive.offset <= member.offset
            or member.name.startswith("/") or ".." in member.name.split("/")
        ):
            raise RegistryError("The skill's archive holds an unsafe entry.")
        count += 1
        if (
            count > _MAX_ARCHIVE_MEMBERS or archive.offset > _MAX_UNPACKED_BYTES
            or len(archive.pax_headers) > _MAX_GLOBAL_RECORDS or time.monotonic() > deadline
        ):
            raise RegistryError("The skill's repository is too large to unpack.")
        yield member
        stream.budget = _MAX_HEADER_BYTES


def _plain(member: tarfile.TarInfo) -> bool:
    """Return whether *member* is a file stored in one piece.

    ``tarfile`` counts a sparse file as a regular one and reads it
    piece by piece from where its map says, forwards or back, and a
    step back in a gzip stream is read again from the start.
    """
    return member.isreg() and not member.issparse()


def _content(stream: _BoundedGzip, member: tarfile.TarInfo, limit: int) -> bytes:
    """Return the first *limit* bytes, or all there are, of the plain file *member*.

    Of a file the archive ends inside of, what there is comes back;
    the walk then fails at its next member.
    """
    size = min(limit, member.size)
    stream.budget = size
    stream.seek(member.offset_data)
    return stream.read(size)


def _open(stream: _BoundedGzip) -> tarfile.TarFile:
    """Return the archive in *stream*, opened to be walked once: no member is kept."""
    return tarfile.TarFile(fileobj=stream, mode="r", stream=True)


def _skill_folder(path: Path, name: str) -> str:
    """Return the folder of the archive at *path* that holds the skill *name*.

    The folder whose ``SKILL.md`` frontmatter names the skill (by its
    registry id, :func:`skill_slug`); when no file does, the folder
    called *name*.  Of several, the one nearest the top of the
    repository, then the first by name.
    """
    # The best folder so far, as (is only called *name*, its depth, the
    # folder): the least of them is the one to return.  Not a list of
    # them all: a path may be 60 KB and a repository may hold 100,000.
    best: tuple[bool, int, str] | None = None
    with _BoundedGzip(path) as stream, _open(stream) as archive:
        for member in _members(archive, stream):
            folder, _, base = member.name.rpartition("/")
            if base == "SKILL.md" and folder and _plain(member):
                head = _content(stream, member, _SKILL_MD_HEAD)
                declared = _DECLARED_NAME_RE.match(head.decode("utf-8-sig", "replace"))
                declares = declared is not None and skill_slug(declared.group(1)) == name
                if declares or folder.rpartition("/")[2] == name:
                    found = (not declares, folder.count("/"), folder)
                    best = min(best or found, found)
    if best is None:
        raise RegistryError(f"The repository has no skill named {name}.")
    return best[2]


def _unpack(path: Path, folder: str, dest: Path) -> None:
    """Write the files of *folder* of the archive at *path* into the new directory *dest*.

    Raises:
        RegistryError: The folder holds something other than plain
            files and directories (a symbolic or hard link, a device,
            a sparse file), a file name with a backslash, a colon, a
            control or invisible character or bytes that are not
            UTF-8, or more than a skill may have.
    """
    prefix = folder + "/"
    files = size = 0
    dest.mkdir()
    with _BoundedGzip(path) as stream, _open(stream) as archive:
        for member in _members(archive, stream):
            if not member.name.startswith(prefix) or member.isdir():
                continue
            relative = member.name[len(prefix):]
            if (
                not _plain(member) or _UNSAFE_NAME_RE.search(relative)
                or reveal_hidden(relative) != relative
            ):
                raise RegistryError(
                    f"The skill holds {relative[:80]!r}, which is not a plain file (a link, a "
                    "device or an unsafe name)."
                )
            files += 1
            size += member.size
            if (
                files > _MAX_SKILL_FILES or member.size > _MAX_FILE_BYTES
                or size > _MAX_SKILL_BYTES or relative.count("/") > _MAX_SKILL_DEPTH
            ):
                raise RegistryError(
                    f"The skill is too large to install (more than {_MAX_SKILL_FILES} files, "
                    f"{_MAX_FILE_BYTES >> 20} MB in one file, {_MAX_SKILL_BYTES >> 20} MB in all "
                    f"or {_MAX_SKILL_DEPTH} folders inside one another)."
                )
            target = dest.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_content(stream, member, member.size))
            target.chmod(0o755 if member.mode & 0o100 else 0o644)


def fetch_skill(source: str, name: str, directory: Path) -> Path:
    """Download the repository *source* and unpack its skill *name*.

    Args:
        source: The GitHub repository, ``owner/repo``.
        name: The skill's registry id within the repository.
        directory: An existing directory of the caller's; the archive
            is downloaded into it and the skill unpacked beside it.

    Returns:
        The unpacked skill folder, ``directory / name``.

    Raises:
        RegistryError: The repository could not be downloaded, holds no
            such skill, or its archive is refused (see :func:`_members`
            and :func:`_unpack`).
    """
    archive = directory / "archive.tar.gz"
    try:
        with open(archive, "wb") as out:
            _copy_url(f"{archive_url()}/{source}/tar.gz/HEAD", "GitHub", _MAX_ARCHIVE_BYTES, out)
    except RegistryError as error:
        if error.status == 404:
            raise RegistryError(f"The repository {source} was not found on GitHub.") from error
        raise
    try:
        _unpack(archive, _skill_folder(archive, name), directory / name)
    # ValueError: tarfile's own, on a malformed sparse-file header;
    # RecursionError: on a thousand long-name or PAX headers in a row.
    except (
        tarfile.TarError, EOFError, zlib.error, OSError, ValueError, RecursionError,
    ) as error:
        raise RegistryError("The skill's archive could not be unpacked.") from error
    return directory / name

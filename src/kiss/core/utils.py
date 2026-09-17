# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Utility functions for the KISS core module."""

import html as html_module
import logging
import os
import posixpath
import re
import stat
import string
import tempfile
import uuid
from pathlib import Path
from typing import IO, Any, cast

import yaml
from yaml.nodes import ScalarNode

from kiss.core import config as config_module

logger = logging.getLogger(__name__)


class _KissDumper(yaml.Dumper):
    """PyYAML dumper carrying KISS's formatting, and only KISS's.

    Registering a representer on ``yaml.Dumper`` (what
    ``yaml.add_representer`` does by default) mutates PyYAML for the
    whole process, so every unrelated ``yaml.dump`` in the interpreter —
    including an embedding application's own — silently changes shape
    the moment anything imports KISS.  Subclassing keeps the change
    where it belongs.
    """


def _str_presenter(dumper: yaml.Dumper, data: str) -> ScalarNode:
    """Represent genuinely multi-line strings as literal blocks.

    Trajectories and agent results are read by humans, so a prompt or a
    tool result is far more legible as a ``|`` block than as one long
    escaped line.  Single-line values (and every mapping key) keep the
    default style: forcing a block scalar on them adds two lines of
    noise per key for no benefit.
    """
    style = "|" if "\n" in data else None
    return dumper.represent_scalar(  # type: ignore[reportUnknownMemberType]
        "tag:yaml.org,2002:str",
        data,
        style=style,
    )


_KissDumper.add_representer(str, _str_presenter)


def dump_yaml(data: Any, stream: IO[str] | None = None, **kwargs: Any) -> Any:
    """Serialize *data* to YAML with KISS's human-readable string style.

    Args:
        data: The object to serialize.
        stream: Optional destination stream; when ``None`` the YAML is
            returned as a string.
        **kwargs: Extra options forwarded to :func:`yaml.dump` (e.g.
            ``indent``, ``sort_keys``).

    Returns:
        The YAML string when *stream* is ``None``, otherwise ``None``.
    """
    return yaml.dump(data, stream, Dumper=_KissDumper, **kwargs)


def atomic_write_text(
    target: Path,
    content: str,
    mode: int | None = None,
    create_mode: int = 0o600,
) -> None:
    """Write *content* to *target* so readers never see a partial file.

    The content is staged in a sibling temp file and then
    ``os.replace``-d into position, which is atomic on every supported
    platform.  A plain ``open(path, "w")`` truncates immediately and
    then fills the file incrementally, so a concurrent reader — the
    trajectory visualizer, another daemon, the VS Code extension — can
    observe an empty or half-written document.

    Permission semantics, in priority order:

    1. An explicit *mode* always wins, for new AND existing targets
       (secret-bearing callers force ``0o600`` regardless of history).
    2. An existing target keeps its current bits — a deliberately
       ``chmod``-ed file is never clobbered on update (``os.replace``
       publishes the STAGED inode's mode, so this must be copied over
       explicitly).
    3. A NEW target is created with *create_mode* filtered by the
       process umask.  The default is ``0o600``: several callers store
       secrets (``config.json`` holds ``remote_password`` and
       ``tunnel_token``, trajectories hold prompts and tool results,
       ``MY_MODELS.json`` holds API keys), so private-by-default is the
       only safe default — exactly what the pre-consolidation
       ``mkstemp``-staged helper published.  Callers whose files are
       meant to be group/world-readable (memory pages) pass
       ``create_mode=0o666`` to get plain ``Path.write_text`` umask
       semantics for fresh files.

    Args:
        target: Destination path; its parent directory is created.
        content: The full text to write.
        mode: Optional permission bits to force on the result (e.g.
            ``0o600`` for files holding secrets).  Best effort: a
            filesystem that refuses ``chmod`` is not treated as a write
            failure.
        create_mode: Permission bits (before the process umask) used
            only when *target* does not exist and *mode* is ``None``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    preserved: int | None = mode
    if preserved is None:
        try:
            preserved = stat.S_IMODE(target.stat().st_mode)
        except OSError:
            preserved = None  # new file: created with create_mode below
    if preserved is None:
        # Stage with create_mode so the kernel applies the process
        # umask at creation, exactly like Path.write_text — without the
        # process-global (and thread-racy) os.umask() probe.
        fd, tmp = _open_staging_file(target, create_mode)
    else:
        # Stage privately; the intended bits are applied only once the
        # content is complete, just before publication.
        fd, tmp = tempfile.mkstemp(prefix=f".{target.name}-", dir=str(target.parent))
    try:
        # A buffered file object rather than a bare os.write, whose
        # POSIX-legal short return count would otherwise be ignored and
        # then published as a permanently truncated file.
        with os.fdopen(fd, "wb") as staged:
            staged.write(content.encode("utf-8"))
        # Applied to the staged file only: os.replace moves the inode,
        # mode included, so a second chmod on the target would be a no-op.
        if preserved is not None:
            _try_chmod(tmp, preserved)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _open_staging_file(target: Path, create_mode: int) -> tuple[int, str]:
    """Exclusively create a unique sibling staging file for *target*.

    Like ``tempfile.mkstemp`` but with a caller-chosen creation mode
    (``mkstemp`` hardwires ``0o600``), so the kernel derives the final
    bits from the process umask at creation time.

    Args:
        target: The destination the staging file will be renamed onto.
        create_mode: Mode bits passed to ``os.open`` (umask applies).

    Returns:
        ``(fd, path)`` of the newly created staging file.

    Raises:
        FileExistsError: If no unique name was found (never in
            practice: 128-bit random suffixes).
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    for _ in range(10):
        tmp = str(target.parent / f".{target.name}-{uuid.uuid4().hex}")
        try:
            return os.open(tmp, flags, create_mode), tmp
        except FileExistsError:  # pragma: no cover — 128-bit collision
            continue
    raise FileExistsError(  # pragma: no cover — unreachable in practice
        f"could not create a staging file for {target}"
    )


def _try_chmod(path: str, mode: int) -> None:
    """Apply *mode* to *path*, ignoring filesystems that refuse it."""
    try:
        os.chmod(path, mode)
    except OSError:
        logger.debug("chmod %o failed on %s", mode, path, exc_info=True)


def is_root_dir(path: str) -> bool:
    """Return whether *path* names a filesystem root.

    Roots — POSIX ``/`` (or ``//``), Windows drive roots such as
    ``C:\\`` / ``C:/`` (including a bare drive ``C:``), and bare
    backslashes — are never legitimate workspace folders: they reach
    the daemon only when a GUI-launched process inherited the root as
    its cwd (a Dock/Finder-launched VS Code window with no folder
    open).  Rooting a task or the ``@``-mention file scan there would
    span the whole disk, so callers treat a root exactly like "no work
    dir" and fall back to a properly configured folder.  Mirrors the
    VS Code extension's ``path.parse(cwd).root === cwd`` guard in
    ``SorcarSidebarView._getWorkDir``.

    Root-equivalent spellings (``/./``, ``/..``, ``C:\\.\\``) are
    normalized before the check so they cannot slip past a literal
    comparison.  Windows UNC roots (``\\\\server\\share``) are NOT
    detected: the daemon serves a POSIX filesystem (its transport is a
    Unix-domain socket), where a double-slash prefix is a legal path,
    so classifying it as a root would blank legitimate directories.

    Args:
        path: The candidate directory path (any string).

    Returns:
        ``True`` when *path* is a filesystem root; ``False`` for every
        other string, including empty or whitespace-only ones.
    """
    p = posixpath.normpath(path.strip().replace("\\", "/"))
    if p in ("/", "//"):
        return True
    # A bare or slash-terminated drive ('C:', 'C:\', 'C:/') from a
    # Windows-side client.  ASCII letters only: Node's path.win32 (the
    # extension-side mirror of this guard) recognizes no other drives.
    return len(p) == 2 and p[1] == ":" and p[0] in string.ascii_letters


def substitute_prompt_args(template: str, arguments: dict[str, str] | None) -> str:
    """Substitute ``{key}`` placeholders in *template* in a single pass.

    Unlike ``str.format()``, literal braces in the template (JSON, code,
    ``${VAR}``) are left untouched instead of raising ``KeyError`` /
    ``ValueError``.  All keys are substituted in ONE pass over the
    template: sequential per-key ``str.replace`` calls would rescan
    previously substituted values, so an argument value that literally
    contains another key's placeholder (e.g. a task string quoting
    ``{result}``) would be re-expanded — leaking the other argument into
    it, dependent on dict insertion order.

    Args:
        template: The prompt template containing ``{key}`` placeholders.
        arguments: Mapping of placeholder names to replacement values.

    Returns:
        The template with every ``{key}`` placeholder replaced.
    """
    if not arguments:
        return template
    pattern = re.compile("|".join(re.escape("{" + key + "}") for key in arguments))
    return pattern.sub(lambda m: str(arguments[m.group(0)[1:-1]]), template)


def _is_secret_config_field(name: str) -> bool:
    """Return True for config field names holding credentials.

    ``WORKSPACE_ID`` covers ``ANTHROPIC_WORKSPACE_ID``: not a key itself,
    but account-identifying and managed alongside the API keys, so it must
    not be serialized into trajectories either.

    Args:
        name: The config field name to classify.

    Returns:
        bool: True when the field must be excluded from serialization.
    """
    return "API_KEY" in name or "WORKSPACE_ID" in name


def config_to_dict() -> dict[Any, Any]:
    """Convert the config to a dictionary.

    Returns:
        dict[Any, Any]: A dictionary representation of the default config.
    """

    def convert_to_json(obj: Any) -> Any:
        if isinstance(obj, dict):  # pragma: no cover – config has no raw dicts
            return {
                k: convert_to_json(v)
                for k, v in obj.items()  # type: ignore[misc]
                if not _is_secret_config_field(k)
            }
        if isinstance(obj, list):  # pragma: no cover – config has no raw lists
            return [convert_to_json(item) for item in obj]  # type: ignore[misc]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if hasattr(obj, "__dict__"):
            return {
                k: convert_to_json(getattr(obj, k))
                for k in obj.__dict__.keys()
                if not _is_secret_config_field(k)
            }
        return obj  # pragma: no cover – all config values have __dict__ or are primitives

    return cast(dict[Any, Any], convert_to_json(config_module.DEFAULT_CONFIG))


def _coerce_bool(value: bool | str) -> bool:
    """Coerce a string or bool tool argument to a Python bool.

    Args:
        value: A string ("true", "1", "yes" → True; anything else → False)
            or an already-boolean value.

    Returns:
        The boolean interpretation of *value*.
    """
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes")
    return bool(value)


_HTML_TAG_RE = re.compile(
    r"</?(?:p|div|h[1-6]|ul|ol|li|br|hr|table|thead|tbody|tr|td|th|pre|code|"
    r"span|b|i|u|strong|em|a|img|blockquote|section|article|details|summary)"
    r"(?:\s[^<>]*)?/?>",
    re.IGNORECASE,
)


def _unescape_escaped_html(text: str) -> str | None:
    """Recover HTML from text whose tags were entity-escaped by mistake.

    Some LLMs emit ``summary_in_html`` with every tag pre-escaped
    (``&lt;h3&gt;`` instead of ``<h3>``), sometimes escaped more than once
    (``&amp;lt;h3&amp;gt;``).  Such a summary contains no real tags, so it
    would otherwise be rendered as plain text and reach the user as
    literal tag soup.  The double-escaping signature is that the text
    *begins* with an entity-escaped HTML tag; escaped entities elsewhere
    (e.g. prose showing ``&lt;h3&gt;`` as an example) are intentional and
    must be preserved.

    Args:
        text: Candidate summary text containing no real HTML tags.

    Returns:
        The fully unescaped HTML if *text* starts (modulo whitespace) with
        an entity-escaped known HTML tag or DOCTYPE, otherwise ``None``.
    """
    candidate = text
    for _ in range(3):
        unescaped = html_module.unescape(candidate)
        if unescaped == candidate:
            return None
        candidate = unescaped
        stripped = candidate.lstrip()
        if stripped[:9].lower() == "<!doctype" or _HTML_TAG_RE.match(stripped):
            return candidate
    return None


def ensure_html(text: str) -> str:
    """Return *text* as HTML, converting from Markdown/plain text if needed.

    Text that already contains HTML markup (a known HTML tag or a full
    document) is passed through unchanged.  Text that is HTML with every
    tag entity-escaped (``&lt;h3&gt;`` instead of ``<h3>``, a known LLM
    mistake) is unescaped back to real HTML.  Anything else is treated as
    Markdown and rendered to HTML, which also HTML-escapes special
    characters in plain text.

    Args:
        text: The summary text: HTML, Markdown, or plain text.  Non-string
            input (some LLMs pass numbers/lists) is coerced with ``str()``.

    Returns:
        The HTML representation of *text* (empty input is returned as-is).
    """
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if text.lstrip()[:9].lower() == "<!doctype" or _HTML_TAG_RE.search(text):
        return text
    unescaped = _unescape_escaped_html(text)
    if unescaped is not None:
        return unescaped
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        # markdown_it can be momentarily unimportable (e.g. the installer
        # is rebuilding the venv with `uv sync`).  finish() runs on error
        # reporting paths, so it must NEVER raise here — degrade to
        # HTML-escaped text instead of masking the original error.
        escaped = html_module.escape(text).replace("\n", "<br/>")
        return f"<p>{escaped}</p>"

    rendered: str = MarkdownIt("commonmark", {"breaks": False}).enable("table").render(text)
    return rendered.strip()


def finish(
    success: bool,
    is_continue: bool = False,
    summary_in_html: str = "",
    suggested_next_task: str = "",
) -> str:
    """Finish execution with status and summary.

    The agent must call this function when it has solved (or cannot solve)
    the given task, passing the final result in ``summary_in_html``.

    Args:
        success: True if the agent has successfully completed the task, False otherwise.
        is_continue: True if the task is incomplete and should continue, False otherwise.
        summary_in_html: The agent's final result, formatted as HTML (never Markdown).
            Use e.g. ``<h3>``, ``<p>``, ``<ul>``, ``<pre><code>``; Markdown or
            plain-text input is converted to HTML.
        suggested_next_task: ONE concrete follow-up task for the user, as one plain-text sentence.
            Shown to the user as "Suggested next"; leave empty when nothing
            sensible follows.

    Returns:
        A YAML string with 'success', 'is_continue' and 'summary' keys, where
        'summary' always holds HTML, plus a 'suggested_next_task' key when a
        non-empty suggestion was given.
    """
    result: dict[str, Any] = {
        "success": _coerce_bool(success),
        "is_continue": _coerce_bool(is_continue),
        "summary": ensure_html(summary_in_html),
    }
    suggestion = str(suggested_next_task).strip() if suggested_next_task else ""
    if suggestion:
        result["suggested_next_task"] = suggestion
    dumped: str = dump_yaml(result, sort_keys=False)
    return dumped

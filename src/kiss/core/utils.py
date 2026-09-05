# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Utility functions for the KISS core module."""

import html as html_module
import logging
import os
import re
import tempfile
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


def atomic_write_text(target: Path, content: str, mode: int | None = None) -> None:
    """Write *content* to *target* so readers never see a partial file.

    The content is staged in a sibling temp file and then
    ``os.replace``-d into position, which is atomic on every supported
    platform.  A plain ``open(path, "w")`` truncates immediately and
    then fills the file incrementally, so a concurrent reader — the
    trajectory visualizer, another daemon, the VS Code extension — can
    observe an empty or half-written document.

    Args:
        target: Destination path; its parent directory is created.
        content: The full text to write.
        mode: Optional permission bits to force on the result (e.g.
            ``0o600`` for files holding secrets).  Best effort: a
            filesystem that refuses ``chmod`` is not treated as a write
            failure.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}-", dir=str(target.parent))
    try:
        # A buffered file object rather than a bare os.write, whose
        # POSIX-legal short return count would otherwise be ignored and
        # then published as a permanently truncated file.
        with os.fdopen(fd, "wb") as staged:
            staged.write(content.encode("utf-8"))
        if mode is not None:
            _try_chmod(tmp, mode)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    if mode is not None:
        _try_chmod(str(target), mode)


def _try_chmod(path: str, mode: int) -> None:
    """Apply *mode* to *path*, ignoring filesystems that refuse it."""
    try:
        os.chmod(path, mode)
    except OSError:
        logger.debug("chmod %o failed on %s", mode, path, exc_info=True)


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

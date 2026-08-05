# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Utility functions for the KISS core module."""

import html as html_module
import re
from typing import Any, cast

import yaml

from kiss.core import config as config_module


def substitute_prompt_args(
    template: str, arguments: dict[str, str] | None
) -> str:
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
    pattern = re.compile(
        "|".join(re.escape("{" + key + "}") for key in arguments)
    )
    return pattern.sub(
        lambda m: str(arguments[m.group(0)[1:-1]]), template
    )


def config_to_dict() -> dict[Any, Any]:
    """Convert the config to a dictionary.

    Returns:
        dict[Any, Any]: A dictionary representation of the default config.
    """

    def convert_to_json(obj: Any) -> Any:
        if isinstance(obj, dict):  # pragma: no cover – config has no raw dicts
            return {k: convert_to_json(v) for k, v in obj.items() if "API_KEY" not in k}  # type: ignore[misc]
        if isinstance(obj, list):  # pragma: no cover – config has no raw lists
            return [convert_to_json(item) for item in obj]  # type: ignore[misc]
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        if hasattr(obj, "__dict__"):
            return {
                k: convert_to_json(getattr(obj, k))
                for k in obj.__dict__.keys()
                if "API_KEY" not in k
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

    rendered: str = (
        MarkdownIt("commonmark", {"breaks": False}).enable("table").render(text)
    )
    return rendered.strip()


def finish(success: bool, is_continue: bool = False, summary_in_html: str = "") -> str:
    """Finish execution with status and summary.

    The agent must call this function when it has solved (or cannot solve)
    the given task, passing the final result in ``summary_in_html``.

    Args:
        success: True if the agent has successfully completed the task, False otherwise.
        is_continue: True if the task is incomplete and should continue, False otherwise.
        summary_in_html: The final result generated by the agent, formatted as
            HTML (e.g. ``<h3>``, ``<p>``, ``<ul>``, ``<pre><code>`` — never
            Markdown). Markdown or plain-text input is converted to HTML.

    Returns:
        A YAML string with 'success', 'is_continue' and 'summary' keys, where
        'summary' always holds HTML.
    """
    dumped: str = yaml.dump(
        {
            "success": _coerce_bool(success),
            "is_continue": _coerce_bool(is_continue),
            "summary": ensure_html(summary_in_html),
        },
        sort_keys=False,
    )
    return dumped

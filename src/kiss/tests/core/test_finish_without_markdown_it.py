# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end regression tests: ``finish()`` must never crash when the
``markdown_it`` package is unavailable.

Reproduces the 2026-08-01 production failure where a task died with
``Task failed with error: No module named 'markdown_it'``:

* The kiss-web installer was rebuilding the extension venv (``uv sync``
  in progress), so ``markdown_it`` was momentarily not importable.
* ``KISSAgent.run()`` raised at startup; ``RelentlessAgent``'s error
  handler called ``finish(False, False, str(exc))`` to report it.
* ``finish()`` -> ``ensure_html()`` lazily imported ``markdown_it``,
  raising ``ModuleNotFoundError`` *inside the error handler* — masking
  the original exception and killing the task with a confusing message.

The fix makes ``ensure_html()`` fall back to HTML-escaped ``<p>`` output
when ``markdown_it`` cannot be imported, so error reporting always works.

Each test runs a real, separate Python interpreter in which
``markdown_it`` is genuinely unimportable (a ``MetaPathFinder`` installed
at process start raises ``ModuleNotFoundError`` for it) — no mocks or
monkeypatching of the code under test.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import yaml

# Python source executed in a fresh interpreter.  It makes `markdown_it`
# truly unimportable in that process (a MetaPathFinder that raises for
# the module, installed before anything else runs), then exercises the
# real `kiss.core.utils.finish` / `ensure_html` code and prints results
# as JSON on stdout.
_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import json
    import sys

    class _BlockMarkdownIt(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "markdown_it" or name.startswith("markdown_it."):
                raise ModuleNotFoundError("No module named 'markdown_it'")
            return None

    sys.meta_path.insert(0, _BlockMarkdownIt())
    sys.modules.pop("markdown_it", None)

    # Sanity: markdown_it really is unimportable in this interpreter.
    try:
        import markdown_it  # noqa: F401
        blocked = False
    except ModuleNotFoundError:
        blocked = True

    from kiss.core.utils import ensure_html, finish

    out = {"blocked": blocked}
    try:
        out["finish_yaml"] = finish(
            False, False, "boom: original error <tag> & text"
        )
        out["finish_raised"] = None
    except BaseException as exc:  # pragma: no cover - pre-fix behavior
        out["finish_yaml"] = None
        out["finish_raised"] = f"{type(exc).__name__}: {exc}"

    try:
        out["ensure_md"] = ensure_html("# Title\\n\\n* item & <x>")
        out["ensure_raised"] = None
    except BaseException as exc:  # pragma: no cover - pre-fix behavior
        out["ensure_md"] = None
        out["ensure_raised"] = f"{type(exc).__name__}: {exc}"

    # HTML input must still pass through untouched.
    out["ensure_html_passthru"] = ensure_html("<p>already html</p>")
    print(json.dumps(out))
    """
)


def _run_without_markdown_it() -> dict[str, object]:
    """Run the scenario script in a fresh interpreter and parse its JSON."""
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"subprocess failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    out: dict[str, object] = json.loads(proc.stdout.strip().splitlines()[-1])
    return out


def test_finish_does_not_raise_when_markdown_it_is_missing() -> None:
    """finish() must return a YAML result, not raise, without markdown_it.

    This is the exact production error path: RelentlessAgent calls
    ``finish(False, False, str(exc))`` to report a startup failure, and
    that call must succeed even when markdown_it is not installed.
    """
    out = _run_without_markdown_it()
    assert out["blocked"] is True, "markdown_it was importable; bad test env"
    assert out["finish_raised"] is None, (
        "finish() crashed while reporting an error, masking the original "
        f"exception: {out['finish_raised']}"
    )
    finish_yaml = out["finish_yaml"]
    assert isinstance(finish_yaml, str)
    parsed = yaml.safe_load(finish_yaml)
    assert parsed["success"] is False
    assert parsed["is_continue"] is False
    # The original error text must survive, HTML-escaped inside HTML.
    assert "boom: original error" in parsed["summary"]
    assert "<tag>" not in parsed["summary"]
    assert "&lt;tag&gt;" in parsed["summary"]
    assert "&amp;" in parsed["summary"]
    assert parsed["summary"].startswith("<")


def test_ensure_html_falls_back_to_escaped_html_without_markdown_it() -> None:
    """ensure_html() must degrade to escaped HTML instead of raising."""
    out = _run_without_markdown_it()
    assert out["blocked"] is True
    assert out["ensure_raised"] is None, (
        f"ensure_html() raised without markdown_it: {out['ensure_raised']}"
    )
    fallback = out["ensure_md"]
    assert isinstance(fallback, str)
    assert "Title" in fallback
    assert "&lt;x&gt;" in fallback
    assert "&amp;" in fallback


def test_ensure_html_passthrough_still_works_without_markdown_it() -> None:
    """HTML input needs no conversion, so it must work unchanged."""
    out = _run_without_markdown_it()
    assert out["ensure_html_passthru"] == "<p>already html</p>"


def test_ensure_html_still_renders_markdown_when_available() -> None:
    """With markdown_it installed (this process), Markdown still converts."""
    from kiss.core.utils import ensure_html

    rendered = ensure_html("# Title\n\n* item")
    assert "<h1>" in rendered
    assert "<li>" in rendered

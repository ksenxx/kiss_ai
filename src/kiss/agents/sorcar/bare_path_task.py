# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Tasks that are nothing but a filesystem path.

A user who submits ``~/Downloads/ROUTING.md`` as the whole task wants
the file opened, the way a path typed into a launcher opens it.  Left
alone, the agent reads the prompt as under-specified and asks what to
do with the file (the system prompt tells it to ask rather than guess),
so :func:`with_open_directive` appends an explicit instruction to such
a prompt: open the path with the platform opener and stop.  The
task-history row, the tab title and the frequent-tasks table still
show the raw path the user typed, because
:class:`~kiss.agents.sorcar.chat_sorcar_agent.ChatSorcarAgent` records
those before the directive is added.
"""

from __future__ import annotations

import shlex
from pathlib import Path


def bare_path(prompt: str, work_dir: str) -> Path | None:
    """Return the existing filesystem path *prompt* consists of.

    The prompt qualifies when, after stripping surrounding whitespace
    and one pair of matching quotes, it names an existing file or
    directory: ``~`` is expanded and a relative path is resolved
    against *work_dir*.  Existence is the whole test — an instruction,
    a paragraph or a mistyped path is not a path on disk.

    Args:
        prompt: The raw task text as the user submitted it.
        work_dir: Directory a relative path is resolved against.

    Returns:
        The absolute path, or ``None`` when the prompt is anything else.
    """
    text = prompt.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    if not text:
        return None
    try:
        path = Path(work_dir, Path(text).expanduser())
        if not path.exists():
            return None
        return path.resolve()
    except RuntimeError:  # ``~nosuchuser/...``: no home directory to expand
        return None
    except OSError:  # Python 3.13 ``exists`` raises on over-long or unreadable paths
        return None


def with_open_directive(prompt: str, work_dir: str) -> str:
    """Append the "open this path" instruction when *prompt* is a bare path.

    Args:
        prompt: The raw task text as the user submitted it.
        work_dir: Directory a relative path is resolved against.

    Returns:
        *prompt* unchanged unless :func:`bare_path` recognises it, in
        which case the prompt followed by the directive to open the
        path with the platform opener (``open`` on macOS, ``xdg-open``
        elsewhere) and finish without reading, editing or asking about
        it.
    """
    path = bare_path(prompt, work_dir)
    if path is None:
        return prompt
    kind = "directory" if path.is_dir() else "file"
    quoted = shlex.quote(str(path))
    return (
        f"{prompt}\n\n"
        f"The task is nothing but the path of an existing {kind}, so open it "
        f"for the user with the platform opener: run `open {quoted}` on macOS "
        f"or `xdg-open {quoted}` on Linux through the Bash tool, then finish "
        f"with a one-sentence summary saying the {kind} was opened. Do not "
        f"read, summarize or edit it, and do not ask what to do with it."
    )

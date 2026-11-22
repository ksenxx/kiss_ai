# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The three replies of the right sidebar's Skills panel.

``skills`` lists what the agent can use (:func:`skills_reply`),
``skillSearch`` what the skills.sh registry offers
(:func:`search_reply`), and ``skillInstall`` reports an install
(:func:`install_reply`): the skill is downloaded into a staging
directory beside the user's skills, checked there
(:mod:`kiss.agents.sorcar.skill_check`) and renamed into place only
when the check allows it.  Each function blocks on the network or on a
model call, so the server runs it in a thread.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from kiss.agents.sorcar import skill_check, skill_registry
from kiss.agents.sorcar.skill_rules import Finding
from kiss.agents.sorcar.skills import Skill, _parse_skill_file, discover_skills, user_skills_dir

logger = logging.getLogger(__name__)

# The names being installed: two installs of one name must not both
# download, pay for a check and race for the directory.  Each install
# holds a thread of the server for a minute or more, so only a few run
# at once.
_MAX_INSTALLS = 4
_installing: set[str] = set()
_installing_lock = threading.Lock()
# What the last ``review`` reply said about a skill id: the digest of
# the files it was about, and its findings.  ``force`` installs those
# files and no others, because a repository can change between the
# review and the user's answer to it.
_MAX_REVIEWS = 64
_reviews: dict[str, tuple[str, list[Finding]]] = {}


def _skill_name(skill: Skill) -> str:
    """Sort key of :func:`skills_reply`."""
    return skill.name


def skills_reply(work_dir: str) -> dict[str, Any]:
    """Return the ``skills`` event: every skill the agent can use in *work_dir*.

    Args:
        work_dir: The folder whose project skills are included.

    Returns:
        ``{"type": "skills", "skills": [{"name", "description",
        "source"}, ...], "error": ""}``, sorted by name; no rows and
        the ``error`` sentence when *work_dir* is no usable path.
    """
    try:
        # Windows resolves a path with a NUL in it without complaint.
        if "\x00" in work_dir:
            raise ValueError("embedded null character in the path of the work folder")
        skills = sorted(discover_skills(work_dir).values(), key=_skill_name)
    # A path the file system refuses: over-long, or with a NUL in it.
    except (OSError, ValueError):
        message = "The skills could not be listed: the path of the work folder is not usable."
        return {"type": "skills", "skills": [], "error": message}
    rows = [{"name": s.name, "description": s.description, "source": s.source} for s in skills]
    return {"type": "skills", "skills": rows, "error": ""}


def search_reply(query: str, work_dir: str) -> dict[str, Any]:
    """Return the ``skillSearch`` event: the registry's skills for *query*.

    Args:
        query: What the user typed; fewer than two characters ask for
            the registry's most installed skills.
        work_dir: The folder whose skills count as installed.

    Returns:
        ``{"type": "skillSearch", "query", "results", "error"}``: the
        rows of :func:`~kiss.agents.sorcar.skill_registry.search`, each
        with ``installed`` (a skill of that name is available already),
        or no rows and the ``error`` sentence of the registry or of
        :func:`skills_reply`.
    """
    listed = skills_reply(work_dir)
    error = listed["error"]
    rows: list[dict[str, Any]] = []
    if not error:
        try:
            rows = skill_registry.search(query)
        except skill_registry.RegistryError as failure:
            error = str(failure)
    available = {row["name"] for row in listed["skills"]}
    for row in rows:
        row["installed"] = row["name"] in available
    return {"type": "skillSearch", "query": query, "results": rows, "error": error}


def _reply(
    skill: str, status: str, name: str, message: str, findings: list[Finding],
) -> dict[str, Any]:
    """Return the ``skillInstall`` event."""
    return {
        "type": "skillInstall", "id": skill, "status": status, "name": name, "message": message,
        "findings": [dataclasses.asdict(finding) for finding in findings],
    }


def _staged_files(folder: Path) -> dict[str, bytes]:
    """Return the files of the unpacked skill in *folder*: path relative to it -> content."""
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _digest(files: dict[str, bytes]) -> str:
    """Return a digest of the names and contents of *files*."""
    digest = hashlib.sha256()
    for name, data in files.items():
        digest.update(f"{len(name.encode())} {len(data)} ".encode() + name.encode() + data)
    return digest.hexdigest()


def _remember_review(skill: str, digest: str, findings: list[Finding]) -> None:
    """Keep what a ``review`` reply said about *skill* for the ``force`` that may answer it."""
    with _installing_lock:
        _reviews.pop(skill, None)
        _reviews[skill] = (digest, findings)
        if len(_reviews) > _MAX_REVIEWS:
            del _reviews[next(iter(_reviews))]  # the oldest


def _install(skill: str, source: str, name: str, force: bool, work_dir: str) -> dict[str, Any]:
    """Install the skill *name* of repository *source* (the body of :func:`install_reply`)."""
    listed = skills_reply(work_dir)
    if listed["error"]:
        return _reply(skill, "error", name, listed["error"], [])
    dest = user_skills_dir() / name
    # A name the agent already has, from any source, is what a search
    # row calls installed: the new skill would replace it unasked.
    if os.path.lexists(dest) or any(row["name"] == name for row in listed["skills"]):
        return _reply(skill, "error", name, f"A skill named {name} is already installed.", [])
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Cleanup errors ignored: a leftover must not turn an install into a failure.
    with tempfile.TemporaryDirectory(
        dir=dest.parent, prefix=".installing-", ignore_cleanup_errors=True,
    ) as staging:
        reviewed = _reviews.get(skill, ("", []))
        try:
            folder = skill_registry.fetch_skill(source, name, Path(staging))
            files = _staged_files(folder)
            digest = _digest(files)
            # ``force`` answers a review: the files that review was
            # about are installed as they were checked, any others are
            # checked first.
            approved = force and reviewed[0] == digest
            findings = reviewed[1] if approved else skill_check.check_skill(skill, files)
        except (skill_registry.RegistryError, skill_check.CheckError) as error:
            return _reply(skill, "error", name, str(error), [])
        status = skill_check.verdict(findings)
        if status == "blocked":
            message = f"{name} was not installed: the security check found a serious problem."
        elif status == "review" and not approved:
            _remember_review(skill, digest, findings)
            message = (
                f"{name} was not installed: it changed after your review, and the security check "
                "found something to review again."
                if force and reviewed[0] else
                f"{name} was not installed: the security check found something to review."
            )
        # Only now, after the rules bounded its frontmatter, is SKILL.md
        # parsed: a skill that discovery would skip is not installed.
        elif _parse_skill_file(folder / "SKILL.md", name, "user") is None:
            status = "error"
            message = "The skill's SKILL.md cannot be read or has no description."
        else:
            # Fails when a file or a directory with files took the name
            # while the check ran; an empty directory is replaced, and
            # nothing is lost with it.
            os.rename(folder, dest)
            message = (
                f"Installed {name} after your review of the findings." if approved
                else f"Installed {name}: the security check passed."
            )
            status = "installed"
    return _reply(skill, status, name, message, findings)


def install_reply(skill: str, force: bool, work_dir: str) -> dict[str, Any]:
    """Fetch, check and install a registry skill; return the ``skillInstall`` event.

    The skill's folder is unpacked from its GitHub repository into a
    staging directory inside the user's skills directory (same file
    system, never seen by skill discovery), checked there, and renamed
    to ``<user skills>/<name>`` only when the check allows it; the
    staging directory is removed whatever happens.

    Args:
        skill: The registry id, ``owner/repo/skill``.
        force: The user's answer to a ``review`` reply: install the
            files that reply was about, without a second check.  Files
            that changed since, or that no ``review`` reply of this
            process was about, are checked as without *force*.  A
            ``blocked`` skill is never installed.
        work_dir: The folder whose project skills count as installed.

    Returns:
        ``{"type": "skillInstall", "id", "status", "name", "message",
        "findings"}``: ``status`` is ``installed``, ``review`` (medium
        findings; nothing installed), ``blocked`` (high or critical
        findings) or ``error`` (not a GitHub skill id, a name the
        agent already has or that is being installed, too many
        installs at once, a failed download, a skills directory that
        cannot be written, the check could not run, a ``SKILL.md`` the
        agent could not load); ``findings`` are the check's, the most
        severe first.
    """
    parsed = skill_registry.parse_skill_id(skill)
    if parsed is None:
        return _reply(
            skill, "error", "", "Not the id of a skill in a GitHub repository (owner/repo/skill).",
            [],
        )
    source, name = parsed
    with _installing_lock:
        if name in _installing:
            return _reply(skill, "error", name, f"{name} is already being installed.", [])
        if len(_installing) >= _MAX_INSTALLS:
            return _reply(
                skill, "error", name,
                f"{_MAX_INSTALLS} skills are being installed; try again when one has finished.",
                [],
            )
        _installing.add(name)
    try:
        reply = _install(skill, source, name, force, work_dir)
    # Whatever fails (a skills directory that cannot be written, a bug
    # in a check), nothing was installed and the page is told so.
    except Exception as error:  # noqa: BLE001
        logger.warning("skill install %s failed", skill, exc_info=True)
        reply = _reply(skill, "error", name, f"The skill could not be installed ({error}).", [])
    finally:
        with _installing_lock:
            _installing.discard(name)
    logger.info("skill install %s: %s (%s)", skill, reply["status"], reply["message"])
    return reply

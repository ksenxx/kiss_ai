# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""GitHub Agent — channel agent for the GitHub REST API.

Provides access to GitHub via its REST API (https://api.github.com)
using a personal access token sent as ``Authorization: Bearer`` with
``Accept: application/vnd.github+json`` and
``X-GitHub-Api-Version: 2022-11-28`` on every call.  Stores config in
``~/.kiss/third_party_agents/github/config.json`` (keys: ``token`` and
optional ``read_only`` — ``"true"`` blocks every mutating tool).

GitHub's REST API has no inbound message stream, so this adapter is
outbound-only and the ``--channel`` poll mode is disabled (``main``
passes ``make_backend=None`` to ``channel_main``).

Usage::

    agent = GitHubAgent()
    agent.run(prompt_template="List the open issues in octocat/Hello-World")
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from kiss.agents.third_party_agents._channel_agent_utils import (
    BaseChannelAgent,
    ChannelConfig,
    ToolMethodBackend,
    channel_main,
)

logger = logging.getLogger(__name__)

_TIMEOUT = 30
_JSON_ACCEPT = "application/vnd.github+json"
_DIFF_ACCEPT = "application/vnd.github.v3.diff"
_RAW_ACCEPT = "application/vnd.github.raw+json"
_MAX_OUTPUT = 8000
_BODY_TRUNCATE = 1000
_MERGE_METHODS = ("merge", "squash", "rebase")

_GITHUB_DIR = Path.home() / ".kiss" / "third_party_agents" / "github"
_config = ChannelConfig(_GITHUB_DIR, ("token",))


def _bad_segment(value: str, name: str) -> str | None:
    """Reject *value* if it cannot safely form a single URL path segment.

    Values containing a path separator or a ``..`` sequence could
    traverse out of the intended API endpoint, so they are refused up
    front (defense in depth on top of ``quote(value, safe="")``).

    Args:
        value: Caller-supplied identifier destined for a URL path segment.
        name: Parameter name used in the error message.

    Returns:
        An ``{"ok": false, "error": ...}`` JSON string if *value* is
        unsafe, otherwise None.
    """
    if not value or "/" in value or "\\" in value or ".." in value:
        return json.dumps(
            {"ok": False, "error": f"invalid {name}: must not contain path separators or '..'"}
        )
    return None


def _bad_file_path(value: str, name: str) -> str | None:
    """Reject a repository file path containing ``..`` or backslashes.

    File paths may contain ``/`` separators (nested files), and the
    empty path is allowed (the GitHub API defines it as the repository
    root directory), but a ``..`` segment or a backslash could traverse
    out of the intended endpoint.

    Args:
        value: Caller-supplied repository-relative file path.
        name: Parameter name used in the error message.

    Returns:
        An ``{"ok": false, "error": ...}`` JSON string if *value* is
        unsafe, otherwise None.
    """
    if "\\" in value or ".." in value:
        return json.dumps({"ok": False, "error": f"invalid {name}: must not contain '..' or '\\'"})
    return None


def _first_line(text: str) -> str:
    """Return the first line of *text*.

    Args:
        text: Possibly multi-line string (e.g. a commit message).

    Returns:
        Everything before the first newline.
    """
    return text.split("\n", 1)[0]


def _condense_repo(repo: dict[str, Any]) -> dict[str, Any]:
    """Condense a repository API object to its useful fields.

    Args:
        repo: Raw repository object from the GitHub API.

    Returns:
        Dict with full_name, description, stars, url, default_branch,
        and private.
    """
    return {
        "full_name": repo.get("full_name", ""),
        "description": repo.get("description") or "",
        "stars": repo.get("stargazers_count", 0),
        "url": repo.get("html_url", ""),
        "default_branch": repo.get("default_branch", ""),
        "private": repo.get("private", False),
    }


def _condense_issue(issue: dict[str, Any], with_body: bool = False) -> dict[str, Any]:
    """Condense an issue or pull-request API object to its useful fields.

    Args:
        issue: Raw issue/PR object from the GitHub API.
        with_body: Whether to include the (truncated) body text.

    Returns:
        Dict with number, title, state, user, created_at, url, and
        optionally body.
    """
    user = issue.get("user") or {}
    condensed = {
        "number": issue.get("number", 0),
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "user": user.get("login", ""),
        "created_at": issue.get("created_at", ""),
        "url": issue.get("html_url", ""),
    }
    if with_body:
        condensed["body"] = (issue.get("body") or "")[:_BODY_TRUNCATE]
    return condensed


def _condense_commit(commit: dict[str, Any]) -> dict[str, Any]:
    """Condense a commit API object to its useful fields.

    Args:
        commit: Raw commit object from the GitHub API.

    Returns:
        Dict with the abbreviated sha, first message line, author, and date.
    """
    meta = commit.get("commit") or {}
    author = meta.get("author") or {}
    return {
        "sha": commit.get("sha", "")[:12],
        "message": _first_line(meta.get("message", "")),
        "author": author.get("name", ""),
        "date": author.get("date", ""),
    }


class GitHubChannelBackend(ToolMethodBackend):
    """Channel backend for the GitHub REST API.

    Talks to GitHub over HTTPS with a personal access token.
    Outbound-only: there is no inbound message stream, so poll mode is
    disabled.  When read-only mode is enabled, every mutating tool
    refuses to run before making any HTTP call.
    """

    def __init__(self) -> None:
        self._base_url: str = "https://api.github.com"
        self._token: str = ""
        self._read_only: bool = False
        self._request_lock = threading.Lock()
        self._connection_info: str = ""

    def connect(self) -> bool:
        """Load the GitHub config from disk.

        Returns:
            True if a valid config with a ``token`` was loaded.
        """
        cfg = _config.load()
        if not cfg:
            self._connection_info = "No GitHub config found."
            return False
        self._token = cfg["token"]
        self._read_only = cfg.get("read_only", "false") == "true"
        mode = "read-only" if self._read_only else "read-write"
        self._connection_info = f"GitHub configured ({mode})."
        return True

    def _api(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        accept: str = _JSON_ACCEPT,
    ) -> tuple[Any, str]:
        """Issue an authenticated GitHub REST request.

        Args:
            method: HTTP method (``"GET"``, ``"POST"``, ``"PATCH"``, ``"PUT"``).
            path: API path starting with ``/``.
            params: Optional query parameters.
            payload: Optional JSON body.
            accept: ``Accept`` header value (JSON by default; diff media
                type for pull-request diffs, raw media type for large
                file contents).

        Returns:
            ``(data, "")`` on success — *data* is the decoded JSON (or
            raw text for the diff/raw media types and non-JSON
            responses) — or ``(None, error)`` on an HTTP error status.
        """
        url = self._base_url.rstrip("/") + path
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": accept,
            "X-GitHub-Api-Version": "2022-11-28",
        }
        with self._request_lock:
            resp = requests.request(
                method, url, headers=headers, params=params, json=payload, timeout=_TIMEOUT
            )
        if resp.status_code >= 400:
            return None, f"HTTP {resp.status_code}: {resp.text[:500]}"
        if accept in (_DIFF_ACCEPT, _RAW_ACCEPT):
            return resp.text, ""
        try:
            return resp.json(), ""
        except ValueError:
            return resp.text, ""

    def _read_only_error(self) -> str | None:
        """Return the read-only refusal, or None when writes are allowed.

        Returns:
            An ``{"ok": false, ...}`` JSON string when read-only mode is
            enabled, otherwise None.
        """
        if self._read_only:
            return json.dumps({"ok": False, "error": "GitHub agent is in read-only mode"})
        return None

    # ------------------------------------------------------------------
    # Read tools
    # ------------------------------------------------------------------

    def gh_get_me(self) -> str:
        """Get the authenticated GitHub user.

        Returns:
            JSON string with ok status and the user's login, name,
            company, url, and public repo count.
        """
        try:
            data, err = self._api("GET", "/user")
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {
                    "ok": True,
                    "user": {
                        "login": data.get("login", ""),
                        "name": data.get("name") or "",
                        "company": data.get("company") or "",
                        "url": data.get("html_url", ""),
                        "public_repos": data.get("public_repos", 0),
                    },
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_search_repositories(self, query: str, per_page: int = 10, page: int = 1) -> str:
        """Search GitHub repositories.

        Args:
            query: Search query (GitHub search syntax, e.g.
                ``"language:python stars:>1000"``).
            per_page: Results per page (default 10).
            page: Page number (default 1).

        Returns:
            JSON string with ok status, total_count, and condensed
            repositories (full_name, description, stars, url,
            default_branch, private).
        """
        try:
            data, err = self._api(
                "GET",
                "/search/repositories",
                params={"q": query, "per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            repos = [_condense_repo(r) for r in data.get("items", [])]
            return json.dumps(
                {"ok": True, "total_count": data.get("total_count", 0), "repositories": repos}
            )[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_get_repository(self, owner: str, repo: str) -> str:
        """Get a repository's metadata.

        Args:
            owner: Repository owner (user or organization login).
            repo: Repository name.

        Returns:
            JSON string with ok status and the condensed repository
            (full_name, description, stars, url, default_branch, private).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET", f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "repository": _condense_repo(data)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_list_issues(
        self, owner: str, repo: str, state: str = "open", labels: str = "",
        per_page: int = 20, page: int = 1,
    ) -> str:
        """List a repository's issues.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: Issue state filter: ``"open"``, ``"closed"``, or
                ``"all"`` (default ``"open"``).
            labels: Optional comma-separated label names to filter by.
            per_page: Results per page (default 20).
            page: Page number (default 1).

        Returns:
            JSON string with ok status and condensed issues (number,
            title, state, user, created_at, url).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            params: dict[str, Any] = {"state": state, "per_page": per_page, "page": page}
            if labels:
                params["labels"] = labels
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/issues",
                params=params,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "issues": [_condense_issue(i) for i in data]})[
                :_MAX_OUTPUT
            ]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_get_issue(self, owner: str, repo: str, number: int) -> str:
        """Get one issue, including its (truncated) body.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue number.

        Returns:
            JSON string with ok status and the condensed issue (number,
            title, state, user, created_at, url, body).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/issues/{int(number)}",
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "issue": _condense_issue(data, with_body=True)})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_list_issue_comments(
        self, owner: str, repo: str, number: int, per_page: int = 20, page: int = 1
    ) -> str:
        """List the comments on an issue or pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue or pull request number.
            per_page: Results per page (default 20).
            page: Page number (default 1).

        Returns:
            JSON string with ok status and condensed comments (user,
            created_at, truncated body).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
                f"/issues/{int(number)}/comments",
                params={"per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            comments = [
                {
                    "user": (c.get("user") or {}).get("login", ""),
                    "created_at": c.get("created_at", ""),
                    "body": (c.get("body") or "")[:_BODY_TRUNCATE],
                }
                for c in data
            ]
            return json.dumps({"ok": True, "comments": comments})[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_search_issues(self, query: str, per_page: int = 10, page: int = 1) -> str:
        """Search issues and pull requests across GitHub.

        Args:
            query: Search query (GitHub search syntax, e.g.
                ``"repo:octocat/Hello-World is:open label:bug"``).
            per_page: Results per page (default 10).
            page: Page number (default 1).

        Returns:
            JSON string with ok status, total_count, and condensed
            issues (number, title, state, user, created_at, url).
        """
        try:
            data, err = self._api(
                "GET",
                "/search/issues",
                params={"q": query, "per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            issues = [_condense_issue(i) for i in data.get("items", [])]
            return json.dumps(
                {"ok": True, "total_count": data.get("total_count", 0), "issues": issues}
            )[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_search_code(self, query: str, per_page: int = 10, page: int = 1) -> str:
        """Search code across GitHub.

        Args:
            query: Search query (GitHub code-search syntax, e.g.
                ``"def connect repo:octocat/Hello-World"``).
            per_page: Results per page (default 10).
            page: Page number (default 1).

        Returns:
            JSON string with ok status, total_count, and matches
            (name, path, repository, url).
        """
        try:
            data, err = self._api(
                "GET",
                "/search/code",
                params={"q": query, "per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            matches = [
                {
                    "name": item.get("name", ""),
                    "path": item.get("path", ""),
                    "repository": (item.get("repository") or {}).get("full_name", ""),
                    "url": item.get("html_url", ""),
                }
                for item in data.get("items", [])
            ]
            return json.dumps(
                {"ok": True, "total_count": data.get("total_count", 0), "matches": matches}
            )[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_list_pull_requests(
        self, owner: str, repo: str, state: str = "open", per_page: int = 20, page: int = 1
    ) -> str:
        """List a repository's pull requests.

        Args:
            owner: Repository owner.
            repo: Repository name.
            state: PR state filter: ``"open"``, ``"closed"``, or
                ``"all"`` (default ``"open"``).
            per_page: Results per page (default 20).
            page: Page number (default 1).

        Returns:
            JSON string with ok status and condensed pull requests
            (number, title, state, user, created_at, url).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls",
                params={"state": state, "per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {"ok": True, "pull_requests": [_condense_issue(p) for p in data]}
            )[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_get_pull_request(self, owner: str, repo: str, number: int) -> str:
        """Get one pull request, including its (truncated) body.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            JSON string with ok status and the condensed pull request
            (number, title, state, user, created_at, url, body, plus
            head/base branch names and merged flag).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls/{int(number)}",
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            pr = _condense_issue(data, with_body=True)
            pr["head"] = (data.get("head") or {}).get("ref", "")
            pr["base"] = (data.get("base") or {}).get("ref", "")
            pr["merged"] = data.get("merged", False)
            return json.dumps({"ok": True, "pull_request": pr})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_get_pull_request_diff(self, owner: str, repo: str, number: int) -> str:
        """Get a pull request's unified diff (truncated to 8000 chars).

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.

        Returns:
            JSON string with ok status and the diff text.
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls/{int(number)}",
                accept=_DIFF_ACCEPT,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "diff": str(data)[:_MAX_OUTPUT]})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_get_file_contents(self, owner: str, repo: str, path: str, ref: str = "") -> str:
        """Get a file's decoded text content, or list a directory.

        Args:
            owner: Repository owner.
            repo: Repository name.
            path: File or directory path within the repository (may
                contain ``/`` but not ``..``); an empty path lists the
                repository root directory.
            ref: Optional branch, tag, or commit SHA (default: the
                repository's default branch).

        Returns:
            JSON string with ok status and either the file text
            (truncated to 8000 chars; large files are re-fetched via the
            raw media type) or the directory entries (name, path, type,
            size).
        """
        try:
            bad = (
                _bad_segment(owner, "owner")
                or _bad_segment(repo, "repo")
                or _bad_file_path(path, "path")
            )
            if bad:
                return bad
            params: dict[str, Any] = {}
            if ref:
                params["ref"] = ref
            api_path = (
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
                f"/contents/{quote(path, safe='/')}"
            )
            data, err = self._api("GET", api_path, params=params or None)
            if err:
                return json.dumps({"ok": False, "error": err})
            if isinstance(data, list):
                entries = [
                    {
                        "name": e.get("name", ""),
                        "path": e.get("path", ""),
                        "type": e.get("type", ""),
                        "size": e.get("size", 0),
                    }
                    for e in data
                ]
                return json.dumps({"ok": True, "directory": entries})[:_MAX_OUTPUT]
            if data.get("encoding") == "base64":
                text = base64.b64decode(data.get("content", "")).decode(
                    "utf-8", errors="replace"
                )
                return json.dumps(
                    {
                        "ok": True,
                        "path": data.get("path", ""),
                        "size": data.get("size", 0),
                        "content": text,
                    }
                )[:_MAX_OUTPUT]
            if data.get("encoding") == "none":
                # Files 1-100 MB: the contents API omits the inline
                # content, so re-fetch the raw text directly.
                raw, raw_err = self._api(
                    "GET", api_path, params=params or None, accept=_RAW_ACCEPT
                )
                if raw_err:
                    return json.dumps({"ok": False, "error": raw_err})
                return json.dumps(
                    {
                        "ok": True,
                        "path": data.get("path", ""),
                        "size": data.get("size", 0),
                        "content": str(raw)[:_MAX_OUTPUT],
                    }
                )
            return json.dumps(
                {
                    "ok": True,
                    "path": data.get("path", ""),
                    "type": data.get("type", ""),
                    "size": data.get("size", 0),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_list_commits(
        self, owner: str, repo: str, sha: str = "", path: str = "",
        per_page: int = 20, page: int = 1,
    ) -> str:
        """List a repository's commits.

        Args:
            owner: Repository owner.
            repo: Repository name.
            sha: Optional branch name or commit SHA to start listing from.
            path: Optional file path — only commits touching it are listed.
            per_page: Results per page (default 20).
            page: Page number (default 1).

        Returns:
            JSON string with ok status and condensed commits
            (abbreviated sha, first message line, author, date).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            params: dict[str, Any] = {"per_page": per_page, "page": page}
            if sha:
                params["sha"] = sha
            if path:
                params["path"] = path
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/commits",
                params=params,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "commits": [_condense_commit(c) for c in data]})[
                :_MAX_OUTPUT
            ]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_list_branches(self, owner: str, repo: str, per_page: int = 50, page: int = 1) -> str:
        """List a repository's branches.

        Args:
            owner: Repository owner.
            repo: Repository name.
            per_page: Results per page (default 50).
            page: Page number (default 1).

        Returns:
            JSON string with ok status and branches (name, abbreviated
            head sha, protected flag).
        """
        try:
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "GET",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/branches",
                params={"per_page": per_page, "page": page},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            branches = [
                {
                    "name": b.get("name", ""),
                    "sha": (b.get("commit") or {}).get("sha", "")[:12],
                    "protected": b.get("protected", False),
                }
                for b in data
            ]
            return json.dumps({"ok": True, "branches": branches})[:_MAX_OUTPUT]
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    # ------------------------------------------------------------------
    # Write tools (blocked in read-only mode)
    # ------------------------------------------------------------------

    def gh_create_issue(
        self, owner: str, repo: str, title: str, body: str = "", labels: str = ""
    ) -> str:
        """Create an issue in a repository.

        Args:
            owner: Repository owner.
            repo: Repository name.
            title: Issue title.
            body: Optional issue body text.
            labels: Optional comma-separated label names.

        Returns:
            JSON string with ok status and the created issue's number
            and url.
        """
        try:
            blocked = self._read_only_error()
            if blocked:
                return blocked
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            payload: dict[str, Any] = {"title": title}
            if body:
                payload["body"] = body
            if labels:
                payload["labels"] = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
            data, err = self._api(
                "POST",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/issues",
                payload=payload,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {"ok": True, "number": data.get("number", 0), "url": data.get("html_url", "")}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_comment_on_issue(self, owner: str, repo: str, number: int, body: str) -> str:
        """Comment on an issue or pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue or pull request number.
            body: Comment text.

        Returns:
            JSON string with ok status and the created comment's url.
        """
        try:
            blocked = self._read_only_error()
            if blocked:
                return blocked
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            data, err = self._api(
                "POST",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
                f"/issues/{int(number)}/comments",
                payload={"body": body},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps({"ok": True, "url": data.get("html_url", "")})
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_update_issue(
        self, owner: str, repo: str, number: int, state: str = "", title: str = "",
        body: str = "",
    ) -> str:
        """Update an issue's state, title, and/or body.

        Only the supplied (non-empty) fields are changed.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Issue number.
            state: Optional new state: ``"open"`` or ``"closed"``.
            title: Optional new title.
            body: Optional new body text.

        Returns:
            JSON string with ok status and the updated issue's number,
            state, and url.
        """
        try:
            blocked = self._read_only_error()
            if blocked:
                return blocked
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            payload: dict[str, Any] = {}
            if state:
                payload["state"] = state
            if title:
                payload["title"] = title
            if body:
                payload["body"] = body
            if not payload:
                return json.dumps(
                    {"ok": False, "error": "nothing to update: supply state, title, or body"}
                )
            data, err = self._api(
                "PATCH",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/issues/{int(number)}",
                payload=payload,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {
                    "ok": True,
                    "number": data.get("number", 0),
                    "state": data.get("state", ""),
                    "url": data.get("html_url", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_create_pull_request(
        self, owner: str, repo: str, title: str, head: str, base: str, body: str = "",
        draft: bool = False,
    ) -> str:
        """Open a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            title: Pull request title.
            head: Branch with the changes (``"branch"`` or ``"user:branch"``).
            base: Branch to merge into (e.g. ``"main"``).
            body: Optional pull request description.
            draft: Whether to open the PR as a draft (default False).

        Returns:
            JSON string with ok status and the created pull request's
            number and url.
        """
        try:
            blocked = self._read_only_error()
            if blocked:
                return blocked
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            payload: dict[str, Any] = {
                "title": title,
                "head": head,
                "base": base,
                "draft": draft,
            }
            if body:
                payload["body"] = body
            data, err = self._api(
                "POST",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}/pulls",
                payload=payload,
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {"ok": True, "number": data.get("number", 0), "url": data.get("html_url", "")}
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})

    def gh_merge_pull_request(
        self, owner: str, repo: str, number: int, merge_method: str = "merge"
    ) -> str:
        """Merge a pull request.

        Args:
            owner: Repository owner.
            repo: Repository name.
            number: Pull request number.
            merge_method: ``"merge"``, ``"squash"``, or ``"rebase"``
                (default ``"merge"``).

        Returns:
            JSON string with ok status, the merged flag, and the API's
            result message.
        """
        try:
            blocked = self._read_only_error()
            if blocked:
                return blocked
            bad = _bad_segment(owner, "owner") or _bad_segment(repo, "repo")
            if bad:
                return bad
            if merge_method not in _MERGE_METHODS:
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"invalid merge_method: must be one of {_MERGE_METHODS}",
                    }
                )
            data, err = self._api(
                "PUT",
                f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
                f"/pulls/{int(number)}/merge",
                payload={"merge_method": merge_method},
            )
            if err:
                return json.dumps({"ok": False, "error": err})
            return json.dumps(
                {
                    "ok": True,
                    "merged": data.get("merged", False),
                    "message": data.get("message", ""),
                }
            )
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})


class GitHubAgent(BaseChannelAgent):
    """Channel agent with GitHub REST API tools."""

    channel_system_prompt = (
        "You are operating GitHub through its REST API. Read tools: "
        "gh_get_me (authenticated user), gh_search_repositories, "
        "gh_get_repository, gh_list_issues, gh_get_issue, "
        "gh_list_issue_comments, gh_search_issues, gh_search_code, "
        "gh_list_pull_requests, gh_get_pull_request, "
        "gh_get_pull_request_diff, gh_get_file_contents (file text or "
        "directory listing), gh_list_commits, and gh_list_branches. "
        "Write tools (refused when the agent is configured read-only): "
        "gh_create_issue, gh_comment_on_issue, gh_update_issue, "
        "gh_create_pull_request, and gh_merge_pull_request. There is no "
        "inbound message stream."
    )

    def __init__(self) -> None:
        super().__init__("GitHub Agent")
        self._backend = GitHubChannelBackend()
        cfg = _config.load()
        if cfg:
            self._backend._token = cfg["token"]
            self._backend._read_only = cfg.get("read_only", "false") == "true"

    def _is_authenticated(self) -> bool:
        """Return True if the backend is authenticated."""
        return bool(self._backend._token)

    def _get_auth_tools(self) -> list:
        """Return channel-specific authentication tool functions."""
        agent = self

        def check_github_auth() -> str:
            """Check if GitHub is configured.

            Returns:
                Configuration status or instructions.
            """
            if not agent._is_authenticated():
                return (
                    "Not configured for GitHub. Use authenticate_github() to "
                    "configure.\n"
                    "You need a personal access token from "
                    "https://github.com/settings/tokens (classic or "
                    "fine-grained). If the GitHub CLI is logged in, "
                    "`gh auth token` prints an existing token."
                )
            return json.dumps({"ok": True, "read_only": agent._backend._read_only})

        def authenticate_github(token: str, read_only: bool = False) -> str:
            """Configure the GitHub personal access token.

            Args:
                token: Personal access token (from
                    https://github.com/settings/tokens or `gh auth token`).
                read_only: When True, every mutating tool (create/update/
                    comment/merge) is refused (default False).

            Returns:
                Configuration result or error message.
            """
            if not token.strip():
                return "token cannot be empty."
            try:
                _config.save(
                    {"token": token.strip(), "read_only": "true" if read_only else "false"}
                )
                agent._backend._token = token.strip()
                agent._backend._read_only = read_only
            except Exception as e:
                return json.dumps(
                    {"ok": False, "error": f"failed to save GitHub config: {e}"}
                )
            return json.dumps({"ok": True, "message": "GitHub configured."})

        def clear_github_auth() -> str:
            """Clear the stored GitHub configuration.

            Returns:
                Status message.
            """
            _config.clear()
            agent._backend._token = ""
            agent._backend._read_only = False
            return "GitHub configuration cleared."

        return [check_github_auth, authenticate_github, clear_github_auth]


def main() -> None:
    """Run the GitHubAgent from the command line with chat persistence.

    Poll mode is disabled (``make_backend=None``): GitHub's REST API
    has no inbound message stream to poll.
    """
    channel_main(
        GitHubAgent,
        "kiss-github",
        channel_name="GitHub",
        make_backend=None,
    )


def get_tools() -> list:
    """Return the GitHub channel tools (``kiss.server.sorcar.run`` tools-file contract).

    Called by the kiss-web daemon when this module's path is passed as
    the API's ``tools=`` argument: builds a fresh agent from the
    credentials persisted under ``~/.kiss`` and returns its
    authentication and backend tools.
    """
    return GitHubAgent()._get_tools()


if __name__ == "__main__":
    main()

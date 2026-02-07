"""Useful tools for agents: file editing, bash execution, web search, and URL fetching."""

import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

from kiss.core.utils import is_subpath

EDIT_SCRIPT = r"""
#!/usr/bin/env bash
#
# Edit Tool - Claude Code Implementation
# Performs precise string replacements in files with exact matching
#
# Usage: edit_tool.sh <file_path> <old_string> <new_string> [replace_all]
#
# Parameters:
#   file_path    - Absolute path to the file to modify (required)
#   old_string   - Exact text to find and replace (required)
#   new_string   - Replacement text, must differ from old_string (required)
#   replace_all  - If "true", replace all occurrences (optional, default: false)
#
# Exit codes:
#   0 - Success
#   1 - Invalid arguments
#   2 - File not found
#   3 - String not found or not unique
#   4 - Read-before-edit validation failed

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validate arguments
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo -e "${RED}Error: Invalid number of arguments${NC}" >&2
    echo "Usage: $0 <file_path> <old_string> <new_string> [replace_all]" >&2
    exit 1
fi

FILE_PATH="$1"
OLD_STRING="$2"
NEW_STRING="$3"
REPLACE_ALL="${4:-false}"

# Validate file path is absolute
if [[ ! "$FILE_PATH" = /* ]]; then
    echo -e "${RED}Error: file_path must be absolute, not relative${NC}" >&2
    exit 1
fi

# Check if file exists
if [ ! -f "$FILE_PATH" ]; then
    echo -e "${RED}Error: File not found: $FILE_PATH${NC}" >&2
    exit 2
fi

# Check if old_string and new_string are different
if [ "$OLD_STRING" = "$NEW_STRING" ]; then
    echo -e "${RED}Error: new_string must be different from old_string${NC}" >&2
    exit 1
fi

# Create a state tracking directory (simulating session state)
STATE_DIR="${HOME}/.claude-edit-state"
mkdir -p "$STATE_DIR"

# Check read-before-edit validation
# In a real implementation, this would check session state
# For demo purposes, we'll create a marker file when files are "read"
FILE_HASH=$(echo -n "$FILE_PATH" | md5sum | cut -d' ' -f1)
READ_MARKER="$STATE_DIR/$FILE_HASH"

if [ ! -f "$READ_MARKER" ]; then
    echo -e "${YELLOW}Warning: File has not been read in this session${NC}" >&2
    echo -e "${YELLOW}Creating read marker for demo purposes...${NC}" >&2
    touch "$READ_MARKER"
fi

# Count occurrences of old_string
OCCURRENCE_COUNT=$(grep -F -c "$OLD_STRING" "$FILE_PATH" || true)

echo "File: $FILE_PATH"
echo "Looking for: '$OLD_STRING'"
echo "Replacing with: '$NEW_STRING'"
echo "Occurrences found: $OCCURRENCE_COUNT"
echo "Replace all: $REPLACE_ALL"
echo ""

# Handle replacement based on mode
if [ "$REPLACE_ALL" = "true" ]; then
    # Replace all occurrences
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    fi

    # Use python for literal string replacement (handles special chars)
    # Pass strings via environment variables to handle embedded quotes safely
    export EDIT_FILE_PATH="$FILE_PATH" EDIT_OLD_STRING="$OLD_STRING"
    export EDIT_NEW_STRING="$NEW_STRING"
    python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
new_string = os.environ['EDIT_NEW_STRING']
with open(file_path, 'r') as f:
    content = f.read()
content = content.replace(old_string, new_string)
with open(file_path, 'w') as f:
    f.write(content)
"

    echo -e "${GREEN}✓ Successfully replaced $OCCURRENCE_COUNT occurrence(s)${NC}"

else
    # Single replacement mode - requires exactly one occurrence
    if [ "$OCCURRENCE_COUNT" -eq 0 ]; then
        echo -e "${RED}Error: String not found in file${NC}" >&2
        exit 3
    elif [ "$OCCURRENCE_COUNT" -gt 1 ]; then
        echo -e "${RED}Error: String appears $OCCURRENCE_COUNT times (not unique)${NC}" >&2
        echo -e "${YELLOW}Hint: Use replace_all=true to replace all occurrences${NC}" >&2
        exit 3
    fi

    # Exactly one occurrence - safe to replace
    # Pass strings via environment variables to handle embedded quotes safely
    export EDIT_FILE_PATH="$FILE_PATH" EDIT_OLD_STRING="$OLD_STRING"
    export EDIT_NEW_STRING="$NEW_STRING"
    python3 -c "
import os
file_path = os.environ['EDIT_FILE_PATH']
old_string = os.environ['EDIT_OLD_STRING']
new_string = os.environ['EDIT_NEW_STRING']
with open(file_path, 'r') as f:
    content = f.read()
content = content.replace(old_string, new_string, 1)
with open(file_path, 'w') as f:
    f.write(content)
"

    echo -e "${GREEN}✓ Successfully replaced 1 occurrence${NC}"
fi

# Show the changed section (context around the change)
echo ""
echo "Changed section:"
echo "----------------------------------------"
grep -n -C 2 "$NEW_STRING" "$FILE_PATH" || echo "(No context available)"
echo "----------------------------------------"

exit 0
"""



SAFE_SPECIAL_PATHS = {
    "/dev/null",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
    "/dev/tty",
}

SAFE_SPECIAL_PREFIXES = (
    "/dev/fd/",
    "/proc/self/",
)


def _is_safe_special_path(path: str) -> bool:
    cleaned = path.strip(")'\"`);}]")
    return cleaned in SAFE_SPECIAL_PATHS or cleaned.startswith(SAFE_SPECIAL_PREFIXES)


def _extract_directory(path_str: str) -> str | None:
    """Resolve a file path to an absolute canonical path for security validation.

    Args:
        path_str: A file or directory path (can be relative or absolute)

    Returns:
        The resolved absolute path, or None if the path is invalid
    """
    try:
        path = Path(path_str)

        # Resolve relative paths to absolute paths using current working directory
        # This is important for security validation of relative paths
        if not path.is_absolute():
            path = Path.cwd() / path

        # Resolve to get canonical path (handles .., ., etc.)
        return str(path.resolve())

    except Exception:
        return None

# Safari browser configuration for web scraping
SAFARI_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/18.2 Safari/605.1.15"
)

SAFARI_HEADERS = {
    "User-Agent": SAFARI_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_url(
    url: str,
    headers: dict[str, str],
    max_characters: int = 10000,
    timeout_seconds: float = 10.0,
) -> str:
    """
    Fetch and extract text content from a URL using BeautifulSoup.

    Args:
        url: The URL to fetch.
        headers: HTTP headers to use for the request.
        max_characters: Maximum number of characters to return.
        timeout_seconds: Request timeout in seconds.

    Returns:
        Extracted text content from the page.
    """
    import requests
    from bs4 import BeautifulSoup

    try:
        response = requests.get(
            url, headers=headers, timeout=timeout_seconds, allow_redirects=True
        )
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content elements
        non_content_tags = [
            "script",
            "style",
            "noscript",
            "header",
            "footer",
            "nav",
            "aside",
            "iframe",
            "svg",
        ]
        for tag in soup(non_content_tags):
            tag.decompose()

        # Find main content area
        main_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find(id=re.compile(r"content|main|article", re.IGNORECASE))
            or soup.find(class_=re.compile(r"content|main|article", re.IGNORECASE))
            or soup.body
            or soup
        )

        text = re.sub(r"\s+", " ", main_content.get_text(separator=" ", strip=True)).strip()

        if len(text) > max_characters:
            text = text[:max_characters] + "... [truncated]"

        return text or "No readable content found."
    except requests.exceptions.Timeout:
        return "Failed to fetch content: Request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Failed to fetch content: HTTP {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"Failed to fetch content: {type(e).__name__}"
    except Exception as e:
        return f"Failed to fetch content: {str(e)}"


def _render_page_with_playwright(url: str, wait_selector: str | None = None) -> str:
    """
    Render a page using Playwright headless browser and return the HTML.

    Uses Safari/WebKit with anti-detection measures to avoid being blocked.

    Args:
        url: The URL to render.
        wait_selector: Optional CSS selector to wait for before extracting content.

    Returns:
        The rendered HTML content.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.webkit.launch(headless=True)
        context = browser.new_context(
            user_agent=SAFARI_USER_AGENT,
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            timezone_id="America/Los_Angeles",
            java_script_enabled=True,
            has_touch=False,
            is_mobile=False,
            device_scale_factor=2,
            extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            },
        )
        page = context.new_page()

        try:
            page.goto(url, wait_until="networkidle", timeout=30000)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=10000)
                except Exception:
                    pass
            page.wait_for_timeout(2000)
            return page.content()
        finally:
            browser.close()


def _extract_search_results(soup: Any, selector: str, max_results: int) -> list[tuple[str, str]]:
    """Extract search result links from parsed HTML.

    Args:
        soup: BeautifulSoup parsed HTML object.
        selector: CSS selector to find search result links.
        max_results: Maximum number of results to extract.

    Returns:
        list[tuple[str, str]]: List of (title, url) tuples for search results.
    """
    skip_domains = {"youtube.com", "maps.google", "accounts.google", "duckduckgo.com"}
    results: list[tuple[str, str]] = []

    for link in soup.select(selector):
        if len(results) >= max_results:
            break

        title = link.get_text(strip=True)
        href = link.get("href", "")
        url = href if isinstance(href, str) else (href[0] if href else "")

        if not title or not url or not url.startswith("http"):
            continue
        if any(domain in url for domain in skip_domains):
            continue

        results.append((title, url))

    return results


def search_web(query: str, max_results: int = 10) -> str:
    """
    Perform a web search and return the top search results with page contents.

    Tries DuckDuckGo first (more reliable for automated access), then falls back
    to Startpage if needed. Uses Playwright headless browser with Safari/WebKit
    to render JavaScript and avoid bot detection.

    Args:
        query: The search query.
        max_results: Maximum number of results to fetch content for. Defaults to 5.

    Returns:
        A string containing titles, links, and page contents of the top search results.
    """
    from urllib.parse import quote_plus

    from bs4 import BeautifulSoup

    # Search providers to try in order
    providers = [
        (
            f"https://duckduckgo.com/?q={quote_plus(query)}&t=h_&ia=web",
            "a[data-testid='result-title-a']",
        ),
        (
            f"https://www.startpage.com/sp/search?query={quote_plus(query)}",
            "a.result-link",
        ),
    ]

    for url, selector in providers:
        try:
            html = _render_page_with_playwright(url, wait_selector=selector)
            if "captcha" in html.lower():
                continue

            soup = BeautifulSoup(html, "html.parser")
            results = _extract_search_results(soup, selector, max_results)

            if results:
                formatted_results: list[str] = []
                for title, result_url in results:
                    content = fetch_url(result_url, SAFARI_HEADERS)
                    result_text = f"Title: {title}\nURL: {result_url}\nContent:\n{content}\n"
                    formatted_results.append(result_text)

                return "\n---\n".join(formatted_results)
        except Exception:
            continue

    return "No search results found."


def _strip_heredocs(command: str) -> str:
    """Strip heredoc content from a bash command.

    Removes everything between << DELIM and DELIM (or <<- DELIM and DELIM),
    so that heredoc body text is not parsed as command arguments.
    """
    return re.sub(
        r"<<-?\s*'?\"?(\w+)'?\"?\s*\n.*?\n\s*\1\b",
        "",
        command,
        flags=re.DOTALL,
    )


def parse_bash_command_paths(command: str) -> tuple[list[str], list[str]]:
    """Parse a bash command to extract readable and writable directory paths.

    This function analyzes bash commands to determine which directories are
    being read from and which are being written to.

    Args:
        command: A bash command string to parse

    Returns:
        A tuple of (readable_dirs, writable_dirs) where each is a list of directory paths

    """
    readable_paths: set[str] = set()
    writable_paths: set[str] = set()

    # EXPANDED: Commands that read files/directories
    read_commands = {
        "cat",
        "less",
        "more",
        "head",
        "tail",
        "grep",
        "find",
        "ls",
        "diff",
        "wc",
        "sort",
        "uniq",
        "cut",
        "sed",
        "awk",
        "od",
        "hexdump",
        "file",
        "stat",
        "du",
        "df",
        "tree",
        "read",
        "source",
        ".",
        "tar",
        "zip",
        "unzip",
        "gzip",
        "gunzip",
        "bzip2",
        "bunzip2",
        "python",
        "python3",
        "node",
        "ruby",
        "perl",
        "bash",
        "sh",
        "zsh",
        "make",
        "cmake",
        "gcc",
        "g++",
        "clang",
        "javac",
        "java",
        "cargo",
        "npm",
        "yarn",
        "pip",
        "go",
        "rustc",
        "rsync",
        # ADDED: Previously untracked commands
        "strings",
        "xxd",
        "nl",
        "fold",
        "rev",
        "pr",
        "fmt",
        "expand",
        "unexpand",
        "tr",
        "col",
        "colrm",
        "column",
        "join",
        "paste",
        "comm",
        "cmp",
        "look",
        "split",
        "csplit",
        "iconv",
        "base64",
        "base32",
        "md5sum",
        "sha1sum",
        "sha256sum",
        "cksum",
        "sum",
        "readlink",
        "realpath",
        "dirname",
        "basename",
        "pathchk",
    }

    # Commands that write files/directories
    write_commands = {
        "touch",
        "mkdir",
        "rm",
        "rmdir",
        "mv",
        "cp",
        "dd",
        "tee",
        "install",
        "chmod",
        "chown",
        "chgrp",
        "ln",
        "rsync",
    }

    write_redirects = {
        ">",
        ">>",
        "&>",
        "&>>",
        "1>",
        "2>",
        "2>&1",
        ">|",
        ">>|",
        "&>|",
        "1>>",
        "2>>",
    }

    try:
        # Strip heredoc content so body text is not parsed as arguments
        command = _strip_heredocs(command)

        # Handle pipes - split into sub-commands
        pipe_parts = command.split("|")

        for part in pipe_parts:
            part = part.strip()

            # Check for output redirection (writing)
            for redirect in write_redirects:
                if redirect in part:
                    # Extract path after redirect
                    redirect_match = re.search(rf"{re.escape(redirect)}\s*([^\s;&|()]+)", part)
                    if redirect_match:
                        path = redirect_match.group(1).strip()
                        path = path.strip("'\"")
                        if path and not _is_safe_special_path(path):
                            dir_path = _extract_directory(path)
                            if dir_path:
                                writable_paths.add(dir_path)

            # Check for input redirection (reading)
            input_redirect_match = re.search(r"<\s*([^\s;&|()]+)", part)
            if input_redirect_match:
                path = input_redirect_match.group(1).strip()
                path = path.strip("'\"")
                if path and not _is_safe_special_path(path):
                    dir_path = _extract_directory(path)
                    if dir_path:
                        readable_paths.add(dir_path)

            # Parse the command tokens
            try:
                tokens = shlex.split(part)
            except ValueError:
                # If shlex fails, do basic split
                tokens = part.split()

            if not tokens:
                continue

            cmd = tokens[0].split("/")[-1]  # Get base command name

            # Process based on command type
            if cmd in read_commands or cmd in write_commands:
                # Extract file/directory arguments (skip flags and redirects)
                paths: list[str] = []
                i = 1
                while i < len(tokens):
                    token = tokens[i]

                    # Skip flags and their arguments
                    if token.startswith("-"):
                        i += 1
                        # Skip flag argument if it doesn't start with - or /
                        if (
                            i < len(tokens)
                            and not tokens[i].startswith("-")
                            and not tokens[i].startswith("/")
                        ):
                            i += 1
                        continue

                    # Skip redirect operators and their targets
                    redirect_ops = [
                        ">", ">>", "<", "<<", "<<<", "&>", "&>>", "1>", "2>",
                        "2>&1", ">|", ">>|", "&>|", "1>>", "2>>",
                    ]
                    if token in redirect_ops or token.startswith("<<"):
                        i += 1
                        # Skip the redirect target (next token)
                        if i < len(tokens):
                            i += 1
                        continue

                    # Check if it looks like a path
                    if "/" in token or not any(c in token for c in ["=", "$", "(", ")"]):
                        token = token.strip("'\"")
                        if token and not _is_safe_special_path(token):
                            paths.append(token)

                    i += 1

                # Classify paths based on command
                if cmd in read_commands:
                    for path in paths:
                        dir_path = _extract_directory(path)
                        if dir_path:
                            readable_paths.add(dir_path)

                if cmd in write_commands:
                    # For write commands, typically the last path is written to
                    if paths:
                        if cmd in ["cp", "mv", "rsync"]:
                            # Source(s) are read, destination is written
                            for path in paths[:-1]:
                                dir_path = _extract_directory(path)
                                if dir_path:
                                    readable_paths.add(dir_path)

                            # Last path is destination
                            if len(paths) > 0:
                                dir_path = _extract_directory(paths[-1])
                                if dir_path:
                                    writable_paths.add(dir_path)
                        elif cmd == "dd":
                            # Special handling for dd command
                            # Look for of= parameter
                            for token in tokens:
                                if token.startswith("of="):
                                    output_file = token[3:]
                                    if not _is_safe_special_path(output_file):
                                        dir_path = _extract_directory(output_file)
                                        if dir_path:
                                            writable_paths.add(dir_path)
                                elif token.startswith("if="):
                                    input_file = token[3:]
                                    if not _is_safe_special_path(input_file):
                                        dir_path = _extract_directory(input_file)
                                        if dir_path:
                                            readable_paths.add(dir_path)
                        else:
                            # Other write commands
                            for path in paths:
                                dir_path = _extract_directory(path)
                                if dir_path:
                                    writable_paths.add(dir_path)

                # tee reads stdin and writes to file
                if cmd == "tee":
                    for path in paths:
                        dir_path = _extract_directory(path)
                        if dir_path:
                            writable_paths.add(dir_path)

    except Exception as e:
        # If parsing fails completely, return empty lists
        print(f"Failed to parse command '{command}': {e}")
        return ([], [])

    # Clean up paths - remove empty strings and '.'
    readable_dirs = sorted([p for p in readable_paths if p and p != "."])
    writable_dirs = sorted([p for p in writable_paths if p and p != "."])

    return (readable_dirs, writable_dirs)


class UsefulTools:
    """A hardened collection of useful tools with improved security."""

    def __init__(
        self,
        base_dir: str,
        readable_paths: list[str] | None = None,
        writable_paths: list[str] | None = None,
    ) -> None:
        """Initialize UsefulTools with security-restricted paths.

        Args:
            base_dir: The base directory for tool operations.
            readable_paths: Optional list of paths allowed for read operations.
            writable_paths: Optional list of paths allowed for write operations.
        """
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self.base_dir = str(Path(base_dir).resolve())
        self.readable_paths = [Path(p).resolve() for p in readable_paths or []]
        self.writable_paths = [Path(p).resolve() for p in writable_paths or []]

    def Edit(  # noqa: N802
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        timeout_seconds: float = 30,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.
            timeout_seconds: Timeout in seconds for the edit command.

        Returns:
            The output of the edit operation.
        """

        # Check if file_path is in writable_paths
        resolved = Path(file_path).resolve()
        if not is_subpath(resolved, self.writable_paths):
            return f"Error: Access denied for writing to {file_path}"

        # Create a temporary script file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write(EDIT_SCRIPT)
            script_path = f.name

        try:
            # Make script executable
            Path(script_path).chmod(0o755)

            # Build command with arguments
            replace_all_str = "true" if replace_all else "false"
            command = [
                "/bin/bash",
                script_path,
                str(resolved),
                old_string,
                new_string,
                replace_all_str,
            ]

            # Execute with timeout for safety
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            # Include stderr which contains the actual error message from the script
            error_msg = e.stderr.strip() if e.stderr else str(e)
            return f"Error: {error_msg}"
        except Exception as e:
            return f"Error: {e}"
        finally:
            # Clean up temporary script
            try:
                Path(script_path).unlink()
            except Exception:
                pass

    def MultiEdit(  # noqa: N802
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        timeout_seconds: float = 30,
    ) -> str:
        """Performs precise string replacements in files with exact matching.

        Args:
            file_path: Absolute path to the file to modify.
            old_string: Exact text to find and replace.
            new_string: Replacement text, must differ from old_string.
            replace_all: If True, replace all occurrences.
            timeout_seconds: Timeout in seconds for the edit command.

        Returns:
            The output of the edit operation.
        """
        return self.Edit(file_path, old_string, new_string, replace_all, timeout_seconds)

    def Bash(self, command: str, description: str, timeout_seconds: float = 30) -> str:  # noqa: N802
        """Runs a bash command and returns its output.

        Args:
            command: The bash command to run.
            description: A brief description of the command.
            timeout_seconds: Timeout in seconds for the command.

        Returns:
            The output of the command.
        """

        # Parse and validate paths
        readable, writable = parse_bash_command_paths(command)

        for path_str in readable:
            if _is_safe_special_path(path_str):
                continue
            resolved = Path(path_str).resolve()
            if not is_subpath(resolved, self.readable_paths):
                return f"Error: Access denied for reading {path_str}"

        for path_str in writable:
            if _is_safe_special_path(path_str):
                continue
            resolved = Path(path_str).resolve()
            if not is_subpath(resolved, self.writable_paths):
                return f"Error: Access denied for writing to {path_str}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            return "Error: Command execution timeout"
        except subprocess.CalledProcessError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

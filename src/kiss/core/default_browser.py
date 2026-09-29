# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""Resolve the machine's default web browser to a streamable executable.

The browser tab (``kiss.server.browser_tab``) drives a real browser
through the Chrome DevTools Protocol, so it needs a Chromium-family
binary: Chrome, Chromium, Brave, Edge, Vivaldi or Opera.  This module
finds the browser the user made their default (the one ``xdg-open``,
``open`` or ``start`` would launch) and reports whether it can be
streamed.  Firefox and Safari have no DevTools protocol (Firefox removed
CDP in version 141), so for them the resolver falls back to any
Chromium-family browser installed on the machine and, failing that, to
Playwright's bundled Chromium.

Detection is best effort and never raises: every probe is wrapped, and
the result always describes something Playwright can launch.
"""

from __future__ import annotations

import importlib
import os
import plistlib
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROBE_TIMEOUT = 5.0

# Substrings (lower-cased) of an executable path or .desktop name that mark
# a Chromium-family browser, mapped to a display name.
_CHROMIUM_FAMILY: tuple[tuple[str, str], ...] = (
    ("brave", "Brave"),
    ("msedge", "Microsoft Edge"),
    ("microsoft-edge", "Microsoft Edge"),
    ("microsoft edge", "Microsoft Edge"),
    ("vivaldi", "Vivaldi"),
    ("opera", "Opera"),
    ("chromium", "Chromium"),
    ("google-chrome", "Google Chrome"),
    ("google chrome", "Google Chrome"),
    ("chrome", "Google Chrome"),
)

_LINUX_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "brave-browser",
    "microsoft-edge",
    "vivaldi",
    "opera",
)

_MAC_CANDIDATES = (
    "/Applications/Google Chrome.app",
    "/Applications/Chromium.app",
    "/Applications/Brave Browser.app",
    "/Applications/Microsoft Edge.app",
    "/Applications/Vivaldi.app",
)

_WINDOWS_CANDIDATES = (
    r"Google\Chrome\Application\chrome.exe",
    r"Microsoft\Edge\Application\msedge.exe",
    r"BraveSoftware\Brave-Browser\Application\brave.exe",
    r"Chromium\Application\chrome.exe",
)


@dataclass(frozen=True)
class ResolvedBrowser:
    """The browser the browser tab will launch.

    Attributes:
        name: Human-readable browser name (``"Google Chrome"``).
        executable: Path of the binary, or ``None`` for Playwright's
            bundled Chromium.
        is_default: True when this is the machine's default browser.
        note: One sentence explaining a fallback, or ``""``.
    """

    name: str
    executable: str | None
    is_default: bool
    note: str = ""


def family_name(path_or_name: str) -> str | None:
    """Return the Chromium-family display name for *path_or_name*.

    Args:
        path_or_name: Executable path, ``.desktop`` file name or
            application bundle path.

    Returns:
        The display name, or ``None`` when the browser is not
        Chromium-based (or unknown).
    """
    low = path_or_name.lower()
    for needle, name in _CHROMIUM_FAMILY:
        if needle in low:
            return name
    return None


def default_browser_executable() -> str | None:
    """Return the executable of the machine's default web browser.

    Honours ``$BROWSER`` first (a command line; its first word is the
    executable), then asks the platform: ``xdg-settings`` / ``xdg-mime``
    on Linux, ``NSWorkspace`` on macOS and the ``UrlAssociations``
    registry key on Windows.

    Returns:
        An absolute executable path, or ``None`` when no default browser
        could be determined.
    """
    env_browser = os.environ.get("BROWSER", "").strip()
    if env_browser:
        words = shlex.split(env_browser, posix=not sys.platform.startswith("win"))
        if words:
            return _which(words[0])
    if sys.platform == "darwin":
        return _mac_default_executable()
    if sys.platform.startswith("win"):
        return _windows_default_executable()
    return _linux_default_executable()


def resolve_browser() -> ResolvedBrowser:
    """Pick the browser the browser tab should launch.

    ``$KISS_BROWSER`` (an executable path) overrides every probe.  The
    machine's default browser is used when it is Chromium-based;
    otherwise the first installed Chromium-family browser is used, and
    finally Playwright's bundled Chromium.

    Returns:
        The resolved browser, never ``None``.
    """
    override = os.environ.get("KISS_BROWSER", "").strip()
    if override:
        exe = _which(override)
        if exe:
            return ResolvedBrowser(family_name(exe) or Path(exe).name, exe, False)
    default = default_browser_executable()
    if default:
        name = family_name(default)
        if name:
            return ResolvedBrowser(name, default, True)
        default_name = Path(default).stem or default
        note = (
            f"The default browser ({default_name}) has no DevTools protocol "
            "and cannot be streamed."
        )
    else:
        default_name = ""
        note = "No default browser is configured on this machine."
    installed = _first_installed_candidate()
    if installed:
        name = family_name(installed) or Path(installed).name
        return ResolvedBrowser(name, installed, False, note)
    return ResolvedBrowser("Chromium (bundled)", None, False, note)


def _which(command: str) -> str | None:
    """Return the absolute path of *command* when it exists and is executable."""
    path = Path(command).expanduser()
    if path.is_file() and os.access(path, os.X_OK):
        return str(path)
    return shutil.which(command)


def _run(argv: list[str]) -> str:
    """Run *argv* and return its stripped stdout, or ``""`` on any failure."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_PROBE_TIMEOUT, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _linux_default_executable() -> str | None:
    """Resolve the ``.desktop`` entry ``xdg-open`` would use for https URLs."""
    desktop = _run(["xdg-settings", "get", "default-web-browser"]) or _run(
        ["xdg-mime", "query", "default", "x-scheme-handler/https"]
    )
    if not desktop:
        return None
    for directory in _xdg_application_dirs():
        candidate = directory / desktop
        if candidate.is_file():
            exe = _desktop_exec(candidate)
            if exe:
                return exe
    # No .desktop file found: the name itself is often the binary
    # ("google-chrome.desktop" -> "google-chrome").
    return _which(desktop.removesuffix(".desktop"))


def _xdg_application_dirs() -> list[Path]:
    """Directories searched for ``.desktop`` files, user dir first."""
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [Path(d) / "applications" for d in [data_home, *data_dirs.split(":")] if d]


def _desktop_exec(desktop_file: Path) -> str | None:
    """Return the executable named by the ``Exec=`` line of a ``.desktop`` file."""
    try:
        lines = desktop_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("Exec="):
            words = shlex.split(line[len("Exec=") :])
            # Skip wrappers such as "env FOO=bar chromium %U".
            for word in words:
                if word == "env" or "=" in word or word.startswith("%"):
                    continue
                return _which(word)
    return None


def _mac_default_executable() -> str | None:
    """Ask LaunchServices which app opens https URLs and return its binary."""
    script = (
        "ObjC.import('AppKit');"
        "$.NSWorkspace.sharedWorkspace.URLForApplicationToOpenURL("
        "$.NSURL.URLWithString('https:')).path.js"
    )
    app = _run(["osascript", "-l", "JavaScript", "-e", script])
    return _mac_app_executable(app) if app else None


def _mac_app_executable(app_path: str) -> str | None:
    """Return the main binary of a macOS ``.app`` bundle."""
    info = Path(app_path) / "Contents" / "Info.plist"
    try:
        with info.open("rb") as fh:
            executable = plistlib.load(fh).get("CFBundleExecutable")
    except (OSError, plistlib.InvalidFileException, ValueError):
        return None
    if not isinstance(executable, str):
        return None
    return _which(str(Path(app_path) / "Contents" / "MacOS" / executable))


def _windows_default_executable() -> str | None:
    """Follow the ``UrlAssociations\\https`` ProgId to its ``open`` command."""
    try:
        winreg: Any = importlib.import_module("winreg")
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        ) as key:
            prog_id = winreg.QueryValueEx(key, "ProgId")[0]
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command") as key:
            command = winreg.QueryValueEx(key, "")[0]
    except OSError:
        return None
    words = shlex.split(str(command), posix=False)
    return _which(words[0].strip('"')) if words else None


def _first_installed_candidate() -> str | None:
    """Return the first well-known Chromium-family browser found on this machine."""
    if sys.platform == "darwin":
        for app in _MAC_CANDIDATES:
            exe = _mac_app_executable(app)
            if exe:
                return exe
        return None
    if sys.platform.startswith("win"):
        root_vars = ("PROGRAMFILES", "PROGRAMFILES(X86)", "LOCALAPPDATA")
        roots = [os.environ.get(v, "") for v in root_vars]
        for root in filter(None, roots):
            for rel in _WINDOWS_CANDIDATES:
                exe = _which(str(Path(root) / rel))
                if exe:
                    return exe
        return None
    for name in _LINUX_CANDIDATES:
        exe = shutil.which(name)
        if exe:
            return exe
    return None

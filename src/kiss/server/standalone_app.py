# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The webapp as a standalone desktop app: the ``kiss-app`` command.

``kiss-app`` opens the webapp the kiss-web daemon serves in its own
window instead of a browser tab: a Chromium-family browser started in
*app mode* (``--app=<URL>``), which draws no address bar, no tab strip
and no browser menus, under its own profile directory
(``$KISS_HOME/webapp-profile``) so it is a separate instance from the
user's everyday browser with its own cookies and window placement.  The
engine is the first of: the browser named by ``$KISS_APP_BROWSER``, a
system Google Chrome / Chromium / Microsoft Edge / Brave, or the
Chromium that Playwright downloads (``install.sh`` installs it with the
app, so it is always there on an installed copy).

``kiss-app --install-launcher`` registers the app with the desktop so
it can be started like any other program: a ``.desktop`` entry under
``~/.local/share/applications`` on Linux, an application bundle under
``~/Applications`` on macOS.  Both carry the brand's product name and
icon (``kiss.core.brand`` and the ``kiss-icon.png`` next to it), so a
white-label brand gets its own launcher.  ``install.sh`` runs this,
then ``kiss-app`` itself, then ``kiss-app --reveal-launcher`` (the
launcher's folder in the file manager, so the user sees where the app
is) at the end of every install.

The running window shows that icon too, not the browser's:

* Linux: the desktop matches the window to the ``.desktop`` entry.  On
  X11 the window's ``WM_CLASS`` class is the ``--class`` we pass (the
  brand slug, also the entry's file name); on Wayland Chromium ignores
  ``--class`` and names an ``--app`` window after its URL and profile
  (:func:`wayland_app_id`), which the entry's ``StartupWMClass`` repeats.
* macOS: the Dock shows the icon of the bundle the running executable
  lives in, so the bundle is a clone of the browser's: a copy of the
  browser's binary and of its ``Contents/Frameworks``, with an
  ``Info.plist`` copied from the browser and renamed after the brand
  (:func:`write_macos_launcher`).  Real copies, not symlinks: Chromium's
  helper processes (GPU, renderers, network) run under a seatbelt
  sandbox that lets them read only the canonical path of the main
  bundle, so a framework reached through a symlink into
  ``/Applications/Google Chrome.app`` is blocked and the browser dies
  with "GPU process isn't usable".  On APFS the copies are clones
  (``cp -c``: instant, no extra disk until either side changes).  The
  copied binary is re-signed ad hoc under the brand's identifier: the
  browser's own signature seals its ``Info.plist``, and the kernel
  kills a binary whose sealed plist differs.  ``kiss-app`` always
  starts the browser through that copy on macOS and refreshes the
  clone when the browser's version changes.

A click on the icon of the running app (in the Dock, in Finder, in
Launchpad) must bring its window to the front.  macOS answers such a
click with a *reopen* event, and Chromium's handler for it focuses the
browser's windows except *app* windows (the ``--app`` kind: since
Chromium 92 they are skipped, as the hidden mirrors of its own web-app
windows live among them), then opens a new, empty browser window when
it found none.  So on macOS the ``--app`` page is not the webapp but
the daemon's ``media/appWindow.html`` (:func:`app_window_url`), which
opens the webapp as a *popup* window of the same bounds and closes
itself: a popup has the same frame (a title bar, no browser UI), is
focused by the reopen handler, and is restored from the Dock when
minimized.  Chromium does not remember the bounds of popups between
launches, so the webapp's ``appWindow.js`` saves them (``localStorage``
of the daemon's origin) and ``appWindow.html`` opens the next popup
with them.  ``--disable-popup-blocking`` lets that page open the popup
without a click.  A second ``kiss-app`` while the window is open
(``install.sh`` after an update) brings it to the front the same way
the icon does (``open -a``, :func:`macos_activate_running`) instead
of opening another window.

Chromium on macOS keeps running after its last window closes, and a
reopen sent to a browser without windows (a click on its icon, that
``open -a``) opens the empty browser window again.  So the popup
marks its WebSocket connection to the daemon (``/ws?appWindow=1``,
:data:`APP_WINDOW_WS_QUERY`), the daemon counts those connections
(:func:`app_window_event`) and quits the browser whose last one
closed with its page and that has no window left
(:func:`quit_closed_app`): the next click starts the app afresh.  A
browser whose popup closed while the daemon was down (a restart) is
caught by the daemon's periodic sweep (:func:`sweep_windowless_app`).
``kiss-app`` asks the daemon whether the popup is connected
(:data:`APP_WINDOW_API`, :func:`app_window_closed`) before it trusts a
reopen, and hands its command line to a running browser without a
window instead (Chromium's process singleton opens the ``--app`` page
in the running instance, which opens a new popup; the page opens none
when the daemon tells it the popup is connected after all).

``kiss-app`` waits for the daemon's URL file (:func:`wait_for_url`)
but not for the daemon: the file outlives a reboot, and macOS
relaunches the app at login a minute before the daemon's LaunchAgent
is spawned.  While the daemon does not answer, the window opens on
the packaged ``media/appStart.html`` (:func:`start_page_url`, a
``file://`` page) which shows the app's name, asks the daemon every
half second and replaces itself with the app at the first answer;
a browser started on the daemon's URL that early would show an error
page instead.
"""

from __future__ import annotations

import argparse
import functools
import http.client
import json
import math
import os
import platform
import plistlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import IO
from xml.parsers.expat import ExpatError

from kiss.core.brand import BRAND
from kiss.core.config import kiss_home, read_remote_urls, remote_url_file
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.processes import run_process_group
from kiss.core.utils import atomic_write_text
from kiss.server.helpers import MEDIA_DIR

ICON_FILE = MEDIA_DIR / "kiss-icon.png"
PROFILE_DIR_NAME = "webapp-profile"
LOOPBACK_URL = "http://127.0.0.1/"
"""The host the launcher opens (the daemon's loopback URL, any port)."""

# Where a stock install puts each engine, by platform.  Checked in order;
# the first executable wins.
_LINUX_BROWSERS = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
    "microsoft-edge-stable",
    "brave-browser",
)
_MACOS_BROWSERS = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
)
_WINDOWS_BROWSERS = (
    r"Google\Chrome\Application\chrome.exe",
    r"Microsoft\Edge\Application\msedge.exe",
    r"BraveSoftware\Brave-Browser\Application\brave.exe",
)


def brand_slug(brand: dict[str, str] = BRAND) -> str:
    """Return the brand's product name as a file-name / WM_CLASS safe slug.

    ``"KISS Sorcar"`` becomes ``kiss-sorcar``; the slug names the
    ``.desktop`` file and is the window class the app's windows carry,
    which is how a Linux desktop matches them to the launcher's icon.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", brand["product_name"].lower()).strip("-")
    return slug or "kiss-sorcar"


def macos_bundle_identifier(brand: dict[str, str] = BRAND) -> str:
    """Return the ``CFBundleIdentifier`` of the brand's app bundle (and its code signature)."""
    return f"org.kiss-sorcar.{brand_slug(brand)}"


def read_loopback_url(url_file: Path) -> str | None:
    """Return the daemon's plain-HTTP loopback URL from *url_file*, or None.

    ``remote-url.json`` carries ``loopback`` (``http://127.0.0.1:PORT``);
    a file written by an older daemon only has ``local``
    (``http://localhost:PORT``), whose host is rewritten to 127.0.0.1.
    None when the file is missing, unreadable or has neither URL.
    """
    urls = read_remote_urls(url_file)
    loopback = urls.get("loopback")
    if isinstance(loopback, str) and loopback:
        return loopback
    local = urls.get("local")
    if isinstance(local, str) and local:
        return local.replace("localhost", "127.0.0.1", 1)
    return None


_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
"""Opens the daemon's URL without the environment's proxy.

The probes below ask the daemon on this machine: an ``http_proxy``
without a matching ``no_proxy`` would send them to a proxy, which cannot
reach this machine's loopback, and the daemon would never seem to answer.
"""


def daemon_answers(url: str) -> bool:
    """Whether an HTTP server answers at *url* (with any status: a 404 is an answer too).

    Another server on the port that does not speak HTTP is no answer.
    """
    try:
        with _DIRECT_OPENER.open(urllib.parse.urljoin(url, APP_WINDOW_API), timeout=2.0):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError, http.client.HTTPException):
        return False


def wait_for_url(url_file: Path, url: str | None, timeout: float) -> str | None:
    """Return *url*, else the loopback URL of *url_file* once it exists (None after *timeout* s).

    The file (``remote-url.json``) is polled for while it is missing:
    the daemon writes it as it starts.  Whether the daemon answers at
    the URL is not waited for (:func:`start_page_url` shows a window
    meanwhile).
    """
    deadline = time.monotonic() + timeout
    while True:
        url = url or read_loopback_url(url_file)
        if url or time.monotonic() >= deadline:
            return url
        time.sleep(0.5)


APP_START_PAGE = MEDIA_DIR / "appStart.html"
"""The packaged page the window shows while the daemon does not answer yet."""


def start_page_url(
    next_url: str, daemon_url: str, brand: dict[str, str] = BRAND, popup: bool = False,
) -> str:
    """Return the ``file://`` URL of :data:`APP_START_PAGE` set to wait for the daemon.

    The page asks ``daemon_url``'s :data:`APP_WINDOW_API` until it
    answers (with any status) and then replaces itself with
    *next_url* (the app); the window is titled after *brand*.  With
    *popup* (macOS, *next_url* the popup-opening page of
    :func:`app_window_url`) the page first moves itself into the
    popup window the app lives in there (a reopen from the Dock
    finds that window; it would find no app window and open an empty
    browser window).
    """
    query = {
        "next": next_url,
        "probe": urllib.parse.urljoin(daemon_url, APP_WINDOW_API),
        "title": brand["product_name"],
    }
    if popup:
        query["popup"] = "1"
    return f"{APP_START_PAGE.as_uri()}?{urllib.parse.urlencode(query)}"


PLAYWRIGHT_TIMEOUT_SECS = 60.0
"""How long :func:`playwright_chromium` waits for Playwright's driver to name the browser.

The lookup is answered within a second or two once the driver is up;
one whose driver starts and never answers must not park ``kiss-app``
before its window, so the lookup is given up on at this deadline.
"""

_PLAYWRIGHT_LOOKUP = (
    "from playwright.sync_api import sync_playwright\n"
    "with sync_playwright() as pw:\n"
    "    print(pw.chromium.executable_path)\n"
)


@functools.cache
def playwright_chromium() -> Path | None:
    """Return the Chromium executable Playwright installed, or None.

    Asks Playwright itself (``BrowserType.executable_path``), which
    resolves the browsers directory and revision the installed
    ``playwright`` package expects; nothing is launched.  None when the
    package or the browser is missing.  The question is put to a child
    interpreter: entering ``sync_playwright()`` starts the Playwright
    node driver and waits for its handshake with no deadline, so a
    driver that stays silent is given up on, and killed with that
    child, after :data:`PLAYWRIGHT_TIMEOUT_SECS`.  Cached: both
    :func:`find_browser` and :func:`sandboxed` need the answer in one
    ``kiss-app`` run (conc1018 H1-F2).
    """
    try:
        result = run_process_group(
            [sys.executable, "-c", _PLAYWRIGHT_LOOKUP], PLAYWRIGHT_TIMEOUT_SECS
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    path = Path(result.stdout.strip())
    return path if path.is_file() else None


def find_browser(system: str = platform.system()) -> Path | None:
    """Return the Chromium-family browser to run the app in, or None.

    ``$KISS_APP_BROWSER`` (a path or a command name), when set, is the
    only candidate: None when it resolves to no executable.  Without it
    the system browsers of *system* (``Linux``, ``Darwin`` or
    ``Windows``) in the order of ``_LINUX_BROWSERS`` /
    ``_MACOS_BROWSERS`` / ``_WINDOWS_BROWSERS``; last the Playwright
    Chromium.
    """
    override = os.environ.get("KISS_APP_BROWSER", "").strip()
    if override:
        found = shutil.which(override)
        return Path(found) if found else None
    if system == "Linux":
        candidates: list[str] = [p for n in _LINUX_BROWSERS if (p := shutil.which(n))]
    elif system == "Darwin":
        candidates = list(_MACOS_BROWSERS)
    elif system == "Windows":
        roots = [
            os.environ.get(v, "") for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")
        ]
        candidates = [os.path.join(r, b) for r in roots if r for b in _WINDOWS_BROWSERS]
    else:
        candidates = []
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file() and os.access(path, os.X_OK):
            return path
    return playwright_chromium()


def chromium_executable_name(browser: Path) -> str:
    """Return Chromium's ``kBrowserProcessExecutableName`` for *browser*.

    The prefix of the Wayland app id: ``brave`` for Brave, ``msedge``
    for Microsoft Edge, ``chrome`` for Google Chrome and Chromium (the
    constant is ``"chrome"`` for both in ``chrome/common/chrome_constants.cc``).
    """
    name = browser.name.lower()
    if "brave" in name:
        return "brave"
    if "edge" in name:
        return "msedge"
    return "chrome"


def wayland_app_id(browser: Path, slug: str, url: str = LOOPBACK_URL) -> str:
    """Return the Wayland app id Chromium gives the ``--app=url`` window.

    Chromium names an app window's ``xdg_toplevel`` after the desktop
    entry it would write for the site, ``<executable>-<app name>-<profile>``
    (``shell_integration_linux::GetXdgAppIdForWebApp``), where the app
    name is the URL's host and path joined by ``_``, the profile is the
    ``--profile-directory`` of :func:`app_command` (the brand *slug*,
    so a stock and a branded install get different ids and the desktop
    can tell their windows apart), and characters a file name cannot
    hold (``/`` among them) become ``_``.  The ``--class`` switch does
    not reach it.  ``http://127.0.0.1:8787/`` in Chrome gives
    ``chrome-127.0.0.1__-kiss-sorcar``.
    """
    parts = urllib.parse.urlsplit(url)
    raw = f"{chromium_executable_name(browser)}-{parts.hostname}_{parts.path or '/'}-{slug}"
    return re.sub(r'[\\/:*?"<>| ]', "_", raw)


def sandboxed(browser: Path, system: str = platform.system()) -> bool:
    """Return whether *browser* can run with Chromium's sandbox on.

    Playwright's Chromium ships without the setuid sandbox helper, so
    on a Linux distribution that restricts unprivileged user namespaces
    (Ubuntu 23.10+ through its AppArmor profile, many containers) it
    exits at once with "No usable sandbox!".  Playwright launches it
    with ``--no-sandbox`` for that reason, and so does the app.  System
    browsers keep their sandbox, as does every browser on macOS and
    Windows.
    """
    if system != "Linux":
        return True
    bundled = playwright_chromium()
    return bundled is None or browser.resolve() != bundled.resolve()


APP_WINDOW_PAGE = "/media/appWindow.html"
"""The daemon's page that opens the webapp as a popup window (macOS, see the module doc)."""


def app_window_url(url: str, brand: dict[str, str] = BRAND) -> str:
    """Return the URL of the daemon page that opens the webapp at *url* as a popup window.

    The page is :data:`APP_WINDOW_PAGE` on *url*'s origin (the popup
    must be the same origin to find the bounds the webapp saved for
    it); its query names the webapp URL to open and the brand's
    product name as the page's title while it is on screen.
    """
    page = urllib.parse.urljoin(url, APP_WINDOW_PAGE)
    return f"{page}?{urllib.parse.urlencode({'url': url, 'title': brand['product_name']})}"


def app_command(
    browser: Path,
    url: str,
    profile_dir: Path,
    slug: str,
    *,
    sandbox: bool = True,
    popups: bool = False,
) -> list[str]:
    """Return the command line that opens *url* as a standalone app window.

    ``--app`` is Chromium's app mode (a window with no browser UI);
    ``--user-data-dir`` keeps the app's profile apart from the user's
    browser profile, which also makes it a separate browser instance;
    ``--profile-directory`` names the profile inside it after the brand
    *slug*, which Chromium folds into the window's Wayland app id
    (:func:`wayland_app_id`); ``--class`` sets the X11 WM_CLASS class to
    the slug, which the Linux launcher's file name matches (other
    platforms ignore it); the two ``--no-*`` flags skip Chromium's
    first-run and default-browser prompts, which have no place in an
    app window; the last two keep the app's profile out of the OS
    keyring (GNOME Keyring / KWallet unlock prompts, the macOS Keychain
    "wants to use Safe Storage" prompt that a bundle with its own
    identity would raise), as Playwright does for its browsers; without
    *sandbox* the browser runs unsandboxed (:func:`sandboxed`); with
    *popups* pages may open popup windows without a click
    (``--disable-popup-blocking``), which the macOS *url*, the page of
    :func:`app_window_url`, needs.
    """
    command = [
        str(browser),
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        f"--profile-directory={slug}",
        f"--class={slug}",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--use-mock-keychain",
    ]
    if not sandbox:
        command.append("--no-sandbox")
    if popups:
        command.append("--disable-popup-blocking")
    return command


STARTUP_GRACE_SECS = 2.0
"""How long a detached browser must survive to count as started."""


def launch(command: list[str], log: Path, *, foreground: bool) -> bool:
    """Run *command*: replace this process with it (*foreground*) or detach it.

    Detached, the browser is started in its own session with stdin and
    stdout closed and stderr in *log*, so it outlives the caller
    (``install.sh``, a terminal) and never inherits its file
    descriptors.  Returns True once it has run for ``STARTUP_GRACE_SECS``
    or exited 0 within them (Chromium hands the URL to an instance that
    already runs this profile and exits at once); False when it died
    with an error (no display, say), its last words in *log*.  In the
    foreground this call does not return.  ``OSError`` when *command*
    cannot be executed or *log* cannot be written: the caller reports it.
    """
    if foreground:
        os.execv(command[0], command)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stderr:
        return start_detached(command, stderr)


def start_detached(command: list[str], stderr: IO[bytes] | int) -> bool:
    """Start *command* in its own session with stdin and stdout closed and stderr on *stderr*.

    Returns True once it has run for ``STARTUP_GRACE_SECS`` or exited 0
    within them, False when it exited with an error within them.
    """
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=stderr,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=STARTUP_GRACE_SECS) == 0
    except subprocess.TimeoutExpired:
        return True


def desktop_exec_arg(value: Path) -> str:
    """Return *value* as one quoted ``Exec`` argument of a desktop entry.

    The spec splits ``Exec`` on spaces, reads ``%x`` as a field code
    (so a literal ``%`` is ``%%``) and lets double quotes group an
    argument, inside which ``"``, backtick, ``$`` and backslash are
    backslash-escaped.
    """
    text = str(value).replace("%", "%%")
    for ch in ("\\", '"', "`", "$"):
        text = text.replace(ch, "\\" + ch)
    return f'"{text}"'


def desktop_exec(exe: Path, kiss_home: Path) -> str:
    """Return the ``Exec`` value that runs *exe* for the state directory *kiss_home*."""
    return f"env KISS_HOME={desktop_exec_arg(kiss_home)} {desktop_exec_arg(exe)}"


def key_file_value(text: str) -> str:
    """Return *text* as a desktop entry string value.

    The entry is a key file first: a value's backslash starts an escape
    sequence (``\\s``, ``\\n``, ``\\t``, ``\\r``, ``\\\\``), so every literal
    backslash, including those :func:`desktop_exec_arg` produced, is
    doubled, and a newline or tab is written as its sequence.
    """
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t").replace("\r", "\\r")


def desktop_entry(
    brand: dict[str, str], icon: Path, exe: Path, kiss_home: Path, browser: Path
) -> str:
    """Return the Linux ``.desktop`` entry for the app.

    The desktop shows the entry's name and icon for the app's windows
    instead of the browser's when it can match them: on X11 the
    window's ``WM_CLASS`` class is the ``--class`` of
    :func:`app_command`, the brand slug, which is also the entry's file
    name; on Wayland the window's app id is :func:`wayland_app_id` for
    *browser*, which ``StartupWMClass`` carries.  ``Exec`` pins the
    state directory the app was installed with (:func:`desktop_exec`):
    a desktop session does not carry the ``KISS_HOME`` of the shell
    that ran the installer.  String values go through
    :func:`key_file_value`.
    """
    return (
        "[Desktop Entry]\n"
        "Version=1.5\n"
        "Type=Application\n"
        f"Name={key_file_value(brand['product_name'])}\n"
        f"Comment={key_file_value(brand['tagline'])}\n"
        f"Exec={key_file_value(desktop_exec(exe, kiss_home))}\n"
        f"Icon={key_file_value(str(icon))}\n"
        "Terminal=false\n"
        "Categories=Utility;Development;\n"
        f"StartupWMClass={key_file_value(wayland_app_id(browser, brand_slug(brand)))}\n"
        "SingleMainWindow=true\n"
    )


def applications_dir(home: Path) -> Path:
    """Return the directory of the user's ``.desktop`` entries.

    ``$XDG_DATA_HOME/applications`` when the variable holds an absolute
    path (the XDG base directory spec says to ignore any other value),
    else ``~/.local/share/applications``.
    """
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg.startswith("/") else home / ".local" / "share"
    return base / "applications"


def linux_launcher_path(home: Path, brand: dict[str, str] = BRAND) -> Path:
    """Return ``<applications dir>/<slug>.desktop``, the brand's launcher entry (written or not)."""
    return applications_dir(home) / f"{brand_slug(brand)}.desktop"


def write_linux_launcher(
    entry: Path, brand: dict[str, str], icon: Path, exe: Path, kiss_home: Path, browser: Path
) -> None:
    """Write the ``.desktop`` file *entry* (see :func:`linux_launcher_path`).

    Published whole (staged next to it, then renamed over it): the
    desktop reads the entry at any moment and must never see it empty
    or cut short.
    """
    atomic_write_text(entry, desktop_entry(brand, icon, exe, kiss_home, browser), mode=0o755)


SETUP_TIMEOUT_SECS = 120.0
"""How long each command that builds the launcher gets before it is given up on.

``sips``, ``iconutil``, ``cp``, ``codesign`` and
``update-desktop-database`` run from the installer and from a macOS
launch that rebuilds a stale bundle; a stuck one must not park either
for good.  The margin is for ``cp`` without clones: a plain copy of a
browser's Frameworks (a gigabyte) on a slow disk.
"""


def write_icns(png: Path, icns: Path) -> bool:
    """Convert *png* to the macOS icon file *icns*; True on success.

    Uses the ``sips`` and ``iconutil`` tools every macOS ships.  False
    (and no file) when either is missing, fails or does not finish
    within :data:`SETUP_TIMEOUT_SECS`, in which case the bundle is
    written without an icon.  The iconset is staged in a directory of
    this call's own (named ``*.iconset``, as ``iconutil`` wants): two
    conversions at once must not delete each other's inputs.
    """
    sips, iconutil = shutil.which("sips"), shutil.which("iconutil")
    if not sips or not iconutil:
        return False
    icns.parent.mkdir(parents=True, exist_ok=True)
    iconset = Path(tempfile.mkdtemp(prefix=f"{icns.stem}.", suffix=".iconset", dir=icns.parent))
    try:
        for size in (16, 32, 64, 128, 256, 512):
            for scale in (1, 2):
                pixels = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                run_process_group(
                    [sips, "-z", str(pixels), str(pixels), str(png), "--out", str(iconset / name)],
                    SETUP_TIMEOUT_SECS,
                ).check_returncode()
        run_process_group(
            [iconutil, "-c", "icns", str(iconset), "-o", str(icns)], SETUP_TIMEOUT_SECS
        ).check_returncode()
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        shutil.rmtree(iconset, ignore_errors=True)
    return icns.is_file()


def macos_launcher_script(exe: Path, kiss_home: Path) -> str:
    """Return the shell script of the macOS bundle: ``kiss-app --foreground`` for *kiss_home*."""
    return (
        "#!/bin/sh\n"
        f"KISS_HOME={shlex.quote(str(kiss_home))}\n"
        "export KISS_HOME\n"
        f"exec {shlex.quote(str(exe))} --foreground\n"
    )


def macos_browser_bundle(browser: Path) -> Path | None:
    """Return the ``.app`` bundle *browser* is the main executable of, or None.

    ``/Applications/Chromium.app/Contents/MacOS/Chromium`` gives
    ``/Applications/Chromium.app``; a bare binary (a ``$KISS_APP_BROWSER``
    outside any bundle) gives None.
    """
    parents = browser.parents
    if (
        len(parents) >= 3
        and parents[0].name == "MacOS"
        and parents[1].name == "Contents"
        and parents[2].suffix == ".app"
    ):
        return parents[2]
    return None


# Keys of the browser's Info.plist the brand bundle must not repeat: the
# registrations that would make it a second handler for the browser's
# URL schemes and document types or a second web browser, the
# asset-catalog icon name that would override CFBundleIconFile, the
# scripting definition and localized name that live in resources the
# clone does not carry, and the browser's updater (Keystone)
# bookkeeping.
_MACOS_PLIST_DROPPED_KEYS = (
    "CFBundleURLTypes",
    "CFBundleDocumentTypes",
    "CFBundleIconName",
    "NSUserActivityTypes",
    "UTExportedTypeDeclarations",
    "UTImportedTypeDeclarations",
    "NSServices",
    "ASWebAuthenticationSessionWebBrowserSupportCapabilities",
    "NSAppleScriptEnabled",
    "OSAScriptingDefinition",
    "LSHasLocalizedDisplayName",
)


def macos_bundle_plist(
    brand: dict[str, str], browser_plist: dict[str, object]
) -> dict[str, object]:
    """Return the brand bundle's ``Info.plist`` built from the browser's.

    Everything Chromium needs at run time comes along (the usage
    descriptions that let it ask for the microphone and camera, the
    high-resolution and graphics-switching flags, the minimum system
    version); the identity becomes the brand's, the executable the
    launcher script, the icon ``AppIcon.icns``; the keys in
    ``_MACOS_PLIST_DROPPED_KEYS`` and the Keystone ``KS*`` keys go.
    An empty *browser_plist* (the launcher-only bundle, no browser
    cloned) yields the brand's identity with version ``1.0``.
    """
    info: dict[str, object] = {"CFBundleVersion": "1.0"}
    info.update(
        (key, value)
        for key, value in browser_plist.items()
        if key not in _MACOS_PLIST_DROPPED_KEYS and not key.startswith("KS")
    )
    info.update(
        {
            "CFBundleName": brand["product_name"],
            "CFBundleDisplayName": brand["product_name"],
            "CFBundleIdentifier": macos_bundle_identifier(brand),
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "launcher",
            "CFBundleIconFile": "AppIcon",
            "LSUIElement": False,
        }
    )
    return info


def read_plist(path: Path) -> dict[str, object] | None:
    """Return the dictionary of the property list at *path*.

    None when the file is missing, cut short, not a property list, or
    one that is damaged: ``plistlib`` raises ``ValueError`` for a value
    or a key it cannot place and ``AttributeError`` for a date it
    cannot read, and loads a list or a string root as readily as a
    dictionary.
    """
    try:
        with path.open("rb") as f:
            info = plistlib.load(f)
    except (OSError, ValueError, AttributeError, ExpatError):
        return None
    return info if isinstance(info, dict) else None


def plist_version(info: dict[str, object]) -> tuple[object, object]:
    """Return the ``(CFBundleShortVersionString, CFBundleVersion)`` of a bundle's *info*."""
    return (info.get("CFBundleShortVersionString"), info.get("CFBundleVersion"))


def macos_bundle_browser(bundle: Path, browser: Path) -> Path | None:
    """Return the bundle's copy of *browser* when the clone is current, else None.

    Current means ``Contents/MacOS/<browser name>`` and
    ``Contents/Frameworks`` are real copies (not the symlinks of an
    older clone) and the bundle's ``Info.plist``, copied from the
    browser's when it was cloned, carries the browser's present version.
    A browser updated in place or replaced by a new Playwright revision
    changes its version; a different browser has a different name.
    """
    browser_bundle = macos_browser_bundle(browser)
    if browser_bundle is None:
        return None
    exe_copy = bundle / "Contents" / "MacOS" / browser.name
    frameworks = bundle / "Contents" / "Frameworks"
    if exe_copy.is_symlink() or not exe_copy.is_file():
        return None
    if frameworks.is_symlink() or not frameworks.is_dir():
        return None
    info = read_plist(bundle / "Contents" / "Info.plist")
    browser_info = read_plist(browser_bundle / "Contents" / "Info.plist")
    if info is None or browser_info is None:
        return None
    version = plist_version(browser_info)
    if None in version or version != plist_version(info):
        return None
    return exe_copy


PROBE_TIMEOUT_SECS = 10.0
"""How long ``ps``, ``osascript`` and ``open`` get before a probe gives up on them.

The probes run from the daemon's watchdog and the popup's close
handler; a stuck command must not hang either.
"""


def _probe(command: list[str], timeout: float) -> subprocess.CompletedProcess[str] | None:
    """Run *command* with *timeout*; None when it did not come back in time or cannot start."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def macos_bundle_processes(
    bundle: Path, timeout: float = PROBE_TIMEOUT_SECS
) -> list[tuple[int, str]] | None:
    """Return ``(pid, command)`` of every process running from *bundle*.

    Every executable of the clone sits under ``Contents/``: the browser
    itself under ``Contents/MacOS/``, its helpers under
    ``Contents/Frameworks/``.  A running one names that directory in
    its command line.  ``-ww`` keeps the whole command line: without it
    procps ``ps`` cuts each line at 80 columns when its output is piped
    and ``TERM`` is unset, which drops a long bundle path.  None (not
    the empty list, which means no such process) when ``ps`` cannot
    start, exits with an error or does not answer within *timeout*
    seconds: the callers that would delete the bundle must not take
    that for "nothing runs".
    """
    listing = _probe(["ps", "-ww", "-axo", "pid=,command="], timeout)
    if listing is None or listing.returncode != 0:
        return None
    prefix = f"{bundle / 'Contents'}/"
    processes = []
    for line in listing.stdout.splitlines():
        pid, _, command = line.strip().partition(" ")
        if prefix in command:
            processes.append((int(pid), command))
    return processes


def macos_bundle_in_use(bundle: Path) -> bool | None:
    """Whether the browser or a helper runs from *bundle*; None when ``ps`` cannot tell."""
    processes = macos_bundle_processes(bundle)
    return None if processes is None else bool(processes)


def macos_bundle_pid(bundle: Path) -> int | None:
    """Return the pid of the browser started from *bundle*; None when none or ``ps`` cannot tell."""
    prefix = f"{bundle / 'Contents' / 'MacOS'}/"
    for pid, command in macos_bundle_processes(bundle) or []:
        if command.startswith(prefix):
            return pid
    return None


_WINDOWS_ON_SCREEN_JXA = """
ObjC.import('CoreGraphics');
function run(argv) {
  var ref = $.CGWindowListCopyWindowInfo(
    $.kCGWindowListOptionOnScreenOnly | $.kCGWindowListExcludeDesktopElements,
    $.kCGNullWindowID);
  var list = ObjC.deepUnwrap(ObjC.castRefToObject(ref));
  var count = 0;
  for (var i = 0; i < list.length; i++) {
    if (list[i].kCGWindowOwnerPID === parseInt(argv[0], 10) && list[i].kCGWindowLayer === 0) {
      count += 1;
    }
  }
  return String(count);
}
"""


def macos_windows_on_screen(pid: int, timeout: float = PROBE_TIMEOUT_SECS) -> int | None:
    """Return how many windows process *pid* has on screen, or None when it cannot be told.

    The count comes from the window server (JXA over CoreGraphics),
    which needs no accessibility permission.  A minimized window is off
    screen and not counted; Chromium's hidden helper windows sit on
    other layers.  None when ``osascript`` fails (no window server
    session, say) or does not answer within *timeout* seconds.
    """
    result = _probe(
        ["osascript", "-l", "JavaScript", "-e", _WINDOWS_ON_SCREEN_JXA, str(pid)], timeout
    )
    if result is None or result.returncode != 0 or not result.stdout.strip():
        return None
    return int(result.stdout)


APP_WINDOW_API = "/api/app-window"
"""The daemon route that tells ``kiss-app`` whether the popup window is open."""

APP_WINDOW_WS_QUERY = "appWindow=1"
"""The query of the popup's WebSocket URL (``/ws?appWindow=1``), which marks it as the popup."""

APP_WINDOW_QUIT_GRACE_SECS = 2.0
"""How long after the popup's page went away the daemon waits before quitting the browser."""

APP_WINDOW_GONE_CLOSE_CODE = 1001
"""The WebSocket close code (going away) Chromium sends when the popup's page is gone."""

APP_WINDOW_SWEEP_STRIKES = 3
"""How many sweeps in a row must find the browser without popup and window before it is quit."""

_app_lock = threading.Lock()
"""Guards :data:`_app_windows` and :data:`_quit_armings`, and the quit decided against them."""

_app_windows = 0
"""How many WebSocket connections of the daemon come from the popup (0: no popup, or none yet)."""

_quit_armings = 0
"""Quit timers :func:`app_window_event` armed plus popups connected; only the latest timer acts,
and the sweeps count their strikes under one value of it (:func:`sweep_windowless_app`)."""

_windowless: tuple[int | None, int, int] = (None, 0, 0)
"""The browser pid the last sweep found without popup and window, :data:`_quit_armings` as it
began to look, and how many sweeps in a row found the two so."""


def app_window_state() -> dict[str, bool]:
    """Return what the daemon serves at :data:`APP_WINDOW_API`: whether the popup is open."""
    return {"open": _app_windows > 0}


def app_window_event(event: str) -> bool:
    """Count the popup's *event*; return whether it was one of the known ones.

    The events are the popup's WebSocket connections: ``open`` when one
    opens, ``closing`` when one closes because its page went away (the
    window closed, or reloaded: the browser started the close, with
    code :data:`APP_WINDOW_GONE_CLOSE_CODE`), ``dropped`` when one
    closes any other way (the daemon's keepalive timed out, or the
    daemon started the close and the page answered with that code)
    while the page lives on and will connect again.  The ``closing``
    that leaves none arranges :func:`quit_closed_app` for the brand
    bundle (:func:`macos_bundle_path`, as of now) after
    :data:`APP_WINDOW_QUIT_GRACE_SECS`, by when a reload has connected
    again; a ``dropped`` one does not, as the page is still there (the
    window may be minimized, which the quit check cannot tell from
    closed), and leaves the browser to :func:`sweep_windowless_app`.
    Only macOS needs either: elsewhere Chromium quits by itself when
    its last window closes.  Each arming counts (:data:`_quit_armings`):
    the timer of a ``closing`` acts only while it is the latest, so a
    reload's timer cannot cut short the grace period of a later reload.
    Each ``open`` counts too: a timer armed before the popup connected
    again quits nothing when that connection is ``dropped``.
    """
    global _app_windows, _quit_armings
    with _app_lock:
        if event == "open":
            _app_windows += 1
            _quit_armings += 1
        elif event in ("closing", "dropped"):
            _app_windows -= 1
            if event == "closing" and _app_windows == 0:
                _quit_armings += 1
                if platform.system() == "Darwin":
                    timer = threading.Timer(
                        APP_WINDOW_QUIT_GRACE_SECS,
                        quit_closed_app,
                        args=(macos_bundle_path(), _quit_armings),
                    )
                    # A daemon's exit must not wait out the grace period.
                    timer.daemon = True
                    timer.start()
        else:
            return False
    return True


def windowless_app_pid(bundle: Path) -> int | None:
    """Return the pid of the browser from *bundle* that has neither popup nor window, or None.

    None when the popup is connected, when no browser runs from the
    bundle, when the browser has a window on screen (a page the popup
    moved to; the popup itself, between connections) or when the
    window server cannot be asked.
    """
    if _app_windows > 0:
        return None
    pid = macos_bundle_pid(bundle)
    if pid is None or macos_windows_on_screen(pid) != 0:
        return None
    if _app_windows > 0:
        # The popup connected while the processes and windows were listed.
        return None
    return pid


def quit_closed_app(bundle: Path, armed: int | None = None) -> bool:
    """Quit the browser running from *bundle* (the brand bundle) if its app window closed.

    Nothing happens, and False is returned, unless the browser has
    neither popup nor window (:func:`windowless_app_pid`).  Otherwise
    the browser gets SIGTERM, which Chromium answers with a regular
    quit, and True is returned (False when it exited meanwhile).
    *armed* is the arming of the quit timer calling (see
    :func:`app_window_event`): an older one than the latest quits
    nothing.
    """
    pid = windowless_app_pid(bundle)
    return pid is not None and _quit_if_still_windowless(pid, armed)


def _quit_if_still_windowless(pid: int, armed: int | None) -> bool:
    """SIGTERM *pid*, found without popup and window, unless the popup connected since.

    Decided under :data:`_app_lock`, the one :func:`app_window_event`
    counts the popup's connections under: a popup that connected
    between the (slow) look at the processes and windows and this
    signal must not lose its browser.  With *armed*,
    :data:`_quit_armings` as the quit timer was armed or as the sweep
    began to look, a ``closing`` or ``open`` since then leaves nothing
    to quit either (:func:`app_window_event`).
    """
    with _app_lock:
        if _app_windows > 0 or (armed is not None and armed != _quit_armings):
            return False
        return _terminate(pid)


def _terminate(pid: int) -> bool:
    """Send *pid* SIGTERM; False when it exited since it was listed (no error on a Timer thread)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def sweep_windowless_app(bundle: Path) -> bool:
    """Quit the browser from *bundle* found without popup and window by several sweeps in a row.

    The daemon's watchdog calls this every few seconds on macOS.  It
    catches the browser whose popup closed while the daemon was down
    or restarting (nobody armed :func:`quit_closed_app` then), which a
    click on the icon would otherwise answer with an empty browser
    window.  A browser is quit only after :data:`APP_WINDOW_SWEEP_STRIKES`
    sweeps of the same pid found it so, which spans the longest the
    popup takes to connect again after a daemon restart: a minimized
    popup between connections is off screen like a closed one.  The
    sweeps count from the popup's last connection
    (:data:`_quit_armings`): one that connected between two sweeps, or
    while one looked, is there, and has the whole span again once its
    connection drops.  True when the browser was quit.
    """
    global _windowless
    armed = _quit_armings  # before the (slow) look: a popup connecting from here on voids it
    pid = windowless_app_pid(bundle)
    if pid is None:
        _windowless = (None, 0, 0)
        return False
    strikes = _windowless[2] + 1 if _windowless[:2] == (pid, armed) else 1
    if strikes < APP_WINDOW_SWEEP_STRIKES:
        _windowless = (pid, armed, strikes)
        return False
    _windowless = (None, 0, 0)
    return _quit_if_still_windowless(pid, armed)


def ask_app_window_open(url: str, timeout: float = 2.0) -> bool | None:
    """Ask the daemon at *url* whether the app window is open.

    None when it does not answer, or answers anything but the state
    (no HTTP, a body cut short as the daemon goes down).
    """
    try:
        api = urllib.parse.urljoin(url, APP_WINDOW_API)
        with _DIRECT_OPENER.open(api, timeout=timeout) as answer:
            state = json.loads(answer.read())
        return bool(state["open"])
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException):
        return None


APP_WINDOW_SETTLE_SECS = 12.0
"""How long kiss-app gives a popup the daemon reports as not connected to connect."""


def app_window_closed(url: str, bundle: Path) -> bool:
    """Whether the browser running from *bundle* has no popup window, so a reopen would open none.

    The daemon at *url* knows (:func:`ask_app_window_open`): the popup
    is open while it is connected.  Not connected is most often the
    popup between connections (the daemon just restarted, and the
    popup connects again within seconds, minimized or not; the app is
    still starting and the popup not open yet), not a closed one, so
    the daemon is asked again for up to :data:`APP_WINDOW_SETTLE_SECS`
    seconds.  Still not connected afterwards, or the daemon not
    answering at all, a window on screen is taken for the popup, and
    none for it closed (a minimized one, which the window server
    cannot tell from a closed one, is connected by then whenever the
    daemon runs).
    """
    deadline = time.monotonic() + APP_WINDOW_SETTLE_SECS
    while True:
        open_ = ask_app_window_open(url)
        if open_:
            return False
        pid = macos_bundle_pid(bundle)
        if pid is None:
            return True
        if open_ is None or time.monotonic() >= deadline:
            windows = macos_windows_on_screen(pid)
            return windows == 0 or (windows is None and open_ is False)
        time.sleep(0.5)


def clone_macos_browser(contents: Path, browser: Path, identifier: str) -> Path | None:
    """Copy *browser* and its bundle's ``Frameworks`` under *contents*; return the copy.

    The copies are ``cp -R`` ones, on macOS with ``-c`` (APFS clones:
    instant, sharing their disk blocks with the originals until either
    side changes; a plain copy on a file system without clones), and
    the copied binary is re-signed ad hoc as *identifier* (see the
    module docstring).  None, with the reason on stderr and nothing
    left under *contents*, when a step fails: no ``Frameworks`` next to
    the browser, no ``codesign`` (part of macOS), a full disk, a step
    that does not finish within :data:`SETUP_TIMEOUT_SECS`.
    """
    browser_bundle = macos_browser_bundle(browser)
    if browser_bundle is None:
        return None
    copy = ["cp", "-R", "-c"] if sys.platform == "darwin" else ["cp", "-R"]
    exe_copy = contents / "MacOS" / browser.name
    steps = [
        [*copy, str(browser_bundle / "Contents" / "Frameworks"), str(contents / "Frameworks")],
        [*copy, str(browser), str(exe_copy)],
    ]
    if sys.platform == "darwin":
        # codesign refuses a file with extended attributes ("resource fork,
        # Finder information, or similar detritus not allowed"), and a Chrome
        # installed from a Safari download carries com.apple.FinderInfo.
        steps.append(["xattr", "-c", str(exe_copy)])
        steps.append(
            ["codesign", "--force", "--sign", "-", "--identifier", identifier, str(exe_copy)]
        )
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    for step in steps:
        try:
            result = run_process_group(step, SETUP_TIMEOUT_SECS)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result = subprocess.CompletedProcess(step, 1, "", str(exc))
        if result.returncode != 0:
            print(
                f"Could not clone {browser_bundle} into {contents.parent}: "
                f"{' '.join(step[:2])} failed: {result.stderr.strip()}",
                file=sys.stderr,
            )
            shutil.rmtree(contents / "Frameworks", ignore_errors=True)
            exe_copy.unlink(missing_ok=True)
            return None
    return exe_copy


def write_macos_launcher(
    apps_dir: Path,
    brand: dict[str, str],
    icon: Path,
    exe: Path,
    kiss_home: Path,
    browser: Path | None,
    in_use: bool | None = None,
) -> Path | None:
    """Write ``<apps_dir>/<Product Name>.app``; return the bundle's copy of *browser*.

    The bundle's executable is a shell script that ``exec``s
    ``kiss-app --foreground`` for the state directory *kiss_home*,
    which in turn ``exec``s the browser, so the process Finder launched
    is the app window itself and the app quits when the window closes.

    When *browser* is the main executable of an application bundle, the
    bundle is also a clone of that browser's (see the module docstring):
    ``Contents/MacOS/<browser name>`` is a copy of *browser*,
    ``Contents/Frameworks`` one of the browser's
    (:func:`clone_macos_browser`), and ``Info.plist`` is the browser's
    with the brand's identity (:func:`macos_bundle_plist`), so
    ``kiss-app`` can start the browser through the copy and the Dock
    shows the brand's icon for the window.  A clone that is still
    current (:func:`macos_bundle_browser`) is kept and only the
    launcher, icon and plist are rewritten; a stale one is replaced,
    unless a window still runs from it, in which case the bundle is
    left as it is until a later launch (deleting the copy would take
    the helper executables away from that window).  Without a browser
    bundle (no browser, a bare binary, a clone that failed) the bundle
    is the launcher alone and the Dock shows the browser's icon.
    *in_use* is whether a window runs from the bundle when the caller
    already listed the processes: True is trusted, anything else is
    found out again here.  A stale bundle is also left alone when that
    cannot be found out (``ps`` does not answer): deleting a bundle a
    window may run from is worse than one more launch under the
    browser's icon.  One writer at a time (a first launch next to an
    ``--install-launcher``, two launches): the checks and the rewrite
    they authorize are one step under a lock file next to the bundle,
    and the process listing that authorizes deleting a stale clone is
    taken under it, so a writer whose own listing predates the clone
    another just built, and the window that started from it, finds
    that clone current, or in use when it clones a different browser,
    instead of deleting it.  A launch holds the lock too, shared, while
    its window starts (:func:`launch_app`): none starts in between.

    Returns:
        The clone's main executable, kept or just written; None when
        the bundle is the launcher alone or a stale clone was left to
        a window still (or possibly) running from it.
    """
    bundle = apps_dir / f"{brand['product_name']}.app"
    contents = bundle / "Contents"
    identifier = macos_bundle_identifier(brand)
    browser_bundle = macos_browser_bundle(browser) if browser else None
    browser_info = None
    if browser_bundle:
        browser_info = read_plist(browser_bundle / "Contents" / "Info.plist")
    # The browser to clone: one in a bundle whose plist can be read.
    source = browser if browser_info is not None else None
    with exclusive_file_lock(apps_dir / f".{bundle.name}.lock", timeout=SETUP_TIMEOUT_SECS):
        clone = macos_bundle_browser(bundle, source) if source else None
        if clone is None:
            # Anything but a current clone is rebuilt from scratch, unless a
            # window still runs from the bundle, or might: an inconclusive
            # listing does not authorize deleting it, nor does the caller's
            # "nothing runs", taken before the lock.
            if in_use is not True:
                in_use = macos_bundle_in_use(bundle)
            if in_use is not False:
                print(
                    f"{bundle} may be in use; it is rewritten at the next launch.",
                    file=sys.stderr,
                )
                return None
            shutil.rmtree(bundle, ignore_errors=True)
            if source:
                clone = clone_macos_browser(contents, source, identifier)
        (contents / "MacOS").mkdir(parents=True, exist_ok=True)
        (contents / "Resources").mkdir(exist_ok=True)
        info = macos_bundle_plist(brand, browser_info if clone and browser_info else {})
        if not write_icns(icon, contents / "Resources" / "AppIcon.icns"):
            info.pop("CFBundleIconFile", None)
        # Both published whole (staged, then renamed into place): Launch
        # Services, Finder and :func:`read_plist` read the plist, and a
        # shell runs the launcher, at any moment, and the lock above keeps
        # out other writers only.
        atomic_write_text(
            contents / "Info.plist", plistlib.dumps(info).decode("utf-8"), create_mode=0o666
        )
        atomic_write_text(
            contents / "MacOS" / "launcher", macos_launcher_script(exe, kiss_home), mode=0o755
        )
        # Launch Services caches bundle metadata by modification time.
        bundle.touch()
    return clone


def install_launcher(
    brand: dict[str, str] = BRAND,
    icon: Path = ICON_FILE,
    exe: Path | None = None,
    home: Path | None = None,
    system: str = platform.system(),
    state_dir: Path | None = None,
    browser: Path | None = None,
) -> Path | None:
    """Register the app with the desktop; return the launcher's path.

    Linux: ``<applications_dir>/<slug>.desktop`` (then
    ``update-desktop-database`` when available, so menus refresh at
    once).  macOS: ``~/Applications/<Product Name>.app``.  None on any
    other platform, which has no launcher format here.  *exe* is the
    ``kiss-app`` command to launch, by default this very script;
    *state_dir* the ``KISS_HOME`` it runs for, by default this one;
    *browser* the engine the launcher is written for (its Wayland app
    id, its macOS bundle), by default :func:`find_browser`.
    """
    exe = exe or Path(sys.argv[0]).resolve()
    home = home or Path.home()
    state_dir = state_dir or kiss_home()
    browser = browser or find_browser(system)
    if system == "Linux":
        entry = linux_launcher_path(home, brand)
        write_linux_launcher(entry, brand, icon, exe, state_dir, browser or Path("chrome"))
        refresh = shutil.which("update-desktop-database")
        if refresh:
            try:
                run_process_group([refresh, str(entry.parent)], SETUP_TIMEOUT_SECS)
            except (OSError, subprocess.TimeoutExpired):  # it cannot start, or never returns
                pass  # the refresh is optional: the menus pick the entry up later
        return entry
    if system == "Darwin":
        bundle = macos_bundle_path(home, brand)
        write_macos_launcher(bundle.parent, brand, icon, exe, state_dir, browser)
        return bundle
    return None


def launcher_path(
    brand: dict[str, str] = BRAND,
    home: Path | None = None,
    system: str = platform.system(),
) -> Path | None:
    """Return where :func:`install_launcher` puts the launcher on *system*, written or not.

    None on a platform that has no launcher format here.
    """
    home = home or Path.home()
    if system == "Linux":
        return linux_launcher_path(home, brand)
    if system == "Darwin":
        return macos_bundle_path(home, brand)
    return None


def macos_bundle_path(home: Path | None = None, brand: dict[str, str] = BRAND) -> Path:
    """Return ``<home>/Applications/<Product Name>.app``, the brand bundle (written or not)."""
    return (home or Path.home()) / "Applications" / f"{brand['product_name']}.app"


def uninstall_launcher(
    brand: dict[str, str] = BRAND,
    home: Path | None = None,
    system: str = platform.system(),
) -> Path | None:
    """Remove the launcher :func:`install_launcher` wrote; return its path if it existed.

    The macOS bundle goes as a writer replaces it
    (:func:`write_macos_launcher`): under the bundle's lock, so that no
    launch starts its clone meanwhile, and not from under a window.

    Raises:
        OSError: The bundle is in use, or may be (``ps`` does not
            answer); its lock is still held after
            :data:`SETUP_TIMEOUT_SECS`; the launcher cannot be deleted.
    """
    path = launcher_path(brand, home, system)
    if path is None or not path.exists():
        return None
    if path.is_dir():
        with exclusive_file_lock(path.parent / f".{path.name}.lock", timeout=SETUP_TIMEOUT_SECS):
            if not path.exists():  # another uninstall removed it while this one waited
                return None
            if macos_bundle_in_use(path) is not False:
                raise OSError(f"{path} may be in use; quit {brand['product_name']} first")
            shutil.rmtree(path)
    else:
        path.unlink()
    return path


def reveal_launcher(
    brand: dict[str, str] = BRAND,
    home: Path | None = None,
    system: str = platform.system(),
) -> Path | None:
    """Show the installed launcher in the desktop's file manager; return its path.

    macOS: Finder opens ``~/Applications`` with the app bundle selected
    (``open -R``).  Linux: ``xdg-open`` opens the applications directory
    the ``.desktop`` entry sits in.  The file manager is started
    detached, like the browser, so it outlives the caller.  None, and
    nothing shown, when there is no launcher to show (a platform
    without one, one not installed yet), no file manager opener, or
    one that cannot be executed or failed at once (no desktop to open
    a window on).
    """
    path = launcher_path(brand, home, system)
    if path is None or not path.exists():
        return None
    if system == "Darwin":
        command = ["open", "-R", str(path)]
    else:
        opener = shutil.which("xdg-open")
        if opener is None:
            return None
        command = [opener, str(path.parent)]
    try:
        return path if start_detached(command, subprocess.DEVNULL) else None
    except OSError:  # the opener cannot be executed
        return None


def macos_launch_browser(
    browser: Path, home: Path | None = None, in_use: bool | None = None
) -> Path:
    """Return the path to start *browser* by on macOS: the brand bundle's copy of it.

    The window then runs under the brand bundle's identity and icon.
    A missing or stale bundle (first run, an updated browser, a
    replaced Playwright Chromium) is rewritten first; when no clone can
    be made (a bare binary, an ``~/Applications`` that cannot be
    written, a stale clone a window still runs from) *browser* itself
    is returned: the window still opens, under the browser's icon (or,
    for a running profile, in the running window).  *in_use* is
    whether a window runs from the bundle when the caller already
    listed the processes (None: the rewrite finds out).
    """
    bundle = macos_bundle_path(home)
    clone = macos_bundle_browser(bundle, browser)
    if clone is None and macos_browser_bundle(browser) is not None:
        try:
            clone = write_macos_launcher(
                bundle.parent,
                BRAND,
                ICON_FILE,
                Path(sys.argv[0]).resolve(),
                kiss_home(),
                browser,
                in_use,
            )
        except OSError as exc:
            print(f"Could not write {bundle}: {exc}", file=sys.stderr)
            return browser
    return clone or browser


def _lock_shared(path: Path) -> int | None:
    """Open the lock file *path* read-only, locked shared until closed; None: no lock to be had.

    Launches hold the lock of a bundle's writers together, never with a
    writer, which locks the file (created when missing) exclusively.
    None when the file cannot be opened or the system refuses to lock
    it: the launch then goes ahead unlocked, as every launch once did.
    TimeoutError when a writer holds it past :data:`SETUP_TIMEOUT_SECS`.
    """
    import fcntl  # POSIX only, like the macOS launch this is for

    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CREAT, 0o600)
    except OSError:
        # No writer can open the file then either; one that already had it open is not kept out.
        return None
    deadline = time.monotonic() + SETUP_TIMEOUT_SECS
    locked = False
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                locked = True
                return descriptor
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
            except OSError:  # refused, not contended (a file system without locks)
                return None
    finally:  # also when the wait is interrupted
        if not locked:
            os.close(descriptor)
    raise TimeoutError(f"{path} still locked by a writer after {SETUP_TIMEOUT_SECS:g}s")


def launch_app(
    command: list[str], log: Path, browser: Path, *, foreground: bool, home: Path | None = None
) -> bool | None:
    """:func:`launch` *command*, the choice of :func:`macos_launch_browser` for *browser*.

    Returns what :func:`launch` returns; None, the reason on stderr,
    when nothing was started.  A file of the brand bundle (the clone,
    by its path, also when ``$KISS_APP_BROWSER`` names it) is started
    under the bundle's lock, held shared, so that no writer deletes it
    under the new window (:func:`write_macos_launcher`), and only while
    it still is the choice for *browser*.  One that is not (a writer
    replaced the clone meanwhile, by another browser's at the same path
    too; the browser was updated) is chosen again, no lock held, three
    attempts in all; *browser* itself is not started in its place, nor
    anything while a writer holds the lock at the deadline.  ``exec``
    in the foreground releases the lock: its descriptor is not inherited.
    """
    bundle = macos_bundle_path(home)
    if not Path(command[0]).resolve().is_relative_to(bundle.resolve()):
        return launch(command, log, foreground=foreground)
    reason = f"no longer a copy of {browser}"
    for attempt in range(3):
        if attempt:  # no longer the choice: choose again, as ``main()`` chose it
            chosen = macos_launch_browser(browser, home)
            if not chosen.resolve().is_relative_to(bundle.resolve()):
                break  # the browser itself: not started in the clone's place
            command = [str(chosen), *command[1:]]
        try:
            lock = _lock_shared(bundle.parent / f".{bundle.name}.lock")
        except TimeoutError as exc:
            reason = str(exc)
            break
        try:
            # Still a file, and the one chosen for *browser*: not another browser's clone there.
            if Path(command[0]).is_file() and command[0] == str(
                macos_bundle_browser(bundle, browser) or browser
            ):
                return launch(command, log, foreground=foreground)
        finally:
            if lock is not None:
                os.close(lock)
    print(f"Could not start {command[0]}: {reason}; try again.", file=sys.stderr)
    return None


def macos_activate_running(bundle: Path, timeout: float = PROBE_TIMEOUT_SECS) -> bool:
    """Bring the window of the app running from *bundle* to the front; whether there was one.

    ``open -a`` on a bundle that is running does what a click on its
    icon does: Launch Services sends the running app a reopen event,
    which focuses the window (see the module doc), so nothing else is
    started.  The caller has found the app running: ``open -a`` starts
    one that is not.  False when ``open`` failed or did not return
    within *timeout* seconds.
    """
    result = _probe(["open", "-a", str(bundle)], timeout)
    return result is not None and result.returncode == 0


def main(argv: list[str] | None = None) -> int:
    """Run the ``kiss-app`` command line; return the exit status."""
    parser = argparse.ArgumentParser(
        prog="kiss-app",
        description=f"Open the {BRAND['product_name']} webapp in its own window.",
    )
    parser.add_argument("--url", help="webapp URL (default: the running daemon's loopback URL)")
    parser.add_argument(
        "--wait",
        type=float,
        default=120.0,
        metavar="SECS",
        help="how long to wait for the daemon's URL file (default 120)",
    )
    parser.add_argument("--foreground", action="store_true", help="run the window in this process")
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="print the browser command instead of running it",
    )
    parser.add_argument(
        "--install-launcher", action="store_true", help="register the desktop launcher and exit"
    )
    parser.add_argument(
        "--uninstall-launcher", action="store_true", help="remove the desktop launcher and exit"
    )
    parser.add_argument(
        "--reveal-launcher",
        action="store_true",
        help="show the desktop launcher's folder in the file manager and exit",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.wait):  # NaN, infinity: a deadline that never comes
        parser.error("argument --wait: SECS must be a finite number")

    if args.install_launcher:
        try:
            path = install_launcher()
        except OSError as exc:  # an unwritable folder, or the macOS bundle's lock stays held
            print(f"Could not install the launcher: {exc}", file=sys.stderr)
            return 1
        print(
            f"Installed {path}"
            if path
            else f"No launcher installed: {platform.system()} has no desktop launcher here."
        )
        return 0
    if args.uninstall_launcher:
        try:
            path = uninstall_launcher()
        except OSError as exc:  # the macOS bundle is in use, or its lock stays held
            print(f"Could not remove the launcher: {exc}", file=sys.stderr)
            return 1
        print(f"Removed {path}" if path else "No launcher to remove.")
        return 0
    if args.reveal_launcher:
        path = reveal_launcher()
        if path is None:
            print(
                "No launcher to show: none installed, or no file manager opened.",
                file=sys.stderr,
            )
            return 1
        print(f"Showing {path.name} in {path.parent}")
        return 0

    url_file = remote_url_file()
    # --print-command only shows the command: it does not wait for the URL file.
    if args.print_command:
        url = args.url or read_loopback_url(url_file)
    else:
        url = wait_for_url(url_file, args.url, args.wait)
    if not url:
        print(
            f"{BRAND['product_name']} is not running: no URL in {url_file} after {args.wait:g}s.",
            file=sys.stderr,
        )
        return 1
    browser = find_browser()
    if browser is None:
        print(
            "No Chromium-family browser found (Google Chrome, Chromium, Microsoft Edge, Brave or "
            "Playwright's Chromium); set KISS_APP_BROWSER to one, or open the URL in a browser: "
            f"{url}",
            file=sys.stderr,
        )
        return 1
    sandbox = sandboxed(browser)
    macos = platform.system() == "Darwin"
    # One process listing serves both the bundle rewrite and the
    # already-open check below (None, ``ps`` did not answer: the
    # rewrite asks again and the check treats it as not running).
    running = macos_bundle_in_use(macos_bundle_path()) if macos else False
    daemon_url = url
    engine = browser
    if macos:
        browser = macos_launch_browser(browser, in_use=running)
        url = app_window_url(url)
    # A daemon not answering yet (the app relaunched at login before
    # the daemon's LaunchAgent spawned it) gets the packaged waiting
    # page, which moves on to the app when the daemon answers; the
    # printed command names the app itself.
    if not args.print_command and not daemon_answers(daemon_url):
        url = start_page_url(url, daemon_url, popup=macos)
    command = app_command(
        browser, url, kiss_home() / PROFILE_DIR_NAME, brand_slug(), sandbox=sandbox, popups=macos
    )
    if args.print_command:
        # Quoted for the platform's shell: a browser or profile path with
        # a space ("Google Chrome.app") must paste as one argument.
        print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command))
        return 0
    # A running browser without its popup window gets the command line
    # instead of a reopen, which would open an empty browser window
    # (see the module doc).
    if (
        running
        and not app_window_closed(daemon_url, macos_bundle_path())
        and macos_activate_running(macos_bundle_path())
    ):
        print(f"{BRAND['product_name']} is already open; its window was brought to the front.")
        return 0
    log = kiss_home() / "kiss-app.log"
    try:
        # macOS starts the clone under its bundle's lock; None: nothing started, the reason printed.
        started = (
            launch_app(command, log, engine, foreground=args.foreground)
            if macos
            else launch(command, log, foreground=args.foreground)
        )
        if started:
            return 0
        if started is None:
            return 1
        last_words = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
    except OSError as exc:  # the browser cannot be executed, or its log not written or read
        print(f"Could not start {browser}: {exc}", file=sys.stderr)
        return 1
    print(
        f"The browser exited at once ({browser}); its stderr is in {log}:",
        *(f"    {line}" for line in last_words),
        sep="\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - console script entry
    sys.exit(main())

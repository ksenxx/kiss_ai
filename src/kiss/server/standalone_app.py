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
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import IO
from xml.parsers.expat import ExpatError

from kiss.core.brand import BRAND
from kiss.core.config import kiss_home

MEDIA_DIR = Path(__file__).resolve().parent.parent / "webapp" / "media"
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


def read_loopback_url(url_file: Path) -> str | None:
    """Return the daemon's plain-HTTP loopback URL from *url_file*, or None.

    ``remote-url.json`` carries ``loopback`` (``http://127.0.0.1:PORT``);
    a file written by an older daemon only has ``local``
    (``http://localhost:PORT``), whose host is rewritten to 127.0.0.1.
    None when the file is missing, unreadable or has neither URL.
    """
    try:
        data = json.loads(url_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    loopback = data.get("loopback")
    if isinstance(loopback, str) and loopback:
        return loopback
    local = data.get("local")
    if isinstance(local, str) and local:
        return local.replace("localhost", "127.0.0.1", 1)
    return None


def wait_for_loopback_url(url_file: Path, timeout: float) -> str | None:
    """Poll *url_file* for up to *timeout* seconds; return the loopback URL or None."""
    deadline = time.monotonic() + timeout
    while True:
        url = read_loopback_url(url_file)
        if url or time.monotonic() >= deadline:
            return url
        time.sleep(0.5)


@functools.cache
def playwright_chromium() -> Path | None:
    """Return the Chromium executable Playwright installed, or None.

    Asks Playwright itself (``BrowserType.executable_path``), which
    resolves the browsers directory and revision the installed
    ``playwright`` package expects; nothing is launched.  None when the
    package or the browser is missing.  Cached: the lookup starts the
    Playwright node driver and waits for its handshake, and both
    :func:`find_browser` and :func:`sandboxed` need the answer in one
    ``kiss-app`` run (conc1018 H1-F2).
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            path = Path(pw.chromium.executable_path)
    except Exception:  # noqa: BLE001 - any failure means "no bundled browser"
        return None
    return path if path.is_file() else None


def find_browser(system: str = platform.system()) -> Path | None:
    """Return the Chromium-family browser to run the app in, or None.

    ``$KISS_APP_BROWSER`` (a path or a command name) wins when it
    resolves to an executable; then the system browsers of *system*
    (``Linux``, ``Darwin`` or ``Windows``) in the order of
    ``_LINUX_BROWSERS`` / ``_MACOS_BROWSERS`` / ``_WINDOWS_BROWSERS``;
    last the Playwright Chromium.
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
    foreground this call does not return.
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


def write_linux_launcher(
    apps_dir: Path, brand: dict[str, str], icon: Path, exe: Path, kiss_home: Path, browser: Path
) -> Path:
    """Write ``<apps_dir>/<slug>.desktop`` and return its path."""
    apps_dir.mkdir(parents=True, exist_ok=True)
    entry = apps_dir / f"{brand_slug(brand)}.desktop"
    entry.write_text(desktop_entry(brand, icon, exe, kiss_home, browser), encoding="utf-8")
    entry.chmod(0o755)
    return entry


def write_icns(png: Path, icns: Path) -> bool:
    """Convert *png* to the macOS icon file *icns*; True on success.

    Uses the ``sips`` and ``iconutil`` tools every macOS ships.  False
    (and no file) when either is missing or fails, in which case the
    bundle is written without an icon.
    """
    sips, iconutil = shutil.which("sips"), shutil.which("iconutil")
    if not sips or not iconutil:
        return False
    iconset = icns.with_suffix(".iconset")
    shutil.rmtree(iconset, ignore_errors=True)
    iconset.mkdir(parents=True)
    try:
        for size in (16, 32, 64, 128, 256, 512):
            for scale in (1, 2):
                pixels = size * scale
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(
                    [sips, "-z", str(pixels), str(pixels), str(png), "--out", str(iconset / name)],
                    check=True,
                    capture_output=True,
                )
        subprocess.run(
            [iconutil, "-c", "icns", str(iconset), "-o", str(icns)], check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
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
    """
    info = {
        key: value
        for key, value in browser_plist.items()
        if key not in _MACOS_PLIST_DROPPED_KEYS and not key.startswith("KS")
    }
    info.update(
        {
            "CFBundleName": brand["product_name"],
            "CFBundleDisplayName": brand["product_name"],
            "CFBundleIdentifier": f"org.kiss-sorcar.{brand_slug(brand)}",
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "launcher",
            "CFBundleIconFile": "AppIcon",
            "LSUIElement": False,
        }
    )
    return info


def read_plist(path: Path) -> dict[str, object] | None:
    """Return the property list at *path*; None when it is missing, not a plist or cut short."""
    try:
        with path.open("rb") as f:
            info: dict[str, object] = plistlib.load(f)
    except (OSError, plistlib.InvalidFileException, ExpatError):
        return None
    return info


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


def macos_bundle_in_use(bundle: Path) -> bool:
    """Whether a process runs from *bundle*: the browser or one of its helpers.

    Every executable of the clone sits under ``Contents/``, so a running
    one names that directory in its command line.
    """
    listing = subprocess.run(["ps", "-axo", "command="], capture_output=True, text=True)
    return f"{bundle / 'Contents'}/" in listing.stdout


def clone_macos_browser(contents: Path, browser: Path, identifier: str) -> Path | None:
    """Copy *browser* and its bundle's ``Frameworks`` under *contents*; return the copy.

    The copies are ``cp -R`` ones, on macOS with ``-c`` (APFS clones:
    instant, sharing their disk blocks with the originals until either
    side changes; a plain copy on a file system without clones), and
    the copied binary is re-signed ad hoc as *identifier* (see the
    module docstring).  None, with the reason on stderr and nothing
    left under *contents*, when a step fails: no ``Frameworks`` next to
    the browser, no ``codesign`` (part of macOS), a full disk.
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
        steps.append(
            ["codesign", "--force", "--sign", "-", "--identifier", identifier, str(exe_copy)]
        )
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    for step in steps:
        try:
            result = subprocess.run(step, capture_output=True, text=True)
        except OSError as exc:
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
) -> Path:
    """Write ``<apps_dir>/<Product Name>.app`` and return its path.

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
    already listed the processes (None: find out here).
    """
    bundle = apps_dir / f"{brand['product_name']}.app"
    contents = bundle / "Contents"
    identifier = f"org.kiss-sorcar.{brand_slug(brand)}"
    browser_bundle = macos_browser_bundle(browser) if browser else None
    browser_info = None
    if browser_bundle:
        browser_info = read_plist(browser_bundle / "Contents" / "Info.plist")
    # The browser to clone: one in a bundle whose plist can be read.
    source = browser if browser_info is not None else None
    clone = macos_bundle_browser(bundle, source) if source else None
    if clone is None:
        # Anything but a current clone is rebuilt from scratch, unless a
        # window still runs from the bundle.
        if macos_bundle_in_use(bundle) if in_use is None else in_use:
            print(f"{bundle} is in use; it is rewritten at the next launch.", file=sys.stderr)
            return bundle
        shutil.rmtree(bundle, ignore_errors=True)
        if source:
            clone = clone_macos_browser(contents, source, identifier)
    (contents / "MacOS").mkdir(parents=True, exist_ok=True)
    (contents / "Resources").mkdir(exist_ok=True)
    info: dict[str, object]
    if clone and browser_info is not None:
        info = macos_bundle_plist(brand, browser_info)
    else:
        info = {
            "CFBundleName": brand["product_name"],
            "CFBundleDisplayName": brand["product_name"],
            "CFBundleIdentifier": identifier,
            "CFBundleVersion": "1.0",
            "CFBundlePackageType": "APPL",
            "CFBundleExecutable": "launcher",
            "CFBundleIconFile": "AppIcon",
            "LSUIElement": False,
        }
    if not write_icns(icon, contents / "Resources" / "AppIcon.icns"):
        info.pop("CFBundleIconFile", None)
    with (contents / "Info.plist").open("wb") as f:
        plistlib.dump(info, f)
    launcher = contents / "MacOS" / "launcher"
    launcher.write_text(macos_launcher_script(exe, kiss_home), encoding="utf-8")
    launcher.chmod(0o755)
    # Launch Services caches bundle metadata by modification time.
    bundle.touch()
    return bundle


def install_launcher(
    brand: dict[str, str] = BRAND,
    icon: Path = ICON_FILE,
    exe: Path | None = None,
    home: Path | None = None,
    system: str = platform.system(),
    state_dir: Path | None = None,
    browser: Path | None = None,
    in_use: bool | None = None,
) -> Path | None:
    """Register the app with the desktop; return the launcher's path.

    Linux: ``<applications_dir>/<slug>.desktop`` (then
    ``update-desktop-database`` when available, so menus refresh at
    once).  macOS: ``~/Applications/<Product Name>.app``.  None on any
    other platform, which has no launcher format here.  *exe* is the
    ``kiss-app`` command to launch, by default this very script;
    *state_dir* the ``KISS_HOME`` it runs for, by default this one;
    *browser* the engine the launcher is written for (its Wayland app
    id, its macOS bundle), by default :func:`find_browser`; *in_use*
    whether a window runs from the macOS bundle, when known (see
    :func:`write_macos_launcher`).
    """
    exe = exe or Path(sys.argv[0]).resolve()
    home = home or Path.home()
    state_dir = state_dir or kiss_home()
    browser = browser or find_browser(system)
    if system == "Linux":
        apps_dir = applications_dir(home)
        entry = write_linux_launcher(
            apps_dir, brand, icon, exe, state_dir, browser or Path("chrome")
        )
        refresh = shutil.which("update-desktop-database")
        if refresh:
            subprocess.run([refresh, str(apps_dir)], check=False, capture_output=True)
        return entry
    if system == "Darwin":
        return write_macos_launcher(
            home / "Applications", brand, icon, exe, state_dir, browser, in_use
        )
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
        return applications_dir(home) / f"{brand_slug(brand)}.desktop"
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
    """Remove the launcher :func:`install_launcher` wrote; return its path if it existed."""
    path = launcher_path(brand, home, system)
    if path is None or not path.exists():
        return None
    if path.is_dir():
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
    one that failed at once (no desktop to open a window on).
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
    return path if start_detached(command, subprocess.DEVNULL) else None


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
            install_launcher(home=home, system="Darwin", browser=browser, in_use=in_use)
        except OSError as exc:
            print(f"Could not write {bundle}: {exc}", file=sys.stderr)
            return browser
        clone = macos_bundle_browser(bundle, browser)
    return clone or browser


def macos_activate_running(bundle: Path, in_use: bool | None = None) -> bool:
    """Bring the window of the app running from *bundle* to the front; whether there was one.

    ``open -a`` on a bundle that is running does what a click on its
    icon does: Launch Services sends the running app a reopen event,
    which focuses the window (see the module doc), so nothing else is
    started.  False when nothing runs from *bundle* (*in_use* when the
    caller already listed the processes, else
    :func:`macos_bundle_in_use`) or ``open`` failed.
    """
    if not (macos_bundle_in_use(bundle) if in_use is None else in_use):
        return False
    return subprocess.run(["open", "-a", str(bundle)], capture_output=True).returncode == 0


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

    if args.install_launcher:
        path = install_launcher()
        print(
            f"Installed {path}"
            if path
            else f"No launcher installed: {platform.system()} has no desktop launcher here."
        )
        return 0
    if args.uninstall_launcher:
        path = uninstall_launcher()
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

    url = args.url or wait_for_loopback_url(kiss_home() / "remote-url.json", args.wait)
    if not url:
        print(
            f"{BRAND['product_name']} is not running: no URL in {kiss_home() / 'remote-url.json'}"
            f" after {args.wait:g}s.",
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
    # already-open check below.
    running = macos and macos_bundle_in_use(macos_bundle_path())
    if macos:
        browser = macos_launch_browser(browser, in_use=running)
        url = app_window_url(url)
    command = app_command(
        browser, url, kiss_home() / PROFILE_DIR_NAME, brand_slug(), sandbox=sandbox, popups=macos
    )
    if args.print_command:
        print(" ".join(command))
        return 0
    if running and macos_activate_running(macos_bundle_path(), in_use=True):
        print(f"{BRAND['product_name']} is already open; its window was brought to the front.")
        return 0
    log = kiss_home() / "kiss-app.log"
    if launch(command, log, foreground=args.foreground):
        return 0
    last_words = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-5:]
    print(
        f"The browser exited at once ({browser}); its stderr is in {log}:",
        *(f"    {line}" for line in last_words),
        sep="\n",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":  # pragma: no cover - console script entry
    sys.exit(main())

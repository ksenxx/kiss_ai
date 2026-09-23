# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Install the local CA (:mod:`kiss.server.tls_certs`) into this user's trust stores.

``kiss-web --trust-ca`` calls :func:`trust_local_ca`.  Browsers keep their
own trust databases, so the CA is added to every one that exists for the
current user:

* NSS databases (Chromium and Firefox on Linux, Firefox on macOS) through
  ``certutil`` from libnss3-tools / nss-tools / Homebrew ``nss``.  Chromium
  reads ``~/.local/share/pki/nssdb`` (since M146) or the older
  ``~/.pki/nssdb``; Firefox keeps one database per profile.
* The macOS login keychain through ``security add-trusted-cert`` (Safari
  and Chrome); macOS asks for the login password in a dialog.
* The Windows current-user Root store through ``certutil -addstore -user``
  (Edge, Chrome and Firefox); Windows shows a confirmation dialog.

Nothing here needs root: the Linux system store is only useful for
non-browser clients (curl, python) and the command for it is printed
rather than run.  Only the CA *certificate* is installed; the CA key never
leaves ``~/.kiss/tls/``.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from kiss.server.tls_certs import ca_common_name, ca_fingerprint

_CMD_TIMEOUT_S = 120.0


def ca_nickname(ca_cert_path: Path) -> str:
    """Return the trust-store nickname for the CA (unique per CA key).

    The fingerprint suffix keeps a regenerated CA from colliding with an
    earlier one still present in a database.
    """
    return f"{ca_common_name(ca_cert_path)} {ca_fingerprint(ca_cert_path)[:23]}"


def _run(argv: list[str]) -> tuple[int, str]:
    """Run *argv*, returning ``(returncode, combined output)``; never raises."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=_CMD_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def nss_databases(home: Path) -> list[str]:
    """Return ``sql:``/``dbm:`` NSS database specs that exist under *home*.

    Args:
        home: The user's home directory (parameterised for tests).
    """
    candidates = [
        home / ".pki" / "nssdb",
        home / ".local" / "share" / "pki" / "nssdb",
        home / "snap" / "chromium" / "current" / ".pki" / "nssdb",
    ]
    for pattern in (
        home / ".mozilla" / "firefox" / "*",
        home / "snap" / "firefox" / "common" / ".mozilla" / "firefox" / "*",
        home / "Library" / "Application Support" / "Firefox" / "Profiles" / "*",
    ):
        candidates.extend(Path(p) for p in sorted(glob.glob(str(pattern))))
    specs: list[str] = []
    for directory in candidates:
        if (directory / "cert9.db").is_file():
            specs.append(f"sql:{directory}")
        elif (directory / "cert8.db").is_file():
            specs.append(f"dbm:{directory}")
    return specs


def _find_certutil() -> str | None:
    """Return the NSS ``certutil`` binary, looking in Homebrew's ``nss`` too."""
    found = shutil.which("certutil")
    if found:
        return found
    for prefix in ("/opt/homebrew/opt/nss", "/usr/local/opt/nss"):
        candidate = Path(prefix) / "bin" / "certutil"
        if candidate.is_file():
            return str(candidate)
    return None


def install_into_nss(ca_cert_path: Path, home: Path) -> list[str]:
    """Add the CA to every NSS database under *home*; return result lines."""
    databases = nss_databases(home)
    if not databases:
        return ["No Chromium/Firefox NSS certificate database found under "
                f"{home} (start the browser once, then rerun)."]
    certutil = _find_certutil()
    if certutil is None:
        hint = (
            "brew install nss" if sys.platform == "darwin"
            else "sudo apt install libnss3-tools  (or: sudo dnf install nss-tools)"
        )
        return [f"certutil not found; install it with `{hint}` and rerun to trust "
                f"the CA in: {', '.join(databases)}"]
    nick = ca_nickname(ca_cert_path)
    lines: list[str] = []
    for spec in databases:
        code, out = _run([
            certutil, "-A", "-d", spec, "-t", "C,,", "-n", nick, "-i", str(ca_cert_path),
        ])
        if code == 0:
            lines.append(f"Trusted in NSS database {spec}")
        else:
            lines.append(f"FAILED for NSS database {spec}: {out}")
    return lines


def install_into_macos_keychain(ca_cert_path: Path, home: Path) -> list[str]:
    """Add the CA to the user's login keychain as a trusted root."""
    keychain = home / "Library" / "Keychains" / "login.keychain-db"
    code, out = _run([
        "security", "add-trusted-cert", "-r", "trustRoot", "-k", str(keychain),
        str(ca_cert_path),
    ])
    if code == 0:
        return [f"Trusted in macOS login keychain {keychain} (Safari, Chrome)"]
    return [f"FAILED for macOS login keychain: {out}"]


def install_into_windows_store(ca_cert_path: Path) -> list[str]:
    """Add the CA to the current user's Trusted Root store on Windows."""
    code, out = _run(["certutil", "-addstore", "-user", "Root", str(ca_cert_path)])
    if code == 0:
        return ["Trusted in the Windows current-user Root store (Edge, Chrome, Firefox)"]
    return [f"FAILED for the Windows Root store: {out}"]


def linux_system_store_hint(ca_cert_path: Path) -> list[str]:
    """Return the (root-only) commands that trust the CA system-wide on Linux."""
    return [
        "Browsers on Linux do not read the system store; for curl/python run one of:",
        f"  sudo cp {ca_cert_path} /usr/local/share/ca-certificates/kiss-sorcar-local-ca.crt"
        " && sudo update-ca-certificates",
        f"  sudo cp {ca_cert_path} /etc/pki/ca-trust/source/anchors/kiss-sorcar-local-ca.pem"
        " && sudo update-ca-trust extract",
    ]


def trust_local_ca(
    ca_cert_path: Path, home: Path | None = None, platform: str | None = None,
) -> list[str]:
    """Install the CA certificate into every trust store of the current user.

    Args:
        ca_cert_path: The PEM CA certificate (``~/.kiss/tls/ca.pem``).
        home: Home directory to scan (defaults to the current user's).
        platform: ``sys.platform`` override for tests.

    Returns:
        Human-readable result lines, one per store, ending with the
        phone/tablet instructions.
    """
    home = home if home is not None else Path(os.path.expanduser("~"))
    platform = platform or sys.platform
    lines = [
        f"CA certificate: {ca_cert_path}",
        f"SHA-256 fingerprint: {ca_fingerprint(ca_cert_path)}",
    ]
    if platform == "win32":
        lines += install_into_windows_store(ca_cert_path)
    elif platform == "darwin":
        lines += install_into_macos_keychain(ca_cert_path, home)
        lines += install_into_nss(ca_cert_path, home)
    else:
        lines += install_into_nss(ca_cert_path, home)
        lines += linux_system_store_hint(ca_cert_path)
    lines += [
        "Restart the browser for the change to take effect.",
        "Phones/tablets: open https://<lan-ip>:PORT/ca.crt from the KISS webapp's "
        "LAN URL, install the downloaded certificate, then enable trust for it "
        "(iOS: Settings > General > About > Certificate Trust Settings; Android: "
        "Settings > Security > Encryption & credentials > Install a certificate > "
        "CA certificate).",
    ]
    return lines

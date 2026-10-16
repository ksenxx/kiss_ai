# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The web app's remote-access password: generation, storage, checking.

``install.sh`` runs :func:`ensure_password` on every run: the first
install draws a random password (:func:`generate_password`, 20 letters
and digits) and stores it in ``config.json`` under ``remote_password``;
every later install or update keeps the stored one, so the password a
user has typed into their devices stays valid.  A new one is drawn only
on request (``python -m kiss.core.remote_password --rotate``) or after
the key is cleared from ``config.json``.  The password is stored as
typed, not hashed: the daemon sends it to authenticated clients in
``configData`` so the Settings panel can show it read-only, and the
password is always random (about 119 bits of entropy), so the file's
0600 mode is what protects it, as it protects ``tunnel_token`` next to
it.  The daemon compares the password a client sends — in its ``auth``
frame and again in every later frame — with the stored one in constant
time (:func:`password_accepted`).  ``src/kiss/scripts/remote_config.py``
(shipped alone to deploy targets) stores the same key with the same
keep-unless-empty rule; keep the two in step.
"""

import hmac
import os
import secrets
import string
import sys

PASSWORD_LENGTH = 20
PASSWORD_ALPHABET = string.ascii_letters + string.digits


def generate_password() -> str:
    """Return a fresh random password of :data:`PASSWORD_LENGTH` letters and digits."""
    return "".join(secrets.choice(PASSWORD_ALPHABET) for _ in range(PASSWORD_LENGTH))


def password_accepted(password: str, stored: str) -> bool:
    """Return whether a client that sent *password* may be served.

    With a password configured, the client's must equal it (a
    constant-time compare).  With none, only the empty password is
    accepted: that keeps a browser on the daemon's own machine working
    before ``install.sh`` ever set a password (the daemon admits no
    other peer while the password is empty).
    """
    return hmac.compare_digest(password.encode("utf-8"), stored.encode("utf-8"))


def stored_password() -> str:
    """Return the password stored in ``config.json`` (empty when none).

    A ``config.json`` that exists but cannot be read raises
    ``PermissionError`` instead of counting as empty (``load_config``
    treats it so): a caller about to store a fresh password must not
    overwrite settings it could not read.
    """
    from kiss.core import user_config

    path = user_config.CONFIG_PATH
    if path.exists() and not os.access(path, os.R_OK):
        raise PermissionError(f"{path} exists but cannot be read")
    return str(user_config.load_config()["remote_password"])


def ensure_password(rotate: bool = False) -> str:
    """Return the remote-access password, storing a fresh one when needed.

    The password already in ``config.json`` is returned as is, so a
    re-run of ``install.sh`` (an update) never invalidates the password
    the user's devices know.  A fresh one is drawn, stored and returned
    when none is stored (the first install, or the key cleared by hand)
    or when *rotate* is true; the previous password then stops working
    at once because the daemon reads ``config.json`` on every check.
    The write also purges the hash-only key of an earlier release
    (``save_config`` drops ``RETIRED_KEYS``).
    """
    from kiss.core.user_config import save_config

    stored = stored_password()
    if stored and not rotate:
        return stored
    password = generate_password()
    save_config({"remote_password": password})
    return password


if __name__ == "__main__":
    # ``--show`` prints the stored password without ever storing one
    # (install.sh reads a password this way); ``--rotate`` stores a new one.
    flag = sys.argv[1] if len(sys.argv) > 1 else ""
    print(stored_password() if flag == "--show" else ensure_password(rotate=flag == "--rotate"))

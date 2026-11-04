# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Build a signed iOS Web Clip profile (``.mobileconfig``) that opens a URL full screen.

A Web Clip is Apple's configuration-profile payload
(``com.apple.webClip.managed``) that puts a named icon on the iPhone
Home Screen; with ``FullScreen`` the URL opens "as a web app without a
browser (there's no URL or search bar, and no bookmarks)" (Apple
Platform Deployment, Web Clips payload settings).  Unlike Safari's
"Add to Home Screen" it works for any site, needs no web app manifest,
and travels as a file: AirDrop it, mail it, or open it in Safari.

Install: open the file on the iPhone (Safari asks to allow the
download), then Settings > Profile Downloaded > Install.  The icon
appears at once; deleting the profile under Settings > General > VPN &
Device Management removes it again.

The profile is signed as CMS/PKCS#7 ``SignedData`` (DER, content
embedded, SHA-256) with an RSA-2048 certificate issued by the daemon's
local CA (:mod:`kiss.core.tls_certs`, ``$KISS_HOME/tls``) and kept next
to it as ``profile-signer.pem`` / ``profile-signer-key.pem``.  The CA
certificate rides along in the signature, so a phone that has
installed the CA (the webapp's ``/ca.crt`` download) shows the profile
as "Verified"; any other phone shows "Not Verified" and installs it
all the same.  The signing certificate is re-issued when it is
missing, mismatched, not issued by the current CA or within 30 days of
expiry, under the same ``.tls.lock`` the daemon holds for the directory.

Usage::

    python -m kiss.scripts.webclip [--url URL] [--label NAME] [--icon PNG]
        [--icon-background '#rrggbb'] [--organization NAME] [--out FILE]
        [--tls-dir DIR] [--unsigned]

``--url`` defaults to this daemon's public tunnel URL (then its first
LAN URL) from ``$KISS_HOME/remote-url.json``; the label, icon and
organization default to the brand's.  The output is deterministic for
the same inputs apart from the signature's signing time: the profile
and payload identifiers derive from the URL, so re-generating for the
same URL replaces the installed profile instead of adding a second one.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import plistlib
import re
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
from PIL import Image, ImageOps

from kiss.core import tls_certs
from kiss.core.brand import PRODUCT_NAME
from kiss.core.config import kiss_home
from kiss.core.file_lock import exclusive_file_lock
from kiss.core.utils import atomic_write_text

SIGNER_CERT_FILE = "profile-signer.pem"
SIGNER_KEY_FILE = "profile-signer-key.pem"
PAYLOAD_TYPE = "com.apple.webClip.managed"
ICON_SIZE = 400
"""Apple: "a square image that's no larger than 400 x 400 pixels and less than 1 MB"."""
_IDENTIFIER_PREFIX = "io.github.kisssorcar.webclip"
DEFAULT_ICON = Path(__file__).resolve().parents[1] / "webapp" / "media" / "kiss-icon.png"
"""The brand icon the profile carries unless ``--icon`` names another."""
DEFAULT_ICON_BACKGROUND = "#ffffff"
"""The colour flattened under the icon's transparent pixels (see :func:`icon_png`)."""
_TLS_LOCK_TIMEOUT_S = 30.0


def default_url() -> str:
    """Return this daemon's public URL from ``$KISS_HOME/remote-url.json``, or ``''``.

    The tunnel URL when the daemon publishes one (it reaches the phone
    anywhere), else the first LAN URL (same Wi-Fi only), else ``''``
    when the file is missing or names neither.
    """
    try:
        data = json.loads((kiss_home() / "remote-url.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    lan = data.get("lan") or [""]
    return str(data.get("tunnel") or lan[0] or "")


def icon_png(path: Path, background: str) -> bytes:
    """Return the image at *path* as an opaque 400x400 PNG for the ``Icon`` key.

    Transparent pixels are flattened onto *background* (``#rrggbb``):
    iOS shows transparency as black and applies its own rounded mask,
    so a logo with transparent corners keeps clean edges only when its
    corners carry the colour of its own background.  A non-square image
    is centre-cropped before scaling.

    Args:
        path: A PNG, JPEG or GIF file.
        background: The colour behind transparent pixels, ``#rrggbb``.
    """
    with Image.open(path) as image:
        rgba = image.convert("RGBA")
    flat = Image.new("RGBA", rgba.size, background)
    flat.alpha_composite(rgba)
    square = ImageOps.fit(flat.convert("RGB"), (ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    square.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def profile_plist(url: str, label: str, icon: bytes, organization: str) -> bytes:
    """Return the unsigned profile as XML plist bytes.

    One ``Configuration`` profile holding one ``com.apple.webClip.managed``
    payload: ``FullScreen`` (no browser UI), ``Precomposed`` (no added
    shine), removable by the user, ``TargetDeviceType`` 1 (iPhone and
    iPad, which spares the "choose a device" alert on install).  The
    identifiers and UUIDs derive from *url* (UUID v5), so the same URL
    always yields the same profile identity.

    Args:
        url: The address the Web Clip opens.
        label: The name under the Home Screen icon (about 12 characters show).
        icon: PNG bytes for the icon, see :func:`icon_png`.
        organization: Shown as the profile's organization on the install screen.
    """
    identifier = f"{_IDENTIFIER_PREFIX}.{hashlib.sha256(url.encode('utf-8')).hexdigest()[:12]}"
    clip = {
        "PayloadType": PAYLOAD_TYPE,
        "PayloadVersion": 1,
        "PayloadIdentifier": f"{identifier}.clip",
        "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_URL, url + "#clip")).upper(),
        "PayloadDisplayName": label,
        "Label": label,
        "URL": url,
        "Icon": icon,
        "FullScreen": True,
        "IgnoreManifestScope": False,
        "IsRemovable": True,
        "Precomposed": True,
    }
    profile = {
        "PayloadType": "Configuration",
        "PayloadVersion": 1,
        "PayloadIdentifier": identifier,
        "PayloadUUID": str(uuid.uuid5(uuid.NAMESPACE_URL, url)).upper(),
        "PayloadDisplayName": f"{label} web clip",
        "PayloadDescription": (
            f"Adds a {label} icon to the Home Screen that opens {url} "
            "full screen, without the Safari address bar."
        ),
        "PayloadOrganization": organization,
        "TargetDeviceType": 1,
        "PayloadContent": [clip],
    }
    return plistlib.dumps(profile, sort_keys=False)


def _load_signer(
    cert_path: Path, key_path: Path, ca_cert: x509.Certificate,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey] | None:
    """Return the signing pair at the paths when it needs no re-issue, else ``None``.

    Re-issue when either file is missing or unreadable, the key is not
    RSA or does not match the certificate, the current CA did not issue
    it, or it is not yet valid (issued under a clock set ahead) or
    within :data:`tls_certs.RENEWAL_THRESHOLD_DAYS` of expiry.
    """
    try:
        cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
        key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
        cert.verify_directly_issued_by(ca_cert)
    except Exception:
        return None
    if not isinstance(key, rsa.RSAPrivateKey) or cert.public_key() != key.public_key():
        return None
    now = datetime.datetime.now(datetime.UTC)
    renewal = datetime.timedelta(days=tls_certs.RENEWAL_THRESHOLD_DAYS)
    if cert.not_valid_before_utc > now or cert.not_valid_after_utc - now <= renewal:
        return None
    return cert, key


def _issue_signer(
    cert_path: Path, key_path: Path, ca_cert: x509.Certificate, ca_key: ec.EllipticCurvePrivateKey,
) -> tuple[x509.Certificate, rsa.RSAPrivateKey]:
    """Issue and write an RSA-2048 profile-signing certificate from the local CA (key file first).

    ``digitalSignature``/``contentCommitment`` key usage and the
    ``emailProtection`` extended key usage, which is what CMS signed
    data (S/MIME) verifiers such as ``openssl smime -verify`` expect;
    iOS checks the chain, not the purpose.  Validity follows the
    server certificate's (:data:`tls_certs.SERVER_CERT_VALIDITY_DAYS`).
    """
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    host = socket.gethostname().split(".")[0] or "localhost"
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, f"{PRODUCT_NAME} profile signing {host}"[:64]),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, PRODUCT_NAME),
    ])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=tls_certs.SERVER_CERT_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, content_commitment=True, key_encipherment=False,
                data_encipherment=False, key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.EMAIL_PROTECTION]), critical=False,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )
    atomic_write_text(key_path, key_pem.decode("ascii"), mode=0o600)
    atomic_write_text(
        cert_path, cert.public_bytes(serialization.Encoding.PEM).decode("ascii"), create_mode=0o644,
    )
    return cert, key


def signer_identity(tls_dir: Path) -> tuple[x509.Certificate, rsa.RSAPrivateKey, x509.Certificate]:
    """Return ``(signing cert, its key, CA cert)`` for *tls_dir*, issuing what is missing.

    Holds the directory's ``.tls.lock`` meanwhile, as the daemon does
    for its own certificates, so a concurrent daemon start and this
    issuance never interleave their check-then-write sequences.  The CA
    itself is generated when *tls_dir* has none (a throwaway directory).

    Args:
        tls_dir: The daemon's TLS directory (``$KISS_HOME/tls``).
    """
    tls_dir.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(tls_dir / tls_certs.TLS_LOCK_FILE, timeout=_TLS_LOCK_TIMEOUT_S):
        ca_cert, ca_key = tls_certs.ensure_local_ca(tls_dir)
        cert_path, key_path = tls_dir / SIGNER_CERT_FILE, tls_dir / SIGNER_KEY_FILE
        pair = _load_signer(cert_path, key_path, ca_cert) or _issue_signer(
            cert_path, key_path, ca_cert, ca_key,
        )
    return pair[0], pair[1], ca_cert


def sign_profile(
    plist: bytes, cert: x509.Certificate, key: rsa.RSAPrivateKey, ca_cert: x509.Certificate,
) -> bytes:
    """Return *plist* wrapped in DER CMS ``SignedData`` with the content embedded.

    The shape ``openssl smime -sign -nodetach -outform der`` produces,
    which iOS requires for a signed profile; ``Binary`` keeps the plist
    bytes exact instead of normalising line endings, and *ca_cert* goes
    into the certificate set so the phone can build the chain.
    """
    return (
        pkcs7.PKCS7SignatureBuilder()
        .set_data(plist)
        .add_signer(cert, key, hashes.SHA256())
        .add_certificate(ca_cert)
        .sign(serialization.Encoding.DER, [pkcs7.PKCS7Options.Binary])
    )


def brand_profile(url: str, tls_dir: Path) -> bytes:
    """Return the signed profile for *url* with the brand's label, icon and organization.

    What ``python -m kiss.scripts.webclip --url URL`` writes with no
    other option; the daemon's ``/webclip.mobileconfig`` route serves
    it for its public URL.

    Args:
        url: The address the Web Clip opens.
        tls_dir: The daemon's TLS directory holding the signing CA (``$KISS_HOME/tls``).
    """
    icon = icon_png(DEFAULT_ICON, DEFAULT_ICON_BACKGROUND)
    plist = profile_plist(url, PRODUCT_NAME, icon, PRODUCT_NAME)
    return sign_profile(plist, *signer_identity(tls_dir))


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser (see the module docstring for the options)."""
    parser = argparse.ArgumentParser(
        prog="python -m kiss.scripts.webclip",
        description="Build a signed iOS Web Clip profile that opens a URL full screen.",
    )
    add = parser.add_argument
    add("--url", default="", help="address to open (default: this daemon's public URL)")
    add("--label", default=PRODUCT_NAME, help="name under the icon (default: %(default)s)")
    add("--icon", type=Path, default=DEFAULT_ICON, help="PNG/JPEG/GIF icon (default: brand icon)")
    add(
        "--icon-background", default=DEFAULT_ICON_BACKGROUND,
        help="colour behind transparent icon pixels (default: %(default)s)",
    )
    add("--organization", default=PRODUCT_NAME, help="profile organization (default: %(default)s)")
    add("--out", type=Path, default=None, help="output file (default: <label>.mobileconfig)")
    add("--tls-dir", type=Path, default=None, help="signing CA directory (default: $KISS_HOME/tls)")
    add("--unsigned", action="store_true", help="write the plain XML profile, unsigned")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and write the profile per *argv* (default ``sys.argv[1:]``); print install steps."""
    parser = build_parser()
    args = parser.parse_args(argv)
    url = args.url or default_url()
    if not url:
        parser.error("no --url given and $KISS_HOME/remote-url.json names no public URL")
    if urlsplit(url).scheme not in ("http", "https") or not urlsplit(url).netloc:
        parser.error(f"--url must be an http(s) URL, got {url!r}")
    if not args.icon.is_file():
        parser.error(f"--icon {args.icon} is not a file")
    icon = icon_png(args.icon, args.icon_background)
    plist = profile_plist(url, args.label, icon, args.organization)
    stem = re.sub(r"[^a-z0-9._-]+", "-", args.label.lower()).strip("-.") or "webclip"
    out = args.out or Path(f"{stem}.mobileconfig")
    if args.unsigned:
        out.write_bytes(plist)
        print(f"Wrote unsigned web clip profile for {url} to {out}")
    else:
        tls_dir = args.tls_dir or kiss_home() / "tls"
        cert, key, ca_cert = signer_identity(tls_dir)
        out.write_bytes(sign_profile(plist, cert, key, ca_cert))
        signer = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
        issuer = tls_certs.ca_common_name(tls_dir / tls_certs.CA_CERT_FILE)
        print(f"Wrote signed web clip profile for {url} to {out}")
        print(f"Signed by {signer!s}, issued by {issuer}")
        print("Shows as Verified on a phone that installed that CA (the webapp's /ca.crt).")
    print("Install: open the file on the iPhone, then Settings > Profile Downloaded > Install.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

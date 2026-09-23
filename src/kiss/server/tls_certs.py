# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""Local certificate authority behind the webapp's always-on TLS.

Browsers warned about the auto-generated certificate for two reasons:
nothing trusted its issuer, and the LAN address (``https://<lan-ip>:PORT``)
was not in its Subject Alternative Name at all, so even a manually trusted
copy still failed on the LAN URL.  This module removes both causes while
keeping TLS on:

* ``ca.pem`` / ``ca-key.pem`` — a root CA generated once per machine and
  kept stable across server-certificate renewals.  Trusting ``ca.pem``
  once (``kiss-web --trust-ca``, see :mod:`kiss.server.tls_trust`, or the
  ``/ca.crt`` download on a phone) silences the warning for every future
  server certificate issued here.
* ``cert.pem`` / ``key.pem`` — the server certificate the daemon presents,
  signed by that CA, whose SAN lists ``localhost``, the hostname, the
  loopback addresses and the machine's current LAN IPs.  It is re-issued
  whenever it is missing, expiring, not issued by the current CA, or
  lacks a current LAN IP.

The server certificate follows the constraints Apple documents for
locally trusted certificates (validity at most 825 days, a SAN, the
``serverAuth`` extended key usage, an RSA >= 2048 or ECC key, SHA-2
signature); Safari and iOS refuse certificates that violate them even
when the root is trusted.  Both keys are ECDSA P-256: generating one
takes milliseconds, so re-issuing the server certificate on a LAN IP
change never stalls the daemon (an RSA key generation would hold the
GIL for a noticeable fraction of a second).  The CA carries a Common
Name because iOS lists trusted roots by it under Settings > General >
About > Certificate Trust Settings.
"""

from __future__ import annotations

import datetime
import ipaddress
import logging
import os
import re
import socket
from collections.abc import Iterable
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

logger = logging.getLogger(__name__)

CA_CERT_FILE = "ca.pem"
CA_KEY_FILE = "ca-key.pem"
SERVER_CERT_FILE = "cert.pem"
SERVER_KEY_FILE = "key.pem"

CA_VALIDITY_DAYS = 3650
# Apple rejects TLS server certificates valid for more than 825 days,
# whatever root they chain to (support.apple.com/en-us/103769).
SERVER_CERT_VALIDITY_DAYS = 820
RENEWAL_THRESHOLD_DAYS = 30

_ORGANIZATION = "KISS Sorcar"
_CA_COMMON_NAME_PREFIX = "KISS Sorcar Local CA"
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?$")


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _host_label() -> str:
    """Return this machine's hostname when it is a usable DNS label, else ''."""
    host = socket.gethostname().strip().rstrip(".")
    return host if _HOSTNAME_RE.match(host) and len(host) <= 253 else ""


def _write_private_key(path: Path, key: ec.EllipticCurvePrivateKey) -> None:
    """Write *key* as PEM to *path*, mode 0600, replacing any old file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        logger.debug("Could not chmod 0700 on %s", path.parent, exc_info=True)
    key_bytes = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    if path.exists():
        path.unlink()
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key_bytes)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def _write_certificate(path: Path, cert: x509.Certificate) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _load_certificate(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _load_private_key(path: Path) -> ec.EllipticCurvePrivateKey:
    key = serialization.load_pem_private_key(path.read_bytes(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError(f"{path} is not an EC private key")
    return key


def _generate_key() -> ec.EllipticCurvePrivateKey:
    return ec.generate_private_key(ec.SECP256R1())


def generate_local_ca(ca_cert_path: Path, ca_key_path: Path) -> None:
    """Generate the machine-local root CA (ECDSA P-256, 10 years).

    The Common Name embeds the hostname so a phone that trusts CAs from
    several machines can tell them apart.  ``pathlen 0`` means the CA
    can only sign end-entity certificates, never further CAs.  The key
    is written before the certificate so a crash in between leaves a
    mismatched pair that :func:`ensure_local_tls_pair` detects and
    regenerates.

    Args:
        ca_cert_path: Where to write the PEM CA certificate (0644).
        ca_key_path: Where to write the PEM CA private key (0600).
    """
    key = _generate_key()
    host = _host_label() or "localhost"
    common_name = f"{_CA_COMMON_NAME_PREFIX} {host}"[:64]
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, _ORGANIZATION),
    ])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=CA_VALIDITY_DAYS))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    _write_private_key(ca_key_path, key)
    _write_certificate(ca_cert_path, cert)


def server_cert_names(
    extra_ips: Iterable[str] = (),
) -> tuple[list[str], list[ipaddress.IPv4Address | ipaddress.IPv6Address]]:
    """Return the (DNS names, IP addresses) the server certificate must cover.

    Args:
        extra_ips: Additional IP addresses (the host's LAN IPs); strings
            that do not parse as IP addresses are skipped.

    Returns:
        DNS names: ``localhost``, ``*.local``, the hostname and
        ``<hostname>.local`` when the hostname is a valid label.  IPs:
        ``127.0.0.1``, ``::1`` and the parsed *extra_ips*, deduplicated
        in order.
    """
    dns_names = ["localhost", "*.local"]
    host = _host_label()
    if host:
        dns_names.append(host)
        if "." not in host:
            dns_names.append(f"{host}.local")
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [
        ipaddress.IPv4Address("127.0.0.1"),
        ipaddress.IPv6Address("::1"),
    ]
    for raw in extra_ips:
        try:
            addr = ipaddress.ip_address(raw)
        except ValueError:
            continue
        if addr not in ips:
            ips.append(addr)
    return dns_names, ips


def issue_server_cert(
    cert_path: Path,
    key_path: Path,
    ca_cert_path: Path,
    ca_key_path: Path,
    extra_ips: Iterable[str] = (),
) -> None:
    """Issue the server certificate signed by the local CA.

    ECDSA P-256 key, 820-day validity, ``digitalSignature`` key usage
    (RFC 5480 forbids ``keyEncipherment`` on EC keys), ``serverAuth``
    extended key usage, SAN from :func:`server_cert_names`.  The key is
    written before the certificate (see :func:`generate_local_ca` for
    why).

    Args:
        cert_path: Where to write the PEM server certificate.
        key_path: Where to write the PEM server private key (0600).
        ca_cert_path: The CA certificate to chain to.
        ca_key_path: The CA private key that signs the certificate.
        extra_ips: LAN IP addresses to include in the SAN.
    """
    ca_cert = _load_certificate(ca_cert_path)
    ca_key = _load_private_key(ca_key_path)
    key = _generate_key()
    dns_names, ips = server_cert_names(extra_ips)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, dns_names[0]),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, _ORGANIZATION),
    ])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=SERVER_CERT_VALIDITY_DAYS))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False,
        )
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(n) for n in dns_names]
                + [x509.IPAddress(ip) for ip in ips],
            ),
            critical=False,
        )
        .add_extension(
            x509.SubjectKeyIdentifier.from_public_key(key.public_key()),
            critical=False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )
    _write_private_key(key_path, key)
    _write_certificate(cert_path, cert)


def cert_needs_renewal(
    cert_path: Path, threshold_days: int = RENEWAL_THRESHOLD_DAYS,
) -> bool:
    """Return True if *cert_path* is unreadable, expired or expires soon.

    Args:
        cert_path: PEM certificate to inspect.
        threshold_days: Remaining validity below which renewal is due.
    """
    try:
        not_after = _load_certificate(cert_path).not_valid_after_utc
    except Exception:
        return True
    return not_after - _now() <= datetime.timedelta(days=threshold_days)


def _ca_pair_is_usable(ca_cert_path: Path, ca_key_path: Path) -> bool:
    """Return True when the CA cert/key exist, match and are not expiring."""
    if not ca_cert_path.is_file() or not ca_key_path.is_file():
        return False
    if cert_needs_renewal(ca_cert_path):
        return False
    try:
        cert = _load_certificate(ca_cert_path)
        key = _load_private_key(ca_key_path)
    except Exception:
        return False
    return cert.public_key() == key.public_key()


def server_cert_is_current(
    cert_path: Path,
    key_path: Path,
    ca_cert_path: Path,
    required_ips: Iterable[str] = (),
) -> bool:
    """Return True when the server certificate needs no re-issue.

    The certificate is current when it and its key exist and match, it is
    not expiring within :data:`RENEWAL_THRESHOLD_DAYS`, it was signed by
    the CA at *ca_cert_path*, and its SAN covers every address in
    *required_ips* (plus the names of :func:`server_cert_names`).

    Args:
        cert_path: PEM server certificate.
        key_path: PEM server private key.
        ca_cert_path: PEM CA certificate the server cert must chain to.
        required_ips: IP addresses (strings) that must be in the SAN.
    """
    if not cert_path.is_file() or not key_path.is_file():
        return False
    if cert_needs_renewal(cert_path):
        return False
    try:
        cert = _load_certificate(cert_path)
        key = _load_private_key(key_path)
        ca_cert = _load_certificate(ca_cert_path)
        cert.verify_directly_issued_by(ca_cert)
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except Exception:
        return False
    if cert.public_key() != key.public_key():
        return False
    dns_names, ips = server_cert_names(required_ips)
    have_dns = set(san.get_values_for_type(x509.DNSName))
    have_ips = set(san.get_values_for_type(x509.IPAddress))
    return set(dns_names) <= have_dns and set(ips) <= have_ips


def ensure_local_tls_pair(
    tls_dir: Path, lan_ips: Iterable[str] = (),
) -> tuple[Path, Path]:
    """Make ``tls_dir`` hold a usable CA and a current server certificate.

    Generates the CA when it is missing, mismatched or expiring (a fresh
    CA must be trusted again; a warning says so), then re-issues the
    server certificate when :func:`server_cert_is_current` says it is
    stale.  The caller serialises concurrent callers (the daemon holds
    ``.tls.lock``).

    Args:
        tls_dir: Directory holding the four PEM files.
        lan_ips: The host's current LAN IP addresses.

    Returns:
        ``(cert_path, key_path)`` of the server certificate to load.
    """
    ca_cert_path = tls_dir / CA_CERT_FILE
    ca_key_path = tls_dir / CA_KEY_FILE
    cert_path = tls_dir / SERVER_CERT_FILE
    key_path = tls_dir / SERVER_KEY_FILE
    lan_ips = list(lan_ips)
    if not _ca_pair_is_usable(ca_cert_path, ca_key_path):
        if ca_cert_path.exists():
            logger.warning(
                "Local CA in %s is expiring or unusable; generating a new "
                "one — browsers that trusted the old CA must trust the new "
                "one (kiss-web --trust-ca)", tls_dir,
            )
        else:
            logger.info("Generating local TLS certificate authority in %s", tls_dir)
        generate_local_ca(ca_cert_path, ca_key_path)
    if not server_cert_is_current(cert_path, key_path, ca_cert_path, lan_ips):
        logger.info(
            "Issuing TLS server certificate in %s for LAN IPs %s",
            tls_dir, sorted(lan_ips),
        )
        issue_server_cert(cert_path, key_path, ca_cert_path, ca_key_path, lan_ips)
    return cert_path, key_path


def ca_fingerprint(ca_cert_path: Path) -> str:
    """Return the SHA-256 fingerprint of the CA certificate as ``AA:BB:...``.

    Shown next to the download link so a user can compare it with what the
    phone displays before trusting the certificate.
    """
    digest = _load_certificate(ca_cert_path).fingerprint(hashes.SHA256())
    return ":".join(f"{b:02X}" for b in digest)


def ca_common_name(ca_cert_path: Path) -> str:
    """Return the CA certificate's Common Name (its display name in trust UIs)."""
    cert = _load_certificate(ca_cert_path)
    attrs = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    return str(attrs[0].value) if attrs else _CA_COMMON_NAME_PREFIX

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The webapp's TLS certificate is signed by a machine-local CA.

Opening the Local (``https://127.0.0.1:PORT``) or LAN
(``https://<lan-ip>:PORT``) URL used to trigger a browser certificate
warning for two independent reasons: the certificate was self-signed (no
trust store knew its issuer) and the LAN IP was not in its SAN.  These
tests exercise the real files, the real ``ssl`` module and a live
``RemoteAccessServer``:

* :mod:`kiss.server.tls_certs` — CA + server certificate generation, the
  browser constraints (SAN, serverAuth EKU, <= 825-day validity), and
  every re-issue trigger (missing, expiring, wrong CA, mismatched key,
  missing LAN IP, unusable CA).
* :func:`kiss.server.web_server._create_ssl_context` — the daemon's
  startup path including the self-heal for a file OpenSSL rejects.
* ``GET /ca.crt`` — the CA download for phones, absent with an explicit
  certificate pair.
* :meth:`RemoteAccessServer._refresh_tls_cert` — a LAN IP change, an
  expiring certificate or a sibling daemon's re-issue makes the live server
  present the new certificate to new connections; a held lock defers it.
* :mod:`kiss.server.tls_trust` and ``kiss-web --trust-ca`` — the trust
  store installer.  The ``certutil`` success path needs libnss3-tools and
  is skipped where the tool is absent; the macOS/Windows commands are run
  for real and report their failure on other platforms.

Not covered on purpose (they depend on the host, not on the code): the
``_host_label`` branches for a hostname that is not a DNS label or that
already contains a dot, the ``chmod`` failure in ``_write_private_key``,
and the success lines of the macOS keychain / Windows store installers.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import ipaddress
import json
import os
import shutil
import ssl
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, skipUnless

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

import kiss.agents.sorcar.persistence as th
from kiss.core.brand import PRODUCT_NAME
from kiss.core.file_lock import lock_exclusive
from kiss.server import tls_certs, tls_trust
from kiss.server import web_server as ws
from kiss.server.web_server import RemoteAccessServer, _generate_self_signed_cert


@contextlib.contextmanager
def _env(**overrides: str | None) -> Iterator[None]:
    """Set (or, with ``None``, unset) environment variables for the block."""
    saved = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load(path: Path) -> x509.Certificate:
    return x509.load_pem_x509_certificate(path.read_bytes())


def _san(cert: x509.Certificate) -> x509.SubjectAlternativeName:
    return cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value


def _san_ips(cert: x509.Certificate) -> set[str]:
    return {str(ip) for ip in _san(cert).get_values_for_type(x509.IPAddress)}


def _write_cert_with_validity(
    path: Path, key_path: Path, subject: x509.Name, days: int, *, ca: bool,
) -> None:
    """Re-sign a self-signed cert for the key at *key_path* with *days* validity."""
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    now = datetime.datetime.now(datetime.UTC)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())  # type: ignore[arg-type]
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    path.write_bytes(
        builder.sign(key, hashes.SHA256())  # type: ignore[arg-type]
        .public_bytes(serialization.Encoding.PEM),
    )


class TestTlsCerts(TestCase):
    """Certificate generation and every re-issue trigger, on real files."""

    def setUp(self) -> None:
        self.tls_dir = Path(tempfile.mkdtemp()) / "tls"
        self.ca = self.tls_dir / tls_certs.CA_CERT_FILE
        self.ca_key = self.tls_dir / tls_certs.CA_KEY_FILE
        self.cert = self.tls_dir / tls_certs.SERVER_CERT_FILE
        self.key = self.tls_dir / tls_certs.SERVER_KEY_FILE

    def tearDown(self) -> None:
        shutil.rmtree(self.tls_dir.parent, ignore_errors=True)

    def test_first_run_creates_a_browser_acceptable_chain(self) -> None:
        cert_path, key_path = tls_certs.ensure_local_tls_pair(
            self.tls_dir, ["192.168.1.20", "10.0.0.5"],
        )
        self.assertEqual((cert_path, key_path), (self.cert, self.key))
        ca, leaf = _load(self.ca), _load(self.cert)

        # CA: a real root with a Common Name (iOS lists roots by it).
        bc = ca.extensions.get_extension_for_class(x509.BasicConstraints)
        self.assertTrue(bc.value.ca)
        self.assertEqual(bc.value.path_length, 0)
        self.assertTrue(
            ca.extensions.get_extension_for_class(x509.KeyUsage).value.key_cert_sign,
        )
        self.assertTrue(tls_certs.ca_common_name(self.ca).startswith(f"{PRODUCT_NAME} Local CA"))
        self.assertGreater(
            ca.not_valid_after_utc - ca.not_valid_before_utc,
            datetime.timedelta(days=3600),
        )

        # Leaf: signed by the CA, within Apple's 825-day cap, serverAuth,
        # SAN with the loopback names and both LAN IPs.
        leaf.verify_directly_issued_by(ca)
        self.assertLessEqual(
            leaf.not_valid_after_utc - leaf.not_valid_before_utc,
            datetime.timedelta(days=825),
        )
        self.assertFalse(
            leaf.extensions.get_extension_for_class(x509.BasicConstraints).value.ca,
        )
        eku = leaf.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        self.assertIn(ExtendedKeyUsageOID.SERVER_AUTH, eku)
        ku = leaf.extensions.get_extension_for_class(x509.KeyUsage).value
        self.assertTrue(ku.digital_signature)
        self.assertFalse(ku.key_encipherment, "RFC 5480: not for EC keys")
        self.assertIsInstance(leaf.public_key(), ec.EllipticCurvePublicKey)
        self.assertIn("localhost", _san(leaf).get_values_for_type(x509.DNSName))
        self.assertEqual(
            _san_ips(leaf), {"127.0.0.1", "::1", "192.168.1.20", "10.0.0.5"},
        )
        aki = leaf.extensions.get_extension_for_class(x509.AuthorityKeyIdentifier).value
        ski = ca.extensions.get_extension_for_class(x509.SubjectKeyIdentifier).value
        self.assertEqual(aki.key_identifier, ski.digest)

        # Private keys are owner-only.
        if os.name == "posix":
            self.assertEqual(self.key.stat().st_mode & 0o777, 0o600)
            self.assertEqual(self.ca_key.stat().st_mode & 0o777, 0o600)

        # The fingerprint is the colon-separated SHA-256 the phone shows.
        fp = tls_certs.ca_fingerprint(self.ca)
        self.assertEqual(len(fp.split(":")), 32)
        self.assertEqual(fp.replace(":", "").lower(), ca.fingerprint(hashes.SHA256()).hex())

    def test_same_ips_is_a_no_op_and_new_ip_reissues_leaf_only(self) -> None:
        tls_certs.ensure_local_tls_pair(self.tls_dir, ["192.168.1.20"])
        ca_bytes, cert_bytes = self.ca.read_bytes(), self.cert.read_bytes()

        tls_certs.ensure_local_tls_pair(self.tls_dir, ["192.168.1.20"])
        self.assertEqual(self.cert.read_bytes(), cert_bytes, "unchanged IPs: no re-issue")

        tls_certs.ensure_local_tls_pair(self.tls_dir, ["192.168.1.20", "172.16.0.9"])
        self.assertNotEqual(self.cert.read_bytes(), cert_bytes, "new LAN IP: re-issued")
        self.assertEqual(self.ca.read_bytes(), ca_bytes, "the trusted CA is kept")
        self.assertIn("172.16.0.9", _san_ips(_load(self.cert)))

        # A subset of the covered IPs is still covered: no churn.
        cert_bytes = self.cert.read_bytes()
        tls_certs.ensure_local_tls_pair(self.tls_dir, ["172.16.0.9"])
        self.assertEqual(self.cert.read_bytes(), cert_bytes)

    def test_leaf_from_another_ca_or_with_wrong_key_is_reissued(self) -> None:
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        other = self.tls_dir.parent / "other"
        tls_certs.ensure_local_tls_pair(other, [])

        # Leaf signed by a different CA (e.g. the old self-signed cert).
        self.cert.write_bytes((other / tls_certs.SERVER_CERT_FILE).read_bytes())
        self.key.write_bytes((other / tls_certs.SERVER_KEY_FILE).read_bytes())
        self.assertFalse(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        _load(self.cert).verify_directly_issued_by(_load(self.ca))

        # Matching CA but a key that does not belong to the cert (crash
        # between the two writes).
        good_cert = self.cert.read_bytes()
        self.key.write_bytes((other / tls_certs.SERVER_KEY_FILE).read_bytes())
        self.assertFalse(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        self.assertNotEqual(self.cert.read_bytes(), good_cert)
        self.assertTrue(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))

        # Missing key, missing SAN, expiring leaf.
        self.key.unlink()
        self.assertFalse(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        _write_cert_with_validity(
            self.cert, self.key, x509.Name([]), days=800, ca=False,
        )
        self.assertFalse(
            tls_certs.server_cert_is_current(self.cert, self.key, self.ca),
            "a cert without a SAN is not current",
        )
        _write_cert_with_validity(self.cert, self.key, x509.Name([]), days=10, ca=False)
        self.assertTrue(tls_certs.cert_needs_renewal(self.cert))
        self.assertFalse(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))

    def test_unusable_ca_is_regenerated_with_a_warning(self) -> None:
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        old_ca = self.ca.read_bytes()

        # CA key that does not match the CA cert.
        other = self.tls_dir.parent / "other"
        tls_certs.ensure_local_tls_pair(other, [])
        self.ca_key.write_bytes((other / tls_certs.CA_KEY_FILE).read_bytes())
        with self.assertLogs("kiss.server.tls_certs", level="WARNING") as logs:
            tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        self.assertIn("kiss-web --trust-ca", "\n".join(logs.output))
        self.assertNotEqual(self.ca.read_bytes(), old_ca)
        _load(self.cert).verify_directly_issued_by(_load(self.ca))

        # Expiring CA.
        subject = _load(self.ca).subject
        _write_cert_with_validity(self.ca, self.ca_key, subject, days=5, ca=True)
        self.assertFalse(tls_certs._ca_pair_is_usable(self.ca, self.ca_key))
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        self.assertTrue(tls_certs._ca_pair_is_usable(self.ca, self.ca_key))

        # CA key that is not an EC key at all.
        ed_key = ed25519.Ed25519PrivateKey.generate()
        self.ca_key.write_bytes(ed_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        self.assertFalse(tls_certs._ca_pair_is_usable(self.ca, self.ca_key))
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        _load(self.cert).verify_directly_issued_by(_load(self.ca))

        # Missing CA files, and a missing CA when checking the leaf.
        self.ca.unlink()
        self.assertFalse(tls_certs._ca_pair_is_usable(self.ca, self.ca_key))
        self.assertFalse(tls_certs.server_cert_is_current(self.cert, self.key, self.ca))
        self.assertTrue(tls_certs.cert_needs_renewal(self.ca), "unreadable = renew")

    def test_server_cert_names_skips_junk_and_dedups(self) -> None:
        dns_names, ips = tls_certs.server_cert_names(
            ["10.0.0.1", "not-an-ip", "127.0.0.1", "10.0.0.1", "fe80::1"],
        )
        self.assertEqual(dns_names[:2], ["localhost", "*.local"])
        self.assertEqual(
            ips,
            [
                ipaddress.ip_address("127.0.0.1"),
                ipaddress.ip_address("::1"),
                ipaddress.ip_address("10.0.0.1"),
                ipaddress.ip_address("fe80::1"),
            ],
        )

    def test_generate_self_signed_cert_helper_uses_the_ca_beside_the_cert(self) -> None:
        cert = self.tls_dir / "srv.pem"
        key = self.tls_dir / "srv-key.pem"
        _generate_self_signed_cert(cert, key)
        _load(cert).verify_directly_issued_by(_load(self.ca))
        ca_bytes = self.ca.read_bytes()
        _generate_self_signed_cert(cert, key)
        self.assertEqual(self.ca.read_bytes(), ca_bytes, "the CA is reused")

    def test_ca_common_name_falls_back_without_a_cn(self) -> None:
        tls_certs.ensure_local_tls_pair(self.tls_dir, [])
        nocn = self.tls_dir / "nocn.pem"
        _write_cert_with_validity(
            nocn, self.ca_key,
            x509.Name([x509.NameAttribute(NameOID.ORGANIZATION_NAME, "x")]),
            days=100, ca=True,
        )
        self.assertEqual(tls_certs.ca_common_name(nocn), f"{PRODUCT_NAME} Local CA")


class TestCreateSslContext(TestCase):
    """The daemon's startup path on the shared ``~/.kiss/tls`` directory."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.saved_tls_dir = ws._TLS_DIR
        ws._TLS_DIR = self.tmp / "tls"

    def tearDown(self) -> None:
        ws._TLS_DIR = self.saved_tls_dir
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_auto_cert_covers_the_given_lan_ips(self) -> None:
        ctx = ws._create_ssl_context(lan_ips=["10.20.30.40"])
        self.assertEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
        leaf = _load(self.tmp / "tls" / tls_certs.SERVER_CERT_FILE)
        self.assertIn("10.20.30.40", _san_ips(leaf))
        leaf.verify_directly_issued_by(_load(self.tmp / "tls" / tls_certs.CA_CERT_FILE))
        self.assertEqual(
            ws._local_ca_cert_bytes(),
            (self.tmp / "tls" / tls_certs.CA_CERT_FILE).read_bytes(),
        )

    def test_auto_cert_probes_lan_ips_when_not_given(self) -> None:
        ws._create_ssl_context()
        leaf = _load(self.tmp / "tls" / tls_certs.SERVER_CERT_FILE)
        self.assertTrue(ws._get_local_ips() <= _san_ips(leaf))

    def test_explicit_pair_is_loaded_as_is(self) -> None:
        cert, key = self.tmp / "c.pem", self.tmp / "k.pem"
        _generate_self_signed_cert(cert, key)
        ctx = ws._create_ssl_context(str(cert), str(key))
        self.assertEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
        self.assertFalse((self.tmp / "tls").exists(), "no auto-generation")
        self.assertIsNone(ws._local_ca_cert_bytes())

    def test_cert_file_openssl_rejects_self_heals(self) -> None:
        """A cert file that parses but OpenSSL refuses is re-issued under the lock.

        ``cryptography`` reads the first PEM block only, so a truncated
        second block (a half-written chain) looks current; OpenSSL's
        ``load_cert_chain`` rejects the file.
        """
        ws._create_ssl_context(lan_ips=[])
        cert_path = self.tmp / "tls" / tls_certs.SERVER_CERT_FILE
        good = cert_path.read_bytes()
        cert_path.write_bytes(
            good + b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----\n",
        )
        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        with self.assertRaises(ssl.SSLError):
            probe.load_cert_chain(
                str(cert_path), str(self.tmp / "tls" / tls_certs.SERVER_KEY_FILE),
            )
        with self.assertLogs("kiss.server.web_server", level="WARNING") as logs:
            ws._create_ssl_context(lan_ips=[])
        self.assertIn("re-issuing", "\n".join(logs.output))
        healed = cert_path.read_bytes()
        self.assertNotEqual(healed, good)
        self.assertEqual(healed.count(b"-----BEGIN CERTIFICATE-----"), 1)
        probe = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        probe.load_cert_chain(
            str(cert_path), str(self.tmp / "tls" / tls_certs.SERVER_KEY_FILE),
        )


def _peer_cert(port: int) -> x509.Certificate:
    """Connect to the live server and return the certificate it presents."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=10) as raw:
        with ctx.wrap_socket(raw) as tls:
            der = tls.getpeercert(binary_form=True)
    assert der is not None
    return x509.load_der_x509_certificate(der)


def _https_get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    """Plain HTTPS GET against the live server (certificate unverified)."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    import socket

    with socket.create_connection(("127.0.0.1", port), timeout=10) as raw:
        with ctx.wrap_socket(raw) as tls:
            tls.sendall(
                f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                "Connection: close\r\n\r\n".encode(),
            )
            chunks = []
            while True:
                chunk = tls.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    head, _, body = b"".join(chunks).partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status = int(lines[0].split()[1])
    headers = {}
    for line in lines[1:]:
        name, _, value = line.partition(":")
        headers[name.strip().lower()] = value.strip()
    return status, headers, body


class TestLiveServer(IsolatedAsyncioTestCase):
    """``/ca.crt`` and the LAN-IP-change re-issue on a live server."""

    async def asyncSetUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        kiss_dir = self.tmp / ".kiss"
        kiss_dir.mkdir()
        self.saved_persistence = (th._DB_PATH, th._db_conn, th._KISS_DIR)
        th._KISS_DIR, th._DB_PATH, th._db_conn = kiss_dir, kiss_dir / "sorcar.db", None
        self.saved_tls_dir = ws._TLS_DIR
        ws._TLS_DIR = kiss_dir / "tls"
        self.server: RemoteAccessServer | None = None

    async def asyncTearDown(self) -> None:
        if self.server is not None:
            await self.server.stop_async()
        ws._TLS_DIR = self.saved_tls_dir
        if th._db_conn is not None:
            th._db_conn.close()
        th._DB_PATH, th._db_conn, th._KISS_DIR = self.saved_persistence  # type: ignore[assignment]
        shutil.rmtree(self.tmp, ignore_errors=True)

    async def _start(self, **kwargs: object) -> int:
        self.server = RemoteAccessServer(
            host="127.0.0.1", port=0,
            url_file=self.tmp / "remote-url.json",
            uds_path=self.tmp / "sorcar.sock",
            **kwargs,  # type: ignore[arg-type]
        )
        await self.server.start_async()
        assert self.server._ws_server is not None
        return int(self.server._ws_server.sockets[0].getsockname()[1])

    async def test_ca_crt_download_and_lan_ip_change(self) -> None:
        port = await self._start()
        server = self.server
        assert server is not None
        tls_dir = ws._TLS_DIR
        assert tls_dir is not None
        ca_path = tls_dir / tls_certs.CA_CERT_FILE
        cert_path = tls_dir / tls_certs.SERVER_CERT_FILE

        # The certificate the server presents chains to the downloadable CA,
        # is the one on disk, and already covers the machine's LAN IPs.
        presented = await asyncio.to_thread(_peer_cert, port)
        presented.verify_directly_issued_by(_load(ca_path))
        self.assertEqual(presented, _load(cert_path))
        self.assertTrue(ws._get_local_ips() <= _san_ips(presented))
        # Startup records nothing (reading the file after the lock was
        # released could name a sibling's newer certificate); the first
        # watchdog tick loads the on-disk pair under the lock instead.
        self.assertEqual(server._tls_loaded_cert, b"")
        server._last_ips = ws._get_local_ips()
        await server._refresh_tls_cert()
        self.assertEqual(server._tls_loaded_cert, cert_path.read_bytes())
        self.assertEqual(await asyncio.to_thread(_peer_cert, port), presented)

        status, headers, body = await asyncio.to_thread(_https_get, port, "/ca.crt")
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/x-x509-ca-cert")
        self.assertIn('filename="kiss-sorcar-local-ca.crt"', headers["content-disposition"])
        self.assertEqual(body, ca_path.read_bytes())

        # The watchdog adopted new LAN IPs: the leaf is re-issued and the
        # live listener presents it to the next connection; the CA is kept.
        ca_before = ca_path.read_bytes()
        server._last_ips = frozenset({"10.99.88.77"})
        await server._refresh_tls_cert()
        presented = await asyncio.to_thread(_peer_cert, port)
        self.assertIn("10.99.88.77", _san_ips(presented))
        presented.verify_directly_issued_by(_load(ca_path))
        self.assertEqual(ca_path.read_bytes(), ca_before)
        self.assertEqual(server._tls_loaded_cert, cert_path.read_bytes())

        # Same IPs again, or no IPs yet: nothing happens.
        cert_before = cert_path.read_bytes()
        await server._refresh_tls_cert()
        server._last_ips = frozenset()
        await server._refresh_tls_cert()
        self.assertEqual(cert_path.read_bytes(), cert_before)

        # An expiring certificate is renewed even though the IPs did not
        # change (a daemon that stays up for years).
        server._last_ips = frozenset({"10.99.88.77"})
        _write_cert_with_validity(
            cert_path, tls_dir / tls_certs.SERVER_KEY_FILE,
            x509.Name([]), days=10, ca=False,
        )
        await server._refresh_tls_cert()
        presented = await asyncio.to_thread(_peer_cert, port)
        self.assertGreater(
            presented.not_valid_after_utc - datetime.datetime.now(datetime.UTC),
            datetime.timedelta(days=700),
        )
        self.assertIn("10.99.88.77", _san_ips(presented))

        # A sibling daemon re-issued the pair (same IPs): the live context
        # picks up the newer certificate on the next tick.
        sibling_cert = cert_path.read_bytes()
        tls_certs.issue_server_cert(
            cert_path, tls_dir / tls_certs.SERVER_KEY_FILE,
            ca_path, tls_dir / tls_certs.CA_KEY_FILE, ["10.99.88.77"],
        )
        self.assertNotEqual(cert_path.read_bytes(), sibling_cert)
        await server._refresh_tls_cert()
        self.assertEqual(
            (await asyncio.to_thread(_peer_cert, port)), _load(cert_path),
        )

        # While a sibling holds the lock the live reload is skipped (the
        # next tick retries); nothing is loaded and nothing is recorded.
        tls_certs.issue_server_cert(
            cert_path, tls_dir / tls_certs.SERVER_KEY_FILE,
            ca_path, tls_dir / tls_certs.CA_KEY_FILE, ["10.99.88.77"],
        )
        loaded_before = server._tls_loaded_cert
        assert server._ssl_context is not None
        with open(tls_dir / ".tls.lock", "w", encoding="utf-8") as held:
            self.assertTrue(lock_exclusive(held, blocking=False))
            self.assertFalse(
                ws._reload_local_tls_pair_if_unlocked(
                    server._ssl_context, cert_path.read_bytes(),
                ),
            )
        self.assertEqual(server._tls_loaded_cert, loaded_before)
        # ... as is a pair a sibling replaced between the two halves.
        self.assertFalse(
            ws._reload_local_tls_pair_if_unlocked(server._ssl_context, b"stale"),
        )
        await server._refresh_tls_cert()
        self.assertEqual(server._tls_loaded_cert, cert_path.read_bytes())

        # With the context gone (shutdown) the refresh is a no-op too.
        saved_ctx = server._ssl_context
        server._ssl_context = None
        server._last_ips = frozenset({"10.1.1.1"})
        await server._refresh_tls_cert()
        self.assertNotIn("10.1.1.1", _san_ips(_load(cert_path)))
        server._ssl_context = saved_ctx

        # The remote_url payload and the URL file advertise the local CA
        # so the webview shows the trust hint.
        self.assertTrue(server._serves_local_ca)
        server._write_url_file_sync(None)
        self.assertTrue(json.loads((self.tmp / "remote-url.json").read_text())["localCa"])

    async def test_explicit_cert_pair_has_no_ca_download_and_no_reissue(self) -> None:
        cert, key = self.tmp / "c.pem", self.tmp / "k.pem"
        _generate_self_signed_cert(cert, key)
        port = await self._start(certfile=str(cert), keyfile=str(key))
        server = self.server
        assert server is not None
        status, _headers, _body = await asyncio.to_thread(_https_get, port, "/ca.crt")
        self.assertEqual(status, 404)
        before = cert.read_bytes()
        server._last_ips = frozenset({"10.99.88.77"})
        await server._refresh_tls_cert()
        self.assertEqual(cert.read_bytes(), before, "explicit pairs are never rewritten")
        self.assertFalse((ws._TLS_DIR / tls_certs.CA_CERT_FILE).exists())  # type: ignore[operator]
        self.assertFalse(server._serves_local_ca)
        server._write_url_file_sync(None)
        self.assertNotIn("localCa", json.loads((self.tmp / "remote-url.json").read_text()))


class TestTrustLocalCa(TestCase):
    """The ``--trust-ca`` installer against real trust-store tooling."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "home"
        self.home.mkdir()
        tls_dir = self.tmp / "tls"
        tls_certs.ensure_local_tls_pair(tls_dir, [])
        self.ca = tls_dir / tls_certs.CA_CERT_FILE

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _nss_dir(self, rel: str, db: str = "cert9.db") -> Path:
        d = self.home / rel
        d.mkdir(parents=True)
        (d / db).write_bytes(b"")
        return d

    def test_nss_database_discovery(self) -> None:
        self.assertEqual(tls_trust.nss_databases(self.home), [])
        legacy = self._nss_dir(".pki/nssdb", "cert8.db")
        chromium = self._nss_dir(".local/share/pki/nssdb")
        firefox = self._nss_dir(".mozilla/firefox/abc.default-release")
        (self.home / ".mozilla/firefox/no-db").mkdir()
        self.assertEqual(
            tls_trust.nss_databases(self.home),
            [f"dbm:{legacy}", f"sql:{chromium}", f"sql:{firefox}"],
        )

    def test_linux_report_without_databases_or_certutil(self) -> None:
        lines = tls_trust.trust_local_ca(self.ca, home=self.home, platform="linux")
        text = "\n".join(lines)
        self.assertIn(str(self.ca), text)
        self.assertIn(tls_certs.ca_fingerprint(self.ca), text)
        self.assertIn("No Chromium/Firefox NSS certificate database found", text)
        self.assertIn("update-ca-certificates", text)
        self.assertIn("/ca.crt", text)

        self._nss_dir(".pki/nssdb")
        # Hide certutil from PATH *and* from the Homebrew keg probe (this
        # machine may have Homebrew nss installed outside PATH).
        empty = self.tmp / "empty-bin"
        with _env(PATH=str(empty), HOMEBREW_PREFIX=str(empty)):
            lines = tls_trust.trust_local_ca(self.ca, home=self.home, platform="linux")
        self.assertTrue(
            any(line.startswith("certutil not found") for line in lines), lines,
        )

    def test_certutil_is_found_in_the_homebrew_keg(self) -> None:
        empty = self.tmp / "empty-bin"
        keg_certutil = self.tmp / "brew" / "opt" / "nss" / "bin" / "certutil"
        keg_certutil.parent.mkdir(parents=True)
        keg_certutil.write_text("#!/bin/sh\nexit 0\n")
        with _env(PATH=str(empty), HOMEBREW_PREFIX=str(self.tmp / "brew")):
            self.assertEqual(tls_trust._find_certutil(), str(keg_certutil))
        with _env(PATH=str(empty), HOMEBREW_PREFIX=None):
            found = tls_trust._find_certutil()
        defaults = [
            p for p in ("/opt/homebrew/opt/nss/bin/certutil", "/usr/local/opt/nss/bin/certutil")
            if Path(p).is_file()
        ]
        self.assertEqual(found, defaults[0] if defaults else None)

    @skipUnless(shutil.which("certutil"), "needs libnss3-tools (certutil)")
    def test_nss_install_is_verified_by_certutil(self) -> None:  # pragma: no cover
        db = self._nss_dir(".pki/nssdb")
        (db / "cert9.db").unlink()
        subprocess.run(
            ["certutil", "-N", "-d", f"sql:{db}", "--empty-password"], check=True,
        )
        lines = tls_trust.trust_local_ca(self.ca, home=self.home, platform="linux")
        self.assertIn(f"Trusted in NSS database sql:{db}", lines)
        listing = subprocess.run(
            ["certutil", "-L", "-d", f"sql:{db}"], capture_output=True, text=True,
            check=True,
        ).stdout
        self.assertIn(tls_trust.ca_nickname(self.ca), listing)
        # Idempotent.
        lines = tls_trust.trust_local_ca(self.ca, home=self.home, platform="linux")
        self.assertIn(f"Trusted in NSS database sql:{db}", lines)

    def test_macos_and_windows_paths_report_their_tool_result(self) -> None:
        """On this platform the macOS/Windows tools are missing: the failure is reported."""
        mac = tls_trust.trust_local_ca(self.ca, home=self.home, platform="darwin")
        win = tls_trust.trust_local_ca(self.ca, home=self.home, platform="win32")
        if sys.platform == "darwin":
            self.assertTrue(any("macOS login keychain" in line for line in mac))
        else:
            self.assertTrue(any(line.startswith("FAILED for macOS login keychain") for line in mac))
        if sys.platform == "win32":
            self.assertTrue(any("Windows current-user Root store" in line for line in win))
        else:
            self.assertTrue(
                any(line.startswith("FAILED for the Windows Root store") for line in win),
            )

    def test_cli_trust_ca_creates_the_ca_and_prints_the_report(self) -> None:
        env = dict(os.environ, KISS_HOME=str(self.tmp / "kiss-home"), HOME=str(self.home))
        proc = subprocess.run(
            [sys.executable, "-m", "kiss.server.web_server", "--trust-ca"],
            capture_output=True, text=True, timeout=180, env=env, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        ca_path = self.tmp / "kiss-home" / "tls" / tls_certs.CA_CERT_FILE
        self.assertTrue(ca_path.is_file())
        self.assertIn(f"CA certificate: {ca_path}", proc.stdout)
        self.assertIn(tls_certs.ca_fingerprint(ca_path), proc.stdout)

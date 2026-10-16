# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""The certificates behind a machine link (mutual TLS between daemons).

:mod:`kiss.core.tls_certs` issues the client certificate a daemon
presents to the machines it links to and recognises, on the other
side, the enrolled CA that issued a peer's certificate;
:func:`kiss.server.web_server._load_trusted_client_cas` loads the
enrolled CAs into a live server context.  These tests exercise the real
files and the real ``ssl`` module; the end-to-end handshakes between two
daemons are in :mod:`kiss.tests.agents.sorcar.test_machines`.
"""

from __future__ import annotations

import datetime
import logging
import os
import ssl
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from kiss.core import tls_certs
from kiss.server import web_server as ws


def _der(path: Path) -> bytes:
    return x509.load_pem_x509_certificate(path.read_bytes()).public_bytes(
        serialization.Encoding.DER,
    )


def _load(ctx: ssl.SSLContext, loaded: set[bytes]) -> list[str]:
    """Read the trusted-clients directory and load its new CAs into *ctx*, as the daemon does."""
    return ws._load_trusted_client_cas(ctx, loaded, ws._read_trusted_client_cas())


def test_client_pair_is_issued_by_the_local_ca_and_renewed_with_it(tmp_path: Path) -> None:
    """The client leaf has ``clientAuth``, no SAN, and follows the CA it chains to."""
    tls_dir = tmp_path / "tls"
    cert_path, key_path = tls_certs.ensure_local_client_pair(tls_dir)  # no CA yet: made too
    assert (cert_path, key_path) == (tls_dir / "client.pem", tls_dir / "client-key.pem")
    ca = x509.load_pem_x509_certificate((tls_dir / "ca.pem").read_bytes())
    cert = x509.load_pem_x509_certificate(cert_path.read_bytes())
    cert.verify_directly_issued_by(ca)
    eku = cert.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
    assert list(eku) == [ExtendedKeyUsageOID.CLIENT_AUTH]
    with pytest.raises(x509.ExtensionNotFound):
        cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
    assert not cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    cn = cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
    assert str(cn).endswith(tls_certs._host_label() or "localhost") and "machine link" in str(cn)
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    assert key.public_key() == cert.public_key()

    # Current: nothing is rewritten.
    before = cert_path.read_bytes(), key_path.read_bytes()
    assert tls_certs.ensure_local_client_pair(tls_dir) == (cert_path, key_path)
    assert (cert_path.read_bytes(), key_path.read_bytes()) == before
    # The CA changed (regenerated): the leaf is re-issued under the new CA.
    tls_certs.generate_local_ca(tls_dir / "ca.pem", tls_dir / "ca-key.pem")
    tls_certs.ensure_local_client_pair(tls_dir)
    assert cert_path.read_bytes() != before[0]
    x509.load_pem_x509_certificate(cert_path.read_bytes()).verify_directly_issued_by(
        x509.load_pem_x509_certificate((tls_dir / "ca.pem").read_bytes()),
    )
    # A key that no longer matches the certificate: re-issued as well.
    key_path.write_bytes(before[1])
    tls_certs.ensure_local_client_pair(tls_dir)
    assert key_path.read_bytes() != before[1]
    # The CA's label: this host and the first 8 hex digits of its fingerprint.
    label = tls_certs.ca_label(tls_dir / "ca.pem")
    digest = x509.load_pem_x509_certificate((tls_dir / "ca.pem").read_bytes()).fingerprint(
        hashes.SHA256(),
    )
    host, _, fp = label.rpartition("-")
    assert fp == digest[:4].hex() and host and "/" not in host and " " not in host


def test_enrolled_client_ca_names_the_issuer_that_is_still_on_disk(tmp_path: Path) -> None:
    """A peer certificate maps to the enrolled CA file that signed it, and only while it exists."""
    alice, bob = tmp_path / "alice", tmp_path / "bob"
    alice_cert, _ = tls_certs.ensure_local_client_pair(alice)
    bob_cert, _ = tls_certs.ensure_local_client_pair(bob)
    trusted = tmp_path / "trusted-clients"
    assert tls_certs.trusted_client_cas(trusted) == []  # no directory yet
    assert tls_certs.enrolled_client_ca(_der(alice_cert), trusted) == ""
    trusted.mkdir()
    (trusted / "alice-1234.pem").write_bytes((alice / "ca.pem").read_bytes())
    (trusted / "notes.txt").write_text("not a certificate", encoding="utf-8")
    (trusted / "junk.pem").write_text("not a certificate either", encoding="utf-8")
    (trusted / "subdir.pem").mkdir()
    assert [name for name, _pem in tls_certs.trusted_client_cas(trusted)] == [
        "alice-1234", "junk",
    ]
    assert tls_certs.enrolled_client_ca(_der(alice_cert), trusted) == "alice-1234"
    assert tls_certs.enrolled_client_ca(_der(bob_cert), trusted) == ""  # another CA
    assert tls_certs.enrolled_client_ca(b"\x30\x03\x02\x01\x01", trusted) == ""  # not a cert
    assert tls_certs.enrolled_client_ca(_der(alice / "ca.pem"), trusted) == ""  # the CA itself
    # A leaf without BasicConstraints (another issuer's convention) is a client.
    ca_key = serialization.load_pem_private_key((alice / "ca-key.pem").read_bytes(), None)
    assert isinstance(ca_key, ec.EllipticCurvePrivateKey)
    ca = x509.load_pem_x509_certificate((alice / "ca.pem").read_bytes())
    now = datetime.datetime.now(datetime.UTC)
    bare = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "bare")]))
        .issuer_name(ca.subject)
        .public_key(ec.generate_private_key(ec.SECP256R1()).public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .sign(ca_key, hashes.SHA256())
    )
    assert tls_certs.enrolled_client_ca(bare.public_bytes(serialization.Encoding.DER), trusted) == (
        "alice-1234"
    )
    # A file the daemon cannot read is skipped.
    (trusted / "secret.pem").write_bytes((alice / "ca.pem").read_bytes())
    (trusted / "secret.pem").chmod(0o000)
    try:
        if os.access(trusted / "secret.pem", os.R_OK):
            pytest.skip("running as root: every file is readable")
        assert [name for name, _pem in tls_certs.trusted_client_cas(trusted)] == [
            "alice-1234", "junk",
        ]
    finally:
        (trusted / "secret.pem").chmod(0o600)
    (trusted / "alice-1234.pem").unlink()  # revoked
    assert tls_certs.enrolled_client_ca(_der(alice_cert), trusted) == "secret"
    (trusted / "secret.pem").unlink()
    assert tls_certs.enrolled_client_ca(_der(alice_cert), trusted) == ""


def test_enrolled_cas_are_loaded_into_a_live_context_once_each(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """New CA files turn on client-certificate requests; junk is logged once; nothing twice."""
    saved = ws._TLS_DIR
    ws._TLS_DIR = tmp_path / "tls"
    try:
        trusted = ws._trusted_clients_dir()
        assert trusted == tmp_path / "tls" / "trusted-clients" and not trusted.exists()
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        loaded: set[bytes] = set()
        assert _load(ctx, loaded) == []
        assert trusted.is_dir() and ctx.verify_mode == ssl.CERT_NONE and loaded == set()
        tls_certs.ensure_local_client_pair(tmp_path / "alice")
        tls_certs.ensure_local_client_pair(tmp_path / "bob")
        (trusted / "alice.pem").write_bytes((tmp_path / "alice" / "ca.pem").read_bytes())
        (trusted / "junk.pem").write_text("junk", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=ws.__name__):
            assert _load(ctx, loaded) == ["alice"]
            assert _load(ctx, loaded) == []
        assert ctx.verify_mode == ssl.CERT_OPTIONAL
        assert [r.getMessage() for r in caplog.records] == [
            f"Ignoring {trusted}/junk.pem: not a PEM certificate",
        ]
        assert len(ctx.get_ca_certs()) == 1
        (trusted / "bob.pem").write_bytes((tmp_path / "bob" / "ca.pem").read_bytes())
        assert _load(ctx, loaded) == ["bob"]
        assert len(ctx.get_ca_certs()) == 2 and len(loaded) == 3
        # A connection that is not TLS (or not a connection at all) has no client CA.
        server = ws.RemoteAccessServer.__new__(ws.RemoteAccessServer)
        assert server._peer_client_ca(object()) == ""
    finally:
        ws._TLS_DIR = saved

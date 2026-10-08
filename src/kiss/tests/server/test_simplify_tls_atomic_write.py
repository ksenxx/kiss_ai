# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here

"""The local CA and server certificate files are published atomically.

``ca.pem`` is read without the TLS lock by the ``/ca.crt`` handler and by
every local client that builds its trust store from the endpoint file.
``_write_certificate`` used to ``write_bytes`` (truncate the existing
inode, then fill it), so a reader that overlapped a CA regeneration could
get an empty or mixed file; ``_write_private_key`` used a bare ``os.write``
whose short-write count was ignored.  Both now go through
:func:`kiss.core.utils.atomic_write_text`.

The torn-read case is reproduced deterministically instead of with a racing
thread: a reader opens ``ca.pem``, reads part of it, the CA is regenerated,
and the reader reads the rest.  With an atomic replace the open descriptor
still refers to the old inode, so the reader gets exactly the old PEM; with
the in-place truncate it got the old head followed by the new tail (neither
certificate).  Windows refuses to replace a file that another handle has
open, so that case is POSIX-only.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest import TestCase

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from kiss.server import tls_certs
from kiss.tests.conftest import posix_only


class TlsFilesAreWrittenAtomicallyTests(TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tls_dir = Path(self._tmp.name) / "tls"
        self.ca = self.tls_dir / tls_certs.CA_CERT_FILE
        self.ca_key = self.tls_dir / tls_certs.CA_KEY_FILE
        self.cert = self.tls_dir / tls_certs.SERVER_CERT_FILE
        self.key = self.tls_dir / tls_certs.SERVER_KEY_FILE

    def test_files_have_expected_mode_and_content(self) -> None:
        tls_certs.ensure_local_tls_pair(self.tls_dir, ["192.168.1.20"])

        # Only the four PEM files remain: no staging file is left behind.
        self.assertEqual(
            sorted(p.name for p in self.tls_dir.iterdir()),
            sorted(
                [
                    tls_certs.CA_CERT_FILE,
                    tls_certs.CA_KEY_FILE,
                    tls_certs.SERVER_CERT_FILE,
                    tls_certs.SERVER_KEY_FILE,
                ]
            ),
        )
        for cert_path, key_path in ((self.ca, self.ca_key), (self.cert, self.key)):
            cert_bytes = cert_path.read_bytes()
            key_bytes = key_path.read_bytes()
            self.assertTrue(cert_bytes.startswith(b"-----BEGIN CERTIFICATE-----\n"))
            self.assertTrue(cert_bytes.endswith(b"-----END CERTIFICATE-----\n"))
            self.assertTrue(key_bytes.startswith(b"-----BEGIN PRIVATE KEY-----\n"))
            self.assertTrue(key_bytes.endswith(b"-----END PRIVATE KEY-----\n"))
            cert = x509.load_pem_x509_certificate(cert_bytes)
            key = serialization.load_pem_private_key(key_bytes, password=None)
            self.assertEqual(cert.public_key(), key.public_key())
            self.assertEqual(
                cert.public_bytes(serialization.Encoding.PEM),
                cert_bytes,
            )
            if os.name == "posix":
                self.assertEqual(key_path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(self.tls_dir.stat().st_mode & 0o777, 0o700)
                mode = cert_path.stat().st_mode & 0o777
                self.assertEqual(mode & 0o600, 0o600)
                self.assertEqual(mode & 0o022, 0)

        # Rewriting keeps the key private and the certificate complete,
        # whatever the previous file's bits were.
        if os.name == "posix":
            self.ca_key.chmod(0o644)
        tls_certs.generate_local_ca(self.ca, self.ca_key)
        if os.name == "posix":
            self.assertEqual(self.ca_key.stat().st_mode & 0o777, 0o600)
        new_ca = x509.load_pem_x509_certificate(self.ca.read_bytes())
        new_key = serialization.load_pem_private_key(
            self.ca_key.read_bytes(),
            password=None,
        )
        self.assertEqual(new_ca.public_key(), new_key.public_key())

    @posix_only("an open handle blocks os.replace on Windows")
    def test_reader_overlapping_ca_regeneration_sees_one_whole_pem(self) -> None:
        tls_certs.generate_local_ca(self.ca, self.ca_key)
        old_pem = self.ca.read_bytes()

        # Unbuffered: a BufferedReader would slurp the whole small file
        # on the first read and hide the second half's origin.
        with self.ca.open("rb", buffering=0) as reader:
            head = reader.read(len(old_pem) // 2)
            for _ in range(5):
                tls_certs.generate_local_ca(self.ca, self.ca_key)
            tail = reader.readall()

        new_pem = self.ca.read_bytes()
        self.assertNotEqual(old_pem, new_pem)
        seen = head + tail
        self.assertIn(seen, (old_pem, new_pem))
        x509.load_pem_x509_certificate(seen)
        # The ``/ca.crt`` handler reads the whole file at once; after the
        # rewrite it serves the new CA, never an empty body.
        self.assertEqual(self.ca.read_bytes(), new_pem)
        self.assertEqual(self.ca.stat().st_size, len(new_pem))

# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests of ``python -m kiss.scripts.webclip`` (signed iOS Web Clip profiles).

The script runs as a subprocess against a throwaway ``KISS_HOME``; the
signed output is checked the way a phone checks it: ``openssl smime
-verify`` against the local CA must accept the CMS signature and give
back the plist, which must carry one full-screen ``com.apple.webClip
.managed`` payload with a 400x400 opaque PNG icon.
"""

from __future__ import annotations

import datetime
import os
import plistlib
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from cryptography.hazmat.primitives.serialization import pkcs7
from cryptography.x509.oid import ExtendedKeyUsageOID
from PIL import Image

from kiss.core import tls_certs
from kiss.scripts import webclip
from kiss.tests.conftest import IS_WINDOWS, openssl_cli

_OPENSSL = openssl_cli()


def _run(home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the script with ``KISS_HOME`` at *home* and *home* as the working directory."""
    return subprocess.run(
        [sys.executable, "-m", "kiss.scripts.webclip", *args],
        env={**os.environ, "KISS_HOME": str(home)}, cwd=home,
        capture_output=True, text=True, timeout=180, check=False,
    )


def _verified_plist(signed: Path, ca: Path) -> dict:
    """Return the plist inside *signed* once ``openssl smime -verify`` accepts it against *ca*."""
    plain = signed.with_suffix(".plist")
    done = subprocess.run(
        [_OPENSSL, "smime", "-verify", "-inform", "der", "-in", str(signed),
         "-CAfile", str(ca), "-out", str(plain)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr
    return dict(plistlib.loads(plain.read_bytes()))


def _clip_url(profile: Path) -> str:
    return str(plistlib.loads(profile.read_bytes())["PayloadContent"][0]["URL"])


def _pem_key(key: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption(),
    )


def _resigned(
    signer: x509.Certificate, ca_cert: x509.Certificate, ca_key: ec.EllipticCurvePrivateKey,
    start: datetime.datetime, end: datetime.datetime,
) -> bytes:
    """Return *signer*'s certificate re-signed by the CA with validity *start*..*end*, as PEM."""
    cert = (
        x509.CertificateBuilder()
        .subject_name(signer.subject).issuer_name(ca_cert.subject)
        .public_key(signer.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(start).not_valid_after(end)
        .sign(ca_key, hashes.SHA256())
    )
    return cert.public_bytes(serialization.Encoding.PEM)


class WebClipScriptTest(unittest.TestCase):
    """The script's contract, exercised through its command line."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.tls = self.home / "tls"
        # A 300x200 logo with transparent corners: exercises the centre
        # crop and the flattening onto the background colour.
        self.icon = self.home / "logo.png"
        logo = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        logo.paste((200, 30, 30, 255), (40, 20, 260, 180))
        logo.save(self.icon)

    def _signer(self) -> x509.Certificate:
        return x509.load_pem_x509_certificate((self.tls / webclip.SIGNER_CERT_FILE).read_bytes())

    def test_signed_profile_verifies_and_reuses_its_signing_identity(self) -> None:
        out = self.home / "clip.mobileconfig"
        url = "https://example.com/app?x=1"
        done = _run(
            self.home, "--url", url, "--label", "Demo Clip", "--icon", str(self.icon),
            "--icon-background", "#102030", "--organization", "Acme", "--out", str(out),
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("Signed by", done.stdout)
        self.assertIn("Settings > Profile Downloaded > Install", done.stdout)
        profile = _verified_plist(out, self.tls / tls_certs.CA_CERT_FILE)
        self.assertEqual(profile["PayloadType"], "Configuration")
        self.assertEqual(profile["PayloadVersion"], 1)
        self.assertEqual(profile["TargetDeviceType"], 1)
        self.assertEqual(profile["PayloadOrganization"], "Acme")
        self.assertTrue(profile["PayloadIdentifier"].startswith("io.github.kisssorcar.webclip."))
        self.assertIn(url, profile["PayloadDescription"])
        (clip,) = profile["PayloadContent"]
        self.assertEqual(clip["PayloadType"], webclip.PAYLOAD_TYPE)
        self.assertEqual(clip["PayloadIdentifier"], profile["PayloadIdentifier"] + ".clip")
        self.assertEqual(
            (clip["URL"], clip["Label"], clip["PayloadDisplayName"]),
            (url, "Demo Clip", "Demo Clip"),
        )
        self.assertTrue(clip["FullScreen"] and clip["IsRemovable"] and clip["Precomposed"])
        self.assertFalse(clip["IgnoreManifestScope"])
        icon = Image.open(BytesIO(clip["Icon"]))
        self.assertEqual((icon.format, icon.mode, icon.size), ("PNG", "RGB", (400, 400)))
        self.assertEqual(icon.getpixel((0, 0)), (16, 32, 48))  # transparent corner flattened
        self.assertEqual(icon.getpixel((200, 200)), (200, 30, 30))  # the logo, centre-cropped
        # The signature carries the signing certificate and the CA; the
        # signer is an RSA-2048 leaf of the CA with signing usages only.
        ca = x509.load_pem_x509_certificate((self.tls / tls_certs.CA_CERT_FILE).read_bytes())
        signer = self._signer()
        self.assertEqual(
            {c.serial_number for c in pkcs7.load_der_pkcs7_certificates(out.read_bytes())},
            {ca.serial_number, signer.serial_number},
        )
        signer.verify_directly_issued_by(ca)
        public = signer.public_key()
        self.assertIsInstance(public, rsa.RSAPublicKey)
        self.assertEqual(getattr(public, "key_size", 0), 2048)
        usage = signer.extensions.get_extension_for_class(x509.KeyUsage).value
        self.assertTrue(usage.digital_signature and usage.content_commitment)
        self.assertFalse(usage.key_cert_sign or usage.key_encipherment)
        eku = signer.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
        self.assertEqual(list(eku), [ExtendedKeyUsageOID.EMAIL_PROTECTION])
        if not IS_WINDOWS:  # NTFS has no mode bits
            self.assertEqual((self.tls / webclip.SIGNER_KEY_FILE).stat().st_mode & 0o777, 0o600)
        # Same URL again: same identity in the plist, same signing certificate.
        done = _run(self.home, "--url", url, "--icon", str(self.icon), "--out", str(out))
        self.assertEqual(done.returncode, 0, done.stderr)
        again = _verified_plist(out, self.tls / tls_certs.CA_CERT_FILE)
        (again_clip,) = again["PayloadContent"]
        self.assertEqual(
            (again["PayloadIdentifier"], again["PayloadUUID"], again_clip["PayloadUUID"]),
            (profile["PayloadIdentifier"], profile["PayloadUUID"], clip["PayloadUUID"]),
        )
        self.assertEqual(self._signer().serial_number, signer.serial_number)
        self.assertEqual(again["PayloadOrganization"], again_clip["Label"])  # both brand defaults

    def test_signing_identity_is_reissued_when_unusable(self) -> None:
        out = self.home / "clip.mobileconfig"
        args = ("--url", "https://example.org/", "--icon", str(self.icon), "--out", str(out))
        self.assertEqual(_run(self.home, *args).returncode, 0)
        key_path = self.tls / webclip.SIGNER_KEY_FILE
        cert_path = self.tls / webclip.SIGNER_CERT_FILE
        serials = [self._signer().serial_number]
        ca_cert, ca_key = tls_certs.ensure_local_ca(self.tls)
        good_cert, good_key = cert_path.read_bytes(), key_path.read_bytes()
        now = datetime.datetime.now(datetime.UTC)
        day = datetime.timedelta(days=1)
        signer = self._signer()
        ec_cert, ec_key = tls_certs.ensure_local_client_pair(self.tls)
        unusable = {
            "garbage key": (good_cert, b"not a key"),
            "ec key": (good_cert, _pem_key(ec.generate_private_key(ec.SECP256R1()))),
            "ec pair issued by the CA": (ec_cert.read_bytes(), ec_key.read_bytes()),
            "other rsa key": (good_cert, _pem_key(rsa.generate_private_key(65537, 2048))),
            "expiring cert": (_resigned(signer, ca_cert, ca_key, now - day, now + day), good_key),
            "not yet valid cert": (
                _resigned(signer, ca_cert, ca_key, now + day, now + 400 * day), good_key,
            ),
            "not issued by the CA": (good_cert, good_key),
        }
        for reason, (cert_bytes, key_bytes) in unusable.items():
            cert_path.write_bytes(cert_bytes)
            key_path.write_bytes(key_bytes)
            if reason == "not issued by the CA":
                tls_certs.generate_local_ca(
                    self.tls / tls_certs.CA_CERT_FILE, self.tls / tls_certs.CA_KEY_FILE,
                )
            done = _run(self.home, *args)
            self.assertEqual(done.returncode, 0, (reason, done.stderr))
            serials.append(self._signer().serial_number)
            self.assertEqual(len(set(serials)), len(serials), reason)
            self.assertIsInstance(self._signer().public_key(), rsa.RSAPublicKey, reason)
            _verified_plist(out, self.tls / tls_certs.CA_CERT_FILE)

    def test_url_defaults_to_the_daemon_public_url(self) -> None:
        out = self.home / "clip.mobileconfig"
        args = ("--icon", str(self.icon), "--out", str(out), "--unsigned")
        done = _run(self.home, *args)
        self.assertEqual(done.returncode, 2, done.stdout)
        self.assertIn("remote-url.json names no public URL", done.stderr)
        url_file = self.home / "remote-url.json"
        # A file that is not the object the daemon writes names no URL either.
        for nameless in ('{"local": "http://localhost:8787"}', "[]", '{"lan": 5}'):
            url_file.write_text(nameless, encoding="utf-8")
            done = _run(self.home, *args)
            self.assertEqual(done.returncode, 2, (nameless, done.stderr))
            self.assertIn("remote-url.json names no public URL", done.stderr)
        url_file.write_text('{"lan": ["https://10.0.0.5:8787"]}', encoding="utf-8")
        self.assertEqual(_run(self.home, *args).returncode, 0)
        self.assertEqual(_clip_url(out), "https://10.0.0.5:8787")
        url_file.write_text(
            '{"tunnel": "https://demo.trycloudflare.com", "lan": ["https://10.0.0.5:8787"]}',
            encoding="utf-8",
        )
        self.assertEqual(_run(self.home, *args).returncode, 0)
        self.assertEqual(_clip_url(out), "https://demo.trycloudflare.com")

    def test_unsigned_output_default_name_and_argument_errors(self) -> None:
        done = _run(self.home, "--url", "https://example.com", "--label", "My App", "--unsigned",
                    "--icon", str(self.icon), "--tls-dir", str(self.home / "elsewhere"))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("unsigned", done.stdout)
        out = self.home / "my-app.mobileconfig"
        self.assertTrue(out.read_bytes().startswith(b"<?xml"))
        self.assertEqual(plistlib.loads(out.read_bytes())["PayloadContent"][0]["Label"], "My App")
        self.assertFalse((self.home / "elsewhere").exists())
        self.assertFalse(self.tls.exists())
        # The default name is a safe file name whatever the label says.
        done = _run(self.home, "--url", "https://example.com", "--label", "Work/Chat: v2",
                    "--unsigned", "--icon", str(self.icon))
        self.assertEqual(done.returncode, 0, done.stderr)
        renamed = plistlib.loads((self.home / "work-chat-v2.mobileconfig").read_bytes())
        self.assertEqual(renamed["PayloadDisplayName"], "Work/Chat: v2 web clip")
        no_image = self.home / "text.png"
        no_image.write_text("not an image", encoding="utf-8")
        for bad in (("--url", "ftp://example.com"), ("--url", "https://"),
                    ("--url", "https://example.com", "--icon", str(self.home / "missing.png")),
                    ("--url", "https://example.com", "--icon", str(no_image)),
                    ("--url", "https://example.com", "--icon", str(self.icon),
                     "--icon-background", "no-colour")):
            done = _run(self.home, *bad)
            self.assertEqual(done.returncode, 2, bad)
            self.assertIn("--url" if bad[1] != "https://example.com" else "--icon", done.stderr)
        # --tls-dir names where the CA and signing identity live.
        elsewhere, signed = self.home / "elsewhere", self.home / "signed.mobileconfig"
        done = _run(self.home, "--url", "https://example.com", "--icon", str(self.icon),
                    "--out", str(signed), "--tls-dir", str(elsewhere))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertTrue((elsewhere / webclip.SIGNER_CERT_FILE).is_file())
        self.assertFalse(self.tls.exists())
        _verified_plist(signed, elsewhere / tls_certs.CA_CERT_FILE)

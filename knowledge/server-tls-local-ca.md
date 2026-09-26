---
title: 'Always-on TLS for kiss-web: local CA, server certificate and kiss-web --trust-ca'
uuid: 8ea43b16-0fe7-4b01-b7ff-39308a2c2569
summary: ~/.kiss/tls ca.pem/ca-key.pem root CA and cert.pem/key.pem server cert (ECDSA
  P-256, SAN localhost/hostname/LAN IPs, 820 days), renewal, .tls.lock, hot reload,
  trust-store install and /ca.crt.
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Always-on TLS for kiss-web: local CA, server certificate and kiss-web --trust-ca

TLS is always on for the HTTPS/WSS port. `_create_ssl_context(certfile, keyfile, lan_ips)` loads an explicit cert/key pair when given one. Otherwise it uses the machine-local CA in `~/.kiss/tls/` (`_tls_dir()`, i.e. `$KISS_HOME/tls`).

## Files (`tls_certs.py`)
| File | What |
|---|---|
| `ca.pem` / `ca-key.pem` | Root CA generated once per machine (`generate_local_ca`), `CA_VALIDITY_DAYS = 3650`. It stays stable across server cert renewals, so trusting it once is enough. It has a Common Name prefixed with `"<product> Local CA"`, because iOS lists roots by CN. |
| `cert.pem` / `key.pem` | Server certificate signed by the CA (`issue_server_cert`). SAN = `localhost`, the hostname, loopback addresses and the current LAN IPs (`server_cert_names`). |

Both keys are ECDSA P-256. Generation takes milliseconds, so re-issuing on a LAN IP change never stalls the daemon, whereas RSA would hold the GIL noticeably. The server cert follows Apple's rules for locally trusted certs, which Safari/iOS enforce even with a trusted root: validity at most 825 days (the code uses `SERVER_CERT_VALIDITY_DAYS = 820`), a SAN, the `serverAuth` EKU, and a SHA-2 signature.

## Renewal
`ensure_local_tls_pair` / `server_cert_is_current` re-issue the server cert when it is missing, expiring within `RENEWAL_THRESHOLD_DAYS = 30`, not signed by the current CA, or missing a current LAN IP. Sibling daemons (tests, a respawn racing install.sh) serialize on `~/.kiss/tls/.tls.lock` (`_tls_lock_path`), taken with the bounded `_flock_with_deadline` and `_TLS_LOCK_TIMEOUT_S = 30.0`.

Every watchdog tick, `_refresh_tls_cert` refreshes the pair off-thread and hot-loads a changed cert into the live `SSLContext` on the loop thread. New handshakes get it, and existing connections are untouched. This matters in tunnel mode: an IP change does not restart the daemon there, and the new `https://<lan-ip>:PORT` would otherwise fail hostname verification. Explicit certfile/keyfile pairs are never rewritten.

## Trusting the CA
- `kiss-web --trust-ca` calls `tls_trust.trust_local_ca(<tls>/ca.pem)` and exits. It installs only the CA certificate (the key never leaves `~/.kiss/tls/`) into every store that exists for the user:
  - NSS databases via `certutil` (Chromium on Linux reads `~/.local/share/pki/nssdb` since M146, or the older `~/.pki/nssdb`; Firefox has one per profile): `install_into_nss`
  - the macOS login keychain via `security add-trusted-cert` (asks for the password): `install_into_macos_keychain`
  - the Windows current-user Root store via `certutil -addstore -user`: `install_into_windows_store`
  - the Linux system store: the command is only printed (`linux_system_store_hint`), because it needs root and only helps non-browser clients.
- Phones download the CA from `GET /ca.crt` (served as `kiss-sorcar-local-ca.crt`) and trust it manually. On iOS: Settings > General > About > Certificate Trust Settings.


## Sources
- `src/kiss/server/tls_certs.py` (module docstring, `generate_local_ca`, `issue_server_cert`, `server_cert_names`, `server_cert_is_current`, `ensure_local_tls_pair`, `CA_VALIDITY_DAYS`, `SERVER_CERT_VALIDITY_DAYS`, `RENEWAL_THRESHOLD_DAYS`)
- `src/kiss/server/tls_trust.py` (`trust_local_ca`, `install_into_nss`, `install_into_macos_keychain`, `install_into_windows_store`, `linux_system_store_hint`)
- `src/kiss/server/web_server.py` (`_create_ssl_context`, `_refresh_tls_cert`, `_tls_dir`, `_tls_lock_path`, `_TLS_LOCK_TIMEOUT_S`)

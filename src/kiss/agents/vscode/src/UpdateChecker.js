// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// On-activation PyPI update check for the KISS Sorcar VS Code extension.
//
// Why this exists
// ---------------
// Before this module, when VS Code launched and KISS Sorcar was **not**
// installing (the "fast path" early-exit in
// ``DependencyInstaller.ensureDependenciesImpl``), the extension did
// nothing to tell the user a newer release was available.  The kiss-web
// daemon polls PyPI hourly and broadcasts ``update_available`` to
// connected webview clients, but:
//
//   1. the daemon may already have been running for hours/days, so its
//      cached "no update" answer can be stale by up to one hour;
//   2. the fast path does not restart the daemon, so there is no fresh
//      check tied to activation;
//   3. users who launch VS Code only briefly never wait for the next
//      hourly poll.
//
// ``checkForExtensionUpdate`` actively queries PyPI on activation,
// compares the result with the locally installed ``__version__``
// shipped in ``kiss_project/src/kiss/core/_version.py``, and surfaces a
// chat-webview notification with a single "Update now" action when an
// upgrade is available.  A cached result at
// ``~/.kiss/.update-check.json`` rate-limits repeated checks so that
// the same VS Code session never hammers PyPI on every window reload.
//
// Authored in plain JS (rather than TS) for the same reason as
// ``reloadGuard.js``, ``daemonHealth.js`` and ``installerPath.js``:
// the ``test/`` harness runs under bare ``node`` without a TypeScript
// compile step, and we want the regression test to drive this code
// directly with no build dependency.
//
// All side effects (HTTP, fs, vscode) are injected through ``opts`` so
// the integration test can drive the real flow against a local HTTP
// stub of PyPI without touching the user's ``~/.kiss/`` or the
// network.

'use strict';

const fs = require('fs');
const http = require('http');
const https = require('https');
const os = require('os');
const path = require('path');
const {URL} = require('url');

const DEFAULT_PYPI_URL = 'https://pypi.org/pypi/kiss-agent-framework/json';
// 6 h between successful checks: matches the "at most a few times per
// day" cadence the daemon's hourly poll would give a long-running
// install, while still being frequent enough that a user who keeps
// VS Code open for the workday sees a same-day release.
const DEFAULT_COOLDOWN_MS = 6 * 60 * 60 * 1000;
const DEFAULT_FETCH_TIMEOUT_MS = 15_000;

/**
 * Parse a CalVer ``YYYY.M.P`` (or shorter) version string into a tuple
 * of integers.  Returns ``null`` for malformed input so callers can
 * treat it as "no information" rather than "newer than everything".
 */
function versionTuple(v) {
  if (typeof v !== 'string') return null;
  const parts = v.trim().split('.').filter(p => p !== '');
  const out = [];
  for (const p of parts) {
    if (!/^\d+$/.test(p)) return null;
    out.push(parseInt(p, 10));
  }
  return out.length > 0 ? out : null;
}

/**
 * Compare two version strings.  Returns ``1`` when *a* > *b*, ``-1``
 * when *a* < *b*, ``0`` otherwise (including malformed input — the
 * conservative answer so a garbled PyPI payload never falsely claims
 * an update).  Shorter tuples are right-padded with zeros so
 * ``"2026.6"`` and ``"2026.6.0"`` compare equal.  Mirrors the Python
 * ``_compare_versions`` helper in ``web_server.py`` so the extension
 * and daemon agree on update direction.
 */
function compareVersions(a, b) {
  const ta = versionTuple(a);
  const tb = versionTuple(b);
  if (!ta || !tb) return 0;
  const n = Math.max(ta.length, tb.length);
  while (ta.length < n) ta.push(0);
  while (tb.length < n) tb.push(0);
  for (let i = 0; i < n; i++) {
    if (ta[i] > tb[i]) return 1;
    if (ta[i] < tb[i]) return -1;
  }
  return 0;
}

/**
 * Read ``__version__`` from a ``_version.py`` file shipped with the
 * embedded kiss_project.  Returns ``null`` when the file cannot be
 * read or parsed — the caller treats this as "unknown", which skips
 * the update check rather than producing a false positive.
 */
function readVersionPy(versionPyPath) {
  try {
    const text = fs.readFileSync(versionPyPath, 'utf-8');
    const m = /__version__\s*=\s*["']([^"']+)["']/.exec(text);
    return m ? m[1] : null;
  } catch {
    return null;
  }
}

// Publisher.name prefix of the KISS Sorcar extension.  Kept in one
// place so the extension-dir scanner and ``install.sh`` /
// ``package.json`` stay in lock-step.
const EXTENSION_DIR_PREFIX = 'ksenxx.kiss-sorcar-';

/**
 * Scan ``extensionsRoot`` (default ``~/.vscode/extensions``) for every
 * installed KISS Sorcar extension directory and return the highest
 * ``__version__`` found.  Returns ``null`` when no such directory
 * exists or no directory contains a parseable ``_version.py``.
 *
 * Mirrors ``_scan_installed_extension_versions`` in ``web_server.py``
 * so the daemon and the extension side agree on which installed
 * version is authoritative — the fix for the sticky "update available"
 * toast that kept re-appearing after the user clicked "Update".  See
 * the comment above ``_INSTALLED_EXTENSIONS_ROOT`` in ``web_server.py``
 * for the full root-cause analysis.
 */
function scanInstalledExtensionVersions(extensionsRoot) {
  const root = extensionsRoot || path.join(os.homedir(), '.vscode', 'extensions');
  let entries;
  try {
    entries = fs.readdirSync(root, {withFileTypes: true});
  } catch {
    return [];
  }
  const versions = [];
  for (const e of entries) {
    try {
      if (!e.isDirectory()) continue;
    } catch {
      continue;
    }
    if (!e.name.startsWith(EXTENSION_DIR_PREFIX)) continue;
    // Canonical location since the version literal moved into
    // ``kiss.core``; the pre-move ``kiss/_version.py`` path is kept
    // as a fallback for extensions installed before the move.
    const kissDir = path.join(root, e.name, 'kiss_project', 'src', 'kiss');
    const v = readVersionPy(path.join(kissDir, 'core', '_version.py')) ||
        readVersionPy(path.join(kissDir, '_version.py'));
    if (v) versions.push(v);
  }
  return versions;
}

/**
 * Default current-version resolver.  Returns the newest
 * ``__version__`` string found in **any** installed KISS Sorcar
 * extension directory under ``extensionsRoot`` (default
 * ``~/.vscode/extensions``), falling back to the bundled
 * ``_version.py`` for developer / Docker installs where no such dir
 * exists.  ``kissProjectPath`` still points at a specific checkout
 * (used by the integration test and the dev-checkout path).
 */
function resolveCurrentVersion(kissProjectPath, extensionsRoot) {
  let best = null;
  let bestTuple = null;
  for (const v of scanInstalledExtensionVersions(extensionsRoot)) {
    const t = versionTuple(v);
    if (!t) continue;
    if (!bestTuple || compareVersions(v, best) > 0) {
      best = v;
      bestTuple = t;
    }
  }
  if (best) return best;
  if (kissProjectPath) {
    // Canonical location first (version literal lives in
    // ``kiss/core/_version.py``); legacy path kept as a fallback.
    const kissDir = path.join(kissProjectPath, 'src', 'kiss');
    const v = readVersionPy(path.join(kissDir, 'core', '_version.py')) ||
        readVersionPy(path.join(kissDir, '_version.py'));
    if (v) return v;
  }
  return null;
}

/**
 * Fetch a URL and resolve with the parsed JSON body.  Rejects with an
 * ``Error`` on non-2xx status, network failure, or timeout.  The
 * integration test injects an ``http://127.0.0.1:<port>/`` URL so the
 * helper must handle both ``http`` and ``https``.
 *
 * No shell is involved, redirects are NOT followed (PyPI's JSON
 * endpoint serves directly), and the timeout fires unconditionally so
 * a slow PyPI cannot wedge extension activation.
 */
function fetchJson(url, timeoutMs) {
  return new Promise((resolve, reject) => {
    let parsed;
    try {
      parsed = new URL(url);
    } catch (err) {
      reject(err);
      return;
    }
    const mod = parsed.protocol === 'http:' ? http : https;
    const req = mod.get(
      url,
      {timeout: timeoutMs, headers: {Accept: 'application/json'}},
      res => {
        const status = res.statusCode || 0;
        if (status < 200 || status >= 300) {
          res.resume();
          reject(new Error(`HTTP ${status} fetching ${url}`));
          return;
        }
        const chunks = [];
        res.on('data', c => chunks.push(c));
        res.on('end', () => {
          try {
            resolve(JSON.parse(Buffer.concat(chunks).toString('utf-8')));
          } catch (err) {
            reject(err);
          }
        });
        res.on('error', reject);
      },
    );
    req.on('error', reject);
    req.on('timeout', () => {
      req.destroy(new Error(`Timeout fetching ${url}`));
    });
  });
}

/**
 * Default PyPI fetcher.  Returns the ``info.version`` string from the
 * JSON document at ``url`` or ``null`` on any error (network, malformed
 * payload, missing key).  Errors are swallowed because an update check
 * failure must never break extension activation.
 */
async function defaultFetchLatest(url, timeoutMs) {
  try {
    const data = await fetchJson(url, timeoutMs);
    if (!data || typeof data !== 'object') return null;
    const info = data.info;
    if (!info || typeof info !== 'object') return null;
    const v = info.version;
    if (typeof v !== 'string' || !v.trim()) return null;
    return v.trim();
  } catch {
    return null;
  }
}

/**
 * Read the JSON cache file and return ``{lastCheckMs, lastLatest}``,
 * or ``null`` when the file is missing/unreadable/malformed.  A missing
 * cache is treated as "never checked" — the caller will perform a
 * check immediately, which is the right behavior for a fresh install.
 */
function readCache(cachePath) {
  try {
    const text = fs.readFileSync(cachePath, 'utf-8');
    const data = JSON.parse(text);
    if (!data || typeof data !== 'object') return null;
    const ts = typeof data.lastCheckMs === 'number' ? data.lastCheckMs : 0;
    const latest =
      typeof data.lastLatest === 'string' ? data.lastLatest : '';
    return {lastCheckMs: ts, lastLatest: latest};
  } catch {
    return null;
  }
}

/**
 * Write the JSON cache file atomically (via tmp + rename) so a crash
 * mid-write cannot leave a partially-written JSON document that
 * ``readCache`` would silently treat as "never checked".  Failures
 * are swallowed — the worst case is an extra PyPI hit on the next
 * activation, which is benign.
 */
function writeCache(cachePath, data) {
  try {
    fs.mkdirSync(path.dirname(cachePath), {recursive: true});
    const tmp = cachePath + '.tmp';
    fs.writeFileSync(tmp, JSON.stringify(data));
    fs.renameSync(tmp, cachePath);
  } catch {
    /* best-effort */
  }
}

/**
 * Check PyPI for a newer ``kiss-agent-framework`` release and, when
 * one is found, surface a chat-webview notification with an
 * "Update now" action.
 *
 * All side effects (network, fs, notification, clock) are injectable
 * via ``opts`` so the integration test can exercise the real flow
 * without touching the network, the user's ``~/.kiss/``, or the VS
 * Code API.
 *
 * Options
 * -------
 *   pypiUrl           PyPI JSON URL (default: kiss-agent-framework).
 *   cacheFilePath     JSON cache file (default: ~/.kiss/.update-check.json).
 *   cooldownMs        Min ms between successful checks (default: 6 h).
 *   fetchTimeoutMs    HTTP timeout (default: 15 s).
 *   currentVersion    Override local version string.
 *   kissProjectPath   Where to look for src/kiss/core/_version.py.
 *   fetchLatest       Override the PyPI fetcher (returns string|null).
 *   notify            Called with ({latest, current}) when an upgrade
 *                     is available; defaults to a no-op so the helper
 *                     is safe to call in non-VS-Code contexts.
 *   now               Clock override (returns ms since epoch).
 *
 * Returns ``{checked, notified, latest, current, reason}`` so callers
 * (and tests) can assert on the decision.
 */
async function checkForExtensionUpdate(opts) {
  const o = opts || {};
  const pypiUrl = o.pypiUrl || DEFAULT_PYPI_URL;
  const cachePath =
    o.cacheFilePath ||
    path.join(os.homedir(), '.kiss', '.update-check.json');
  const cooldownMs =
    typeof o.cooldownMs === 'number' ? o.cooldownMs : DEFAULT_COOLDOWN_MS;
  const fetchTimeoutMs =
    typeof o.fetchTimeoutMs === 'number'
      ? o.fetchTimeoutMs
      : DEFAULT_FETCH_TIMEOUT_MS;
  const now = typeof o.now === 'function' ? o.now : () => Date.now();
  const notify = typeof o.notify === 'function' ? o.notify : () => {};
  const fetchLatest =
    typeof o.fetchLatest === 'function'
      ? o.fetchLatest
      : url => defaultFetchLatest(url, fetchTimeoutMs);

  const current =
    o.currentVersion ||
    resolveCurrentVersion(o.kissProjectPath, o.extensionsRoot);
  if (!current) {
    return {
      checked: false,
      notified: false,
      latest: null,
      current: null,
      reason: 'unknown-current-version',
    };
  }

  const cached = readCache(cachePath);
  const nowMs = now();
  if (cached && nowMs - cached.lastCheckMs < cooldownMs) {
    // Within cooldown — replay the cached decision so the user is
    // still notified if the previous check found an upgrade and they
    // dismissed (or missed) the toast on a prior window.  This is
    // intentionally idempotent: the webview notification system
    // de-duplicates by id, so an identical toast just refreshes.
    if (compareVersions(cached.lastLatest, current) > 0) {
      notify({latest: cached.lastLatest, current});
      return {
        checked: false,
        notified: true,
        latest: cached.lastLatest,
        current,
        reason: 'cooldown-replay',
      };
    }
    return {
      checked: false,
      notified: false,
      latest: cached.lastLatest || null,
      current,
      reason: 'cooldown',
    };
  }

  const latest = await fetchLatest(pypiUrl);
  if (!latest) {
    // Fetcher returned null (network failure / malformed payload).
    // Do NOT poison the cache with an empty answer — leave the prior
    // cache (if any) intact so the next activation can retry promptly.
    return {
      checked: true,
      notified: false,
      latest: null,
      current,
      reason: 'fetch-failed',
    };
  }

  writeCache(cachePath, {lastCheckMs: nowMs, lastLatest: latest});

  if (compareVersions(latest, current) > 0) {
    notify({latest, current});
    return {
      checked: true,
      notified: true,
      latest,
      current,
      reason: 'update-available',
    };
  }
  return {
    checked: true,
    notified: false,
    latest,
    current,
    reason: 'up-to-date',
  };
}

module.exports = {
  checkForExtensionUpdate,
  compareVersions,
  versionTuple,
  readVersionPy,
  resolveCurrentVersion,
  scanInstalledExtensionVersions,
  defaultFetchLatest,
  DEFAULT_PYPI_URL,
  DEFAULT_COOLDOWN_MS,
  DEFAULT_FETCH_TIMEOUT_MS,
  EXTENSION_DIR_PREFIX,
};

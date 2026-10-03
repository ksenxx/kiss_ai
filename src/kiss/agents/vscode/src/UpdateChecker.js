// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const fs = require('fs');
const http = require('http');
const https = require('https');
const os = require('os');
const path = require('path');
const {URL} = require('url');
// $KISS_HOME, else the brand's ~/<home_dir>: the cooldown cache must live
// under the SAME directory every other extension path resolves.
const {kissHomeDir} = require('./kissHome');

const DEFAULT_PYPI_URL = 'https://pypi.org/pypi/kiss-agent-framework/json';
const DEFAULT_COOLDOWN_MS = 6 * 60 * 60 * 1000;
const DEFAULT_SNOOZE_MS = 24 * 60 * 60 * 1000;
const DEFAULT_FETCH_TIMEOUT_MS = 15_000;

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

function readVersionPy(versionPyPath) {
  try {
    const text = fs.readFileSync(versionPyPath, 'utf-8');
    const m = /__version__\s*=\s*["']([^"']+)["']/.exec(text);
    return m ? m[1] : null;
  } catch {
    return null;
  }
}

const EXTENSION_DIR_PREFIX = 'ksenxx.kiss-sorcar-';

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
    const kissDir = path.join(root, e.name, 'kiss_project', 'src', 'kiss');
    const v = readVersionPy(path.join(kissDir, 'core', '_version.py')) ||
        readVersionPy(path.join(kissDir, '_version.py'));
    if (v) versions.push(v);
  }
  return versions;
}

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
    const kissDir = path.join(kissProjectPath, 'src', 'kiss');
    const v = readVersionPy(path.join(kissDir, 'core', '_version.py')) ||
        readVersionPy(path.join(kissDir, '_version.py'));
    if (v) return v;
  }
  return null;
}

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

function readCache(cachePath) {
  try {
    const text = fs.readFileSync(cachePath, 'utf-8');
    const data = JSON.parse(text);
    if (!data || typeof data !== 'object') return null;
    const ts = typeof data.lastCheckMs === 'number' ? data.lastCheckMs : 0;
    const latest =
      typeof data.lastLatest === 'string' ? data.lastLatest : '';
    const snoozeUntilMs =
      typeof data.snoozeUntilMs === 'number' ? data.snoozeUntilMs : 0;
    const snoozedLatest =
      typeof data.snoozedLatest === 'string' ? data.snoozedLatest : '';
    const skippedVersion =
      typeof data.skippedVersion === 'string' ? data.skippedVersion : '';
    return {
      lastCheckMs: ts,
      lastLatest: latest,
      snoozeUntilMs,
      snoozedLatest,
      skippedVersion,
    };
  } catch {
    return null;
  }
}

// "Remind me later" state: the notification stays suppressed while the
// snooze window is open UNLESS a release NEWER than the snoozed one
// appears (compareVersions returns 0 for an unparsable snoozedLatest,
// so a snooze whose version was lost still suppresses everything until
// it expires).
function isSnoozeActive(cached, candidateLatest, nowMs) {
  if (!cached || !cached.snoozeUntilMs) return false;
  if (nowMs >= cached.snoozeUntilMs) return false;
  return compareVersions(candidateLatest, cached.snoozedLatest) <= 0;
}

// "Skip this version" state: releases up to and including the skipped
// one are never announced again; a newer release still notifies.
function isSkipped(cached, candidateLatest) {
  if (!cached || !cached.skippedVersion) return false;
  return compareVersions(candidateLatest, cached.skippedVersion) <= 0;
}

// Whether the popup for candidateLatest is suppressed by either a
// running snooze or a permanent skip.
function isSuppressed(cached, candidateLatest, nowMs) {
  return (
    isSnoozeActive(cached, candidateLatest, nowMs) ||
    isSkipped(cached, candidateLatest)
  );
}

// audit0902-coverage:start
function writeCache(cachePath, data) {
  // Every window writes this file at activation.  The temp name must be
  // unique per writer (pid + timestamp temp file, then rename): with one
  // shared `<cache>.tmp`, a second window truncates the first one's temp
  // file before it is renamed and the cache ends up empty -- unparsable,
  // so the cooldown is lost.
  const tmp = `${cachePath}.${process.pid}.${Date.now()}.tmp`;
  try {
    fs.mkdirSync(path.dirname(cachePath), {recursive: true});
    fs.writeFileSync(tmp, JSON.stringify(data));
    fs.renameSync(tmp, cachePath);
  } catch {
    try {
      fs.unlinkSync(tmp);
    } catch {
    }
  }
}
// audit0902-coverage:end

async function checkForExtensionUpdate(opts) {
  const o = opts || {};
  const pypiUrl = o.pypiUrl || DEFAULT_PYPI_URL;
  const cachePath =
    o.cacheFilePath || path.join(kissHomeDir(), '.update-check.json');
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

  // The `kissSorcar.checkForUpdates` setting: the caller reads the
  // configuration and passes `enabled: false` to turn the check off
  // entirely (no network request, no cache write, no popup).
  if (o.enabled === false) {
    return {
      checked: false,
      notified: false,
      latest: null,
      current: null,
      reason: 'disabled',
    };
  }

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
    if (compareVersions(cached.lastLatest, current) > 0) {
      if (isSuppressed(cached, cached.lastLatest, nowMs)) {
        return {
          checked: false,
          notified: false,
          latest: cached.lastLatest,
          current,
          reason: isSkipped(cached, cached.lastLatest) ? 'skipped' : 'snoozed',
        };
      }
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
    return {
      checked: true,
      notified: false,
      latest: null,
      current,
      reason: 'fetch-failed',
    };
  }

  // Re-read the cache: the network fetch takes seconds, and a sibling
  // window may have recorded a "Remind me later" snooze meanwhile.
  // Rewriting from the pre-fetch snapshot would erase that snooze and
  // re-notify the user who just dismissed the popup.
  const fresh = readCache(cachePath) || cached;

  // Preserve a still-running snooze across the cache refresh: writeCache
  // replaces the whole file, and dropping the snooze fields here would
  // resurface the popup as soon as the 6h fetch cooldown lapsed.
  const nextCache = {lastCheckMs: nowMs, lastLatest: latest};
  if (fresh && fresh.snoozeUntilMs > nowMs) {
    nextCache.snoozeUntilMs = fresh.snoozeUntilMs;
    nextCache.snoozedLatest = fresh.snoozedLatest;
  }
  if (fresh && fresh.skippedVersion) {
    nextCache.skippedVersion = fresh.skippedVersion;
  }
  writeCache(cachePath, nextCache);

  if (compareVersions(latest, current) > 0) {
    if (isSuppressed(fresh, latest, nowMs)) {
      return {
        checked: true,
        notified: false,
        latest,
        current,
        reason: isSkipped(fresh, latest) ? 'skipped' : 'snoozed',
      };
    }
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

// Records a "Remind me later" click: suppresses the update popup for
// snoozeMs (default 24h) for releases up to and including `latest`.
// A release newer than `latest` published inside the window still
// notifies.  Existing cooldown fields in the cache are preserved.
function snoozeUpdateNotification(opts) {
  const o = opts || {};
  const cachePath =
    o.cacheFilePath || path.join(kissHomeDir(), '.update-check.json');
  const snoozeMs = typeof o.snoozeMs === 'number' ? o.snoozeMs : DEFAULT_SNOOZE_MS;
  const now = typeof o.now === 'function' ? o.now : () => Date.now();
  const nowMs = now();
  const cached = readCache(cachePath) || {lastCheckMs: 0, lastLatest: ''};
  const snoozedLatest =
    typeof o.latest === 'string' && o.latest ? o.latest : cached.lastLatest;
  const snoozeUntilMs = nowMs + snoozeMs;
  const next = {
    lastCheckMs: cached.lastCheckMs,
    lastLatest: cached.lastLatest,
    snoozeUntilMs,
    snoozedLatest,
  };
  if (cached.skippedVersion) next.skippedVersion = cached.skippedVersion;
  writeCache(cachePath, next);
  return {snoozeUntilMs, snoozedLatest};
}

// Records a "Skip this version" click: `latest` (default: the last
// release seen) and every older release are never announced again.
// A newer release still notifies.  Cooldown and snooze fields in the
// cache are preserved.
function skipUpdateVersion(opts) {
  const o = opts || {};
  const cachePath =
    o.cacheFilePath || path.join(kissHomeDir(), '.update-check.json');
  const cached = readCache(cachePath) || {lastCheckMs: 0, lastLatest: ''};
  const skippedVersion =
    typeof o.latest === 'string' && o.latest ? o.latest : cached.lastLatest;
  const next = {
    lastCheckMs: cached.lastCheckMs,
    lastLatest: cached.lastLatest,
    skippedVersion,
  };
  if (cached.snoozeUntilMs) {
    next.snoozeUntilMs = cached.snoozeUntilMs;
    next.snoozedLatest = cached.snoozedLatest;
  }
  writeCache(cachePath, next);
  return {skippedVersion};
}

module.exports = {
  checkForExtensionUpdate,
  compareVersions,
  readVersionPy,
  resolveCurrentVersion,
  scanInstalledExtensionVersions,
  skipUpdateVersion,
  snoozeUpdateNotification,
};

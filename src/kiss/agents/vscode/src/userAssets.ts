// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

export function kissHomeDir(): string {
  return process.env.KISS_HOME || path.join(os.homedir(), '.kiss');
}

/**
 * The kiss-web daemon's local-endpoint file.
 *
 * The daemon writes its WSS URL, per-start local token and CA path
 * there once its listener is bound (`kiss.agents.sorcar.local_endpoint`),
 * and removes it on shutdown, so the file doubles as the daemon's
 * presence marker.  Must mirror the daemon's own resolution (it writes
 * under $KISS_HOME): every extension-host probe, kill and startup-poll
 * of the daemon has to look at the SAME file, or a window with
 * KISS_HOME set kills a healthy daemon and then polls a path it never
 * writes.
 */
export function sorcarEndpointPath(): string {
  return (
    process.env.KISS_SORCAR_LOCAL ||
    path.join(kissHomeDir(), 'sorcar-local.json')
  );
}

/** What a local client needs to reach and authenticate to the daemon. */
export interface LocalEndpoint {
  /** `wss://127.0.0.1:8787/ws` (or `ws://...` from a test daemon). */
  url: string;
  /** The daemon's per-start local token, sent in the `auth` frame. */
  token: string;
  /** PEM file to trust for TLS, or null for the system store. */
  ca: string | null;
  /** The daemon's pid (0 when unknown). */
  pid: number;
}

/**
 * Read the endpoint file at `endpointPath` (default:
 * {@link sorcarEndpointPath}).  Returns null when the file is missing,
 * unreadable, half-written or not an endpoint record.
 */
export function readLocalEndpoint(endpointPath?: string): LocalEndpoint | null {
  const file = endpointPath ?? sorcarEndpointPath();
  let data: unknown;
  try {
    data = JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch {
    return null;
  }
  if (!data || typeof data !== 'object') return null;
  const rec = data as Record<string, unknown>;
  if (typeof rec.url !== 'string' || !rec.url) return null;
  if (typeof rec.token !== 'string' || !rec.token) return null;
  if (rec.ca !== null && rec.ca !== undefined && typeof rec.ca !== 'string') {
    return null;
  }
  return {
    url: rec.url,
    token: rec.token,
    ca: typeof rec.ca === 'string' ? rec.ca : null,
    pid: typeof rec.pid === 'number' ? rec.pid : 0,
  };
}

/**
 * Return `~/.kiss/<name>`, seeding it with `defaultContent` if absent.
 *
 * Mirrors the daemon's `user_assets.ensure_user_asset_from_default`: the
 * seed is atomic and non-clobbering.  The default is staged in a sibling
 * temp file and hard-linked into place, so a concurrent reader (the
 * daemon's autocomplete worker reads MY_INJECTION.md on every keystroke)
 * never observes an empty or partially-written file, and a concurrent
 * seeder (the daemon on a fresh install) cannot truncate this one's
 * payload: whoever links first wins, the loser gets EEXIST and returns
 * the winner's path.  Returns null when `~/.kiss/` is not writable so
 * callers can skip silently.
 */
export function ensureUserAssetFromDefault(
  name: string,
  defaultContent: string,
): string | null {
  const userPath = path.join(kissHomeDir(), name);
  // audit0902-coverage:start
  try {
    if (fs.existsSync(userPath)) return userPath;
    const dir = path.dirname(userPath);
    fs.mkdirSync(dir, {recursive: true});
    const tmp = path.join(dir, `.${name}-${process.pid}-${Date.now()}.tmp`);
    try {
      fs.writeFileSync(tmp, defaultContent);
      fs.linkSync(tmp, userPath);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') throw err;
    } finally {
      fs.rmSync(tmp, {force: true});
    }
    return userPath;
  } catch {
    return null;
  }
  // audit0902-coverage:end
}

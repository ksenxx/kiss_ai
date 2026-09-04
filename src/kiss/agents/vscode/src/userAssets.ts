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
 * The Unix socket the kiss-web daemon listens on.
 *
 * Must mirror the daemon's own resolution (web_server.py binds under
 * $KISS_HOME): every extension-host probe, kill and startup-poll of the
 * daemon has to look at the SAME socket, or a window with KISS_HOME set
 * kills a healthy daemon and then polls a path it never binds.
 */
export function sorcarSockPath(): string {
  return (
    process.env.KISS_SORCAR_SOCK || path.join(kissHomeDir(), 'sorcar.sock')
  );
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

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

export function ensureUserAssetFromDefault(
  name: string,
  defaultContent: string,
): string | null {
  const userPath = path.join(kissHomeDir(), name);
  try {
    if (fs.existsSync(userPath)) return userPath;
    fs.mkdirSync(path.dirname(userPath), {recursive: true});
    fs.writeFileSync(userPath, defaultContent);
    return userPath;
  } catch {
    return null;
  }
}

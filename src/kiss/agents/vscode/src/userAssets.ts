// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.
 *
 * TypeScript counterpart to ``kiss/agents/vscode/user_assets.py``.
 * Both keep ``INJECTIONS.md`` and ``SAMPLE_TASKS.md`` consistent
 * across the kiss-web daemon (Python) and the VS Code extension
 * webview (TypeScript): ``~/.kiss/<name>`` is the runtime source of
 * truth, and the bundled package copy is the seed used the very
 * first time a user opens the welcome screen / Tricks button on a
 * fresh machine.
 *
 * Once the user copy exists, **user edits win unconditionally** —
 * ``ensureUserAsset`` never refreshes a present user copy from the
 * package copy, even after the package copy is bumped by ``git pull``
 * or ``Update``.  To pull in a new bundled default, the user removes
 * ``~/.kiss/<name>`` and reads again.  Matches the Python helper.
 *
 * ``install.sh`` performs an eager ``cp -n`` seed so a fresh install
 * is immediately reflected without waiting for the first read; this
 * helper covers the cases ``install.sh`` does not (tests, dev
 * checkouts, sandboxed envs).
 *
 * Honours the ``KISS_HOME`` env var, matching ``persistence.py``,
 * ``web_server.py``, and ``vscode_config.py``.
 */

import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

/** Return ``~/.kiss/`` (or ``$KISS_HOME`` when set). */
export function kissHomeDir(): string {
  return process.env.KISS_HOME || path.join(os.homedir(), '.kiss');
}

/**
 * Return the path the runtime should read ``name`` from.
 *
 * Prefers ``~/.kiss/<name>``; seeds it from ``packagePath`` only when
 * the user copy is missing.  Once seeded, user edits survive every
 * read — the package copy is never silently restored.  Falls back to
 * ``packagePath`` when ``~/.kiss/`` is not writable.
 *
 * @param name Asset filename (e.g. ``"INJECTIONS.md"``).
 * @param packagePath Path to the bundled package copy used as the
 *   seed and the read-only fallback.
 * @returns Path the caller should read.
 */
export function ensureUserAsset(name: string, packagePath: string): string {
  const userPath = path.join(kissHomeDir(), name);
  try {
    if (fs.existsSync(userPath)) {
      // User copy wins unconditionally; user edits are never silently
      // clobbered by a newer package copy.  To pull in a fresh
      // bundled copy the user removes ``~/.kiss/<name>``.
      return userPath;
    }
    if (fs.existsSync(packagePath)) {
      fs.mkdirSync(path.dirname(userPath), {recursive: true});
      fs.copyFileSync(packagePath, userPath);
      return userPath;
    }
  } catch {
    /* fall through to package path */
  }
  return packagePath;
}

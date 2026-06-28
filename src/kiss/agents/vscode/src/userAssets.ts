// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Resolve and lazily seed ``~/.kiss/<asset>`` markdown assets.
 *
 * TypeScript counterpart to ``kiss/agents/vscode/user_assets.py``.
 *
 * Two helpers:
 *
 * * ``ensureUserAsset(name, packagePath)`` — seeds a user copy from a
 *   bundled package file.  Generic helper retained in the public API
 *   for future package-seeded assets; no production caller currently
 *   uses it (``INJECTIONS.md`` is now read directly from the bundled
 *   package, never copied into ``~/.kiss/`` — see ``getTricks`` in
 *   ``SorcarTab.ts`` and ``kiss.agents.vscode.tricks.read_tricks``).
 *   At runtime user edits survive every read.
 *
 * * ``ensureUserAssetFromDefault(name, defaultContent)`` — seeds a user
 *   copy from an inline string default, with no package file
 *   involved.  Used for ``MY_TASK_TEMPLATES.md`` (welcome-screen chips)
 *   and ``MY_INJECTION.md`` (Inject instruction panel), both purely
 *   user-curated files whose only "bundled" content is a tiny inline
 *   starter (``## Task\n\nHi!\n`` and a ``## Trick`` test-first
 *   starter, respectively).  Returns ``null`` when ``~/.kiss/`` is not
 *   writable so the caller can skip silently.
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

/**
 * Return ``~/.kiss/<name>``, seeding it with ``defaultContent`` when
 * absent.  Used for assets like ``MY_TASK_TEMPLATES.md`` whose source
 * of truth is the user's local copy — there is no bundled package
 * file, only a tiny inline default written on first read.  Returns
 * ``null`` when ``~/.kiss/`` is not writable (read-only FS, missing
 * HOME) so the caller can skip silently.
 *
 * @param name Asset filename (e.g. ``"MY_TASK_TEMPLATES.md"``).
 * @param defaultContent UTF-8 string written to the user copy on
 *   first read.  Never overwrites an existing file.
 * @returns Path to the user copy, or ``null`` when ``~/.kiss/`` cannot
 *   be written.
 */
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

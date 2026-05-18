/**
 * Shared filesystem helpers for locating the embedded ``kiss_project``
 * directory and the ``uv`` binary.
 *
 * These used to live inside ``AgentProcess.ts`` because the extension
 * spawned ``uv run python -m kiss.agents.vscode.server`` directly.  Now
 * that the extension talks to the kiss-web daemon over a UDS socket
 * (see ``AgentClient.ts``) the only consumers of these helpers are
 * ``DependencyInstaller`` (which still installs / restarts the daemon)
 * and ``SorcarTab.ts`` (which loads bundled HTML).  Lifted into a
 * dedicated module so ``AgentProcess.ts`` can be deleted.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {execSync} from 'child_process';

function isValidKissProject(dir: string): boolean {
  try {
    const pyproject = path.join(dir, 'pyproject.toml');
    if (!fs.existsSync(pyproject)) return false;
    const content = fs.readFileSync(pyproject, 'utf-8');
    return content.includes('name = "kiss') || content.includes("name = 'kiss");
  } catch {
    return false;
  }
}

/**
 * Find the KISS project root directory.
 *
 * Direct VSIX install is the only supported installation model, so the
 * embedded ``kiss_project`` bundled inside the VSIX is the default —
 * even when the user originally installed from a cloned checkout via
 * the top-level ``install.sh``.  The env-var / setting escape hatches
 * remain for developers who want the daemon to run against a working
 * checkout, but they are only honoured in a trusted workspace.
 *
 * Search order:
 * 1. ``KISS_PROJECT_PATH`` environment variable (explicit override,
 *    e.g. Docker containers).  Trusted workspaces only.
 * 2. ``kissSorcar.kissProjectPath`` configuration setting.  Trusted
 *    workspaces only.
 * 3. Embedded ``kiss_project`` directory bundled with the extension.
 */
export function findKissProject(): string | null {
  // H5 — only honour explicit workspace-scoped overrides (env var or
  // setting) inside a *trusted* workspace.  A malicious workspace's
  // .vscode/settings.json must not be able to redirect the daemon
  // installer at attacker-controlled code, since the daemon later runs
  // arbitrary shell commands on user request.
  const isTrusted = vscode.workspace.isTrusted;

  if (isTrusted) {
    const envPath = process.env.KISS_PROJECT_PATH;
    if (envPath && isValidKissProject(envPath)) return envPath;

    const configPath = vscode.workspace
      .getConfiguration('kissSorcar')
      .get<string>('kissProjectPath');
    if (configPath && isValidKissProject(configPath)) return configPath;
  }

  const embeddedPath = path.join(__dirname, '..', 'kiss_project');
  if (isValidKissProject(embeddedPath)) return embeddedPath;

  return null;
}

/**
 * Find the uv binary path, or null if not installed anywhere.
 */
export function findUvPath(): string | null {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const suffix = process.platform === 'win32' ? '.exe' : '';
  const candidates = [
    path.join(homeDir, '.local', 'bin', `uv${suffix}`),
    path.join(homeDir, '.cargo', 'bin', `uv${suffix}`),
  ];
  if (process.platform !== 'win32') {
    candidates.push('/usr/local/bin/uv', '/opt/homebrew/bin/uv');
  }
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      continue;
    }
  }
  try {
    execSync(process.platform === 'win32' ? 'where uv' : 'which uv', {
      stdio: 'ignore',
    });
    return 'uv';
  } catch {
    return null;
  }
}


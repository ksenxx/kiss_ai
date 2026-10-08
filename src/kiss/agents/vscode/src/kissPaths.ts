// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {execFileSync} from 'child_process';

// Synchronous PATH probes run on the extension host's event loop, so
// they must never wait on a hung child (e.g. a PATH entry on a stalled
// network mount).
const WHICH_TIMEOUT_MS = 5_000;

/**
 * Whether *cmd* resolves on PATH (`which` / `where`).  Synchronous, so
 * bounded: execFileSync (no shell layer to orphan the probe) with a
 * timeout and SIGKILL, since a probe stalled on a dead network mount
 * can ignore the default SIGTERM and block past the timeout.
 */
export function commandExists(cmd: string): boolean {
  try {
    execFileSync(process.platform === 'win32' ? 'where' : 'which', [cmd], {
      stdio: 'ignore',
      timeout: WHICH_TIMEOUT_MS,
      killSignal: 'SIGKILL',
    });
    return true;
  } catch {
    return false;
  }
}

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

export function findKissProject(): string | null {
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
  return commandExists('uv') ? 'uv' : null;
}

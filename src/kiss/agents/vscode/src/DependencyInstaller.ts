/**
 * Auto-installation of binary dependencies for the KISS Sorcar extension.
 * Ensures uv, git, Python environment, and Playwright Chromium are available.
 * Called during extension activation.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { exec, execSync, spawn } from 'child_process';
import { findKissProject, findUvPath } from './AgentProcess';

/**
 * Ensure all required dependencies are installed.
 * Shows a progress notification during first-time installation.
 * Safe to call multiple times — skips already-installed dependencies.
 */
export async function ensureDependencies(): Promise<void> {
  const kissProjectPath = findKissProject();
  if (!kissProjectPath) {
    return; // Can't set up without a project path; AgentProcess.start() will show error
  }

  let uvPath = findUvPath();
  const venvExists = fs.existsSync(path.join(kissProjectPath, '.venv'));

  // Fast path: everything looks ready, ensure playwright in background
  if (uvPath && venvExists) {
    spawnBackground(uvPath, ['run', 'python', '-m', 'playwright', 'install', 'chromium'], kissProjectPath);
    return;
  }

  // Slow path: show progress notification and install missing deps
  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'KISS Sorcar', cancellable: false },
    async (progress) => {
      // 1. Install uv if needed
      if (!uvPath) {
        progress.report({ message: 'Installing uv package manager...' });
        uvPath = await installUv();
        if (!uvPath) {
          vscode.window.showErrorMessage(
            'KISS Sorcar: Failed to install uv. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh'
          );
          return;
        }
      }

      // 2. Warn about git
      if (!commandExists('git')) {
        vscode.window.showWarningMessage(
          'KISS Sorcar: git is required but not found. Please install git.'
        );
      }

      // 3. Set up Python environment (installs Python 3.13+ and all pip dependencies)
      if (!venvExists) {
        progress.report({ message: 'Setting up Python environment (first time, may take a minute)...' });
        await runAsync(uvPath, ['sync'], kissProjectPath);
      }

      // 4. Install Playwright Chromium
      progress.report({ message: 'Installing Chromium browser...' });
      await runAsync(
        uvPath,
        ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
        kissProjectPath
      );
    }
  );
}

/**
 * Install uv and return its path, or null on failure.
 */
async function installUv(): Promise<string | null> {
  const cmd =
    process.platform === 'win32'
      ? 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
      : 'curl -LsSf https://astral.sh/uv/install.sh | sh';
  try {
    await execPromise(cmd);
    return findUvPath();
  } catch (err) {
    console.error('[KISS Sorcar] Failed to install uv:', err);
    return null;
  }
}

/**
 * Check if a command is available on the system.
 */
function commandExists(cmd: string): boolean {
  try {
    execSync(process.platform === 'win32' ? `where ${cmd}` : `which ${cmd}`, {
      stdio: 'ignore',
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Run a command with args and return a promise that resolves on exit code 0.
 */
function runAsync(cmd: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd,
      stdio: 'pipe',
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    let stderr = '';
    proc.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on('close', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${cmd} ${args.join(' ')} failed (code ${code}): ${stderr}`));
    });
    proc.on('error', reject);
  });
}

/**
 * Spawn a command in the background (fire-and-forget).
 */
function spawnBackground(cmd: string, args: string[], cwd: string): void {
  const proc = spawn(cmd, args, {
    cwd,
    stdio: 'ignore',
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  });
  proc.on('error', (err) => {
    console.error(`[KISS Sorcar] Background command failed: ${cmd} ${args.join(' ')}:`, err);
  });
  proc.unref();
}

/**
 * Execute a shell command and return stdout.
 */
function execPromise(cmd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(cmd, { timeout: 300_000 }, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}

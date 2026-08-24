// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as path from 'path';
import * as os from 'os';
import * as fs from 'fs';
import * as https from 'https';
import * as crypto from 'crypto';
import {exec, execSync, execFileSync, spawn} from 'child_process';
import {findKissProject, findUvPath} from './kissPaths';
import {
  probeDaemonHealth,
  daemonHasActiveTasks,
  decideRestart,
} from './daemonHealth';
import {verifyDaemonStartup} from './daemonRestartVerify';
import {restartLaunchAgent} from './macLaunchd';
import {kissHomeDir, sorcarSockPath} from './userAssets';
import {
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
  withWebviewNotificationProgress,
} from './WebviewNotifications';

const HOME_DIR = process.env.HOME || process.env.USERPROFILE || '';
// The daemon resolves its state directory from $KISS_HOME (see
// kiss/core/config.py), so everything the extension shares with it —
// sockets, config.json, markers, logs — must live under the same root.
const LOG_DIR = kissHomeDir();
const LOG_FILE = path.join(LOG_DIR, 'install.log');
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 13;
const UV_VERSION = '0.11.2';
const NODE_VERSION = 'v22.16.0';

function xmlEscape(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

function unitEscape(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/%/g, '%%');
}

export function downloadFile(
  url: string,
  destPath: string,
  maxRedirects = 5,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const get = (u: string, hops: number): void => {
      const req = https.get(u, {timeout: 60000}, res => {
        const status = res.statusCode || 0;
        if (status >= 300 && status < 400 && res.headers.location) {
          if (hops <= 0) {
            res.resume();
            reject(new Error(`Too many redirects from ${url}`));
            return;
          }
          res.resume();
          const next = new URL(res.headers.location, u).toString();
          get(next, hops - 1);
          return;
        }
        if (status !== 200) {
          res.resume();
          reject(new Error(`HTTP ${status} fetching ${u}`));
          return;
        }
        const out = fs.createWriteStream(destPath);
        const fail = (err: Error): void => {
          out.destroy();
          fs.unlink(destPath, () => {
            reject(err);
          });
        };
        res.on('error', fail);
        res.on('aborted', () =>
          fail(new Error(`Connection aborted downloading ${u}`)),
        );
        res.pipe(out);
        out.on('finish', () =>
          out.close(err => (err ? reject(err) : resolve())),
        );
        out.on('error', fail);
      });
      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy(new Error(`Timeout downloading ${u}`));
      });
    };
    get(url, maxRedirects);
  });
}

function sha256OfFile(filePath: string): string {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

function verifyDownloadHash(
  filePath: string,
  expectedHashHex: string | null,
): void {
  const got = sha256OfFile(filePath);
  if (!expectedHashHex) {
    log(
      `No SHA256 expectation for ${path.basename(filePath)}; ` +
        `computed hash = ${got}`,
    );
    return;
  }
  if (got.toLowerCase() !== expectedHashHex.toLowerCase()) {
    try {
      fs.unlinkSync(filePath);
    } catch {}
    throw new Error(
      `SHA256 mismatch for ${path.basename(filePath)}: ` +
        `expected ${expectedHashHex}, got ${got}`,
    );
  }
  log(`SHA256 ok for ${path.basename(filePath)}`);
}

/**
 * GET *url* over HTTPS and return the response body as UTF-8 text.
 *
 * Returns null on any failure (non-200 status, network error, abort,
 * 15s timeout, or a malformed/non-HTTPS URL).  Redirects are not
 * followed.  Shared transport for the SHA-256 manifest fetchers
 * below, which previously duplicated this boilerplate.
 */
export function httpsGetText(url: string): Promise<string | null> {
  return new Promise(resolve => {
    let req: ReturnType<typeof https.get>;
    try {
      req = https.get(url, {timeout: 15000}, res => {
        if ((res.statusCode || 0) !== 200) {
          res.resume();
          resolve(null);
          return;
        }
        const chunks: Buffer[] = [];
        res.on('data', d => chunks.push(d));
        res.on('end', () => resolve(Buffer.concat(chunks).toString('utf-8')));
        res.on('error', () => resolve(null));
        res.on('aborted', () => resolve(null));
      });
    } catch {
      // e.g. malformed URL or non-HTTPS protocol throws synchronously.
      resolve(null);
      return;
    }
    req.on('error', () => resolve(null));
    req.on('timeout', () => {
      req.destroy();
      resolve(null);
    });
  });
}

/**
 * Fetch the uv-style `<assetUrl>.sha256` manifest and return the
 * leading 64-hex-digit digest, or null when unavailable.
 */
export async function fetchUvStyleSha256(
  assetUrl: string,
): Promise<string | null> {
  const text = await httpsGetText(assetUrl + '.sha256');
  if (text === null) return null;
  const m = /^([0-9a-fA-F]{64})/.exec(text.trim());
  return m ? m[1] : null;
}

/**
 * Look up *assetName*'s digest in the Node.js SHASUMS256.txt manifest
 * for NODE_VERSION, or null when unavailable or unlisted.
 */
export async function fetchNodeSha256(
  assetName: string,
  manifestUrl?: string,
): Promise<string | null> {
  const url =
    manifestUrl || `https://nodejs.org/dist/${NODE_VERSION}/SHASUMS256.txt`;
  const text = await httpsGetText(url);
  if (text === null) return null;
  for (const line of text.split('\n')) {
    const m = /^([0-9a-fA-F]{64})\s+(.+?)\s*$/.exec(line);
    if (m && m[2] === assetName) return m[1];
  }
  return null;
}

function sleepSync(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

function spawnCollect(
  cmd: string,
  args: string[],
  opts: {cwd?: string; env?: NodeJS.ProcessEnv; timeoutMs?: number},
): Promise<{code: number | null; stdout: string; stderr: string}> {
  return new Promise((resolve, reject) => {
    const proc = spawn(cmd, args, {
      cwd: opts.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: opts.env,
    });
    let stdout = '';
    let stderr = '';
    const timer = opts.timeoutMs
      ? setTimeout(() => {
          proc.kill('SIGKILL');
          reject(
            new Error(
              `${cmd} ${args.join(' ')} timed out after ${opts.timeoutMs}ms`,
            ),
          );
        }, opts.timeoutMs)
      : undefined;
    proc.stdout?.on('data', (d: Buffer) => {
      stdout += d.toString();
    });
    proc.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on('close', code => {
      if (timer) clearTimeout(timer);
      resolve({code, stdout, stderr});
    });
    proc.on('error', err => {
      if (timer) clearTimeout(timer);
      reject(err);
    });
  });
}

async function spawnPromise(
  cmd: string,
  args: string[],
  cwd?: string,
  timeoutMs = 300_000,
): Promise<string> {
  const r = await spawnCollect(cmd, args, {cwd, timeoutMs});
  if (r.code === 0) return r.stdout.trim();
  throw new Error(
    `${cmd} ${args.join(' ')} exited ${r.code}: ${r.stderr.trim()}`,
  );
}

let pendingDeps: Promise<void> | null = null;

function log(message: string): void {
  const line = `[${new Date().toISOString()}] ${message}`;
  console.log('[KISS Sorcar]', message);
  try {
    fs.mkdirSync(LOG_DIR, {recursive: true});
    fs.appendFileSync(LOG_FILE, line + '\n');
  } catch {}
}

function prependToProcessPath(dir: string): void {
  const parts = (process.env.PATH || '').split(path.delimiter);
  if (!parts.includes(dir)) {
    process.env.PATH = `${dir}${path.delimiter}${process.env.PATH || ''}`;
  }
}

export function ensureLocalBinInPath(): void {
  if (!HOME_DIR) return;
  prependToProcessPath(path.join(HOME_DIR, '.local', 'bin'));
}

function windowsZipInstall(
  url: string,
  zipPath: string,
  destDir: string,
  extraPsCommands = '',
): Promise<string> {
  return execPromise(
    'powershell -Command "' +
      `Invoke-WebRequest -Uri '${url}' -OutFile '${zipPath}'; ` +
      `Expand-Archive -Force -Path '${zipPath}' -DestinationPath '${destDir}'; ` +
      extraPsCommands +
      `Remove-Item -Force '${zipPath}'"`,
  );
}

function findNodeDirWindows(baseDir: string): string {
  try {
    for (const entry of fs.readdirSync(baseDir)) {
      const candidate = path.join(baseDir, entry);
      if (fs.existsSync(path.join(candidate, 'node.exe'))) return candidate;
    }
  } catch {}
  return baseDir;
}

export function getFallbackDefaultModel(): string {
  const env = process.env;
  if (env.ANTHROPIC_API_KEY) return 'claude-opus-4-7';
  if (env.OPENAI_API_KEY) return 'gpt-5.6-luna';
  if (env.GEMINI_API_KEY) return 'gemini-3.6-flash';
  if (env.OPENROUTER_API_KEY) return 'openrouter/anthropic/claude-opus-4.7';
  if (env.TOGETHER_API_KEY) return 'moonshotai/Kimi-K3';
  const whichCmd = process.platform === 'win32' ? 'where' : 'which';
  try {
    execFileSync(whichCmd, ['claude'], {stdio: 'ignore', timeout: 2_000});
    return 'cc/opus';
  } catch {}
  try {
    execFileSync(whichCmd, ['codex'], {stdio: 'ignore', timeout: 2_000});
    return 'codex/default';
  } catch {}
  return 'No model';
}

export function getDefaultModel(): string {
  const uvPath = findUvPath();
  const kissProject = findKissProject();
  if (!uvPath || !kissProject) return getFallbackDefaultModel();
  try {
    const out = execFileSync(
      uvPath,
      [
        'run',
        '--directory',
        kissProject,
        'python',
        '-c',
        'from kiss.core.models.model_info import get_default_model; ' +
          'print(get_default_model())',
      ],
      {encoding: 'utf-8', timeout: 15_000, stdio: ['ignore', 'pipe', 'ignore']},
    ).trim();
    return out || getFallbackDefaultModel();
  } catch {
    return getFallbackDefaultModel();
  }
}

async function runFinalization(
  progress: vscode.Progress<{message?: string; increment?: number}> | null,
  kissProjectPath: string,
  uvPath: string | null,
): Promise<boolean> {
  if (uvPath) {
    if (progress) progress.report({message: 'Installing CLI wrapper...'});
    installCliScript(kissProjectPath, uvPath);
  }

  log(
    'MODEL_INFO.json and INJECTIONS.md are read directly from the bundled ' +
      'package; no copies are made into ~/.kiss/ (user overrides live in ' +
      'MY_MODELS.json and MY_INJECTION.md).',
  );

  if (progress) progress.report({message: 'Checking cloudflared...'});
  await installCloudflaredIfNeeded();

  if (progress) progress.report({message: 'Restarting kiss-web daemon...'});
  const webWorkDir =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || kissProjectPath;
  await restartKissWebDaemon(kissProjectPath, webWorkDir);

  if (progress) progress.report({message: 'Updating shell PATH...'});
  try {
    const rcPath = getShellRcPath();
    const localBin = path.join(HOME_DIR, '.local', 'bin');
    ensurePathInShellRc(rcPath, localBin);
    if (process.platform === 'win32') {
      const gitCmdDir = path.join(HOME_DIR, '.local', 'git', 'cmd');
      if (fs.existsSync(gitCmdDir)) {
        ensurePathInShellRc(rcPath, gitCmdDir);
      }
      const nodeBaseDir = path.join(HOME_DIR, '.local', 'node');
      const nodeDir = findNodeDirWindows(nodeBaseDir);
      if (fs.existsSync(nodeDir)) {
        ensurePathInShellRc(rcPath, nodeDir);
      }
    }
  } catch (err) {
    log(
      `Failed to update shell rc PATH: ${err instanceof Error ? err.message : err}`,
    );
  }

  if (progress) progress.report({message: 'Checking API keys...'});
  const apiKeysReady = await ensureApiKeys();

  if (progress) progress.report({message: 'Checking remote password...'});
  await ensureRemotePassword();

  return apiKeysReady;
}

export function ensureDependencies(): Promise<void> {
  if (pendingDeps) return pendingDeps;
  pendingDeps = ensureDependenciesImpl().finally(() => {
    pendingDeps = null;
  });
  return pendingDeps;
}

async function ensureDependenciesImpl(): Promise<void> {
  ensureLocalBinInPath();
  log('=== Dependency check started ===');

  const kissProjectPath = findKissProject();
  if (!kissProjectPath) {
    log('KISS project not found — skipping dependency setup');
    showErrorNotification(
      'KISS Sorcar: Could not find the KISS project directory. ' +
        'Please set "kissSorcar.kissProjectPath" in VS Code settings. ' +
        `See ${path.join(LOG_DIR, 'install.log')} for details.`,
    );
    return;
  }
  log(`KISS project: ${kissProjectPath}`);

  const updateMarker = path.join(LOG_DIR, '.extension-updated');
  let uvPath = findUvPath();
  let venvExists = fs.existsSync(path.join(kissProjectPath, '.venv'));
  if (
    uvPath &&
    venvExists &&
    isChromiumInstalled() &&
    (await isDaemonRunning()) &&
    !fs.existsSync(updateMarker)
  ) {
    log('All dependencies satisfied and daemon running — nothing to do');
    log('=== Dependency check finished ===');
    loadApiKeysFromShellRc();
    return;
  }

  if (uvPath && venvExists) {
    const pyStatus = checkPythonVersion(uvPath, kissProjectPath);
    if (pyStatus === 'too_old') {
      log('Python version too old — removing .venv for recreation');
      try {
        fs.rmSync(path.join(kissProjectPath, '.venv'), {
          recursive: true,
          force: true,
        });
      } catch {}
      venvExists = false;
    } else if (pyStatus === 'error') {
      log('Python version check failed (transient) — keeping .venv');
    }
  }

  let showRestartNotification = false;
  let apiKeysReady = false;
  if (fs.existsSync(updateMarker)) {
    showRestartNotification = true;
    try {
      fs.unlinkSync(updateMarker);
    } catch {}
    log('Extension-updated marker found — will show restart notification');
  }

  if (uvPath && venvExists) {
    log('Fast path: uv and .venv present, ensuring Playwright in background');
    const uv = uvPath;
    runAsync(
      uv,
      ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
      kissProjectPath,
    )
      .then(async () => {
        if (process.platform === 'linux') {
          await runAsync(
            uv,
            ['run', 'python', '-m', 'playwright', 'install-deps', 'chromium'],
            kissProjectPath,
          );
        }
      })
      .catch(err => {
        log(
          `Fast-path Playwright install failed: ${err instanceof Error ? err.message : err}`,
        );
        if (!isChromiumInstalled()) {
          showWarningNotification(
            'KISS Sorcar: Chromium browser update failed in background. ' +
              `See ${path.join(LOG_DIR, 'install.log')} for details.`,
          );
        }
      });
    if (!gitWorks()) {
      void installGit().then(installed => {
        if (!installed) {
          showWarningNotification(
            `KISS Sorcar: git is not available. ${gitInstallHint()}`,
          );
        }
      });
    }
    if (!commandExists('node')) {
      void installNode().then(installed => {
        if (!installed) {
          showWarningNotification(
            'KISS Sorcar: Node.js could not be installed automatically. Some agent tools may be unavailable.',
          );
        }
      });
    }
    if (!commandExists('code')) {
      void installCodeCli();
    }
    apiKeysReady = await runFinalization(null, kissProjectPath, uvPath);
  } else {
    const result = await withWebviewNotificationProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'KISS Sorcar: Setting up',
        cancellable: false,
      },
      async progress => {
        if (!uvPath) {
          if (process.platform !== 'win32') {
            for (const bin of ['curl', 'tar']) {
              if (!commandExists(bin)) {
                showErrorNotification(
                  `KISS Sorcar: '${bin}' is required to install uv but was not found. Please install '${bin}' and restart VS Code.`,
                );
                return {success: false, apiKeysReady: false};
              }
            }
          }
          progress.report({
            message: 'Installing uv package manager...',
            increment: 0,
          });
          uvPath = await installUv();
          if (!uvPath) {
            showErrorNotification(
              'KISS Sorcar: Failed to install uv. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh',
            );
            return {success: false, apiKeysReady: false};
          }
          progress.report({increment: 20});
        }

        if (!gitWorks()) {
          progress.report({message: 'Installing git...'});
          const gitInstalled = await installGit();
          if (!gitInstalled) {
            showWarningNotification(
              `KISS Sorcar: git could not be installed automatically. ${gitInstallHint()}`,
            );
          }
        }

        if (!commandExists('node')) {
          progress.report({message: 'Installing Node.js...'});
          const nodeInstalled = await installNode();
          if (!nodeInstalled) {
            log('Node.js could not be installed automatically');
            showWarningNotification(
              'KISS Sorcar: Node.js could not be installed automatically. ' +
                'Some agent tools may be unavailable. Install from https://nodejs.org',
            );
          }
        }

        if (!commandExists('code')) {
          progress.report({message: 'Setting up VS Code CLI...'});
          const codeInstalled = await installCodeCli();
          if (!codeInstalled) {
            log('VS Code CLI could not be set up on PATH');
          }
        }

        if (!venvExists) {
          progress.report({
            message:
              'Setting up Python environment (first time, may take a minute)...',
          });
          await runAsync(uvPath, ['sync'], kissProjectPath);
          progress.report({increment: 50});
        }

        if (checkPythonVersion(uvPath, kissProjectPath) !== 'ok') {
          showErrorNotification(
            `KISS Sorcar requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+. ` +
              `Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or later and restart VS Code.`,
          );
          return {success: false, apiKeysReady: false};
        }

        progress.report({message: 'Installing dependencies...'});
        await runAsync(
          uvPath,
          ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
          kissProjectPath,
        );
        if (process.platform === 'linux') {
          await runAsync(
            uvPath,
            ['run', 'python', '-m', 'playwright', 'install-deps', 'chromium'],
            kissProjectPath,
          ).catch(err =>
            log(
              `Playwright deps install failed (may need sudo): ${err instanceof Error ? err.message : err}`,
            ),
          );
        }
        progress.report({increment: 30});

        progress.report({message: 'Finalizing setup...'});
        const finalizedKeys = await runFinalization(
          progress,
          kissProjectPath,
          uvPath,
        );
        return {success: true, apiKeysReady: finalizedKeys};
      },
    );

    showRestartNotification = !!result.success;
    apiKeysReady = result.apiKeysReady;
  }

  log('=== Dependency check finished ===');

  if (showRestartNotification) {
    if (apiKeysReady) {
      showInformationNotification('KISS Sorcar: Installation complete!');
    } else {
      showWarningNotification(
        'KISS Sorcar: Installation complete, but at least one of Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY is required. ' +
          'Set an API key in your environment, then reload the window (Developer: Reload Window) to be prompted again.',
      );
    }
  }
}

export function pidsOnPort(port: number): string[] {
  try {
    return execFileSync('lsof', ['-ti', `tcp:${port}`, '-sTCP:LISTEN'], {
      encoding: 'utf-8',
      timeout: 3000,
      stdio: ['ignore', 'pipe', 'ignore'],
    })
      .trim()
      .split('\n')
      .filter(Boolean);
  } catch {
    return [];
  }
}

function killPids(pids: string[], signal: NodeJS.Signals): void {
  for (const pid of pids) {
    try {
      process.kill(parseInt(pid, 10), signal);
    } catch {}
  }
}

function killProcessOnPort(port: number): void {
  const pids = pidsOnPort(port);
  if (pids.length === 0) return;
  killPids(pids, 'SIGTERM');
  for (let i = 0; i < 6; i++) {
    if (pidsOnPort(port).length === 0) return;
    sleepSync(500);
  }
  killPids(pidsOnPort(port), 'SIGKILL');
}

function spawnKissWebDirect(kissWebBin: string, workDir: string): void {
  const binDir = path.join(HOME_DIR, '.local', 'bin');
  try {
    fs.mkdirSync(LOG_DIR, {recursive: true});
    const outFd = fs.openSync(path.join(LOG_DIR, 'kiss-web-stdout.log'), 'a');
    const errFd = fs.openSync(path.join(LOG_DIR, 'kiss-web-stderr.log'), 'a');
    const child = spawn(kissWebBin, [], {
      cwd: workDir,
      detached: true,
      stdio: ['ignore', outFd, errFd],
      env: {
        ...process.env,
        PATH: `${binDir}:${process.env.PATH || '/usr/local/bin:/usr/bin:/bin'}`,
      },
    });
    child.unref();
    fs.closeSync(outFd);
    fs.closeSync(errFd);
    log(
      `kiss-web started directly (no systemd): pid ${child.pid ?? '<unknown>'}, ` +
        `cwd ${workDir}`,
    );
  } catch (err) {
    log(
      `Failed to start kiss-web directly: ${err instanceof Error ? err.message : err}`,
    );
  }
}

// A restart is a probe-then-act on a resource every window shares: the
// daemon on port 8787.  Without a cross-process lock two windows opened
// together both see "dead", and the second SIGTERMs the daemon the first
// has just started -- while it is still booting, so it has not yet
// accepted a UDS connection and cannot report the active tasks that
// decideRestart() exists to protect.
const RESTART_LOCK_FILE = path.join(LOG_DIR, '.kiss-web.restart.lock');
// How long a lock whose owner cannot be identified -- an empty file
// caught mid-write, or one written by an older extension build -- is
// honoured before it is assumed abandoned.
const RESTART_LOCK_STALE_MS = 120_000;
// The backstop for a lock whose owner is still ALIVE.  Age is no
// evidence that a live window is finished: verifyDaemonStartup() alone
// is allowed 180s, and a laptop suspended mid-restart adds however long
// it slept.  Only a window that has been in the restart path for longer
// than any restart could conceivably take is treated as wedged.
const RESTART_LOCK_MAX_HOLD_MS = 600_000;

interface RestartLockOwner {
  pid: number;
  token: string;
}

/**
 * Read the identity stamped in a restart lock file.
 *
 * @param lockFile Path of the lock file.
 * @returns The owner, or null when the file is missing, half-written or
 *     not in the current format.
 */
function readRestartLockOwner(lockFile: string): RestartLockOwner | null {
  try {
    const data: unknown = JSON.parse(fs.readFileSync(lockFile, 'utf-8'));
    const {pid, token} = data as {pid?: unknown; token?: unknown};
    if (typeof pid !== 'number' || !pid) return null;
    if (typeof token !== 'string' || !token) return null;
    return {pid, token};
  } catch {
    return null;
  }
}

/**
 * Report whether a process id still exists.
 *
 * @param pid The process id stamped in the lock.
 * @returns True when the process is running (EPERM counts: it exists,
 *     it just belongs to another user).
 */
function processIsAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch (err) {
    return (err as NodeJS.ErrnoException).code === 'EPERM';
  }
}

/**
 * Delete a restart lock that nobody is using any more, if there is one.
 *
 * @param lockFile Path of the lock file.
 * @returns True when the caller should retry taking the lock.
 */
function breakAbandonedRestartLock(lockFile: string): boolean {
  let ageMs: number;
  try {
    ageMs = Date.now() - fs.statSync(lockFile).mtimeMs;
  } catch {
    // The holder released it between our open and our stat; retry.
    return true;
  }
  const owner = readRestartLockOwner(lockFile);
  if (owner && processIsAlive(owner.pid)) {
    if (ageMs < RESTART_LOCK_MAX_HOLD_MS) return false;
    log(
      `breaking kiss-web restart lock wedged by live pid ${owner.pid} ` +
        `(${Math.round(ageMs)}ms old)`,
    );
  } else if (owner) {
    log(`breaking kiss-web restart lock left by dead pid ${owner.pid}`);
  } else if (ageMs < RESTART_LOCK_STALE_MS) {
    // No readable owner: most likely a lock created moments ago whose
    // identity has not been written yet.  Assume it is live.
    return false;
  } else {
    log(
      'breaking unreadable kiss-web restart lock ' +
        `(${Math.round(ageMs)}ms old)`,
    );
  }
  try {
    fs.unlinkSync(lockFile);
  } catch {
    return false;
  }
  return true;
}

/**
 * Take the cross-process kiss-web restart lock.
 *
 * The lock is an exclusively created file, which is atomic across
 * processes on every POSIX filesystem the extension runs on, and it
 * carries the owner's pid and a one-off token.
 *
 * Both are load-bearing.  A lock is broken only when its owner is
 * provably gone -- or has held it for longer than any restart could
 * take -- because age alone says nothing about whether a window is
 * still inside the restart path, and evicting one that is puts us back
 * to a window SIGTERMing the daemon another just started.  And because
 * a lock CAN change hands that way, the release checks the token: an
 * evicted owner that finishes later must not delete its successor's
 * lock and let a third window in beside it.
 *
 * @param lockFile Path of the lock file (overridable for tests).
 * @returns A release function, or null when another window holds it.
 */
export function acquireDaemonRestartLock(
  lockFile: string = RESTART_LOCK_FILE,
): (() => void) | null {
  const token = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      fs.mkdirSync(path.dirname(lockFile), {recursive: true});
      const fd = fs.openSync(lockFile, 'wx');
      try {
        fs.writeSync(fd, JSON.stringify({pid: process.pid, token}));
      } finally {
        fs.closeSync(fd);
      }
      return () => releaseDaemonRestartLock(lockFile, token);
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code !== 'EEXIST') return null;
      if (!breakAbandonedRestartLock(lockFile)) return null;
    }
  }
  return null;
}

/**
 * Release a restart lock, but only while we still own it.
 *
 * @param lockFile Path of the lock file.
 * @param token The token stamped when the lock was taken.
 */
function releaseDaemonRestartLock(lockFile: string, token: string): void {
  const owner = readRestartLockOwner(lockFile);
  if (!owner || owner.token !== token) {
    if (owner) {
      log(
        `kiss-web restart lock now belongs to pid ${owner.pid}; ` +
          'leaving it alone',
      );
    }
    return;
  }
  try {
    fs.unlinkSync(lockFile);
  } catch {}
}

export async function restartKissWebDaemon(
  kissProjectPath: string,
  workDir: string,
): Promise<void> {
  if (process.platform === 'win32') return;

  const kissWebBin = path.join(kissProjectPath, '.venv', 'bin', 'kiss-web');
  if (!fs.existsSync(kissWebBin)) {
    log(`kiss-web binary not found at ${kissWebBin} — skipping daemon setup`);
    return;
  }

  const releaseLock = acquireDaemonRestartLock();
  if (!releaseLock) {
    log('another window is restarting kiss-web — skipping this one');
    return;
  }
  try {
    await restartKissWebDaemonLocked(kissProjectPath, workDir, kissWebBin);
  } finally {
    releaseLock();
  }
}

async function restartKissWebDaemonLocked(
  kissProjectPath: string,
  workDir: string,
  kissWebBin: string,
): Promise<void> {
  const binDir = path.join(HOME_DIR, '.local', 'bin');

  const fpFile = path.join(LOG_DIR, '.kiss-web.fingerprint');
  const currentFp = computeKissWebFingerprint(
    kissProjectPath,
    kissWebBin,
    workDir,
  );
  let savedFp = '';
  try {
    savedFp = fs.readFileSync(fpFile, 'utf-8').trim();
  } catch {}
  const sockPath = sorcarSockPath();

  const health = await probeDaemonHealth(8787, 1500);
  const sockExists = fs.existsSync(sockPath);

  // Always query the UDS, even when the TCP listener looks dead: the task
  // worker can be alive behind a transiently refused HTTP port, and
  // decideRestart() protects any reported active task regardless of health.
  const activeTasks:
    {ok: true; count: number; tabs: string[]} | {ok: false; reason: string} =
    await daemonHasActiveTasks(sockPath, 1500);

  const decision = decideRestart({
    fingerprintMatches: !!currentFp && currentFp === savedFp,
    health,
    activeTasks,
  });
  if (decision.skip) {
    if (decision.reason === 'active-tasks') {
      log(
        `kiss-web has ${(activeTasks as {ok: true; count: number}).count} ` +
          'active task(s) — deferring restart to avoid aborting in-flight work',
      );
    } else if (decision.reason.startsWith('alive-uncertain')) {
      log(
        'kiss-web alive but active-tasks probe inconclusive ' +
          `(${decision.reason}) — deferring restart to next activation`,
      );
    } else {
      log(
        `kiss-web fingerprint unchanged (${currentFp.slice(0, 8)}) and ` +
          `daemon healthy (health=${health}, sock=${sockExists}) — ` +
          'skipping restart to preserve tunnel URL',
      );
    }
    return;
  }
  log(
    `kiss-web restart: fingerprint ${savedFp.slice(0, 8) || '<none>'} → ` +
      `${currentFp.slice(0, 8) || '<none>'}, health=${health}, ` +
      `sock=${sockExists}, activeTasks=` +
      `${activeTasks.ok ? activeTasks.count : 'unknown(' + activeTasks.reason + ')'}`,
  );

  killProcessOnPort(8787);

  let reissueRestart: (() => void | Promise<void>) | null = null;

  if (process.platform === 'darwin') {
    const plistLabel = 'com.kiss.web-server';
    const plistDir = path.join(HOME_DIR, 'Library', 'LaunchAgents');
    const plistFile = path.join(plistDir, `${plistLabel}.plist`);

    log('Restarting kiss-web macOS LaunchAgent...');
    try {
      fs.mkdirSync(plistDir, {recursive: true});
      const xLabel = xmlEscape(plistLabel);
      const xBin = xmlEscape(kissWebBin);
      const xProj = xmlEscape(workDir);
      const xLogDir = xmlEscape(LOG_DIR);
      const xPath = xmlEscape(
        `/opt/homebrew/bin:${binDir}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`,
      );
      // The daemon resolves its state dir — and the socket it binds —
      // from $KISS_HOME, so a KISS_HOME visible only to the VS Code
      // process must reach the launchd service too; otherwise the
      // extension probes $KISS_HOME/sorcar.sock while the daemon binds
      // ~/.kiss/sorcar.sock, and every health poll restarts a healthy
      // daemon.  KISS_SORCAR_SOCK is a client-side override only (the
      // daemon does not read it), so it is deliberately NOT propagated.
      const kissHomeEnv = process.env.KISS_HOME || '';
      const xKissHomeEntry = kissHomeEnv
        ? `\n        <key>KISS_HOME</key>\n        <string>${xmlEscape(
            kissHomeEnv,
          )}</string>`
        : '';
      const plistContent = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${xLabel}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${xBin}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${xProj}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>StandardOutPath</key>
    <string>${xLogDir}/kiss-web-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${xLogDir}/kiss-web-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${xPath}</string>${xKissHomeEntry}
    </dict>
</dict>
</plist>`;

      fs.writeFileSync(plistFile, plistContent);

      const uid = execFileSync('id', ['-u'], {encoding: 'utf-8'}).trim();
      reissueRestart = async () => {
        const res = await restartLaunchAgent({
          serviceTarget: `gui/${uid}/${plistLabel}`,
          domainTarget: `gui/${uid}`,
          plistFile,
          log,
        });
        log(
          `kiss-web LaunchAgent restart (${plistFile}): ` +
            `drained=${res.drained} (${res.drainedMs}ms), ` +
            `bootstrapped=${res.bootstrapped} ` +
            `(${res.bootstrapAttempts} attempt(s)), ` +
            `registered=${res.registered}, kickstarted=${res.kickstarted}`,
        );
      };
      await reissueRestart();
    } catch (err) {
      log(
        `Failed to restart kiss-web daemon (macOS): ${err instanceof Error ? err.message : err}`,
      );
    }
  } else if (process.platform === 'linux') {
    const systemdDir = path.join(HOME_DIR, '.config', 'systemd', 'user');
    const serviceFile = path.join(systemdDir, 'kiss-web.service');

    log('Restarting kiss-web systemd user service...');
    let systemdOk = false;
    try {
      fs.mkdirSync(systemdDir, {recursive: true});
      const uBin = unitEscape(kissWebBin);
      const uProj = unitEscape(workDir);
      const uPath = unitEscape(`${binDir}:/usr/local/bin:/usr/bin:/bin`);
      const uLogDir = unitEscape(LOG_DIR);
      // Same KISS_HOME propagation as the launchd plist above: the
      // daemon binds its socket under $KISS_HOME, so the service must
      // see the same value the extension host sees.  KISS_SORCAR_SOCK
      // is a client-side override only (the daemon does not read it),
      // so it is deliberately NOT propagated.
      const kissHomeEnv = process.env.KISS_HOME || '';
      const kissHomeLine = kissHomeEnv
        ? `Environment=KISS_HOME=${unitEscape(kissHomeEnv)}\n`
        : '';
      const serviceContent = `[Unit]
Description=KISS Sorcar Remote Web Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=${uBin}
WorkingDirectory=${uProj}
Restart=always
RestartSec=5
Environment=PATH=${uPath}
${kissHomeLine}StandardOutput=append:${uLogDir}/kiss-web-stdout.log
StandardError=append:${uLogDir}/kiss-web-stderr.log

[Install]
WantedBy=default.target
`;
      fs.writeFileSync(serviceFile, serviceContent);
      execSync('systemctl --user daemon-reload', {
        stdio: 'ignore',
        timeout: 10000,
      });
      // --no-block queues the restart job and returns immediately.  A
      // blocking restart waits for the old daemon to finish shutting
      // down, which can take longer than the 10s timeout (tunnel
      // cleanup; the daemon's SIGTERM failsafe allows 30s).  The
      // ETIMEDOUT that execSync then threw was misread as "systemd
      // failed" and triggered the direct-spawn fallback below — while
      // systemd's restart job was still in flight — leaving TWO
      // daemons racing for port 8787 and systemd crash-looping every
      // RestartSec against the rogue's listener.
      execSync('systemctl --user restart --no-block kiss-web', {
        stdio: 'ignore',
        timeout: 10000,
      });
      const username = os.userInfo().username;
      try {
        execFileSync('loginctl', ['enable-linger', username], {
          stdio: 'ignore',
          timeout: 5000,
        });
      } catch {}
      log(`kiss-web systemd user service restarted: ${serviceFile}`);
      systemdOk = true;
      reissueRestart = () => {
        execSync('systemctl --user restart --no-block kiss-web', {
          stdio: 'ignore',
          timeout: 10000,
        });
      };
    } catch (err) {
      log(
        'Failed to restart kiss-web daemon via systemd (Linux): ' +
          `${err instanceof Error ? err.message : err} — ` +
          'falling back to direct background spawn',
      );
    }
    if (!systemdOk) {
      reissueRestart = () => spawnKissWebDirect(kissWebBin, workDir);
      void reissueRestart();
    }
  }

  const verdict = await verifyDaemonStartup({
    binPath: kissWebBin,
    sockPath,
    port: 8787,
    restart: reissueRestart,
    log,
  });
  if (!verdict.ok) {
    log(
      `kiss-web daemon did NOT come up within ${verdict.waitedMs}ms of the ` +
        `restart (reason=${verdict.reason}, extra restart attempts=` +
        `${verdict.restarts}) — fingerprint not recorded so the next ` +
        'activation retries',
    );
    return;
  }
  log(
    `kiss-web daemon verified up ${verdict.waitedMs}ms after restart` +
      (verdict.restarts > 0
        ? ` (needed ${verdict.restarts} extra restart attempt(s))`
        : '') +
      (verdict.binaryVanished
        ? ' (kiss-web binary was transiently missing — concurrent reinstall)'
        : ''),
  );

  try {
    fs.writeFileSync(fpFile, currentFp + '\n');
  } catch (err) {
    log(
      `Failed to write kiss-web fingerprint: ${err instanceof Error ? err.message : err}`,
    );
  }
}

function computeKissWebFingerprint(
  kissProjectPath: string,
  kissWebBin: string,
  workDir: string,
): string {
  try {
    const hash = crypto.createHash('sha256');
    hash.update(fs.readFileSync(kissWebBin));
    hash.update(workDir);
    const srcDir = path.join(kissProjectPath, 'src', 'kiss');
    let latestMtimeNs = BigInt(0);
    const walk = (dir: string): void => {
      let entries: fs.Dirent[];
      try {
        entries = fs.readdirSync(dir, {withFileTypes: true});
      } catch {
        return;
      }
      for (const entry of entries) {
        if (entry.name === '__pycache__' || entry.name === 'tests') continue;
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
          walk(full);
        } else if (entry.isFile() && entry.name.endsWith('.py')) {
          try {
            const st = fs.statSync(full, {bigint: true});
            if (st.mtimeNs > latestMtimeNs) latestMtimeNs = st.mtimeNs;
          } catch {}
        }
      }
    };
    walk(srcDir);
    hash.update(latestMtimeNs.toString());
    return hash.digest('hex');
  } catch (err) {
    log(
      `computeKissWebFingerprint failed: ${err instanceof Error ? err.message : err}`,
    );
    return '';
  }
}

function installCliScript(kissProjectPath: string, uvPath: string): void {
  if (!HOME_DIR) return;

  const binDir = path.join(HOME_DIR, '.local', 'bin');

  let absUvPath = uvPath;
  if (uvPath === 'uv' || !path.isAbsolute(uvPath)) {
    try {
      const whichCmd =
        process.platform === 'win32' ? `where ${uvPath}` : `which ${uvPath}`;
      // `where` on Windows emits CRLF line endings and may print several
      // matches; splitting on '\n' alone left a trailing '\r' on the first
      // line, which then got baked into the generated sorcar.cmd.
      absUvPath = execSync(whichCmd, {encoding: 'utf-8'})
        .trim()
        .split(/\r?\n/)[0]
        .trim();
    } catch {
      const suffix = process.platform === 'win32' ? '.exe' : '';
      absUvPath = path.join(HOME_DIR, '.local', 'bin', `uv${suffix}`);
    }
  }

  try {
    fs.mkdirSync(binDir, {recursive: true});

    if (process.platform === 'win32') {
      const cmdPath = path.join(binDir, 'sorcar.cmd');
      const script =
        '@echo off\r\n' +
        'REM Installed by KISS Sorcar VS Code extension\r\n' +
        'set "KISS_WORKDIR=%CD%"\r\n' +
        `"${absUvPath}" run --directory "${kissProjectPath}" sorcar %*\r\n`;
      fs.writeFileSync(cmdPath, script);
    } else {
      const scriptPath = path.join(binDir, 'sorcar');
      const script =
        '#!/bin/bash\n' +
        '# Installed by KISS Sorcar VS Code extension\n' +
        'export KISS_WORKDIR="$PWD"\n' +
        `exec "${absUvPath}" run --directory "${kissProjectPath}" sorcar "$@"\n`;
      fs.writeFileSync(scriptPath, script, {mode: 0o755});
    }
  } catch (err) {
    log(
      `Failed to install CLI script: ${err instanceof Error ? err.message : err}`,
    );
  }
}

function uvAssetInfo(): {
  archName: string;
  triplet: string;
  ext: string;
} | null {
  const archMap: Record<string, string> = {
    arm64: 'aarch64',
    x64: 'x86_64',
  };
  const arch = archMap[process.arch];
  if (!arch) return null;

  if (process.platform === 'darwin') {
    return {archName: arch, triplet: `${arch}-apple-darwin`, ext: 'tar.gz'};
  } else if (process.platform === 'linux') {
    return {
      archName: arch,
      triplet: `${arch}-unknown-linux-gnu`,
      ext: 'tar.gz',
    };
  } else if (process.platform === 'win32') {
    return {archName: arch, triplet: `${arch}-pc-windows-msvc`, ext: 'zip'};
  }
  return null;
}

async function installUv(): Promise<string | null> {
  const asset = uvAssetInfo();
  if (!asset) {
    log(
      `Unsupported platform/arch for uv: ${process.platform}/${process.arch}`,
    );
    return null;
  }

  const installDir = path.join(HOME_DIR, '.local', 'bin');
  const assetName = `uv-${asset.triplet}`;
  const url = `https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/${assetName}.${asset.ext}`;
  log(`Downloading uv ${UV_VERSION} from ${url}`);

  try {
    fs.mkdirSync(installDir, {recursive: true});

    if (process.platform === 'win32') {
      const zipPath = path.join(installDir, `${assetName}.zip`);
      await windowsZipInstall(
        url,
        zipPath,
        installDir,
        `Move-Item -Force '${path.join(installDir, assetName, 'uv.exe')}' '${path.join(installDir, 'uv.exe')}'; ` +
          `Move-Item -Force '${path.join(installDir, assetName, 'uvx.exe')}' '${path.join(installDir, 'uvx.exe')}'; ` +
          `Remove-Item -Recurse -Force '${path.join(installDir, assetName)}'; `,
      );
    } else {
      const tarPath = path.join(installDir, `${assetName}.${asset.ext}`);
      await downloadFile(url, tarPath);
      const expectedHash = await fetchUvStyleSha256(url);
      verifyDownloadHash(tarPath, expectedHash);
      await spawnPromise('tar', ['xzf', tarPath, '-C', installDir]);
      const extractedDir = path.join(installDir, assetName);
      for (const bin of ['uv', 'uvx']) {
        const src = path.join(extractedDir, bin);
        const dst = path.join(installDir, bin);
        try {
          fs.unlinkSync(dst);
        } catch {}
        fs.renameSync(src, dst);
        fs.chmodSync(dst, 0o755);
      }
      try {
        fs.rmSync(extractedDir, {recursive: true, force: true});
      } catch {}
      try {
        fs.unlinkSync(tarPath);
      } catch {}
    }

    log('uv installed successfully');
    return findUvPath();
  } catch (err) {
    log(`Failed to install uv: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

function checkPythonVersion(
  uvPath: string,
  cwd: string,
): 'ok' | 'too_old' | 'error' {
  try {
    const output = execFileSync(uvPath, ['run', 'python', '--version'], {
      cwd,
      encoding: 'utf-8',
      timeout: 30_000,
    }).trim();
    const match = output.match(/Python\s+(\d+)\.(\d+)/);
    if (!match) return 'error';
    const major = parseInt(match[1], 10);
    const minor = parseInt(match[2], 10);
    if (
      major > MIN_PYTHON_MAJOR ||
      (major === MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR)
    ) {
      return 'ok';
    }
    return 'too_old';
  } catch {
    return 'error';
  }
}

function playwrightBrowsersPath(): string {
  const env = process.env.PLAYWRIGHT_BROWSERS_PATH;
  if (env) return env;
  if (process.platform === 'darwin') {
    return path.join(HOME_DIR, 'Library', 'Caches', 'ms-playwright');
  } else if (process.platform === 'win32') {
    return path.join(
      process.env.LOCALAPPDATA || path.join(HOME_DIR, 'AppData', 'Local'),
      'ms-playwright',
    );
  }
  return path.join(HOME_DIR, '.cache', 'ms-playwright');
}

async function isDaemonRunning(): Promise<boolean> {
  if (process.platform === 'win32') return false;
  const sockPath = sorcarSockPath();
  for (let attempt = 0; attempt < 3; attempt++) {
    const health = await probeDaemonHealth(8787);
    if (health === 'alive' && fs.existsSync(sockPath)) return true;
    if (attempt < 2) {
      await new Promise(r => setTimeout(r, 300));
    }
  }
  return false;
}

function isChromiumInstalled(): boolean {
  try {
    const cacheDir = playwrightBrowsersPath();
    if (!fs.existsSync(cacheDir)) return false;
    return fs.readdirSync(cacheDir).some(e => e.startsWith('chromium-'));
  } catch {
    return false;
  }
}

function commandExists(cmd: string): boolean {
  try {
    execFileSync(process.platform === 'win32' ? 'where' : 'which', [cmd], {
      stdio: 'ignore',
    });
    return true;
  } catch {
    return false;
  }
}

function gitWorks(): boolean {
  try {
    const output = execSync('git --version', {
      encoding: 'utf-8',
      timeout: 10_000,
      stdio: ['ignore', 'pipe', 'ignore'],
    });
    return output.includes('git version');
  } catch {
    return false;
  }
}

function gitInstallHint(): string {
  if (process.platform === 'darwin') {
    return 'Run "xcode-select --install" in Terminal, or install Homebrew (https://brew.sh) and run "brew install git".';
  } else if (process.platform === 'linux') {
    return 'Run "sudo apt-get install git" (Debian/Ubuntu), "sudo dnf install git" (Fedora), or the equivalent for your distribution.';
  } else if (process.platform === 'win32') {
    return 'Download Git from https://git-scm.com/download/win';
  }
  return 'Download Git from https://git-scm.com';
}

async function installGit(): Promise<boolean> {
  log('Git not found, attempting to install...');

  if (process.platform === 'darwin') {
    if (commandExists('brew')) {
      log('Installing git via Homebrew...');
      try {
        await execPromise('brew install git');
        if (gitWorks()) {
          log('Git installed via Homebrew');
          return true;
        }
      } catch (err) {
        log(
          `Homebrew git install failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }

    try {
      execSync('xcode-select -p', {stdio: 'ignore'});
      log('Xcode CLT present but git not working');
      return false;
    } catch {}

    log('Triggering Xcode Command Line Tools installation...');
    try {
      execSync('xcode-select --install', {stdio: 'ignore', timeout: 5_000});
    } catch {}

    for (let i = 0; i < 120; i++) {
      await new Promise(resolve => setTimeout(resolve, 5_000));
      if (gitWorks()) {
        log('Git installed via Xcode Command Line Tools');
        return true;
      }
    }
    return false;
  } else if (process.platform === 'linux') {
    const attempts: [string, string][] = [
      [
        'apt-get',
        'sudo -n sh -c "apt-get update -y && apt-get install -y git"',
      ],
      ['dnf', 'sudo -n dnf install -y git'],
      ['yum', 'sudo -n yum install -y git'],
      ['pacman', 'sudo -n pacman -S --noconfirm git'],
      ['apk', 'sudo -n apk add git'],
    ];
    for (const [bin, cmd] of attempts) {
      if (commandExists(bin)) {
        log(`Installing git via ${bin}...`);
        try {
          await execPromise(cmd);
          if (gitWorks()) {
            log(`Git installed via ${bin}`);
            return true;
          }
        } catch (err) {
          log(`Failed via ${bin}: ${err instanceof Error ? err.message : err}`);
        }
      }
    }
    return false;
  } else if (process.platform === 'win32') {
    return installMinGitWindows();
  }

  return false;
}

async function installMinGitWindows(): Promise<boolean> {
  const GIT_VERSION = '2.49.0';
  const archSuffix = process.arch === 'arm64' ? 'arm64' : '64';
  const assetName = `MinGit-${GIT_VERSION}-${archSuffix}-bit`;
  const url = `https://github.com/git-for-windows/git/releases/download/v${GIT_VERSION}.windows.1/${assetName}.zip`;
  const gitDir = path.join(HOME_DIR, '.local', 'git');

  log(`Downloading MinGit from ${url}`);

  try {
    fs.mkdirSync(gitDir, {recursive: true});

    const zipPath = path.join(gitDir, `${assetName}.zip`);
    await windowsZipInstall(url, zipPath, gitDir);

    const gitCmdDir = path.join(gitDir, 'cmd');
    if (fs.existsSync(path.join(gitCmdDir, 'git.exe'))) {
      prependToProcessPath(gitCmdDir);
      log('MinGit installed successfully');
      return true;
    }
    log('MinGit extracted but git.exe not found in cmd/');
  } catch (err) {
    log(
      `MinGit installation failed: ${err instanceof Error ? err.message : err}`,
    );
  }
  return false;
}

async function installCloudflaredIfNeeded(): Promise<boolean> {
  if (process.platform === 'win32') return false;
  if (commandExists('cloudflared')) return true;

  const archMap: Record<string, string> = {arm64: 'arm64', x64: 'amd64'};
  const arch = archMap[process.arch];
  if (!arch) {
    log(`Unsupported architecture for cloudflared: ${process.arch}`);
    return false;
  }

  const binDir = path.join(HOME_DIR, '.local', 'bin');
  fs.mkdirSync(binDir, {recursive: true});

  try {
    if (process.platform === 'darwin') {
      if (commandExists('brew')) {
        try {
          await execPromise('brew install cloudflared');
          if (commandExists('cloudflared')) return true;
        } catch (err) {
          log(
            `Homebrew cloudflared install failed: ${err instanceof Error ? err.message : err}`,
          );
        }
      }

      const url = `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-${arch}.tgz`;
      const tarPath = path.join(binDir, `cloudflared-darwin-${arch}.tgz`);
      await downloadFile(url, tarPath);
      await spawnPromise('tar', ['xzf', tarPath, '-C', binDir]);
      try {
        fs.unlinkSync(tarPath);
      } catch {}
    } else if (process.platform === 'linux') {
      const url = `https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${arch}`;
      const dst = path.join(binDir, 'cloudflared');
      await downloadFile(url, dst);
    } else {
      return false;
    }

    const cloudflaredPath = path.join(binDir, 'cloudflared');
    if (fs.existsSync(cloudflaredPath)) {
      fs.chmodSync(cloudflaredPath, 0o755);
    }
    log('cloudflared installed successfully');
    return commandExists('cloudflared') || fs.existsSync(cloudflaredPath);
  } catch (err) {
    log(
      `cloudflared installation failed: ${err instanceof Error ? err.message : err}`,
    );
    return false;
  }
}

async function installNode(): Promise<boolean> {
  const archMap: Record<string, string> = {arm64: 'arm64', x64: 'x64'};
  const arch = archMap[process.arch];
  if (!arch) {
    log(`Unsupported architecture for Node.js: ${process.arch}`);
    return false;
  }

  if (process.platform === 'win32') {
    const assetName = `node-${NODE_VERSION}-win-${arch}`;
    const url = `https://nodejs.org/dist/${NODE_VERSION}/${assetName}.zip`;
    const installDir = path.join(HOME_DIR, '.local', 'node');
    log(`Downloading Node.js from ${url}`);
    try {
      fs.mkdirSync(installDir, {recursive: true});
      const zipPath = path.join(installDir, `${assetName}.zip`);
      await windowsZipInstall(url, zipPath, installDir);
      const nodeDir = path.join(installDir, assetName);
      if (fs.existsSync(path.join(nodeDir, 'node.exe'))) {
        prependToProcessPath(nodeDir);
        log('Node.js installed successfully (Windows)');
        return true;
      }
    } catch (err) {
      log(
        `Node.js installation failed: ${err instanceof Error ? err.message : err}`,
      );
    }
    return false;
  }

  const osName = process.platform === 'darwin' ? 'darwin' : 'linux';
  const assetName = `node-${NODE_VERSION}-${osName}-${arch}`;
  const url = `https://nodejs.org/dist/${NODE_VERSION}/${assetName}.tar.gz`;
  log(`Downloading Node.js from ${url}`);

  try {
    const installDir = path.join(HOME_DIR, '.local');
    fs.mkdirSync(installDir, {recursive: true});
    const tarPath = path.join(installDir, `${assetName}.tar.gz`);
    await downloadFile(url, tarPath);
    const expectedHash = await fetchNodeSha256(`${assetName}.tar.gz`);
    verifyDownloadHash(tarPath, expectedHash);
    await spawnPromise('tar', [
      'xzf',
      tarPath,
      '-C',
      installDir,
      '--strip-components=1',
    ]);
    try {
      fs.unlinkSync(tarPath);
    } catch {}
    log('Node.js installed successfully');
    return commandExists('node');
  } catch (err) {
    log(
      `Node.js installation failed: ${err instanceof Error ? err.message : err}`,
    );
    return false;
  }
}

async function installCodeCli(): Promise<boolean> {
  if (commandExists('code')) return true;

  if (process.platform === 'darwin') {
    const vscodeApp =
      '/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code';
    if (fs.existsSync(vscodeApp)) {
      const binDir = path.join(HOME_DIR, '.local', 'bin');
      try {
        fs.mkdirSync(binDir, {recursive: true});
        const linkPath = path.join(binDir, 'code');
        try {
          fs.unlinkSync(linkPath);
        } catch {}
        fs.symlinkSync(vscodeApp, linkPath);
        log('VS Code CLI symlinked to ~/.local/bin/code');
        return true;
      } catch (err) {
        log(
          `Failed to symlink VS Code CLI: ${err instanceof Error ? err.message : err}`,
        );
      }
    }
  } else if (process.platform === 'linux') {
    if (commandExists('snap')) {
      try {
        await execPromise('sudo -n snap install --classic code');
        if (commandExists('code')) {
          log('VS Code CLI installed via snap');
          return true;
        }
      } catch (err) {
        log(`snap install failed: ${err instanceof Error ? err.message : err}`);
      }
    }
    if (commandExists('apt-get')) {
      try {
        await execPromise(
          'curl -fsSL https://packages.microsoft.com/keys/microsoft.asc | sudo -n gpg --dearmor -o /usr/share/keyrings/microsoft.gpg && ' +
            'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/repos/code stable main" | ' +
            'sudo -n tee /etc/apt/sources.list.d/vscode.list >/dev/null && ' +
            'sudo -n apt-get update -y && sudo -n apt-get install -y code',
        );
        if (commandExists('code')) {
          log('VS Code CLI installed via apt');
          return true;
        }
      } catch (err) {
        log(`apt install failed: ${err instanceof Error ? err.message : err}`);
      }
    }
  }
  return commandExists('code');
}

async function runAsync(
  cmd: string,
  args: string[],
  cwd: string,
): Promise<void> {
  const cmdLine = `${cmd} ${args.join(' ')}`;
  log(`Running: ${cmdLine}`);
  let r: {code: number | null; stdout: string; stderr: string};
  try {
    r = await spawnCollect(cmd, args, {
      cwd,
      env: {...process.env, PYTHONUNBUFFERED: '1'},
      timeoutMs: 0,
    });
  } catch (err) {
    log(`Spawn error [${cmdLine}]: ${(err as Error).message}`);
    throw err;
  }
  const output = r.stdout + r.stderr;
  if (output.trim()) log(`Output [${cmdLine}]:\n${output.trim()}`);
  if (r.code === 0) {
    log(`Completed: ${cmdLine}`);
    return;
  }
  throw new Error(`${cmdLine} failed (exit code ${r.code}): ${output}`);
}

function execPromise(cmd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(cmd, {timeout: 300_000}, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}

function getShellRcPath(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  if (process.platform === 'win32') {
    const docsDir = path.join(homeDir, 'Documents', 'PowerShell');
    return path.join(docsDir, 'Microsoft.PowerShell_profile.ps1');
  }

  const shell = process.env.SHELL || '';
  if (shell.endsWith('/zsh') || shell.endsWith('/zsh-5')) {
    return path.join(homeDir, '.zshrc');
  } else if (shell.endsWith('/fish')) {
    return path.join(homeDir, '.config', 'fish', 'config.fish');
  } else {
    return path.join(homeDir, '.bashrc');
  }
}

function validateAnthropicKey(key: string): Promise<boolean> {
  return new Promise(resolve => {
    const req = https.request(
      {
        hostname: 'api.anthropic.com',
        path: '/v1/models',
        method: 'GET',
        headers: {
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
        },
        timeout: 15000,
      },
      res => {
        resolve(res.statusCode === 200);
        res.resume();
      },
    );
    req.on('error', () => resolve(false));
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.end();
  });
}

function readShellRc(rcPath: string): string {
  try {
    return fs.readFileSync(rcPath, 'utf-8');
  } catch {
    fs.mkdirSync(path.dirname(rcPath), {recursive: true});
    return '';
  }
}

function writeShellRc(rcPath: string, content: string): void {
  if (content.length > 0 && !content.endsWith('\n')) {
    content += '\n';
  }
  fs.writeFileSync(rcPath, content);
}

function addToShellRc(rcPath: string, envName: string, value: string): void {
  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  const exportLine = isPs1
    ? `$env:${envName} = "${value}"`
    : isFish
      ? `set -gx ${envName} "${value}"`
      : `export ${envName}="${value}"`;

  let content = readShellRc(rcPath);

  const linePattern = isPs1
    ? new RegExp(`^\\s*\\$env:${envName}\\s*=.*$`, 'gm')
    : isFish
      ? new RegExp(`^\\s*set\\s+-gx\\s+${envName}\\s.*$`, 'gm')
      : new RegExp(`^\\s*export\\s+${envName}=.*$`, 'gm');

  if (linePattern.test(content)) {
    linePattern.lastIndex = 0;
    content = content.replace(linePattern, exportLine);
  } else {
    if (content.length > 0 && !content.endsWith('\n')) {
      content += '\n';
    }
    content += exportLine + '\n';
  }

  writeShellRc(rcPath, content);
}

function ensurePathInShellRc(rcPath: string, dirPath: string): void {
  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  let dirRef = dirPath;
  if (homeDir && dirPath.startsWith(homeDir)) {
    dirRef = dirPath.replace(homeDir, '$HOME');
  }

  let content = readShellRc(rcPath);

  const escaped = dirRef
    .replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    .replace('\\$HOME', '(\\$HOME|~)');
  const alreadyPresent = isPs1
    ? new RegExp(`\\$env:PATH.*${escaped}`, 'm').test(content)
    : isFish
      ? new RegExp(`fish_add_path.*${escaped}`, 'm').test(content)
      : new RegExp(`PATH.*${escaped}`, 'm').test(content);

  if (alreadyPresent) return;

  const pathSep = isPs1 ? ';' : ':';
  const exportLine = isPs1
    ? `$env:PATH = "${dirRef};$env:PATH"`
    : isFish
      ? `fish_add_path "${dirRef}"`
      : `export PATH="${dirRef}${pathSep}$PATH"`;

  if (content.length > 0 && !content.endsWith('\n')) {
    content += '\n';
  }
  content += exportLine + '\n';

  writeShellRc(rcPath, content);
  log(`Added ${dirRef} to PATH in ${rcPath}`);
}

async function promptForApiKey(
  displayName: string,
  placeholder: string,
  validate?: (key: string) => Promise<boolean>,
  optional?: boolean,
): Promise<string | undefined> {
  while (true) {
    const prompt = optional
      ? `${displayName} (optional — press Esc to skip):`
      : `${displayName} is not set. Please enter your key:`;
    const key = await vscode.window.showInputBox({
      title: displayName,
      prompt,
      placeHolder: placeholder,
      ignoreFocusOut: true,
    });

    if (key === undefined) {
      if (!optional) {
        const choice = await showWarningNotification(
          `${displayName} is required for KISS Sorcar to function.`,
          'Enter Key',
          'Skip',
        );
        if (choice === 'Enter Key') {
          continue;
        }
      }
      return undefined;
    }

    const trimmed = key.trim();
    if (!trimmed) {
      continue;
    }

    if (validate) {
      const valid = await withWebviewNotificationProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Validating ${displayName}...`,
        },
        () => validate(trimmed),
      );

      if (!valid) {
        const choice = await showWarningNotification(
          `The ${displayName} is not valid. Please try again.`,
          'Try Again',
          'Cancel',
        );
        if (choice !== 'Try Again') {
          return undefined;
        }
        continue;
      }
    }

    return trimmed;
  }
}

function loadApiKeysFromShellRc(): void {
  const rcPath = getShellRcPath();
  const content = readShellRc(rcPath);
  if (!content) return;

  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  const pattern = isPs1
    ? /^\s*\$env:(\w+)\s*=\s*(.+)$/gm
    : isFish
      ? /^\s*set\s+-gx\s+(\w+)\s+(.+)$/gm
      : /^\s*export\s+(\w+)=(.+)$/gm;

  let match;
  while ((match = pattern.exec(content)) !== null) {
    const name = match[1];
    let value = match[2].trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    if (name && value && !process.env[name]) {
      process.env[name] = value;
    }
  }
}

// Prompt-then-save of an API key is a read-modify-write of the shell rc
// shared by every window, so it reuses the restart-lock pattern: the
// file is exclusively created, stamped with pid+token, and broken only
// when its owner is provably gone (see acquireDaemonRestartLock).
const API_KEYS_LOCK_FILE = path.join(LOG_DIR, '.api-keys.lock');
// How long a window without the lock waits for the prompting window to
// finish before giving up and reporting whatever keys exist by then.
const API_KEYS_LOCK_WAIT_MS = 600_000;

export async function ensureApiKeys(
  lockFile: string = API_KEYS_LOCK_FILE,
): Promise<boolean> {
  loadApiKeysFromShellRc();

  const keys = [
    {
      envName: 'ANTHROPIC_API_KEY',
      displayName: 'Anthropic API Key',
      placeholder: 'sk-ant-...',
      validate: validateAnthropicKey,
    },
    {
      envName: 'OPENAI_API_KEY',
      displayName: 'OpenAI API Key',
      placeholder: 'sk-...',
    },
  ];

  const hasClaudeCli = commandExists('claude');
  const hasAnyKey = () =>
    hasClaudeCli || keys.some(k => !!process.env[k.envName]);

  if (hasAnyKey()) return true;

  const releaseLock = acquireDaemonRestartLock(lockFile);
  if (!releaseLock) {
    // Another window is already prompting. Prompting here too would
    // race it: both windows would read the same shell rc, each append
    // its own key line, and the second write would drop the first.
    // Wait for that window to finish, then use whatever it saved.
    log('another window is prompting for API keys — waiting for it');
    const deadline = Date.now() + API_KEYS_LOCK_WAIT_MS;
    while (fs.existsSync(lockFile) && Date.now() < deadline) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    loadApiKeysFromShellRc();
    return hasAnyKey();
  }
  try {
    // Re-check under the lock: the window that held it before us may
    // have saved a key while we were waiting to create the lock file.
    loadApiKeysFromShellRc();
    if (hasAnyKey()) return true;

    const markerPath = path.join(LOG_DIR, '.api-keys-prompted');
    const alreadyPrompted = fs.existsSync(markerPath);
    const rcPath = getShellRcPath();

    while (true) {
      for (const {envName, displayName, placeholder, validate} of keys) {
        // A prompt can sit open for minutes; keys saved elsewhere in
        // the meantime (e.g. by `sorcar` in a terminal) make the
        // remaining prompts unnecessary, so re-read before each one.
        loadApiKeysFromShellRc();
        if (process.env[envName]) continue;
        if (hasAnyKey() && alreadyPrompted) break;

        const key = await promptForApiKey(
          displayName,
          placeholder,
          validate,
          true,
        );
        if (key) {
          process.env[envName] = key;
          addToShellRc(rcPath, envName, key);
          log(`${displayName} saved to ~/${path.basename(rcPath)}`);
        }
      }

      if (hasAnyKey()) break;

      const choice = await showWarningNotification(
        'KISS Sorcar requires Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY to work.',
        'Enter Key',
        'Skip',
      );
      if (choice !== 'Enter Key') break;
    }

    if (!alreadyPrompted) {
      try {
        fs.mkdirSync(LOG_DIR, {recursive: true});
        fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
        log('API key prompt marker written');
      } catch {}
    }

    return hasAnyKey();
  } finally {
    releaseLock();
  }
}

function readKissConfigOnce():
  | {
      ok: true;
      value: Record<string, unknown>;
    }
  | {
      ok: false;
      reason: 'missing' | 'empty' | 'parse' | 'shape' | 'io';
      err?: unknown;
    } {
  const configPath = path.join(LOG_DIR, 'config.json');
  let raw: string;
  try {
    raw = fs.readFileSync(configPath, 'utf-8');
  } catch (err) {
    const code = (err as NodeJS.ErrnoException | undefined)?.code;
    if (code === 'ENOENT') {
      return {ok: false, reason: 'missing', err};
    }
    return {ok: false, reason: 'io', err};
  }
  if (!raw.trim()) {
    return {ok: false, reason: 'empty'};
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    return {ok: false, reason: 'parse', err};
  }
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    return {ok: true, value: parsed as Record<string, unknown>};
  }
  return {ok: false, reason: 'shape'};
}

function readKissConfig(): Record<string, unknown> {
  const configPath = path.join(LOG_DIR, 'config.json');
  const RETRIES = 5;
  const BACKOFF_MS = 100;
  let last: ReturnType<typeof readKissConfigOnce> = {ok: false, reason: 'io'};
  for (let attempt = 0; attempt < RETRIES; attempt++) {
    last = readKissConfigOnce();
    if (last.ok) {
      return last.value;
    }
    if (last.reason === 'missing') {
      log(`readKissConfig: ${configPath} does not exist`);
      return {};
    }
    if (attempt < RETRIES - 1) {
      sleepSync(BACKOFF_MS);
    }
  }
  if (last.reason === 'empty') {
    log(
      `readKissConfig: ${configPath} exists but is empty after ${RETRIES} retries`,
    );
  } else if (last.reason === 'parse') {
    log(
      `readKissConfig: failed to parse ${configPath} after ${RETRIES} retries: ${
        last.err instanceof Error ? last.err.message : String(last.err)
      }`,
    );
  } else if (last.reason === 'shape') {
    log(`readKissConfig: ${configPath} parsed but not a plain object`);
  } else {
    log(
      `readKissConfig: failed to read ${configPath} after ${RETRIES} retries: ${
        last.err instanceof Error ? last.err.message : String(last.err)
      }`,
    );
  }
  return {};
}

function writeKissConfig(cfg: Record<string, unknown>): void {
  fs.mkdirSync(LOG_DIR, {recursive: true});
  const target = path.join(LOG_DIR, 'config.json');
  const tmp = path.join(
    LOG_DIR,
    `.config.json.${process.pid}.${Date.now()}.tmp`,
  );
  fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2) + '\n');
  fs.renameSync(tmp, target);
}

function getStoredRemotePassword(): string {
  const cfg = readKissConfig();
  const existing = cfg['remote_password'];
  if (typeof existing === 'string' && existing.length > 0) {
    return existing;
  }
  return '';
}

async function ensureRemotePassword(): Promise<void> {
  if (getStoredRemotePassword()) {
    log('ensureRemotePassword: password already set — skipping prompt');
    return;
  }

  log(
    'ensureRemotePassword: password not found on first read — retrying after 2 s',
  );
  await new Promise(resolve => setTimeout(resolve, 2000));

  if (getStoredRemotePassword()) {
    log('ensureRemotePassword: password found on retry — skipping prompt');
    return;
  }

  log('ensureRemotePassword: password still empty — prompting user');
  const password = await vscode.window.showInputBox({
    title: 'KISS Sorcar — Remote Access Password',
    prompt:
      'Set a password for the KISS Sorcar web / mobile app (press Esc to skip):',
    placeHolder: 'Enter a password',
    password: true,
    ignoreFocusOut: true,
  });

  if (password === undefined || password.trim() === '') {
    showInformationNotification(
      'KISS Sorcar: You can set the remote access password later in the ' +
        'KISS Sorcar settings panel (Remote password field).',
    );
    return;
  }

  const cfg = readKissConfig();
  cfg['remote_password'] = password.trim();
  writeKissConfig(cfg);
  log(`Remote access password saved to ${path.join(LOG_DIR, 'config.json')}`);
}

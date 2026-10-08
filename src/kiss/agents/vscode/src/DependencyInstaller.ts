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
import {exec, execFile, execSync, spawn} from 'child_process';
import {commandExists, findKissProject, findUvPath} from './kissPaths';
import {
  probeDaemonHealth,
  daemonHasActiveTasks,
  decideRestart,
  sleep,
} from './daemonHealth';
import {verifyDaemonStartup} from './daemonRestartVerify';
import {restartLaunchAgent} from './macLaunchd';
import {kissHomeDir, readLocalEndpoint, sorcarEndpointPath} from './userAssets';
import {
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
  withWebviewNotificationProgress,
} from './WebviewNotifications';
import {PRODUCT_NAME} from './brand';

const HOME_DIR = process.env.HOME || process.env.USERPROFILE || '';
// The daemon resolves its state directory from $KISS_HOME (see
// kiss/core/config.py), so everything the extension shares with it —
// the endpoint file (sorcar-local.json), config.json, markers, logs —
// must live under the same root.
const LOG_DIR = kissHomeDir();
const LOG_FILE = path.join(LOG_DIR, 'install.log');

// Written when a kiss-web restart is needed (the installed kiss_project no
// longer matches the fingerprint the running daemon was started with) but
// had to be deferred because tasks were in flight.  While it exists,
// ensureDependencies() skips its "nothing to do" fast path so the next
// activation re-evaluates the restart, and a timer retries in-process.
// Without it a deferred restart was never retried: the .extension-updated
// marker that forces the full check is consumed before the restart runs, so
// the daemon kept old code until VS Code happened to find it dead.
// The record is shared by every window; a window whose fingerprint already
// matches the running daemon clears it (the writer's own retry timer still
// re-evaluates), and a verified restart clears it.
const RESTART_PENDING_FILE = path.join(LOG_DIR, '.kiss-web.restart-pending');
const RESTART_RETRY_MS = Number(process.env.KISS_RESTART_RETRY_MS) || 60_000;
let restartRetryTimer: ReturnType<typeof setTimeout> | undefined;

// Synchronous probes run on the extension-host event loop, so they must
// never wait on a hung child (e.g. a PATH entry on a stalled mount).
const SYNC_PROBE_TIMEOUT_MS = 5_000;

// Ceiling for one install step (`uv sync`, Playwright downloads).  Even
// a slow first-time install finishes well inside it; without a ceiling
// a stalled child left setup and its progress toast pending for ever.
const INSTALL_STEP_TIMEOUT_MS = 30 * 60_000;

const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 13;
// Keep at the newest release (https://github.com/astral-sh/uv/releases):
// releases before 0.11.15 carry GHSA-4gg8-gxpx-9rph (arbitrary file write
// through entry-point names) and GHSA-pjjw-68hj-v9mw (file deletion through
// RECORD entries).  Dockerfile pins the same version.
const UV_VERSION = '0.12.19';

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
 * followed.  Transport for the SHA-256 manifest fetcher below.
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

function spawnCollect(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    timeoutMs?: number;
    /**
     * Run the child in its own process group (POSIX) so that on timeout
     * the whole tree -- e.g. `uv sync` and the build backends it spawned
     * -- is killed, not just the direct child.
     */
    killGroup?: boolean;
    /**
     * The setup's cancellation token: cancelling kills the child (and
     * its group, see killGroup) and rejects with code 'ECANCELLED'.
     */
    token?: vscode.CancellationToken;
  },
): Promise<{code: number | null; stdout: string; stderr: string}> {
  return new Promise((resolve, reject) => {
    const ownGroup = !!opts.killGroup && process.platform !== 'win32';
    const proc = spawn(cmd, args, {
      cwd: opts.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: opts.env,
      detached: ownGroup,
    });
    let stdout = '';
    let stderr = '';
    const killTree = () => {
      if (ownGroup && proc.pid) {
        try {
          process.kill(-proc.pid, 'SIGKILL');
        } catch {}
      }
      proc.kill('SIGKILL');
    };
    const timer = opts.timeoutMs
      ? setTimeout(() => {
          killTree();
          const err: NodeJS.ErrnoException = new Error(
            `${cmd} ${args.join(' ')} timed out after ${opts.timeoutMs}ms`,
          );
          err.code = 'ETIMEDOUT';
          reject(err);
        }, opts.timeoutMs)
      : undefined;
    // Guarded like the other optional host APIs (test stubs may pass a
    // bare token without the event).
    const cancelSub =
      typeof opts.token?.onCancellationRequested === 'function'
        ? opts.token.onCancellationRequested(() => {
            killTree();
            reject(new SetupCancelledError());
          })
        : undefined;
    proc.stdout?.on('data', (d: Buffer) => {
      stdout += d.toString();
    });
    proc.stderr?.on('data', (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on('close', code => {
      if (timer) clearTimeout(timer);
      cancelSub?.dispose();
      resolve({code, stdout, stderr});
    });
    proc.on('error', err => {
      if (timer) clearTimeout(timer);
      cancelSub?.dispose();
      reject(err);
    });
  });
}

/**
 * Thrown when the user cancels the first-run setup from its progress
 * notification: every step checks the token between commands, and a
 * running command is killed (spawnCollect).
 */
export class SetupCancelledError extends Error {
  code = 'ECANCELLED';
  constructor() {
    super('Setup was cancelled.');
  }
}

/** Throw SetupCancelledError when *token* has been cancelled. */
function throwIfCancelled(token: vscode.CancellationToken | undefined): void {
  if (token?.isCancellationRequested) throw new SetupCancelledError();
}

const CAUSE_MAX_CHARS = 200;

/**
 * The one line of a failed command's output that explains the failure,
 * for the error toast: the first line mentioning an error, a missing
 * file or a denied permission, else the first non-empty line — capped
 * at CAUSE_MAX_CHARS.  The full output goes to the log, never the toast.
 */
export function summarizeFailure(output: string): string {
  const lines = output
    .split(/\r?\n/)
    .map(l => l.trim())
    .filter(l => l !== '');
  const cause =
    lines.find(l => /error|failed|not found|no such|denied|cannot/i.test(l)) ||
    lines[0] ||
    'no output';
  return cause.length > CAUSE_MAX_CHARS
    ? cause.slice(0, CAUSE_MAX_CHARS - 1) + '…'
    : cause;
}

async function spawnPromise(
  cmd: string,
  args: string[],
  cwd?: string,
  timeoutMs = 300_000,
): Promise<string> {
  const r = await spawnCollect(cmd, args, {cwd, timeoutMs});
  if (r.code === 0) return r.stdout.trim();
  log(`${cmd} ${args.join(' ')} exited ${r.code}\n${r.stderr}`);
  throw new Error(
    `${cmd} ${args.join(' ')} exited ${r.code}: ` +
      summarizeFailure(r.stderr || r.stdout),
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

/** The default model implied by the API keys in the environment, if any. */
function envDefaultModel(): string | null {
  const env = process.env;
  if (env.ANTHROPIC_API_KEY) return 'claude-opus-4-7';
  if (env.OPENAI_API_KEY) return 'gpt-5.6-luna';
  if (env.GEMINI_API_KEY) return 'gemini-3.6-flash';
  if (env.OPENROUTER_API_KEY) return 'openrouter/anthropic/claude-opus-4.7';
  if (env.TOGETHER_API_KEY) return 'moonshotai/Kimi-K3';
  return null;
}

async function commandExistsAsync(cmd: string): Promise<boolean> {
  try {
    const r = await spawnCollect(
      process.platform === 'win32' ? 'where' : 'which',
      [cmd],
      {timeoutMs: 2_000},
    );
    return r.code === 0;
  } catch {
    return false;
  }
}

/**
 * Default model when KISS's own catalog cannot be consulted: the
 * environment's API keys, then an installed Claude Code / Codex CLI.
 */
export async function getFallbackDefaultModel(): Promise<string> {
  const fromEnv = envDefaultModel();
  if (fromEnv) return fromEnv;
  if (await commandExistsAsync('claude')) return 'cc/opus';
  if (await commandExistsAsync('codex')) return 'codex/default';
  return 'No model';
}

let lastResolvedDefaultModel: string | null = null;
let defaultModelInFlight: Promise<string> | null = null;

/**
 * A default model available synchronously WITHOUT spawning anything:
 * the most recent `resolveDefaultModel()` result, else the model implied
 * by the environment's API keys, else "No model".  Callers on the
 * extension-host event loop (view constructors, activation) start from
 * this and adopt `resolveDefaultModel()`'s answer when it arrives.
 */
export function provisionalDefaultModel(): string {
  return lastResolvedDefaultModel ?? envDefaultModel() ?? 'No model';
}

/**
 * Resolve the default model asynchronously: KISS's `get_default_model()`
 * via `uv run` (15s deadline), falling back to
 * `getFallbackDefaultModel()`.  Concurrent callers share one lookup.
 */
export function resolveDefaultModel(): Promise<string> {
  if (!defaultModelInFlight) {
    defaultModelInFlight = resolveDefaultModelImpl()
      .then(model => {
        lastResolvedDefaultModel = model;
        return model;
      })
      .finally(() => {
        defaultModelInFlight = null;
      });
  }
  return defaultModelInFlight;
}

async function resolveDefaultModelImpl(): Promise<string> {
  const uvPath = findUvPath();
  const kissProject = findKissProject();
  if (uvPath && kissProject) {
    try {
      const out = await spawnPromise(
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
        undefined,
        15_000,
      );
      if (out) return out;
    } catch {}
  }
  return getFallbackDefaultModel();
}

/**
 * The last setup steps shared by the fast and the slow path: CLI
 * wrapper, model catalog, cloudflared, kiss-web restart, shell PATH,
 * API-key and remote-password prompts.
 *
 * @param token The slow path's cancellation token: checked right before
 *     every side effect and before reporting success, so a Cancel
 *     pressed while a step's message is showing stops the setup at that
 *     step (SetupCancelledError) instead of restarting the daemon or
 *     prompting for credentials.  The fast path passes none.
 * @returns whether an API key (or the Claude CLI) is available.
 */
async function runFinalization(
  progress: vscode.Progress<{message?: string; increment?: number}> | null,
  kissProjectPath: string,
  uvPath: string | null,
  token?: vscode.CancellationToken,
): Promise<boolean> {
  throwIfCancelled(token);
  if (uvPath) {
    if (progress) progress.report({message: 'Installing CLI wrapper...'});
    installCliScript(kissProjectPath, uvPath);
  }

  // Refresh the user-local model catalog from the freshly installed
  // bundle: an installed KISS Sorcar reads $KISS_HOME/MODEL_INFO.json at
  // runtime (kiss.core.models.model_info._select_catalog_path), and the
  // settings panel's "Update Models" button updates that copy in place.
  // INJECTIONS.md stays bundled-only (user overrides live in
  // MY_MODELS.json and MY_INJECTION.md).
  try {
    const modelInfoSrc = path.join(
      kissProjectPath,
      'src',
      'kiss',
      'core',
      'models',
      'MODEL_INFO.json',
    );
    const modelInfoDst = path.join(kissHomeDir(), 'MODEL_INFO.json');
    fs.mkdirSync(kissHomeDir(), {recursive: true});
    fs.copyFileSync(modelInfoSrc, modelInfoDst);
    log(`Copied MODEL_INFO.json to ${modelInfoDst}`);
  } catch (err) {
    log(
      `Failed to copy MODEL_INFO.json into ${kissHomeDir()}: ` +
        `${err instanceof Error ? err.message : err}`,
    );
  }

  if (progress) progress.report({message: 'Checking cloudflared...'});
  throwIfCancelled(token);
  await installCloudflaredIfNeeded();

  if (progress) progress.report({message: 'Restarting kiss-web daemon...'});
  throwIfCancelled(token);
  const webWorkDir =
    vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || kissProjectPath;
  await restartKissWebDaemon(kissProjectPath, webWorkDir);

  if (progress) progress.report({message: 'Updating shell PATH...'});
  throwIfCancelled(token);
  try {
    const rcPath = getShellRcPath();
    const localBin = path.join(HOME_DIR, '.local', 'bin');
    ensurePathInShellRc(rcPath, localBin);
    if (process.platform === 'win32') {
      const gitCmdDir = path.join(HOME_DIR, '.local', 'git', 'cmd');
      if (fs.existsSync(gitCmdDir)) {
        ensurePathInShellRc(rcPath, gitCmdDir);
      }
    }
  } catch (err) {
    log(
      `Failed to update shell rc PATH: ${err instanceof Error ? err.message : err}`,
    );
  }

  if (progress) progress.report({message: 'Checking API keys...'});
  throwIfCancelled(token);
  const apiKeysReady = await ensureApiKeys();

  if (progress) progress.report({message: 'Checking remote password...'});
  throwIfCancelled(token);
  await ensureRemotePassword(uvPath, kissProjectPath);

  // A Cancel pressed during the last prompt must not end in
  // "Installation complete".
  throwIfCancelled(token);
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
      `${PRODUCT_NAME}: Could not find the KISS project directory. ` +
        'Please set "kissSorcar.kissProjectPath" in VS Code settings. ' +
        `See ${path.join(LOG_DIR, 'install.log')} for details.`,
    );
    return;
  }
  log(`KISS project: ${kissProjectPath}`);

  const updateMarker = path.join(LOG_DIR, '.extension-updated');
  const uvPath = findUvPath();
  let venvExists = fs.existsSync(path.join(kissProjectPath, '.venv'));
  if (
    uvPath &&
    venvExists &&
    isChromiumInstalled() &&
    (await isDaemonRunning()) &&
    !fs.existsSync(updateMarker) &&
    !fs.existsSync(RESTART_PENDING_FILE)
  ) {
    log('All dependencies satisfied and daemon running — nothing to do');
    log('=== Dependency check finished ===');
    loadApiKeysFromShellRc();
    return;
  }

  if (uvPath && venvExists) {
    const pyStatus = await checkPythonVersion(uvPath, kissProjectPath);
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
            `${PRODUCT_NAME}: Chromium browser update failed in background. ` +
              `See ${path.join(LOG_DIR, 'install.log')} for details.`,
          );
        }
      });
    if (!(await gitWorks())) {
      void installGit().then(installed => {
        if (!installed) {
          showWarningNotification(
            `${PRODUCT_NAME}: git is not available. ${gitInstallHint()}`,
          );
        }
      });
    }
    if (!commandExists('code')) {
      void installCodeCli();
    }
    apiKeysReady = await runFinalization(null, kissProjectPath, uvPath);
  } else {
    // The first-run install can take minutes (uv sync, Chromium).  It
    // is cancellable from the progress toast: the token is checked
    // between steps and handed to the long-running commands, which are
    // killed on cancel.  A cancelled setup says how to start it again.
    let result: {success: boolean; apiKeysReady: boolean};
    try {
      result = await withWebviewNotificationProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `${PRODUCT_NAME}: Setting up`,
          cancellable: true,
        },
        (progress, token) =>
          runSlowPathSetup(
            progress,
            token,
            kissProjectPath,
            uvPath,
            venvExists,
          ),
      );
    } catch (err) {
      if (!(err instanceof SetupCancelledError)) throw err;
      log('Setup cancelled by the user');
      void showInformationNotification(
        `${PRODUCT_NAME}: setup was cancelled. Run "KISS: Retry Setup" ` +
          'from the Command Palette to start it again.',
        'Retry setup',
      ).then(action => {
        if (action === 'Retry setup') {
          void vscode.commands.executeCommand('kissSorcar.retrySetup');
        }
      });
      return;
    }

    showRestartNotification = !!result.success;
    apiKeysReady = result.apiKeysReady;
  }

  log('=== Dependency check finished ===');

  if (showRestartNotification) {
    if (apiKeysReady) {
      showInformationNotification(`${PRODUCT_NAME}: Installation complete!`);
    } else {
      void showWarningNotification(
        `${PRODUCT_NAME}: Installation complete, but at least one of Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY is required.`,
        'Enter API key',
      ).then(action => {
        if (action === 'Enter API key') {
          void vscode.commands.executeCommand('kissSorcar.enterApiKey');
        }
      });
    }
  }
}

/**
 * The slow-path install steps (no uv and/or no .venv yet), run inside
 * the cancellable "Setting up" progress notification.
 *
 * @returns success=false when a prerequisite is missing (the user was
 *     told what to install); apiKeysReady from runFinalization.
 * @throws SetupCancelledError when the user cancels; any other error
 *     from a failing command, its message already summarised for the
 *     failure toast (the full output is in install.log).
 */
async function runSlowPathSetup(
  progress: vscode.Progress<{message?: string; increment?: number}>,
  token: vscode.CancellationToken,
  kissProjectPath: string,
  uvPath: string | null,
  venvExists: boolean,
): Promise<{success: boolean; apiKeysReady: boolean}> {
  if (!uvPath) {
    if (process.platform !== 'win32') {
      for (const bin of ['curl', 'tar']) {
        if (!commandExists(bin)) {
          showErrorNotification(
            `${PRODUCT_NAME}: '${bin}' is required to install uv but was not found. Please install '${bin}' and restart VS Code.`,
          );
          return {success: false, apiKeysReady: false};
        }
      }
    }
    progress.report({
      message: 'Installing uv package manager...',
      increment: 0,
    });
    uvPath = await installUv(token);
    if (!uvPath) {
      // uv's official one-liners: PowerShell on Windows, curl | sh
      // elsewhere.
      const manual =
        process.platform === 'win32'
          ? 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
          : 'curl -LsSf https://astral.sh/uv/install.sh | sh';
      showErrorNotification(
        `${PRODUCT_NAME}: Failed to install uv. Install manually: ${manual}`,
      );
      return {success: false, apiKeysReady: false};
    }
    progress.report({increment: 20});
  }

  throwIfCancelled(token);
  if (!(await gitWorks())) {
    progress.report({message: 'Installing git...'});
    const gitInstalled = await installGit(token);
    if (!gitInstalled) {
      showWarningNotification(
        `${PRODUCT_NAME}: git could not be installed automatically. ${gitInstallHint()}`,
      );
    }
  }

  throwIfCancelled(token);
  if (!commandExists('code')) {
    progress.report({message: 'Setting up VS Code CLI...'});
    const codeInstalled = await installCodeCli(token);
    if (!codeInstalled) {
      log('VS Code CLI could not be set up on PATH');
    }
  }

  throwIfCancelled(token);
  if (!venvExists) {
    progress.report({
      message:
        'Setting up Python environment (first time, may take a minute)...',
    });
    await runAsync(uvPath, ['sync'], kissProjectPath, token);
    progress.report({increment: 50});
  }

  throwIfCancelled(token);
  if ((await checkPythonVersion(uvPath, kissProjectPath)) !== 'ok') {
    showErrorNotification(
      `${PRODUCT_NAME} requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+. ` +
        `Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or later and restart VS Code.`,
    );
    return {success: false, apiKeysReady: false};
  }

  progress.report({message: 'Installing dependencies...'});
  await runAsync(
    uvPath,
    ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
    kissProjectPath,
    token,
  );
  if (process.platform === 'linux') {
    await runAsync(
      uvPath,
      ['run', 'python', '-m', 'playwright', 'install-deps', 'chromium'],
      kissProjectPath,
      token,
    ).catch(err => {
      if (err instanceof SetupCancelledError) throw err;
      log(
        `Playwright deps install failed (may need sudo): ${err instanceof Error ? err.message : err}`,
      );
    });
  }
  progress.report({increment: 30});

  throwIfCancelled(token);
  progress.report({message: 'Finalizing setup...'});
  const finalizedKeys = await runFinalization(
    progress,
    kissProjectPath,
    uvPath,
    token,
  );
  return {success: true, apiKeysReady: finalizedKeys};
}

/**
 * PIDs of kiss-web daemons listening on TCP *port*, via `lsof` and
 * `ps`; empty when none or when either tool fails.  Asynchronous: a
 * slow `lsof` must not freeze the extension host for its 3s deadline
 * on every poll.
 *
 * Only processes whose command line names `kiss-web` are returned.
 * Other listeners share the port legitimately: on macOS a VS Code
 * Remote window's extension host binds `127.0.0.1:8787` when it
 * forwards a remote kiss-web, and signalling it would tear down that
 * whole window.
 */
export async function pidsOnPort(port: number): Promise<string[]> {
  if (process.platform === 'win32') return windowsDaemonPidsOnPort(port);
  try {
    const r = await spawnCollect(
      'lsof',
      ['-ti', `tcp:${port}`, '-sTCP:LISTEN'],
      {timeoutMs: 3000},
    );
    if (r.code !== 0) return [];
    const pids = r.stdout.trim().split('\n').filter(Boolean);
    if (pids.length === 0) return [];
    const ps = await spawnCollect(
      'ps',
      ['-o', 'pid=,command=', '-p', pids.join(',')],
      {timeoutMs: 3000},
    );
    if (ps.code !== 0) return [];
    // A whole argv token ending in `kiss-web` (the `.venv/bin/kiss-web`
    // entry point that launchd, systemd and spawnKissWebDirect run), not
    // any substring: `--directory /x/kiss-web-client` is not a daemon.
    const kissWebPids: string[] = [];
    for (const line of ps.stdout.split('\n')) {
      const m = /^\s*(\d+)\s+(.*)$/.exec(line);
      if (m && pids.includes(m[1]) && /(^|[\s/])kiss-web(\s|$)/.test(m[2])) {
        kissWebPids.push(m[1]);
      }
    }
    return kissWebPids;
  } catch {
    return [];
  }
}

/**
 * Windows: the kiss-web pid listening on *port*, or empty.
 *
 * Windows has no `lsof`.  `netstat -ano` names the pid that owns the
 * LISTENING socket, and `tasklist` names its image; a pid counts only
 * when both agree with the daemon's own endpoint file (which records
 * the pid that bound the port) and the image is a Python / kiss-web
 * executable.  A stale endpoint file whose pid was reused by an
 * unrelated process therefore never selects that process, and a VS
 * Code Remote port forward (Code.exe) is never mistaken for the daemon.
 */
async function windowsDaemonPidsOnPort(port: number): Promise<string[]> {
  const endpoint = readLocalEndpoint(sorcarEndpointPath());
  if (!endpoint || !endpoint.pid) return [];
  let url: URL;
  try {
    url = new URL(endpoint.url);
  } catch {
    return [];
  }
  if (Number(url.port) !== port) return [];
  const pid = String(endpoint.pid);
  try {
    const ns = await spawnCollect('netstat', ['-ano', '-p', 'tcp'], {
      timeoutMs: 5000,
    });
    if (ns.code !== 0) return [];
    const listening = ns.stdout.split(/\r?\n/).some(line => {
      const m = /^\s*TCP\s+\S+:(\d+)\s+\S+\s+LISTENING\s+(\d+)\s*$/.exec(line);
      return m !== null && Number(m[1]) === port && m[2] === pid;
    });
    if (!listening) return [];
    const tl = await spawnCollect(
      'tasklist',
      ['/FI', `PID eq ${pid}`, '/FO', 'CSV', '/NH'],
      {timeoutMs: 5000},
    );
    if (tl.code !== 0) return [];
    const image = /^"([^"]*)"/.exec(tl.stdout.trim());
    if (!image || !/^(kiss-web|python[\w.]*)\.exe$/i.test(image[1])) return [];
    return [pid];
  } catch {
    return [];
  }
}

function killPids(pids: string[], signal: NodeJS.Signals): void {
  for (const pid of pids) {
    try {
      if (process.platform === 'win32') {
        // Node's process.kill is TerminateProcess on Windows whatever
        // the signal; `taskkill /T` also takes the daemon's children
        // (agent subprocesses) so none is left holding the port.
        spawn('taskkill', ['/PID', pid, '/T', '/F'], {
          stdio: 'ignore',
          windowsHide: true,
        }).unref();
      } else {
        process.kill(parseInt(pid, 10), signal);
      }
    } catch {}
  }
}

/** Where `kiss-web` lives in the bundled project's venv. */
export function kissWebBinPath(kissProjectPath: string): string {
  return process.platform === 'win32'
    ? path.join(kissProjectPath, '.venv', 'Scripts', 'kiss-web.exe')
    : path.join(kissProjectPath, '.venv', 'bin', 'kiss-web');
}

/**
 * SIGTERM whatever listens on *port* and wait (asynchronously) for it to
 * go away, escalating to SIGKILL after 3s.
 *
 * The wait between polls must not block: this runs on the extension
 * host's event loop, and a synchronous 3s sleep froze every other
 * extension, the daemon client's socket and the webview for the whole
 * of the old daemon's shutdown.
 */
async function killProcessOnPort(port: number): Promise<void> {
  const pids = await pidsOnPort(port);
  if (pids.length === 0) return;
  killPids(pids, 'SIGTERM');
  for (let i = 0; i < 6; i++) {
    if ((await pidsOnPort(port)).length === 0) return;
    await sleep(500);
  }
  killPids(await pidsOnPort(port), 'SIGKILL');
}

async function systemctlRestartKissWeb(): Promise<void> {
  await spawnPromise(
    'systemctl',
    ['--user', 'restart', '--no-block', 'kiss-web'],
    undefined,
    10000,
  );
}

/**
 * Start kiss-web as a detached background process.
 *
 * The fallback when no service manager runs it (systemd unavailable on
 * Linux) and the only supervisor on Windows, where the extension's
 * activation check and restart retry take the place of `Restart=always`:
 * a daemon that dies is started again the next time a window activates
 * or a restart is due.  The child outlives the extension host
 * (`detached`, `unref`) and, on Windows, gets no console window.
 */
function spawnKissWebDirect(kissWebBin: string, workDir: string): void {
  const binDir = path.join(HOME_DIR, '.local', 'bin');
  try {
    fs.mkdirSync(LOG_DIR, {recursive: true});
    const outFd = fs.openSync(path.join(LOG_DIR, 'kiss-web-stdout.log'), 'a');
    const errFd = fs.openSync(path.join(LOG_DIR, 'kiss-web-stderr.log'), 'a');
    const inheritedPath =
      process.env.PATH ||
      (process.platform === 'win32' ? '' : '/usr/local/bin:/usr/bin:/bin');
    const child = spawn(kissWebBin, [], {
      cwd: workDir,
      detached: true,
      windowsHide: true,
      stdio: ['ignore', outFd, errFd],
      env: {
        ...process.env,
        PATH: `${binDir}${path.delimiter}${inheritedPath}`,
      },
    });
    child.unref();
    fs.closeSync(outFd);
    fs.closeSync(errFd);
    log(
      'kiss-web started directly (no service manager): pid ' +
        `${child.pid ?? '<unknown>'}, cwd ${workDir}`,
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
// published its endpoint and cannot report the active tasks that
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

/**
 * Restart the kiss-web daemon if its code changed and it is safe to.
 *
 * @param kissProjectPath The bundled kiss_project directory.
 * @param workDir The working directory kiss-web is started in.
 * @param force Restart even though the daemon reports active tasks:
 *     the user chose "Restart now" on the deferred-update notification.
 * @returns False when the restart lock could not be taken (another
 *     window holds it, or the lock file could not be created), so this
 *     call made no decision and the caller may try again; true
 *     otherwise (a decision was made, or there is nothing to restart).
 */
export async function restartKissWebDaemon(
  kissProjectPath: string,
  workDir: string,
  force = false,
): Promise<boolean> {
  const kissWebBin = kissWebBinPath(kissProjectPath);
  if (!fs.existsSync(kissWebBin)) {
    log(`kiss-web binary not found at ${kissWebBin} — skipping daemon setup`);
    return true;
  }

  const releaseLock = acquireDaemonRestartLock();
  if (!releaseLock) {
    log('another window is restarting kiss-web — skipping this one');
    return false;
  }
  try {
    await restartKissWebDaemonLocked(
      kissProjectPath,
      workDir,
      kissWebBin,
      force,
    );
  } finally {
    releaseLock();
  }
  return true;
}

async function restartKissWebDaemonLocked(
  kissProjectPath: string,
  workDir: string,
  kissWebBin: string,
  force: boolean,
): Promise<void> {
  // Checked under the lock: a forced restart answers a notification
  // that may have outlived its deferral.  If the pending record is
  // gone, a retry or another window already restarted the daemon with
  // the new code (and cleared the record under this same lock), and
  // restarting again would only abort whatever started since.
  if (force && !fs.existsSync(RESTART_PENDING_FILE)) {
    log('kiss-web update already applied — ignoring the forced restart');
    return;
  }
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
  const endpointPath = sorcarEndpointPath();

  const health = await probeDaemonHealth(8787, 1500);
  const endpointExists = fs.existsSync(endpointPath);

  // Always query the local endpoint, even when the TCP probe looks dead:
  // the task worker can be alive behind a transiently refused port, and
  // decideRestart() protects any reported active task regardless of health.
  const activeTasks:
    {ok: true; count: number; tabs: string[]} | {ok: false; reason: string} =
    await daemonHasActiveTasks(endpointPath, 1500);

  const fingerprintMatches = !!currentFp && currentFp === savedFp;
  const decision = decideRestart({
    fingerprintMatches,
    health,
    activeTasks,
    force,
  });
  if (decision.skip) {
    if (fingerprintMatches) {
      clearRestartPending();
    } else {
      markRestartPending(decision.reason, kissProjectPath, workDir);
    }
    if (decision.reason === 'active-tasks') {
      const count = (activeTasks as {ok: true; count: number}).count;
      log(
        `kiss-web has ${count} active task(s) — deferring restart to avoid ` +
          'aborting in-flight work',
      );
      if (!fingerprintMatches) {
        offerForcedRestart(count, kissProjectPath, workDir);
      }
    } else if (decision.reason.startsWith('alive-uncertain')) {
      log(
        'kiss-web alive but active-tasks probe inconclusive ' +
          `(${decision.reason}) — deferring restart to next activation`,
      );
    } else {
      log(
        `kiss-web fingerprint unchanged (${currentFp.slice(0, 8)}) and ` +
          `daemon healthy (health=${health}, endpoint=${endpointExists}) — ` +
          'skipping restart to preserve tunnel URL',
      );
    }
    return;
  }
  log(
    `kiss-web restart (${decision.reason}): fingerprint ` +
      `${savedFp.slice(0, 8) || '<none>'} → ` +
      `${currentFp.slice(0, 8) || '<none>'}, health=${health}, ` +
      `endpoint=${endpointExists}, activeTasks=` +
      `${activeTasks.ok ? activeTasks.count : 'unknown(' + activeTasks.reason + ')'}`,
  );

  await killProcessOnPort(8787);

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
      // The daemon resolves its state dir — and the endpoint file it
      // writes — from $KISS_HOME, so a KISS_HOME visible only to the VS
      // Code process must reach the launchd service too; otherwise the
      // extension probes $KISS_HOME/sorcar-local.json while the daemon
      // writes ~/.kiss/sorcar-local.json, and every health poll restarts
      // a healthy daemon.  KISS_SORCAR_LOCAL is a client-side override
      // only (the daemon does not read it), so it is NOT propagated.
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

      const uid = await spawnPromise('id', ['-u'], undefined, 5_000);
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
      // daemon writes its endpoint file under $KISS_HOME, so the service
      // must see the same value the extension host sees.
      // KISS_SORCAR_LOCAL is a client-side override only (the daemon
      // does not read it), so it is deliberately NOT propagated.
      const kissHomeEnv = process.env.KISS_HOME || '';
      const kissHomeLine = kissHomeEnv
        ? `Environment=KISS_HOME=${unitEscape(kissHomeEnv)}\n`
        : '';
      const serviceContent = `[Unit]
Description=${PRODUCT_NAME} Remote Web Server
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
      await spawnPromise(
        'systemctl',
        ['--user', 'daemon-reload'],
        undefined,
        10000,
      );
      // --no-block queues the restart job and returns immediately.  A
      // blocking restart waits for the old daemon to finish shutting
      // down, which can take longer than the 10s timeout (tunnel
      // cleanup; the daemon's SIGTERM failsafe allows 30s).  The
      // ETIMEDOUT that the restart call then threw was misread as
      // "systemd failed" and triggered the direct-spawn fallback below
      // — while systemd's restart job was still in flight — leaving
      // TWO daemons racing for port 8787 and systemd crash-looping
      // every RestartSec against the rogue's listener.
      //
      // All of these are awaited rather than run synchronously: a slow
      // systemd DBus round-trip must not freeze the extension host.
      await systemctlRestartKissWeb();
      const username = os.userInfo().username;
      try {
        await spawnPromise(
          'loginctl',
          ['enable-linger', username],
          undefined,
          5000,
        );
      } catch {}
      log(`kiss-web systemd user service restarted: ${serviceFile}`);
      systemdOk = true;
      reissueRestart = systemctlRestartKissWeb;
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
  } else {
    // Windows (and any other platform without launchd/systemd): the
    // detached spawn IS the service; the extension re-issues it when
    // the daemon is found dead.
    log('Starting kiss-web as a detached background process...');
    reissueRestart = () => spawnKissWebDirect(kissWebBin, workDir);
    void reissueRestart();
  }

  const verdict = await verifyDaemonStartup({
    binPath: kissWebBin,
    endpointPath,
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
  clearRestartPending();
}

/**
 * Record that a needed kiss-web restart was deferred and retry it later.
 *
 * Writes RESTART_PENDING_FILE (so the next activation skips the "nothing
 * to do" fast path) and arms a single RESTART_RETRY_MS timer that calls
 * restartKissWebDaemon again, so the daemon picks up the new code as soon
 * as the in-flight tasks finish, without waiting for a window reload.
 *
 * @param reason The decideRestart() reason the restart was skipped for.
 * @param kissProjectPath The bundled kiss_project directory.
 * @param workDir The working directory kiss-web is started in.
 */
function markRestartPending(
  reason: string,
  kissProjectPath: string,
  workDir: string,
): void {
  try {
    fs.writeFileSync(RESTART_PENDING_FILE, reason + '\n');
  } catch (err) {
    log(
      `Failed to write ${RESTART_PENDING_FILE}: ` +
        `${err instanceof Error ? err.message : err}`,
    );
  }
  log(
    `kiss-web restart pending (${reason}) — retrying in ` +
      `${RESTART_RETRY_MS / 1000}s`,
  );
  armRestartRetry(kissProjectPath, workDir);
}

function armRestartRetry(kissProjectPath: string, workDir: string): void {
  if (restartRetryTimer) return;
  restartRetryTimer = setTimeout(
    retryPendingRestart,
    RESTART_RETRY_MS,
    kissProjectPath,
    workDir,
  );
  restartRetryTimer.unref();
}

/**
 * Timer callback: run the full restart path again and keep retrying while
 * the pending record survives it (tasks still active, or another window
 * held the restart lock so this attempt was skipped without a decision).
 */
async function retryPendingRestart(
  kissProjectPath: string,
  workDir: string,
): Promise<void> {
  restartRetryTimer = undefined;
  try {
    await restartKissWebDaemon(kissProjectPath, workDir);
  } catch (err) {
    log(
      `kiss-web restart retry failed: ${err instanceof Error ? err.message : err}`,
    );
  }
  if (fs.existsSync(RESTART_PENDING_FILE)) {
    armRestartRetry(kissProjectPath, workDir);
  }
}

// Whether the "Restart now" notification for the current deferred
// update has been shown.  One offer per deferral: the retry timer
// re-enters the restart path every RESTART_RETRY_MS and must not
// re-raise the notification each time.
let forcedRestartOffered = false;

/**
 * Let the user break a deferral that may never end on its own.
 *
 * A pending update waits for the daemon's active tasks to finish, but
 * the daemon's report can be wrong: a tab wedged on a stale busy claim
 * (2026-09-22, ~/.kiss/kiss-web-stderr.log) counted as active for hours
 * and deferred the very restart that would have loaded the fix.  The
 * notification says what the update is waiting for and lets the user
 * decide to restart anyway, accepting that any task really running in
 * the daemon is aborted.
 *
 * @param count Active tasks the daemon reported.
 * @param kissProjectPath The bundled kiss_project directory.
 * @param workDir The working directory kiss-web is started in.
 */
function offerForcedRestart(
  count: number,
  kissProjectPath: string,
  workDir: string,
): void {
  if (forcedRestartOffered) return;
  forcedRestartOffered = true;
  void showWarningNotification(
    `${PRODUCT_NAME}: the kiss-web update is waiting for ${count} running ` +
      'task(s) to finish and retries every minute. If no task is ' +
      'actually running, the daemon is stuck on a stale task and only a ' +
      'restart will clear it. Restarting now aborts any task that IS ' +
      'running.',
    'Restart now',
    'Keep waiting',
  ).then(choice => {
    if (choice === undefined) {
      // No answer: the toast was dismissed, or its webview poster was
      // replaced (a re-created sidebar resolves every pending toast
      // with undefined).  Unlatch so the next retry can offer again.
      forcedRestartOffered = false;
      return;
    }
    if (choice !== 'Restart now') return;
    log('user chose to restart kiss-web despite reported active tasks');
    forceRestartKissWebDaemon(kissProjectPath, workDir).catch(err => {
      // The unattended retry catches its own failures; a click that
      // fails must not leave the offer latched, or the user is never
      // asked again while the retry keeps deferring.
      log(
        `forced kiss-web restart failed: ${err instanceof Error ? err.message : err}`,
      );
      forcedRestartOffered = false;
    });
  });
}

// How long a "Restart now" click keeps trying to take the restart lock
// before giving up: as long as a live lock holder is honoured
// (RESTART_LOCK_MAX_HOLD_MS), because that is how long a legitimate
// restart in another window can take.
const FORCED_RESTART_RETRY_MS = 2_000;

/**
 * Carry out the user's "Restart now" choice.
 *
 * The click is not dropped while the restart lock is busy: this
 * window's own retry timer holds it for the few seconds its probes
 * take, and another window's restart may hold it for minutes, and in
 * both cases restartKissWebDaemon() returns false without deciding
 * anything.  Keep trying until a forced attempt gets to decide, or the
 * lock's maximum hold time passes — then re-arm the offer so the user
 * can click again instead of waiting on a click that went nowhere.
 *
 * A click on a notification that outlived its deferral is ignored by
 * the locked restart path itself (see restartKissWebDaemonLocked), so
 * the check cannot race another window's completion.
 *
 * @param kissProjectPath The bundled kiss_project directory.
 * @param workDir The working directory kiss-web is started in.
 */
async function forceRestartKissWebDaemon(
  kissProjectPath: string,
  workDir: string,
): Promise<void> {
  const deadline = Date.now() + RESTART_LOCK_MAX_HOLD_MS;
  while (true) {
    if (await restartKissWebDaemon(kissProjectPath, workDir, true)) return;
    if (Date.now() > deadline) {
      log(
        'forced kiss-web restart could not take the restart lock within ' +
          `${RESTART_LOCK_MAX_HOLD_MS / 1000}s — offering it again`,
      );
      forcedRestartOffered = false;
      return;
    }
    await sleep(FORCED_RESTART_RETRY_MS);
  }
}

/**
 * Forget a pending restart: remove the record and cancel the retry timer,
 * so a later deferral arms a fresh timer with its own project/work dir
 * and may offer the forced restart again.
 */
function clearRestartPending(): void {
  try {
    fs.unlinkSync(RESTART_PENDING_FILE);
  } catch {}
  if (restartRetryTimer) {
    clearTimeout(restartRetryTimer);
    restartRetryTimer = undefined;
  }
  forcedRestartOffered = false;
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

/**
 * Follow *target*'s symlink chain by hand when `realpathSync` cannot.
 *
 * A DANGLING dotfile symlink (`.bashrc -> dotfiles/bashrc` whose target
 * is not checked out yet) makes `fs.realpathSync` throw, and renaming
 * over the link would silently turn it into a regular file.  Walking
 * `readlinkSync` — a relative link resolved against its containing
 * directory — finds the missing referent so the atomic write can create
 * it, exactly like the plain `fs.writeFileSync` this replaced used to.
 * The walk is bounded: on a link loop it gives up on a node in the
 * cycle, which the rename then replaces (the old code threw ELOOP; a
 * loop is broken garbage either way).
 *
 * @param target Path whose links to follow.
 * @returns The first non-symlink path in the chain (it may not exist).
 */
function resolveSymlinkTargetSync(target: string): string {
  // audit0903-coverage:start
  let current = target;
  for (let depth = 0; depth < 32; depth += 1) {
    let link: string;
    try {
      // Throws EINVAL for a regular file, ENOENT for a missing one.
      link = fs.readlinkSync(current);
    } catch {
      return current;
    }
    current = path.resolve(path.dirname(current), link);
  }
  return current;
  // audit0903-coverage:end
}

/**
 * Replace *target* with *content* atomically.
 *
 * `fs.writeFileSync(target, ...)` truncates the file before writing, so
 * a concurrent reader — a shell `exec`ing `~/.local/bin/sorcar`, a new
 * terminal sourcing the shell rc — can observe an empty or partial
 * file.  This writes a uniquely named temp file in the target's
 * directory and `rename(2)`s it into place, so every reader sees either
 * the old or the new content, whole.
 *
 * A symlinked target (dotfile-managed shell rc) is resolved first —
 * through `realpath`, or link by link when the chain dangles — so the
 * rename replaces (or creates) the file the link points at, never the
 * link itself; the existing file's permission bits are preserved unless
 * *mode* overrides them.
 *
 * @param target Path of the file to replace (created when missing).
 * @param content The full new file content.
 * @param mode Permission bits for the result; defaults to the existing
 *   file's bits, or the process umask default for a new file.
 */
export function writeFileAtomicSync(
  target: string,
  content: string,
  mode?: number,
): void {
  // audit0903-coverage:start
  let resolved: string;
  try {
    resolved = fs.realpathSync(target);
  } catch {
    resolved = resolveSymlinkTargetSync(target);
  }
  if (mode === undefined) {
    try {
      mode = fs.statSync(resolved).mode & 0o777;
    } catch {}
  }
  const tmp = path.join(
    path.dirname(resolved),
    `.${path.basename(resolved)}.${process.pid}.${Date.now()}.` +
      `${Math.random().toString(36).slice(2)}.tmp`,
  );
  // The whole temp lifecycle is guarded: a write that fails AFTER
  // creating the temp (the EFBIG/ENOSPC class leaves a partial file)
  // must clean up exactly like a failed chmod or rename.
  let renamed = false;
  try {
    fs.writeFileSync(tmp, content);
    if (mode !== undefined) fs.chmodSync(tmp, mode);
    fs.renameSync(tmp, resolved);
    renamed = true;
  } finally {
    if (!renamed) {
      try {
        fs.unlinkSync(tmp);
      } catch {
        // The write failed before the temp existed: nothing to clean.
      }
    }
  }
  // audit0903-coverage:end
}

/**
 * Install the `sorcar` CLI wrapper into `~/.local/bin`.
 *
 * The wrapper is an executable other processes run at any moment, so it
 * is replaced atomically (see {@link writeFileAtomicSync}) — the old
 * truncate-then-write left a window in which a user's shell `exec`ed an
 * empty or truncated script.  Exported so the e2e tests can hammer it
 * from concurrent processes.
 *
 * @param kissProjectPath Root of the kiss checkout the wrapper targets.
 * @param uvPath The uv binary the wrapper runs (made absolute here).
 */
export function installCliScript(
  kissProjectPath: string,
  uvPath: string,
): void {
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
      absUvPath = execSync(whichCmd, {
        encoding: 'utf-8',
        timeout: SYNC_PROBE_TIMEOUT_MS,
      })
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
        `REM Installed by ${PRODUCT_NAME} VS Code extension\r\n` +
        'set "KISS_WORKDIR=%CD%"\r\n' +
        `"${absUvPath}" run --directory "${kissProjectPath}" sorcar %*\r\n`;
      writeFileAtomicSync(cmdPath, script);
    } else {
      // audit0903-coverage:start
      const scriptPath = path.join(binDir, 'sorcar');
      const script =
        '#!/bin/bash\n' +
        `# Installed by ${PRODUCT_NAME} VS Code extension\n` +
        'export KISS_WORKDIR="$PWD"\n' +
        `exec "${absUvPath}" run --directory "${kissProjectPath}" sorcar "$@"\n`;
      writeFileAtomicSync(scriptPath, script, 0o755);
      // audit0903-coverage:end
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

/**
 * Download and install uv into ~/.local/bin.
 *
 * @param token Setup cancellation token, checked between the download,
 *     the hash check and the extraction; a cancel is rethrown as
 *     SetupCancelledError instead of being reported as a failed install.
 * @returns the installed uv path, or null when the install failed.
 */
async function installUv(
  token?: vscode.CancellationToken,
): Promise<string | null> {
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
      throwIfCancelled(token);
      const expectedHash = await fetchUvStyleSha256(url);
      throwIfCancelled(token);
      verifyDownloadHash(tarPath, expectedHash);
      await spawnPromise('tar', ['xzf', tarPath, '-C', installDir]);
      throwIfCancelled(token);
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
    if (err instanceof SetupCancelledError) throw err;
    log(`Failed to install uv: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

async function checkPythonVersion(
  uvPath: string,
  cwd: string,
): Promise<'ok' | 'too_old' | 'error'> {
  try {
    const output = await spawnPromise(
      uvPath,
      ['run', 'python', '--version'],
      cwd,
      30_000,
    );
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
  const endpointPath = sorcarEndpointPath();
  for (let attempt = 0; attempt < 3; attempt++) {
    const health = await probeDaemonHealth(8787);
    if (health === 'alive' && fs.existsSync(endpointPath)) return true;
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

async function gitWorks(): Promise<boolean> {
  try {
    const r = await spawnCollect('git', ['--version'], {timeoutMs: 10_000});
    return r.code === 0 && r.stdout.includes('git version');
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

/**
 * Install git with the platform's package manager (or Xcode CLT).
 *
 * @param token Setup cancellation token, checked between install
 *     attempts and on every poll of the Xcode CLT installer (up to ten
 *     minutes); a cancel throws SetupCancelledError.
 * @returns whether a working git is available afterwards.
 */
async function installGit(token?: vscode.CancellationToken): Promise<boolean> {
  log('Git not found, attempting to install...');

  if (process.platform === 'darwin') {
    if (commandExists('brew')) {
      log('Installing git via Homebrew...');
      try {
        await execPromise('brew install git');
        if (await gitWorks()) {
          log('Git installed via Homebrew');
          return true;
        }
      } catch (err) {
        log(
          `Homebrew git install failed: ${err instanceof Error ? err.message : err}`,
        );
      }
    }
    // A Cancel pressed while Homebrew ran must not open the Xcode
    // Command Line Tools installer as a fallback.
    throwIfCancelled(token);

    try {
      execSync('xcode-select -p', {
        stdio: 'ignore',
        timeout: SYNC_PROBE_TIMEOUT_MS,
      });
      log('Xcode CLT present but git not working');
      return false;
    } catch {}

    log('Triggering Xcode Command Line Tools installation...');
    try {
      execSync('xcode-select --install', {stdio: 'ignore', timeout: 5_000});
    } catch {}

    for (let i = 0; i < 120; i++) {
      await new Promise(resolve => setTimeout(resolve, 5_000));
      throwIfCancelled(token);
      if (await gitWorks()) {
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
      throwIfCancelled(token);
      if (commandExists(bin)) {
        log(`Installing git via ${bin}...`);
        try {
          await execPromise(cmd);
          if (await gitWorks()) {
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
  // git-for-windows tags a release `v<git>.windows.<n>` and names its
  // assets `MinGit-<git>[.<n>]-64-bit.zip` / `MinGit-<git>[.<n>]-arm64.zip`
  // (the `.<n>` suffix is dropped when n == 1).  Newest release:
  // https://github.com/git-for-windows/git/releases/latest
  const GIT_RELEASE_TAG = 'v2.55.0.windows.5';
  const GIT_VERSION = '2.55.0.5';
  const archSuffix = process.arch === 'arm64' ? 'arm64' : '64-bit';
  const assetName = `MinGit-${GIT_VERSION}-${archSuffix}`;
  const url = `https://github.com/git-for-windows/git/releases/download/${GIT_RELEASE_TAG}/${assetName}.zip`;
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

/**
 * Put the `code` CLI on PATH (symlink on macOS, snap/apt on Linux).
 *
 * @param token Setup cancellation token, checked between the snap and
 *     the apt attempt; a cancel throws SetupCancelledError.
 * @returns whether `code` is on PATH afterwards.
 */
async function installCodeCli(
  token?: vscode.CancellationToken,
): Promise<boolean> {
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
    throwIfCancelled(token);
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
  token?: vscode.CancellationToken,
): Promise<void> {
  const cmdLine = `${cmd} ${args.join(' ')}`;
  log(`Running: ${cmdLine}`);
  let r: {code: number | null; stdout: string; stderr: string};
  try {
    r = await spawnCollect(cmd, args, {
      cwd,
      env: {...process.env, PYTHONUNBUFFERED: '1'},
      timeoutMs: INSTALL_STEP_TIMEOUT_MS,
      killGroup: true,
      token,
    });
  } catch (err) {
    log(`Spawn error [${cmdLine}]: ${(err as Error).message}`);
    if ((err as NodeJS.ErrnoException).code === 'ETIMEDOUT') {
      throw new Error(
        `${cmdLine} did not finish within ` +
          `${INSTALL_STEP_TIMEOUT_MS / 60_000} minutes and was killed.`,
      );
    }
    throw err;
  }
  const output = r.stdout + r.stderr;
  if (output.trim()) log(`Output [${cmdLine}]:\n${output.trim()}`);
  if (r.code === 0) {
    log(`Completed: ${cmdLine}`);
    return;
  }
  // The toast gets the one-line cause; the full output is in the log
  // (written above) behind the toast's 'Open log' action.
  throw new Error(
    `${cmdLine} failed (exit code ${r.code}): ${summarizeFailure(output)}`,
  );
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

/**
 * Outcome of probing a provider with a key: 'ok', 'rejected' (the
 * provider answered and refused the key) or 'unreachable' (no answer —
 * offline, DNS, firewall — so nothing is known about the key).
 */
export type KeyValidation = 'ok' | 'rejected' | 'unreachable';

function validateAnthropicKey(key: string): Promise<KeyValidation> {
  return new Promise(resolve => {
    const headers: Record<string, string> = {
      'x-api-key': key,
      'anthropic-version': '2023-06-01',
    };
    // An identity-linked API key is rejected (400) unless the request
    // names the workspace it acts in, so a valid key would fail this
    // probe without the header.
    const workspaceId = (process.env.ANTHROPIC_WORKSPACE_ID || '').trim();
    if (workspaceId) {
      headers['anthropic-workspace-id'] = workspaceId;
    }
    const req = https.request(
      {
        hostname: 'api.anthropic.com',
        path: '/v1/models',
        method: 'GET',
        headers,
        timeout: 15000,
      },
      res => {
        resolve(res.statusCode === 200 ? 'ok' : 'rejected');
        res.resume();
      },
    );
    req.on('error', () => resolve('unreachable'));
    req.on('timeout', () => {
      req.destroy();
      resolve('unreachable');
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

/**
 * Write the user's shell rc file atomically.
 *
 * The rc file is sourced by every new shell, so a truncate-then-write
 * (`fs.writeFileSync`) risked a just-opened terminal sourcing an empty
 * or partial rc — dropping the user's PATH and API keys for that
 * session.  {@link writeFileAtomicSync} also preserves a symlinked rc
 * (dotfile repos) and its permission bits.  Exported so the e2e tests
 * can hammer it from concurrent processes.
 *
 * @param rcPath Path of the shell rc file.
 * @param content The full new rc content (newline-terminated here).
 */
export function writeShellRc(rcPath: string, content: string): void {
  // audit0903-coverage:start
  if (content.length > 0 && !content.endsWith('\n')) {
    content += '\n';
  }
  writeFileAtomicSync(rcPath, content);
  // audit0903-coverage:end
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

/**
 * Ask for one API key in a password input box and, when a validator is
 * given, probe the provider with it before accepting.  Exported so the
 * e2e tests can drive the prompt with a fake validator.
 *
 * @param displayName Human name of the key ("Anthropic API Key").
 * @param placeholder Input-box placeholder ("sk-ant-...").
 * @param validate Optional provider probe; see {@link KeyValidation}.
 * @param optional Esc skips silently instead of asking "Enter Key / Skip".
 * @param providerHost Host named in the validation messages.
 * @returns The trimmed key, or undefined when the user skipped/cancelled.
 */
export async function promptForApiKey(
  displayName: string,
  placeholder: string,
  validate?: (key: string) => Promise<KeyValidation>,
  optional?: boolean,
  providerHost = '',
): Promise<string | undefined> {
  // The value typed last time: a 'Try again' after a failed validation
  // reopens the box with it so a one-character typo is fixed, not
  // retyped from scratch.
  let lastValue = '';
  while (true) {
    const prompt = optional
      ? `${displayName} (optional — press Esc to skip):`
      : `${displayName} is not set. Please enter your key:`;
    const key = await vscode.window.showInputBox({
      title: displayName,
      prompt,
      placeHolder: placeholder,
      ignoreFocusOut: true,
      password: true,
      value: lastValue,
    });

    if (key === undefined) {
      if (!optional) {
        const choice = await showWarningNotification(
          `${displayName} is required for ${PRODUCT_NAME} to function.`,
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
    lastValue = trimmed;

    if (validate) {
      const outcome = await withWebviewNotificationProgress(
        {
          location: vscode.ProgressLocation.Notification,
          title: `Validating ${displayName}...`,
        },
        () => validate(trimmed),
      );

      if (outcome === 'unreachable') {
        // Nothing is known about the key: offer to keep it unvalidated
        // rather than calling a possibly fine key "not valid".
        const choice = await showWarningNotification(
          `Could not reach ${providerHost || 'the provider'} to validate ` +
            'the key; check your connection.',
          'Try again',
          'Save without validating',
          'Cancel',
        );
        if (choice === 'Save without validating') return trimmed;
        if (choice !== 'Try again') return undefined;
        continue;
      }
      if (outcome === 'rejected') {
        const choice = await showWarningNotification(
          `${providerHost || 'The provider'} rejected this ${displayName}. ` +
            'Check the key and try again.',
          'Try again',
          'Cancel',
        );
        if (choice !== 'Try again') {
          return undefined;
        }
        continue;
      }
    }

    return trimmed;
  }
}

function importEnvAssignments(content: string, pattern: RegExp): void {
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

function loadApiKeysFromShellRc(): void {
  // The canonical key store first: $KISS_HOME/api_keys.env is where the
  // settings panel and ./rsorcar persist keys (bash `export KEY=value`
  // syntax), and save_api_key scrubs assignments out of the shell RC, so
  // a key saved through the panel exists ONLY here.  Already-set process
  // environment variables always win.
  const canonical = readShellRc(path.join(kissHomeDir(), 'api_keys.env'));
  if (canonical) {
    importEnvAssignments(canonical, /^\s*export\s+(\w+)=(.+)$/gm);
  }

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
  importEnvAssignments(content, pattern);
}

// Prompt-then-save of an API key is a read-modify-write of the shell rc
// shared by every window, so it reuses the restart-lock pattern: the
// file is exclusively created, stamped with pid+token, and broken only
// when its owner is provably gone (see acquireDaemonRestartLock).
const API_KEYS_LOCK_FILE = path.join(LOG_DIR, '.api-keys.lock');
// How long a window without the lock waits for the prompting window to
// finish before giving up and reporting whatever keys exist by then.
const API_KEYS_LOCK_WAIT_MS = 600_000;

// Written when the user skips the API-key prompt.  While it exists no
// activation prompts again (the skip was an answer, not a request to be
// asked at every window); "KISS: Enter API Key" clears it.
const API_KEYS_DECLINED_FILE = path.join(LOG_DIR, '.api-keys-declined');

/** Persist a one-shot user decision as a marker file under $KISS_HOME. */
function writeDeclinedMarker(markerPath: string): void {
  try {
    fs.mkdirSync(path.dirname(markerPath), {recursive: true});
    fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
  } catch {}
}

/**
 * "KISS: Enter API Key": forget an earlier skip and run the API-key
 * prompt now.
 *
 * @returns true when a key (or the Claude CLI) is available afterwards.
 */
export function promptApiKeysNow(): Promise<boolean> {
  fs.rmSync(API_KEYS_DECLINED_FILE, {force: true});
  return ensureApiKeys();
}

export async function ensureApiKeys(
  lockFile: string = API_KEYS_LOCK_FILE,
  declinedFile: string = API_KEYS_DECLINED_FILE,
): Promise<boolean> {
  loadApiKeysFromShellRc();

  const keys = [
    {
      envName: 'ANTHROPIC_API_KEY',
      displayName: 'Anthropic API Key',
      placeholder: 'sk-ant-...',
      validate: validateAnthropicKey,
      providerHost: 'api.anthropic.com',
    },
    {
      envName: 'OPENAI_API_KEY',
      displayName: 'OpenAI API Key',
      placeholder: 'sk-...',
      validate: undefined,
      providerHost: 'api.openai.com',
    },
  ];

  const hasClaudeCli = commandExists('claude');
  const hasAnyKey = () =>
    hasClaudeCli || keys.some(k => !!process.env[k.envName]);

  if (hasAnyKey()) return true;
  if (fs.existsSync(declinedFile)) {
    log('API key prompt skipped earlier — not asking again');
    return false;
  }

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
      for (const {
        envName,
        displayName,
        placeholder,
        validate,
        providerHost,
      } of keys) {
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
          providerHost,
        );
        if (key) {
          process.env[envName] = key;
          addToShellRc(rcPath, envName, key);
          log(`${displayName} saved to ~/${path.basename(rcPath)}`);
        }
      }

      if (hasAnyKey()) break;

      const choice = await showWarningNotification(
        `${PRODUCT_NAME} requires Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY to work.`,
        'Enter Key',
        'Skip',
      );
      if (choice !== 'Enter Key') {
        // Skip (or closing the toast) is remembered: no window asks
        // again until the user runs the command offered here.
        writeDeclinedMarker(declinedFile);
        void showInformationNotification(
          `${PRODUCT_NAME} will not ask for an API key again. ` +
            'Add one any time with "KISS: Enter API Key" from the Command Palette.',
        );
        break;
      }
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

async function readKissConfig(): Promise<Record<string, unknown>> {
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
      // The file is being rewritten by the daemon; back off without
      // blocking the extension host's event loop.
      await sleep(BACKOFF_MS);
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

// Runs inside the kiss venv: save_config() merges the update read from
// stdin into config.json under the daemon's fcntl lock (.config.lock).
const SAVE_CONFIG_PY =
  'import json, sys; from kiss.core.vscode_config import save_config; ' +
  'save_config(json.load(sys.stdin))';
const SAVE_CONFIG_TIMEOUT_MS = 60_000;

/**
 * Merge `update` into `$KISS_HOME/config.json`.
 *
 * The daemon is the single writer of config.json: its
 * `vscode_config.save_config` merges under an `fcntl.flock` on
 * `.config.lock`.  Node has no flock, so instead of emulating the lock
 * the extension hands the update to that writer through `uv run python`
 * (the payload travels on stdin, never argv, so a password never shows
 * up in `ps`).  An unlocked read-modify-replace here raced the daemon's
 * own saves and lost either the password or the daemon's settings, so
 * there is deliberately no direct-write fallback: when the writer is
 * unavailable the save fails and config.json is left untouched.
 *
 * The child runs asynchronously (execFile) so a slow `uv run` never
 * freezes the extension host; it is killed after
 * {@link SAVE_CONFIG_TIMEOUT_MS}.
 *
 * @param update Keys to merge into config.json.
 * @param uvPath Path of the uv binary, or null when uv was not found.
 * @param kissProjectPath Root of the kiss checkout whose venv runs Python.
 * @returns Resolves once save_config has written the file; rejects with
 *   the reason (uv missing, spawn error, or the exit code and stderr).
 */
export function saveKissConfig(
  update: Record<string, unknown>,
  uvPath: string | null,
  kissProjectPath: string,
): Promise<void> {
  // audit0902-coverage:start
  if (!uvPath) {
    log('saveKissConfig: uv not found — cannot save config.json');
    return Promise.reject(new Error('uv not found'));
  }
  return new Promise<void>((resolve, reject) => {
    const child = execFile(
      uvPath,
      ['run', 'python', '-c', SAVE_CONFIG_PY],
      {
        cwd: kissProjectPath,
        env: {...process.env, KISS_HOME: LOG_DIR},
        timeout: SAVE_CONFIG_TIMEOUT_MS,
      },
      (err, _stdout, stderr) => {
        if (!err) {
          resolve();
          return;
        }
        const code = (err as {code?: number | string}).code;
        const reason =
          typeof code === 'number' ? `exited with code ${code}` : err.message;
        const detail = stderr ? String(stderr).trim() : '';
        log(
          `saveKissConfig: Python save_config failed: ${reason}` +
            (detail ? `\n${detail}` : ''),
        );
        reject(new Error(reason + (detail ? `: ${detail}` : '')));
      },
    );
    // A spawn failure (ENOENT, EACCES) is delivered to the callback as
    // well; the stdin write is guarded so a dead pipe cannot throw here.
    child.stdin?.on('error', () => {});
    child.stdin?.end(JSON.stringify(update));
  });
  // audit0902-coverage:end
}

async function getStoredRemotePassword(): Promise<string> {
  const cfg = await readKissConfig();
  const existing = cfg['remote_password'];
  if (typeof existing === 'string' && existing.length > 0) {
    return existing;
  }
  return '';
}

// The remote-password prompt is one-shot cross-window UI state, exactly
// like the API-key prompts: without a lock two windows finalizing
// together both showed the input box, the user typed two passwords, and
// the saves raced last-write-wins through save_config.  The lock is an
// exclusively created pid+token file like the daemon-restart lock, but
// with a PROMPT-specific lifecycle: a password box can legitimately sit
// open for many minutes, so a LIVE holder is never evicted by age; a
// holder that DIED mid-prompt is detected by pid and taken over within
// seconds; and a finished holder records a terminal outcome (saved /
// skipped / failed) next to the lock so a waiter can tell a deliberate
// Esc from a crash without ever stacking a second prompt on the user.
const REMOTE_PASSWORD_LOCK_FILE = path.join(LOG_DIR, '.remote-password.lock');
// How long a window without the lock waits for the prompting window
// before giving up (a prompt can sit open for minutes).
const REMOTE_PASSWORD_LOCK_WAIT_MS = 600_000;
// How often a waiter re-checks the holder (liveness and outcome).
const REMOTE_PASSWORD_POLL_MS = 1000;
// Written when the user skips the remote-password prompt (Esc / empty).
// While it exists no later session prompts again; the settings panel's
// Remote password field is the way to set one afterwards.
const REMOTE_PASSWORD_DECLINED_FILE = path.join(
  LOG_DIR,
  '.remote-password-declined',
);
// An unreadable lock younger than this is assumed to be one caught
// mid-write and is honoured as live.
const PROMPT_LOCK_UNREADABLE_STALE_MS = 120_000;

type RemotePasswordOutcome = 'saved' | 'skipped' | 'failed';

/**
 * Read the terminal outcome the last prompt holder recorded, if any.
 *
 * @param lockFile Path of the prompt lock; the outcome lives beside it.
 * @returns The token-stamped outcome, or null when there is none or it
 *     is unreadable / not in the current format.
 */
function readPromptOutcome(
  lockFile: string,
): {token: string; outcome: RemotePasswordOutcome} | null {
  // audit0903-coverage:start
  try {
    const raw = fs.readFileSync(`${lockFile}.outcome`, 'utf-8');
    const data: unknown = JSON.parse(raw);
    const {token, outcome} = data as {token?: unknown; outcome?: unknown};
    if (typeof token !== 'string' || !token) return null;
    if (outcome !== 'saved' && outcome !== 'skipped' && outcome !== 'failed') {
      return null;
    }
    return {token, outcome};
  } catch {
    return null;
  }
  // audit0903-coverage:end
}

/**
 * One waiter poll round: how does the prompt holder's session stand?
 *
 * A dead or stale-unreadable lock is removed here (read-verify-unlink,
 * the same small accepted race as breakAbandonedRestartLock) so that a
 * 'contend' caller can immediately try the exclusive create.
 *
 * An outcome binds only a caller that actually WAITED on its session
 * (*honorOutcome*): a fresh activation finding a leftover outcome from
 * some past session must still prompt, exactly as it always did after
 * a normally released skip — the record exists so a waiter never turns
 * the holder's deliberate Esc into a second prompt.
 *
 * @param lockFile Path of the prompt lock.
 * @param honorOutcome Whether this caller waited on the current
 *     holder's session and must respect its recorded outcome.
 * @returns 'settled' when the session this caller waited on ended with
 *     a recorded outcome (a deliberate save / skip / failure — never
 *     re-prompt), 'contend' when the lock is free or its owner died
 *     mid-prompt (the caller should try to take it), 'wait' while a
 *     live holder prompts.
 */
function checkPromptHolder(
  lockFile: string,
  honorOutcome: boolean,
): 'settled' | 'contend' | 'wait' {
  // audit0903-coverage:start
  let st: fs.Stats;
  try {
    st = fs.statSync(lockFile);
  } catch {
    // No lock: the holder we waited on finished (its outcome is
    // recorded), or this is a fresh start.
    return honorOutcome && readPromptOutcome(lockFile) ? 'settled' : 'contend';
  }
  const owner = readRestartLockOwner(lockFile);
  if (!owner) {
    // Unreadable: honour it while young (a lock caught mid-write),
    // treat it as abandoned once stale.
    if (Date.now() - st.mtimeMs < PROMPT_LOCK_UNREADABLE_STALE_MS) {
      return 'wait';
    }
    log('breaking unreadable remote-password prompt lock');
    try {
      fs.unlinkSync(lockFile);
    } catch {}
    return 'contend';
  }
  if (!processIsAlive(owner.pid)) {
    const outcome = readPromptOutcome(lockFile);
    if (honorOutcome && outcome && outcome.token === owner.token) {
      // The holder we waited on finished (outcome recorded) and died
      // before its unlink: honour the result, just clean the lock up.
      log('remote-password prompt holder finished and exited — cleaning up');
      try {
        fs.unlinkSync(lockFile);
      } catch {}
      return 'settled';
    }
    log(`remote-password prompt holder pid ${owner.pid} died — taking over`);
    try {
      fs.unlinkSync(lockFile);
    } catch {}
    return 'contend';
  }
  // A live holder is NEVER evicted by age: a prompt may stay open
  // longer than any timeout we could pick.
  return 'wait';
  // audit0903-coverage:end
}

interface RemotePasswordPromptHold {
  /** Record the terminal outcome, then release the lock. */
  finish: (outcome: RemotePasswordOutcome) => void;
}

/**
 * Take the cross-window remote-password prompt lock.
 *
 * A single exclusive create: breaking dead or stale locks is
 * {@link checkPromptHolder}'s job, so a loser here simply keeps
 * waiting.
 *
 * @param lockFile Path of the lock file.
 * @returns A hold whose finish() records the prompt's outcome and
 *     releases the lock, or null when another window holds it.
 */
function acquireRemotePasswordPromptLock(
  lockFile: string,
): RemotePasswordPromptHold | null {
  // audit0903-coverage:start
  const token = `${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  try {
    fs.mkdirSync(path.dirname(lockFile), {recursive: true});
    const fd = fs.openSync(lockFile, 'wx');
    try {
      fs.writeSync(fd, JSON.stringify({pid: process.pid, token}));
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    // Another window created the lock between our poll and this open,
    // or the directory is unwritable: wait like any other loser.
    return null;
  }
  // A new prompt session begins: a leftover outcome belongs to an OLDER
  // session and must not be mistaken for this one's result.
  try {
    fs.unlinkSync(`${lockFile}.outcome`);
  } catch {}
  return {
    finish: (outcome: RemotePasswordOutcome) => {
      try {
        // Atomic, so a waiter polling the file never reads a torn
        // record; written BEFORE the release so no gap exists in which
        // the lock is gone but the outcome not yet visible.
        writeFileAtomicSync(
          `${lockFile}.outcome`,
          JSON.stringify({token, outcome}),
        );
      } catch (err) {
        log(
          'remote-password outcome not recorded: ' +
            `${err instanceof Error ? err.message : err}`,
        );
      }
      releaseDaemonRestartLock(lockFile, token);
    },
  };
  // audit0903-coverage:end
}

/**
 * Prompt for the remote-access password when config.json has none and
 * save it through {@link saveKissConfig}.  Runs after the daemon restart
 * in runFinalization, so `uvPath`/`kissProjectPath` point at a usable
 * venv; exported so the e2e tests can drive the prompt end to end.  When
 * the save fails the password is left unsaved and the user is pointed at
 * the settings panel's Remote password field (saved by the daemon).
 *
 * The prompt is guarded by a cross-window lock: two windows finalizing
 * together would otherwise both prompt and the two saves would race
 * last-write-wins.  The window without the lock waits for the holder
 * and then respects its recorded outcome (saved, deliberately skipped
 * or failed) instead of stacking a second prompt on the user; a holder
 * that DIES mid-prompt is taken over within seconds.
 *
 * @param uvPath Path of the uv binary, or null when uv was not found.
 * @param kissProjectPath Root of the kiss checkout whose venv runs Python.
 * @param lockFile Path of the cross-window prompt lock (overridable for
 *   tests).
 * @param waitMs How long to wait for another window's open prompt
 *   before giving up (overridable for tests).
 * @param pollMs Interval between holder liveness checks (overridable
 *   for tests).
 * @param declinedFile Marker written when the user skipped the prompt
 *   in an earlier session; while it exists the prompt is not repeated
 *   (overridable for tests).
 */
export async function ensureRemotePassword(
  uvPath: string | null,
  kissProjectPath: string,
  lockFile: string = REMOTE_PASSWORD_LOCK_FILE,
  waitMs: number = REMOTE_PASSWORD_LOCK_WAIT_MS,
  pollMs: number = REMOTE_PASSWORD_POLL_MS,
  declinedFile: string = REMOTE_PASSWORD_DECLINED_FILE,
): Promise<void> {
  // audit0903-coverage:start
  if (await getStoredRemotePassword()) {
    log('ensureRemotePassword: password already set — skipping prompt');
    return;
  }
  if (fs.existsSync(declinedFile)) {
    log('ensureRemotePassword: prompt skipped earlier — not asking again');
    return;
  }

  log(
    'ensureRemotePassword: password not found on first read — retrying after 2 s',
  );
  await new Promise(resolve => setTimeout(resolve, 2000));

  if (await getStoredRemotePassword()) {
    log('ensureRemotePassword: password found on retry — skipping prompt');
    return;
  }

  const deadline = Date.now() + waitMs;
  let waitLogged = false;
  // True once this call has observed another window's session (a live
  // or unreadable lock, or a lost exclusive create): only then does a
  // recorded outcome bind us.
  let observedHolder = false;
  for (;;) {
    const verdict = checkPromptHolder(lockFile, observedHolder);
    if (verdict === 'settled') {
      // The holder saved a password, or the user deliberately skipped
      // (or the save failed and was reported); every recorded outcome
      // stands — never double-prompt.
      log('ensureRemotePassword: another window finished the prompt');
      return;
    }
    if (verdict === 'contend') {
      const hold = acquireRemotePasswordPromptLock(lockFile);
      if (hold) {
        let outcome: RemotePasswordOutcome = 'failed';
        try {
          outcome = await ensureRemotePasswordLocked(
            uvPath,
            kissProjectPath,
            declinedFile,
          );
        } finally {
          hold.finish(outcome);
        }
        return;
      }
      // Lost the exclusive create: a holder exists right now.
      observedHolder = true;
    } else {
      observedHolder = true;
    }
    if (Date.now() >= deadline) {
      log('ensureRemotePassword: gave up waiting for the prompting window');
      return;
    }
    if (!waitLogged) {
      log('ensureRemotePassword: another window is prompting — waiting');
      waitLogged = true;
    }
    await new Promise(resolve => setTimeout(resolve, pollMs));
  }
  // audit0903-coverage:end
}

/**
 * The prompt-and-save body of {@link ensureRemotePassword}, run while
 * holding the cross-window prompt lock.
 *
 * @param uvPath Path of the uv binary, or null when uv was not found.
 * @param kissProjectPath Root of the kiss checkout whose venv runs Python.
 * @param declinedFile Marker to write when the user skips the prompt.
 * @returns The terminal outcome the holder records beside the lock.
 */
async function ensureRemotePasswordLocked(
  uvPath: string | null,
  kissProjectPath: string,
  declinedFile: string,
): Promise<RemotePasswordOutcome> {
  // audit0903-coverage:start
  // Re-check under the lock: a previous holder (or a takeover) may have
  // saved a password between our pre-lock read and the lock creation.
  if (await getStoredRemotePassword()) {
    log(
      'ensureRemotePassword: password saved while acquiring the lock — ' +
        'skipping prompt',
    );
    return 'saved';
  }

  log('ensureRemotePassword: password still empty — prompting user');
  const password = await vscode.window.showInputBox({
    title: `${PRODUCT_NAME} — Remote Access Password`,
    prompt: `Set a password for the ${PRODUCT_NAME} web / mobile app (press Esc to skip):`,
    placeHolder: 'Enter a password',
    password: true,
    ignoreFocusOut: true,
  });

  if (password === undefined || password.trim() === '') {
    // The skip is remembered across sessions; the one-time follow-up
    // says where the password can be set later and opens it.
    writeDeclinedMarker(declinedFile);
    void showInformationNotification(
      `${PRODUCT_NAME} will not ask for a remote access password again. ` +
        `Set one any time in the ${PRODUCT_NAME} settings panel ` +
        '(Remote password field).',
      'Open settings',
    ).then(action => {
      if (action === 'Open settings') {
        void vscode.commands.executeCommand('kissSorcar.openSettings');
      }
    });
    return 'skipped';
  }

  try {
    await saveKissConfig(
      {remote_password: password.trim()},
      uvPath,
      kissProjectPath,
    );
  } catch (err) {
    const reason = err instanceof Error ? err.message : String(err);
    log(`ensureRemotePassword: saving the password failed: ${reason}`);
    showErrorNotification(
      `${PRODUCT_NAME}: could not save the remote access password ` +
        `(${reason}). Set it in the ${PRODUCT_NAME} settings panel ` +
        '(Remote password field) once the daemon is running.',
    );
    return 'failed';
  }
  log(`Remote access password saved to ${path.join(LOG_DIR, 'config.json')}`);
  return 'saved';
  // audit0903-coverage:end
}

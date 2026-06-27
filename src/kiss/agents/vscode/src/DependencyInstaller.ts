// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Auto-installation of binary dependencies for the KISS Sorcar extension.
 * Ensures uv, git, Node.js, VS Code CLI, Python environment, and Playwright
 * Chromium are available.  Called during extension activation.
 */

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
import {
  showErrorNotification,
  showInformationNotification,
  showWarningNotification,
  withWebviewNotificationProgress,
} from './WebviewNotifications';

const HOME_DIR = process.env.HOME || process.env.USERPROFILE || '';
const LOG_DIR = path.join(HOME_DIR, '.kiss');
const LOG_FILE = path.join(LOG_DIR, 'install.log');
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 13;
const UV_VERSION = '0.11.2';
const NODE_VERSION = 'v22.16.0';

/**
 * Escape a string so it is safe to embed inside an XML ``<string>``
 * element (e.g. macOS LaunchAgent plist).  Without escaping a path
 * containing ``&`` or ``<`` produces malformed XML and ``launchctl``
 * silently rejects the unit.
 */
function xmlEscape(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

/**
 * Escape a string so it is safe to embed inside a systemd unit value.
 * Backslashes and newlines must not appear unescaped — they otherwise
 * silently corrupt the unit.
 */
function unitEscape(s: string): string {
  return s.replace(/\\/g, '\\\\').replace(/\n/g, '\\n').replace(/%/g, '%%');
}

/**
 * Download a URL into ``destPath`` using the Node ``https`` module.
 * Follows up to 5 redirects.  Resolves only after the response has been
 * fully written to disk.  Never spawns a shell.
 */
function downloadFile(
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
        res.pipe(out);
        out.on('finish', () =>
          out.close(err => (err ? reject(err) : resolve())),
        );
        out.on('error', reject);
      });
      req.on('error', reject);
      req.on('timeout', () => {
        req.destroy(new Error(`Timeout downloading ${u}`));
      });
    };
    get(url, maxRedirects);
  });
}

/**
 * Compute the lowercase-hex SHA256 digest of a file on disk.
 */
function sha256OfFile(filePath: string): string {
  const buf = fs.readFileSync(filePath);
  return crypto.createHash('sha256').update(buf).digest('hex');
}

/**
 * Best-effort integrity check for a downloaded asset against a vendor-
 * published SHA256 file (e.g. ``<url>.sha256`` for uv, ``SHASUMS256.txt``
 * for nodejs.org).  When ``expectedHashHex`` is given and does not match,
 * the file is unlinked and an error is thrown.  When omitted (vendor
 * does not publish a hash), the function logs a warning and returns —
 * we do not silently skip integrity for binaries we know how to verify.
 *
 * Pinning the digest in the extension source is preferable but requires
 * release-time bookkeeping; this helper at minimum compares against the
 * SHA256 file fetched from the same release URL, defending against
 * CDN/mirror corruption and accidental wrong-asset downloads.
 */
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
    } catch {
      /* ignore */
    }
    throw new Error(
      `SHA256 mismatch for ${path.basename(filePath)}: ` +
        `expected ${expectedHashHex}, got ${got}`,
    );
  }
  log(`SHA256 ok for ${path.basename(filePath)}`);
}

/**
 * Fetch ``<assetUrl>.sha256`` (uv-style sidecar) and return the hex digest,
 * or null when the sidecar isn't available.
 */
function fetchUvStyleSha256(assetUrl: string): Promise<string | null> {
  return new Promise(resolve => {
    const req = https.get(assetUrl + '.sha256', {timeout: 15000}, res => {
      if ((res.statusCode || 0) !== 200) {
        res.resume();
        resolve(null);
        return;
      }
      const chunks: Buffer[] = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf-8').trim();
        // sidecar is "<hex>  filename"
        const m = /^([0-9a-fA-F]{64})/.exec(text);
        resolve(m ? m[1] : null);
      });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => {
      req.destroy();
      resolve(null);
    });
  });
}

/**
 * Fetch ``https://nodejs.org/dist/<NODE_VERSION>/SHASUMS256.txt`` and find
 * the hash for ``assetName``.  Returns null on any error.
 */
function fetchNodeSha256(assetName: string): Promise<string | null> {
  const url = `https://nodejs.org/dist/${NODE_VERSION}/SHASUMS256.txt`;
  return new Promise(resolve => {
    const req = https.get(url, {timeout: 15000}, res => {
      if ((res.statusCode || 0) !== 200) {
        res.resume();
        resolve(null);
        return;
      }
      const chunks: Buffer[] = [];
      res.on('data', d => chunks.push(d));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf-8');
        for (const line of text.split('\n')) {
          const m = /^([0-9a-fA-F]{64})\s+(.+?)\s*$/.exec(line);
          if (m && m[2] === assetName) {
            resolve(m[1]);
            return;
          }
        }
        resolve(null);
      });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => {
      req.destroy();
      resolve(null);
    });
  });
}

/** Synchronous sleep that neither spawns a process nor spins the CPU. */
function sleepSync(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/**
 * Core no-shell process runner: spawn ``cmd`` with ``args``, capture
 * stdout and stderr, and resolve on exit (any code).  ``timeoutMs = 0``
 * disables the kill timer.  Rejects only on spawn failure or timeout.
 * Shared by :func:`spawnPromise` and :func:`runAsync`.
 */
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

/**
 * Run a child process without invoking a shell and resolve with the
 * trimmed stdout on exit code 0.  Used for ``tar``, ``mv``, etc. where
 * a shell would be required only to interpolate user-controlled paths.
 */
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

/** Guard against concurrent ensureDependencies calls. */
let pendingDeps: Promise<void> | null = null;

/**
 * Write a timestamped message to ~/.kiss/install.log and the developer console.
 */
function log(message: string): void {
  const line = `[${new Date().toISOString()}] ${message}`;
  console.log('[KISS Sorcar]', message);
  try {
    fs.mkdirSync(LOG_DIR, {recursive: true});
    fs.appendFileSync(LOG_FILE, line + '\n');
  } catch {
    /* ignore write errors */
  }
}

/**
 * Prepend *dir* to ``process.env.PATH`` so binaries in it are found by
 * all child processes.  Safe to call multiple times — skips if already
 * present.
 */
function prependToProcessPath(dir: string): void {
  const parts = (process.env.PATH || '').split(path.delimiter);
  if (!parts.includes(dir)) {
    process.env.PATH = `${dir}${path.delimiter}${process.env.PATH || ''}`;
  }
}

/**
 * Prepend ~/.local/bin to process.env.PATH so that binaries installed by
 * the extension (uv, node, sorcar) are found by all child processes.
 */
export function ensureLocalBinInPath(): void {
  if (!HOME_DIR) return;
  prependToProcessPath(path.join(HOME_DIR, '.local', 'bin'));
}

/**
 * Windows install helper: download a zip with PowerShell, extract it
 * into ``destDir``, run any ``extraPsCommands`` (e.g. Move-Item
 * cleanup), and delete the zip — all in one PowerShell invocation.
 */
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

/**
 * Find the actual Node.js directory inside the base install dir on Windows.
 * Node.js extracts to a nested subdirectory like node-v22.16.0-win-x64/.
 * Returns the nested directory containing node.exe, or baseDir as fallback.
 */
function findNodeDirWindows(baseDir: string): string {
  try {
    for (const entry of fs.readdirSync(baseDir)) {
      const candidate = path.join(baseDir, entry);
      if (fs.existsSync(path.join(candidate, 'node.exe'))) return candidate;
    }
  } catch {
    /* ignore */
  }
  return baseDir;
}

/**
 * Pure-TypeScript replica of Python's ``get_default_model()`` priority
 * order from ``kiss.core.models.model_info``.  Used as a fallback when
 * the .venv is not yet available (first-time setup) so the model
 * picker is never rendered blank.
 *
 * Priority order (must match Python):
 *   Anthropic > OpenAI > Gemini > OpenRouter > Together AI >
 *   Claude Code CLI > Codex CLI > ``"No model"``.
 *
 * Keep the strings in sync with ``get_default_model`` in
 * ``src/kiss/core/models/model_info.py``.
 */
export function getFallbackDefaultModel(): string {
  const env = process.env;
  if (env.ANTHROPIC_API_KEY) return 'claude-opus-4-7';
  if (env.OPENAI_API_KEY) return 'gpt-5.5';
  if (env.GEMINI_API_KEY) return 'gemini-3.1-pro-preview';
  if (env.OPENROUTER_API_KEY) return 'openrouter/anthropic/claude-opus-4.7';
  if (env.TOGETHER_API_KEY) return 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8';
  const whichCmd = process.platform === 'win32' ? 'where' : 'which';
  try {
    execFileSync(whichCmd, ['claude'], {stdio: 'ignore', timeout: 2_000});
    return 'cc/opus';
  } catch {
    /* not installed */
  }
  try {
    execFileSync(whichCmd, ['codex'], {stdio: 'ignore', timeout: 2_000});
    return 'codex/default';
  } catch {
    /* not installed */
  }
  return 'No model';
}

/**
 * Return the best default model name by calling Python's canonical
 * ``get_default_model()`` from ``kiss.core.models.model_info``.
 *
 * Requires ``uv`` and the KISS project ``.venv`` to be available.
 * When the Python call fails (e.g. during first-time setup before
 * dependencies are installed), falls back to a pure-TypeScript
 * replica that checks the same API-key env vars and CLI executables.
 * This guarantees the model picker never renders blank on a fresh
 * install — the backend still overrides the value via the ``models``
 * event once the daemon is up.
 */
export function getDefaultModel(): string {
  const uvPath = findUvPath();
  const kissProject = findKissProject();
  if (!uvPath || !kissProject) return getFallbackDefaultModel();
  try {
    // H1 — argv-form so quotes / shell metacharacters in either path
    // (both come from user-controlled HOME / settings) cannot inject.
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

/**
 * Run the post-install finalization steps: install the ``sorcar`` CLI
 * wrapper, install cloudflared, restart the kiss-web daemon, persist
 * PATH entries to the user's shell rc file, and prompt for any missing
 * API keys.  Returns whether at least one API key (or the Claude CLI)
 * is available.
 *
 * When called with a non-null ``progress`` reporter this function
 * publishes brief sub-step messages so the surrounding "KISS Sorcar:
 * Setting up" notification stays visible continuously from the first
 * install step until the completion message is shown.  Passing ``null``
 * skips the progress reports (used on the fast path, which has no
 * progress notification to annotate).
 */
async function runFinalization(
  progress: vscode.Progress<{message?: string; increment?: number}> | null,
  kissProjectPath: string,
  uvPath: string | null,
): Promise<boolean> {
  if (uvPath) {
    if (progress) progress.report({message: 'Installing CLI wrapper...'});
    installCliScript(kissProjectPath, uvPath);
  }

  if (progress) progress.report({message: 'Refreshing model info...'});
  installModelInfoJson(kissProjectPath);
  installMarkdownAssets(kissProjectPath);

  if (progress) progress.report({message: 'Checking cloudflared...'});
  await installCloudflaredIfNeeded();

  if (progress) progress.report({message: 'Restarting kiss-web daemon...'});
  // Point the kiss-web daemon's WorkingDirectory (its process getcwd, and
  // therefore the default task work_dir) at the VS Code workspace root —
  // i.e. USER_PWD, the directory the user ran install.sh from.  Falls back
  // to kissProjectPath when no folder is open (preserving prior behavior).
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

/**
 * Ensure all required dependencies are installed.
 * Shows a progress notification during first-time installation.
 * Safe to call multiple times — uses a concurrency guard so overlapping
 * calls reuse the in-flight check instead of racing.
 */
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
        'See ~/.kiss/install.log for details.',
    );
    return;
  }
  log(`KISS project: ${kissProjectPath}`);

  // Early exit: when all dependencies are fully installed, the daemon is
  // running, and no extension update is pending, skip all setup work.
  // This avoids the 162 MB Playwright re-download, daemon restart, and
  // CLI script rewrite that previously ran on every VS Code activation.
  const updateMarker = path.join(LOG_DIR, '.extension-updated');
  if (
    findUvPath() &&
    fs.existsSync(path.join(kissProjectPath, '.venv')) &&
    isChromiumInstalled() &&
    (await isDaemonRunning()) &&
    !fs.existsSync(updateMarker)
  ) {
    log('All dependencies satisfied and daemon running — nothing to do');
    log('=== Dependency check finished ===');
    loadApiKeysFromShellRc();
    return;
  }

  let uvPath = findUvPath();
  let venvExists = fs.existsSync(path.join(kissProjectPath, '.venv'));

  // If .venv exists but Python is genuinely too old, remove it so uv sync
  // recreates it.  Transient errors (timeout, spawn failure) must NOT delete
  // .venv — that causes unnecessary slow-path rebuilds on every activation.
  if (uvPath && venvExists) {
    const pyStatus = checkPythonVersion(uvPath, kissProjectPath);
    if (pyStatus === 'too_old') {
      log('Python version too old — removing .venv for recreation');
      try {
        fs.rmSync(path.join(kissProjectPath, '.venv'), {
          recursive: true,
          force: true,
        });
      } catch {
        /* ignored */
      }
      venvExists = false;
    } else if (pyStatus === 'error') {
      log('Python version check failed (transient) — keeping .venv');
    }
  }

  // Check if build-extension.sh just ran (marker file written by the script).
  // When the marker exists, always show the restart notification — even on
  // the fast path where uv + .venv are already present.
  // (updateMarker is declared above for the early-exit guard.)
  let showRestartNotification = false;
  // ``apiKeysReady`` is set by ``runFinalization`` in either the fast
  // or slow path below.  Declared up here so the post-progress
  // restart-notification block below can read it regardless of which
  // path was taken.
  let apiKeysReady = false;
  if (fs.existsSync(updateMarker)) {
    showRestartNotification = true;
    try {
      fs.unlinkSync(updateMarker);
    } catch {
      /* ignore */
    }
    log('Extension-updated marker found — will show restart notification');
  }

  // Fast path: everything looks ready, ensure playwright in background
  if (uvPath && venvExists) {
    log('Fast path: uv and .venv present, ensuring Playwright in background');
    const uv = uvPath; // capture narrowed non-null type for closure
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
        // Only warn the user if Chromium is genuinely missing.  Playwright
        // browsers are cached system-wide (outside .venv), so a transient
        // background update failure is benign when chromium is already cached.
        if (!isChromiumInstalled()) {
          showWarningNotification(
            'KISS Sorcar: Chromium browser update failed in background. See ~/.kiss/install.log for details.',
          );
        }
      });
    // Ensure git, Node.js, and VS Code CLI are available even on fast path
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
    // Fast-path finalization: when we did not enter the slow-path
    // withProgress block we still need to run the finalization
    // steps.  The fast path does NOT show a progress notification
    // (because there was no "installing" notification to keep
    // visible to begin with), so the user-visible UX gap that
    // motivated wrapping finalization in withProgress does not
    // apply here.
    apiKeysReady = await runFinalization(null, kissProjectPath, uvPath);
  } else {
    // Slow path: show progress bar and install missing deps
    const result = await withWebviewNotificationProgress(
      {
        location: vscode.ProgressLocation.Notification,
        title: 'KISS Sorcar: Setting up',
        cancellable: false,
      },
      async progress => {
        // 1. Install uv if needed
        if (!uvPath) {
          // curl and tar are required to download and extract uv
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

        // 2. Install git if needed
        if (!gitWorks()) {
          progress.report({message: 'Installing git...'});
          const gitInstalled = await installGit();
          if (!gitInstalled) {
            showWarningNotification(
              `KISS Sorcar: git could not be installed automatically. ${gitInstallHint()}`,
            );
          }
        }

        // 3. Install Node.js if needed (provides node, npm, npx for agent tasks)
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

        // 4. Ensure VS Code CLI is on PATH
        if (!commandExists('code')) {
          progress.report({message: 'Setting up VS Code CLI...'});
          const codeInstalled = await installCodeCli();
          if (!codeInstalled) {
            log('VS Code CLI could not be set up on PATH');
          }
        }

        // 5. Set up Python environment (installs Python 3.13+ and all pip dependencies)
        if (!venvExists) {
          progress.report({
            message:
              'Setting up Python environment (first time, may take a minute)...',
          });
          await runAsync(uvPath, ['sync'], kissProjectPath);
          progress.report({increment: 50});
        }

        // 6. Verify Python version meets minimum requirement
        if (checkPythonVersion(uvPath, kissProjectPath) !== 'ok') {
          showErrorNotification(
            `KISS Sorcar requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+. ` +
              `Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or later and restart VS Code.`,
          );
          return {success: false, apiKeysReady: false};
        }

        // 7. Install Playwright Chromium
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

        // Finalization runs INSIDE the withProgress callback so the
        // "KISS Sorcar: Setting up" notification stays visible
        // continuously from the first install step until the
        // completion message is shown.  Previously these steps ran
        // after the callback returned, leaving a visible gap between
        // "Installing dependencies..." and the final notification
        // (often several seconds — much longer when ensureApiKeys
        // had to prompt the user for an API key).
        progress.report({message: 'Finalizing setup...'});
        const finalizedKeys = await runFinalization(
          progress,
          kissProjectPath,
          uvPath,
        );
        return {success: true, apiKeysReady: finalizedKeys};
      },
    );

    showRestartNotification = showRestartNotification || !!result.success;
    apiKeysReady = result.apiKeysReady;
  }

  log('=== Dependency check finished ===');

  // Post-install notification.  A VS Code window reload is NOT
  // required here:
  //
  //   * When a new VSIX was installed while VS Code was running, the
  //     ``fs.watchFile`` watchers in ``extension.ts`` (on ``out/
  //     extension.js`` and on ``~/.kiss/.extension-updated``) already
  //     fired ``workbench.action.reloadWindow`` BEFORE this function
  //     ran in the new activation — the extension code in memory is
  //     already current.
  //   * For a from-scratch install (slow path), the current Node host
  //     already has the updated ``process.env.PATH`` (set by
  //     ``ensureLocalBinInPath()`` at activation start) and the API
  //     keys (set by ``ensureApiKeys`` / ``loadApiKeysFromShellRc``),
  //     the Python venv exists on disk, and the kiss-web daemon was
  //     just (re)started.  Child processes spawned for chat tasks
  //     inherit all of this.
  //
  // The only thing a reload would refresh is already-open integrated
  // terminals — and opening a new terminal achieves the same effect
  // without disrupting an in-flight chat task.  So we now show a
  // non-prompting info message instead of forcing a reload.
  if (showRestartNotification) {
    if (apiKeysReady) {
      showInformationNotification(
        'KISS Sorcar: Installation complete! You are ready to go. ' +
          'Already-open terminals will not see the updated PATH until you open a new one.',
      );
    } else {
      showWarningNotification(
        'KISS Sorcar: Installation complete, but at least one of Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY is required. ' +
          'Set an API key in your environment, then reload the window (Developer: Reload Window) to be prompted again.',
      );
    }
  }
}

/**
 * Reliably kill every process listening on ``port``.
 *
 * Sends SIGTERM first, polls up to 3 s for graceful exit, then SIGKILLs
 * any stragglers.  Returns silently when nothing is listening.
 *
 * Used in place of ``pkill -x kiss-web``: kiss-web is a Python shebang
 * script, so the kernel's ``comm`` field is the truncated interpreter
 * path rather than the literal ``kiss-web``, making name-based pkill
 * unreliable on macOS.  Killing by listening port works on both macOS
 * and Linux without relying on process-name heuristics.
 *
 * @param port - TCP port whose listening processes should be killed.
 */
/** PIDs of processes listening on *port* (empty when none / lsof fails). */
function pidsOnPort(port: number): string[] {
  try {
    return execFileSync('lsof', ['-ti', `:${port}`], {
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

/** Send *signal* to every PID, ignoring already-gone processes. */
function killPids(pids: string[], signal: NodeJS.Signals): void {
  for (const pid of pids) {
    try {
      process.kill(parseInt(pid, 10), signal);
    } catch {
      /* already gone */
    }
  }
}

function killProcessOnPort(port: number): void {
  const pids = pidsOnPort(port);
  if (pids.length === 0) return; // Nothing listening — ok.
  killPids(pids, 'SIGTERM');
  // Wait up to ~3 s for the port to free up.
  for (let i = 0; i < 6; i++) {
    if (pidsOnPort(port).length === 0) return; // Port is free.
    sleepSync(500);
  }
  // Force-kill any survivors.
  killPids(pidsOnPort(port), 'SIGKILL');
}

/**
 * Start the kiss-web daemon directly as a detached background process.
 *
 * Fallback for Linux environments that have no usable systemd **user**
 * session — most notably Docker containers (where PID 1 is not systemd
 * and ``$DBUS_SESSION_BUS_ADDRESS`` / ``$XDG_RUNTIME_DIR`` are unset), as
 * well as minimal VMs and some WSL setups.  In those environments
 * ``systemctl --user`` fails, so without this fallback the daemon never
 * starts, ``~/.kiss/sorcar.sock`` is never created, and the extension can
 * never run a task.
 *
 * The child is detached and ``unref``-ed so it survives the extension host
 * exiting, inherits the current environment (API keys, etc.), runs with
 * ``cwd = workDir``, and appends its output to the same log files the
 * systemd/launchd units use.
 *
 * @param kissWebBin - Absolute path to ``.venv/bin/kiss-web``.
 * @param workDir - Working directory (process ``getcwd()``) for the daemon.
 */
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

/**
 * Restart the kiss-web remote access daemon.
 *
 * Always restarts the daemon so that code changes from an editable install
 * are picked up (the Python code is loaded at process start time).  Kills
 * any existing kiss-web and cloudflared processes, waits for port 8787 to
 * be free, then (re-)creates the service definition and starts it.
 *
 * On macOS uses ``~/Library/LaunchAgents/com.kiss.web-server.plist``
 * with ``KeepAlive`` and ``RunAtLoad``.
 * On Linux uses ``~/.config/systemd/user/kiss-web.service`` with
 * ``Restart=always`` and enables lingering.
 *
 * @param kissProjectPath - Absolute path to the KISS project directory.
 *   The ``kiss-web`` binary is always taken from
 *   ``${kissProjectPath}/.venv/bin/kiss-web``.
 * @param workDir - Absolute path used as the daemon's ``WorkingDirectory``
 *   (the process ``getcwd()``).  This is the VS Code workspace root (i.e.
 *   ``USER_PWD`` — the directory the user ran ``install.sh`` from), so that
 *   the web app's default task work_dir resolves to the user's project
 *   instead of the embedded ``kiss_project`` bundled inside the VSIX.
 *   ``web_server.py`` launches kiss-web with no ``--workdir``, so its
 *   ``os.getcwd()`` (= this ``WorkingDirectory``) becomes the fallback
 *   work_dir for sessions that don't send an explicit ``workDir``.
 */
async function restartKissWebDaemon(
  kissProjectPath: string,
  workDir: string,
): Promise<void> {
  if (process.platform === 'win32') return; // Not supported on Windows

  const kissWebBin = path.join(kissProjectPath, '.venv', 'bin', 'kiss-web');
  if (!fs.existsSync(kissWebBin)) {
    log(`kiss-web binary not found at ${kissWebBin} — skipping daemon setup`);
    return;
  }

  const binDir = path.join(HOME_DIR, '.local', 'bin');

  // Skip the kill+restart when the kiss-web binary and the editable
  // kiss source tree are byte-for-byte identical to the last time the
  // daemon was started AND the daemon is currently healthy on port
  // 8787.  This preserves any running ``cloudflared`` quick-tunnel
  // across VS Code window reloads / re-activations, which would
  // otherwise mint a brand-new random ``*.trycloudflare.com`` URL
  // every time the extension activates.
  //
  // The fingerprint changes whenever the user rebuilds, reinstalls,
  // or edits any Python source under ``src/kiss/`` — so the
  // editable-install code-pickup behavior is preserved on real
  // changes.
  const fpFile = path.join(LOG_DIR, '.kiss-web.fingerprint');
  const currentFp = computeKissWebFingerprint(
    kissProjectPath,
    kissWebBin,
    workDir,
  );
  let savedFp = '';
  try {
    savedFp = fs.readFileSync(fpFile, 'utf-8').trim();
  } catch {
    /* missing or unreadable — treat as mismatch */
  }
  const sockPath = path.join(HOME_DIR, '.kiss', 'sorcar.sock');

  // Tri-state TCP probe.  The OLD implementation here was::
  //
  //     try { execSync('lsof -i :8787 -t', {timeout: 2000});
  //           return fs.existsSync(sockPath); }
  //     catch { return false; }
  //
  // — which conflates "no listener" with "lsof slow under load".
  // On a heavily-loaded host the 2 s ``lsof`` timeout would fire,
  // the catch returned ``false``, the guard fell through, and the
  // installer SIGTERMed a perfectly healthy daemon that was mid-task.
  // ``probeDaemonHealth`` returns ``'alive' | 'dead' | 'unknown'`` so
  // the guard can treat a transient probe failure as "do not restart"
  // when the fingerprint matches.
  const health = await probeDaemonHealth(8787, 1500);
  const sockExists = fs.existsSync(sockPath);

  // Ask the live daemon whether any agent tasks are in flight.  The
  // SIGTERM regression report shows the kill happened with
  // ``active_tasks=[ad4ecb65(task=74)]`` on the daemon's own log — i.e.
  // the restart logic had no awareness of in-flight work at all.  When
  // the daemon reports tasks, defer the restart unconditionally (even
  // on a fingerprint change): the new code will be picked up on the
  // next activation once the running task has finished.
  let activeTasks:
    | {ok: true; count: number; tabs: string[]}
    | {ok: false; reason: string} = {ok: false, reason: 'not-probed'};
  if (health !== 'dead' && sockExists) {
    activeTasks = await daemonHasActiveTasks(sockPath, 1500);
  }

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
      // The daemon's TCP probe says ``alive`` but the UDS active-tasks
      // round-trip could not be completed.  Deferring is the safe
      // call: SIGTERMing an alive daemon whose task status is unknown
      // is exactly what aborted task_history row 3192 mid-flight.
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

  // Kill the existing kiss-web process before restarting.  ``kiss-web``
  // is a Python shebang script, so the kernel's ``comm`` field is the
  // (15-char-truncated) interpreter path — NOT the literal string
  // ``kiss-web``.  ``pkill -x kiss-web`` is therefore a silent no-op
  // on macOS; killing by listening port is the only reliable approach
  // that works across both macOS and Linux.
  //
  // ``cloudflared`` is intentionally NOT killed here: the Python
  // ``web_server`` adopts the existing cloudflared on startup
  // (matching pid in ``~/.kiss/cloudflared.pid`` and a healthy
  // metrics endpoint), keeping the same public URL across kiss-web
  // restarts.  When the kiss-web binary genuinely changed we still
  // restart kiss-web; the surviving cloudflared keeps forwarding to
  // ``https://localhost:8787``, which the new kiss-web re-binds.
  killProcessOnPort(8787);

  if (process.platform === 'darwin') {
    const plistLabel = 'com.kiss.web-server';
    const plistDir = path.join(HOME_DIR, 'Library', 'LaunchAgents');
    const plistFile = path.join(plistDir, `${plistLabel}.plist`);

    log('Restarting kiss-web macOS LaunchAgent...');
    try {
      fs.mkdirSync(plistDir, {recursive: true});
      // XML-escape every interpolated path so that paths containing
      // ``&``, ``<``, ``>``, ``"`` or ``'`` cannot corrupt the plist
      // structure or inject alternate ProgramArguments via XML entities.
      const xLabel = xmlEscape(plistLabel);
      const xBin = xmlEscape(kissWebBin);
      const xProj = xmlEscape(workDir);
      const xLogDir = xmlEscape(LOG_DIR);
      const xPath = xmlEscape(
        `/opt/homebrew/bin:${binDir}:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin`,
      );
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
    <key>StandardOutPath</key>
    <string>${xLogDir}/kiss-web-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>${xLogDir}/kiss-web-stderr.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${xPath}</string>
    </dict>
</dict>
</plist>`;

      fs.writeFileSync(plistFile, plistContent);

      // Unload existing service if present, then bootstrap the new one.
      // H1 — pass paths as separate argv entries via execFileSync so a
      // plistFile path containing single/double quotes or shell
      // metacharacters (HOME_DIR is user-controlled) cannot inject
      // arbitrary shell commands.
      const uid = execFileSync('id', ['-u'], {encoding: 'utf-8'}).trim();
      try {
        execFileSync('launchctl', ['bootout', `gui/${uid}/${plistLabel}`], {
          stdio: 'ignore',
          timeout: 5000,
        });
      } catch {
        /* not loaded — ok */
      }
      try {
        execFileSync('launchctl', ['bootstrap', `gui/${uid}`, plistFile], {
          stdio: 'ignore',
          timeout: 5000,
        });
      } catch {
        // Fall back to older load command — same argv-form to avoid
        // shell interpolation of plistFile.
        execFileSync('launchctl', ['load', '-w', plistFile], {
          stdio: 'ignore',
          timeout: 5000,
        });
      }
      // ``bootstrap`` (and ``load``) register the unit but do not
      // guarantee the process is running — on a fresh install or
      // after ``bootout`` the agent may sit in "loaded but not
      // started" state.  ``kickstart -k`` forces an immediate
      // (re)start so the daemon is actually alive when the
      // extension continues to query models via the UDS.
      try {
        execFileSync(
          'launchctl',
          ['kickstart', '-k', `gui/${uid}/${plistLabel}`],
          {stdio: 'ignore', timeout: 5000},
        );
      } catch {
        /* best-effort — KeepAlive will start it eventually */
      }
      log(`kiss-web macOS LaunchAgent restarted: ${plistFile}`);
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
      // unit-escape every interpolated path so that backslashes / newlines
      // / percent signs in user paths cannot corrupt the unit.
      const uBin = unitEscape(kissWebBin);
      const uProj = unitEscape(workDir);
      const uPath = unitEscape(`${binDir}:/usr/local/bin:/usr/bin:/bin`);
      const uLogDir = unitEscape(LOG_DIR);
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
StandardOutput=append:${uLogDir}/kiss-web-stdout.log
StandardError=append:${uLogDir}/kiss-web-stderr.log

[Install]
WantedBy=default.target
`;
      fs.writeFileSync(serviceFile, serviceContent);
      execSync('systemctl --user daemon-reload', {
        stdio: 'ignore',
        timeout: 10000,
      });
      execSync('systemctl --user restart kiss-web', {
        stdio: 'ignore',
        timeout: 10000,
      });
      // Enable lingering so service runs without active login session
      const username = os.userInfo().username;
      try {
        // H1 — argv-form so a username with shell metacharacters cannot
        // inject commands.
        execFileSync('loginctl', ['enable-linger', username], {
          stdio: 'ignore',
          timeout: 5000,
        });
      } catch {
        /* may require elevated privileges — non-fatal */
      }
      log(`kiss-web systemd user service restarted: ${serviceFile}`);
      systemdOk = true;
    } catch (err) {
      log(
        'Failed to restart kiss-web daemon via systemd (Linux): ' +
          `${err instanceof Error ? err.message : err} — ` +
          'falling back to direct background spawn',
      );
    }
    // Environments without a usable systemd user session (Docker
    // containers, minimal VMs, some WSL setups) reach here.  Start the
    // daemon directly so the UDS socket is created and tasks can run.
    if (!systemdOk) {
      spawnKissWebDirect(kissWebBin, workDir);
    }
  }

  // Record the fingerprint so the next activation can skip the
  // restart when nothing has changed.  Best-effort: a write failure
  // simply forces an unconditional restart next time, which is safe.
  try {
    fs.writeFileSync(fpFile, currentFp + '\n');
  } catch (err) {
    log(
      `Failed to write kiss-web fingerprint: ${err instanceof Error ? err.message : err}`,
    );
  }
}

/**
 * Compute a fingerprint of the installed kiss-web binary and the
 * editable kiss source tree.
 *
 * The fingerprint is the SHA-256 of:
 *   - the bytes of the ``kiss-web`` entrypoint script, and
 *   - the latest mtime (in nanoseconds) across all ``*.py`` files
 *     under ``${kissProjectPath}/src/kiss/``, excluding ``__pycache__``
 *     and ``tests/`` directories.
 *
 * This changes whenever the user rebuilds kiss-web, reinstalls the
 * package, edits any source file picked up by the editable install, or
 * opens a different workspace (``workDir`` changes the daemon's
 * ``WorkingDirectory``) — exactly the situations where the daemon needs
 * to restart, either to pick up new code or to re-point its default
 * work_dir at the new workspace root.  Returns ``""`` if the fingerprint
 * cannot be computed (caller treats this as a mismatch and restarts
 * unconditionally).
 */
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
          } catch {
            /* skip unreadable file */
          }
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
 * Always overwrite ``~/.kiss/INJECTIONS.md`` and ``~/.kiss/SAMPLE_TASKS.md``
 * with the copies bundled inside the installed kiss_project.  Called on
 * every install/update (both fast and slow paths in ``runFinalization``) so
 * that the kiss-web daemon's Tricks button and the VS Code extension's
 * welcome-screen sample-task chips always reflect the latest bundled
 * Markdown after a version upgrade — matching the ``installModelInfoJson``
 * pattern.  Failures are logged and swallowed so a read-only ``~/.kiss/``
 * cannot break the overall install flow.
 */
function installMarkdownAssets(kissProjectPath: string): void {
  if (!HOME_DIR) return;
  const kissHomeDir = LOG_DIR; // ~/.kiss
  const assets: Array<[string, string]> = [
    [
      path.join(kissProjectPath, 'src', 'kiss', 'INJECTIONS.md'),
      path.join(kissHomeDir, 'INJECTIONS.md'),
    ],
    [
      path.join(kissProjectPath, 'src', 'kiss', 'SAMPLE_TASKS.md'),
      path.join(kissHomeDir, 'SAMPLE_TASKS.md'),
    ],
  ];
  for (const [src, dst] of assets) {
    try {
      if (!fs.existsSync(src)) {
        log(`${path.basename(src)} not found at ${src}; skipping copy`);
        continue;
      }
      fs.mkdirSync(kissHomeDir, {recursive: true});
      fs.copyFileSync(src, dst);
      log(`Installed ${path.basename(src)} at ${dst}`);
    } catch (e: unknown) {
      log(`Failed to install ${path.basename(src)}: ${(e as Error).message}`);
    }
  }
}

/**
 * Copy the kiss_project's bundled ``MODEL_INFO.json`` to
 * ``~/.kiss/MODEL_INFO.json``.  The Python loader in
 * ``kiss.core.models.model_info`` reads the user-local copy at runtime
 * (with an mtime-based auto-refresh from the package copy), so seeding
 * it here on every install means the freshly installed extension serves
 * the up-to-date pricing/context table immediately.  Failures are logged
 * and swallowed: model_info.py's loader falls back to the package copy
 * if ~/.kiss/MODEL_INFO.json is missing or unreadable.
 */
function installModelInfoJson(kissProjectPath: string): void {
  if (!HOME_DIR) return;
  const src = path.join(
    kissProjectPath,
    'src',
    'kiss',
    'core',
    'models',
    'MODEL_INFO.json',
  );
  const dst = path.join(LOG_DIR, 'MODEL_INFO.json');
  try {
    if (!fs.existsSync(src)) {
      log(`MODEL_INFO.json not found at ${src}; skipping copy`);
      return;
    }
    fs.mkdirSync(LOG_DIR, {recursive: true});
    fs.copyFileSync(src, dst);
    log(`Installed MODEL_INFO.json at ${dst}`);
  } catch (e: unknown) {
    log(`Failed to install MODEL_INFO.json: ${(e as Error).message}`);
  }
}

/**
 * Install a `sorcar` CLI wrapper script in ~/.local/bin/ so users can
 * invoke the agent from any terminal after the extension is installed.
 * The wrapper calls `uv run sorcar` from the bundled kiss_project directory.
 */
function installCliScript(kissProjectPath: string, uvPath: string): void {
  if (!HOME_DIR) return;

  const binDir = path.join(HOME_DIR, '.local', 'bin');

  // Resolve uv to an absolute path for the wrapper script
  let absUvPath = uvPath;
  if (uvPath === 'uv' || !path.isAbsolute(uvPath)) {
    try {
      const whichCmd =
        process.platform === 'win32' ? `where ${uvPath}` : `which ${uvPath}`;
      absUvPath = execSync(whichCmd, {encoding: 'utf-8'}).trim().split('\n')[0];
    } catch {
      const suffix = process.platform === 'win32' ? '.exe' : '';
      absUvPath = path.join(HOME_DIR, '.local', 'bin', `uv${suffix}`);
    }
  }

  try {
    fs.mkdirSync(binDir, {recursive: true});

    // ``uv run --directory <kissProjectPath>`` changes the child
    // process's working directory to the bundled project before the
    // CLI starts, so the CLI's ``Path.cwd()`` would otherwise report
    // the project directory instead of the user's shell directory.
    // Capture the launch directory in ``KISS_WORKDIR`` *before* uv
    // runs so the CLI can recover it (see ``_launch_work_dir`` in
    // ``cli_helpers.py``) and default the task ``work_dir`` to where
    // the user actually invoked ``sorcar``.
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

/**
 * Map Node.js platform/arch to the uv GitHub release asset triplet.
 * Returns [archName, platformSuffix, extension] or null if unsupported.
 */
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
 * Install uv from a pinned GitHub binary release and return its path, or null on failure.
 * Downloads the platform-specific binary from releases.astral.sh and extracts to ~/.local/bin/.
 */
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
    // Ensure install directory exists
    fs.mkdirSync(installDir, {recursive: true});

    if (process.platform === 'win32') {
      // Windows: download zip and extract with PowerShell
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
      // macOS/Linux: download tar.gz and extract with tar (no shell so
      // that ``HOME`` containing single-quotes can never inject shell
      // commands).  Verify SHA256 against the vendor sidecar before
      // extracting so a corrupted/MITM'd download is rejected.
      const tarPath = path.join(installDir, `${assetName}.${asset.ext}`);
      await downloadFile(url, tarPath);
      const expectedHash = await fetchUvStyleSha256(url);
      verifyDownloadHash(tarPath, expectedHash);
      // Extract with argv-form spawn — no shell.
      await spawnPromise('tar', ['xzf', tarPath, '-C', installDir]);
      // Move uv / uvx into installDir, then remove the extracted folder.
      const extractedDir = path.join(installDir, assetName);
      for (const bin of ['uv', 'uvx']) {
        const src = path.join(extractedDir, bin);
        const dst = path.join(installDir, bin);
        try {
          fs.unlinkSync(dst);
        } catch {
          /* not present */
        }
        fs.renameSync(src, dst);
        fs.chmodSync(dst, 0o755);
      }
      try {
        fs.rmSync(extractedDir, {recursive: true, force: true});
      } catch {
        /* ignore */
      }
      try {
        fs.unlinkSync(tarPath);
      } catch {
        /* ignore */
      }
    }

    log('uv installed successfully');
    return findUvPath();
  } catch (err) {
    log(`Failed to install uv: ${err instanceof Error ? err.message : err}`);
    return null;
  }
}

/**
 * Check that the Python version in the project's .venv is >= MIN_PYTHON.
 * Runs `uv run python --version` and parses the output (e.g. "Python 3.13.2").
 *
 * Returns ``'ok'`` when the version meets the requirement, ``'too_old'``
 * when a valid version was parsed but is below the minimum, and ``'error'``
 * when the check itself failed (timeout, spawn error, etc.).  Callers
 * should only delete `.venv` on ``'too_old'`` — transient errors must not
 * destroy a potentially valid environment.
 */
function checkPythonVersion(
  uvPath: string,
  cwd: string,
): 'ok' | 'too_old' | 'error' {
  try {
    // H1 — argv-form so a uvPath containing quotes or shell
    // metacharacters cannot inject commands.
    const output = execFileSync(uvPath, ['run', 'python', '--version'], {
      cwd,
      encoding: 'utf-8',
      timeout: 30_000,
    }).trim();
    // Output format: "Python 3.13.2"
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

/**
 * Return the Playwright browsers cache directory for the current platform.
 */
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

/**
 * Check if the kiss-web daemon is fully running by probing **both**
 * the TCP listener on port 8787 and the Unix-domain socket at
 * ``~/.kiss/sorcar.sock``.  Requiring both endpoints ensures the
 * extension never treats a half-started daemon (e.g. TCP bound but
 * UDS not yet created, or vice-versa) as healthy — the model-list
 * and agent RPCs need the UDS, while the remote-access / tunnel path
 * needs the TCP port.
 *
 * Returns true only when both are present (macOS/Linux).
 * On Windows the daemon is not supported, so always returns false.
 *
 * The TCP probe is retried up to 3 times with 300 ms gaps before
 * returning false.  A single probe races with the LaunchAgent's
 * ~1-3 s respawn window after a previous kiss-web restart; without
 * the retry the extension would nuke a healthy daemon that was merely
 * mid-startup, burning the current Cloudflare tunnel URL on every
 * VS Code activation that lost that race.
 *
 * Uses the non-blocking ``net`` probe from ``daemonHealth`` instead of
 * shelling out to ``lsof`` — the old lsof+timeout pattern could block
 * extension activation for up to ~10 s on a loaded host.
 */
async function isDaemonRunning(): Promise<boolean> {
  if (process.platform === 'win32') return false;
  const sockPath = path.join(HOME_DIR, '.kiss', 'sorcar.sock');
  for (let attempt = 0; attempt < 3; attempt++) {
    const health = await probeDaemonHealth(8787);
    if (health === 'alive' && fs.existsSync(sockPath)) return true;
    // TCP not up (or UDS missing — daemon still initialising); retry.
    if (attempt < 2) {
      await new Promise(r => setTimeout(r, 300));
    }
  }
  return false;
}

/**
 * Check if Playwright Chromium is already installed in the browser cache.
 * Returns true if any chromium-* directory exists in the cache.
 */
function isChromiumInstalled(): boolean {
  try {
    const cacheDir = playwrightBrowsersPath();
    if (!fs.existsSync(cacheDir)) return false;
    return fs.readdirSync(cacheDir).some(e => e.startsWith('chromium-'));
  } catch {
    return false;
  }
}

/**
 * Check if a command is available on the system.
 */
function commandExists(cmd: string): boolean {
  try {
    // H1 — argv-form so a caller-supplied ``cmd`` containing shell
    // metacharacters cannot inject extra commands.
    execFileSync(process.platform === 'win32' ? 'where' : 'which', [cmd], {
      stdio: 'ignore',
    });
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if git is functional (not just a macOS shim that triggers a CLT dialog).
 * Returns true only if `git --version` actually succeeds and outputs a version string.
 */
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

/**
 * Return a user-facing hint for how to install git manually on the current platform.
 */
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
 * Attempt to install git from prebuilt binaries.
 * Returns true if git is available after the attempt.
 *
 * macOS: tries Homebrew (binary bottles), then triggers Xcode Command Line Tools.
 * Linux: tries common package managers with non-interactive sudo.
 * Windows: downloads MinGit portable from Git for Windows releases.
 */
async function installGit(): Promise<boolean> {
  log('Git not found, attempting to install...');

  if (process.platform === 'darwin') {
    // Try Homebrew first — downloads a prebuilt bottle, no user interaction needed
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

    // Fall back to Xcode Command Line Tools (installs Apple's prebuilt git binary)
    try {
      execSync('xcode-select -p', {stdio: 'ignore'});
      // CLT already installed but git not working — unusual, nothing more we can do
      log('Xcode CLT present but git not working');
      return false;
    } catch {
      // CLT not installed — trigger the system installer dialog
    }

    log('Triggering Xcode Command Line Tools installation...');
    try {
      execSync('xcode-select --install', {stdio: 'ignore', timeout: 5_000});
    } catch {
      // Expected: opens a system dialog and may exit non-zero
    }

    // Poll for git to become available while the user completes the CLT dialog
    for (let i = 0; i < 120; i++) {
      // up to 10 minutes
      await new Promise(resolve => setTimeout(resolve, 5_000));
      if (gitWorks()) {
        log('Git installed via Xcode Command Line Tools');
        return true;
      }
    }
    return false;
  } else if (process.platform === 'linux') {
    // Try common package managers with non-interactive sudo (-n)
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

/**
 * Download MinGit (portable git for Windows) from Git for Windows releases
 * and extract it to ~/.local/git/. Adds the git cmd directory to PATH.
 */
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

    // Add MinGit's cmd directory to PATH so git.exe is found
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

/**
 * Install cloudflared from the official release asset when it is missing.
 * This is best-effort: the local web UI still works without a tunnel, so
 * failures are logged instead of blocking extension setup.
 */
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
      } catch {
        /* ignore */
      }
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
 * Install Node.js from the official binary tarball and return true on success.
 * Downloads a platform-specific archive and extracts to ~/.local/ so that
 * node, npm, and npx are available on PATH via ~/.local/bin/.
 */
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
      // Add node directory to PATH
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

  // macOS / Linux: download tar.gz and extract to ~/.local/
  const osName = process.platform === 'darwin' ? 'darwin' : 'linux';
  const assetName = `node-${NODE_VERSION}-${osName}-${arch}`;
  const url = `https://nodejs.org/dist/${NODE_VERSION}/${assetName}.tar.gz`;
  log(`Downloading Node.js from ${url}`);

  try {
    const installDir = path.join(HOME_DIR, '.local');
    fs.mkdirSync(installDir, {recursive: true});
    // Download with the Node ``https`` module (no shell), verify the
    // SHA256 against ``SHASUMS256.txt`` from nodejs.org, and extract
    // with argv-form ``tar`` (no shell).
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
    } catch {
      /* ignore */
    }
    log('Node.js installed successfully');
    return commandExists('node');
  } catch (err) {
    log(
      `Node.js installation failed: ${err instanceof Error ? err.message : err}`,
    );
    return false;
  }
}

/**
 * Ensure the VS Code CLI (`code`) is available on PATH.
 * On macOS, symlinks from the app bundle to ~/.local/bin/code.
 * On Linux, attempts snap or apt installation.
 * On Windows, VS Code's installer normally adds `code` to PATH.
 * Returns true if `code` is available after the attempt.
 */
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
        } catch {
          /* doesn't exist */
        }
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
    // Try snap first, then apt with Microsoft repo
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
  // Windows: VS Code installer normally adds `code` to PATH; nothing to do.
  return commandExists('code');
}

/**
 * Run a command with args and return a promise that resolves on exit code 0.
 */
async function runAsync(
  cmd: string,
  args: string[],
  cwd: string,
): Promise<void> {
  const cmdLine = `${cmd} ${args.join(' ')}`;
  log(`Running: ${cmdLine}`);
  let r: {code: number | null; stdout: string; stderr: string};
  try {
    // timeoutMs: 0 — installs (uv sync, Playwright download) can
    // legitimately run for many minutes; never kill them.
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

/**
 * Execute a shell command and return stdout.
 */
function execPromise(cmd: string): Promise<string> {
  return new Promise((resolve, reject) => {
    exec(cmd, {timeout: 300_000}, (err, stdout) => {
      if (err) reject(err);
      else resolve(stdout);
    });
  });
}

// ---------------------------------------------------------------------------
// API Key Setup
// ---------------------------------------------------------------------------

/**
 * Get the path to the user's shell rc file based on the SHELL environment variable.
 */
function getShellRcPath(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';

  if (process.platform === 'win32') {
    // PowerShell profile
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
 * Validate an Anthropic API key by calling the /v1/models endpoint.
 * Returns true if the key is valid (HTTP 200), false otherwise.
 */
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
        res.resume(); // consume response body
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

/**
 * Read the content of a shell rc file, creating its parent directory if needed.
 * Returns an empty string if the file doesn't exist yet.
 */
function readShellRc(rcPath: string): string {
  try {
    return fs.readFileSync(rcPath, 'utf-8');
  } catch {
    fs.mkdirSync(path.dirname(rcPath), {recursive: true});
    return '';
  }
}

/**
 * Write content to a shell rc file, ensuring a trailing newline.
 */
function writeShellRc(rcPath: string, content: string): void {
  if (content.length > 0 && !content.endsWith('\n')) {
    content += '\n';
  }
  fs.writeFileSync(rcPath, content);
}

/**
 * Add an environment variable export line to a shell rc file.
 * If the variable already exists in the file, replaces it. Otherwise appends.
 */
function addToShellRc(rcPath: string, envName: string, value: string): void {
  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  const exportLine = isPs1
    ? `$env:${envName} = "${value}"`
    : isFish
      ? `set -gx ${envName} "${value}"`
      : `export ${envName}="${value}"`;

  let content = readShellRc(rcPath);

  // Check if an export for this variable already exists
  const linePattern = isPs1
    ? new RegExp(`^\\s*\\$env:${envName}\\s*=.*$`, 'gm')
    : isFish
      ? new RegExp(`^\\s*set\\s+-gx\\s+${envName}\\s.*$`, 'gm')
      : new RegExp(`^\\s*export\\s+${envName}=.*$`, 'gm');

  if (linePattern.test(content)) {
    linePattern.lastIndex = 0; // reset after test() so replace() starts from beginning
    content = content.replace(linePattern, exportLine);
  } else {
    if (content.length > 0 && !content.endsWith('\n')) {
      content += '\n';
    }
    content += exportLine + '\n';
  }

  writeShellRc(rcPath, content);
}

/**
 * Ensure a directory is on PATH in the user's shell rc file.
 * Adds an idempotent PATH export/prepend line if not already present.
 * Uses $HOME instead of a hardcoded path for portability.
 */
function ensurePathInShellRc(rcPath: string, dirPath: string): void {
  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  // Use $HOME-relative form for portability
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  let dirRef = dirPath;
  if (homeDir && dirPath.startsWith(homeDir)) {
    dirRef = dirPath.replace(homeDir, '$HOME');
  }

  let content = readShellRc(rcPath);

  // Check if the directory is already referenced in a PATH line
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
 * Prompt the user for an API key. If a validate function is provided,
 * the key is validated and the user is re-prompted if invalid.
 * When optional is true, the prompt indicates the key can be skipped with Esc.
 * Returns the key string, or undefined if the user cancelled.
 */
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
      continue; // Empty input — re-prompt
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

/**
 * Load API keys from the user's shell rc file into process.env.
 * VS Code launched from macOS Dock/Spotlight does not source ~/.zshrc,
 * so API keys set there are invisible to process.env.  This function
 * reads the rc file and populates any missing env vars.
 */
function loadApiKeysFromShellRc(): void {
  const rcPath = getShellRcPath();
  const content = readShellRc(rcPath);
  if (!content) return;

  const isPs1 = rcPath.endsWith('.ps1');
  const isFish = rcPath.endsWith('config.fish');
  // Match uncommented export lines per shell syntax
  const pattern = isPs1
    ? /^\s*\$env:(\w+)\s*=\s*(.+)$/gm
    : isFish
      ? /^\s*set\s+-gx\s+(\w+)\s+(.+)$/gm
      : /^\s*export\s+(\w+)=(.+)$/gm;

  let match;
  while ((match = pattern.exec(content)) !== null) {
    const name = match[1];
    let value = match[2].trim();
    // Strip surrounding quotes
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

/**
 * Ensure at least one LLM API key is configured.
 * Loads existing keys from the shell rc file (needed on macOS Dock launch),
 * then skips prompting if any key is already set.  Otherwise prompts the
 * user for each provider until at least one key is provided.
 * Validates the Anthropic key if the user enters one.
 * Saves provided keys to the user's shell rc file and current process env.
 *
 * Uses a marker file (~/.kiss/.api-keys-prompted) to suppress re-prompting
 * for additional keys on subsequent VS Code restarts once at least one key
 * has been collected.
 *
 * Returns true when at least one API key is available.
 */
async function ensureApiKeys(): Promise<boolean> {
  // Load keys from shell rc into process.env so that keys saved in
  // ~/.zshrc are picked up even when VS Code wasn't launched from a shell.
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

  // If claude CLI or at least one key is already set, no prompting needed
  if (hasAnyKey()) return true;

  const markerPath = path.join(LOG_DIR, '.api-keys-prompted');
  const alreadyPrompted = fs.existsSync(markerPath);
  const rcPath = getShellRcPath();

  // Prompt for keys until at least one is provided
  while (true) {
    for (const {envName, displayName, placeholder, validate} of keys) {
      if (process.env[envName]) continue;
      // Once we have at least one key and were already prompted, skip remaining
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

    // No key provided — warn and offer retry
    const choice = await showWarningNotification(
      'KISS Sorcar requires Claude Code, ANTHROPIC_API_KEY, or OPENAI_API_KEY to work.',
      'Enter Key',
      'Skip',
    );
    if (choice !== 'Enter Key') break;
  }

  // Write marker so additional keys aren't re-prompted on next restart
  if (!alreadyPrompted) {
    try {
      fs.mkdirSync(LOG_DIR, {recursive: true});
      fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
      log('API key prompt marker written');
    } catch {
      /* ignore */
    }
  }

  return hasAnyKey();
}

/**
 * Read ``~/.kiss/config.json`` and return the parsed object, or an empty
 * object when the file is missing / unreadable.
 *
 * Retries up to ``RETRIES`` times with a small backoff to defend against
 * a concurrent writer (the Python ``save_config`` or ``install.sh``)
 * briefly truncating the file mid-write.  Only ``ENOENT`` (file does
 * not exist) short-circuits without retrying — every other failure
 * mode (empty file, ``SyntaxError`` from a half-written JSON document,
 * EBUSY, etc.) is treated as transient.
 */
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
      // The file genuinely does not exist — no point retrying.
      log(`readKissConfig: ${configPath} does not exist`);
      return {};
    }
    if (attempt < RETRIES - 1) {
      // Wait briefly (without spinning) for a concurrent writer to finish.
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

/**
 * Write ``cfg`` to ``~/.kiss/config.json`` atomically: stage in a
 * sibling temp file and then ``rename`` into place so that concurrent
 * readers never observe an empty or half-written ``config.json``.
 */
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

/**
 * Return the ``remote_password`` value from ``~/.kiss/config.json``,
 * or an empty string when it is missing / not a non-empty string.
 */
function getStoredRemotePassword(): string {
  const cfg = readKissConfig();
  const existing = cfg['remote_password'];
  if (typeof existing === 'string' && existing.length > 0) {
    return existing;
  }
  return '';
}

/**
 * Ensure the remote-access password for the KISS Sorcar web / mobile app
 * is configured.  Reads ``remote_password`` from ``~/.kiss/config.json``;
 * if it is empty or missing, prompts the user to set one.
 *
 * To guard against transient file-read failures (e.g. a concurrent
 * writer holding the file or the daemon restart briefly truncating it),
 * the config is re-read after a short delay before prompting the user.
 *
 * When the user provides a password it is persisted to ``config.json``.
 * When the user cancels, an informational message tells them the password
 * can be set later from the KISS Sorcar settings panel.
 */
async function ensureRemotePassword(): Promise<void> {
  // First read — fast path when password is already set.
  if (getStoredRemotePassword()) {
    log('ensureRemotePassword: password already set — skipping prompt');
    return;
  }

  // The daemon was just restarted; config.json may be mid-write by a
  // concurrent process (install.sh or kiss-web).  Wait briefly and retry.
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

  // Re-read config before writing so we don't clobber keys added by
  // a concurrent process between our read and this write.
  const cfg = readKissConfig();
  cfg['remote_password'] = password.trim();
  writeKissConfig(cfg);
  log('Remote access password saved to ~/.kiss/config.json');
}

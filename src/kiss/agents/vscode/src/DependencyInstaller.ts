/**
 * Auto-installation of binary dependencies for the KISS Sorcar extension.
 * Ensures uv, git, Node.js, VS Code CLI, Python environment, and Playwright
 * Chromium are available.  Called during extension activation.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as https from 'https';
import { exec, execSync, spawn } from 'child_process';
import { findKissProject, findUvPath } from './AgentProcess';

const HOME_DIR = process.env.HOME || process.env.USERPROFILE || '';
const LOG_DIR = path.join(HOME_DIR, '.kiss');
const LOG_FILE = path.join(LOG_DIR, 'install.log');
const MIN_PYTHON_MAJOR = 3;
const MIN_PYTHON_MINOR = 13;
const UV_VERSION = '0.11.2';
const NODE_VERSION = 'v22.16.0';

/**
 * Write a timestamped message to ~/.kiss/install.log and the developer console.
 */
function log(message: string): void {
  const line = `[${new Date().toISOString()}] ${message}`;
  console.log('[KISS Sorcar]', message);
  try {
    fs.mkdirSync(LOG_DIR, { recursive: true });
    fs.appendFileSync(LOG_FILE, line + '\n');
  } catch { /* ignore write errors */ }
}

/**
 * Prepend ~/.local/bin to process.env.PATH so that binaries installed by
 * the extension (uv, node, sorcar) are found by all child processes.
 * Safe to call multiple times — skips if already present.
 */
export function ensureLocalBinInPath(): void {
  if (!HOME_DIR) return;
  const localBin = path.join(HOME_DIR, '.local', 'bin');
  const parts = (process.env.PATH || '').split(path.delimiter);
  if (!parts.includes(localBin)) {
    process.env.PATH = `${localBin}${path.delimiter}${process.env.PATH || ''}`;
  }
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
  } catch { /* ignore */ }
  return baseDir;
}

/**
 * Return the best default model name based on which LLM API keys are set
 * in ``process.env``.  Priority: Anthropic > OpenRouter > Gemini > OpenAI > Together AI.
 * Falls back to ``"claude-opus-4-6"`` when no keys are present.
 */
export function getDefaultModel(): string {
  if (process.env.ANTHROPIC_API_KEY) return 'claude-opus-4-6';
  if (process.env.OPENROUTER_API_KEY) return 'openrouter/anthropic/claude-opus-4.6';
  if (process.env.GEMINI_API_KEY) return 'gemini-3.1-pro-preview';
  if (process.env.OPENAI_API_KEY) return 'gpt-5.4';
  if (process.env.TOGETHER_API_KEY) return 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8';
  return 'claude-opus-4-6';
}

/**
 * Ensure all required dependencies are installed.
 * Shows a progress notification during first-time installation.
 * Safe to call multiple times — skips already-installed dependencies.
 */
export async function ensureDependencies(): Promise<void> {
  ensureLocalBinInPath();
  log('=== Dependency check started ===');

  const kissProjectPath = findKissProject();
  if (!kissProjectPath) {
    log('KISS project not found — skipping dependency setup');
    vscode.window.showErrorMessage(
      'KISS Sorcar: Could not find the KISS project directory. ' +
      'Please set "kissSorcar.kissProjectPath" in VS Code settings. ' +
      'See ~/.kiss/install.log for details.'
    );
    return;
  }
  log(`KISS project: ${kissProjectPath}`);

  let uvPath = findUvPath();
  let venvExists = fs.existsSync(path.join(kissProjectPath, '.venv'));

  // If .venv exists but Python is too old, remove it so uv sync recreates it
  if (uvPath && venvExists && !checkPythonVersion(uvPath, kissProjectPath)) {
    try {
      fs.rmSync(path.join(kissProjectPath, '.venv'), { recursive: true, force: true });
    } catch { /* ignored */ }
    venvExists = false;
  }

  let showRestartNotification = false;

  // Fast path: everything looks ready, ensure playwright in background
  if (uvPath && venvExists) {
    log('Fast path: uv and .venv present, ensuring Playwright in background');
    const uv = uvPath; // capture narrowed non-null type for closure
    runAsync(uv, ['run', 'python', '-m', 'playwright', 'install', 'chromium'], kissProjectPath).then(() => {
      if (process.platform === 'linux') {
        return runAsync(uv, ['run', 'python', '-m', 'playwright', 'install-deps', 'chromium'], kissProjectPath);
      }
    }).catch(err => {
      log(`Fast-path Playwright install failed: ${err instanceof Error ? err.message : err}`);
      vscode.window.showWarningMessage(
        'KISS Sorcar: Chromium browser update failed in background. See ~/.kiss/install.log for details.'
      );
    });
    // Ensure git, Node.js, and VS Code CLI are available even on fast path
    if (!gitWorks()) {
      installGit().then(installed => {
        if (!installed) {
          vscode.window.showWarningMessage(
            `KISS Sorcar: git is not available. ${gitInstallHint()}`
          );
        }
      });
    }
    if (!commandExists('node')) {
      installNode().then(installed => {
        if (!installed) {
          vscode.window.showWarningMessage(
            'KISS Sorcar: Node.js could not be installed automatically. Some agent tools may be unavailable.'
          );
        }
      });
    }
    if (!commandExists('code')) { installCodeCli(); }
  } else {
    // Slow path: show progress bar and install missing deps
    const success = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'KISS Sorcar: Setting up', cancellable: false },
      async (progress) => {
        // 1. Install uv if needed
        if (!uvPath) {
          // curl and tar are required to download and extract uv
          if (process.platform !== 'win32') {
            for (const bin of ['curl', 'tar']) {
              if (!commandExists(bin)) {
                vscode.window.showErrorMessage(
                  `KISS Sorcar: '${bin}' is required to install uv but was not found. Please install '${bin}' and restart VS Code.`
                );
                return false;
              }
            }
          }
          progress.report({ message: 'Installing uv package manager...', increment: 0 });
          uvPath = await installUv();
          if (!uvPath) {
            vscode.window.showErrorMessage(
              'KISS Sorcar: Failed to install uv. Install manually: curl -LsSf https://astral.sh/uv/install.sh | sh'
            );
            return false;
          }
          progress.report({ increment: 20 });
        }

        // 2. Install git if needed
        if (!gitWorks()) {
          progress.report({ message: 'Installing git...' });
          const gitInstalled = await installGit();
          if (!gitInstalled) {
            vscode.window.showWarningMessage(
              `KISS Sorcar: git could not be installed automatically. ${gitInstallHint()}`
            );
          }
        }

        // 3. Install Node.js if needed (provides node, npm, npx for agent tasks)
        if (!commandExists('node')) {
          progress.report({ message: 'Installing Node.js...' });
          const nodeInstalled = await installNode();
          if (!nodeInstalled) {
            log('Node.js could not be installed automatically');
            vscode.window.showWarningMessage(
              'KISS Sorcar: Node.js could not be installed automatically. ' +
              'Some agent tools may be unavailable. Install from https://nodejs.org'
            );
          }
        }

        // 4. Ensure VS Code CLI is on PATH
        if (!commandExists('code')) {
          progress.report({ message: 'Setting up VS Code CLI...' });
          const codeInstalled = await installCodeCli();
          if (!codeInstalled) {
            log('VS Code CLI could not be set up on PATH');
          }
        }

        // 5. Set up Python environment (installs Python 3.13+ and all pip dependencies)
        if (!venvExists) {
          progress.report({ message: 'Setting up Python environment (first time, may take a minute)...' });
          await runAsync(uvPath, ['sync'], kissProjectPath);
          progress.report({ increment: 50 });
        }

        // 6. Verify Python version meets minimum requirement
        if (!checkPythonVersion(uvPath, kissProjectPath)) {
          vscode.window.showErrorMessage(
            `KISS Sorcar requires Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR}+. ` +
            `Please install Python ${MIN_PYTHON_MAJOR}.${MIN_PYTHON_MINOR} or later and restart VS Code.`
          );
          return false;
        }

        // 7. Install Playwright Chromium
        progress.report({ message: 'Installing Chromium browser...' });
        await runAsync(
          uvPath,
          ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
          kissProjectPath
        );
        if (process.platform === 'linux') {
          await runAsync(
            uvPath,
            ['run', 'python', '-m', 'playwright', 'install-deps', 'chromium'],
            kissProjectPath
          ).catch(err => log(`Playwright deps install failed (may need sudo): ${err instanceof Error ? err.message : err}`));
        }
        progress.report({ increment: 30 });
        return true;
      }
    );

    showRestartNotification = !!success;
  }

  // Install CLI wrapper so `sorcar` is available from any terminal
  if (uvPath) {
    installCliScript(kissProjectPath, uvPath);
  }

  // Persist PATH entries to the user's shell rc file so new terminals find
  // installed binaries (uv, node, sorcar, etc.) without manual setup.
  try {
    const rcPath = getShellRcPath();
    const localBin = path.join(HOME_DIR, '.local', 'bin');
    ensurePathInShellRc(rcPath, localBin);
    // If MinGit was installed on Windows, persist its PATH entry too
    if (process.platform === 'win32') {
      const gitCmdDir = path.join(HOME_DIR, '.local', 'git', 'cmd');
      if (fs.existsSync(gitCmdDir)) {
        ensurePathInShellRc(rcPath, gitCmdDir);
      }
      // Find the actual nested node directory (e.g. node-v22.16.0-win-x64)
      const nodeBaseDir = path.join(HOME_DIR, '.local', 'node');
      const nodeDir = findNodeDirWindows(nodeBaseDir);
      if (fs.existsSync(nodeDir)) {
        ensurePathInShellRc(rcPath, nodeDir);
      }
    }
  } catch (err) {
    log(`Failed to update shell rc PATH: ${err instanceof Error ? err.message : err}`);
  }

  log('=== Dependency check finished ===');

  // Prompt for missing API keys (returns true when at least one key is set)
  const apiKeysReady = await ensureApiKeys();

  // Show restart notification only after API key prompting has completed.
  if (showRestartNotification) {
    if (apiKeysReady) {
      vscode.window.showInformationMessage(
        'KISS Sorcar: Installation complete! Please restart VS Code and any open terminal for changes to take effect.',
        'Restart VS Code'
      ).then(choice => {
        if (choice === 'Restart VS Code') {
          vscode.commands.executeCommand('workbench.action.reloadWindow');
        }
      });
    } else {
      vscode.window.showWarningMessage(
        'KISS Sorcar: Installation complete, but at least one LLM API key is required. ' +
        'Set an API key (Anthropic, OpenAI, Gemini, Together AI, or OpenRouter) in your environment or restart VS Code to be prompted again.'
      );
    }
  }
}

/**
 * Install a `sorcar` CLI wrapper script in ~/.local/bin/ so users can
 * invoke the agent from any terminal after the extension is installed.
 * The wrapper calls `uv run sorcar` from the bundled kiss_project directory.
 */
function installCliScript(kissProjectPath: string, uvPath: string): void {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  if (!homeDir) return;

  const binDir = path.join(homeDir, '.local', 'bin');

  // Resolve uv to an absolute path for the wrapper script
  let absUvPath = uvPath;
  if (uvPath === 'uv' || !path.isAbsolute(uvPath)) {
    try {
      const whichCmd = process.platform === 'win32' ? `where ${uvPath}` : `which ${uvPath}`;
      absUvPath = execSync(whichCmd, { encoding: 'utf-8' }).trim().split('\n')[0];
    } catch {
      const suffix = process.platform === 'win32' ? '.exe' : '';
      absUvPath = path.join(homeDir, '.local', 'bin', `uv${suffix}`);
    }
  }

  try {
    fs.mkdirSync(binDir, { recursive: true });

    if (process.platform === 'win32') {
      const cmdPath = path.join(binDir, 'sorcar.cmd');
      const script =
        `@echo off\r\n` +
        `REM Installed by KISS Sorcar VS Code extension\r\n` +
        `"${absUvPath}" run --directory "${kissProjectPath}" sorcar %*\r\n`;
      fs.writeFileSync(cmdPath, script);
    } else {
      const scriptPath = path.join(binDir, 'sorcar');
      const script =
        `#!/bin/bash\n` +
        `# Installed by KISS Sorcar VS Code extension\n` +
        `exec "${absUvPath}" run --directory "${kissProjectPath}" sorcar "$@"\n`;
      fs.writeFileSync(scriptPath, script, { mode: 0o755 });
    }
  } catch (err) {
    log(`Failed to install CLI script: ${err instanceof Error ? err.message : err}`);
  }
}

/**
 * Map Node.js platform/arch to the uv GitHub release asset triplet.
 * Returns [archName, platformSuffix, extension] or null if unsupported.
 */
function uvAssetInfo(): { archName: string; triplet: string; ext: string } | null {
  const archMap: Record<string, string> = {
    'arm64': 'aarch64',
    'x64': 'x86_64',
  };
  const arch = archMap[process.arch];
  if (!arch) return null;

  if (process.platform === 'darwin') {
    return { archName: arch, triplet: `${arch}-apple-darwin`, ext: 'tar.gz' };
  } else if (process.platform === 'linux') {
    return { archName: arch, triplet: `${arch}-unknown-linux-gnu`, ext: 'tar.gz' };
  } else if (process.platform === 'win32') {
    return { archName: arch, triplet: `${arch}-pc-windows-msvc`, ext: 'zip' };
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
    log(`Unsupported platform/arch for uv: ${process.platform}/${process.arch}`);
    return null;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const installDir = path.join(homeDir, '.local', 'bin');
  const assetName = `uv-${asset.triplet}`;
  const url = `https://releases.astral.sh/github/uv/releases/download/${UV_VERSION}/${assetName}.${asset.ext}`;
  log(`Downloading uv ${UV_VERSION} from ${url}`);

  try {
    // Ensure install directory exists
    fs.mkdirSync(installDir, { recursive: true });

    if (process.platform === 'win32') {
      // Windows: download zip and extract with PowerShell
      const zipPath = path.join(installDir, `${assetName}.zip`);
      await execPromise(
        `powershell -Command "Invoke-WebRequest -Uri '${url}' -OutFile '${zipPath}'; ` +
        `Expand-Archive -Force -Path '${zipPath}' -DestinationPath '${installDir}'; ` +
        `Move-Item -Force '${path.join(installDir, assetName, 'uv.exe')}' '${path.join(installDir, 'uv.exe')}'; ` +
        `Move-Item -Force '${path.join(installDir, assetName, 'uvx.exe')}' '${path.join(installDir, 'uvx.exe')}'; ` +
        `Remove-Item -Force '${zipPath}'; Remove-Item -Recurse -Force '${path.join(installDir, assetName)}'"`
      );
    } else {
      // macOS/Linux: download tar.gz and extract with tar, then move binaries
      await execPromise(
        `curl -fsSL '${url}' | tar xz -C '${installDir}' && ` +
        `mv -f '${path.join(installDir, assetName, 'uv')}' '${installDir}/' && ` +
        `mv -f '${path.join(installDir, assetName, 'uvx')}' '${installDir}/' && ` +
        `rm -rf '${path.join(installDir, assetName)}' && ` +
        `chmod +x '${path.join(installDir, 'uv')}' '${path.join(installDir, 'uvx')}'`
      );
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
 * Returns true if the version meets the requirement, false otherwise.
 */
function checkPythonVersion(uvPath: string, cwd: string): boolean {
  try {
    const output = execSync(`"${uvPath}" run python --version`, {
      cwd,
      encoding: 'utf-8',
      timeout: 30_000,
    }).trim();
    // Output format: "Python 3.13.2"
    const match = output.match(/Python\s+(\d+)\.(\d+)/);
    if (!match) return false;
    const major = parseInt(match[1], 10);
    const minor = parseInt(match[2], 10);
    return major > MIN_PYTHON_MAJOR || (major === MIN_PYTHON_MAJOR && minor >= MIN_PYTHON_MINOR);
  } catch {
    return false;
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
        log(`Homebrew git install failed: ${err instanceof Error ? err.message : err}`);
      }
    }

    // Fall back to Xcode Command Line Tools (installs Apple's prebuilt git binary)
    try {
      execSync('xcode-select -p', { stdio: 'ignore' });
      // CLT already installed but git not working — unusual, nothing more we can do
      log('Xcode CLT present but git not working');
      return false;
    } catch {
      // CLT not installed — trigger the system installer dialog
    }

    log('Triggering Xcode Command Line Tools installation...');
    try {
      execSync('xcode-select --install', { stdio: 'ignore', timeout: 5_000 });
    } catch {
      // Expected: opens a system dialog and may exit non-zero
    }

    // Poll for git to become available while the user completes the CLT dialog
    for (let i = 0; i < 120; i++) {   // up to 10 minutes
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
      ['apt-get', 'sudo -n sh -c "apt-get update -y && apt-get install -y git"'],
      ['dnf',     'sudo -n dnf install -y git'],
      ['yum',     'sudo -n yum install -y git'],
      ['pacman',  'sudo -n pacman -S --noconfirm git'],
      ['apk',     'sudo -n apk add git'],
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
    fs.mkdirSync(gitDir, { recursive: true });

    const zipPath = path.join(gitDir, `${assetName}.zip`);
    await execPromise(
      `powershell -Command "` +
      `Invoke-WebRequest -Uri '${url}' -OutFile '${zipPath}'; ` +
      `Expand-Archive -Force -Path '${zipPath}' -DestinationPath '${gitDir}'; ` +
      `Remove-Item -Force '${zipPath}'"`
    );

    // Add MinGit's cmd directory to PATH so git.exe is found
    const gitCmdDir = path.join(gitDir, 'cmd');
    if (fs.existsSync(path.join(gitCmdDir, 'git.exe'))) {
      const parts = (process.env.PATH || '').split(path.delimiter);
      if (!parts.includes(gitCmdDir)) {
        process.env.PATH = `${gitCmdDir}${path.delimiter}${process.env.PATH || ''}`;
      }
      log('MinGit installed successfully');
      return true;
    }
    log('MinGit extracted but git.exe not found in cmd/');
  } catch (err) {
    log(`MinGit installation failed: ${err instanceof Error ? err.message : err}`);
  }
  return false;
}

/**
 * Install Node.js from the official binary tarball and return true on success.
 * Downloads a platform-specific archive and extracts to ~/.local/ so that
 * node, npm, and npx are available on PATH via ~/.local/bin/.
 */
async function installNode(): Promise<boolean> {
  const archMap: Record<string, string> = { 'arm64': 'arm64', 'x64': 'x64' };
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
      fs.mkdirSync(installDir, { recursive: true });
      const zipPath = path.join(installDir, `${assetName}.zip`);
      await execPromise(
        `powershell -Command "` +
        `Invoke-WebRequest -Uri '${url}' -OutFile '${zipPath}'; ` +
        `Expand-Archive -Force -Path '${zipPath}' -DestinationPath '${installDir}'; ` +
        `Remove-Item -Force '${zipPath}'"`
      );
      // Add node directory to PATH
      const nodeDir = path.join(installDir, assetName);
      if (fs.existsSync(path.join(nodeDir, 'node.exe'))) {
        const parts = (process.env.PATH || '').split(path.delimiter);
        if (!parts.includes(nodeDir)) {
          process.env.PATH = `${nodeDir}${path.delimiter}${process.env.PATH || ''}`;
        }
        log('Node.js installed successfully (Windows)');
        return true;
      }
    } catch (err) {
      log(`Node.js installation failed: ${err instanceof Error ? err.message : err}`);
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
    fs.mkdirSync(installDir, { recursive: true });
    await execPromise(
      `curl -fsSL '${url}' | tar xz -C '${installDir}' --strip-components=1`
    );
    log('Node.js installed successfully');
    return commandExists('node');
  } catch (err) {
    log(`Node.js installation failed: ${err instanceof Error ? err.message : err}`);
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
    const vscodeApp = '/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code';
    if (fs.existsSync(vscodeApp)) {
      const binDir = path.join(HOME_DIR, '.local', 'bin');
      try {
        fs.mkdirSync(binDir, { recursive: true });
        const linkPath = path.join(binDir, 'code');
        try { fs.unlinkSync(linkPath); } catch { /* doesn't exist */ }
        fs.symlinkSync(vscodeApp, linkPath);
        log('VS Code CLI symlinked to ~/.local/bin/code');
        return true;
      } catch (err) {
        log(`Failed to symlink VS Code CLI: ${err instanceof Error ? err.message : err}`);
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
          'sudo -n apt-get update -y && sudo -n apt-get install -y code'
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
function runAsync(cmd: string, args: string[], cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const cmdLine = `${cmd} ${args.join(' ')}`;
    log(`Running: ${cmdLine}`);
    const proc = spawn(cmd, args, {
      cwd,
      stdio: 'pipe',
      env: { ...process.env, PYTHONUNBUFFERED: '1' },
    });
    let output = '';
    proc.stdout?.on('data', (d: Buffer) => { output += d.toString(); });
    proc.stderr?.on('data', (d: Buffer) => { output += d.toString(); });
    proc.on('close', (code) => {
      if (output.trim()) log(`Output [${cmdLine}]:\n${output.trim()}`);
      if (code === 0) {
        log(`Completed: ${cmdLine}`);
        resolve();
      } else {
        reject(new Error(`${cmdLine} failed (exit code ${code}): ${output}`));
      }
    });
    proc.on('error', (err) => {
      log(`Spawn error [${cmdLine}]: ${err.message}`);
      reject(err);
    });
  });
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
  return new Promise((resolve) => {
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
      (res) => {
        resolve(res.statusCode === 200);
        res.resume(); // consume response body
      }
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
    fs.mkdirSync(path.dirname(rcPath), { recursive: true });
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
    dirRef = isPs1
      ? dirPath.replace(homeDir, '$HOME')
      : dirPath.replace(homeDir, '$HOME');
  }

  let content = readShellRc(rcPath);

  // Check if the directory is already referenced in a PATH line
  const escaped = dirRef.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
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
  optional?: boolean
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
        const choice = await vscode.window.showWarningMessage(
          `${displayName} is required for KISS Sorcar to function.`,
          'Enter Key', 'Skip'
        );
        if (choice === 'Enter Key') { continue; }
      }
      return undefined;
    }

    const trimmed = key.trim();
    if (!trimmed) {
      continue; // Empty input — re-prompt
    }

    if (validate) {
      const valid = await vscode.window.withProgress(
        { location: vscode.ProgressLocation.Notification, title: `Validating ${displayName}...` },
        () => validate(trimmed)
      );

      if (!valid) {
        const choice = await vscode.window.showWarningMessage(
          `The ${displayName} is not valid. Please try again.`,
          'Try Again',
          'Cancel'
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
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
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
    { envName: 'ANTHROPIC_API_KEY', displayName: 'Anthropic API Key', placeholder: 'sk-ant-...', validate: validateAnthropicKey },
    { envName: 'OPENAI_API_KEY', displayName: 'OpenAI API Key', placeholder: 'sk-...' },
    { envName: 'GEMINI_API_KEY', displayName: 'Gemini API Key', placeholder: 'AI...' },
    { envName: 'TOGETHER_API_KEY', displayName: 'Together API Key', placeholder: 'tok-...' },
    { envName: 'OPENROUTER_API_KEY', displayName: 'OpenRouter API Key', placeholder: 'sk-or-...' },
  ];

  const hasAnyKey = () => keys.some(k => !!process.env[k.envName]);

  // If at least one key is already set, no prompting needed
  if (hasAnyKey()) return true;

  const markerPath = path.join(LOG_DIR, '.api-keys-prompted');
  const alreadyPrompted = fs.existsSync(markerPath);
  const rcPath = getShellRcPath();

  // Prompt for keys until at least one is provided
  while (true) {
    for (const { envName, displayName, placeholder, validate } of keys) {
      if (process.env[envName]) continue;
      // Once we have at least one key and were already prompted, skip remaining
      if (hasAnyKey() && alreadyPrompted) break;

      const key = await promptForApiKey(displayName, placeholder, validate, true);
      if (key) {
        process.env[envName] = key;
        addToShellRc(rcPath, envName, key);
        log(`${displayName} saved to ~/${path.basename(rcPath)}`);
      }
    }

    if (hasAnyKey()) break;

    // No key provided — warn and offer retry
    const choice = await vscode.window.showWarningMessage(
      'KISS Sorcar requires at least one LLM API key (Anthropic, OpenAI, Gemini, Together AI, or OpenRouter).',
      'Enter Key', 'Skip'
    );
    if (choice !== 'Enter Key') break;
  }

  // Write marker so additional keys aren't re-prompted on next restart
  if (!alreadyPrompted) {
    try {
      fs.mkdirSync(LOG_DIR, { recursive: true });
      fs.writeFileSync(markerPath, new Date().toISOString() + '\n');
      log('API key prompt marker written');
    } catch { /* ignore */ }
  }

  return hasAnyKey();
}

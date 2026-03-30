/**
 * Auto-installation of binary dependencies for the KISS Sorcar extension.
 * Ensures uv, git, Python environment, and Playwright Chromium are available.
 * Called during extension activation.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import * as https from 'https';
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
  } else {
    // Slow path: show progress bar and install missing deps
    const success = await vscode.window.withProgress(
      { location: vscode.ProgressLocation.Notification, title: 'KISS Sorcar: Setting up', cancellable: false },
      async (progress) => {
        // 1. Install uv if needed
        if (!uvPath) {
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

        // 2. Warn about git
        if (!commandExists('git')) {
          vscode.window.showWarningMessage(
            'KISS Sorcar: git is required but not found. Please install git.'
          );
        }

        // 3. Set up Python environment (installs Python 3.13+ and all pip dependencies)
        if (!venvExists) {
          progress.report({ message: 'Setting up Python environment (first time, may take a minute)...' });
          try {
            await runAsync(uvPath, ['sync'], kissProjectPath);
          } catch (err) {
            const msg = err instanceof Error ? err.message : String(err);
            vscode.window.showErrorMessage(
              `KISS Sorcar: Python environment setup failed — ${msg}. Check ~/.kiss/install.log for details.`
            );
            return false;
          }
          progress.report({ increment: 50 });
        }

        // 4. Install Playwright Chromium
        progress.report({ message: 'Installing Chromium browser...' });
        try {
          await runAsync(
            uvPath,
            ['run', 'python', '-m', 'playwright', 'install', 'chromium'],
            kissProjectPath
          );
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          vscode.window.showErrorMessage(
            `KISS Sorcar: Chromium browser installation failed — ${msg}. Check ~/.kiss/install.log for details.`
          );
          return false;
        }
        progress.report({ increment: 30 });
        return true;
      }
    );

    // Show restart notification only when installation actually succeeded
    if (success) {
      vscode.window.showInformationMessage(
        'KISS Sorcar: Installation complete! Please restart VS Code and any open terminal for changes to take effect.',
        'Restart VS Code'
      ).then(choice => {
        if (choice === 'Restart VS Code') {
          vscode.commands.executeCommand('workbench.action.reloadWindow');
        }
      });
    }
  }

  // Install CLI wrapper so `sorcar` is available from any terminal
  if (uvPath) {
    installCliScript(kissProjectPath, uvPath);
  }

  // Prompt for missing API keys
  await ensureApiKeys();
}

/**
 * Install a `sorcar` CLI wrapper script in ~/.local/bin/ so users can
 * invoke the agent from any terminal after the extension is installed.
 * The wrapper calls `uv run sorcar` from the bundled kiss_project directory.
 */
function installCliScript(kissProjectPath: string, uvPath: string): void {
  if (process.platform === 'win32') {
    return; // TODO: Windows .cmd wrapper
  }

  const homeDir = process.env.HOME || '';
  if (!homeDir) return;

  const binDir = path.join(homeDir, '.local', 'bin');
  const scriptPath = path.join(binDir, 'sorcar');

  // Resolve uv to an absolute path for the wrapper script
  let absUvPath = uvPath;
  if (uvPath === 'uv' || !path.isAbsolute(uvPath)) {
    try {
      absUvPath = execSync(`which ${uvPath}`, { encoding: 'utf-8' }).trim();
    } catch {
      absUvPath = path.join(homeDir, '.local', 'bin', 'uv');
    }
  }

  const script =
    `#!/bin/bash\n` +
    `# Installed by KISS Sorcar VS Code extension\n` +
    `exec "${absUvPath}" run --directory "${kissProjectPath}" sorcar "$@"\n`;

  try {
    fs.mkdirSync(binDir, { recursive: true });
    fs.writeFileSync(scriptPath, script, { mode: 0o755 });
  } catch (err) {
    console.error('[KISS Sorcar] Failed to install CLI script:', err);
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
 * Install uv from the latest GitHub binary release and return its path, or null on failure.
 * Downloads the platform-specific binary from https://github.com/astral-sh/uv/releases/latest
 * and extracts it to ~/.local/bin/.
 */
async function installUv(): Promise<string | null> {
  const asset = uvAssetInfo();
  if (!asset) {
    console.error(`[KISS Sorcar] Unsupported platform/arch: ${process.platform}/${process.arch}`);
    return null;
  }

  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const installDir = path.join(homeDir, '.local', 'bin');
  const assetName = `uv-${asset.triplet}`;
  const url = `https://github.com/astral-sh/uv/releases/latest/download/${assetName}.${asset.ext}`;

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

    return findUvPath();
  } catch (err) {
    console.error('[KISS Sorcar] Failed to install uv from GitHub release:', err);
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

// ---------------------------------------------------------------------------
// API Key Setup
// ---------------------------------------------------------------------------

/**
 * Get the path to the user's shell rc file based on the SHELL environment variable.
 */
function getShellRcPath(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
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
 * Add an environment variable export line to a shell rc file.
 * If the variable already exists in the file, replaces it. Otherwise appends.
 */
function addToShellRc(rcPath: string, envName: string, value: string): void {
  const isFish = rcPath.endsWith('config.fish');
  const exportLine = isFish
    ? `set -gx ${envName} "${value}"`
    : `export ${envName}="${value}"`;

  let content = '';
  try {
    content = fs.readFileSync(rcPath, 'utf-8');
  } catch {
    // File doesn't exist yet — will be created
    const dir = path.dirname(rcPath);
    fs.mkdirSync(dir, { recursive: true });
  }

  // Check if an export for this variable already exists
  const linePattern = isFish
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

  fs.writeFileSync(rcPath, content);
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
      return undefined; // User pressed Escape
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
 * Ensure all LLM API keys are configured.
 * Prompts the user for each missing key. Validates the Anthropic key.
 * Saves provided keys to the user's shell rc file and current process env.
 */
async function ensureApiKeys(): Promise<void> {
  const rcPath = getShellRcPath();

  const keys = [
    { envName: 'ANTHROPIC_API_KEY', displayName: 'Anthropic API Key', placeholder: 'sk-ant-...', validate: true },
    { envName: 'OPENAI_API_KEY', displayName: 'OpenAI API Key', placeholder: 'sk-...', validate: false },
    { envName: 'GEMINI_API_KEY', displayName: 'Gemini API Key', placeholder: 'AI...', validate: false },
    { envName: 'TOGETHER_API_KEY', displayName: 'Together API Key', placeholder: 'tok-...', validate: false },
    { envName: 'OPENROUTER_API_KEY', displayName: 'OpenRouter API Key', placeholder: 'sk-or-...', validate: false },
  ];

  for (const { envName, displayName, placeholder, validate } of keys) {
    if (process.env[envName]) {
      continue; // Already set in environment
    }

    const key = await promptForApiKey(
      displayName,
      placeholder,
      validate ? validateAnthropicKey : undefined,
      true
    );

    if (key) {
      process.env[envName] = key;
      addToShellRc(rcPath, envName, key);
      vscode.window.showInformationMessage(
        `${displayName} saved to ~/${path.basename(rcPath)}`
      );
    }
  }
}

/**
 * Python agent subprocess manager.
 * Spawns and communicates with the Sorcar agent backend.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ChildProcess, spawn, execSync } from 'child_process';
import { EventEmitter } from 'events';
import { AgentCommand, ToWebviewMessage } from './types';

// ---------------------------------------------------------------------------
// Module-level utility functions (shared with DependencyInstaller)
// ---------------------------------------------------------------------------

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

function searchUpward(startDir: string): string | null {
  let dir = startDir;
  for (let i = 0; i < 10; i++) {
    if (isValidKissProject(dir)) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return null;
}

/**
 * Find the KISS project root directory.
 * Search order:
 * 0. Embedded kiss_project/ inside the extension (standalone mode)
 * 1. Configuration setting
 * 2. Search up from workspace folders
 * 3. Search up from extension directory
 * 4. Environment variable
 * 5. Common locations
 */
export function findKissProject(): string | null {
  // 0. Check for embedded kiss_project/ (standalone mode)
  const embeddedPath = path.join(__dirname, '..', 'kiss_project');
  if (isValidKissProject(embeddedPath)) return embeddedPath;

  // 1. Check configuration setting
  const configPath = vscode.workspace.getConfiguration('kissSorcar').get<string>('kissProjectPath');
  if (configPath && isValidKissProject(configPath)) return configPath;

  // 2. Search up from workspace folders
  const workspaceFolders = vscode.workspace.workspaceFolders;
  if (workspaceFolders) {
    for (const folder of workspaceFolders) {
      const found = searchUpward(folder.uri.fsPath);
      if (found) return found;
    }
  }

  // 3. Search from this file's directory upward
  const found = searchUpward(__dirname);
  if (found) return found;

  // 4. Environment variable
  const envPath = process.env.KISS_PROJECT_PATH;
  if (envPath && isValidKissProject(envPath)) return envPath;

  // 5. Common locations
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  for (const p of [
    path.join(homeDir, 'work', 'kiss'),
    path.join(homeDir, 'projects', 'kiss'),
    path.join(homeDir, 'dev', 'kiss'),
    path.join(homeDir, 'kiss'),
  ]) {
    if (isValidKissProject(p)) return p;
  }

  return null;
}

/**
 * Find the uv binary path, or null if not installed anywhere.
 */
export function findUvPath(): string | null {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '';
  const candidates = [
    path.join(homeDir, '.local', 'bin', 'uv'),
    path.join(homeDir, '.cargo', 'bin', 'uv'),
    '/usr/local/bin/uv',
    '/opt/homebrew/bin/uv',
  ];
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate)) return candidate;
    } catch {
      continue;
    }
  }
  // Check if uv is on PATH
  try {
    execSync(process.platform === 'win32' ? 'where uv' : 'which uv', { stdio: 'ignore' });
    return 'uv';
  } catch {
    return null;
  }
}

/**
 * Find the uv binary path, with 'uv' fallback for spawning.
 */
export function findUvBinary(): string {
  return findUvPath() ?? 'uv';
}

// ---------------------------------------------------------------------------
// AgentProcess class
// ---------------------------------------------------------------------------

export class AgentProcess extends EventEmitter {
  private process: ChildProcess | null = null;
  private kissProjectPath: string | null = null;
  private buffer: string = '';

  constructor() {
    super();
  }

  /**
   * Start the Python backend process.
   */
  start(workDir: string): boolean {
    if (this.process) {
      return true; // Already running
    }

    this.kissProjectPath = findKissProject();
    if (!this.kissProjectPath) {
      this.emit('message', {
        type: 'error',
        text: 'Could not find KISS project. Please set kissSorcar.kissProjectPath in settings.'
      } as ToWebviewMessage);
      return false;
    }

    const serverModule = 'kiss.agents.vscode.server';
    const pythonArgs = ['-u', '-m', serverModule];
    const uvBin = findUvBinary();
    const args = ['run', 'python', ...pythonArgs];

    try {
      this.process = spawn(uvBin, args, {
        cwd: this.kissProjectPath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          KISS_WORKDIR: workDir,
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      });

      this.process.stdout?.on('data', (data: Buffer) => {
        this.handleStdout(data.toString());
      });

      this.process.stderr?.on('data', (data: Buffer) => {
        const text = data.toString();
        console.error('[AgentProcess stderr]', text);
      });

      this.process.on('close', (code) => {
        console.log(`[AgentProcess] Process exited with code ${code}`);
        this.process = null;
        this.emit('message', { type: 'status', running: false } as ToWebviewMessage);
      });

      this.process.on('error', (err) => {
        console.error('[AgentProcess error]', err);
        this.emit('message', {
          type: 'error',
          text: `Failed to start agent: ${err.message}`
        } as ToWebviewMessage);
        this.process = null;
      });

      return true;
    } catch (err) {
      console.error('[AgentProcess] Failed to spawn:', err);
      return false;
    }
  }

  /**
   * Handle stdout data, parsing JSON events.
   */
  private handleStdout(data: string): void {
    this.buffer += data;
    const lines = this.buffer.split('\n');
    // Keep the last incomplete line in the buffer
    this.buffer = lines.pop() || '';

    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const event = JSON.parse(line) as ToWebviewMessage;
        this.emit('message', event);
      } catch {
        // Not JSON, might be raw output
        console.log('[AgentProcess raw]', line);
      }
    }
  }

  /**
   * Send a command to the Python backend.
   */
  sendCommand(cmd: AgentCommand): void {
    if (!this.process?.stdin?.writable) {
      this.emit('message', {
        type: 'error',
        text: 'Agent process not running'
      } as ToWebviewMessage);
      return;
    }

    try {
      const line = JSON.stringify(cmd) + '\n';
      this.process.stdin.write(line);
    } catch {
      this.emit('message', {
        type: 'error',
        text: 'Failed to send command to agent'
      } as ToWebviewMessage);
    }
  }

  /**
   * Stop the current task.
   */
  stop(): void {
    this.sendCommand({ type: 'stop' });
  }

  /**
   * Cleanup and terminate the process.
   *
   * Closes stdin first so the Python server's ``for line in sys.stdin``
   * loop sees EOF and exits cleanly.  SIGTERM and SIGKILL follow as
   * fallbacks to avoid leaving zombie processes that block VS Code
   * shutdown.
   */
  dispose(): void {
    if (this.process) {
      const proc = this.process;
      this.process = null;
      this.removeAllListeners();
      // Close stdin to unblock the Python server's main loop
      try { proc.stdin?.end(); } catch { /* ignored */ }
      // SIGTERM as backup
      try { proc.kill('SIGTERM'); } catch { /* ignored */ }
      // SIGKILL after 2s if the process is still alive
      setTimeout(() => {
        try { proc.kill('SIGKILL'); } catch { /* ignored */ }
      }, 2000);
    } else {
      this.removeAllListeners();
    }
  }
}

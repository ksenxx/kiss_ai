/**
 * Python agent subprocess manager.
 * Spawns and communicates with the Sorcar agent backend.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import {ChildProcess, spawn, execSync} from 'child_process';
import {EventEmitter} from 'events';
import {AgentCommand, ToWebviewMessage} from './types';

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

/**
 * Find the KISS project root directory.
 * Search order:
 * 1. Environment variable (explicit override, e.g. Docker containers)
 * 2. Configuration setting (kissSorcar.kissProjectPath)
 * 3. Embedded kiss_project directory bundled with the extension
 */
export function findKissProject(): string | null {
  // H5 — only honour explicit workspace-scoped overrides (env var or
  // setting) inside a *trusted* workspace.  A malicious workspace's
  // .vscode/settings.json must not be able to redirect the agent
  // process at attacker-controlled code, since the agent later runs
  // arbitrary shell commands on user request.  When the workspace is
  // not trusted, we fall back to the bundled embedded project.
  const isTrusted = vscode.workspace.isTrusted;

  if (isTrusted) {
    // 1. Environment variable (highest priority — explicit user/Docker override)
    const envPath = process.env.KISS_PROJECT_PATH;
    if (envPath && isValidKissProject(envPath)) return envPath;

    // 2. Check configuration setting
    const configPath = vscode.workspace
      .getConfiguration('kissSorcar')
      .get<string>('kissProjectPath');
    if (configPath && isValidKissProject(configPath)) return configPath;
  }

  // 3. Embedded kiss_project bundled inside the extension directory
  // (always allowed; this is shipped with the extension and trusted by
  //  installation).
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
  // Check if uv is on PATH
  try {
    execSync(process.platform === 'win32' ? 'where uv' : 'which uv', {
      stdio: 'ignore',
    });
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

/**
 * Maximum size of the per-line stdout buffer for the Python backend.
 * If a single JSON line ever exceeds this limit the process is killed
 * to avoid OOMing the extension host (M3 — robustness fix).
 */
const MAX_STDOUT_BUFFER_BYTES = 32 * 1024 * 1024; // 32 MB

export class AgentProcess extends EventEmitter {
  private process: ChildProcess | null = null;
  private kissProjectPath: string | null = null;
  private buffer: string = '';
  /** Tab ID this process is associated with (empty for shared processes). */
  public readonly tabId: string;

  constructor(tabId: string = '') {
    super();
    this.tabId = tabId;
  }

  /** True when the underlying child process is still alive. */
  get isAlive(): boolean {
    return this.process !== null;
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
        text: 'Could not find KISS project. Please set kissSorcar.kissProjectPath in settings.',
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

      this.process.on('close', code => {
        console.log(`[AgentProcess] Process exited with code ${code}`);
        // RC11 fix: flush any remaining buffer content
        if (this.buffer.trim()) {
          try {
            const event = JSON.parse(this.buffer) as ToWebviewMessage;
            this.emit('message', event);
          } catch {
            /* not valid JSON — discard */
          }
        }
        this.buffer = '';
        this.process = null;
        this.emit('message', {
          type: 'status',
          running: false,
        } as ToWebviewMessage);
      });

      this.process.on('error', err => {
        console.error('[AgentProcess error]', err);
        this.emit('message', {
          type: 'error',
          text: `Failed to start agent: ${err.message}`,
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
    // M3: cap the unparsed-line buffer.  A backend that emits one huge
    // JSON line with no newline would otherwise grow the buffer until
    // the extension host runs out of memory.
    if (this.buffer.length > MAX_STDOUT_BUFFER_BYTES) {
      console.error(
        '[AgentProcess] stdout buffer exceeded limit ' +
          `(${this.buffer.length} > ${MAX_STDOUT_BUFFER_BYTES}); ` +
          'killing the agent.',
      );
      this.buffer = '';
      this.dispose();
      return;
    }
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
        text: 'Agent process not running',
      } as ToWebviewMessage);
      return;
    }

    try {
      const line = JSON.stringify(cmd) + '\n';
      this.process.stdin.write(line);
    } catch {
      this.emit('message', {
        type: 'error',
        text: 'Failed to send command to agent',
      } as ToWebviewMessage);
    }
  }

  /**
   * Stop the current task.
   */
  stop(): void {
    this.sendCommand({type: 'stop'});
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
      try {
        proc.stdin?.end();
      } catch {
        /* ignored */
      }
      if (process.platform === 'win32') {
        // On Windows, SIGTERM/SIGKILL both map to TerminateProcess — one call suffices
        try {
          proc.kill();
        } catch {
          /* ignored */
        }
      } else {
        // SIGTERM as backup
        try {
          proc.kill('SIGTERM');
        } catch {
          /* ignored */
        }
        // SIGKILL after 2s if the process is still alive
        setTimeout(() => {
          try {
            proc.kill('SIGKILL');
          } catch {
            /* ignored */
          }
        }, 2000);
      }
    } else {
      this.removeAllListeners();
    }
  }
}

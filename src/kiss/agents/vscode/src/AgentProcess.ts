/**
 * Python agent subprocess manager.
 * Spawns and communicates with the Sorcar agent backend.
 */

import * as vscode from 'vscode';
import * as path from 'path';
import * as fs from 'fs';
import { ChildProcess, spawn } from 'child_process';
import { EventEmitter } from 'events';
import { AgentCommand, ToWebviewMessage } from './types';

export class AgentProcess extends EventEmitter {
  private process: ChildProcess | null = null;
  private kissProjectPath: string | null = null;
  private buffer: string = '';

  constructor() {
    super();
  }

  /**
   * Find the KISS project root directory.
   * Search order:
   * 1. Configuration setting
   * 2. Search up from workspace folders
   * 3. Search up from extension directory
   * 4. Environment variable
   * 5. Common locations
   */
  findKissProject(): string | null {
    // 1. Check configuration setting
    const configPath = vscode.workspace.getConfiguration('kissSorcar').get<string>('kissProjectPath');
    if (configPath && this.isValidKissProject(configPath)) {
      return configPath;
    }

    // 2. Search up from workspace folders
    const workspaceFolders = vscode.workspace.workspaceFolders;
    if (workspaceFolders) {
      for (const folder of workspaceFolders) {
        const found = this.searchUpward(folder.uri.fsPath);
        if (found) return found;
      }
    }

    // 3. Search from this file's directory upward
    // The extension is at src/kiss/agents/vscode, so go up 4 levels
    const extensionDir = __dirname;
    const found = this.searchUpward(extensionDir);
    if (found) return found;

    // 4. Environment variable
    const envPath = process.env.KISS_PROJECT_PATH;
    if (envPath && this.isValidKissProject(envPath)) {
      return envPath;
    }

    // 5. Common locations
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const commonPaths = [
      path.join(homeDir, 'work', 'kiss'),
      path.join(homeDir, 'projects', 'kiss'),
      path.join(homeDir, 'dev', 'kiss'),
      path.join(homeDir, 'kiss'),
    ];
    for (const p of commonPaths) {
      if (this.isValidKissProject(p)) {
        return p;
      }
    }

    return null;
  }

  private searchUpward(startDir: string): string | null {
    let dir = startDir;
    for (let i = 0; i < 10; i++) {
      if (this.isValidKissProject(dir)) {
        return dir;
      }
      const parent = path.dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
    return null;
  }

  private isValidKissProject(dir: string): boolean {
    try {
      const pyproject = path.join(dir, 'pyproject.toml');
      if (!fs.existsSync(pyproject)) return false;
      const content = fs.readFileSync(pyproject, 'utf-8');
      // Check if it contains kiss project marker (kiss or kiss-agent-framework)
      return content.includes('name = "kiss') || content.includes("name = 'kiss");
    } catch {
      return false;
    }
  }

  /**
   * Find the uv binary path.
   * Checks common locations since VSCode Desktop may not have user's PATH.
   */
  private findUvBinary(): string {
    const homeDir = process.env.HOME || process.env.USERPROFILE || '';
    const candidates = [
      path.join(homeDir, '.local', 'bin', 'uv'),
      path.join(homeDir, '.cargo', 'bin', 'uv'),
      '/usr/local/bin/uv',
      '/opt/homebrew/bin/uv',
      'uv', // fallback to PATH
    ];
    for (const candidate of candidates) {
      if (candidate === 'uv') return candidate;
      try {
        if (fs.existsSync(candidate)) return candidate;
      } catch {
        continue;
      }
    }
    return 'uv';
  }

  /**
   * Start the Python backend process.
   */
  start(workDir: string): boolean {
    if (this.process) {
      return true; // Already running
    }

    this.kissProjectPath = this.findKissProject();
    if (!this.kissProjectPath) {
      this.emit('message', {
        type: 'error',
        text: 'Could not find KISS project. Please set kissSorcar.kissProjectPath in settings.'
      } as ToWebviewMessage);
      return false;
    }

    const serverModule = 'kiss.agents.vscode.server';
    const pythonArgs = ['-u', '-m', serverModule];
    const uvBin = this.findUvBinary();
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
    if (!this.process?.stdin) {
      this.emit('message', {
        type: 'error',
        text: 'Agent process not running'
      } as ToWebviewMessage);
      return;
    }

    const line = JSON.stringify(cmd) + '\n';
    this.process.stdin.write(line);
  }

  /**
   * Stop the current task.
   */
  stop(): void {
    this.sendCommand({ type: 'stop' });
  }

  /**
   * Cleanup and terminate the process.
   */
  dispose(): void {
    if (this.process) {
      this.process.kill('SIGTERM');
      this.process = null;
    }
    this.removeAllListeners();
  }

  /**
   * Check if process is running.
   */
  isRunning(): boolean {
    return this.process !== null;
  }
}

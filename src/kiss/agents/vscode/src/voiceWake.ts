// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Host-side "Sorcar" wake-word listener for the VS Code extension.
 *
 * Extension webviews cannot capture the microphone (VS Code denies
 * ``getUserMedia`` for webview origins), so the always-on listener runs
 * here in the extension host instead: a local Python process
 * (``kiss.agents.vscode.voice_wake``) streams microphone audio into the
 * lightweight offline Vosk small English model and prints one line per
 * event on stdout:
 *
 * - ``READY`` — model loaded, microphone open, listening.
 * - ``WAKE``  — the wake word "Sorcar" was heard.
 *
 * The service forwards those events to the webview, where voice.js
 * types "sorcar" into the task input and the listener simply keeps
 * running.  Everything is fully local; no audio leaves the machine.
 */

import {spawn, ChildProcess} from 'child_process';
import {findKissProject, findUvPath} from './kissPaths';

/** Callback invoked whenever the wake word is detected. */
export type WakeCallback = () => void;

/** Callback reporting listener state changes to the UI. */
export type StateCallback = (listening: boolean, error?: string) => void;

export class VoiceWakeService {
  private _proc: ChildProcess | undefined;

  constructor(
    private readonly _onWake: WakeCallback,
    private readonly _onState: StateCallback,
  ) {}

  /** Whether the listener process is currently running. */
  get running(): boolean {
    return this._proc !== undefined;
  }

  /**
   * Start the wake-word listener process (idempotent).
   *
   * Reports failures through the state callback instead of throwing so
   * the webview toggle always settles into a definite on/off/error
   * state.
   */
  start(): void {
    if (this._proc) {
      this._onState(true);
      return;
    }
    const kissProject = findKissProject();
    const uv = findUvPath();
    if (!kissProject || !uv) {
      this._onState(
        false,
        'KISS project or uv binary not found; cannot start voice listener',
      );
      return;
    }
    let proc: ChildProcess;
    try {
      proc = spawn(
        uv,
        ['run', 'python', '-m', 'kiss.agents.vscode.voice_wake'],
        {cwd: kissProject, stdio: ['ignore', 'pipe', 'pipe']},
      );
    } catch (err) {
      this._onState(false, `voice listener failed to start: ${String(err)}`);
      return;
    }
    this._proc = proc;

    let stdoutBuf = '';
    proc.stdout?.on('data', (chunk: Buffer) => {
      stdoutBuf += chunk.toString('utf-8');
      let idx = stdoutBuf.indexOf('\n');
      while (idx >= 0) {
        const line = stdoutBuf.slice(0, idx).trim();
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (line === 'WAKE') this._onWake();
        else if (line === 'READY') this._onState(true);
        idx = stdoutBuf.indexOf('\n');
      }
    });

    let stderrTail = '';
    proc.stderr?.on('data', (chunk: Buffer) => {
      // Keep only the tail; vosk/kaldi are chatty on stderr.
      stderrTail = (stderrTail + chunk.toString('utf-8')).slice(-2000);
    });

    proc.on('error', (err: Error) => {
      if (this._proc === proc) {
        this._proc = undefined;
        this._onState(false, `voice listener error: ${err.message}`);
      }
    });

    proc.on('exit', (code: number | null) => {
      if (this._proc !== proc) return; // superseded or stopped on purpose
      this._proc = undefined;
      if (code === 0 || code === null) {
        this._onState(false);
      } else {
        const detail = stderrTail.trim().split('\n').pop() || '';
        this._onState(
          false,
          `voice listener exited (code ${code})${detail ? ': ' + detail : ''}`,
        );
      }
    });
  }

  /** Stop the listener process (idempotent). */
  stop(): void {
    const proc = this._proc;
    this._proc = undefined;
    if (proc) {
      try {
        proc.kill();
      } catch {
        // Already dead — nothing to do.
      }
    }
    this._onState(false);
  }

  /** Release all resources; the service must not be reused after this. */
  dispose(): void {
    this.stop();
  }
}

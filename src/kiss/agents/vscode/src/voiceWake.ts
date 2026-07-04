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
 * - ``READY``         — model loaded, microphone open, listening.
 * - ``WAKE``          — the wake word "Sorcar" was heard.
 * - ``TRANSCRIBING``  — speech capture ended; the gpt-audio call started.
 * - ``SPEECH <json>`` — the speech that followed the wake word,
 *   translated to English by the gpt-audio model (JSON string payload).
 * - ``NO_SPEECH``     — only silence followed the wake word.
 *
 * The service forwards those events to the webview, where voice.js
 * inserts the translated text into the task input and the listener
 * simply keeps running.  Wake-word detection is fully local; only the
 * post-wake utterance is sent to the GPT translation API.
 */

import {spawn, ChildProcess} from 'child_process';
import {findKissProject, findUvPath} from './kissPaths';

/** Callback invoked whenever the wake word is detected. */
export type WakeCallback = () => void;

/** Callback reporting listener state changes to the UI. */
export type StateCallback = (listening: boolean, error?: string) => void;

/**
 * Callback receiving the English translation of the speech following
 * the wake word ('' when only silence was heard).
 */
export type SpeechCallback = (text: string) => void;

/**
 * Callback invoked when the post-wake capture ends and the gpt-audio
 * transcription/translation call starts (the UI shows a "transcribing"
 * indicator until the speech or silence result arrives).
 */
export type TranscribingCallback = () => void;

/**
 * Extra CLI arguments for the Python listener, JSON-encoded in the
 * ``KISS_VOICE_WAKE_ARGS`` environment variable (e.g. ``["--wav",
 * "f.wav"]``).  Used by the end-to-end tests to feed recorded audio.
 */
function extraListenerArgs(): string[] {
  const raw = process.env.KISS_VOICE_WAKE_ARGS;
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every(a => typeof a === 'string')) {
      return parsed;
    }
  } catch {
    // Malformed override — ignore it.
  }
  return [];
}

export class VoiceWakeService {
  private _proc: ChildProcess | undefined;

  constructor(
    private readonly _onWake: WakeCallback,
    private readonly _onState: StateCallback,
    private readonly _onSpeech: SpeechCallback,
    private readonly _onTranscribing: TranscribingCallback,
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
        [
          'run',
          'python',
          '-m',
          'kiss.agents.vscode.voice_wake',
          ...extraListenerArgs(),
        ],
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
        else if (line === 'TRANSCRIBING') this._onTranscribing();
        else if (line === 'NO_SPEECH') this._onSpeech('');
        else if (line.startsWith('SPEECH ')) {
          try {
            const text = JSON.parse(line.slice('SPEECH '.length));
            if (typeof text === 'string') this._onSpeech(text);
          } catch {
            // Malformed payload — drop the event.
          }
        }
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

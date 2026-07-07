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
 * - ``SPEECH <json>`` — the speech that followed the wake word, as a
 *   JSON object ``{"text": <english translation>, "speaker": <int or
 *   null>, "language": <language tag or null>}``; the text and the
 *   spoken language come from a KISS transcription agent, and the
 *   speaker number comes from local voice recognition (each distinct
 *   voice gets a unique number starting from 1).  Legacy plain
 *   JSON-string payloads are also accepted.
 * - ``NO_SPEECH``     — only silence followed the wake word.
 *
 * The service forwards those events to the webview, where voice.js
 * inserts the translated text into the task input and the listener
 * simply keeps running.  Wake-word detection is fully local; only the
 * post-wake utterance is sent to the GPT translation API.
 */

import {spawn, spawnSync, ChildProcess} from 'child_process';
import {findKissProject, findUvPath} from './kissPaths';

/** Callback invoked whenever the wake word is detected. */
export type WakeCallback = () => void;

/** Callback reporting listener state changes to the UI. */
export type StateCallback = (listening: boolean, error?: string) => void;

/**
 * Callback receiving the English translation of the speech following
 * the wake word ('' when only silence was heard) plus the speaker
 * number assigned by voice recognition — a unique integer starting
 * from 1 per distinct voice — or undefined when identification was
 * unavailable, plus the language tag of the spoken speech (e.g.
 * "en", "fr") reported by the transcription agent, or undefined when
 * the language is unknown.
 */
export type SpeechCallback = (
  text: string,
  speaker?: number,
  language?: string,
) => void;

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
   *
   * @param sensitivity Wake-word sensitivity 0..100 (settings-panel
   *   slider), forwarded to the Python listener as ``--sensitivity``;
   *   omitted or non-finite values fall back to the listener default.
   */
  start(sensitivity?: number): void {
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
    const sensitivityArgs =
      typeof sensitivity === 'number' && Number.isFinite(sensitivity)
        ? [
            '--sensitivity',
            String(Math.min(100, Math.max(0, Math.round(sensitivity)))),
          ]
        : [];
    let proc: ChildProcess;
    try {
      proc = spawn(
        uv,
        [
          'run',
          'python',
          '-m',
          'kiss.agents.vscode.voice_wake',
          ...sensitivityArgs,
          ...extraListenerArgs(),
        ],
        // On POSIX, ``detached`` puts the listener in its own process
        // group so stop() can signal the WHOLE tree: killing only the
        // ``uv run`` wrapper leaves the actual Python child alive as
        // an orphan still holding the microphone (observed live).
        // Windows has no negative-PID process-group signal; stop()
        // uses taskkill /T there, so avoid detached's "keep running
        // after parent exits" side effect on that platform.  stdio
        // stays piped, so no unref() — the event loop keeps watching it.
        {
          cwd: kissProject,
          stdio: ['ignore', 'pipe', 'pipe'],
          detached: process.platform !== 'win32',
        },
      );
    } catch (err) {
      this._onState(false, `voice listener failed to start: ${String(err)}`);
      return;
    }
    this._proc = proc;

    let stdoutBuf = '';
    proc.stdout?.on('data', (chunk: Buffer) => {
      if (this._proc !== proc) return; // stale output after stop/restart
      stdoutBuf += chunk.toString('utf-8');
      let idx = stdoutBuf.indexOf('\n');
      while (idx >= 0) {
        const line = stdoutBuf.slice(0, idx).trim();
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (this._proc !== proc) return; // callback may have stopped us
        if (line === 'WAKE') this._onWake();
        else if (line === 'READY') this._onState(true);
        else if (line === 'TRANSCRIBING') this._onTranscribing();
        else if (line === 'NO_SPEECH') this._onSpeech('');
        else if (line.startsWith('SPEECH ')) {
          // Parse first, deliver once: a SPEECH line is the terminal
          // event of a voice round, and dropping it (as the old code
          // did for malformed payloads) leaked the webview's round
          // counter, leaving the mic button blinking yellow after
          // every later utterance.  A malformed or unusable payload
          // therefore degrades to an empty terminal instead of no
          // terminal, and the callback runs outside the try so its
          // own exceptions are never mistaken for a parse failure.
          let text = '';
          let speaker: number | undefined;
          let language: string | undefined;
          try {
            const payload = JSON.parse(line.slice('SPEECH '.length));
            if (typeof payload === 'string') {
              // Legacy payload shape: the bare translated text.
              text = payload;
            } else if (
              payload &&
              typeof payload === 'object' &&
              typeof payload.text === 'string'
            ) {
              text = payload.text;
              const spk = payload.speaker;
              const lang = payload.language;
              if (typeof spk === 'number' && Number.isInteger(spk) && spk >= 1)
                speaker = spk;
              if (typeof lang === 'string' && lang) language = lang;
            }
          } catch {
            // Malformed JSON — fall through to the empty terminal.
          }
          this._onSpeech(text, speaker, language);
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

    proc.on('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      if (this._proc !== proc) return; // superseded or stopped on purpose
      this._proc = undefined;
      if (code === 0 || (code === null && signal === null)) {
        this._onState(false);
      } else {
        const detail = stderrTail.trim().split('\n').pop() || '';
        const reason = signal ? `signal ${signal}` : `code ${code}`;
        this._onState(
          false,
          `voice listener exited (${reason})${detail ? ': ' + detail : ''}`,
        );
      }
    });
  }

  /**
   * Stop the listener process tree (idempotent).
   *
   * The listener runs under a ``uv run`` wrapper; signalling only the
   * wrapper can orphan the Python child, which keeps the microphone
   * open (observed live).  The process was spawned detached into its
   * own process group, so the whole tree is killed with one negative-
   * PID signal, falling back to a plain kill of the wrapper.
   */
  stop(): void {
    const proc = this._proc;
    this._proc = undefined;
    if (proc) {
      try {
        if (typeof proc.pid === 'number') {
          if (process.platform === 'win32') {
            const result = spawnSync(
              'taskkill',
              ['/PID', String(proc.pid), '/T', '/F'],
              {stdio: 'ignore', windowsHide: true},
            );
            if (result.error || result.status !== 0) proc.kill();
          } else {
            process.kill(-proc.pid, 'SIGTERM');
          }
        } else {
          proc.kill();
        }
      } catch {
        try {
          proc.kill();
        } catch {
          // Already dead — nothing to do.
        }
      }
    }
    this._onState(false);
  }

  /** Release all resources; the service must not be reused after this. */
  dispose(): void {
    this.stop();
  }
}

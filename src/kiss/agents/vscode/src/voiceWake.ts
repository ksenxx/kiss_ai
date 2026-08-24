// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import {spawn, spawnSync, ChildProcess} from 'child_process';
import {findKissProject, findUvPath} from './kissPaths';

// A wake and the transcript that answers it carry the same round id. The
// listener resumes wake detection while the previous utterance is still being
// transcribed, so rounds overlap and can finish out of order; the id is what
// lets the webview pair a transcript with the conversation that was on screen
// when those particular words were spoken.
export type WakeCallback = (roundId: number) => void;

export type StateCallback = (listening: boolean, error?: string) => void;

export type SpeechCallback = (
  roundId: number,
  text: string,
  speaker?: number,
  language?: string,
) => void;

export type TranscribingCallback = () => void;

function extraListenerArgs(): string[] {
  const raw = process.env.KISS_VOICE_WAKE_ARGS;
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed) && parsed.every(a => typeof a === 'string')) {
      return parsed;
    }
  } catch {}
  return [];
}

export class VoiceWakeService {
  // Stays assigned until the child's exit event fires: a listener that is
  // still dying holds the exclusive microphone, so it must keep counting
  // as "present" until it has actually exited.
  private _proc: ChildProcess | undefined;

  // Resolves when the child being stopped has exited. start() calls that
  // arrive while this is pending are queued (see _queuedStart) instead of
  // spawning a second listener against the mic the dying one still holds.
  private _stopping: Promise<void> | undefined;

  // The start() request deferred until the in-flight stop completes.
  // A later stop() cancels it (hide → show → hide must end stopped).
  private _queuedStart: {sensitivity?: number} | undefined;

  // The child whose exit was requested by stop(), so the exit handler
  // reports a clean shutdown instead of a listener error.
  private _stopRequestedFor: ChildProcess | undefined;

  // Monotonic across the whole session, so a round id is never reused and a
  // transcript can only ever be paired with the wake it came from.
  private _roundId: number = 0;

  // The round the listener is currently transcribing. voice_wake prints one
  // SPEECH/NO_SPEECH per WAKE, in that order, so the id of the last WAKE is
  // the id the next transcript answers.
  private _speechRoundId: number = 0;

  constructor(
    private readonly _onWake: WakeCallback,
    private readonly _onState: StateCallback,
    private readonly _onSpeech: SpeechCallback,
    private readonly _onTranscribing: TranscribingCallback,
  ) {}

  get running(): boolean {
    // A dying listener still holds the microphone and a queued start
    // will hold it again momentarily; both count as "running" so that
    // visibility handlers suspend/resume symmetrically around them.
    return this._proc !== undefined || this._queuedStart !== undefined;
  }

  start(sensitivity?: number): void {
    if (this._stopping) {
      // The previous listener is still dying; spawning now would fight
      // it for the exclusive microphone. Defer until it has exited
      // (stop() cancels the deferred start).
      this._queuedStart = {sensitivity};
      return;
    }
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
          'kiss.server.voice_wake',
          ...sensitivityArgs,
          ...extraListenerArgs(),
        ],
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

    // Output from a listener whose stop has been requested is stale:
    // the process may keep printing while it shuts down (or while it
    // ignores SIGTERM until the SIGKILL escalation), and none of it may
    // flip the UI back to "listening".
    const isActive = () =>
      this._proc === proc && this._stopRequestedFor !== proc;
    let stdoutBuf = '';
    proc.stdout?.on('data', (chunk: Buffer) => {
      if (!isActive()) return;
      stdoutBuf += chunk.toString('utf-8');
      let idx = stdoutBuf.indexOf('\n');
      while (idx >= 0) {
        const line = stdoutBuf.slice(0, idx).trim();
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (!isActive()) return;
        if (line === 'WAKE') {
          this._speechRoundId = ++this._roundId;
          this._onWake(this._speechRoundId);
        } else if (line === 'READY') this._onState(true);
        else if (line === 'TRANSCRIBING') this._onTranscribing();
        else if (line === 'NO_SPEECH') this._onSpeech(this._speechRoundId, '');
        else if (line.startsWith('SPEECH ')) {
          let text = '';
          let speaker: number | undefined;
          let language: string | undefined;
          try {
            const payload = JSON.parse(line.slice('SPEECH '.length));
            if (typeof payload === 'string') {
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
          } catch {}
          this._onSpeech(this._speechRoundId, text, speaker, language);
        }
        idx = stdoutBuf.indexOf('\n');
      }
    });

    let stderrTail = '';
    proc.stderr?.on('data', (chunk: Buffer) => {
      stderrTail = (stderrTail + chunk.toString('utf-8')).slice(-2000);
    });

    proc.on('error', (err: Error) => {
      if (this._proc === proc) {
        this._proc = undefined;
        this._onState(false, `voice listener error: ${err.message}`);
      }
    });

    proc.on('exit', (code: number | null, signal: NodeJS.Signals | null) => {
      if (this._proc !== proc) return;
      this._proc = undefined;
      const requested = this._stopRequestedFor === proc;
      if (requested) this._stopRequestedFor = undefined;
      if (requested || code === 0 || (code === null && signal === null)) {
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
   * Stop the listener and report when it has actually exited.
   *
   * The kill is asynchronous: on exclusive-capture audio backends a new
   * listener spawned while the old process is still dying fails to open
   * the microphone. The child therefore stays tracked (and `running`
   * stays true) until its exit event fires, and a start() issued while
   * the stop is in flight is queued behind it instead of overlapping it
   * — so even fire-and-forget callers cannot double-open the mic.
   * Resolves immediately when no listener is running; a process that
   * ignores SIGTERM is SIGKILLed after 5s as an escalation, but the
   * promise still resolves only from the exit event itself.
   *
   * @returns A promise that resolves once the child process has exited.
   */
  stop(): Promise<void> {
    this._queuedStart = undefined;
    const proc = this._proc;
    if (!proc) {
      this._onState(false);
      return this._stopping ?? Promise.resolve();
    }
    if (this._stopping) return this._stopping;
    this._stopRequestedFor = proc;
    let exited: Promise<void> = Promise.resolve();
    if (proc.exitCode === null && proc.signalCode === null) {
      exited = new Promise<void>(resolve => {
        // SIGKILL is an escalation only: resolution comes solely from
        // the exit event, never from the timer, so the promise cannot
        // claim the child is gone while it still holds the microphone.
        const killTimer = setTimeout(() => {
          try {
            if (typeof proc.pid === 'number' && process.platform !== 'win32') {
              process.kill(-proc.pid, 'SIGKILL');
            } else {
              proc.kill('SIGKILL');
            }
          } catch {
            try {
              proc.kill('SIGKILL');
            } catch {}
          }
        }, 5000);
        proc.once('exit', () => {
          clearTimeout(killTimer);
          resolve();
        });
      });
    }
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
      } catch {}
    }
    this._onState(false);
    const settled: Promise<void> = exited.then(() => {
      if (this._proc === proc) this._proc = undefined;
      if (this._stopping === settled) this._stopping = undefined;
      const queued = this._queuedStart;
      if (queued) {
        this._queuedStart = undefined;
        this.start(queued.sensitivity);
      }
    });
    this._stopping = settled;
    return settled;
  }

  dispose(): void {
    void this.stop();
  }
}

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
  private _proc: ChildProcess | undefined;

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
    return this._proc !== undefined;
  }

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

    let stdoutBuf = '';
    proc.stdout?.on('data', (chunk: Buffer) => {
      if (this._proc !== proc) return;
      stdoutBuf += chunk.toString('utf-8');
      let idx = stdoutBuf.indexOf('\n');
      while (idx >= 0) {
        const line = stdoutBuf.slice(0, idx).trim();
        stdoutBuf = stdoutBuf.slice(idx + 1);
        if (this._proc !== proc) return;
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
        } catch {}
      }
    }
    this._onState(false);
  }

  dispose(): void {
    this.stop();
  }
}

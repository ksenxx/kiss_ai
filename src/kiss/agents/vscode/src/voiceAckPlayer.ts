// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import {spawn} from 'child_process';
import {commandExists} from './kissPaths';

function shellSplit(command: string): string[] {
  const parts: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(command)) !== null) {
    parts.push(m[1] ?? m[2] ?? m[3]);
  }
  return parts;
}

const FALLBACK_PLAYERS: string[][] = [
  ['mpg123', '-q'],
  ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'],
  ['mpv', '--no-video', '--really-quiet'],
];

// Result of the PATH probes, memoised per process: every spoken
// acknowledgement would otherwise run up to four synchronous `which`
// calls on the extension host's event loop.
let probedPlayer: string[] | null | undefined;

function probePlayer(): string[] | null {
  if (process.platform === 'darwin' && commandExists('afplay')) {
    return ['afplay'];
  }
  for (const candidate of FALLBACK_PLAYERS) {
    if (commandExists(candidate[0])) return candidate;
  }
  return null;
}

/**
 * The argv prefix that plays an mp3 file, or null when no player is
 * installed.  `KISS_SORCAR_PLAY_CMD` in `env` overrides the PATH probe.
 */
export function ackPlayerCommand(
  env: NodeJS.ProcessEnv = process.env,
): string[] | null {
  const override = (env.KISS_SORCAR_PLAY_CMD || '').trim();
  if (override) {
    const argv = shellSplit(override);
    if (argv.length) return argv;
  }
  if (probedPlayer === undefined) probedPlayer = probePlayer();
  return probedPlayer ? [...probedPlayer] : null;
}

export function playVoiceAckClip(mp3Path: string): void {
  try {
    const argv = ackPlayerCommand();
    if (!argv) return;
    const child = spawn(argv[0], [...argv.slice(1), mp3Path], {
      stdio: 'ignore',
      detached: true,
    });
    child.on('error', () => {});
    child.unref();
  } catch {}
}

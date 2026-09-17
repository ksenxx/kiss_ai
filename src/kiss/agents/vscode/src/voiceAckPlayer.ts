// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import {spawn, spawnSync} from 'child_process';

// Synchronous PATH probes run on the extension host's event loop (the
// webview's voiceAck message handler calls ackPlayerCommand), so they
// must never wait on a hung child (e.g. a PATH entry on a stalled
// network mount).  Twin of DependencyInstaller's SYNC_PROBE_TIMEOUT_MS.
const PROBE_TIMEOUT_MS = 5_000;

function shellSplit(command: string): string[] {
  const parts: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(command)) !== null) {
    parts.push(m[1] ?? m[2] ?? m[3]);
  }
  return parts;
}

function commandExists(cmd: string): boolean {
  try {
    const probe = process.platform === 'win32' ? 'where' : 'which';
    // killSignal SIGKILL: Node's default timeout kill is SIGTERM, which
    // a stalled probe (e.g. blocked on a dead network mount) can ignore,
    // leaving spawnSync blocked past its timeout.
    return (
      spawnSync(probe, [cmd], {
        stdio: 'ignore',
        timeout: PROBE_TIMEOUT_MS,
        killSignal: 'SIGKILL',
      }).status === 0
    );
  } catch {
    return false;
  }
}

const FALLBACK_PLAYERS: string[][] = [
  ['mpg123', '-q'],
  ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'],
  ['mpv', '--no-video', '--really-quiet'],
];

export function ackPlayerCommand(
  env: NodeJS.ProcessEnv = process.env,
): string[] | null {
  const override = (env.KISS_SORCAR_PLAY_CMD || '').trim();
  if (override) {
    const argv = shellSplit(override);
    if (argv.length) return argv;
  }
  if (process.platform === 'darwin' && commandExists('afplay')) {
    return ['afplay'];
  }
  for (const candidate of FALLBACK_PLAYERS) {
    if (commandExists(candidate[0])) return [...candidate];
  }
  return null;
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

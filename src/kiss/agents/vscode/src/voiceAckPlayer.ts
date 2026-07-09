// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Native playback of the "Working on it." voice-dictation
// acknowledgement clip (media/working-on-it.mp3).
//
// The chat webview cannot play the clip itself: Chromium's autoplay
// policy rejects `Audio.play()` in a VS Code webview unless the user
// clicked it moments earlier (microsoft/vscode#197937), and a
// voice-dictated task involves no click.  The old Web Speech fallback
// produced the loud robotic "alien voice" on every spoken task, so
// voice.js now posts `{type: 'voiceAck'}` and the extension host plays
// the clip natively on this machine's speakers — the same strategy the
// kiss-web daemon uses for agent `talk` clips
// (kiss.agents.sorcar.cli_talk / web_server._play_talk_clip_locally).
//
// Player resolution mirrors cli_talk.player_command(): the
// KISS_SORCAR_PLAY_CMD environment override (shell-split; the audio
// path is appended as the last argument — tests substitute a real
// scripted child process), otherwise `afplay` on macOS, otherwise the
// first of mpg123 / ffplay / mpv found on PATH.  A machine with no
// player degrades to silence — never to the robotic voice.

import {spawn, spawnSync} from 'child_process';

/** Shell-ish split honouring double/single quotes (KISS_SORCAR_PLAY_CMD). */
function shellSplit(command: string): string[] {
  const parts: string[] = [];
  const re = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(command)) !== null) {
    parts.push(m[1] ?? m[2] ?? m[3]);
  }
  return parts;
}

/** True when `cmd` resolves to an executable on PATH. */
function commandExists(cmd: string): boolean {
  try {
    const probe = process.platform === 'win32' ? 'where' : 'which';
    return spawnSync(probe, [cmd], {stdio: 'ignore'}).status === 0;
  } catch {
    return false;
  }
}

// MP3-capable players, most common first (cli_talk parity); each entry
// is the argv prefix — the audio path is appended as the last argument.
const FALLBACK_PLAYERS: string[][] = [
  ['mpg123', '-q'],
  ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'],
  ['mpv', '--no-video', '--really-quiet'],
];

/**
 * Return the argv prefix used to play an audio file, or null when this
 * machine has no known audio player (callers then stay silent).
 */
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

/**
 * Play the acknowledgement clip at `mp3Path` natively, fire-and-forget.
 *
 * Spawns the resolved player fully detached with stdio ignored so the
 * extension host never blocks on (or crashes with) the child; every
 * failure — no player, spawn error — degrades to silence.
 */
export function playVoiceAckClip(mp3Path: string): void {
  try {
    const argv = ackPlayerCommand();
    if (!argv) return;
    const child = spawn(argv[0], [...argv.slice(1), mp3Path], {
      stdio: 'ignore',
      detached: true,
    });
    child.on('error', () => {
      /* player refused to start; stay silent */
    });
    child.unref();
  } catch {
    /* never let ack playback break the message handler */
  }
}

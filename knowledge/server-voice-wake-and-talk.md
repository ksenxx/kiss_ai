---
title: 'Voice: "Hey Sorcar" wake word (listener, VS Code host, daemon control, browser
  fallback) and talk() playback'
uuid: 8e9d12a0-ff96-4172-a4ee-b0d78186c934
summary: 'Hey Sorcar wake word: Vosk listener protocol, VS Code VoiceWakeService and
  voice.js, daemon voiceWakeStart/Stop, hostMicUnavailable browser fallback, and talk()
  playback (_fanout_talk, TalkPlayer).'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:24Z'
---
# Voice: "Hey Sorcar" wake word and talk() playback

## Listener (`kiss.server.voice_wake`)
A child process running offline Vosk `vosk-model-small-en-us-0.15` (~40 MB) on the mic. Stdout, one event per line: `READY` (mic open), `WAKE`, `TRANSCRIBING`, `SPEECH {"text", "speaker", "language"}` (English translation), `NO_SPEECH` (silence, failed translation, false wake).
- **Grammar:** only phonetic aliases of "Hey Sorcar" (`WAKE_ALIASES`; "sorcar" is not in the vocabulary) plus mandatory `[unk]`, without which Kaldi stalls on out-of-grammar audio.
- **No substring matching** ("yes sir the car is ready" → `[unk] sir car [unk]` used to fire). Wakes on exactly one alias (optionally after a short `[unk]` onset), no very low-confidence alias word, partials only after ~100 ms pause. At `SUFFIX_MATCH_SENSITIVITY = 75` or above (default `DEFAULT_SENSITIVITY = 80`) an utterance ending in an alias also wakes. `--sensitivity` 0-100 moves the gates.
- **Dual check:** speech after a wake is RMS-endpointed, the 2 s pre-wake ring (`WAKE_PREAMBLE_SECONDS`) is prepended, and a non-agentic `KISSAgent` run of `gpt-audio` (`DEFAULT_AUDIO_MODEL`) returns translation and language. The transcript must start with "Sorcar"-like text (`split_wake_prefix`) or it is a false wake; non-English speech relies on the local check only.
- Speaker ids: `vosk-model-spk-0.4` x-vectors, stable numbers from 1. Translation runs on one worker, so a new `WAKE` may precede the previous `SPEECH`.

## VS Code host (`VoiceWakeService`)
`media/voice.js` drives `#voice-btn`/`#task-input`; phrases mirror `WAKE_ALIASES`; sensitivity and auto-submit are in `localStorage` (`kissVoiceSensitivity`, `kissVoiceAutoSubmit`); config comes from `window.__VOICE__` (`VOICE_CONFIG`: `{mode: 'webview', ackAudioUrl, voskSrc, modelUrl, nonce}`).

On `voiceToggle {enabled, sensitivity}`, `SorcarSidebarView` lazily creates a `VoiceWakeService`, whose `start` spawns `uv run python -m kiss.server.voice_wake --sensitivity N` in the Python project (detached group on POSIX). Relays: `READY`→`voiceState {listening: true}`, `WAKE`→`voiceWake {roundId}`, `TRANSCRIBING`→`voiceTranscribing`, `NO_SPEECH`→`voiceSpeech {roundId, text: ''}`, `SPEECH`→`voiceSpeech {roundId, text, speaker, language}`. The service keeps a single `_speechRoundId`, overwritten on each `WAKE`, so overlapping wakes can label a late transcript with the newer round. Output from a stopping listener is ignored; no respawn while the old one still holds the mic.

**hostMicUnavailable:** if the child dies before `READY` (PortAudio missing on a headless host, uv/project missing, spawn failure) the view clears `_voiceEnabled` and posts `voiceState {listening: false, hostMicUnavailable: true}`; the webview falls back to the browser mic via `media/vosk.js` or shows "voice capture unavailable". Hence the CSP allows `'wasm-unsafe-eval'`, `worker-src blob:` and `connect-src` to the model origin.

**Lifecycle:** hide stops the listener (`_voiceWakeSuspendedByHide`), show restarts it, dispose stops it and clears intent. The webview fetches `VOICE_MODEL_URL` (ccoreilly.github.io) directly since it cannot reach the daemon's HTTPS port; the remote webapp always uses the browser path, with the daemon proxying `/voice-model.tar.gz`. `voiceAckPlayer.ts` plays `media/working-on-it.mp3` (`KISS_SORCAR_PLAY_CMD`, `afplay`, `mpg123`, `ffplay`, `mpv`).

## Daemon control (`voice_wake_control.py`)
API commands `voiceWakeStart`/`voiceWakeStop` (`voice_wake_start`/`voice_wake_stop`) let a client run the listener as a daemon child over the UDS; the current extension does not use them. `VoiceWakeController` keeps one listener per `conn_id`; disconnect cleanup calls `stop(conn_id)`. `parse_protocol_line` yields `voiceWakeEvent`; start/exit are `voiceWakeState`; generation counters (`_bump_generation`, `accepts`) drop stale events. `voiceToggle`, `voiceSensitivity`, `voiceAck` are catalog `drop` commands.

## talk() playback
`talk` emits a `talk` event (base64 MP3, `talkId`). `WebPrinter._fanout_talk`: Chromium autoplay blocks `Audio.play()` in local webviews (microsoft/vscode#197937), so when a local tab is subscribed (`_local_tab_shown`) the daemon plays natively (`_play_talk_clip_locally`) and marks local UDS copies `"muted": true`. Remote WSS copies stay playable, as do all copies without a player or clip.

`TalkPlayer` (singleton) plays via `afplay`/`mpg123`/`ffplay`/`mpv`, falling back to TTS (`say`, `espeak`, `spd-say`); dedupes by `talkId`; serializes on one worker. Env: `KISS_SORCAR_PLAY_CMD`, `KISS_SORCAR_SAY_CMD`, `KISS_SORCAR_PLAY_TIMEOUT` (600 s). Timeouts kill the child's process group (grandchild players hold the device), else `proc.kill()`.

## Sources
- `src/kiss/server/voice_wake.py` (`WAKE_ALIASES`, `DEFAULT_SENSITIVITY`, `SUFFIX_MATCH_SENSITIVITY`, `split_wake_prefix`)
- `src/kiss/server/voice_wake_control.py` (`VoiceWakeController`, `parse_protocol_line`)
- `src/kiss/server/sorcar.py` (`voice_wake_start`, `voice_wake_stop`)
- `src/kiss/server/web_server.py` (`_fanout_talk`, `_play_talk_clip_locally`)
- `src/kiss/server/talk_player.py` (`TalkPlayer`, `_signal_group`)
- `src/kiss/agents/vscode/src/voiceWake.ts` (`VoiceWakeService`)
- `src/kiss/agents/vscode/src/SorcarSidebarView.ts` (`voiceToggle` handler)
- `src/kiss/agents/vscode/src/SorcarTab.ts` (`VOICE_MODEL_URL`, `VOICE_CONFIG`)
- `src/kiss/agents/vscode/src/voiceAckPlayer.ts` (`playVoiceAckClip`)
- `src/kiss/agents/vscode/media/voice.js`, `media/vosk.js`

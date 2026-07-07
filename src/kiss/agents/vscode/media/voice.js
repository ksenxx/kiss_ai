// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Voice wake-word support for KISS Sorcar.
 *
 * Always-on, fully local listener for the trigger word "Sorcar".  When
 * the wake word is heard, the mic button flashes RED as a visual cue
 * (no text is ever typed into the task input — the literal word
 * "sorcar" must never appear there) while the extension host records
 * the speech that follows.  When the host starts the gpt-audio
 * transcription/translation call it sends ``{type:
 * 'voiceTranscribing'}`` and the flash turns YELLOW; the resulting
 * ``{type: 'voiceSpeech', text}`` clears the flash and types the
 * translated text into the task input (or appends it to an existing
 * draft), and listening continues.
 *
 * Two modes, selected by the ``window.__VOICE__`` config injected by
 * the page template:
 *
 * - ``browser`` (remote web app): the microphone is captured in the
 *   page itself and the wake word is recognized with vosk-browser, a
 *   WASM build of the lightweight Kaldi/Vosk small English model
 *   running in a Web Worker.  After a wake, the utterance that
 *   follows is captured right here (RMS endpointing that mirrors the
 *   Python listener's SpeechCapture), downsampled to 16kHz s16le PCM
 *   and posted to the server as ``{type: 'voiceTranscribe', audio:
 *   <base64 pcm>}``; the server runs the same gpt-audio translation
 *   as webview mode and replies with ``{type: 'voiceSpeech', text,
 *   speaker}``.  Wake-word detection never leaves the machine.
 *
 * - ``webview`` (VS Code extension): extension webviews cannot use
 *   getUserMedia (VS Code denies microphone access to webview
 *   origins), so the extension host runs the same Vosk small model in
 *   a local Python process and forwards wake events to this page as
 *   ``{type: 'voiceWake'}`` messages.  The toggle button posts
 *   ``{type: 'voiceToggle', enabled}`` back to the host through the
 *   ``kiss-voice-post`` bridge event handled in main.js (which owns
 *   the single allowed ``acquireVsCodeApi()`` instance).
 */
(function () {
  'use strict';

  const cfg = window.__VOICE__ || {mode: 'browser'};
  const btn = document.getElementById('voice-btn');
  const inp = document.getElementById('task-input');
  if (!btn || !inp) return;

  // "sorcar" is not in the small English model's vocabulary, so the
  // grammar also lists in-vocabulary words/phrases that sound like
  // "Sorcar".  Because the grammar forces every sound into an alias
  // or [unk], detection is strict at the default sensitivity: the
  // utterance has to decode to exactly one alias (allowing only a
  // brief leading [unk] — the breathy onset of softly spoken speech,
  // see wakeWithLeadingNoise), every alias word
  // needs a sufficient confidence, and a partial result only fires
  // after a short quiet pause proves the speaker paused instead of
  // talking on ("soccer is my favorite sport").  The settings slider
  // adjusts those gates; high sensitivity also accepts utterances
  // that END with an alias ("[unk] sore car" from "hey there Sorcar")
  // but never an alias followed by more speech/unknown audio.
  // Common standalone words ("soccer", "circa", "so car", "saw car")
  // are deliberately NOT aliases, keeping the grammar to genuinely
  // Sorcar-shaped phrases; real human "Sorcar" decodes to "sir car"/
  // "sore car"/"sar car" (verified live).
  const WAKE_ALIASES = ['sorcar', 'sir car', 'sore car', 'sar car'];
  const COOLDOWN_MS = 2000;
  // Wake-word sensitivity (0..100, settings-panel slider).  The value
  // scales the per-word confidence gate and the post-alias pause
  // linearly, and at >= TRAILING_ALIAS_SENSITIVITY also accepts
  // utterances that merely END with an alias ("hey there Sorcar"
  // decodes to "[unk] sore car", measured live).  Sensitivity 50
  // reproduces the historical gates (conf 0.4, pause 200ms); the
  // default 70 is deliberately more sensitive (conf 0.24, pause
  // 120ms).  Mirrors kiss/agents/vscode/voice_wake.py.
  const DEFAULT_SENSITIVITY = 70;
  const TRAILING_ALIAS_SENSITIVITY = 75;
  const SENSITIVITY_KEY = 'kissVoiceSensitivity';

  /**
   * Per-word confidence floor for a sensitivity: linear 0.8 -> 0.0.
   * Vosk grammar-mode word confidences are posteriors in [0, 1] but
   * do not separate true wakes from sound-alikes (real human "Sorcar"
   * scored 0.53 live; TTS "soccer" force-fit 0.55) — the gate is a
   * sanity net against egregious force-fits that a LOW slider setting
   * raises high enough to reject sound-alikes.
   */
  function sensitivityMinWordConf(s) {
    return 0.8 * (1 - s / 100);
  }

  /**
   * Quiet audio (ms) required after the alias before a partial result
   * may fire the wake word: linear 400ms -> 100ms floor (without any
   * pause continuous speech would fire mid-utterance).
   */
  function sensitivityWakePauseMs(s) {
    return Math.max(100, 400 * (1 - s / 100));
  }

  /** The stored slider value, or the default when absent/garbage. */
  function storedSensitivity() {
    try {
      const v = parseInt(localStorage.getItem(SENSITIVITY_KEY), 10);
      if (isFinite(v) && v >= 0 && v <= 100) return v;
    } catch (_e) {
      /* storage unavailable; use the default */
    }
    return DEFAULT_SENSITIVITY;
  }

  // Live wake-word sensitivity; updated by the settings-panel slider.
  let sensitivity = storedSensitivity();
  // Blocks at or above this normalized RMS count as speech.
  const SPEECH_RMS_THRESHOLD = 0.01;
  // Browser-mode post-wake speech capture (mirrors SpeechCapture in
  // kiss/agents/vscode/voice_wake.py): trailing silence that ends the
  // utterance, silence-only timeout after the wake, hard length cap,
  // and the PCM sample rate the server-side gpt-audio call expects.
  const CAPTURE_END_SILENCE_MS = 2000;
  const CAPTURE_NO_SPEECH_TIMEOUT_MS = 5000;
  const CAPTURE_MAX_MS = 30000;
  const CAPTURE_SAMPLE_RATE = 16000;
  const STORAGE_KEY = 'kissVoiceEnabled';
  const DEBUG_KEY = 'kissVoiceDebug';

  function debugEnabled() {
    try {
      return localStorage.getItem(DEBUG_KEY) === '1';
    } catch (_e) {
      return false;
    }
  }

  function debugLog(kind, text) {
    if (!debugEnabled()) return;
    console.log('[voice] ' + kind + ':', JSON.stringify(text));
  }

  let enabled = false; // user intent: wake-word listening is on
  let busy = false; // async start/stop in progress (browser mode)
  let lastWakeAt = 0;
  // Webview mode: voice rounds (host WAKE events) whose terminal
  // voiceSpeech result has not arrived yet.  Translations run on a
  // background worker in the listener, so a second round can start
  // before the first round's text arrives; a late terminal event must
  // then NOT clear the flash that belongs to the newer round.
  let outstandingRounds = 0;

  // Browser-mode audio pipeline handles.
  let model = null;
  let recognizer = null;
  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let voskLoadPromise = null;

  function setUi(state, message) {
    // state: 'off' | 'loading' | 'listening' | 'error'
    btn.classList.remove(
      'voice-off',
      'voice-loading',
      'voice-listening',
      'voice-error',
      'active',
    );
    btn.classList.add('voice-' + state);
    if (state === 'listening') btn.classList.add('active');
    let tip;
    if (state === 'listening') {
      tip =
        "Voice trigger on: say 'Sorcar' and pause briefly " +
        '(click to turn off)';
    } else if (state === 'loading') {
      tip = 'Voice trigger: starting ...';
    } else if (state === 'error') {
      tip = 'Voice trigger error: ' + (message || 'unavailable');
    } else {
      tip = "Voice trigger: listen for the word 'Sorcar'";
    }
    btn.setAttribute('data-tooltip', tip);
  }

  function normalize(text) {
    return String(text || '')
      .toLowerCase()
      .replace(/\s+/g, ' ')
      .trim();
  }

  /**
   * True when the whole normalized utterance is exactly one wake
   * alias.  Substring matching is deliberately avoided; it fired on
   * everyday sentences that merely contain an alias-sounding word.
   * At high sensitivity an utterance that ENDS with an alias also
   * matches ("hey there Sorcar" -> "[unk] sore car"), but a
   * mid-utterance alias (anything after it) still never does.
   */
  function matchesWake(text) {
    const t = normalize(text);
    if (!t) return false;
    if (WAKE_ALIASES.indexOf(t) !== -1) return true;
    if (sensitivity >= TRAILING_ALIAS_SENSITIVITY) {
      for (let i = 0; i < WAKE_ALIASES.length; i++) {
        const suffix = ' ' + WAKE_ALIASES[i];
        if (t.length > suffix.length && t.endsWith(suffix)) return true;
      }
    }
    return false;
  }

  // Softly spoken "Sorcar" carries a breathy onset that the grammar
  // decodes as a brief leading [unk] before the alias ("[unk] sore
  // car" with a ~60ms [unk], measured with whispered speech); exact
  // whole-utterance matching rejected those wakes, so the wake word
  // seemed to need a loud voice.  Spoken-word prefixes decode to
  // [unk] spans of ~0.5s and up ("hey there" 0.61s, "he wrecked his"
  // 0.60s, measured), so this budget keeps sentences and "hey there
  // Sorcar" rejected.  Mirrors kiss/agents/vscode/voice_wake.py.
  const MAX_LEADING_NOISE_SECONDS = 0.35;

  /**
   * True when `words` (the word list of a Vosk FINAL result, entries
   * carrying word/start/end) is exactly one wake alias preceded only
   * by [unk] noise totaling at most MAX_LEADING_NOISE_SECONDS —
   * softly spoken "Sorcar" whose breathy onset decoded as a short
   * [unk].  Word entries without numeric timings reject; anything
   * after the alias rejects.  Companion to matchesWake; vosk-browser
   * partial results carry no word list, so this gate only sees
   * finals.
   */
  function wakeWithLeadingNoise(words) {
    if (!Array.isArray(words) || words.length === 0) return false;
    let i = 0;
    let noiseSeconds = 0;
    while (i < words.length && words[i] && words[i].word === '[unk]') {
      const start = words[i].start;
      const end = words[i].end;
      if (typeof start !== 'number' || typeof end !== 'number') {
        return false;
      }
      noiseSeconds += Math.max(0, end - start);
      i++;
    }
    if (i === 0 || noiseSeconds > MAX_LEADING_NOISE_SECONDS) return false;
    const tail = [];
    for (; i < words.length; i++) {
      tail.push(words[i] && words[i].word);
    }
    return WAKE_ALIASES.indexOf(tail.join(' ')) !== -1;
  }

  /**
   * True when every recognized word clears the sensitivity-scaled
   * confidence floor.  `words` is the word list of a Vosk result
   * (each entry has a `conf` field when setWords(true) is on).  Only
   * confidences on the [0, 1] posterior scale are gated; larger
   * values and missing word lists pass, so the gate can only tighten
   * detection, never lose a clean wake.
   */
  function wordsConfident(words) {
    if (!Array.isArray(words)) return true;
    const minConf = sensitivityMinWordConf(sensitivity);
    for (let i = 0; i < words.length; i++) {
      const conf = words[i] && words[i].conf;
      if (typeof conf === 'number' && conf <= 1.0 && conf < minConf) {
        return false;
      }
    }
    return true;
  }

  let flashTimer = null;

  /**
   * Show a transient color state on the mic button: 'voice-triggered'
   * (red — wake word heard, capturing speech) or 'voice-transcribing'
   * (yellow — gpt-audio transcription in flight).  Passing a falsy
   * class clears both.  Only one flash (and one safety timer) is
   * active at a time.
   */
  function flash(cls, timeoutMs) {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
    btn.classList.remove('voice-triggered', 'voice-transcribing');
    if (!cls) return;
    btn.classList.add(cls);
    flashTimer = setTimeout(() => {
      flashTimer = null;
      btn.classList.remove(cls);
    }, timeoutMs);
  }

  /**
   * React to the wake word: flash the mic button red and focus the
   * task input so the user sees they were heard.  No text is inserted
   * — the literal word "sorcar" must never appear in the input box;
   * the translated speech arrives later as a voiceSpeech message.
   * Debounced so one long utterance (partial + final results) only
   * triggers once; returns true when the wake actually fired (not
   * debounced) so browser mode knows to start a speech capture.  The
   * red flash persists while the speech that follows is captured —
   * by the extension host in webview mode, by this page in browser
   * mode — until the voiceTranscribing/voiceSpeech message that
   * follows replaces or clears it (the long timeout is only a safety
   * net beyond the 30s capture cap).
   */
  function triggerWake() {
    const now = Date.now();
    if (now - lastWakeAt < COOLDOWN_MS) return false;
    lastWakeAt = now;
    try {
      inp.focus();
    } catch (_e) {
      /* focus can fail in background documents; ignore */
    }
    flash('voice-triggered', 45000);
    return true;
  }

  /**
   * Insert the English translation of the speech that followed the
   * wake word into the task input and submit it to the agent of the
   * highlighted tab.  When *speaker* is a positive integer (the
   * host's voice-recognition speaker number, unique per distinct
   * voice, starting from 1) the text is prefixed with
   * ``Speaker #N says that: `` so the agent knows who spoke.  An
   * empty input receives the text; an existing draft is appended to
   * with a space.  After a non-empty insert a ``kiss-voice-submit``
   * window event asks main.js to send the input exactly like a click
   * on the send button — a fresh task for an idle tab, or a steering
   * user message injected into a running agent.  An empty translation
   * (silence or a failed translation) never touches user text and
   * never submits.  The mic-button flash is cleared unless
   * *keepFlash* is true (a newer voice round is still in flight and
   * owns the indicator).
   */
  function insertSpeech(text, keepFlash, speaker) {
    if (!keepFlash) flash(null);
    let translated = String(typeof text === 'string' ? text : '').trim();
    if (!translated) return;
    if (
      typeof speaker === 'number' &&
      isFinite(speaker) &&
      speaker >= 1 &&
      Math.floor(speaker) === speaker
    ) {
      translated = 'Speaker #' + speaker + ' says that: ' + translated;
    }
    if (!inp.value) {
      inp.value = translated;
    } else {
      inp.value = inp.value + ' ' + translated;
    }
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    try {
      inp.focus();
    } catch (_e) {
      /* focus can fail in background documents; ignore */
    }
    // Hand the spoken task to the agent: main.js owns sendMessage()
    // (submit vs. steering of a running task) and listens for this
    // event next to the 'kiss-voice-post' bridge.
    window.dispatchEvent(new CustomEvent('kiss-voice-submit'));
  }

  // Browser-mode post-wake capture in progress, or null.  Owns every
  // audio block while active (the wake recognizer does not hear
  // capture audio, mirroring the Python listener).
  let capture = null;

  /** Start capturing the utterance that follows a browser-mode wake. */
  function beginCapture() {
    // Count the round at wake time (not when capture finishes), so a
    // late voiceSpeech from an older transcription cannot clear the
    // red flash that belongs to this newer capture.
    outstandingRounds++;
    capture = {
      chunks: [], // Int16Array blocks, already at CAPTURE_SAMPLE_RATE
      sinceWakeMs: 0,
      elapsedMs: 0,
      speechStarted: false,
      trailingSilenceMs: 0,
    };
  }

  /**
   * Convert one Float32 audio block at *sourceRate* into 16-bit PCM
   * at CAPTURE_SAMPLE_RATE using linear interpolation.  Mic capture
   * rates (44.1/48kHz) are at or above the 16kHz target, so simple
   * interpolation without a low-pass filter is adequate for speech.
   */
  function downsampleTo16k(samples, sourceRate) {
    sourceRate = Number(sourceRate);
    if (!Number.isFinite(sourceRate) || sourceRate <= 0) {
      sourceRate = CAPTURE_SAMPLE_RATE;
    }
    const ratio = sourceRate / CAPTURE_SAMPLE_RATE;
    const outLength = Math.floor(samples.length / ratio);
    const out = new Int16Array(outLength);
    for (let i = 0; i < outLength; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, samples.length - 1);
      const frac = pos - i0;
      let v = samples[i0] * (1 - frac) + samples[i1] * frac;
      if (v > 1) v = 1;
      else if (v < -1) v = -1;
      out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
    }
    return out;
  }

  /** Base64-encode captured Int16Array blocks as little-endian PCM. */
  function pcmBase64(chunks) {
    let totalSamples = 0;
    for (let i = 0; i < chunks.length; i++) totalSamples += chunks[i].length;
    const bytes = new Uint8Array(totalSamples * 2);
    let off = 0;
    for (let i = 0; i < chunks.length; i++) {
      const chunk = chunks[i];
      for (let j = 0; j < chunk.length; j++) {
        const s = chunk[j];
        bytes[off++] = s & 0xff;
        bytes[off++] = (s >> 8) & 0xff;
      }
    }
    let binary = '';
    const STRIDE = 0x8000;
    for (let i = 0; i < bytes.length; i += STRIDE) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + STRIDE));
    }
    return window.btoa(binary);
  }

  /**
   * End the active capture.  With speech captured, the PCM is posted
   * to the server for gpt-audio translation ({type: 'voiceTranscribe',
   * audio: <base64 16kHz mono s16le>}) and the mic button turns
   * yellow until the server's voiceSpeech reply arrives; silence just
   * clears the red wake flash (the Python listener's NO_SPEECH).
   */
  function finishCapture() {
    const done = capture;
    capture = null;
    if (!done.speechStarted || !done.chunks.length) {
      outstandingRounds = Math.max(0, outstandingRounds - 1);
      if (outstandingRounds > 0) flash('voice-transcribing', 60000);
      else flash(null);
      return;
    }
    flash('voice-transcribing', 60000);
    postToHost({type: 'voiceTranscribe', audio: pcmBase64(done.chunks)});
  }

  /**
   * Feed one audio block to the active capture — the browser-mode
   * mirror of SpeechCapture.feed in voice_wake.py.  Leading silence
   * is skipped (until CAPTURE_NO_SPEECH_TIMEOUT_MS gives up), speech
   * starts on the first loud block, and the capture ends after
   * CAPTURE_END_SILENCE_MS of trailing silence or CAPTURE_MAX_MS of
   * audio.
   */
  function feedCapture(samples, rms, blockMs, sourceRate) {
    const loud = rms >= SPEECH_RMS_THRESHOLD;
    capture.sinceWakeMs += blockMs;
    if (!capture.speechStarted) {
      if (!loud) {
        if (capture.sinceWakeMs >= CAPTURE_NO_SPEECH_TIMEOUT_MS) {
          finishCapture();
        }
        return;
      }
      capture.speechStarted = true;
    }
    capture.chunks.push(downsampleTo16k(samples, sourceRate));
    capture.elapsedMs += blockMs;
    capture.trailingSilenceMs = loud ? 0 : capture.trailingSilenceMs + blockMs;
    if (
      capture.trailingSilenceMs >= CAPTURE_END_SILENCE_MS ||
      capture.elapsedMs >= CAPTURE_MAX_MS
    ) {
      finishCapture();
    }
  }

  function loadVosk() {
    if (window.Vosk) return Promise.resolve();
    if (voskLoadPromise) return voskLoadPromise;
    voskLoadPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = cfg.voskSrc;
      s.onload = () => {
        resolve();
      };
      s.onerror = () => {
        voskLoadPromise = null;
        reject(new Error('failed to load speech engine'));
      };
      document.head.appendChild(s);
    });
    return voskLoadPromise;
  }

  function stopBrowserPipeline() {
    capture = null;
    if (processorNode) {
      try {
        processorNode.disconnect();
      } catch (_e) {
        /* already disconnected */
      }
      processorNode.onaudioprocess = null;
      processorNode = null;
    }
    if (sourceNode) {
      try {
        sourceNode.disconnect();
      } catch (_e) {
        /* already disconnected */
      }
      sourceNode = null;
    }
    if (mediaStream) {
      const tracks = mediaStream.getTracks();
      for (let i = 0; i < tracks.length; i++) tracks[i].stop();
      mediaStream = null;
    }
    if (audioContext) {
      try {
        audioContext.close();
      } catch (_e) {
        /* already closed */
      }
      audioContext = null;
    }
    if (recognizer) {
      try {
        recognizer.remove();
      } catch (_e) {
        /* worker already gone */
      }
      recognizer = null;
    }
  }

  function startBrowserPipeline() {
    busy = true;
    setUi('loading');
    return loadVosk()
      .then(() => {
        if (model) return model;
        return window.Vosk.createModel(cfg.modelUrl).then(m => {
          model = m;
          return m;
        });
      })
      .then(() => {
        return navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            channelCount: 1,
          },
        });
      })
      .then(stream => {
        if (!enabled) {
          // User toggled off while we were starting.
          const tracks = stream.getTracks();
          for (let i = 0; i < tracks.length; i++) tracks[i].stop();
          return;
        }
        mediaStream = stream;
        if (debugEnabled()) {
          const track = stream.getAudioTracks()[0];
          if (track) {
            debugLog(
              'track',
              track.label + ' ' + JSON.stringify(track.getSettings()),
            );
          }
          navigator.mediaDevices
            .enumerateDevices()
            .then(devs => {
              for (let i = 0; i < devs.length; i++) {
                if (devs[i].kind === 'audioinput') {
                  debugLog(
                    'mic',
                    devs[i].label + ' [' + devs[i].deviceId + ']',
                  );
                }
              }
            })
            .catch(() => {});
        }
        const Ctx = window.AudioContext || window.webkitAudioContext;
        audioContext = new Ctx();
        if (audioContext.state === 'suspended') {
          // Autoplay policy can start the context suspended; the click
          // that enabled voice counts as the required user gesture.
          audioContext.resume().catch(() => {});
        }
        const grammar = JSON.stringify(WAKE_ALIASES.concat(['[unk]']));
        recognizer = new model.KaldiRecognizer(
          audioContext.sampleRate,
          grammar,
        );
        if (typeof recognizer.setWords === 'function') {
          // Word-level confidences in final results (partial results
          // carry no confidences in vosk-browser).
          recognizer.setWords(true);
        }
        // Milliseconds of quiet audio since the last speech block;
        // updated by onaudioprocess, read by the partial-result gate.
        let quietMs = 0;
        // True while the 'result' event carrying the flushed wake
        // utterance (the retrieveFinalResult() call in fireWake) is
        // still in flight; that result's text is the wake word itself
        // and must never re-trigger.
        let awaitingFlush = false;
        // React to a recognized wake word: start the post-wake capture
        // and flush/reset the recognizer.  Without the flush the
        // decoded "sorcar" utterance stays pending inside Vosk while
        // the capture owns the audio (the recognizer hears nothing),
        // and the first silence endpoint AFTER the capture finalizes
        // it as a stale wake-word result — re-arming a capture and
        // transcribing speech nobody prefixed with "Sorcar".  Mirrors
        // the Python listener's Reset() after a wake;
        // retrieveFinalResult() is the only reset vosk-browser
        // exposes, and it delivers the flushed utterance as one
        // 'result' event that awaitingFlush consumes.
        function fireWake() {
          if (!triggerWake()) return;
          quietMs = 0;
          beginCapture();
          if (
            recognizer &&
            typeof recognizer.retrieveFinalResult === 'function'
          ) {
            awaitingFlush = true;
            try {
              recognizer.retrieveFinalResult();
            } catch (_e) {
              awaitingFlush = false;
            }
          }
        }
        recognizer.on('result', message => {
          if (message && message.result) {
            debugLog('result', message.result.text);
            if (awaitingFlush) {
              // The flushed final of the utterance that fired the
              // wake; its text is the wake word and must not
              // re-trigger.
              awaitingFlush = false;
              return;
            }
            // A capture owns the current voice round; any result
            // arriving now decodes pre-wake audio and must not wipe
            // the capture by re-waking.
            if (capture) return;
            // A final result means Vosk saw the endpoint: the whole
            // utterance is over, so no pause gate is needed.  Softly
            // spoken "Sorcar" decodes with a brief leading [unk]
            // (breathy onset), which wakeWithLeadingNoise accepts.
            if (
              (matchesWake(message.result.text) ||
                wakeWithLeadingNoise(message.result.result)) &&
              wordsConfident(message.result.result)
            ) {
              fireWake();
            }
          }
        });
        recognizer.on('partialresult', message => {
          if (message && message.result) {
            if (message.result.partial) {
              debugLog('partial', message.result.partial);
            }
            // Stale partials decoded from pre-wake audio must not
            // re-wake while the capture owns the round.
            if (capture) return;
            // Fire only when the speaker paused right after the
            // alias; continuous speech keeps quietMs at 0.
            if (
              quietMs >= sensitivityWakePauseMs(sensitivity) &&
              matchesWake(message.result.partial)
            ) {
              fireWake();
            }
          }
        });
        // Debug-only free-decode recognizer (no grammar): logs what the
        // model hears without grammar constraints so wake aliases can
        // be tuned against real human speech.
        let freeRecognizer = null;
        if (debugEnabled()) {
          freeRecognizer = new model.KaldiRecognizer(audioContext.sampleRate);
          freeRecognizer.on('result', message => {
            if (message && message.result) {
              debugLog('free-result', message.result.text);
            }
          });
          freeRecognizer.on('partialresult', message => {
            if (message && message.result && message.result.partial) {
              debugLog('free-partial', message.result.partial);
            }
          });
        }
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        let lastRmsAt = 0;
        processorNode.onaudioprocess = event => {
          if (!recognizer) return;
          const samples = event.inputBuffer.getChannelData(0);
          let sumSquares = 0;
          for (let i = 0; i < samples.length; i++) {
            sumSquares += samples[i] * samples[i];
          }
          const rms = Math.sqrt(sumSquares / samples.length);
          const blockMs =
            (samples.length / event.inputBuffer.sampleRate) * 1000;
          quietMs = rms >= SPEECH_RMS_THRESHOLD ? 0 : quietMs + blockMs;
          if (debugEnabled()) {
            const now = Date.now();
            if (now - lastRmsAt > 2000) {
              lastRmsAt = now;
              debugLog('rms', rms.toFixed(5));
            }
          }
          if (capture) {
            // A post-wake capture owns the audio; the wake recognizer
            // does not hear it (mirrors the Python listener).
            feedCapture(samples, rms, blockMs, event.inputBuffer.sampleRate);
            return;
          }
          try {
            recognizer.acceptWaveform(event.inputBuffer);
            if (freeRecognizer)
              freeRecognizer.acceptWaveform(event.inputBuffer);
          } catch (_e) {
            /* recognizer torn down mid-flight; ignore */
          }
        };
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        setUi('listening');
      })
      .catch(err => {
        enabled = false;
        persist();
        stopBrowserPipeline();
        setUi('error', err && err.message);
      })
      .then(() => {
        busy = false;
        if (!enabled && (mediaStream || audioContext)) {
          // Turned off during a successful start: tear down now.
          stopBrowserPipeline();
          setUi('off');
        } else if (enabled && !processorNode) {
          // Re-enabled while a cancelled start was unwinding (the
          // off-toggle made this chain drop its stream): start over so
          // the UI cannot get stuck in 'loading'.
          startBrowserPipeline();
        }
      });
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');
    } catch (_e) {
      /* storage unavailable (private mode); voice still works */
    }
  }

  /**
   * Whether wake-word listening should be on when the page loads.
   *
   * VS Code webview mode: the mic defaults ON when VS Code launches —
   * only an explicit user opt-out (stored '0', written by clicking the
   * mic button off or by a listener error) keeps it off.  Browser mode
   * (remote web app) never auto-starts: getUserMedia without a user
   * gesture would fail or prompt unexpectedly, so it only restores an
   * explicit opt-in (stored '1').
   */
  function wasEnabled() {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (cfg.mode === 'webview') return stored !== '0';
      return stored === '1';
    } catch (_e) {
      return cfg.mode === 'webview';
    }
  }

  function postToHost(message) {
    // main.js owns the single acquireVsCodeApi() handle; it forwards
    // 'kiss-voice-post' events to the extension host.
    window.dispatchEvent(new CustomEvent('kiss-voice-post', {detail: message}));
  }

  // Settings-panel sensitivity slider (#cfg-voice-sensitivity).  The
  // slider reflects the stored value on load and applies changes
  // LIVE: browser mode reads the `sensitivity` variable on every
  // recognizer result, and webview mode reports the value to the
  // extension host, which restarts the Python listener with
  // --sensitivity.  Pages without the settings panel (no slider in
  // the DOM) keep working with the stored/default value.
  const sensSlider = document.getElementById('cfg-voice-sensitivity');
  const sensValue = document.getElementById('cfg-voice-sensitivity-value');

  function renderSensitivity() {
    if (sensSlider) sensSlider.value = String(sensitivity);
    if (sensValue) sensValue.textContent = String(sensitivity);
  }

  renderSensitivity();
  if (sensSlider) {
    sensSlider.addEventListener('input', () => {
      const v = parseInt(sensSlider.value, 10);
      if (!isFinite(v)) return;
      sensitivity = Math.min(100, Math.max(0, v));
      try {
        localStorage.setItem(SENSITIVITY_KEY, String(sensitivity));
      } catch (_e) {
        /* storage unavailable; the live value still applies */
      }
      renderSensitivity();
      if (cfg.mode === 'webview') {
        postToHost({type: 'voiceSensitivity', value: sensitivity});
      }
    });
  }

  function setEnabled(next) {
    if (enabled === next) return;
    enabled = next;
    persist();
    // Turning listening off ends any in-flight voice round-trip; a
    // stale red/yellow flash must not wait for a host message that
    // may never come.
    if (!next) {
      outstandingRounds = 0;
      capture = null;
      flash(null);
    }
    if (cfg.mode === 'webview') {
      setUi(next ? 'loading' : 'off');
      // The sensitivity rides along so the host can pass
      // --sensitivity to the Python listener it starts.
      postToHost({type: 'voiceToggle', enabled: next, sensitivity});
      return;
    }
    if (next) {
      // When a start/stop is already in flight, the running chain's
      // finalizer notices ``enabled === true`` and restarts itself —
      // launching a second chain here would leak a mic pipeline.
      if (!busy) startBrowserPipeline();
    } else if (!busy) {
      stopBrowserPipeline();
      setUi('off');
    }
    // When busy, the start chain notices ``enabled === false`` and
    // tears the pipeline down itself.
  }

  btn.addEventListener('click', () => {
    setEnabled(!enabled);
  });

  window.addEventListener('message', event => {
    const msg = event && event.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'voiceWake') {
      // Count every host WAKE (even one debounced visually): the
      // listener emits exactly one terminal voiceSpeech per WAKE.
      outstandingRounds++;
      triggerWake();
    } else if (msg.type === 'voiceTranscribing') {
      // Capture ended; the gpt-audio call is in flight — flash yellow
      // until the voiceSpeech result clears it (60s safety timeout).
      flash('voice-transcribing', 60000);
    } else if (msg.type === 'voiceSpeech') {
      // Terminal results arrive in spoken (FIFO) order; keep the
      // flash when a newer round is still capturing or transcribing.
      outstandingRounds = Math.max(0, outstandingRounds - 1);
      insertSpeech(msg.text, outstandingRounds > 0, msg.speaker);
    } else if (msg.type === 'voiceState') {
      // Extension host reports the real listener state.  A listener
      // that stopped (error or off) can no longer deliver the
      // voiceTranscribing/voiceSpeech that would clear an in-flight
      // flash, so clear it here.
      if (msg.error) {
        outstandingRounds = 0;
        flash(null);
        enabled = false;
        persist();
        setUi('error', msg.error);
      } else if (msg.listening) {
        setUi('listening');
      } else {
        outstandingRounds = 0;
        flash(null);
        if (!enabled) setUi('off');
      }
    }
  });

  setUi('off');
  // Turn the mic on at load: in webview mode this runs when VS Code
  // launches (with the Sorcar view open), so listening starts without
  // a click unless the user explicitly opted out.
  if (wasEnabled()) {
    setEnabled(true);
  }
})();

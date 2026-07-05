// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Voice wake-word support for KISS Sorcar.
 *
 * Always-on, fully local listener for the trigger word "Sorcar".  When
 * the wake word is heard, the mic button flashes GREEN as a visual cue
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
 *   page itself and speech is recognized with vosk-browser, a WASM
 *   build of the lightweight Kaldi/Vosk small English model running in
 *   a Web Worker.  Nothing ever leaves the machine.
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
  // or [unk], detection must be strict: the WHOLE utterance has to
  // decode to exactly one alias (never an alias embedded in [unk]
  // context, e.g. "[unk] sir car [unk]" from "yes sir the car is
  // ready"), every alias word needs a high confidence, and a partial
  // result only fires after ~200ms of quiet audio proves the speaker
  // paused instead of talking on ("soccer is my favorite sport").
  // Common standalone words ("soccer", "circa", "so car", "saw car")
  // are deliberately NOT aliases, keeping the grammar to genuinely
  // Sorcar-shaped phrases; real human "Sorcar" decodes to "sir car"/
  // "sore car"/"sar car" (verified live).
  const WAKE_ALIASES = ['sorcar', 'sir car', 'sore car', 'sar car'];
  const COOLDOWN_MS = 2000;
  // Vosk grammar-mode word confidences are posteriors in [0, 1] but
  // do not separate true wakes from sound-alikes (real human "Sorcar"
  // scored 0.53 live; TTS "soccer" force-fit 0.55) — this is only a
  // sanity net against egregious force-fits.  Values above 1.0 would
  // be raw acoustic likelihoods on another scale and are not gated.
  const MIN_WORD_CONF = 0.4;
  // Quiet audio required after the alias before a partial result may
  // fire the wake word.
  const WAKE_PAUSE_MS = 200;
  // Blocks at or above this normalized RMS count as speech.
  const SPEECH_RMS_THRESHOLD = 0.01;
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
   */
  function matchesWake(text) {
    const t = normalize(text);
    if (!t) return false;
    return WAKE_ALIASES.indexOf(t) !== -1;
  }

  /**
   * True when every recognized word clears MIN_WORD_CONF.  `words` is
   * the word list of a Vosk result (each entry has a `conf` field
   * when setWords(true) is on).  Only confidences on the [0, 1]
   * posterior scale are gated; larger values and missing word lists
   * pass, so the gate can only tighten detection, never lose a clean
   * wake.
   */
  function wordsConfident(words) {
    if (!Array.isArray(words)) return true;
    for (let i = 0; i < words.length; i++) {
      const conf = words[i] && words[i].conf;
      if (typeof conf === 'number' && conf <= 1.0 && conf < MIN_WORD_CONF) {
        return false;
      }
    }
    return true;
  }

  let flashTimer = null;

  /**
   * Show a transient color state on the mic button: 'voice-triggered'
   * (green — wake word heard, capturing speech) or 'voice-transcribing'
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
   * React to the wake word: flash the mic button green and focus the
   * task input so the user sees they were heard.  No text is inserted
   * — the literal word "sorcar" must never appear in the input box;
   * the translated speech arrives later as a voiceSpeech message.
   * Debounced so one long utterance (partial + final results) only
   * triggers once.  In webview mode the green flash persists while
   * the host captures the speech (the voiceTranscribing/voiceSpeech
   * message that follows replaces or clears it; the long timeout is
   * only a safety net beyond the listener's 30s capture cap).
   */
  function triggerWake() {
    const now = Date.now();
    if (now - lastWakeAt < COOLDOWN_MS) return;
    lastWakeAt = now;
    try {
      inp.focus();
    } catch (_e) {
      /* focus can fail in background documents; ignore */
    }
    flash('voice-triggered', cfg.mode === 'webview' ? 45000 : 600);
  }

  /**
   * Insert the English translation of the speech that followed the
   * wake word into the task input.  An empty input receives the text;
   * an existing draft is appended to with a space.  An empty
   * translation (silence or a failed translation) never touches user
   * text.  The mic-button flash is cleared unless *keepFlash* is true
   * (a newer voice round is still in flight and owns the indicator).
   */
  function insertSpeech(text, keepFlash) {
    if (!keepFlash) flash(null);
    const translated = String(typeof text === 'string' ? text : '').trim();
    if (!translated) return;
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
        recognizer.on('result', message => {
          if (message && message.result) {
            debugLog('result', message.result.text);
            // A final result means Vosk saw the endpoint: the whole
            // utterance is over, so no pause gate is needed.
            if (
              matchesWake(message.result.text) &&
              wordsConfident(message.result.result)
            ) {
              triggerWake();
            }
          }
        });
        recognizer.on('partialresult', message => {
          if (message && message.result) {
            if (message.result.partial) {
              debugLog('partial', message.result.partial);
            }
            // Fire only when the speaker paused right after the
            // alias; continuous speech keeps quietMs at 0.
            if (
              quietMs >= WAKE_PAUSE_MS &&
              matchesWake(message.result.partial)
            ) {
              triggerWake();
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

  function wasEnabled() {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch (_e) {
      return false;
    }
  }

  function postToHost(message) {
    // main.js owns the single acquireVsCodeApi() handle; it forwards
    // 'kiss-voice-post' events to the extension host.
    window.dispatchEvent(new CustomEvent('kiss-voice-post', {detail: message}));
  }

  function setEnabled(next) {
    if (enabled === next) return;
    enabled = next;
    persist();
    // Turning listening off ends any in-flight voice round-trip; a
    // stale green/yellow flash must not wait for a host message that
    // may never come.
    if (!next) {
      outstandingRounds = 0;
      flash(null);
    }
    if (cfg.mode === 'webview') {
      setUi(next ? 'loading' : 'off');
      postToHost({type: 'voiceToggle', enabled: next});
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
      insertSpeech(msg.text, outstandingRounds > 0);
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
  if (wasEnabled()) {
    setEnabled(true);
  }
})();

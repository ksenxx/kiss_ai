// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Voice wake-word support for KISS Sorcar.
 *
 * Always-on, fully local listener for the trigger word "Sorcar".  When
 * the wake word is heard, the "sorcar" placeholder is typed into the
 * task input textbox while the extension host records the speech that
 * follows, translates it to English with the gpt-audio model, and sends
 * it back as ``{type: 'voiceSpeech', text}``; the translated text then
 * replaces the placeholder (or is appended to an existing draft) and
 * listening continues.
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

  const WAKE_WORD = 'sorcar';
  // "sorcar" is not in the small English model's vocabulary, so the
  // grammar also lists in-vocabulary words/phrases that sound like
  // "Sorcar".  Any of them heard in a partial or final result counts
  // as the wake word.
  const WAKE_ALIASES = [
    'sorcar',
    'soccer',
    'circa',
    'sir car',
    'sore car',
    'so car',
    'saw car',
    'sar car',
  ];
  const COOLDOWN_MS = 2000;
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
      tip = "Voice trigger on: say 'Sorcar' (click to turn off)";
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

  function matchesWake(text) {
    const t = normalize(text);
    if (!t) return false;
    for (let i = 0; i < WAKE_ALIASES.length; i++) {
      if (t.indexOf(WAKE_ALIASES[i]) !== -1) return true;
    }
    return false;
  }

  /**
   * Type the wake word into the task input and keep listening.
   * Debounced so one long utterance (partial + final results) only
   * triggers once.
   */
  function triggerWake() {
    const now = Date.now();
    if (now - lastWakeAt < COOLDOWN_MS) return;
    lastWakeAt = now;
    inp.value = WAKE_WORD;
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    try {
      inp.focus();
    } catch (_e) {
      /* focus can fail in background documents; ignore */
    }
    btn.classList.add('voice-triggered');
    setTimeout(() => {
      btn.classList.remove('voice-triggered');
    }, 600);
  }

  /**
   * Insert the English translation of the speech that followed the
   * wake word into the task input.  The "sorcar" wake placeholder is
   * replaced; any other existing draft is appended to.  An empty
   * translation (silence or a failed translation) only clears the
   * placeholder and never touches user text.
   */
  function insertSpeech(text) {
    const translated = String(typeof text === 'string' ? text : '').trim();
    const hasPlaceholder = inp.value === WAKE_WORD;
    if (!translated) {
      if (!hasPlaceholder) return;
      inp.value = '';
    } else if (hasPlaceholder || !inp.value) {
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
        recognizer.on('result', message => {
          if (message && message.result) {
            debugLog('result', message.result.text);
            if (matchesWake(message.result.text)) triggerWake();
          }
        });
        recognizer.on('partialresult', message => {
          if (message && message.result) {
            if (message.result.partial) {
              debugLog('partial', message.result.partial);
            }
            if (matchesWake(message.result.partial)) triggerWake();
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
          if (debugEnabled()) {
            const now = Date.now();
            if (now - lastRmsAt > 2000) {
              lastRmsAt = now;
              const d = event.inputBuffer.getChannelData(0);
              let sum = 0;
              for (let i = 0; i < d.length; i++) sum += d[i] * d[i];
              debugLog('rms', Math.sqrt(sum / d.length).toFixed(5));
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
      triggerWake();
    } else if (msg.type === 'voiceSpeech') {
      insertSpeech(msg.text);
    } else if (msg.type === 'voiceState') {
      // Extension host reports the real listener state.
      if (msg.error) {
        enabled = false;
        persist();
        setUi('error', msg.error);
      } else if (msg.listening) {
        setUi('listening');
      } else if (!enabled) {
        setUi('off');
      }
    }
  });

  setUi('off');
  if (wasEnabled()) {
    setEnabled(true);
  }
})();

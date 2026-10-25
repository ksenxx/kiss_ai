// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
(function () {
  'use strict';

  // The page's voice config (web_server.py): the Vosk bundle and model
  // URLs for the in-page wake-word pipeline and the ack sound.
  const cfg = window.__VOICE__ || {};
  const btn = document.getElementById('voice-btn');
  const inp = document.getElementById('task-input');
  if (!btn || !inp) return;

  // The wake word (mirrors WAKE_WORD in kiss/server/speech.py).  It is
  // in the Vosk model's vocabulary, long and rare in conversation, so
  // the grammar cannot force-fit everyday speech ("hey soccer", "sorry
  // car", ...) onto it the way the earlier "Hey Sorcar" allowed.
  const WAKE_ALIASES = ['abracadabra'];
  const COOLDOWN_MS = 2000;
  const DEFAULT_SENSITIVITY = 80;
  const TRAILING_ALIAS_SENSITIVITY = 75;
  const SENSITIVITY_KEY = 'kissVoiceSensitivity';
  const AUTO_SUBMIT_KEY = 'kissVoiceAutoSubmit';

  function sensitivityMinWordConf(s) {
    return 0.8 * (1 - s / 100);
  }

  function storedSensitivity() {
    try {
      const v = parseInt(localStorage.getItem(SENSITIVITY_KEY), 10);
      if (isFinite(v) && v >= 0 && v <= 100) return v;
    } catch (_e) {}
    return DEFAULT_SENSITIVITY;
  }

  function storedAutoSubmit() {
    try {
      return localStorage.getItem(AUTO_SUBMIT_KEY) !== 'off';
    } catch (_e) {
      return true;
    }
  }

  let sensitivity = storedSensitivity();
  let autoSubmit = storedAutoSubmit();
  const SPEECH_RMS_THRESHOLD = 0.01;
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

  let enabled = false;
  let busy = false;
  let lastWakeAt = 0;

  // Generation stamp for the async voice lifecycle. Every mic toggle
  // bumps it; pipeline-start continuations compare the value they
  // captured and stand down when superseded, so an ON→OFF→ON burst can
  // never start two capture paths or resurrect a cancelled one.
  let voiceGen = 0;

  // tableak-coverage:start
  // The conversation that was on screen when each outstanding round began,
  // oldest first. A transcript arrives seconds after the words were spoken
  // and the user may have moved to a different task in the meantime, so
  // submitting it against whatever tab is visible on arrival would put one
  // task's words into another task's conversation. The page captures,
  // transcribes and answers its rounds in order, so the owners queue.
  const roundOwners = [];

  // Owner recorded for a round that began in a page with no conversations at
  // all (a page without the tab machinery). Such a round has nothing to leak
  // into, so it is not the same thing as a CANCELLED round, which must still
  // fail closed.
  const UNSCOPED = {unscoped: true};

  // Owner handed back for a round that was cancelled while its audio was
  // still being transcribed: its late transcript is refused.
  const CANCELLED = {cancelled: true};

  // Rounds cancelled while their transcript may still be on its way. They
  // are only counted: a late transcript that no live round claims is
  // charged to them and fails closed.
  let cancelledRounds = 0;

  /**
   * The conversation on screen, as `{tabId, taskId}`, or null when the
   * page has not published one.
   */
  function currentOwner() {
    const read = window.kissVoiceOwner;
    if (typeof read !== 'function') return null;
    const owner = read();
    return owner && owner.tabId ? owner : null;
  }

  /** Record the owner of an utterance that is starting now. */
  function markSpeechStart() {
    roundOwners.push(currentOwner() || UNSCOPED);
  }

  /**
   * Cancel every outstanding round: the mic was turned off, or the
   * transcribe-flash window ran out.  A cancelled round's transcript may
   * still be on its way, so the rounds are counted: that is what makes
   * the late words fail closed.
   */
  function cancelSpeechRounds() {
    cancelledRounds += roundOwners.length;
    roundOwners.length = 0;
  }

  /**
   * Close the oldest round and hand back the owner it was started with:
   * the live rounds in order, then CANCELLED while the cancelled credit
   * lasts.  A live round always wins: the page cannot tell a cancelled
   * round whose reply is still coming from one whose reply was lost (a
   * socket that dropped mid-transcription), and charging the credit
   * first would let a lost reply eat the next valid dictation for good.
   * With nothing outstanding at all the result is undefined: this
   * transcript belongs to no round this page ever saw, so it was never
   * tied to a moment in time and there is no tab switch to detect.
   */
  function retireRound() {
    if (roundOwners.length) return roundOwners.shift();
    if (cancelledRounds > 0) {
      cancelledRounds--;
      return CANCELLED;
    }
    return undefined;
  }

  /**
   * True when an utterance recorded against `owner` may be typed into the
   * conversation now on screen.
   *
   * Fails CLOSED whenever ownership cannot be proved: a round that was
   * CANCELLED in flight, a page that has stopped publishing an owner, or a
   * round whose owner is not the conversation now on screen. Two different
   * tabs showing the SAME task are one conversation, which is the single
   * exemption -- it matches isForActiveTab() in main.js.
   *
   * Two entries are not failures. `undefined` means no round was ever
   * recorded for this transcript (see retireRound). UNSCOPED means the
   * round began in a page that has no conversations at all. Neither has
   * another conversation to leak into.
   */
  function ownerIsOnScreen(owner) {
    if (owner === undefined || owner === UNSCOPED) return true;
    if (owner === CANCELLED) return false;
    const now = currentOwner();
    if (!now) return false;
    if (owner.tabId === now.tabId) return true;
    return !!owner.taskId && owner.taskId === now.taskId;
  }

  /** The tab an utterance was recorded against, or '' when it had none. */
  function ownerTabId(owner) {
    return owner && owner.tabId ? owner.tabId : '';
  }
  // tableak-coverage:end

  /**
   * Retire the NEWEST round — the one begun most recently.
   *
   * Used when a just-begun capture is abandoned before its audio was ever
   * posted (silence timeout, or the pipeline stopping mid-capture): no
   * transcript will ever answer that round, and the round to close is the
   * one beginCapture just recorded — the LAST entry, not the oldest,
   * which may still be awaiting its own words. When the rounds were
   * reset in the meantime there is nothing left to retire.
   */
  function retireNewestRound() {
    if (!roundOwners.length) return;
    roundOwners.pop();
  }

  let model = null;
  let recognizer = null;
  // The extra grammar-free recognizer run alongside `recognizer` when
  // kissVoiceDebug=1, so stopBrowserPipeline can free it: each one holds a
  // worker-side Kaldi recognizer and a model.recognizers entry for the
  // model's page-long lifetime.
  let freeRecognizer = null;
  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let voskLoadPromise = null;

  let lastUiState = 'off';
  let lastUiTip = "Voice trigger: listen for the word 'Abracadabra'";
  let lastFlashCls = null;

  function applyUiClasses() {
    btn.classList.remove(
      'voice-off',
      'voice-loading',
      'voice-listening',
      'voice-error',
      'active',
    );
    btn.classList.add('voice-' + lastUiState);
    if (lastUiState === 'listening') btn.classList.add('active');
    // The toggle's on/off state for assistive tech, not just a CSS class.
    btn.setAttribute(
      'aria-pressed',
      lastUiState === 'listening' ? 'true' : 'false',
    );
    btn.setAttribute('data-tooltip', lastUiTip);
  }

  function applyFlashClasses() {
    btn.classList.remove('voice-triggered', 'voice-transcribing');
    if (lastFlashCls) btn.classList.add(lastFlashCls);
  }

  // The one toast id every voice failure reuses, so a repeated failure
  // updates the visible notification instead of stacking a second one,
  // and a later successful start can close it.
  const VOICE_NOTIFICATION_ID = 'voice-trigger-error';

  /**
   * Show (or close) the voice trigger's visible notification.
   *
   * voice.js runs in its own closure and cannot call main.js's
   * showNotification directly, so it hands main.js the same
   * `{type: 'notification'}` message the daemon would send: main.js's
   * window 'message' handler routes an untagged notification to the
   * window-level toast container.  Errors are sticky: the user closes
   * them once they have read the remedy.
   */
  function notifyUser(severity, message) {
    const data = message
      ? {
          type: 'notification',
          id: VOICE_NOTIFICATION_ID,
          severity: severity,
          sticky: true,
          message: message,
        }
      : {type: 'notification', id: VOICE_NOTIFICATION_ID, close: true};
    window.dispatchEvent(new window.MessageEvent('message', {data: data}));
  }

  /**
   * Translate a capture failure into plain language with a remedy.
   *
   * `err` is the getUserMedia / AudioContext / model-load rejection (an
   * Error with `name`, or a string from the server).
   */
  function describeVoiceError(err) {
    const name = (err && err.name) || '';
    const text = String((err && err.message) || err || '').trim();
    if (
      name === 'NotAllowedError' ||
      name === 'SecurityError' ||
      /denied|not allowed|permission/i.test(text)
    ) {
      return (
        'Microphone access was denied. Allow the microphone for this ' +
        'site in the browser settings, then turn the voice trigger on again.'
      );
    }
    if (name === 'NotFoundError' || /not found|no device/i.test(text)) {
      return (
        'No microphone was found. Connect a microphone, then turn the ' +
        'voice trigger on again.'
      );
    }
    if (name === 'NotReadableError' || /in use|could not start/i.test(text)) {
      return (
        'The microphone could not be started; another application may ' +
        'be using it. Close that application, then turn the voice trigger ' +
        'on again.'
      );
    }
    return (
      'The voice trigger could not start' +
      (text ? ' (' + text + ')' : '') +
      '. Check the microphone, then turn the voice trigger on again.'
    );
  }

  function setUi(state, message) {
    let tip;
    if (state === 'listening') {
      tip =
        "Voice trigger on: say 'Abracadabra' and pause " +
        'briefly (click to turn off)';
      notifyUser(null, null);
    } else if (state === 'loading') {
      tip = 'Voice trigger: starting ...';
    } else if (state === 'error') {
      const plain = describeVoiceError(message);
      tip = 'Voice trigger error: ' + plain;
      notifyUser('error', plain);
    } else {
      tip = "Voice trigger: listen for the word 'Abracadabra'";
    }
    lastUiState = state;
    lastUiTip = tip;
    applyUiClasses();
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
    if (WAKE_ALIASES.indexOf(t) !== -1) return true;
    if (sensitivity >= TRAILING_ALIAS_SENSITIVITY) {
      for (let i = 0; i < WAKE_ALIASES.length; i++) {
        const suffix = ' ' + WAKE_ALIASES[i];
        if (t.length > suffix.length && t.endsWith(suffix)) return true;
      }
    }
    return false;
  }

  const MAX_LEADING_NOISE_SECONDS = 0.35;

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

  function showListening(on) {
    const overlay = document.getElementById('listening-overlay');
    const wrap = overlay && overlay.parentElement;
    if (!wrap) return;
    if (on) wrap.classList.add('listening');
    else wrap.classList.remove('listening');
  }

  /**
   * Show "transcribing" for up to 60 s while a posted round still
   * awaits its transcript (that timer retires the round when none
   * comes), else clear the flash; a wake's overlay comes down either
   * way.
   */
  function flashPendingRounds() {
    if (roundOwners.length > 0) flash('voice-transcribing', 60000);
    else flash(null);
  }

  function flash(cls, timeoutMs) {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
    lastFlashCls = cls || null;
    applyFlashClasses();
    showListening(cls === 'voice-triggered');
    if (!cls) return;
    flashTimer = setTimeout(() => {
      flashTimer = null;
      lastFlashCls = null;
      applyFlashClasses();
      showListening(false);
      cancelSpeechRounds();
    }, timeoutMs);
  }

  function triggerWake() {
    const now = Date.now();
    if (now - lastWakeAt < COOLDOWN_MS) return false;
    lastWakeAt = now;
    try {
      inp.focus();
    } catch (_e) {}
    flash('voice-triggered', 45000);
    return true;
  }

  // True while the conversation on screen is blocked in ask_user_question:
  // main.js flags the body and routes the composer text to the answer.
  function answeringQuestion() {
    return document.body.classList.contains('ask-answering');
  }

  // owner is this round's conversation, already retired by the caller. It is
  // taken as a parameter rather than read here so that the round is closed
  // before anything can return early: an empty transcript still completes a
  // round, and leaving its owner in the queue would pair the NEXT transcript
  // with the wrong utterance.
  function insertSpeech(text, keepFlash, speaker, language, owner) {
    if (!keepFlash) flash(null);
    const spoken = String(typeof text === 'string' ? text : '').trim();
    if (!spoken) return;
    let translated = spoken;
    if (
      typeof speaker === 'number' &&
      isFinite(speaker) &&
      speaker >= 1 &&
      Math.floor(speaker) === speaker
    ) {
      const lang = typeof language === 'string' ? language.trim() : '';
      translated = lang
        ? 'Speaker #' +
          speaker +
          ' says in the language ' +
          lang +
          ' that: ' +
          spoken
        : 'Speaker #' + speaker + ' says that: ' + spoken;
    }
    // tableak-coverage:start
    // The user switched tasks while speaking: the words belong to the
    // conversation that was on screen when the utterance began, and that
    // tab's input is no longer the one in the DOM. Drop the transcript
    // instead of typing it into a stranger's conversation.
    if (!ownerIsOnScreen(owner)) return;
    // tableak-coverage:end
    // Non-auto mode drafts into the input for the user to edit, so insert
    // the exact spoken words: the speaker/language prefix is only useful to
    // the agent, and would just be noise the user has to delete by hand.
    // An answer to a pending question is always submitted: the agent is
    // waiting on it.
    if (!autoSubmit && !answeringQuestion()) {
      insertAtCursor(spoken);
      return;
    }
    if (!inp.value) {
      inp.value = translated;
    } else {
      inp.value = inp.value + ' ' + translated;
    }
    inp.dispatchEvent(new Event('input', {bubbles: true}));
    try {
      inp.focus();
    } catch (_e) {}
    window.dispatchEvent(
      new CustomEvent('kiss-voice-submit', {
        detail: {tabId: ownerTabId(owner)},
      }),
    );
    speakWorkingOnIt();
  }

  function insertAtCursor(text) {
    const current = inp.value;
    const start =
      typeof inp.selectionStart === 'number'
        ? inp.selectionStart
        : current.length;
    const end =
      typeof inp.selectionEnd === 'number' ? inp.selectionEnd : current.length;
    const before = current.slice(0, start);
    const after = current.slice(end);
    const leadPad = before.length === 0 || /\s$/.test(before) ? '' : ' ';
    const trailPad = after.length === 0 || /^\s/.test(after) ? '' : ' ';
    const injected = leadPad + text + trailPad;
    inp.value = before + injected + after;
    // Restore the caret before notifying listeners: assigning `value` parks
    // the selection at the end, and main.js reads `selectionStart` inside its
    // synchronous `input` handler to decide whether to request a completion.
    const caret = start + leadPad.length + text.length;
    try {
      inp.focus();
      inp.setSelectionRange(caret, caret);
    } catch (_e) {}
    inp.dispatchEvent(new Event('input', {bubbles: true}));
  }

  function speakWorkingOnIt() {
    try {
      if (cfg.ackAudioUrl && typeof window.Audio === 'function') {
        const audio = new window.Audio(cfg.ackAudioUrl);
        const p = audio.play();
        if (p && typeof p.catch === 'function') {
          p.catch(() => {});
        }
      }
    } catch (_e) {}
  }

  let capture = null;

  // Rolling ring of the most recent pre-wake audio (16kHz Int16Array
  // chunks). When the wake word fires, the ring ends with the audio of
  // the wake word itself; it is prepended to the capture so the server
  // can re-confirm the wake word from the transcript (dual wake check).
  const WAKE_PREAMBLE_MS = 2000;
  let preambleChunks = [];
  let preambleMs = 0;

  function rememberPreamble(samples, sourceRate, blockMs) {
    preambleChunks.push({
      chunk: downsampleTo16k(samples, sourceRate),
      ms: blockMs,
    });
    preambleMs += blockMs;
    // Keep at least one chunk, and never evict below the cap: the wake
    // word must not be trimmed away by a block boundary.
    while (
      preambleChunks.length > 1 &&
      preambleMs - preambleChunks[0].ms >= WAKE_PREAMBLE_MS
    ) {
      preambleMs -= preambleChunks[0].ms;
      preambleChunks.shift();
    }
  }

  function clearPreamble() {
    preambleChunks = [];
    preambleMs = 0;
  }

  function beginCapture() {
    markSpeechStart();
    capture = {
      preamble: preambleChunks.map(entry => entry.chunk),
      chunks: [],
      sinceWakeMs: 0,
      elapsedMs: 0,
      speechStarted: false,
      trailingSilenceMs: 0,
    };
    clearPreamble();
  }

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

  function finishCapture() {
    const done = capture;
    capture = null;
    if (!done.speechStarted || !done.chunks.length) {
      // This round produced no audio, so no transcript will ever come back
      // for it. Retire its owner with it, or the next transcript would be
      // paired with this abandoned utterance's conversation. Rounds
      // overlap — this capture may have begun while an earlier round's
      // audio was still being transcribed — so the round to close is the
      // NEWEST one (this capture's own), never the oldest, which
      // is still waiting for its words.
      retireNewestRound();
      flashPendingRounds();
      return;
    }
    flash('voice-transcribing', 60000);
    // The wake word's own audio goes first: the server-side transcript
    // check (wakePrefixed) re-confirms the wake word before accepting
    // the rest of the utterance as a command. wakeSamples tells the
    // server where the preamble ends so speaker identification can use
    // the capture alone (the pre-wake ring may carry another voice).
    let wakeSamples = 0;
    for (let i = 0; i < done.preamble.length; i++) {
      wakeSamples += done.preamble[i].length;
    }
    postToDaemon({
      type: 'voiceTranscribe',
      audio: pcmBase64(done.preamble.concat(done.chunks)),
      wakePrefixed: true,
      wakeSamples: wakeSamples,
    });
  }

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
        s.remove();
        voskLoadPromise = null;
        reject(new Error('failed to load speech engine'));
      };
      document.head.appendChild(s);
    });
    return voskLoadPromise;
  }

  // How long a speech-model load may take before it is declared failed.
  // The vendored createModel settles when its worker posts 'load' or
  // fires a Worker 'error' event; a worker that does neither (the model
  // fetch hangs on a dead tunnel, WASM init stalls) leaves the promise
  // pending for ever, so without a bound `busy` stays true for the page
  // lifetime and the mic is wedged on 'loading' with every re-enable
  // deferring to `busy`.  The server never sets `cfg.modelLoadTimeoutMs`;
  // it is a test override of the 30 s bound.
  const MODEL_LOAD_TIMEOUT_MS =
    typeof cfg.modelLoadTimeoutMs === 'number' && cfg.modelLoadTimeoutMs > 0
      ? cfg.modelLoadTimeoutMs
      : 30000;

  // The one createModel in flight (or done) for the page: the model is a
  // 50 MB download and a worker, so every attempt shares it rather than
  // starting a second one next to it.  A failed load is forgotten so
  // the next attempt downloads afresh.
  let modelPromise = null;

  /**
   * Load (or reuse) the speech model, never hanging: this attempt is
   * raced against MODEL_LOAD_TIMEOUT_MS so a worker that never reports
   * settles it into the ordinary failure path (error painted, `busy`
   * cleared, re-enable retries).  The download itself goes on: a
   * re-enable waits for it instead of starting another, and a model
   * that arrives after its attempt timed out is kept for the next one.
   */
  function loadModel() {
    if (!modelPromise) {
      modelPromise = window.Vosk.createModel(cfg.modelUrl).then(
        m => {
          model = m;
          return m;
        },
        err => {
          modelPromise = null;
          // Realm-agnostic error check (`instanceof Error` fails across
          // window boundaries): anything carrying a message is reported
          // as-is; a rejection without one gets a message of its own.
          throw err && err.message
            ? err
            : new Error('failed to load speech model');
        },
      );
    }
    const load = modelPromise;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('speech model load timed out')),
        MODEL_LOAD_TIMEOUT_MS,
      );
      load.then(
        m => {
          clearTimeout(timer);
          resolve(m);
        },
        err => {
          clearTimeout(timer);
          reject(err);
        },
      );
    });
  }

  /**
   * The microphone went away under the live pipeline of generation
   * *gen* (headset unplugged, device taken by another app, the site's
   * permission revoked): its track fired `ended` and no more audio
   * comes.  Bring the pipeline down the way a failed start comes
   * down — mic off, the reason on the button — so the user is not
   * shown a "listening" mic that hears nothing.  A later toggle has
   * bumped the generation and torn that pipeline down already: then
   * there is nothing to do.  A wake's "Listening ..." overlay and
   * flash come down with the pipeline; the rounds whose audio was
   * already posted keep their owners (and the 60 s "transcribing"
   * timer that retires them if no transcript comes), their
   * transcripts still stand.
   */
  function onMicEnded(gen) {
    if (gen !== voiceGen) return;
    ++voiceGen;
    stopBrowserPipeline();
    flashPendingRounds();
    enabled = false;
    persist();
    setUi(
      'error',
      'the microphone was disconnected or taken by another application',
    );
  }

  function stopBrowserPipeline() {
    if (capture) {
      // An active capture dies with the pipeline before its audio was
      // posted (the mic was turned off mid-utterance), so no
      // transcript will ever answer its round. Retire it, or the NEXT
      // transcript would be paired with this abandoned utterance's
      // conversation and typed into (or dropped for) the wrong task.
      retireNewestRound();
      capture = null;
    }
    clearPreamble();
    if (processorNode) {
      try {
        processorNode.disconnect();
      } catch (_e) {}
      processorNode.onaudioprocess = null;
      processorNode = null;
    }
    if (sourceNode) {
      try {
        sourceNode.disconnect();
      } catch (_e) {}
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
      } catch (_e) {}
      audioContext = null;
    }
    if (recognizer) {
      try {
        recognizer.remove();
      } catch (_e) {}
      recognizer = null;
    }
    if (freeRecognizer) {
      try {
        freeRecognizer.remove();
      } catch (_e) {}
      freeRecognizer = null;
    }
  }

  function startBrowserPipeline() {
    // The generation this start belongs to. A later toggle or hide/show
    // bumps voiceGen; every continuation below re-checks it and stands
    // down instead of acting on a superseded intent.
    const gen = voiceGen;
    // Whether the catch below painted the red error state (and turned the
    // mic off): the settle handler at the end must not repaint it as a
    // calm "off".
    let paintedError = false;
    busy = true;
    setUi('loading');
    return loadVosk()
      .then(() => loadModel())
      .then(() => {
        if (gen !== voiceGen || !enabled) {
          // Superseded while the engine/model loaded: never even ask for
          // the microphone. The settle handler below fixes up the UI.
          return null;
        }
        return navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            channelCount: 1,
          },
        });
      })
      .then(stream => {
        if (!stream) return;
        if (gen !== voiceGen || !enabled) {
          // Superseded while getUserMedia was pending: release the just
          // granted microphone immediately.
          const tracks = stream.getTracks();
          for (let i = 0; i < tracks.length; i++) tracks[i].stop();
          return;
        }
        mediaStream = stream;
        const audioTracks = stream.getAudioTracks();
        for (let i = 0; i < audioTracks.length; i++) {
          audioTracks[i].addEventListener('ended', () => onMicEnded(gen));
        }
        if (debugEnabled()) {
          const track = audioTracks[0];
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
          audioContext.resume().catch(() => {});
        }
        const grammar = JSON.stringify(WAKE_ALIASES.concat(['[unk]']));
        recognizer = new model.KaldiRecognizer(
          audioContext.sampleRate,
          grammar,
        );
        if (typeof recognizer.setWords === 'function') {
          recognizer.setWords(true);
        }
        // Only a FINAL result can wake.  Partial hypotheses are not
        // trustworthy with a two-word grammar: Kaldi must label every
        // sound as either "abracadabra" or [unk], and mid-utterance it
        // routinely guesses "[unk] abracadabra" for ordinary speech
        // ("abstract algebra", "the data is in the dashboard", "sorry,
        // I could not hear you") before the final decode settles on
        // plain "[unk]".  Waking on such a partial after a 100 ms
        // pause, as this page once did, made the wake fire at random
        // during conversation.  The final result lands about a second
        // after the word ends, with a confidence per word.
        recognizer.on('result', message => {
          if (message && message.result) {
            debugLog('result', message.result.text);
            if (capture) return;
            if (
              (matchesWake(message.result.text) ||
                wakeWithLeadingNoise(message.result.result)) &&
              wordsConfident(message.result.result)
            ) {
              if (triggerWake()) beginCapture();
            }
          }
        });
        recognizer.on('partialresult', message => {
          if (message && message.result && message.result.partial) {
            debugLog('partial', message.result.partial);
          }
        });
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
          if (debugEnabled()) {
            const now = Date.now();
            if (now - lastRmsAt > 2000) {
              lastRmsAt = now;
              debugLog('rms', rms.toFixed(5));
            }
          }
          if (capture) {
            feedCapture(samples, rms, blockMs, event.inputBuffer.sampleRate);
            return;
          }
          rememberPreamble(samples, event.inputBuffer.sampleRate, blockMs);
          try {
            recognizer.acceptWaveform(event.inputBuffer);
            if (freeRecognizer)
              freeRecognizer.acceptWaveform(event.inputBuffer);
          } catch (_e) {}
        };
        sourceNode.connect(processorNode);
        processorNode.connect(audioContext.destination);
        setUi('listening');
      })
      .catch(err => {
        stopBrowserPipeline();
        if (gen === voiceGen && enabled) {
          // The failure belongs to an attempt the user still wants: turn
          // the mic off and show why. A superseded attempt's failure is
          // nobody's news — the settle handler below paints the state the
          // newer intent asked for instead of a stale red error.
          enabled = false;
          persist();
          paintedError = true;
          setUi('error', err);
        }
      })
      .then(() => {
        // Settle handler: runs after success, supersession, or failure.
        busy = false;
        if (gen === voiceGen && enabled) {
          // Still current and still wanted: the stream handler above ran
          // to completion (every path that could stop it either bumps the
          // generation or clears `enabled`), so the pipeline is live.
          return;
        }
        stopBrowserPipeline();
        if (enabled) {
          // A newer toggle-on arrived while this start was in flight (it
          // deferred to `busy`); run it now under its own generation.
          startBrowserPipeline();
        } else if (!paintedError) {
          setUi('off');
        }
      });
  }

  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, enabled ? '1' : '0');
    } catch (_e) {}
  }

  function wasEnabled() {
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch (_e) {
      return false;
    }
  }

  // voice.js runs in its own closure: main.js forwards the event's detail
  // to the daemon as a command.
  function postToDaemon(message) {
    window.dispatchEvent(new CustomEvent('kiss-voice-post', {detail: message}));
  }

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
      } catch (_e) {}
      renderSensitivity();
      // The pipeline reads `sensitivity` live in its wake callbacks.
    });
  }

  const autoSubmitSelect = document.getElementById('cfg-voice-auto-submit');

  function renderAutoSubmit() {
    if (autoSubmitSelect) autoSubmitSelect.value = autoSubmit ? 'on' : 'off';
  }

  renderAutoSubmit();
  if (autoSubmitSelect) {
    autoSubmitSelect.addEventListener('change', () => {
      autoSubmit = autoSubmitSelect.value !== 'off';
      try {
        localStorage.setItem(AUTO_SUBMIT_KEY, autoSubmit ? 'on' : 'off');
      } catch (_e) {}
    });
  }

  // Remote-web clients can have the same chat open in several same-origin
  // tabs; keep them in step with the tab that changed the setting.
  window.addEventListener('storage', event => {
    if (!event || event.key !== AUTO_SUBMIT_KEY) return;
    autoSubmit = event.newValue !== 'off';
    renderAutoSubmit();
  });

  function setEnabled(next) {
    if (enabled === next) return;
    // An explicit toggle supersedes whatever pipeline start is still in
    // flight.
    ++voiceGen;
    enabled = next;
    persist();
    if (next) {
      // When a superseded start is still unwinding (`busy`), its settle
      // handler observes this newer generation and restarts the pipeline.
      if (!busy) startBrowserPipeline();
      else setUi('loading');
    } else {
      // Stopping the pipeline retires a capture still under way (its
      // audio was never posted, so no transcript will answer it); the
      // rounds whose audio is being transcribed are cancelled. While a
      // start is in flight nothing has been captured yet and the
      // superseded start's settle handler releases whatever it acquired.
      if (!busy) stopBrowserPipeline();
      cancelSpeechRounds();
      flash(null);
      setUi('off');
    }
  }

  btn.addEventListener('click', () => {
    setEnabled(!enabled);
  });

  window.addEventListener('message', event => {
    // Daemon events arrive as synthetic MessageEvents dispatched by
    // media/transport.js, which carry no source window.  A real
    // postMessage from any (sandboxed) iframe on the page does, so it
    // cannot impersonate a transcript.
    if (!event || event.source) return;
    const msg = event.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'voiceSpeech' && typeof msg.text === 'string') {
      // tableak-coverage:start
      const owner = retireRound();
      insertSpeech(
        msg.text,
        roundOwners.length > 0,
        msg.speaker,
        msg.language,
        owner,
      );
      // tableak-coverage:end
    }
  });

  setUi('off');
  if (wasEnabled()) {
    setEnabled(true);
  }
})();

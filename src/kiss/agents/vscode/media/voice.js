// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
(function () {
  'use strict';

  const cfg = window.__VOICE__ || {mode: 'browser'};
  const btn = document.getElementById('voice-btn');
  const inp = document.getElementById('task-input');
  if (!btn || !inp) return;

  const WAKE_ALIASES = ['sorcar', 'sir car', 'sore car', 'sar car'];
  const COOLDOWN_MS = 2000;
  const DEFAULT_SENSITIVITY = 80;
  const TRAILING_ALIAS_SENSITIVITY = 75;
  const SENSITIVITY_KEY = 'kissVoiceSensitivity';
  const AUTO_SUBMIT_KEY = 'kissVoiceAutoSubmit';

  function sensitivityMinWordConf(s) {
    return 0.8 * (1 - s / 100);
  }

  function sensitivityWakePauseMs(s) {
    return Math.max(100, 400 * (1 - s / 100));
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
  let outstandingRounds = 0;

  // Generation stamp for the async voice lifecycle. Every transition that
  // invalidates work still in flight — a mic toggle, or hiding/showing a
  // webview page — bumps it; pending permission probes and pipeline-start
  // continuations compare the value they captured and stand down when
  // superseded, so an ON→OFF→ON burst can never start two capture paths
  // or resurrect a cancelled one.
  let voiceGen = 0;

  // True while the in-page pipeline is parked because the webview page is
  // hidden (VS Code retains hidden webviews, so scripts — and an open
  // microphone — would keep running unseen). Mirrors the host listener's
  // suspend-on-hide; cleared when the page shows again (the pipeline
  // restarts) or on any explicit toggle.
  let suspendedByHide = false;

  // Whether capture runs IN THIS PAGE (the browser-mic pipeline).
  // Browser-mode pages (the remote webapp) always capture in-page. A
  // webview-mode page starts on the host listener but switches here the
  // moment its embedder grants getUserMedia — the machine hosting the
  // extension may have no microphone at all (a headless remote host),
  // while the browser rendering this page does.
  let usingBrowserPipeline = cfg.mode !== 'webview';

  /**
   * True when this page is even equipped to try in-page capture: the
   * host handed it the vosk bundle + model URL and the browser exposes
   * getUserMedia. Checked before probing so pages without the wiring
   * (older hosts, test harnesses) keep the synchronous host-toggle path.
   */
  function browserCapturePossible() {
    return !!(
      cfg.voskSrc &&
      cfg.modelUrl &&
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function'
    );
  }

  /**
   * Ask the embedder for a microphone stream once, then release it.
   *
   * Resolves true when capture is permitted (the pipeline's own
   * getUserMedia then reuses the fresh grant without a second prompt),
   * false when it is refused — VS Code webviews currently reject with
   * NotAllowedError ("Permissions policy violation: microphone is not
   * allowed in this document", microsoft/vscode#323602), so the probe
   * costs nothing there and never prompts.
   */
  function probeBrowserCapture() {
    return navigator.mediaDevices.getUserMedia({audio: true}).then(
      stream => {
        const tracks = stream.getTracks();
        for (let i = 0; i < tracks.length; i++) tracks[i].stop();
        return true;
      },
      () => false,
    );
  }

  // How long a CANCELLED round entry is kept. A cancelled round's
  // transcript can only still arrive within the transcribe-flash window
  // (60s, see the flash('voice-transcribing', 60000) calls); entries older
  // than that are dead weight that repeated listener restarts would grow
  // without bound.
  const CANCELLED_ROUND_TTL_MS = 60000;

  // When each CANCELLED entry in roundOwners was cancelled, by round key.
  const cancelledRoundAt = new Map();

  /** Drop CANCELLED entries older than the transcribe-flash window. */
  function evictStaleCancelledRounds(now) {
    for (const [key, at] of cancelledRoundAt) {
      if (now - at <= CANCELLED_ROUND_TTL_MS) continue;
      cancelledRoundAt.delete(key);
      if (roundOwners.get(key) === CANCELLED) roundOwners.delete(key);
    }
  }

  // tableak-coverage:start
  // The conversation that was on screen when each outstanding round began,
  // keyed by that round's id. A transcript arrives seconds after the words
  // were spoken and the user may have moved to a different task in the
  // meantime, so submitting it against whatever tab is visible on arrival
  // would put one task's words into another task's conversation.
  //
  // Rounds are keyed, not queued. They overlap -- the wake detector starts
  // listening again while the previous utterance is still being transcribed
  // -- and they finish out of order, and any of them can be cancelled
  // mid-flight. Position in a queue therefore stops meaning anything: the
  // shift that answers a late transcript from a cancelled round would hand
  // it the owner of a round that is still waiting for its own words. The
  // round id is carried through wake -> transcript by the producer, so it
  // pairs each transcript with the utterance it actually belongs to.
  const roundOwners = new Map();

  // Owner recorded for a round that began in a page with no conversations at
  // all -- voice.js also runs in hosts that have no tab machinery. Such a
  // round has nothing to leak into, so it is not the same thing as a
  // CANCELLED round, which must still fail closed.
  const UNSCOPED = {unscoped: true};

  // Owner recorded for a round that was cancelled while its audio was still
  // being transcribed. The entry stays in the map so that the round's own
  // late transcript is recognised and refused, instead of being charged to
  // whichever round happens to be outstanding when it lands.
  const CANCELLED = {cancelled: true};

  // Rounds this webview started but could not name, because the producer sent
  // no id. They can only be answered in order, so they keep the old
  // positional behaviour among themselves.
  const unkeyedOwners = [];

  /**
   * The conversation on screen, as `{tabId, taskId}`, or null when the
   * webview has not published one.
   */
  function currentOwner() {
    const read = window.kissVoiceOwner;
    if (typeof read !== 'function') return null;
    const owner = read();
    return owner && owner.tabId ? owner : null;
  }

  /** The round id of a wake/speech message, or null when it carries none. */
  function roundKey(msg) {
    const id = msg ? msg.roundId : undefined;
    if (id === undefined || id === null || id === '') return null;
    return String(id);
  }

  /**
   * Record the owner of an utterance that is starting now.
   *
   * `key` is the round id the producer will echo back with the transcript, or
   * null when it sends none.
   */
  function markSpeechStart(key) {
    evictStaleCancelledRounds(Date.now());
    const owner = currentOwner() || UNSCOPED;
    if (key === null) {
      unkeyedOwners.push(owner);
    } else {
      roundOwners.set(key, owner);
      cancelledRoundAt.delete(key);
    }
    outstandingRounds++;
  }

  /**
   * Cancel every outstanding round, keeping each one recognisable.
   *
   * A cancelled round's transcript may still be on its way, so its id is kept
   * with a CANCELLED marker: that is what makes the late words fail closed
   * without stealing the owner of a round that is still live. Unkeyed rounds
   * have nothing to be recognised by, so they are only counted.
   */
  function resetSpeechRounds() {
    const now = Date.now();
    evictStaleCancelledRounds(now);
    for (const key of roundOwners.keys()) {
      roundOwners.set(key, CANCELLED);
      if (!cancelledRoundAt.has(key)) cancelledRoundAt.set(key, now);
    }
    cancelledUnkeyedRounds += unkeyedOwners.length;
    unkeyedOwners.length = 0;
    outstandingRounds = 0;
  }

  // Cancelled rounds that had no id. Their late transcripts must fail closed,
  // but there is nothing to key them by, so they are charged only once every
  // live unkeyed round has been answered.
  let cancelledUnkeyedRounds = 0;

  /**
   * Close the round `key` names and hand back the owner it was started with.
   *
   * A round that is still on record returns its own owner -- CANCELLED when
   * it was cancelled in flight, so its late words fail closed. An unkeyed
   * transcript is answered from the unkeyed rounds in order, then from the
   * cancelled unkeyed credit. With nothing outstanding at all the result is
   * undefined: this transcript belongs to no round this webview ever saw --
   * the host transcribes on its own too -- so it was never tied to a moment
   * in time and there is no tab switch to detect.
   */
  function retireRound(key) {
    if (key !== null && roundOwners.has(key)) {
      const owner = roundOwners.get(key);
      roundOwners.delete(key);
      cancelledRoundAt.delete(key);
      outstandingRounds = Math.max(0, outstandingRounds - 1);
      return owner;
    }
    if (key !== null) return undefined;
    outstandingRounds = Math.max(0, outstandingRounds - 1);
    if (unkeyedOwners.length) return unkeyedOwners.shift();
    if (cancelledUnkeyedRounds > 0) {
      cancelledUnkeyedRounds--;
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
   * recorded for this transcript (see retireRound) -- the host transcribes
   * on its own too. UNSCOPED means the round began in a page that has no
   * conversations at all. Neither has another conversation to leak into.
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
   * Retire the NEWEST unkeyed round — the one begun most recently.
   *
   * Used when a just-begun capture is abandoned before its audio was ever
   * posted (silence timeout, or the pipeline stopping mid-capture): no
   * transcript will ever answer that round, and the round to close is the
   * one beginCapture just recorded — the LAST unkeyed entry, not the
   * oldest, which may still be awaiting its own words. When the rounds
   * were reset in the meantime there is nothing left to retire.
   */
  function retireNewestUnkeyedRound() {
    if (!unkeyedOwners.length) return;
    unkeyedOwners.pop();
    outstandingRounds = Math.max(0, outstandingRounds - 1);
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
  let lastUiTip = "Voice trigger: listen for the word 'Sorcar'";
  let lastFlashCls = null;

  function askMicButtons() {
    return document.querySelectorAll('.ask-user-mic');
  }

  function applyUiClasses(el) {
    el.classList.remove(
      'voice-off',
      'voice-loading',
      'voice-listening',
      'voice-error',
      'voice-unavailable',
      'active',
    );
    el.classList.add('voice-' + lastUiState);
    if (lastUiState === 'listening') el.classList.add('active');
    el.setAttribute('data-tooltip', lastUiTip);
  }

  function applyFlashClasses(el) {
    el.classList.remove('voice-triggered', 'voice-transcribing');
    if (lastFlashCls) el.classList.add(lastFlashCls);
  }

  function applyFlashToAll() {
    applyFlashClasses(btn);
    const mics = askMicButtons();
    for (let i = 0; i < mics.length; i++) applyFlashClasses(mics[i]);
  }

  function syncAskMics() {
    const mics = askMicButtons();
    for (let i = 0; i < mics.length; i++) {
      applyUiClasses(mics[i]);
      applyFlashClasses(mics[i]);
    }
  }

  function setUi(state, message) {
    let tip;
    if (state === 'listening') {
      tip =
        "Voice trigger on: say 'Sorcar' and pause briefly " +
        '(click to turn off)';
    } else if (state === 'loading') {
      tip = 'Voice trigger: starting ...';
    } else if (state === 'error') {
      tip = 'Voice trigger error: ' + (message || 'unavailable');
    } else if (state === 'unavailable') {
      // Calm, non-error state: the machine running KISS has no
      // microphone and this window's embedder refused in-page capture,
      // so there is simply nothing to record with HERE — voice still
      // works in the remote web app, which uses the browser's mic.
      tip =
        'Voice capture unavailable in this window: the machine running ' +
        'KISS has no microphone. Use the remote web app to dictate with ' +
        "your browser's microphone.";
    } else {
      tip = "Voice trigger: listen for the word 'Sorcar'";
    }
    lastUiState = state;
    lastUiTip = tip;
    applyUiClasses(btn);
    syncAskMics();
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

  function flash(cls, timeoutMs) {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
    lastFlashCls = cls || null;
    applyFlashToAll();
    showListening(cls === 'voice-triggered');
    if (!cls) return;
    flashTimer = setTimeout(() => {
      flashTimer = null;
      lastFlashCls = null;
      applyFlashToAll();
      showListening(false);
      resetSpeechRounds();
    }, timeoutMs);
  }

  function triggerWake() {
    const now = Date.now();
    if (now - lastWakeAt < COOLDOWN_MS) return false;
    lastWakeAt = now;
    try {
      (askAnswerInput() || inp).focus();
    } catch (_e) {}
    flash('voice-triggered', 45000);
    return true;
  }

  function askAnswerInput() {
    const modal = document.getElementById('ask-user-modal');
    if (!modal || !modal.style.display || modal.style.display === 'none') {
      return null;
    }
    return modal.querySelector('.ask-user-input');
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
    // tab's input is no longer the one in the DOM. Hand the transcript back
    // to the host instead of typing it into a stranger's conversation.
    if (!ownerIsOnScreen(owner)) {
      postToHost({
        type: 'voiceDropped',
        tabId: ownerTabId(owner),
        text: translated,
      });
      return;
    }
    // tableak-coverage:end
    const askInp = askAnswerInput();
    if (askInp) {
      askInp.value = askInp.value
        ? askInp.value + ' ' + translated
        : translated;
      askInp.dispatchEvent(new Event('input', {bubbles: true}));
      try {
        askInp.focus();
      } catch (_e) {}
      window.dispatchEvent(
        new CustomEvent('kiss-voice-answer', {
          detail: {tabId: ownerTabId(owner)},
        }),
      );
      speakWorkingOnIt();
      return;
    }
    // Non-auto mode drafts into the input for the user to edit, so insert
    // the exact spoken words: the speaker/language prefix is only useful to
    // the agent, and would just be noise the user has to delete by hand.
    if (!autoSubmit) {
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
    if (cfg.mode === 'webview' && !usingBrowserPipeline) {
      postToHost({type: 'voiceAck'});
      return;
    }
    // In-page pipelines (remote webapp, webview browser-mic fallback)
    // play the ack in the page: the fallback exists precisely because
    // the host machine has no working audio hardware.
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
    // The browser pipeline captures, transcribes and answers one round at a
    // time in this closure, so there is no id to carry: the round is unkeyed.
    markSpeechStart(null);
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
      // NEWEST unkeyed one (this capture's own), never the oldest, which
      // is still waiting for its words.
      retireNewestUnkeyedRound();
      if (outstandingRounds > 0) flash('voice-transcribing', 60000);
      else flash(null);
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
    postToHost({
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
      // Webview pages run under a nonce-gated CSP; the host passes the
      // page nonce through the voice config so this injected tag is
      // allowed to execute. Browser-mode pages have no CSP nonce.
      if (cfg.nonce) s.nonce = cfg.nonce;
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

  // How long a speech-model load may take before it is declared failed.
  // The vendored createModel settles only when its worker posts a 'load'
  // message; a worker that fails to boot at all (its blob: URL blocked by
  // CSP, a top-level script/WASM crash) fires only a Worker 'error' event
  // that nobody listens for, so without a bound the promise never settles,
  // `busy` stays true for the page lifetime, and the mic is wedged on
  // 'loading' with every re-enable deferring to `busy`.
  const MODEL_LOAD_TIMEOUT_MS =
    typeof cfg.modelLoadTimeoutMs === 'number' && cfg.modelLoadTimeoutMs > 0
      ? cfg.modelLoadTimeoutMs
      : 30000;

  /**
   * Load (or reuse) the speech model, never hanging: the load is raced
   * against MODEL_LOAD_TIMEOUT_MS so a worker that never reports settles
   * this attempt into the ordinary failure path (error painted, `busy`
   * cleared, re-enable retries). A model that arrives after its timeout
   * already failed the attempt belongs to nobody: its worker is freed.
   */
  function loadModel() {
    if (model) return Promise.resolve(model);
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(() => {
        settled = true;
        reject(new Error('speech model load timed out'));
      }, MODEL_LOAD_TIMEOUT_MS);
      window.Vosk.createModel(cfg.modelUrl).then(
        m => {
          if (settled) {
            try {
              m.terminate();
            } catch (_e) {}
            return;
          }
          settled = true;
          clearTimeout(timer);
          model = m;
          resolve(m);
        },
        err => {
          if (settled) return;
          settled = true;
          clearTimeout(timer);
          // Realm-agnostic error check (`instanceof Error` fails across
          // window boundaries): anything carrying a message is reported
          // as-is, the vendored bare reject() gets a message of its own.
          reject(
            err && err.message ? err : new Error('failed to load speech model'),
          );
        },
      );
    });
  }

  function stopBrowserPipeline() {
    if (capture) {
      // An active capture dies with the pipeline before its audio was
      // posted (e.g. the webview was hidden mid-utterance), so no
      // transcript will ever answer its round. Retire it, or the NEXT
      // transcript would be paired with this abandoned utterance's
      // conversation and typed into (or dropped for) the wrong task.
      retireNewestUnkeyedRound();
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
        let quietMs = 0;
        let awaitingFlush = false;
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
              awaitingFlush = false;
              return;
            }
            if (capture) return;
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
            if (capture) return;
            if (
              quietMs >= sensitivityWakePauseMs(sensitivity) &&
              matchesWake(message.result.partial)
            ) {
              fireWake();
            }
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
          quietMs = rms >= SPEECH_RMS_THRESHOLD ? 0 : quietMs + blockMs;
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
          setUi('error', err && err.message);
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
        if (enabled && !suspendedByHide) {
          // A newer toggle-on arrived while this start was in flight (it
          // deferred to `busy`); run it now under its own generation.
          startBrowserPipeline();
        } else if (!enabled && !paintedError) {
          setUi('off');
        }
        // enabled && suspendedByHide: the page went hidden mid-start; the
        // visibility handler restarts the pipeline on the next show.
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

  function postToHost(message) {
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
      // The in-page pipeline reads `sensitivity` live in its wake
      // callbacks, so only a running HOST listener needs the value
      // shipped over (it restarts with the new --sensitivity).
      if (cfg.mode === 'webview' && !usingBrowserPipeline) {
        postToHost({type: 'voiceSensitivity', value: sensitivity});
      }
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
    // An explicit toggle supersedes whatever async voice work is still in
    // flight (a permission probe, a pipeline start) and overrides a
    // pending hide-suspend resume.
    const gen = ++voiceGen;
    suspendedByHide = false;
    enabled = next;
    persist();
    if (!next) {
      resetSpeechRounds();
      capture = null;
      flash(null);
    }
    if (cfg.mode === 'webview' && !usingBrowserPipeline) {
      if (next && browserCapturePossible()) {
        // Try in-page capture first: the machine hosting the extension
        // may have no microphone at all, while the browser rendering
        // this page does. When the embedder grants the mic, the same
        // in-page pipeline the remote webapp uses takes over and the
        // host listener is never started (so a mic-less host can never
        // throw). VS Code webviews currently refuse getUserMedia
        // (microsoft/vscode#323602): the probe rejects instantly and
        // falls through to the host listener exactly as before.
        setUi('loading');
        probeBrowserCapture().then(granted => {
          // Only the newest toggle's probe may act: a stale one could
          // otherwise start the host listener AND the in-page pipeline
          // together, or double-post voiceToggle(true).
          if (gen !== voiceGen || !enabled) return;
          if (granted) {
            usingBrowserPipeline = true;
            if (document.visibilityState === 'hidden') {
              // The page went hidden while the probe was pending: never
              // open a microphone for a page nobody can see. The
              // visibility handler starts the pipeline on the next show.
              suspendedByHide = true;
              return;
            }
            if (!busy) startBrowserPipeline();
          } else {
            postToHost({type: 'voiceToggle', enabled: true, sensitivity});
          }
        });
        return;
      }
      setUi(next ? 'loading' : 'off');
      postToHost({type: 'voiceToggle', enabled: next, sensitivity});
      return;
    }
    if (next) {
      // When a superseded start is still unwinding (`busy`), its settle
      // handler observes this newer generation and restarts the pipeline.
      if (!busy) startBrowserPipeline();
      else setUi('loading');
    } else {
      if (!busy) stopBrowserPipeline();
      // Paint off immediately even while a start is in flight; the
      // superseded start's settle handler releases whatever it acquired.
      setUi('off');
    }
  }

  // Webview pages outlive their visibility (VS Code retains them), so an
  // in-page microphone would keep recording after the user hides the
  // panel. Mirror the host listener's suspend-on-hide (see
  // SorcarSidebarView's onDidChangeVisibility): park the pipeline when the
  // page hides, restart it when the page shows again. Browser-mode pages
  // (the remote webapp) are untouched — a backgrounded browser tab
  // listening for its wake word is expected behavior there, and so is a
  // host listener kept alive by the host across webview hides.
  document.addEventListener('visibilitychange', () => {
    if (cfg.mode !== 'webview' || !usingBrowserPipeline) return;
    if (document.visibilityState === 'hidden') {
      if (!enabled || suspendedByHide) return;
      suspendedByHide = true;
      ++voiceGen; // strand any in-flight start; its settle handler cleans up
      if (!busy) stopBrowserPipeline();
    } else if (suspendedByHide) {
      suspendedByHide = false;
      ++voiceGen;
      if (enabled && !busy) startBrowserPipeline();
      // When busy, the stranded start's settle handler restarts it.
    }
  });

  btn.addEventListener('click', () => {
    setEnabled(!enabled);
  });

  document.addEventListener('click', event => {
    const t = event.target;
    const mic = t && t.closest ? t.closest('.ask-user-mic') : null;
    if (mic) setEnabled(!enabled);
  });

  window.addEventListener('kiss-ask-mic-mounted', () => {
    syncAskMics();
  });

  window.addEventListener('message', event => {
    const msg = event && event.data;
    if (!msg || typeof msg !== 'object') return;
    if (msg.type === 'voiceWake') {
      // tableak-coverage:start
      // The host stamps a monotonic round id on the wake and echoes it on the
      // transcript, so this owner is looked up by the round it belongs to
      // rather than by its position among the rounds still outstanding.
      markSpeechStart(roundKey(msg));
      // tableak-coverage:end
      triggerWake();
    } else if (msg.type === 'voiceTranscribing') {
      flash('voice-transcribing', 60000);
    } else if (msg.type === 'voiceSpeech') {
      // tableak-coverage:start
      const owner = retireRound(roundKey(msg));
      insertSpeech(
        msg.text,
        outstandingRounds > 0,
        msg.speaker,
        msg.language,
        owner,
      );
      // tableak-coverage:end
    } else if (msg.type === 'voiceState') {
      if (msg.hostMicUnavailable) {
        // The machine running KISS cannot capture audio and this
        // window's embedder already refused in-page capture (the
        // browser pipeline is tried BEFORE the host listener). Not an
        // error — nothing is broken, there is just no microphone
        // reachable from this window — so show a calm explanatory
        // state instead of the red error the raw listener failure
        // (e.g. "OSError: PortAudio library not found") used to paint.
        resetSpeechRounds();
        flash(null);
        enabled = false;
        persist();
        setUi('unavailable');
      } else if (msg.error) {
        resetSpeechRounds();
        flash(null);
        enabled = false;
        persist();
        setUi('error', msg.error);
      } else if (msg.listening) {
        setUi('listening');
      } else {
        resetSpeechRounds();
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

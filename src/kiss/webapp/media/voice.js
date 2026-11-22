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
  // The composer's mic (see "Dictation" below).
  const dictateBtn = document.getElementById('dictate-btn');
  const inp = document.getElementById('task-input');
  if (!btn || !dictateBtn || !inp) return;

  // The wake phrase "Hey Sorcar" (mirrors WAKE_PHRASE in
  // kiss/server/speech.py).  "Sorcar" is not in the Vosk model's
  // vocabulary, so the grammar spells it with the dictionary words the
  // model hears for it.
  const WAKE_ALIASES = ['hey sore car', 'hey sir car', 'hey sar car'];
  // How the grammar-free recognizer confirms a wake.  It cannot spell
  // the name: for "Hey Sorcar" in 21 macOS voices it wrote "hey
  // soccer", "hey saga", "hey saw car", "hey sort of", "who's or car",
  // and through this Mac's speaker and microphone "he's order".  So a
  // final confirms when it has the word "hey" (18 of the 21 voices) or
  // when it ends in words whose letters are within FREE_WAKE_MAX_EDITS
  // of FREE_WAKE_SPELLING ("he's order" and "who's or car" are 3 away;
  // everyday finals such as "his car" or "he's sorry" are 4 or more).
  const FREE_WAKE_WORD = 'hey';
  const FREE_WAKE_SPELLING = 'heysorcar';
  const FREE_WAKE_MAX_EDITS = 3;
  const FREE_WAKE_MAX_WORDS = 4;
  const COOLDOWN_MS = 2000;
  // The inaudible level the audio processor writes to its output so the
  // browser keeps the real audio sink (see onaudioprocess).
  const SINK_KEEP_AWAKE_LEVEL = 1e-7;
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
  // The voice trigger was on when dictation took the microphone: it is
  // off (`enabled`) until dictation ends and turns it on again (see
  // "Dictation" below).
  let wakePaused = false;

  // Generation stamp for the async voice lifecycle. Every mic toggle
  // bumps it; pipeline-start continuations compare the value they
  // captured and stand down when superseded, so an ON→OFF→ON burst can
  // never start two capture paths or resurrect a cancelled one.
  let voiceGen = 0;

  // tableak-coverage:start
  // The outstanding rounds, oldest first, as `{owner, id}`. `owner` is the
  // conversation that was on screen when the round's wake word was heard
  // (see heardWake): a transcript arrives seconds after the words were
  // spoken and the user may have moved to a different task in the
  // meantime, so submitting it against whatever tab is visible on arrival
  // would put one task's words into another task's conversation. `id` is
  // null while the round is still capturing and becomes the round's tag
  // once its audio is posted (see finishCapture). Only the newest round
  // can still be capturing.
  const rounds = [];

  // The tag of the round posted last.
  let lastRoundId = 0;

  // Owner recorded for a round that began in a page with no conversations at
  // all (a page without the tab machinery). Such a round has nothing to leak
  // into, so it is not the same thing as a CANCELLED round, which must still
  // fail closed.
  const UNSCOPED = {unscoped: true};

  // Owner handed back for a transcript that answers no outstanding round
  // (its round was cancelled while the audio was being transcribed, or it
  // names none): it is refused.
  const CANCELLED = {cancelled: true};

  /**
   * The conversation on screen, as main.js publishes it (`{tabId,
   * taskId, chatId, question, questionId}`, see kissVoiceOwner), or null
   * when the page has not published one.
   */
  function currentOwner() {
    const read = window.kissVoiceOwner;
    if (typeof read !== 'function') return null;
    const owner = read();
    return owner && owner.tabId ? owner : null;
  }

  /** Record and return a round begun while `owner` was on screen. */
  function markSpeechStart(owner) {
    const round = {owner: owner, id: null};
    rounds.push(round);
    return round;
  }

  /**
   * Cancel every outstanding round: the mic was turned off, or the
   * transcribe-flash window ran out.  A cancelled round's transcript may
   * still be on its way: its tag is outstanding no more, which is what
   * makes the late words fail closed (see retireRound).
   */
  function cancelSpeechRounds() {
    rounds.length = 0;
  }

  /**
   * Close the round a transcript answers and hand back the owner it was
   * started with.
   *
   * The daemon echoes the round tag `id` this page posted with the audio
   * (see finishCapture), which names the round: a tag no longer
   * outstanding belongs to a round this page cancelled, so its words are
   * CANCELLED, and so are those of a transcript that carries no tag (it
   * names no round, least of all one still capturing, whose `id` is
   * null).
   */
  function retireRound(id) {
    const at = typeof id === 'number' ? rounds.findIndex(r => r.id === id) : -1;
    return at === -1 ? CANCELLED : rounds.splice(at, 1)[0].owner;
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
   * The same tab is not enough: the daemon can rebind a tab to another
   * chat while a transcript is on its way (a replay, a run from another
   * surface), so the tab must still be on the owner's chat (`chatId`).
   * A chat has no id until its first task names it; until then its tab
   * alone tells it, and the id the tab gets is that chat's.
   *
   * The conversation must also still be waiting on the same thing: the
   * owner's pending ask_user_question (`question`, null when none, and
   * `questionId`, which tells two asks of the same words apart) must be
   * unchanged, so that words spoken as an answer do not answer a later
   * question, nor words spoken as a task answer one.
   *
   * UNSCOPED is not a failure: the round began in a page that has no
   * conversations at all, so there is no other one to leak into.
   */
  function ownerIsOnScreen(owner) {
    if (owner === UNSCOPED) return true;
    if (owner === CANCELLED) return false;
    const now = currentOwner();
    if (!now) return false;
    if (owner.question !== now.question) return false;
    if (owner.questionId !== now.questionId) return false;
    if (owner.tabId === now.tabId) {
      return !owner.chatId || owner.chatId === now.chatId;
    }
    return !!owner.taskId && owner.taskId === now.taskId;
  }

  /** The tab an utterance was recorded against, or '' when it had none. */
  function ownerTabId(owner) {
    return owner.tabId ? owner.tabId : '';
  }
  // tableak-coverage:end

  /**
   * Forget `round`, a capture abandoned before its audio was posted
   * (silence timeout, or the pipeline stopping mid-capture): no
   * transcript will ever answer it. A capture's round is outstanding for
   * as long as the capture lives: what cancels the rounds ends it too.
   */
  function dropRound(round) {
    rounds.splice(rounds.indexOf(round), 1);
  }

  let model = null;
  let recognizer = null;
  // The grammar-free recognizer run alongside `recognizer`: a wake needs
  // both to hear the word (see heardWake).  Kept here so
  // stopBrowserPipeline can free it: each one holds a worker-side Kaldi
  // recognizer and a model.recognizers entry for the model's page-long
  // lifetime.
  let freeRecognizer = null;
  let audioContext = null;
  let mediaStream = null;
  let sourceNode = null;
  let processorNode = null;
  let voskLoadPromise = null;

  // The one toast id every failure of the voice trigger reuses, and the
  // one every failure of dictation does, so a repeated failure updates
  // the visible notification instead of stacking a second one, and a
  // later successful start can close it.
  const VOICE_NOTIFICATION_ID = 'voice-trigger-error';
  const DICTATION_NOTIFICATION_ID = 'dictation-error';

  /**
   * Show (or, without a `message`, close) the visible notification `id`
   * of the voice trigger or of dictation.
   *
   * voice.js runs in its own closure and cannot call main.js's
   * showNotification directly, so it hands main.js the same
   * `{type: 'notification'}` message the daemon would send: main.js's
   * window 'message' handler routes an untagged notification to the
   * window-level toast container.  Errors are sticky: the user closes
   * them once they have read the remedy.
   */
  function notifyUser(id, severity, message) {
    const data = message
      ? {
          type: 'notification',
          id: id,
          severity: severity,
          sticky: true,
          message: message,
        }
      : {type: 'notification', id: id, close: true};
    window.dispatchEvent(new window.MessageEvent('message', {data: data}));
  }

  /**
   * Translate a capture failure into plain language with a remedy.
   *
   * `err` is the getUserMedia / AudioContext / model-load rejection (an
   * Error with `name`, or a string from the server).  `feature` names
   * what could not listen, as the remedy tells the user to turn it on
   * again: 'the voice trigger' or 'dictation'.
   */
  function describeVoiceError(err, feature) {
    const name = (err && err.name) || '';
    const text = String((err && err.message) || err || '').trim();
    const retry = ', then turn ' + feature + ' on again.';
    if (
      name === 'NotAllowedError' ||
      name === 'SecurityError' ||
      /denied|not allowed|permission/i.test(text)
    ) {
      return (
        'Microphone access was denied. Allow the microphone for this ' +
        'site in the browser settings' +
        retry
      );
    }
    if (name === 'NotFoundError' || /not found|no device/i.test(text)) {
      return 'No microphone was found. Connect a microphone' + retry;
    }
    if (name === 'NotReadableError' || /in use|could not start/i.test(text)) {
      return (
        'The microphone could not be started; another application may ' +
        'be using it. Close that application' +
        retry
      );
    }
    return (
      feature.charAt(0).toUpperCase() +
      feature.slice(1) +
      ' could not start' +
      (text ? ' (' + text + ')' : '') +
      '. Check the microphone' +
      retry
    );
  }

  function setUi(state, message) {
    let tip;
    if (state === 'listening') {
      tip =
        "Voice trigger on: say 'Hey Sorcar' and pause " +
        'briefly (click to turn off)';
      notifyUser(VOICE_NOTIFICATION_ID, null, null);
    } else if (state === 'loading') {
      tip = 'Voice trigger: starting ...';
    } else if (state === 'error') {
      const plain = describeVoiceError(message, 'the voice trigger');
      tip = 'Voice trigger error: ' + plain;
      notifyUser(VOICE_NOTIFICATION_ID, 'error', plain);
    } else {
      tip = "Voice trigger: listen for the phrase 'Hey Sorcar'";
    }
    btn.classList.remove(
      'voice-off',
      'voice-loading',
      'voice-listening',
      'voice-error',
      'active',
    );
    btn.classList.add('voice-' + state);
    if (state === 'listening') btn.classList.add('active');
    // The toggle's on/off state for assistive tech, not just a CSS class.
    btn.setAttribute('aria-pressed', state === 'listening' ? 'true' : 'false');
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
    if (WAKE_ALIASES.indexOf(t) !== -1) return true;
    if (sensitivity >= TRAILING_ALIAS_SENSITIVITY) {
      for (let i = 0; i < WAKE_ALIASES.length; i++) {
        const suffix = ' ' + WAKE_ALIASES[i];
        if (t.endsWith(suffix)) return true;
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
      if (typeof conf === 'number' && conf < minConf) {
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
    if (rounds.length > 0) flash('voice-transcribing', 60000);
    else flash(null);
  }

  function flash(cls, timeoutMs) {
    if (flashTimer !== null) {
      clearTimeout(flashTimer);
      flashTimer = null;
    }
    btn.classList.remove('voice-triggered', 'voice-transcribing');
    if (cls) btn.classList.add(cls);
    showListening(cls === 'voice-triggered');
    if (!cls) return;
    flashTimer = setTimeout(() => {
      flash(null);
      cancelSpeechRounds();
      // A capture still open (it ends on the audio clock, which can fall
      // behind this wall-clock timer) goes with its round.
      capture = null;
    }, timeoutMs);
  }

  // How far apart (ms) the two recognizers' final results may land and
  // still count as one wake.  The grammar-free final usually comes one
  // 4096-sample block (~0.26 s) after the grammar's, and up to ~1.5 s
  // later when a command follows the word without a pause.
  const WAKE_AGREE_MS = 2500;
  // The capture that a wake word heard by one recognizer alone would
  // begin, kept while the other may still confirm it (null: neither has
  // heard the word since the last wake or pipeline stop); see heardWake.
  let candidate = null;

  /** Return the Levenshtein edit distance between strings *a* and *b*. */
  function editDistance(a, b) {
    let previous = [];
    for (let j = 0; j <= b.length; j++) previous.push(j);
    for (let i = 1; i <= a.length; i++) {
      const current = [i];
      for (let j = 1; j <= b.length; j++) {
        current.push(
          Math.min(
            previous[j] + 1,
            current[j - 1] + 1,
            previous[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1),
          ),
        );
      }
      previous = current;
    }
    return previous[b.length];
  }

  /**
   * Whether the grammar-free transcript *text* confirms the wake phrase:
   * it has the word "hey", or its last one to FREE_WAKE_MAX_WORDS words
   * spell something within FREE_WAKE_MAX_EDITS of "heysorcar".
   */
  function freeTextHasWake(text) {
    const t = normalize(text);
    if (!t) return false;
    const words = t.split(' ');
    if (words.indexOf(FREE_WAKE_WORD) !== -1) return true;
    const first = Math.max(0, words.length - FREE_WAKE_MAX_WORDS);
    for (let i = first; i < words.length; i++) {
      const letters = words
        .slice(i)
        .join('')
        .replace(/[^a-z]/g, '');
      if (editDistance(letters, FREE_WAKE_SPELLING) <= FREE_WAKE_MAX_EDITS) {
        return true;
      }
    }
    return false;
  }

  /**
   * Record that one recognizer heard the wake word, and wake when the
   * other one heard it too within WAKE_AGREE_MS.
   *
   * The grammar recognizer alone is not enough: its grammar has only
   * the wake phrase and [unk], so Kaldi can force-fit any loud enough
   * stretch of speech onto it (Russian and Bengali talk did with the
   * earlier "Abracadabra", 2026-11-08).  The grammar-free recognizer
   * has the full vocabulary, so it writes the foreign words instead
   * (see freeTextHasWake); English sound-alikes it lets through ("hey,
   * so the car.") are rejected by the daemon's transcript check
   * (speech.py).
   *
   * The first hearing opens the candidate: the capture as it would begin
   * right then, with the conversation on screen and the ring (it ends
   * with the word's own audio), to which the audio that follows is added
   * (feedCandidate).  The confirmed wake adopts it (beginCapture), so a
   * command spoken before the confirmation is neither lost nor handed to
   * a conversation selected meanwhile.  When the candidate's conversation
   * has left the screen it is dropped together with the final that would
   * have confirmed it: both heard words spoken there.
   *
   * @param {boolean} fromGrammar true for the grammar recognizer's
   *   final, false for the grammar-free one's.
   * @returns {boolean} true when the wake fired.
   */
  function heardWake(fromGrammar) {
    const now = Date.now();
    if (
      candidate &&
      candidate.fromGrammar !== fromGrammar &&
      now - candidate.heardAt <= WAKE_AGREE_MS
    ) {
      if (ownerIsOnScreen(candidate.owner) && triggerWake()) return true;
      candidate = null;
      return false;
    }
    candidate = {
      fromGrammar: fromGrammar,
      heardAt: now,
      owner: currentOwner() || UNSCOPED,
      preamble: preamble.entries.map(entry => entry.chunk),
      chunks: [],
      sinceWakeMs: 0,
      elapsedMs: 0,
      speechStarted: false,
      trailingSilenceMs: 0,
    };
    return false;
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

  /**
   * Whether the composer may take the words of an utterance spoken into
   * the conversation `owner`, which it is made to show first.
   */
  function composerTakes(owner) {
    // tableak-coverage:start
    // The user switched tasks while speaking: the words belong to the
    // conversation that was on screen when the utterance began, and that
    // tab's input is no longer the one in the DOM. Drop the transcript
    // instead of typing it into a stranger's conversation.
    if (!ownerIsOnScreen(owner)) return false;
    // tableak-coverage:end
    // The composer read and written by the caller must be this
    // conversation's.  While a file is on screen the composer is hidden,
    // so main.js brings the chat the file screen stands for
    // (kissVoiceOwner) back first, with its own draft and its pending
    // question. It cancels the event when no chat is left to take the
    // words (the file on screen outlived its chats): they are dropped.
    const compose = new CustomEvent('kiss-voice-compose', {
      detail: {tabId: ownerTabId(owner)},
      cancelable: true,
    });
    return window.dispatchEvent(compose);
  }

  // owner is this round's conversation, already retired by the caller. It is
  // taken as a parameter rather than read here so that the round is closed
  // before anything can return early: an empty transcript still completes a
  // round.
  function insertSpeech(text, keepFlash, speaker, language, owner) {
    if (!keepFlash) flash(null);
    const spoken = text.trim();
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
    if (!composerTakes(owner)) return;
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

  // A rolling ring of the audio heard last while nothing was being
  // recorded, as `{entries, ms, keepMs}`: 16kHz Int16Array chunks with
  // their durations, `ms` of audio in all, of which the last `keepMs`
  // are kept.
  //
  // This one is the wake pipeline's.  When a recognizer hears the wake
  // word, the ring ends with the audio of the word itself; the candidate
  // takes it (heardWake) and it is prepended to the capture so the
  // server can re-confirm the wake word from the transcript (dual wake
  // check).
  const WAKE_PREAMBLE_MS = 2000;
  const preamble = {entries: [], ms: 0, keepMs: WAKE_PREAMBLE_MS};

  /** Add one audio block to `ring` and drop what it no longer keeps. */
  function rememberAudio(ring, samples, sourceRate, blockMs) {
    ring.entries.push({
      chunk: downsampleTo16k(samples, sourceRate),
      ms: blockMs,
    });
    ring.ms += blockMs;
    // Keep at least one chunk, and never evict below the cap: the wake
    // word must not be trimmed away by a block boundary.
    while (
      ring.entries.length > 1 &&
      ring.ms - ring.entries[0].ms >= ring.keepMs
    ) {
      ring.ms -= ring.entries[0].ms;
      ring.entries.shift();
    }
  }

  /** Empty `ring` and hand back the chunks it held, oldest first. */
  function takeAudio(ring) {
    const chunks = ring.entries.map(entry => entry.chunk);
    ring.entries = [];
    ring.ms = 0;
    return chunks;
  }

  /**
   * The wake is confirmed: the candidate becomes the capture, its round
   * owned by the conversation that was on screen when the word was first
   * heard.
   */
  function beginCapture() {
    capture = candidate;
    candidate = null;
    capture.round = markSpeechStart(capture.owner);
    takeAudio(preamble);
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
    if (!done.speechStarted) {
      // This round produced no audio, so no transcript will ever come back
      // for it: forget it, never an earlier round still awaiting its words.
      dropRound(done.round);
      flashPendingRounds();
      return;
    }
    done.round.id = ++lastRoundId;
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
      // Every capture starts from heardWake, i.e. both recognizers heard
      // the word; the server needs this to accept a non-English
      // transcript that lost the word in translation.
      wakeAgreed: true,
      // The daemon echoes the tag on the transcript (see retireRound).
      round: done.round.id,
    });
  }

  /**
   * Add one audio block to `c`, the capture or the candidate that may
   * become it; true once it is over: no speech within
   * CAPTURE_NO_SPEECH_TIMEOUT_MS of the wake word, CAPTURE_END_SILENCE_MS
   * of silence after speech, or CAPTURE_MAX_MS of speech.
   */
  function feedCapture(c, samples, rms, blockMs, sourceRate) {
    const loud = rms >= SPEECH_RMS_THRESHOLD;
    c.sinceWakeMs += blockMs;
    if (!c.speechStarted) {
      if (!loud) return c.sinceWakeMs >= CAPTURE_NO_SPEECH_TIMEOUT_MS;
      c.speechStarted = true;
    }
    c.chunks.push(downsampleTo16k(samples, sourceRate));
    c.elapsedMs += blockMs;
    c.trailingSilenceMs = loud ? 0 : c.trailingSilenceMs + blockMs;
    return (
      c.trailingSilenceMs >= CAPTURE_END_SILENCE_MS ||
      c.elapsedMs >= CAPTURE_MAX_MS
    );
  }

  /**
   * Add one audio block to the candidate, which holds it until the other
   * recognizer confirms the wake word; past WAKE_AGREE_MS it no longer
   * can, and the candidate is dropped.
   */
  function feedCandidate(samples, rms, blockMs, sourceRate) {
    if (Date.now() - candidate.heardAt > WAKE_AGREE_MS) candidate = null;
    else feedCapture(candidate, samples, rms, blockMs, sourceRate);
  }

  // How long the speech engine (vosk.js), a speech-model load, the
  // microphone request or the audio context's start may take before it
  // is declared failed.  The engine's script tag settles on `load` or
  // `error`, the vendored createModel when its worker posts 'load' or
  // fires a Worker 'error' event, getUserMedia when the browser answers
  // and resume() when the context runs; one that does none of these (a
  // stalled response, a service worker that never answers, the model
  // fetch hanging on a dead tunnel, WASM init stalling, a permission
  // prompt nobody answers, a context the browser keeps suspended) stays
  // pending for ever, so without a bound `busy` stays true for the page
  // lifetime and the mic is wedged on 'loading' with every re-enable
  // deferring to `busy`.  The server never sets `cfg.modelLoadTimeoutMs`;
  // it is a test override of the 30 s bound.
  const MODEL_LOAD_TIMEOUT_MS =
    typeof cfg.modelLoadTimeoutMs === 'number' && cfg.modelLoadTimeoutMs > 0
      ? cfg.modelLoadTimeoutMs
      : 30000;

  /**
   * Load the speech engine bundle once.  A load that fails, or does not
   * settle within MODEL_LOAD_TIMEOUT_MS, removes its tag and is
   * forgotten, so the next attempt injects a fresh one.
   */
  function loadVosk() {
    if (window.Vosk) return Promise.resolve();
    if (voskLoadPromise) return voskLoadPromise;
    voskLoadPromise = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      const timer = setTimeout(() => s.onerror(), MODEL_LOAD_TIMEOUT_MS);
      s.src = cfg.voskSrc;
      s.onload = () => {
        clearTimeout(timer);
        resolve();
      };
      s.onerror = () => {
        clearTimeout(timer);
        s.remove();
        voskLoadPromise = null;
        reject(new Error('failed to load speech engine'));
      };
      document.head.appendChild(s);
    });
    return voskLoadPromise;
  }

  // The one createModel in flight (or done) for the page: the model is a
  // 50 MB download and a worker, so every attempt shares it rather than
  // starting a second one next to it.  A failed load is forgotten so
  // the next attempt downloads afresh.
  let modelPromise = null;
  // The attempts that timed out waiting on the load in flight: the first
  // keeps sharing it (a slow download finishes for the next attempt),
  // the second gives it up as stalled (a worker that never reports) and
  // cancels it, so the attempt after that downloads afresh instead of
  // waiting on it for ever.
  let modelTimeouts = 0;

  /**
   * Load (or reuse) the speech model, never hanging: this attempt is
   * raced against MODEL_LOAD_TIMEOUT_MS so a worker that never reports
   * settles it into the ordinary failure path (error painted, `busy`
   * cleared, re-enable retries).  The download itself goes on: a
   * re-enable waits for it instead of starting another, and a model
   * that arrives after its attempt timed out is kept for the next one.
   * A load two attempts in a row timed out on is given up: the engine
   * cancels it (its worker is terminated, see vosk.js createModel) and
   * the next attempt starts a fresh one.
   */
  function loadModel() {
    if (!modelPromise) {
      modelTimeouts = 0;
      const engineLoad = window.Vosk.createModel(cfg.modelUrl);
      const load = engineLoad.then(
        m => {
          model = m;
          // The engine's own worker 'error' listener is one-shot (boot
          // failures): a worker that dies after its load is this page's
          // to notice, for the model's whole life.
          if (m.worker && typeof m.worker.addEventListener === 'function') {
            m.worker.addEventListener('error', err =>
              onModelWorkerError(m, err),
            );
          }
          return m;
        },
        err => {
          if (modelPromise === load) modelPromise = null;
          // Realm-agnostic error check (`instanceof Error` fails across
          // window boundaries): anything carrying a message is reported
          // as-is; a rejection without one gets a message of its own.
          throw err && err.message
            ? err
            : new Error('failed to load speech model');
        },
      );
      load.cancel = engineLoad.cancel;
      modelPromise = load;
    }
    const load = modelPromise;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (++modelTimeouts > 1 && modelPromise === load) {
          modelPromise = null;
          load.cancel();
        }
        reject(new Error('speech model load timed out'));
      }, MODEL_LOAD_TIMEOUT_MS);
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

  // Why a listener whose microphone track ended can no longer listen.
  const MIC_LOST =
    'the microphone was disconnected or taken by another application';

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
    failLivePipeline(gen, MIC_LOST);
  }

  /**
   * The speech worker reported an error for the wake-word recognizer of
   * generation *gen* (it could not build it, or could not decode a
   * chunk): nothing can wake through this pipeline any more, so it comes
   * down like a lost microphone, with the worker's reason on the button.
   * The loaded model is kept: the next click retries on it.
   */
  function onRecognizerError(gen, message) {
    failLivePipeline(
      gen,
      (message && message.error) || 'the speech recognizer failed',
    );
  }

  /**
   * The speech worker of the loaded model *m* died (an uncaught error
   * in the worker, a WASM abort): it decodes nothing any more, so a live
   * pipeline on it would stay "listening" to a microphone nothing hears.
   * The worker is reclaimed and the model forgotten, so the next enable
   * loads a fresh one; a pipeline that is live or starting on it comes
   * down like a lost microphone, with the worker's reason on the button.
   * A model forgotten already (its worker erred before) is only reclaimed.
   */
  function onModelWorkerError(m, err) {
    m.worker.terminate();
    if (model !== m) return;
    model = null;
    modelPromise = null;
    if (enabled) {
      failLivePipeline(
        voiceGen,
        (err && err.message) || 'the speech worker failed',
      );
    }
  }

  function failLivePipeline(gen, reason) {
    if (gen !== voiceGen) return;
    ++voiceGen;
    stopBrowserPipeline();
    flashPendingRounds();
    enabled = false;
    persist();
    setUi('error', reason);
  }

  function stopTracks(stream) {
    const tracks = stream.getTracks();
    for (let i = 0; i < tracks.length; i++) tracks[i].stop();
  }

  /**
   * Ask for the microphone, never hanging: the request is raced against
   * MODEL_LOAD_TIMEOUT_MS (a permission prompt nobody answers keeps
   * getUserMedia pending for ever, and the browser offers no way to
   * abort it), so a stalled request settles its attempt into the
   * ordinary failure path instead of wedging `busy`, and with it every
   * re-enable, for the page lifetime.  The request itself goes on: a
   * stream granted after the deadline belongs to an attempt that is
   * over and is released at once.
   */
  function requestMicrophone() {
    const request = navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        channelCount: 1,
      },
    });
    return new Promise((resolve, reject) => {
      let expired = false;
      const timer = setTimeout(() => {
        expired = true;
        reject(new Error('microphone request timed out'));
      }, MODEL_LOAD_TIMEOUT_MS);
      request.then(
        stream => {
          if (expired) {
            stopTracks(stream);
            return;
          }
          clearTimeout(timer);
          resolve(stream);
        },
        err => {
          clearTimeout(timer);
          reject(err);
        },
      );
    });
  }

  /**
   * Start *context* when the browser created it suspended, never
   * hanging: its resume() stays pending for as long as the browser keeps
   * the audio suspended and may be refused, and a pipeline built on a
   * context that does not run would show "listening" and hear nothing.
   * So the resume is raced against MODEL_LOAD_TIMEOUT_MS like the
   * microphone request, and either failure settles the attempt into the
   * ordinary failure path.
   */
  function resumeAudio(context) {
    if (context.state !== 'suspended') return Promise.resolve();
    const resumed = context.resume();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(
        () => reject(new Error('the browser kept the audio suspended')),
        MODEL_LOAD_TIMEOUT_MS,
      );
      resumed.then(
        () => {
          clearTimeout(timer);
          resolve();
        },
        err => {
          clearTimeout(timer);
          reject(err);
        },
      );
    });
  }

  /**
   * Take down what a listener holds of its audio graph, each of which
   * may be null: the processor and source nodes, the microphone
   * `stream` and the audio `context`.
   */
  function releaseAudio(processor, source, stream, context) {
    if (processor) {
      try {
        processor.disconnect();
      } catch (_e) {}
      processor.onaudioprocess = null;
    }
    if (source) {
      try {
        source.disconnect();
      } catch (_e) {}
    }
    if (stream) stopTracks(stream);
    if (context) {
      try {
        context.close();
      } catch (_e) {}
    }
  }

  /**
   * Take the block of microphone audio a processor node hands over in
   * `event`, as `{samples, rms, rate, blockMs}`: the samples, their
   * loudness (root mean square), their sample rate and their duration.
   */
  function readBlock(event) {
    // Never output pure silence: after 30 s of zeros Chrome swaps
    // the audio sink for a fake one that ran this callback at
    // under half speed (measured 2026-11-08 in Google Chrome with
    // the real mic: 43 s of audio in 60 s), so the recognizers got
    // chopped audio and missed the wake phrase.  1e-7 is about
    // -140 dBFS, far below hearing.
    event.outputBuffer.getChannelData(0).fill(SINK_KEEP_AWAKE_LEVEL);
    const samples = event.inputBuffer.getChannelData(0);
    let sumSquares = 0;
    for (let i = 0; i < samples.length; i++) {
      sumSquares += samples[i] * samples[i];
    }
    const rate = event.inputBuffer.sampleRate;
    return {
      samples: samples,
      rms: Math.sqrt(sumSquares / samples.length),
      rate: rate,
      blockMs: (samples.length / rate) * 1000,
    };
  }

  function stopBrowserPipeline() {
    if (capture) {
      // An active capture dies with the pipeline before its audio was
      // posted (the mic was turned off mid-utterance), so no
      // transcript will ever answer its round.
      dropRound(capture.round);
      capture = null;
    }
    candidate = null;
    takeAudio(preamble);
    releaseAudio(processorNode, sourceNode, mediaStream, audioContext);
    processorNode = null;
    sourceNode = null;
    mediaStream = null;
    audioContext = null;
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
    // The generation this start belongs to. A later toggle (setEnabled)
    // or a lost microphone (onMicEnded) bumps voiceGen; every
    // continuation below re-checks it and stands down instead of acting
    // on a superseded intent.
    const gen = voiceGen;
    busy = true;
    setUi('loading');
    return loadVosk()
      .then(() => loadModel())
      .then(() => {
        if (gen !== voiceGen || !enabled) {
          // Superseded while the engine/model loaded: never even ask for
          // the microphone.
          return null;
        }
        return requestMicrophone();
      })
      .then(stream => {
        if (!stream) return;
        if (gen !== voiceGen || !enabled) {
          // Superseded while getUserMedia was pending: release the just
          // granted microphone immediately.
          stopTracks(stream);
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
        return resumeAudio(audioContext);
      })
      .then(() => {
        if (gen !== voiceGen || !enabled) {
          // Superseded before the audio ran (an earlier step stood down,
          // or a toggle or a failure came during the resume): what
          // superseded this start has released what it held.
          return;
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
        // trustworthy with a tiny grammar: Kaldi must label every
        // sound as a wake-phrase word or [unk], and mid-utterance it
        // routinely guesses the wake phrase for ordinary speech
        // (measured with the earlier "Abracadabra": "abstract
        // algebra", "sorry, I could not hear you") before the final
        // decode settles on plain "[unk]".  Waking on such a partial
        // after a 100 ms pause, as this page once did, made the wake
        // fire at random during conversation.  The final result lands
        // about a second after the word ends, with a confidence per
        // word.  Even a final only counts once the grammar-free
        // recognizer below agrees (heardWake).
        recognizer.on('result', message => {
          if (message && message.result) {
            debugLog('result', message.result.text);
            if (capture) return;
            if (
              (matchesWake(message.result.text) ||
                wakeWithLeadingNoise(message.result.result)) &&
              wordsConfident(message.result.result) &&
              heardWake(true)
            ) {
              beginCapture();
            }
          }
        });
        recognizer.on('partialresult', message => {
          if (message && message.result && message.result.partial) {
            debugLog('partial', message.result.partial);
          }
        });
        recognizer.on('error', message => onRecognizerError(gen, message));
        freeRecognizer = new model.KaldiRecognizer(audioContext.sampleRate);
        freeRecognizer.on('result', message => {
          if (message && message.result) {
            debugLog('free-result', message.result.text);
            if (capture) return;
            if (freeTextHasWake(message.result.text) && heardWake(false)) {
              beginCapture();
            }
          }
        });
        freeRecognizer.on('partialresult', message => {
          if (message && message.result && message.result.partial) {
            debugLog('free-partial', message.result.partial);
          }
        });
        freeRecognizer.on('error', message => onRecognizerError(gen, message));
        sourceNode = audioContext.createMediaStreamSource(mediaStream);
        processorNode = audioContext.createScriptProcessor(4096, 1, 1);
        let lastRmsAt = 0;
        processorNode.onaudioprocess = event => {
          const {samples, rms, rate, blockMs} = readBlock(event);
          if (debugEnabled()) {
            const now = Date.now();
            if (now - lastRmsAt > 2000) {
              lastRmsAt = now;
              debugLog('rms', rms.toFixed(5));
            }
          }
          if (capture) {
            if (feedCapture(capture, samples, rms, blockMs, rate)) {
              finishCapture();
            }
            return;
          }
          rememberAudio(preamble, samples, rate, blockMs);
          if (candidate) feedCandidate(samples, rms, blockMs, rate);
          try {
            recognizer.acceptWaveform(event.inputBuffer);
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
          // nobody's news: what superseded it has painted the state that
          // stands.
          enabled = false;
          persist();
          setUi('error', err);
        }
      })
      .then(() => {
        // Settle handler: runs after success, supersession, or failure.
        busy = false;
        if (gen === voiceGen && enabled) {
          // Still current and still wanted: the steps above ran to
          // completion (every path that could stop them either bumps the
          // generation or clears `enabled`), so the pipeline is live.
          return;
        }
        stopBrowserPipeline();
        if (enabled) {
          // A newer toggle-on arrived while this start was in flight (it
          // deferred to `busy`); run it now under its own generation.
          startBrowserPipeline();
        }
        // Otherwise the button keeps what the path that turned the mic off
        // painted: "off" (setEnabled) or its error (the catch above,
        // failLivePipeline).
      });
  }

  /**
   * Store whether the user has the voice trigger on.  One that
   * dictation has paused (`wakePaused`) still is.
   */
  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, enabled || wakePaused ? '1' : '0');
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
    if (!event) return;
    if (event.key === AUTO_SUBMIT_KEY) {
      autoSubmit = event.newValue !== 'off';
      renderAutoSubmit();
    } else if (event.key === SENSITIVITY_KEY) {
      sensitivity = storedSensitivity();
      renderSensitivity();
    }
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
      // rounds whose audio is being transcribed are cancelled. A start
      // still in flight loses what it holds at once too (the microphone
      // and the audio context of one waiting in resumeAudio): the mic is
      // released when the button says so, not when that start settles.
      stopBrowserPipeline();
      cancelSpeechRounds();
      flash(null);
      setUi('off');
    }
  }

  // dictation-coverage:start
  // Dictation: the composer's mic (#dictate-btn).
  //
  // While the mic is on, what the user says is typed at the cursor of
  // the text box, in English.  No wake word is involved: an utterance
  // starts with the first loud block of audio and ends like the speech
  // after a wake word does (CAPTURE_END_SILENCE_MS of silence, or
  // CAPTURE_MAX_MS of speech).  Its audio goes to the daemon in the same
  // `voiceTranscribe` command, marked `dictation` instead of carrying the
  // wake word's fields, and the answer is its English transcript.
  // Nothing is ever submitted.
  //
  // Dictation and the voice trigger never listen together, or the same
  // words would be typed twice: the mic pauses a trigger that is on and
  // turns it on again when dictation ends, and turning the trigger on
  // ends dictation.

  // An utterance takes the audio heard just before its first loud block
  // along: the start of a word is often too soft to count as loud.
  const DICTATION_LEAD_IN_MS = 500;
  // An utterance with less loud audio than this is a click or a knock,
  // not speech: it is not sent.
  const DICTATION_MIN_SPEECH_MS = 200;
  // Someone who speaks on without a long pause is cut at a short one
  // (DICTATION_BREATH_MS) once the utterance is this long, rather than
  // in the middle of a word at CAPTURE_MAX_MS.
  const DICTATION_BREATH_AFTER_MS = 20000;
  const DICTATION_BREATH_MS = 300;
  // How long a transcript may take before its round is given up.
  const DICTATION_REPLY_TIMEOUT_MS = 60000;

  // What the dictation that is on holds, null while the mic is off: the
  // microphone and its audio graph (`stream`, `context`, `source`,
  // `processor`, each null until the start has it), the ring of the
  // audio heard before an utterance (`leadIn`) and the utterance being
  // recorded (`utterance`, or null).  A start in flight keeps the object
  // it began with and stands down once this is another: the mic was
  // toggled since.
  let dictation = null;
  // Why dictation failed, in plain language, until it is turned on again.
  let dictationError = null;
  // The utterances awaiting their transcript, oldest first, as `{id,
  // owner, timer}`: the tag the daemon echoes, the conversation that was
  // on screen when the utterance began (see `rounds`) and the timer that
  // gives the round up.  They outlive the mic: what was said before it
  // was turned off is still typed.
  const dictationRounds = [];

  /** Paint the mic button from what dictation is doing. */
  function renderDictation() {
    const on = dictation !== null;
    const listening = on && dictation.processor !== null;
    const hearing = listening && dictation.utterance !== null;
    let tip =
      'Dictate: speak and your words are typed at the cursor, in English';
    if (dictationError) {
      tip = 'Dictation error: ' + dictationError;
    } else if (listening) {
      tip =
        'Dictation on: your words are typed at the cursor, in English ' +
        '(click to turn off)';
    } else if (on) {
      tip = 'Dictation: starting ...';
    }
    dictateBtn.classList.toggle('dictate-loading', on && !listening);
    dictateBtn.classList.toggle('active', listening);
    dictateBtn.classList.toggle('dictate-hearing', hearing);
    dictateBtn.classList.toggle(
      'dictate-transcribing',
      !hearing && dictationRounds.length > 0,
    );
    dictateBtn.classList.toggle('dictate-error', dictationError !== null);
    dictateBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
    dictateBtn.setAttribute('data-tooltip', tip);
  }

  /** Give the dictation round `round` up: no transcript came in time. */
  function expireDictationRound(round) {
    dictationRounds.splice(dictationRounds.indexOf(round), 1);
    renderDictation();
  }

  /**
   * Close the dictation round that the transcript tagged `id` answers
   * and hand it back; null when it answers none (a round of the voice
   * trigger, or one given up).
   */
  function takeDictationRound(id) {
    const at = dictationRounds.findIndex(round => round.id === id);
    if (at === -1) return null;
    const round = dictationRounds.splice(at, 1)[0];
    clearTimeout(round.timer);
    renderDictation();
    return round;
  }

  /**
   * End the utterance `held` is recording and send it for its
   * transcript, unless it has too little loud audio to be speech.
   */
  function endUtterance(held) {
    const said = held.utterance;
    held.utterance = null;
    if (said.loudMs >= DICTATION_MIN_SPEECH_MS) {
      const round = {id: ++lastRoundId, owner: said.owner, timer: null};
      round.timer = setTimeout(
        expireDictationRound,
        DICTATION_REPLY_TIMEOUT_MS,
        round,
      );
      dictationRounds.push(round);
      postToDaemon({
        type: 'voiceTranscribe',
        audio: pcmBase64(said.chunks),
        // No wake word precedes these words: the daemon cuts none from
        // their start and looks for no speaker.
        dictation: true,
        // The daemon echoes the tag on the transcript.
        round: round.id,
      });
    }
    renderDictation();
  }

  /** Take one block of microphone audio for the dictation `held`. */
  function hearDictation(held, event) {
    const {samples, rms, rate, blockMs} = readBlock(event);
    const loud = rms >= SPEECH_RMS_THRESHOLD;
    if (!held.utterance) {
      if (!loud) {
        rememberAudio(held.leadIn, samples, rate, blockMs);
        return;
      }
      // What feedCapture keeps of a capture, with the conversation the
      // words are spoken to and how much of the audio is loud.
      held.utterance = {
        owner: currentOwner() || UNSCOPED,
        chunks: takeAudio(held.leadIn),
        sinceWakeMs: 0,
        elapsedMs: 0,
        speechStarted: true,
        trailingSilenceMs: 0,
        loudMs: 0,
      };
      renderDictation();
    }
    const said = held.utterance;
    if (loud) said.loudMs += blockMs;
    if (
      feedCapture(said, samples, rms, blockMs, rate) ||
      (said.elapsedMs >= DICTATION_BREATH_AFTER_MS &&
        said.trailingSilenceMs >= DICTATION_BREATH_MS)
    ) {
      endUtterance(held);
    }
  }

  /**
   * Turn the mic off.  What was being said is still sent for its
   * transcript, and a voice trigger that dictation paused is turned on
   * again.
   */
  function stopDictation() {
    const held = dictation;
    dictation = null;
    if (held.utterance) endUtterance(held);
    releaseAudio(held.processor, held.source, held.stream, held.context);
    renderDictation();
    if (wakePaused) {
      wakePaused = false;
      setEnabled(true);
    }
  }

  /**
   * The dictation `held` cannot listen (it got no microphone, or lost
   * it): turn the mic off and say why.  A mic toggled since is another
   * dictation's, or nobody's: there is nothing to do.
   */
  function failDictation(held, err) {
    if (dictation !== held) return;
    dictationError = describeVoiceError(err, 'dictation');
    notifyUser(DICTATION_NOTIFICATION_ID, 'error', dictationError);
    stopDictation();
  }

  /** Turn the mic on. */
  function startDictation() {
    if (enabled) {
      // The voice trigger gives the microphone up the way it does when
      // it is turned off (the capture under way and the transcripts
      // awaited are dropped), but stays on for the user: stopDictation
      // turns it on again.
      wakePaused = true;
      setEnabled(false);
    }
    const held = {
      stream: null,
      context: null,
      source: null,
      processor: null,
      leadIn: {entries: [], ms: 0, keepMs: DICTATION_LEAD_IN_MS},
      utterance: null,
    };
    dictation = held;
    dictationError = null;
    renderDictation();
    // The microphone is asked for within the click.  A browser that has
    // no microphone to ask for throws here, which fails this start the
    // way a refusal does.
    new Promise(resolve => resolve(requestMicrophone()))
      .then(stream => {
        if (dictation !== held) {
          // Turned off while the browser asked: the microphone it has
          // just granted is released.
          stopTracks(stream);
          return;
        }
        held.stream = stream;
        const tracks = stream.getAudioTracks();
        for (let i = 0; i < tracks.length; i++) {
          tracks[i].addEventListener('ended', () =>
            failDictation(held, MIC_LOST),
          );
        }
        const Ctx = window.AudioContext || window.webkitAudioContext;
        held.context = new Ctx();
        return resumeAudio(held.context);
      })
      .then(() => {
        // Turned off before the audio ran: stopDictation has released
        // what this start held.
        if (dictation !== held) return;
        held.source = held.context.createMediaStreamSource(held.stream);
        held.processor = held.context.createScriptProcessor(4096, 1, 1);
        held.processor.onaudioprocess = event => hearDictation(held, event);
        held.source.connect(held.processor);
        held.processor.connect(held.context.destination);
        notifyUser(DICTATION_NOTIFICATION_ID, null, null);
        renderDictation();
      })
      .catch(err => failDictation(held, err));
  }

  dictateBtn.addEventListener('click', () => {
    if (dictation) stopDictation();
    else startDictation();
  });

  btn.addEventListener('click', () => {
    if (dictation) {
      // The trigger takes the microphone over: dictation ends, which
      // turns on the trigger it had paused, or this one.
      wakePaused = true;
      stopDictation();
    } else {
      setEnabled(!enabled);
    }
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
      const dictated = takeDictationRound(msg.round);
      if (dictated) {
        // The words of a dictated utterance go into the text box of the
        // conversation they were spoken to, where the cursor is, and
        // are left there for the user.
        const spoken = msg.text.trim();
        if (spoken && composerTakes(dictated.owner)) insertAtCursor(spoken);
        return;
      }
      // dictation-coverage:end
      // tableak-coverage:start
      const owner = retireRound(msg.round);
      insertSpeech(
        msg.text,
        rounds.length > 0,
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

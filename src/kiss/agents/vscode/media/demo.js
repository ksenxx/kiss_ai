// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * KISS Sorcar Demo Mode
 *
 * Replays task history in a streaming fashion for demonstrations.
 * When demo mode is on and a user clicks a task in the history sidebar,
 * the clicked task's recorded events are replayed (legacy callers may
 * replay multiple sessions sequentially):
 *   1. Events are grouped into logical panels and each panel is loaded
 *      in 0.5s then collapsed before moving to the next
 *   2. The result panel streams word-by-word
 *
 * The replay PAUSES while speech is playing: replayed ``talk`` tool
 * calls — and ``prompt`` events (user messages), which are narrated
 * with a "User says " prefix — are awaited before the demo advances,
 * so the visuals never run ahead of the audio.  Cancelling the demo
 * resolves any pending speech promise (via ``api.stopSpeech``) so the
 * paused replay exits immediately.
 *
 * Communicates with main.js via window._demoApi (set by main.js).
 */
(function () {
  'use strict';

  let cancelRequested = false;

  // Generation counter for replay runs.  Each _startDemoReplay call and
  // each _cancelDemoReplay bumps it; a replay captures its generation
  // at start and treats any mismatch as a cancel.  This is what keeps a
  // cancelled replay suspended at an ``await`` from resuming as a
  // ZOMBIE after a newer replay reset the shared ``cancelRequested``
  // flag — the stale generation stops it at its next checkpoint, so a
  // finished/stopped demo can never interleave its panels into (or
  // tear the running-demo UI down under) the next demo.
  let replayGen = 0;

  // Resolver of the currently pending requestEvents promise, kept so a
  // cancel can resolve it: otherwise a replay cancelled while its
  // task_events request is in flight would stay suspended forever and
  // leave a stale ``api.resolveEvents`` behind to swallow the NEXT
  // demo's events.
  let pendingEventsResolve = null;

  // Pause/play state for the demo pause button: while paused, every
  // replay checkpoint awaits in pauseGate until resume (or cancel)
  // flushes the stored resolvers.
  let pauseRequested = false;
  let pauseResolvers = [];

  // Arguments of the most recent replay, kept so the ENDED-state play
  // button can restart the finished/stopped demo from the beginning
  // (see window._restartDemoReplay).  Cleared when the ended UI is
  // dismissed (demo mode off, tab switch) via window._clearDemoReplay.
  let lastReplayArgs = null;

  /**
   * Notify main.js helpers waiting to start/resume demo speech that the
   * pause state changed.  A tiny DOM event keeps the demo.js-owned pause
   * state decoupled from the talk-queue implementation in main.js.
   */
  function notifyPauseChanged() {
    try {
      let ev;
      if (typeof window.CustomEvent === 'function') {
        ev = new window.CustomEvent('kiss-demo-pause-change', {
          detail: {paused: pauseRequested, cancelled: cancelRequested},
        });
      } else {
        ev = new window.Event('kiss-demo-pause-change');
        ev.detail = {paused: pauseRequested, cancelled: cancelRequested};
      }
      window.dispatchEvent(ev);
    } catch (_e) {
      // Pause/resume still works through pauseResolvers; this event only
      // gates speech clips that have not started yet.
    }
  }

  /**
   * Pause or resume the demo replay (wired to the pause/play button
   * in main.js).  Pausing also pauses the currently playing speech;
   * resuming flushes every checkpoint waiting in ``pauseGate`` and
   * resumes the speech.
   *
   * @param {boolean} paused - True to pause, false to resume.
   */
  window._setDemoPaused = function (paused) {
    pauseRequested = !!paused;
    const api = getApi();
    if (pauseRequested) {
      if (api && typeof api.pauseSpeech === 'function') api.pauseSpeech();
    } else {
      const resolvers = pauseResolvers;
      pauseResolvers = [];
      for (let i = 0; i < resolvers.length; i++) resolvers[i]();
      if (api && typeof api.resumeSpeech === 'function') api.resumeSpeech();
    }
    notifyPauseChanged();
  };

  /**
   * Check whether the demo replay is currently paused.
   *
   * @returns {boolean} - True while the pause button is engaged.
   */
  window._isDemoPaused = function () {
    return pauseRequested;
  };

  /**
   * True when the replay run that captured *gen* must stop: either the
   * user cancelled, or a newer replay run superseded it (generation
   * mismatch).
   *
   * @param {number} gen - Generation captured by the replay run.
   * @returns {boolean} - Whether the run is cancelled or stale.
   */
  function replayStopped(gen) {
    return cancelRequested || gen !== replayGen;
  }

  /**
   * Replay checkpoint: block while the demo is paused.  Resuming (or
   * cancelling, which flushes the resolvers too) releases the wait,
   * so a paused replay can never hang forever.
   *
   * @param {number} gen - Generation of the calling replay run; a
   *     stale run passes straight through so it can exit.
   */
  async function pauseGate(gen) {
    while (pauseRequested && !replayStopped(gen)) {
      await new Promise(resolve => {
        pauseResolvers.push(resolve);
      });
    }
  }

  // Monotonic sequence for demo sub-agent tab ids — guarantees unique
  // ids even when two fan-outs replay within the same millisecond.
  let demoSubTabSeq = 0;

  /** Sanitize markdown HTML before innerHTML — see kissSanitize in main.js. */
  function kissSanitize(html) {
    const t = document.createElement('template');
    t.innerHTML = String(html == null ? '' : html);
    const BAD_TAGS = new Set([
      'SCRIPT',
      'IFRAME',
      'OBJECT',
      'EMBED',
      'FORM',
      'META',
      'LINK',
      'STYLE',
      'BASE',
      'FRAME',
      'FRAMESET',
    ]);
    const URL_ATTRS = new Set([
      'href',
      'src',
      'action',
      'formaction',
      'xlink:href',
    ]);
    for (const el of Array.from(t.content.querySelectorAll('*'))) {
      if (BAD_TAGS.has(el.tagName)) {
        el.remove();
        continue;
      }
      for (const attr of Array.from(el.attributes)) {
        const name = attr.name.toLowerCase();
        if (name.startsWith('on')) {
          el.removeAttribute(attr.name);
          continue;
        }
        if (
          URL_ATTRS.has(name) &&
          /^(javascript|data|vbscript):/i.test((attr.value || '').trim())
        ) {
          el.removeAttribute(attr.name);
        }
      }
    }
    return t.innerHTML;
  }

  function sleep(ms) {
    return new Promise(resolve => {
      setTimeout(resolve, ms);
    });
  }

  /**
   * Wait for the demo API to be available (main.js sets it after init).
   * Returns the API object.
   */
  function getApi() {
    return window._demoApi;
  }

  /**
   * Request task events for a history session from the backend.
   * ``taskId`` pins the request to the session row's OWN task —
   * without it the backend loads the LATEST task of the chat, so a
   * multi-task chat would replay the same (newest) events for every
   * task.  Returns a promise that resolves with the events array when
   * the task_events message arrives (main.js routes it here in demo
   * mode).
   */
  function requestEvents(api, session) {
    const reqTabId = api.getActiveTabId();
    const reqTaskId = session.task_id;
    return new Promise(resolve => {
      const deliver = function (events, envelope) {
        // Identity check: main.js passes the full task_events message
        // as *envelope* — a reply addressed to a DIFFERENT tab or task
        // (a stopped demo's late reply arriving on the same tab, a
        // legacy same-tab restart) must not settle this fetch with the
        // wrong task's events.  Keep waiting for the right reply; a
        // cancel still frees the fetch via discardPendingEvents.
        // Legacy callers pass no envelope — always accepted.
        if (envelope && !eventsReplyMatches(envelope, reqTabId, reqTaskId)) {
          return;
        }
        // Ownership check: a STALE deliver hook (captured by a caller
        // before this fetch was discarded) may fire late — it must
        // only settle its own abandoned promise, never clear the hooks
        // now owned by a NEWER fetch.
        if (api.resolveEvents === deliver) api.resolveEvents = null;
        if (pendingEventsResolve === resolve) pendingEventsResolve = null;
        resolve(events);
      };
      pendingEventsResolve = resolve;
      api.resolveEvents = deliver;
      api.sendMessage({
        type: 'resumeSession',
        id: session.id,
        taskId: reqTaskId,
        tabId: reqTabId,
      });
    });
  }

  /**
   * True when a ``task_events`` reply *envelope* answers the fetch
   * that requested *reqTabId* / *reqTaskId*.  Fields absent from the
   * envelope or from the request (legacy histories, older backends)
   * are not compared — only a POSITIVE mismatch rejects the reply.
   *
   * @param {Object} envelope - The task_events message from main.js.
   * @param {string} reqTabId - Tab id the fetch was requested for.
   * @param {*} reqTaskId - task_id of the requested history session.
   * @returns {boolean} - Whether the reply belongs to this fetch.
   */
  function eventsReplyMatches(envelope, reqTabId, reqTaskId) {
    const evTab = envelope.tabId;
    if (
      evTab !== undefined &&
      evTab !== null &&
      reqTabId &&
      String(evTab) !== String(reqTabId)
    ) {
      return false;
    }
    const evTask = envelope.task_id;
    if (
      evTask !== undefined &&
      evTask !== null &&
      reqTaskId !== undefined &&
      reqTaskId !== null &&
      reqTaskId !== '' &&
      String(evTask) !== String(reqTaskId)
    ) {
      return false;
    }
    return true;
  }

  /**
   * Discard a pending requestEvents fetch: clear the ``resolveEvents``
   * hook (so a late task_events reply can never be mistaken for the
   * NEXT demo's events) and resolve the suspended promise with no
   * events so the cancelled replay run wakes up, observes its stale
   * generation, and exits instead of leaking forever.
   *
   * @param {?Object} api - The demo API from main.js (may be absent).
   */
  function discardPendingEvents(api) {
    if (api) api.resolveEvents = null;
    const resolve = pendingEventsResolve;
    pendingEventsResolve = null;
    if (resolve) resolve([]);
  }

  /**
   * Format a number with thousand separators.
   */
  function fmtN(n) {
    return Number(n).toLocaleString('en-US');
  }

  /**
   * Stream the result panel content word-by-word.
   *
   * @param {Object} api - The demo API from main.js.
   * @param {Object} ev - The recorded ``result`` event.
   * @param {number} gen - Generation of the calling replay run;
   *     streaming stops as soon as the run is cancelled or stale.
   */
  async function streamResultEvent(api, ev, gen) {
    const O = document.getElementById('output');
    if (!O) return;

    const rc = document.createElement('div');
    rc.className = 'ev rc';

    // Header
    const header = document.createElement('div');
    header.className = 'rc-h';
    const h3 = document.createElement('h3');
    h3.textContent = 'Result';
    header.appendChild(h3);

    const rs = document.createElement('div');
    rs.className = 'rs';
    const tokSpan = document.createElement('span');
    tokSpan.innerHTML = 'Tokens <b>' + fmtN(ev.total_tokens || 0) + '</b>';
    rs.appendChild(tokSpan);
    const costSpan = document.createElement('span');
    costSpan.innerHTML = 'Cost <b>' + esc(ev.cost || 'N/A') + '</b>';
    rs.appendChild(costSpan);
    header.appendChild(rs);
    rc.appendChild(header);

    // Status banner — mirrors handleOutputEvent's 'result' case in
    // main.js: ``is_continue`` (agent paused to continue in a new
    // session) takes precedence over the failure banner, because the
    // backend emits continue-results with ``success: false``.
    if (ev.is_continue) {
      const contDiv = document.createElement('div');
      contDiv.style.cssText =
        'color:var(--yellow);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px';
      contDiv.textContent = 'Status: Continue';
      rc.appendChild(contDiv);
    } else if (ev.success === false) {
      const failDiv = document.createElement('div');
      failDiv.style.cssText =
        'color:var(--red);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px';
      failDiv.textContent = 'Status: FAILED';
      rc.appendChild(failDiv);
    }

    // Body
    const body = document.createElement('div');
    body.className = 'rc-body md-body';
    rc.appendChild(body);
    O.appendChild(rc);
    api.scrollToBottom();

    // Stream content word by word
    const text = (ev.summary || ev.text || '(no result)')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    const words = text.split(/(\s+)/);
    let accumulated = '';
    const WORDS_PER_TICK = 3;
    const TICK_MS = 10;

    for (let i = 0; i < words.length; i++) {
      if (replayStopped(gen)) break;
      accumulated += words[i];
      if (i % WORDS_PER_TICK === WORDS_PER_TICK - 1 || i === words.length - 1) {
        if (typeof marked !== 'undefined') {
          body.innerHTML = kissSanitize(marked.parse(accumulated));
        } else {
          body.textContent = accumulated;
        }
        api.scrollToBottom();
        await sleep(TICK_MS);
        await pauseGate(gen);
      }
    }

    // Highlight code blocks
    if (typeof hljs !== 'undefined') {
      body.querySelectorAll('pre code').forEach(bl => {
        hljs.highlightElement(bl);
      });
    }
  }

  /**
   * Escape HTML entities.
   */
  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  /** Lifecycle event types to skip during replay. */
  const SKIP_TYPES = {
    task_done: 1,
    task_error: 1,
    task_stopped: 1,
    task_interrupted: 1,
    followup_suggestion: 1,
  };

  /**
   * Group a flat list of events into logical panel groups.
   *
   * Each group corresponds to one visual panel in the output:
   *   - LLM panel: starts at thinking_start/text_delta after a tool_result
   *     (or at step 0), includes all thinking/text events until the next
   *     tool_call or result.
   *   - Tool call panel: starts at tool_call, includes system_output and
   *     tool_result events.
   *   - Prompt panel: a single prompt event (a mid-task user message);
   *     it is narrated aloud during replay with a "User says " prefix.
   *   - Result panel: a single result event.
   *
   * @param {Array} events - Flat list of task events.
   * @returns {Array<Array>} - Array of event groups.
   */
  function groupEventsIntoPanels(events) {
    const panels = [];
    let current = [];
    let afterToolResult = true; // start true so first thought gets a panel

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const t = ev.type;

      if (SKIP_TYPES[t]) continue;

      // tool_call starts a new tool-call panel group
      if (t === 'tool_call') {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = false;
        continue;
      }

      // thinking_start or text_delta after a tool_result starts a new llm-panel
      if ((t === 'thinking_start' || t === 'text_delta') && afterToolResult) {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = false;
        continue;
      }

      // result is always its own group
      if (t === 'result') {
        if (current.length > 0) panels.push(current);
        panels.push([ev]);
        current = [];
        afterToolResult = false;
        continue;
      }

      // prompt (a mid-task user message) is always its own group so it
      // can be shown and narrated as one panel; the following
      // thinking/text starts a fresh LLM panel.
      if (t === 'prompt') {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = true;
        continue;
      }

      // tool_result marks the end of a tool-call group
      if (t === 'tool_result') {
        current.push(ev);
        afterToolResult = true;
        continue;
      }

      // Everything else (deltas, system_output, usage_info) stays in current group
      current.push(ev);
    }

    if (current.length > 0) panels.push(current);
    return panels;
  }

  // Expose for testing
  window._groupEventsIntoPanels = groupEventsIntoPanels;

  /**
   * Parse the ``tasks`` argument of a recorded ``run_parallel`` tool
   * call.  The agent passes a JSON-encoded list of task description
   * strings; anything unparseable (truncated history, non-list JSON)
   * yields an empty list so the replay degrades gracefully.
   *
   * @param {*} raw - Raw ``extras.tasks`` value from the event.
   * @returns {Array<string>} - Task description strings.
   */
  function parseDemoTasks(raw) {
    if (Array.isArray(raw)) return raw.map(String);
    if (typeof raw !== 'string' || !raw) return [];
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (_e) {
      return [];
    }
  }

  // Expose for testing
  window._parseDemoTasks = parseDemoTasks;

  /**
   * Select which history sessions the demo replays.
   *
   * The demo-mode spec: clicking a task in the history replays ONLY
   * that task — never the chat's other tasks (the old behavior of
   * replaying the whole chain oldest-to-newest) and never other
   * chats' tasks ("random tasks").  This holds for regular rows and
   * sub-agent rows alike (a sub-agent task's own fan-outs are still
   * replayed inside it via ``executeDemoToolCall``).  Rows without a
   * usable ``task_id`` (legacy histories: undefined/null/'') fall
   * back to the clicked chat's top-level tasks, oldest first.
   * Without *clicked* (tests, legacy callers) every session with
   * stored events is replayed, oldest first.
   *
   * @param {Array} sessions - All history sessions (newest first).
   * @param {?Object} clicked - The history row the user clicked.
   * @returns {Array} - The sessions to replay.
   */
  function selectReplaySessions(sessions, clicked) {
    let items = sessions.filter(s => {
      return s.has_events && s.id;
    });
    if (clicked && clicked.id) {
      const clickedTaskId = clicked.task_id;
      if (
        clickedTaskId !== undefined &&
        clickedTaskId !== null &&
        clickedTaskId !== ''
      ) {
        // The clicked chat id is matched too so a legacy row of
        // ANOTHER chat whose task_id happens to coincide can never
        // sneak into the replay.
        items = items.filter(s => {
          return (
            String(s.id) === String(clicked.id) &&
            String(s.task_id) === String(clickedTaskId)
          );
        });
      } else {
        items = items.filter(s => {
          return String(s.id) === String(clicked.id) && !s.parent_task_id;
        });
      }
    }
    return items.slice().reverse();
  }

  // Expose for testing
  window._selectReplaySessions = selectReplaySessions;

  /**
   * True when a panel group contains a ``run_parallel`` tool call —
   * its replay opened sub-agent tabs that stay visible only until the
   * panel collapses, so the group earns a longer show pause.
   *
   * @param {Array} group - One panel group of task events.
   * @returns {boolean} - Whether the group is a fan-out panel.
   */
  function groupHasFanOut(group) {
    for (let i = 0; i < group.length; i++) {
      if (group[i].type === 'tool_call' && group[i].name === 'run_parallel') {
        return true;
      }
    }
    return false;
  }

  /**
   * Actually execute a replayed ``talk`` or ``run_parallel`` tool call
   * (or narrate a mid-task user ``prompt``) so the demo behaves like a
   * live run:
   *   - ``talk``: speak the recorded text aloud through the talk queue;
   *   - ``prompt``: read the user's mid-task message aloud through the
   *     same talk queue, prefixed with "User says ";
   *   - ``run_parallel``: materialise one sub-agent tab per task via
   *     the real ``openSubagentTab`` path (the tabs close again when
   *     the fan-out's panel collapses, exactly like a live fan-out).
   *
   * Must run AFTER ``api.processEvent(ev)`` rendered the tool-call
   * panel: the sub-agent tab machinery locates the fan-out's
   * ``.tc-run-parallel`` panel in the chat DOM.
   *
   * @param {Object} api - The demo API from main.js.
   * @param {Object} ev - A replayed task event.
   * @returns {?Promise} - A promise that resolves when the replayed
   *     ``talk`` speech or ``prompt`` narration finishes (so the caller
   *     can PAUSE the demo until the talking ends), or null when
   *     nothing awaitable ran.
   */
  function executeDemoToolCall(api, ev) {
    if (!ev) return null;
    if (ev.type === 'prompt' && typeof api.playTalkEvent === 'function') {
      const promptText = ev.text || '';
      if (promptText) {
        return api.playTalkEvent({text: 'User says ' + promptText});
      }
      return null;
    }
    if (ev.type !== 'tool_call') return null;
    const extras = ev.extras || {};
    if (ev.name === 'talk' && typeof api.playTalkEvent === 'function') {
      const talkText = extras.text || '';
      if (talkText) {
        return api.playTalkEvent({
          text: talkText,
          language: extras.language,
          emotion: extras.emotion,
          audioB64: extras.audioB64,
          audioMime: extras.audioMime,
        });
      }
      return null;
    }
    if (
      ev.name === 'run_parallel' &&
      typeof api.openSubagentTab === 'function'
    ) {
      const tasks = parseDemoTasks(extras.tasks);
      const parentId = api.getActiveTabId();
      for (let i = 0; i < tasks.length; i++) {
        api.openSubagentTab({
          type: 'openSubagentTab',
          tab_id: 'demo-sub-' + ++demoSubTabSeq + '-' + i,
          parent_tab_id: parentId,
          description: tasks[i],
          taskIndex: i,
          isDone: false,
        });
      }
    }
    return null;
  }

  /**
   * Replay *items* (history sessions, oldest first) panel-by-panel:
   * fetch each session's recorded events, group them into panels, and
   * render them on the demo cadence.  Extracted from
   * ``_startDemoReplay`` so the caller can guard it with try/catch —
   * even a throwing host hook must not leave the demo marked active.
   *
   * @param {Object} api - The demo API from main.js.
   * @param {Array} items - The sessions to replay, oldest first.
   * @param {number} myGen - Generation captured by the calling run.
   */
  async function replaySessions(api, items, myGen) {
    for (let i = 0; i < items.length; i++) {
      if (replayStopped(myGen)) break;
      const session = items[i];
      const taskText = session.preview || session.title || 'Untitled';

      // Step 1: Prepare output area
      if (i === 0) {
        api.clearForReplay();
      } else {
        // Continuation: keep existing output, just reset panel state
        api.resetOutputState();
      }
      api.setTaskText(taskText);
      api.updateTabTitle(taskText);

      // Step 2: Request events from backend
      const events = await requestEvents(api, session);
      await pauseGate(myGen);
      if (replayStopped(myGen)) break;

      // Step 3: Group events into panels and replay panel-by-panel
      const panelGroups = groupEventsIntoPanels(events);

      for (let j = 0; j < panelGroups.length; j++) {
        if (replayStopped(myGen)) break;
        await pauseGate(myGen);
        if (replayStopped(myGen)) break;
        const group = panelGroups[j];

        // Check if this group is a result panel
        if (group.length === 1 && group[0].type === 'result') {
          await streamResultEvent(api, group[0], myGen);
          continue;
        }

        // Process all events in this panel group at once.  Replayed
        // ``talk`` / ``run_parallel`` tool calls are ACTUALLY executed
        // (speech playback / sub-agent tab creation) right after their
        // panel is rendered.  When a ``talk`` is played the replay
        // PAUSES until the speech ends (cancel resolves the promise
        // via api.stopSpeech, so this can never hang forever).
        for (let k = 0; k < group.length; k++) {
          api.processEvent(group[k]);
          const speech = executeDemoToolCall(api, group[k]);
          if (speech && typeof speech.then === 'function') {
            api.scrollToBottom();
            await speech;
            await pauseGate(myGen);
            if (replayStopped(myGen)) break;
          }
        }
        if (replayStopped(myGen)) break;
        api.scrollToBottom();

        // Brief pause to show the panel, then collapse it.  A fan-out
        // group pauses much longer: its sub-agent tabs (opened by
        // executeDemoToolCall) close again when the panel collapses,
        // so this pause is the only window in which the viewer can
        // actually SEE the tabs materialise.
        await sleep(groupHasFanOut(group) ? 2500 : 500);
        await pauseGate(myGen);
        if (!replayStopped(myGen)) {
          api.collapsePanels();
          api.scrollToBottom();
        }
      }

      // Brief pause between tasks
      if (i < items.length - 1) {
        await sleep(1000);
        await pauseGate(myGen);
      }
    }
  }

  /**
   * Start the demo replay for the clicked history task.
   * Called from main.js when a history item is clicked in demo mode.
   *
   * @param {Array} sessions - All history sessions (newest first from server).
   * @param {?Object} clicked - The history session row the user
   *     clicked; only that task is replayed — never the chat's other
   *     tasks.  Omitted by legacy callers/tests to replay every
   *     session.
   */
  window._startDemoReplay = async function (sessions, clicked) {
    const api = getApi();
    if (!api || api.active) return;
    api.active = true;
    cancelRequested = false;
    // Remember what is being replayed so the ended-state play button
    // can restart this exact demo (window._restartDemoReplay).
    lastReplayArgs = {sessions: sessions, clicked: clicked};
    const myGen = ++replayGen;
    // A replay stopped while its event fetch was in flight may have
    // left a stale resolver behind; discard it so this run's fetch can
    // never be resolved by the PREVIOUS run's late task_events reply.
    discardPendingEvents(api);

    try {
      // Show stop button and spinner for the duration of the replay,
      // hide the input controls, and show the pause/play button reset
      // to its "pause" state.  Inside the try so even a throwing
      // setup hook cannot leave the demo stuck active.
      api.setRunningState(true);
      api.showSpinner();
      pauseRequested = false;
      if (typeof api.setDemoUi === 'function') api.setDemoUi(true);

      const items = selectReplaySessions(sessions, clicked);
      await replaySessions(api, items, myGen);
    } catch (_e) {
      // A throwing host hook (processEvent / sendMessage / a rejected
      // speech promise) must not leave the demo stuck "active": that
      // would block every later demo (_startDemoReplay early-returns
      // while api.active) with the spinner and demo UI up forever.
      // Swallow and fall through to the teardown below — the replay
      // is started fire-and-forget from the history-row click handler,
      // so rethrowing would only surface as an unhandled rejection.
    }

    // A newer replay owns the demo UI now: this run was cancelled (the
    // cancel already restored the controls) and superseded while it
    // was suspended at an await.  Leave every flag and control alone —
    // resetting them here would tear the running-demo state down under
    // the CURRENT replay (spinner gone, _demoActive off, its
    // task_events falling into the instant-replay path).
    if (myGen !== replayGen) return;

    // An error may have escaped between installing the event resolver
    // and its resolution — never leave a dangling fetch behind on the
    // way out.
    discardPendingEvents(api);
    api.setRunningState(false);
    api.removeSpinner();
    pauseRequested = false;
    // The demo ENDED: keep the input controls hidden and show ONLY
    // the play button, which restarts this demo from the beginning.
    if (typeof api.setDemoUi === 'function') api.setDemoUi('ended');
    api.active = false;
  };

  /**
   * Restart the most recently replayed demo from the beginning (the
   * ended-state play button in main.js).  Replays in the CURRENT tab
   * — the finished demo's own tab — re-fetching the recorded events.
   *
   * @returns {boolean} - True when a restart was started.
   */
  window._restartDemoReplay = function () {
    const args = lastReplayArgs;
    const api = getApi();
    if (!args || !api || api.active) return false;
    // Fire-and-forget like the history-row click handler.
    window._startDemoReplay(args.sessions, args.clicked);
    return true;
  };

  /**
   * Dismiss the ended-state play button and forget the replayed demo
   * (demo mode turned off, or the user navigated to another tab).
   * Never touches a RUNNING replay's UI.
   */
  window._clearDemoReplay = function () {
    lastReplayArgs = null;
    const api = getApi();
    if (api && !api.active && typeof api.setDemoUi === 'function') {
      api.setDemoUi(false);
    }
  };

  /**
   * Cancel an in-progress demo replay.
   *
   * @param {?Object} opts - ``{restoreUi: true}`` restores the normal
   *     input controls (demo mode turned off mid-replay) instead of
   *     the default ENDED state, which shows only the play button
   *     that restarts the stopped demo.
   */
  window._cancelDemoReplay = function (opts) {
    const restoreUi = !!(opts && opts.restoreUi);
    cancelRequested = true;
    // Invalidate the running replay's generation: even if a NEW replay
    // resets ``cancelRequested`` before the cancelled run reaches its
    // next checkpoint, the stale generation still stops it — no zombie
    // replay can interleave its panels into the next demo's output.
    replayGen++;
    // Release a PAUSED replay so it observes the cancel and exits
    // instead of hanging in pauseGate forever.
    pauseRequested = false;
    const resolvers = pauseResolvers;
    pauseResolvers = [];
    for (let i = 0; i < resolvers.length; i++) resolvers[i]();
    notifyPauseChanged();
    const api = getApi();
    // Wake a replay suspended on an in-flight event fetch and clear
    // the stale resolveEvents hook it would otherwise leave behind.
    discardPendingEvents(api);
    if (restoreUi) lastReplayArgs = null;
    if (api) {
      api.active = false;
      api.setRunningState(false);
      api.removeSpinner();
      // A stop leaves the ENDED play-button UI up (press play to
      // restart the stopped demo); restoreUi brings the normal input
      // controls back instead.
      if (typeof api.setDemoUi === 'function') {
        api.setDemoUi(restoreUi || !lastReplayArgs ? false : 'ended');
      }
      // Stop any queued/in-flight demo speech immediately.
      if (typeof api.stopSpeech === 'function') api.stopSpeech();
    }
  };

  /**
   * Check whether a demo replay is currently running.
   */
  window._isDemoActive = function () {
    const api = getApi();
    return api ? api.active : false;
  };
})();

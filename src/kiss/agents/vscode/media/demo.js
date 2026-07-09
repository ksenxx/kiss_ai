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

  // Pause/play state for the demo pause button: while paused, every
  // replay checkpoint awaits in pauseGate until resume (or cancel)
  // flushes the stored resolvers.
  let pauseRequested = false;
  let pauseResolvers = [];

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
   * Replay checkpoint: block while the demo is paused.  Resuming (or
   * cancelling, which flushes the resolvers too) releases the wait,
   * so a paused replay can never hang forever.
   */
  async function pauseGate() {
    while (pauseRequested && !cancelRequested) {
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
    return new Promise(resolve => {
      api.resolveEvents = function (events) {
        api.resolveEvents = null;
        resolve(events);
      };
      api.sendMessage({
        type: 'resumeSession',
        id: session.id,
        taskId: session.task_id,
        tabId: api.getActiveTabId(),
      });
    });
  }

  /**
   * Format a number with thousand separators.
   */
  function fmtN(n) {
    return Number(n).toLocaleString('en-US');
  }

  /**
   * Stream the result panel content word-by-word.
   */
  async function streamResultEvent(api, ev) {
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
    const TICK_MS = 50;

    for (let i = 0; i < words.length; i++) {
      if (cancelRequested) break;
      accumulated += words[i];
      if (i % WORDS_PER_TICK === WORDS_PER_TICK - 1 || i === words.length - 1) {
        if (typeof marked !== 'undefined') {
          body.innerHTML = kissSanitize(marked.parse(accumulated));
        } else {
          body.textContent = accumulated;
        }
        api.scrollToBottom();
        await sleep(TICK_MS);
        await pauseGate();
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

    // Show stop button and spinner for the duration of the replay,
    // hide the input controls, and show the pause/play button reset
    // to its "pause" state.
    api.setRunningState(true);
    api.showSpinner();
    pauseRequested = false;
    if (typeof api.setDemoUi === 'function') api.setDemoUi(true);

    const items = selectReplaySessions(sessions, clicked);

    for (let i = 0; i < items.length; i++) {
      if (cancelRequested) break;
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
      await pauseGate();
      if (cancelRequested) break;

      // Step 3: Group events into panels and replay panel-by-panel
      const panelGroups = groupEventsIntoPanels(events);

      for (let j = 0; j < panelGroups.length; j++) {
        if (cancelRequested) break;
        await pauseGate();
        const group = panelGroups[j];

        // Check if this group is a result panel
        if (group.length === 1 && group[0].type === 'result') {
          await streamResultEvent(api, group[0]);
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
            await pauseGate();
            if (cancelRequested) break;
          }
        }
        if (cancelRequested) break;
        api.scrollToBottom();

        // Brief pause to show the panel, then collapse it.  A fan-out
        // group pauses much longer: its sub-agent tabs (opened by
        // executeDemoToolCall) close again when the panel collapses,
        // so this pause is the only window in which the viewer can
        // actually SEE the tabs materialise.
        await sleep(groupHasFanOut(group) ? 2500 : 500);
        await pauseGate();
        if (!cancelRequested) {
          api.collapsePanels();
          api.scrollToBottom();
        }
      }

      // Brief pause between tasks
      if (i < items.length - 1) {
        await sleep(1000);
        await pauseGate();
      }
    }

    api.setRunningState(false);
    api.removeSpinner();
    pauseRequested = false;
    if (typeof api.setDemoUi === 'function') api.setDemoUi(false);
    api.active = false;
  };

  /**
   * Cancel an in-progress demo replay.
   */
  window._cancelDemoReplay = function () {
    cancelRequested = true;
    // Release a PAUSED replay so it observes the cancel and exits
    // instead of hanging in pauseGate forever.
    pauseRequested = false;
    const resolvers = pauseResolvers;
    pauseResolvers = [];
    for (let i = 0; i < resolvers.length; i++) resolvers[i]();
    notifyPauseChanged();
    const api = getApi();
    if (api) {
      api.active = false;
      api.setRunningState(false);
      api.removeSpinner();
      if (typeof api.setDemoUi === 'function') api.setDemoUi(false);
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

/**
 * KISS Sorcar Demo Mode
 *
 * Replays task history in a streaming fashion for demonstrations.
 * When demo mode is on and a user clicks a task in the history sidebar,
 * all tasks in the history are replayed sequentially:
 *   1. Task text appears in the input box (2-second pause)
 *   2. Events are grouped into logical panels and each panel is loaded
 *      in 0.5s then collapsed before moving to the next
 *   3. The result panel streams word-by-word
 *
 * Communicates with main.js via window._demoApi (set by main.js).
 */
(function () {
  'use strict';

  let cancelRequested = false;

  /** Sanitize markdown HTML before innerHTML — see kissSanitize in main.js. */
  function kissSanitize(html) {
    const t = document.createElement('template');
    t.innerHTML = String(html == null ? '' : html);
    const BAD_TAGS = new Set([
      'SCRIPT', 'IFRAME', 'OBJECT', 'EMBED', 'FORM', 'META', 'LINK',
      'STYLE', 'BASE', 'FRAME', 'FRAMESET',
    ]);
    const URL_ATTRS = new Set(['href', 'src', 'action', 'formaction',
                               'xlink:href']);
    for (const el of Array.from(t.content.querySelectorAll('*'))) {
      if (BAD_TAGS.has(el.tagName)) { el.remove(); continue; }
      for (const attr of Array.from(el.attributes)) {
        const name = attr.name.toLowerCase();
        if (name.startsWith('on')) {
          el.removeAttribute(attr.name);
          continue;
        }
        if (URL_ATTRS.has(name) &&
            /^(javascript|data|vbscript):/i.test((attr.value || '').trim())) {
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
   * Request task events for a session from the backend.
   * Returns a promise that resolves with the events array when the
   * task_events message arrives (main.js routes it here in demo mode).
   */
  function requestEvents(api, sessionId) {
    return new Promise(resolve => {
      api.resolveEvents = function (events) {
        api.resolveEvents = null;
        resolve(events);
      };
      api.sendMessage({
        type: 'resumeSession',
        id: sessionId,
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

    // Failure banner
    if (ev.success === false) {
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
   * Start the demo replay for all sessions in the history.
   * Called from main.js when a history item is clicked in demo mode.
   *
   * @param {Array} sessions - All history sessions (newest first from server).
   */
  window._startDemoReplay = async function (sessions) {
    const api = getApi();
    if (!api || api.active) return;
    api.active = true;
    cancelRequested = false;

    // Show stop button and spinner for the duration of the replay
    api.setRunningState(true);
    api.showSpinner();

    // Filter sessions that have stored events, reverse to oldest-first
    const items = sessions
      .filter(s => {
        return s.has_events && s.id;
      })
      .slice()
      .reverse();

    for (let i = 0; i < items.length; i++) {
      if (cancelRequested) break;
      const session = items[i];
      const taskText = session.preview || session.title || 'Untitled';

      // Hide welcome immediately so it's never visible between tasks
      api.hideWelcome();

      // Step 1: Show task text in the input box
      api.setInput(taskText);
      await sleep(2000);
      if (cancelRequested) break;

      // Step 2: Clear input and prepare output area
      api.clearInput();
      if (i === 0) {
        api.clearForReplay();
      } else {
        // Continuation: keep existing output, just reset panel state
        api.resetOutputState();
      }
      api.setTaskText(taskText);
      api.updateTabTitle(taskText);

      // Step 3: Request events from backend
      const events = await requestEvents(api, session.id);
      if (cancelRequested) break;

      // Step 4: Group events into panels and replay panel-by-panel
      const panelGroups = groupEventsIntoPanels(events);

      for (let j = 0; j < panelGroups.length; j++) {
        if (cancelRequested) break;
        const group = panelGroups[j];

        // Check if this group is a result panel
        if (group.length === 1 && group[0].type === 'result') {
          await streamResultEvent(api, group[0]);
          continue;
        }

        // Process all events in this panel group at once
        for (let k = 0; k < group.length; k++) {
          api.processEvent(group[k]);
        }
        api.scrollToBottom();

        // Brief pause to show the panel, then collapse it
        await sleep(500);
        if (!cancelRequested) {
          api.collapsePanels();
          api.scrollToBottom();
        }
      }

      // Brief pause between tasks
      if (i < items.length - 1) {
        await sleep(1000);
      }
    }

    api.setRunningState(false);
    api.removeSpinner();
    api.active = false;
  };

  /**
   * Cancel an in-progress demo replay.
   */
  window._cancelDemoReplay = function () {
    cancelRequested = true;
    const api = getApi();
    if (api) {
      api.active = false;
      api.setRunningState(false);
      api.removeSpinner();
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

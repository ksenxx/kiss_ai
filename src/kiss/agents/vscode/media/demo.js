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

  function sleep(ms) {
    return new Promise(function (resolve) {
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
    return new Promise(function (resolve) {
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
    var O = document.getElementById('output');
    if (!O) return;

    var rc = document.createElement('div');
    rc.className = 'ev rc';

    // Header
    var header = document.createElement('div');
    header.className = 'rc-h';
    var h3 = document.createElement('h3');
    h3.textContent = 'Result';
    header.appendChild(h3);

    var rs = document.createElement('div');
    rs.className = 'rs';
    var tokSpan = document.createElement('span');
    tokSpan.innerHTML = 'Tokens <b>' + fmtN(ev.total_tokens || 0) + '</b>';
    rs.appendChild(tokSpan);
    var costSpan = document.createElement('span');
    costSpan.innerHTML = 'Cost <b>' + esc(ev.cost || 'N/A') + '</b>';
    rs.appendChild(costSpan);
    header.appendChild(rs);
    rc.appendChild(header);

    // Failure banner
    if (ev.success === false) {
      var failDiv = document.createElement('div');
      failDiv.style.cssText =
        'color:var(--red);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px';
      failDiv.textContent = 'Status: FAILED';
      rc.appendChild(failDiv);
    }

    // Body
    var body = document.createElement('div');
    body.className = 'rc-body md-body';
    rc.appendChild(body);
    O.appendChild(rc);
    api.scrollToBottom();

    // Stream content word by word
    var text = (ev.summary || ev.text || '(no result)')
      .replace(/\n{3,}/g, '\n\n')
      .trim();
    var words = text.split(/(\s+)/);
    var accumulated = '';
    var WORDS_PER_TICK = 3;
    var TICK_MS = 50;

    for (var i = 0; i < words.length; i++) {
      if (cancelRequested) break;
      accumulated += words[i];
      if (i % WORDS_PER_TICK === WORDS_PER_TICK - 1 || i === words.length - 1) {
        if (typeof marked !== 'undefined') {
          body.innerHTML = marked.parse(accumulated);
        } else {
          body.textContent = accumulated;
        }
        api.scrollToBottom();
        await sleep(TICK_MS);
      }
    }

    // Highlight code blocks
    if (typeof hljs !== 'undefined') {
      body.querySelectorAll('pre code').forEach(function (bl) {
        hljs.highlightElement(bl);
      });
    }
  }

  /**
   * Escape HTML entities.
   */
  function esc(t) {
    var d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  /** Lifecycle event types to skip during replay. */
  var SKIP_TYPES = {
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
    var panels = [];
    var current = [];
    var afterToolResult = true; // start true so first thought gets a panel

    for (var i = 0; i < events.length; i++) {
      var ev = events[i];
      var t = ev.type;

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
    var api = getApi();
    if (!api || api.active) return;
    api.active = true;
    cancelRequested = false;

    // Filter sessions that have stored events, reverse to oldest-first
    var items = sessions
      .filter(function (s) {
        return s.has_events && s.id;
      })
      .slice()
      .reverse();

    for (var i = 0; i < items.length; i++) {
      if (cancelRequested) break;
      var session = items[i];
      var taskText = session.preview || session.title || 'Untitled';

      // Create a new tab for each task after the first
      if (i > 0) api.createNewTab();

      // Step 1: Show task text in the input box
      api.setInput(taskText);
      await sleep(2000);
      if (cancelRequested) break;

      // Step 2: Clear input and prepare output area
      api.clearInput();
      api.clearForReplay();
      api.setTaskText(taskText);
      api.updateTabTitle(taskText);
      api.hideWelcome();

      // Step 3: Request events from backend
      var events = await requestEvents(api, session.id);
      if (cancelRequested) break;

      // Step 4: Group events into panels and replay panel-by-panel
      var panelGroups = groupEventsIntoPanels(events);

      for (var j = 0; j < panelGroups.length; j++) {
        if (cancelRequested) break;
        var group = panelGroups[j];

        // Check if this group is a result panel
        if (group.length === 1 && group[0].type === 'result') {
          await streamResultEvent(api, group[0]);
          continue;
        }

        // Process all events in this panel group at once
        for (var k = 0; k < group.length; k++) {
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

    api.active = false;
  };

  /**
   * Cancel an in-progress demo replay.
   */
  window._cancelDemoReplay = function () {
    cancelRequested = true;
    var api = getApi();
    if (api) api.active = false;
  };

  /**
   * Check whether a demo replay is currently running.
   */
  window._isDemoActive = function () {
    var api = getApi();
    return api ? api.active : false;
  };
})();

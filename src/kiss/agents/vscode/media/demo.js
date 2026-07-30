// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
(function () {
  'use strict';

  let cancelRequested = false;

  let replayGen = 0;

  let pendingEventsResolve = null;

  let pauseRequested = false;
  let pauseResolvers = [];

  let lastReplayArgs = null;

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
    } catch (_e) {}
  }

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

  window._isDemoPaused = function () {
    return pauseRequested;
  };

  function replayStopped(gen) {
    return cancelRequested || gen !== replayGen;
  }

  async function pauseGate(gen) {
    while (pauseRequested && !replayStopped(gen)) {
      await new Promise(resolve => {
        pauseResolvers.push(resolve);
      });
    }
  }

  let demoSubTabSeq = 0;

  // Delegates to main.js's sanitizer (exposed on window._demoApi) so demo
  // replay strips exactly what live chat rendering strips — demo.js used to
  // carry its own weaker copy that let custom elements through.
  function kissSanitize(html) {
    return getApi().kissSanitize(html);
  }

  function sleep(ms) {
    return new Promise(resolve => {
      setTimeout(resolve, ms);
    });
  }

  function getApi() {
    return window._demoApi;
  }

  function requestEvents(api, session) {
    const reqTabId = api.getActiveTabId();
    const reqTaskId = session.task_id;
    return new Promise(resolve => {
      const deliver = function (events, envelope) {
        if (envelope && !eventsReplyMatches(envelope, reqTabId, reqTaskId)) {
          return;
        }
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

  function discardPendingEvents(api) {
    if (api) api.resolveEvents = null;
    const resolve = pendingEventsResolve;
    pendingEventsResolve = null;
    if (resolve) resolve([]);
  }

  function fmtN(n) {
    return Number(n).toLocaleString('en-US');
  }

  async function streamResultEvent(api, ev, gen) {
    const O = document.getElementById('output');
    if (!O) return;

    const rc = document.createElement('div');
    rc.className = 'ev rc';

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

    if (ev.is_continue) {
      const contDiv = document.createElement('div');
      contDiv.className = 'rc-status';
      contDiv.textContent = 'Status: Continue';
      rc.appendChild(contDiv);
    } else if (ev.success === false) {
      const failDiv = document.createElement('div');
      failDiv.className = 'rc-status rc-status-fail';
      failDiv.textContent = 'Status: FAILED';
      rc.appendChild(failDiv);
    }

    const body = document.createElement('div');
    body.className = 'rc-body md-body';
    rc.appendChild(body);
    O.appendChild(rc);

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
        // The summary wire format is always HTML (see finish() in
        // kiss/core/utils.py); render it sanitized, never via Markdown —
        // matching createResultPanel() in main.js.  Fall back to plain
        // text when the host api exposes no sanitizer.
        if (typeof getApi().kissSanitize === 'function') {
          body.innerHTML = kissSanitize(accumulated);
        } else {
          body.textContent = accumulated;
        }
        await sleep(TICK_MS);
        await pauseGate(gen);
      }
    }

    if (typeof hljs !== 'undefined') {
      body.querySelectorAll('pre code').forEach(bl => {
        hljs.highlightElement(bl);
      });
    }
  }

  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  const SKIP_TYPES = {
    task_done: 1,
    task_error: 1,
    task_stopped: 1,
    task_interrupted: 1,
    followup_suggestion: 1,
  };

  function groupEventsIntoPanels(events) {
    const panels = [];
    let current = [];
    let afterToolResult = true;

    for (let i = 0; i < events.length; i++) {
      const ev = events[i];
      const t = ev.type;

      if (SKIP_TYPES[t]) continue;

      if (t === 'tool_call') {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = false;
        continue;
      }

      if ((t === 'thinking_start' || t === 'text_delta') && afterToolResult) {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = false;
        continue;
      }

      if (t === 'result') {
        if (current.length > 0) panels.push(current);
        panels.push([ev]);
        current = [];
        afterToolResult = false;
        continue;
      }

      if (t === 'prompt') {
        if (current.length > 0) panels.push(current);
        current = [ev];
        afterToolResult = true;
        continue;
      }

      if (t === 'tool_result') {
        current.push(ev);
        afterToolResult = true;
        continue;
      }

      current.push(ev);
    }

    if (current.length > 0) panels.push(current);
    return panels;
  }

  window._groupEventsIntoPanels = groupEventsIntoPanels;

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

  window._parseDemoTasks = parseDemoTasks;

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

  window._selectReplaySessions = selectReplaySessions;

  function groupHasFanOut(group) {
    for (let i = 0; i < group.length; i++) {
      if (group[i].type === 'tool_call' && group[i].name === 'run_parallel') {
        return true;
      }
    }
    return false;
  }

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

  async function replaySessions(api, items, myGen) {
    for (let i = 0; i < items.length; i++) {
      if (replayStopped(myGen)) break;
      const session = items[i];
      const taskText = session.preview || session.title || 'Untitled';

      if (i === 0) {
        api.clearForReplay();
      } else {
        api.resetOutputState();
      }
      api.setTaskText(taskText);
      api.updateTabTitle(taskText);

      const events = await requestEvents(api, session);
      await pauseGate(myGen);
      if (replayStopped(myGen)) break;

      const panelGroups = groupEventsIntoPanels(events);

      for (let j = 0; j < panelGroups.length; j++) {
        if (replayStopped(myGen)) break;
        await pauseGate(myGen);
        if (replayStopped(myGen)) break;
        const group = panelGroups[j];

        if (group.length === 1 && group[0].type === 'result') {
          await streamResultEvent(api, group[0], myGen);
          continue;
        }

        for (let k = 0; k < group.length; k++) {
          api.processEvent(group[k]);
          const speech = executeDemoToolCall(api, group[k]);
          if (speech && typeof speech.then === 'function') {
            await speech;
            await pauseGate(myGen);
            if (replayStopped(myGen)) break;
          }
        }
        if (replayStopped(myGen)) break;

        await sleep(groupHasFanOut(group) ? 2500 : 500);
        await pauseGate(myGen);
        if (!replayStopped(myGen)) {
          api.collapsePanels();
        }
      }

      if (i < items.length - 1) {
        await sleep(1000);
        await pauseGate(myGen);
      }
    }
  }

  window._startDemoReplay = async function (sessions, clicked) {
    const api = getApi();
    if (!api || api.active) return;
    api.active = true;
    cancelRequested = false;
    lastReplayArgs = {sessions: sessions, clicked: clicked};
    const myGen = ++replayGen;
    discardPendingEvents(api);

    try {
      api.setRunningState(true);
      api.showSpinner();
      pauseRequested = false;
      if (typeof api.setDemoUi === 'function') api.setDemoUi(true);

      const items = selectReplaySessions(sessions, clicked);
      await replaySessions(api, items, myGen);
    } catch (_e) {}

    if (myGen !== replayGen) return;

    discardPendingEvents(api);
    api.setRunningState(false);
    api.removeSpinner();
    pauseRequested = false;
    if (typeof api.setDemoUi === 'function') api.setDemoUi('ended');
    api.active = false;
  };

  window._restartDemoReplay = function () {
    const args = lastReplayArgs;
    const api = getApi();
    if (!args || !api || api.active) return false;
    window._startDemoReplay(args.sessions, args.clicked);
    return true;
  };

  window._clearDemoReplay = function () {
    lastReplayArgs = null;
    const api = getApi();
    if (api && !api.active && typeof api.setDemoUi === 'function') {
      api.setDemoUi(false);
    }
  };

  window._cancelDemoReplay = function (opts) {
    const restoreUi = !!(opts && opts.restoreUi);
    cancelRequested = true;
    replayGen++;
    pauseRequested = false;
    const resolvers = pauseResolvers;
    pauseResolvers = [];
    for (let i = 0; i < resolvers.length; i++) resolvers[i]();
    notifyPauseChanged();
    const api = getApi();
    discardPendingEvents(api);
    if (restoreUi) lastReplayArgs = null;
    if (api) {
      api.active = false;
      api.setRunningState(false);
      api.removeSpinner();
      if (typeof api.setDemoUi === 'function') {
        api.setDemoUi(restoreUi || !lastReplayArgs ? false : 'ended');
      }
      if (typeof api.stopSpeech === 'function') api.stopSpeech();
    }
  };

  window._isDemoActive = function () {
    const api = getApi();
    return api ? api.active : false;
  };
})();

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

(function () {
  // @ts-ignore - vscode is injected by the webview
  const vscode = acquireVsCodeApi();
  const api = createSorcarApi(msg => vscode.postMessage(msg));

  function fmtN(n) {
    return Number(n).toLocaleString('en-US');
  }

  function fmtElapsedMs(ms) {
    const n = Math.max(0, Math.round(Number(ms) || 0));
    if (n < 1000) return n + 'ms';
    const s = n / 1000;
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s / 60);
    const sec = (s - m * 60).toFixed(1);
    return m + 'm ' + sec + 's';
  }

  // Reads a pixel-valued CSS custom property off <body> so layout
  // bounds live in the stylesheet only.  Returns `fallback` when the
  // property is missing (VS Code webview, which never loads
  // remote-codex.css) or is not a positive length.
  function cssPxVar(name, fallback) {
    if (typeof window.getComputedStyle !== 'function') return fallback;
    const style = window.getComputedStyle(document.body);
    const px = parseFloat(style.getPropertyValue(name));
    return Number.isFinite(px) && px > 0 ? px : fallback;
  }

  const _activePanels = new Set();
  let _activePanelTickIv = null;

  function stampPanelStart(el) {
    if (!el || _deferHighlight) return;
    if (el.dataset.startMs) return;
    el.dataset.startMs = String(Date.now());
    _activePanels.add(el);
    _renderPanelTime(el);
    _startActivePanelTick();
  }

  // panelts-coverage:start
  function _renderPanelTime(el) {
    if (!el) return;
    const startMs = Number(el.dataset.startMs || 0);
    if (!startMs) return;
    const ms = Date.now() - startMs;
    const footer = window.PanelCopy.ensurePanelFoot(el);
    if (footer !== el.lastElementChild) el.appendChild(footer);
    let span = footer.querySelector(':scope > .panel-elapsed');
    if (!span) {
      span = document.createElement('span');
      span.className = 'panel-elapsed';
      footer.appendChild(span);
    }
    span.textContent = fmtElapsedMs(ms);
  }
  // panelts-coverage:end

  function _startActivePanelTick() {
    if (_activePanelTickIv) return;
    if (_activePanels.size === 0) return;
    _activePanelTickIv = setInterval(() => {
      for (const el of Array.from(_activePanels)) {
        if (!el || !el.isConnected) {
          _activePanels.delete(el);
          continue;
        }
        _renderPanelTime(el);
      }
      if (_activePanels.size === 0) {
        clearInterval(_activePanelTickIv);
        _activePanelTickIv = null;
      }
    }, 1000);
  }

  function finalizePanelTime(el) {
    if (!el) return;
    const startMs = Number(el.dataset.startMs || 0);
    if (!startMs) return;
    _renderPanelTime(el);
    el.dataset.timeDone = '1';
    _activePanels.delete(el);
    if (_activePanels.size === 0 && _activePanelTickIv) {
      clearInterval(_activePanelTickIv);
      _activePanelTickIv = null;
    }
  }

  function reviveActivePanelTimes(root) {
    if (!root || !root.querySelectorAll) return;
    const stamped = root.querySelectorAll(
      '[data-start-ms]:not([data-time-done])',
    );
    for (let i = 0; i < stamped.length; i++) {
      _activePanels.add(stamped[i]);
      _renderPanelTime(stamped[i]);
    }
    _startActivePanelTick();
  }

  function discardProvisionalPanel(el) {
    if (!el) return;
    _activePanels.delete(el);
    if (_activePanels.size === 0 && _activePanelTickIv) {
      clearInterval(_activePanelTickIv);
      _activePanelTickIv = null;
    }
    if (el.parentNode) el.parentNode.removeChild(el);
  }

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
    const walk = root => {
      const elements = Array.from(root.querySelectorAll('*'));
      for (const el of elements) {
        if (BAD_TAGS.has(el.tagName)) {
          el.remove();
          continue;
        }
        if (el.tagName.includes('-')) {
          while (el.firstChild) el.before(el.firstChild);
          el.remove();
          continue;
        }
        for (const attr of Array.from(el.attributes)) {
          const name = attr.name.toLowerCase();
          if (name.startsWith('on')) {
            el.removeAttribute(attr.name);
            continue;
          }
          if (URL_ATTRS.has(name)) {
            const value = attr.value || '';
            let schemeProbe = '';
            for (const char of value) {
              const code = char.charCodeAt(0);
              if (code > 0x20 && (code < 0x7f || code > 0x9f)) {
                schemeProbe += char;
              }
            }
            if (/^(javascript|data|vbscript):/i.test(schemeProbe)) {
              el.removeAttribute(attr.name);
            }
          }
        }
      }
    };
    walk(t.content);
    return t.innerHTML;
  }

  const RESULT_HTML_TAG_RE = new RegExp(
    '</?(?:p|div|h[1-6]|ul|ol|li|br|hr|table|thead|tbody|tr|td|th|' +
      'pre|code|span|b|i|u|strong|em|a|img|blockquote|section|article|' +
      'details|summary)(?:\\s[^<>]*)?/?>',
    'i',
  );

  /**
   * Normalize a result summary for HTML rendering.
   *
   * New finish() results are already HTML. Persisted events from before the
   * HTML wire-format migration still contain Markdown, so convert only input
   * that has no known HTML tag. This mirrors kiss.core.utils.ensure_html().
   */
  function resultSummaryHtml(summary) {
    const text = String(summary == null ? '' : summary);
    const detectionText = text
      .replace(/```[\s\S]*?```/g, '')
      .replace(/`[^`\n]*`/g, '');
    if (
      text.trimStart().slice(0, 9).toLowerCase() === '<!doctype' ||
      RESULT_HTML_TAG_RE.test(detectionText)
    ) {
      return text;
    }
    if (typeof marked !== 'undefined') return marked.parse(text);
    return '<p>' + esc(text).replace(/\n/g, '<br>') + '</p>';
  }

  const notificationTimers = new Map();

  function ensureNotificationContainer() {
    let container = document.getElementById('kiss-notification-container');
    if (!container) {
      container = document.createElement('section');
      container.id = 'kiss-notification-container';
      container.className = 'kiss-notification-container';
      container.setAttribute('aria-label', 'KISS Sorcar notifications');
      document.body.appendChild(container);
    }
    let liveRegion = document.getElementById('kiss-notification-live-region');
    if (!liveRegion) {
      liveRegion = document.createElement('div');
      liveRegion.id = 'kiss-notification-live-region';
      liveRegion.className = 'kiss-sr-only';
      liveRegion.setAttribute('role', 'status');
      liveRegion.setAttribute('aria-live', 'polite');
      liveRegion.setAttribute('aria-atomic', 'true');
      document.body.appendChild(liveRegion);
    }
    return container;
  }

  function notificationIcon(severity) {
    if (severity === 'error') return '\u2715';
    if (severity === 'warning') return '\u26A0';
    return '\u2139';
  }

  function notificationTitle(severity) {
    if (severity === 'error') return 'Error';
    if (severity === 'warning') return 'Warning';
    return 'Information';
  }

  function clearNotificationTimer(id) {
    const timer = notificationTimers.get(id);
    if (timer) clearTimeout(timer);
    notificationTimers.delete(id);
  }

  function notificationSelector(id) {
    return (
      '.kiss-notification[data-notification-id="' +
      String(id).replace(/\\/g, '\\\\').replace(/"/g, '\\"') +
      '"]'
    );
  }

  function removeNotification(id, action, notifyExtension) {
    clearNotificationTimer(id);
    const toast = document.querySelector(notificationSelector(id));
    if (toast && toast.parentNode) toast.parentNode.removeChild(toast);
    if (notifyExtension) {
      api.notificationAction({id: id, action: action});
    }
  }

  function scheduleNotificationDismiss(id, severity, sticky) {
    clearNotificationTimer(id);
    if (sticky) return;
    const delay =
      severity === 'error' ? 7500 : severity === 'warning' ? 6000 : 5000;
    notificationTimers.set(
      id,
      setTimeout(() => removeNotification(id, undefined, false), delay),
    );
  }

  function showNotification(ev) {
    const container = ensureNotificationContainer();
    const id = ev.id || String(Date.now());
    let toast = container.querySelector(notificationSelector(id));
    const severity = ev.severity || 'info';
    const actions = Array.isArray(ev.actions) ? ev.actions : [];
    const hasLocalActions = actions.some(
      action =>
        action &&
        typeof action === 'object' &&
        !Array.isArray(action) &&
        typeof action.onClick === 'function',
    );
    const notifyOnClose = actions.length > 0 && !hasLocalActions;
    const sticky = !!ev.sticky || actions.length > 0 || !!ev.progress;
    if (!toast) {
      toast = document.createElement('article');
      toast.className = 'kiss-notification';
      toast.dataset.notificationId = String(id);
      toast.tabIndex = -1;
      container.insertBefore(toast, container.firstChild);
      toast.addEventListener('mouseenter', () => clearNotificationTimer(id));
      toast.addEventListener('focusin', () => clearNotificationTimer(id));
      toast.addEventListener('mouseleave', () => {
        const state = toast.kissNotificationState || {
          id: id,
          severity: 'info',
          sticky: false,
        };
        scheduleNotificationDismiss(state.id, state.severity, state.sticky);
      });
      toast.addEventListener('focusout', () => {
        const state = toast.kissNotificationState || {
          id: id,
          severity: 'info',
          sticky: false,
        };
        scheduleNotificationDismiss(state.id, state.severity, state.sticky);
      });
    }
    toast.kissNotificationState = {id: id, severity: severity, sticky: sticky};
    toast.className = 'kiss-notification kiss-notification-' + severity;
    toast.dataset.notificationSticky = sticky ? 'true' : 'false';
    toast.setAttribute('role', severity === 'error' ? 'alert' : 'status');
    toast.setAttribute(
      'aria-label',
      notificationTitle(severity) + ': ' + (ev.message || ''),
    );

    const body = document.createElement('div');
    body.className = 'kiss-notification-body';
    const icon = document.createElement('div');
    icon.className = 'kiss-notification-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = notificationIcon(severity);
    const content = document.createElement('div');
    content.className = 'kiss-notification-content';
    const title = document.createElement('div');
    title.className = 'kiss-notification-title';
    title.textContent = notificationTitle(severity);
    const message = document.createElement('div');
    message.className = 'kiss-notification-message';
    message.textContent = ev.message || '';
    content.appendChild(title);
    content.appendChild(message);
    if (ev.progress && ev.progressMessage) {
      const progress = document.createElement('div');
      progress.className = 'kiss-notification-progress-message';
      progress.textContent = ev.progressMessage;
      content.appendChild(progress);
    }
    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'kiss-notification-close';
    closeBtn.setAttribute('aria-label', 'Dismiss notification');
    closeBtn.textContent = '\u00d7';
    closeBtn.addEventListener('click', () =>
      removeNotification(id, undefined, notifyOnClose),
    );
    body.appendChild(icon);
    body.appendChild(content);
    body.appendChild(closeBtn);
    toast.replaceChildren(body);
    if (ev.progress) {
      const progressBar = document.createElement('div');
      progressBar.className = 'kiss-notification-progress';
      progressBar.setAttribute('aria-hidden', 'true');
      toast.appendChild(progressBar);
    }
    if (actions.length > 0) {
      const actionRow = document.createElement('div');
      actionRow.className = 'kiss-notification-actions';
      actions.forEach(action => {
        const isObj =
          action && typeof action === 'object' && !Array.isArray(action);
        const label = isObj ? String(action.label || '') : String(action);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'kiss-notification-action';
        if (isObj && action.svg) {
          const cleaned = kissSanitize(String(action.svg));
          const parser = new window.DOMParser();
          const doc = parser.parseFromString(cleaned, 'image/svg+xml');
          const svgEl = doc.documentElement;
          if (
            svgEl &&
            svgEl.namespaceURI === 'http://www.w3.org/2000/svg' &&
            svgEl.localName === 'svg'
          ) {
            svgEl.setAttribute('class', 'kiss-notification-action-icon');
            svgEl.setAttribute('aria-hidden', 'true');
            button.appendChild(document.importNode(svgEl, true));
          }
        }
        if (label) {
          const labelEl = document.createElement('span');
          labelEl.className = 'kiss-notification-action-label';
          labelEl.textContent = label;
          button.appendChild(labelEl);
        }
        if (isObj && action.ariaLabel) {
          button.setAttribute('aria-label', String(action.ariaLabel));
        } else if (label) {
          button.setAttribute('aria-label', label);
        }
        button.addEventListener('click', () => {
          if (isObj && typeof action.onClick === 'function') {
            try {
              action.onClick();
            } catch (_err) {}
            removeNotification(id, undefined, false);
            return;
          }
          removeNotification(id, isObj ? label : action, true);
        });
        actionRow.appendChild(button);
      });
      toast.appendChild(actionRow);
    }
    const liveRegion = document.getElementById('kiss-notification-live-region');
    if (liveRegion) {
      liveRegion.textContent = '';
      setTimeout(() => {
        liveRegion.textContent =
          notificationTitle(severity) + ': ' + (ev.message || '');
      }, 0);
    }
    scheduleNotificationDismiss(id, severity, sticky);
  }

  function updateNotification(ev) {
    if (ev.close) {
      removeNotification(ev.id, undefined, false);
      return;
    }
    showNotification(ev);
  }

  let isRunning = false;
  // The model the user last picked in the picker. It is what a submit
  // runs with and what the picker shows, EXCEPT while a running agent
  // has switched models on itself -- that transient override lives in
  // `agentModel` and is dropped the moment the task ends, so the user's
  // own choice is never silently replaced by the agent's.
  let selectedModel = '';
  let agentModel = '';
  let allModels = [];
  let modelDDIdx = -1;
  let attachments = [];
  // Human-readable reasons why a picked file could not be attached (an
  // undecodable HEIC on a browser without HEIC support, an unreadable file).
  // Rendered next to the file chips so a failed attachment is never silent.
  let attachErrors = [];
  // Set while a send is parked waiting for an attachment to finish converting,
  // so repeated Enter presses cannot submit the same prompt twice.
  let awaitingAttachments = false;
  let _deferHighlight = false;
  let acIdx = -1;

  let histCache = [];
  let histIdx = -1;

  let ghostTimer = null;
  let currentGhost = '';

  let allHistSessions = [];

  let historyOffset = 0;
  let historyLoading = false;
  let historyHasMore = true;
  let historyGeneration = 0;
  let historyDateRangeUserSet = false;
  const historyLastRunningTaskIds = new Set();
  const historyJustCompletedTaskIds = new Set();

  let currentTaskName = '';
  let currentTaskId = null;
  let oldestLoadedTaskId = null;
  let newestLoadedTaskId = null;
  let adjacentLoading = false;
  let noPrevTask = false;
  let noNextTask = false;
  let overscrollAccum = 0;
  let overscrollDir = '';
  let overscrollTimer = null;
  const OVERSCROLL_THRESHOLD = 150;
  let currentTaskMetrics = {tokens: '', budget: '', steps: ''};

  function genTabId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID)
      return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = (Math.random() * 16) | 0;
      return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
  }

  let tabs = [];
  let activeTabId = '';
  // The chat tab the host believes is on screen. Kept in step with the host
  // so a tab that gets closed can never be left standing there.
  let reportedChatTabId = '';

  let configWorkDir = '';

  function makeTab(title) {
    const _id = genTabId();
    return {
      id: _id,
      title: title || 'new chat',
      backendChatId: '',
      currentTaskId: null,
      // Set by sendMessage() the moment a submit leaves this tab, so the tab
      // owns the task before the daemon has told anyone its real id.
      pendingTaskId: null,
      isRunning: false,
      // Raised by the Stop button until the task actually ends, so a
      // stop the agent has not reached yet looks different from a stop
      // that never arrived (see stop_button_delay_2026-08-05.html).
      isStopping: false,
      outputFragment: null,
      taskPanelHTML: '',
      taskPanelVisible: false,
      statusTextContent: 'Ready',
      statusTextColor: 'var(--green)',
      statusTokensText: '',
      statusBudgetText: '',
      statusStepsText: '',
      welcomeVisible: true,
      selectedModel: selectedModel,
      agentModel: '',
      attachments: [],
      attachErrors: [],
      inputValue: '',
      worktreeBarEl: null,
      t0: null,
      endTs: 0,
      workDir: '',
      streamState: null,
      streamLlmPanel: null,
      streamLlmPanelState: null,
      streamLastToolName: '',
      streamPendingPanel: false,
      streamStepCount: 0,
      // A content tab shows a file instead of a conversation: it keeps
      // its own detached view and never owns task output.
      isContentTab: false,
      contentPath: '',
      contentViewEl: null,
      contentEditor: null,
      // Set on a sub-agent tab opened by a run_parallel fan-out, naming
      // the conversation that started it.
      isSubagentTab: false,
      parentTabId: null,
      isDone: false,
      lastTaskFailed: false,
      hasRunTask: false,
      askPendingQuestion: null,
      askQuestionEl: null,
      askInputEl: null,
    };
  }

  function getTab(id) {
    return tabs.find(t => t.id === id) || null;
  }

  // modelpick-coverage:start
  /** Repaint the picker label from the active tab's model state. */
  function refreshModelLabel() {
    if (modelName) modelName.textContent = agentModel || selectedModel;
  }

  /**
   * Apply a `modelPick` event to `tabId`.
   *
   * `source` says what the model means:
   *   'agent'   - a running agent switched models; show it while the
   *               task lasts without touching the user's own pick.
   *   'restore' - the task ended: drop the override and show the pick
   *               the user made in that tab again.
   */
  function applyModelPick(tabId, model, source) {
    if (!model || !tabId) return;
    const isAgent = source === 'agent';
    const t = getTab(tabId);
    if (t) {
      if (isAgent) {
        t.agentModel = model;
      } else {
        t.agentModel = '';
        t.selectedModel = model;
      }
    }
    if (tabId !== activeTabId) return;
    if (isAgent) {
      agentModel = model;
    } else {
      agentModel = '';
      selectedModel = model;
    }
    refreshModelLabel();
    if (modelDropdown && modelDropdown.classList.contains('open')) {
      renderModelList(modelSearch ? modelSearch.value : '');
    }
  }

  /** Drop *tab*'s agent override; its picker shows the user's pick again. */
  function clearAgentModel(tabId) {
    const t = getTab(tabId);
    if (t) t.agentModel = '';
    if (tabId === activeTabId && agentModel) {
      agentModel = '';
      refreshModelLabel();
    }
  }
  // modelpick-coverage:end

  function getTabByBackendChatId(chatId) {
    if (chatId === undefined || chatId === null || chatId === '') return null;
    const key = String(chatId);
    return tabs.find(t => String(t.backendChatId || '') === key) || null;
  }

  function placeSubagentTabAfterParent(subTab, parentId) {
    const curIdx = tabs.indexOf(subTab);
    if (curIdx >= 0) tabs.splice(curIdx, 1);
    const parentIdx = tabs.findIndex(t => {
      return t.id === parentId;
    });
    if (parentIdx < 0) {
      tabs.push(subTab);
      return;
    }
    let insertAt = parentIdx + 1;
    while (insertAt < tabs.length && tabs[insertAt].parentTabId === parentId) {
      insertAt += 1;
    }
    tabs.splice(insertAt, 0, subTab);
  }

  // panelts-coverage:start
  function normalizeEventTs(ev) {
    if (
      ev &&
      ev.ts == null &&
      typeof ev._timestamp === 'number' &&
      ev._timestamp > 0 &&
      ev._timestamp <= 8.64e12
    ) {
      ev.ts = Math.round(ev._timestamp * 1000);
    }
    return ev;
  }
  // panelts-coverage:end

  function mkThoughtsPanel(ts) {
    const panel = mkEl('div', 'llm-panel');
    const hdr = mkEl('div', 'llm-panel-hdr');
    hdr.textContent = 'Thoughts';
    addCollapse(panel, hdr, ts);
    panel.appendChild(hdr);
    stampPanelStart(panel);
    return panel;
  }

  function isActiveTabRunning() {
    const tab = getTab(activeTabId);
    return tab ? tab.isRunning : false;
  }

  function findTabByEvt(ev) {
    return ev && ev.tabId !== undefined ? getTab(ev.tabId) : null;
  }

  function workDirForTab(tabId) {
    const tab = getTab(tabId);
    if (tab && tab.workDir) return tab.workDir;
    return configWorkDir || '';
  }

  function saveCurrentTab() {
    const tab = getTab(activeTabId);
    if (!tab) return;
    if (tab.isContentTab) return;
    // visibletask-coverage:start
    // The panel and the status row may be describing a neighbouring task
    // the reader scrolled into. That is a viewing position, not the tab's
    // identity, so a tab is always saved under its own task. This has to
    // be read before the transcript is detached below, while #output
    // still has the geometry the reader was looking at.
    const shownRegion = visibleRegion();
    const neighbour = shownRegion && regionNeighbour(shownRegion);
    // visibletask-coverage:end
    tab.welcomeVisible = welcome ? welcome.style.display !== 'none' : true;
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    tab.outputFragment = document.createDocumentFragment();
    while (O.firstChild) tab.outputFragment.appendChild(O.firstChild);
    // visibletask-coverage:start
    tab.taskPanelHTML = neighbour
      ? currentTaskName
      : taskPanelText
        ? taskPanelText.textContent
        : '';
    tab.taskPanelVisible = neighbour
      ? !!currentTaskName
      : taskPanel
        ? taskPanel.classList.contains('visible')
        : false;
    tab.statusTextContent = statusText ? statusText.textContent : 'Ready';
    tab.statusTextColor = statusText ? statusText.style.color : 'var(--green)';
    tab.statusTokensText = neighbour
      ? currentTaskMetrics.tokens
      : statusTokens
        ? statusTokens.textContent
        : '';
    tab.statusBudgetText = neighbour
      ? currentTaskMetrics.budget
      : statusBudget
        ? statusBudget.textContent
        : '';
    tab.statusStepsText = neighbour
      ? currentTaskMetrics.steps
      : statusSteps
        ? statusSteps.textContent
        : '';
    // visibletask-coverage:end
    tab.selectedModel = selectedModel;
    tab.agentModel = agentModel;
    tab.attachments = attachments;
    tab.attachErrors = attachErrors;
    tab.inputValue = inp.value;
    tab.isRunning = isActiveTabRunning();
    tab.t0 = t0;
    tab.endTs = endTs;
    tab.streamState = state;
    tab.streamLlmPanel = llmPanel;
    tab.streamLlmPanelState = llmPanelState;
    tab.streamLastToolName = lastToolName;
    tab.streamPendingPanel = pendingPanel;
    tab.streamStepCount = stepCount;
    if (worktreeBar && worktreeBar.parentNode) {
      tab.worktreeBarEl = worktreeBar;
      worktreeBar.parentNode.removeChild(worktreeBar);
    } else {
      tab.worktreeBarEl = null;
    }
    worktreeBar = null;
    if (inputContainer) inputContainer.style.display = '';
    persistTabState();
  }

  // The single place that moves the host's idea of the on-screen chat tab.
  // The empty string is a legitimate value meaning "no chat tab at all", so
  // it is reported like any other change rather than swallowed.
  function reportChatTab(tabId) {
    const id = tabId || '';
    if (id === reportedChatTabId) return;
    reportedChatTabId = id;
    api.activeTabChanged({tabId: id});
  }

  // Closing tabs must never leave the host naming one that is gone: it would
  // keep matching merges (and completions) against a dead chat. When the
  // reported tab is removed and the tab taking its place on screen is a
  // content tab, point the host at whichever chat tab survives — and when
  // none survives, clear it, so the deleted chat stops owning the editor.
  function reportSurvivingChatTab() {
    if (tabs.some(t => t.id === reportedChatTabId)) return;
    const chat = tabs.find(t => !t.isContentTab);
    reportChatTab(chat ? chat.id : '');
  }

  function restoreTab(tab) {
    hideContentArea();
    activeTabId = tab.id;
    // Every chat tab activation funnels through here — switching, creating,
    // and falling back after a close — so this is the one place that tells
    // the host which chat is on screen. The host only lets that chat take
    // over the editor (e.g. to open a merge for review), so a stale id would
    // yank an editor in front of a user looking at a different tab.
    // Content tabs deliberately do not report: the host compares this id
    // against chat tab ids only, so viewing a file leaves it untouched.
    reportChatTab(tab.id);
    O.innerHTML = '';
    // autoscroll-coverage:start
    // A switched-to (or newly created) tab is a fresh view, so any
    // scroll lock the user engaged on the previous tab's chat no
    // longer applies.
    resetUserScrollLock();
    // autoscroll-coverage:end
    if (tab.outputFragment) {
      O.appendChild(tab.outputFragment);
      tab.outputFragment = null;
      reviveActivePanelTimes(O);
      // autoscroll-coverage:start
      // Events may have streamed into the fragment while the tab was
      // hidden: land the restored chat at the end of its latest panel.
      autoScrollLatestEventPanel(O.lastElementChild);
      // autoscroll-coverage:end
    }
    if (taskPanel && taskPanelText) {
      const restoredTask = (tab.taskPanelHTML || '').trim();
      taskPanelText.textContent = restoredTask;
      if (restoredTask) {
        taskPanelText.setAttribute('data-tooltip', restoredTask);
      } else {
        taskPanelText.removeAttribute('data-tooltip');
      }
      if (tab.taskPanelVisible) taskPanel.classList.add('visible');
      else taskPanel.classList.remove('visible');
    }
    currentTaskName = (tab.taskPanelHTML || '').trim();
    currentTaskId = tab.currentTaskId !== undefined ? tab.currentTaskId : null;
    if (statusText) {
      statusText.textContent = tab.statusTextContent || 'Ready';
      statusText.style.color = tab.statusTextColor || 'var(--green)';
    }
    if (statusTokens) statusTokens.textContent = tab.statusTokensText;
    if (statusBudget) statusBudget.textContent = tab.statusBudgetText;
    if (statusSteps) statusSteps.textContent = tab.statusStepsText;
    // visibletask-coverage:start
    // The restored numbers are this tab's own: they are what the status
    // row must come back to after the reader visits a neighbour.
    currentTaskMetrics = {
      tokens: tab.statusTokensText || '',
      budget: tab.statusBudgetText || '',
      steps: tab.statusStepsText || '',
    };
    // visibletask-coverage:end
    if (welcome) {
      if (tab.welcomeVisible) {
        showWelcomeScreen();
      } else {
        welcome.style.display = 'none';
        refreshWelcomeLayout();
      }
    }
    selectedModel = tab.selectedModel || '';
    agentModel = tab.agentModel || '';
    refreshModelLabel();
    attachments = tab.attachments || [];
    attachErrors = tab.attachErrors || [];
    renderFileChips();
    inp.value = tab.inputValue || '';
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
    t0 = tab.t0 || null;
    endTs = tab.endTs || 0;
    state = tab.streamState || mkS();
    llmPanel = tab.streamLlmPanel || null;
    llmPanelState = tab.streamLlmPanelState || mkS();
    lastToolName = tab.streamLastToolName || '';
    pendingPanel = tab.streamPendingPanel || false;
    stepCount = tab.streamStepCount || 0;
    if (worktreeBar && worktreeBar.parentNode)
      worktreeBar.parentNode.removeChild(worktreeBar);
    worktreeBar = null;
    if (tab.worktreeBarEl) {
      worktreeBar = tab.worktreeBarEl;
      tab.worktreeBarEl = null;
      const area = document.getElementById('input-area');
      area.insertBefore(worktreeBar, area.firstChild);
    }
    const hideInput = worktreeBar || (tab.isSubagentTab && !tab.isRunning);
    if (hideInput) {
      if (inputContainer) inputContainer.style.display = 'none';
    } else {
      if (inputContainer) inputContainer.style.display = '';
    }
    updateInputDisabled();
    // A tab that ran while hidden comes back looking like one that ran
    // on screen: everything but its latest panel collapsed.
    collapseOlderPanels(O, tab.id);
    resetAdjacentState();
    syncAskModalToActiveTab();
    // visibletask-coverage:start
    // The transcript comes back where the reader left it, which may well
    // be inside a neighbouring task, so the panel is derived from the
    // restored transcript rather than from the tab's own name.
    updateVisibleTask();
    // visibletask-coverage:end
  }

  // Light / dark theme toggle for the REMOTE webapp only.  The VS Code
  // webview always follows the editor theme, so none of this runs there
  // (the toggle button is only created for body.remote-chat).  The dark
  // palette is the default; "light" mimics VS Code's Light Modern theme
  // (see remote-codex.css).  The choice is persisted in localStorage.
  const REMOTE_THEME_KEY = 'kissRemoteTheme';

  const THEME_SUN_SVG =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';

  const THEME_MOON_SVG =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  function getSavedRemoteTheme() {
    try {
      return localStorage.getItem(REMOTE_THEME_KEY) === 'light'
        ? 'light'
        : 'dark';
    } catch (_e) {
      return 'dark';
    }
  }

  // The button shows the theme it switches TO: a sun while in dark
  // mode, a moon while in light mode.
  function updateThemeButton(btn) {
    const light = document.body.classList.contains('light-theme');
    btn.innerHTML = light ? THEME_MOON_SVG : THEME_SUN_SVG;
    const label = light ? 'Switch to dark mode' : 'Switch to light mode';
    btn.title = label;
    btn.setAttribute('aria-label', label);
  }

  function applyRemoteTheme(theme) {
    if (!document.body.classList.contains('remote-chat')) return;
    document.body.classList.toggle('light-theme', theme === 'light');
    const hljsLink = document.getElementById('hljs-theme');
    const hljsUrls = window.__HLJS_THEME_CSS__;
    if (hljsLink && hljsUrls && hljsUrls[theme]) {
      hljsLink.setAttribute('href', hljsUrls[theme]);
    }
    const btn = document.querySelector('#tab-bar .chat-tab-theme');
    if (btn) updateThemeButton(btn);
  }

  function toggleRemoteTheme() {
    const next = document.body.classList.contains('light-theme')
      ? 'dark'
      : 'light';
    try {
      localStorage.setItem(REMOTE_THEME_KEY, next);
    } catch (_e) {
      /* private browsing: theme simply won't persist */
    }
    applyRemoteTheme(next);
  }

  // Arrow-key navigation for the tablist (WAI-ARIA tabs pattern,
  // manual activation): ArrowLeft/ArrowRight move focus to the
  // previous/next tab with wrap-around, Home/End jump to the
  // first/last tab, and the roving Tab stop follows the focus so the
  // user can Tab away and come back to where they were.  Focus
  // movement alone never activates; Enter/Space do.  The per-tab close
  // '×' controls keep tabindex=0 (they are separate buttons, not tabs,
  // and this keeps them directly Tab-reachable -- the simplest correct
  // option under the pattern).
  function moveTabFocus(fromEl, key) {
    const tabList = document.getElementById('tab-list');
    if (!tabList) return;
    const els = Array.from(tabList.querySelectorAll('[role="tab"]'));
    if (els.length === 0) return;
    let target;
    if (key === 'Home') {
      target = els[0];
    } else if (key === 'End') {
      target = els[els.length - 1];
    } else {
      const i = els.indexOf(fromEl);
      const d = key === 'ArrowLeft' ? -1 : 1;
      target = i < 0 ? els[0] : els[(i + d + els.length) % els.length];
    }
    els.forEach(t => t.setAttribute('tabindex', t === target ? '0' : '-1'));
    target.focus();
  }

  function renderTabBar() {
    const tabList = document.getElementById('tab-list');
    const tabBar = document.getElementById('tab-bar');
    if (!tabList || !tabBar) return;

    tabBar.style.display = '';

    // Chat tabs are proper a11y tabs: keyboard users reach them with
    // Tab, screen readers announce "<title>, tab, selected", and
    // Enter/Space activates them exactly like a click.
    tabList.setAttribute('role', 'tablist');
    tabList.setAttribute('aria-label', 'Chat tabs');

    tabList.innerHTML = '';
    // Roving tabindex (WAI-ARIA tabs pattern, manual activation): only
    // the active tab is a Tab stop; the arrow keys move focus between
    // tabs (see moveTabFocus) and Enter/Space activate.  If the active
    // tab is somehow not in the list, the first tab is the stop so the
    // tablist never becomes keyboard-unreachable.
    const rovingStopId = tabs.some(t => t.id === activeTabId)
      ? activeTabId
      : tabs.length > 0
        ? tabs[0].id
        : null;
    tabs.forEach(tab => {
      const el = document.createElement('div');
      el.className =
        'chat-tab' +
        (tab.id === activeTabId ? ' active' : '') +
        (tab.isSubagentTab ? ' subagent-tab' : '') +
        (tab.isContentTab ? ' content-tab' : '');
      el.dataset.tabId = tab.id;
      el.setAttribute('role', 'tab');
      el.setAttribute('tabindex', tab.id === rovingStopId ? '0' : '-1');
      el.setAttribute(
        'aria-selected',
        tab.id === activeTabId ? 'true' : 'false',
      );
      el.setAttribute('aria-label', tab.title);
      // All chat tabs swap the one shared chat surface (#output), so a
      // single shared tabpanel is the correct association.
      el.setAttribute('aria-controls', 'output');

      if (tab.isContentTab) {
        const fileIcon = document.createElement('span');
        fileIcon.className = 'content-tab-icon';
        fileIcon.textContent = '\uD83D\uDCC4';
        fileIcon.title = tab.contentPath || '';
        el.appendChild(fileIcon);
      } else if (tab.isSubagentTab) {
        const subIndicator = document.createElement('span');
        subIndicator.className =
          'subagent-indicator' + (tab.isDone ? ' done' : '');
        subIndicator.textContent = '\u25C9';
        subIndicator.title = tab.isDone ? 'Done' : 'Running';
        el.appendChild(subIndicator);
      } else {
        if (tab.isRunning) {
          const spinner = document.createElement('span');
          spinner.className = 'chat-tab-spinner';
          el.appendChild(spinner);
        } else if (tab.hasRunTask) {
          const icon = document.createElement('span');
          icon.className = tab.lastTaskFailed
            ? 'chat-tab-status chat-tab-fail'
            : 'chat-tab-status chat-tab-ok';
          icon.textContent = '\u25CF';
          el.appendChild(icon);
        }
      }

      if (tab.askPendingQuestion !== null && tab.id !== activeTabId) {
        const attention = document.createElement('span');
        attention.className = 'chat-tab-attention';
        attention.textContent = '?';
        attention.title = 'Waiting for your answer';
        el.appendChild(attention);
      }

      const label = document.createElement('span');
      label.className = 'chat-tab-label';
      label.textContent = tab.title;
      el.appendChild(label);

      const closeBtn = document.createElement('span');
      closeBtn.className = 'chat-tab-close';
      closeBtn.textContent = '\u00d7';
      closeBtn.setAttribute('role', 'button');
      closeBtn.setAttribute('tabindex', '0');
      closeBtn.setAttribute('aria-label', 'Close tab');
      closeBtn.addEventListener('click', e => {
        e.stopPropagation();
        closeTab(tab.id);
      });
      closeBtn.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          e.stopPropagation();
          closeTab(tab.id);
        }
      });
      el.appendChild(closeBtn);

      el.addEventListener('click', () => {
        switchToTab(tab.id);
      });
      el.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          switchToTab(tab.id);
        } else if (
          e.key === 'ArrowLeft' ||
          e.key === 'ArrowRight' ||
          e.key === 'Home' ||
          e.key === 'End'
        ) {
          e.preventDefault();
          moveTabFocus(el, e.key);
        }
      });
      el.addEventListener('contextmenu', e => {
        e.preventDefault();
        e.stopPropagation();
        showTabContextMenu(e.clientX, e.clientY, tab.id);
      });
      tabList.appendChild(el);
    });

    const existingAdd = tabBar.querySelector('.chat-tab-add');
    if (!existingAdd) {
      const addBtn = document.createElement('div');
      addBtn.className = 'chat-tab chat-tab-add';
      addBtn.textContent = '+';
      addBtn.title = 'New chat';
      addBtn.setAttribute('role', 'button');
      addBtn.setAttribute('tabindex', '0');
      addBtn.setAttribute('aria-label', 'New chat');
      addBtn.addEventListener('click', () => {
        createNewTab();
      });
      addBtn.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          createNewTab();
        }
      });
      tabBar.appendChild(addBtn);
    }

    if (
      document.body.classList.contains('remote-chat') &&
      !tabBar.querySelector('.chat-tab-theme')
    ) {
      const themeBtn = document.createElement('div');
      themeBtn.className = 'chat-tab chat-tab-theme';
      themeBtn.setAttribute('role', 'button');
      themeBtn.setAttribute('tabindex', '0');
      themeBtn.addEventListener('click', toggleRemoteTheme);
      themeBtn.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleRemoteTheme();
        }
      });
      tabBar.appendChild(themeBtn);
      updateThemeButton(themeBtn);
    }

    const existingSettings = tabBar.querySelector('.chat-tab-settings');
    if (!existingSettings) {
      const settingsBtn = document.createElement('div');
      settingsBtn.className = 'chat-tab chat-tab-settings';
      settingsBtn.title = 'Settings';
      settingsBtn.setAttribute('role', 'button');
      settingsBtn.setAttribute('tabindex', '0');
      settingsBtn.setAttribute('aria-label', 'Settings');
      settingsBtn.innerHTML =
        '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
      settingsBtn.addEventListener('click', () => {
        openSettingsPanel();
      });
      settingsBtn.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          openSettingsPanel();
        }
      });
      tabBar.appendChild(settingsBtn);
    }

    const activeEl = tabList.querySelector('.chat-tab.active');
    if (activeEl)
      activeEl.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }

  function switchToTab(tabId) {
    if (tabId === activeTabId) return;
    const tab = getTab(tabId);
    if (!tab) return;
    saveCurrentTab();
    if (tab.isContentTab) {
      activeTabId = tabId;
      showContentTab(tab);
      renderTabBar();
      return;
    }
    restoreTab(tab);
    renderTabBar();
    persistTabState();
    setRunningState(tab.isRunning);
    if (!tab.isRunning) {
      stopTimer();
      removeSpinner();
    }
    applyChevronState(currentTaskName);
    focusInputWithRetry();
  }

  // Which tab takes over when the tab the user was on is closed.
  //
  // A close the user asked for is a user action, so plain index
  // adjacency is right: whatever sits where the tab was - a file the
  // user opened included - may come forward.  A close the agent did on
  // its own (a sub-agent finishing) must not switch the user to a
  // content tab: the sub-agent may have parked its report there moments
  // earlier while the parent task is still running, and showing it would
  // break the "no tab switch unless finished" rule.  For those closes
  // prefer the closed tab's parent chat tab, then the nearest surviving
  // chat tab, falling back to adjacency only if no chat tab is left.
  function pickSuccessorTab(closed, origIdx, agentInitiated) {
    const adjacent = tabs[Math.min(origIdx, tabs.length - 1)];
    if (!agentInitiated) return adjacent;
    const parent = closed.parentTabId ? getTab(closed.parentTabId) : null;
    if (parent && !parent.isContentTab) return parent;
    for (let d = 0; d < tabs.length; d++) {
      const after = tabs[origIdx + d];
      if (after && !after.isContentTab) return after;
      const before = tabs[origIdx - 1 - d];
      if (before && !before.isContentTab) return before;
    }
    return adjacent;
  }

  // agentInitiated marks a close the agent performed by itself rather
  // than one the user asked for; see pickSuccessorTab.
  // fromServer marks a close applied FROM a daemon broadcast (e.g.
  // `closeSubagentTab`): it must not be echoed back as a `closeTab`
  // command, or two clients would bounce the close between each other.
  function closeTab(tabId, agentInitiated, fromServer) {
    const origIdx = tabs.findIndex(t => {
      return t.id === tabId;
    });
    if (origIdx < 0) return;
    if (tabs[origIdx].isContentTab) {
      closeContentTab(tabId);
      return;
    }
    const toClose = new Set([tabId]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const t of tabs) {
        if (t.parentTabId && toClose.has(t.parentTabId) && !toClose.has(t.id)) {
          toClose.add(t.id);
          grew = true;
        }
      }
    }
    const activeWasClosed = toClose.has(activeTabId);
    const closed = tabs[origIdx];
    for (const id of toClose) {
      const i = tabs.findIndex(t => t.id === id);
      if (i >= 0) tabs.splice(i, 1);
      forgetPendingFileLinks(id);
      // report-coverage:start
      discardReadyReports(id);
      // report-coverage:end
      if (!fromServer) api.closeTab({tabId: id});
    }
    rpAfterTabsClosed(toClose);
    if (activeWasClosed) {
      if (tabs.length === 0) {
        createNewTab();
        return;
      }
      activateAdjacentTab(pickSuccessorTab(closed, origIdx, agentInitiated));
    }
    reportSurvivingChatTab();
    renderTabBar();
    persistTabState();
  }

  let contentArea = null;
  let _monacoPromise = null;

  function ensureContentArea() {
    if (contentArea) return contentArea;
    contentArea = document.createElement('div');
    contentArea.id = 'content-tab-area';
    contentArea.style.display = 'none';
    const app = document.getElementById('app');
    const inputArea = document.getElementById('input-area');
    if (app) app.insertBefore(contentArea, inputArea || null);
    return contentArea;
  }

  function setChatSurfaceVisible(visible) {
    ['output', 'task-panel', 'input-area'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.style.display = visible ? '' : 'none';
    });
  }

  function showContentTab(tab) {
    const area = ensureContentArea();
    setChatSurfaceVisible(false);
    area.style.display = '';
    Array.from(area.children).forEach(v => {
      v.style.display = 'none';
    });
    if (tab.contentViewEl) tab.contentViewEl.style.display = '';
    if (tab.contentEditor && tab.contentEditor.layout) {
      try {
        tab.contentEditor.layout();
      } catch (_e) {}
    }
  }

  function hideContentArea() {
    if (contentArea) contentArea.style.display = 'none';
    setChatSurfaceVisible(true);
  }

  function activateAdjacentTab(newTab) {
    if (newTab.isContentTab) {
      activeTabId = newTab.id;
      showContentTab(newTab);
      return;
    }
    restoreTab(newTab);
    setRunningState(newTab.isRunning);
    if (!newTab.isRunning) {
      stopTimer();
      removeSpinner();
    }
    applyChevronState(currentTaskName);
    focusInputWithRetry();
  }

  function disposeTabContentView(tab) {
    if (tab.contentEditor) {
      try {
        tab.contentEditor.dispose();
      } catch (_e) {}
      tab.contentEditor = null;
    }
    if (tab.contentViewEl && tab.contentViewEl.parentNode) {
      tab.contentViewEl.parentNode.removeChild(tab.contentViewEl);
    }
    tab.contentViewEl = null;
  }

  function closeContentTab(tabId) {
    const idx = tabs.findIndex(t => {
      return t.id === tabId;
    });
    if (idx < 0) return;
    const tab = tabs[idx];
    tabs.splice(idx, 1);
    disposeTabContentView(tab);
    if (activeTabId === tabId) {
      if (tabs.length === 0) {
        hideContentArea();
        createNewTab();
        return;
      }
      const newIdx = Math.min(idx, tabs.length - 1);
      activateAdjacentTab(tabs[newIdx]);
    }
    renderTabBar();
    persistTabState();
  }

  function languageFromPath(lowerName) {
    const dot = lowerName.lastIndexOf('.');
    const ext = dot >= 0 ? lowerName.slice(dot + 1) : '';
    const map = {
      py: 'python',
      js: 'javascript',
      mjs: 'javascript',
      cjs: 'javascript',
      jsx: 'javascript',
      ts: 'typescript',
      tsx: 'typescript',
      json: 'json',
      md: 'markdown',
      css: 'css',
      scss: 'scss',
      less: 'less',
      html: 'html',
      htm: 'html',
      xml: 'xml',
      svg: 'xml',
      sh: 'shell',
      bash: 'shell',
      zsh: 'shell',
      yaml: 'yaml',
      yml: 'yaml',
      toml: 'ini',
      ini: 'ini',
      c: 'c',
      h: 'c',
      cpp: 'cpp',
      hpp: 'cpp',
      cc: 'cpp',
      java: 'java',
      go: 'go',
      rs: 'rust',
      rb: 'ruby',
      php: 'php',
      sql: 'sql',
      swift: 'swift',
      kt: 'kotlin',
      dockerfile: 'dockerfile',
      tex: 'latex',
    };
    return map[ext] || 'plaintext';
  }

  function ensureMonaco() {
    if (_monacoPromise) return _monacoPromise;
    _monacoPromise = new Promise((resolve, reject) => {
      if (window.monaco && window.monaco.editor) {
        resolve(window.monaco);
        return;
      }
      const base = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.52.2/min';
      const timer = setTimeout(() => {
        _monacoPromise = null;
        reject(new Error('Monaco load timeout'));
      }, 10000);
      const script = document.createElement('script');
      script.src = base + '/vs/loader.js';
      script.onload = () => {
        try {
          window.require.config({paths: {vs: base + '/vs'}});
          window.require(
            ['vs/editor/editor.main'],
            () => {
              clearTimeout(timer);
              resolve(window.monaco);
            },
            err => {
              clearTimeout(timer);
              _monacoPromise = null;
              reject(err);
            },
          );
        } catch (e) {
          clearTimeout(timer);
          _monacoPromise = null;
          reject(e);
        }
      };
      script.onerror = () => {
        clearTimeout(timer);
        _monacoPromise = null;
        reject(new Error('Monaco loader failed'));
      };
      document.head.appendChild(script);
    });
    return _monacoPromise;
  }

  function renderCodeContent(tab, holder, text, language) {
    ensureMonaco()
      .then(monaco => {
        if (!holder.isConnected || tab.contentEditor) return;
        tab.contentEditor = monaco.editor.create(holder, {
          value: text,
          language: language,
          readOnly: true,
          automaticLayout: true,
          minimap: {enabled: false},
          scrollBeyondLastLine: false,
          theme: 'vs-dark',
        });
      })
      .catch(() => {
        if (!holder.isConnected || holder.firstChild) return;
        const pre = document.createElement('pre');
        pre.className = 'content-code-fallback';
        const code = document.createElement('code');
        code.textContent = text;
        pre.appendChild(code);
        holder.appendChild(pre);
        try {
          if (window.hljs) window.hljs.highlightElement(code);
        } catch (_e) {}
      });
  }

  // ctxmenu-coverage:start
  // An opened .html file renders inside an iframe sandboxed with
  // `allow-scripts` only, i.e. an opaque origin the webview cannot script.
  // The Copy / Select All menu therefore has to be shipped *into* that
  // document: contentContextMenuBootstrapHtml() serialises the very same
  // implementation the parent document uses.
  function withContentContextMenu(html) {
    const api = window.ContentContextMenu;
    if (!api) return html;
    const boot = api.contentContextMenuBootstrapHtml();
    // The ORIGINAL string is searched case-insensitively: lower-casing is
    // not length preserving in Unicode (U+0130 becomes two UTF-16 units),
    // so an index taken from a lower-cased copy can land mid-tag.
    const re = /<\/body\s*>/gi;
    let at = -1;
    let m = re.exec(html);
    while (m) {
      at = m.index;
      m = re.exec(html);
    }
    if (at < 0) return html + boot;
    return html.slice(0, at) + boot + html.slice(at);
  }

  // The menu belongs to read-only content surfaces only — the tab that
  // shows an opened file or report.  Everywhere else (composer, settings,
  // panels, tab strip) the chat UI has its own menus and native editing
  // affordances, which this must never replace.
  function contentContextMenuAllowed(e) {
    if (e.defaultPrevented) return false;
    const target = e.target;
    if (!target || typeof target.closest !== 'function') return false;
    return !!target.closest('#content-tab-area');
  }

  function installParentContentContextMenu() {
    const api = window.ContentContextMenu;
    if (!api) return null;
    return api.installContentContextMenu(document, {
      shouldOpen: contentContextMenuAllowed,
    });
  }
  // ctxmenu-coverage:end

  function renderContentView(tab, ev) {
    const area = ensureContentArea();
    disposeTabContentView(tab);
    const view = document.createElement('div');
    view.className = 'content-tab-view';
    view.style.display = 'none';
    area.appendChild(view);
    tab.contentViewEl = view;
    const lower = (ev.name || '').toLowerCase();
    // report-coverage:start
    if (ev.isReport || lower.endsWith('.html') || lower.endsWith('.htm')) {
      // report-coverage:end
      const iframe = document.createElement('iframe');
      iframe.className = 'content-html-frame';
      iframe.setAttribute('sandbox', 'allow-scripts');
      // ctxmenu-coverage:start
      iframe.srcdoc = withContentContextMenu(ev.content || '');
      // ctxmenu-coverage:end
      view.appendChild(iframe);
      return;
    }
    const holder = document.createElement('div');
    holder.className = 'content-monaco-holder';
    view.appendChild(holder);
    renderCodeContent(tab, holder, ev.content || '', languageFromPath(lower));
  }

  // mayFocus tells whether this content tab is allowed to become the
  // active tab. It defaults to true because every caller but one acts on
  // a user request or a finished task; pass false to open the tab in the
  // background instead.
  function handleFileContent(ev, mayFocus) {
    if (mayFocus === undefined) mayFocus = true;
    if (ev.error) {
      // tableak-coverage:start
      // A background task's failed file open is that task's problem. Toasting
      // it would interrupt the conversation the user is actually reading.
      if (mayFocus) {
        updateNotification({
          id: 'file-open-error',
          message: ev.error,
          severity: 'error',
        });
      }
      // tableak-coverage:end
      return;
    }
    const path = ev.path || '';
    const existing = tabs.find(t => {
      return t.isContentTab && t.contentPath === path;
    });
    if (existing) {
      renderContentView(existing, ev);
      if (activeTabId === existing.id) showContentTab(existing);
      else if (mayFocus) switchToTab(existing.id);
      return;
    }
    const tab = makeTab(ev.name || path || 'file');
    tab.isContentTab = true;
    tab.contentPath = path;
    tabs.push(tab);
    renderContentView(tab, ev);
    if (mayFocus) switchToTab(tab.id);
    else renderTabBar();
  }

  // report-coverage:start
  function reportPathInfo(p) {
    const raw = String(p || '').split(/[\\/]/);
    const segs = [];
    for (let i = 0; i < raw.length; i++) {
      const s = raw[i];
      if (s === '' || s === '.') continue;
      if (s === '..') segs.pop();
      else segs.push(s);
    }
    const file = segs.pop() || '';
    const dot = file.lastIndexOf('.');
    const ext = dot >= 0 ? file.slice(dot + 1).toLowerCase() : '';
    const isMarkdown = ext === 'md' || ext === 'markdown';
    if (!isMarkdown && ext !== 'html' && ext !== 'htm') return null;
    const inReports = segs.some(s => {
      return s.toLowerCase() === 'reports';
    });
    if (!inReports) return null;
    return {name: file, isMarkdown: isMarkdown};
  }

  function markdownReportToHtml(text) {
    let body = null;
    if (typeof marked !== 'undefined') {
      try {
        body = marked.parse(text || '');
      } catch (_e) {
        body = null;
      }
    }
    if (body === null) body = '<pre>' + esc(text || '') + '</pre>';
    return (
      '<!DOCTYPE html><html><head><meta charset="utf-8"><style>' +
      'body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",' +
      'Helvetica,Arial,sans-serif;line-height:1.6;color:#24292f;' +
      'background:#fff;max-width:860px;margin:0 auto;padding:2em 1.5em}' +
      'pre{background:#f6f8fa;padding:12px;border-radius:6px;' +
      'overflow-x:auto}code{background:#f6f8fa;padding:.1em .3em;' +
      'border-radius:4px}pre code{background:none;padding:0}' +
      'table{border-collapse:collapse}td,th{border:1px solid #d0d7de;' +
      'padding:5px 10px}img{max-width:100%}' +
      'blockquote{border-left:4px solid #d0d7de;margin-left:0;' +
      'padding-left:1em;color:#57606a}' +
      '</style></head><body>' +
      body +
      '</body></html>'
    );
  }

  function stashPendingReport(tState, ev) {
    tState.pendingReport = null;
    if (ev.name !== 'Write' || !ev.path) return;
    const info = reportPathInfo(ev.path);
    if (!info) return;
    tState.pendingReport = {
      path: ev.path,
      name: info.name,
      isMarkdown: info.isMarkdown,
      content: ev.content || '',
    };
  }

  // Reports confirmed by a successful Write, keyed by owning tab id and
  // kept in write order; their tabs open only when that task finishes.
  const readyReportsByTab = Object.create(null);

  function reportTabKey(evTabId) {
    const key =
      evTabId === undefined || evTabId === null ? activeTabId : evTabId;
    return String(key);
  }

  function confirmReadyReport(tState, ev) {
    const rep = tState.pendingReport;
    tState.pendingReport = null;
    if (!rep || tState.suppressReportOpen) return;
    if (ev.tool_name !== 'Write') return;
    if (ev.path && ev.path !== rep.path) return;
    const rc = String(ev.content || '');
    if (rc.lastIndexOf('Successfully wrote ', 0) !== 0) return;
    const key = reportTabKey(ev.tabId);
    const list = readyReportsByTab[key] || (readyReportsByTab[key] = []);
    for (let i = list.length - 1; i >= 0; i--) {
      if (list[i].path === rep.path) list.splice(i, 1);
    }
    list.push(rep);
  }

  function discardReadyReports(evTabId) {
    delete readyReportsByTab[reportTabKey(evTabId)];
  }

  // mayFocus defaults to true: a task that reached a terminal event may
  // put its report on screen. Pass false when the owning work is not the
  // user's task finishing (see the subagentDone case).
  function openReadyReportTabs(evTabId, mayFocus) {
    if (mayFocus === undefined) mayFocus = true;
    const key = reportTabKey(evTabId);
    const reps = readyReportsByTab[key];
    delete readyReportsByTab[key];
    if (!reps) return;
    reps.forEach(rep => {
      handleFileContent(
        {
          path: rep.path,
          name: rep.name,
          content: rep.isMarkdown
            ? markdownReportToHtml(rep.content)
            : rep.content,
          isReport: true,
        },
        mayFocus,
      );
    });
  }
  // report-coverage:end

  const tabCtxMenu = document.createElement('div');
  tabCtxMenu.id = 'tab-context-menu';
  document.body.appendChild(tabCtxMenu);

  function closeTabContextMenu() {
    tabCtxMenu.classList.remove('open');
  }

  function showTabContextMenu(x, y, tabId) {
    tabCtxMenu.innerHTML = '';
    const items = [
      {
        label: 'Close',
        action: function () {
          closeTab(tabId);
        },
      },
      {
        label: 'Close Others',
        action: function () {
          const ids = tabs
            .filter(t => {
              return t.id !== tabId;
            })
            .map(t => {
              return t.id;
            });
          if (tabId !== activeTabId) switchToTab(tabId);
          ids.forEach(id => {
            closeTab(id);
          });
        },
      },
      {
        label: 'Close All',
        action: function () {
          const ids = tabs.map(t => {
            return t.id;
          });
          ids.forEach(id => {
            closeTab(id);
          });
        },
      },
      {
        label: 'Close Inactive',
        action: function () {
          const ids = tabs
            .filter(t => {
              return !t.isRunning;
            })
            .map(t => {
              return t.id;
            });
          ids.forEach(id => {
            closeTab(id);
          });
        },
      },
    ];
    items.forEach(item => {
      const el = document.createElement('div');
      el.className = 'tab-ctx-item';
      el.textContent = item.label;
      el.addEventListener('click', () => {
        closeTabContextMenu();
        item.action();
      });
      tabCtxMenu.appendChild(el);
    });
    tabCtxMenu.classList.add('open');
    const mw = tabCtxMenu.offsetWidth;
    const mh = tabCtxMenu.offsetHeight;
    const px = Math.min(x, window.innerWidth - mw - 4);
    const py = Math.min(y, window.innerHeight - mh - 4);
    tabCtxMenu.style.left = Math.max(0, px) + 'px';
    tabCtxMenu.style.top = Math.max(0, py) + 'px';
  }

  document.addEventListener('click', () => {
    closeTabContextMenu();
  });

  const BLUR_AFTER_CLICK_SELECTOR = [
    '#menu-btn',
    '#model-btn',
    '#upload-btn',
    '#tricks-btn',
    '#voice-btn',
    '#send-btn',
    '#stop-btn',
    '.chat-tab-add',
    '.chat-tab-settings',
    '.chat-tab-close',
    '#input-clear-btn',
    '.search-clear-btn',
    '#task-panel-drawer-btn',
    '#input-drawer-btn',
  ].join(', ');
  document.addEventListener(
    'click',
    e => {
      if (!e.target || typeof e.target.closest !== 'function') return;
      const btn = e.target.closest(BLUR_AFTER_CLICK_SELECTOR);
      if (btn && typeof btn.blur === 'function') btn.blur();
    },
    true,
  );
  document.addEventListener('contextmenu', e => {
    if (
      !e.target.closest('#tab-context-menu') &&
      !e.target.closest('.chat-tab')
    ) {
      closeTabContextMenu();
    }
  });
  // ctxmenu-coverage:start
  installParentContentContextMenu();
  // ctxmenu-coverage:end
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeTabContextMenu();
  });

  function createBackgroundSubagentTab(parentId) {
    const subTab = makeTab('new chat');
    if (parentId) subTab.parentTabId = parentId;
    subTab.isSubagentTab = true;
    placeSubagentTabAfterParent(subTab, parentId);
    renderTabBar();
    persistTabState();
    return subTab;
  }

  function createNewTab() {
    // Opening a chat is the user taking over: the launch is over, and no
    // backend event may move them off the tab they just asked for.
    closeLaunchSwitch();
    const pendingText = inp.value || '';
    saveCurrentTab();
    const tab = makeTab('new chat');
    tab.inputValue = pendingText;
    tabs.push(tab);
    activeTabId = tab.id;
    restoreTab(tab);
    renderTabBar();
    persistTabState();
    setRunningState(tab.isRunning);
    if (!tab.isRunning) {
      t0 = null;
      stopTimer();
      removeSpinner();
    }
    registerTab(tab);
    api.newChat({tabId: tab.id});
    api.getWelcomeSuggestions();
    focusInputWithRetry();
  }

  function updateActiveTabTitle(title) {
    const tab = getTab(activeTabId);
    if (!tab) return;
    const t = (title || '').trim();
    tab.title = t
      ? t.length > 30
        ? t.substring(0, 30) + '\u2026'
        : t
      : 'new chat';
    renderTabBar();
    persistTabState();
  }

  // The tab SET is server-canonical (the daemon's shared tab registry,
  // mirrored to every client via `tabs_state`), so the only webview
  // state persisted locally is what stays client-local by design: the
  // selected tab and the drawer preferences.
  function persistTabState() {
    vscode.setState({
      chatId: activeTabId,
      taskDrawerCollapsed: taskDrawerCollapsed,
      inputDrawerCollapsed: inputDrawerCollapsed,
      taskDrawerUserSet: taskDrawerUserSet,
      inputDrawerUserSet: inputDrawerUserSet,
      drawersVersion: DRAWERS_VERSION,
    });
  }

  // tabmirror-coverage:start
  // Tab ids announced to the daemon in an `openTab` whose `tabs_state`
  // echo has not arrived yet, mapped to how many snapshots have since
  // arrived WITHOUT the id. A snapshot broadcast inside that window
  // predates the registration and must not remove the brand-new tab —
  // but an id no snapshot ever confirms (e.g. the daemon rejected or
  // dropped the open) must not stay snapshot-immune forever, so it
  // expires after PENDING_OPEN_MAX_MISSES missed snapshots.
  const pendingOpenTabs = new Map();
  const PENDING_OPEN_MAX_MISSES = 3;

  // Announce a locally created chat tab to the daemon's shared tab
  // registry so every other client opens the same tab.
  function registerTab(tab) {
    pendingOpenTabs.set(tab.id, 0);
    api.openTab({
      tabId: tab.id,
      title: tab.title || 'new chat',
      workDir: tab.workDir || '',
    });
  }

  function clipTabTitle(title) {
    const t = (title || '').trim();
    if (!t) return 'new chat';
    return t.length > 30 ? t.substring(0, 30) + '\u2026' : t;
  }

  // Reconcile the local tab bar against the canonical `tabs_state`
  // snapshot: adopt tabs other clients opened, drop tabs they closed,
  // and follow the registry's titles, order and chat bindings.
  //
  // Client-local state survives untouched: the active-tab selection,
  // and each tab's own composer draft, model pick and task panel
  // (mirroring covers the tab SET and transcript CONTENTS, not what
  // the user is typing). Two kinds of tabs stay client-local by
  // design and are never removed here: sub-agent tabs (derived state,
  // rebuilt on every client from `openSubagentTab` broadcasts and
  // replays; they re-anchor after their parent) and content tabs
  // (the remote web app's stand-in for the VS Code editor — editors
  // are per-user surfaces on every client, so file views are not
  // mirrored).
  function reconcileTabs(list) {
    const byId = new Map(
      tabs.map(t => {
        return [t.id, t];
      }),
    );
    const inSnapshot = new Set();
    const next = [];
    // One tab per chat: the daemon's registry enforces the invariant,
    // and this backstop drops any duplicate chat binding a legacy or
    // buggy snapshot might still carry (keep-first, deterministic on
    // every client).
    const seenChats = new Set();
    list.forEach(e => {
      if (!e || !e.tabId || inSnapshot.has(e.tabId)) return;
      // The daemon listed the id, so its `openTab` is confirmed —
      // clear the pending shield even when the entry is dropped as a
      // duplicate below, or the local duplicate tab would survive
      // reconciliation until the shield expires.
      pendingOpenTabs.delete(e.tabId);
      if (e.chatId) {
        if (seenChats.has(String(e.chatId))) return;
        seenChats.add(String(e.chatId));
      }
      inSnapshot.add(e.tabId);
      let tab = byId.get(e.tabId);
      if (!tab) {
        tab = makeTab(clipTabTitle(e.title));
        tab.id = e.tabId;
        if (e.chatId) tab.hasRunTask = true;
      } else if (!tab.isSubagentTab && e.title) {
        tab.title = clipTabTitle(e.title);
      }
      // Once the registry has listed a tab, its removal from a later
      // snapshot means another client closed it.
      tab.inRegistry = true;
      if (e.chatId && String(tab.backendChatId || '') !== String(e.chatId)) {
        tab.backendChatId = String(e.chatId);
      }
      if (e.workDir && !tab.workDir) tab.workDir = e.workDir;
      next.push(tab);
    });

    // Expire pending opens the daemon never confirmed: after a few
    // snapshots without the id, the open is considered lost/rejected
    // and the id stops shielding its local tab from reconciliation.
    pendingOpenTabs.forEach((misses, id) => {
      if (inSnapshot.has(id)) return;
      if (misses + 1 >= PENDING_OPEN_MAX_MISSES) pendingOpenTabs.delete(id);
      else pendingOpenTabs.set(id, misses + 1);
    });

    // A removed registry tab takes its local sub-agent descendants
    // with it, exactly like a local close would. An EMPTY snapshot
    // spares tabs the registry never listed (the boot placeholder):
    // there is nothing to mirror, so destroying and recreating the
    // placeholder would only lose its identity and composer draft.
    const removedIds = new Set();
    const snapshotEmpty = inSnapshot.size === 0;
    tabs.forEach(t => {
      if (
        !inSnapshot.has(t.id) &&
        !t.isSubagentTab &&
        !t.isContentTab &&
        !pendingOpenTabs.has(t.id) &&
        !(snapshotEmpty && !t.inRegistry)
      ) {
        removedIds.add(t.id);
      }
    });
    let grew = removedIds.size > 0;
    while (grew) {
      grew = false;
      tabs.forEach(t => {
        if (
          t.parentTabId &&
          removedIds.has(t.parentTabId) &&
          !removedIds.has(t.id)
        ) {
          removedIds.add(t.id);
          grew = true;
        }
      });
    }
    tabs.forEach(t => {
      if (inSnapshot.has(t.id) || removedIds.has(t.id)) return;
      next.push(t);
    });
    removedIds.forEach(id => {
      const doomed = byId.get(id);
      if (doomed && doomed.isContentTab) disposeTabContentView(doomed);
      forgetPendingFileLinks(id);
      // report-coverage:start
      discardReadyReports(id);
      // report-coverage:end
    });
    tabs = next;
    if (removedIds.size > 0) rpAfterTabsClosed(removedIds);
    tabs.forEach(t => {
      if (t.isSubagentTab && t.parentTabId) {
        placeSubagentTabAfterParent(t, t.parentTabId);
      }
    });

    if (tabs.length === 0) {
      // Empty registry: keep one local, unregistered placeholder so
      // the composer always exists. The daemon adopts it the moment
      // it runs a task; until then it is a welcome screen only.
      tabs.push(makeTab('new chat'));
    }
    if (!getTab(activeTabId)) {
      const saved = savedActiveTabId ? getTab(savedActiveTabId) : null;
      activateAdjacentTab(saved || tabs[0]);
    }
    savedActiveTabId = '';
    reportSurvivingChatTab();
    renderTabBar();
    persistTabState();
  }
  // tabmirror-coverage:end

  // drawer-coverage:start
  function isMobileRemoteWebApp() {
    if (!document.body.classList.contains('remote-chat')) return false;
    const uaData = navigator.userAgentData;
    if (uaData && uaData.mobile === true) return true;
    const ua = navigator.userAgent || '';
    if (/Android|iPhone|iPad|iPod|Mobile|IEMobile|Opera Mini/i.test(ua)) {
      return true;
    }
    return /Macintosh/i.test(ua) && navigator.maxTouchPoints > 1;
  }

  const DRAWERS_VERSION = 3;
  const isMobileRemote = isMobileRemoteWebApp();
  // The static task panel opens collapsed in every chat webview -- the
  // extension sidebar and the remote web app, phone or desktop. It is a
  // header, not content: the transcript deserves the room. Only a click on
  // #task-panel-drawer-btn may expand it, so an expanded panel is restored
  // only when the persisted blob says a click put it there. `*UserSet`
  // records that click; without it the defaults below always win, which is
  // what keeps a reload, a reconnect or a new task from re-expanding a
  // panel the user never opened.
  let taskDrawerCollapsed = true;
  let taskDrawerUserSet = false;
  // The composer stays reachable by default. On a phone it folds away while
  // a task is running (see syncMobileInputDrawer) because the transcript
  // needs the whole screen, but with nothing running the textbox and its
  // buttons are the only thing worth showing.
  let inputDrawerCollapsed = false;
  let inputDrawerUserSet = false;
  {
    const _saved = vscode.getState();
    const _drawersTrusted =
      _saved &&
      typeof _saved === 'object' &&
      _saved.drawersVersion >= DRAWERS_VERSION;
    if (_drawersTrusted && _saved.taskDrawerUserSet) {
      taskDrawerCollapsed = !!_saved.taskDrawerCollapsed;
      taskDrawerUserSet = true;
    }
    if (_drawersTrusted && _saved.inputDrawerUserSet) {
      inputDrawerCollapsed = !!_saved.inputDrawerCollapsed;
      inputDrawerUserSet = true;
    }
  }
  // drawer-coverage:end

  {
    const _initialModelEl = document.getElementById('model-name');
    if (_initialModelEl && _initialModelEl.textContent) {
      selectedModel = _initialModelEl.textContent;
    }
  }

  // Tabs come from the server's shared registry (`tabs_state`), never
  // from local storage. The boot placeholder below is replaced by the
  // first snapshot; `savedActiveTabId` restores this client's own tab
  // selection (selection stays client-local) once that snapshot lands.
  // `legacyRestoredTabs` carries a pre-registry client's locally
  // persisted tab set into `ready` exactly once, so the first daemon
  // with an empty registry can adopt it (one-time migration).
  let savedActiveTabId = '';
  const legacyRestoredTabs = [];
  (function () {
    const saved = vscode.getState();
    if (saved && saved.tabs && saved.tabs.length > 0) {
      const seenChatIds = new Set();
      saved.tabs.forEach(st => {
        if (!st || st.isSubagentTab) return;
        const chatId = st.backendChatId ? String(st.backendChatId) : '';
        if (!st.chatId || !chatId || seenChatIds.has(chatId)) return;
        seenChatIds.add(chatId);
        legacyRestoredTabs.push({
          tabId: String(st.chatId),
          chatId: chatId,
          title: st.title || '',
          workDir: st.workDir || '',
        });
      });
    }
    if (saved && saved.chatId) savedActiveTabId = String(saved.chatId);
    const initial = makeTab('new chat');
    tabs.push(initial);
    activeTabId = initial.id;
  })();

  const O = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  const inp = document.getElementById('task-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const uploadBtn = document.getElementById('upload-btn');

  const modelBtn = document.getElementById('model-btn');
  const modelDropdown = document.getElementById('model-dropdown');
  const modelSearch = document.getElementById('model-search');
  const modelList = document.getElementById('model-list');
  const modelName = document.getElementById('model-name');
  const fileChips = document.getElementById('file-chips');

  const statusText = document.getElementById('status-text');
  const menuBtn = document.getElementById('menu-btn');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const sidebarClose = document.getElementById('sidebar-close');
  const historySearch = document.getElementById('history-search');
  const modelSearchClear = document.getElementById('model-search-clear');
  const historySearchClear = document.getElementById('history-search-clear');
  const historyList = document.getElementById('history-list');
  const autocomplete = document.getElementById('autocomplete');
  const askUserModal = document.getElementById('ask-user-modal');
  const askUserSlot = document.getElementById('ask-user-slot');

  const settingsPanel = document.getElementById('settings-panel');
  const settingsOverlay = document.getElementById('settings-overlay');
  const settingsPanelClose = document.getElementById('settings-panel-close');
  const frequentPanel = document.getElementById('frequent-panel');
  const frequentOverlay = document.getElementById('frequent-overlay');
  const frequentPanelClose = document.getElementById('frequent-panel-close');
  const frequentTasksBtn = document.getElementById('frequent-tasks-btn');
  const frequentList = document.getElementById('frequent-list');
  const tricksPanel = document.getElementById('tricks-panel');
  const tricksOverlay = document.getElementById('tricks-overlay');
  const tricksPanelClose = document.getElementById('tricks-panel-close');
  const tricksBtn = document.getElementById('tricks-btn');
  const tricksList = document.getElementById('tricks-list');
  const waitSpinner = document.getElementById('wait-spinner');
  const ghostOverlay = document.getElementById('ghost-overlay');
  const inputContainer = document.getElementById('input-container');
  const inputClearBtn = document.getElementById('input-clear-btn');
  const worktreeToggleBtn = document.getElementById('cfg-use-worktree');
  const autocommitBtn = document.getElementById('autocommit-btn');
  const updateBtn = document.getElementById('cfg-update-btn');
  const serverResetBtn = document.getElementById('cfg-server-reset-btn');
  const serverResetConfirmModal = document.getElementById(
    'server-reset-confirm-modal',
  );
  const serverResetConfirmOkBtn = document.getElementById(
    'server-reset-confirm-ok',
  );
  const serverResetConfirmCancelBtn = document.getElementById(
    'server-reset-confirm-cancel',
  );
  const autocommitToggleBtn = document.getElementById('cfg-auto-commit');
  const taskPanel = document.getElementById('task-panel');
  const taskPanelText = document.getElementById('task-panel-text');
  const taskPanelCopy = document.getElementById('task-panel-copy');
  const taskPanelDrawerBtn = document.getElementById('task-panel-drawer-btn');
  const inputDrawerBtn = document.getElementById('input-drawer-btn');
  const inputAreaEl = document.getElementById('input-area');
  const statusTokens = document.getElementById('status-tokens');
  const statusBudget = document.getElementById('status-budget');
  const statusSteps = document.getElementById('status-steps');

  // The welcome screen lives inside the scrolling chat container, so
  // whatever scroll offset the previous content left behind (a finished
  // conversation is parked at its bottom) would otherwise hide the
  // greeting and the first suggestions.  Showing the welcome screen
  // always lands it at the top.
  function showWelcomeScreen() {
    if (!welcome) return;
    welcome.style.display = '';
    if (!O.contains(welcome)) O.appendChild(welcome);
    refreshWelcomeLayout();
    O.scrollTop = 0;
  }

  function refreshWelcomeLayout() {
    if (!document.body.classList.contains('remote-chat')) return;
    const ia = document.getElementById('input-area');
    const app = document.getElementById('app');
    if (!ia || !app || !welcome) return;
    if (ia.parentNode === welcome) {
      const sbar = document.getElementById('sidebar');
      if (sbar) app.insertBefore(ia, sbar);
      else app.appendChild(ia);
    }
  }

  refreshWelcomeLayout();

  function setTaskText(text) {
    if (!taskPanel || !taskPanelText) return;
    const t = (text || '').trim();
    if (t) {
      taskPanelText.textContent = t;
      taskPanelText.setAttribute('data-tooltip', t);
      taskPanel.classList.add('visible');
    } else {
      taskPanelText.textContent = '';
      taskPanelText.removeAttribute('data-tooltip');
      taskPanel.classList.remove('visible');
    }
  }

  // chevron-coverage:start
  function applyChevronState(taskName) {
    if (!O) return;
    const panels = O.querySelectorAll('.collapsible');
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      const adjacentContainer = p.closest('.adjacent-task');
      const inAdjacent = !!adjacentContainer;
      const inRunning = isRunning && !inAdjacent;
      const panelTask = inAdjacent
        ? adjacentContainer.dataset.task || ''
        : currentTaskName;
      if (taskName && panelTask !== taskName) continue;
      if (inRunning || p.classList.contains('rc')) {
        p.classList.remove('chv-hidden');
        continue;
      }
      if (p.classList.contains('tc-summary')) {
        p.classList.remove('chv-hidden');
        if (!p.classList.contains('user-pinned')) p.classList.add('collapsed');
        if (p.classList.contains('collapsed')) collapseNestedRunParallel(p);
        continue;
      }
      if (p.closest('.summary-sub')) {
        p.classList.remove('chv-hidden');
        continue;
      }
      p.classList.add('chv-hidden');
      if (p.classList.contains('tc-run-parallel')) {
        p.classList.add('collapsed');
        p.classList.remove('user-pinned');
        collapsePreview(p);
        syncRunParallelPanel(p);
      } else {
        // A hidden panel takes any fan-out panel it swallowed off
        // screen with it, so those sub-agent tabs must close too.
        collapseNestedRunParallel(p);
      }
    }
  }
  // chevron-coverage:end

  // drawer-coverage:start
  function applyDrawerState() {
    if (taskPanel && taskPanelDrawerBtn) {
      taskPanel.classList.toggle('drawer-collapsed', taskDrawerCollapsed);
      taskPanelDrawerBtn.setAttribute(
        'aria-expanded',
        taskDrawerCollapsed ? 'false' : 'true',
      );
      const taskLabel = taskDrawerCollapsed
        ? 'Expand task panel'
        : 'Collapse task panel';
      taskPanelDrawerBtn.setAttribute('aria-label', taskLabel);
      taskPanelDrawerBtn.removeAttribute('data-tooltip');
    }
    if (inputAreaEl && inputDrawerBtn) {
      inputAreaEl.classList.toggle('drawer-collapsed', inputDrawerCollapsed);
      inputDrawerBtn.setAttribute(
        'aria-expanded',
        inputDrawerCollapsed ? 'false' : 'true',
      );
      const inputLabel = inputDrawerCollapsed
        ? 'Expand input panel'
        : 'Collapse input panel';
      inputDrawerBtn.setAttribute('aria-label', inputLabel);
      inputDrawerBtn.removeAttribute('data-tooltip');
    }
  }

  if (taskPanelDrawerBtn) {
    taskPanelDrawerBtn.addEventListener('click', e => {
      e.stopPropagation();
      taskDrawerCollapsed = !taskDrawerCollapsed;
      taskDrawerUserSet = true;
      applyDrawerState();
      persistTabState();
    });
  }
  if (inputDrawerBtn) {
    inputDrawerBtn.addEventListener('click', e => {
      e.stopPropagation();
      inputDrawerCollapsed = !inputDrawerCollapsed;
      inputDrawerUserSet = true;
      applyDrawerState();
      persistTabState();
    });
  }
  applyDrawerState();

  // A phone screen holds either the transcript or the composer, not both.
  // While a task runs the transcript wins; the moment nothing is running the
  // input textbox and its buttons come back so the user can start the next
  // task. Once the user works the handle themselves that choice is final.
  function syncMobileInputDrawer() {
    if (!isMobileRemote || inputDrawerUserSet) return;
    const wantCollapsed = tabs.some(isLaunchRunning);
    if (inputDrawerCollapsed === wantCollapsed) return;
    inputDrawerCollapsed = wantCollapsed;
    applyDrawerState();
    persistTabState();
  }
  // drawer-coverage:end

  // launchswitch-coverage:start
  // A chat window is often opened while agents are still working: the
  // daemon replays every chat-bound tab of the shared registry after
  // `ready`, including a `status` for each one that is still running.
  // What the user wants to see then is the task that started last, not
  // whichever tab happened to be active when the window was last closed.
  //
  // The window this permission lives in closes at the first real gesture --
  // a tap or a keystroke -- so a snapshot that arrives while the user is
  // already working can never yank them off the transcript they are reading.
  // The wall-clock bound closes it for a window that is simply left alone, so
  // a task started much later cannot steal a tab either.
  const LAUNCH_SWITCH_WINDOW_MS = 15000;
  let launchStartedAt = 0;
  let launchSwitchDone = false;
  let launchNewsSeen = false;

  // A launch begins when the backend becomes live, which is not the moment the
  // page loads: until then the chat is hidden behind the "KISS Sorcar Server
  // is starting ..." overlay, so nothing the user did to it counted and the
  // wall-clock bound would be measuring the wait rather than the launch.
  //
  // Every such transition restarts the launch, because every one of them ends
  // a spell with the chat off screen. That is what carries the remote web app
  // through its password prompt: the socket shim reports a live backend once
  // to reveal the prompt (the modal lives inside #app) and again once the
  // password is accepted -- and only then does it let `ready`, and the
  // running-task news it triggers, through. Without the restart the keystrokes
  // spent on the prompt would spend the launch they precede.
  //
  // Once news HAS arrived the launch has had its chance, and a later hiccup --
  // a daemon that dies and comes back mid-session -- must not hand it a second
  // one: by then the user has picked the tab they want to be on.
  function beginLaunch() {
    if (launchNewsSeen) return;
    launchStartedAt = Date.now();
    launchSwitchDone = false;
  }

  function closeLaunchSwitch() {
    // A tap on the loading overlay or the password prompt is not the user
    // taking over the chat -- the chat is not even on screen yet.
    if (!launchStartedAt) return;
    launchSwitchDone = true;
  }

  function launchSwitchAllowed() {
    if (launchSwitchDone || !launchStartedAt) return false;
    if (Date.now() - launchStartedAt > LAUNCH_SWITCH_WINDOW_MS) {
      closeLaunchSwitch();
      return false;
    }
    return true;
  }

  // A tab counts for the launch only while its own replayed `status`
  // says it is running; sub-agent and content tabs are implementation
  // details of some chat tab, never launch targets themselves.
  function isLaunchRunning(tab) {
    return !tab.isContentTab && !tab.isSubagentTab && !!tab.isRunning;
  }

  // Ties -- two tasks whose start timestamp is missing, so both read 0 --
  // resolve to the later tab, because the backend hands out its running
  // tasks oldest first and they are opened in that order.
  function switchToLatestRunningTab() {
    if (!launchSwitchAllowed()) return;
    launchNewsSeen = true;
    let best = null;
    let bestTs = -1;
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      if (!isLaunchRunning(tab)) continue;
      const ts = Number(tab.t0) || 0;
      if (ts >= bestTs) {
        bestTs = ts;
        best = tab;
      }
    }
    if (best && best.id !== activeTabId) switchToTab(best.id);
  }
  // launchswitch-coverage:end

  function fallbackCopyText(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      document.execCommand('copy');
      ok = true;
    } catch {}
    document.body.removeChild(ta);
    return ok;
  }

  if (taskPanelCopy && taskPanelText) {
    let copyResetTimer = null;
    taskPanelCopy.addEventListener('click', async e => {
      e.stopPropagation();
      const text = (taskPanelText.textContent || '').trim();
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
      } catch {
        fallbackCopyText(text);
      }
      const iconCopy = taskPanelCopy.querySelector('.icon-copy');
      const iconCheck = taskPanelCopy.querySelector('.icon-check');
      if (iconCopy && iconCheck) {
        iconCopy.style.display = 'none';
        iconCheck.style.display = '';
      }
      taskPanelCopy.classList.add('copied');
      if (copyResetTimer) clearTimeout(copyResetTimer);
      copyResetTimer = setTimeout(() => {
        if (iconCopy && iconCheck) {
          iconCopy.style.display = '';
          iconCheck.style.display = 'none';
        }
        taskPanelCopy.classList.remove('copied');
      }, 1500);
    });
  }

  function syncClearBtn() {
    if (inputClearBtn) inputClearBtn.style.display = inp.value ? '' : 'none';
  }

  let state = mkS();
  let lastToolName = '';
  let llmPanel = null;
  let llmPanelState = mkS();
  let pendingPanel = false;
  let stepCount = 0;

  let t0 = null;
  let timerIv = null;
  let _spinnerTimer = null;

  function mkS() {
    return {
      thinkEl: null,
      thinkCnt: null,
      thinkBuf: '',
      thinkRaf: 0,
      txtEl: null,
      txtBuf: '',
      txtNode: null,
      txtPending: '',
      txtRaf: 0,
      bashPanel: null,
      bashBuf: '',
      bashRaf: 0,
      lastToolCallEl: null,
      pendingReport: null,
    };
  }

  function resetOutputState() {
    state = mkS();
    llmPanel = null;
    llmPanelState = mkS();
    lastToolName = '';
    pendingPanel = false;
    stepCount = 0;
  }

  function resetAdjacentState() {
    adjacentLoading = false;
    oldestLoadedTaskId = currentTaskId;
    newestLoadedTaskId = currentTaskId;
    noPrevTask = false;
    noNextTask = false;
    overscrollAccum = 0;
    overscrollDir = '';
    if (overscrollTimer) {
      clearTimeout(overscrollTimer);
      overscrollTimer = null;
    }
    // taskwheel-coverage:start
    taskWheelPendingDir = '';
    taskWheelLastTarget = null;
    taskWheelAccum = 0;
    taskWheelDir = '';
    if (taskWheelTimer) {
      clearTimeout(taskWheelTimer);
      taskWheelTimer = null;
    }
    // taskwheel-coverage:end
  }

  function showAdjacentLoader(direction) {
    removeAdjacentLoader();
    const loader = mkEl('div', 'adjacent-loader');
    loader.id = 'adjacent-loader';
    loader.textContent =
      'Loading ' + (direction === 'prev' ? 'previous' : 'next') + ' task…';
    if (direction === 'prev') {
      O.insertBefore(loader, O.firstChild);
    } else {
      O.appendChild(loader);
    }
  }

  function removeAdjacentLoader() {
    const el = document.getElementById('adjacent-loader');
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  /**
   * Splice a neighbouring task's transcript into the visible #output.
   *
   * ownerTabId names the tab the transcript belongs to. It is threaded into
   * the replay so file links are cached and stamped against that tab rather
   * than against whichever tab happens to be on screen.
   */
  function renderAdjacentTask(direction, task, events, taskId, ownerTabId) {
    removeAdjacentLoader();
    adjacentLoading = false;
    // taskwheel-coverage:start
    const wheelScrollPending = taskWheelPendingDir === direction;
    taskWheelPendingDir = '';
    // taskwheel-coverage:end

    const hasTaskId = taskId !== undefined && taskId !== null && taskId !== '';
    if (!hasTaskId && !task) {
      if (direction === 'prev') noPrevTask = true;
      else noNextTask = true;
      return;
    }

    const taskLabel = task || '(untitled task)';

    const container = mkEl('div', 'adjacent-task');
    container.dataset.task = taskLabel;
    if (hasTaskId) container.dataset.taskId = String(taskId);

    const savedTokens = statusTokens ? statusTokens.textContent : '';
    const savedBudget = statusBudget ? statusBudget.textContent : '';
    const savedSteps = statusSteps ? statusSteps.textContent : '';
    // visibletask-coverage:start
    // The replay below renders a neighbour's transcript through the very
    // renderers the live stream uses, so it walks the live task's step
    // counter and metrics along with it. They are put back afterwards.
    const savedMetrics = currentTaskMetrics;
    const savedStepCount = stepCount;
    const savedVisibleTab = activeTabId;
    currentTaskMetrics = {tokens: '', budget: '', steps: ''};
    // visibletask-coverage:end
    if (events && events.length > 0) {
      // tableak-coverage:start
      replayEventsInto(container, events, {
        ownerTabId: ownerTabId || activeTabId,
      });
      // tableak-coverage:end
    }
    if (!container.firstChild) {
      const ph = mkEl('div', 'adjacent-task-placeholder');
      ph.textContent = taskLabel + ' — (no output recorded)';
      container.appendChild(ph);
    }
    container.dataset.metricTokens = statusTokens
      ? statusTokens.textContent
      : '';
    container.dataset.metricBudget = statusBudget
      ? statusBudget.textContent
      : '';
    container.dataset.metricSteps = statusSteps ? statusSteps.textContent : '';
    if (statusTokens) statusTokens.textContent = savedTokens;
    if (statusBudget) statusBudget.textContent = savedBudget;
    if (statusSteps) statusSteps.textContent = savedSteps;
    // visibletask-coverage:start
    // Same rule as everywhere else: if the replay swapped the tab on
    // screen, its numbers are already up and must be left alone.
    if (activeTabId === savedVisibleTab) {
      currentTaskMetrics = savedMetrics;
      stepCount = savedStepCount;
    }
    // visibletask-coverage:end

    if (direction === 'prev') {
      const prevScrollHeight = O.scrollHeight;
      O.insertBefore(container, O.firstChild);
      const newScrollHeight = O.scrollHeight;
      O.scrollTop += newScrollHeight - prevScrollHeight;
      if (hasTaskId) oldestLoadedTaskId = taskId;
    } else {
      O.appendChild(container);
      if (hasTaskId) newestLoadedTaskId = taskId;
    }
    applyChevronState(taskLabel);
    // taskwheel-coverage:start
    if (wheelScrollPending)
      scrollTaskRegionToTop({
        task: taskLabel,
        first: container,
        last: container,
      });
    // taskwheel-coverage:end
    // Splicing the transcript changes what is on screen even when nothing
    // scrolls — a short transcript cannot scroll at all — so the panel is
    // re-derived here rather than waiting for a scroll that may never come.
    updateVisibleTask();
  }

  function clearOutput() {
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    forgetPendingFileLinks(activeTabId);
    O.innerHTML = '';
    // autoscroll-coverage:start
    // The output was rebuilt from scratch (a new task's `clear`, a
    // replay, or a welcome reset): any user scroll lock is stale.
    resetUserScrollLock();
    // autoscroll-coverage:end
  }

  function removeSpinner() {
    if (_spinnerTimer) {
      clearTimeout(_spinnerTimer);
      _spinnerTimer = null;
    }
    if (waitSpinner) waitSpinner.classList.remove('active');
  }
  function showSpinner() {
    removeSpinner();
    _spinnerTimer = setTimeout(() => {
      _spinnerTimer = null;
      if (waitSpinner) waitSpinner.classList.add('active');
    }, 250);
  }

  function clearGhost() {
    currentGhost = '';
    if (ghostOverlay) ghostOverlay.innerHTML = '';
    if (ghostTimer) {
      clearTimeout(ghostTimer);
      ghostTimer = null;
    }
  }

  function updateGhost(suggestion) {
    currentGhost = suggestion || '';
    if (!ghostOverlay || !currentGhost) {
      clearGhost();
      return;
    }
    const val = inp.value;
    ghostOverlay.innerHTML =
      '<span style="visibility:hidden">' +
      esc(val) +
      '</span>' +
      '<span class="ghost-text">' +
      esc(currentGhost) +
      '</span>';
  }

  function acceptGhost() {
    if (!currentGhost) return false;
    inp.value += currentGhost;
    if (/\S$/.test(inp.value)) inp.value += ' ';
    clearGhost();
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
    return true;
  }

  function cycleHistoryUp() {
    if (histCache.length > 0 && (histIdx >= 0 || !inp.value)) {
      histIdx = Math.min(histIdx + 1, histCache.length - 1);
      inp.value = histCache[histIdx];
      inp.style.height = 'auto';
      inp.style.height = inp.scrollHeight + 'px';
      syncClearBtn();
      clearGhost();
      return true;
    }
    return false;
  }

  function cycleHistoryDown() {
    if (histIdx < 0) return false;
    histIdx--;
    inp.value = histIdx >= 0 ? histCache[histIdx] : '';
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
    syncClearBtn();
    clearGhost();
    return true;
  }

  let _touchStartX = 0;
  let _touchStartY = 0;
  const SWIPE_THRESHOLD = 30;

  function handleInputTouchStart(e) {
    if (e.touches.length === 1) {
      _touchStartX = e.touches[0].clientX;
      _touchStartY = e.touches[0].clientY;
    }
  }

  function handleInputTouchEnd(e) {
    if (e.changedTouches.length !== 1) return;
    const dx = e.changedTouches[0].clientX - _touchStartX;
    const dy = e.changedTouches[0].clientY - _touchStartY;
    const absDx = Math.abs(dx);
    const absDy = Math.abs(dy);

    if (absDx < SWIPE_THRESHOLD && absDy < SWIPE_THRESHOLD) return;

    if (absDx > absDy && dx > SWIPE_THRESHOLD) {
      if (acceptGhost()) e.preventDefault();
    } else if (absDy > absDx) {
      if (dy < -SWIPE_THRESHOLD && autocomplete.style.display !== 'block') {
        if (cycleHistoryUp()) e.preventDefault();
      } else if (dy > SWIPE_THRESHOLD) {
        if (cycleHistoryDown()) e.preventDefault();
      }
    }
  }

  function requestGhost() {
    clearGhost();
    if (isRunning || !inp.value) return;
    if (getAtCtx()) return;
    if (inp.selectionStart < inp.value.length) return;
    if (inp.value.replace(/\s/g, '').length < 2) return;
    ghostTimer = setTimeout(() => {
      ghostTimer = null;
      // Stamp the owning tab so the daemon completes against this tab's
      // chat context, not the host's stale notion of the active tab.
      api.complete({query: inp.value, tabId: activeTabId || undefined});
    }, 300);
  }

  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  const tooltipEl = document.createElement('div');
  tooltipEl.id = 'custom-tooltip';
  document.body.appendChild(tooltipEl);
  let tooltipTimer = null;
  document.addEventListener('mouseover', e => {
    const target = e.target.closest('[data-tooltip]');
    if (!target) return;
    clearTimeout(tooltipTimer);
    tooltipTimer = setTimeout(() => {
      tooltipEl.textContent = target.dataset.tooltip;
      tooltipEl.classList.toggle(
        'task-panel-tooltip',
        target.id === 'task-panel-text',
      );
      const rect = target.getBoundingClientRect();
      tooltipEl.style.left = rect.left + 'px';
      tooltipEl.style.top = rect.bottom + 4 + 'px';
      tooltipEl.classList.add('visible');
    }, 400);
  });
  document.addEventListener('mouseout', e => {
    const target = e.target.closest('[data-tooltip]');
    if (!target) return;
    clearTimeout(tooltipTimer);
    tooltipEl.classList.remove('visible');
  });
  document.addEventListener(
    'scroll',
    () => {
      clearTimeout(tooltipTimer);
      tooltipEl.classList.remove('visible');
    },
    true,
  );
  function mkEl(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  const _LINK_FILEPATH_RE =
    /(?<![\w@:%/.~-])((?:(?:~|\.{1,2})?\/|[A-Za-z0-9_+-]+\/)[A-Za-z0-9_./+-]*[A-Za-z0-9_+/-](?::\d+)?)/g;
  const _LINK_SKIP_TAGS = new Set([
    'A',
    'SCRIPT',
    'STYLE',
    'TEXTAREA',
    'INPUT',
    'BUTTON',
    'SELECT',
  ]);

  // ownerTabId names the tab whose transcript `root` belongs to. It is not
  // always the active tab: background fragments are linkified too.
  function linkifyFilePaths(root, workDir, ownerTabId) {
    if (!root || root.nodeType !== 1) return;
    // Stamp the root with the workDir/tab it is linkified under so the
    // links can be re-created after hljs.highlightElement() rewrites a
    // code block inside it (see highlightBlockPreservingLinks).
    if (root.dataset) {
      root.dataset.linkWd =
        typeof workDir === 'string'
          ? workDir
          : workDirForTab(activeTabId) || '';
      root.dataset.linkTab = String(
        ownerTabId === undefined ? activeTabId : ownerTabId,
      );
    }
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        let p = node.parentNode;
        while (p && p !== root.parentNode) {
          if (p.nodeType === 1) {
            if (_LINK_SKIP_TAGS.has(p.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            if (
              p.dataset &&
              (p.dataset.path ||
                p.dataset.pathCandidate ||
                p.dataset.pathMissing)
            ) {
              return NodeFilter.FILTER_REJECT;
            }
          }
          p = p.parentNode;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    const matches = [];
    let n;
    while ((n = walker.nextNode())) {
      const text = n.nodeValue;
      if (!text || text.indexOf('/') < 0) continue;
      _LINK_FILEPATH_RE.lastIndex = 0;
      if (_LINK_FILEPATH_RE.test(text)) {
        matches.push(n);
      }
    }
    for (const node of matches) {
      const text = node.nodeValue;
      _LINK_FILEPATH_RE.lastIndex = 0;
      const frag = node.ownerDocument.createDocumentFragment();
      let last = 0;
      let m;
      while ((m = _LINK_FILEPATH_RE.exec(text)) !== null) {
        const start = m.index;
        const end = start + m[1].length;
        if (start > last) {
          frag.appendChild(
            node.ownerDocument.createTextNode(text.slice(last, start)),
          );
        }
        const span = node.ownerDocument.createElement('span');
        span.setAttribute('data-path-candidate', m[1]);
        span.textContent = m[1];
        frag.appendChild(span);
        last = end;
      }
      if (last < text.length) {
        frag.appendChild(node.ownerDocument.createTextNode(text.slice(last)));
      }
      if (node.parentNode) node.parentNode.replaceChild(frag, node);
    }
    verifyFileLinkCandidates(root, workDir, ownerTabId);
  }

  // File paths found by linkifyFilePaths start as inert
  // [data-path-candidate] spans and become clickable [data-path] links
  // ONLY after the host confirms the file exists (checkPaths ->
  // pathsExist round-trip).  Each candidate is stamped with the workDir
  // it was checked under (data-path-wd) so replies for one workDir never
  // resolve spans checked under another.  Existence results are NOT
  // cached: only in-flight checks are deduped (per workDir+path), so
  // paths in NEW panels are re-checked and files created or deleted
  // mid-run get fresh clickability.  Candidate spans awaiting a reply
  // are tracked in a registry because panels are often linkified before
  // they are attached to the document, where a document query could not
  // find them.
  const _pendingPathChecks = new Set();
  const _pendingFileLinkSpans = new Set();

  // tableak-coverage:start
  // Keyed by tab as well as workDir: a check in flight for one tab must
  // not suppress the same check for another, or that tab's links stay
  // permanently inert.
  function _fileLinkCacheKey(tabId, workDir, p) {
    return String(tabId) + '\u0000' + workDir + '\u0000' + p;
  }
  // tableak-coverage:end

  function _stripLineSuffix(p) {
    const m = p.match(/^(.+):\d+$/);
    return m ? m[1] : p;
  }

  function promoteFileLink(span) {
    const raw = span.getAttribute('data-path-candidate');
    span.removeAttribute('data-path-candidate');
    span.removeAttribute('data-path-wd');
    span.removeAttribute('data-path-tab');
    span.setAttribute('data-path', raw);
    span.classList.add('kiss-filelink');
    span.title = 'Open ' + raw;
    _pendingFileLinkSpans.delete(span);
  }

  function demoteFileLink(span) {
    span.removeAttribute('data-path-candidate');
    span.removeAttribute('data-path-wd');
    span.removeAttribute('data-path-tab');
    span.setAttribute('data-path-missing', '1');
    span.classList.remove('kiss-filelink');
    _pendingFileLinkSpans.delete(span);
  }

  function verifyFileLinkCandidates(root, workDir, ownerTabId) {
    const spans = root.querySelectorAll('[data-path-candidate]');
    if (!spans.length) return;
    const wd =
      typeof workDir === 'string' ? workDir : workDirForTab(activeTabId) || '';
    // tableak-coverage:start
    const owner = ownerTabId === undefined ? activeTabId : ownerTabId;
    // tableak-coverage:end
    const toCheck = [];
    for (const span of spans) {
      const p = _stripLineSuffix(span.getAttribute('data-path-candidate'));
      span.setAttribute('data-path-wd', wd);
      // tableak-coverage:start
      span.setAttribute('data-path-tab', String(owner));
      // tableak-coverage:end
      _pendingFileLinkSpans.add(span);
      const key = _fileLinkCacheKey(owner, wd, p);
      if (!_pendingPathChecks.has(key)) {
        _pendingPathChecks.add(key);
        toCheck.push(p);
      }
    }
    if (toCheck.length) {
      api.send({
        type: 'checkPaths',
        paths: toCheck,
        workDir: wd,
        tabId: owner,
      });
    }
  }

  function handlePathsExist(ev) {
    const results = ev.results;
    if (!results || typeof results !== 'object') return;
    const workDir = typeof ev.workDir === 'string' ? ev.workDir : '';
    // tableak-coverage:start
    // The reply resolves only the spans of the tab that asked. Sweeping
    // every span in the document would promote or grey out another
    // conversation's file links whenever both tabs share a workspace.
    const owner = ev.tabId === undefined ? activeTabId : ev.tabId;
    for (const p of Object.keys(results)) {
      _pendingPathChecks.delete(_fileLinkCacheKey(owner, workDir, p));
    }
    // tableak-coverage:end
    const spans = new Set(_pendingFileLinkSpans);
    for (const span of document.querySelectorAll('[data-path-candidate]')) {
      spans.add(span);
    }
    for (const span of spans) {
      // tableak-coverage:start
      if ((span.getAttribute('data-path-tab') || '') !== String(owner)) {
        continue;
      }
      // tableak-coverage:end
      if ((span.getAttribute('data-path-wd') || '') !== workDir) continue;
      const p = _stripLineSuffix(
        span.getAttribute('data-path-candidate') || '',
      );
      if (Object.prototype.hasOwnProperty.call(results, p)) {
        if (results[p]) promoteFileLink(span);
        else demoteFileLink(span);
      }
    }
  }

  /**
   * Release the file-link bookkeeping of a transcript that is going away.
   *
   * The spans of a closed or cleared tab are detached DOM that no reply
   * will ever promote, and the registry would keep them -- and the work
   * of walking them on every later reply -- alive for the rest of the
   * session. Their in-flight keys go with them, so the same paths are
   * checked afresh if the tab's transcript is rebuilt.
   *
   * @param {string} tabId The tab whose transcript is being discarded.
   */
  function forgetPendingFileLinks(tabId) {
    const owner = String(tabId);
    for (const span of Array.from(_pendingFileLinkSpans)) {
      if ((span.getAttribute('data-path-tab') || '') === owner) {
        _pendingFileLinkSpans.delete(span);
      }
    }
    const prefix = owner + '\u0000';
    for (const key of Array.from(_pendingPathChecks)) {
      if (key.indexOf(prefix) === 0) _pendingPathChecks.delete(key);
    }
  }

  /**
   * Forget which checks are in flight after losing the daemon.
   *
   * Nothing can answer a request that was on the wire when the socket
   * died, so leaving its key in the dedup set would suppress every later
   * check of that path and the links would stay inert for the rest of
   * the session. The spans themselves are kept: they are still on
   * screen, and the re-issued check promotes them.
   */
  function forgetInFlightPathChecks() {
    _pendingPathChecks.clear();
  }

  /**
   * Re-ask about the candidate spans whose reply the outage swallowed.
   *
   * forgetInFlightPathChecks() drops the keys of checks that can never
   * be answered, but the spans themselves stay on screen -- grey and
   * unclickable. A finished task renders no further panels, so unless
   * some later output happened to mention the very same path, nothing
   * would ever ask about them again and those links would stay dead for
   * the rest of the session. Reconnecting is their last chance.
   *
   * Each span carries the tab and the workDir it was checked under, and
   * a reply only resolves spans stamped with the same pair, so the
   * reissue is grouped by both. Spans already covered by a live check
   * are skipped, and a reconnect with nothing outstanding sends
   * nothing: every open window reconnects at once after a daemon
   * restart.
   */
  function reissueFileLinkChecks() {
    const groups = new Map();
    for (const span of _pendingFileLinkSpans) {
      const raw = span.getAttribute('data-path-candidate');
      if (!raw) continue;
      const owner = span.getAttribute('data-path-tab') || '';
      const wd = span.getAttribute('data-path-wd') || '';
      const p = _stripLineSuffix(raw);
      const key = _fileLinkCacheKey(owner, wd, p);
      if (_pendingPathChecks.has(key)) continue;
      _pendingPathChecks.add(key);
      const groupKey = owner + '\u0000' + wd;
      let group = groups.get(groupKey);
      if (!group) {
        group = {type: 'checkPaths', paths: [], workDir: wd, tabId: owner};
        groups.set(groupKey, group);
      }
      group.paths.push(p);
    }
    for (const group of groups.values()) api.send(group);
  }

  /**
   * Highlight *bl* without permanently destroying its file links.
   *
   * hljs.highlightElement() rewrites the block's innerHTML from its
   * text, wiping the [data-path]/[data-path-candidate] spans that
   * linkifyFilePaths() put there (deferred highlighting runs AFTER the
   * panel was linkified).  Unwrap the spans first — so hljs sees plain
   * text and stale spans are dropped from the pending registry — then
   * re-linkify the block under the workDir/tab stamped on the panel
   * root that was linkified originally.
   */
  function highlightBlockPreservingLinks(bl) {
    const spans = bl.querySelectorAll(
      '[data-path], [data-path-candidate], [data-path-missing]',
    );
    for (const span of spans) {
      _pendingFileLinkSpans.delete(span);
      span.replaceWith(span.ownerDocument.createTextNode(span.textContent));
    }
    hljs.highlightElement(bl);
    const holder = bl.closest('[data-link-wd]');
    if (holder) {
      linkifyFilePaths(bl, holder.dataset.linkWd, holder.dataset.linkTab);
    }
  }

  function hlBlock(el) {
    if (typeof hljs === 'undefined') return;
    el.querySelectorAll('pre code').forEach(bl => {
      if (_deferHighlight) {
        bl.classList.add('needs-hl');
      } else {
        highlightBlockPreservingLinks(bl);
      }
    });
  }

  function highlightPending(root) {
    if (typeof hljs === 'undefined' || !root) return;
    root.querySelectorAll('code.needs-hl').forEach(bl => {
      bl.classList.remove('needs-hl');
      highlightBlockPreservingLinks(bl);
    });
  }

  function toggleThink(el) {
    const p = el.parentElement;
    p.querySelector('.cnt').classList.toggle('hidden');
    el.querySelector('.arrow').classList.toggle('collapsed');
  }

  function collectText(node) {
    if (node.nodeType === 3) return node.textContent || '';
    if (node.nodeType === 1 && node.classList) {
      if (
        node.classList.contains('panel-copy-btn') ||
        node.classList.contains('collapse-chv') ||
        node.classList.contains('collapse-preview') ||
        node.classList.contains('panel-ts') ||
        node.classList.contains('panel-time')
      )
        return '';
    }
    let out = '';
    for (let i = 0; i < node.childNodes.length; i++) {
      const child = node.childNodes[i];
      const t = collectText(child);
      if (child.nodeType === 1 && out.length > 0 && t.length > 0) out += ' ';
      out += t;
    }
    return out;
  }

  function collapsePreview(panelEl) {
    const prev = panelEl.querySelector('.collapse-preview');
    if (!prev) return;
    if (panelEl.classList.contains('tc-summary')) {
      prev.textContent = '';
      return;
    }
    if (!panelEl.classList.contains('collapsed')) {
      prev.textContent = '';
      return;
    }
    let txt = '';
    for (let i = 0; i < panelEl.children.length; i++) {
      const ch = panelEl.children[i];
      if (
        ch.classList.contains('collapse-chv') ||
        ch === prev ||
        ch.querySelector('.collapse-chv')
      )
        continue;
      txt += collectText(ch) + ' ';
    }
    txt = txt.replace(/\s+/g, ' ').trim();
    prev.textContent = txt;
  }

  /**
   * Collapse every run_parallel panel inside *root* and close the
   * sub-agent tabs those fan-outs own.
   *
   * Called with a panel that just collapsed, and with a whole
   * transcript that is about to be hidden or thrown away.  A collapsed
   * panel hides its children (``.tc.collapsed > :not(.tc-h,
   * .panel-copy-btn){display:none}`` in main.css), so a run_parallel
   * panel that another panel swallowed -- the ``summary`` tool adopts
   * the event panels preceding it into a ``.summary-sub`` child -- is
   * just as collapsed as one the user closed by hand. Its chevron is
   * off screen, so leaving its sub-agent tabs open would strand tabs
   * that no reachable panel can ever close again.
   *
   * @param {Element|DocumentFragment|null} root Panel or transcript
   *     whose fan-outs are going off screen. Null is a no-op, so a
   *     transcript that was already discarded needs no guard.
   */
  function collapseNestedRunParallel(root) {
    if (!root) return;
    const nested = root.querySelectorAll('.tc-run-parallel');
    for (let i = 0; i < nested.length; i++) {
      const p = nested[i];
      // A neighbouring task's replayed transcript owns no tab of this
      // conversation, so its fan-out panels are left untouched.
      if (p.closest('.adjacent-task')) continue;
      if (!p.classList.contains('collapsed')) {
        p.classList.add('collapsed');
        p.classList.remove('user-pinned');
        collapsePreview(p);
      }
      syncRunParallelPanel(p);
    }
  }

  function addCollapse(panelEl, headerEl, ts) {
    panelEl.classList.add('collapsible');
    const chv = mkEl('span', 'collapse-chv');
    chv.textContent = '\u25BE';
    const prev = mkEl('span', 'collapse-preview');
    headerEl.insertBefore(chv, headerEl.firstChild);
    headerEl.appendChild(prev);
    headerEl.classList.add('collapse-header');
    headerEl.style.cursor = 'pointer';
    headerEl.style.userSelect = 'none';
    headerEl.addEventListener('click', e => {
      e.stopPropagation();
      panelEl.classList.toggle('collapsed');
      if (panelEl.classList.contains('collapsed')) {
        panelEl.classList.remove('user-pinned');
      } else {
        panelEl.classList.add('user-pinned');
        highlightPending(panelEl);
      }
      collapsePreview(panelEl);
      syncRunParallelPanel(panelEl);
      if (panelEl.classList.contains('collapsed'))
        collapseNestedRunParallel(panelEl);
    });
    addCopyButton(panelEl);
    addPanelTimestamp(panelEl, ts);
  }

  const _rpTabPanel = new Map();
  const _rpClosedSubagentTabs = new Set();
  let _rpSyncing = false;
  // Sub-agent tabs a collapse wants closed while a transcript is being
  // replayed wait here until the replay is done; see rpCloseSubagentTab.
  let _rpDeferredCloses = null;

  /**
   * Close sub-agent tab *tabId* on behalf of its collapsed fan-out
   * panel, or queue the close when a transcript is being replayed.
   *
   * Closing the tab the user is looking at moves them to another tab,
   * and moving them onto the very chat whose transcript is being
   * replayed would put that half-written transcript on screen and
   * detach the rest of the replay into a fragment nobody sees. The
   * replay finishes first, then the tabs close.
   *
   * @param {string} tabId The sub-agent tab to close.
   */
  function rpCloseSubagentTab(tabId) {
    if (_rpDeferredCloses) _rpDeferredCloses.push(tabId);
    else closeTab(tabId);
  }

  /**
   * Close the sub-agent tabs queued during a replay.
   *
   * The closes are performed as the collapse they came from, not as a
   * close the user asked for, so a sub-agent stays reopenable by
   * expanding its panel again (see rpAfterTabsClosed).
   */
  function rpFlushDeferredCloses() {
    const ids = _rpDeferredCloses;
    _rpDeferredCloses = null;
    if (!ids.length) return;
    _rpSyncing = true;
    try {
      for (const id of ids) closeTab(id);
    } finally {
      _rpSyncing = false;
    }
  }

  function rpTaskDomRootForParent(parentId) {
    if (parentId === activeTabId) return O;
    const parentTab = getTab(parentId);
    return parentTab ? parentTab.outputFragment : null;
  }

  function rpDirectPanelsForParent(parentId) {
    const root = rpTaskDomRootForParent(parentId);
    if (!root || !root.querySelectorAll) return [];
    return Array.from(root.querySelectorAll('.tc-run-parallel')).filter(
      p => !(p.closest && p.closest('.adjacent-task')),
    );
  }

  function runParallelPanelForParent(parentId) {
    const panels = rpDirectPanelsForParent(parentId);
    return panels.length ? panels[panels.length - 1] : null;
  }

  function rpExpectedTaskCount(rawTasks) {
    if (Array.isArray(rawTasks)) return rawTasks.length;
    if (typeof rawTasks !== 'string' || !rawTasks) return null;
    try {
      const parsed = JSON.parse(rawTasks);
      return Array.isArray(parsed) ? parsed.length : null;
    } catch (_e) {
      return null;
    }
  }

  function rpPanelOwningTask(parentId, taskId) {
    if (taskId === undefined || taskId === null || taskId === '') return null;
    for (const p of rpDirectPanelsForParent(parentId)) {
      const entries = p._rpSubagents || [];
      if (entries.some(en => String(en.taskId) === String(taskId))) return p;
    }
    return null;
  }

  function rpPanelForNewSubagent(parentId, taskId) {
    const owner = rpPanelOwningTask(parentId, taskId);
    if (owner) return owner;
    const panels = rpDirectPanelsForParent(parentId);
    for (const p of panels) {
      const expected = p._rpExpectedCount;
      if (typeof expected !== 'number') continue;
      if ((p._rpSubagents || []).length < expected) return p;
    }
    return panels.length ? panels[panels.length - 1] : null;
  }

  function rpPanelHasOpenTabs(panelEl) {
    const entries = panelEl._rpSubagents;
    if (!entries) return false;
    return entries.some(en => en.tabId && getTab(en.tabId));
  }

  /**
   * The open sub-agent tab that is running sub-agent task *taskId*, or
   * null when no tab shows that sub-agent.
   *
   * A sub-agent's identity is its TASK id, never a tab id: the daemon
   * addresses one sub-agent by several tab ids over its life -- the live
   * fan-out id minted by the parent agent, the deterministic
   * ``<parentTab>__sub_<taskId>`` id used when the parent's history row
   * is replayed, and whatever id this client minted when it reopened
   * the tab after its run_parallel panel was expanded again. Keying
   * tabs on the tab id alone therefore stacks one tab per id for a
   * single sub-agent.
   *
   * @param {string} taskId Sub-agent task id to look for.
   * @param {string} exceptTabId Tab id to ignore, or '' for none.
   * @returns {object|null} The tab object, or null.
   */
  function openSubagentTabForTask(taskId, exceptTabId) {
    if (taskId === undefined || taskId === null || taskId === '') return null;
    const key = String(taskId);
    for (const tab of tabs) {
      if (!tab.isSubagentTab) continue;
      if (exceptTabId && tab.id === exceptTabId) continue;
      if (tabTaskId(tab) === key) return tab;
    }
    return null;
  }

  /**
   * True when the user closed sub-agent task *taskId*'s tab by hand.
   *
   * Such a sub-agent stays closed until its run_parallel panel is
   * collapsed and expanded again, so no later announcement from the
   * daemon -- under any of the tab ids it addresses that sub-agent by
   * -- may reopen it.
   *
   * @param {Element|null} panelEl The owning run_parallel panel.
   * @param {string} taskId Sub-agent task id.
   * @returns {boolean} True when the sub-agent must stay closed.
   */
  function rpSubagentHandClosed(panelEl, taskId) {
    if (!panelEl || taskId === undefined || taskId === null || taskId === '')
      return false;
    return (panelEl._rpSubagents || []).some(
      en =>
        String(en.taskId) === String(taskId) &&
        en.userClosed &&
        !getTab(en.tabId),
    );
  }

  /**
   * Move the already-open sub-agent tab *tab* onto *newTabId*, the tab
   * id the daemon now addresses that sub-agent by.
   *
   * Following the daemon's rename (instead of opening a second tab)
   * keeps one sub-agent on one tab and keeps both sides in agreement:
   * every later event for this sub-agent, and every ``closeTab`` this
   * client sends for it, then names the same id.  The host is told to
   * release the old id, because the daemon opened the new one as an
   * additional viewer of the same sub-agent rather than as a rename:
   * left alone, the retired id would keep host resources (merge
   * managers, tab ownership, worktree bars) alive for a tab that no
   * longer exists on this client.
   *
   * @param {object} tab The open sub-agent tab to re-id.
   * @param {string} newTabId The tab id the daemon uses from now on.
   */
  function retagSubagentTab(tab, newTabId) {
    if (!tab || !newTabId || tab.id === newTabId) return;
    const oldId = tab.id;
    const panel = _rpTabPanel.get(oldId) || null;
    tab.id = newTabId;
    if (panel) {
      _rpTabPanel.delete(oldId);
      _rpTabPanel.set(newTabId, panel);
      for (const en of panel._rpSubagents || []) {
        if (en.tabId === oldId) en.tabId = newTabId;
      }
    }
    _rpClosedSubagentTabs.delete(newTabId);
    // report-coverage:start
    // Reports this sub-agent already wrote are pending under its old
    // tab id; they must still open when the sub-agent finishes.
    const oldReports = readyReportsByTab[reportTabKey(oldId)];
    if (oldReports) {
      delete readyReportsByTab[reportTabKey(oldId)];
      readyReportsByTab[reportTabKey(newTabId)] = oldReports;
    }
    // report-coverage:end
    if (activeTabId === oldId) activeTabId = newTabId;
    // The host is told which CHAT tab is on screen even while a content
    // tab is active, so the reported id must follow the rename on its
    // own -- a stale one keeps the host matching merges against a tab
    // that is gone.
    if (reportedChatTabId === oldId) reportChatTab(newTabId);
    api.closeTab({tabId: oldId});
  }

  function rpOwnerTabIdForContainer(container, fallbackTabId) {
    if (fallbackTabId !== undefined && fallbackTabId !== null)
      return fallbackTabId;
    if (container === O) return activeTabId;
    for (const tab of tabs) {
      if (tab.outputFragment === container) return tab.id;
      if (
        tab.outputFragment &&
        tab.outputFragment.contains &&
        tab.outputFragment.contains(container)
      ) {
        return tab.id;
      }
    }
    return '';
  }

  function rpOwnerTabIdForPanel(panelEl) {
    if (panelEl._rpParentTabId) return panelEl._rpParentTabId;
    if (panelEl.closest && panelEl.closest('.adjacent-task')) return '';
    if (O && O.contains(panelEl)) return activeTabId;
    const root = panelEl.getRootNode ? panelEl.getRootNode() : null;
    return rpOwnerTabIdForContainer(root);
  }

  function rpAdoptOpenSubagents(panelEl, parentId) {
    if (!panelEl.classList.contains('tc-run-parallel') || !parentId) return;
    const livePanels = new Set(rpDirectPanelsForParent(parentId));
    if (!livePanels.has(panelEl)) return;
    const newest = runParallelPanelForParent(parentId);
    const openChildren = tabs.filter(
      tab => tab.isSubagentTab && tab.parentTabId === parentId,
    );
    for (const tab of openChildren) {
      const previousPanel = _rpTabPanel.get(tab.id);
      if (previousPanel) {
        if (livePanels.has(previousPanel)) continue;
        if (previousPanel._rpCallIndex !== panelEl._rpCallIndex) continue;
      } else if (panelEl !== newest) {
        continue;
      }
      const previousEntry = previousPanel
        ? (previousPanel._rpSubagents || []).find(en => en.tabId === tab.id)
        : null;
      let taskId = previousEntry ? previousEntry.taskId : '';
      if (
        (taskId === undefined || taskId === null || taskId === '') &&
        tab.currentTaskId !== undefined &&
        tab.currentTaskId !== null
      ) {
        taskId = tab.currentTaskId;
      }
      rpRegisterSubagent(panelEl, parentId, taskId || '', tab.id);
    }
  }

  function rpRegisterSubagent(panelEl, parentId, taskId, tabId) {
    if (!panelEl._rpSubagents) panelEl._rpSubagents = [];
    panelEl._rpParentTabId = parentId;
    const taskKey = taskId === undefined || taskId === null ? '' : taskId;
    const tabKey = tabId || '';
    let entry = null;
    if (tabKey) {
      entry = panelEl._rpSubagents.find(en => en.tabId === tabKey) || null;
    }
    if (!entry && taskKey !== '') {
      entry =
        panelEl._rpSubagents.find(
          en => String(en.taskId) === String(taskKey),
        ) || null;
    }
    if (!entry) {
      entry = {taskId: taskKey, tabId: tabKey};
      panelEl._rpSubagents.push(entry);
    } else {
      if (taskKey !== '') entry.taskId = taskKey;
      if (tabKey && entry.tabId !== tabKey) {
        if (entry.tabId) _rpTabPanel.delete(entry.tabId);
        entry.tabId = tabKey;
      }
    }
    if (tabKey) {
      entry.userClosed = false;
      _rpClosedSubagentTabs.delete(tabKey);
      _rpTabPanel.set(tabKey, panelEl);
    }
    rpMergeDuplicateEntries(panelEl, entry);
  }

  /**
   * Fold every other entry of *panelEl* that names *entry*'s sub-agent
   * task into *entry*.
   *
   * One sub-agent must own exactly one entry, or expanding the panel
   * would open one tab per entry for it. Duplicates appear because an
   * entry can be created before its task id is known (a tab adopted
   * from a re-rendered panel) or before its tab exists (a sub-agent
   * spawned while the panel was collapsed), and the two only turn out
   * to be the same sub-agent once the daemon names both.
   *
   * @param {Element} panelEl The run_parallel panel to clean up.
   * @param {object} entry The surviving entry.
   */
  function rpMergeDuplicateEntries(panelEl, entry) {
    const key = entry.taskId === undefined ? '' : String(entry.taskId);
    if (key === '') return;
    const entries = panelEl._rpSubagents;
    for (let i = entries.length - 1; i >= 0; i--) {
      const other = entries[i];
      if (other === entry || String(other.taskId) !== key) continue;
      // Keep whichever tab is actually open: an entry recorded while
      // the panel was collapsed carries no tab.
      if (!entry.tabId || !getTab(entry.tabId)) {
        if (other.tabId && getTab(other.tabId)) {
          entry.tabId = other.tabId;
          _rpTabPanel.set(other.tabId, panelEl);
        }
      }
      if (other.tabId && other.tabId !== entry.tabId) {
        _rpTabPanel.delete(other.tabId);
      }
      // An open tab settles it; otherwise a hand-close recorded on
      // either entry still holds (the user's close wins).
      if (entry.tabId && getTab(entry.tabId)) entry.userClosed = false;
      else entry.userClosed = !!(entry.userClosed || other.userClosed);
      entries.splice(i, 1);
    }
  }

  function syncRunParallelPanel(panelEl) {
    if (!panelEl.classList.contains('tc-run-parallel')) return;
    rpAdoptOpenSubagents(panelEl, rpOwnerTabIdForPanel(panelEl));
    const entries = panelEl._rpSubagents;
    if (!entries) return;
    if (_rpSyncing) return;
    _rpSyncing = true;
    try {
      const collapsed = panelEl.classList.contains('collapsed');
      for (const en of entries) {
        const openTab = en.tabId ? getTab(en.tabId) : null;
        if (collapsed && openTab) {
          // The tab stays this panel's until the close actually lands:
          // rpAfterTabsClosed does the bookkeeping, and a tab whose
          // close is still queued must keep looking owned so another
          // panel's adoption pass cannot claim it as unowned.
          rpCloseSubagentTab(en.tabId);
        } else if (collapsed) {
          en.userClosed = false;
        } else if (!openTab && en.taskId !== '' && !en.userClosed) {
          // This sub-agent may already have a tab under another id
          // (the daemon renames sub-agent tabs across replays); adopt
          // it rather than opening a second tab for one sub-agent.
          const existing = openSubagentTabForTask(en.taskId, '');
          if (existing) {
            en.tabId = existing.id;
            _rpTabPanel.set(existing.id, panelEl);
            continue;
          }
          const subTab = createBackgroundSubagentTab(panelEl._rpParentTabId);
          subTab.currentTaskId = en.taskId;
          en.tabId = subTab.id;
          _rpTabPanel.set(subTab.id, panelEl);
          api.resumeSession({taskId: en.taskId, tabId: subTab.id});
        }
      }
    } finally {
      _rpSyncing = false;
    }
  }

  /**
   * Forget the sub-agent tabs in *closedIds* and, when the user was the
   * one who closed them, collapse the fan-out panels left with none.
   *
   * The bookkeeping runs even while a collapse is closing tabs
   * (``_rpSyncing``): closing a sub-agent's tab also closes the tabs of
   * the fan-out that sub-agent ran itself, and a grandchild the client
   * still believes is owned by a panel from a chat that no longer
   * exists would be reopened by the next announcement naming it.
   *
   * @param {Set<string>|Array<string>} closedIds Tab ids just closed.
   */
  function rpAfterTabsClosed(closedIds) {
    const panels = new Set();
    for (const id of closedIds) {
      const p = _rpTabPanel.get(id);
      if (p) {
        _rpClosedSubagentTabs.add(id);
        _rpTabPanel.delete(id);
        for (const en of p._rpSubagents || []) {
          if (en.tabId === id) {
            en.tabId = '';
            // Only a close the user asked for keeps this sub-agent shut
            // while its panel stays expanded; collapsing the panel
            // reopens every sub-agent when it is expanded again.
            en.userClosed = !_rpSyncing;
          }
        }
        panels.add(p);
      }
    }
    if (_rpSyncing) return;
    for (const p of panels) {
      const parentOpen =
        p._rpParentTabId === activeTabId || getTab(p._rpParentTabId);
      if (!parentOpen) continue;
      if (rpPanelHasOpenTabs(p)) continue;
      if (!p.classList.contains('collapsed')) {
        p.classList.add('collapsed');
        p.classList.remove('user-pinned');
        collapsePreview(p);
      }
      syncRunParallelPanel(p);
    }
  }

  const addCopyButton = window.PanelCopy.addCopyButton;
  const addPanelTimestamp = window.PanelCopy.addPanelTimestamp;
  const PANEL_COPY_SVG = window.PanelCopy.PANEL_COPY_SVG;
  const PANEL_CHECK_SVG = window.PanelCopy.PANEL_CHECK_SVG;

  function collapseAllExceptResult(container, ownerTabId) {
    const ownerId = rpOwnerTabIdForContainer(container, ownerTabId);
    const panels = container.querySelectorAll('.collapsible');
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      if (p.classList.contains('rc')) continue;
      if (p.classList.contains('tc-run-parallel'))
        rpAdoptOpenSubagents(p, ownerId);
      if (rpPanelHasOpenTabs(p) && !p._rpDone) continue;
      p.classList.add('collapsed');
      collapsePreview(p);
      syncRunParallelPanel(p);
      collapseNestedRunParallel(p);
    }
  }

  /**
   * True while the task of *tabId* is running.
   *
   * The visible tab's flag is the module-level `isRunning` that
   * setRunningState keeps in step with it; a tab that is not on screen
   * carries the flag on itself. Reading it per tab is what lets a
   * background transcript collapse its panels exactly like a visible
   * one -- it used to consult the visible tab's flag and so never
   * collapsed anything.
   *
   * @param {string} tabId The tab that owns a transcript.
   * @returns {boolean} Whether that tab's task is running.
   */
  function streamTabIsRunning(tabId) {
    if (tabId === activeTabId) return isRunning;
    const tab = getTab(tabId);
    return !!(tab && tab.isRunning);
  }

  /**
   * Collapse every top-level panel of a running transcript but the last.
   *
   * @param {Element|DocumentFragment} container The transcript.
   * @param {string} tabId The tab that owns it.
   */
  function collapseOlderPanels(container, tabId) {
    // Only an attached transcript is collapsed as it streams.  A
    // background tab's fragment is collapsed once, when it is restored
    // (see restoreTab): collapsing a run_parallel panel adopts its open
    // sub-agent tabs into the newest fan-out call, and mid-stream that
    // call does not exist yet, so a live sub-agent tab would be closed
    // by the very panel it is about to move out of.
    if (!container || container.nodeType !== 1) return;
    if (!streamTabIsRunning(tabId)) return;
    const panels = Array.from(container.children).filter(
      el => el.classList && el.classList.contains('collapsible'),
    );
    for (let i = 0; i < panels.length - 1; i++) {
      const p = panels[i];
      if (p.classList.contains('rc') || p.classList.contains('user-pinned'))
        continue;
      if (p.classList.contains('tc-run-parallel'))
        rpAdoptOpenSubagents(p, tabId);
      if (rpPanelHasOpenTabs(p) && !p._rpDone) continue;
      p.classList.add('collapsed');
      collapsePreview(p);
      syncRunParallelPanel(p);
      collapseNestedRunParallel(p);
    }
  }

  function splitMultiSessionSummary(summary) {
    const text = typeof summary === 'string' ? summary : '';
    // The summary wire format is HTML (<h3> session markers); old
    // persisted events may still carry Markdown '###' markers.
    const finalMarkers = [
      '\n\n---\n\n<h3>Final Session</h3>\n',
      '\n\n---\n\n### Final Session\n',
    ];
    let markerIdx = -1;
    let markerLen = 0;
    for (const finalMarker of finalMarkers) {
      markerIdx = text.indexOf(finalMarker);
      if (markerIdx > 0) {
        markerLen = finalMarker.length;
        break;
      }
    }
    if (markerIdx <= 0) {
      const separator = '\n\n---\n\n';
      markerIdx = text.lastIndexOf(separator);
      markerLen = separator.length;
    }
    if (markerIdx <= 0) return null;
    const previous = text.substring(0, markerIdx).trim();
    const final = text.substring(markerIdx + markerLen).trim();
    if (!previous || !final) return null;
    if (
      !previous.includes('<h3>Previous Session') &&
      !previous.includes('### Previous Session')
    )
      return null;
    return {previous: previous, final: final};
  }

  function removeResultPanels(container) {
    if (!container || !container.children) return;
    for (let i = container.children.length - 1; i >= 0; i--) {
      const child = container.children[i];
      if (child.classList && child.classList.contains('rc')) child.remove();
    }
  }

  function createResultPanel(
    ev,
    summaryOverride,
    titleOverride,
    showStatus,
    workDir,
    ownerTabId,
  ) {
    const rc = mkEl('div', 'ev rc');
    let rb = '';
    let rawBody = '';
    if (showStatus && ev.is_continue) {
      rb += '<div class="rc-status">Status: Continue</div>';
      rawBody += 'Status: Continue\n\n';
    } else if (showStatus && ev.success === false) {
      rb += '<div class="rc-status rc-status-fail">Status: FAILED</div>';
      rawBody += 'Status: FAILED\n\n';
    }
    let usePre = true;
    const summaryText =
      summaryOverride !== undefined ? summaryOverride : ev.summary;
    if (summaryText) {
      const sum = String(summaryText)
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      // New summaries are HTML; legacy persisted events are Markdown.
      rb += kissSanitize(resultSummaryHtml(sum));
      usePre = false;
      rawBody += sum;
    } else {
      const txt = (ev.text || '(no result)').replace(/\n{3,}/g, '\n\n').trim();
      rb += esc(txt);
      rawBody += txt;
    }
    rc.dataset.rawText = rawBody;
    rc.innerHTML =
      '<div class="rc-h"><h3>' +
      esc(titleOverride || 'Result') +
      '</h3><div class="rs">' +
      '<span>Tokens <b>' +
      fmtN(ev.total_tokens || 0) +
      '</b></span>' +
      '<span>Cost <b>' +
      esc(ev.cost || 'N/A') +
      '</b></span>' +
      '</div></div><div class="rc-body md-body' +
      (usePre ? ' pre' : '') +
      '">' +
      rb +
      '</div>';
    hlBlock(rc);
    addCopyButton(rc);
    addPanelTimestamp(rc, ev.ts);
    const rcBody = rc.querySelector('.rc-body');
    if (rcBody) linkifyFilePaths(rcBody, workDir, ownerTabId);
    return rc;
  }

  window.toggleThink = toggleThink;

  function lineDiff(a, b) {
    const al = a.split('\n'),
      bl = b.split('\n'),
      m = al.length,
      n = bl.length;
    const dp = [];
    for (let i = 0; i <= m; i++) {
      dp[i] = new Array(n + 1);
      dp[i][0] = 0;
    }
    for (let j = 0; j <= n; j++) dp[0][j] = 0;
    for (let i = 1; i <= m; i++)
      for (let j = 1; j <= n; j++)
        dp[i][j] =
          al[i - 1] === bl[j - 1]
            ? dp[i - 1][j - 1] + 1
            : Math.max(dp[i - 1][j], dp[i][j - 1]);
    const ops = [];
    let ci = m,
      cj = n;
    while (ci > 0 || cj > 0) {
      if (ci > 0 && cj > 0 && al[ci - 1] === bl[cj - 1]) {
        ops.unshift({t: '=', o: al[--ci], n: bl[--cj]});
      } else if (cj > 0 && (ci === 0 || dp[ci][cj - 1] >= dp[ci - 1][cj])) {
        ops.unshift({t: '+', n: bl[--cj]});
      } else {
        ops.unshift({t: '-', o: al[--ci]});
      }
    }
    return ops;
  }

  function hlInline(oldL, newL) {
    const mn = Math.min(oldL.length, newL.length);
    let pre = 0,
      suf = 0;
    while (pre < mn && oldL[pre] === newL[pre]) pre++;
    while (
      suf < mn - pre &&
      oldL[oldL.length - 1 - suf] === newL[newL.length - 1 - suf]
    )
      suf++;
    const pf = oldL.substring(0, pre),
      sf = suf ? oldL.substring(oldL.length - suf) : '';
    return {
      o:
        esc(pf) +
        '<span class="diff-hl-del">' +
        esc(oldL.substring(pre, oldL.length - suf)) +
        '</span>' +
        esc(sf),
      n:
        esc(pf) +
        '<span class="diff-hl-add">' +
        esc(newL.substring(pre, newL.length - suf)) +
        '</span>' +
        esc(sf),
    };
  }

  function renderDiff(oldStr, newStr) {
    const ops = lineDiff(oldStr, newStr);
    let html = '',
      i = 0;
    while (i < ops.length) {
      const dels = [],
        adds = [];
      while (i < ops.length && ops[i].t === '-') {
        dels.push(ops[i++]);
      }
      while (i < ops.length && ops[i].t === '+') {
        adds.push(ops[i++]);
      }
      if (dels.length || adds.length) {
        const pairs = Math.min(dels.length, adds.length);
        for (let p = 0; p < pairs; p++) {
          const h = hlInline(dels[p].o, adds[p].n);
          html += '<div class="diff-old">- ' + h.o + '</div>';
          html += '<div class="diff-new">+ ' + h.n + '</div>';
        }
        for (let p = pairs; p < dels.length; p++)
          html += '<div class="diff-old">- ' + esc(dels[p].o) + '</div>';
        for (let p = pairs; p < adds.length; p++)
          html += '<div class="diff-new">+ ' + esc(adds[p].n) + '</div>';
        continue;
      }
      html += '<div class="diff-ctx">  ' + esc(ops[i].o) + '</div>';
      i++;
    }
    return html;
  }

  // autoscroll-coverage:start
  // Auto-scroll: the chat webview (extension and remote webapp alike)
  // follows the tail of the latest event panel — unless the user
  // scroll lock below is engaged — and every scrollable subpanel of an
  // event panel follows its own tail as streamed text appears inside
  // it.
  const AUTO_SCROLL_SUBPANEL_SEL =
    '.think, .bash-panel-content, .llm-panel, .tc-b, .tr, ' +
    '.prompt-body, .system-prompt-body';

  function scrollPanelToEnd(el) {
    const top = Math.max(0, el.scrollHeight - el.clientHeight);
    if (el.scrollTop !== top) el.scrollTop = top;
  }

  // User scroll lock: when a task is running and the user scrolls the
  // chat up by at least 1/8th of its visible height, outer auto-scroll
  // is disabled; it resumes once the user scrolls back to the bottom
  // of the chat.
  let userScrollLock = false;

  function chatDistanceFromBottom() {
    return Math.max(0, O.scrollHeight - O.clientHeight - O.scrollTop);
  }

  function updateUserScrollLock() {
    const dist = chatDistanceFromBottom();
    if (dist >= O.clientHeight / 8) {
      if (isRunning) userScrollLock = true;
    } else if (dist <= 1) {
      userScrollLock = false;
    }
  }

  function resetUserScrollLock() {
    userScrollLock = false;
  }

  function autoScrollChat() {
    // The outer chat follows the tail only while the user has not
    // scrolled up (the lock re-arms when they return to the bottom).
    if (!userScrollLock) scrollPanelToEnd(O);
  }

  function autoScrollStreamed(el) {
    // Scroll every scrollable panel enclosing a streamed text update,
    // then the outer chat, so the newest text stays visible.  Nodes
    // still inside a background tab's detached fragment are skipped.
    if (!el || !O.contains(el)) return;
    let n = el;
    while (n && n !== O) {
      if (n.matches && n.matches(AUTO_SCROLL_SUBPANEL_SEL)) scrollPanelToEnd(n);
      n = n.parentElement;
    }
    autoScrollChat();
  }

  function autoScrollLatestEventPanel(panel) {
    // Scroll the latest event panel's scrollable subpanels to their
    // end, then the outer chat to the end of that panel.
    if (panel && O.contains(panel)) {
      if (panel.matches && panel.matches(AUTO_SCROLL_SUBPANEL_SEL))
        scrollPanelToEnd(panel);
      const subs = panel.querySelectorAll(AUTO_SCROLL_SUBPANEL_SEL);
      for (let i = 0; i < subs.length; i++) scrollPanelToEnd(subs[i]);
    }
    autoScrollChat();
  }
  // autoscroll-coverage:end

  function handleOutputEvent(ev, target, tState, ownerWorkDir, ownerTabId) {
    const evWorkDir =
      typeof ownerWorkDir === 'string'
        ? ownerWorkDir
        : workDirForTab(activeTabId) || '';
    // tableak-coverage:start
    // File-link candidates are stamped with the tab that owns this
    // transcript so a later pathsExist reply resolves only its own spans.
    const evOwnerTab = ownerTabId === undefined ? activeTabId : ownerTabId;
    // tableak-coverage:end
    const t = ev.type;
    switch (t) {
      case 'thinking_start':
        tState.thinkEl = mkEl('div', 'ev think');
        tState.thinkEl.innerHTML =
          '<div class="lbl" onclick="toggleThink(this)">' +
          '<span class="arrow">\u25BE</span> Thinking</div>' +
          '<div class="cnt"></div>';
        tState.thinkCnt = tState.thinkEl.querySelector('.cnt');
        tState.thinkBuf = '';
        tState.thinkRaf = 0;
        target.appendChild(tState.thinkEl);
        break;
      case 'thinking_delta':
        if (tState.thinkCnt) {
          tState.thinkBuf += (ev.text || '').replace(/\n\n+/g, '\n');
          if (!tState.thinkRaf) {
            tState.thinkRaf = requestAnimationFrame(() => {
              tState.thinkRaf = 0;
              if (!tState.thinkCnt) {
                tState.thinkBuf = '';
                return;
              }
              const cnt = tState.thinkCnt;
              const last = cnt.lastChild;
              if (last && last.nodeType === 3) {
                last.appendData(tState.thinkBuf);
              } else {
                cnt.appendChild(document.createTextNode(tState.thinkBuf));
              }
              tState.thinkBuf = '';
              // autoscroll-coverage:start
              autoScrollStreamed(cnt);
              // autoscroll-coverage:end
            });
          }
        }
        break;
      case 'thinking_end':
        if (tState.thinkRaf) {
          cancelAnimationFrame(tState.thinkRaf);
          tState.thinkRaf = 0;
          if (tState.thinkCnt && tState.thinkBuf) {
            const cnt = tState.thinkCnt;
            const last = cnt.lastChild;
            if (last && last.nodeType === 3) last.appendData(tState.thinkBuf);
            else cnt.appendChild(document.createTextNode(tState.thinkBuf));
            // autoscroll-coverage:start
            autoScrollStreamed(cnt);
            // autoscroll-coverage:end
          }
          tState.thinkBuf = '';
        }
        tState.thinkEl = null;
        tState.thinkCnt = null;
        break;
      case 'text_delta':
        if (!tState.txtEl) {
          tState.txtEl = mkEl('div', 'txt');
          target.appendChild(tState.txtEl);
          tState.txtBuf = '';
          tState.txtNode = document.createTextNode('');
          tState.txtEl.appendChild(tState.txtNode);
          tState.txtPending = '';
          tState.txtRaf = 0;
        }
        {
          const _td = ev.text || '';
          tState.txtBuf += _td;
          tState.txtPending += _td.replace(/\n\n+/g, '\n');
        }
        if (!tState.txtRaf) {
          tState.txtRaf = requestAnimationFrame(() => {
            tState.txtRaf = 0;
            if (tState.txtNode && tState.txtPending) {
              tState.txtNode.appendData(tState.txtPending);
              // autoscroll-coverage:start
              autoScrollStreamed(tState.txtEl);
              // autoscroll-coverage:end
            }
            tState.txtPending = '';
          });
        }
        break;
      case 'text_end':
        if (tState.txtRaf) {
          cancelAnimationFrame(tState.txtRaf);
          tState.txtRaf = 0;
        }
        if (tState.txtEl) {
          if (typeof marked !== 'undefined') {
            tState.txtEl.classList.add('md-body');
            tState.txtEl.innerHTML = kissSanitize(
              marked.parse(tState.txtBuf || ''),
            );
            hlBlock(tState.txtEl);
            tState.txtEl.dataset.rawText = tState.txtBuf || '';
          } else if (tState.txtNode && tState.txtPending) {
            tState.txtNode.appendData(tState.txtPending);
          }
          linkifyFilePaths(tState.txtEl, evWorkDir, evOwnerTab);
          tState.txtEl = null;
          tState.txtBuf = '';
          tState.txtNode = null;
          tState.txtPending = '';
        }
        break;
      case 'tool_call': {
        if (tState.bashPanel && tState.bashBuf) {
          tState.bashPanel.textContent += tState.bashBuf;
          tState.bashBuf = '';
          linkifyFilePaths(tState.bashPanel, evWorkDir, evOwnerTab);
          // autoscroll-coverage:start
          // The flushed text belongs to the PREVIOUS tool's bash
          // subpanel; scroll it now, before this new tool call
          // becomes the latest event panel.
          autoScrollStreamed(tState.bashPanel);
          // autoscroll-coverage:end
        }
        tState.bashPanel = null;
        tState.bashRaf = 0;
        // report-coverage:start
        stashPendingReport(tState, ev);
        // report-coverage:end
        const c = mkEl('div', 'ev tc');
        const hdr = mkEl('div', 'tc-h');
        hdr.textContent = ev.name || 'Tool';
        if (ev.name === 'Bash') {
          hdr.classList.add('tc-h-bash');
          c.classList.add('tc-bash');
        }
        if (ev.name === 'run_parallel') {
          c.classList.add('tc-run-parallel');
          if (ev.tabId !== undefined && ev.tabId !== null)
            c._rpParentTabId = ev.tabId;
          tState.runParallelCount = (tState.runParallelCount || 0) + 1;
          c._rpCallIndex = tState.runParallelCount;
          c._rpExpectedCount = rpExpectedTaskCount(
            ev.extras ? ev.extras.tasks : undefined,
          );
        }
        const isSummary = ev.name === 'summary';
        if (isSummary) {
          // summaryhint-coverage:start
          c.classList.add('tc-summary');
          const hint = mkEl('span', 'tc-summary-hint');
          hint.textContent = ' (click to expand)';
          hint.dataset.rawText = '';
          hdr.appendChild(hint);
          // summaryhint-coverage:end
        }
        let b = '';
        if (ev.path) {
          const ep = esc(ev.path).replace(/"/g, '&quot;');
          b +=
            '<div class="tc-arg"><span class="tc-arg-name">path:</span> <span class="tp" data-path-candidate="' +
            ep +
            '">' +
            esc(ev.path) +
            '</span></div>';
        }
        if (ev.description && !isSummary)
          b +=
            '<div class="tc-arg"><span class="tc-arg-name">description:</span> ' +
            esc(ev.description) +
            '</div>';
        if (ev.command)
          b +=
            '<pre><code class="language-bash">' +
            esc(ev.command) +
            '</code></pre>';
        if (ev.content) {
          const lc = ev.lang ? 'language-' + esc(ev.lang) : '';
          b +=
            '<pre><code class="' +
            lc +
            '">' +
            esc(ev.content) +
            '</code></pre>';
        }
        if (ev.old_string !== undefined && ev.new_string !== undefined) {
          b += renderDiff(ev.old_string, ev.new_string);
        } else {
          if (ev.old_string !== undefined)
            b += '<div class="diff-old">- ' + esc(ev.old_string) + '</div>';
          if (ev.new_string !== undefined)
            b += '<div class="diff-new">+ ' + esc(ev.new_string) + '</div>';
        }
        if (ev.extras) {
          for (const k in ev.extras) {
            if (k === 'audioB64' || k === 'audioMime') continue;
            b +=
              '<div class="extra">' +
              esc(k) +
              ': ' +
              esc(ev.extras[k]) +
              '</div>';
          }
        }
        const tcBody = mkEl('div', 'tc-b');
        tcBody.innerHTML =
          b || '<em style="color:var(--dim)">No arguments</em>';
        c.appendChild(hdr);
        if (isSummary) {
          const sd = mkEl('div', 'tc-summary-desc');
          const rawDesc = ev.description || '';
          if (typeof marked !== 'undefined' && rawDesc) {
            sd.classList.add('md-body');
            sd.innerHTML = kissSanitize(marked.parse(rawDesc));
            hlBlock(sd);
            linkifyFilePaths(sd, evWorkDir, evOwnerTab);
          } else {
            sd.textContent = rawDesc;
          }
          sd.dataset.rawText = rawDesc;
          c.appendChild(sd);
        } else {
          c.appendChild(tcBody);
          verifyFileLinkCandidates(tcBody, evWorkDir, evOwnerTab);
        }
        addCollapse(c, hdr, ev.ts);
        target.appendChild(c);
        if (isSummary) {
          const sub = mkEl('div', 'summary-sub');
          const adopt = [];
          let sib = c.previousElementSibling;
          while (sib) {
            if (
              sib.classList.contains('tc-summary') ||
              sib.classList.contains('prompt') ||
              sib.classList.contains('system-prompt') ||
              sib.classList.contains('adjacent-task') ||
              sib.classList.contains('rc')
            )
              break;
            if (
              !sib.classList.contains('ev') &&
              !sib.classList.contains('llm-panel')
            )
              break;
            adopt.push(sib);
            sib = sib.previousElementSibling;
          }
          for (let ai = adopt.length - 1; ai >= 0; ai--)
            sub.appendChild(adopt[ai]);
          c.appendChild(sub);
          c.classList.add('collapsed');
          // The adopted panels are now hidden behind this collapsed
          // summary; a fan-out panel among them must give its
          // sub-agent tabs up like any other collapsed fan-out.
          collapseNestedRunParallel(c);
        }
        tState.lastToolCallEl = c;
        stampPanelStart(c);
        if (ev.command) {
          const bp = mkEl('div', 'bash-panel');
          const bpContent = mkEl('div', 'bash-panel-content');
          bp.appendChild(bpContent);
          addCopyButton(bp);
          c.appendChild(bp);
          tState.bashPanel = bpContent;
        }
        hlBlock(c);
        break;
      }
      case 'tool_result': {
        if (tState.bashPanel && tState.bashBuf) {
          tState.bashPanel.textContent += tState.bashBuf;
          tState.bashBuf = '';
          linkifyFilePaths(tState.bashPanel, evWorkDir, evOwnerTab);
        } else if (tState.bashPanel) {
          linkifyFilePaths(tState.bashPanel, evWorkDir, evOwnerTab);
        }
        const hadBash = !!tState.bashPanel;
        tState.bashPanel = null;
        tState.bashRaf = 0;
        if (tState.lastToolCallEl) finalizePanelTime(tState.lastToolCallEl);
        if (
          tState.lastToolCallEl &&
          tState.lastToolCallEl.classList.contains('tc-run-parallel')
        ) {
          tState.lastToolCallEl._rpDone = true;
        }
        // report-coverage:start
        if (ev.is_error) tState.pendingReport = null;
        else confirmReadyReport(tState, ev);
        // report-coverage:end
        if (hadBash && !ev.is_error) break;
        const resultTarget = tState.lastToolCallEl || target;
        if (ev.is_error) {
          const r = mkEl('div', 'ev tr err');
          r.innerHTML =
            '<div class="rl fail">FAILED</div><div class="tr-content">' +
            esc(ev.content) +
            '</div>';
          r.dataset.rawText = 'FAILED\n' + (ev.content || '');
          addCollapse(
            r,
            r.querySelector('.rl'),
            tState.lastToolCallEl ? undefined : ev.ts,
          );
          resultTarget.appendChild(r);
          const trBody = r.querySelector('.tr-content');
          if (trBody) linkifyFilePaths(trBody, evWorkDir, evOwnerTab);
        } else {
          const op = mkEl('div', 'bash-panel');
          const opContent = mkEl('div', 'bash-panel-content');
          opContent.textContent = ev.content;
          linkifyFilePaths(opContent, evWorkDir, evOwnerTab);
          op.appendChild(opContent);
          addCopyButton(op);
          if (!tState.lastToolCallEl) addPanelTimestamp(op, ev.ts);
          resultTarget.appendChild(op);
        }
        break;
      }
      case 'system_output': {
        if (tState.bashPanel) {
          if (!tState.bashBuf) tState.bashBuf = '';
          tState.bashBuf += ev.text || '';
          if (!tState.bashRaf) {
            tState.bashRaf = requestAnimationFrame(() => {
              if (tState.bashPanel) {
                tState.bashPanel.textContent += tState.bashBuf;
                linkifyFilePaths(tState.bashPanel, evWorkDir, evOwnerTab);
                // autoscroll-coverage:start
                autoScrollStreamed(tState.bashPanel);
                // autoscroll-coverage:end
              }
              tState.bashBuf = '';
              tState.bashRaf = 0;
            });
          }
        } else {
          const s = mkEl('div', 'ev sys');
          s.textContent = (ev.text || '').replace(/\n\n+/g, '\n');
          linkifyFilePaths(s, evWorkDir, evOwnerTab);
          target.appendChild(s);
        }
        break;
      }
      case 'result': {
        const multiSummary = splitMultiSessionSummary(ev.summary);
        if (multiSummary) {
          removeResultPanels(target);
          target.appendChild(
            createResultPanel(
              ev,
              multiSummary.previous,
              'Previous Sessions',
              false,
              evWorkDir,
              evOwnerTab,
            ),
          );
          target.appendChild(
            createResultPanel(
              ev,
              multiSummary.final,
              'Result',
              true,
              evWorkDir,
              evOwnerTab,
            ),
          );
        } else {
          target.appendChild(
            createResultPanel(
              ev,
              undefined,
              'Result',
              true,
              evWorkDir,
              evOwnerTab,
            ),
          );
        }
        if (statusTokens && ev.total_tokens)
          statusTokens.textContent = 'Tokens: ' + fmtN(ev.total_tokens);
        if (statusBudget && ev.cost && ev.cost !== 'N/A')
          statusBudget.textContent = 'Cost: ' + ev.cost;
        if (ev.step_count) updateStepCount(ev.step_count);
        break;
      }
      case 'system_prompt':
      case 'prompt': {
        const cls = t === 'system_prompt' ? 'system-prompt' : 'prompt';
        const label = t === 'system_prompt' ? 'System Prompt' : 'Prompt';
        let el = null;
        if (!ev.early) {
          const pending = target.querySelectorAll(
            '.ev.' + cls + '[data-early="1"]',
          );
          if (pending.length) el = pending[pending.length - 1];
        }
        const fresh = !el;
        if (fresh) el = mkEl('div', 'ev ' + cls);
        const body =
          typeof marked !== 'undefined'
            ? kissSanitize(marked.parse(ev.text || ''))
            : esc(ev.text || '');
        el.innerHTML =
          '<div class="' +
          cls +
          '-h">' +
          label +
          '</div>' +
          '<div class="' +
          cls +
          '-body md-body">' +
          body +
          '</div>';
        if (ev.early) {
          el.dataset.early = '1';
        } else {
          delete el.dataset.early;
        }
        el.dataset.rawText = ev.text || '';
        addCollapse(el, el.querySelector('.' + cls + '-h'), ev.ts);
        hlBlock(el);
        if (fresh) target.appendChild(el);
        const bodyEl = el.querySelector('.' + cls + '-body');
        if (bodyEl) {
          linkifyFilePaths(bodyEl, evWorkDir, evOwnerTab);
        }
        break;
      }
      case 'usage_info': {
        if (ev.total_tokens != null && ev.cost != null) {
          if (statusTokens)
            statusTokens.textContent = 'Tokens: ' + fmtN(ev.total_tokens);
          if (statusBudget && ev.cost !== 'N/A')
            statusBudget.textContent = 'Cost: ' + ev.cost;
          if (statusSteps && ev.total_steps != null)
            statusSteps.textContent = 'Steps: ' + ev.total_steps;
        } else {
          updateUsageMetrics(ev.text || '');
        }
        break;
      }
      case 'autocommit_progress':
      case 'worktree_progress': {
        renderActionProgress(target, ev.message || ev.text || '');
        break;
      }
      case 'autocommit_done': {
        clearActionProgress(target);
        // A successful manual Git Commit is reported by a toast
        // notification instead; only failures earn transcript text.
        if (ev && ev.manual && ev.success) break;
        const cls2 = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
        const acDiv = mkEl('div', 'ev ' + cls2);
        acDiv.textContent = (ev && ev.message) || '';
        target.appendChild(acDiv);
        break;
      }
      case 'warning': {
        const warnDiv = mkEl('div', 'ev tr warn');
        warnDiv.innerHTML =
          '<strong>Warning:</strong> ' + esc(ev.message || ev.text || '');
        target.appendChild(warnDiv);
        break;
      }
      case 'error': {
        const errDiv = mkEl('div', 'ev tr err');
        errDiv.innerHTML =
          '<strong>Error:</strong> ' + esc(ev.text || ev.message || '');
        target.appendChild(errDiv);
        break;
      }
    }
  }

  function updateStepCount(count) {
    stepCount = count;
    // visibletask-coverage:start
    // Remembered as well as painted: the status row is lent out to the
    // neighbouring tasks the reader scrolls through, and this is the
    // number it has to come back to.
    currentTaskMetrics.steps = 'Steps: ' + count;
    // visibletask-coverage:end
    if (statusSteps) statusSteps.textContent = 'Steps: ' + count;
  }

  // "Steps: 7/100" -- the daemon writes its step count into the usage
  // line as well as into the numeric fields beside it.
  const STEPS_TEXT_RE = /Steps:\s*(\d+)\/\d+/;

  /**
   * The step count the daemon itself reports in *ev*, or 0.
   *
   * ``usage_info`` carries it as ``total_steps``, and -- for a payload
   * without the numeric fields -- inside its ``Steps: N/M`` text, which
   * is the same pair of forms the renderer reads.
   *
   * @param {object} ev A usage_info event.
   * @returns {number} The reported count, or 0 when it reports none.
   */
  function reportedStepCount(ev) {
    if (ev.total_steps != null) return ev.total_steps;
    const m = STEPS_TEXT_RE.exec(ev.text || '');
    return m ? parseInt(m[1], 10) : 0;
  }

  /**
   * The mutable state of ONE transcript's event stream.
   *
   * Three transcripts run the same state machine: the visible tab, a
   * background tab's detached fragment, and a replay. Holding that state
   * in one shape -- and the machine itself in one place (streamBegin /
   * streamEnd) -- is what stops the three from drifting apart.
   *
   * @param {Element|DocumentFragment} container Where panels are added.
   * @param {string} tabId The tab that owns the transcript.
   * @returns {object} A fresh stream context.
   */
  function mkStreamCtx(container, tabId) {
    return {
      container: container,
      tabId: tabId,
      // Renderer state for events that land in the transcript itself.
      state: mkS(),
      lastToolName: '',
      llmPanel: null,
      llmPanelState: mkS(),
      pendingPanel: false,
      stepCount: 0,
      // Called with the new count whenever a step is counted.
      onStep: null,
    };
  }

  function streamCountStep(ctx) {
    ctx.stepCount += 1;
    if (ctx.onStep) ctx.onStep(ctx.stepCount);
  }

  function streamOpenThoughts(ctx, ts, provisional) {
    const panel = mkThoughtsPanel(ts);
    if (provisional) panel._provisional = true;
    ctx.llmPanel = panel;
    ctx.container.appendChild(panel);
    collapseOlderPanels(ctx.container, ctx.tabId);
    ctx.llmPanelState = mkS();
    ctx.pendingPanel = false;
  }

  const STREAM_PANEL_TYPES = new Set([
    'thinking_start',
    'thinking_delta',
    'thinking_end',
    'text_delta',
    'text_end',
  ]);

  /**
   * Apply the panel transitions an event triggers before it is rendered.
   *
   * A tool_call ends the current thoughts panel, a tool_result arms the
   * next one, and the first thinking_start or text_delta after that
   * opens it and counts a step.
   *
   * @param {object} ctx The transcript's stream context.
   * @param {object} ev The event about to be rendered.
   * @returns {{target: (Element|DocumentFragment), state: object}} Where
   *   the event must be rendered, and the renderer state to use.
   */
  function streamBegin(ctx, ev) {
    const t = ev.type;
    if (t === 'tool_call') {
      ctx.lastToolName = ev.name || '';
      if (ctx.llmPanel && ctx.llmPanel._provisional)
        discardProvisionalPanel(ctx.llmPanel);
      else if (ctx.llmPanel) finalizePanelTime(ctx.llmPanel);
      ctx.llmPanel = null;
      ctx.llmPanelState = mkS();
      ctx.pendingPanel = true;
    }
    if (t === 'tool_result' && ctx.lastToolName !== 'finish') {
      ctx.pendingPanel = true;
    }
    const opensPanel = t === 'thinking_start' || t === 'text_delta';
    if (ctx.llmPanel && ctx.llmPanel._provisional && opensPanel) {
      streamCountStep(ctx);
      ctx.llmPanel._provisional = false;
    } else if ((ctx.pendingPanel || ctx.stepCount === 0) && opensPanel) {
      streamCountStep(ctx);
      streamOpenThoughts(ctx, ev.ts, false);
    }
    if (ctx.llmPanel && STREAM_PANEL_TYPES.has(t)) {
      return {target: ctx.llmPanel, state: ctx.llmPanelState};
    }
    return {target: ctx.container, state: ctx.state};
  }

  /**
   * Apply the panel transitions that follow an event being rendered.
   *
   * @param {object} ctx The transcript's stream context.
   * @param {object} ev The event that was just rendered.
   * @param {Element|DocumentFragment} target Where it was rendered.
   */
  function streamEnd(ctx, ev, target) {
    const t = ev.type;
    if (target === ctx.container) {
      collapseOlderPanels(ctx.container, ctx.tabId);
    }
    if (t === 'tool_result' && ctx.lastToolName !== 'finish' && !ctx.llmPanel) {
      // The agent is thinking again; the panel its words will land in is
      // opened now so the transcript does not sit empty, and withdrawn
      // again if nothing is ever said into it.
      streamOpenThoughts(ctx, ev.ts, true);
    }
    if (t === 'usage_info' && ctx.stepCount > 0) {
      // The daemon's own count outranks the panel counting, which only
      // estimates the steps between two of its reports -- a run_parallel
      // fan-out reports the sub-agents' steps too, so the estimate is
      // far behind. Adopted only once this transcript has counted a step
      // of its own: until then stepCount === 0 is also what tells
      // streamBegin the first thoughts panel is still to be opened, and
      // the daemon reports a step in progress before its first token.
      const reported = reportedStepCount(ev);
      if (reported) ctx.stepCount = reported;
    }
    if (t === 'result') {
      if (ctx.llmPanel && ctx.llmPanel._provisional)
        discardProvisionalPanel(ctx.llmPanel);
      else if (ctx.llmPanel) finalizePanelTime(ctx.llmPanel);
      ctx.llmPanel = null;
      // The daemon's own count is the authoritative one.
      if (ev.step_count) ctx.stepCount = ev.step_count;
      collapseAllExceptResult(ctx.container, ctx.tabId);
      if (ev.success === false && !ev.is_continue) {
        const rTab = getTab(ctx.tabId);
        if (rTab) rTab.lastTaskFailed = true;
      }
      ctx.pendingPanel = true;
    }
  }

  // The visible transcript's stream state lives in module globals
  // because a tab switch saves and restores them; they are lent to the
  // shared machine for the length of one event.
  function liveStreamCtx() {
    return {
      container: O,
      tabId: activeTabId,
      state: state,
      lastToolName: lastToolName,
      llmPanel: llmPanel,
      llmPanelState: llmPanelState,
      pendingPanel: pendingPanel,
      stepCount: stepCount,
      onStep: updateStepCount,
    };
  }

  function saveLiveStreamCtx(ctx) {
    lastToolName = ctx.lastToolName;
    llmPanel = ctx.llmPanel;
    llmPanelState = ctx.llmPanelState;
    pendingPanel = ctx.pendingPanel;
    stepCount = ctx.stepCount;
  }

  function processOutputEvent(ev) {
    normalizeEventTs(ev);
    // visibletask-coverage:start
    // A live event reads and rewrites the status row, so the row must be
    // showing the live task's own numbers while it is handled — the
    // reader may have left it on a neighbouring task. updateVisibleTask()
    // at the end hands it back.
    showLiveMetrics();
    // visibletask-coverage:end
    const ctx = liveStreamCtx();
    const t = ev.type;
    const where = streamBegin(ctx, ev);
    const target = where.target;
    const tState = where.state;
    handleOutputEvent(ev, target, tState);
    // autoscroll-coverage:start
    // Capture the latest event panel now: right below, a provisional
    // thoughts panel may be appended after a tool_result, which would
    // hide the tool panel that actually received the result output.
    const autoScrollPanel =
      target !== O
        ? target
        : (t === 'tool_result' && tState.lastToolCallEl) || O.lastElementChild;
    // autoscroll-coverage:end
    streamEnd(ctx, ev, target);
    saveLiveStreamCtx(ctx);
    if (t === 'result' || t === 'usage_info') {
      currentTaskMetrics.tokens = statusTokens ? statusTokens.textContent : '';
      currentTaskMetrics.budget = statusBudget ? statusBudget.textContent : '';
      currentTaskMetrics.steps = statusSteps ? statusSteps.textContent : '';
    }
    // autoscroll-coverage:start
    autoScrollLatestEventPanel(autoScrollPanel);
    // autoscroll-coverage:end
    applyChevronState(currentTaskName);
    // visibletask-coverage:start
    // The event may have changed the transcript's shape as well as the
    // status row, so both the panel and the row are re-derived from what
    // is actually on screen.
    updateVisibleTask();
    // visibletask-coverage:end
  }

  function processOutputEventForBgTab(ev, tab) {
    normalizeEventTs(ev);
    if (!tab.outputFragment)
      tab.outputFragment = document.createDocumentFragment();

    const ctx = mkStreamCtx(tab.outputFragment, tab.id);
    ctx.state = tab.streamState || mkS();
    ctx.lastToolName = tab.streamLastToolName || '';
    ctx.llmPanel = tab.streamLlmPanel || null;
    ctx.llmPanelState = tab.streamLlmPanelState || mkS();
    ctx.pendingPanel = tab.streamPendingPanel || false;
    ctx.stepCount = tab.streamStepCount || 0;
    ctx.onStep = count => {
      tab.statusStepsText = 'Steps: ' + count;
    };

    const where = streamBegin(ctx, ev);
    const target = where.target;

    // The window owns ONE status row, so the hidden tab's numbers are
    // lent to it for the length of this event and taken back after. That
    // is what lets a background tab go through exactly the same
    // renderers as a visible one — usage_info in both its numeric and
    // its text form included.
    const prevStepCount = stepCount;
    const prevTokensText = statusTokens ? statusTokens.textContent : '';
    const prevBudgetText = statusBudget ? statusBudget.textContent : '';
    const prevStepsText = statusSteps ? statusSteps.textContent : '';
    // visibletask-coverage:start
    // A hidden tab's event runs through the same renderers, so it also
    // moves the visible tab's remembered numbers unless they are put
    // back with the status row below.
    const prevMetrics = currentTaskMetrics;
    const prevVisibleTab = activeTabId;
    currentTaskMetrics = {tokens: '', budget: '', steps: ''};
    // visibletask-coverage:end
    if (statusTokens) statusTokens.textContent = tab.statusTokensText || '';
    if (statusBudget) statusBudget.textContent = tab.statusBudgetText || '';
    if (statusSteps) statusSteps.textContent = tab.statusStepsText || '';

    handleOutputEvent(
      ev,
      target,
      where.state,
      tab.workDir || configWorkDir || '',
      tab.id,
    );

    if (statusTokens) tab.statusTokensText = statusTokens.textContent;
    if (statusBudget) tab.statusBudgetText = statusBudget.textContent;
    if (statusSteps) tab.statusStepsText = statusSteps.textContent;

    stepCount = prevStepCount;
    if (statusTokens) statusTokens.textContent = prevTokensText;
    if (statusBudget) statusBudget.textContent = prevBudgetText;
    if (statusSteps) statusSteps.textContent = prevStepsText;
    // visibletask-coverage:start
    // Collapsing a finished run_parallel panel closes its sub-agent
    // tabs, so this event may have swapped the tab on screen; the
    // borrowed numbers only go back to the tab they came from.
    if (activeTabId === prevVisibleTab) currentTaskMetrics = prevMetrics;
    // visibletask-coverage:end

    streamEnd(ctx, ev, target);
    if (ev.type === 'result' && ev.step_count) {
      tab.statusStepsText = 'Steps: ' + ev.step_count;
    }

    tab.streamState = ctx.state;
    tab.streamLlmPanel = ctx.llmPanel;
    tab.streamLlmPanelState = ctx.llmPanelState;
    tab.streamLastToolName = ctx.lastToolName;
    tab.streamPendingPanel = ctx.pendingPanel;
    tab.streamStepCount = ctx.stepCount;
    tab.welcomeVisible = false;
  }

  function accumulateOverscroll(dir, delta, taskId) {
    if (taskId === undefined || taskId === null || taskId === '') return;
    if (overscrollDir !== dir) {
      overscrollAccum = 0;
      overscrollDir = dir;
    }
    overscrollAccum += Math.abs(delta);
    clearTimeout(overscrollTimer);
    overscrollTimer = setTimeout(() => {
      overscrollAccum = 0;
      overscrollDir = '';
    }, 500);
    if (overscrollAccum >= OVERSCROLL_THRESHOLD) {
      overscrollAccum = 0;
      overscrollDir = '';
      adjacentLoading = true;
      showAdjacentLoader(dir);
      api.getAdjacentTask({tabId: activeTabId, taskId: taskId, direction: dir});
    }
  }

  O.addEventListener('wheel', e => {
    const _activeTabForAdj = getTab(activeTabId);
    const _isSubagentActive = !!(
      _activeTabForAdj && _activeTabForAdj.isSubagentTab
    );
    if (
      !adjacentLoading &&
      activeTabId &&
      currentTaskName &&
      !_isSubagentActive
    ) {
      const atTop = O.scrollTop <= 0;
      const atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 2;

      if (atTop && e.deltaY < 0 && !noPrevTask && oldestLoadedTaskId != null) {
        accumulateOverscroll('prev', e.deltaY, oldestLoadedTaskId);
      } else if (
        atBottom &&
        e.deltaY > 0 &&
        !noNextTask &&
        newestLoadedTaskId != null
      ) {
        accumulateOverscroll('next', e.deltaY, newestLoadedTaskId);
      } else {
        overscrollAccum = 0;
        overscrollDir = '';
      }
    }
  });

  let _touchOutputLastY = 0;

  O.addEventListener(
    'touchstart',
    e => {
      if (e.touches.length === 1) {
        _touchOutputLastY = e.touches[0].clientY;
      }
    },
    {passive: true},
  );

  O.addEventListener(
    'touchmove',
    e => {
      if (e.touches.length !== 1) return;
      const currentY = e.touches[0].clientY;
      const touchDelta = _touchOutputLastY - currentY;
      _touchOutputLastY = currentY;

      if (adjacentLoading || !activeTabId || !currentTaskName) return;
      const _activeTabT = getTab(activeTabId);
      if (_activeTabT && _activeTabT.isSubagentTab) return;

      const atTop = O.scrollTop <= 0;
      const atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 2;

      if (
        atTop &&
        touchDelta < 0 &&
        !noPrevTask &&
        oldestLoadedTaskId != null
      ) {
        accumulateOverscroll('prev', touchDelta, oldestLoadedTaskId);
      } else if (
        atBottom &&
        touchDelta > 0 &&
        !noNextTask &&
        newestLoadedTaskId != null
      ) {
        accumulateOverscroll('next', touchDelta, newestLoadedTaskId);
      } else {
        overscrollAccum = 0;
        overscrollDir = '';
      }
    },
    {passive: true},
  );

  O.addEventListener(
    'touchend',
    () => {
      overscrollAccum = 0;
      overscrollDir = '';
      if (overscrollTimer) {
        clearTimeout(overscrollTimer);
        overscrollTimer = null;
      }
    },
    {passive: true},
  );

  // visibletask-coverage:start
  /**
   * Split #output into task regions, top to bottom.
   *
   * Every spliced-in neighbour is one `.adjacent-task` container; each run
   * of plain children between them belongs to the task this tab is on.
   */
  function getTaskRegions() {
    const regions = [];
    let mainFirst = null;
    let mainLast = null;
    const children = O.children;
    for (let i = 0; i < children.length; i++) {
      const el = children[i];
      if (el.id === 'welcome' || el.id === 'adjacent-loader') continue;
      if (el.classList.contains('chv-hidden')) continue;
      if (el.classList.contains('adjacent-task')) {
        if (mainFirst) {
          regions.push({
            task: currentTaskName,
            first: mainFirst,
            last: mainLast,
          });
          mainFirst = null;
          mainLast = null;
        }
        regions.push({task: el.dataset.task || '', first: el, last: el});
      } else {
        if (!mainFirst) mainFirst = el;
        mainLast = el;
      }
    }
    if (mainFirst)
      regions.push({task: currentTaskName, first: mainFirst, last: mainLast});
    return regions;
  }

  /**
   * Index of the region the reader is looking at: the one owning the most
   * visible pixels, ties going to the upper region.
   *
   * Visible height, rather than a fixed probe line, is what keeps the
   * first and the last region selectable. The scroller clamps at both
   * ends, so a task shorter than the probe offset can never be moved onto
   * that line even when its events are the only ones worth reading.
   *
   * The overlap is deliberately left signed: for a viewport that sits off
   * the transcript entirely it degrades into "the nearest region", which
   * is the first one above the content and the last one below it.
   */
  function getVisibleRegionIndex(regions) {
    const pinned = wheelPinnedTarget();
    if (pinned)
      for (let i = 0; i < regions.length; i++)
        if (regions[i].first === pinned.el) return i;
    const outputRect = O.getBoundingClientRect();
    let bestIdx = 0;
    let bestVisible = -Infinity;
    for (let i = 0; i < regions.length; i++) {
      const top = regions[i].first.getBoundingClientRect().top;
      const bottom = regions[i].last.getBoundingClientRect().bottom;
      const visible =
        Math.min(bottom, outputRect.bottom) - Math.max(top, outputRect.top);
      if (visible > bestVisible) {
        bestVisible = visible;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  /**
   * The task region the reader is looking at, or null when the transcript
   * holds nothing but this tab's own task.
   *
   * With no neighbour spliced in there is nothing to disambiguate, and the
   * panel may be showing a read-only history preview that has no region of
   * its own, so callers must leave the panel alone.
   */
  function visibleRegion() {
    if (!O.querySelector('.adjacent-task[data-task]')) return null;
    const regions = getTaskRegions();
    return regions[getVisibleRegionIndex(regions)];
  }

  /** The `.adjacent-task` container of a region, null for the tab's own. */
  function regionNeighbour(region) {
    return region.first.classList.contains('adjacent-task')
      ? region.first
      : null;
  }

  /**
   * Put the live task's own numbers back into the shared status row.
   *
   * The row is lent to whichever neighbouring task the reader scrolls
   * into, so anything that works with the live task's numbers has to
   * reclaim it first.
   */
  function showLiveMetrics() {
    if (!O.querySelector('.adjacent-task[data-task]')) return;
    if (statusTokens) statusTokens.textContent = currentTaskMetrics.tokens;
    if (statusBudget) statusBudget.textContent = currentTaskMetrics.budget;
    if (statusSteps) statusSteps.textContent = currentTaskMetrics.steps;
  }

  function updateVisibleTask() {
    const region = visibleRegion();
    if (!region) return;
    const container = regionNeighbour(region);
    setTaskText(region.task || currentTaskName);
    if (container) {
      if (statusTokens)
        statusTokens.textContent = container.dataset.metricTokens || '';
      if (statusBudget)
        statusBudget.textContent = container.dataset.metricBudget || '';
      if (statusSteps)
        statusSteps.textContent = container.dataset.metricSteps || '';
    } else {
      if (statusTokens) statusTokens.textContent = currentTaskMetrics.tokens;
      if (statusBudget) statusBudget.textContent = currentTaskMetrics.budget;
      if (statusSteps) statusSteps.textContent = currentTaskMetrics.steps;
    }
  }
  // visibletask-coverage:end

  O.addEventListener('scroll', () => {
    // autoscroll-coverage:start
    updateUserScrollLock();
    // autoscroll-coverage:end
    updateVisibleTask();
  });

  const TASK_WHEEL_STEP = 60;
  // taskwheel-coverage:start
  let taskWheelAccum = 0;
  let taskWheelDir = '';
  let taskWheelTimer = null;
  let taskWheelPendingDir = '';
  let taskWheelLastTarget = null;

  function wheelPinnedTarget() {
    if (!taskWheelLastTarget) return null;
    if (O.scrollTop !== taskWheelLastTarget.scrollTop) {
      taskWheelLastTarget = null;
      return null;
    }
    if (!O.contains(taskWheelLastTarget.el)) {
      taskWheelLastTarget = null;
      return null;
    }
    return taskWheelLastTarget;
  }

  function scrollTaskRegionToTop(region) {
    const outputRect = O.getBoundingClientRect();
    const top = region.first.getBoundingClientRect().top;
    O.scrollTop += top - outputRect.top;
    taskWheelLastTarget = {el: region.first, scrollTop: O.scrollTop};
    updateVisibleTask();
  }

  // taskwheel-coverage:end
  /**
   * Scroll the transcript so the task with `taskId` sits at the top of
   * the viewport.  `scrollTaskRegionToTop` pins the region and re-derives
   * the static task panel, so the clicked task is what the panel names.
   *
   * Returns true when the task is shown by this tab — a region in the
   * transcript, either the tab's own task (`currentTaskId`) or a
   * spliced-in neighbour (`.adjacent-task[data-task-id]`), or the tab's
   * own task with no rendered region (nothing was output yet).  Returns
   * false when the task's events are not loaded, so the caller can
   * fetch them instead.
   */
  function scrollChatToTask(taskId) {
    if (taskId === undefined || taskId === null || taskId === '') return false;
    const idStr = String(taskId);
    const ownIdStr =
      currentTaskId === undefined || currentTaskId === null
        ? ''
        : String(currentTaskId);
    const regions = getTaskRegions();
    for (let i = 0; i < regions.length; i++) {
      const region = regions[i];
      const neighbour = regionNeighbour(region);
      const regionId = neighbour ? neighbour.dataset.taskId || '' : ownIdStr;
      if (regionId === idStr) {
        scrollTaskRegionToTop(region);
        return true;
      }
    }
    // The tab's own task may have no rendered region at all (nothing
    // was output yet), so there is nothing to scroll to and nothing to
    // fetch.  The panel and the status row may still be lent to a
    // spliced-in neighbour the reader scrolled into; reclaim them so
    // the clicked task is what the panel names.
    if (ownIdStr === '' || idStr !== ownIdStr) return false;
    if (O.querySelector('.adjacent-task[data-task]')) {
      taskWheelLastTarget = null;
      setTaskText(currentTaskName);
      showLiveMetrics();
    }
    return true;
  }
  // taskwheel-coverage:start

  function stepTaskFromPanel(dir) {
    const tab = getTab(activeTabId);
    if (tab && tab.isSubagentTab) return;
    const regions = getTaskRegions();
    if (!regions.length) return;
    const idx = getVisibleRegionIndex(regions);
    const targetIdx = dir === 'next' ? idx + 1 : idx - 1;
    if (targetIdx >= 0 && targetIdx < regions.length) {
      scrollTaskRegionToTop(regions[targetIdx]);
      return;
    }
    if (adjacentLoading || !activeTabId || !currentTaskName) return;
    if (dir === 'prev' ? noPrevTask : noNextTask) return;
    const anchorId = dir === 'prev' ? oldestLoadedTaskId : newestLoadedTaskId;
    if (anchorId === undefined || anchorId === null || anchorId === '') return;
    taskWheelPendingDir = dir;
    adjacentLoading = true;
    showAdjacentLoader(dir);
    api.getAdjacentTask({tabId: activeTabId, taskId: anchorId, direction: dir});
  }

  if (taskPanel) {
    taskPanel.addEventListener(
      'wheel',
      e => {
        e.preventDefault();
        e.stopPropagation();
        if (!e.deltaY) return;
        const dir = e.deltaY > 0 ? 'next' : 'prev';
        if (taskWheelDir !== dir) {
          taskWheelAccum = 0;
          taskWheelDir = dir;
        }
        taskWheelAccum += Math.abs(e.deltaY);
        clearTimeout(taskWheelTimer);
        taskWheelTimer = setTimeout(() => {
          taskWheelAccum = 0;
          taskWheelDir = '';
        }, 300);
        if (taskWheelAccum >= TASK_WHEEL_STEP) {
          taskWheelAccum = 0;
          stepTaskFromPanel(dir);
        }
      },
      {passive: false},
    );
  }
  // taskwheel-coverage:end

  let endTs = 0;
  function doneLabelFor(startMs, endMs) {
    const ds = Math.max(0, Math.floor((endMs - startMs) / 1000));
    const dm = Math.floor(ds / 60);
    return 'Done (' + (dm > 0 ? dm + 'm ' : '') + (ds % 60) + 's)';
  }
  function formatDurationHms(ms) {
    const total = Math.max(0, Math.floor(Number(ms) / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = n => (n < 10 ? '0' + n : String(n));
    return pad(h) + ':' + pad(m) + ':' + pad(s);
  }
  function _renderTimerTick() {
    if (endTs > 0 && t0 && Date.now() >= endTs) {
      statusText.textContent = doneLabelFor(t0, endTs);
      stopTimer();
      setRunningState(false);
      return;
    }
    const s = Math.floor((Date.now() - t0) / 1000);
    const m = Math.floor(s / 60);
    statusText.textContent =
      'Running ' + (m > 0 ? m + 'm ' : '') + (s % 60) + 's';
  }
  function startTimer() {
    if (!t0) t0 = Date.now();
    if (timerIv) clearInterval(timerIv);
    statusText.style.color = 'var(--red)';
    _renderTimerTick();
    timerIv = setInterval(_renderTimerTick, 1000);
  }
  function stopTimer() {
    if (timerIv) {
      clearInterval(timerIv);
      timerIv = null;
    }
    statusText.style.color = 'var(--green)';
  }

  function updateUsageMetrics(text) {
    if (!statusTokens || !statusBudget) return;
    const tm =
      text.match(/Context:\s*([\d,]+)\/[\d,]+/) ||
      text.match(/Tokens:\s*([\d,]+)\/[\d,]+/);
    const bm = text.match(/Budget:\s*(\$[0-9.]+)\/\$[0-9.]+/);
    const sm = STEPS_TEXT_RE.exec(text);
    if (tm) statusTokens.textContent = 'Tokens: ' + tm[1];
    if (bm) statusBudget.textContent = 'Cost: ' + bm[1];
    if (sm) updateStepCount(parseInt(sm[1], 10));
  }

  function clearUsageMetrics() {
    if (statusTokens) statusTokens.textContent = '';
    if (statusBudget) statusBudget.textContent = '';
    if (statusSteps) statusSteps.textContent = '';
    stepCount = 0;
    currentTaskMetrics = {tokens: '', budget: '', steps: ''};
  }

  function focusInputWithRetry() {
    inp.focus();
    setTimeout(() => {
      inp.focus();
    }, 100);
    setTimeout(() => {
      inp.focus();
    }, 300);
  }

  function resetHistoryPagination() {
    historyOffset = 0;
    historyHasMore = true;
    historyLoading = false;
    historyGeneration++;
  }

  function refreshHistory() {
    if (sidebar.classList.contains('open')) {
      resetHistoryPagination();
      api.getHistory({
        query: historySearch.value,
        generation: historyGeneration,
      });
    }
  }

  function setServerLoading(loading) {
    const overlay = document.getElementById('kiss-server-loading');
    const app = document.getElementById('app');
    if (overlay) overlay.style.display = loading ? '' : 'none';
    if (app) app.style.display = loading ? 'none' : '';
  }

  const spokenTalkIds = new Set();

  const talkQueue = [];
  let talkQueueBusy = false;

  function pumpTalkQueue() {
    if (talkQueueBusy) return;
    const job = talkQueue.shift();
    if (!job) return;
    talkQueueBusy = true;
    let finished = false;
    job(() => {
      if (finished) return;
      finished = true;
      talkQueueBusy = false;
      pumpTalkQueue();
    });
  }

  function enqueueTalkPlayback(job) {
    talkQueue.push(job);
    pumpTalkQueue();
  }

  function playTalkAudio(ev, onDone) {
    const done = typeof onDone === 'function' ? onDone : function () {};
    let player = null;
    try {
      if (typeof window.Audio !== 'function') return false;
      const mime = ev.audioMime || 'audio/mpeg';
      player = new window.Audio('data:' + mime + ';base64,' + ev.audioB64);
      player.muted = !!ev.muted;
      player.onended = done;
      player.onerror = done;
      player.onabort = done;
      const played = player.play();
      if (played && typeof played.catch === 'function') {
        played.catch(() => {
          player.onended = null;
          player.onerror = null;
          player.onabort = null;
          done();
        });
      }
      return true;
    } catch (_e) {
      if (player) {
        player.onended = null;
        player.onerror = null;
        player.onabort = null;
      }
      return false;
    }
  }

  // tableak-coverage:start
  // The streaming transcript types. Each one carries a fragment of one
  // task's output, so it is meaningless without a tab to attribute it to.
  const TASK_SCOPED_STREAM_TYPES = new Set([
    'thinking_start',
    'thinking_delta',
    'thinking_end',
    'text_delta',
    'text_end',
    'tool_call',
    'tool_result',
    'system_output',
    'system_prompt',
    'prompt',
    'result',
    'usage_info',
  ]);

  /**
   * The task a tab owns, as a string, or '' when it owns none yet.
   *
   * A tab adopts a task id from the first event the daemon sends back, but
   * output can arrive before that. sendMessage() therefore stamps
   * pendingTaskId on the tab it submits from, so the tab is a legitimate
   * owner from the instant the request leaves the webview.
   */
  function tabTaskId(tab) {
    if (!tab) return '';
    const owned = tab.currentTaskId || tab.pendingTaskId;
    return owned === undefined || owned === null ? '' : String(owned);
  }

  /**
   * True when two tabs are showing the SAME task, and so are two views of one
   * conversation rather than two conversations.
   *
   * Task identity is the only sound test. A backend chat can host several
   * tasks, so backendChatId equality would let a tab running task A display
   * task B's transcript -- exactly the leak this module exists to prevent.
   */
  function isSameTaskTab(a, b) {
    if (!a || !b) return false;
    if (a.id === b.id) return true;
    const ta = tabTaskId(a);
    return ta !== '' && ta === tabTaskId(b);
  }

  /**
   * True when the visible tab is the only conversation that could own an
   * unaddressed message.
   *
   * Content tabs (file previews) are not conversations: they never own task
   * output, so they must not make a lone chat look like a crowd. Sub-agent
   * tabs ARE conversations and do count, unless they prove they are showing
   * the same task as the visible tab.
   */
  function isOnlyActiveConversation() {
    const active = getTab(activeTabId);
    if (!active || active.isContentTab) return false;
    return tabs.every(t => t.isContentTab || isSameTaskTab(t, active));
  }

  // Every task-scoped message must prove it belongs to the conversation on
  // screen before it may touch a shared surface (#output, #task-input, the
  // autocomplete dropdown, the ghost overlay, the file-link spans, the toast
  // container). The webview keeps ONE copy of each of those, so an
  // unattributed write is attributed to whichever tab happens to be visible
  // and is then baked into that tab's snapshot by the next saveCurrentTab().
  //
  // The decision therefore FAILS CLOSED: when the message cannot prove where
  // it belongs it is dropped rather than shown in an arbitrary conversation,
  // because guessing is exactly the bug. Proof comes from a tabId, or failing
  // that from a taskId (see isForActiveTaskId). Genuinely global messages
  // (daemon status, model list, remote URL, workspace suggestions, and the
  // user-initiated input commands) never reach this helper -- they are
  // handled by their own cases and stay global by construction.
  //
  // A message addressed to another tab is shown only when that tab is
  // showing the SAME task as the visible one; that is the single exemption
  // the product allows.
  function isForActiveTab(ev) {
    const evTabId = ev ? ev.tabId : undefined;
    if (evTabId === undefined || evTabId === null || evTabId === '') {
      return isForActiveTaskId(ev);
    }
    if (evTabId === activeTabId) return true;
    return isSameTaskTab(getTab(evTabId), getTab(activeTabId));
  }

  // Fallback addressing for a message that names no tab. A taskId names the
  // conversation just as precisely as a tabId would, so the message is not a
  // guess: it belongs to the visible tab exactly when that tab owns the same
  // task.
  //
  // A message with neither id is truly unaddressed. It is safe only when a
  // single conversation is on screen, because then there is no other tab it
  // could belong to and nothing to leak into; dropping it there would merely
  // throw away the output of an ordinary one-tab session.
  function isForActiveTaskId(ev) {
    const evTaskId = ev ? ev.taskId : undefined;
    if (evTaskId === undefined || evTaskId === null || evTaskId === '') {
      return isOnlyActiveConversation();
    }
    const owned = tabTaskId(getTab(activeTabId));
    if (owned === '') return mayPrecedeAdoption(ev);
    return owned === String(evTaskId);
  }

  // The only message types that legitimately reach a tab BEFORE it knows
  // which task it is running. The daemon reports these two aggregates as soon
  // as it starts working, which can be before the reply carrying the task id
  // has been processed, and neither is adopted (see the adoption guard in the
  // streaming branch) -- so a tab that owns nothing yet would never be able
  // to show its own header counters if they were held to the same rule as a
  // transcript.
  const PRE_ADOPTION_TYPES = new Set(['result', 'usage_info']);

  /**
   * True when `ev` may be shown in a visible tab that owns no task yet.
   *
   * "I have not adopted a task" is NOT proof of ownership, so a tab in that
   * state fails closed for everything that carries a task's words: those are
   * only ever produced for the tab that asked for them, and that tab proves
   * itself with a tabId or with the pendingTaskId it claimed when it
   * submitted. Only the header aggregates are exempt, and only because they
   * are the ones that legitimately race ahead of adoption.
   */
  function mayPrecedeAdoption(ev) {
    return !!ev && PRE_ADOPTION_TYPES.has(ev.type);
  }

  /**
   * True when the visible tab may take `ev`'s taskId as its own.
   *
   * Adoption is how a tab decides which task's traffic it will accept from
   * then on, so it must be driven by something that names this tab: a reply
   * addressed to it, or the pendingTaskId it stamped on itself when it
   * submitted. Adopting a bare task id off the wire instead mis-binds the tab
   * to a task another tab is running, and every subsequent event for the task
   * it was really showing is then rejected as foreign.
   */
  function mayAdoptTaskId(ev) {
    if (!ev) return false;
    if (ev.tabId !== undefined && ev.tabId !== null && ev.tabId !== '') {
      return true;
    }
    const tab = getTab(activeTabId);
    return !!tab && !!tab.pendingTaskId;
  }

  /**
   * True when a message names the conversation it belongs to.
   *
   * An addressed message must satisfy isForActiveTab() before it may touch a
   * shared surface. An unaddressed one is judged separately, because for
   * legacy window-level traffic (install toasts, daemon diagnostics) the
   * absence of an id means "everybody", not "nobody".
   */
  function isAddressed(ev) {
    if (!ev) return false;
    const hasTab = ev.tabId !== undefined && ev.tabId !== null;
    const hasTask = ev.taskId !== undefined && ev.taskId !== null;
    return hasTab || hasTask;
  }

  // A spoken transcript belongs to the tab that was on screen when the words
  // were said, not to the tab that happens to be on screen when the audio
  // finishes transcribing. voice.js stamps that tab on the event; an
  // unstamped event predates any tab switch and stays allowed.
  function isFromSpeechTab(event) {
    const detail = event ? event.detail : null;
    const tabId = detail ? detail.tabId : null;
    if (tabId === undefined || tabId === null || tabId === '') return true;
    return isForActiveTab({tabId: tabId});
  }

  /**
   * The conversation on screen, for voice.js.
   *
   * voice.js runs in the same webview but in its own closure, so it reads the
   * visible tab through this accessor rather than a shared variable. It
   * returns the task id too, so voice can apply the same-task exemption that
   * isForActiveTab() applies.
   */
  window.kissVoiceOwner = function () {
    return {tabId: activeTabId, taskId: tabTaskId(getTab(activeTabId))};
  };
  // Retained for callers that only need the visible tab id.
  window.kissActiveTabId = function () {
    return activeTabId;
  };
  // tableak-coverage:end

  // Raised while the daemon is unreachable so the reconnect can
  // re-announce `ready` (tab-registry sync + transcript replay).
  let daemonWasDown = false;

  function handleEvent(ev) {
    const t = ev.type;
    switch (t) {
      case 'daemonStatus':
        setServerLoading(!ev.connected);
        if (!ev.connected) {
          forgetInFlightPathChecks();
          daemonWasDown = true;
        }
        if (ev.connected) {
          // The backend is live, so this window's `ready` is on its way and
          // the running-task news it triggers is about to arrive: the launch
          // starts here (see beginLaunch).
          beginLaunch();
          // A daemon that went away and came back is a fresh daemon as
          // far as this client is concerned (it may have restarted):
          // re-announce `ready` so it re-syncs the shared tab registry
          // and replays the transcripts this window shows. The remote
          // web app reloads the whole page on reconnect instead, so
          // only the VS Code webview takes this path in practice.
          if (daemonWasDown) {
            daemonWasDown = false;
            sendReady();
          }
          // modelpick-coverage:start
          // While the daemon was away this window may have missed both a
          // task ending and its picker hand-back, so no agent override
          // can be trusted any more. The user's own pick is the honest
          // thing to show; an agent still running re-announces its model
          // to any tab that re-joins its task.
          tabs.forEach(t => clearAgentModel(t.id));
          // modelpick-coverage:end
          // The checks the outage swallowed have to be asked again, or
          // the file links they were for stay grey for ever.
          reissueFileLinkChecks();
          refreshHistory();
        }
        return;
      case 'notification':
        // tableak-coverage:start
        // An untagged toast is window-level (install progress, updates);
        // a tagged one belongs to a task and must stay with its tab.
        // The chat tab this window currently REPRESENTS also counts:
        // when a content tab is on screen, actions taken on its
        // behalf (the settings panel's Git Commit targets
        // reportedChatTabId then) must still toast here — the toast
        // container is window-level, not transcript-bound, so nothing
        // can leak into another conversation's transcript.
        if (
          ev.tabId !== undefined &&
          ev.tabId !== reportedChatTabId &&
          !isForActiveTab(ev)
        )
          break;
        // tableak-coverage:end
        updateNotification(ev);
        break;
      case 'fileContent':
        // tableak-coverage:start
        // A file opened for a background task must never pull the user away
        // from the conversation they are reading. It is still the user's
        // file though, so open it in the background rather than throw it
        // away -- the tab is waiting for them when they switch over.
        if (ev.tabId !== undefined && !isForActiveTab(ev)) {
          handleFileContent(ev, false);
          return;
        }
        // tableak-coverage:end
        handleFileContent(ev, true);
        return;
      case 'pathsExist':
        handlePathsExist(ev);
        return;
      case 'status': {
        const evTab = findTabByEvt(ev);
        if (evTab) {
          setTabRunning(evTab, !!ev.running);
          // modelpick-coverage:start
          // Belt and braces for the daemon's `modelPick` restore: a task
          // that stops without one (a killed daemon, a submit refused at
          // shutdown) must still not strand the picker on the model the
          // agent happened to end on.
          if (!ev.running) clearAgentModel(evTab.id);
          // modelpick-coverage:end
        }
        if (ev.running && typeof ev.startTs === 'number' && ev.startTs > 0) {
          if (evTab) {
            evTab.t0 = ev.startTs;
            evTab.endTs = 0;
          }
          if (ev.tabId === undefined || ev.tabId === activeTabId) {
            t0 = ev.startTs;
            endTs = 0;
          }
        }
        if (ev.tabId === undefined || ev.tabId === activeTabId) {
          setRunningState(ev.running);
          if (!ev.running) {
            const stTab = getTab(activeTabId);
            if (stTab && stTab.isSubagentTab && inputContainer)
              inputContainer.style.display = 'none';
          }
          if (ev.running) applyChevronState(currentTaskName);
        }
        renderTabBar();
        refreshHistory();
        syncMobileInputDrawer();
        // Only news of a task that IS running may move the user. A task
        // finishing must not: the launch already brought them to it, and the
        // result they were brought to see is the last thing to pull them off.
        if (ev.running) switchToLatestRunningTab();
        break;
      }
      case 'models':
        allModels = ev.models || [];
        if (ev.selected) {
          // `selected` is the daemon-wide default, so it may only adopt
          // tabs that were still tracking it -- a tab the user gave its
          // own model keeps it. It must also not blank the override of
          // a tab whose agent is still running, hence refreshModelLabel
          // rather than writing the label directly.
          const _prevSelected = selectedModel;
          tabs.forEach(t => {
            const cur = t.selectedModel || '';
            if (cur === '' || cur === 'No model' || cur === _prevSelected) {
              t.selectedModel = ev.selected;
            }
          });
          selectedModel = ev.selected;
          refreshModelLabel();
        }
        renderModelList('');
        break;
      case 'stop_ack':
        // The daemon found nothing to stop for this tab — the click
        // would otherwise have been swallowed in silence, which is
        // exactly what makes people click again.  No running task owns
        // the tab, so its "running" look was stale too: setReady drops
        // the spinner, timer and Stop button along with the message.
        if (!ev.accepted) {
          markStopping(ev.tabId || activeTabId, false);
          setReady('No running task to stop', ev.tabId || activeTabId);
        }
        break;
      // modelpick-coverage:start
      case 'modelPick':
        applyModelPick(ev.tabId || '', ev.model, ev.source);
        break;
      // modelpick-coverage:end
      case 'configData':
        populateConfigForm(ev.config || {}, ev.apiKeys || {});
        break;
      case 'history':
        renderHistory(ev.sessions || [], ev.offset || 0, ev.generation || 0);
        autofillHistoryDateRange(ev.dateRange);
        break;
      case 'frequentTasks':
        renderFrequentTasks(ev.tasks || []);
        break;
      case 'files': {
        // tableak-coverage:start
        if (!isForActiveTab(ev)) break;
        // tableak-coverage:end
        const filesCtx = getAtCtx();
        if (!filesCtx) {
          hideAC();
          break;
        }
        if (ev.prefix !== undefined && ev.prefix !== filesCtx.query) {
          break;
        }
        renderAutocomplete(ev.files || []);
        break;
      }
      case 'askUser': {
        const askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
        const askTab = getTab(askTabId);
        if (!askTab) break;
        const askQuestion = ev.question || '';
        // Duplicate delivery of the SAME pending question (the server
        // re-emits it on every session replay so clients that connect
        // mid-question also show the modal). Re-initializing would
        // wipe the answer the user is typing here just because
        // another client reloaded; a genuinely new question always
        // follows an askUserDone, which resets the state to null.
        if (askTab.askPendingQuestion === askQuestion) break;
        askTab.askPendingQuestion = askQuestion;
        showAskForTab(askTab);
        renderTabBar();
        break;
      }
      case 'askUserDone': {
        const askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
        const askTab = getTab(askTabId);
        if (!askTab) break;
        clearAskForMatchingChatTabs(askTab);
        break;
      }
      case 'talk': {
        if (ev.muted) break;
        const talkText = ev.text || '';
        if (!talkText) break;
        if (ev.tabId !== undefined && !getTab(ev.tabId)) break;
        if (ev.talkId) {
          if (spokenTalkIds.has(ev.talkId)) break;
          spokenTalkIds.add(ev.talkId);
          if (spokenTalkIds.size > 500) {
            const ids = spokenTalkIds.values();
            for (let i = 0; i < 250; i++) {
              spokenTalkIds.delete(ids.next().value);
            }
          }
        }
        enqueueTalkPlayback(finish => {
          if (!ev.audioB64 || !playTalkAudio(ev, finish)) finish();
        });
        break;
      }
      case 'error':
        // tableak-coverage:start
        // Diagnostics are task output like any other: a message that names a
        // task or a tab belongs to that conversation and nowhere else.
        if (isAddressed(ev) && !isForActiveTab(ev)) {
          const bgErrTab = findTabByEvt(ev);
          if (bgErrTab) processOutputEventForBgTab(ev, bgErrTab);
          break;
        }
        // tableak-coverage:end
        addError(ev.text);
        break;
      case 'notice':
        // tableak-coverage:start
        if (isAddressed(ev) && !isForActiveTab(ev)) {
          const bgNoticeTab = findTabByEvt(ev);
          if (bgNoticeTab) processOutputEventForBgTab(ev, bgNoticeTab);
          break;
        }
        // tableak-coverage:end
        addNotice(ev.text);
        break;
      case 'warning': {
        // tableak-coverage:start
        if (isAddressed(ev) && !isForActiveTab(ev)) {
          const bgWarnTab = findTabByEvt(ev);
          if (bgWarnTab) processOutputEventForBgTab(ev, bgWarnTab);
          break;
        }
        // tableak-coverage:end
        addWarning(ev.message || ev.text || '');
        break;
      }
      case 'clear': {
        // report-coverage:start
        // A new task is starting in this tab: any report queued by a
        // previous task that never reached a terminal event is stale.
        discardReadyReports(ev.tabId);
        // report-coverage:end
        const clearTab =
          ev.tabId !== undefined ? getTab(ev.tabId) : getTab(activeTabId);
        if (clearTab) {
          clearTab.lastTaskFailed = false;
          clearTab.hasRunTask = true;
        }
        if (ev.chat_id && clearTab) {
          clearTab.backendChatId = ev.chat_id;
          if (!clearTab.workDir && configWorkDir) {
            clearTab.workDir = configWorkDir;
          }
          persistTabState();
        }
        const evTabId = ev.tabId;
        if (evTabId === undefined || evTabId === activeTabId) {
          // The new task replaces this chat's transcript: the fan-out
          // panels in it are about to stop existing, so they must hand
          // their sub-agent tabs in first.
          collapseNestedRunParallel(O);
          clearOutput();
          resetOutputState();
          showSpinner();
        } else if (clearTab) {
          collapseNestedRunParallel(clearTab.outputFragment);
          forgetPendingFileLinks(clearTab.id);
          clearTab.outputFragment = null;
          clearTab.streamState = null;
          clearTab.streamLlmPanel = null;
          clearTab.streamLlmPanelState = null;
          clearTab.streamLastToolName = '';
          clearTab.streamPendingPanel = false;
          clearTab.streamStepCount = 0;
        }
        renderTabBar();
        break;
      }
      case 'clearChat': {
        const ccTab = getTab(activeTabId);
        const ccWelcome =
          welcome && welcome.style.display !== 'none' && O.contains(welcome);
        if (ccTab && !ccTab.backendChatId && ccWelcome) {
          focusInputWithRetry();
        } else {
          createNewTab();
        }
        break;
      }
      case 'showWelcome': {
        const swTabId = ev.tabId || activeTabId;
        const swTab = getTab(swTabId);
        if (swTab) {
          if (ev.model) applyModelPick(swTabId, ev.model, 'restore');

          if (swTabId === activeTabId) {
            // Resetting the chat to the welcome screen discards its
            // transcript, fan-out panels and all; their sub-agent tabs
            // must not outlive them.
            collapseNestedRunParallel(O);
            clearOutput();
            resetOutputState();
            showWelcomeScreen();
          } else {
            collapseNestedRunParallel(swTab.outputFragment);
            forgetPendingFileLinks(swTabId);
            swTab.outputFragment = null;
            swTab.welcomeVisible = true;
          }
        }
        break;
      }
      case 'welcome_suggestions':
        renderWelcomeSuggestions(ev.suggestions);
        break;
      case 'remote_url':
        renderRemoteUrl(ev.url, ev.ntfyUrl, ev.tunnelActive);
        break;
      case 'update_available':
        renderUpdateAvailable(
          !!ev.available,
          ev.latest || '',
          ev.current || '',
        );
        break;
      case 'followup_suggestion': {
        // tableak-coverage:start
        if (!isForActiveTab(ev)) break;
        // tableak-coverage:end
        const fu = mkEl('div', 'followup-bar');
        fu.innerHTML =
          '<span class="fu-label">Suggested next</span>' +
          '<span class="fu-text">' +
          esc(ev.text) +
          '</span>';
        fu.addEventListener('click', () => {
          inp.value = ev.text;
          syncClearBtn();
          inp.focus();
        });
        O.appendChild(fu);
        // autoscroll-coverage:start
        autoScrollLatestEventPanel(fu);
        // autoscroll-coverage:end
        break;
      }
      case 'tasks_updated':
        refreshHistory();
        api.getInputHistory();
        break;

      case 'task_events': {
        const teTabId = ev.tabId || activeTabId;
        const teTab = getTab(teTabId);
        if (ev.chat_id && teTab) {
          teTab.backendChatId = ev.chat_id;
          if (!teTab.workDir && configWorkDir) {
            teTab.workDir = configWorkDir;
          }
          persistTabState();
        }
        if (teTab && ev.task_id !== undefined && ev.task_id !== null) {
          teTab.currentTaskId = ev.task_id;
          teTab.pendingTaskId = null;
          const rpPanel = _rpTabPanel.get(teTabId);
          if (rpPanel && teTab.isSubagentTab) {
            rpRegisterSubagent(
              rpPanel,
              rpPanel._rpParentTabId || teTab.parentTabId || '',
              ev.task_id,
              teTabId,
            );
          }
        }
        if (teTabId !== activeTabId) {
          if (!teTab) break;

          const taskTitle = (ev.task || '').trim();
          if (taskTitle) {
            teTab.title =
              taskTitle.length > 30
                ? taskTitle.substring(0, 30) + '\u2026'
                : taskTitle;
            teTab.taskPanelHTML = taskTitle;
            teTab.taskPanelVisible = true;
            renderTabBar();
          }
          if (ev.extra) {
            try {
              const bgExtra = JSON.parse(ev.extra);
              if (bgExtra.work_dir) teTab.workDir = bgExtra.work_dir;
              if (typeof bgExtra.startTs === 'number' && bgExtra.startTs > 0)
                teTab.t0 = bgExtra.startTs;
              if (typeof bgExtra.endTs === 'number' && bgExtra.endTs > 0) {
                teTab.endTs = bgExtra.endTs;
                if (teTab.t0) {
                  teTab.statusTextContent = doneLabelFor(teTab.t0, teTab.endTs);
                  teTab.statusTextColor = 'var(--green)';
                }
              }
            } catch (_e) {}
          }
          const frag = document.createDocumentFragment();
          // Hand the tab its new transcript BEFORE replaying into it:
          // the replay's collapse pass asks this tab which run_parallel
          // panels it owns, and against the outgoing fragment it would
          // disown the replacement panel -- leaving that collapsed
          // panel's sub-agent tabs open with nothing to close them.
          // A replay that fails hands the old transcript back rather
          // than leaving the tab showing half a new one.
          const teOldFrag = teTab.outputFragment;
          teTab.outputFragment = frag;
          // visibletask-coverage:start
          // This transcript belongs to a hidden tab, but it renders
          // through the same renderers as the live stream: the visible
          // tab's status row, step counter and remembered numbers are
          // put back exactly as they were found.
          const teMetrics = currentTaskMetrics;
          const teStepCount = stepCount;
          const teTokens = statusTokens ? statusTokens.textContent : '';
          const teBudget = statusBudget ? statusBudget.textContent : '';
          const teSteps = statusSteps ? statusSteps.textContent : '';
          const teVisibleTab = activeTabId;
          currentTaskMetrics = {tokens: '', budget: '', steps: ''};
          // visibletask-coverage:end
          let bgSteps = 0;
          try {
            bgSteps = replayEventsInto(frag, ev.events || [], {
              ownerTabId: teTabId,
              onFollowupClick: function (text) {
                inp.value = text;
                syncClearBtn();
                inp.focus();
              },
            });
          } catch (e) {
            teTab.outputFragment = teOldFrag;
            throw e;
          } finally {
            // visibletask-coverage:start
            // The replay can close the tab that was on screen — a
            // finished run_parallel panel takes its sub-agent tabs with
            // it — and the tab that takes its place has already put its
            // own numbers up. Only give the borrowed ones back to the tab
            // they were borrowed from.
            if (activeTabId === teVisibleTab) {
              currentTaskMetrics = teMetrics;
              stepCount = teStepCount;
              if (statusTokens) statusTokens.textContent = teTokens;
              if (statusBudget) statusBudget.textContent = teBudget;
              if (statusSteps) statusSteps.textContent = teSteps;
            }
            // visibletask-coverage:end
          }
          teTab.welcomeVisible = false;
          if (bgSteps > 0) teTab.statusStepsText = 'Steps: ' + bgSteps;
          break;
        }
        if (ev.task) {
          currentTaskName = ev.task;
          if (ev.task_id !== undefined && ev.task_id !== null)
            currentTaskId = ev.task_id;
          resetAdjacentState();
          setTaskText(ev.task);
          if (welcome) {
            welcome.style.display = 'none';
            refreshWelcomeLayout();
          }
          updateActiveTabTitle(ev.task);
        } else if (ev.task_id !== undefined && ev.task_id !== null) {
          currentTaskId = ev.task_id;
          if (!currentTaskName) {
            const tetTab = getTab(activeTabId);
            currentTaskName = (tetTab && tetTab.title) || 'Task';
          }
          resetAdjacentState();
        }
        if (ev.extra) {
          try {
            const extra = JSON.parse(ev.extra);
            if (typeof extra.startTs === 'number' && extra.startTs > 0) {
              t0 = extra.startTs;
            }
            if (typeof extra.endTs === 'number' && extra.endTs > 0) {
              endTs = extra.endTs;
            }
            if (extra.work_dir) {
              const wdTab = getTab(activeTabId);
              if (wdTab) wdTab.workDir = extra.work_dir;
            }
          } catch (_e) {}
        }
        replayTaskEvents(ev.events || []);
        break;
      }
      case 'adjacent_task_events':
        // tableak-coverage:start
        // A neighbouring task's transcript replays into the visible #output,
        // so it must belong to the conversation on screen.
        if (isAddressed(ev) && !isForActiveTab(ev)) break;
        renderAdjacentTask(
          ev.direction,
          ev.task,
          ev.events || [],
          ev.task_id,
          ev.tabId || activeTabId,
        );
        // tableak-coverage:end
        break;
      case 'setTaskText': {
        const stt = (ev.text || '').trim();
        // tableak-coverage:start
        // The header names the task the user is looking at. A foreign task
        // may only retitle its OWN tab, never the one on screen.
        if (!isAddressed(ev) || isForActiveTab(ev)) {
          // tableak-coverage:end
          if (stt) {
            currentTaskName = stt;
            currentTaskId = null;
            resetAdjacentState();
            if (welcome) {
              welcome.style.display = 'none';
              refreshWelcomeLayout();
            }
            updateActiveTabTitle(stt);
          }
          setTaskText(ev.text || '');
        } else if (stt) {
          const sttTab = getTab(ev.tabId);
          if (sttTab) {
            sttTab.title =
              stt.length > 30 ? stt.substring(0, 30) + '\u2026' : stt;
            sttTab.taskPanelHTML = stt;
            sttTab.taskPanelVisible = true;
            renderTabBar();
            persistTabState();
          }
        }
        break;
      }
      case 'tabs_state':
        // A snapshot without a well-formed tab list is junk, not an
        // empty registry: ignore it rather than close every tab.
        if (Array.isArray(ev.tabs)) reconcileTabs(ev.tabs);
        break;

      case 'triggerStop':
        markStopping(activeTabId, true);
        api.stop({tabId: activeTabId});
        break;
      case 'appendToInput':
        if (ev.text) {
          inp.value = inp.value ? inp.value + '\n' + ev.text : ev.text;
          inp.dispatchEvent(new Event('input', {bubbles: true}));
        }
        focusInputWithRetry();
        break;
      case 'insertAndSubmit':
        if (ev.text) {
          inp.value = ev.text;
          inp.dispatchEvent(new Event('input', {bubbles: true}));
          focusInputWithRetry();
          sendMessage();
        }
        break;
      case 'focusInput':
        focusInputWithRetry();
        break;

      case 'measureSize':
        try {
          api.sizeReport({
            innerWidth: window.innerWidth || 0,
            screenWidth:
              (window.screen && window.screen.availWidth) ||
              window.innerWidth ||
              0,
          });
        } catch (_e) {}
        break;

      case 'inputHistory':
        histCache = ev.tasks || [];
        if (histIdx < 0) histIdx = -1;
        break;
      case 'ghost':
        // tableak-coverage:start
        if (!isForActiveTab(ev)) break;
        // tableak-coverage:end
        if (ev.suggestion && ev.query === inp.value) {
          updateGhost(ev.suggestion);
        }
        break;
      case 'completions': {
        // tableak-coverage:start
        if (!isForActiveTab(ev)) break;
        // tableak-coverage:end
        if (ev.query !== undefined && ev.query !== inp.value) {
          break;
        }
        renderCompletions(ev.completions || []);
        break;
      }

      case 'commitMessage':
        break;
      case 'droppedPaths':
        if (ev.paths && ev.paths.length > 0) {
          const pos = inp.selectionStart || inp.value.length;
          const before = inp.value.substring(0, pos);
          const after = inp.value.substring(pos);
          const insert = ev.paths
            .map(p => {
              return './' + p;
            })
            .join(' ');
          const needSpace = before.length > 0 && !/\s$/.test(before);
          const trailSpace = after.length > 0 && !/^\s/.test(after) ? ' ' : '';
          inp.value =
            before + (needSpace ? ' ' : '') + insert + trailSpace + after;
          const np =
            before.length +
            (needSpace ? 1 : 0) +
            insert.length +
            trailSpace.length;
          inp.setSelectionRange(np, np);
          syncClearBtn();
          inp.focus();
        }
        break;
      case 'worktree_done':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgWtTab = getTab(ev.tabId);
          if (bgWtTab) {
            bgWtTab.worktreeBarEl = createWorktreeBar(ev.tabId);
          }
          break;
        }
        showWorktreeActions(ev);
        break;
      case 'worktree_result':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgWrTab = getTab(ev.tabId);
          if (bgWrTab) {
            bgWrTab.worktreeBarEl = null;
            clearActionProgress(bgWrTab.outputFragment);
            if (bgWrTab.outputFragment && !isSilentDiscardMessage(ev)) {
              const cls = ev.success ? 'wt-result-ok' : 'wt-result-err';
              const div = mkEl('div', 'ev ' + cls);
              div.textContent = ev.message || '';
              bgWrTab.outputFragment.appendChild(div);
            }
          }
          break;
        }
        handleWorktreeResult(ev);
        break;
      case 'autocommit_done':
        // Terminal for any manual Git Commit in flight (the daemon
        // broadcasts one per request, including refusals), so the
        // settings button re-arms no matter which tab was targeted.
        setAutocommitInFlight(false);
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgAdTab = getTab(ev.tabId);
          if (bgAdTab) {
            clearActionProgress(bgAdTab.outputFragment);
            // A successful manual Git Commit is reported by a toast
            // notification instead; only failures earn transcript text.
            if (bgAdTab.outputFragment && !(ev && ev.manual && ev.success)) {
              const cls = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
              const div = mkEl('div', 'ev ' + cls);
              div.textContent = (ev && ev.message) || '';
              bgAdTab.outputFragment.appendChild(div);
            }
          }
          break;
        }
        handleAutocommitResult(ev);
        break;
      case 'task_done': {
        let doneT0 = t0;
        if (!doneT0 && ev.tabId !== undefined) {
          const rt = getTab(ev.tabId);
          if (rt) doneT0 = rt.t0;
        }
        const ms =
          ev.startTs > 0 && ev.endTs > 0
            ? ev.endTs - ev.startTs
            : Date.now() - (doneT0 || Date.now());
        const el = Math.max(0, Math.floor(ms / 1000));
        const em = Math.floor(el / 60);
        markTabDone(ev.tabId, ev.success === false);
        clearActionProgressForTab(ev.tabId);
        setReady(
          'Done (' + (em > 0 ? em + 'm ' : '') + (el % 60) + 's)',
          ev.tabId,
          ev.startTs,
          ev.endTs,
        );
        focusFinishedTab(ev.tabId);
        // report-coverage:start
        openReadyReportTabs(ev.tabId);
        // report-coverage:end
        break;
      }
      case 'task_error':
      case 'task_interrupted':
      case 'task_stopped': {
        markTabDone(ev.tabId, true);
        clearActionProgressForTab(ev.tabId);
        if (ev.tabId === undefined || ev.tabId === activeTabId) {
          if (llmPanel && llmPanel._provisional)
            discardProvisionalPanel(llmPanel);
          else if (llmPanel) finalizePanelTime(llmPanel);
          llmPanel = null;
          pendingPanel = true;
        } else {
          const endTab = getTab(ev.tabId);
          if (endTab && endTab.streamLlmPanel) {
            if (endTab.streamLlmPanel._provisional)
              discardProvisionalPanel(endTab.streamLlmPanel);
            else finalizePanelTime(endTab.streamLlmPanel);
            endTab.streamLlmPanel = null;
            endTab.streamPendingPanel = true;
          }
        }
        const label =
          t === 'task_error'
            ? 'Error'
            : t === 'task_interrupted'
              ? 'Interrupted'
              : 'Stopped';
        setReady(label, ev.tabId, ev.startTs, ev.endTs);
        focusFinishedTab(ev.tabId);
        // report-coverage:start
        // The task finished (with an error / stop): a successfully
        // written report is still a real artifact — open it.
        openReadyReportTabs(ev.tabId);
        // report-coverage:end
        break;
      }
      case 'new_tab': {
        if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
          break;
        if (ev.task_id === undefined || ev.task_id === null) break;
        const parentTabBeforeNew = ev.parent_tab_id || '';
        // One sub-agent, one tab: a re-delivered spawn for a sub-agent
        // that already has a tab must not open a second one.
        const spawned = openSubagentTabForTask(ev.task_id, '');
        if (spawned) {
          const spawnPanel = _rpTabPanel.get(spawned.id) || null;
          if (spawnPanel) {
            rpRegisterSubagent(
              spawnPanel,
              spawnPanel._rpParentTabId || parentTabBeforeNew,
              ev.task_id,
              spawned.id,
            );
          }
          break;
        }
        let subAgentTabId;
        if (parentTabBeforeNew) {
          // Attribute the spawn to the fan-out that owns it, not to the
          // newest panel: with several run_parallel calls in one task a
          // late spawn belongs to an earlier call, whose collapsed state
          // decides whether it may have a tab.
          const rpPanel = rpPanelForNewSubagent(parentTabBeforeNew, ev.task_id);
          if (
            rpPanel &&
            (rpPanel.classList.contains('collapsed') ||
              rpSubagentHandClosed(rpPanel, ev.task_id))
          ) {
            rpRegisterSubagent(rpPanel, parentTabBeforeNew, ev.task_id, '');
            break;
          }
          const subTab = createBackgroundSubagentTab(parentTabBeforeNew);
          subTab.currentTaskId = ev.task_id;
          subAgentTabId = subTab.id;
          if (rpPanel) {
            rpRegisterSubagent(
              rpPanel,
              parentTabBeforeNew,
              ev.task_id,
              subTab.id,
            );
          }
        } else {
          subAgentTabId = createBackgroundSubagentTab('').id;
        }
        api.resumeSession({taskId: ev.task_id, tabId: subAgentTabId});
        break;
      }
      case 'openSubagentTab': {
        if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
          break;
        if (!ev.parent_tab_id && !getTab(ev.tab_id)) break;
        const subDesc = (ev.description || 'Sub-agent').trim();
        const subIdx =
          typeof ev.taskIndex === 'number' ? ev.taskIndex + 1 : null;
        const titlePrefix = subIdx !== null ? subIdx + '. ' : '';
        const title = titlePrefix + subDesc.substring(0, 40);
        const parentId = ev.parent_tab_id || ev.tabId || '';
        const subTaskId =
          ev.task_id === undefined || ev.task_id === null ? '' : ev.task_id;
        let rpPanel = _rpTabPanel.get(ev.tab_id) || null;
        if (!rpPanel && parentId) {
          rpPanel = rpPanelForNewSubagent(parentId, subTaskId);
        }
        let subTab = getTab(ev.tab_id);
        // A sub-agent the user closed by hand stays closed until its
        // run_parallel panel is collapsed and expanded again -- also
        // when the daemon re-announces it under a different tab id.
        if (!subTab && rpSubagentHandClosed(rpPanel, subTaskId)) {
          _rpClosedSubagentTabs.add(ev.tab_id);
          break;
        }
        if (!subTab && _rpClosedSubagentTabs.has(ev.tab_id)) {
          if (rpPanel) rpRegisterSubagent(rpPanel, parentId, subTaskId, '');
          break;
        }
        // One sub-agent, one tab: the daemon addresses a sub-agent by
        // different tab ids across replays, so an announcement for a
        // sub-agent that already has a tab renames that tab instead of
        // opening another one for the same conversation.
        if (!subTab) {
          const openForTask = openSubagentTabForTask(subTaskId, ev.tab_id);
          if (openForTask) {
            retagSubagentTab(openForTask, ev.tab_id);
            subTab = openForTask;
          }
        }
        if (rpPanel && rpPanel.classList.contains('collapsed')) {
          rpRegisterSubagent(
            rpPanel,
            parentId,
            subTaskId,
            subTab ? subTab.id : '',
          );
          if (subTab) syncRunParallelPanel(rpPanel);
          else _rpClosedSubagentTabs.add(ev.tab_id);
          break;
        }
        const needsPlacement = !subTab || !subTab.isSubagentTab;
        if (!subTab) {
          subTab = makeTab(title);
          subTab.id = ev.tab_id;
        } else {
          subTab.title = title;
        }
        if (needsPlacement) {
          placeSubagentTabAfterParent(subTab, parentId);
        }
        subTab.isSubagentTab = true;
        // Stamp the sub-agent's task on its tab right away: that task
        // id is this tab's identity for every later announcement (see
        // openSubagentTabForTask).
        if (subTaskId !== '') {
          subTab.currentTaskId = subTaskId;
          subTab.pendingTaskId = null;
        }
        if (parentId && parentId !== subTab.id) {
          subTab.parentTabId = parentId;
        }
        const subDone = !!ev.isDone;
        subTab.isDone = subDone;
        setTabRunning(subTab, !subDone);
        subTab.taskPanelHTML = subDesc;
        subTab.taskPanelVisible = true;
        if (rpPanel)
          rpRegisterSubagent(rpPanel, parentId, subTaskId, subTab.id);
        renderTabBar();
        if (subTab.id === activeTabId) {
          if (inputContainer) {
            if (subTab.isRunning) inputContainer.style.display = '';
            else inputContainer.style.display = 'none';
          }
          setRunningState(subTab.isRunning);
          if (subTab.isRunning) applyChevronState(currentTaskName);
        }
        persistTabState();
        break;
      }
      case 'closeSubagentTab': {
        // Another client closed this sub-agent tab; mirror the close.
        // Applied without echoing `closeTab` back to the daemon (the
        // origin client already sent it) — see closeTab(fromServer).
        if (getTab(ev.tab_id)) closeTab(ev.tab_id, true, true);
        break;
      }
      case 'openTabRejected': {
        // The daemon refused to register this tab (registry cap): it
        // will never appear in a snapshot, so drop the local copy —
        // unless it is the only tab, which stays as the same local,
        // unregistered placeholder an empty registry gets. No
        // re-registration happens either way, so a full registry
        // cannot start an openTab/reject loop.
        pendingOpenTabs.delete(ev.tabId);
        const rejTab = getTab(ev.tabId);
        if (rejTab && !rejTab.isSubagentTab && !rejTab.isContentTab) {
          const chatTabs = tabs.filter(t => !t.isContentTab);
          if (chatTabs.length > 1) closeTab(ev.tabId, true, true);
        }
        if (ev.text) {
          showNotification({
            id: 'open-tab-rejected',
            severity: 'warning',
            message: ev.text,
          });
        }
        break;
      }
      case 'subagentDone': {
        const doneTab = getTab(ev.tab_id);
        if (doneTab) {
          // report-coverage:start
          // A sub-agent finishing is not the task finishing: the parent
          // task keeps running, so its report opens in the background.
          openReadyReportTabs(doneTab.id, false);
          // report-coverage:end
          doneTab.isDone = true;
          setTabRunning(doneTab, false);
          if (doneTab.id === activeTabId) {
            setRunningState(false);
            if (inputContainer) inputContainer.style.display = 'none';
          }
          closeTab(doneTab.id, true);
        }
        break;
      }
      default:
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgTab = findTabByEvt(ev);
          if (bgTab) processOutputEventForBgTab(ev, bgTab);
          if (!isForActiveTab(ev)) break;
        }
        // tableak-coverage:start
        // Streaming transcript events are always task-scoped, so one that
        // names no tab cannot be attributed and must not be shown.
        if (TASK_SCOPED_STREAM_TYPES.has(t) && !isForActiveTab(ev)) break;
        // tableak-coverage:end
        if (
          ev.taskId !== undefined &&
          ev.taskId !== null &&
          ev.taskId !== '' &&
          !PRE_ADOPTION_TYPES.has(ev.type) &&
          // tableak-coverage:start
          // A tab learns which task it is running from a reply that names it
          // (tabId) or from the request it sent itself (pendingTaskId). A bare
          // task id off the wire is not evidence of anything, and adopting one
          // binds this tab to a task it never started -- permanently, since
          // every later event for the real owner is then rejected.
          mayAdoptTaskId(ev)
          // tableak-coverage:end
        ) {
          const adoptTab = getTab(activeTabId);
          if (
            adoptTab &&
            String(adoptTab.currentTaskId) !== String(ev.taskId)
          ) {
            adoptTab.currentTaskId = ev.taskId;
            adoptTab.pendingTaskId = null;
            currentTaskId = ev.taskId;
            if (
              (oldestLoadedTaskId === null || oldestLoadedTaskId === '') &&
              (newestLoadedTaskId === null || newestLoadedTaskId === '')
            ) {
              oldestLoadedTaskId = ev.taskId;
              newestLoadedTaskId = ev.taskId;
            }
          }
        }
        if (ev.taskId && (ev.type === 'result' || ev.type === 'usage_info')) {
          const activeTab = getTab(activeTabId);
          if (
            activeTab &&
            activeTab.currentTaskId !== undefined &&
            activeTab.currentTaskId !== null &&
            String(activeTab.currentTaskId) !== String(ev.taskId)
          ) {
            console.warn(
              'Dropping mis-routed',
              ev.type,
              'event for task',
              ev.taskId,
              'in active tab whose currentTaskId is',
              activeTab.currentTaskId,
            );
            break;
          }
        }
        processOutputEvent(ev);
        if (isActiveTabRunning()) showSpinner();
        break;
    }
  }

  function updateInputDisabled() {
    inp.disabled = false;
    // A photo still being converted must not be raced by a send: block the
    // button until every attachment slot holds real bytes.
    sendBtn.disabled = hasPendingAttachments();
  }

  /**
   * Repaint the Stop button from the active tab's pending-stop state.
   *
   * A stop the agent has not acted on yet (it is inside a long model
   * request) is indistinguishable from a stop that never arrived unless
   * the button says so — which is why the button looked dead for three
   * minutes in the post-mortem `stop_button_delay_2026-08-05.html`.
   */
  function renderStopButton() {
    const tab = getTab(activeTabId);
    const stopping = !!(tab && tab.isStopping) && isRunning;
    stopBtn.classList.toggle('stopping', stopping);
    stopBtn.setAttribute(
      'data-tooltip',
      stopping ? 'Stopping — waiting for the agent' : 'Stop agent',
    );
  }

  /**
   * Record whether `tabId` has a stop in flight and repaint the button.
   *
   * @param {string} tabId Tab whose Stop button was pressed.
   * @param {boolean} stopping True while the stop is pending.
   */
  function markStopping(tabId, stopping) {
    const tab = getTab(tabId);
    if (tab) tab.isStopping = stopping;
    if (tabId === activeTabId) renderStopButton();
  }

  /**
   * Set a tab's running state, dropping any pending stop along with it.
   *
   * `isStopping` means "a stop is in flight for the task THIS tab is
   * running", so it must not outlive that task: a background tab that
   * finished while stopping would otherwise open its next task with the
   * Stop button already pulsing.
   *
   * @param {object} tab The tab to update.
   * @param {boolean} running Whether that tab is now running a task.
   */
  function setTabRunning(tab, running) {
    tab.isRunning = running;
    if (!running) tab.isStopping = false;
    if (tab.id === activeTabId) renderStopButton();
  }

  function setRunningState(running) {
    isRunning = running;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';
    // A tab that is not running has nothing left to stop, so the
    // pending state never survives the task it belonged to.
    if (!running) {
      const activeTab = getTab(activeTabId);
      if (activeTab) activeTab.isStopping = false;
    }
    renderStopButton();

    updateInputDisabled();
    if (running) {
      startTimer();
      showSpinner();
    } else {
      stopTimer();
      removeSpinner();
      if (statusText.textContent.startsWith('Running')) {
        statusText.textContent =
          t0 && endTs > 0 ? doneLabelFor(t0, endTs) : 'Done';
      }
    }
  }

  function markTabDone(tabId, failed) {
    const tid = tabId !== undefined ? tabId : activeTabId;
    const tab = getTab(tid);
    if (tab) {
      tab.hasRunTask = true;
      tab.lastTaskFailed = !!failed;
      // A question can only be answered while its task is alive. The server
      // sends `askUserDone` only for an accepted answer, so a task that ends
      // with a question outstanding must retire it here. Only THIS task
      // ended, so tabs sharing its backend chat keep their own questions.
      if (tab.askPendingQuestion !== null) clearAskForTab(tab);
    }
  }

  function focusFinishedTab(tabId) {
    if (tabId === undefined || tabId === null) return;
    if (tabId === activeTabId) return;
    if (!getTab(tabId)) return;
    switchToTab(tabId);
  }

  function setReady(label, tabId, doneStartTs, doneEndTs) {
    const hasStart = typeof doneStartTs === 'number' && doneStartTs > 0;
    const hasEnd = typeof doneEndTs === 'number' && doneEndTs > 0;
    let doneTab = null;
    if (tabId !== undefined) {
      doneTab = getTab(tabId);
      if (doneTab) {
        setTabRunning(doneTab, false);
        if (hasStart) doneTab.t0 = doneStartTs;
        doneTab.endTs = hasEnd ? doneEndTs : Date.now();
        doneTab.statusTextContent = label || 'Ready';
        doneTab.statusTextColor = 'var(--green)';
      }
    }
    if (tabId === undefined || tabId === activeTabId) {
      if (hasStart) t0 = doneStartTs;
      endTs = hasEnd ? doneEndTs : Date.now();
      setRunningState(false);
      stopTimer();
      removeSpinner();
      statusText.textContent = label || 'Ready';
      inp.focus();
    }
    renderTabBar();
  }

  function addBanner(cls, label, text) {
    const div = mkEl('div', 'ev tr ' + cls);
    div.innerHTML = '<strong>' + label + '</strong> ' + esc(text);
    O.appendChild(div);
    // autoscroll-coverage:start
    autoScrollLatestEventPanel(div);
    // autoscroll-coverage:end
  }

  function addError(text) {
    addBanner('err', 'Error:', text);
  }

  function addNotice(text) {
    addBanner('note', 'Note:', text);
  }

  function addWarning(text) {
    addBanner('warn', 'Warning:', text);
  }

  function _buildRemoteUrlBar(displayUrl, isNtfy) {
    const wrapper = document.createElement('div');
    wrapper.className = 'remote-url-bar';
    const label = document.createElement('div');
    label.className = 'remote-url-label';
    label.textContent = isNtfy
      ? 'Webapp: click the link in the first post at URL:'
      : 'Web/mobile app';
    const row = document.createElement('div');
    row.className = 'remote-url-row';
    const link = document.createElement('a');
    link.href = displayUrl;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = displayUrl;
    link.className = 'remote-url-link';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'remote-url-copy';
    copyBtn.title = 'Copy URL';
    const copySvg =
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
    const checkSvg =
      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>';
    copyBtn.innerHTML = copySvg;
    copyBtn.addEventListener('click', e => {
      e.preventDefault();
      navigator.clipboard.writeText(displayUrl).then(() => {
        copyBtn.innerHTML = checkSvg;
        setTimeout(() => {
          copyBtn.innerHTML = copySvg;
        }, 1500);
      });
    });
    row.appendChild(link);
    row.appendChild(copyBtn);
    wrapper.appendChild(label);
    wrapper.appendChild(row);
    return wrapper;
  }

  function renderRemoteUrl(url, ntfyUrl, tunnelActive) {
    const displayUrl = ntfyUrl || url;
    const containerIds = ['remote-url', 'welcome-remote-url'];
    for (const id of containerIds) {
      const container = document.getElementById(id);
      if (!container) continue;
      container.innerHTML = '';
      if (!displayUrl) continue;
      container.appendChild(_buildRemoteUrlBar(displayUrl, !!ntfyUrl));
    }
    const welcomeCfg = document.getElementById('welcome-config');
    if (welcomeCfg) {
      const isRemoteChat = document.body.classList.contains('remote-chat');
      const visible = isRemoteChat
        ? false
        : tunnelActive === undefined
          ? !!displayUrl
          : !!tunnelActive;
      welcomeCfg.style.display = visible ? '' : 'none';
    }
  }

  const UPDATE_NOTIFICATION_ID = 'kiss-update-available';

  const UPDATE_DOWNLOAD_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" ' +
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
    '<polyline points="7 10 12 15 17 10"/>' +
    '<line x1="12" y1="15" x2="12" y2="3"/>' +
    '</svg>';

  const UPDATE_BADGE_SVG =
    '<svg class="update-available-icon" width="12" height="12" ' +
    'viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">' +
    '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>' +
    '<polyline points="7 10 12 15 17 10"/>' +
    '<line x1="12" y1="15" x2="12" y2="3"/>' +
    '</svg>';

  function renderUpdateAvailable(available, latest, current) {
    renderUpdateAvailableBadge(available, latest, current);
    renderUpdateAvailableNotification(available, latest, current);
  }

  function renderUpdateAvailableBadge(available, latest, current) {
    const btn = document.getElementById('cfg-update-btn');
    if (!btn) return;
    const prior = btn.querySelector('.update-available-icon');
    if (prior) prior.remove();
    if (!available) {
      btn.classList.remove('has-update');
      btn.removeAttribute('title');
      return;
    }
    btn.classList.add('has-update');
    const tip =
      latest && current
        ? `New version ${latest} available (installed ${current}) ` +
          '— click to update'
        : 'A new version is available — click to update';
    btn.setAttribute('title', tip);
    btn.insertAdjacentHTML('afterbegin', UPDATE_BADGE_SVG);
  }

  function renderUpdateAvailableNotification(available, latest, current) {
    if (!available) {
      removeNotification(UPDATE_NOTIFICATION_ID, undefined, false);
      return;
    }
    const message =
      latest && current
        ? `KISS Sorcar ${latest} is available (you have ${current}).`
        : 'A new KISS Sorcar release is available.';
    showNotification({
      id: UPDATE_NOTIFICATION_ID,
      severity: 'info',
      message,
      sticky: true,
      actions: [
        {
          label: 'Update',
          ariaLabel: latest
            ? `Update KISS Sorcar to ${latest}`
            : 'Update KISS Sorcar',
          svg: UPDATE_DOWNLOAD_SVG,
          onClick: () => {
            api.runUpdate();
          },
        },
      ],
    });
  }

  function renderWelcomeSuggestions(suggestions) {
    const container = document.getElementById('suggestions');
    if (!container) return;
    container.innerHTML = '';
    if (!suggestions || suggestions.length === 0) return;
    suggestions.forEach(s => {
      const chip = document.createElement('div');
      chip.className = 'suggestion-chip';
      chip.dataset.prompt = s.text;
      chip.dataset.tooltip = s.text;
      chip.innerHTML =
        '<span class="chip-label">Suggested prompt</span>' +
        '<span class="chip-text">' +
        esc(s.text) +
        '</span>';
      chip.addEventListener('click', () => {
        inp.value = s.text;
        syncClearBtn();
        inp.focus();
      });
      container.appendChild(chip);
    });
  }

  /**
   * Render *events* into *container*, holding back the sub-agent tab
   * closes the replay's collapse passes ask for until the whole
   * transcript exists (see rpCloseSubagentTab).
   *
   * A nested replay leaves the outer replay's queue in charge, and a
   * replay that throws drops its queue instead of applying closes to a
   * half-written transcript -- either way no close is left queued for
   * ever, which would silently strand sub-agent tabs of every later
   * collapse.
   *
   * @param {Element|DocumentFragment} container Where to render.
   * @param {Array<object>} events The transcript to replay.
   * @param {object} [opts] ownerTabId / onFollowupClick.
   * @returns {number} The number of steps the transcript records.
   */
  function replayEventsInto(container, events, opts) {
    if (_rpDeferredCloses !== null) {
      return renderReplayedEvents(container, events, opts);
    }
    _rpDeferredCloses = [];
    let steps;
    try {
      steps = renderReplayedEvents(container, events, opts);
    } catch (e) {
      _rpDeferredCloses = null;
      throw e;
    }
    rpFlushDeferredCloses();
    return steps;
  }

  function renderReplayedEvents(container, events, opts) {
    const ownerTabId = opts ? opts.ownerTabId : undefined;
    const rWorkDir =
      ownerTabId !== undefined ? workDirForTab(ownerTabId) || '' : undefined;
    const ctx = mkStreamCtx(container, ownerTabId);
    // report-coverage:start
    ctx.state.suppressReportOpen = true;
    // report-coverage:end
    const prevDefer = _deferHighlight;
    _deferHighlight = true;
    try {
      events.forEach(ev => {
        normalizeEventTs(ev);
        const t = ev.type;
        if (
          t === 'task_done' ||
          t === 'task_error' ||
          t === 'task_stopped' ||
          t === 'task_interrupted'
        ) {
          return;
        }
        if (t === 'followup_suggestion') {
          const fu = mkEl('div', 'followup-bar');
          fu.innerHTML =
            '<span class="fu-label">Suggested next</span>' +
            '<span class="fu-text">' +
            esc(ev.text) +
            '</span>';
          if (opts && opts.onFollowupClick) {
            fu.addEventListener('click', () => {
              opts.onFollowupClick(ev.text);
            });
          }
          container.appendChild(fu);
          return;
        }
        const where = streamBegin(ctx, ev);
        handleOutputEvent(ev, where.target, where.state, rWorkDir, ownerTabId);
        streamEnd(ctx, ev, where.target);
      });
    } finally {
      _deferHighlight = prevDefer;
    }
    collapseAllExceptResult(container, ownerTabId);
    if (typeof hljs !== 'undefined') {
      container.querySelectorAll('code.needs-hl').forEach(bl => {
        if (!bl.closest('.collapsible.collapsed')) {
          bl.classList.remove('needs-hl');
          highlightBlockPreservingLinks(bl);
        }
      });
    }
    return ctx.stepCount;
  }

  function replayTaskEvents(events) {
    clearOutput();
    resetOutputState();
    clearUsageMetrics();
    const rSteps = replayEventsInto(O, events, {
      ownerTabId: activeTabId,
      onFollowupClick: function (text) {
        inp.value = text;
        syncClearBtn();
        inp.focus();
      },
    });
    if (rSteps > 0) updateStepCount(rSteps);
    // autoscroll-coverage:start
    // clearOutput() above released any user scroll lock: the replayed
    // chat lands at the end of its latest event panel.
    autoScrollLatestEventPanel(O.lastElementChild);
    // autoscroll-coverage:end
    currentTaskMetrics.tokens = statusTokens ? statusTokens.textContent : '';
    currentTaskMetrics.budget = statusBudget ? statusBudget.textContent : '';
    currentTaskMetrics.steps = statusSteps ? statusSteps.textContent : '';
    applyChevronState(currentTaskName);
  }

  function createActionBar(labelText, buttons) {
    const bar = mkEl('div', 'wt-bar');
    const label = mkEl('span', 'wt-label');
    label.textContent = labelText;
    bar.appendChild(label);

    const btns = mkEl('div', 'wt-btns');
    buttons.forEach(b => {
      const btn = mkEl('button', 'wt-btn ' + b.cls);
      btn.textContent = b.text;
      btn.addEventListener('click', () => {
        disableActionBarBtns(bar);
        api.send(b.msg());
      });
      btns.appendChild(btn);
    });
    bar.appendChild(btns);
    return bar;
  }

  function disableActionBarBtns(bar) {
    if (!bar) return;
    bar.querySelectorAll('.wt-btn').forEach(b => {
      b.disabled = true;
    });
  }

  function detachActionBar(bar) {
    if (bar && bar.parentNode) bar.parentNode.removeChild(bar);
    if (inputContainer) inputContainer.style.display = '';
  }

  function attachActionBar(bar) {
    if (inputContainer) inputContainer.style.display = 'none';
    const area = document.getElementById('input-area');
    area.insertBefore(bar, area.firstChild);
  }

  function appendActionResult(ev) {
    const cls = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
    const div = mkEl('div', 'ev ' + cls);
    div.textContent = (ev && ev.message) || '';
    O.appendChild(div);
    // autoscroll-coverage:start
    autoScrollLatestEventPanel(div);
    // autoscroll-coverage:end
  }

  /**
   * Show the daemon's live progress line for a commit or merge flow.
   *
   * `autocommit_progress` and `worktree_progress` are the steps of ONE
   * operation ("Staging changes…" → "Generating commit message…" →
   * "Committing…"), so they share a single line that is replaced in
   * place and removed again by the terminal event. Rendering them here
   * is what gives the remote web client the feedback the VS Code host
   * gets from its native progress toast.
   *
   * @param {Element|DocumentFragment} target Transcript to render into.
   * @param {string} message The user-facing progress text.
   */
  function renderActionProgress(target, message) {
    if (!target) return;
    let el = target.querySelector('.ev.wt-progress');
    if (!el) {
      el = mkEl('div', 'ev wt-progress');
      target.appendChild(el);
    }
    el.textContent = message || '';
  }

  /**
   * Remove the live progress line, if any, from *target*.
   *
   * @param {Element|DocumentFragment} target Transcript to clean up.
   */
  function clearActionProgress(target) {
    const el = target ? target.querySelector('.ev.wt-progress') : null;
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  /**
   * Drop the live progress line owned by *tabId*, wherever it lives.
   *
   * `autocommit_done` and `worktree_result` are the intended way for
   * the line to go, but neither is guaranteed: the daemon runs both
   * flows on its post-task path inside a swallow-all handler
   * (`task_runner._run_task_inner`), and the progress events are
   * broadcast from inside the call it wraps. Anything raised after the
   * first message -- a git binary that dies, an LLM call that throws
   * while writing the commit message, a stopped task unwinding through
   * the merge -- is logged and dropped, and the terminal event is never
   * sent. The line then reads as if the operation were still running,
   * forever.
   *
   * The task-end event that follows immediately is the last thing the
   * flow is bracketed by, so it is where the line is taken down. In the
   * normal case the terminal event has already removed it and this is a
   * no-op.
   *
   * @param {string|undefined} tabId Tab whose task ended. `undefined`
   *   means the visible tab, as everywhere else in the dispatcher.
   */
  function clearActionProgressForTab(tabId) {
    if (tabId === undefined || tabId === activeTabId) {
      clearActionProgress(O);
      return;
    }
    const tab = getTab(tabId);
    if (tab) clearActionProgress(tab.outputFragment);
  }

  let worktreeBar = null;

  function clearWorktreeBar() {
    detachActionBar(worktreeBar);
    worktreeBar = null;
  }

  function createWorktreeBar(ownerTabId) {
    return createActionBar('Auto-commit and merge or Discard?', [
      {
        cls: 'wt-merge',
        text: 'Auto-commit and merge',
        msg: () => ({
          type: 'worktreeAction',
          action: 'merge',
          tabId: ownerTabId,
        }),
      },
      {
        cls: 'wt-discard',
        text: 'Discard',
        msg: () => ({
          type: 'worktreeAction',
          action: 'discard',
          tabId: ownerTabId,
        }),
      },
    ]);
  }

  function showWorktreeActions(ev) {
    clearWorktreeBar();
    worktreeBar = createWorktreeBar((ev && ev.tabId) || activeTabId);
    attachActionBar(worktreeBar);
  }

  function isSilentDiscardMessage(ev) {
    if (!ev || !ev.success) return false;
    const msg = ev.message || '';
    return /^Discarded branch '[^']+'\.$/.test(msg);
  }

  function handleWorktreeResult(ev) {
    clearWorktreeBar();
    clearActionProgress(O);
    if (isSilentDiscardMessage(ev)) {
      return;
    }
    appendActionResult(ev);
  }

  function handleAutocommitResult(ev) {
    clearActionProgress(O);
    // A successful manual Git Commit is reported by a toast
    // notification instead; only failures earn transcript text.
    if (!(ev && ev.manual && ev.success)) appendActionResult(ev);
    focusInputWithRetry();
  }

  // The settings panel's Git Commit button is disabled while its
  // manual autocommit is in flight (the daemon silently drops
  // duplicate requests, so a still-enabled button would look dead).
  // The timer is a failsafe: if the daemon dies mid-commit and the
  // terminal autocommit_done never arrives, the button re-arms on its
  // own instead of staying wedged forever.
  let autocommitRearmTimer = null;

  function setAutocommitInFlight(pending) {
    if (autocommitRearmTimer) {
      clearTimeout(autocommitRearmTimer);
      autocommitRearmTimer = null;
    }
    if (autocommitBtn) autocommitBtn.disabled = pending;
    if (pending) {
      autocommitRearmTimer = setTimeout(() => {
        setAutocommitInFlight(false);
      }, 120000);
    }
  }

  // Content tabs (opened HTML files, subagent viewers…) have no
  // transcript of their own — a result addressed to one would render
  // into the hidden shared output and be destroyed on the next tab
  // switch.  Commit on behalf of the chat tab the host currently
  // considers active instead.
  function autocommitTargetTabId() {
    const active = getTab(activeTabId);
    if (active && !active.isContentTab) return activeTabId;
    return reportedChatTabId || activeTabId;
  }

  // The `ready` announcement: hands the daemon this client's legacy
  // locally-persisted tabs exactly once (adopted only into an empty
  // registry) — or, on a re-`ready` after a daemon restart, the tabs
  // currently on screen, so a daemon whose registry file was wiped
  // re-adopts them. The daemon answers with the canonical `tabs_state`
  // snapshot and replays every chat-bound tab.
  function collectRestoredTabs() {
    const current = tabs
      .filter(t => {
        return !t.isSubagentTab && !t.isContentTab && t.backendChatId;
      })
      .map(t => {
        return {
          tabId: t.id,
          chatId: t.backendChatId,
          title: t.title || '',
          workDir: t.workDir || '',
        };
      });
    return current.length > 0 ? current : legacyRestoredTabs;
  }

  function sendReady() {
    // `ready` seeds the host's active chat tab exactly like an
    // `activeTabChanged` would, so the local mirror has to start out
    // agreeing with it — otherwise the first real change looks like a
    // no-op and is never sent.
    api.ready({tabId: activeTabId, restoredTabs: collectRestoredTabs()});
    reportedChatTabId = activeTabId;
  }

  function init() {
    setupEventListeners();
    renderTabBar();
    sendReady();
    api.getConfig();
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    // The first gesture ends the launch: from here on the user drives which
    // tab is on screen. Capture phase, because a handler further down may
    // stop the event from bubbling back up.
    document.addEventListener('pointerdown', closeLaunchSwitch, true);
    document.addEventListener('keydown', closeLaunchSwitch, true);
    window.addEventListener('focus', () => {
      api.webviewFocusChanged({focused: true});
    });
    window.addEventListener('blur', () => {
      api.webviewFocusChanged({focused: false});
    });
    document.addEventListener('keydown', e => {
      if (
        (e.metaKey || e.ctrlKey) &&
        e.key === 'd' &&
        !e.shiftKey &&
        !e.altKey
      ) {
        e.preventDefault();
        api.focusEditor();
      }
      if (e.key === 'Escape' && sidebar.classList.contains('open')) {
        e.preventDefault();
        closeSidebar();
      }
    });
    inp.addEventListener('keydown', e => {
      if (autocomplete.style.display === 'block') {
        const items = autocomplete.querySelectorAll('.ac-item');
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          acIdx = Math.min(acIdx + 1, items.length - 1);
          updateSel(items, acIdx);
          return;
        }
        if (e.key === 'ArrowUp') {
          e.preventDefault();
          acIdx = Math.max(acIdx - 1, -1);
          updateSel(items, acIdx);
          return;
        }
        if (e.key === 'Tab') {
          e.preventDefault();
          const ti = acIdx >= 0 ? acIdx : 0;
          if (items[ti]) items[ti].click();
          return;
        }
        if (e.key === 'Enter') {
          const atCtx = getAtCtx();
          if (atCtx && acIdx >= 0) {
            e.preventDefault();
            items[acIdx].click();
            return;
          }
          hideAC();
        }
        if (e.key === 'Escape') {
          hideAC();
          return;
        }
      }
      if (e.key === 'Tab' && currentGhost) {
        e.preventDefault();
        acceptGhost();
        return;
      }
      if (e.key === 'ArrowUp' && autocomplete.style.display !== 'block') {
        if (cycleHistoryUp()) {
          e.preventDefault();
          return;
        }
      }
      if (e.key === 'ArrowDown' && histIdx >= 0) {
        e.preventDefault();
        cycleHistoryDown();
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
        return;
      }
      if (e.key !== 'Tab') clearGhost();
    });
    let _shiftHeld = false;
    document.addEventListener('keydown', e => {
      if (e.key === 'Shift') _shiftHeld = true;
    });
    document.addEventListener('keyup', e => {
      if (e.key === 'Shift') _shiftHeld = false;
    });
    inp.addEventListener('beforeinput', e => {
      if (e.inputType === 'insertLineBreak' && !_shiftHeld) {
        e.preventDefault();
        sendMessage();
      }
    });
    inp.addEventListener('input', () => {
      inp.style.height = 'auto';
      inp.style.height = inp.scrollHeight + 'px';
      checkAutocomplete();
      requestGhost();
      histIdx = -1;
      syncClearBtn();
    });
    inp.addEventListener('blur', () => {
      clearGhost();
      hideAC();
    });
    inp.addEventListener('touchstart', handleInputTouchStart, {passive: true});
    inp.addEventListener('touchend', handleInputTouchEnd);
    autocomplete.addEventListener('mousedown', e => {
      e.preventDefault();
    });
    stopBtn.addEventListener('click', () => {
      markStopping(activeTabId, true);
      api.stop({tabId: activeTabId});
    });
    uploadBtn.addEventListener('click', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      // The explicit HEIC/HEIF extensions matter for the iOS "Browse" path,
      // which filters by extension rather than by MIME type.  Whichever way
      // the photo arrives, prepareAttachment() converts it to JPEG.
      input.accept = 'image/*,.heic,.heif,application/pdf';
      input.onchange = handleFileSelect;
      input.click();
    });
    setupPasswordToggle('cfg-remote-password-toggle', 'cfg-remote-password');
    setupPasswordToggle(
      'welcome-cfg-remote-password-toggle',
      'welcome-cfg-remote-password',
    );
    [
      'cfg-key-GEMINI_API_KEY',
      'cfg-key-OPENAI_API_KEY',
      'cfg-key-ANTHROPIC_API_KEY',
      'cfg-key-TOGETHER_API_KEY',
      'cfg-key-OPENROUTER_API_KEY',
      'cfg-key-ZAI_API_KEY',
      'cfg-key-MOONSHOT_API_KEY',
      'cfg-custom-api-key',
    ].forEach(setupSecretInput);
    const welcomePwInp = document.getElementById('welcome-cfg-remote-password');
    const settingsPwInp = document.getElementById('cfg-remote-password');
    function _flushPw() {
      saveSettingsIfPopulated();
    }
    if (welcomePwInp && settingsPwInp) {
      welcomePwInp.addEventListener('input', () => {
        settingsPwInp.value = welcomePwInp.value;
        // Assigning .value fires no input event, so the mirrored field
        // has to be marked by hand or the very first password a user
        // sets from the welcome screen is dropped.
        markSettingsFieldEdited('cfg-remote-password');
      });
      settingsPwInp.addEventListener('input', () => {
        welcomePwInp.value = settingsPwInp.value;
      });
    }
    if (welcomePwInp) {
      welcomePwInp.addEventListener('change', _flushPw);
      welcomePwInp.addEventListener('blur', _flushPw);
      welcomePwInp.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          _flushPw();
          welcomePwInp.blur();
        }
      });
    }
    if (settingsPwInp) {
      settingsPwInp.addEventListener('keydown', e => {
        if (e.key === 'Enter') {
          e.preventDefault();
          _flushPw();
          settingsPwInp.blur();
        }
      });
    }

    if (autocommitBtn) {
      autocommitBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (autocommitBtn.disabled) return;
        const commitTabId = autocommitTargetTabId();
        setAutocommitInFlight(true);
        // Close the drawer so the transcript's autocommit_progress /
        // autocommit_done lines are visible instead of hidden behind
        // the opaque settings sheet.
        closeSettingsPanel();
        api.autocommitAction({
          tabId: commitTabId,
          workDir: workDirForTab(commitTabId),
        });
      });
    }

    if (updateBtn) {
      updateBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        api.runUpdate();
      });
    }

    function openServerResetConfirm() {
      if (!serverResetConfirmModal) return;
      serverResetConfirmModal.classList.add('open');
      if (serverResetConfirmOkBtn) {
        try {
          serverResetConfirmOkBtn.focus();
        } catch (_err) {}
      }
    }
    function closeServerResetConfirm() {
      if (!serverResetConfirmModal) return;
      serverResetConfirmModal.classList.remove('open');
    }
    function isServerResetConfirmOpen() {
      return !!(
        serverResetConfirmModal &&
        serverResetConfirmModal.classList.contains('open')
      );
    }

    if (serverResetBtn) {
      serverResetBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        if (isServerResetConfirmOpen()) return;
        const agentRunning = tabs.some(tab => tab && tab.isRunning);
        if (!agentRunning) {
          api.serverReset();
          return;
        }
        openServerResetConfirm();
      });
    }

    if (serverResetConfirmOkBtn) {
      serverResetConfirmOkBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        closeServerResetConfirm();
        api.serverReset();
      });
    }
    if (serverResetConfirmCancelBtn) {
      serverResetConfirmCancelBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        closeServerResetConfirm();
      });
    }
    if (serverResetConfirmModal) {
      serverResetConfirmModal.addEventListener('click', e => {
        if (e.target === serverResetConfirmModal) closeServerResetConfirm();
      });
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && isServerResetConfirmOpen()) {
          e.preventDefault();
          e.stopPropagation();
          closeServerResetConfirm();
        }
      });
    }

    if (inputClearBtn) {
      inputClearBtn.addEventListener('click', () => {
        inp.value = '';
        inp.style.height = 'auto';
        inputClearBtn.style.display = 'none';
        clearGhost();
        hideAC();
        inp.focus();
      });
    }
    modelBtn.addEventListener('click', e => {
      e.stopPropagation();
      if (modelDropdown.classList.contains('open')) {
        closeModelDD();
        return;
      }
      modelDropdown.classList.add('open');
      modelSearch.value = '';
      if (modelSearchClear) modelSearchClear.style.display = 'none';
      renderModelList('');
      modelSearch.focus();
    });
    modelSearch.addEventListener('input', function () {
      renderModelList(this.value);
      if (modelSearchClear)
        modelSearchClear.style.display = this.value ? '' : 'none';
    });
    if (modelSearchClear) {
      modelSearchClear.addEventListener('click', e => {
        e.stopPropagation();
        modelSearch.value = '';
        renderModelList('');
        modelSearchClear.style.display = 'none';
        modelSearch.focus();
      });
    }
    modelSearch.addEventListener('keydown', e => {
      const items = modelList.querySelectorAll('.model-item');
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        modelDDIdx = Math.min(modelDDIdx + 1, items.length - 1);
        updateSel(items, modelDDIdx);
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        modelDDIdx = Math.max(modelDDIdx - 1, -1);
        updateSel(items, modelDDIdx);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        const ti = modelDDIdx >= 0 ? modelDDIdx : 0;
        if (items[ti]) items[ti].click();
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        closeModelDD();
        return;
      }
    });
    document.addEventListener('click', e => {
      if (!document.getElementById('model-picker').contains(e.target))
        closeModelDD();
      if (!autocomplete.contains(e.target) && e.target !== inp) {
        hideAC();
      }
    });
    function toggleHistorySidebar() {
      if (sidebar.classList.contains('open')) {
        closeSidebar(true);
      } else {
        sidebar.classList.add('open');
        if (!document.body.classList.contains('remote-desktop')) {
          sidebarOverlay.classList.add('open');
        }
        resetHistoryPagination();
        api.getHistory({
          query: historySearch ? historySearch.value : '',
          generation: historyGeneration,
        });
      }
    }
    if (menuBtn) {
      menuBtn.addEventListener('click', toggleHistorySidebar);
    }
    sidebarClose.addEventListener('click', () => closeSidebar(true));
    sidebarOverlay.addEventListener('click', closeSidebar);
    applyRemoteTheme(getSavedRemoteTheme());
    if (
      document.body.classList.contains('remote-chat') &&
      typeof window.matchMedia === 'function'
    ) {
      const desktopMq = window.matchMedia('(min-width: 900px)');
      const applyRemoteDesktop = () => {
        if (desktopMq.matches) {
          document.body.classList.add('remote-desktop');
          if (!sidebar.classList.contains('open')) {
            sidebar.classList.add('open');
            resetHistoryPagination();
            api.getHistory({
              query: historySearch ? historySearch.value : '',
              generation: historyGeneration,
            });
          }
          sidebarOverlay.classList.remove('open');
        } else {
          document.body.classList.remove('remote-desktop');
          sidebar.classList.remove('open');
          sidebarOverlay.classList.remove('open');
        }
      };
      if (typeof desktopMq.addEventListener === 'function') {
        desktopMq.addEventListener('change', applyRemoteDesktop);
      } else if (typeof desktopMq.addListener === 'function') {
        desktopMq.addListener(applyRemoteDesktop);
      }
      applyRemoteDesktop();
    }
    const sidebarResizer = document.getElementById('sidebar-resizer');
    if (document.body.classList.contains('remote-chat') && sidebarResizer) {
      // Bounds come from remote-codex.css: the minimum is the width at
      // which every history filter toggle fits on one line, so neither
      // a drag nor a stale persisted value can wrap them again.
      const SB_MIN = cssPxVar('--sidebar-min-w', 520);
      const SB_MAX = cssPxVar('--sidebar-max-w', 820);
      const CHAT_MIN = cssPxVar('--chat-min-w', 360);
      const SB_KEY = 'kiss-sidebar-w';
      // Widest the panel may become on the CURRENT window: a wide
      // panel dragged on a big monitor must not squeeze the chat into
      // an unusable sliver after the window shrinks.
      const sidebarWindowMax = () =>
        Math.max(SB_MIN, Math.min(SB_MAX, window.innerWidth - CHAT_MIN));
      const sidebarDefaultW = () =>
        Math.max(
          SB_MIN,
          Math.min(sidebarWindowMax(), Math.round(window.innerWidth * 0.34)),
        );
      const setSidebarW = px => {
        const max = sidebarWindowMax();
        const w = Math.max(SB_MIN, Math.min(max, Math.round(px)));
        document.documentElement.style.setProperty('--sidebar-w', w + 'px');
        sidebarResizer.setAttribute('aria-valuemax', String(max));
        sidebarResizer.setAttribute('aria-valuenow', String(w));
        return w;
      };
      sidebarResizer.setAttribute('aria-valuemin', String(SB_MIN));
      sidebarResizer.setAttribute('aria-valuemax', String(sidebarWindowMax()));
      sidebarResizer.setAttribute('aria-valuenow', String(sidebarDefaultW()));
      let sidebarW = sidebarDefaultW();
      let persisted = null;
      try {
        persisted = window.localStorage.getItem(SB_KEY);
      } catch {}
      if (persisted !== null && /^\d+$/.test(persisted)) {
        sidebarW = setSidebarW(parseInt(persisted, 10));
      }
      const persistSidebarW = () => {
        try {
          window.localStorage.setItem(SB_KEY, String(sidebarW));
        } catch {}
      };
      let sidebarResizing = false;
      const endSidebarResize = e => {
        if (!sidebarResizing) return;
        sidebarResizing = false;
        document.body.classList.remove('sidebar-resizing');
        try {
          if (
            e.pointerId !== undefined &&
            typeof sidebarResizer.releasePointerCapture === 'function'
          ) {
            sidebarResizer.releasePointerCapture(e.pointerId);
          }
        } catch {}
        persistSidebarW();
      };
      sidebarResizer.addEventListener('pointerdown', e => {
        if (e.button !== 0) return;
        if (!document.body.classList.contains('remote-desktop')) return;
        e.preventDefault();
        sidebarResizing = true;
        document.body.classList.add('sidebar-resizing');
        try {
          if (
            e.pointerId !== undefined &&
            typeof sidebarResizer.setPointerCapture === 'function'
          ) {
            sidebarResizer.setPointerCapture(e.pointerId);
          }
        } catch {}
      });
      sidebarResizer.addEventListener('pointermove', e => {
        if (!sidebarResizing) return;
        if (!document.body.classList.contains('remote-desktop')) return;
        sidebarW = setSidebarW(e.clientX);
      });
      sidebarResizer.addEventListener('pointerup', endSidebarResize);
      sidebarResizer.addEventListener('pointercancel', endSidebarResize);
      sidebarResizer.addEventListener('dblclick', () => {
        if (!document.body.classList.contains('remote-desktop')) return;
        sidebarW = setSidebarW(sidebarDefaultW());
        try {
          window.localStorage.removeItem(SB_KEY);
        } catch {}
      });
      sidebarResizer.addEventListener('keydown', e => {
        if (!document.body.classList.contains('remote-desktop')) return;
        if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
        e.preventDefault();
        sidebarW = setSidebarW(sidebarW + (e.key === 'ArrowRight' ? 16 : -16));
        persistSidebarW();
      });
      // Re-apply the width whenever the window changes size so a wide
      // panel narrows instead of crushing the chat.  The preferred
      // width in `sidebarW` is intentionally left untouched.
      window.addEventListener('resize', () => {
        if (!document.body.classList.contains('remote-desktop')) return;
        setSidebarW(sidebarW);
      });
    }
    if (frequentTasksBtn) {
      frequentTasksBtn.addEventListener('click', () => {
        if (frequentPanel && frequentPanel.classList.contains('open')) {
          closeFrequentPanel();
        } else {
          openFrequentPanel();
        }
      });
    }
    if (frequentPanelClose) {
      frequentPanelClose.addEventListener('click', closeFrequentPanel);
    }
    if (frequentOverlay) {
      frequentOverlay.addEventListener('click', closeFrequentPanel);
    }
    if (tricksBtn) {
      tricksBtn.addEventListener('click', () => {
        if (tricksPanel && tricksPanel.classList.contains('open')) {
          closeTricksPanel();
        } else {
          openTricksPanel();
        }
      });
    }
    if (tricksPanelClose) {
      tricksPanelClose.addEventListener('click', closeTricksPanel);
    }
    if (tricksOverlay) {
      tricksOverlay.addEventListener('click', closeTricksPanel);
    }
    if (settingsPanelClose) {
      settingsPanelClose.addEventListener('click', closeSettingsPanel);
    }
    if (settingsOverlay) {
      settingsOverlay.addEventListener('click', closeSettingsPanel);
    }
    if (settingsPanel) {
      const noteSettingsEdit = e => {
        if (e.target && e.target.id) markSettingsFieldEdited(e.target.id);
      };
      settingsPanel.addEventListener('input', noteSettingsEdit);
      settingsPanel.addEventListener('change', noteSettingsEdit);
    }
    historySearch.addEventListener('input', () => {
      resetHistoryPagination();
      api.getHistory({
        query: historySearch.value,
        generation: historyGeneration,
      });
      if (historySearchClear)
        historySearchClear.style.display = historySearch.value ? '' : 'none';
    });
    if (historySearchClear) {
      historySearchClear.addEventListener('click', () => {
        historySearch.value = '';
        if (historySearchClear) historySearchClear.style.display = 'none';
        resetHistoryPagination();
        api.getHistory({query: '', generation: historyGeneration});
        historySearch.focus();
      });
    }
    const {
      hfRunning,
      hfErrors,
      hfCompleted,
      hfWorkspace,
      hfFavorite,
      hfFrom,
      hfTo,
    } = getHistoryFilterEls();
    [
      hfRunning,
      hfErrors,
      hfCompleted,
      hfWorkspace,
      hfFavorite,
      hfFrom,
      hfTo,
    ].forEach(el => {
      if (el) el.addEventListener('change', applyHistoryFilterVisibility);
    });
    [hfFrom, hfTo].forEach(el => {
      if (el) {
        el.addEventListener('change', () => {
          historyDateRangeUserSet = true;
        });
      }
    });
    const historyFiltersToggle = document.getElementById(
      'history-filters-toggle',
    );
    const historyFiltersBody = document.getElementById('history-filters-body');
    const HISTORY_FILTERS_COLLAPSED_KEY = 'kissSorcar.historyFiltersCollapsed';
    if (historyFiltersToggle && historyFiltersBody) {
      const setHistoryFiltersExpanded = expanded => {
        historyFiltersBody.hidden = !expanded;
        historyFiltersToggle.setAttribute(
          'aria-expanded',
          expanded ? 'true' : 'false',
        );
        historyFiltersToggle.classList.toggle('expanded', expanded);
      };
      let filtersCollapsed = true;
      try {
        filtersCollapsed =
          window.localStorage.getItem(HISTORY_FILTERS_COLLAPSED_KEY) !== '0';
      } catch {}
      setHistoryFiltersExpanded(!filtersCollapsed);
      historyFiltersToggle.addEventListener('click', () => {
        const nowCollapsed =
          historyFiltersToggle.getAttribute('aria-expanded') === 'true';
        setHistoryFiltersExpanded(!nowCollapsed);
        try {
          window.localStorage.setItem(
            HISTORY_FILTERS_COLLAPSED_KEY,
            nowCollapsed ? '1' : '0',
          );
        } catch {}
      });
    }
    const hfDateClear = document.getElementById('hf-date-clear');
    if (hfDateClear) {
      hfDateClear.addEventListener('click', e => {
        e.stopPropagation();
        if (hfFrom) hfFrom.value = '';
        if (hfTo) hfTo.value = '';
        historyDateRangeUserSet = true;
        applyHistoryFilterVisibility();
      });
    }
    const hfFromBtn = document.getElementById('hf-from-btn');
    const hfToBtn = document.getElementById('hf-to-btn');
    if (hfFromBtn) {
      hfFromBtn.addEventListener('click', e => {
        e.stopPropagation();
        openCustomDatePicker(hfFrom, hfFromBtn);
      });
    }
    if (hfToBtn) {
      hfToBtn.addEventListener('click', e => {
        e.stopPropagation();
        openCustomDatePicker(hfTo, hfToBtn);
      });
    }
    historyList.addEventListener('scroll', () => {
      if (historyLoading || !historyHasMore) return;
      if (
        historyList.scrollTop + historyList.clientHeight >=
        historyList.scrollHeight - 50
      ) {
        historyLoading = true;
        const loader = document.createElement('div');
        loader.className = 'sidebar-loading';
        loader.id = 'history-loader';
        loader.textContent = 'Loading...';
        historyList.appendChild(loader);
        api.getHistory({
          query: historySearch.value,
          offset: historyOffset,
          generation: historyGeneration,
        });
      }
    });
    document.addEventListener('click', e => {
      const el = e.target.closest('[data-path]');
      if (el && el.dataset.path) {
        const raw = el.dataset.path;
        const msg = {
          type: 'openFile',
          path: raw,
          workDir: workDirForTab(activeTabId),
          tabId: activeTabId,
        };
        const match = raw.match(/^(.+):(\d+)$/);
        if (match) {
          msg.path = match[1];
          msg.line = parseInt(match[2], 10);
        }
        api.send(msg);
      }
    });

    inp.addEventListener('paste', e => {
      const items = (e.clipboardData || {}).items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind !== 'file') continue;
        const file = item.getAsFile();
        // The pasted item's own `type` is authoritative here, but Safari
        // leaves it empty for some pictures, so the file name decides too.
        if (file && isAttachableFile(file)) {
          e.preventDefault();
          readFileAsAttachment(file);
        }
      }
    });

    if (inputContainer) {
      inputContainer.addEventListener('dragover', e => {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.add('drag-over');
      });
      inputContainer.addEventListener('dragleave', e => {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.remove('drag-over');
      });
      inputContainer.addEventListener('drop', e => {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.remove('drag-over');
        const uriList =
          e.dataTransfer && e.dataTransfer.getData('text/uri-list');
        if (uriList) {
          const uris = uriList.split(/\r?\n/).filter(u => {
            return u && !u.startsWith('#');
          });
          if (uris.length > 0) {
            api.resolveDroppedPaths({uris: uris});
            return;
          }
        }
        const files = e.dataTransfer && e.dataTransfer.files;
        if (!files) return;
        Array.from(files).forEach(file => {
          if (isAttachableFile(file)) readFileAsAttachment(file);
        });
      });
    }

    window.addEventListener('message', event => {
      handleEvent(event.data);
    });

    window.addEventListener('kiss-voice-post', event => {
      const detail = event && event.detail;
      if (detail && detail.type) api.send(detail);
    });

    // tableak-coverage:start
    window.addEventListener('kiss-voice-submit', event => {
      if (!isFromSpeechTab(event)) return;
      sendMessage();
    });

    window.addEventListener('kiss-voice-answer', event => {
      if (!isFromSpeechTab(event)) return;
      const tab = getTab(activeTabId);
      if (tab && tab.askPendingQuestion !== null) submitAskForTab(tab);
    });
    // tableak-coverage:end
  }

  // ---------------------------------------------------------------------
  // Attachment intake
  //
  // An iPhone whose Settings > Camera > Formats is "High Efficiency" (the
  // factory default since the iPhone 7) stores every photo as HEIC.  WebKit
  // transcodes a picked file only when the picker's `accept` list does not
  // already admit the file's own type, so a list containing `image/*` leaves
  // the HEIC alone and `file.type` arrives as "image/heic".  The OpenAI and
  // Anthropic vision APIs reject that MIME type outright (of the providers
  // used here only Gemini accepts it), so the photo used to be dropped
  // further down the pipeline with no user-visible error.  Safari 17+ (macOS
  // and iOS) decodes HEIC natively, so the photo is re-encoded to JPEG here,
  // in the browser, with no extra dependency.  Huge photos are downscaled in
  // the same pass.
  // ---------------------------------------------------------------------

  // Anthropic downsizes anything longer than 1568px on its long edge before
  // the model sees it, and that is well inside what the other providers
  // accept, so it is the target for every attachment.
  const ATTACH_MAX_EDGE = 1568;
  const ATTACH_JPEG_QUALITY = 0.82;
  // Above this size an image is re-encoded even if its format is already
  // supported: base64 inflates payloads by a third and iPhone captures run to
  // several megabytes.
  const ATTACH_MAX_BYTES = 1500 * 1024;
  // A valid JPEG of any real photo is far bigger than this; mobile Safari has
  // been seen returning byte-stub blobs instead of an encoded image.
  const ATTACH_MIN_JPEG_BYTES = 256;
  // Image formats every supported vision API understands.
  const MODEL_IMAGE_MIME_TYPES = [
    'image/jpeg',
    'image/png',
    'image/webp',
    'image/gif',
  ];
  // ISO base media file format major brands (bytes 8..12) used by HEIF
  // containers, including the burst/Live-Photo variants.
  const HEIF_BRANDS = [
    'heic',
    'heix',
    'heim',
    'heis',
    'hevc',
    'hevx',
    'hevm',
    'hevs',
    'mif1',
    'msf1',
  ];
  const ATTACH_EXT_MIME = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    gif: 'image/gif',
    webp: 'image/webp',
    heic: 'image/heic',
    heif: 'image/heif',
    hif: 'image/heif',
    pdf: 'application/pdf',
  };

  function attachMimeFromName(name) {
    const m = /\.([A-Za-z0-9]+)$/.exec(name || '');
    return (m && ATTACH_EXT_MIME[m[1].toLowerCase()]) || '';
  }

  /** Best guess at a picked file's MIME type; iOS often reports none. */
  function attachMimeOf(file) {
    return (file.type || attachMimeFromName(file.name) || '').toLowerCase();
  }

  /**
   * True for files the chat accepts: images (whatever the camera called them)
   * and PDFs.  Used by the paste and drag-and-drop handlers, which see files
   * whose `type` iOS and macOS sometimes leave empty.
   */
  function isAttachableFile(file) {
    const mime = attachMimeOf(file);
    return mime.startsWith('image/') || mime === 'application/pdf';
  }

  /** True when `data`'s ISO-BMFF header identifies a HEIF/HEIC container. */
  function isHeifHeader(head) {
    if (head.length < 12) return false;
    let brand = '';
    for (let i = 4; i < 12; i++) brand += String.fromCharCode(head[i]);
    if (brand.slice(0, 4) !== 'ftyp') return false;
    return HEIF_BRANDS.indexOf(brand.slice(4).toLowerCase()) >= 0;
  }

  async function isHeifFile(file) {
    if (attachMimeOf(file).indexOf('heic') >= 0) return true;
    if (attachMimeOf(file).indexOf('heif') >= 0) return true;
    if (typeof file.slice !== 'function') return false;
    try {
      const head = await file.slice(0, 12).arrayBuffer();
      return isHeifHeader(new Uint8Array(head));
    } catch (_e) {
      return false;
    }
  }

  /**
   * Decode `file` and re-encode it as a downscaled JPEG Blob.
   *
   * Args:
   *   file: The picked File/Blob, in any format the browser can decode.
   *
   * Returns:
   *   A promise for an `image/jpeg` Blob at most ATTACH_MAX_EDGE px per edge,
   *   with EXIF rotation baked into the pixels.
   *
   * Throws:
   *   Error if the browser cannot decode the format (every engine except
   *   Safari 17+ for HEIC) or cannot encode a JPEG.
   */
  async function encodeAsJpeg(file) {
    if (typeof window.createImageBitmap !== 'function') {
      throw new Error('this browser cannot convert the image');
    }
    let bitmap;
    try {
      // 'from-image' bakes in the EXIF rotation, which every vision API
      // ignores.  The option name only landed in Safari 16, hence the retry.
      bitmap = await window.createImageBitmap(file, {
        imageOrientation: 'from-image',
      });
    } catch (_e) {
      bitmap = await window.createImageBitmap(file);
    }
    const scale = Math.min(
      1,
      ATTACH_MAX_EDGE / Math.max(bitmap.width, bitmap.height),
    );
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const h = Math.max(1, Math.round(bitmap.height * scale));
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    try {
      const ctx = canvas.getContext('2d', {alpha: false});
      if (!ctx) throw new Error('this browser cannot convert the image');
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(bitmap, 0, 0, w, h);
      return await canvasToJpeg(canvas);
    } finally {
      if (typeof bitmap.close === 'function') bitmap.close();
      // Release the decoded pixels: mobile Safari kills tabs that hold on to
      // multi-megapixel canvas backing stores.
      canvas.width = 1;
      canvas.height = 1;
    }
  }

  function canvasToJpeg(canvas) {
    return new Promise((resolve, reject) => {
      if (typeof canvas.toBlob !== 'function') {
        reject(new Error('this browser cannot convert the image'));
        return;
      }
      canvas.toBlob(
        blob => {
          if (!blob || blob.size < ATTACH_MIN_JPEG_BYTES) {
            reject(new Error('the converted image came back empty'));
          } else {
            resolve(blob);
          }
        },
        'image/jpeg',
        ATTACH_JPEG_QUALITY,
      );
    });
  }

  function jpegNameFor(name) {
    const base = String(name || 'photo').replace(/\.[^./\\]*$/, '');
    return (base || 'photo') + '.jpg';
  }

  function blobToBase64(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const url = String(reader.result || '');
        const comma = url.indexOf(',');
        if (comma < 0) reject(new Error('the file could not be read'));
        else resolve(url.slice(comma + 1));
      };
      reader.onerror = () => {
        reject(reader.error || new Error('the file could not be read'));
      };
      reader.readAsDataURL(blob);
    });
  }

  /**
   * Turn a picked file into the payload the daemon should receive, converting
   * formats no vision API accepts and shrinking oversized photos.
   */
  async function prepareAttachment(file) {
    const mime = attachMimeOf(file);
    const heif =
      mime.startsWith('image/') || !mime ? await isHeifFile(file) : false;
    const reencode =
      heif ||
      (mime.startsWith('image/') &&
        // An unknown image codec (HEIF, AVIF, TIFF...) has to be converted;
        // a supported one only when it is too big to ship as base64.  GIFs
        // are left alone because re-encoding would drop the animation.
        (MODEL_IMAGE_MIME_TYPES.indexOf(mime) < 0 ||
          (mime !== 'image/gif' && file.size > ATTACH_MAX_BYTES)));
    if (!reencode) {
      return {
        name: file.name || 'attachment',
        type: mime || 'application/octet-stream',
        data: await blobToBase64(file),
      };
    }
    const jpeg = await encodeAsJpeg(file);
    return {
      name: jpegNameFor(file.name),
      type: 'image/jpeg',
      data: await blobToBase64(jpeg),
    };
  }

  /**
   * Fill `slot` with the bytes of `file`, or drop it and explain why.
   *
   * The tab's own lists are captured up front: the user may switch tabs while
   * a photo is being converted, and the outcome belongs to the tab that
   * picked the file, not to whichever tab is on screen when it lands.
   *
   * Returns:
   *   A promise for whether the slot now holds a usable attachment.
   */
  async function fillAttachmentSlot(file, slot) {
    const ownerFiles = attachments;
    const ownerErrors = attachErrors;
    try {
      const ready = await prepareAttachment(file);
      slot.name = ready.name;
      slot.type = ready.type;
      slot.data = ready.data;
      return true;
    } catch (err) {
      const idx = ownerFiles.indexOf(slot);
      if (idx >= 0) ownerFiles.splice(idx, 1);
      const why = (err && err.message) || 'it could not be attached';
      ownerErrors.push((file.name || 'attachment') + ': ' + why);
      return false;
    } finally {
      slot.pending = false;
      updateInputDisabled();
      renderFileChips();
    }
  }

  /**
   * Add `file` to the pending attachments of the active chat tab.
   *
   * A placeholder chip appears immediately and keeps the slot's position
   * while the file is read (and, for a camera HEIC, converted); the send
   * button stays disabled until every slot is filled.
   *
   * Args:
   *   file: A File from the picker, a paste or a drop.
   *
   * Returns:
   *   A promise that settles once the slot is filled or has been dropped.
   */
  function readFileAsAttachment(file) {
    const slot = {
      name: file.name || 'attachment',
      type: attachMimeOf(file),
      data: '',
      pending: true,
    };
    attachments.push(slot);
    slot.promise = fillAttachmentSlot(file, slot);
    updateInputDisabled();
    renderFileChips();
    return slot.promise;
  }

  function hasPendingAttachments() {
    return attachments.some(a => a.pending);
  }

  /**
   * Wait for every in-flight attachment of the active tab.
   *
   * Returns:
   *   A promise for whether all of them arrived intact.
   */
  async function attachmentsReady() {
    const results = await Promise.all(
      attachments.filter(a => a.pending).map(a => a.promise),
    );
    return results.every(ok => ok);
  }

  async function sendMessage() {
    let prompt = inp.value.trim();
    if (!prompt) return;

    // Enter and the voice trigger bypass the disabled send button, so a photo
    // that is still being converted has to be waited for rather than lost.
    // With nothing to wait for, no await runs and the send stays synchronous:
    // callers rely on the message being posted before they return.
    if (hasPendingAttachments()) {
      // Only the first waiting caller may proceed, or a burst of Enters would
      // submit the same prompt several times.
      if (awaitingAttachments) return;
      const tabAtEntry = activeTabId;
      awaitingAttachments = true;
      let ready = false;
      try {
        ready = await attachmentsReady();
      } finally {
        awaitingAttachments = false;
      }
      // A conversion that failed leaves its error chip in place: sending the
      // prompt without the photo is exactly the silent loss to avoid.  A tab
      // switch means this submission no longer matches what the user sees.
      if (!ready || activeTabId !== tabAtEntry) return;
      // The composer may have been edited during the wait.
      prompt = inp.value.trim();
      if (!prompt) return;
    }

    if (histCache[0] !== prompt) {
      histCache.unshift(prompt);
    }
    const curTab = getTab(activeTabId);

    if (isRunning) {
      api.appendUserMessage({prompt: prompt, tabId: activeTabId});
      inp.value = '';
      inp.style.height = 'auto';
      attachments = [];
      attachErrors = [];
      updateInputDisabled();
      renderFileChips();
      clearGhost();
      histIdx = -1;
      if (inputClearBtn) inputClearBtn.style.display = 'none';
      return;
    }

    const msg = {
      type: 'submit',
      prompt: prompt,
      model: selectedModel,
      tabId: activeTabId,
      attachments: attachments
        .filter(a => a.data)
        .map(a => {
          return {name: a.name, mimeType: a.type, data: a.data};
        }),
      useWorktree: !!(worktreeToggleBtn && worktreeToggleBtn.checked),
      useParallel: true,
      autoCommit: !!(autocommitToggleBtn && autocommitToggleBtn.checked),
    };
    if (curTab && curTab.workDir) msg.workDir = curTab.workDir;
    api.send(msg);
    t0 = Date.now();
    endTs = 0;
    if (curTab) {
      curTab.t0 = t0;
      curTab.endTs = 0;
      // tableak-coverage:start
      // Claim the task before its id exists. Until the daemon replies with a
      // real task id, this marker is what makes the submitting tab -- and
      // only it -- a legitimate owner of the output that is about to arrive.
      if (!curTab.currentTaskId) curTab.pendingTaskId = 'pending:' + curTab.id;
      // tableak-coverage:end
    }
    inp.value = '';
    inp.style.height = 'auto';
    attachments = [];
    attachErrors = [];
    updateInputDisabled();
    renderFileChips();
    clearGhost();
    histIdx = -1;
    if (inputClearBtn) inputClearBtn.style.display = 'none';
  }

  function ensureAskElementsForTab(tab) {
    if (tab.askQuestionEl) return;
    const q = document.createElement('div');
    q.className = 'ask-user-question';
    const i = document.createElement('textarea');
    i.className = 'ask-user-input';
    i.placeholder = 'Your answer...';
    const s = document.createElement('button');
    s.className = 'ask-user-submit';
    s.setAttribute('data-tooltip', 'Submit answer');
    s.textContent = 'Submit';
    s.addEventListener('click', () => {
      submitAskForTab(tab);
    });
    i.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitAskForTab(tab);
      }
    });
    const m = document.createElement('button');
    m.className = 'ask-user-mic';
    m.setAttribute(
      'data-tooltip',
      "Voice trigger: listen for the word 'Sorcar'",
    );
    m.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
      'stroke-linejoin="round">' +
      '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>' +
      '<path d="M19 10v2a7 7 0 0 1-14 0v-2"/>' +
      '<line x1="12" y1="19" x2="12" y2="23"/>' +
      '<line x1="8" y1="23" x2="16" y2="23"/></svg>';
    const row = document.createElement('div');
    row.className = 'ask-user-actions';
    row.appendChild(s);
    row.appendChild(m);
    tab.askQuestionEl = q;
    tab.askInputEl = i;
    tab.askActionsEl = row;
  }

  function setAskQuestionTextForTab(tab, text) {
    const t = text || '';
    if (typeof marked !== 'undefined') {
      tab.askQuestionEl.innerHTML = kissSanitize(marked.parse(t));
      tab.askQuestionEl.classList.add('md-body');
      hlBlock(tab.askQuestionEl);
    } else {
      tab.askQuestionEl.textContent = t;
    }
  }

  function clearAskSlot() {
    if (!askUserSlot) return;
    while (askUserSlot.firstChild)
      askUserSlot.removeChild(askUserSlot.firstChild);
    if (askUserModal) askUserModal.style.display = 'none';
  }

  function mountAskForTab(tab) {
    if (!askUserSlot) return;
    while (askUserSlot.firstChild)
      askUserSlot.removeChild(askUserSlot.firstChild);
    askUserSlot.appendChild(tab.askQuestionEl);
    askUserSlot.appendChild(tab.askInputEl);
    askUserSlot.appendChild(tab.askActionsEl);
    askUserModal.style.display = 'flex';
    window.dispatchEvent(new CustomEvent('kiss-ask-mic-mounted'));
    setTimeout(() => {
      if (tab.id === activeTabId && tab.askInputEl) tab.askInputEl.focus();
    }, 0);
  }

  function showAskForTab(tab) {
    if (tab.askPendingQuestion === null) {
      if (tab.id === activeTabId) clearAskSlot();
      return;
    }
    ensureAskElementsForTab(tab);
    setAskQuestionTextForTab(tab, tab.askPendingQuestion);
    tab.askInputEl.value = '';
    if (tab.id === activeTabId) mountAskForTab(tab);
  }

  function isAskSameChatTab(sourceTab, candidate) {
    if (!sourceTab || !candidate) return false;
    if (candidate.id === sourceTab.id) return true;
    const chatId = String(sourceTab.backendChatId || '');
    return !!chatId && String(candidate.backendChatId || '') === chatId;
  }

  // Retire the question of exactly one tab. Returns true when the retired
  // question was the one on screen, so callers can drop the modal once.
  function retireAskForTab(tab) {
    tab.askPendingQuestion = null;
    if (tab.askInputEl) tab.askInputEl.value = '';
    return tab.id === activeTabId;
  }

  // A tab's own task ended, so only its question dies. Sibling tabs sharing
  // the backend chat run their own tasks and may still be waiting on answers.
  function clearAskForTab(tab) {
    if (retireAskForTab(tab)) clearAskSlot();
    renderTabBar();
  }

  function clearAskForMatchingChatTabs(sourceTab) {
    let shouldClearSlot = false;
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      if (!isAskSameChatTab(sourceTab, tab)) continue;
      if (retireAskForTab(tab)) shouldClearSlot = true;
    }
    if (shouldClearSlot) clearAskSlot();
    renderTabBar();
  }

  function submitAskForTab(tab) {
    const answer = tab.askInputEl ? tab.askInputEl.value : '';
    api.userAnswer({answer: answer, tabId: tab.id});
    clearAskForMatchingChatTabs(tab);
  }

  function syncAskModalToActiveTab() {
    clearAskSlot();
    const tab = getTab(activeTabId);
    if (!tab || tab.askPendingQuestion === null) return;
    ensureAskElementsForTab(tab);
    mountAskForTab(tab);
  }

  function handleFileSelect(e) {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    Array.from(files).forEach(file => {
      readFileAsAttachment(file);
    });
  }

  function renderFileChips() {
    fileChips.innerHTML = '';
    attachments.forEach((att, idx) => {
      const chip = document.createElement('div');
      chip.className = 'file-chip' + (att.pending ? ' pending' : '');
      const isImage = !att.pending && (att.type || '').startsWith('image/');
      chip.innerHTML =
        (isImage
          ? '<img src="data:' + att.type + ';base64,' + att.data + '">'
          : '<span class="fc-icon">' +
            (att.pending ? '\u22EF' : '\uD83D\uDCC4') +
            '</span>') +
        '<span>' +
        esc(att.name) +
        '</span>' +
        '<span class="fc-rm" data-idx="' +
        idx +
        '">&times;</span>';
      chip.querySelector('.fc-rm').addEventListener('click', () => {
        attachments.splice(idx, 1);
        updateInputDisabled();
        renderFileChips();
      });
      fileChips.appendChild(chip);
    });
    attachErrors.forEach((msg, idx) => {
      const chip = document.createElement('div');
      chip.className = 'file-chip error';
      chip.innerHTML =
        '<span class="fc-icon">\u26A0</span><span>' +
        esc(msg) +
        '</span><span class="fc-rm">&times;</span>';
      chip.querySelector('.fc-rm').addEventListener('click', () => {
        attachErrors.splice(idx, 1);
        renderFileChips();
      });
      fileChips.appendChild(chip);
    });
  }

  function renderModelItem(m) {
    const d = mkEl(
      'div',
      'model-item' + (m.name === selectedModel ? ' active' : ''),
    );
    const price = '$' + m.inp.toFixed(2) + ' / $' + m.out.toFixed(2);
    d.innerHTML =
      '<span>' +
      esc(m.name) +
      '</span><span class="model-cost">' +
      price +
      '</span>';
    d.addEventListener('click', () => {
      selectModel(m.name);
    });
    return d;
  }

  function renderModelList(q) {
    modelList.innerHTML = '';
    modelDDIdx = -1;
    const ql = q.toLowerCase();
    const used = [],
      rest = [];
    allModels.forEach(m => {
      if (ql && m.name.toLowerCase().indexOf(ql) < 0) return;
      if (m.uses > 0) used.push(m);
      else rest.push(m);
    });
    used.sort((a, b) => {
      return b.uses - a.uses;
    });
    if (used.length) {
      const hdr = mkEl('div', 'model-group-hdr');
      hdr.textContent = 'Recently Used';
      modelList.appendChild(hdr);
      used.forEach(m => {
        modelList.appendChild(renderModelItem(m));
      });
    }
    let lastVendor = '';
    rest.forEach(m => {
      const v = m.vendor;
      if (v !== lastVendor) {
        const hdr = mkEl('div', 'model-group-hdr');
        hdr.textContent = v;
        modelList.appendChild(hdr);
        lastVendor = v;
      }
      modelList.appendChild(renderModelItem(m));
    });
  }

  function selectModel(name) {
    selectedModel = name;
    agentModel = '';
    // Record it on the tab straight away rather than waiting for the
    // next saveCurrentTab: a `models` refresh arriving in between reads
    // the tab's value to decide whether the tab has its own pick.
    const picked = getTab(activeTabId);
    if (picked) {
      picked.selectedModel = name;
      picked.agentModel = '';
    }
    refreshModelLabel();
    closeModelDD();
    renderModelList('');
    api.selectModel({model: name, tabId: activeTabId});
  }

  function closeModelDD() {
    modelDropdown.classList.remove('open');
    modelSearch.value = '';
    if (modelSearchClear) modelSearchClear.style.display = 'none';
    modelDDIdx = -1;
  }

  function updateSel(items, idx) {
    items.forEach((it, i) => {
      it.classList.toggle('sel', i === idx);
    });
    if (idx >= 0) items[idx].scrollIntoView({block: 'nearest'});
  }

  function makeSidebarDeleteConfirm(opts) {
    const delBtn = document.createElement('button');
    delBtn.className = 'sidebar-item-delete';
    delBtn.dataset.tooltip = 'Delete';
    delBtn.setAttribute('aria-label', opts.ariaLabel);
    delBtn.innerHTML =
      '<svg width="11" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>';
    const confirmWrap = document.createElement('span');
    confirmWrap.className = 'sidebar-item-confirm';
    confirmWrap.style.display = 'none';
    const confirmBtn = document.createElement('button');
    confirmBtn.className = 'sidebar-confirm-yes';
    confirmBtn.dataset.tooltip = 'Confirm delete';
    confirmBtn.textContent = 'Delete';
    const cancelBtn = document.createElement('button');
    cancelBtn.className = 'sidebar-confirm-no';
    cancelBtn.dataset.tooltip = 'Cancel';
    cancelBtn.textContent = 'Cancel';
    confirmWrap.appendChild(confirmBtn);
    confirmWrap.appendChild(cancelBtn);
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      delBtn.style.display = 'none';
      confirmWrap.style.display = '';
      if (opts.onShowConfirm) opts.onShowConfirm();
    });
    confirmBtn.addEventListener('click', e => {
      e.stopPropagation();
      opts.onConfirm();
    });
    cancelBtn.addEventListener('click', e => {
      e.stopPropagation();
      confirmWrap.style.display = 'none';
      delBtn.style.display = '';
      if (opts.onCancel) opts.onCancel();
    });
    return {delBtn: delBtn, confirmWrap: confirmWrap};
  }

  // Expanded-state store for history task panels. Keyed by a stable
  // per-task key (not the transient session objects) so that the
  // expanded/collapsed choice survives backend-driven history
  // re-renders, which always deliver freshly constructed sessions.
  const historyExpandedTaskKeys = new Set();

  function historyCollapseKey(session) {
    if (session.task_id) return 'task:' + session.task_id;
    return (
      'chat:' + String(session.id || '') + ':' + String(session.timestamp || 0)
    );
  }

  function makeSidebarCollapseToggle(itemDiv, session) {
    const key = historyCollapseKey(session);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sidebar-item-collapse';
    btn.innerHTML =
      '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>';
    const applyCollapseState = () => {
      const collapsed = !historyExpandedTaskKeys.has(key);
      itemDiv.classList.toggle('collapsed', collapsed);
      btn.dataset.tooltip = collapsed ? 'Show details' : 'Hide details';
      btn.setAttribute(
        'aria-label',
        collapsed ? 'Expand task details' : 'Collapse task details',
      );
      btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    };
    applyCollapseState();
    btn.addEventListener('click', e => {
      e.stopPropagation();
      e.preventDefault();
      if (historyExpandedTaskKeys.has(key)) {
        historyExpandedTaskKeys.delete(key);
      } else {
        historyExpandedTaskKeys.add(key);
      }
      applyCollapseState();
    });
    return btn;
  }

  function makeSidebarCopyButton(text) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sidebar-item-copy';
    btn.setAttribute('aria-label', 'Copy task to clipboard');
    wireCopyButton(btn, text, false, false);
    return btn;
  }

  function wireCopyButton(btn, text, retryFallback, resetFlashTimer) {
    btn.innerHTML = PANEL_COPY_SVG;

    let flashTimer = 0;
    const flash = () => {
      btn.innerHTML = PANEL_CHECK_SVG;
      btn.classList.add('copied');
      if (resetFlashTimer) clearTimeout(flashTimer);
      flashTimer = setTimeout(() => {
        btn.innerHTML = PANEL_COPY_SVG;
        btn.classList.remove('copied');
      }, 1500);
    };

    btn.addEventListener('click', e => {
      e.stopPropagation();
      e.preventDefault();
      const payload = String(text == null ? '' : text);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(payload).then(flash, () => {
          if (retryFallback && fallbackCopyText(payload)) flash();
        });
      } else if (fallbackCopyText(payload)) {
        flash();
      }
    });
  }

  function makeIdCopyButton(idText, kind) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'ids-copy-btn ids-copy-' + kind;
    btn.dataset.tooltip = 'Copy ' + kind + ' id';
    btn.setAttribute('aria-label', 'Copy ' + kind + ' id to clipboard');
    wireCopyButton(btn, idText, true, true);
    return btn;
  }

  function chatIdBgColor(chatId) {
    if (!chatId) return 'hsl(0, 0%, 75%)';
    let hash = 5381;
    for (let i = 0; i < chatId.length; i++) {
      hash = (hash << 5) + hash + chatId.charCodeAt(i);
      hash |= 0;
    }
    const hue = Math.abs(hash) % 360;
    return 'hsl(' + hue + ', 55%, 75%)';
  }

  function renderHistory(sessions, offset, generation) {
    if (generation !== historyGeneration) return;

    historyLoading = false;
    const loader = document.getElementById('history-loader');
    if (loader) loader.remove();

    if (offset === 0) {
      allHistSessions = [];
      if (sessions.length === 0) {
        historyList.innerHTML =
          '<div class="sidebar-empty">No conversations yet</div>';
        historyHasMore = false;
        return;
      }
      historyList.innerHTML = '';
    }
    allHistSessions = allHistSessions.concat(sessions);

    const newRunningTaskIds = new Set();
    allHistSessions.forEach(s => {
      if (s.is_running && s.task_id) newRunningTaskIds.add(s.task_id);
    });
    historyLastRunningTaskIds.forEach(id => {
      if (!newRunningTaskIds.has(id)) historyJustCompletedTaskIds.add(id);
    });
    historyLastRunningTaskIds.clear();
    newRunningTaskIds.forEach(id => historyLastRunningTaskIds.add(id));

    sessions.forEach(s => {
      const div = document.createElement('div');
      div.className = 'sidebar-item running-item';
      div.tabIndex = 0;
      div.setAttribute('role', 'button');
      div.addEventListener('keydown', e => {
        if (e.target !== div) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          div.click();
        }
      });
      div.dataset.category = s.is_running
        ? 'running'
        : s.failed
          ? 'errors'
          : 'completed';
      div.dataset.timestamp = String(Number(s.timestamp || 0));
      div.dataset.favorite = s.is_favorite ? '1' : '0';
      div.dataset.workDir = s.work_dir || '';
      const itemText = s.title || s.preview || 'Untitled';
      div.dataset.tooltip = s.preview || itemText;
      div.style.setProperty('--task-color', chatIdBgColor(String(s.id)));

      if (s.is_running) {
        const runningDot = document.createElement('span');
        runningDot.className = 'sidebar-item-running';
        runningDot.dataset.tooltip = 'Task running';
        runningDot.setAttribute('aria-label', 'Task running');
        div.appendChild(runningDot);
      } else if (s.failed) {
        const failedDot = document.createElement('span');
        failedDot.className = 'sidebar-item-failed';
        failedDot.dataset.tooltip = 'Task failed';
        failedDot.setAttribute('aria-label', 'Task failed');
        div.appendChild(failedDot);
      } else if (s.task_id && historyJustCompletedTaskIds.has(s.task_id)) {
        const completedDot = document.createElement('span');
        completedDot.className = 'sidebar-item-completed';
        completedDot.dataset.tooltip = 'Task completed';
        completedDot.setAttribute('aria-label', 'Task completed');
        div.appendChild(completedDot);
      }

      const textSpan = document.createElement('span');
      textSpan.className = 'sidebar-item-text';
      textSpan.textContent = itemText;
      div.appendChild(textSpan);

      const actions = document.createElement('div');
      actions.className = 'sidebar-item-actions';

      if (s.task_id) {
        const FAV_FILLED_SVG =
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
        const FAV_OUTLINE_SVG =
          '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"/></svg>';
        const favBtn = document.createElement('button');
        favBtn.type = 'button';
        favBtn.className = 'sidebar-item-favorite';
        const applyFavState = () => {
          if (s.is_favorite) {
            favBtn.classList.add('favorited');
            favBtn.dataset.tooltip = 'Unfavourite';
            favBtn.setAttribute('aria-label', 'Unfavourite task');
            favBtn.innerHTML = FAV_FILLED_SVG;
          } else {
            favBtn.classList.remove('favorited');
            favBtn.dataset.tooltip = 'Favourite';
            favBtn.setAttribute('aria-label', 'Favourite task');
            favBtn.innerHTML = FAV_OUTLINE_SVG;
          }
        };
        applyFavState();
        favBtn.addEventListener('click', e => {
          e.stopPropagation();
          e.preventDefault();
          const next = !s.is_favorite;
          s.is_favorite = next;
          applyFavState();
          div.dataset.favorite = next ? '1' : '0';
          applyHistoryFilterVisibility();
          api.setFavorite({taskId: s.task_id, isFavorite: next});
        });
        actions.appendChild(favBtn);

        const copyBtn = makeSidebarCopyButton(s.preview || itemText);
        actions.appendChild(copyBtn);
      }

      actions.appendChild(makeSidebarCollapseToggle(div, s));
      div.appendChild(actions);

      const info = document.createElement('div');
      info.className = 'running-item-info';

      const metrics = document.createElement('span');
      metrics.className = 'running-item-metrics';
      const tokens = Number(s.tokens || 0);
      const cost = Number(s.cost || 0);
      const steps = Number(s.steps || 0);
      const ts = Number(s.timestamp || 0);
      let when = '';
      if (ts > 0) {
        const d = new Date(ts * 1000);
        if (!isNaN(d.getTime())) {
          when =
            ' • ' +
            d.toLocaleString(undefined, {
              year: 'numeric',
              month: 'short',
              day: 'numeric',
              hour: '2-digit',
              minute: '2-digit',
            });
        }
      }
      const startTsMs = Number(s.startTs || 0);
      const endTsMs = Number(s.endTs || 0);
      let durMs = 0;
      if (startTsMs > 0) {
        if (endTsMs > startTsMs) {
          durMs = endTsMs - startTsMs;
        } else if (s.is_running || endTsMs === 0) {
          durMs = Date.now() - startTsMs;
        }
      }
      const dur = durMs > 0 ? ' • ' + formatDurationHms(durMs) : '';
      metrics.textContent =
        steps +
        ' steps • ' +
        tokens.toLocaleString() +
        ' tok • $' +
        cost.toFixed(4) +
        dur +
        when;
      info.appendChild(metrics);

      const workDir = typeof s.work_dir === 'string' ? s.work_dir : '';
      const modelName = typeof s.model === 'string' ? s.model : '';
      const parts = [];
      if (workDir) {
        parts.push(workDir);
      }
      if (modelName) {
        const wtLabel = s.is_worktree ? 'wt' : 'no-wt';
        const parLabel = s.is_parallel ? 'parallel' : 'sequential';
        const acLabel = s.auto_commit_mode ? 'auto-commit' : 'manual-commit';
        parts.push(modelName, wtLabel, parLabel, acLabel);
      }
      if (parts.length > 0) {
        const workspace = document.createElement('span');
        workspace.className = 'running-item-workspace';
        const text = parts.join(' • ');
        workspace.textContent = text;
        workspace.title = text;
        info.appendChild(workspace);
      }

      const chatId = typeof s.id === 'string' ? s.id : '';
      const taskIdRaw = s.task_id;
      const taskIdStr =
        taskIdRaw === undefined || taskIdRaw === null ? '' : String(taskIdRaw);
      const parentIdRaw = s.parent_task_id;
      const parentIdStr =
        parentIdRaw === undefined || parentIdRaw === null
          ? ''
          : String(parentIdRaw);
      const idSegments = [];
      if (chatId) {
        idSegments.push({text: 'chat ' + chatId, copy: chatId, kind: 'chat'});
      }
      if (taskIdStr) {
        idSegments.push({
          text: 'task ' + taskIdStr,
          copy: taskIdStr,
          kind: 'task',
        });
      }
      if (parentIdStr) {
        idSegments.push({text: 'parent ' + parentIdStr});
      }
      if (idSegments.length > 0) {
        const idsSpan = document.createElement('span');
        idsSpan.className = 'running-item-ids';
        idSegments.forEach((seg, i) => {
          if (i > 0) {
            idsSpan.appendChild(document.createTextNode(' • '));
          }
          idsSpan.appendChild(document.createTextNode(seg.text));
          if (seg.copy) {
            idsSpan.appendChild(makeIdCopyButton(seg.copy, seg.kind));
          }
        });
        idsSpan.title = idSegments.map(seg => seg.text).join(' • ');
        info.appendChild(idsSpan);
      }

      div.appendChild(info);

      div.addEventListener('click', () => {
        // The task text goes to the read-only task panel only.  #task-input
        // holds the user's own draft for the NEXT prompt and is never written.
        const taskText = s.preview || s.title || '';
        const existingChatTab = getTabByBackendChatId(s.id);
        if (existingChatTab) {
          switchToTab(existingChatTab.id);
          // The tab may be parked on a different task of the same chat.
          // Scroll the clicked task's region into view so the static
          // task panel names it; when its events are not spliced into
          // the transcript yet, replay the tab at that task instead.
          if (
            !existingChatTab.isContentTab &&
            !scrollChatToTask(s.task_id) &&
            s.task_id !== undefined &&
            s.task_id !== null &&
            s.task_id !== ''
          ) {
            api.resumeSession({
              id: s.id,
              taskId: s.task_id,
              tabId: existingChatTab.id,
            });
          }
        } else if (s.id && (s.has_events || s.is_running)) {
          // A running task is resumable even before its first event is
          // persisted: the server reattaches the live chat on replay.
          createNewTab();
          setTaskText(taskText);
          api.resumeSession({id: s.id, taskId: s.task_id, tabId: activeTabId});
        } else {
          // Nothing to resume, but the row still knows what the task was, so
          // show it read-only in the fresh tab.
          createNewTab();
          setTaskText(taskText);
          inp.focus();
        }
        closeSidebar();
      });
      historyList.appendChild(div);
    });

    historyOffset += sessions.length;
    if (sessions.length < 50) {
      historyHasMore = false;
    }
    applyHistoryFilterVisibility();
  }

  function autofillHistoryDateRange(range) {
    if (historyDateRangeUserSet || !range) return;
    const hfFrom = document.getElementById('hf-from');
    const hfTo = document.getElementById('hf-to');
    if (!hfFrom || !hfTo) return;
    if (range.min == null || range.max == null) {
      hfFrom.value = '';
      hfTo.value = '';
      applyHistoryFilterVisibility();
      return;
    }
    const isoOfSec = sec => {
      const n = Number(sec);
      if (!isFinite(n)) return '';
      const d = new Date(n * 1000);
      if (isNaN(d.getTime())) return '';
      const p = x => (x < 10 ? '0' + x : '' + x);
      return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate());
    };
    const fromIso = isoOfSec(range.min);
    const toIso = isoOfSec(range.max);
    if (!fromIso || !toIso) return;
    hfFrom.value = fromIso;
    hfTo.value = toIso;
    applyHistoryFilterVisibility();
  }

  function openCustomDatePicker(input, anchorBtn) {
    if (!input) return;
    const existing = document.getElementById('kiss-datepicker-pop');
    if (existing) {
      const sameInput = existing._kissInput === input;
      // Close through the picker's own teardown so the document-level
      // mousedown/keydown/resize listeners it registered are removed too;
      // a bare remove() would leave them behind on every toggle.
      if (typeof existing._kissClose === 'function') existing._kissClose();
      else existing.remove();
      if (sameInput) return;
    }
    const MONTHS = [
      'January',
      'February',
      'March',
      'April',
      'May',
      'June',
      'July',
      'August',
      'September',
      'October',
      'November',
      'December',
    ];
    const DAYS = ['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'];
    const pop = document.createElement('div');
    pop.id = 'kiss-datepicker-pop';
    pop.className = 'kiss-datepicker';
    pop._kissInput = input;
    let cursor = null;
    if (input.value) {
      const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(input.value);
      if (m) cursor = new Date(+m[1], +m[2] - 1, +m[3]);
    }
    if (!cursor || isNaN(cursor.getTime())) cursor = new Date();
    let viewYear = cursor.getFullYear();
    let viewMonth = cursor.getMonth();

    function pad2(n) {
      return n < 10 ? '0' + n : '' + n;
    }
    function isoOf(y, m, d) {
      return y + '-' + pad2(m + 1) + '-' + pad2(d);
    }

    function render() {
      const today = new Date();
      const selVal = input.value;
      let html = '';
      html += '<div class="dp-hdr">';
      html +=
        '<button type="button" class="dp-nav" data-nav="prev" ' +
        'aria-label="Previous month">&lsaquo;</button>';
      html +=
        '<span class="dp-title">' +
        esc(MONTHS[viewMonth] + ' ' + viewYear) +
        '</span>';
      html +=
        '<button type="button" class="dp-nav" data-nav="next" ' +
        'aria-label="Next month">&rsaquo;</button>';
      html += '</div>';
      html += '<div class="dp-grid">';
      for (let i = 0; i < DAYS.length; i++)
        html += '<div class="dp-dn">' + DAYS[i] + '</div>';
      const firstDow = new Date(viewYear, viewMonth, 1).getDay();
      const daysInMonth = new Date(viewYear, viewMonth + 1, 0).getDate();
      for (let i = 0; i < firstDow; i++)
        html += '<div class="dp-day dp-empty"></div>';
      for (let d = 1; d <= daysInMonth; d++) {
        const iso = isoOf(viewYear, viewMonth, d);
        const isToday =
          d === today.getDate() &&
          viewMonth === today.getMonth() &&
          viewYear === today.getFullYear();
        const isSel = iso === selVal;
        let cls = 'dp-day';
        if (isToday) cls += ' dp-today';
        if (isSel) cls += ' dp-sel';
        html +=
          '<button type="button" class="' +
          cls +
          '" data-date="' +
          iso +
          '">' +
          d +
          '</button>';
      }
      html += '</div>';
      html += '<div class="dp-foot">';
      html += '<button type="button" class="dp-clear">Clear</button>';
      html += '<button type="button" class="dp-today-btn">Today</button>';
      html += '</div>';
      pop.innerHTML = html;
    }

    function position() {
      const anchor = anchorBtn || input;
      const rect = anchor.getBoundingClientRect();
      const popH = pop.offsetHeight || 260;
      const popW = pop.offsetWidth || 220;
      let top = rect.bottom + 4;
      if (top + popH > window.innerHeight - 4)
        top = Math.max(4, rect.top - popH - 4);
      let left = rect.left;
      if (left + popW > window.innerWidth - 4)
        left = Math.max(4, window.innerWidth - popW - 4);
      pop.style.top = top + 'px';
      pop.style.left = left + 'px';
    }

    function commit(value) {
      input.value = value;
      input.dispatchEvent(new Event('change', {bubbles: true}));
      closePicker();
    }

    let pickerClosed = false;

    function closePicker() {
      pickerClosed = true;
      document.removeEventListener('mousedown', onDocClick, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', position);
      if (pop.parentNode) pop.parentNode.removeChild(pop);
    }
    pop._kissClose = closePicker;

    function onDocClick(e) {
      if (pop.contains(e.target)) return;
      if (anchorBtn && anchorBtn.contains(e.target)) return;
      closePicker();
    }

    function onKey(e) {
      if (e.key === 'Escape') closePicker();
    }

    pop.addEventListener('click', e => {
      const nav = e.target.closest('[data-nav]');
      if (nav) {
        if (nav.dataset.nav === 'prev') {
          viewMonth--;
          if (viewMonth < 0) {
            viewMonth = 11;
            viewYear--;
          }
        } else {
          viewMonth++;
          if (viewMonth > 11) {
            viewMonth = 0;
            viewYear++;
          }
        }
        render();
        return;
      }
      const day = e.target.closest('[data-date]');
      if (day) {
        commit(day.dataset.date);
        return;
      }
      if (e.target.closest('.dp-clear')) {
        commit('');
        return;
      }
      if (e.target.closest('.dp-today-btn')) {
        const t = new Date();
        commit(isoOf(t.getFullYear(), t.getMonth(), t.getDate()));
        return;
      }
    });

    render();
    document.body.appendChild(pop);
    position();
    setTimeout(() => {
      if (pickerClosed) return;
      document.addEventListener('mousedown', onDocClick, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('resize', position);
    }, 0);
  }

  function normalizeHistoryWorkDir(p) {
    if (typeof p !== 'string' || p === '') return '';
    const isWindowsPath = /^[A-Za-z]:[\\/]/.test(p) || p.startsWith('\\\\');
    let normalized = isWindowsPath ? p.replace(/\\/g, '/') : p;
    const minLength = /^[A-Za-z]:\/$/.test(normalized) ? 3 : 1;
    while (normalized.length > minLength && normalized.endsWith('/')) {
      normalized = normalized.slice(0, -1);
    }
    return isWindowsPath ? normalized.toLowerCase() : normalized;
  }

  function getHistoryFilterEls() {
    return {
      hfRunning: document.getElementById('hf-running'),
      hfErrors: document.getElementById('hf-errors'),
      hfCompleted: document.getElementById('hf-completed'),
      hfWorkspace: document.getElementById('hf-workspace'),
      hfFavorite: document.getElementById('hf-favorite'),
      hfFrom: document.getElementById('hf-from'),
      hfTo: document.getElementById('hf-to'),
    };
  }

  function applyHistoryFilterVisibility() {
    const {
      hfRunning,
      hfErrors,
      hfCompleted,
      hfWorkspace,
      hfFavorite,
      hfFrom,
      hfTo,
    } = getHistoryFilterEls();
    if (!hfRunning || !hfErrors || !hfCompleted) return;
    const showRunning = hfRunning.checked;
    const showErrors = hfErrors.checked;
    const showCompleted = hfCompleted.checked;
    const onlyFavorite = hfFavorite && hfFavorite.checked;
    const onlyWorkspace = hfWorkspace && hfWorkspace.checked;
    const normClientWorkDir = normalizeHistoryWorkDir(configWorkDir || '');
    // "/Users/me/proj" -> "/Users/me/proj/", but "/" stays "/", so that
    // subdirectory matching below never looks for a doubled separator.
    const clientWorkDirPrefix = normClientWorkDir.endsWith('/')
      ? normClientWorkDir
      : normClientWorkDir + '/';
    let fromTs = -Infinity;
    let toTs = Infinity;
    if (hfFrom && hfFrom.value) {
      const d = new Date(hfFrom.value + 'T00:00:00');
      if (!isNaN(d.getTime())) fromTs = d.getTime() / 1000;
    }
    if (hfTo && hfTo.value) {
      const d = new Date(hfTo.value + 'T23:59:59.999');
      if (!isNaN(d.getTime())) toTs = d.getTime() / 1000;
    }
    const rows = historyList.querySelectorAll('.sidebar-item');
    let visible = 0;
    rows.forEach(row => {
      const cat = row.dataset.category;
      const ts = Number(row.dataset.timestamp || 0);
      let catOk = false;
      if (cat === 'running') catOk = showRunning;
      else if (cat === 'errors') catOk = showErrors;
      else if (cat === 'completed') catOk = showCompleted;
      const dateOk = ts >= fromTs && ts <= toTs;
      const favOk = !onlyFavorite || row.dataset.favorite === '1';
      const rowWorkDir = normalizeHistoryWorkDir(row.dataset.workDir || '');
      // A task that ran in a git worktree (".kiss-worktrees/kiss_wt-...")
      // or any other subdirectory still belongs to this workspace.
      const wsOk =
        !onlyWorkspace ||
        cat === 'running' ||
        rowWorkDir === '' ||
        normClientWorkDir === '' ||
        rowWorkDir === normClientWorkDir ||
        rowWorkDir.startsWith(clientWorkDirPrefix);
      if (catOk && dateOk && favOk && wsOk) {
        row.style.display = '';
        visible++;
      } else {
        row.style.display = 'none';
      }
    });
    let placeholder = historyList.querySelector('.sidebar-empty-filter');
    if (rows.length > 0 && visible === 0) {
      if (!placeholder) {
        placeholder = document.createElement('div');
        placeholder.className = 'sidebar-empty sidebar-empty-filter';
        placeholder.textContent = 'No tasks match the filter';
        historyList.appendChild(placeholder);
      }
    } else if (placeholder) {
      placeholder.remove();
    }
    const hfDateClear = document.getElementById('hf-date-clear');
    if (hfDateClear) {
      const hasDate = !!((hfFrom && hfFrom.value) || (hfTo && hfTo.value));
      hfDateClear.style.display = hasDate ? '' : 'none';
    }
  }

  /**
   * Flush the settings form if there is anything to flush.
   *
   * A form the reply never reached is not empty of intent: whatever the
   * user typed into it must still be saved, so those fields alone are
   * sent. A form nobody touched and nobody populated has nothing to say
   * and stays silent.
   */
  function saveSettingsIfPopulated() {
    const edited = settingsEditedFields;
    if (!configFormPopulated && edited.size === 0) return;
    const data = configFormPopulated
      ? collectConfigForm()
      : collectConfigForm(edited);
    api.saveConfig({...data});
    if (
      document.body.classList.contains('remote-chat') &&
      typeof data.config.work_dir === 'string' &&
      data.config.work_dir
    ) {
      api.setWorkDir({workDir: data.config.work_dir});
    }
  }

  function closeSidebar(force) {
    if (force !== true && document.body.classList.contains('remote-desktop')) {
      sidebarOverlay.classList.remove('open');
      return;
    }
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
  }

  function setPanelOpen(panel, overlay, open) {
    if (panel) panel.classList.toggle('open', open);
    if (overlay) overlay.classList.toggle('open', open);
  }

  function openSettingsPanel() {
    if (!settingsPanel) return;
    setPanelOpen(settingsPanel, settingsOverlay, true);
    configFormPopulated = false;
    settingsEditedFields.clear();
    api.getConfig();
  }

  function closeSettingsPanel() {
    saveSettingsIfPopulated();
    settingsEditedFields.clear();
    setPanelOpen(settingsPanel, settingsOverlay, false);
  }

  function openFrequentPanel() {
    if (!frequentPanel) return;
    setPanelOpen(frequentPanel, frequentOverlay, true);
    api.getFrequentTasks({limit: 50});
  }

  function closeFrequentPanel() {
    setPanelOpen(frequentPanel, frequentOverlay, false);
  }

  function openTricksPanel() {
    if (!tricksPanel) return;
    setPanelOpen(tricksPanel, tricksOverlay, true);
    renderTricks(window.__TRICKS__ || []);
  }

  function closeTricksPanel() {
    setPanelOpen(tricksPanel, tricksOverlay, false);
  }

  function renderTricks(tricks) {
    if (!tricksList) return;
    if (!tricks || tricks.length === 0) {
      tricksList.innerHTML =
        '<div class="sidebar-empty">No tricks available</div>';
      return;
    }
    tricksList.innerHTML = '';
    tricks.forEach(text => {
      const div = document.createElement('div');
      div.className = 'sidebar-item tricks-item';
      div.dataset.tooltip = text;
      const textSpan = document.createElement('span');
      textSpan.className = 'sidebar-item-text';
      textSpan.textContent = text;
      div.appendChild(textSpan);
      div.addEventListener('click', () => {
        const current = inp.value;
        const start =
          typeof inp.selectionStart === 'number'
            ? inp.selectionStart
            : current.length;
        const end =
          typeof inp.selectionEnd === 'number'
            ? inp.selectionEnd
            : current.length;
        const before = current.slice(0, start);
        const after = current.slice(end);
        const leadPad = before.length === 0 || /\s$/.test(before) ? '' : ' ';
        const trailPad = after.length === 0 || /^\s/.test(after) ? '' : ' ';
        const injected = leadPad + text + trailPad;
        inp.value = before + injected + after;
        const caret = start + injected.length;
        syncClearBtn();
        inp.style.height = 'auto';
        inp.style.height = inp.scrollHeight + 'px';
        inp.focus();
        try {
          inp.setSelectionRange(caret, caret);
        } catch (_e) {}
        closeTricksPanel();
      });
      tricksList.appendChild(div);
    });
  }

  function renderFrequentTasks(tasks) {
    if (!frequentList) return;
    if (!tasks || tasks.length === 0) {
      frequentList.innerHTML = '<div class="sidebar-empty">No tasks yet</div>';
      return;
    }
    frequentList.innerHTML = '';
    tasks.forEach(t => {
      const div = document.createElement('div');
      div.className = 'sidebar-item frequent-item';
      const text = String(t.task || '');
      div.dataset.tooltip = text;
      div.style.backgroundColor = chatIdBgColor(text);
      div.style.color = '#1a1a1a';

      const textSpan = document.createElement('span');
      textSpan.className = 'sidebar-item-text';
      textSpan.textContent = text;
      div.appendChild(textSpan);

      const cnt = document.createElement('span');
      cnt.className = 'frequent-item-count';
      cnt.textContent = String(t.count);
      div.appendChild(cnt);

      const copyBtn = makeSidebarCopyButton(text);
      div.appendChild(copyBtn);

      const {delBtn, confirmWrap} = makeSidebarDeleteConfirm({
        ariaLabel: 'Delete frequent task',
        onShowConfirm: () => {
          cnt.style.display = 'none';
        },
        onCancel: () => {
          cnt.style.display = '';
        },
        onConfirm: () => {
          api.deleteFrequentTask({task: text});
          div.remove();
        },
      });

      div.appendChild(delBtn);
      div.appendChild(confirmWrap);

      div.addEventListener('click', () => {
        inp.value = text;
        syncClearBtn();
        inp.style.height = 'auto';
        inp.style.height = inp.scrollHeight + 'px';
        inp.focus();
        closeFrequentPanel();
      });
      frequentList.appendChild(div);
    });
  }
  function setupPasswordToggle(toggleId, inputId, secretName) {
    const btn = document.getElementById(toggleId);
    const inp = document.getElementById(inputId);
    if (!btn || !inp) return;
    const noun = secretName || 'password';
    btn.addEventListener('click', () => {
      const showing = inp.type === 'text';
      inp.type = showing ? 'password' : 'text';
      const eye = btn.querySelector('.icon-eye');
      const eyeOff = btn.querySelector('.icon-eye-off');
      if (eye) eye.style.display = showing ? '' : 'none';
      if (eyeOff) eyeOff.style.display = showing ? 'none' : '';
      btn.setAttribute('aria-pressed', showing ? 'false' : 'true');
      const lbl = (showing ? 'Show ' : 'Hide ') + noun;
      btn.setAttribute('aria-label', lbl);
      btn.setAttribute('title', lbl);
    });
  }

  function setupSecretInput(inputId) {
    const inp = document.getElementById(inputId);
    const proto = document.getElementById('cfg-remote-password-toggle');
    if (!inp || !proto) return;
    inp.type = 'password';
    inp.setAttribute('autocomplete', 'off');
    const wrap = document.createElement('div');
    wrap.className = 'config-password-wrap';
    inp.parentNode.insertBefore(wrap, inp);
    wrap.appendChild(inp);
    const btn = proto.cloneNode(true);
    btn.id = inputId + '-toggle';
    btn.setAttribute('aria-pressed', 'false');
    btn.setAttribute('aria-label', 'Show API key');
    btn.setAttribute('title', 'Show API key');
    const eye = btn.querySelector('.icon-eye');
    const eyeOff = btn.querySelector('.icon-eye-off');
    if (eye) eye.style.display = '';
    if (eyeOff) eyeOff.style.display = 'none';
    wrap.appendChild(btn);
    setupPasswordToggle(btn.id, inputId, 'API key');
  }

  let configFormPopulated = false;
  // Ids of the settings fields the user has edited since the panel was
  // opened.  `configData` is not a one-shot reply: the host re-pushes it
  // from a 2-second poll of ~/.kiss/config.json and on every daemon
  // reconnect, so a field that is being typed into is never repainted --
  // and a panel closed before any reply arrived still saves what was
  // touched instead of dropping it.
  const settingsEditedFields = new Set();

  /**
   * Remember that a settings field now holds the user's own value.
   *
   * @param {string} id The element id of the edited field.
   */
  function markSettingsFieldEdited(id) {
    if (id) settingsEditedFields.add(id);
  }

  function populateConfigForm(cfg, apiKeys) {
    const el = id => document.getElementById(id);
    const setValue = (id, value) => {
      const node = el(id);
      if (!node || settingsEditedFields.has(id)) return;
      node.value = value;
    };
    const setChecked = (node, checked) => {
      if (!node || settingsEditedFields.has(node.id)) return;
      node.checked = checked;
    };
    const prevConfigWorkDir = configWorkDir;
    configWorkDir = cfg.work_dir || '';
    if (prevConfigWorkDir !== configWorkDir) {
      try {
        applyHistoryFilterVisibility();
      } catch (_e) {}
    }
    const wdInp = el('cfg-work-dir');
    if (wdInp) {
      setValue('cfg-work-dir', cfg.work_dir || '');
      if (!document.body.classList.contains('remote-chat')) {
        wdInp.readOnly = true;
        wdInp.title = 'Set by the workspace folder open in this window';
      } else {
        let pinned = '';
        try {
          // eslint-disable-next-line no-undef -- sessionStorage is a browser global
          pinned = sessionStorage.getItem('sorcar-work-dir') || '';
        } catch (_e) {}
        if (pinned) {
          setValue('cfg-work-dir', pinned);
          configWorkDir = pinned;
        } else if (cfg.work_dir) {
          api.setWorkDir({workDir: cfg.work_dir});
        }
      }
    }
    // The budget default belongs to Python (`config.DEFAULT_MAX_BUDGET`,
    // read by `vscode_config.DEFAULTS`), and `load_config()` seeds every
    // reply from it, so an effective value is always in `configData`.
    // A copy of the number here could only ever drift away from it, so
    // the box shows what the daemon sent and nothing when it sent none.
    if (cfg.max_budget != null) setValue('cfg-max-budget', cfg.max_budget);
    // Initialize the run toggles from the persisted config instead of
    // leaving whatever hardcoded `checked` state chat.html shipped with,
    // so a fresh session (VS Code webview or remote web client) reflects
    // the user's saved preference.  Missing keys default to true, the
    // same defaults as vscode_config.DEFAULTS.
    setChecked(autocommitToggleBtn, cfg.auto_commit_mode !== false);
    setChecked(worktreeToggleBtn, cfg.is_worktree !== false);
    setValue('cfg-custom-endpoint', cfg.custom_endpoint || '');
    setValue('cfg-custom-api-key', cfg.custom_api_key || '');
    setValue('cfg-custom-headers', cfg.custom_headers || '');
    setValue('cfg-remote-password', cfg.remote_password || '');
    // The welcome screen's password box mirrors the settings one, so it
    // follows the same edited mark.
    if (!settingsEditedFields.has('cfg-remote-password')) {
      setValue('welcome-cfg-remote-password', cfg.remote_password || '');
    }
    configFormPopulated = true;
    const keyIds = [
      'GEMINI_API_KEY',
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'TOGETHER_API_KEY',
      'OPENROUTER_API_KEY',
      'ZAI_API_KEY',
      'MOONSHOT_API_KEY',
    ];
    keyIds.forEach(k => {
      setValue('cfg-key-' + k, (apiKeys && apiKeys[k]) || '');
    });
  }
  /**
   * Read the settings form into a `saveConfig` payload.
   *
   * @param {Set<string>} [onlyIds] When given, only these field ids are
   *   read. The daemon MERGES what it is sent (see
   *   ``vscode_config.save_config``), so a partial payload updates just
   *   those fields and leaves the rest of config.json alone -- which is
   *   what lets a panel closed before its `configData` reply arrived
   *   save the edit instead of flushing a form full of blanks over the
   *   stored settings.
   * @returns {{config: object, apiKeys: object}} The payload.
   */
  function collectConfigForm(onlyIds) {
    const el = id => document.getElementById(id);
    const want = id => !onlyIds || onlyIds.has(id);
    const cfg = {};
    if (want('cfg-max-budget')) {
      // An unparseable box -- cleared by the user, or never filled
      // because no `configData` arrived -- means the client has no
      // budget to report, NOT that it should supply one: the daemon
      // merges this payload, so leaving the key out keeps whatever is
      // stored. Writing a locally invented default here is how a user
      // on a 250 budget silently ended up back on 100.
      const budget = parseFloat(el('cfg-max-budget').value);
      if (Number.isFinite(budget)) cfg.max_budget = budget;
    }
    if (want('cfg-auto-commit')) {
      cfg.auto_commit_mode = !!(
        autocommitToggleBtn && autocommitToggleBtn.checked
      );
    }
    if (want('cfg-use-worktree')) {
      cfg.is_worktree = !!(worktreeToggleBtn && worktreeToggleBtn.checked);
    }
    if (want('cfg-custom-endpoint')) {
      cfg.custom_endpoint = el('cfg-custom-endpoint').value.trim();
    }
    if (want('cfg-custom-api-key')) {
      cfg.custom_api_key = el('cfg-custom-api-key').value.trim();
    }
    if (want('cfg-custom-headers')) {
      cfg.custom_headers = el('cfg-custom-headers').value.trim();
    }
    if (want('cfg-remote-password')) {
      cfg.remote_password = el('cfg-remote-password').value.trim();
    }
    const wdInp = el('cfg-work-dir');
    if (wdInp && !wdInp.readOnly && want('cfg-work-dir')) {
      cfg.work_dir = wdInp.value.trim();
    }
    const apiKeys = {};
    const keyIds = [
      'GEMINI_API_KEY',
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'TOGETHER_API_KEY',
      'OPENROUTER_API_KEY',
      'ZAI_API_KEY',
      'MOONSHOT_API_KEY',
    ];
    keyIds.forEach(k => {
      if (!want('cfg-key-' + k)) return;
      const v = el('cfg-key-' + k).value.trim();
      if (v) apiKeys[k] = v;
    });
    return {config: cfg, apiKeys};
  }

  function getAtCtx() {
    const val = inp.value,
      pos = inp.selectionStart || 0;
    const before = val.substring(0, pos);
    const m = before.match(/@([^\s]*)$/);
    return m ? {start: before.length - m[0].length, query: m[1]} : null;
  }

  function checkAutocomplete() {
    const atCtx = getAtCtx();
    if (atCtx) {
      api.getFiles({
        prefix: atCtx.query,
        workDir: workDirForTab(activeTabId),
        // Stamp the owning tab so a reply can never render over a sibling
        // tab whose input happens to hold the same half-typed mention.
        tabId: activeTabId || undefined,
      });
    } else {
      hideAC();
    }
  }

  const _acSvg = {
    file: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    star: '<svg viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
    bolt: '<svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    spark:
      '<svg viewBox="0 0 24 24"><path d="M12 2l1.5 5L19 8.5 13.5 10 12 15 10.5 10 5 8.5 10.5 7 12 2z"/></svg>',
    code: '<svg viewBox="0 0 24 24"><path d="M8 4H6a2 2 0 00-2 2v4a2 2 0 01-2 2 2 2 0 012 2v4a2 2 0 002 2h2M16 4h2a2 2 0 012 2v4a2 2 0 002 2 2 2 0 00-2 2v4a2 2 0 01-2 2h-2"/></svg>',
  };
  function _acIcon(type) {
    if (type === 'frequent') return _acSvg.star;
    if (type === 'task') return _acSvg.bolt;
    if (type === 'trick') return _acSvg.spark;
    if (type === 'identifier') return _acSvg.code;
    return _acSvg.file;
  }
  function hlMatch(text, query) {
    if (!query) return esc(text);
    const idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) return esc(text);
    return (
      esc(text.substring(0, idx)) +
      '<strong class="ac-hl">' +
      esc(text.substring(idx, idx + query.length)) +
      '</strong>' +
      esc(text.substring(idx + query.length))
    );
  }
  function _acPathHtml(text) {
    const last = text.lastIndexOf('/');
    if (last < 0 || last === text.length - 1) return esc(text);
    const dir = text.substring(0, last + 1);
    const fname = text.substring(last + 1);
    return (
      '<span class="ac-dir">' +
      esc(dir) +
      '</span>' +
      '<span class="ac-fname">' +
      esc(fname) +
      '</span>'
    );
  }
  function hideAC() {
    autocomplete.style.display = 'none';
    acIdx = -1;
  }

  function renderAcDropdown(data, order, labels, itemHtml, onAccept) {
    autocomplete.innerHTML = '';
    acIdx = -1;
    const groups = {};
    data.forEach(item => {
      const t = item.type;
      if (!groups[t]) groups[t] = [];
      groups[t].push(item);
    });
    let isFirst = true;
    order.forEach(type => {
      const g = groups[type];
      if (!g) return;
      const lbl = labels[type] || type;
      const hdr = mkEl('div', 'ac-section');
      hdr.textContent = lbl;
      autocomplete.appendChild(hdr);
      g.forEach(item => {
        const d = mkEl('div', 'ac-item');
        d.dataset.text = item.text;
        d.innerHTML =
          '<span class="ac-icon">' +
          _acIcon(item.type) +
          '</span>' +
          '<span class="ac-text">' +
          itemHtml(item) +
          '</span>';
        if (isFirst) {
          d.innerHTML += '<span class="ac-hint">tab</span>';
          isFirst = false;
        }
        d.addEventListener('click', () => {
          onAccept(item.text);
        });
        autocomplete.appendChild(d);
      });
    });
    const footer = mkEl('div', 'ac-footer');
    footer.innerHTML =
      '<span><kbd>\u2191\u2193</kbd> navigate</span>' +
      '<span><kbd>Tab</kbd> accept</span>' +
      '<span><kbd>Esc</kbd> dismiss</span>';
    autocomplete.appendChild(footer);
    autocomplete.style.display = 'block';
    acIdx = 0;
    updateSel(autocomplete.querySelectorAll('.ac-item'), acIdx);
  }

  function renderAutocomplete(data) {
    if (!data || !data.length) {
      hideAC();
      return;
    }
    const atMatch = getAtCtx();
    const searchQ = atMatch ? atMatch.query : '';
    renderAcDropdown(
      data,
      ['frequent', 'file'],
      {frequent: 'Frequent', file: 'Files'},
      item =>
        searchQ && searchQ.length > 0
          ? hlMatch(item.text, searchQ)
          : _acPathHtml(item.text),
      insertAtMention,
    );
  }

  function insertAtMention(file) {
    const atCtx = getAtCtx();
    if (atCtx) {
      const before = inp.value.substring(0, atCtx.start);
      const after = inp.value.substring(inp.selectionStart || inp.value.length);
      const sep = /^\s/.test(after) ? '' : ' ';
      const mention = './' + file;
      inp.value = before + mention + sep + after;
      syncClearBtn();
      const np = before.length + mention.length + sep.length;
      inp.setSelectionRange(np, np);
      api.recordFileUsage({path: file, workDir: workDirForTab(activeTabId)});
    }
    hideAC();
    inp.focus();
  }

  function acceptCompletion(full) {
    const cur = inp.value;
    let overlap = Math.min(cur.length, full.length);
    while (overlap > 0 && !cur.endsWith(full.slice(0, overlap))) {
      overlap -= 1;
    }
    inp.value = cur.slice(0, cur.length - overlap) + full;
    if (/\S$/.test(inp.value)) inp.value += ' ';
    clearGhost();
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
    const np = inp.value.length;
    inp.setSelectionRange(np, np);
    hideAC();
    inp.focus();
  }

  function renderCompletions(data) {
    if (getAtCtx()) {
      return;
    }
    if (!data || !data.length) {
      hideAC();
      return;
    }
    if (isRunning) {
      hideAC();
      return;
    }
    if (!inp.value) {
      hideAC();
      return;
    }
    if (inp.selectionStart < inp.value.length) {
      hideAC();
      return;
    }
    renderAcDropdown(
      data,
      ['task', 'frequent', 'trick', 'identifier'],
      {
        task: 'History',
        frequent: 'Frequent',
        trick: 'Suggestions',
        identifier: 'From editor',
      },
      item => hlMatch(item.text, inp.value),
      acceptCompletion,
    );
  }

  // Everything above lives inside this IIFE, so the end-to-end webview tests
  // (jsdom and Playwright) have no other way to drive a real conversation.
  // These few entry points are the whole surface they need: which tab is on
  // screen, open another one, feed it a backend event, end the launch, and
  // get the welcome screen out of the way.
  window._testApi = {
    getActiveTabId: function () {
      return activeTabId;
    },
    createNewTab: createNewTab,
    processEvent: processOutputEvent,
    // Stands in for the first tap or keystroke: after this the window is no
    // longer launching, so no backend event may switch tabs on its own.
    endLaunch: closeLaunchSwitch,
    hideWelcome: function () {
      if (welcome) {
        welcome.style.display = 'none';
        refreshWelcomeLayout();
      }
    },
    // How much file-link work is still being retained. Both numbers must
    // come back down; see forgetPendingFileLinks().
    pendingFileLinkCounts: function () {
      return {
        spans: _pendingFileLinkSpans.size,
        checks: _pendingPathChecks.size,
      };
    },
  };

  init();
})();

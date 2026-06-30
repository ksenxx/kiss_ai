// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * KISS Sorcar Webview JavaScript
 * Uses the same event protocol and rendering as the browser-based Sorcar.
 */

(function () {
  // @ts-ignore - vscode is injected by the webview
  const vscode = acquireVsCodeApi();

  /** Format a number with thousand separators (e.g. 12345 → "12,345"). */
  function fmtN(n) {
    return Number(n).toLocaleString('en-US');
  }

  /**
   * Format an elapsed duration in milliseconds for the per-panel time
   * footer: ``"850ms"`` below one second, ``"3.4s"`` below one minute,
   * ``"1m 12.0s"`` otherwise.
   */
  function fmtElapsedMs(ms) {
    const n = Math.max(0, Math.round(Number(ms) || 0));
    if (n < 1000) return n + 'ms';
    const s = n / 1000;
    if (s < 60) return s.toFixed(1) + 's';
    const m = Math.floor(s / 60);
    const sec = (s - m * 60).toFixed(1);
    return m + 'm ' + sec + 's';
  }

  // Set of in-progress panel elements currently being stamped with a
  // live ``.panel-time`` footer.  Populated by ``stampPanelStart`` and
  // drained by ``finalizePanelTime``.  A single 1-second interval
  // (``_activePanelTickIv``) walks this set and re-renders each
  // panel's footer so the user sees the elapsed time update every
  // second while the panel is still active.  Without this loop the
  // footer would only appear/refresh when the panel closes — which is
  // the bug we are fixing.
  const _activePanels = new Set();
  let _activePanelTickIv = null;

  /**
   * Stamp a panel element with its creation time (``data-start-ms``).
   *
   * No-op if the panel already carries a start stamp or if we are
   * currently replaying persisted events (``_deferHighlight`` is set
   * only during ``replayEventsInto``).  Replayed events arrive
   * back-to-back so per-panel wall-clock measurements would be
   * meaningless; we deliberately skip stamping then.
   *
   * In addition to stamping, the panel is registered with the live
   * 1-second ticker so its ``.panel-time`` footer starts rendering
   * immediately and refreshes every second while the panel is active.
   */
  function stampPanelStart(el) {
    if (!el || _deferHighlight) return;
    if (el.dataset.startMs) return;
    el.dataset.startMs = String(Date.now());
    _activePanels.add(el);
    _renderPanelTime(el);
    _startActivePanelTick();
  }

  /**
   * Render (create or refresh) the ``.panel-time`` footer for ``el``
   * using its ``data-start-ms`` stamp.  Shared by the live 1-second
   * ticker and by ``finalizePanelTime`` so the in-progress footer and
   * the final footer use identical anchoring/formatting logic.
   *
   * No-op if the panel was never stamped (e.g. replayed events) so the
   * historical view stays clean.
   */
  function _renderPanelTime(el) {
    if (!el) return;
    const startMs = Number(el.dataset.startMs || 0);
    if (!startMs) return;
    const ms = Date.now() - startMs;
    let footer = null;
    // Find an existing direct-child footer (avoid matching footers
    // inside nested panels).
    for (let i = el.children.length - 1; i >= 0; i--) {
      const c = el.children[i];
      if (c.classList && c.classList.contains('panel-time')) {
        footer = c;
        break;
      }
    }
    if (!footer) {
      footer = document.createElement('div');
      footer.className = 'panel-time';
      el.appendChild(footer);
    } else if (footer !== el.lastElementChild) {
      // Keep the footer anchored as the LAST child so it always
      // renders visually at the bottom of the panel, even when later
      // content (e.g. a tool_result bash-panel) is appended after the
      // initial finalisation.
      el.appendChild(footer);
    }
    footer.textContent = fmtElapsedMs(ms);
  }

  /**
   * Start the shared 1-second interval that re-renders the
   * ``.panel-time`` footer of every panel still in ``_activePanels``.
   * Idempotent: a no-op if the interval is already running or if no
   * panels are active.  Each tick prunes panels that are no longer
   * connected to the DOM so detached panels don't keep the loop alive
   * forever, and stops the interval once the active set is empty.
   */
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

  /**
   * Append (or refresh) the final "time spent" footer as the LAST
   * child of the given panel, then deregister the panel from the live
   * ticker so its footer freezes at the closing time.  Reads
   * ``data-start-ms`` set by ``stampPanelStart``.
   *
   * No-op if the panel was never stamped (e.g. replayed events), so
   * the historical view stays clean.
   */
  function finalizePanelTime(el) {
    if (!el) return;
    const startMs = Number(el.dataset.startMs || 0);
    if (!startMs) return;
    _renderPanelTime(el);
    _activePanels.delete(el);
    if (_activePanels.size === 0 && _activePanelTickIv) {
      clearInterval(_activePanelTickIv);
      _activePanelTickIv = null;
    }
  }

  /**
   * Sanitize an HTML string before assigning to innerHTML.
   *
   * Strips dangerous tags (script/iframe/object/embed/form/meta/link/style/
   * base), all event-handler attributes (onclick, onerror, ...) and
   * javascript:/data:/vbscript: URLs in href/src/action.  Used to wrap every
   * marked.parse() result that flows into innerHTML so that agent-supplied
   * markdown can never inject script/iframe/form via the webview.
   */
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
        for (const attr of Array.from(el.attributes)) {
          const name = attr.name.toLowerCase();
          // Strip every event-handler attribute (onclick, onerror, ...).
          if (name.startsWith('on')) {
            el.removeAttribute(attr.name);
            continue;
          }
          if (URL_ATTRS.has(name)) {
            const v = (attr.value || '').trim();
            if (/^(javascript|data|vbscript):/i.test(v)) {
              el.removeAttribute(attr.name);
            }
          }
        }
      }
    };
    walk(t.content);
    return t.innerHTML;
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
      vscode.postMessage({type: 'notificationAction', id: id, action: action});
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
    // Expose `sticky` on the DOM so downstream tests (and any future
    // a11y tooling) can verify that a notification will not auto-
    // dismiss — the existing `scheduleNotificationDismiss` already
    // honours it for the timer, but the flag was otherwise invisible
    // from the rendered DOM.
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
        // Each action is either a plain string label OR an object of
        // shape ``{label, svg?, ariaLabel?, onClick?}``.  The object
        // form is used by in-webview callers (e.g. the permanent
        // "update available" notification) that want to render an
        // inline ``<svg>`` icon inside the button and/or run a local
        // click handler instead of round-tripping through the
        // extension via ``notificationAction``.
        const isObj =
          action && typeof action === 'object' && !Array.isArray(action);
        const label = isObj ? String(action.label || '') : String(action);
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'kiss-notification-action';
        if (isObj && action.svg) {
          // Parse + sanitise the SVG XML in an off-DOM template, then
          // adopt the resulting SVG element.  This guarantees the
          // browser parses it as SVG (correct namespace) and that
          // ``kissSanitize`` strips any ``<script>``/``on*``/javascript:
          // payload that may have slipped in.
          const cleaned = kissSanitize(String(action.svg));
          const parser = new window.DOMParser();
          const doc = parser.parseFromString(cleaned, 'image/svg+xml');
          const svgEl = doc.documentElement;
          // DOMParser returns a ``<parsererror>`` element on invalid
          // input — only adopt real SVG roots so we never inject
          // arbitrary error HTML into the button.
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
            } catch (_err) {
              // Swallow handler errors so the notification still
              // closes — the click already dismissed it from the
              // user's point of view.
            }
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

  // State — isRunning mirrors the active tab's tab.isRunning for UI controls
  let isRunning = false;
  let selectedModel = '';
  let allModels = [];
  let modelDDIdx = -1;
  let attachments = [];
  let _scrollLock = false;
  let _noScroll = false;
  // When true (only during bulk replay in replayEventsInto), hlBlock defers
  // syntax highlighting: code blocks are tagged `needs-hl` instead of being
  // highlighted, so panels collapsed by collapseAllExceptResult() are only
  // highlighted lazily when the user expands them (see highlightPending).
  let _deferHighlight = false;
  let scrollRaf = 0;
  let acIdx = -1;

  // History cycling state
  let histCache = [];
  let histIdx = -1;

  // Ghost text state
  let ghostTimer = null;
  let currentGhost = '';

  // Per-tab ask-user modal routing: each tab owns its own pending question
  // string and askQuestionEl / askInputEl / askSubmitEl DOM nodes (see
  // makeTab).  The shared #ask-user-slot hosts the active tab's triplet;
  // switching tabs detaches and re-attaches so each tab's half-typed answer
  // is preserved.  Because the modal blocks the tab's agent, at most one
  // ask-user request is pending per tab at any time — no queue is needed.

  // Demo mode state
  let demoMode = false;
  let _demoActive = false;
  let allHistSessions = [];

  // Infinite scroll state for history sidebar
  let historyOffset = 0;
  let historyLoading = false;
  let historyHasMore = true;
  let historyGeneration = 0;
  // Session-scoped sets tracking the live running→completed
  // transition for the History panel's status dot.  The invariant:
  //   * A running row renders the pulsing green dot.
  //   * On completion, the dot becomes SOLID green and STAYS that
  //     way for the rest of the page session, even across
  //     ``refreshHistory()`` reloads.
  //   * A completed row that we never saw running in this session
  //     (e.g. on a fresh history load) renders NO dot.
  // ``historyLastRunningTaskIds`` is the snapshot of which task_ids
  // were rendered as ``is_running:true`` on the previous
  // ``renderHistory`` call.  When the next render drops a task_id
  // (it transitioned to is_running:false / failed:false), we move
  // it into ``historyJustCompletedTaskIds``, which sticks until the
  // page is reloaded.
  const historyLastRunningTaskIds = new Set();
  const historyJustCompletedTaskIds = new Set();

  // Adjacent task scroll state (Cursor-style chat thread navigation)
  // Tab.id is a frontend-only UUID string; chat_id is an int assigned by the DB.
  let currentTaskName = ''; // the originally loaded task
  // DB row ids identifying the topmost / bottommost tasks currently
  // rendered in #output.  These are the values sent over the wire to
  // the backend's getAdjacentTask handler — using the row id (rather
  // than the task description string) ensures that duplicate task
  // texts within a chat are navigated unambiguously.  ``null`` means
  // "no id known yet" (e.g. fresh tab before any task ran).
  let currentTaskId = null; // task_id of the originally loaded task
  let oldestLoadedTaskId = null; // task_id of the topmost loaded task
  let newestLoadedTaskId = null; // task_id of the bottommost loaded task
  let adjacentLoading = false;
  let noPrevTask = false; // true when server says no prev exists
  let noNextTask = false; // true when server says no next exists
  let overscrollAccum = 0;
  let overscrollDir = '';
  let overscrollTimer = null;
  const OVERSCROLL_THRESHOLD = 150; // pixels of accumulated overscroll to trigger load
  // Per-task metrics for adjacent scrolling: when the user scrolls between
  // the current task and adjacent tasks, the header tokens/cost/steps should
  // reflect the currently visible task.  currentTaskMetrics stores the main
  // task's metrics; adjacent containers store theirs in dataset attributes.
  let currentTaskMetrics = {tokens: '', budget: '', steps: ''};

  // --- Chat tabs state ---
  /** Generate a UUID v4 string for tab identification. */
  function genTabId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID)
      return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = (Math.random() * 16) | 0;
      return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
  }

  let tabs = []; // array of tab objects (see makeTab for fields)
  let activeTabId = '';

  // Configured work directory reported by the backend's ``getConfig``
  // reply (``configData`` → ``populateConfigForm``).  Used as the
  // fallback repo path for ``workDirForTab`` when a tab has not yet
  // learned its own ``workDir`` from a background-task event, so
  // commands like the settings-panel "git commit" act on the
  // configured repo rather than the daemon-wide ``self.work_dir``.
  let configWorkDir = '';

  function makeTab(title) {
    const _id = genTabId();
    return {
      id: _id,
      title: title || 'new chat',
      backendChatId: '',
      // task_id (DB row id) of the task currently displayed as the
      // header / current task in this tab.  null when unknown (fresh
      // tab before any chat is loaded).  Used by the 'taskDeleted'
      // handler to close tabs whose current task was deleted.
      currentTaskId: null,
      isRunning: false,
      outputFragment: null,
      taskPanelHTML: '',
      taskPanelVisible: false,
      panelsExpandedMap: {},
      statusTextContent: 'Ready',
      statusTextColor: 'var(--green)',
      statusTokensText: '',
      statusBudgetText: '',
      statusStepsText: '',
      welcomeVisible: true,
      selectedModel: selectedModel,
      attachments: [],
      inputValue: '',
      isMerging: false,
      worktreeBarEl: null,
      autocommitBarEl: null,
      mergeToolbarEl: null,
      t0: null,
      // Agent-recorded end timestamp (ms since epoch) of this tab's
      // last task; 0 while running / unknown.  Together with ``t0``
      // it lets every tab render "Done (Xm Ys)" as endTs - t0 no
      // matter when the user switches to the tab.
      endTs: 0,
      workDir: '',
      streamState: null,
      streamLlmPanel: null,
      streamLlmPanelState: null,
      streamLastToolName: '',
      streamPendingPanel: false,
      lastTaskFailed: false,
      hasRunTask: false,
      // Ask-user modal: the currently-pending question for this tab (or
      // null if none) and per-tab DOM nodes.  Only one ask can be pending
      // at a time because the agent blocks on the user answer.
      askPendingQuestion: null,
      askQuestionEl: null,
      askInputEl: null,
      askSubmitEl: null,
    };
  }

  /** Find a tab object by its id (null when absent). */
  function getTab(id) {
    return tabs.find(t => t.id === id) || null;
  }

  /** Find the local tab that already displays a backend chat id. */
  function getTabByBackendChatId(chatId) {
    if (chatId === undefined || chatId === null || chatId === '') return null;
    const key = String(chatId);
    return tabs.find(t => String(t.backendChatId || '') === key) || null;
  }

  /**
   * Place *subTab* immediately to the RIGHT of its parent tab — after
   * any sub-agent tabs of the same parent already sitting there — so
   * fan-out layouts always read parent → sub-agents left-to-right.
   * When the parent tab is not present locally, the sub tab is
   * appended at the end (matching the old behavior).  *subTab* may or
   * may not already be in ``tabs``; it is (re)inserted exactly once.
   */
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

  /** Create a fresh collapsible 'Thoughts' llm-panel. */
  function mkThoughtsPanel() {
    const panel = mkEl('div', 'llm-panel');
    const hdr = mkEl('div', 'llm-panel-hdr');
    hdr.textContent = 'Thoughts';
    addCollapse(panel, hdr);
    panel.appendChild(hdr);
    stampPanelStart(panel);
    return panel;
  }

  /** Check if the active tab has a running task. */
  function isActiveTabRunning() {
    const tab = getTab(activeTabId);
    return tab ? tab.isRunning : false;
  }

  /** Find the tab object that owns a backend message by tabId. */
  function findTabByEvt(ev) {
    return ev && ev.tabId !== undefined ? getTab(ev.tabId) : null;
  }

  /**
   * Resolve the working directory for a given tab id.  Returns the
   * tab's own ``workDir`` (set from background-task events) when known,
   * else falls back to ``configWorkDir`` — the configured work
   * directory the backend reports via ``getConfig`` (``configData`` →
   * ``populateConfigForm``).  Returns an empty string only when neither
   * is known, in which case the backend falls back to its global
   * ``work_dir``.  Used to stamp ``workDir`` on commands (e.g.
   * ``autocommitAction``) so they act on the tab's actual repo rather
   * than a possibly-stale daemon-wide directory.
   */
  function workDirForTab(tabId) {
    const tab = getTab(tabId);
    if (tab && tab.workDir) return tab.workDir;
    return configWorkDir || '';
  }

  function saveCurrentTab() {
    const tab = getTab(activeTabId);
    if (!tab) return;
    // Save welcome visibility and detach from O before capturing fragment
    tab.welcomeVisible = welcome ? welcome.style.display !== 'none' : true;
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    // Save DOM subtree as fragment (preserves element references for streaming state)
    tab.outputFragment = document.createDocumentFragment();
    while (O.firstChild) tab.outputFragment.appendChild(O.firstChild);
    tab.taskPanelHTML = taskPanelText ? taskPanelText.textContent : '';
    tab.taskPanelVisible = taskPanel
      ? taskPanel.classList.contains('visible')
      : false;
    tab.statusTextContent = statusText ? statusText.textContent : 'Ready';
    tab.statusTextColor = statusText ? statusText.style.color : 'var(--green)';
    tab.statusTokensText = statusTokens ? statusTokens.textContent : '';
    tab.statusBudgetText = statusBudget ? statusBudget.textContent : '';
    tab.statusStepsText = statusSteps ? statusSteps.textContent : '';
    // Save per-tab state
    tab.selectedModel = selectedModel;
    tab.attachments = attachments;
    tab.inputValue = inp.value;
    tab.isMerging = isMerging;
    tab.isRunning = isActiveTabRunning();
    tab.t0 = t0;
    tab.endTs = endTs;
    // Save streaming state (DOM refs preserved via fragment)
    tab.streamState = state;
    tab.streamLlmPanel = llmPanel;
    tab.streamLlmPanelState = llmPanelState;
    tab.streamLastToolName = lastToolName;
    tab.streamPendingPanel = pendingPanel;
    tab.streamStepCount = stepCount;
    // Save worktree bar (detach from DOM)
    if (worktreeBar && worktreeBar.parentNode) {
      tab.worktreeBarEl = worktreeBar;
      worktreeBar.parentNode.removeChild(worktreeBar);
    } else {
      tab.worktreeBarEl = null;
    }
    worktreeBar = null;
    // Save autocommit bar (detach from DOM)
    if (autocommitBar && autocommitBar.parentNode) {
      tab.autocommitBarEl = autocommitBar;
      autocommitBar.parentNode.removeChild(autocommitBar);
    } else {
      tab.autocommitBarEl = null;
    }
    autocommitBar = null;
    // Save merge toolbar (detach from DOM)
    const mergeBar = document.getElementById('merge-toolbar');
    if (mergeBar && mergeBar.parentNode) {
      tab.mergeToolbarEl = mergeBar;
      mergeBar.parentNode.removeChild(mergeBar);
    } else {
      tab.mergeToolbarEl = null;
    }
    // Restore inputContainer visibility (may have been hidden by worktree/merge bar)
    if (inputContainer) inputContainer.style.display = '';
    persistTabState();
  }

  function restoreTab(tab) {
    activeTabId = tab.id;
    // Restore DOM subtree from fragment (preserves element references)
    O.innerHTML = '';
    if (tab.outputFragment) {
      O.appendChild(tab.outputFragment);
      tab.outputFragment = null;
    }
    if (taskPanel && taskPanelText) {
      // Trim trailing/leading whitespace so the user-visible task text
      // — which is also what gets selected and copied to the clipboard
      // — never carries stray newlines.  Regression: setTaskText() trims
      // its input, but tab.taskPanelHTML can also be written by
      // background-tab handlers ('taskExecuted', 'setTaskText',
      // 'openSubagentTab') from raw event fields, so this restore path
      // must defensively trim too.  See test_task_panel_no_trailing_newlines.py.
      taskPanelText.textContent = (tab.taskPanelHTML || '').trim();
      if (tab.taskPanelVisible) taskPanel.classList.add('visible');
      else taskPanel.classList.remove('visible');
    }
    currentTaskName = (tab.taskPanelHTML || '').trim();
    currentTaskId = tab.currentTaskId !== undefined ? tab.currentTaskId : null;
    updateChevronIcon(!!tab.panelsExpandedMap[currentTaskName]);
    if (statusText) {
      statusText.textContent = tab.statusTextContent || 'Ready';
      statusText.style.color = tab.statusTextColor || 'var(--green)';
    }
    if (statusTokens) statusTokens.textContent = tab.statusTokensText;
    if (statusBudget) statusBudget.textContent = tab.statusBudgetText;
    if (statusSteps) statusSteps.textContent = tab.statusStepsText;
    if (welcome) {
      if (tab.welcomeVisible) {
        welcome.style.display = '';
        if (!O.contains(welcome)) O.appendChild(welcome);
      } else {
        welcome.style.display = 'none';
      }
      refreshWelcomeLayout();
    }
    // Restore per-tab state
    selectedModel = tab.selectedModel || '';
    if (modelName) modelName.textContent = selectedModel;
    attachments = tab.attachments || [];
    renderFileChips();
    inp.value = tab.inputValue || '';
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = inp.scrollHeight + 'px';
    isMerging = tab.isMerging || false;
    t0 = tab.t0 || null;
    endTs = tab.endTs || 0;
    // Restore streaming state (DOM refs valid since fragment preserves elements)
    state = tab.streamState || mkS();
    llmPanel = tab.streamLlmPanel || null;
    llmPanelState = tab.streamLlmPanelState || mkS();
    lastToolName = tab.streamLastToolName || '';
    pendingPanel = tab.streamPendingPanel || false;
    stepCount = tab.streamStepCount || 0;
    _scrollLock = false;
    // Restore worktree bar
    if (worktreeBar && worktreeBar.parentNode)
      worktreeBar.parentNode.removeChild(worktreeBar);
    worktreeBar = null;
    if (tab.worktreeBarEl) {
      worktreeBar = tab.worktreeBarEl;
      tab.worktreeBarEl = null;
      const area = document.getElementById('input-area');
      area.insertBefore(worktreeBar, area.firstChild);
    }
    // Restore autocommit bar
    if (autocommitBar && autocommitBar.parentNode)
      autocommitBar.parentNode.removeChild(autocommitBar);
    autocommitBar = null;
    if (tab.autocommitBarEl) {
      autocommitBar = tab.autocommitBarEl;
      tab.autocommitBarEl = null;
      const acArea = document.getElementById('input-area');
      acArea.insertBefore(autocommitBar, acArea.firstChild);
    }
    // Restore merge toolbar
    const existingMerge = document.getElementById('merge-toolbar');
    if (existingMerge) existingMerge.remove();
    if (tab.mergeToolbarEl) {
      document.getElementById('input-area').appendChild(tab.mergeToolbarEl);
      tab.mergeToolbarEl = null;
    } else if (isMerging) {
      showMergeToolbar(tab.id);
    }
    // Set inputContainer visibility based on active bars and subagent tab status
    const hideInput =
      worktreeBar ||
      autocommitBar ||
      document.getElementById('merge-toolbar') ||
      tab.isSubagentTab;
    if (hideInput) {
      if (inputContainer) inputContainer.style.display = 'none';
    } else {
      if (inputContainer) inputContainer.style.display = '';
    }
    updateInputDisabled();
    resetAdjacentState();
    syncAskModalToActiveTab();
  }

  function renderTabBar() {
    const tabList = document.getElementById('tab-list');
    const tabBar = document.getElementById('tab-bar');
    if (!tabList || !tabBar) return;

    // Always show the tab bar
    tabBar.style.display = '';

    tabList.innerHTML = '';
    tabs.forEach(tab => {
      const el = document.createElement('div');
      el.className =
        'chat-tab' +
        (tab.id === activeTabId ? ' active' : '') +
        (tab.isSubagentTab ? ' subagent-tab' : '');
      el.dataset.tabId = tab.id;

      if (tab.isSubagentTab) {
        // Subagent tab indicator — purple ◉ (fisheye) glyph in the
        // tab title.  While the sub-agent is running we pulse its
        // opacity via the default ``.subagent-indicator`` animation.
        // Once the sub-agent is done we keep the same ◉ glyph but
        // add the ``.done`` modifier class, which (via the
        // corresponding CSS rule) kills the pulse animation and
        // pins opacity at 1 — giving the user a clear "this
        // sub-agent finished" signal: a SOLID (non-pulsing) purple
        // ◉ instead of the pulsing running one.
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
          // Show a filled circle (●) coloured green for success or red
          // for failure via the .chat-tab-ok / .chat-tab-fail classes,
          // replacing the previous ✓ / ✗ glyphs.
          icon.textContent = '\u25CF';
          el.appendChild(icon);
        }
      }

      const label = document.createElement('span');
      label.className = 'chat-tab-label';
      label.textContent = tab.title;
      el.appendChild(label);

      // Show close button for all tabs (regular and subagent)
      const closeBtn = document.createElement('span');
      closeBtn.className = 'chat-tab-close';
      closeBtn.textContent = '\u00d7';
      closeBtn.addEventListener('click', e => {
        e.stopPropagation();
        closeTab(tab.id);
      });
      el.appendChild(closeBtn);

      el.addEventListener('click', () => {
        switchToTab(tab.id);
      });
      el.addEventListener('contextmenu', e => {
        e.preventDefault();
        e.stopPropagation();
        showTabContextMenu(e.clientX, e.clientY, tab.id);
      });
      tabList.appendChild(el);
    });

    // Add "+" button as a direct child of tab-bar, positioned between
    // #tab-list and the action buttons (frequent / history / settings).
    const existingAdd = tabBar.querySelector('.chat-tab-add');
    if (!existingAdd) {
      const addBtn = document.createElement('div');
      addBtn.className = 'chat-tab chat-tab-add';
      addBtn.textContent = '+';
      addBtn.title = 'New chat';
      addBtn.addEventListener('click', () => {
        createNewTab();
      });
      tabBar.appendChild(addBtn);
    }

    // Settings button (gear icon) sits to the right of the "+" button.
    const existingSettings = tabBar.querySelector('.chat-tab-settings');
    if (!existingSettings) {
      const settingsBtn = document.createElement('div');
      settingsBtn.className = 'chat-tab chat-tab-settings';
      settingsBtn.title = 'Settings';
      settingsBtn.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>';
      settingsBtn.addEventListener('click', () => {
        openSettingsPanel();
      });
      tabBar.appendChild(settingsBtn);
    }

    // Scroll the active tab into view
    const activeEl = tabList.querySelector('.chat-tab.active');
    if (activeEl)
      activeEl.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }

  function switchToTab(tabId) {
    if (tabId === activeTabId) return;
    saveCurrentTab();
    const tab = getTab(tabId);
    if (!tab) return;
    restoreTab(tab);
    renderTabBar();
    persistTabState();
    // Restore running state for the target tab.  Keep the restored
    // ``t0``/``endTs`` anchors: a finished tab needs them to render
    // "Done (Xm Ys)" as agent end - start wall-clock.
    setRunningState(tab.isRunning);
    if (!tab.isRunning) {
      stopTimer();
      removeSpinner();
    }
    applyChevronState(
      !!tab.panelsExpandedMap[currentTaskName],
      currentTaskName,
    );
    focusInputWithRetry();
  }

  function closeTab(tabId) {
    const origIdx = tabs.findIndex(t => {
      return t.id === tabId;
    });
    if (origIdx < 0) return;
    // Collect *tabId* and every (transitive) descendant via
    // ``parentTabId`` chains so closing a parent tab also closes the
    // tabs of its sub-agents — and the sub-agents of those sub-agents,
    // recursively.  Closure detection runs against a snapshot of the
    // current ``tabs`` array; mutation happens afterwards.
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
    // Remove every doomed tab from the ``tabs`` array and notify the
    // backend.  Iterate over an explicit id list (not over ``tabs``)
    // because we mutate ``tabs`` inside the loop.
    for (const id of toClose) {
      const i = tabs.findIndex(t => t.id === id);
      if (i >= 0) tabs.splice(i, 1);
      vscode.postMessage({type: 'closeTab', tabId: id});
    }
    if (activeWasClosed) {
      if (tabs.length === 0) {
        // Last tab closed — create a fresh chat instead of closing
        // the secondary sidebar.
        createNewTab();
        return;
      }
      // Switch to an adjacent tab (clamp to the new array length).
      const newIdx = Math.min(origIdx, tabs.length - 1);
      const newTab = tabs[newIdx];
      restoreTab(newTab);
      // Restore running state for the new tab.  Keep the restored
      // ``t0``/``endTs`` anchors (see switchToTab).
      setRunningState(newTab.isRunning);
      if (!newTab.isRunning) {
        stopTimer();
        removeSpinner();
      }
      applyChevronState(
        !!newTab.panelsExpandedMap[currentTaskName],
        currentTaskName,
      );
      focusInputWithRetry();
    }
    renderTabBar();
    persistTabState();
  }

  // --- Tab context menu ---
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
    // Position the menu, clamping to viewport
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
  document.addEventListener('contextmenu', e => {
    if (
      !e.target.closest('#tab-context-menu') &&
      !e.target.closest('.chat-tab')
    ) {
      closeTabContextMenu();
    }
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') closeTabContextMenu();
  });

  /**
   * Create a new chat tab.
   *
   * Always allocates a fresh tab with a newly minted uuid.  The
   * frontend never dedupes by tab id when the user clicks a history
   * row — the backend is the multi-client source of truth and may
   * be observed concurrently from several browsers/webviews, so the
   * "focus the existing tab keyed by this id" shortcut would only
   * be correct for a single-client setup.
   */
  /**
   * Materialise a sub-agent tab in the background — without changing
   * ``activeTabId`` and without any of the side effects that
   * ``createNewTab`` triggers for a user-initiated new chat
   * (``saveCurrentTab``/``restoreTab`` DOM swap, ``newChat`` and
   * ``getWelcomeSuggestions`` posts to the backend, focus theft).
   *
   * Called from the new_tab message handler when the backend's
   * broadcast carries a ``parent_tab_id`` — i.e. when the new tab is
   * a sub-agent tab spawned under a ``run_parallel`` call.  The
   * sub-agent run shares the parent's ``chat_id`` so minting a fresh
   * backend chat (which ``createNewTab`` does via ``newChat``) would
   * be incorrect.
   *
   * The fresh tab is anchored immediately to the right of its parent
   * via ``placeSubagentTabAfterParent`` so the tab bar reads
   * parent → sub-agents left-to-right.  Returns the new tab object;
   * callers use ``returned.id`` to address it in subsequent
   * ``resumeSession`` posts.
   */
  function createBackgroundSubagentTab(parentId) {
    const subTab = makeTab('new chat');
    if (parentId) subTab.parentTabId = parentId;
    // Mark the tab as a sub-agent tab immediately so the brief window
    // between this ``new_tab`` and the follow-up ``openSubagentTab``
    // event is consistent with the tab's final identity.  In
    // particular, ``persistTabState`` filters sub-agent tabs out of
    // the persisted set (they are reopened by the parent's
    // ``resumeSession`` flow on restart — see
    // ``_open_persisted_subagent_tabs`` in server.py).  Without this
    // flag a window reload landing inside the
    // ``new_tab → openSubagentTab`` window would persist a stray
    // regular tab with no backend chat id.
    subTab.isSubagentTab = true;
    placeSubagentTabAfterParent(subTab, parentId);
    renderTabBar();
    persistTabState();
    return subTab;
  }

  function createNewTab() {
    // Preserve any typed text so it carries over to the new tab
    const pendingText = inp.value || '';
    saveCurrentTab();
    const tab = makeTab('new chat');
    tab.inputValue = pendingText;
    tabs.push(tab);
    activeTabId = tab.id;
    // Reset UI for fresh tab
    // (empty fragment, "Ready" status, welcome visible, no merge,
    // no worktree bar, etc.).  `restoreTab` applies that state to
    // the shared DOM, so no additional manual resets are needed.
    restoreTab(tab);
    renderTabBar();
    persistTabState();
    // Sync the module-global running state with the fresh tab (isRunning
    // is false on newly made tabs).  Without this, restoreTab's final
    // updateInputDisabled() would read the *previous* tab's stale
    // isRunning and leave inp / sendBtn disabled.  Mirrors switchToTab
    // and closeTab.
    setRunningState(tab.isRunning);
    if (!tab.isRunning) {
      t0 = null;
      stopTimer();
      removeSpinner();
    }
    vscode.postMessage({type: 'newChat', tabId: tab.id});
    vscode.postMessage({type: 'getWelcomeSuggestions'});
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

  /** Persist lightweight tab metadata via vscode.setState for cross-restart restore. */
  function persistTabState() {
    // Sub-agent tabs are NOT persisted.  They share the parent's
    // backend chat id, so a chat-id-only resumeSession after a
    // restart cannot identify which sub-agent row a restored sub tab
    // should replay.  Instead, the parent tab's own resumeSession
    // deterministically reopens one sub-agent tab per persisted
    // sub-agent row — with that row's own events — to the right of
    // the parent (see _open_persisted_subagent_tabs in server.py and
    // the 'openSubagentTab' handler below).  Persisting sub tabs
    // would duplicate those reopened tabs and load the parent's
    // events into them.
    const persistable = tabs.filter(t => {
      return !t.isSubagentTab;
    });
    const serialized = persistable.map(t => {
      // Always use activeTabId for the active tab so the persisted
      // chatId stays in sync even when saveCurrentTab() hasn't run.
      return {
        title: t.title,
        chatId: t.id,
        backendChatId: t.backendChatId || '',
        parentTabId: t.parentTabId || '',
        // Persist the tab's pinned work_dir so a window reload that
        // restores the tab keeps the same effective work_dir even
        // before ``resumeSession`` replays ``task_events`` (and even
        // for older persisted rows whose ``extra`` carries no
        // ``work_dir``).  See INVARIANTS.md → Tabs & chat webview.
        workDir: t.workDir || '',
      };
    });
    let activeIdx = persistable.findIndex(t => {
      return t.id === activeTabId;
    });
    if (activeIdx < 0) {
      // The active tab is a sub-agent tab (filtered out above): fall
      // back to its parent so the restored window focuses the chat
      // the user was working in.
      const active = getTab(activeTabId);
      const parentId = active && active.parentTabId ? active.parentTabId : '';
      activeIdx = persistable.findIndex(t => {
        return t.id === parentId;
      });
    }
    vscode.setState({
      tabs: serialized,
      activeTabIndex: activeIdx,
      chatId: activeTabId,
    });
  }

  // Initialize tabs — restore from saved state if available, else create one default tab
  // Race-fix: seed the closure-scoped ``selectedModel`` from the DOM
  // BEFORE the launch IIFE creates any tab.  ``makeTab`` reads the
  // closure variable to populate ``tab.selectedModel``; without this
  // seeding every tab built during init (including the ones restored
  // from ``vscode.getState()``) records ``''`` and the picker turns
  // blank on the next tab switch — even though ``#model-name`` shows
  // the correct template value on launch and the daemon's ``models``
  // event later updates the live label.
  {
    const _initialModelEl = document.getElementById('model-name');
    if (_initialModelEl && _initialModelEl.textContent) {
      selectedModel = _initialModelEl.textContent;
    }
  }

  (function () {
    const saved = vscode.getState();
    if (saved && saved.tabs && saved.tabs.length > 0) {
      tabs = [];
      const restoredBackendChatIds = new Set();
      saved.tabs.forEach(st => {
        // Sub-agent tabs (persisted by older versions of
        // persistTabState) are dropped: they cannot be resumed by
        // their (shared, parent-owned) chat id.  The parent tab's
        // resumeSession reopens one fresh sub-agent tab per persisted
        // sub-agent row, with the row's own events, right of the
        // parent.
        if (st.isSubagentTab) return;
        const persistedBackendChatId = st.backendChatId
          ? String(st.backendChatId)
          : '';
        if (
          persistedBackendChatId &&
          restoredBackendChatIds.has(persistedBackendChatId)
        ) {
          return;
        }
        const tab = makeTab(st.title);
        // Restore tab.id from persisted chatId (frontend tab identifier)
        if (st.chatId) tab.id = st.chatId;
        if (persistedBackendChatId) {
          tab.backendChatId = persistedBackendChatId;
          restoredBackendChatIds.add(persistedBackendChatId);
        }
        if (st.parentTabId) tab.parentTabId = st.parentTabId;
        // Restore the tab's pinned work_dir (see persistTabState).
        // Survives a settings-panel change made before the next
        // ``resumeSession`` replay re-pins it from ``extra.work_dir``.
        if (st.workDir) tab.workDir = st.workDir;
        tabs.push(tab);
      });
    }
    if (tabs.length > 0) {
      const idx = (saved && saved.activeTabIndex) || 0;
      if (idx >= 0 && idx < tabs.length) {
        activeTabId = tabs[idx].id;
        // Tab IDs restored from persisted state
      } else {
        activeTabId = tabs[0].id;
      }
    } else {
      const initial = makeTab('new chat');
      tabs.push(initial);
      activeTabId = initial.id;
    }
  })();

  // Elements
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
  // Read initial model from DOM (injected by the backend template)
  if (modelName && modelName.textContent) selectedModel = modelName.textContent;
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

  // #sidebar hosts only the History list now.  The Frequent tasks list
  // lives in its own standalone bottom-anchored panel (#frequent-panel),
  // and Settings has its own standalone right-anchored panel
  // (#settings-panel).
  const settingsPanel = document.getElementById('settings-panel');
  const settingsOverlay = document.getElementById('settings-overlay');
  const settingsPanelClose = document.getElementById('settings-panel-close');
  const frequentPanel = document.getElementById('frequent-panel');
  const frequentOverlay = document.getElementById('frequent-overlay');
  const frequentPanelClose = document.getElementById('frequent-panel-close');
  const frequentTasksBtn = document.getElementById('frequent-tasks-btn');
  const frequentList = document.getElementById('frequent-list');
  // Tricks panel — mirrors the Frequent tasks panel structure.  The
  // trick texts are parsed from src/kiss/INJECTIONS.md by the HTML
  // builder and injected as window.__TRICKS__ before main.js loads.
  const tricksPanel = document.getElementById('tricks-panel');
  const tricksOverlay = document.getElementById('tricks-overlay');
  const tricksPanelClose = document.getElementById('tricks-panel-close');
  const tricksBtn = document.getElementById('tricks-btn');
  const tricksList = document.getElementById('tricks-list');
  const autocommitBtn = document.getElementById('autocommit-btn');
  const waitSpinner = document.getElementById('wait-spinner');
  const ghostOverlay = document.getElementById('ghost-overlay');
  const inputContainer = document.getElementById('input-container');
  const inputClearBtn = document.getElementById('input-clear-btn');
  const worktreeToggleBtn = document.getElementById('cfg-use-worktree');
  const parallelToggleBtn = document.getElementById('cfg-use-parallel');
  const demoToggleBtn = document.getElementById('cfg-demo-mode');
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
  const taskPanelChevron = document.getElementById('task-panel-chevron');
  const taskPanelCopy = document.getElementById('task-panel-copy');
  const statusTokens = document.getElementById('status-tokens');
  const statusBudget = document.getElementById('status-budget');
  const statusSteps = document.getElementById('status-steps');

  // In the remote chat webview (body.remote-chat) the welcome page hides
  // the SAMPLE_TASKS suggestions and shows the input textbox + buttons
  // centered inside #welcome.  We achieve the centering by physically
  // moving #input-area into #welcome while welcome is visible, and back
  // to its original position (between #output and #sidebar inside #app)
  // when a task starts and welcome is hidden.  Outside the remote
  // webview this helper is a no-op so the VS Code extension layout is
  // unchanged.
  function refreshWelcomeLayout() {
    // In remote-chat mode the input area stays pinned at the bottom of
    // #app in both the welcome and running states so its width is
    // always consistent.  The welcome content is displayed in the
    // output area above it.
    if (!document.body.classList.contains('remote-chat')) return;
    const ia = document.getElementById('input-area');
    const app = document.getElementById('app');
    if (!ia || !app || !welcome) return;
    // If the input-area was previously moved into #welcome (e.g. by an
    // older code path), move it back to #app so it always sits at the
    // bottom with full width.
    if (ia.parentNode === welcome) {
      const sbar = document.getElementById('sidebar');
      if (sbar) app.insertBefore(ia, sbar);
      else app.appendChild(ia);
    }
  }

  // Apply the centered remote welcome layout on initial load (welcome
  // is visible by default in the static HTML).
  refreshWelcomeLayout();

  function setTaskText(text) {
    if (!taskPanel || !taskPanelText) return;
    const t = (text || '').trim();
    if (t) {
      taskPanelText.textContent = t;
      taskPanel.classList.add('visible');
    } else {
      taskPanelText.textContent = '';
      taskPanel.classList.remove('visible');
    }
  }

  /**
   * Return the task name currently visible in the viewport.
   * Uses the same heuristic as updateVisibleTask: the first
   * .adjacent-task whose bounds straddle the 30%-from-top line wins;
   * if none match, the current (main) task is visible.
   */
  function getVisibleTaskName() {
    const adjacentTasks = O.querySelectorAll('.adjacent-task[data-task]');
    if (!adjacentTasks.length) return currentTaskName;
    const outputRect = O.getBoundingClientRect();
    const checkY = outputRect.top + outputRect.height * 0.3;
    for (let i = 0; i < adjacentTasks.length; i++) {
      const rect = adjacentTasks[i].getBoundingClientRect();
      if (rect.top <= checkY && rect.bottom > checkY) {
        return adjacentTasks[i].dataset.task || currentTaskName;
      }
    }
    return currentTaskName;
  }

  /**
   * Apply the chevron expand/collapse state to panels belonging to
   * a specific task.
   * @param {boolean} expanded - true = expand, false = collapse
   * @param {string} taskName - the task whose panels to affect;
   *   panels belonging to other tasks are left untouched.
   *   If empty/falsy, affects only the current (main) task panels.
   *
   * - expanded=false (chevron right, default): hide every .collapsible
   *   panel that belongs to the specified task (display:none via
   *   .chv-hidden) except result panels (.rc) and panels belonging
   *   to the currently running task.
   * - expanded=true (chevron down): reveal every hidden panel belonging
   *   to the specified task and expand every .collapsible panel except
   *   those belonging to the currently running task.
   * Running task panels are direct children of #output (not inside
   *   .adjacent-task) while a task is running; adjacent-task containers
   *   hold previously-completed tasks.
   */
  function applyChevronState(expanded, taskName) {
    if (!O) return;
    const panels = O.querySelectorAll('.collapsible');
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      const adjacentContainer = p.closest('.adjacent-task');
      const inAdjacent = !!adjacentContainer;
      const inRunning = isRunning && !inAdjacent;
      // Determine which task this panel belongs to
      const panelTask = inAdjacent
        ? adjacentContainer.dataset.task || ''
        : currentTaskName;
      // Skip panels that don't belong to the target task
      if (taskName && panelTask !== taskName) continue;
      if (!expanded) {
        if (inRunning || p.classList.contains('rc')) {
          p.classList.remove('chv-hidden');
          continue;
        }
        p.classList.add('chv-hidden');
      } else {
        p.classList.remove('chv-hidden');
        if (inRunning) continue;
        p.classList.remove('collapsed');
        // Highlight code deferred while this panel stayed collapsed.
        highlightPending(p);
        collapsePreview(p);
      }
    }
  }

  /** Update the chevron icon to reflect the current expanded state. */
  function updateChevronIcon(expanded) {
    if (!taskPanelChevron) return;
    if (expanded) taskPanelChevron.classList.add('expanded');
    else taskPanelChevron.classList.remove('expanded');
  }

  if (taskPanelChevron) {
    taskPanelChevron.addEventListener('click', e => {
      e.stopPropagation();
      const tab = getTab(activeTabId);
      const visibleTask = getVisibleTaskName();
      const wasExpanded = tab ? !!tab.panelsExpandedMap[visibleTask] : false;
      const expanded = !wasExpanded;
      if (tab) tab.panelsExpandedMap[visibleTask] = expanded;
      updateChevronIcon(expanded);
      applyChevronState(expanded, visibleTask);
    });
  }

  /**
   * Copy *text* via a temporary textarea + ``document.execCommand('copy')``
   * — fallback for environments without the async clipboard API.
   * Returns true when ``execCommand`` ran without throwing.
   */
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
    } catch {
      /* ignore */
    }
    document.body.removeChild(ta);
    return ok;
  }

  // Copy-task button: trims the visible task text and copies it to the
  // system clipboard.  Briefly swaps the clipboard icon for a green
  // check mark to confirm.
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

  // Merge state
  let isMerging = false;

  // Streaming state (mirrors browser handleOutputEvent)
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
      // Cached descendants/buffers for RAF-batched streaming.  Coalescing
      // many small token deltas into a single DOM mutation per frame
      // avoids per-token layout thrash during high-rate LLM streaming.
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
    };
  }

  function resetOutputState() {
    state = mkS();
    llmPanel = null;
    llmPanelState = mkS();
    lastToolName = '';
    pendingPanel = false;
    stepCount = 0;
    _scrollLock = false;
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

  function renderAdjacentTask(direction, task, events, taskId) {
    removeAdjacentLoader();
    adjacentLoading = false;

    if (!task || !events || events.length === 0) {
      if (direction === 'prev') noPrevTask = true;
      else noNextTask = true;
      return;
    }

    // Create a container for the adjacent task
    const container = mkEl('div', 'adjacent-task');
    container.dataset.task = task;
    // Stamp the row id so a 'taskDeleted' broadcast from the backend
    // can locate and remove this exact block via
    //   .adjacent-task[data-task-id="<id>"]
    if (taskId !== undefined && taskId !== null && taskId !== '')
      container.dataset.taskId = String(taskId);

    // Replay events into the container (save/restore header metrics so
    // adjacent-task replay doesn't overwrite the current task's values)
    const savedTokens = statusTokens ? statusTokens.textContent : '';
    const savedBudget = statusBudget ? statusBudget.textContent : '';
    const savedSteps = statusSteps ? statusSteps.textContent : '';
    replayEventsInto(container, events);
    // Capture the adjacent task's metrics before restoring the current ones
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

    if (direction === 'prev') {
      // Save scroll position, prepend, then restore
      const prevScrollHeight = O.scrollHeight;
      O.insertBefore(container, O.firstChild);
      const newScrollHeight = O.scrollHeight;
      O.scrollTop += newScrollHeight - prevScrollHeight;
      if (taskId !== undefined && taskId !== null && taskId !== '')
        oldestLoadedTaskId = taskId;
    } else {
      O.appendChild(container);
      if (taskId !== undefined && taskId !== null && taskId !== '')
        newestLoadedTaskId = taskId;
    }
    const tab = getTab(activeTabId);
    if (tab) applyChevronState(!!tab.panelsExpandedMap[task], task);
  }

  function clearOutput() {
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    O.innerHTML = '';
  }

  // --- Spinner ---
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

  // --- Ghost text ---
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

  /** Accept the current ghost text suggestion into the input. */
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

  /** Cycle to the previous (older) history item. Returns true if acted. */
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

  /** Cycle to the next (newer) history item. Returns true if acted. */
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

  // --- Mobile touch gestures ---
  // Swipe right on input to accept ghost text (replaces Tab key).
  // Swipe up/down on input to cycle history (replaces ArrowUp/ArrowDown).
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
      // Swipe right: accept ghost text
      if (acceptGhost()) e.preventDefault();
    } else if (absDy > absDx) {
      if (dy < -SWIPE_THRESHOLD && autocomplete.style.display !== 'block') {
        // Swipe up: previous history item
        if (cycleHistoryUp()) e.preventDefault();
      } else if (dy > SWIPE_THRESHOLD) {
        // Swipe down: next history item
        if (cycleHistoryDown()) e.preventDefault();
      }
    }
  }

  function requestGhost() {
    clearGhost();
    if (isRunning || !inp.value) return;
    // Don't request ghost when in file picker mode (@-mention autocomplete)
    if (getAtCtx()) return;
    // Don't request ghost when cursor isn't at end
    if (inp.selectionStart < inp.value.length) return;
    // Minimum query length check (2 non-whitespace chars)
    if (inp.value.replace(/\s/g, '').length < 2) return;
    ghostTimer = setTimeout(() => {
      ghostTimer = null;
      vscode.postMessage({type: 'complete', query: inp.value});
    }, 300);
  }

  // --- File path detection (matches web Sorcar) ---
  // --- Shared rendering (ported from browser EVENT_HANDLER_JS) ---

  function esc(t) {
    const d = document.createElement('div');
    d.textContent = t;
    return d.innerHTML;
  }

  // --- Custom tooltip (native title doesn't work in VS Code webviews) ---
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

  // ------------------------------------------------------------------
  // Filepath linkifier — walks every text node under ``root`` and
  // wraps slash-bearing tokens that look like absolute paths
  // (``/foo/bar``), home-relative paths (``~/foo``), dot-relative
  // paths (``./foo``, ``../foo``), or workspace-relative paths with
  // at least one directory component (``src/foo``) — with optional
  // ``:line`` suffix — in a ``<span class="kiss-filelink"
  // data-path="...">``.  The existing global click handler (see
  // bottom of this file) dispatches on ``[data-path]`` to post an
  // ``openFile`` message to the extension, which validates the path
  // and dispatches it to the VS Code editor or the native viewer.
  //
  // We skip text nodes inside ``<a>`` (already a hyperlink — marked
  // autolinks URLs) and inside any element that already carries a
  // ``data-path`` attribute (e.g. the existing tool_call ``.tp``
  // hooks).  The leading character class lookbehind avoids matching
  // a URL's path component (``https://x/y``) as a filepath: the
  // character before ``/y`` is the alphanumeric host suffix, which
  // ``\w`` rejects.
  //
  // The path regex deliberately requires at least one ``/`` so bare
  // filenames like ``package.json`` — which would noise-up sentences
  // and ambiguous tokens like ``v1.0`` — are NOT linkified.  Trailing
  // sentence punctuation (``,``, ``.``, ``;``, ``)``, ``]``) is
  // excluded by the closing character class so ``/tmp/foo.py,`` is
  // captured as ``/tmp/foo.py``.
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

  function linkifyFilePaths(root) {
    if (!root || root.nodeType !== 1) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        let p = node.parentNode;
        while (p && p !== root.parentNode) {
          if (p.nodeType === 1) {
            if (_LINK_SKIP_TAGS.has(p.tagName)) {
              return NodeFilter.FILTER_REJECT;
            }
            if (p.dataset && p.dataset.path) {
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
        span.className = 'kiss-filelink';
        span.setAttribute('data-path', m[1]);
        span.title = 'Open ' + m[1];
        span.textContent = m[1];
        frag.appendChild(span);
        last = end;
      }
      if (last < text.length) {
        frag.appendChild(node.ownerDocument.createTextNode(text.slice(last)));
      }
      if (node.parentNode) node.parentNode.replaceChild(frag, node);
    }
  }

  function hlBlock(el) {
    if (typeof hljs === 'undefined') return;
    el.querySelectorAll('pre code').forEach(bl => {
      // During bulk replay, defer the (expensive) highlight: tag the block so
      // it is only highlighted once its panel is shown.  See highlightPending.
      if (_deferHighlight) {
        bl.classList.add('needs-hl');
      } else {
        hljs.highlightElement(bl);
      }
    });
  }

  /**
   * Highlight code blocks under `root` that were deferred during replay.
   *
   * Blocks tagged with `needs-hl` by hlBlock() while `_deferHighlight` was
   * active are syntax-highlighted now and the tag removed, so each block is
   * highlighted at most once.  Called when a collapsed panel is expanded so a
   * long task's off-screen code is highlighted lazily instead of all at once.
   */
  function highlightPending(root) {
    if (typeof hljs === 'undefined' || !root) return;
    root.querySelectorAll('code.needs-hl').forEach(bl => {
      bl.classList.remove('needs-hl');
      hljs.highlightElement(bl);
    });
  }

  function toggleThink(el) {
    const p = el.parentElement;
    p.querySelector('.cnt').classList.toggle('hidden');
    el.querySelector('.arrow').classList.toggle('collapsed');
  }

  /**
   * Recursively collect text from a DOM node, inserting a space before each
   * element-node boundary so that adjacent block-level elements (divs, pres)
   * produce separated words.  Unlike innerText, this works correctly even
   * when the node is hidden (display:none), where innerText falls back to
   * textContent and concatenates block children without separators.
   */
  function collectText(node) {
    if (node.nodeType === 3) return node.textContent || '';
    if (node.nodeType === 1 && node.classList) {
      // Skip UI-only chrome injected by the panel helpers so it never
      // ends up in either the collapse preview or the clipboard payload.
      if (
        node.classList.contains('panel-copy-btn') ||
        node.classList.contains('collapse-chv') ||
        node.classList.contains('collapse-preview')
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

  function addCollapse(panelEl, headerEl) {
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
      _noScroll = true;
      panelEl.classList.toggle('collapsed');
      if (panelEl.classList.contains('collapsed')) {
        panelEl.classList.remove('user-pinned');
      } else {
        panelEl.classList.add('user-pinned');
        // Highlight code that was deferred while this panel stayed collapsed.
        highlightPending(panelEl);
      }
      collapsePreview(panelEl);
      setTimeout(() => {
        _noScroll = false;
      }, 0);
    });
    addCopyButton(panelEl);
  }

  // Panel Copy button + raw-text walker live in media/panelCopy.js so
  // they can be loaded both in the webview and from a Node + jsdom test.
  const addCopyButton = window.PanelCopy.addCopyButton;
  const PANEL_COPY_SVG = window.PanelCopy.PANEL_COPY_SVG;
  const PANEL_CHECK_SVG = window.PanelCopy.PANEL_CHECK_SVG;

  function collapseAllExceptResult(container) {
    const panels = container.querySelectorAll('.collapsible');
    for (let i = 0; i < panels.length; i++) {
      if (!panels[i].classList.contains('rc')) {
        panels[i].classList.add('collapsed');
        collapsePreview(panels[i]);
      }
    }
  }

  function collapseOlderPanels() {
    if (!isRunning) return;
    const panels = O.querySelectorAll(':scope > .collapsible');
    for (let i = 0; i < panels.length - 1; i++) {
      if (
        !panels[i].classList.contains('rc') &&
        !panels[i].classList.contains('user-pinned')
      ) {
        panels[i].classList.add('collapsed');
        collapsePreview(panels[i]);
      }
    }
  }

  function splitMultiSessionSummary(summary) {
    const text = typeof summary === 'string' ? summary : '';
    const finalMarker = '\n\n---\n\n### Final Session\n';
    let markerIdx = text.indexOf(finalMarker);
    let markerLen = finalMarker.length;
    if (markerIdx <= 0) {
      const separator = '\n\n---\n\n';
      markerIdx = text.lastIndexOf(separator);
      markerLen = separator.length;
    }
    if (markerIdx <= 0) return null;
    const previous = text.substring(0, markerIdx).trim();
    const final = text.substring(markerIdx + markerLen).trim();
    if (!previous || !final) return null;
    if (!previous.includes('### Previous Session')) return null;
    return {previous: previous, final: final};
  }

  function removeResultPanels(container) {
    if (!container || !container.children) return;
    for (let i = container.children.length - 1; i >= 0; i--) {
      const child = container.children[i];
      if (child.classList && child.classList.contains('rc')) child.remove();
    }
  }

  function createResultPanel(ev, summaryOverride, titleOverride, showStatus) {
    const rc = mkEl('div', 'ev rc');
    let rb = '';
    let rawBody = '';
    if (showStatus && ev.is_continue) {
      rb +=
        '<div style="color:var(--yellow);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px">Status: Continue</div>';
      rawBody += 'Status: Continue\n\n';
    } else if (showStatus && ev.success === false) {
      rb +=
        '<div style="color:var(--red);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px">Status: FAILED</div>';
      rawBody += 'Status: FAILED\n\n';
    }
    let usePre = true;
    const summaryText =
      summaryOverride !== undefined ? summaryOverride : ev.summary;
    if (summaryText) {
      const sum = String(summaryText)
        .replace(/\n{3,}/g, '\n\n')
        .trim();
      if (typeof marked !== 'undefined') {
        rb += kissSanitize(marked.parse(sum));
        usePre = false;
      } else {
        rb += esc(sum);
      }
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
    const rcBody = rc.querySelector('.rc-body');
    if (rcBody) linkifyFilePaths(rcBody);
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

  function handleOutputEvent(ev, target, tState) {
    const t = ev.type;
    switch (t) {
      case 'thinking_start':
        tState.thinkEl = mkEl('div', 'ev think');
        tState.thinkEl.innerHTML =
          '<div class="lbl" onclick="toggleThink(this)">' +
          '<span class="arrow">\u25BE</span> Thinking</div>' +
          '<div class="cnt"></div>';
        // Cache the .cnt child so per-delta updates do not pay
        // querySelector cost on every streamed token.
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
              // appendData on a single Text node is far cheaper than
              // reassigning textContent (which discards/rebuilds nodes).
              const cnt = tState.thinkCnt;
              const last = cnt.lastChild;
              if (last && last.nodeType === 3) {
                last.appendData(tState.thinkBuf);
              } else {
                cnt.appendChild(document.createTextNode(tState.thinkBuf));
              }
              tState.thinkBuf = '';
              if (tState.thinkEl)
                tState.thinkEl.scrollTop = tState.thinkEl.scrollHeight;
            });
          }
        }
        break;
      case 'thinking_end':
        // Keep the thinking panel expanded so the streamed thinking
        // tokens remain visible after the block ends.  The user can
        // still click the "Thinking" label to manually collapse.
        if (tState.thinkRaf) {
          cancelAnimationFrame(tState.thinkRaf);
          tState.thinkRaf = 0;
          if (tState.thinkCnt && tState.thinkBuf) {
            const cnt = tState.thinkCnt;
            const last = cnt.lastChild;
            if (last && last.nodeType === 3) last.appendData(tState.thinkBuf);
            else cnt.appendChild(document.createTextNode(tState.thinkBuf));
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
            // Preserve raw markdown so the panel Copy button reproduces
            // the original markdown rather than the rendered HTML's
            // textContent (which loses #/`*/` markers).
            tState.txtEl.dataset.rawText = tState.txtBuf || '';
          } else if (tState.txtNode && tState.txtPending) {
            tState.txtNode.appendData(tState.txtPending);
          }
          // Surface clickable filepaths inside the streamed text
          // body so the global click handler can route them to the
          // extension's ``openFile`` viewer.  Runs regardless of
          // whether ``marked`` is available — when it is not, the
          // text node fallback still needs the linkifier.
          linkifyFilePaths(tState.txtEl);
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
          linkifyFilePaths(tState.bashPanel);
        }
        tState.bashPanel = null;
        tState.bashRaf = 0;
        const c = mkEl('div', 'ev tc');
        const hdr = mkEl('div', 'tc-h');
        hdr.textContent = ev.name || 'Tool';
        // The Bash tool-call panel header is painted in the cyan
        // theme colour to visually distinguish shell invocations from
        // every other tool (which use the orange accent).  Tag the
        // header (and outer container) with a Bash-specific CSS hook
        // so ``main.css`` can target it without touching the generic
        // ``.tc-h`` rule that every other tool depends on.
        if (ev.name === 'Bash') {
          hdr.classList.add('tc-h-bash');
          c.classList.add('tc-bash');
        }
        let b = '';
        if (ev.path) {
          const ep = esc(ev.path).replace(/"/g, '&quot;');
          b +=
            '<div class="tc-arg"><span class="tc-arg-name">path:</span> <span class="tp" data-path="' +
            ep +
            '">' +
            esc(ev.path) +
            '</span></div>';
        }
        if (ev.description)
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
          for (const k in ev.extras)
            b +=
              '<div class="extra">' +
              esc(k) +
              ': ' +
              esc(ev.extras[k]) +
              '</div>';
        }
        const tcBody = mkEl('div', 'tc-b');
        tcBody.innerHTML =
          b || '<em style="color:var(--dim)">No arguments</em>';
        c.appendChild(hdr);
        c.appendChild(tcBody);
        addCollapse(c, hdr);
        target.appendChild(c);
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
          linkifyFilePaths(tState.bashPanel);
        } else if (tState.bashPanel) {
          // Even if no pending buffer, the panel may have been
          // populated by prior system_output flushes — make sure
          // those text nodes are linkified before the bash panel is
          // finalised.
          linkifyFilePaths(tState.bashPanel);
        }
        const hadBash = !!tState.bashPanel;
        tState.bashPanel = null;
        tState.bashRaf = 0;
        // Close out the tool-call panel's time footer here — BEFORE the
        // ``hadBash && !is_error`` early exit — so every tool_result
        // path (bash, plain output, error) stamps the elapsed time.
        if (tState.lastToolCallEl) finalizePanelTime(tState.lastToolCallEl);
        if (hadBash && !ev.is_error) break;
        const resultTarget = tState.lastToolCallEl || target;
        if (ev.is_error) {
          const r = mkEl('div', 'ev tr err');
          r.innerHTML =
            '<div class="rl fail">FAILED</div><div class="tr-content">' +
            esc(ev.content) +
            '</div>';
          // Raw tool-result error text for the Copy button.
          r.dataset.rawText = 'FAILED\n' + (ev.content || '');
          addCollapse(r, r.querySelector('.rl'));
          resultTarget.appendChild(r);
          const trBody = r.querySelector('.tr-content');
          if (trBody) linkifyFilePaths(trBody);
        } else {
          const op = mkEl('div', 'bash-panel');
          const opContent = mkEl('div', 'bash-panel-content');
          opContent.textContent = ev.content;
          linkifyFilePaths(opContent);
          op.appendChild(opContent);
          addCopyButton(op);
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
                linkifyFilePaths(tState.bashPanel);
              }
              tState.bashBuf = '';
              tState.bashRaf = 0;
              if (tState.bashPanel)
                tState.bashPanel.scrollTop = tState.bashPanel.scrollHeight;
            });
          }
        } else {
          const s = mkEl('div', 'ev sys');
          s.textContent = (ev.text || '').replace(/\n\n+/g, '\n');
          linkifyFilePaths(s);
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
            ),
          );
          target.appendChild(
            createResultPanel(ev, multiSummary.final, 'Result', true),
          );
        } else {
          target.appendChild(createResultPanel(ev, undefined, 'Result', true));
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
        const el = mkEl('div', 'ev ' + cls);
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
        // Preserve the raw markdown so the panel Copy button reproduces
        // the original markdown rather than the rendered HTML.
        el.dataset.rawText = ev.text || '';
        addCollapse(el, el.querySelector('.' + cls + '-h'));
        hlBlock(el);
        target.appendChild(el);
        const bodyEl = el.querySelector('.' + cls + '-body');
        if (bodyEl) {
          linkifyFilePaths(bodyEl);
          bodyEl.scrollTop = bodyEl.scrollHeight;
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
      case 'autocommit_done': {
        const cls2 = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
        const acDiv = mkEl('div', 'ev ' + cls2);
        acDiv.textContent = (ev && ev.message) || '';
        target.appendChild(acDiv);
        break;
      }
      case 'warning': {
        // Persisted backend warning (e.g. the worktree agent's
        // stash-pop failure) replayed from the stored event stream —
        // ``warning`` is in json_printer's ``_DISPLAY_EVENT_TYPES``
        // so it survives a chat reopen / demo replay.  Rendered
        // identically to the live banner (``addWarning``); the LIVE
        // path never reaches here because the top-level message
        // switch handles ``case 'warning'`` and breaks.
        const warnDiv = mkEl('div', 'ev tr warn');
        warnDiv.innerHTML =
          '<strong>Warning:</strong> ' + esc(ev.message || ev.text || '');
        target.appendChild(warnDiv);
        break;
      }
    }
  }

  function updateStepCount(count) {
    stepCount = count;
    if (statusSteps) statusSteps.textContent = 'Steps: ' + count;
  }

  function processOutputEvent(ev) {
    const t = ev.type;
    if (t === 'tool_call') {
      lastToolName = ev.name || '';
      // Close out the previous Thoughts panel — the next streaming
      // step will create a new one — so the time-spent footer covers
      // the period between this panel and the tool call that ends it.
      if (llmPanel) finalizePanelTime(llmPanel);
      llmPanel = null;
      llmPanelState = mkS();
      // Set true (not false) so that non-core tools (screenshot,
      // go_to_url, scroll, etc.) whose tool_result is suppressed by
      // the backend still trigger Thoughts-panel creation for the
      // subsequent thinking/text block.
      pendingPanel = true;
    }
    if (t === 'tool_result' && lastToolName !== 'finish') {
      pendingPanel = true;
    }
    // First thought (stepCount === 0) also gets a panel, like every other turn.
    if (
      (pendingPanel || stepCount === 0) &&
      (t === 'thinking_start' || t === 'text_delta')
    ) {
      updateStepCount(stepCount + 1);
      llmPanel = mkThoughtsPanel();
      O.appendChild(llmPanel);
      collapseOlderPanels();
      llmPanelState = mkS();
      pendingPanel = false;
    }
    let target = O,
      tState = state;
    if (
      llmPanel &&
      (t === 'thinking_start' ||
        t === 'thinking_delta' ||
        t === 'thinking_end' ||
        t === 'text_delta' ||
        t === 'text_end')
    ) {
      target = llmPanel;
      tState = llmPanelState;
    }
    handleOutputEvent(ev, target, tState);
    if (target === O) collapseOlderPanels();
    if (t === 'result' || t === 'usage_info') {
      // Snapshot current task metrics so adjacent-scroll can restore them
      currentTaskMetrics.tokens = statusTokens ? statusTokens.textContent : '';
      currentTaskMetrics.budget = statusBudget ? statusBudget.textContent : '';
      currentTaskMetrics.steps = statusSteps ? statusSteps.textContent : '';
    }
    if (t === 'result') {
      // Close out the last live Thoughts panel so its bottom-anchored
      // time footer reflects the duration up to the result event.
      if (llmPanel) finalizePanelTime(llmPanel);
      collapseAllExceptResult(O);
      if (ev.success === false && !ev.is_continue) {
        const rTab = getTab(activeTabId);
        if (rTab) rTab.lastTaskFailed = true;
      }
      // After a result, the next thinking/text (e.g. from a new
      // RelentlessAgent sub-session) must create its own Thoughts panel.
      pendingPanel = true;
    }
    // Batch the llm-panel auto-scroll into one RAF tick so a burst of
    // streaming deltas does not force a synchronous layout per event.
    if (target === llmPanel && llmPanel && !llmPanel._scrollRaf) {
      const _lp = llmPanel;
      _lp._scrollRaf = requestAnimationFrame(() => {
        _lp._scrollRaf = 0;
        _lp.scrollTop = _lp.scrollHeight;
      });
    }
    // Keep the chevron "right" state consistent across new panels added by streaming.
    // Skip during demo replay — demo mode never sets isRunning so
    // applyChevronState(false) would hide every non-result panel via chv-hidden.
    const tab = getTab(activeTabId);
    if (tab && !tab.panelsExpandedMap[currentTaskName] && !_demoActive)
      applyChevronState(false, currentTaskName);
  }

  /**
   * Process a streaming output event for a background (non-active) tab.
   * Mirrors processOutputEvent but operates on the tab's saved outputFragment
   * and streaming state so panels are built even when the tab is not visible.
   */
  function processOutputEventForBgTab(ev, tab) {
    const t = ev.type;

    if (!tab.outputFragment)
      tab.outputFragment = document.createDocumentFragment();

    // Load the tab's streaming state into locals
    let bgLastToolName = tab.streamLastToolName || '';
    let bgLlmPanel = tab.streamLlmPanel || null;
    let bgLlmPanelState = tab.streamLlmPanelState || mkS();
    let bgPendingPanel = tab.streamPendingPanel || false;
    let bgStepCount = tab.streamStepCount || 0;
    const bgState = tab.streamState || mkS();

    // Advance the streaming state machine
    if (t === 'tool_call') {
      bgLastToolName = ev.name || '';
      // Mirror processOutputEvent: close out the previous Thoughts
      // panel so its time-spent footer covers up to this tool call.
      if (bgLlmPanel) finalizePanelTime(bgLlmPanel);
      bgLlmPanel = null;
      bgLlmPanelState = mkS();
      bgPendingPanel = true;
    }
    if (t === 'tool_result' && bgLastToolName !== 'finish') {
      bgPendingPanel = true;
    }

    // Create a new llm-panel when needed
    if (
      (bgPendingPanel || bgStepCount === 0) &&
      (t === 'thinking_start' || t === 'text_delta')
    ) {
      bgStepCount++;
      tab.statusStepsText = 'Steps: ' + bgStepCount;
      bgLlmPanel = mkThoughtsPanel();
      tab.outputFragment.appendChild(bgLlmPanel);
      bgLlmPanelState = mkS();
      bgPendingPanel = false;
    }

    // Handle usage_info: save to tab state without touching DOM
    if (t === 'usage_info') {
      if (ev.total_tokens != null && ev.cost != null) {
        tab.statusTokensText = 'Tokens: ' + fmtN(ev.total_tokens);
        if (ev.cost !== 'N/A') tab.statusBudgetText = 'Cost: ' + ev.cost;
        if (ev.total_steps != null)
          tab.statusStepsText = 'Steps: ' + ev.total_steps;
      }
    } else {
      let target = tab.outputFragment;
      let tState = bgState;
      if (
        bgLlmPanel &&
        (t === 'thinking_start' ||
          t === 'thinking_delta' ||
          t === 'thinking_end' ||
          t === 'text_delta' ||
          t === 'text_end')
      ) {
        target = bgLlmPanel;
        tState = bgLlmPanelState;
      }

      // Protect active-tab globals from side effects in handleOutputEvent
      // (result events update statusTokens/statusBudget/stepCount via DOM)
      const prevStepCount = stepCount;
      const prevTokensText = statusTokens ? statusTokens.textContent : '';
      const prevBudgetText = statusBudget ? statusBudget.textContent : '';
      const prevStepsText = statusSteps ? statusSteps.textContent : '';

      handleOutputEvent(ev, target, tState);

      // Restore active-tab globals
      stepCount = prevStepCount;
      if (statusTokens) statusTokens.textContent = prevTokensText;
      if (statusBudget) statusBudget.textContent = prevBudgetText;
      if (statusSteps) statusSteps.textContent = prevStepsText;

      if (t === 'result') {
        // Close out the last bg-tab Thoughts panel so its footer
        // reflects the duration up to this result event.
        if (bgLlmPanel) finalizePanelTime(bgLlmPanel);
        if (ev.step_count) {
          bgStepCount = ev.step_count;
          tab.statusStepsText = 'Steps: ' + ev.step_count;
        }
        if (ev.total_tokens)
          tab.statusTokensText = 'Tokens: ' + fmtN(ev.total_tokens);
        if (ev.cost && ev.cost !== 'N/A')
          tab.statusBudgetText = 'Cost: ' + ev.cost;
        collapseAllExceptResult(tab.outputFragment);
        if (ev.success === false && !ev.is_continue) tab.lastTaskFailed = true;
        // After a result, the next thinking/text must create a new panel.
        bgPendingPanel = true;
      }
    }

    // Save streaming state back to the tab
    tab.streamState = bgState;
    tab.streamLlmPanel = bgLlmPanel;
    tab.streamLlmPanelState = bgLlmPanelState;
    tab.streamLastToolName = bgLastToolName;
    tab.streamPendingPanel = bgPendingPanel;
    tab.streamStepCount = bgStepCount;
    tab.welcomeVisible = false;
  }

  // --- Scrolling ---

  function sb() {
    if (
      !_scrollLock &&
      !_noScroll &&
      !scrollRaf &&
      !(welcome && welcome.style.display !== 'none')
    ) {
      scrollRaf = requestAnimationFrame(() => {
        O.scrollTo({top: O.scrollHeight, behavior: 'instant'});
        scrollRaf = 0;
      });
    }
  }

  /**
   * Shared overscroll accumulator for wheel/touch adjacent-task loading.
   * Accumulates |delta| while the user keeps overscrolling in `dir`
   * ('prev' | 'next'); once OVERSCROLL_THRESHOLD is reached, resets the
   * accumulator and requests the adjacent task relative to `taskId`.
   */
  function accumulateOverscroll(dir, delta, taskId) {
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
      vscode.postMessage({
        type: 'getAdjacentTask',
        tabId: activeTabId,
        taskId: taskId,
        direction: dir,
      });
    }
  }

  O.addEventListener('wheel', e => {
    if (isRunning && e.deltaY < 0) _scrollLock = true;

    // Adjacent task loading via overscroll detection.  Sub-agent
    // tabs MUST NOT show siblings from the same chat_id — they
    // render exactly one task, the sub-agent's own row.
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
        // Scrolling up at top — load task before the oldest loaded
        accumulateOverscroll('prev', e.deltaY, oldestLoadedTaskId);
      } else if (
        atBottom &&
        e.deltaY > 0 &&
        !noNextTask &&
        newestLoadedTaskId != null
      ) {
        // Scrolling down at bottom — load task after the newest loaded
        accumulateOverscroll('next', e.deltaY, newestLoadedTaskId);
      } else {
        overscrollAccum = 0;
        overscrollDir = '';
      }
    }
  });

  // --- Touch-based adjacent scrolling on #output ---
  // Mirrors the wheel handler above for mobile/tablet devices where wheel
  // events do not fire.  Tracks incremental finger movement while the
  // scroll position is pinned at a boundary.
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
      // Positive touchDelta = finger moved up = scroll down ("next")
      // Negative touchDelta = finger moved down = scroll up ("prev")
      const touchDelta = _touchOutputLastY - currentY;
      _touchOutputLastY = currentY;

      if (adjacentLoading || !activeTabId || !currentTaskName) return;
      // Sub-agent tabs MUST NOT load sibling tasks from the same
      // chat_id — they render exactly the sub-agent's own row.
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
        // Pulling down at top — load previous task
        accumulateOverscroll('prev', touchDelta, oldestLoadedTaskId);
      } else if (
        atBottom &&
        touchDelta > 0 &&
        !noNextTask &&
        newestLoadedTaskId != null
      ) {
        // Pushing up at bottom — load next task
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
      // Reset overscroll state when the finger lifts
      overscrollAccum = 0;
      overscrollDir = '';
      if (overscrollTimer) {
        clearTimeout(overscrollTimer);
        overscrollTimer = null;
      }
    },
    {passive: true},
  );

  function updateVisibleTask() {
    const adjacentTasks = O.querySelectorAll('.adjacent-task[data-task]');
    if (!adjacentTasks.length) return;
    const outputRect = O.getBoundingClientRect();
    const checkY = outputRect.top + outputRect.height * 0.3;
    let visibleTask = currentTaskName;
    let visibleContainer = null;
    for (let i = 0; i < adjacentTasks.length; i++) {
      const rect = adjacentTasks[i].getBoundingClientRect();
      if (rect.top <= checkY && rect.bottom > checkY) {
        visibleTask = adjacentTasks[i].dataset.task;
        visibleContainer = adjacentTasks[i];
        break;
      }
    }
    setTaskText(visibleTask);
    // Sync chevron icon to the visible task's expanded state
    const vTab = getTab(activeTabId);
    if (vTab) updateChevronIcon(!!vTab.panelsExpandedMap[visibleTask]);
    // Update header metrics to match the visible task
    if (visibleContainer) {
      // Scrolled to an adjacent task — show its metrics
      if (statusTokens)
        statusTokens.textContent = visibleContainer.dataset.metricTokens || '';
      if (statusBudget)
        statusBudget.textContent = visibleContainer.dataset.metricBudget || '';
      if (statusSteps)
        statusSteps.textContent = visibleContainer.dataset.metricSteps || '';
    } else {
      // Back on the current (main) task — restore its metrics
      if (statusTokens) statusTokens.textContent = currentTaskMetrics.tokens;
      if (statusBudget) statusBudget.textContent = currentTaskMetrics.budget;
      if (statusSteps) statusSteps.textContent = currentTaskMetrics.steps;
    }
  }

  O.addEventListener('scroll', () => {
    if (_scrollLock) {
      const atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 150;
      if (atBottom) _scrollLock = false;
    }
    updateVisibleTask();
  });
  new MutationObserver(() => {
    if (isRunning) sb();
  }).observe(O, {childList: true, subtree: true, characterData: true});

  // --- Timer ---
  // ``endTs`` (ms since epoch) is the agent's recorded end timestamp
  // for the currently-displayed task.  When the agent has already
  // finished but the frontend joined late (history load) this lets
  // ``_renderTimerTick`` flip the label from "Running …" to
  // "Done (Xm Ys)" without waiting for a live ``task_done`` event.
  // Zero means "no recorded end yet" (still running, or a legacy
  // row that pre-dates endTs persistence).
  let endTs = 0;
  /** Format a "Done (Xm Ys)" label from start/end timestamps (ms). */
  function doneLabelFor(startMs, endMs) {
    const ds = Math.max(0, Math.floor((endMs - startMs) / 1000));
    const dm = Math.floor(ds / 60);
    return 'Done (' + (dm > 0 ? dm + 'm ' : '') + (ds % 60) + 's)';
  }
  /**
   * Format a millisecond duration as ``hh:mm:ss`` (zero-padded).
   * Hours have no upper bound (e.g. ``101:02:03`` is valid for a
   * 101-hour task).  Negative or non-finite inputs clamp to 0.
   * Used by the History sidebar's per-task panel to show the time
   * spent on the task after the cost figure.
   */
  function formatDurationHms(ms) {
    const total = Math.max(0, Math.floor(Number(ms) / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = n => (n < 10 ? '0' + n : String(n));
    return pad(h) + ':' + pad(m) + ':' + pad(s);
  }
  function _renderTimerTick() {
    // If the agent's persisted end timestamp has already passed,
    // surface "Done (Xm Ys)" computed from agent wall-clock
    // (endTs - t0) and shut down the live tick.  Without this
    // branch a chat loaded from history for a task that finished
    // while the client was disconnected would render
    // "Running …" forever.
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
    // Render the first tick immediately so the "Running …" label
    // appears the instant the running state turns on (e.g. when a
    // running task is loaded into a new tab from history).  Without
    // this the statusText stays at "Ready" / blank for up to 1s
    // until the first ``setInterval`` callback fires.
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

  // --- Usage metrics (tokens / budget) in header ---
  function updateUsageMetrics(text) {
    if (!statusTokens || !statusBudget) return;
    const tm = text.match(/Tokens:\s*([\d,]+)\/[\d,]+/);
    const bm = text.match(/Budget:\s*(\$[0-9.]+)\/\$[0-9.]+/);
    const sm = text.match(/Steps:\s*(\d+)\/\d+/);
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

  // --- Refresh history ---
  function resetHistoryPagination() {
    historyOffset = 0;
    historyHasMore = true;
    historyLoading = false;
    historyGeneration++;
  }

  function refreshHistory() {
    if (sidebar.classList.contains('open')) {
      resetHistoryPagination();
      vscode.postMessage({
        type: 'getHistory',
        query: historySearch.value,
        generation: historyGeneration,
      });
    }
  }

  /**
   * Toggle the "KISS Sorcar Server is starting ..." overlay.
   *
   * When the kiss-web daemon socket is NOT yet connected the overlay
   * covers the whole webview and #app is hidden so the user does not
   * see a non-functional tab bar / welcome page.  When connected the
   * overlay is removed and #app becomes visible.
   *
   * Driven by the ``daemonStatus`` message posted from the extension
   * host (see SorcarSidebarView.ts ``connect``/``disconnect`` handlers
   * and the ``ready`` handler).
   */
  function setServerLoading(loading) {
    const overlay = document.getElementById('kiss-server-loading');
    const app = document.getElementById('app');
    if (overlay) overlay.style.display = loading ? '' : 'none';
    if (app) app.style.display = loading ? 'none' : '';
  }

  // --- Main event handler ---
  function handleEvent(ev) {
    const t = ev.type;
    switch (t) {
      case 'daemonStatus':
        // The extension host has told us whether the kiss-web daemon
        // socket is connected.  Hide #app and show the loading overlay
        // until ``connected === true``, then reveal the regular tabs.
        setServerLoading(!ev.connected);
        return;
      case 'notification':
        updateNotification(ev);
        break;
      case 'status': {
        const evTab = findTabByEvt(ev);
        if (evTab) {
          evTab.isRunning = !!ev.running;
        }
        // Anchor the chat webview's "Running …" timer to the
        // agent's TRUE start timestamp (ms since epoch) supplied by
        // the backend — not to the client's Date.now() at the
        // moment this status event arrives.  Without this anchor a
        // chat resumed from history would show "Running 0s" no
        // matter how long the agent has actually been running.
        if (ev.running && typeof ev.startTs === 'number' && ev.startTs > 0) {
          if (evTab) {
            evTab.t0 = ev.startTs;
            // A freshly-running task has no recorded end yet — clear
            // any endTs left over from the tab's previous task.
            evTab.endTs = 0;
          }
          // Anchor the GLOBAL timer only when the event targets the
          // active tab (or carries no tabId).  The daemon broadcasts
          // tab-stamped status events for background tabs and for
          // tabs owned by OTHER VS Code windows; those must not
          // clobber the active tab's running clock (same routing
          // rule as the UI-update gate below).
          if (ev.tabId === undefined || ev.tabId === activeTabId) {
            t0 = ev.startTs;
            // A freshly-running task on this tab has no recorded end
            // yet — clear any stale ``endTs`` captured from a prior
            // task's ``task_events`` so the timer doesn't immediately
            // jump to "Done".
            endTs = 0;
          }
        }
        // Update UI only when the event targets the active tab (or no
        // tabId).  A tabId that resolves to NO local tab belongs to a
        // tab in another window (the daemon broadcasts tab-stamped
        // events to every connected client) — it must not flip this
        // window's running state.
        if (ev.tabId === undefined || ev.tabId === activeTabId) {
          setRunningState(ev.running);
          // Refresh chevron-driven visibility so panels of the now-running
          // task become visible even if the chevron is collapsed.  Needed
          // after resuming a running task from history: the prior
          // ``task_events`` replay called applyChevronState() while
          // isRunning was still false, marking every replayed panel
          // chv-hidden.  Re-applying it here (with isRunning=true) unhides
          // panels of the running task per applyChevronState's
          // ``inRunning`` branch, so subsequent live events from the
          // re-attached agent are visible immediately.
          if (ev.running) {
            const aTab = getTab(activeTabId);
            if (aTab) {
              applyChevronState(
                !!aTab.panelsExpandedMap[currentTaskName],
                currentTaskName,
              );
            }
          }
        }
        renderTabBar();
        refreshHistory();
        break;
      }
      case 'models':
        allModels = ev.models || [];
        if (ev.selected) {
          // Race-fix: propagate the new default into every tab whose
          // ``selectedModel`` still mirrors the prior default (or is
          // empty / ``"No model"`` from the launch IIFE).  Tabs where
          // the user explicitly picked a different model are
          // preserved.  Without this, ``restoreTab`` reverts the
          // picker to the stale launch-time value as soon as the user
          // switches tabs.
          const _prevSelected = selectedModel;
          selectedModel = ev.selected;
          modelName.textContent = ev.selected;
          tabs.forEach(t => {
            const cur = t.selectedModel || '';
            if (cur === '' || cur === 'No model' || cur === _prevSelected) {
              t.selectedModel = ev.selected;
            }
          });
        }
        renderModelList('');
        break;
      case 'configData':
        populateConfigForm(ev.config || {}, ev.apiKeys || {});
        break;
      case 'history':
        renderHistory(ev.sessions || [], ev.offset || 0, ev.generation || 0);
        break;
      case 'frequentTasks':
        renderFrequentTasks(ev.tasks || []);
        break;
      case 'files': {
        // Staleness guard (mirrors the ``ghost`` handler's
        // ``ev.query === inp.value`` check): the populated reply for
        // a cache miss arrives asynchronously after a background
        // directory scan — potentially seconds later — so only render
        // it while the user is still typing the @-mention it answers.
        // Without this, a late reply re-opened the picker over the
        // input (with acIdx = 0) after the user had deleted the
        // mention, and the phantom picker swallowed the next Enter.
        const filesCtx = getAtCtx();
        if (!filesCtx) {
          hideAC();
          break;
        }
        if (ev.prefix !== undefined && ev.prefix !== filesCtx.query) {
          break; // reply ranked for an older prefix — a fresh one is coming
        }
        renderAutocomplete(ev.files || []);
        break;
      }
      case 'askUser': {
        const askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
        const askTab = getTab(askTabId);
        if (!askTab) break;
        askTab.askPendingQuestion = ev.question || '';
        if (askTab.id !== activeTabId) {
          switchToTab(askTab.id);
        }
        showAskForTab(askTab);
        break;
      }
      case 'askUserDone': {
        const askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
        const askTab = getTab(askTabId);
        if (!askTab) break;
        clearAskForMatchingChatTabs(askTab);
        break;
      }
      case 'error':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
        addError(ev.text);
        break;
      case 'notice':
        // Informational (non-error) server notification — e.g. the
        // remote webapp's runUpdate acknowledgement.  Counterpart of
        // the extension's vscode.window.showInformationMessage().
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
        addNotice(ev.text);
        break;
      case 'warning': {
        // Backend warning the user must see — e.g. the worktree
        // agent's stash-pop failure or merge-conflict warning
        // (WorktreeSorcarAgent._flush_warnings broadcasts
        // {type: 'warning', message: ...}).
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Route to the owning BACKGROUND tab's saved fragment
          // (mirrors the default display-event route) so the warning
          // is visible when the user switches to that tab.  Unknown
          // tab ids (other VS Code windows) are dropped as before.
          const bgWarnTab = findTabByEvt(ev);
          if (bgWarnTab) processOutputEventForBgTab(ev, bgWarnTab);
          break;
        }
        addWarning(ev.message || ev.text || '');
        break;
      }
      case 'clear': {
        const clearTab =
          ev.tabId !== undefined ? getTab(ev.tabId) : getTab(activeTabId);
        if (clearTab) {
          clearTab.lastTaskFailed = false;
          clearTab.hasRunTask = true;
        }
        if (ev.chat_id && clearTab) {
          clearTab.backendChatId = ev.chat_id;
          // Pin the tab's ``workDir`` the moment a chat-id of a real
          // persisted task is bound to it.  Once bound, a later
          // settings-panel change to ``configWorkDir`` MUST NOT shift
          // this tab's effective work_dir (INVARIANTS.md → Tabs &
          // chat webview).  ``workDirForTab`` would otherwise fall
          // back to the daemon-global ``configWorkDir`` and route
          // follow-up commands (submit, autocommitAction, …) to the
          // wrong repo.
          if (!clearTab.workDir && configWorkDir) {
            clearTab.workDir = configWorkDir;
          }
          persistTabState();
        }
        const evTabId = ev.tabId;
        if (evTabId === undefined || evTabId === activeTabId) {
          clearOutput();
          resetOutputState();
          showSpinner();
        } else if (clearTab) {
          // Reset background tab streaming state so the first thinking
          // event of the new task creates a fresh Thoughts panel.
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
      case 'ensureChat':
        if (tabs.length === 0) {
          createNewTab();
        }
        break;
      case 'showWelcome': {
        const swTabId = ev.tabId || activeTabId;
        const swTab = getTab(swTabId);
        if (swTab) {
          // Update model picker to last user-picked model from DB
          if (ev.model) {
            swTab.selectedModel = ev.model;
            if (swTabId === activeTabId) {
              selectedModel = ev.model;
              if (modelName) modelName.textContent = ev.model;
            }
          }
          if (swTabId === activeTabId) {
            clearOutput();
            resetOutputState();
            if (welcome) {
              welcome.style.display = '';
              O.appendChild(welcome);
              refreshWelcomeLayout();
            }
          } else {
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
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
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
        // Do not invoke the scroll-to-bottom helper here: rendering
        // "Suggested next" must not force the chat to scroll to the bottom
        // (the user may have scrolled up to read earlier content).  The
        // replay path in replayEventsInto also does not scroll for this
        // event — the two code paths are now consistent.
        break;
      }
      case 'tasks_updated':
        refreshHistory();
        vscode.postMessage({type: 'getInputHistory'});
        break;

      case 'taskDeleted': {
        // Backend deleted a task from the history table.  For every
        // open tab that displays the deleted task or its chat:
        //   * remove any .adjacent-task[data-task-id="<id>"] block
        //     (whether it lives in the active output area O or in
        //     this tab's detached outputFragment),
        //   * close the tab if its current (header) task was the
        //     deleted one, or if the chat now has no remaining
        //     tasks left in the database.
        const tdChatId = ev.chatId;
        const tdTaskId = ev.taskId;
        const tdHasMore = !!ev.chatHasMoreTasks;
        if (tdTaskId === undefined || tdTaskId === null) break;
        const tdSelector =
          '.adjacent-task[data-task-id="' + String(tdTaskId) + '"]';
        // Iterate over a snapshot since closeTab() mutates `tabs`.
        const tdSnapshot = tabs.slice();
        tdSnapshot.forEach(t => {
          if (!t || t.backendChatId !== tdChatId) return;
          // Remove the matching adjacent-task block from active DOM
          // when this is the active tab.
          if (t.id === activeTabId && O) {
            const liveBlock = O.querySelector(tdSelector);
            if (liveBlock && liveBlock.parentNode)
              liveBlock.parentNode.removeChild(liveBlock);
          }
          // Remove the matching adjacent-task block from the saved
          // outputFragment (used by inactive tabs to hold their DOM).
          if (t.outputFragment) {
            const fragBlock = t.outputFragment.querySelector(tdSelector);
            if (fragBlock && fragBlock.parentNode)
              fragBlock.parentNode.removeChild(fragBlock);
          }
          // Close the tab when the deleted task was the tab's current
          // (header) task, or when the underlying chat is now empty.
          const isCurrent =
            t.currentTaskId !== undefined &&
            t.currentTaskId !== null &&
            String(t.currentTaskId) === String(tdTaskId);
          if (isCurrent || !tdHasMore) closeTab(t.id);
        });
        break;
      }

      case 'task_events': {
        const teTabId = ev.tabId || activeTabId;
        const teTab = getTab(teTabId);
        if (ev.chat_id && teTab) {
          teTab.backendChatId = ev.chat_id;
          // Pin the tab's ``workDir`` when a chat-id of a real
          // persisted task is bound (INVARIANTS.md → Tabs & chat
          // webview).  ``extra.work_dir`` (parsed further down) takes
          // priority and may overwrite this value with the task's
          // recorded directory; this fallback only kicks in for
          // replays whose ``extra`` is missing ``work_dir`` (older
          // rows), keeping the tab pinned to whatever ``configWorkDir``
          // was at bind time instead of leaking later settings-panel
          // changes through ``workDirForTab``'s fallback.
          if (!teTab.workDir && configWorkDir) {
            teTab.workDir = configWorkDir;
          }
          persistTabState();
        }
        // Track the task_id of the currently displayed task so a later
        // 'taskDeleted' broadcast can decide whether to close this tab.
        if (teTab && ev.task_id !== undefined && ev.task_id !== null)
          teTab.currentTaskId = ev.task_id;
        // Non-active tab: render into a document fragment without touching the DOM.
        // When teTabId targets a different tab but that tab hasn't been
        // created yet (teTab is null), silently drop the event so
        // sub-agent events never fall through to the active (parent) tab.
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
              // Do NOT stamp ``teTab.selectedModel`` from
              // ``bgExtra.model``: a background tab whose chat is
              // being LOADED from history would otherwise capture
              // the loaded task's historical model snapshot into
              // its per-tab state — and ``restoreTab`` would then
              // promote it to the live ``selectedModel`` when the
              // user later switches to that tab, silently making
              // the next submit run with the loaded task's old
              // model instead of the user's current global one.
              // The backend (``_extra_for_replay`` in server.py)
              // already strips ``model`` from history replays; this
              // guard mirrors the active-tab path so the rule holds
              // even if a future replay path forgets to strip.
              if (bgExtra.work_dir) teTab.workDir = bgExtra.work_dir;
              // Capture the agent's persisted start / end timestamps
              // so switching to this background tab renders
              // "Running …" / "Done (Xm Ys)" from agent wall-clock.
              if (typeof bgExtra.startTs === 'number' && bgExtra.startTs > 0)
                teTab.t0 = bgExtra.startTs;
              if (typeof bgExtra.endTs === 'number' && bgExtra.endTs > 0) {
                teTab.endTs = bgExtra.endTs;
                // Finished task: pre-render the done label so a later
                // switch to this tab shows the agent's end - start
                // duration (mirrors _renderTimerTick's Done branch).
                if (teTab.t0) {
                  teTab.statusTextContent = doneLabelFor(teTab.t0, teTab.endTs);
                  teTab.statusTextColor = 'var(--green)';
                }
              }
            } catch (_e) {
              /* ignore */
            }
          }
          const frag = document.createDocumentFragment();
          replayEventsInto(frag, ev.events || [], {
            onFollowupClick: function (text) {
              inp.value = text;
              syncClearBtn();
              inp.focus();
            },
          });
          teTab.outputFragment = frag;
          teTab.welcomeVisible = false;
          // Count steps from replayed events
          const bgSteps = countReplayedSteps(ev.events || []);
          if (bgSteps > 0) teTab.statusStepsText = 'Steps: ' + bgSteps;
          break;
        }
        // Active tab: render directly into the DOM
        if (ev.task) {
          currentTaskName = ev.task;
          // Keep currentTaskId in sync with the active task whose
          // events are reaching this tab so that adjacent scrolling
          // queries the right DB row id.
          if (ev.task_id !== undefined && ev.task_id !== null)
            currentTaskId = ev.task_id;
          resetAdjacentState(); // sets oldest/newest to current task
          setTaskText(ev.task);
          if (welcome) {
            welcome.style.display = 'none';
            refreshWelcomeLayout();
          }
          updateActiveTabTitle(ev.task);
        }
        if (ev.extra) {
          try {
            const extra = JSON.parse(ev.extra);
            // Capture the agent's persisted start / end timestamps
            // (ms since epoch) so the chat webview's "Running …" /
            // "Done (Xm Ys)" header can be computed from agent
            // wall-clock — see ``_renderTimerTick``.  ``endTs > 0``
            // means the task has already ended; the timer-tick
            // branch will flip the label to "Done (…)" as soon as
            // ``Date.now() >= endTs`` even when no live
            // ``task_done`` event arrives (history-resume case).
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
            // Do NOT clobber the live toggle / model state from this
            // task's historical ``extra.is_worktree`` /
            // ``extra.is_parallel`` / ``extra.auto_commit_mode`` /
            // ``extra.model``: those values are a SNAPSHOT of what
            // the toggles read AT THE TIME this task ran, not the
            // user's CURRENT global settings.  The live toggles in
            // this webview already mirror ``~/.kiss/config.json``
            // (kept in sync via ``updateSetting`` / ``configData``);
            // overwriting them with the loaded task's stale snapshot
            // would silently make the NEXT task submitted in this
            // tab run with the loaded task's old settings instead of
            // whatever the user just picked globally.  The backend's
            // ``_extra_for_replay`` (server.py) defensively strips
            // these keys from the broadcast ``extra`` for the same
            // reason — this guard keeps the bug from creeping back
            // in if an older/local backend still ships them.
          } catch (_e) {
            /* ignore malformed extra */
          }
        }
        if (_demoActive && window._demoApi && window._demoApi.resolveEvents) {
          window._demoApi.resolveEvents(ev.events || []);
        } else {
          replayTaskEvents(ev.events || []);
        }
        break;
      }
      case 'adjacent_task_events':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
        renderAdjacentTask(ev.direction, ev.task, ev.events || [], ev.task_id);
        break;
      case 'setTaskText': {
        const stt = (ev.text || '').trim();
        if (ev.tabId === undefined || ev.tabId === activeTabId) {
          if (stt) {
            currentTaskName = stt;
            // setTaskText fires before a task_id is assigned (the row
            // is created later by taskExecuted), so clear the id —
            // adjacent scrolling stays disabled until taskExecuted
            // delivers the real row id.
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
          // Update background tab's saved title without touching active tab
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
      case 'triggerStop':
        vscode.postMessage({type: 'stop', tabId: activeTabId});
        break;
      case 'appendToInput':
        if (ev.text) {
          inp.value = inp.value ? inp.value + '\n' + ev.text : ev.text;
          inp.dispatchEvent(new Event('input', {bubbles: true}));
        }
        focusInputWithRetry();
        break;
      case 'focusInput':
        focusInputWithRetry();
        break;

      case 'measureSize':
        // The extension is asking how wide the sidebar webview is so it
        // can iteratively resize the secondary side bar to ~1/3 of the
        // VS Code window.  window.innerWidth gives the webview iframe's
        // width (= sidebar width); screen.availWidth is the best proxy
        // for the host VS Code window width that the sandboxed webview
        // can read (works correctly when VS Code is maximized, which is
        // the common case on first install).
        try {
          vscode.postMessage({
            type: 'sizeReport',
            innerWidth: window.innerWidth || 0,
            screenWidth:
              (window.screen && window.screen.availWidth) ||
              window.innerWidth ||
              0,
          });
        } catch (_e) {
          /* ignored */
        }
        break;

      case 'updateSetting': {
        const sKey = ev.key;
        const sVal = ev.value;
        if (sKey === 'is_parallel' && parallelToggleBtn) {
          parallelToggleBtn.checked = !!sVal;
        } else if (sKey === 'is_worktree' && worktreeToggleBtn) {
          worktreeToggleBtn.checked = !!sVal;
        } else if (sKey === 'model' && typeof sVal === 'string') {
          // Update #model-name (not #model-btn.textContent, which would
          // destroy the button's SVG icon).
          if (modelName) modelName.textContent = sVal;
          selectedModel = sVal;
        } else if (sKey === 'max_budget') {
          // Budget updated server-side; UI may show in config panel
        } else if (sKey === 'use_web_browser') {
          // Web browser setting updated server-side
        } else if (sKey === 'demo_mode' && demoToggleBtn) {
          demoMode = !!sVal;
          demoToggleBtn.checked = demoMode;
        } else if (sKey === 'auto_commit') {
          // Auto-commit triggered server-side
        } else if (sKey === 'auto_commit_mode' && autocommitToggleBtn) {
          autocommitToggleBtn.checked = !!sVal;
        } else if (sKey === 'custom_endpoint' && typeof sVal === 'string') {
          const epEl = document.getElementById('cfg-custom-endpoint');
          if (epEl) epEl.value = sVal;
        } else if (sKey === 'custom_api_key' && sVal === true) {
          // Custom API key updated server-side; mask in UI
          const akEl = document.getElementById('cfg-custom-api-key');
          if (akEl && !akEl.value) akEl.value = '••••••••';
        } else if (sKey === 'custom_headers' && sVal === true) {
          // Custom headers updated server-side
        } else if (sKey.endsWith('_api_key') && sVal === true) {
          // API key updated server-side; map param name to env var
          const envMap = {
            gemini_api_key: 'GEMINI_API_KEY',
            openai_api_key: 'OPENAI_API_KEY',
            anthropic_api_key: 'ANTHROPIC_API_KEY',
            together_api_key: 'TOGETHER_API_KEY',
            openrouter_api_key: 'OPENROUTER_API_KEY',
            zai_api_key: 'ZAI_API_KEY',
            moonshot_api_key: 'MOONSHOT_API_KEY',
          };
          const envVar = envMap[sKey];
          if (envVar) {
            const keyEl = document.getElementById('cfg-key-' + envVar);
            if (keyEl && !keyEl.value) keyEl.value = '••••••••';
          }
        }
        break;
      }

      case 'inputHistory':
        histCache = ev.tasks || [];
        if (histIdx < 0) histIdx = -1;
        break;
      case 'ghost':
        if (ev.suggestion && ev.query === inp.value) {
          updateGhost(ev.suggestion);
        }
        break;
      case 'completions': {
        // Staleness guard (mirrors the ``ghost`` and ``files``
        // handlers): the populated reply arrives asynchronously after
        // the backend worker drains its queue, so only render while
        // the user's input still matches the query this reply
        // answers.  ``ev.query`` may be ``undefined`` on older
        // backends (back-compat) — then we skip the check.
        if (ev.query !== undefined && ev.query !== inp.value) {
          break;
        }
        renderCompletions(ev.completions || []);
        break;
      }

      case 'merge_data': {
        const mdEl = renderMergeData(ev);
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab: append to saved output fragment
          const bgMdTab = getTab(ev.tabId);
          if (bgMdTab && bgMdTab.outputFragment) {
            bgMdTab.outputFragment.appendChild(mdEl);
          }
          break;
        }
        O.appendChild(mdEl);
        // Highlight the first hunk by default; the server confirms or
        // updates the selection on each subsequent prev/next/accept/reject
        // via merge_nav.
        setCurrentMergeHunk(mdEl, 0, 0);
        scrollHunkIntoView(mdEl, 0, 0);
        collapseOlderPanels();
        break;
      }
      case 'merge_started':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab's merge: mark it and auto-switch so the user
          // sees the merge/diff interface immediately.
          const bgMergeTab = getTab(ev.tabId);
          if (bgMergeTab) {
            bgMergeTab.isMerging = true;
            switchToTab(ev.tabId);
          }
          break;
        }
        isMerging = true;
        showMergeToolbar((ev && ev.tabId) || activeTabId);
        updateInputDisabled();
        sb();
        break;
      case 'merge_ended':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const mrt2 = getTab(ev.tabId);
          if (mrt2) {
            mrt2.isMerging = false;
            mrt2.mergeToolbarEl = null;
          }
          break;
        }
        isMerging = false;
        hideMergeToolbar();
        updateInputDisabled();
        break;
      case 'merge_nav': {
        // Apply resolved-hunk styles + scroll/highlight the current hunk.
        // The most recent merge_data panel for the targeted tab owns the
        // hunk DOM; for the active tab it's in O, for a background tab
        // it's inside that tab's outputFragment.
        const navTabId = ev.tabId || activeTabId;
        const navHost =
          navTabId === activeTabId
            ? O
            : (getTab(navTabId) || {}).outputFragment;
        if (!navHost) break;
        // Update the merge toolbar's remaining-hunk count ONLY when
        // the event targets the active tab.  The daemon broadcasts
        // tab-stamped events to every connected client, and
        // '.merge-toolbar-title' is the toolbar of THIS window's
        // active tab — a merge_nav for a tab in another window (or a
        // background tab here) must not overwrite its counts.
        if (navTabId === activeTabId) {
          const mergeTitle = document.querySelector('.merge-toolbar-title');
          if (mergeTitle && ev.remaining !== undefined) {
            mergeTitle.textContent =
              'Review Changes (' +
              ev.remaining +
              '/' +
              ev.total +
              ' remaining)';
          }
        }
        // Find the most recent merge-info panel that contains hunks.
        const mergePanels = navHost.querySelectorAll('.merge-info');
        const mergePanel = mergePanels[mergePanels.length - 1];
        if (!mergePanel) break;
        applyMergeResolutions(mergePanel, ev.resolved || []);
        if (ev.cur && ev.cur.fi !== undefined && ev.cur.hi !== undefined) {
          setCurrentMergeHunk(mergePanel, ev.cur.fi, ev.cur.hi);
          scrollHunkIntoView(mergePanel, ev.cur.fi, ev.cur.hi);
        } else {
          // No remaining hunks: clear .current highlight.
          mergePanel.querySelectorAll('.merge-hunk.current').forEach(el => {
            el.classList.remove('current');
          });
        }
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
          // Background tab: create bar and save on tab state for restoreTab
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
          // Background tab: clear saved bar and append result to fragment
          const bgWrTab = getTab(ev.tabId);
          if (bgWrTab) {
            bgWrTab.worktreeBarEl = null;
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
      case 'autocommit_prompt':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab: create bar and save on tab state for restoreTab
          const bgAcTab = getTab(ev.tabId);
          if (bgAcTab) {
            bgAcTab.autocommitBarEl = createAutocommitBar(ev);
          }
          break;
        }
        showAutocommitActions(ev);
        break;
      case 'autocommit_done':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab: clear saved bar and append result to fragment
          const bgAdTab = getTab(ev.tabId);
          if (bgAdTab) {
            bgAdTab.autocommitBarEl = null;
            if (bgAdTab.outputFragment) {
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
        setReady(
          'Done (' + (em > 0 ? em + 'm ' : '') + (el % 60) + 's)',
          ev.tabId,
          ev.startTs,
          ev.endTs,
        );
        focusFinishedTab(ev.tabId);
        break;
      }
      case 'task_error':
      case 'task_interrupted':
      case 'task_stopped': {
        markTabDone(ev.tabId, true);
        // ``task_interrupted`` is a graceful server shutdown / restart
        // (e.g. an extension update restarting the daemon), distinct
        // from the user clicking "Stop" (``task_stopped``) and from a
        // genuine failure (``task_error``).
        const label =
          t === 'task_error'
            ? 'Error'
            : t === 'task_interrupted'
              ? 'Interrupted'
              : 'Stopped';
        setReady(label, ev.tabId, ev.startTs, ev.endTs);
        focusFinishedTab(ev.tabId);
        break;
      }
      case 'new_tab': {
        // Backend → frontend request to open a fresh chat tab and
        // resume an existing task into it.  ``task_id`` is the
        // backend's identity for the task; the frontend allocates a
        // tab id (frontend-only concept) and then posts
        // ``resumeSession`` back to the backend.  The server's
        // ``_cmd_resume_session`` handler supports a task-id-only
        // resume (no ``chatId`` required).
        //
        // Sub-agent ``new_tab`` events carry ``parent_tab_id``.  The
        // backend broadcasts them to ALL connected webviews (no
        // per-client routing for global system events), so a webview
        // that doesn't own the parent run_parallel tab must NOT
        // materialise a phantom sub-agent tab.  Skip when the parent
        // tab is not present locally.
        if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
          break;
        if (ev.task_id === undefined || ev.task_id === null) break;
        const parentTabBeforeNew = ev.parent_tab_id || '';
        let subAgentTabId;
        if (parentTabBeforeNew) {
          // Sub-agent path: build the new tab in the BACKGROUND so the
          // user's foreground tab is never disturbed.  Calling
          // ``createNewTab`` here would (1) flip ``activeTabId`` to
          // the new tab, painting its empty welcome screen for one
          // frame before a follow-up ``switchToTab`` reverted it,
          // (2) post a spurious ``newChat`` for what is really the
          // parent's chat session, (3) post ``getWelcomeSuggestions``
          // for a tab that will never show a welcome screen, and (4)
          // steal keyboard focus from the parent the user is typing
          // in.  ``createBackgroundSubagentTab`` does none of that.
          const subTab = createBackgroundSubagentTab(parentTabBeforeNew);
          subAgentTabId = subTab.id;
        } else {
          // Defensive path: a ``new_tab`` event with no
          // ``parent_tab_id`` is not produced by any current backend
          // emitter, but if a future code path emits one we keep the
          // legacy "create + activate" behaviour for it.
          createNewTab();
          subAgentTabId = activeTabId;
        }
        vscode.postMessage({
          type: 'resumeSession',
          taskId: ev.task_id,
          tabId: subAgentTabId,
        });
        break;
      }
      case 'openSubagentTab': {
        // ``openSubagentTab`` is broadcast verbatim to ALL connected
        // webviews (no per-client routing).  A webview whose local
        // ``tabs[]`` does not contain the ``parent_tab_id`` does not
        // own the parent run_parallel tab and must NOT materialise a
        // phantom sub-agent tab — otherwise sub-tabs leak across
        // unrelated chats / chat_ids.
        if (ev.parent_tab_id && !tabs.find(t => t.id === ev.parent_tab_id))
          break;
        // Trim so trailing newlines from the backend description don't
        // bleed into taskPanelHTML and resurface in the user's clipboard
        // when they copy-select the task panel.
        const subDesc = (ev.description || 'Sub-agent').trim();
        // Include the 1-based task index in the title for live
        // spawns so tabs whose descriptions share a long common
        // prefix (e.g. "Research and summarize: ...") stay visually
        // distinct in the truncated tab bar.  History-reopened
        // sub-agent rows have no ``taskIndex`` (the persisted
        // payload is just ``{parent_task_id}``); they fall back to
        // the bare description — the purple .subagent-tab accent
        // already makes them unambiguously a sub-agent tab.
        const subIdx =
          typeof ev.taskIndex === 'number' ? ev.taskIndex + 1 : null;
        const titlePrefix = subIdx !== null ? subIdx + '. ' : '';
        const title = titlePrefix + subDesc.substring(0, 40);
        // ``ev.parent_tab_id`` is set by the sorcar/server emitters;
        // for the chat_sorcar broadcast path the daemon stamps
        // ``ev.tabId`` with the subscriber's (= parent's) tab id, so we
        // fall back to that.
        const parentId = ev.parent_tab_id || ev.tabId || '';
        // Idempotent: if a tab with the same id already exists, update
        // it in place rather than pushing a duplicate.  Defends against
        // accidental duplicate events from the backend.
        let subTab = getTab(ev.tab_id);
        const needsPlacement = !subTab || !subTab.isSubagentTab;
        if (!subTab) {
          subTab = makeTab(title);
          subTab.id = ev.tab_id;
        } else {
          subTab.title = title;
        }
        if (needsPlacement) {
          // First conversion of this tab into a sub-agent tab: anchor
          // it immediately to the right of its parent tab so restored
          // and history-reopened fan-outs mirror the live layout.
          placeSubagentTabAfterParent(subTab, parentId);
        }
        subTab.isSubagentTab = true;
        // Remember the parent → child relationship so closing the parent
        // tab can recursively close every (nested) sub-agent tab it
        // spawned.
        if (parentId && parentId !== subTab.id) {
          subTab.parentTabId = parentId;
        }
        // ``isDone`` is set by the backend for history-loaded sub-agent
        // tabs whose execution already completed — without this flag
        // the tab would forever pulse the running ◉ indicator (no
        // ``subagentDone`` event arrives for an already-finished
        // sub-agent).  Default to "running" for fresh launches.
        const subDone = !!ev.isDone;
        subTab.isDone = subDone;
        subTab.isRunning = !subDone;
        subTab.taskPanelHTML = subDesc;
        subTab.taskPanelVisible = true;
        renderTabBar();
        // If the backend converted the active tab into a sub-agent tab
        // (e.g. the user clicked a sub-agent row in the history panel,
        // which created a fresh chat tab that ``_replay_session`` then
        // flips via ``openSubagentTab``), hide the input textbox + the
        // buttons below it.  ``restoreTab`` already enforces this rule
        // when *switching* tabs, but the tab-switch ran BEFORE
        // ``isSubagentTab`` was set, so the input bar is still showing
        // for the active tab at this point.
        if (subTab.id === activeTabId) {
          if (inputContainer) inputContainer.style.display = 'none';
          // History-load case: the new tab was created and switched
          // to before this handler fired, so ``restoreTab`` initialised
          // the global running state from the brand-new tab's default
          // ``isRunning=false``.  Sync the global state to the
          // sub-agent's actual state now so a still-running sub-agent
          // tab loaded from history shows the same "Running" status,
          // timer and chevron-visible panels as the freshly-launched
          // sub-agent tab the user originally clicked through to.
          setRunningState(subTab.isRunning);
          if (subTab.isRunning) {
            applyChevronState(
              !!subTab.panelsExpandedMap[currentTaskName],
              currentTaskName,
            );
          }
        }
        persistTabState();
        break;
      }
      case 'subagentDone': {
        const doneTab = getTab(ev.tab_id);
        if (doneTab) {
          doneTab.isDone = true;
          doneTab.isRunning = false;
          renderTabBar();
          // Mirror the regular task's status:false handling when the
          // finished sub-agent tab is the one the user is viewing.
          // Without this the status header stays at "Running …" and
          // the timer keeps ticking on a tab whose sub-agent has
          // already completed — diverging from the fresh-launch path
          // where ``restoreTab(setRunningState(false))`` would
          // eventually run when the user clicks back to the tab.
          if (doneTab.id === activeTabId) {
            setRunningState(false);
          }
          persistTabState();
        }
        break;
      }
      default:
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgTab = findTabByEvt(ev);
          if (bgTab) processOutputEventForBgTab(ev, bgTab);
          break;
        }
        // Defensive guard: a misrouted result / usage_info event
        // whose ``taskId`` does not match the active tab's
        // ``currentTaskId`` would otherwise stamp a sub-agent's
        // Result panel (tokens, cost, summary) onto the parent
        // tab's DOM after the parent's own Result + SUGGESTED NEXT.
        // The wire-level path is supposed to fan such events out
        // ONLY to the sub-agent's subscriber tab, but any future
        // regression (or any third path that broadcasts without
        // tabId stamping) would surface as duplicate Result panels
        // in the parent.  Drop the event here so the symptom never
        // reaches the user.
        // Keep ``currentTaskId`` in sync with the active task whose
        // events are reaching this tab.  ``currentTaskId`` is set by
        // ``task_events`` when the user loads a task from history,
        // but no ``task_events`` replay fires for a freshly-submitted
        // task — so without this adoption step a tab that previously
        // loaded an OLD task would keep its stale ``currentTaskId``
        // and the misroute guard below would drop the NEW task's
        // terminal ``result`` / ``usage_info`` events, leaving the
        // chat frozen at the last ``tool_call(finish)`` panel.  Only
        // events routed to the active tab reach this branch (the
        // bg-tab early-return above already peels sub-agent events
        // off), so updating ``currentTaskId`` here is safe and does
        // not weaken the cross-tab defense.  Skip ``result`` /
        // ``usage_info`` themselves so a genuinely misrouted result
        // can still be dropped by the guard below.
        if (
          ev.taskId !== undefined &&
          ev.taskId !== null &&
          ev.type !== 'result' &&
          ev.type !== 'usage_info'
        ) {
          const adoptTab = getTab(activeTabId);
          if (
            adoptTab &&
            String(adoptTab.currentTaskId) !== String(ev.taskId)
          ) {
            adoptTab.currentTaskId = ev.taskId;
            // Keep the global currentTaskId (used by adjacent
            // scrolling to identify the boundary task by DB row id)
            // in sync with the active tab's adopted id.
            currentTaskId = ev.taskId;
            // Re-seed oldest/newest boundary ids whenever none has
            // been loaded yet (no .adjacent-task containers), so a
            // fresh task immediately becomes the scroll anchor.
            if (oldestLoadedTaskId === null && newestLoadedTaskId === null) {
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
        sb();
        break;
    }
  }

  function updateInputDisabled() {
    // Only block input during merge.  While a task is running the
    // user can still type — ``sendMessage`` then forwards the prompt
    // as an ``appendUserMessage`` so it gets injected into the live
    // agent's conversation before its next model call.
    const blocked = isMerging;
    inp.disabled = blocked;
    sendBtn.disabled = blocked;
    if (blocked) {
      clearGhost();
      hideAC();
    }
  }

  function setRunningState(running) {
    isRunning = running;
    sendBtn.style.display = 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';

    updateInputDisabled();
    if (running) {
      startTimer();
      // Show the wait-spinner whenever the UI flips to running.
      // Without this, the spinner is only (re)started inside
      // ``processOutputEvent`` when an event arrives on the active
      // tab.  During ``run_parallel`` the parent agent emits one
      // ``tool_call`` event and then blocks while sub-agents run —
      // meanwhile each sub-agent's ``new_tab`` broadcast causes the
      // frontend to ``createNewTab`` (which calls
      // ``setRunningState(false)`` + ``removeSpinner`` on the new
      // sub-tab) and ``switchToTab`` back to the still-running parent
      // (calling ``setRunningState(true)`` here).  Without
      // ``showSpinner`` in this branch the parent tab is left with a
      // cancelled timer and no visible spinner for the entire
      // duration of the parallel fan-out.  Calling ``showSpinner``
      // here makes the spinner consistent across (a) task start
      // (``status running:true``), (b) tab switch back to a running
      // tab, and (c) ``run_parallel`` sub-agent spawn/switch-back.
      showSpinner();
    } else {
      // Safety net: ensure the timer always stops when the running
      // state flips to false.  Without this, if a ``status: running:
      // false`` event arrives without a matching ``task_done`` (e.g.
      // an ill-formed task_done with a non-matching tabId), the
      // header keeps showing "Running …" forever.
      stopTimer();
      removeSpinner();
      if (statusText.textContent.startsWith('Running')) {
        // Render the done duration from the agent's wall-clock
        // anchors when they are known; plain "Done" only as a
        // legacy fallback (no recorded timestamps).
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
    }
  }

  /**
   * Auto-switch the active tab to ``tabId`` after the task running in
   * that tab has just ended (task_done / task_error / task_stopped /
   * task_interrupted).  Without this the user, who may have moved to
   * a different chat tab while waiting for the background task to
   * finish, would have to manually click back to see the result; the
   * product contract is that the webview MUST switch to the tab
   * whose task just completed so the result panel is immediately
   * visible.
   *
   * Silently a no-op when:
   *
   *   * ``tabId`` is missing (legacy/global events without a
   *     per-tab identity — there is no specific tab to focus),
   *   * the tab is not present in this webview's ``tabs`` array
   *     (events are broadcast to every connected client; a webview
   *     that does not own the tab must ignore the focus request),
   *   * the tab is already the active tab (no-op; ``switchToTab``
   *     also short-circuits, but we filter here for clarity).
   */
  function focusFinishedTab(tabId) {
    if (tabId === undefined || tabId === null) return;
    if (tabId === activeTabId) return;
    if (!getTab(tabId)) return;
    switchToTab(tabId);
  }

  function setReady(label, tabId, doneStartTs, doneEndTs) {
    // Mark the tab as no longer running.  Keep (and refine) the tab's
    // ``t0`` / ``endTs`` anchors instead of discarding them: every tab
    // must be able to re-render its done duration as ``endTs - t0``
    // (agent wall-clock) whenever the user switches back to it.
    const hasStart = typeof doneStartTs === 'number' && doneStartTs > 0;
    const hasEnd = typeof doneEndTs === 'number' && doneEndTs > 0;
    let doneTab = null;
    if (tabId !== undefined) {
      doneTab = getTab(tabId);
      if (doneTab) {
        doneTab.isRunning = false;
        if (hasStart) doneTab.t0 = doneStartTs;
        doneTab.endTs = hasEnd ? doneEndTs : Date.now();
        // Persist the final label + colour so restoreTab paints them
        // when the user switches back to a tab that finished while it
        // was in the background.
        doneTab.statusTextContent = label || 'Ready';
        doneTab.statusTextColor = 'var(--green)';
      }
    }
    // Update UI only if the event targets the active tab (or no tabId)
    if (tabId === undefined || tabId === activeTabId) {
      if (hasStart) t0 = doneStartTs;
      // Record the end timestamp so the timer-tick / tab-switch logic
      // renders "Done (Xm Ys)" from agent wall-clock.  The next task
      // on this tab resets both anchors (submit path and the
      // ``status running:true`` handler clear ``endTs`` and re-anchor
      // ``t0``).
      endTs = hasEnd ? doneEndTs : Date.now();
      setRunningState(false);
      stopTimer();
      removeSpinner();
      statusText.textContent = label || 'Ready';
      inp.focus();
    }
    renderTabBar();
  }

  function addError(text) {
    const div = mkEl('div', 'ev tr err');
    div.innerHTML = '<strong>Error:</strong> ' + esc(text);
    O.appendChild(div);
    sb();
  }

  /** Render an informational server notice (green-tinted banner). */
  function addNotice(text) {
    const div = mkEl('div', 'ev tr note');
    div.innerHTML = '<strong>Note:</strong> ' + esc(text);
    O.appendChild(div);
    sb();
  }

  /** Render a backend warning (amber-tinted banner). */
  function addWarning(text) {
    const div = mkEl('div', 'ev tr warn');
    div.innerHTML = '<strong>Warning:</strong> ' + esc(text);
    O.appendChild(div);
    sb();
  }

  // --- Remote URL (dynamic) ---
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
    // Hide the welcome-page remote-password panel when the Cloudflare
    // tunnel is not active — there is no point exposing the password
    // field for a tunnel that does not exist.  When tunnelActive is
    // undefined (older backend) fall back to "show when we have a URL"
    // so existing deployments keep working.
    // In the remote webapp (body.remote-chat) never show the welcome-config
    // panel — the webapp URL and remote password are irrelevant there.
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

  /**
   * Update the settings-panel "Update" button to advertise that a
   * newer ``kiss-agent-framework`` release is available on PyPI.
   *
   * When ``available`` is true, the button receives the
   * ``has-update`` CSS class and a green download SVG icon is
   * injected before the "Update" label.  When false (or when called
   * before the first PyPI poll completes) the icon is removed and
   * the button reverts to its default appearance.
   *
   * The hourly poll runs in ``RemoteAccessServer._version_check_loop``
   * on the kiss-web daemon and broadcasts an ``update_available``
   * event over both UDS (VS Code webview) and WSS (remote-chat
   * webview).
   *
   * @param {boolean} available - true when ``latest > current``.
   * @param {string} latest - The latest version reported by PyPI.
   * @param {string} current - The version installed locally.
   */
  // Stable id for the permanent "update available" notification.  Using
  // a fixed string (instead of the auto-generated ``Date.now()``)
  // ensures the hourly PyPI re-broadcast re-uses the existing toast
  // rather than stacking duplicate notifications on top of each other.
  const UPDATE_NOTIFICATION_ID = 'kiss-update-available';

  // Inline SVG markup for the Feather "download" arrow used by both the
  // small settings-button badge and the action-button icon inside the
  // permanent update notification.  Kept as a single source of truth so
  // the two surfaces always look identical.
  const UPDATE_DOWNLOAD_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" ' +
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
    // Strip any previously-injected icon so repeated broadcasts do
    // not stack badges on top of each other.
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
    // Build an inline SVG download-arrow icon (matches the 12px
    // visual weight of the autocommit button's circle icon).  Use
    // namespaced createElementNS so the browser parses it as SVG
    // rather than HTML.
    const svgNs = 'http://www.w3.org/2000/svg';
    const icon = document.createElementNS(svgNs, 'svg');
    icon.setAttribute('class', 'update-available-icon');
    icon.setAttribute('width', '12');
    icon.setAttribute('height', '12');
    icon.setAttribute('viewBox', '0 0 24 24');
    icon.setAttribute('fill', 'none');
    icon.setAttribute('stroke', 'currentColor');
    icon.setAttribute('stroke-width', '2');
    icon.setAttribute('stroke-linecap', 'round');
    icon.setAttribute('stroke-linejoin', 'round');
    // Download arrow path (Feather "download" icon).
    const parts = [
      ['path', {d: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4'}],
      ['polyline', {points: '7 10 12 15 17 10'}],
      ['line', {x1: '12', y1: '15', x2: '12', y2: '3'}],
    ];
    for (const [tag, attrs] of parts) {
      const el = document.createElementNS(svgNs, tag);
      for (const k of Object.keys(attrs)) el.setAttribute(k, attrs[k]);
      icon.appendChild(el);
    }
    btn.insertBefore(icon, btn.firstChild);
  }

  /**
   * Show (or dismiss) the permanent "KISS Sorcar update available"
   * notification.
   *
   * The settings-panel "Update" button is only visible while the
   * settings panel is expanded, which left users who never opened
   * the panel unaware that a new release was waiting.  This helper
   * surfaces the same event in the always-visible chat-webview
   * notification stack with an SVG-iconed action button.  The
   * notification is sticky (never auto-dismisses) so it stays put
   * until the user clicks the update button or the next PyPI poll
   * reports the user is current.
   */
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
            vscode.postMessage({type: 'runUpdate'});
          },
        },
      ],
    });
  }

  // --- Welcome suggestions (dynamic) ---
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

  // --- Task replay ---
  function replayEventsInto(container, events, opts) {
    const rState = mkS();
    let rLlmPanel = null;
    let rLlmPanelState = mkS();
    let rLastToolName = '';
    // Start true so the first thought also gets its own panel.
    let rPendingPanel = true;
    // Defer syntax highlighting for the whole replay: every code block is
    // tagged `needs-hl` instead of being highlighted up front.  After the
    // panels are collapsed below, only the still-visible blocks are
    // highlighted; the collapsed ones wait until the user expands them.
    const prevDefer = _deferHighlight;
    _deferHighlight = true;
    try {
      events.forEach(ev => {
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
        if (t === 'tool_call') {
          rLastToolName = ev.name || '';
          rLlmPanel = null;
          rLlmPanelState = mkS();
          rPendingPanel = true;
        }
        if (t === 'tool_result' && rLastToolName !== 'finish') {
          rPendingPanel = true;
        }
        if (rPendingPanel && (t === 'thinking_start' || t === 'text_delta')) {
          rLlmPanel = mkThoughtsPanel();
          container.appendChild(rLlmPanel);
          rLlmPanelState = mkS();
          rPendingPanel = false;
        }
        let target = container,
          tState = rState;
        if (
          rLlmPanel &&
          (t === 'thinking_start' ||
            t === 'thinking_delta' ||
            t === 'thinking_end' ||
            t === 'text_delta' ||
            t === 'text_end')
        ) {
          target = rLlmPanel;
          tState = rLlmPanelState;
        }
        handleOutputEvent(ev, target, tState);
      });
    } finally {
      _deferHighlight = prevDefer;
    }
    collapseAllExceptResult(container);
    // Highlight only blocks that remain visible after collapsing; blocks
    // inside a collapsed panel stay deferred and are highlighted lazily when
    // the user expands the panel (see addCollapse / highlightPending).
    if (typeof hljs !== 'undefined') {
      container.querySelectorAll('code.needs-hl').forEach(bl => {
        if (!bl.closest('.collapsible.collapsed')) {
          bl.classList.remove('needs-hl');
          hljs.highlightElement(bl);
        }
      });
    }
  }

  /**
   * Count agent steps from a replayed event list: step 1 = the first
   * thinking/text event, each thinking/text following a tool call/result
   * (other than `finish`) = +1; a final `result` event's step_count wins
   * when present.
   */
  function countReplayedSteps(events) {
    let steps = 0,
      pending = false,
      lastTool = '';
    (events || []).forEach(ev => {
      const t = ev.type;
      if (t === 'tool_call') {
        lastTool = ev.name || '';
        pending = true;
      }
      if (t === 'tool_result' && lastTool !== 'finish') pending = true;
      if (steps === 0 && (t === 'thinking_start' || t === 'text_delta'))
        steps = 1;
      if (pending && (t === 'thinking_start' || t === 'text_delta')) {
        steps++;
        pending = false;
      }
      if (t === 'result' && ev.step_count) steps = ev.step_count;
    });
    return steps;
  }

  function replayTaskEvents(events) {
    clearOutput();
    resetOutputState();
    clearUsageMetrics();
    replayEventsInto(O, events, {
      onFollowupClick: function (text) {
        inp.value = text;
        syncClearBtn();
        inp.focus();
      },
    });
    // Count steps from replayed events: step 1 = first thinking, each llm-panel = +1
    const rSteps = countReplayedSteps(events);
    if (rSteps > 0) updateStepCount(rSteps);
    // Snapshot the current task's metrics for adjacent-scroll restoration
    currentTaskMetrics.tokens = statusTokens ? statusTokens.textContent : '';
    currentTaskMetrics.budget = statusBudget ? statusBudget.textContent : '';
    currentTaskMetrics.steps = statusSteps ? statusSteps.textContent : '';
    const tab = getTab(activeTabId);
    if (tab)
      applyChevronState(
        !!tab.panelsExpandedMap[currentTaskName],
        currentTaskName,
      );
    sb();
  }

  // --- Worktree merge/discard + autocommit bars (shared machinery) ---

  /**
   * Build a two-button action bar (shared by the worktree merge/discard
   * bar and the autocommit bar).  Each button entry is
   * `{cls, text, msg}` where `msg()` builds the postMessage payload at
   * click time; clicking any button first disables all of the bar's
   * buttons.
   */
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
        vscode.postMessage(b.msg());
      });
      btns.appendChild(btn);
    });
    bar.appendChild(btns);
    return bar;
  }

  /** Disable every button of an action bar (no-op when bar is null). */
  function disableActionBarBtns(bar) {
    if (!bar) return;
    bar.querySelectorAll('.wt-btn').forEach(b => {
      b.disabled = true;
    });
  }

  /** Detach an action bar from the DOM and restore the input container. */
  function detachActionBar(bar) {
    if (bar && bar.parentNode) bar.parentNode.removeChild(bar);
    if (inputContainer) inputContainer.style.display = '';
  }

  /** Hide the input container and show `bar` at the top of #input-area. */
  function attachActionBar(bar) {
    if (inputContainer) inputContainer.style.display = 'none';
    const area = document.getElementById('input-area');
    area.insertBefore(bar, area.firstChild);
  }

  /** Append a success/error result line for a worktree/autocommit action. */
  function appendActionResult(ev) {
    const cls = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
    const div = mkEl('div', 'ev ' + cls);
    div.textContent = (ev && ev.message) || '';
    O.appendChild(div);
    sb();
  }

  let worktreeBar = null;

  function clearWorktreeBar() {
    detachActionBar(worktreeBar);
    worktreeBar = null;
  }

  /** Create a worktree merge/discard bar element. ownerTabId is captured
   *  in button closures so the correct tab is targeted even if the user
   *  switches tabs before clicking. */
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
    // Hide the input container and show the worktree bar in its place
    worktreeBar = createWorktreeBar((ev && ev.tabId) || activeTabId);
    attachActionBar(worktreeBar);
  }

  // Suppress the trivial "Discarded branch '<name>'." confirmation that
  // would otherwise be appended to the chat output every time a
  // worktree is discarded.  Any discard message that also carries a
  // warning (e.g. ``"… ⚠️  Could not checkout '<orig>': …"``) still
  // gets shown so the user sees the warning.  Merge results and
  // partial-discard results (which always include a warning) are also
  // unaffected.
  function isSilentDiscardMessage(ev) {
    if (!ev || !ev.success) return false;
    const msg = ev.message || '';
    return /^Discarded branch '[^']+'\.$/.test(msg);
  }

  function handleWorktreeResult(ev) {
    clearWorktreeBar();
    if (isSilentDiscardMessage(ev)) {
      sb();
      return;
    }
    appendActionResult(ev);
  }

  // --- Autocommit prompt UI (non-worktree mode) ---
  // After the user resolves all merge-diff hunks, the backend sends an
  // `autocommit_prompt` event when the main branch still has dirty
  // state.  We show "Auto commit" / "Do nothing" buttons in the input
  // area, matching the worktree merge/discard bar.

  let autocommitBar = null;

  function clearAutocommitBar() {
    detachActionBar(autocommitBar);
    autocommitBar = null;
  }

  /** Create an autocommit bar element. ownerTabId is captured in button
   *  closures so the correct tab is targeted even after a tab switch. */
  function createAutocommitBar(ev) {
    const ownerTabId = (ev && ev.tabId) || activeTabId;
    const n = (ev && ev.changedFiles && ev.changedFiles.length) || 0;
    const labelText =
      n === 1
        ? '1 uncommitted change on main. Auto commit?'
        : n + ' uncommitted changes on main. Auto commit?';
    const msgFor = action => () => ({
      type: 'autocommitAction',
      action: action,
      tabId: ownerTabId,
      workDir: workDirForTab(ownerTabId),
    });
    return createActionBar(labelText, [
      {cls: 'wt-merge', text: 'Auto commit', msg: msgFor('commit')},
      {cls: 'wt-discard', text: 'Do nothing', msg: msgFor('skip')},
    ]);
  }

  function showAutocommitActions(ev) {
    clearAutocommitBar();
    autocommitBar = createAutocommitBar(ev);
    attachActionBar(autocommitBar);
  }

  function handleAutocommitResult(ev) {
    clearAutocommitBar();
    appendActionResult(ev);
    focusInputWithRetry();
  }

  // --- Merge diff rendering (web view) ---
  /** Build the DOM for a ``merge_data`` event.
   *
   * Each hunk is wrapped in its own ``<div class="merge-hunk"
   * data-fi=... data-hi=...>`` so the merge toolbar's Prev/Next can
   * scroll a specific hunk into view and Accept/Reject can mark a
   * specific hunk visually (via ``applyMergeResolutions``).  Context
   * lines are interleaved between hunks so the diff still reads
   * naturally.
   */
  function renderMergeData(ev) {
    const mdEl = mkEl('div', 'ev merge-info');
    const hdr = mkEl('div', 'merge-info-hdr');
    hdr.textContent = '✱ Reviewing ' + (ev.hunk_count || 0) + ' change(s)';
    mdEl.appendChild(hdr);

    const body = mkEl('div', 'merge-info-body');
    body.textContent =
      'Red = old lines, Green = new lines. Use the merge toolbar to ' +
      'navigate and accept or reject changes.';
    mdEl.appendChild(body);

    const mergeFiles = (ev.data && ev.data.files) || [];
    for (let mfi = 0; mfi < mergeFiles.length; mfi++) {
      const mf = mergeFiles[mfi];
      if (mf.base_text === undefined || mf.current_text === undefined) continue;
      const fileEl = mkEl('div', 'merge-file-diff');
      fileEl.dataset.fi = String(mfi);
      const fileName = mkEl('div', 'merge-file-name');
      fileName.textContent = mf.name || 'unknown';
      fileEl.appendChild(fileName);

      const baseLines = (mf.base_text || '').split('\n');
      const curLines = (mf.current_text || '').split('\n');
      const hunks = mf.hunks || [];
      let curIdx = 0;
      for (let mhi = 0; mhi < hunks.length; mhi++) {
        const h = hunks[mhi];
        // Context lines before the hunk (rendered outside the
        // hunk container so we never scroll context into the
        // highlight box).
        if (curIdx < h.cs) {
          const ctxBefore = mkEl('pre', 'merge-ctx');
          let ctxText = '';
          while (curIdx < h.cs) {
            ctxText += ' ' + (curLines[curIdx] || '') + '\n';
            curIdx++;
          }
          ctxBefore.textContent = ctxText;
          fileEl.appendChild(ctxBefore);
        }
        const hunkEl = mkEl('pre', 'merge-hunk');
        hunkEl.dataset.fi = String(mfi);
        hunkEl.dataset.hi = String(mhi);
        const hunkHdr = mkEl('span', 'merge-hunk-label');
        hunkHdr.textContent =
          'Hunk ' + (mhi + 1) + ' / ' + hunks.length + ' @ line ' + (h.cs + 1);
        hunkEl.appendChild(hunkHdr);
        // Old (base) lines - red
        for (let bi = h.bs; bi < h.bs + h.bc; bi++) {
          const oldLine = mkEl('span', 'diff-del');
          oldLine.textContent = '-' + (baseLines[bi] || '') + '\n';
          hunkEl.appendChild(oldLine);
        }
        // New (current) lines - green
        for (let ci = h.cs; ci < h.cs + h.cc; ci++) {
          const newLine = mkEl('span', 'diff-add');
          newLine.textContent = '+' + (curLines[ci] || '') + '\n';
          hunkEl.appendChild(newLine);
        }
        fileEl.appendChild(hunkEl);
        curIdx = h.cs + h.cc;
      }
      // Trailing context (after last hunk).
      if (curIdx < curLines.length) {
        const ctxAfter = mkEl('pre', 'merge-ctx');
        let ctxText = '';
        while (curIdx < curLines.length) {
          ctxText += ' ' + (curLines[curIdx] || '') + '\n';
          curIdx++;
        }
        ctxAfter.textContent = ctxText;
        fileEl.appendChild(ctxAfter);
      }
      mdEl.appendChild(fileEl);
    }
    addCollapse(mdEl, hdr);
    return mdEl;
  }

  /** Mark the hunk identified by ``(fi, hi)`` as the active one. */
  function setCurrentMergeHunk(mergePanel, fi, hi) {
    mergePanel.querySelectorAll('.merge-hunk.current').forEach(el => {
      el.classList.remove('current');
    });
    const hunk = mergePanel.querySelector(
      '.merge-hunk[data-fi="' + fi + '"][data-hi="' + hi + '"]',
    );
    if (hunk) hunk.classList.add('current');
  }

  /** Scroll the hunk identified by ``(fi, hi)`` into view.
   *
   * The remote-web shell sets ``html, body { overflow: hidden }`` (so the
   * page itself never scrolls) and delegates scrolling to ``#output``.
   * Native ``Element.scrollIntoView`` is unreliable in that layout —
   * Chromium/Webkit sometimes try to scroll the (non-scrollable) document
   * instead of bubbling to the nearest scrollable ancestor, so clicking
   * Accept/Reject/Prev/Next in the inline merge toolbar would highlight
   * the new hunk but leave it off-screen.  We walk up to the nearest
   * scrollable ancestor explicitly and animate ``scrollTop`` to centre
   * the hunk ourselves.  Falls back to ``scrollIntoView`` if no
   * scrollable ancestor is found (e.g. when the hunk lives in a detached
   * background-tab fragment). */
  function scrollHunkIntoView(mergePanel, fi, hi) {
    const hunk = mergePanel.querySelector(
      '.merge-hunk[data-fi="' + fi + '"][data-hi="' + hi + '"]',
    );
    if (!hunk) return;
    let container = hunk.parentElement;
    while (container && container !== document.body) {
      const style = window.getComputedStyle(container);
      const oy = style.overflowY;
      if (
        (oy === 'auto' || oy === 'scroll') &&
        container.scrollHeight > container.clientHeight
      ) {
        break;
      }
      container = container.parentElement;
    }
    if (!container || container === document.body) {
      if (typeof hunk.scrollIntoView === 'function') {
        hunk.scrollIntoView({block: 'center', behavior: 'smooth'});
      }
      return;
    }
    const containerRect = container.getBoundingClientRect();
    const hunkRect = hunk.getBoundingClientRect();
    const target =
      container.scrollTop +
      (hunkRect.top - containerRect.top) -
      Math.max(0, (container.clientHeight - hunkRect.height) / 2);
    const top = Math.max(
      0,
      Math.min(target, container.scrollHeight - container.clientHeight),
    );
    if (typeof container.scrollTo === 'function') {
      container.scrollTo({top: top, behavior: 'smooth'});
    } else {
      container.scrollTop = top;
    }
  }

  /** Apply ``accepted`` / ``rejected`` classes to every resolved hunk.
   *
   * resolutions is an array of ``{fi, hi, status}`` objects sent by
   * the server in each ``merge_nav`` event.  Classes are cleared from
   * hunks no longer in the list so undo-like behaviour (if added
   * later) works correctly.
   */
  function applyMergeResolutions(mergePanel, resolutions) {
    mergePanel
      .querySelectorAll('.merge-hunk.accepted, .merge-hunk.rejected')
      .forEach(el => {
        el.classList.remove('accepted');
        el.classList.remove('rejected');
      });
    for (let i = 0; i < resolutions.length; i++) {
      const r = resolutions[i];
      if (!r || r.fi === undefined || r.hi === undefined) continue;
      const hunk = mergePanel.querySelector(
        '.merge-hunk[data-fi="' + r.fi + '"][data-hi="' + r.hi + '"]',
      );
      if (hunk)
        hunk.classList.add(r.status === 'rejected' ? 'rejected' : 'accepted');
    }
  }

  // --- Merge toolbar (shown in input area, replacing textarea) ---
  function showMergeToolbar(ownerTabId) {
    if (document.getElementById('merge-toolbar')) return;
    const capturedTabId = ownerTabId || activeTabId;
    inputContainer.style.display = 'none';
    const bar = mkEl('div', 'merge-toolbar-card');
    bar.id = 'merge-toolbar';
    bar.innerHTML =
      '<div class="merge-toolbar-header">' +
      '<span class="merge-toolbar-title">Review Changes</span>' +
      '<span class="merge-toolbar-hint">Red = old \u00b7 Green = new</span>' +
      '</div>' +
      '<div class="merge-toolbar-actions">' +
      '<div class="merge-toolbar-row">' +
      '<button class="merge-btn merge-nav" id="merge-prev-btn">Prev</button>' +
      '<button class="merge-btn merge-nav" id="merge-next-btn">Next</button>' +
      '<button class="merge-btn merge-accept" id="merge-accept-btn">Accept</button>' +
      '<button class="merge-btn merge-reject" id="merge-reject-btn">Reject</button>' +
      '</div>' +
      '<div class="merge-toolbar-row">' +
      '<button class="merge-btn merge-accept" id="merge-accept-file-btn">Accept File</button>' +
      '<button class="merge-btn merge-reject" id="merge-reject-file-btn">Reject File</button>' +
      '<button class="merge-btn merge-accept" id="merge-accept-all-btn">Accept Rest</button>' +
      '<button class="merge-btn merge-reject" id="merge-reject-all-btn">Reject Rest</button>' +
      '</div>' +
      '</div>';
    document.getElementById('input-area').appendChild(bar);
    const mergeActions = {
      'merge-accept-btn': 'accept',
      'merge-reject-btn': 'reject',
      'merge-prev-btn': 'prev',
      'merge-next-btn': 'next',
      'merge-accept-file-btn': 'accept-file',
      'merge-reject-file-btn': 'reject-file',
      'merge-accept-all-btn': 'accept-all',
      'merge-reject-all-btn': 'reject-all',
    };
    Object.keys(mergeActions).forEach(id => {
      document.getElementById(id).addEventListener('click', () => {
        vscode.postMessage({
          type: 'mergeAction',
          action: mergeActions[id],
          tabId: capturedTabId,
        });
      });
    });
    sb();
  }

  function hideMergeToolbar() {
    const bar = document.getElementById('merge-toolbar');
    if (bar) bar.remove();
    inputContainer.style.display = '';
  }

  // --- Init and event listeners ---

  function init() {
    setupEventListeners();
    renderTabBar();
    // Include restored tabs with backend chat IDs so the extension can auto-reload their events
    const restoredTabs = tabs
      .filter(t => {
        return t.backendChatId;
      })
      .map(t => {
        return {tabId: t.id, chatId: t.backendChatId};
      });
    vscode.postMessage({
      type: 'ready',
      tabId: activeTabId,
      restoredTabs: restoredTabs,
    });
    // Request the current config so the welcome-page remote-password
    // mirror (welcome-cfg-remote-password) is populated before the user
    // ever opens the Settings panel.
    vscode.postMessage({type: 'getConfig'});
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    window.addEventListener('focus', () => {
      vscode.postMessage({type: 'webviewFocusChanged', focused: true});
    });
    window.addEventListener('blur', () => {
      vscode.postMessage({type: 'webviewFocusChanged', focused: false});
    });
    document.addEventListener('keydown', e => {
      if (
        (e.metaKey || e.ctrlKey) &&
        e.key === 'd' &&
        !e.shiftKey &&
        !e.altKey
      ) {
        e.preventDefault();
        vscode.postMessage({type: 'focusEditor'});
      }
      if (e.key === 'Escape' && sidebar.classList.contains('open')) {
        e.preventDefault();
        closeSidebar();
      }
    });
    inp.addEventListener('keydown', e => {
      // Autocomplete navigation
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
        if (e.key === 'Enter' && acIdx >= 0) {
          e.preventDefault();
          items[acIdx].click();
          return;
        }
        if (e.key === 'Escape') {
          hideAC();
          return;
        }
      }
      // Ghost text accept
      if (e.key === 'Tab' && currentGhost) {
        e.preventDefault();
        acceptGhost();
        return;
      }
      // History cycling (ArrowUp/Down only when textbox is empty and no autocomplete)
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
      // Any other key clears ghost
      if (e.key !== 'Tab') clearGhost();
    });
    // Fallback for mobile virtual keyboards that don't fire keydown for Enter.
    // Track Shift state so Shift+Enter still inserts a newline on desktop.
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
    // Mobile touch gestures on the input textarea
    inp.addEventListener('touchstart', handleInputTouchStart, {passive: true});
    inp.addEventListener('touchend', handleInputTouchEnd);
    autocomplete.addEventListener('mousedown', e => {
      e.preventDefault();
    });
    stopBtn.addEventListener('click', () => {
      if (_demoActive) {
        if (typeof window._cancelDemoReplay === 'function')
          window._cancelDemoReplay();
        _demoActive = false;
        return;
      }
      vscode.postMessage({type: 'stop', tabId: activeTabId});
    });
    uploadBtn.addEventListener('click', () => {
      const input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.accept = 'image/*,application/pdf';
      input.onchange = handleFileSelect;
      input.click();
    });
    setupPasswordToggle('cfg-remote-password-toggle', 'cfg-remote-password');
    setupPasswordToggle(
      'welcome-cfg-remote-password-toggle',
      'welcome-cfg-remote-password',
    );
    // API keys are secret by default; each gets an eye toggle like the
    // remote-password field.
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
    // The welcome-page remote-password input and the settings-panel
    // input mirror each other so the existing collectConfigForm +
    // saveConfig flow keeps working without changes.  On Enter, blur,
    // or change we flush via saveSettingsIfPopulated so the password
    // is persisted promptly.  The Enter handler also blurs the input
    // so the on-screen keyboard collapses on mobile.
    const welcomePwInp = document.getElementById('welcome-cfg-remote-password');
    const settingsPwInp = document.getElementById('cfg-remote-password');
    function _flushPw() {
      // Only save if the settings form has already been populated by
      // configData; otherwise collectConfigForm would post stale empty
      // fields and clobber the user's real config.  When the welcome
      // panel is the only thing shown (before settings is ever opened)
      // configData has already populated both inputs, so this is true.
      saveSettingsIfPopulated();
    }
    if (welcomePwInp && settingsPwInp) {
      welcomePwInp.addEventListener('input', () => {
        settingsPwInp.value = welcomePwInp.value;
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

    if (demoToggleBtn) {
      demoToggleBtn.addEventListener('change', () => {
        if (_demoActive && !demoToggleBtn.checked) {
          // Cancel running demo when unchecked mid-replay
          if (typeof window._cancelDemoReplay === 'function')
            window._cancelDemoReplay();
          demoMode = false;
          _demoActive = false;
          return;
        }
        demoMode = demoToggleBtn.checked;
      });
    }

    if (updateBtn) {
      updateBtn.addEventListener('click', e => {
        e.preventDefault();
        e.stopPropagation();
        vscode.postMessage({type: 'runUpdate'});
      });
    }

    // Open/close helpers for the in-settings-panel floating confirmation
    // box.  The dialog replaces the native VS Code modal warning so the
    // confirmation lives WITHIN the settings panel itself.  Only OK
    // forwards ``{type:'serverReset'}`` to the extension; Cancel, the
    // backdrop, and Escape all simply close the box.
    function openServerResetConfirm() {
      if (!serverResetConfirmModal) return;
      serverResetConfirmModal.classList.add('open');
      // Focus the OK button so keyboard users can confirm with Enter
      // (or dismiss with Escape) without reaching for the mouse.
      if (serverResetConfirmOkBtn) {
        try {
          serverResetConfirmOkBtn.focus();
        } catch (_err) {
          // ``focus()`` can throw if the element is detached or in a
          // hidden subtree (some JSDOM versions); the dialog still
          // works without focus, so swallow.
        }
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
        // Server reset SIGTERMs the kiss-web daemon, killing every
        // in-flight agent.  When any tab still has a running agent we
        // surface an in-webview floating confirmation box anchored to
        // the settings panel; only OK forwards the reset.  When no
        // agent is running we fast-path the reset.
        //
        // Guard: if the confirmation box is already open, ignore the
        // click — otherwise rapid double-clicks would re-open / stack
        // the dialog.
        if (isServerResetConfirmOpen()) return;
        const agentRunning = tabs.some(tab => tab && tab.isRunning);
        if (!agentRunning) {
          vscode.postMessage({type: 'serverReset'});
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
        vscode.postMessage({type: 'serverReset'});
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
      // Clicking the dimmed backdrop (the modal element itself, not its
      // inner content box) dismisses the dialog without confirming.
      serverResetConfirmModal.addEventListener('click', e => {
        if (e.target === serverResetConfirmModal) closeServerResetConfirm();
      });
      // Escape closes the dialog — only when it's open, so we don't
      // swallow Escape for the rest of the webview.
      document.addEventListener('keydown', e => {
        if (e.key === 'Escape' && isServerResetConfirmOpen()) {
          e.preventDefault();
          e.stopPropagation();
          closeServerResetConfirm();
        }
      });
    }

    if (autocommitBtn) {
      autocommitBtn.addEventListener('click', e => {
        // The button now lives inside the cfg-auto-commit <label>; stop
        // the click from propagating to the label and toggling the
        // sibling checkbox.
        e.preventDefault();
        e.stopPropagation();
        vscode.postMessage({
          type: 'autocommitAction',
          action: 'commit',
          tabId: activeTabId,
          workDir: workDirForTab(activeTabId),
        });
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
        closeSidebar();
      } else {
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('open');
        resetHistoryPagination();
        vscode.postMessage({
          type: 'getHistory',
          query: historySearch ? historySearch.value : '',
          generation: historyGeneration,
        });
      }
    }
    if (menuBtn) {
      menuBtn.addEventListener('click', toggleHistorySidebar);
    }
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
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
    historySearch.addEventListener('input', () => {
      resetHistoryPagination();
      vscode.postMessage({
        type: 'getHistory',
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
        vscode.postMessage({
          type: 'getHistory',
          query: '',
          generation: historyGeneration,
        });
        historySearch.focus();
      });
    }
    // History filter bar: 3 category checkboxes + From/To date range.
    // Filtering is purely client-side over rows already in the DOM.
    // ``applyHistoryFilterVisibility()`` toggles ``display`` on each
    // row based on the row's ``data-category`` and ``data-timestamp``
    // attributes set in ``renderHistory``.
    const hfRunning = document.getElementById('hf-running');
    const hfErrors = document.getElementById('hf-errors');
    const hfCompleted = document.getElementById('hf-completed');
    const hfWorkspace = document.getElementById('hf-workspace');
    const hfFavorite = document.getElementById('hf-favorite');
    const hfFrom = document.getElementById('hf-from');
    const hfTo = document.getElementById('hf-to');
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
    // The calendar selector buttons sit next to each date textbox and
    // open a custom in-webview calendar popup.  The native
    // <input type=date> picker (showPicker / focus+click) is unreliable
    // inside VS Code webviews — it often does nothing because the
    // embedded Chromium build either blocks ``showPicker`` without
    // recent user activation or shows the picker behind the webview.
    // A custom popup avoids both issues and gives consistent styling
    // across the extension and the remote browser chat.
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
        vscode.postMessage({
          type: 'getHistory',
          query: historySearch.value,
          offset: historyOffset,
          generation: historyGeneration,
        });
      }
    });
    // Click handler for file paths in tool call headers — parse :line suffix
    document.addEventListener('click', e => {
      const el = e.target.closest('[data-path]');
      if (el && el.dataset.path) {
        const raw = el.dataset.path;
        const match = raw.match(/^(.+):(\d+)$/);
        if (match) {
          vscode.postMessage({
            type: 'openFile',
            path: match[1],
            line: parseInt(match[2], 10),
          });
        } else {
          vscode.postMessage({type: 'openFile', path: raw});
        }
      }
    });
    // Per-tab ask-user submit/keydown listeners are wired in
    // ensureAskElementsForTab() so each tab gets its own input/submit.

    // Paste images/PDFs
    inp.addEventListener('paste', e => {
      const items = (e.clipboardData || {}).items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (
          item.kind === 'file' &&
          (item.type.startsWith('image/') || item.type === 'application/pdf')
        ) {
          e.preventDefault();
          const file = item.getAsFile();
          if (file) readFileAsAttachment(file);
        }
      }
    });

    // Drag and drop
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
        // Handle file URIs from VS Code explorer (text/uri-list)
        const uriList =
          e.dataTransfer && e.dataTransfer.getData('text/uri-list');
        if (uriList) {
          const uris = uriList.split(/\r?\n/).filter(u => {
            return u && !u.startsWith('#');
          });
          if (uris.length > 0) {
            vscode.postMessage({type: 'resolveDroppedPaths', uris: uris});
            return;
          }
        }
        // Handle image/PDF file drops
        const files = e.dataTransfer && e.dataTransfer.files;
        if (!files) return;
        Array.from(files).forEach(file => {
          if (
            file.type.startsWith('image/') ||
            file.type === 'application/pdf'
          ) {
            readFileAsAttachment(file);
          }
        });
      });
    }

    window.addEventListener('message', event => {
      handleEvent(event.data);
    });
  }

  function readFileAsAttachment(file) {
    const reader = new FileReader();
    reader.onload = function (event) {
      attachments.push({
        name: file.name,
        type: file.type,
        data: event.target.result.split(',')[1],
      });
      renderFileChips();
    };
    reader.readAsDataURL(file);
  }

  function sendMessage() {
    const prompt = inp.value.trim();
    if (!prompt) return;

    if (histCache[0] !== prompt) {
      histCache.unshift(prompt);
    }
    const curTab = getTab(activeTabId);

    // If a task is already running for this tab, forward the prompt
    // to the backend as an ``appendUserMessage`` so it gets injected
    // into the live agent's conversation as a follow-up user message
    // before its next model call.  Clear the input afterwards just
    // like a normal submit, so the user can keep typing further
    // messages while the task runs.
    if (isRunning) {
      vscode.postMessage({
        type: 'appendUserMessage',
        prompt: prompt,
        tabId: activeTabId,
      });
      inp.value = '';
      inp.style.height = 'auto';
      attachments = [];
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
      attachments: attachments.map(a => {
        return {name: a.name, mimeType: a.type, data: a.data};
      }),
      useWorktree: !!(worktreeToggleBtn && worktreeToggleBtn.checked),
      useParallel: !!(parallelToggleBtn && parallelToggleBtn.checked),
      autoCommit: !!(autocommitToggleBtn && autocommitToggleBtn.checked),
    };
    if (curTab && curTab.workDir) msg.workDir = curTab.workDir;
    vscode.postMessage(msg);
    // Fresh local run: anchor the optimistic timer at submit time and
    // clear the previous task's end timestamp (the extension host
    // sends a startTs-less ``status running:true`` right away; the
    // daemon's tab-stamped status re-anchors ``t0`` to the agent's
    // true startTs moments later).
    t0 = Date.now();
    endTs = 0;
    if (curTab) {
      curTab.t0 = t0;
      curTab.endTs = 0;
    }
    inp.value = '';
    inp.style.height = 'auto';
    attachments = [];
    renderFileChips();
    clearGhost();
    histIdx = -1;
    if (inputClearBtn) inputClearBtn.style.display = 'none';
  }

  /**
   * Create the per-tab ask-user DOM nodes (question div, answer textarea,
   * submit button) and wire them to the per-tab submit handler.  Idempotent.
   */
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
    tab.askQuestionEl = q;
    tab.askInputEl = i;
    tab.askSubmitEl = s;
  }

  /** Render the given question text into the tab's question element. */
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

  /** Detach any ask elements currently in the shared slot (hide modal). */
  function clearAskSlot() {
    if (!askUserSlot) return;
    while (askUserSlot.firstChild)
      askUserSlot.removeChild(askUserSlot.firstChild);
    if (askUserModal) askUserModal.style.display = 'none';
  }

  /** Mount the tab's current ask-user elements into the slot and focus input. */
  function mountAskForTab(tab) {
    if (!askUserSlot) return;
    while (askUserSlot.firstChild)
      askUserSlot.removeChild(askUserSlot.firstChild);
    askUserSlot.appendChild(tab.askQuestionEl);
    askUserSlot.appendChild(tab.askInputEl);
    askUserSlot.appendChild(tab.askSubmitEl);
    askUserModal.style.display = 'flex';
    setTimeout(() => {
      if (tab.id === activeTabId && tab.askInputEl) tab.askInputEl.focus();
    }, 0);
  }

  /**
   * Render the tab's pending ask-user question into its triplet and, if
   * the tab is active, mount the triplet into the shared slot and show the
   * modal.  If the tab has no pending question and is active, hide the
   * modal.
   */
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

  /** Return true when an ask-clear for sourceTab should also clear candidate. */
  function isAskSameChatTab(sourceTab, candidate) {
    if (!sourceTab || !candidate) return false;
    if (candidate.id === sourceTab.id) return true;
    const chatId = String(sourceTab.backendChatId || '');
    return !!chatId && String(candidate.backendChatId || '') === chatId;
  }

  /** Clear pending ask-user UI for sourceTab and sibling tabs on the same chat. */
  function clearAskForMatchingChatTabs(sourceTab) {
    let shouldClearSlot = false;
    for (let i = 0; i < tabs.length; i++) {
      const tab = tabs[i];
      if (!isAskSameChatTab(sourceTab, tab)) continue;
      tab.askPendingQuestion = null;
      if (tab.askInputEl) tab.askInputEl.value = '';
      if (tab.id === activeTabId) shouldClearSlot = true;
    }
    if (shouldClearSlot) clearAskSlot();
  }

  /** Submit the current answer for the given tab; clear pending question. */
  function submitAskForTab(tab) {
    const answer = tab.askInputEl ? tab.askInputEl.value : '';
    vscode.postMessage({type: 'userAnswer', answer: answer, tabId: tab.id});
    clearAskForMatchingChatTabs(tab);
  }

  /**
   * Synchronise the shared modal slot with the active tab after a tab
   * switch: detach previous contents and mount the active tab's ask UI if
   * it has a pending question.
   */
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
      chip.className = 'file-chip';
      const isImage = att.type.startsWith('image/');
      chip.innerHTML =
        (isImage
          ? '<img src="data:' + att.type + ';base64,' + att.data + '">'
          : '<span class="fc-icon">\uD83D\uDCC4</span>') +
        '<span>' +
        esc(att.name) +
        '</span>' +
        '<span class="fc-rm" data-idx="' +
        idx +
        '">&times;</span>';
      chip.querySelector('.fc-rm').addEventListener('click', () => {
        attachments.splice(idx, 1);
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
    modelName.textContent = name;
    closeModelDD();
    renderModelList('');
    vscode.postMessage({type: 'selectModel', model: name, tabId: activeTabId});
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

  /**
   * Build a sidebar copy-to-clipboard button for a history/frequent task row.
   *
   * Clicking the returned button copies the supplied task text to the
   * system clipboard via ``navigator.clipboard.writeText`` (falling
   * back to a temporary textarea + ``document.execCommand('copy')``
   * when the async clipboard API is unavailable, e.g. in older
   * webview hosts).  After a successful copy the trash-shaped icon
   * briefly swaps to a check mark for visual confirmation.  Click
   * propagation is stopped so the surrounding row's click handler
   * (which would reopen the task / fill the input) does not fire.
   *
   * @param {string} text - the full task text to copy.
   * @returns {HTMLButtonElement}
   */
  function makeSidebarCopyButton(text) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'sidebar-item-copy';
    btn.setAttribute('aria-label', 'Copy task to clipboard');
    btn.innerHTML = PANEL_COPY_SVG;

    const flash = () => {
      btn.innerHTML = PANEL_CHECK_SVG;
      btn.classList.add('copied');
      setTimeout(() => {
        btn.innerHTML = PANEL_COPY_SVG;
        btn.classList.remove('copied');
      }, 1500);
    };

    btn.addEventListener('click', e => {
      e.stopPropagation();
      e.preventDefault();
      const payload = String(text == null ? '' : text);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(payload).then(flash, () => {});
      } else if (fallbackCopyText(payload)) {
        flash();
      }
    });

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

    // Compute the live running→completed transitions BEFORE
    // rendering: any task_id that was rendered as running on the
    // previous ``renderHistory`` call but is no longer running on
    // this call has just completed in the user's current session,
    // so its row must show the SOLID green dot from now on (and
    // STAY that way for the rest of the page session).  We compute
    // the new running set from the full ``allHistSessions`` (not
    // just this batch) so pagination batches don't artificially
    // drop ids that are simply absent from the current chunk.
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
      // Match the Running tab's visual layout: the ``running-item``
      // class flips the row into a wrap-with-metrics layout
      // (multi-line text, metrics row on its own line).
      div.className = 'sidebar-item running-item';
      // Stamp the row with its filter-bar category and timestamp so
      // ``applyHistoryFilterVisibility()`` can toggle ``display`` on
      // each row in O(n) without re-rendering the list.
      div.dataset.category = s.is_running
        ? 'running'
        : s.failed
          ? 'errors'
          : 'completed';
      div.dataset.timestamp = String(Number(s.timestamp || 0));
      // ``data-favorite`` mirrors the persisted ``is_favorite`` flag
      // so ``applyHistoryFilterVisibility()`` can include/exclude the
      // row when the Favorite checkbox in the filter bar is toggled.
      div.dataset.favorite = s.is_favorite ? '1' : '0';
      // ``data-work-dir`` mirrors the persisted ``extra.work_dir``
      // so ``applyHistoryFilterVisibility()`` can hide rows whose
      // ``work_dir`` differs from the client's currently-configured
      // workspace when the Workspace filter checkbox is checked.
      div.dataset.workDir = s.work_dir || '';
      const itemText = s.title || s.preview || 'Untitled';
      div.dataset.tooltip = s.preview || itemText;
      div.style.backgroundColor = chatIdBgColor(String(s.id));
      div.style.color = '#1a1a1a';

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
        // The row was rendered as ``is_running:true`` earlier in
        // this page session and has now transitioned to
        // finished-cleanly.  Render the SOLID green circle (no
        // animation) and KEEP it for the rest of the session — even
        // after subsequent ``refreshHistory()`` reloads.  Tasks
        // that the user never saw running in this session (e.g.
        // every row on a fresh page-load) intentionally render NO
        // dot, so the History panel doesn't show a sea of solid
        // green circles for old completed tasks.
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

      if (s.task_id) {
        // Container that stacks the three per-row action buttons
        // (favourite / copy / delete) vertically with a 6px gap.
        // The confirm-delete prompt is appended to the same column
        // so it visually replaces the delete button when shown.
        const actions = document.createElement('div');
        actions.className = 'sidebar-item-actions';

        // Favourite (star) button — flips the persisted
        // ``is_favorite`` flag on the task's ``extra`` JSON column.
        // The icon shows a filled star when favourited, outline
        // otherwise.  Click toggles both the UI and the backend
        // state optimistically.
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
          // Keep ``data-favorite`` in sync with the toggled state so
          // the Favorite filter checkbox immediately reflects the
          // change without waiting for a re-render.
          div.dataset.favorite = next ? '1' : '0';
          applyHistoryFilterVisibility();
          vscode.postMessage({
            type: 'setFavorite',
            taskId: s.task_id,
            isFavorite: next,
          });
        });
        actions.appendChild(favBtn);

        // Copy-to-clipboard button — sits immediately left of the
        // trash icon so the user can grab the full task text without
        // first reopening the task.  ``s.preview`` carries the full
        // task text (see server._get_history where ``preview`` is set
        // to the task string verbatim); ``itemText`` is the fallback.
        const copyBtn = makeSidebarCopyButton(s.preview || itemText);
        actions.appendChild(copyBtn);

        const delBtn = document.createElement('button');
        delBtn.className = 'sidebar-item-delete';
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
        });

        confirmBtn.addEventListener('click', e => {
          e.stopPropagation();
          vscode.postMessage({type: 'deleteTask', taskId: s.task_id});
          div.remove();
        });

        cancelBtn.addEventListener('click', e => {
          e.stopPropagation();
          confirmWrap.style.display = 'none';
          delBtn.style.display = '';
        });

        actions.appendChild(delBtn);
        actions.appendChild(confirmWrap);
        div.appendChild(actions);
      }

      // Per-row info column — wraps the three stacked detail lines
      // (metrics, workspace+meta, chat/task/parent ids) so they
      // render flush, with no flex row-gap between them.  The
      // container itself uses ``flex-basis: 100%`` to drop onto its
      // own line below the running/failed dot, the task text, and
      // the action column (mirroring the trick the metrics span used
      // when it was a direct sibling).  Its inner ``flex-direction:
      // column`` + ``gap: 0`` rule stacks the three lines tightly.
      const info = document.createElement('div');
      info.className = 'running-item-info';

      // Metrics row (steps • tokens • cost • duration) — matches the
      // Running tab.  Rendered as the first child of the info
      // container above; ``.running-item-metrics`` no longer needs
      // ``flex-basis: 100%`` (the container handles the line break)
      // but the rule is kept for backwards compatibility with any
      // other surface that may still render it as a direct child of
      // ``.sidebar-item``.
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
      // Time spent on the task in hh:mm:ss format, rendered AFTER the
      // cost and BEFORE the date suffix.  For finished rows the
      // backend surfaces ``endTs - startTs``; for running rows
      // (``is_running=true`` or ``endTs==0``) we use ``Date.now() -
      // startTs`` so the user sees a live estimate at history-load
      // time.  Rows with no usable startTs or a non-positive duration
      // omit the token entirely — we never display 00:00:00.
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

      // Workspace + meta row — the task's ``work_dir`` and the
      // persisted run metadata (model name, wt/no-wt,
      // parallel/sequential, auto-commit/manual-commit) rendered
      // as a single dot-separated line IMMEDIATELY after the
      // metrics line inside the per-row info column.  Format:
      //
      //   <work_dir> • <model> • <wt|no-wt>
      //     • <parallel|sequential> • <auto-commit|manual-commit>
      //
      // The metadata fields come from the per-task ``extra`` JSON
      // (``model``, ``is_worktree``, ``is_parallel``,
      // ``auto_commit_mode``) persisted by
      // ``_TaskRunnerMixin._run_task_inner``.  Missing pieces are
      // simply skipped:
      //   * no ``work_dir``      → line starts with the model;
      //   * no ``model``         → line shows only the workspace;
      //   * neither present      → no line at all (no placeholder).
      // Booleans default to ``false`` when missing → no-wt /
      // sequential / manual-commit.
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
        // Native HTML tooltip — useful when the combined line is
        // long enough to be clipped by overflow:hidden in the
        // sidebar.
        workspace.title = text;
        info.appendChild(workspace);
      }

      // Ids row — chat id, task id, and parent task id rendered as
      // a single dot-separated line right below the workspace+meta
      // line.  Format:
      //
      //   chat <chat_id> • task <task_id> • parent <parent_task_id>
      //
      // Each field is omitted when not present so legacy rows that
      // pre-date a particular id, plus regular (non-sub-agent) rows
      // that have no ``parent_task_id``, render cleanly without
      // dangling bullets or placeholder text.  When NONE of the
      // three ids is set we skip the span entirely so the History
      // panel does not show an empty third line.
      const chatId = typeof s.id === 'string' ? s.id : '';
      const taskIdRaw = s.task_id;
      const taskIdStr =
        taskIdRaw === undefined || taskIdRaw === null ? '' : String(taskIdRaw);
      const parentIdRaw = s.parent_task_id;
      const parentIdStr =
        parentIdRaw === undefined || parentIdRaw === null
          ? ''
          : String(parentIdRaw);
      const idParts = [];
      if (chatId) idParts.push('chat ' + chatId);
      if (taskIdStr) idParts.push('task ' + taskIdStr);
      if (parentIdStr) idParts.push('parent ' + parentIdStr);
      if (idParts.length > 0) {
        const idsSpan = document.createElement('span');
        idsSpan.className = 'running-item-ids';
        const idsText = idParts.join(' • ');
        idsSpan.textContent = idsText;
        // Native HTML tooltip — useful when the combined line is
        // long enough to be clipped by overflow:hidden in the
        // sidebar.
        idsSpan.title = idsText;
        info.appendChild(idsSpan);
      }

      // Finally attach the per-row info column to the row itself.
      // Appended LAST so the running/failed dot, the task text, and
      // the action buttons sit on the row's first visual line and
      // the info container drops onto the second visual line.
      div.appendChild(info);

      div.addEventListener('click', () => {
        if (demoMode && typeof window._startDemoReplay === 'function') {
          closeSidebar();
          createNewTab();
          window._startDemoReplay(allHistSessions);
          return;
        }
        // A client must never display the same backend chat id in two
        // local tabs.  If this history row's chat is already open (for
        // example the user is on a blank tab and clicks an older row
        // for a chat that is open to the left), simply switch focus to
        // that tab instead of creating a duplicate tab and issuing a
        // second resumeSession for the same chat.
        const existingChatTab = getTabByBackendChatId(s.id);
        if (existingChatTab) {
          switchToTab(existingChatTab.id);
        } else if (s.has_events && s.id) {
          // Sub-agent history rows reopen as a regular chat tab that
          // the backend (``_replay_session``) will then flip into a
          // sub-agent tab via ``openSubagentTab`` (purple accent,
          // no input bar, no adjacent-task loading).  We do
          // not look up "is the original sub-agent tab still open?" —
          // sub-agent rows are persisted with just their parent
          // task_history.id, so the simplest UX is a fresh tab whose
          // events are replayed from the row's own events table.
          // When the clicked history row has a known chat_id (s.id)
          // and persisted events, allocate a fresh tab id and let the
          // backend route the chat lookup by chat_id (passed in the
          // ``resumeSession`` payload).  ``tab_id`` and ``chat_id``
          // are orthogonal — each local tab has its own routing key,
          // but there is at most one local tab per backend chat id.
          createNewTab();
          const taskText = s.preview || s.title || '';
          setTaskText(taskText);
          // Also copy the task text into the chat input textbox so the
          // user can edit and resubmit it without retyping.
          inp.value = taskText;
          syncClearBtn();
          vscode.postMessage({
            type: 'resumeSession',
            id: s.id,
            taskId: s.task_id,
            tabId: activeTabId,
          });
        } else {
          createNewTab();
          inp.value = s.preview || s.title || '';
          syncClearBtn();
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

  /**
   * Open a custom in-webview calendar popup anchored next to the
   * supplied date input.  Picking a day sets ``input.value`` to the
   * ISO ``YYYY-MM-DD`` string (the same format ``<input type=date>``
   * produces) and dispatches a ``change`` event so the existing
   * history filter listener runs.  Closes on outside click or Escape.
   *
   * @param {HTMLInputElement} input - the adjacent date text input to
   *   populate (e.g. ``#hf-from`` or ``#hf-to``).
   * @param {HTMLElement} anchorBtn - the calendar icon button used
   *   to anchor the popup position.
   */
  function openCustomDatePicker(input, anchorBtn) {
    if (!input) return;
    const existing = document.getElementById('kiss-datepicker-pop');
    if (existing) {
      const sameInput = existing._kissInput === input;
      existing.remove();
      if (sameInput) return; // toggle off when same button clicked twice
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
    // Seed viewed month from input value if present, else today
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

    function closePicker() {
      document.removeEventListener('mousedown', onDocClick, true);
      document.removeEventListener('keydown', onKey, true);
      window.removeEventListener('resize', position);
      if (pop.parentNode) pop.parentNode.removeChild(pop);
    }

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
    // Defer outside-click registration so the originating click that
    // opened the picker does not immediately close it.
    setTimeout(() => {
      document.addEventListener('mousedown', onDocClick, true);
      document.addEventListener('keydown', onKey, true);
      window.addEventListener('resize', position);
    }, 0);
  }

  /**
   * Show/hide rows in the history sidebar based on the filter bar
   * state (Running / Errors / Completed / Favorite checkboxes and
   * From / To date inputs).  Reads ``data-category``,
   * ``data-timestamp`` and ``data-favorite`` stamped on each row by
   * ``renderHistory``.  When every row is hidden, replaces the list
   * with a "No matching tasks" placeholder unless the unfiltered list
   * was itself empty.
   */
  function applyHistoryFilterVisibility() {
    const hfRunning = document.getElementById('hf-running');
    const hfErrors = document.getElementById('hf-errors');
    const hfCompleted = document.getElementById('hf-completed');
    const hfWorkspace = document.getElementById('hf-workspace');
    const hfFavorite = document.getElementById('hf-favorite');
    const hfFrom = document.getElementById('hf-from');
    const hfTo = document.getElementById('hf-to');
    if (!hfRunning || !hfErrors || !hfCompleted) return;
    const showRunning = hfRunning.checked;
    const showErrors = hfErrors.checked;
    const showCompleted = hfCompleted.checked;
    const onlyFavorite = hfFavorite && hfFavorite.checked;
    // Workspace filter — when checked, only show rows whose
    // ``data-work-dir`` (mirroring ``extra.work_dir`` set at task
    // completion) equals the client's currently-configured work
    // directory.  An empty client work_dir or an empty row work_dir
    // both pass the filter so the user sees rows that pre-date the
    // ``extra.work_dir`` persistence change and rows running in the
    // "no folder open" state.
    const onlyWorkspace = hfWorkspace && hfWorkspace.checked;
    const clientWorkDir = configWorkDir || '';
    // Date inputs are <input type=date> with value="YYYY-MM-DD".
    // Convert to local-midnight epoch seconds for inclusive bounds.
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
      const rowWorkDir = row.dataset.workDir || '';
      // Workspace match honors the documented contract above: an
      // empty client work_dir or an empty row work_dir BOTH pass so
      // (a) freshly-started running tasks whose ``extra.work_dir``
      //     hasn't been persisted yet still appear in History when
      //     the user opens the burger menu in a real workspace, and
      // (b) standalone web clients with no folder open ("client
      //     work_dir empty") see every row.
      // Without this, the strict ``rowWorkDir === clientWorkDir``
      // test silently hides every running-task row whose work_dir
      // hasn't been written yet — exactly the user-reported
      // "task panel does not show up in History after burger open"
      // regression.
      const wsOk =
        !onlyWorkspace ||
        rowWorkDir === '' ||
        clientWorkDir === '' ||
        rowWorkDir === clientWorkDir;
      if (catOk && dateOk && favOk && wsOk) {
        row.style.display = '';
        visible++;
      } else {
        row.style.display = 'none';
      }
    });
    // Manage the "no matches" placeholder.  Only show it when there
    // are rows in the list but the filter hides them all — never
    // when the unfiltered list was already empty (the existing "No
    // conversations yet" placeholder handles that case).
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
  }

  /**
   * Save the settings form to the backend if the form is currently
   * populated.  Used both when switching away from the Settings sub-tab
   * and when closing the unified sidebar while Settings is active.
   */
  function saveSettingsIfPopulated() {
    if (configFormPopulated) {
      const data = collectConfigForm();
      vscode.postMessage({type: 'saveConfig', ...data});
      // Standalone web client: editing the work_dir in Settings also
      // re-pins THIS instance (sessionStorage via the WS shim + the
      // server's per-connection work_dir), so the change applies to
      // this browser tab immediately while other instances keep their
      // own pinned folders.  saveConfig above still persists the value
      // globally as the default for NEW instances.
      if (
        document.body.classList.contains('remote-chat') &&
        typeof data.config.work_dir === 'string' &&
        data.config.work_dir
      ) {
        vscode.postMessage({
          type: 'setWorkDir',
          workDir: data.config.work_dir,
        });
      }
    }
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
  }

  /**
   * Open the standalone Settings panel (slides in from the right) and
   * request the current config from the backend so the form is freshly
   * populated.
   */
  function openSettingsPanel() {
    if (!settingsPanel) return;
    settingsPanel.classList.add('open');
    if (settingsOverlay) settingsOverlay.classList.add('open');
    configFormPopulated = false;
    vscode.postMessage({type: 'getConfig'});
  }

  /**
   * Close the standalone Settings panel.  If the config form is
   * populated, flush it to the backend via ``saveConfig`` first.
   */
  function closeSettingsPanel() {
    saveSettingsIfPopulated();
    if (settingsPanel) settingsPanel.classList.remove('open');
    if (settingsOverlay) settingsOverlay.classList.remove('open');
  }

  /**
   * Open the standalone Frequent tasks panel (slides up from the
   * bottom) and request the current frequent tasks from the backend.
   */
  function openFrequentPanel() {
    if (!frequentPanel) return;
    frequentPanel.classList.add('open');
    if (frequentOverlay) frequentOverlay.classList.add('open');
    vscode.postMessage({type: 'getFrequentTasks', limit: 50});
  }

  /** Close the standalone Frequent tasks panel. */
  function closeFrequentPanel() {
    if (frequentPanel) frequentPanel.classList.remove('open');
    if (frequentOverlay) frequentOverlay.classList.remove('open');
  }

  /**
   * Open the standalone Tricks panel (slides up from the bottom).
   * Trick texts are read from ``window.__TRICKS__`` which is injected
   * by the HTML builder after parsing ``src/kiss/INJECTIONS.md``.
   */
  function openTricksPanel() {
    if (!tricksPanel) return;
    tricksPanel.classList.add('open');
    if (tricksOverlay) tricksOverlay.classList.add('open');
    renderTricks(window.__TRICKS__ || []);
  }

  /** Close the standalone Tricks panel. */
  function closeTricksPanel() {
    if (tricksPanel) tricksPanel.classList.remove('open');
    if (tricksOverlay) tricksOverlay.classList.remove('open');
  }

  /**
   * Render the list of tricks inside the Tricks panel.  Each row is
   * clickable: clicking copies the trick text into the prompt textarea
   * and closes the panel — mirroring the click handler used by the
   * Frequent tasks list.
   */
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
        // Pad the injected text with whitespace on either side so it never
        // visually merges with adjacent input.  Skip the pad when the
        // neighbouring character is already whitespace or we are at the
        // boundary of the textarea — that avoids creating "  " runs.
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
        } catch (_e) {
          /* ignore selection errors on non-text inputs */
        }
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
      // Show the full text; CSS line-clamp on .frequent-item > .sidebar-item-text
      // clips it to two lines with an ellipsis.
      textSpan.textContent = text;
      div.appendChild(textSpan);

      const cnt = document.createElement('span');
      cnt.className = 'frequent-item-count';
      cnt.textContent = String(t.count);
      div.appendChild(cnt);

      // Copy-to-clipboard button — placed immediately left of the
      // trash icon so the user can copy the full task text without
      // first selecting/reopening the task.  ``text`` is the full
      // task string straight from the ``frequent_tasks.task`` column.
      const copyBtn = makeSidebarCopyButton(text);
      div.appendChild(copyBtn);

      // Delete button + inline confirm/cancel — mirrors the layout used
      // by the History sidebar rows so the user gets a consistent
      // "click trash → confirm Delete or Cancel" flow.  On confirm we
      // optimistically remove the row from the DOM and ask the backend
      // to delete the row from the ``frequent_tasks`` table.
      const delBtn = document.createElement('button');
      delBtn.className = 'sidebar-item-delete';
      delBtn.dataset.tooltip = 'Delete';
      delBtn.setAttribute('aria-label', 'Delete frequent task');
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
        cnt.style.display = 'none';
        confirmWrap.style.display = '';
      });

      confirmBtn.addEventListener('click', e => {
        e.stopPropagation();
        vscode.postMessage({type: 'deleteFrequentTask', task: text});
        div.remove();
      });

      cancelBtn.addEventListener('click', e => {
        e.stopPropagation();
        confirmWrap.style.display = 'none';
        delBtn.style.display = '';
        cnt.style.display = '';
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
  /**
   * Wire a show/hide eye-toggle button to a password input.
   *
   * Used both for the settings-panel password (cfg-remote-password) and
   * for the welcome-page mirror (welcome-cfg-remote-password).  The
   * function is a no-op if either DOM node is missing, so it is safe to
   * call unconditionally from setupEventListeners().
   */
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

  /**
   * Make a settings-panel input secret (masked) by default with a
   * show/hide eye toggle, matching the remote-password field.
   *
   * The input is switched to type="password", wrapped in a
   * .config-password-wrap div, and an eye-toggle button cloned from
   * the remote-password toggle is appended (so the SVG icon markup
   * lives in exactly one place: chat.html).  Used for the API-key
   * fields.  No-op when either DOM node is missing.
   */
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
  function populateConfigForm(cfg, apiKeys) {
    const el = id => document.getElementById(id);
    // Remember the configured work directory so ``workDirForTab`` can
    // fall back to it when a tab has not yet learned its own
    // ``workDir`` from a background-task event.
    const prevConfigWorkDir = configWorkDir;
    configWorkDir = cfg.work_dir || '';
    // The history sidebar's Workspace filter compares each row's
    // ``data-work-dir`` against ``configWorkDir`` — a change here
    // must re-run the visibility pass so already-rendered rows
    // immediately reflect the new client work_dir.
    if (prevConfigWorkDir !== configWorkDir) {
      try {
        applyHistoryFilterVisibility();
      } catch (_e) {
        /* history list not yet rendered */
      }
    }
    const wdInp = el('cfg-work-dir');
    if (wdInp) {
      wdInp.value = cfg.work_dir || '';
      if (!document.body.classList.contains('remote-chat')) {
        // In VS Code each window's work_dir is ALWAYS the workspace
        // folder open in that window (the extension overwrites
        // ``configData.config.work_dir`` with its own folder before
        // forwarding), so the field is informational and read-only.
        // Only the standalone web client — which has no workspace —
        // may edit it.
        wdInp.readOnly = true;
        wdInp.title = 'Set by the workspace folder open in this window';
      } else {
        // Standalone web client: each browser tab (= one webapp
        // instance) pins its own work_dir in sessionStorage under
        // 'sorcar-work-dir' (written by the WS shim's postMessage
        // hook and replayed to the server on every reconnect).
        // Prefer the pinned value over the globally persisted one so
        // another instance saving a different work_dir can never
        // change what THIS instance displays or uses; when no pin
        // exists yet (first configData after a fresh tab), adopt the
        // global value as this instance's pin.
        let pinned = '';
        try {
          // eslint-disable-next-line no-undef -- sessionStorage is a browser global
          pinned = sessionStorage.getItem('sorcar-work-dir') || '';
        } catch (_e) {
          /* sessionStorage may be unavailable in VS Code webviews */
        }
        if (pinned) {
          wdInp.value = pinned;
          // The pinned per-instance folder overrides the global
          // default for this browser tab, so prefer it as the
          // ``workDirForTab`` fallback too.
          configWorkDir = pinned;
        } else if (cfg.work_dir) {
          vscode.postMessage({type: 'setWorkDir', workDir: cfg.work_dir});
        }
      }
    }
    el('cfg-max-budget').value = cfg.max_budget != null ? cfg.max_budget : 100;
    el('cfg-custom-endpoint').value = cfg.custom_endpoint || '';
    el('cfg-custom-api-key').value = cfg.custom_api_key || '';
    el('cfg-custom-headers').value = cfg.custom_headers || '';
    el('cfg-use-web-browser').checked = cfg.use_web_browser !== false;
    el('cfg-demo-mode').checked = !!cfg.demo_mode || demoMode;
    demoMode = el('cfg-demo-mode').checked;
    el('cfg-remote-password').value = cfg.remote_password || '';
    // Also populate the welcome-page mirror (may not exist on some
    // alternate views; guarded by ``if (welcomePw)``).
    const welcomePw = el('welcome-cfg-remote-password');
    if (welcomePw) welcomePw.value = cfg.remote_password || '';
    configFormPopulated = true;
    // Populate API key fields from current environment values
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
      el('cfg-key-' + k).value = (apiKeys && apiKeys[k]) || '';
    });
  }
  function collectConfigForm() {
    const el = id => document.getElementById(id);
    const cfg = {
      max_budget: parseFloat(el('cfg-max-budget').value) || 100,
      custom_endpoint: el('cfg-custom-endpoint').value.trim(),
      custom_api_key: el('cfg-custom-api-key').value.trim(),
      custom_headers: el('cfg-custom-headers').value.trim(),
      use_web_browser: el('cfg-use-web-browser').checked,
      demo_mode: el('cfg-demo-mode').checked,
      remote_password: el('cfg-remote-password').value.trim(),
    };
    // Only the standalone web client may change the work_dir; in
    // VS Code the field is read-only (the workspace folder is the
    // work_dir) and is omitted so one window's save can never
    // overwrite the persisted work_dir with its own folder.
    const wdInp = el('cfg-work-dir');
    if (wdInp && !wdInp.readOnly) {
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
      vscode.postMessage({
        type: 'getFiles',
        prefix: atCtx.query,
        workDir: workDirForTab(activeTabId),
      });
    } else {
      hideAC();
    }
  }

  const _acSvg = {
    file: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    star: '<svg viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
    // Lightning bolt for the fast-complete picker (history/tricks/identifiers).
    bolt: '<svg viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    // Sparkle for INJECTIONS.md trick suggestions.
    spark:
      '<svg viewBox="0 0 24 24"><path d="M12 2l1.5 5L19 8.5 13.5 10 12 15 10.5 10 5 8.5 10.5 7 12 2z"/></svg>',
    // Curly-brace identifier glyph.
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

  function renderAutocomplete(data) {
    if (!data || !data.length) {
      hideAC();
      return;
    }
    autocomplete.innerHTML = '';
    acIdx = -1;
    const atMatch = getAtCtx();
    const searchQ = atMatch ? atMatch.query : '';
    const order = ['frequent', 'file'];
    const labels = {frequent: 'Frequent', file: 'Files'};
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
        const useSearch = searchQ && searchQ.length > 0;
        const textHtml = useSearch
          ? hlMatch(item.text, searchQ)
          : _acPathHtml(item.text);
        d.innerHTML =
          '<span class="ac-icon">' +
          _acIcon(item.type) +
          '</span>' +
          '<span class="ac-text">' +
          textHtml +
          '</span>';
        if (isFirst) {
          d.innerHTML += '<span class="ac-hint">tab</span>';
          isFirst = false;
        }
        d.addEventListener('click', () => {
          insertAtMention(item.text);
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
    const allItems = autocomplete.querySelectorAll('.ac-item');
    updateSel(allItems, acIdx);
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
      vscode.postMessage({
        type: 'recordFileUsage',
        path: file,
        workDir: workDirForTab(activeTabId),
      });
    }
    hideAC();
    inp.focus();
  }

  /**
   * Replace the input value with *full* and append a trailing space.
   * Used by the fast-complete picker: every completion item carries the
   * full line text (history task, trick, or identifier completion)
   * that should land in the textarea on accept — mirroring the CLI
   * dropdown's whole-line replacement contract.
   */
  function acceptCompletion(full) {
    inp.value = full;
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

  /**
   * Render the fast-complete dropdown picker for *data*: a list of
   * ``{type, text}`` items where ``text`` is the full replacement
   * line.  Reuses ``#autocomplete`` (and therefore the existing
   * keyboard handler at line ~5334), grouped by section with the
   * same DOM as the ``@``-mention file picker.
   *
   * The picker is suppressed while an ``@``-mention is active (file
   * picker takes precedence), while a task is running, when the
   * cursor is not at end, or when the input is empty.  The caller
   * is responsible for the staleness guard (query == inp.value).
   */
  function renderCompletions(data) {
    if (getAtCtx()) {
      // The ``@``-mention file picker owns ``#autocomplete`` in this
      // mode; never let completions clobber it — not even an empty
      // reply, which would otherwise call ``hideAC()`` and close the
      // file picker that arrived between request and reply.
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
    autocomplete.innerHTML = '';
    acIdx = -1;
    const order = ['task', 'frequent', 'trick', 'identifier'];
    const labels = {
      task: 'History',
      frequent: 'Frequent',
      trick: 'Suggestions',
      identifier: 'From editor',
    };
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
        const textHtml = hlMatch(item.text, inp.value);
        d.innerHTML =
          '<span class="ac-icon">' +
          _acIcon(item.type) +
          '</span>' +
          '<span class="ac-text">' +
          textHtml +
          '</span>';
        if (isFirst) {
          d.innerHTML += '<span class="ac-hint">tab</span>';
          isFirst = false;
        }
        d.addEventListener('click', () => {
          acceptCompletion(item.text);
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
    const allItems = autocomplete.querySelectorAll('.ac-item');
    updateSel(allItems, acIdx);
  }

  // Expose minimal API for demo.js
  window._demoApi = {
    get active() {
      return _demoActive;
    },
    set active(v) {
      _demoActive = !!v;
    },
    resolveEvents: null,
    createNewTab: createNewTab,
    setInput: function (text) {
      inp.value = text;
      syncClearBtn();
    },
    clearInput: function () {
      inp.value = '';
      syncClearBtn();
    },
    clearForReplay: function () {
      clearOutput();
      resetOutputState();
      clearUsageMetrics();
    },
    resetOutputState: function () {
      resetOutputState();
    },
    processEvent: processOutputEvent,
    setTaskText: setTaskText,
    updateTabTitle: updateActiveTabTitle,
    hideWelcome: function () {
      if (welcome) {
        welcome.style.display = 'none';
        refreshWelcomeLayout();
      }
    },
    scrollToBottom: sb,
    getActiveTabId: function () {
      return activeTabId;
    },
    sendMessage: function (msg) {
      vscode.postMessage(msg);
    },
    collapsePanels: function () {
      collapseAllExceptResult(O);
    },
    setRunningState: setRunningState,
    showSpinner: showSpinner,
    removeSpinner: removeSpinner,
  };

  // Start
  init();
})();

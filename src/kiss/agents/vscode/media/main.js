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

  // State — isRunning mirrors the active tab's tab.isRunning for UI controls
  let isRunning = false;
  let selectedModel = 'claude-opus-4-6';
  let allModels = [];
  let modelDDIdx = -1;
  let attachments = [];
  let _scrollLock = false;
  let _noScroll = false;
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

  // Adjacent task scroll state (Cursor-style chat thread navigation)
  // Tab.id is a frontend-only UUID string; chat_id is an int assigned by the DB.
  let currentTaskName = ''; // the originally loaded task
  let oldestLoadedTask = ''; // topmost task in the view (for scrolling up)
  let newestLoadedTask = ''; // bottommost task in the view (for scrolling down)
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

  function makeTab(title) {
    const _id = genTabId();
    return {
      id: _id,
      title: title || 'new chat',
      backendChatId: '',
      isRunning: false,
      outputFragment: null,
      taskPanelHTML: '',
      taskPanelVisible: false,
      panelsExpanded: false,
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
      taskQueue: [],
    };
  }

  /** Check if the active tab has a running task. */
  function isActiveTabRunning() {
    const tab = tabs.find(t => {
      return t.id === activeTabId;
    });
    return tab ? tab.isRunning : false;
  }

  /** Find the tab object that owns a backend message by tabId. */
  function findTabByEvt(ev) {
    if (ev && ev.tabId !== undefined) {
      return (
        tabs.find(t => {
          return t.id === ev.tabId;
        }) || null
      );
    }
    return null;
  }

  function saveCurrentTab() {
    const tab = tabs.find(t => {
      return t.id === activeTabId;
    });
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
      taskPanelText.textContent = tab.taskPanelHTML;
      if (tab.taskPanelVisible) taskPanel.classList.add('visible');
      else taskPanel.classList.remove('visible');
    }
    currentTaskName = tab.taskPanelHTML || '';
    updateChevronIcon(!!tab.panelsExpanded);
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
    }
    // Restore per-tab state
    selectedModel = tab.selectedModel || 'claude-opus-4-6';
    if (modelName) modelName.textContent = selectedModel;
    attachments = tab.attachments || [];
    renderFileChips();
    inp.value = tab.inputValue || '';
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
    isMerging = tab.isMerging || false;
    t0 = tab.t0 || null;
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
    // Set inputContainer visibility based on active bars
    if (
      worktreeBar ||
      autocommitBar ||
      document.getElementById('merge-toolbar')
    ) {
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
      el.className = 'chat-tab' + (tab.id === activeTabId ? ' active' : '');
      el.dataset.tabId = tab.id;

      if (tab.isRunning) {
        const spinner = document.createElement('span');
        spinner.className = 'chat-tab-spinner';
        el.appendChild(spinner);
      } else if (tab.hasRunTask) {
        const icon = document.createElement('span');
        icon.className = tab.lastTaskFailed
          ? 'chat-tab-status chat-tab-fail'
          : 'chat-tab-status chat-tab-ok';
        icon.textContent = tab.lastTaskFailed ? '\u2717' : '\u2713';
        el.appendChild(icon);
      }

      const label = document.createElement('span');
      label.className = 'chat-tab-label';
      label.textContent = tab.title;
      el.appendChild(label);

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

    // Add "+" button as a direct child of tab-bar (between tab-list and history-btn)
    const existingAdd = tabBar.querySelector('.chat-tab-add');
    if (!existingAdd) {
      const addBtn = document.createElement('div');
      addBtn.className = 'chat-tab chat-tab-add';
      addBtn.textContent = '+';
      addBtn.title = 'New chat';
      addBtn.addEventListener('click', () => {
        createNewTab();
      });
      tabBar.insertBefore(addBtn, document.getElementById('config-btn'));
    }

    // Scroll the active tab into view
    const activeEl = tabList.querySelector('.chat-tab.active');
    if (activeEl)
      activeEl.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }

  function switchToTab(tabId) {
    if (tabId === activeTabId) return;
    saveCurrentTab();
    const tab = tabs.find(t => {
      return t.id === tabId;
    });
    if (!tab) return;
    restoreTab(tab);
    renderTabBar();
    persistTabState();
    // Restore running state for the target tab
    setRunningState(tab.isRunning);
    if (!tab.isRunning) {
      t0 = null;
      stopTimer();
      removeSpinner();
    }
    applyChevronState(!!tab.panelsExpanded);
    focusInputWithRetry();
  }

  function closeTab(tabId) {
    const idx = tabs.findIndex(t => {
      return t.id === tabId;
    });
    if (idx < 0) return;
    tabs.splice(idx, 1);
    vscode.postMessage({type: 'closeTab', tabId: tabId});
    if (activeTabId === tabId) {
      if (tabs.length === 0) {
        // Last tab closed — create a fresh chat instead of closing
        // the secondary sidebar.
        createNewTab();
        return;
      }
      // Switch to an adjacent tab
      const newIdx = Math.min(idx, tabs.length - 1);
      const newTab = tabs[newIdx];
      restoreTab(newTab);
      // Restore running state for the new tab
      setRunningState(newTab.isRunning);
      if (!newTab.isRunning) {
        t0 = null;
        stopTimer();
        removeSpinner();
      }
      applyChevronState(!!newTab.panelsExpanded);
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
    vscode.postMessage({type: 'getWelcomeSuggestions'});
    focusInputWithRetry();
  }

  function updateActiveTabTitle(title) {
    const tab = tabs.find(t => {
      return t.id === activeTabId;
    });
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
    const serialized = tabs.map(t => {
      // Always use activeTabId for the active tab so the persisted
      // chatId stays in sync even when saveCurrentTab() hasn't run.
      return {
        title: t.title,
        chatId: t.id,
        backendChatId: t.backendChatId || '',
      };
    });
    const activeIdx = tabs.findIndex(t => {
      return t.id === activeTabId;
    });
    vscode.setState({
      tabs: serialized,
      activeTabIndex: activeIdx,
      chatId: activeTabId,
    });
  }

  // Initialize tabs — restore from saved state if available, else create one default tab
  (function () {
    const saved = vscode.getState();
    if (saved && saved.tabs && saved.tabs.length > 0) {
      tabs = [];
      saved.tabs.forEach(st => {
        const tab = makeTab(st.title);
        // Restore tab.id from persisted chatId (frontend tab identifier)
        if (st.chatId) tab.id = st.chatId;
        if (st.backendChatId) tab.backendChatId = st.backendChatId;
        tabs.push(tab);
      });
      const idx = saved.activeTabIndex || 0;
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
  const fileChips = document.getElementById('file-chips');

  const statusText = document.getElementById('status-text');
  const historyBtn = document.getElementById('history-btn');
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

  // Config sidebar elements
  const configBtn = document.getElementById('config-btn');
  const configSidebar = document.getElementById('config-sidebar');
  const configSidebarOverlay = document.getElementById(
    'config-sidebar-overlay',
  );
  const configSidebarClose = document.getElementById('config-sidebar-close');
  const cfgSaveBtn = document.getElementById('cfg-save-btn');
  const autocommitBtn = document.getElementById('autocommit-btn');
  const waitSpinner = document.getElementById('wait-spinner');
  const ghostOverlay = document.getElementById('ghost-overlay');
  const inputContainer = document.getElementById('input-container');
  const inputClearBtn = document.getElementById('input-clear-btn');
  const worktreeToggleBtn = document.getElementById('worktree-toggle-btn');
  const parallelToggleBtn = document.getElementById('parallel-toggle-btn');
  const demoToggleBtn = document.getElementById('demo-toggle-btn');
  const taskPanel = document.getElementById('task-panel');
  const taskPanelText = document.getElementById('task-panel-text');
  const taskPanelChevron = document.getElementById('task-panel-chevron');
  const statusTokens = document.getElementById('status-tokens');
  const statusBudget = document.getElementById('status-budget');
  const statusSteps = document.getElementById('status-steps');

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
   * Apply the chevron expand/collapse state to panels in the tab.
   * - expanded=false (chevron right, default): hide every .collapsible
   *   panel in #output (display:none via .chv-hidden) except result panels
   *   (.rc) and panels belonging to the currently running task.
   * - expanded=true (chevron down): reveal every hidden panel and expand
   *   every .collapsible panel in #output except those belonging to the
   *   currently running task.
   * Running task panels are direct children of #output (not inside
   *   .adjacent-task) while a task is running; adjacent-task containers
   *   hold previously-completed tasks.
   */
  function applyChevronState(expanded) {
    if (!O) return;
    const panels = O.querySelectorAll('.collapsible');
    for (let i = 0; i < panels.length; i++) {
      const p = panels[i];
      const inAdjacent = !!p.closest('.adjacent-task');
      const inRunning = isRunning && !inAdjacent;
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
      const tab = tabs.find(t => {
        return t.id === activeTabId;
      });
      const expanded = tab ? !tab.panelsExpanded : true;
      if (tab) tab.panelsExpanded = expanded;
      updateChevronIcon(expanded);
      applyChevronState(expanded);
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
      txtEl: null,
      txtBuf: '',
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
    oldestLoadedTask = currentTaskName;
    newestLoadedTask = currentTaskName;
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

  function renderAdjacentTask(direction, task, events) {
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
      oldestLoadedTask = task;
    } else {
      O.appendChild(container);
      newestLoadedTask = task;
    }
    const tab = tabs.find(x => {
      return x.id === activeTabId;
    });
    if (tab) applyChevronState(!!tab.panelsExpanded);
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
    inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
    return true;
  }

  /** Cycle to the previous (older) history item. Returns true if acted. */
  function cycleHistoryUp() {
    if (histCache.length > 0 && (histIdx >= 0 || !inp.value)) {
      histIdx = Math.min(histIdx + 1, histCache.length - 1);
      inp.value = histCache[histIdx];
      inp.style.height = 'auto';
      inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
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
    inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
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
  function mkEl(tag, cls) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    return e;
  }

  function hlBlock(el) {
    if (typeof hljs !== 'undefined')
      el.querySelectorAll('pre code').forEach(bl => {
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
      }
      collapsePreview(panelEl);
      setTimeout(() => {
        _noScroll = false;
      }, 0);
    });
  }

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
        target.appendChild(tState.thinkEl);
        break;
      case 'thinking_delta':
        if (tState.thinkEl) {
          const tc = tState.thinkEl.querySelector('.cnt');
          tc.textContent += (ev.text || '').replace(/\n\n+/g, '\n');
          tState.thinkEl.scrollTop = tState.thinkEl.scrollHeight;
        }
        break;
      case 'thinking_end':
        // Keep the thinking panel expanded so the streamed thinking
        // tokens remain visible after the block ends.  The user can
        // still click the "Thinking" label to manually collapse.
        tState.thinkEl = null;
        break;
      case 'text_delta':
        if (!tState.txtEl) {
          tState.txtEl = mkEl('div', 'txt');
          target.appendChild(tState.txtEl);
          tState.txtBuf = '';
        }
        tState.txtBuf += ev.text || '';
        tState.txtEl.textContent += (ev.text || '').replace(/\n\n+/g, '\n');
        break;
      case 'text_end':
        if (tState.txtEl) {
          if (typeof marked !== 'undefined') {
            tState.txtEl.classList.add('md-body');
            tState.txtEl.innerHTML = marked.parse(tState.txtBuf || '');
            hlBlock(tState.txtEl);
          }
          tState.txtEl = null;
          tState.txtBuf = '';
        }
        break;
      case 'tool_call': {
        if (tState.bashPanel && tState.bashBuf) {
          tState.bashPanel.textContent += tState.bashBuf;
          tState.bashBuf = '';
        }
        tState.bashPanel = null;
        tState.bashRaf = 0;
        const c = mkEl('div', 'ev tc');
        const hdr = mkEl('div', 'tc-h');
        hdr.textContent = ev.name || 'Tool';
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
        if (ev.command) {
          const bp = mkEl('div', 'bash-panel');
          const bpContent = mkEl('div', 'bash-panel-content');
          bp.appendChild(bpContent);
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
        }
        const hadBash = !!tState.bashPanel;
        tState.bashPanel = null;
        tState.bashRaf = 0;
        if (hadBash && !ev.is_error) break;
        const resultTarget = tState.lastToolCallEl || target;
        if (ev.is_error) {
          const r = mkEl('div', 'ev tr err');
          r.innerHTML =
            '<div class="rl fail">FAILED</div><div class="tr-content">' +
            esc(ev.content) +
            '</div>';
          addCollapse(r, r.querySelector('.rl'));
          resultTarget.appendChild(r);
        } else {
          const op = mkEl('div', 'bash-panel');
          const opContent = mkEl('div', 'bash-panel-content');
          opContent.textContent = ev.content;
          op.appendChild(opContent);
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
              if (tState.bashPanel)
                tState.bashPanel.textContent += tState.bashBuf;
              tState.bashBuf = '';
              tState.bashRaf = 0;
              if (tState.bashPanel)
                tState.bashPanel.scrollTop = tState.bashPanel.scrollHeight;
            });
          }
        } else {
          const s = mkEl('div', 'ev sys');
          s.textContent = (ev.text || '').replace(/\n\n+/g, '\n');
          target.appendChild(s);
        }
        break;
      }
      case 'result': {
        const rc = mkEl('div', 'ev rc');
        let rb = '';
        if (ev.success === false) {
          rb +=
            '<div style="color:var(--red);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px">Status: FAILED</div>';
        }
        let usePre = true;
        if (ev.summary) {
          const sum = (ev.summary || '').replace(/\n{3,}/g, '\n\n').trim();
          if (typeof marked !== 'undefined') {
            rb += marked.parse(sum);
            usePre = false;
          } else {
            rb += esc(sum);
          }
        } else {
          rb += esc(
            (ev.text || '(no result)').replace(/\n{3,}/g, '\n\n').trim(),
          );
        }
        rc.innerHTML =
          '<div class="rc-h"><h3>Result</h3><div class="rs">' +
          '<span>Tokens <b>' +
          fmtN(ev.total_tokens || 0) +
          '</b></span>' +
          '<span>Cost <b>' +
          (ev.cost || 'N/A') +
          '</b></span>' +
          '</div></div><div class="rc-body md-body' +
          (usePre ? ' pre' : '') +
          '">' +
          rb +
          '</div>';
        hlBlock(rc);
        target.appendChild(rc);
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
            ? marked.parse(ev.text || '')
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
        addCollapse(el, el.querySelector('.' + cls + '-h'));
        hlBlock(el);
        target.appendChild(el);
        const bodyEl = el.querySelector('.' + cls + '-body');
        if (bodyEl) bodyEl.scrollTop = bodyEl.scrollHeight;
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
      llmPanel = null;
      llmPanelState = mkS();
      pendingPanel = false;
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
      llmPanel = mkEl('div', 'llm-panel');
      const lHdr = mkEl('div', 'llm-panel-hdr');
      lHdr.textContent = 'Thoughts';
      addCollapse(llmPanel, lHdr);
      llmPanel.appendChild(lHdr);
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
      collapseAllExceptResult(O);
      if (ev.success === false) {
        const rTab = tabs.find(x => {
          return x.id === activeTabId;
        });
        if (rTab) rTab.lastTaskFailed = true;
      }
    }
    if (target === llmPanel) llmPanel.scrollTop = llmPanel.scrollHeight;
    // Keep the chevron "right" state consistent across new panels added by streaming.
    // Skip during demo replay — demo mode never sets isRunning so
    // applyChevronState(false) would hide every non-result panel via chv-hidden.
    const tab = tabs.find(x => {
      return x.id === activeTabId;
    });
    if (tab && !tab.panelsExpanded && !_demoActive) applyChevronState(false);
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
      bgLlmPanel = null;
      bgLlmPanelState = mkS();
      bgPendingPanel = false;
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
      bgLlmPanel = mkEl('div', 'llm-panel');
      const lHdr = mkEl('div', 'llm-panel-hdr');
      lHdr.textContent = 'Thoughts';
      addCollapse(bgLlmPanel, lHdr);
      bgLlmPanel.appendChild(lHdr);
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
        if (ev.step_count) {
          bgStepCount = ev.step_count;
          tab.statusStepsText = 'Steps: ' + ev.step_count;
        }
        if (ev.total_tokens)
          tab.statusTokensText = 'Tokens: ' + fmtN(ev.total_tokens);
        if (ev.cost && ev.cost !== 'N/A')
          tab.statusBudgetText = 'Cost: ' + ev.cost;
        collapseAllExceptResult(tab.outputFragment);
        if (ev.success === false) tab.lastTaskFailed = true;
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

  O.addEventListener('wheel', e => {
    if (isRunning && e.deltaY < 0) _scrollLock = true;

    // Adjacent task loading via overscroll detection
    if (!adjacentLoading && activeTabId && currentTaskName) {
      const atTop = O.scrollTop <= 0;
      const atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 2;

      if (atTop && e.deltaY < 0 && !noPrevTask && oldestLoadedTask) {
        // Scrolling up at top — load task before the oldest loaded
        if (overscrollDir !== 'prev') {
          overscrollAccum = 0;
          overscrollDir = 'prev';
        }
        overscrollAccum += Math.abs(e.deltaY);
        clearTimeout(overscrollTimer);
        overscrollTimer = setTimeout(() => {
          overscrollAccum = 0;
          overscrollDir = '';
        }, 500);
        if (overscrollAccum >= OVERSCROLL_THRESHOLD) {
          overscrollAccum = 0;
          overscrollDir = '';
          adjacentLoading = true;
          showAdjacentLoader('prev');
          vscode.postMessage({
            type: 'getAdjacentTask',
            tabId: activeTabId,
            task: oldestLoadedTask,
            direction: 'prev',
          });
        }
      } else if (atBottom && e.deltaY > 0 && !noNextTask && newestLoadedTask) {
        // Scrolling down at bottom — load task after the newest loaded
        if (overscrollDir !== 'next') {
          overscrollAccum = 0;
          overscrollDir = 'next';
        }
        overscrollAccum += Math.abs(e.deltaY);
        clearTimeout(overscrollTimer);
        overscrollTimer = setTimeout(() => {
          overscrollAccum = 0;
          overscrollDir = '';
        }, 500);
        if (overscrollAccum >= OVERSCROLL_THRESHOLD) {
          overscrollAccum = 0;
          overscrollDir = '';
          adjacentLoading = true;
          showAdjacentLoader('next');
          vscode.postMessage({
            type: 'getAdjacentTask',
            tabId: activeTabId,
            task: newestLoadedTask,
            direction: 'next',
          });
        }
      } else {
        overscrollAccum = 0;
        overscrollDir = '';
      }
    }
  });
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
  function startTimer() {
    if (!t0) t0 = Date.now();
    if (timerIv) clearInterval(timerIv);
    statusText.style.color = 'var(--red)';
    timerIv = setInterval(() => {
      const s = Math.floor((Date.now() - t0) / 1000);
      const m = Math.floor(s / 60);
      statusText.textContent =
        'Running ' + (m > 0 ? m + 'm ' : '') + (s % 60) + 's';
    }, 1000);
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

  // --- Main event handler ---
  function handleEvent(ev) {
    const t = ev.type;
    switch (t) {
      case 'status': {
        const evTab = findTabByEvt(ev);
        if (evTab) {
          evTab.isRunning = !!ev.running;
        }
        // Update UI only when the event targets the active tab (or no tabId)
        if (!evTab || evTab.id === activeTabId) {
          setRunningState(ev.running);
        }
        renderTabBar();
        break;
      }
      case 'models':
        allModels = ev.models || [];
        if (ev.selected) {
          selectedModel = ev.selected;
          modelName.textContent = ev.selected;
        }
        renderModelList('');
        break;
      case 'configData':
        populateConfigForm(ev.config || {}, ev.apiKeys || {});
        break;
      case 'history':
        renderHistory(ev.sessions || [], ev.offset || 0, ev.generation || 0);
        break;
      case 'files':
        renderAutocomplete(ev.files || []);
        break;
      case 'askUser': {
        const askTabId = ev.tabId !== undefined ? ev.tabId : activeTabId;
        const askTab = tabs.find(t => {
          return t.id === askTabId;
        });
        if (!askTab) break;
        askTab.askPendingQuestion = ev.question || '';
        showAskForTab(askTab);
        break;
      }
      case 'error':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
        addError(ev.text);
        break;
      case 'clear': {
        const clearTab =
          ev.tabId !== undefined
            ? tabs.find(t => {
                return t.id === ev.tabId;
              })
            : tabs.find(t => {
                return t.id === activeTabId;
              });
        if (clearTab) {
          clearTab.lastTaskFailed = false;
          clearTab.hasRunTask = true;
        }
        if (ev.chat_id && clearTab) {
          clearTab.backendChatId = ev.chat_id;
          persistTabState();
        }
        const evTabId = ev.tabId;
        if (evTabId === undefined || evTabId === activeTabId) {
          clearOutput();
          resetOutputState();
          showSpinner();
        }
        renderTabBar();
        break;
      }
      case 'clearChat': {
        const ccTab = tabs.find(t => t.id === activeTabId);
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
        const swTab = tabs.find(t => {
          return t.id === swTabId;
        });
        if (swTab) {
          if (swTabId === activeTabId) {
            clearOutput();
            resetOutputState();
            if (welcome) {
              welcome.style.display = '';
              O.appendChild(welcome);
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
        renderRemoteUrl(ev.url);
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
        sb();
        break;
      }
      case 'tasks_updated':
        refreshHistory();
        vscode.postMessage({type: 'getInputHistory'});
        break;

      case 'task_events': {
        const teTabId = ev.tabId || activeTabId;
        const teTab = tabs.find(t => {
          return t.id === teTabId;
        });
        if (ev.chat_id && teTab) {
          teTab.backendChatId = ev.chat_id;
          persistTabState();
        }
        // Non-active tab: render into a document fragment without touching the DOM
        if (teTabId !== activeTabId && teTab) {
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
              if (bgExtra.model) teTab.selectedModel = bgExtra.model;
              if (bgExtra.work_dir) teTab.workDir = bgExtra.work_dir;
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
          let bgSteps = 0,
            bgPending = false,
            bgLastTool = '';
          (ev.events || []).forEach(e => {
            const t = e.type;
            if (t === 'tool_call') {
              bgLastTool = e.name || '';
              bgPending = false;
            }
            if (t === 'tool_result' && bgLastTool !== 'finish')
              bgPending = true;
            if (bgSteps === 0 && (t === 'thinking_start' || t === 'text_delta'))
              bgSteps = 1;
            if (bgPending && (t === 'thinking_start' || t === 'text_delta')) {
              bgSteps++;
              bgPending = false;
            }
            if (t === 'result' && e.step_count) bgSteps = e.step_count;
          });
          if (bgSteps > 0) teTab.statusStepsText = 'Steps: ' + bgSteps;
          break;
        }
        // Active tab: render directly into the DOM
        if (ev.task) {
          currentTaskName = ev.task;
          resetAdjacentState(); // sets oldest/newest to currentTaskName
          setTaskText(ev.task);
          if (welcome) welcome.style.display = 'none';
          updateActiveTabTitle(ev.task);
        }
        if (ev.extra) {
          try {
            const extra = JSON.parse(ev.extra);
            if (extra.model) {
              selectedModel = extra.model;
              if (modelName) modelName.textContent = selectedModel;
              const curTab = tabs.find(t => {
                return t.id === activeTabId;
              });
              if (curTab) curTab.selectedModel = selectedModel;
            }
            if (extra.work_dir) {
              const wdTab = tabs.find(t => {
                return t.id === activeTabId;
              });
              if (wdTab) wdTab.workDir = extra.work_dir;
            }
            if (worktreeToggleBtn) {
              if (extra.is_worktree) worktreeToggleBtn.classList.add('active');
              else worktreeToggleBtn.classList.remove('active');
            }
            if (parallelToggleBtn) {
              if (extra.is_parallel) parallelToggleBtn.classList.add('active');
              else parallelToggleBtn.classList.remove('active');
            }
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
        renderAdjacentTask(ev.direction, ev.task, ev.events || []);
        break;
      case 'setTaskText': {
        const stt = (ev.text || '').trim();
        if (ev.tabId === undefined || ev.tabId === activeTabId) {
          if (stt) {
            currentTaskName = stt;
            resetAdjacentState();
            if (welcome) welcome.style.display = 'none';
            updateActiveTabTitle(stt);
          }
          setTaskText(ev.text || '');
        } else if (stt) {
          // Update background tab's saved title without touching active tab
          const sttTab = tabs.find(t => {
            return t.id === ev.tabId;
          });
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

      case 'inputHistory':
        histCache = ev.tasks || [];
        if (histIdx < 0) histIdx = -1;
        break;
      case 'ghost':
        if (ev.suggestion && ev.query === inp.value) {
          updateGhost(ev.suggestion);
        }
        break;

      case 'merge_data': {
        const mdEl = mkEl('div', 'ev merge-info');
        let mergeHtml =
          '<div class="merge-info-hdr" style="color:var(--yellow);font-weight:600;font-size:var(--fs-base);margin-bottom:4px">' +
          '\u2731 Reviewing ' +
          (ev.hunk_count || 0) +
          ' change(s)</div>' +
          '<div class="merge-info-body" style="font-size:var(--fs-md);color:var(--dim)">Red = old lines, Green = new lines. ' +
          'Use the merge toolbar to accept or reject changes.</div>';
        // Render inline diff for web clients (when file contents are present)
        const mergeFiles = (ev.data && ev.data.files) || [];
        for (let mfi = 0; mfi < mergeFiles.length; mfi++) {
          const mf = mergeFiles[mfi];
          if (mf.base_text !== undefined && mf.current_text !== undefined) {
            mergeHtml +=
              '<div class="merge-file-diff" style="margin-top:8px;">';
            mergeHtml +=
              '<div style="font-weight:600;color:var(--vscode-textLink-foreground);margin-bottom:4px;">' +
              (mf.name || 'unknown') +
              '</div>';
            mergeHtml +=
              '<pre style="font-size:12px;line-height:1.4;overflow-x:auto;background:var(--vscode-input-background);padding:8px;border-radius:4px;margin:0;">';
            const baseLines = (mf.base_text || '').split('\n');
            const curLines = (mf.current_text || '').split('\n');
            const hunks = mf.hunks || [];
            // Build a unified diff view
            let curIdx = 0;
            for (let mhi = 0; mhi < hunks.length; mhi++) {
              const h = hunks[mhi];
              // Context lines before hunk
              while (curIdx < h.cs) {
                mergeHtml +=
                  '<span style="color:var(--dim);">' +
                  esc(' ' + curLines[curIdx]) +
                  '</span>\n';
                curIdx++;
              }
              // Old (base) lines - red
              for (let bi = h.bs; bi < h.bs + h.bc; bi++) {
                mergeHtml +=
                  '<span style="color:#f44;background:rgba(255,60,60,0.15);">-' +
                  esc(baseLines[bi] || '') +
                  '</span>\n';
              }
              // New (current) lines - green
              for (let ci = h.cs; ci < h.cs + h.cc; ci++) {
                mergeHtml +=
                  '<span style="color:#4c4;background:rgba(60,255,60,0.15);">+' +
                  esc(curLines[ci] || '') +
                  '</span>\n';
              }
              curIdx = h.cs + h.cc;
            }
            // Remaining context
            while (curIdx < curLines.length) {
              mergeHtml +=
                '<span style="color:var(--dim);">' +
                esc(' ' + curLines[curIdx]) +
                '</span>\n';
              curIdx++;
            }
            mergeHtml += '</pre></div>';
          }
        }
        mdEl.innerHTML = mergeHtml;
        addCollapse(mdEl, mdEl.querySelector('.merge-info-hdr'));
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab: append to saved output fragment
          const bgMdTab = tabs.find(t => t.id === ev.tabId);
          if (bgMdTab && bgMdTab.outputFragment) {
            bgMdTab.outputFragment.appendChild(mdEl);
          }
          break;
        }
        O.appendChild(mdEl);
        collapseOlderPanels();
        break;
      }
      case 'merge_started':
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          // Background tab's merge: mark it and auto-switch so the user
          // sees the merge/diff interface immediately.
          const bgMergeTab = tabs.find(t => {
            return t.id === ev.tabId;
          });
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
          const mrt2 = tabs.find(t => {
            return t.id === ev.tabId;
          });
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
        // Update merge toolbar with remaining hunk count
        const mergeTitle = document.querySelector('.merge-toolbar-title');
        if (mergeTitle && ev.remaining !== undefined) {
          mergeTitle.textContent =
            'Review Changes (' + ev.remaining + '/' + ev.total + ' remaining)';
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
              return 'PWD/' + p;
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
          const bgWtTab = tabs.find(t => t.id === ev.tabId);
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
          const bgWrTab = tabs.find(t => t.id === ev.tabId);
          if (bgWrTab) {
            bgWrTab.worktreeBarEl = null;
            if (bgWrTab.outputFragment) {
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
          const bgAcTab = tabs.find(t => t.id === ev.tabId);
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
          const bgAdTab = tabs.find(t => t.id === ev.tabId);
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
          const rt = tabs.find(t => {
            return t.id === ev.tabId;
          });
          if (rt) doneT0 = rt.t0;
        }
        const el = doneT0 ? Math.floor((Date.now() - doneT0) / 1000) : 0;
        const em = Math.floor(el / 60);
        markTabDone(ev.tabId, ev.success === false);
        setReady(
          'Done (' + (em > 0 ? em + 'm ' : '') + (el % 60) + 's)',
          ev.tabId,
        );
        break;
      }
      case 'task_error':
      case 'task_stopped': {
        const isErr = t === 'task_error';
        markTabDone(ev.tabId, true);
        setReady(isErr ? 'Error' : 'Stopped', ev.tabId);
        break;
      }
      default:
        if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
          const bgTab = findTabByEvt(ev);
          if (bgTab) processOutputEventForBgTab(ev, bgTab);
          break;
        }
        processOutputEvent(ev);
        if (isActiveTabRunning()) showSpinner();
        sb();
        break;
    }
  }

  function updateInputDisabled() {
    // Only block input during merge — not while running (tasks queue locally)
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
    // Show both send and stop buttons when running so users can queue tasks
    sendBtn.style.display = 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';

    if (uploadBtn) uploadBtn.disabled = running;
    if (autocommitBtn) autocommitBtn.disabled = running;
    if (worktreeToggleBtn) worktreeToggleBtn.disabled = running;
    if (parallelToggleBtn) parallelToggleBtn.disabled = running;
    if (demoToggleBtn) demoToggleBtn.disabled = running;
    if (modelBtn) {
      modelBtn.disabled = running;
      if (running) closeModelDD();
    }
    updateInputDisabled();
    updateQueueIndicator();
    if (running) {
      startTimer();
    }
  }

  function markTabDone(tabId, failed) {
    const tid = tabId !== undefined ? tabId : activeTabId;
    const tab = tabs.find(t => {
      return t.id === tid;
    });
    if (tab) {
      tab.hasRunTask = true;
      tab.lastTaskFailed = !!failed;
    }
  }

  function setReady(label, tabId) {
    // Mark the tab as no longer running
    const tid = tabId !== undefined ? tabId : activeTabId;
    let doneTab = null;
    if (tabId !== undefined) {
      doneTab = tabs.find(t => {
        return t.id === tabId;
      });
      if (doneTab) {
        doneTab.isRunning = false;
        doneTab.t0 = null;
      }
    }
    // Update UI only if the event targets the active tab (or no tabId)
    if (tabId === undefined || tabId === activeTabId) {
      t0 = null;
      setRunningState(false);
      stopTimer();
      removeSpinner();
      statusText.textContent = label || 'Ready';
      inp.focus();
    }
    renderTabBar();

    // Process the next queued task if any
    if (!doneTab) doneTab = tabs.find(t => t.id === tid);
    if (doneTab && doneTab.taskQueue && doneTab.taskQueue.length > 0) {
      setTimeout(() => {
        submitNextQueuedTask(doneTab);
      }, 200);
    }
  }

  function addError(text) {
    const div = mkEl('div', 'ev tr err');
    div.innerHTML = '<strong>Error:</strong> ' + esc(text);
    O.appendChild(div);
    sb();
  }

  // --- Remote URL (dynamic) ---
  function renderRemoteUrl(url) {
    const container = document.getElementById('remote-url');
    if (!container) return;
    container.innerHTML = '';
    if (!url) return;
    const wrapper = document.createElement('div');
    wrapper.className = 'remote-url-bar';
    const label = document.createElement('div');
    label.className = 'remote-url-label';
    label.textContent = 'Web/mobile app';
    const row = document.createElement('div');
    row.className = 'remote-url-row';
    const link = document.createElement('a');
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = url;
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
      navigator.clipboard.writeText(url).then(() => {
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
    container.appendChild(wrapper);
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
      chip.innerHTML =
        '<span class="chip-label">Suggested</span>' + esc(s.text);
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
    events.forEach(ev => {
      const t = ev.type;
      if (t === 'task_done' || t === 'task_error' || t === 'task_stopped') {
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
        rPendingPanel = false;
      }
      if (t === 'tool_result' && rLastToolName !== 'finish') {
        rPendingPanel = true;
      }
      if (rPendingPanel && (t === 'thinking_start' || t === 'text_delta')) {
        rLlmPanel = mkEl('div', 'llm-panel');
        const lHdr = mkEl('div', 'llm-panel-hdr');
        lHdr.textContent = 'Thoughts';
        addCollapse(rLlmPanel, lHdr);
        rLlmPanel.appendChild(lHdr);
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
    collapseAllExceptResult(container);
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
    let rSteps = 0,
      rPending = false,
      rLastTool = '';
    events.forEach(ev => {
      const t = ev.type;
      if (t === 'tool_call') {
        rLastTool = ev.name || '';
        rPending = false;
      }
      if (t === 'tool_result' && rLastTool !== 'finish') rPending = true;
      if (rSteps === 0 && (t === 'thinking_start' || t === 'text_delta'))
        rSteps = 1;
      if (rPending && (t === 'thinking_start' || t === 'text_delta')) {
        rSteps++;
        rPending = false;
      }
      if (t === 'result' && ev.step_count) rSteps = ev.step_count;
    });
    if (rSteps > 0) updateStepCount(rSteps);
    // Snapshot the current task's metrics for adjacent-scroll restoration
    currentTaskMetrics.tokens = statusTokens ? statusTokens.textContent : '';
    currentTaskMetrics.budget = statusBudget ? statusBudget.textContent : '';
    currentTaskMetrics.steps = statusSteps ? statusSteps.textContent : '';
    const tab = tabs.find(x => {
      return x.id === activeTabId;
    });
    if (tab) applyChevronState(!!tab.panelsExpanded);
    sb();
  }

  // --- Worktree merge/discard UI ---

  let worktreeBar = null;

  function clearWorktreeBar() {
    if (worktreeBar && worktreeBar.parentNode) {
      worktreeBar.parentNode.removeChild(worktreeBar);
    }
    worktreeBar = null;
    if (inputContainer) inputContainer.style.display = '';
  }

  /** Create a worktree merge/discard bar element. ownerTabId is captured
   *  in button closures so the correct tab is targeted even if the user
   *  switches tabs before clicking. */
  function createWorktreeBar(ownerTabId) {
    const bar = mkEl('div', 'wt-bar');
    const label = mkEl('span', 'wt-label');
    label.textContent = 'Auto-commit and merge or Discard?';
    bar.appendChild(label);

    const btns = mkEl('div', 'wt-btns');
    const mergeBtn = mkEl('button', 'wt-btn wt-merge');
    mergeBtn.textContent = 'Auto-commit and merge';
    mergeBtn.addEventListener('click', () => {
      disableWtBtns();
      vscode.postMessage({
        type: 'worktreeAction',
        action: 'merge',
        tabId: ownerTabId,
      });
    });

    const discardBtn = mkEl('button', 'wt-btn wt-discard');
    discardBtn.textContent = 'Discard';
    discardBtn.addEventListener('click', () => {
      disableWtBtns();
      vscode.postMessage({
        type: 'worktreeAction',
        action: 'discard',
        tabId: ownerTabId,
      });
    });

    btns.appendChild(mergeBtn);
    btns.appendChild(discardBtn);
    bar.appendChild(btns);
    return bar;
  }

  function showWorktreeActions(ev) {
    clearWorktreeBar();
    const ownerTabId = (ev && ev.tabId) || activeTabId;
    const bar = createWorktreeBar(ownerTabId);
    // Hide the input container and show the worktree bar in its place
    if (inputContainer) inputContainer.style.display = 'none';
    const area = document.getElementById('input-area');
    area.insertBefore(bar, area.firstChild);
    worktreeBar = bar;
  }

  function disableWtBtns() {
    if (!worktreeBar) return;
    const btns = worktreeBar.querySelectorAll('.wt-btn');
    btns.forEach(b => {
      b.disabled = true;
    });
  }

  function handleWorktreeResult(ev) {
    clearWorktreeBar();
    const cls = ev.success ? 'wt-result-ok' : 'wt-result-err';
    const div = mkEl('div', 'ev ' + cls);
    const msg = ev.message || '';
    div.textContent = msg;
    O.appendChild(div);
    sb();
  }

  // --- Autocommit prompt UI (non-worktree mode) ---
  // After the user resolves all merge-diff hunks, the backend sends an
  // `autocommit_prompt` event when the main branch still has dirty
  // state.  We show "Auto commit" / "Do nothing" buttons in the input
  // area, matching the worktree merge/discard bar.

  let autocommitBar = null;

  function clearAutocommitBar() {
    if (autocommitBar && autocommitBar.parentNode) {
      autocommitBar.parentNode.removeChild(autocommitBar);
    }
    autocommitBar = null;
    if (inputContainer) inputContainer.style.display = '';
  }

  /** Create an autocommit bar element. ownerTabId is captured in button
   *  closures so the correct tab is targeted even after a tab switch. */
  function createAutocommitBar(ev) {
    const ownerTabId = (ev && ev.tabId) || activeTabId;
    const bar = mkEl('div', 'wt-bar');
    const label = mkEl('span', 'wt-label');
    const n = (ev && ev.changedFiles && ev.changedFiles.length) || 0;
    label.textContent =
      n === 1
        ? '1 uncommitted change on main. Auto commit?'
        : n + ' uncommitted changes on main. Auto commit?';
    bar.appendChild(label);

    const btns = mkEl('div', 'wt-btns');
    const commitBtn = mkEl('button', 'wt-btn wt-merge');
    commitBtn.textContent = 'Auto commit';
    commitBtn.addEventListener('click', () => {
      disableAutocommitBtns();
      vscode.postMessage({
        type: 'autocommitAction',
        action: 'commit',
        tabId: ownerTabId,
      });
    });

    const skipBtn = mkEl('button', 'wt-btn wt-discard');
    skipBtn.textContent = 'Do nothing';
    skipBtn.addEventListener('click', () => {
      disableAutocommitBtns();
      vscode.postMessage({
        type: 'autocommitAction',
        action: 'skip',
        tabId: ownerTabId,
      });
    });

    btns.appendChild(commitBtn);
    btns.appendChild(skipBtn);
    bar.appendChild(btns);
    return bar;
  }

  function showAutocommitActions(ev) {
    clearAutocommitBar();
    const bar = createAutocommitBar(ev);
    if (inputContainer) inputContainer.style.display = 'none';
    const area = document.getElementById('input-area');
    area.insertBefore(bar, area.firstChild);
    autocommitBar = bar;
  }

  function disableAutocommitBtns() {
    if (!autocommitBar) return;
    const btns = autocommitBar.querySelectorAll('.wt-btn');
    btns.forEach(b => {
      b.disabled = true;
    });
  }

  function handleAutocommitResult(ev) {
    clearAutocommitBar();
    const cls = ev && ev.success ? 'wt-result-ok' : 'wt-result-err';
    const div = mkEl('div', 'ev ' + cls);
    div.textContent = (ev && ev.message) || '';
    O.appendChild(div);
    sb();
    focusInputWithRetry();
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
      inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
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
    if (worktreeToggleBtn) {
      worktreeToggleBtn.addEventListener('click', () => {
        worktreeToggleBtn.classList.toggle('active');
      });
    }
    if (parallelToggleBtn) {
      parallelToggleBtn.addEventListener('click', () => {
        parallelToggleBtn.classList.toggle('active');
      });
    }

    if (demoToggleBtn) {
      demoToggleBtn.addEventListener('click', () => {
        if (_demoActive) {
          // Cancel running demo
          if (typeof window._cancelDemoReplay === 'function')
            window._cancelDemoReplay();
          demoMode = false;
          _demoActive = false;
          demoToggleBtn.classList.remove('active');
          return;
        }
        demoMode = !demoMode;
        demoToggleBtn.classList.toggle('active', demoMode);
      });
    }

    if (autocommitBtn) {
      autocommitBtn.addEventListener('click', () => {
        vscode.postMessage({
          type: 'autocommitAction',
          action: 'commit',
          tabId: activeTabId,
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
      if (isRunning) return;
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
    historyBtn.addEventListener('click', () => {
      if (sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        resetHistoryPagination();
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('open');
        historyBtn.classList.add('open');
        vscode.postMessage({
          type: 'getHistory',
          query: historySearch.value,
          generation: historyGeneration,
        });
      }
    });
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    configBtn.addEventListener('click', () => {
      if (configSidebar.classList.contains('open')) {
        closeConfigSidebar();
      } else {
        openConfigSidebar();
      }
    });
    configSidebarClose.addEventListener('click', closeConfigSidebar);
    configSidebarOverlay.addEventListener('click', closeConfigSidebar);
    cfgSaveBtn.addEventListener('click', () => {
      const data = collectConfigForm();
      vscode.postMessage({type: 'saveConfig', ...data});
      closeConfigSidebar();
    });
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
    const curTab = tabs.find(t => {
      return t.id === activeTabId;
    });

    // If a task is running, queue this task locally for later execution
    if (isRunning) {
      if (curTab) {
        curTab.taskQueue.push({
          prompt: prompt,
          model: selectedModel,
          attachments: attachments.map(a => ({
            name: a.name,
            mimeType: a.type,
            data: a.data,
          })),
          useWorktree: !!(
            worktreeToggleBtn && worktreeToggleBtn.classList.contains('active')
          ),
          useParallel: !!(
            parallelToggleBtn && parallelToggleBtn.classList.contains('active')
          ),
          workDir: curTab.workDir || '',
        });
        // Tell backend to skip merge/diff for the currently running task
        // since there are now queued tasks that should run first.
        vscode.postMessage({
          type: 'setSkipMerge',
          tabId: activeTabId,
          skip: true,
        });
        inp.value = '';
        inp.style.height = 'auto';
        attachments = [];
        renderFileChips();
        clearGhost();
        histIdx = -1;
        if (inputClearBtn) inputClearBtn.style.display = 'none';
        updateQueueIndicator();
      }
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
      useWorktree: !!(
        worktreeToggleBtn && worktreeToggleBtn.classList.contains('active')
      ),
      useParallel: !!(
        parallelToggleBtn && parallelToggleBtn.classList.contains('active')
      ),
    };
    if (curTab && curTab.workDir) msg.workDir = curTab.workDir;
    vscode.postMessage(msg);
    inp.value = '';
    inp.style.height = 'auto';
    attachments = [];
    renderFileChips();
    clearGhost();
    histIdx = -1;
    if (inputClearBtn) inputClearBtn.style.display = 'none';
  }

  /** Submit the next queued task for a tab. */
  function submitNextQueuedTask(tab) {
    if (!tab || !tab.taskQueue || tab.taskQueue.length === 0) return;
    const task = tab.taskQueue.shift();
    updateQueueIndicator();
    // If there are still more queued tasks, skip merge for this task.
    // If this is the last task, allow merge to run.
    const hasMoreQueued = tab.taskQueue.length > 0;
    const msg = {
      type: 'submit',
      prompt: task.prompt,
      model: task.model,
      tabId: tab.id,
      attachments: task.attachments || [],
      useWorktree: !!task.useWorktree,
      useParallel: !!task.useParallel,
      skipMerge: hasMoreQueued,
      reuseProcess: true,
    };
    if (task.workDir) msg.workDir = task.workDir;
    vscode.postMessage(msg);
  }

  /** Update the queue count indicator badge near the stop button. */
  function updateQueueIndicator() {
    let badge = document.getElementById('queue-badge');
    const curTab = tabs.find(t => t.id === activeTabId);
    const count = curTab && curTab.taskQueue ? curTab.taskQueue.length : 0;
    if (count === 0) {
      if (badge) badge.style.display = 'none';
      return;
    }
    if (!badge) {
      badge = document.createElement('span');
      badge.id = 'queue-badge';
      badge.style.cssText =
        'display:inline-flex;align-items:center;justify-content:center;' +
        'min-width:18px;height:18px;padding:0 5px;border-radius:9px;' +
        'background:var(--vscode-badge-background, #4d4d4d);' +
        'color:var(--vscode-badge-foreground, #fff);' +
        'font-size:11px;font-weight:600;margin-right:4px;';
      // Insert before stopBtn
      stopBtn.parentNode.insertBefore(badge, stopBtn);
    }
    badge.textContent = count + ' queued';
    badge.style.display = 'inline-flex';
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
      if (e.key === 'Enter') {
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
      tab.askQuestionEl.innerHTML = marked.parse(t);
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

  /** Submit the current answer for the given tab; clear pending question. */
  function submitAskForTab(tab) {
    const answer = tab.askInputEl ? tab.askInputEl.value : '';
    vscode.postMessage({type: 'userAnswer', answer: answer, tabId: tab.id});
    tab.askPendingQuestion = null;
    if (tab.askInputEl) tab.askInputEl.value = '';
    if (tab.id === activeTabId) clearAskSlot();
  }

  /**
   * Synchronise the shared modal slot with the active tab after a tab
   * switch: detach previous contents and mount the active tab's ask UI if
   * it has a pending question.
   */
  function syncAskModalToActiveTab() {
    clearAskSlot();
    const tab = tabs.find(t => {
      return t.id === activeTabId;
    });
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

    sessions.forEach(s => {
      const div = document.createElement('div');
      div.className = 'sidebar-item';
      const itemText = s.title || s.preview || 'Untitled';
      div.dataset.tooltip = s.preview || itemText;
      div.style.backgroundColor = chatIdBgColor(String(s.id));
      div.style.color = '#1a1a1a';

      const textSpan = document.createElement('span');
      textSpan.className = 'sidebar-item-text';
      textSpan.textContent = itemText;
      div.appendChild(textSpan);

      if (s.task_id) {
        const delBtn = document.createElement('button');
        delBtn.className = 'sidebar-item-delete';
        delBtn.dataset.tooltip = 'Delete task';
        delBtn.innerHTML =
          '<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor"><path d="M5.5 5.5a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm5 0a.5.5 0 0 1 .5.5v6a.5.5 0 0 1-1 0V6a.5.5 0 0 1 .5-.5zm-7-1a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1H14a.5.5 0 0 1 0 1h-.5l-.8 8.4A2 2 0 0 1 10.71 15H5.29a2 2 0 0 1-1.99-1.8L2.5 4.8H2a.5.5 0 0 1 0-1h1.5zM3.51 5.5l.79 8.2a1 1 0 0 0 .99.8h5.42a1 1 0 0 0 .99-.8l.79-8.2H3.51zM6 2.5a.5.5 0 0 0-.5.5h5a.5.5 0 0 0-.5-.5H6z"/></svg>';

        const confirmWrap = document.createElement('span');
        confirmWrap.className = 'sidebar-item-confirm';
        confirmWrap.style.display = 'none';

        const confirmBtn = document.createElement('button');
        confirmBtn.className = 'sidebar-confirm-yes';
        confirmBtn.dataset.tooltip = 'Confirm delete';
        confirmBtn.innerHTML =
          '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M13.78 4.22a.75.75 0 0 1 0 1.06l-7.25 7.25a.75.75 0 0 1-1.06 0L2.22 9.28a.75.75 0 1 1 1.06-1.06L6 10.94l6.72-6.72a.75.75 0 0 1 1.06 0z"/></svg>';

        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'sidebar-confirm-no';
        cancelBtn.dataset.tooltip = 'Cancel';
        cancelBtn.innerHTML =
          '<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M3.72 3.72a.75.75 0 0 1 1.06 0L8 6.94l3.22-3.22a.75.75 0 1 1 1.06 1.06L9.06 8l3.22 3.22a.75.75 0 1 1-1.06 1.06L8 9.06l-3.22 3.22a.75.75 0 0 1-1.06-1.06L6.94 8 3.72 4.78a.75.75 0 0 1 0-1.06z"/></svg>';

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

        div.appendChild(delBtn);
        div.appendChild(confirmWrap);
      }

      div.addEventListener('click', () => {
        if (demoMode && typeof window._startDemoReplay === 'function') {
          closeSidebar();
          createNewTab();
          window._startDemoReplay(allHistSessions);
          return;
        }
        createNewTab();
        if (s.has_events && s.id) {
          setTaskText(s.preview || s.title || '');
          vscode.postMessage({
            type: 'resumeSession',
            id: s.id,
            tabId: activeTabId,
          });
        } else {
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
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    sidebarOverlay.classList.remove('open');
    historyBtn.classList.remove('open');
  }

  function openConfigSidebar() {
    closeConfigSidebar();
    closeSidebar();
    vscode.postMessage({type: 'getConfig'});
    configSidebar.classList.add('open');
    configSidebarOverlay.classList.add('open');
    configBtn.classList.add('open');
  }
  function closeConfigSidebar() {
    configSidebar.classList.remove('open');
    configSidebarOverlay.classList.remove('open');
    configBtn.classList.remove('open');
  }
  function populateConfigForm(cfg, apiKeys) {
    const el = id => document.getElementById(id);
    el('cfg-max-budget').value = cfg.max_budget != null ? cfg.max_budget : 100;
    el('cfg-custom-endpoint').value = cfg.custom_endpoint || '';
    el('cfg-custom-api-key').value = cfg.custom_api_key || '';
    el('cfg-use-web-browser').checked = cfg.use_web_browser !== false;
    el('cfg-remote-password').value = cfg.remote_password || '';
    // Populate API key fields from current environment values
    const keyIds = [
      'GEMINI_API_KEY',
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'TOGETHER_API_KEY',
      'OPENROUTER_API_KEY',
      'MINIMAX_API_KEY',
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
      use_web_browser: el('cfg-use-web-browser').checked,
      remote_password: el('cfg-remote-password').value.trim(),
    };
    const apiKeys = {};
    const keyIds = [
      'GEMINI_API_KEY',
      'OPENAI_API_KEY',
      'ANTHROPIC_API_KEY',
      'TOGETHER_API_KEY',
      'OPENROUTER_API_KEY',
      'MINIMAX_API_KEY',
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
      vscode.postMessage({type: 'getFiles', prefix: atCtx.query});
    } else {
      hideAC();
    }
  }

  const _acSvg = {
    file: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    star: '<svg viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>',
  };
  function _acIcon(type) {
    if (type === 'frequent') return _acSvg.star;
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
      const mention = 'PWD/' + file;
      inp.value = before + mention + sep + after;
      syncClearBtn();
      const np = before.length + mention.length + sep.length;
      inp.setSelectionRange(np, np);
      vscode.postMessage({type: 'recordFileUsage', path: file});
    }
    hideAC();
    inp.focus();
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
      if (welcome) welcome.style.display = 'none';
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

/**
 * KISS Sorcar Webview JavaScript
 * Uses the same event protocol and rendering as the browser-based Sorcar.
 */

(function() {
  // @ts-ignore - vscode is injected by the webview
  const vscode = acquireVsCodeApi();

  /** Format a number with thousand separators (e.g. 12345 → "12,345"). */
  function fmtN(n) { return Number(n).toLocaleString('en-US'); }

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

  // Infinite scroll state for history sidebar
  var historyOffset = 0;
  var historyLoading = false;
  var historyHasMore = true;
  var historyGeneration = 0;

  // Adjacent task scroll state (Cursor-style chat thread navigation)
  var currentChatId = '';     // chat session identifier
  var currentTaskName = '';   // the originally loaded task
  var oldestLoadedTask = '';  // topmost task in the view (for scrolling up)
  var newestLoadedTask = '';  // bottommost task in the view (for scrolling down)
  var adjacentLoading = false;
  var noPrevTask = false;     // true when server says no prev exists
  var noNextTask = false;     // true when server says no next exists
  var overscrollAccum = 0;
  var overscrollDir = '';
  var overscrollTimer = null;
  var OVERSCROLL_THRESHOLD = 150; // pixels of accumulated overscroll to trigger load


  // --- Chat tabs state ---
  var tabIdCounter = 0;
  var tabs = [];        // array of tab objects (see makeTab for fields)
  var activeTabId = -1;

  function makeTab(title) {
    var id = ++tabIdCounter;
    return {
      id: id,
      title: title || 'new chat',
      isRunning: false,
      outputFragment: null,
      taskPanelHTML: '',
      taskPanelVisible: false,
      chatId: '',
      statusTokensText: '',
      statusBudgetText: '',
      statusStepsText: '',
      welcomeVisible: true,
      selectedModel: selectedModel,
      attachments: [],
      inputValue: '',
      isMerging: false,
      worktreeBarEl: null,
      mergeToolbarEl: null,
      t0: null,
      workDir: '',
      streamState: null,
      streamLlmPanel: null,
      streamLlmPanelState: null,
      streamLastToolName: '',
      streamPendingPanel: false,
    };
  }

  /** Check if the active tab has a running task. */
  function isActiveTabRunning() {
    var tab = tabs.find(function(t) { return t.id === activeTabId; });
    return tab ? tab.isRunning : false;
  }

  /** Find the tab object that owns a backend message by tabId. */
  function findTabByEvt(ev) {
    if (ev && ev.tabId !== undefined) {
      return tabs.find(function(t) { return t.id === ev.tabId; }) || null;
    }
    return null;
  }

  function saveCurrentTab() {
    var tab = tabs.find(function(t) { return t.id === activeTabId; });
    if (!tab) return;
    // Save welcome visibility and detach from O before capturing fragment
    tab.welcomeVisible = welcome ? welcome.style.display !== 'none' : true;
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    // Save DOM subtree as fragment (preserves element references for streaming state)
    tab.outputFragment = document.createDocumentFragment();
    while (O.firstChild) tab.outputFragment.appendChild(O.firstChild);
    tab.taskPanelHTML = taskPanel ? taskPanel.textContent : '';
    tab.taskPanelVisible = taskPanel ? taskPanel.classList.contains('visible') : false;
    tab.chatId = currentChatId;
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
    // Save merge toolbar (detach from DOM)
    var mergeBar = document.getElementById('merge-toolbar');
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
    if (taskPanel) {
      taskPanel.textContent = tab.taskPanelHTML;
      if (tab.taskPanelVisible) taskPanel.classList.add('visible');
      else taskPanel.classList.remove('visible');
    }
    currentChatId = tab.chatId || '';
    currentTaskName = tab.taskPanelHTML || '';
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
    if (worktreeBar && worktreeBar.parentNode) worktreeBar.parentNode.removeChild(worktreeBar);
    worktreeBar = null;
    if (tab.worktreeBarEl) {
      worktreeBar = tab.worktreeBarEl;
      tab.worktreeBarEl = null;
      var area = document.getElementById('input-area');
      area.insertBefore(worktreeBar, area.firstChild);
    }
    // Restore merge toolbar
    var existingMerge = document.getElementById('merge-toolbar');
    if (existingMerge) existingMerge.remove();
    if (tab.mergeToolbarEl) {
      document.getElementById('input-area').appendChild(tab.mergeToolbarEl);
      tab.mergeToolbarEl = null;
    } else if (isMerging) {
      showMergeToolbar();
    }
    // Set inputContainer visibility based on active bars
    if (worktreeBar || document.getElementById('merge-toolbar')) {
      if (inputContainer) inputContainer.style.display = 'none';
    } else {
      if (inputContainer) inputContainer.style.display = '';
    }
    updateInputDisabled();
    resetAdjacentState();
  }

  function renderTabBar() {
    var tabList = document.getElementById('tab-list');
    var tabBar = document.getElementById('tab-bar');
    if (!tabList || !tabBar) return;

    // Always show the tab bar
    tabBar.style.display = '';

    tabList.innerHTML = '';
    tabs.forEach(function(tab) {
      var el = document.createElement('div');
      el.className = 'chat-tab' + (tab.id === activeTabId ? ' active' : '');
      el.dataset.tabId = tab.id;

      var label = document.createElement('span');
      label.className = 'chat-tab-label';
      label.textContent = tab.title;
      el.appendChild(label);

      var closeBtn = document.createElement('span');
      closeBtn.className = 'chat-tab-close';
      closeBtn.textContent = '\u00d7';
      closeBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        closeTab(tab.id);
      });
      el.appendChild(closeBtn);

      el.addEventListener('click', function() {
        switchToTab(tab.id);
      });
      tabList.appendChild(el);
    });

    // Add "+" button as a direct child of tab-bar (between tab-list and history-btn)
    var existingAdd = tabBar.querySelector('.chat-tab-add');
    if (!existingAdd) {
      var addBtn = document.createElement('div');
      addBtn.className = 'chat-tab chat-tab-add';
      addBtn.textContent = '+';
      addBtn.title = 'New chat';
      addBtn.addEventListener('click', function() {
        createNewTab();
      });
      tabBar.insertBefore(addBtn, document.getElementById('history-btn'));
    }

    // Scroll the active tab into view
    var activeEl = tabList.querySelector('.chat-tab.active');
    if (activeEl) activeEl.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  function switchToTab(tabId) {
    if (tabId === activeTabId) return;
    saveCurrentTab();
    var tab = tabs.find(function(t) { return t.id === tabId; });
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
      statusText.textContent = 'Ready';
    }
    // Tell backend to resume this tab's session
    if (tab.chatId) {
      vscode.postMessage({ type: 'resumeSession', id: tab.chatId });
    } else {
      vscode.postMessage({ type: 'newChat' });
      vscode.postMessage({ type: 'getWelcomeSuggestions' });
    }
  }

  function closeTab(tabId) {
    var idx = tabs.findIndex(function(t) { return t.id === tabId; });
    if (idx < 0) return;
    tabs.splice(idx, 1);
    if (activeTabId === tabId) {
      if (tabs.length === 0) {
        // Last tab closed — open a fresh new chat
        createNewTab();
        return;
      }
      // Switch to an adjacent tab
      var newIdx = Math.min(idx, tabs.length - 1);
      var newTab = tabs[newIdx];
      restoreTab(newTab);
      // Restore running state for the new tab
      setRunningState(newTab.isRunning);
      if (!newTab.isRunning) {
        t0 = null;
        stopTimer();
        removeSpinner();
        statusText.textContent = 'Ready';
      }
      if (newTab.chatId) {
        vscode.postMessage({ type: 'resumeSession', id: newTab.chatId });
      } else {
        vscode.postMessage({ type: 'newChat' });
        vscode.postMessage({ type: 'getWelcomeSuggestions' });
      }
    }
    renderTabBar();
    persistTabState();
  }

  function createNewTab() {
    // Preserve any typed text so it carries over to the new tab
    var pendingText = inp.value || '';
    saveCurrentTab();
    var tab = makeTab('new chat');
    tabs.push(tab);
    activeTabId = tab.id;
    // Reset UI for fresh tab
    clearOutput();
    resetOutputState();
    resetAdjacentState();
    currentChatId = '';
    currentTaskName = '';
    removeSpinner();
    clearWorktreeBar();
    clearUsageMetrics();
    setTaskText('');
    // Reset per-tab state for new tab (selectedModel inherited via makeTab)
    attachments = [];
    renderFileChips();
    inp.value = pendingText;
    syncClearBtn();
    inp.style.height = 'auto';
    inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
    isMerging = false;
    hideMergeToolbar();
    t0 = null;
    if (welcome) {
      welcome.style.display = '';
      O.appendChild(welcome);
    }
    renderTabBar();
    persistTabState();
    // New tab is never running
    setRunningState(false);
    stopTimer();
    statusText.textContent = 'Ready';
    vscode.postMessage({ type: 'newChat' });
    vscode.postMessage({ type: 'getWelcomeSuggestions' });
  }

  function updateActiveTabTitle(title) {
    var tab = tabs.find(function(t) { return t.id === activeTabId; });
    if (!tab) return;
    var t = (title || '').trim();
    tab.title = t ? (t.length > 30 ? t.substring(0, 30) + '\u2026' : t) : 'new chat';
    tab.chatId = currentChatId;
    renderTabBar();
    persistTabState();
  }

  /** Persist lightweight tab metadata via vscode.setState for cross-restart restore. */
  function persistTabState() {
    var serialized = tabs.map(function(t) {
      // Always use currentChatId for the active tab so the persisted
      // chatId stays in sync even when saveCurrentTab() hasn't run.
      var chatId = (t.id === activeTabId) ? currentChatId : t.chatId;
      return { title: t.title, chatId: chatId };
    });
    var activeIdx = tabs.findIndex(function(t) { return t.id === activeTabId; });
    vscode.setState({ tabs: serialized, activeTabIndex: activeIdx, chatId: currentChatId });
  }

  // Initialize tabs — restore from saved state if available, else create one default tab
  var _restoredActiveChatId = '';
  (function() {
    var saved = vscode.getState();
    if (saved && saved.tabs && saved.tabs.length > 0) {
      tabs = [];
      tabIdCounter = 0;
      saved.tabs.forEach(function(st) {
        var tab = makeTab(st.title);
        tab.chatId = st.chatId || '';
        tabs.push(tab);
      });
      var idx = saved.activeTabIndex || 0;
      if (idx >= 0 && idx < tabs.length) {
        activeTabId = tabs[idx].id;
        currentChatId = tabs[idx].chatId || '';
        _restoredActiveChatId = currentChatId;
      } else {
        activeTabId = tabs[0].id;
      }
    } else {
      var initial = makeTab('new chat');
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
  const runPromptBtn = document.getElementById('run-prompt-btn');
  const sidebar = document.getElementById('sidebar');
  const sidebarOverlay = document.getElementById('sidebar-overlay');
  const sidebarClose = document.getElementById('sidebar-close');
  const historySearch = document.getElementById('history-search');
  const historyList = document.getElementById('history-list');
  const autocomplete = document.getElementById('autocomplete');
  const askUserModal = document.getElementById('ask-user-modal');
  const askUserQuestion = document.getElementById('ask-user-question');
  const askUserInput = document.getElementById('ask-user-input');
  const askUserSubmit = document.getElementById('ask-user-submit');
  const waitSpinner = document.getElementById('wait-spinner');
  const ghostOverlay = document.getElementById('ghost-overlay');
  const inputContainer = document.getElementById('input-container');
  const inputClearBtn = document.getElementById('input-clear-btn');
  const worktreeToggleBtn = document.getElementById('worktree-toggle-btn');
  const parallelToggleBtn = document.getElementById('parallel-toggle-btn');
  const taskPanel = document.getElementById('task-panel');
  const statusTokens = document.getElementById('status-tokens');
  const statusBudget = document.getElementById('status-budget');
  const statusSteps = document.getElementById('status-steps');


  function setTaskText(text) {
    if (!taskPanel) return;
    var t = (text || '').trim();
    if (t) {
      taskPanel.textContent = t;
      taskPanel.classList.add('visible');
    } else {
      taskPanel.textContent = '';
      taskPanel.classList.remove('visible');
    }
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

  function mkS() { return { thinkEl: null, txtEl: null, bashPanel: null, bashBuf: '', bashRaf: 0, lastToolCallEl: null }; }

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
    if (overscrollTimer) { clearTimeout(overscrollTimer); overscrollTimer = null; }
  }

  function showAdjacentLoader(direction) {
    removeAdjacentLoader();
    var loader = mkEl('div', 'adjacent-loader');
    loader.id = 'adjacent-loader';
    loader.textContent = 'Loading ' + (direction === 'prev' ? 'previous' : 'next') + ' task…';
    if (direction === 'prev') {
      O.insertBefore(loader, O.firstChild);
    } else {
      O.appendChild(loader);
    }
  }

  function removeAdjacentLoader() {
    var el = document.getElementById('adjacent-loader');
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
    var container = mkEl('div', 'adjacent-task');
    container.dataset.task = task;
    var separator = mkEl('div', 'adjacent-separator');
    var taskLabel = task.length > 80 ? task.substring(0, 80) + '…' : task;
    separator.innerHTML = '<span class="adjacent-sep-line"></span>'
      + '<span class="adjacent-sep-label">' + esc(taskLabel) + '</span>'
      + '<span class="adjacent-sep-line"></span>';
    container.appendChild(separator);

    // Replay events into the container (save/restore header metrics so
    // adjacent-task replay doesn't overwrite the current task's values)
    var savedTokens = statusTokens ? statusTokens.textContent : '';
    var savedBudget = statusBudget ? statusBudget.textContent : '';
    replayEventsInto(container, events);
    if (statusTokens) statusTokens.textContent = savedTokens;
    if (statusBudget) statusBudget.textContent = savedBudget;

    if (direction === 'prev') {
      // Save scroll position, prepend, then restore
      var prevScrollHeight = O.scrollHeight;
      O.insertBefore(container, O.firstChild);
      var newScrollHeight = O.scrollHeight;
      O.scrollTop += (newScrollHeight - prevScrollHeight);
      oldestLoadedTask = task;
    } else {
      O.appendChild(container);
      newestLoadedTask = task;
    }
  }

  function clearOutput() {
    if (welcome && welcome.parentNode === O) O.removeChild(welcome);
    O.innerHTML = '';
  }

  // --- Spinner ---
  function removeSpinner() {
    if (_spinnerTimer) { clearTimeout(_spinnerTimer); _spinnerTimer = null; }
    if (waitSpinner) waitSpinner.classList.remove('active');
  }
  function showSpinner() {
    removeSpinner();
    _spinnerTimer = setTimeout(function() {
      _spinnerTimer = null;
      if (waitSpinner) waitSpinner.classList.add('active');
    }, 250);
  }

  // --- Ghost text ---
  function clearGhost() {
    currentGhost = '';
    if (ghostOverlay) ghostOverlay.innerHTML = '';
    if (ghostTimer) { clearTimeout(ghostTimer); ghostTimer = null; }
  }

  function updateGhost(suggestion) {
    currentGhost = suggestion || '';
    if (!ghostOverlay || !currentGhost) { clearGhost(); return; }
    var val = inp.value;
    ghostOverlay.innerHTML = '<span style="visibility:hidden">' + esc(val) + '</span>'
      + '<span class="ghost-text">' + esc(currentGhost) + '</span>';
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
    ghostTimer = setTimeout(function() {
      ghostTimer = null;
      vscode.postMessage({ type: 'complete', query: inp.value });
    }, 300);
  }

  // --- File path detection (matches web Sorcar) ---
  // --- Shared rendering (ported from browser EVENT_HANDLER_JS) ---

  function esc(t) { var d = document.createElement('div'); d.textContent = t; return d.innerHTML; }

  // --- Custom tooltip (native title doesn't work in VS Code webviews) ---
  var tooltipEl = document.createElement('div');
  tooltipEl.id = 'custom-tooltip';
  document.body.appendChild(tooltipEl);
  var tooltipTimer = null;
  document.addEventListener('mouseover', function(e) {
    var target = e.target.closest('[data-tooltip]');
    if (!target) return;
    clearTimeout(tooltipTimer);
    tooltipTimer = setTimeout(function() {
      tooltipEl.textContent = target.dataset.tooltip;
      var rect = target.getBoundingClientRect();
      tooltipEl.style.left = rect.left + 'px';
      tooltipEl.style.top = (rect.bottom + 4) + 'px';
      tooltipEl.classList.add('visible');
    }, 400);
  });
  document.addEventListener('mouseout', function(e) {
    var target = e.target.closest('[data-tooltip]');
    if (!target) return;
    clearTimeout(tooltipTimer);
    tooltipEl.classList.remove('visible');
  });
  function mkEl(tag, cls) { var e = document.createElement(tag); if (cls) e.className = cls; return e; }

  function hlBlock(el) {
    if (typeof hljs !== 'undefined') el.querySelectorAll('pre code').forEach(function(bl) { hljs.highlightElement(bl); });
  }

  function toggleThink(el) {
    var p = el.parentElement;
    p.querySelector('.cnt').classList.toggle('hidden');
    el.querySelector('.arrow').classList.toggle('collapsed');
  }

  function collapsePreview(panelEl) {
    var prev = panelEl.querySelector('.collapse-preview');
    if (!prev) return;
    if (!panelEl.classList.contains('collapsed')) { prev.textContent = ''; return; }
    var txt = '';
    for (var i = 0; i < panelEl.children.length; i++) {
      var ch = panelEl.children[i];
      if (ch.classList.contains('collapse-chv') || ch === prev
        || ch.querySelector('.collapse-chv')) continue;
      txt += (ch.textContent || '') + ' ';
    }
    txt = txt.replace(/\s+/g, ' ').trim();
    prev.textContent = txt;
  }

  function addCollapse(panelEl, headerEl) {
    panelEl.classList.add('collapsible');
    var chv = mkEl('span', 'collapse-chv');
    chv.textContent = '\u25BE';
    var prev = mkEl('span', 'collapse-preview');
    headerEl.insertBefore(chv, headerEl.firstChild);
    headerEl.appendChild(prev);
    headerEl.classList.add('collapse-header');
    headerEl.style.cursor = 'pointer';
    headerEl.style.userSelect = 'none';
    headerEl.addEventListener('click', function(e) {
      e.stopPropagation();
      _noScroll = true;
      panelEl.classList.toggle('collapsed');
      if (panelEl.classList.contains('collapsed')) {
        panelEl.classList.remove('user-pinned');
      } else {
        panelEl.classList.add('user-pinned');
      }
      collapsePreview(panelEl);
      setTimeout(function() { _noScroll = false; }, 0);
    });
  }

  function collapseAllExceptResult(container) {
    var panels = container.querySelectorAll('.collapsible');
    for (var i = 0; i < panels.length; i++) {
      if (!panels[i].classList.contains('rc')) {
        panels[i].classList.add('collapsed');
        collapsePreview(panels[i]);
      }
    }
  }

  function collapseOlderPanels() {
    if (!isRunning) return;
    var panels = O.querySelectorAll(':scope > .collapsible');
    for (var i = 0; i < panels.length - 1; i++) {
      if (!panels[i].classList.contains('rc') && !panels[i].classList.contains('user-pinned')) {
        panels[i].classList.add('collapsed');
        collapsePreview(panels[i]);
      }
    }
  }

  window.toggleThink = toggleThink;

  function lineDiff(a, b) {
    var al = a.split('\n'), bl = b.split('\n'), m = al.length, n = bl.length;
    var dp = [];
    for (var i = 0; i <= m; i++) { dp[i] = new Array(n + 1); dp[i][0] = 0; }
    for (var j = 0; j <= n; j++) dp[0][j] = 0;
    for (var i = 1; i <= m; i++)
      for (var j = 1; j <= n; j++)
        dp[i][j] = al[i-1] === bl[j-1] ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
    var ops = [], ci = m, cj = n;
    while (ci > 0 || cj > 0) {
      if (ci > 0 && cj > 0 && al[ci-1] === bl[cj-1]) { ops.unshift({t:'=',o:al[--ci],n:bl[--cj]}); }
      else if (cj > 0 && (ci === 0 || dp[ci][cj-1] >= dp[ci-1][cj])) { ops.unshift({t:'+',n:bl[--cj]}); }
      else { ops.unshift({t:'-',o:al[--ci]}); }
    }
    return ops;
  }

  function hlInline(oldL, newL) {
    var mn = Math.min(oldL.length, newL.length), pre = 0, suf = 0;
    while (pre < mn && oldL[pre] === newL[pre]) pre++;
    while (suf < mn - pre && oldL[oldL.length-1-suf] === newL[newL.length-1-suf]) suf++;
    var pf = oldL.substring(0, pre), sf = suf ? oldL.substring(oldL.length - suf) : '';
    return {
      o: esc(pf) + '<span class="diff-hl-del">' + esc(oldL.substring(pre, oldL.length - suf)) + '</span>' + esc(sf),
      n: esc(pf) + '<span class="diff-hl-add">' + esc(newL.substring(pre, newL.length - suf)) + '</span>' + esc(sf)
    };
  }

  function renderDiff(oldStr, newStr) {
    var ops = lineDiff(oldStr, newStr), html = '', i = 0;
    while (i < ops.length) {
      var dels = [], adds = [];
      while (i < ops.length && ops[i].t === '-') { dels.push(ops[i++]); }
      while (i < ops.length && ops[i].t === '+') { adds.push(ops[i++]); }
      if (dels.length || adds.length) {
        var pairs = Math.min(dels.length, adds.length);
        for (var p = 0; p < pairs; p++) {
          var h = hlInline(dels[p].o, adds[p].n);
          html += '<div class="diff-old">- ' + h.o + '</div>';
          html += '<div class="diff-new">+ ' + h.n + '</div>';
        }
        for (var p = pairs; p < dels.length; p++)
          html += '<div class="diff-old">- ' + esc(dels[p].o) + '</div>';
        for (var p = pairs; p < adds.length; p++)
          html += '<div class="diff-new">+ ' + esc(adds[p].n) + '</div>';
        continue;
      }
      html += '<div class="diff-ctx">  ' + esc(ops[i].o) + '</div>'; i++;
    }
    return html;
  }

  function handleOutputEvent(ev, target, tState) {
    var t = ev.type;
    switch (t) {
    case 'thinking_start':
      tState.thinkEl = mkEl('div', 'ev think');
      tState.thinkEl.innerHTML =
        '<div class="lbl" onclick="toggleThink(this)">'
        + '<span class="arrow">\u25BE</span> Thinking</div>'
        + '<div class="cnt"></div>';
      target.appendChild(tState.thinkEl); break;
    case 'thinking_delta':
      if (tState.thinkEl) {
        var tc = tState.thinkEl.querySelector('.cnt');
        tc.textContent += (ev.text || '').replace(/\n\n+/g, '\n');
        tState.thinkEl.scrollTop = tState.thinkEl.scrollHeight;
      } break;
    case 'thinking_end':
      if (tState.thinkEl) {
        tState.thinkEl.querySelector('.lbl').innerHTML =
          '<span class="arrow collapsed">\u25BE</span> Thinking (click to expand)';
        tState.thinkEl.querySelector('.cnt').classList.add('hidden');
      }
      tState.thinkEl = null; break;
    case 'text_delta':
      if (!tState.txtEl) { tState.txtEl = mkEl('div', 'txt'); target.appendChild(tState.txtEl); }
      tState.txtEl.textContent += (ev.text || '').replace(/\n\n+/g, '\n'); break;
    case 'text_end':
      tState.txtEl = null; break;
    case 'tool_call': {
      if (tState.bashPanel && tState.bashBuf) { tState.bashPanel.textContent += tState.bashBuf; tState.bashBuf = ''; }
      tState.bashPanel = null; tState.bashRaf = 0;
      var c = mkEl('div', 'ev tc');
      var hdr = mkEl('div', 'tc-h');
      hdr.textContent = ev.name || 'Tool';
      var b = '';
      if (ev.path) {
        var ep = esc(ev.path).replace(/"/g, '&quot;');
        b += '<div class="tc-arg"><span class="tc-arg-name">path:</span> <span class="tp" data-path="' + ep + '">' + esc(ev.path) + '</span></div>';
      }
      if (ev.description) b += '<div class="tc-arg"><span class="tc-arg-name">description:</span> ' + esc(ev.description) + '</div>';
      if (ev.command) b += '<pre><code class="language-bash">' + esc(ev.command) + '</code></pre>';
      if (ev.content) {
        var lc = ev.lang ? 'language-' + esc(ev.lang) : '';
        b += '<pre><code class="' + lc + '">' + esc(ev.content) + '</code></pre>';
      }
      if (ev.old_string !== undefined && ev.new_string !== undefined) {
        b += renderDiff(ev.old_string, ev.new_string);
      } else {
        if (ev.old_string !== undefined) b += '<div class="diff-old">- ' + esc(ev.old_string) + '</div>';
        if (ev.new_string !== undefined) b += '<div class="diff-new">+ ' + esc(ev.new_string) + '</div>';
      }
      if (ev.extras) { for (var k in ev.extras) b += '<div class="extra">' + esc(k) + ': ' + esc(ev.extras[k]) + '</div>'; }
      var tcBody = mkEl('div', 'tc-b');
      tcBody.innerHTML = b || '<em style="color:var(--dim)">No arguments</em>';
      c.appendChild(hdr);
      c.appendChild(tcBody);
      addCollapse(c, hdr);
      target.appendChild(c);
      tState.lastToolCallEl = c;
      if (ev.command) {
        var bp = mkEl('div', 'bash-panel');
        var bpContent = mkEl('div', 'bash-panel-content');
        bp.appendChild(bpContent);
        c.appendChild(bp);
        tState.bashPanel = bpContent;
      }
      hlBlock(c);
      break;
    }
    case 'tool_result': {
      if (tState.bashPanel && tState.bashBuf) { tState.bashPanel.textContent += tState.bashBuf; tState.bashBuf = ''; }
      var hadBash = !!tState.bashPanel;
      tState.bashPanel = null; tState.bashRaf = 0;
      if (hadBash && !ev.is_error) break;
      var resultTarget = tState.lastToolCallEl || target;
      if (ev.is_error) {
        var r = mkEl('div', 'ev tr err');
        r.innerHTML = '<div class="rl fail">FAILED</div><div class="tr-content">' + esc(ev.content) + '</div>';
        addCollapse(r, r.querySelector('.rl'));
        resultTarget.appendChild(r);
      } else {
        var op = mkEl('div', 'bash-panel');
        var opContent = mkEl('div', 'bash-panel-content');
        opContent.textContent = ev.content;
        op.appendChild(opContent);
        resultTarget.appendChild(op);
      }
      break;
    }
    case 'system_output': {
      if (tState.bashPanel) {
        if (!tState.bashBuf) tState.bashBuf = '';
        tState.bashBuf += (ev.text || '');
        if (!tState.bashRaf) {
          tState.bashRaf = requestAnimationFrame(function() {
            if (tState.bashPanel) tState.bashPanel.textContent += tState.bashBuf;
            tState.bashBuf = '';
            tState.bashRaf = 0;
            if (tState.bashPanel) tState.bashPanel.scrollTop = tState.bashPanel.scrollHeight;
          });
        }
      } else {
        var s = mkEl('div', 'ev sys');
        s.textContent = (ev.text || '').replace(/\n\n+/g, '\n');
        target.appendChild(s);
      }
      break;
    }
    case 'result': {
      var rc = mkEl('div', 'ev rc');
      var rb = '';
      if (ev.success === false) {
        rb += '<div style="color:var(--red);font-weight:700;font-size:var(--fs-xl);margin-bottom:10px">Status: FAILED</div>';
      }
      var usePre = true;
      if (ev.summary) {
        var sum = (ev.summary || '').replace(/\n{3,}/g, '\n\n').trim();
        if (typeof marked !== 'undefined') { rb += marked.parse(sum); usePre = false; }
        else { rb += esc(sum); }
      } else {
        rb += esc((ev.text || '(no result)').replace(/\n{3,}/g, '\n\n').trim());
      }
      rc.innerHTML = '<div class="rc-h"><h3>Result</h3><div class="rs">'
        + '<span>Tokens <b>' + fmtN(ev.total_tokens || 0) + '</b></span>'
        + '<span>Cost <b>' + (ev.cost || 'N/A') + '</b></span>'
        + '</div></div><div class="rc-body md-body' + (usePre ? ' pre' : '') + '">' + rb + '</div>';
      hlBlock(rc);
      target.appendChild(rc);
      if (statusTokens && ev.total_tokens) statusTokens.textContent = 'Tokens: ' + fmtN(ev.total_tokens);
      if (statusBudget && ev.cost && ev.cost !== 'N/A') statusBudget.textContent = 'Cost: ' + ev.cost;
      if (ev.step_count) updateStepCount(ev.step_count);
      break;
    }
    case 'system_prompt':
    case 'prompt': {
      var cls = t === 'system_prompt' ? 'system-prompt' : 'prompt';
      var label = t === 'system_prompt' ? 'System Prompt' : 'Prompt';
      var el = mkEl('div', 'ev ' + cls);
      var body = typeof marked !== 'undefined' ? marked.parse(ev.text || '') : esc(ev.text || '');
      el.innerHTML = '<div class="' + cls + '-h">' + label + '</div>'
        + '<div class="' + cls + '-body md-body">' + body + '</div>';
      addCollapse(el, el.querySelector('.' + cls + '-h'));
      hlBlock(el);
      target.appendChild(el);
      var bodyEl = el.querySelector('.' + cls + '-body');
      if (bodyEl) bodyEl.scrollTop = bodyEl.scrollHeight;
      break;
    }
    case 'usage_info': {
      var u = mkEl('div', 'ev usage');
      var uHdr = mkEl('div', 'usage-hdr');
      uHdr.textContent = 'Usage';
      var uBody = mkEl('div', 'usage-content');
      uBody.textContent = ev.text || '';
      u.appendChild(uHdr);
      u.appendChild(uBody);
      addCollapse(u, uHdr);
      target.appendChild(u);
      if (ev.total_tokens != null && ev.cost != null) {
        if (statusTokens) statusTokens.textContent = 'Tokens: ' + fmtN(ev.total_tokens);
        if (statusBudget && ev.cost !== 'N/A') statusBudget.textContent = 'Cost: ' + ev.cost;
        if (statusSteps && ev.total_steps != null) statusSteps.textContent = 'Steps: ' + ev.total_steps;
      } else {
        updateUsageMetrics(ev.text || '');
      }
      break;
    }
    }
  }

  function updateStepCount(count) {
    stepCount = count;
    if (statusSteps) statusSteps.textContent = 'Steps: ' + count;
  }

  function processOutputEvent(ev) {
    var t = ev.type;
    if (t === 'tool_call') {
      lastToolName = ev.name || '';
      llmPanel = null; llmPanelState = mkS(); pendingPanel = false;
    }
    if (t === 'tool_result' && lastToolName !== 'finish') { pendingPanel = true; }
    // Count initial thinking as step 1
    if (stepCount === 0 && (t === 'thinking_start' || t === 'text_delta')) {
      updateStepCount(1);
    }
    if (pendingPanel && (t === 'thinking_start' || t === 'text_delta')) {
      updateStepCount(stepCount + 1);
      llmPanel = mkEl('div', 'llm-panel');
      var lHdr = mkEl('div', 'llm-panel-hdr');
      lHdr.textContent = 'Thoughts';
      addCollapse(llmPanel, lHdr);
      llmPanel.appendChild(lHdr);
      O.appendChild(llmPanel);
      collapseOlderPanels();
      llmPanelState = mkS(); pendingPanel = false;
    }
    var target = O, tState = state;
    if (llmPanel && (t === 'thinking_start' || t === 'thinking_delta' || t === 'thinking_end'
      || t === 'text_delta' || t === 'text_end')) {
      target = llmPanel; tState = llmPanelState;
    }
    handleOutputEvent(ev, target, tState);
    if (target === O) collapseOlderPanels();
    if (target === llmPanel) llmPanel.scrollTop = llmPanel.scrollHeight;
  }

  // --- Scrolling ---

  function sb() {
    if (!_scrollLock && !_noScroll && !scrollRaf && !(welcome && welcome.style.display !== 'none')) {
      scrollRaf = requestAnimationFrame(function() {
        O.scrollTo({ top: O.scrollHeight, behavior: 'instant' });
        scrollRaf = 0;
      });
    }
  }

  O.addEventListener('wheel', function(e) {
    if (isRunning && e.deltaY < 0) _scrollLock = true;

    // Adjacent task loading via overscroll detection
    if (!isRunning && !adjacentLoading && currentChatId && currentTaskName) {
      var atTop = O.scrollTop <= 0;
      var atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 2;

      if (atTop && e.deltaY < 0 && !noPrevTask && oldestLoadedTask) {
        // Scrolling up at top — load task before the oldest loaded
        if (overscrollDir !== 'prev') { overscrollAccum = 0; overscrollDir = 'prev'; }
        overscrollAccum += Math.abs(e.deltaY);
        clearTimeout(overscrollTimer);
        overscrollTimer = setTimeout(function() { overscrollAccum = 0; overscrollDir = ''; }, 500);
        if (overscrollAccum >= OVERSCROLL_THRESHOLD) {
          overscrollAccum = 0;
          overscrollDir = '';
          adjacentLoading = true;
          showAdjacentLoader('prev');
          vscode.postMessage({ type: 'getAdjacentTask', chatId: currentChatId, task: oldestLoadedTask, direction: 'prev' });
        }
      } else if (atBottom && e.deltaY > 0 && !noNextTask && newestLoadedTask) {
        // Scrolling down at bottom — load task after the newest loaded
        if (overscrollDir !== 'next') { overscrollAccum = 0; overscrollDir = 'next'; }
        overscrollAccum += Math.abs(e.deltaY);
        clearTimeout(overscrollTimer);
        overscrollTimer = setTimeout(function() { overscrollAccum = 0; overscrollDir = ''; }, 500);
        if (overscrollAccum >= OVERSCROLL_THRESHOLD) {
          overscrollAccum = 0;
          overscrollDir = '';
          adjacentLoading = true;
          showAdjacentLoader('next');
          vscode.postMessage({ type: 'getAdjacentTask', chatId: currentChatId, task: newestLoadedTask, direction: 'next' });
        }
      } else {
        overscrollAccum = 0;
        overscrollDir = '';
      }
    }
  });
  function updateVisibleTask() {
    var adjacentTasks = O.querySelectorAll('.adjacent-task[data-task]');
    if (!adjacentTasks.length) return;
    var outputRect = O.getBoundingClientRect();
    var checkY = outputRect.top + outputRect.height * 0.3;
    var visibleTask = currentTaskName;
    for (var i = 0; i < adjacentTasks.length; i++) {
      var rect = adjacentTasks[i].getBoundingClientRect();
      if (rect.top <= checkY && rect.bottom > checkY) {
        visibleTask = adjacentTasks[i].dataset.task;
        break;
      }
    }
    setTaskText(visibleTask);
  }

  O.addEventListener('scroll', function() {
    if (_scrollLock) {
      var atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 150;
      if (atBottom) _scrollLock = false;
    }
    updateVisibleTask();
  });
  new MutationObserver(function() { if (isRunning) sb(); }).observe(O, { childList: true, subtree: true, characterData: true });

  // --- Timer ---
  function startTimer() {
    if (!t0) t0 = Date.now();
    if (timerIv) clearInterval(timerIv);
    statusText.style.color = 'var(--red)';
    timerIv = setInterval(function() {
      var s = Math.floor((Date.now() - t0) / 1000);
      var m = Math.floor(s / 60);
      statusText.textContent = 'Running ' + (m > 0 ? m + 'm ' : '') + s % 60 + 's';
    }, 1000);
  }
  function stopTimer() { if (timerIv) { clearInterval(timerIv); timerIv = null; } statusText.style.color = 'var(--green)'; }



  // --- Usage metrics (tokens / budget) in header ---
  function updateUsageMetrics(text) {
    if (!statusTokens || !statusBudget) return;
    var tm = text.match(/Tokens:\s*([\d,]+)\/[\d,]+/);
    var bm = text.match(/Budget:\s*(\$[0-9.]+)\/\$[0-9.]+/);
    var sm = text.match(/Steps:\s*(\d+)\/\d+/);
    if (tm) statusTokens.textContent = 'Tokens: ' + tm[1];
    if (bm) statusBudget.textContent = 'Cost: ' + bm[1];
    if (sm) updateStepCount(parseInt(sm[1], 10));
  }

  function clearUsageMetrics() {
    if (statusTokens) statusTokens.textContent = '';
    if (statusBudget) statusBudget.textContent = '';
    if (statusSteps) statusSteps.textContent = '';
    stepCount = 0;
  }

  function focusInputWithRetry() {
    inp.focus();
    setTimeout(function() { inp.focus(); }, 100);
    setTimeout(function() { inp.focus(); }, 300);
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
      vscode.postMessage({ type: 'getHistory', query: historySearch.value, generation: historyGeneration });
    }
  }

  // --- Main event handler ---
  function handleEvent(ev) {
    var t = ev.type;
    switch (t) {
    case 'status': {
      var evTab = findTabByEvt(ev);
      if (evTab) {
        evTab.isRunning = !!ev.running;
      }
      // Update UI only when the event targets the active tab (or no tabId)
      if (!evTab || evTab.id === activeTabId) {
        setRunningState(ev.running);
      }
      break;
    }
    case 'models':
      allModels = ev.models || [];
      if (ev.selected) { selectedModel = ev.selected; modelName.textContent = ev.selected; }
      renderModelList('');
      break;
    case 'history':
      renderHistory(ev.sessions || [], ev.offset || 0, ev.generation || 0);
      break;
    case 'files':
      renderAutocomplete(ev.files || []);
      break;
    case 'askUser':
      showAskUserModal(ev.question);
      break;
    case 'error':
      addError(ev.text);
      break;
    case 'clear': {
      var evTabId = ev.tabId;
      if (evTabId === undefined || evTabId === activeTabId) {
        clearOutput();
        resetOutputState();
        showSpinner();
      }
      break;
    }
    case 'clearChat':
      createNewTab();
      break;
    case 'followup_suggestion': {
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
      var fu = mkEl('div', 'followup-bar');
      fu.innerHTML = '<span class="fu-label">Suggested next</span>'
        + '<span class="fu-text">' + esc(ev.text) + '</span>';
      fu.addEventListener('click', function() {
        inp.value = ev.text; syncClearBtn();
        inp.focus();
      });
      O.appendChild(fu);
      sb();
      break;
    }
    case 'tasks_updated':
      refreshHistory();
      vscode.postMessage({ type: 'getWelcomeSuggestions' });
      vscode.postMessage({ type: 'getInputHistory' });
      break;
    case 'welcome_suggestions':
      renderWelcomeSuggestions(ev.suggestions || []);
      break;
    case 'chatId': {
      var newChatId = ev.chat_id || '';
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
        // Event is for a background tab; store chatId in that tab's saved state
        var cidTab = tabs.find(function(t) { return t.id === ev.tabId; });
        if (cidTab) { cidTab.chatId = newChatId; persistTabState(); }
      } else {
        currentChatId = newChatId;
        persistTabState();
      }
      break;
    }
    case 'task_events':
      if (ev.chat_id) currentChatId = ev.chat_id;
      if (ev.task) {
        currentTaskName = ev.task;
        resetAdjacentState();  // sets oldest/newest to currentTaskName
        setTaskText(ev.task);
        if (welcome) welcome.style.display = 'none';
        updateActiveTabTitle(ev.task);
      }
      if (ev.extra) {
        try {
          var extra = JSON.parse(ev.extra);
          if (extra.model) {
            selectedModel = extra.model;
            if (modelName) modelName.textContent = selectedModel;
            var curTab = tabs.find(function(t) { return t.id === activeTabId; });
            if (curTab) curTab.selectedModel = selectedModel;
          }
          if (extra.work_dir) {
            var wdTab = tabs.find(function(t) { return t.id === activeTabId; });
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
        } catch (e) { /* ignore malformed extra */ }
      }
      replayTaskEvents(ev.events || []);
      break;
    case 'adjacent_task_events':
      renderAdjacentTask(ev.direction, ev.task, ev.events || []);
      break;
    case 'setTaskText': {
      var stt = (ev.text || '').trim();
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
        var sttTab = tabs.find(function(t) { return t.id === ev.tabId; });
        if (sttTab) {
          sttTab.title = stt.length > 30 ? stt.substring(0, 30) + '\u2026' : stt;
          sttTab.taskPanelHTML = stt;
          sttTab.taskPanelVisible = true;
          renderTabBar();
          persistTabState();
        }
      }
      break;
    }
    case 'appendToInput':
      if (ev.text) {
        inp.value = inp.value ? inp.value + '\n' + ev.text : ev.text;
        inp.dispatchEvent(new Event('input', { bubbles: true }));
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
    case 'activeFileInfo':
      if (runPromptBtn) {
        if (!isRunning && ev.isPrompt) {
          runPromptBtn.disabled = false;
          runPromptBtn.dataset.tooltip = 'Run prompt: ' + ev.filename;
        } else {
          runPromptBtn.disabled = true;
          runPromptBtn.dataset.tooltip = ev.isPrompt
            ? 'Run current file as prompt'
            : 'Run current file as prompt (no prompt detected)';
        }
      }
      break;
    case 'merge_data': {
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
      var mc = mkEl('div', 'ev merge-info');
      mc.innerHTML = '<div class="merge-info-hdr" style="color:var(--yellow);font-weight:600;font-size:var(--fs-base);margin-bottom:4px">'
        + '\u2731 Reviewing ' + (ev.hunk_count || 0) + ' change(s)</div>'
        + '<div class="merge-info-body" style="font-size:var(--fs-md);color:var(--dim)">Red = old lines, Green = new lines. '
        + 'Use the merge toolbar to accept or reject changes.</div>';
      addCollapse(mc, mc.querySelector('.merge-info-hdr'));
      O.appendChild(mc);
      collapseOlderPanels();
      break;
    }
    case 'merge_started':
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
        switchToTab(ev.tabId);
      }
      isMerging = true;
      showMergeToolbar();
      updateInputDisabled();
      sb();
      break;
    case 'merge_ended':
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) {
        var mrt2 = tabs.find(function(t) { return t.id === ev.tabId; });
        if (mrt2) mrt2.isMerging = false;
        break;
      }
      isMerging = false;
      hideMergeToolbar();
      updateInputDisabled();
      break;
    case 'commitMessage':
      break;
    case 'droppedPaths':
      if (ev.paths && ev.paths.length > 0) {
        var pos = inp.selectionStart || inp.value.length;
        var before = inp.value.substring(0, pos);
        var after = inp.value.substring(pos);
        var insert = ev.paths.map(function(p) { return 'WORK_DIR/' + p; }).join(' ');
        var needSpace = before.length > 0 && !/\s$/.test(before);
        var trailSpace = after.length > 0 && !/^\s/.test(after) ? ' ' : '';
        inp.value = before + (needSpace ? ' ' : '') + insert + trailSpace + after;
        var np = before.length + (needSpace ? 1 : 0) + insert.length + trailSpace.length;
        inp.setSelectionRange(np, np);
        syncClearBtn();
        inp.focus();
      }
      break;
    case 'worktree_done':
      showWorktreeActions(ev);
      break;
    case 'worktree_result':
      handleWorktreeResult(ev);
      break;
    case 'task_done': {
      var doneT0 = t0;
      if (!doneT0 && ev.tabId !== undefined) {
        var rt = tabs.find(function(t) { return t.id === ev.tabId; });
        if (rt) doneT0 = rt.t0;
      }
      var el = doneT0 ? Math.floor((Date.now() - doneT0) / 1000) : 0;
      var em = Math.floor(el / 60);
      setReady('Done (' + (em > 0 ? em + 'm ' : '') + el % 60 + 's)', ev.tabId);
      break;
    }
    case 'task_error':
    case 'task_stopped': {
      var isErr = t === 'task_error';
      if (ev.tabId === undefined || ev.tabId === activeTabId) {
        var banner = mkEl('div', 'ev tr err');
        banner.innerHTML = '<div class="rl fail">' + (isErr ? 'ERROR' : 'STOPPED') + '</div>'
          + '<div class="tr-content">' + esc(isErr ? (ev.text || 'Unknown error') : 'Agent execution stopped by user') + '</div>';
        addCollapse(banner, banner.querySelector('.rl'));
        O.appendChild(banner);
        collapseOlderPanels();
      }
      setReady(isErr ? 'Error' : 'Stopped', ev.tabId);
      break;
    }
    default:
      if (ev.tabId !== undefined && ev.tabId !== activeTabId) break;
      processOutputEvent(ev);
      if (isActiveTabRunning()) showSpinner();
      sb();
      break;
    }
  }

  function updateInputDisabled() {
    var blocked = isRunning || isMerging;
    inp.disabled = blocked;
    if (blocked) { clearGhost(); hideAC(); }
  }

  function setRunningState(running) {
    isRunning = running;
    sendBtn.style.display = running ? 'none' : 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';
    sendBtn.disabled = running || isMerging;

    if (uploadBtn) uploadBtn.disabled = running;
    if (worktreeToggleBtn) worktreeToggleBtn.disabled = running;
    if (parallelToggleBtn) parallelToggleBtn.disabled = running;
    if (runPromptBtn && running) runPromptBtn.disabled = true;
    if (modelBtn) { modelBtn.disabled = running; if (running) closeModelDD(); }
    updateInputDisabled();
    if (running) {
      startTimer();
    }
  }

  function setReady(label, tabId) {
    // Mark the tab as no longer running
    if (tabId !== undefined) {
      var doneTab = tabs.find(function(t) { return t.id === tabId; });
      if (doneTab) { doneTab.isRunning = false; doneTab.t0 = null; }
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
  }

  function addError(text) {
    var div = mkEl('div', 'ev tr err');
    div.innerHTML = '<strong>Error:</strong> ' + esc(text);
    O.appendChild(div);
    sb();
  }

  // --- Welcome suggestions (dynamic) ---
  function renderWelcomeSuggestions(suggestions) {
    var container = document.getElementById('suggestions');
    if (!container) return;
    container.innerHTML = '';
    if (!suggestions || suggestions.length === 0) return;
    suggestions.forEach(function(s) {
      var chip = document.createElement('div');
      chip.className = 'suggestion-chip';
      chip.dataset.prompt = s.text;
      chip.innerHTML = '<span class="chip-label">Suggested</span>' + esc(s.text);
      chip.addEventListener('click', function() {
        inp.value = s.text; syncClearBtn();
        inp.focus();
      });
      container.appendChild(chip);
    });
  }

  // --- Task replay ---
  function replayEventsInto(container, events, opts) {
    var rState = mkS();
    var rLlmPanel = null;
    var rLlmPanelState = mkS();
    var rLastToolName = '';
    var rPendingPanel = false;
    events.forEach(function(ev) {
      var t = ev.type;
      if (t === 'task_done' || t === 'task_error' || t === 'task_stopped') {
        if (t === 'task_error') {
          var banner = mkEl('div', 'ev tr err');
          banner.innerHTML = '<div class="rl fail">ERROR</div><div class="tr-content">' + esc(ev.text || 'Unknown error') + '</div>';
          addCollapse(banner, banner.querySelector('.rl'));
          container.appendChild(banner);
        } else if (t === 'task_stopped') {
          var banner = mkEl('div', 'ev tr err');
          banner.innerHTML = '<div class="rl fail">STOPPED</div><div class="tr-content">Agent execution stopped by user</div>';
          addCollapse(banner, banner.querySelector('.rl'));
          container.appendChild(banner);
        }
        return;
      }
      if (t === 'followup_suggestion') {
        var fu = mkEl('div', 'followup-bar');
        fu.innerHTML = '<span class="fu-label">Suggested next</span>'
          + '<span class="fu-text">' + esc(ev.text) + '</span>';
        if (opts && opts.onFollowupClick) {
          fu.addEventListener('click', function() { opts.onFollowupClick(ev.text); });
        }
        container.appendChild(fu);
        return;
      }
      if (t === 'tool_call') {
        rLastToolName = ev.name || '';
        rLlmPanel = null; rLlmPanelState = mkS(); rPendingPanel = false;
      }
      if (t === 'tool_result' && rLastToolName !== 'finish') { rPendingPanel = true; }
      if (rPendingPanel && (t === 'thinking_start' || t === 'text_delta')) {
        rLlmPanel = mkEl('div', 'llm-panel');
        var lHdr = mkEl('div', 'llm-panel-hdr');
        lHdr.textContent = 'Thoughts';
        addCollapse(rLlmPanel, lHdr);
        rLlmPanel.appendChild(lHdr);
        container.appendChild(rLlmPanel);
        rLlmPanelState = mkS(); rPendingPanel = false;
      }
      var target = container, tState = rState;
      if (rLlmPanel && (t === 'thinking_start' || t === 'thinking_delta' || t === 'thinking_end'
        || t === 'text_delta' || t === 'text_end')) {
        target = rLlmPanel; tState = rLlmPanelState;
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
      onFollowupClick: function(text) { inp.value = text; syncClearBtn(); inp.focus(); }
    });
    // Count steps from replayed events: step 1 = first thinking, each llm-panel = +1
    var rSteps = 0, rPending = false, rLastTool = '';
    events.forEach(function(ev) {
      var t = ev.type;
      if (t === 'tool_call') { rLastTool = ev.name || ''; rPending = false; }
      if (t === 'tool_result' && rLastTool !== 'finish') rPending = true;
      if (rSteps === 0 && (t === 'thinking_start' || t === 'text_delta')) rSteps = 1;
      if (rPending && (t === 'thinking_start' || t === 'text_delta')) { rSteps++; rPending = false; }
      if (t === 'result' && ev.step_count) rSteps = ev.step_count;
    });
    if (rSteps > 0) updateStepCount(rSteps);
    sb();
  }

  // --- Worktree merge/discard UI ---

  var worktreeBar = null;

  function clearWorktreeBar() {
    if (worktreeBar && worktreeBar.parentNode) {
      worktreeBar.parentNode.removeChild(worktreeBar);
    }
    worktreeBar = null;
    if (inputContainer) inputContainer.style.display = '';
  }

  function showWorktreeActions(ev) {
    clearWorktreeBar();
    var bar = mkEl('div', 'wt-bar');
    var label = mkEl('span', 'wt-label');
    label.textContent = 'Auto-commit and merge, Discard, or Do Nothing?';
    bar.appendChild(label);

    var btns = mkEl('div', 'wt-btns');
    var mergeBtn = mkEl('button', 'wt-btn wt-merge');
    mergeBtn.textContent = 'Auto-commit and merge';
    mergeBtn.addEventListener('click', function() {
      disableWtBtns();
      vscode.postMessage({ type: 'worktreeAction', action: 'merge' });
    });

    var discardBtn = mkEl('button', 'wt-btn wt-discard');
    discardBtn.textContent = 'Discard';
    discardBtn.addEventListener('click', function() {
      disableWtBtns();
      vscode.postMessage({ type: 'worktreeAction', action: 'discard' });
    });

    var doNothingBtn = mkEl('button', 'wt-btn wt-donothing');
    doNothingBtn.textContent = 'Do Nothing';
    doNothingBtn.addEventListener('click', function() {
      disableWtBtns();
      vscode.postMessage({ type: 'worktreeAction', action: 'do_nothing' });
    });

    btns.appendChild(mergeBtn);
    btns.appendChild(discardBtn);
    btns.appendChild(doNothingBtn);
    bar.appendChild(btns);

    // Hide the input container and show the worktree bar in its place
    if (inputContainer) inputContainer.style.display = 'none';
    var area = document.getElementById('input-area');
    area.insertBefore(bar, area.firstChild);
    worktreeBar = bar;
  }

  function disableWtBtns() {
    if (!worktreeBar) return;
    var btns = worktreeBar.querySelectorAll('.wt-btn');
    btns.forEach(function(b) { b.disabled = true; });
  }

  function handleWorktreeResult(ev) {
    clearWorktreeBar();
    var cls = ev.success ? 'wt-result-ok' : 'wt-result-err';
    var div = mkEl('div', 'ev ' + cls);
    var msg = ev.message || '';
    div.textContent = msg;
    O.appendChild(div);
    sb();
  }

  // --- Merge toolbar (shown in input area, replacing textarea) ---
  function showMergeToolbar() {
    if (document.getElementById('merge-toolbar')) return;
    inputContainer.style.display = 'none';
    var bar = mkEl('div', 'merge-toolbar-card');
    bar.id = 'merge-toolbar';
    bar.innerHTML =
      '<div class="merge-toolbar-header">'
      + '<span class="merge-toolbar-title">Review Changes</span>'
      + '<span class="merge-toolbar-hint">Red = old \u00b7 Green = new</span>'
      + '</div>'
      + '<div class="merge-toolbar-actions">'
      + '<div class="merge-toolbar-row">'
      + '<button class="merge-btn merge-nav" id="merge-prev-btn">Prev</button>'
      + '<button class="merge-btn merge-nav" id="merge-next-btn">Next</button>'
      + '<button class="merge-btn merge-accept" id="merge-accept-btn">Accept</button>'
      + '<button class="merge-btn merge-reject" id="merge-reject-btn">Reject</button>'
      + '</div>'
      + '<div class="merge-toolbar-row">'
      + '<button class="merge-btn merge-accept" id="merge-accept-file-btn">Accept File</button>'
      + '<button class="merge-btn merge-reject" id="merge-reject-file-btn">Reject File</button>'
      + '<button class="merge-btn merge-accept" id="merge-accept-all-btn">Accept Rest</button>'
      + '<button class="merge-btn merge-reject" id="merge-reject-all-btn">Reject Rest</button>'
      + '</div>'
      + '</div>';
    document.getElementById('input-area').appendChild(bar);
    var mergeActions = {
      'merge-accept-btn': 'accept', 'merge-reject-btn': 'reject',
      'merge-prev-btn': 'prev', 'merge-next-btn': 'next',
      'merge-accept-file-btn': 'accept-file', 'merge-reject-file-btn': 'reject-file',
      'merge-accept-all-btn': 'accept-all', 'merge-reject-all-btn': 'reject-all',
    };
    Object.keys(mergeActions).forEach(function(id) {
      document.getElementById(id).addEventListener('click', function() {
        vscode.postMessage({ type: 'mergeAction', action: mergeActions[id] });
      });
    });
    sb();
  }

  function hideMergeToolbar() {
    var bar = document.getElementById('merge-toolbar');
    if (bar) bar.remove();
    inputContainer.style.display = '';
  }

  // --- Init and event listeners ---

  function init() {
    setupEventListeners();
    renderTabBar();
    vscode.postMessage({ type: 'ready', activeChatId: _restoredActiveChatId });
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
    document.addEventListener('keydown', function(e) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'd' && !e.shiftKey && !e.altKey) {
        e.preventDefault();
        vscode.postMessage({ type: 'focusEditor' });
      }
      if (e.key === 'Escape' && sidebar.classList.contains('open')) {
        e.preventDefault();
        closeSidebar();
      }
    });
    inp.addEventListener('keydown', function(e) {
      // Autocomplete navigation
      if (autocomplete.style.display === 'block') {
        var items = autocomplete.querySelectorAll('.ac-item');
        if (e.key === 'ArrowDown') { e.preventDefault(); acIdx = Math.min(acIdx + 1, items.length - 1); updateSel(items, acIdx); return; }
        if (e.key === 'ArrowUp') { e.preventDefault(); acIdx = Math.max(acIdx - 1, -1); updateSel(items, acIdx); return; }
        if (e.key === 'Tab') { e.preventDefault(); var ti = acIdx >= 0 ? acIdx : 0; if (items[ti]) items[ti].click(); return; }
        if (e.key === 'Enter' && acIdx >= 0) { e.preventDefault(); items[acIdx].click(); return; }
        if (e.key === 'Escape') { hideAC(); return; }
      }
      // Ghost text accept
      if (e.key === 'Tab' && currentGhost) {
        e.preventDefault();
        inp.value += currentGhost;
        if (/\S$/.test(inp.value)) inp.value += ' ';
        clearGhost();
        syncClearBtn();
        inp.style.height = 'auto';
        inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
        return;
      }
      // History cycling (ArrowUp/Down only when textbox is empty and no autocomplete)
      if (e.key === 'ArrowUp' && autocomplete.style.display !== 'block') {
        if (histCache.length > 0 && (histIdx >= 0 || !inp.value)) {
          e.preventDefault();
          histIdx = Math.min(histIdx + 1, histCache.length - 1);
          inp.value = histCache[histIdx]; syncClearBtn();
          return;
        }
      }
      if (e.key === 'ArrowDown' && histIdx >= 0) {
        e.preventDefault();
        histIdx--;
        inp.value = histIdx >= 0 ? histCache[histIdx] : ''; syncClearBtn();
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
      // Any other key clears ghost
      if (e.key !== 'Tab') clearGhost();
    });
    inp.addEventListener('input', function() {
      inp.style.height = 'auto';
      inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
      checkAutocomplete();
      requestGhost();
      histIdx = -1;
      syncClearBtn();
    });
    inp.addEventListener('blur', function() { clearGhost(); hideAC(); });
    autocomplete.addEventListener('mousedown', function(e) { e.preventDefault(); });
    stopBtn.addEventListener('click', function() {
      vscode.postMessage({ type: 'stop' });
    });
    uploadBtn.addEventListener('click', function() {
      var input = document.createElement('input');
      input.type = 'file';
      input.multiple = true;
      input.accept = 'image/*,application/pdf';
      input.onchange = handleFileSelect;
      input.click();
    });
    if (worktreeToggleBtn) {
      worktreeToggleBtn.addEventListener('click', function() {
        worktreeToggleBtn.classList.toggle('active');
      });
    }
    if (parallelToggleBtn) {
      parallelToggleBtn.addEventListener('click', function() {
        parallelToggleBtn.classList.toggle('active');
      });
    }

    if (inputClearBtn) {
      inputClearBtn.addEventListener('click', function() {
        inp.value = '';
        inp.style.height = 'auto';
        inputClearBtn.style.display = 'none';
        clearGhost();
        hideAC();
        inp.focus();
      });
    }
    modelBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      if (isRunning) return;
      if (modelDropdown.classList.contains('open')) { closeModelDD(); return; }
      modelDropdown.classList.add('open');
      modelSearch.value = '';
      renderModelList('');
      modelSearch.focus();
    });
    modelSearch.addEventListener('input', function() { renderModelList(this.value); });
    modelSearch.addEventListener('keydown', function(e) {
      var items = modelList.querySelectorAll('.model-item');
      if (e.key === 'ArrowDown') { e.preventDefault(); modelDDIdx = Math.min(modelDDIdx + 1, items.length - 1); updateSel(items, modelDDIdx); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); modelDDIdx = Math.max(modelDDIdx - 1, -1); updateSel(items, modelDDIdx); return; }
      if (e.key === 'Enter') { e.preventDefault(); var ti = modelDDIdx >= 0 ? modelDDIdx : 0; if (items[ti]) items[ti].click(); return; }
      if (e.key === 'Escape') { e.preventDefault(); closeModelDD(); return; }
    });
    document.addEventListener('click', function(e) {
      if (!document.getElementById('model-picker').contains(e.target)) closeModelDD();
      if (!autocomplete.contains(e.target) && e.target !== inp) {
        hideAC();
      }
    });
    if (runPromptBtn) {
      runPromptBtn.addEventListener('click', function() {
        if (runPromptBtn.disabled || isRunning) return;
        vscode.postMessage({ type: 'runPrompt' });
      });
    }
    historyBtn.addEventListener('click', function() {
      if (sidebar.classList.contains('open')) {
        closeSidebar();
      } else {
        resetHistoryPagination();
        sidebar.classList.add('open');
        sidebarOverlay.classList.add('open');
        historyBtn.classList.add('open');
        vscode.postMessage({ type: 'getHistory', query: historySearch.value, generation: historyGeneration });
      }
    });
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    historySearch.addEventListener('input', function() {
      resetHistoryPagination();
      vscode.postMessage({ type: 'getHistory', query: historySearch.value, generation: historyGeneration });
    });
    historyList.addEventListener('scroll', function() {
      if (historyLoading || !historyHasMore) return;
      if (historyList.scrollTop + historyList.clientHeight >= historyList.scrollHeight - 50) {
        historyLoading = true;
        var loader = document.createElement('div');
        loader.className = 'sidebar-loading';
        loader.id = 'history-loader';
        loader.textContent = 'Loading...';
        historyList.appendChild(loader);
        vscode.postMessage({ type: 'getHistory', query: historySearch.value, offset: historyOffset, generation: historyGeneration });
      }
    });
    // Click handler for file paths in tool call headers — parse :line suffix
    document.addEventListener('click', function(e) {
      var el = e.target.closest('[data-path]');
      if (el && el.dataset.path) {
        var raw = el.dataset.path;
        var match = raw.match(/^(.+):(\d+)$/);
        if (match) {
          vscode.postMessage({ type: 'openFile', path: match[1], line: parseInt(match[2], 10) });
        } else {
          vscode.postMessage({ type: 'openFile', path: raw });
        }
      }
    });
    askUserSubmit.addEventListener('click', function() {
      var answer = askUserInput.value;
      vscode.postMessage({ type: 'userAnswer', answer: answer });
      askUserModal.style.display = 'none';
      askUserInput.value = '';
    });
    askUserInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        askUserSubmit.click();
      }
    });



    // Paste images/PDFs
    inp.addEventListener('paste', function(e) {
      var items = (e.clipboardData || {}).items;
      if (!items) return;
      for (var i = 0; i < items.length; i++) {
        var item = items[i];
        if (item.kind === 'file' && (item.type.startsWith('image/') || item.type === 'application/pdf')) {
          e.preventDefault();
          var file = item.getAsFile();
          if (file) readFileAsAttachment(file);
        }
      }
    });

    // Drag and drop
    if (inputContainer) {
      inputContainer.addEventListener('dragover', function(e) {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.add('drag-over');
      });
      inputContainer.addEventListener('dragleave', function(e) {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.remove('drag-over');
      });
      inputContainer.addEventListener('drop', function(e) {
        e.preventDefault();
        e.stopPropagation();
        inputContainer.classList.remove('drag-over');
        // Handle file URIs from VS Code explorer (text/uri-list)
        var uriList = e.dataTransfer && e.dataTransfer.getData('text/uri-list');
        if (uriList) {
          var uris = uriList.split(/\r?\n/).filter(function(u) { return u && !u.startsWith('#'); });
          if (uris.length > 0) {
            vscode.postMessage({ type: 'resolveDroppedPaths', uris: uris });
            return;
          }
        }
        // Handle image/PDF file drops
        var files = e.dataTransfer && e.dataTransfer.files;
        if (!files) return;
        Array.from(files).forEach(function(file) {
          if (file.type.startsWith('image/') || file.type === 'application/pdf') {
            readFileAsAttachment(file);
          }
        });
      });
    }

    window.addEventListener('message', function(event) {
      handleEvent(event.data);
    });
  }

  function readFileAsAttachment(file) {
    var reader = new FileReader();
    reader.onload = function(event) {
      attachments.push({
        name: file.name,
        type: file.type,
        data: event.target.result.split(',')[1]
      });
      renderFileChips();
    };
    reader.readAsDataURL(file);
  }

  function sendMessage() {
    var prompt = inp.value.trim();
    if (!prompt || isRunning) return;

    if (histCache[0] !== prompt) {
      histCache.unshift(prompt);
    }
    var curTab = tabs.find(function(t) { return t.id === activeTabId; });
    var msg = {
      type: 'submit',
      prompt: prompt,
      model: selectedModel,
      tabId: activeTabId,
      attachments: attachments.map(function(a) {
        return { name: a.name, mimeType: a.type, data: a.data };
      }),
      useWorktree: !!(worktreeToggleBtn && worktreeToggleBtn.classList.contains('active')),
      useParallel: !!(parallelToggleBtn && parallelToggleBtn.classList.contains('active'))
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

  function showAskUserModal(question) {
    askUserQuestion.textContent = question;
    askUserModal.style.display = 'flex';
    askUserInput.focus();
  }

  function handleFileSelect(e) {
    var files = e.target.files;
    if (!files || files.length === 0) return;
    Array.from(files).forEach(function(file) { readFileAsAttachment(file); });
  }

  function renderFileChips() {
    fileChips.innerHTML = '';
    attachments.forEach(function(att, idx) {
      var chip = document.createElement('div');
      chip.className = 'file-chip';
      var isImage = att.type.startsWith('image/');
      chip.innerHTML =
        (isImage ? '<img src="data:' + att.type + ';base64,' + att.data + '">' : '<span class="fc-icon">\uD83D\uDCC4</span>')
        + '<span>' + esc(att.name) + '</span>'
        + '<span class="fc-rm" data-idx="' + idx + '">&times;</span>';
      chip.querySelector('.fc-rm').addEventListener('click', function() {
        attachments.splice(idx, 1);
        renderFileChips();
      });
      fileChips.appendChild(chip);
    });
  }

  function renderModelItem(m) {
    var d = mkEl('div', 'model-item' + (m.name === selectedModel ? ' active' : ''));
    var price = '$' + m.inp.toFixed(2) + ' / $' + m.out.toFixed(2);
    d.innerHTML = '<span>' + esc(m.name) + '</span><span class="model-cost">' + price + '</span>';
    d.addEventListener('click', function() { selectModel(m.name); });
    return d;
  }

  function renderModelList(q) {
    modelList.innerHTML = ''; modelDDIdx = -1;
    var ql = q.toLowerCase();
    var used = [], rest = [];
    allModels.forEach(function(m) {
      if (ql && m.name.toLowerCase().indexOf(ql) < 0) return;
      if (m.uses > 0) used.push(m); else rest.push(m);
    });
    used.sort(function(a, b) { return b.uses - a.uses; });
    if (used.length) {
      var hdr = mkEl('div', 'model-group-hdr');
      hdr.textContent = 'Recently Used';
      modelList.appendChild(hdr);
      used.forEach(function(m) { modelList.appendChild(renderModelItem(m)); });
    }
    var lastVendor = '';
    rest.forEach(function(m) {
      var v = m.vendor;
      if (v !== lastVendor) {
        var hdr = mkEl('div', 'model-group-hdr');
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
    vscode.postMessage({ type: 'selectModel', model: name });
  }

  function closeModelDD() {
    modelDropdown.classList.remove('open');
    modelSearch.value = '';
    modelDDIdx = -1;
  }

  function updateSel(items, idx) {
    items.forEach(function(it, i) { it.classList.toggle('sel', i === idx); });
    if (idx >= 0) items[idx].scrollIntoView({ block: 'nearest' });
  }

  function chatIdBgColor(chatId) {
    if (!chatId) return 'hsl(0, 0%, 92%)';
    var hash = 5381;
    for (var i = 0; i < chatId.length; i++) {
      hash = ((hash << 5) + hash) + chatId.charCodeAt(i);
      hash |= 0;
    }
    var hue = Math.abs(hash) % 360;
    return 'hsl(' + hue + ', 40%, 92%)';
  }

  function renderHistory(sessions, offset, generation) {
    if (generation !== historyGeneration) return;

    historyLoading = false;
    var loader = document.getElementById('history-loader');
    if (loader) loader.remove();

    if (offset === 0) {
      if (sessions.length === 0) {
        historyList.innerHTML = '<div class="sidebar-empty">No conversations yet</div>';
        historyHasMore = false;
        return;
      }
      historyList.innerHTML = '';
    }

    sessions.forEach(function(s) {
      var div = document.createElement('div');
      div.className = 'sidebar-item';
      var itemText = s.title || s.preview || 'Untitled';
      div.textContent = itemText;
      div.dataset.tooltip = s.text || itemText;
      div.style.backgroundColor = chatIdBgColor(s.chat_id || String(s.id));
      div.style.color = '#000';
      div.addEventListener('click', function() {
        if (s.has_events && (s.chat_id || s.id)) {
          setTaskText(s.preview || s.title || '');
          vscode.postMessage({ type: 'resumeSession', id: s.chat_id || s.id });
        } else {
          inp.value = s.preview || s.title || ''; syncClearBtn();
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

  function getAtCtx() {
    var val = inp.value, pos = inp.selectionStart || 0;
    var before = val.substring(0, pos);
    var m = before.match(/@([^\s]*)$/);
    return m ? { start: before.length - m[0].length, query: m[1] } : null;
  }

  function checkAutocomplete() {
    var atCtx = getAtCtx();
    if (atCtx) {
      vscode.postMessage({ type: 'getFiles', prefix: atCtx.query });
    } else {
      hideAC();
    }
  }

  var _acSvg = {
    file: '<svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>',
    star: '<svg viewBox="0 0 24 24"><path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/></svg>'
  };
  function _acIcon(type) {
    if (type === 'frequent') return _acSvg.star;
    return _acSvg.file;
  }
  function hlMatch(text, query) {
    if (!query) return esc(text);
    var idx = text.toLowerCase().indexOf(query.toLowerCase());
    if (idx < 0) return esc(text);
    return esc(text.substring(0, idx))
      + '<strong class="ac-hl">' + esc(text.substring(idx, idx + query.length)) + '</strong>'
      + esc(text.substring(idx + query.length));
  }
  function _acPathHtml(text) {
    var last = text.lastIndexOf('/');
    if (last < 0 || last === text.length - 1) return esc(text);
    var dir = text.substring(0, last + 1);
    var fname = text.substring(last + 1);
    return '<span class="ac-dir">' + esc(dir) + '</span>'
      + '<span class="ac-fname">' + esc(fname) + '</span>';
  }
  function hideAC() { autocomplete.style.display = 'none'; acIdx = -1; }

  function renderAutocomplete(data) {
    if (!data || !data.length) { hideAC(); return; }
    autocomplete.innerHTML = ''; acIdx = -1;
    var atMatch = getAtCtx();
    var searchQ = atMatch ? atMatch.query : '';
    var order = ['frequent', 'file'];
    var labels = { frequent: 'Frequent', file: 'Files' };
    var groups = {};
    data.forEach(function(item) {
      var t = item.type;
      if (!groups[t]) groups[t] = [];
      groups[t].push(item);
    });
    var isFirst = true;
    order.forEach(function(type) {
      var g = groups[type]; if (!g) return;
      var lbl = labels[type] || type;
      var hdr = mkEl('div', 'ac-section');
      hdr.textContent = lbl; autocomplete.appendChild(hdr);
      g.forEach(function(item) {
        var d = mkEl('div', 'ac-item');
        d.dataset.text = item.text;
        var useSearch = searchQ && searchQ.length > 0;
        var textHtml = useSearch
          ? hlMatch(item.text, searchQ)
          : _acPathHtml(item.text);
        d.innerHTML = '<span class="ac-icon">' + _acIcon(item.type) + '</span>'
          + '<span class="ac-text">' + textHtml + '</span>';
        if (isFirst) {
          d.innerHTML += '<span class="ac-hint">tab</span>';
          isFirst = false;
        }
        d.addEventListener('click', function() {
          insertAtMention(item.text);
        });
        autocomplete.appendChild(d);
      });
    });
    var footer = mkEl('div', 'ac-footer');
    footer.innerHTML = '<span><kbd>\u2191\u2193</kbd> navigate</span>'
      + '<span><kbd>Tab</kbd> accept</span>'
      + '<span><kbd>Esc</kbd> dismiss</span>';
    autocomplete.appendChild(footer);
    autocomplete.style.display = 'block';
    acIdx = 0;
    var allItems = autocomplete.querySelectorAll('.ac-item');
    updateSel(allItems, acIdx);
  }

  function insertAtMention(file) {
    var atCtx = getAtCtx();
    if (atCtx) {
      var before = inp.value.substring(0, atCtx.start);
      var after = inp.value.substring(inp.selectionStart || inp.value.length);
      var sep = /^\s/.test(after) ? '' : ' ';
      var mention = 'WORK_DIR/' + file;
      inp.value = before + mention + sep + after; syncClearBtn();
      var np = before.length + mention.length + sep.length;
      inp.setSelectionRange(np, np);
      vscode.postMessage({ type: 'recordFileUsage', path: file });
    }
    hideAC();
    inp.focus();
  }

  // Start
  init();
})();

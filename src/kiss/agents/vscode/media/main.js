/**
 * KISS Sorcar Webview JavaScript
 * Uses the same event protocol and rendering as the browser-based Sorcar.
 */

(function() {
  // @ts-ignore - vscode is injected by the webview
  const vscode = acquireVsCodeApi();

  // State
  let isRunning = false;
  let selectedModel = 'claude-opus-4-6';
  let allModels = [];
  let modelDDIdx = -1;
  let attachments = [];
  let _scrollLock = false;
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


  // Elements
  const O = document.getElementById('output');
  const welcome = document.getElementById('welcome');
  const inp = document.getElementById('task-input');
  const sendBtn = document.getElementById('send-btn');
  const stopBtn = document.getElementById('stop-btn');
  const uploadBtn = document.getElementById('upload-btn');
  const clearBtn = document.getElementById('clear-btn');
  const modelBtn = document.getElementById('model-btn');
  const modelDropdown = document.getElementById('model-dropdown');
  const modelSearch = document.getElementById('model-search');
  const modelList = document.getElementById('model-list');
  const modelName = document.getElementById('model-name');
  const fileChips = document.getElementById('file-chips');
  const statusDot = document.getElementById('status-dot');
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
  const taskPanel = document.getElementById('task-panel');


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

  // Streaming state (mirrors browser handleOutputEvent)
  let state = mkS();
  let lastToolName = '';
  let llmPanel = null;
  let llmPanelState = mkS();
  let pendingPanel = false;

  let t0 = null;
  let timerIv = null;
  let _spinnerTimer = null;

  function mkS() { return { thinkEl: null, txtEl: null, bashPanel: null, bashBuf: '', bashRaf: 0 }; }

  function resetOutputState() {
    state = mkS();
    llmPanel = null;
    llmPanelState = mkS();
    lastToolName = '';
    pendingPanel = false;
    _scrollLock = false;
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

  function toggleTC(el) {
    el.nextElementSibling.classList.toggle('hide');
    el.querySelector('.chv').classList.toggle('open');
  }

  function toggleThink(el) {
    var p = el.parentElement;
    p.querySelector('.cnt').classList.toggle('hidden');
    el.querySelector('.arrow').classList.toggle('collapsed');
  }
  window.toggleTC = toggleTC;
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
      var h = '<span class="chv open">\u25B6</span><span class="tn">' + esc(ev.name) + '</span>';
      if (ev.path) {
        var ep = esc(ev.path).replace(/"/g, '&quot;');
        h += '<span class="tp" data-path="' + ep + '"> ' + esc(ev.path) + '</span>';
      }
      if (ev.description) h += '<span class="td"> ' + esc(ev.description) + '</span>';
      var b = '';
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
      var body = b || '<em style="color:var(--dim)">No arguments</em>';
      c.innerHTML = '<div class="tc-h" onclick="toggleTC(this)">' + h + '</div>'
        + '<div class="tc-b' + (b ? '' : ' hide') + '">' + body + '</div>';
      target.appendChild(c);
      if (ev.command) { var bp = mkEl('div', 'bash-panel'); target.appendChild(bp); tState.bashPanel = bp; }
      hlBlock(c);
      break;
    }
    case 'tool_result': {
      if (tState.bashPanel && tState.bashBuf) { tState.bashPanel.textContent += tState.bashBuf; tState.bashBuf = ''; }
      var hadBash = !!tState.bashPanel;
      tState.bashPanel = null; tState.bashRaf = 0;
      if (hadBash && !ev.is_error) break;
      var r = mkEl('div', 'ev tr' + (ev.is_error ? ' err' : ''));
      var lb = ev.is_error ? 'FAILED' : 'OK';
      var lc = ev.is_error ? 'fail' : 'ok';
      r.innerHTML = '<div class="rl ' + lc + '">' + lb + '</div>' + esc(ev.content);
      target.appendChild(r); break;
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
        + '<span>Tokens <b>' + (ev.total_tokens || 0) + '</b></span>'
        + '<span>Cost <b>' + (ev.cost || 'N/A') + '</b></span>'
        + '</div></div><div class="rc-body md-body' + (usePre ? ' pre' : '') + '">' + rb + '</div>';
      hlBlock(rc);
      target.appendChild(rc); break;
    }
    case 'system_prompt':
    case 'prompt': {
      var cls = t === 'system_prompt' ? 'system-prompt' : 'prompt';
      var label = t === 'system_prompt' ? 'System Prompt' : 'Prompt';
      var el = mkEl('div', 'ev ' + cls);
      var body = typeof marked !== 'undefined' ? marked.parse(ev.text || '') : esc(ev.text || '');
      el.innerHTML = '<div class="' + cls + '-h">' + label + '</div>'
        + '<div class="' + cls + '-body md-body">' + body + '</div>';
      hlBlock(el);
      target.appendChild(el);
      var bodyEl = el.querySelector('.' + cls + '-body');
      if (bodyEl) bodyEl.scrollTop = bodyEl.scrollHeight;
      break;
    }
    case 'usage_info': {
      var u = mkEl('div', 'ev usage');
      u.textContent = ev.text || '';
      target.appendChild(u); break;
    }
    }
  }

  function processOutputEvent(ev) {
    var t = ev.type;
    if (t === 'tool_call') {
      lastToolName = ev.name || '';
      llmPanel = null; llmPanelState = mkS(); pendingPanel = false;
    }
    if (t === 'tool_result' && lastToolName !== 'finish') { pendingPanel = true; }
    if (pendingPanel && (t === 'thinking_start' || t === 'text_delta')) {
      llmPanel = mkEl('div', 'llm-panel');
      O.appendChild(llmPanel);
      llmPanelState = mkS(); pendingPanel = false;
    }
    var target = O, tState = state;
    if (llmPanel && (t === 'thinking_start' || t === 'thinking_delta' || t === 'thinking_end'
      || t === 'text_delta' || t === 'text_end')) {
      target = llmPanel; tState = llmPanelState;
    }
    handleOutputEvent(ev, target, tState);
    if (target === llmPanel) llmPanel.scrollTop = llmPanel.scrollHeight;
  }

  // --- Scrolling ---

  function sb() {
    if (!_scrollLock && !scrollRaf) {
      scrollRaf = requestAnimationFrame(function() {
        O.scrollTo({ top: O.scrollHeight, behavior: 'instant' });
        scrollRaf = 0;
      });
    }
  }

  O.addEventListener('wheel', function(e) {
    if (isRunning && e.deltaY < 0) _scrollLock = true;
  });
  O.addEventListener('scroll', function() {
    if (_scrollLock) {
      var atBottom = O.scrollTop + O.clientHeight >= O.scrollHeight - 150;
      if (atBottom) _scrollLock = false;
    }
  });
  new MutationObserver(function() { sb(); }).observe(O, { childList: true, subtree: true, characterData: true });

  // --- Timer ---
  function startTimer() {
    t0 = Date.now();
    if (timerIv) clearInterval(timerIv);
    timerIv = setInterval(function() {
      var s = Math.floor((Date.now() - t0) / 1000);
      var m = Math.floor(s / 60);
      statusText.textContent = 'Running ' + (m > 0 ? m + 'm ' : '') + s % 60 + 's';
    }, 1000);
  }
  function stopTimer() { if (timerIv) { clearInterval(timerIv); timerIv = null; } }



  // --- Clear chat ---
  function resetChatUI() {
    clearOutput();
    resetOutputState();
    removeSpinner();
    clearWorktreeBar();
    setTaskText('');
    vscode.setState(null);
    if (welcome) {
      welcome.style.display = '';
      O.appendChild(welcome);
    }
  }

  function doClearChat() {
    resetChatUI();
    vscode.postMessage({ type: 'newChat' });
    vscode.postMessage({ type: 'getWelcomeSuggestions' });
  }

  // --- Refresh history ---
  function refreshHistory() {
    if (sidebar.classList.contains('open')) {
      historyOffset = 0;
      historyHasMore = true;
      historyLoading = false;
      historyGeneration++;
      vscode.postMessage({ type: 'getHistory', query: historySearch.value, generation: historyGeneration });
    }
  }

  // --- Main event handler ---
  function handleEvent(ev) {
    var t = ev.type;
    switch (t) {
    case 'status':
      setRunningState(ev.running);
      break;
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
    case 'waitForUser':
      showWaitForUser(ev.instruction, ev.url);
      break;
    case 'error':
      addError(ev.text);
      break;
    case 'test_file_picker':
      inp.value = '@'; syncClearBtn();
      inp.selectionStart = inp.selectionEnd = 1;
      inp.focus();
      renderAutocomplete(ev.files || []);
      break;
    case 'test_model_picker':
      allModels = ev.models || [];
      if (ev.selected) { selectedModel = ev.selected; modelName.textContent = ev.selected; }
      modelDropdown.classList.add('open');
      modelSearch.value = '';
      renderModelList('');
      modelSearch.focus();
      break;
    case 'clear':
      clearOutput();
      resetOutputState();
      showSpinner();
      break;
    case 'clearChat':
      resetChatUI();
      vscode.postMessage({ type: 'getWelcomeSuggestions' });
      break;
    case 'followup_suggestion': {
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
    case 'task_events':
      if (ev.task) {
        setTaskText(ev.task);
        vscode.setState({ task: ev.task });
        if (welcome) welcome.style.display = 'none';
      }
      replayTaskEvents(ev.events || []);
      break;
    case 'setTaskText':
      setTaskText(ev.text || '');
      if (welcome) welcome.style.display = 'none';
      break;
    case 'focusInput':
      inp.focus();
      setTimeout(function() { inp.focus(); }, 100);
      setTimeout(function() { inp.focus(); }, 300);
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
      var el = t0 ? Math.floor((Date.now() - t0) / 1000) : 0;
      var em = Math.floor(el / 60);
      setReady('Done (' + (em > 0 ? em + 'm ' : '') + el % 60 + 's)');
      break;
    }
    case 'task_error':
    case 'task_stopped': {
      var isErr = t === 'task_error';
      var banner = mkEl('div', 'ev tr err');
      banner.innerHTML = '<div class="rl fail">' + (isErr ? 'ERROR' : 'STOPPED') + '</div>'
        + esc(isErr ? (ev.text || 'Unknown error') : 'Agent execution stopped by user');
      O.appendChild(banner);
      setReady(isErr ? 'Error' : 'Stopped');
      break;
    }
    default:
      processOutputEvent(ev);
      if (isRunning) showSpinner();
      break;
    }
    sb();
  }

  function updateInputDisabled() {
    inp.disabled = isRunning;
    if (isRunning) { clearGhost(); hideAC(); }
  }

  function setRunningState(running) {
    isRunning = running;
    sendBtn.style.display = running ? 'none' : 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';
    sendBtn.disabled = running;
    statusDot.classList.toggle('running', running);
    if (clearBtn) clearBtn.disabled = running;
    if (historyBtn) historyBtn.disabled = running;
    if (runPromptBtn && running) runPromptBtn.disabled = true;
    updateInputDisabled();
    if (running) {
      startTimer();
    }
  }

  function setReady(label) {
    setRunningState(false);
    stopTimer();
    removeSpinner();
    statusText.textContent = label || 'Ready';
    inp.focus();
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
  function replayTaskEvents(events) {
    clearOutput();
    resetOutputState();
    events.forEach(function(ev) {
      var t = ev.type;
      if (t === 'task_done' || t === 'task_error' || t === 'task_stopped'
        || t === 'followup_suggestion') {
        handleEvent(ev);
        return;
      }
      processOutputEvent(ev);
    });
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

  // --- Init and event listeners ---

  function init() {
    setupEventListeners();
    vscode.postMessage({ type: 'ready' });
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
    if (clearBtn) {
      clearBtn.addEventListener('click', function() {
        doClearChat();
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
      historyOffset = 0;
      historyHasMore = true;
      historyLoading = false;
      historyGeneration++;
      sidebar.classList.add('open');
      sidebarOverlay.classList.add('open');
      vscode.postMessage({ type: 'getHistory', generation: historyGeneration });
    });
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    historySearch.addEventListener('input', function() {
      historyOffset = 0;
      historyHasMore = true;
      historyLoading = false;
      historyGeneration++;
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
    setTaskText(prompt);
    vscode.setState({ task: prompt });
    vscode.postMessage({
      type: 'submit',
      prompt: prompt,
      model: selectedModel,
      attachments: attachments.map(function(a) {
        return { name: a.name, mimeType: a.type, data: a.data };
      }),
      useWorktree: !!(worktreeToggleBtn && worktreeToggleBtn.classList.contains('active'))
    });
    inp.value = '';
    inp.style.height = 'auto';
    attachments = [];
    renderFileChips();
    clearGhost();
    histIdx = -1;
    if (inputClearBtn) inputClearBtn.style.display = 'none';
    if (welcome) welcome.style.display = 'none';
  }

  function showAskUserModal(question) {
    askUserQuestion.textContent = question;
    askUserModal.style.display = 'flex';
    askUserInput.focus();
  }

  function showWaitForUser(instruction, url) {
    var div = mkEl('div', 'ev user-action-card');
    div.innerHTML =
      '<div style="border:2px solid var(--yellow);border-radius:8px;padding:14px;margin:8px 0;background:color-mix(in srgb, var(--yellow) 8%, transparent)">'
      + '<div style="font-weight:600;color:var(--yellow);margin-bottom:6px">\u23F8\uFE0F User Action Required</div>'
      + '<div style="margin-bottom:10px">' + esc(instruction || '') + '</div>'
      + '<button id="wait-user-done-btn" style="padding:6px 16px;background:var(--green);color:#000;'
      + 'border:none;border-radius:6px;cursor:pointer;font-weight:600">I\'m Done</button></div>';
    O.appendChild(div);
    div.querySelector('#wait-user-done-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'userActionDone' });
      this.disabled = true;
      this.textContent = 'Resumed';
    });
    sb();
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
        if (s.has_events) {
          setTaskText(s.preview || s.title || '');
          vscode.postMessage({ type: 'resumeSession', id: s.id });
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

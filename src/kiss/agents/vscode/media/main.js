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
  let histSaved = '';

  // Ghost text state
  let ghostTimer = null;
  let currentGhost = '';

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

  // Merge state
  let isMerging = false;

  // Streaming state (mirrors browser handleOutputEvent)
  let state = mkS();
  let lastToolName = '';
  let llmPanel = null;
  let llmPanelState = mkS();
  let pendingPanel = false;
  let pendingUserMsg = null;
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
    if (isRunning || !inp.value.trim()) return;
    ghostTimer = setTimeout(function() {
      ghostTimer = null;
      vscode.postMessage({ type: 'complete', query: inp.value });
    }, 300);
  }

  // --- File path detection (matches web Sorcar) ---
  function looksLikeFilePath(s) {
    if (s.startsWith('/') || s.startsWith('./') || s.startsWith('../') || s.startsWith('~/')) return true;
    if (!/\s/.test(s) && (s.indexOf('/') >= 0 || /\.\w{1,10}$/.test(s))) return true;
    return false;
  }

  // --- Shared rendering (ported from browser EVENT_HANDLER_JS) ---

  function esc(t) { var d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
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
        + '</div></div><div class="rc-body' + (usePre ? ' pre' : '') + '">' + rb + '</div>';
      hlBlock(rc);
      target.appendChild(rc); break;
    }
    case 'system_prompt': {
      var sp = mkEl('div', 'ev system-prompt');
      var spBody = typeof marked !== 'undefined' ? marked.parse(ev.text || '') : esc(ev.text || '');
      sp.innerHTML = '<div class="system-prompt-h">System Prompt</div>'
        + '<div class="system-prompt-body">' + spBody + '</div>';
      hlBlock(sp);
      target.appendChild(sp); break;
    }
    case 'prompt': {
      var pr = mkEl('div', 'ev prompt');
      var pBody = typeof marked !== 'undefined' ? marked.parse(ev.text || '') : esc(ev.text || '');
      pr.innerHTML = '<div class="prompt-h">Prompt</div>'
        + '<div class="prompt-body">' + pBody + '</div>';
      hlBlock(pr);
      target.appendChild(pr); break;
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

  // --- User message ---
  function showUserMsg(msg) {
    if (!msg) return;
    var um = mkEl('div', 'user-msg');
    var html = '';
    if (msg.images && msg.images.length) {
      html += '<div class="user-msg-images">';
      msg.images.forEach(function(url) { html += '<img src="' + url + '" class="user-msg-img">'; });
      html += '</div>';
    }
    html += esc(msg.text);
    um.innerHTML = html;
    O.appendChild(um);
  }

  // --- Clear chat ---
  function doClearChat() {
    O.innerHTML = '';
    resetOutputState();
    removeSpinner();
    if (welcome) welcome.style.display = '';
    vscode.postMessage({ type: 'getWelcomeSuggestions' });
  }

  // --- Refresh history ---
  function refreshHistory() {
    vscode.postMessage({ type: 'getHistory' });
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
      renderHistory(ev.sessions || []);
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
      inp.value = '@';
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
    case 'user_msg':
      pendingUserMsg = { text: ev.text, images: ev.images || [] };
      break;
    case 'clear':
      O.innerHTML = '';
      resetOutputState();
      showUserMsg(pendingUserMsg);
      pendingUserMsg = null;
      showSpinner();
      break;
    case 'clearChat':
      doClearChat();
      break;
    case 'followup_suggestion': {
      var fu = mkEl('div', 'followup-bar');
      fu.title = ev.text;
      fu.innerHTML = '<span class="fu-label">Suggested next</span>'
        + '<span class="fu-text">' + esc(ev.text) + '</span>';
      fu.addEventListener('click', function() {
        inp.value = ev.text;
        inp.focus();
      });
      O.appendChild(fu);
      sb();
      break;
    }
    case 'tasks_updated':
      refreshHistory();
      vscode.postMessage({ type: 'getWelcomeSuggestions' });
      // Refresh histCache for input cycling
      histCache = [];
      histIdx = -1;
      break;
    case 'welcome_suggestions':
      renderWelcomeSuggestions(ev.suggestions || []);
      break;
    case 'task_events':
      replayTaskEvents(ev.events || []);
      break;
    case 'ghost':
      updateGhost(ev.suggestion || '');
      break;
    case 'merge_data': {
      var mc = mkEl('div', 'ev merge-info');
      mc.innerHTML = '<div style="color:var(--yellow);font-weight:600;font-size:var(--fs-base);margin-bottom:4px">'
        + '\u2731 Reviewing ' + (ev.hunk_count || 0) + ' change(s)</div>'
        + '<div style="font-size:var(--fs-md);color:var(--dim)">Red = old lines, Blue = new lines. '
        + 'Use the merge toolbar to accept or reject changes.</div>';
      O.appendChild(mc);
      break;
    }
    case 'merge_started':
      isMerging = true;
      showMergeToolbar();
      break;
    case 'merge_ended':
      isMerging = false;
      hideMergeToolbar();
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

  function setRunningState(running) {
    isRunning = running;
    sendBtn.style.display = running ? 'none' : 'flex';
    stopBtn.style.display = running ? 'flex' : 'none';
    sendBtn.disabled = running;
    statusDot.classList.toggle('running', running);
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
    if (!suggestions || suggestions.length === 0) return;
    container.innerHTML = '';
    suggestions.forEach(function(s) {
      var chip = document.createElement('div');
      chip.className = 'suggestion-chip';
      chip.dataset.prompt = s.text;
      if (s.has_events) chip.dataset.hasEvents = 'true';
      var displayText = s.text.length > 80 ? s.text.substring(0, 77) + '...' : s.text;
      chip.innerHTML = '<span class="chip-label">Recent</span>' + esc(displayText);
      chip.addEventListener('click', function() {
        if (s.has_events) {
          vscode.postMessage({ type: 'resumeSession', id: s.text });
          if (welcome) welcome.style.display = 'none';
        } else {
          inp.value = s.text;
          inp.focus();
        }
      });
      container.appendChild(chip);
    });
    // Cache task strings for history cycling
    histCache = suggestions.map(function(s) { return s.text; });
    histIdx = -1;
  }

  // --- Task replay ---
  function replayTaskEvents(events) {
    O.innerHTML = '';
    resetOutputState();
    if (welcome) welcome.style.display = 'none';
    events.forEach(function(ev) {
      var t = ev.type;
      if (t === 'task_done' || t === 'task_error' || t === 'task_stopped'
        || t === 'followup_suggestion') {
        handleEvent(ev);
        return;
      }
      if (t === 'user_msg') {
        showUserMsg({ text: ev.text, images: ev.images || [] });
        return;
      }
      processOutputEvent(ev);
    });
    sb();
  }

  // --- Merge toolbar ---
  function showMergeToolbar() {
    if (document.getElementById('merge-toolbar')) return;
    var bar = mkEl('div', 'merge-toolbar');
    bar.id = 'merge-toolbar';
    bar.innerHTML =
      '<span class="merge-label">\u2731 Reviewing Changes</span>'
      + '<button class="merge-btn merge-accept" id="merge-accept-btn" title="Accept current change">\u2713 Accept</button>'
      + '<button class="merge-btn merge-reject" id="merge-reject-btn" title="Reject current change">\u2717 Reject</button>'
      + '<button class="merge-btn merge-nav" id="merge-prev-btn" title="Previous change">\u25C0 Prev</button>'
      + '<button class="merge-btn merge-nav" id="merge-next-btn" title="Next change">Next \u25B6</button>'
      + '<span class="merge-sep"></span>'
      + '<button class="merge-btn merge-accept-all" id="merge-accept-all-btn" title="Accept all changes">\u2713\u2713 Accept All</button>'
      + '<button class="merge-btn merge-reject-all" id="merge-reject-all-btn" title="Reject all changes">\u2717\u2717 Reject All</button>';
    O.parentElement.insertBefore(bar, O);
    document.getElementById('merge-accept-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'accept' });
    });
    document.getElementById('merge-reject-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'reject' });
    });
    document.getElementById('merge-prev-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'prev' });
    });
    document.getElementById('merge-next-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'next' });
    });
    document.getElementById('merge-accept-all-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'accept-all' });
    });
    document.getElementById('merge-reject-all-btn').addEventListener('click', function() {
      vscode.postMessage({ type: 'mergeAction', action: 'reject-all' });
    });
  }

  function hideMergeToolbar() {
    var bar = document.getElementById('merge-toolbar');
    if (bar) bar.remove();
  }

  // --- Init and event listeners ---

  function init() {
    setupEventListeners();
    vscode.postMessage({ type: 'ready' });
  }

  function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);
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
        clearGhost();
        inp.style.height = 'auto';
        inp.style.height = Math.min(inp.scrollHeight, 200) + 'px';
        return;
      }
      // History cycling (ArrowUp/Down when no autocomplete)
      if (e.key === 'ArrowUp' && autocomplete.style.display !== 'block') {
        if (histCache.length > 0 && (inp.value === '' || histIdx >= 0)) {
          e.preventDefault();
          if (histIdx < 0) histSaved = inp.value;
          histIdx = Math.min(histIdx + 1, histCache.length - 1);
          inp.value = histCache[histIdx];
          return;
        }
      }
      if (e.key === 'ArrowDown' && histIdx >= 0) {
        e.preventDefault();
        histIdx--;
        inp.value = histIdx >= 0 ? histCache[histIdx] : histSaved;
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
    });
    inp.addEventListener('blur', function() { clearGhost(); });
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
    if (clearBtn) {
      clearBtn.addEventListener('click', function() {
        doClearChat();
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
    historyBtn.addEventListener('click', function() {
      sidebar.classList.add('open');
      sidebarOverlay.classList.add('open');
      vscode.postMessage({ type: 'getHistory' });
    });
    sidebarClose.addEventListener('click', closeSidebar);
    sidebarOverlay.addEventListener('click', closeSidebar);
    historySearch.addEventListener('input', function() {
      vscode.postMessage({ type: 'getHistory', query: historySearch.value });
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
    document.querySelectorAll('.suggestion-chip').forEach(function(chip) {
      chip.addEventListener('click', function() {
        inp.value = chip.dataset.prompt;
        inp.focus();
      });
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

    // File path detection: @path shortcut or bare file path
    var fileMatch = prompt.match(/^@(\S+)$/);
    if (fileMatch) {
      vscode.postMessage({ type: 'openFile', path: fileMatch[1] });
      inp.value = '';
      return;
    }
    if (looksLikeFilePath(prompt)) {
      vscode.postMessage({ type: 'openFile', path: prompt });
      inp.value = '';
      return;
    }

    vscode.postMessage({
      type: 'submit',
      prompt: prompt,
      model: selectedModel,
      attachments: attachments.map(function(a) {
        return { name: a.name, mimeType: a.type, data: a.data };
      })
    });
    inp.value = '';
    inp.style.height = 'auto';
    attachments = [];
    renderFileChips();
    clearGhost();
    histIdx = -1;
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
      '<div style="border:2px solid var(--yellow);border-radius:8px;padding:14px;margin:8px 0;background:rgba(210,153,34,.08)">'
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

  function renderHistory(sessions) {
    if (sessions.length === 0) {
      historyList.innerHTML = '<div class="sidebar-empty">No conversations yet</div>';
      return;
    }
    historyList.innerHTML = '';
    sessions.forEach(function(s) {
      var div = document.createElement('div');
      div.className = 'sidebar-item';
      div.textContent = s.title || s.preview || 'Untitled';
      div.addEventListener('click', function() {
        if (s.has_events) {
          vscode.postMessage({ type: 'resumeSession', id: s.id });
        } else {
          inp.value = s.preview || s.title || '';
          inp.focus();
        }
        closeSidebar();
      });
      historyList.appendChild(div);
    });
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
      var sep = (!after || /^\s/.test(after)) ? '' : ' ';
      inp.value = before + file + sep + after;
      var np = before.length + file.length + sep.length;
      inp.setSelectionRange(np, np);
      vscode.postMessage({ type: 'recordFileUsage', path: file });
    }
    hideAC();
    inp.focus();
  }

  // Start
  init();
})();

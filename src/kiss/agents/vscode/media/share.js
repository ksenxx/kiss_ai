// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Standalone viewer script for shared chat pages
// (reports/chat-<id>.html, written by the daemon's shareChat
// handler).  The page body holds one section per task of the chat —
// a clone of the webview's static task panel above the task's
// transcript — so this script re-creates the interactions the webview
// attaches through JavaScript: collapsing / expanding event panels
// (media/main.js addCollapse), each section's task-panel drawer
// button, and the "Thinking" section toggle.  The styling
// comes from the page's inlined main.css, driven purely by the same
// classes this script toggles.
(function () {
  'use strict';

  /**
   * Collect the visible text of *node* for a collapsed panel's
   * one-line preview, skipping the chrome elements the webview's
   * collectText (media/main.js) also skips.
   *
   * @param {Node} node Panel content node.
   * @returns {string} The concatenated text.
   */
  function collectText(node) {
    if (node.nodeType === 3) return node.textContent || '';
    if (node.nodeType === 1 && node.classList) {
      if (
        node.classList.contains('panel-copy-btn') ||
        node.classList.contains('panel-stop-btn') ||
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

  /**
   * Fill or clear a panel's one-line collapsed preview, mirroring
   * collapsePreview in media/main.js: an expanded panel and a summary
   * panel show no preview; a collapsed one previews its content text.
   *
   * @param {Element} panelEl The collapsible panel.
   */
  function collapsePreview(panelEl) {
    const prev = panelEl.querySelector('.collapse-preview');
    if (!prev) return;
    if (
      panelEl.classList.contains('tc-summary') ||
      !panelEl.classList.contains('collapsed')
    ) {
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
    prev.textContent = txt.replace(/\s+/g, ' ').trim();
  }

  /**
   * Toggle a "Thinking" section open or closed.  The transcript's
   * think headers carry the webview's inline
   * onclick="toggleThink(this)", so the shared page defines the same
   * global (media/main.js exposes it as window.toggleThink too).
   *
   * @param {Element} el The clicked .lbl header of the think section.
   */
  window.toggleThink = function (el) {
    const p = el.parentElement;
    if (!p) return;
    const cnt = p.querySelector('.cnt');
    if (cnt) cnt.classList.toggle('hidden');
    const arrow = el.querySelector('.arrow');
    if (arrow) arrow.classList.toggle('collapsed');
  };

  /**
   * Collapse every run_parallel panel inside *root*, mirroring
   * collapseNestedRunParallel in media/main.js: a collapsed panel
   * hides its children, so a fan-out panel it swallowed must show as
   * collapsed too when it is expanded again — and, exactly like the
   * live webview, the swallowed fan-out's sub-agent tabs close with
   * it.
   *
   * @param {Element} root The panel that just collapsed.
   */
  function collapseNestedRunParallel(root) {
    const nested = root.querySelectorAll('.tc-run-parallel');
    for (let i = 0; i < nested.length; i++) {
      const p = nested[i];
      if (p.closest('.adjacent-task')) continue;
      if (!p.classList.contains('collapsed')) {
        p.classList.add('collapsed');
        p.classList.remove('user-pinned');
        collapsePreview(p);
      }
      syncSubagentTabs(p);
    }
  }

  // sharesub-coverage:start
  // The shared page's sub-agent tabs.
  //
  // The export (buildShareableHtml in media/main.js) renders every
  // sub-agent's transcript into a hidden .share-subagent section and
  // stamps each run_parallel panel with the task ids of the
  // sub-agents it fanned out (data-rp-subagents). This block
  // re-creates the live webview's tab behaviour on top of that
  // static data: expanding a fan-out panel opens its sub-agents'
  // tabs, collapsing it closes them, a tab closed by hand stays
  // closed until its panel is collapsed and expanded again, closing
  // a tab takes its descendants' tabs with it, and selecting a tab
  // swaps the transcript on screen (the root tab being the chat
  // itself). The tab strip reuses the webview's own markup and
  // classes (#tab-bar / .chat-tab..., styled by the inlined
  // main.css).

  // Ordered ids of the open sub-agent tabs (the root chat tab is
  // always open and lives outside this list).
  const openSubTabs = [];
  // The selected tab: a sub-agent task id, or null for the root tab.
  let activeSubTab = null;

  /**
   * The section holding *taskId*'s exported transcript, or null.
   *
   * @param {string} taskId The sub-agent's task id.
   * @returns {Element|null} Its .share-subagent section.
   */
  function subagentSection(taskId) {
    const sections = document.querySelectorAll('.share-subagent');
    for (let i = 0; i < sections.length; i++) {
      if (sections[i].getAttribute('data-task-id') === taskId)
        return sections[i];
    }
    return null;
  }

  /**
   * The sub-agent task ids a fan-out panel was stamped with.
   *
   * @param {Element} panelEl A .tc-run-parallel panel.
   * @returns {Array<string>} The ids (possibly empty).
   */
  function panelSubagentIds(panelEl) {
    const raw = panelEl.getAttribute('data-rp-subagents') || '';
    return raw.split(/\s+/).filter(Boolean);
  }

  /**
   * The tab strip, created on first use: the webview's own
   * #tab-bar > #tab-list markup inserted above #app (so the inlined
   * main.css styles it), holding the root chat tab.
   *
   * @returns {Element} The #tab-list element.
   */
  function ensureTabBar() {
    let list = document.getElementById('tab-list');
    if (list) return list;
    const bar = document.createElement('div');
    bar.id = 'tab-bar';
    list = document.createElement('div');
    list.id = 'tab-list';
    list.setAttribute('role', 'tablist');
    list.setAttribute('aria-label', 'Chat tabs');
    bar.appendChild(list);
    const app = document.getElementById('app');
    document.body.insertBefore(bar, app || document.body.firstChild);
    return list;
  }

  /**
   * Show or hide *sectionEl* (also syncing the [hidden] attribute the
   * page CSS keys on).
   *
   * @param {Element} sectionEl A .share-task section.
   * @param {boolean} show Whether it may be visible.
   */
  function setSectionShown(sectionEl, show) {
    if (show) sectionEl.removeAttribute('hidden');
    else sectionEl.setAttribute('hidden', '');
  }

  /**
   * Select the root chat tab (*taskId* null) or a sub-agent's tab:
   * swap which sections are on screen, repaint the strip, and put the
   * reader back at the top — the shared page's switchToTab.
   *
   * @param {string|null} taskId The tab to select.
   */
  function selectSubTab(taskId) {
    activeSubTab = taskId;
    const sections = document.querySelectorAll('.share-task');
    for (let i = 0; i < sections.length; i++) {
      const s = sections[i];
      if (s.classList.contains('share-subagent')) {
        setSectionShown(s, s.getAttribute('data-task-id') === taskId);
      } else {
        setSectionShown(s, taskId === null);
      }
    }
    renderShareTabBar();
    window.scrollTo(0, 0);
  }

  /**
   * Rebuild the tab strip from the open-tab list, mirroring the
   * webview's renderTabBar: the root chat tab first, then one
   * .chat-tab.subagent-tab per open sub-agent (done indicator, title,
   * close button), roving tabindex on the active tab. The bar only
   * shows when there is something beyond the root chat to switch to.
   */
  function renderShareTabBar() {
    const list = ensureTabBar();
    const bar = list.parentElement;
    bar.style.display = openSubTabs.length > 0 ? '' : 'none';
    list.innerHTML = '';
    const rootTitle = (document.title || '').trim() || 'Chat';

    function makeTab(taskId, title) {
      const el = document.createElement('div');
      el.className =
        'chat-tab' +
        (taskId === null ? '' : ' subagent-tab') +
        (taskId === activeSubTab ? ' active' : '');
      el.setAttribute('role', 'tab');
      el.setAttribute('tabindex', taskId === activeSubTab ? '0' : '-1');
      el.setAttribute(
        'aria-selected',
        taskId === activeSubTab ? 'true' : 'false',
      );
      el.setAttribute('aria-label', title);
      el.setAttribute('aria-controls', 'app');
      if (taskId !== null) {
        el.setAttribute('data-sub-tab-id', taskId);
        const dot = document.createElement('span');
        dot.className = 'subagent-indicator done status-tick';
        dot.title = 'Done';
        el.appendChild(dot);
      }
      const label = document.createElement('span');
      label.className = 'chat-tab-label';
      label.textContent = title;
      el.appendChild(label);
      if (taskId !== null) {
        const closeBtn = document.createElement('span');
        closeBtn.className = 'chat-tab-close';
        closeBtn.textContent = '\u00d7';
        closeBtn.setAttribute('role', 'button');
        closeBtn.setAttribute('tabindex', '0');
        closeBtn.setAttribute('aria-label', 'Close tab');
        el.appendChild(closeBtn);
      }
      list.appendChild(el);
    }

    makeTab(null, rootTitle.substring(0, 40));
    for (let i = 0; i < openSubTabs.length; i++) {
      const section = subagentSection(openSubTabs[i]);
      const title = section
        ? section.getAttribute('data-sub-title') || 'Sub-agent'
        : 'Sub-agent';
      makeTab(openSubTabs[i], title);
    }
  }

  /**
   * Open the tab of sub-agent *taskId* (a no-op when it has no
   * exported section or is open already), then open the tabs of the
   * expanded fan-outs INSIDE its transcript — the webview does the
   * same for a sub-agent that ran fan-outs of its own.
   *
   * @param {string} taskId The sub-agent to open.
   */
  function openSubagentTab(taskId) {
    const section = subagentSection(taskId);
    if (!section || openSubTabs.indexOf(taskId) !== -1) return;
    openSubTabs.push(taskId);
    renderShareTabBar();
    const nested = section.querySelectorAll(
      '.tc-run-parallel[data-rp-subagents]',
    );
    for (let i = 0; i < nested.length; i++) {
      if (!nested[i].classList.contains('collapsed'))
        syncSubagentTabs(nested[i]);
    }
    openOrphanChildTabs(taskId);
  }

  /**
   * Open the tabs of *taskId*'s orphan children — sub-agents no
   * fan-out panel claims (exported with `data-sub-orphan`; e.g. a
   * `run_agent` sub-task whose panel never learned its task id). The
   * static page has no panel entry to expand for them, so their tabs
   * ride along with their parent's transcript.
   *
   * @param {string} taskId The parent whose transcript just opened.
   */
  function openOrphanChildTabs(taskId) {
    const orphans = document.querySelectorAll(
      '.share-subagent[data-sub-orphan]',
    );
    for (let i = 0; i < orphans.length; i++) {
      if (orphans[i].getAttribute('data-parent-task-id') !== taskId) continue;
      openSubagentTab(orphans[i].getAttribute('data-task-id'));
    }
  }

  /**
   * True when *taskId* is *ancestorId* or a descendant of it, walked
   * over the sections' data-parent-task-id chain (cycle-guarded).
   *
   * @param {string} taskId The candidate descendant.
   * @param {string} ancestorId The candidate ancestor.
   * @returns {boolean} Whether the chain passes through *ancestorId*.
   */
  function isSubagentDescendant(taskId, ancestorId) {
    let cur = taskId;
    const seen = {};
    while (cur && !seen[cur]) {
      if (cur === ancestorId) return true;
      seen[cur] = true;
      const section = subagentSection(cur);
      cur = section ? section.getAttribute('data-parent-task-id') : '';
    }
    return false;
  }

  /**
   * Close the tab of sub-agent *taskId* along with every open
   * descendant's tab (closing a sub-agent also closes the tabs of the
   * fan-outs that sub-agent ran itself — the webview's rule). A close
   * the USER asked for is remembered on the owning panel, so the
   * sub-agent stays closed until the panel is collapsed and expanded
   * again; a close done by a collapsing panel is not.
   *
   * @param {string} taskId The sub-agent tab to close.
   * @param {boolean} byUser Whether the user clicked the close button.
   */
  function closeSubagentTab(taskId, byUser) {
    // Which tab takes over when the SELECTED tab is closed: a close
    // the user asked for falls back to plain index adjacency (the
    // webview's rule — whatever now sits where the tab was), while a
    // close done by a collapsing panel returns to the root chat.
    const before = [null].concat(openSubTabs);
    const activeAt = before.indexOf(activeSubTab);
    let activeClosed = false;
    for (let i = openSubTabs.length - 1; i >= 0; i--) {
      const open = openSubTabs[i];
      if (!isSubagentDescendant(open, taskId)) continue;
      openSubTabs.splice(i, 1);
      if (activeSubTab === open) activeClosed = true;
      const section = subagentSection(open);
      if (!section) continue;
      setSectionShown(section, false);
      // A sub-agent tab the webview reopens replays its transcript
      // with the fan-outs collapsed; fold this section's the same
      // way, so reopening the tab never resurrects grandchild tabs
      // on its own.
      const panels = section.querySelectorAll(
        '.tc-run-parallel[data-rp-subagents]',
      );
      for (let j = 0; j < panels.length; j++) {
        panels[j].classList.add('collapsed');
        panels[j].classList.remove('user-pinned');
        collapsePreview(panels[j]);
        panels[j]._shareClosedSubs = {};
      }
    }
    if (byUser) {
      const panels = document.querySelectorAll(
        '.tc-run-parallel[data-rp-subagents]',
      );
      for (let i = 0; i < panels.length; i++) {
        if (panelSubagentIds(panels[i]).indexOf(taskId) === -1) continue;
        if (!panels[i]._shareClosedSubs) panels[i]._shareClosedSubs = {};
        panels[i]._shareClosedSubs[taskId] = true;
      }
    }
    if (activeClosed && byUser) {
      const now = before.filter(
        id => id === null || openSubTabs.indexOf(id) !== -1,
      );
      selectSubTab(now[Math.min(activeAt, now.length - 1)]);
    } else if (activeClosed) {
      selectSubTab(null);
    } else {
      renderShareTabBar();
    }
  }

  /**
   * Put a fan-out panel's sub-agent tabs in step with its collapsed
   * state — the shared page's syncRunParallelPanel. A collapsed panel
   * takes its sub-agents' tabs with it and forgives every hand-close
   * (so expanding reopens them all); an expanded one opens a tab per
   * sub-agent the user has not closed by hand.
   *
   * @param {Element} panelEl The .tc-run-parallel panel.
   */
  function syncSubagentTabs(panelEl) {
    const ids = panelSubagentIds(panelEl);
    if (ids.length === 0) return;
    const collapsed = panelEl.classList.contains('collapsed');
    if (collapsed) {
      for (let i = 0; i < ids.length; i++) closeSubagentTab(ids[i], false);
      panelEl._shareClosedSubs = {};
      return;
    }
    const closed = panelEl._shareClosedSubs || {};
    for (let i = 0; i < ids.length; i++) {
      if (!closed[ids[i]]) openSubagentTab(ids[i]);
    }
  }

  /**
   * Move focus between tabs with the arrow / Home / End keys — the
   * webview's roving-tabindex pattern.
   *
   * @param {Element} fromEl The tab that has focus.
   * @param {string} key The pressed navigation key.
   */
  function moveShareTabFocus(fromEl, key) {
    const list = document.getElementById('tab-list');
    if (!list) return;
    const els = Array.prototype.slice.call(list.querySelectorAll('.chat-tab'));
    const at = els.indexOf(fromEl);
    if (at === -1) return;
    let to = at;
    if (key === 'ArrowLeft') to = at > 0 ? at - 1 : els.length - 1;
    else if (key === 'ArrowRight') to = at < els.length - 1 ? at + 1 : 0;
    else if (key === 'Home') to = 0;
    else if (key === 'End') to = els.length - 1;
    if (!els[to]) return;
    // Roving tabindex, exactly like the webview's moveTabFocus: the
    // freshly focused tab becomes the tablist's one Tab stop.
    for (let i = 0; i < els.length; i++) {
      els[i].setAttribute('tabindex', i === to ? '0' : '-1');
    }
    els[to].focus();
  }

  /**
   * Activate the tab element *tabEl* (the root tab has no
   * data-sub-tab-id).
   *
   * @param {Element} tabEl The clicked or activated .chat-tab.
   */
  function activateShareTab(tabEl) {
    const id = tabEl.getAttribute('data-sub-tab-id');
    selectSubTab(id === null ? null : id);
  }

  document.addEventListener('keydown', e => {
    const target = e.target;
    if (!target || typeof target.closest !== 'function') return;
    const closeEl = target.closest('.chat-tab-close');
    if (closeEl && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      e.stopPropagation();
      const tabEl = closeEl.closest('.chat-tab');
      const id = tabEl && tabEl.getAttribute('data-sub-tab-id');
      if (id) closeSubagentTab(id, true);
      return;
    }
    const tabEl = target.closest('.chat-tab');
    if (!tabEl) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      activateShareTab(tabEl);
    } else if (
      e.key === 'ArrowLeft' ||
      e.key === 'ArrowRight' ||
      e.key === 'Home' ||
      e.key === 'End'
    ) {
      e.preventDefault();
      moveShareTabFocus(tabEl, e.key);
    }
  });

  // The exported page may hold fan-out panels the user left expanded:
  // their sub-agent tabs open on load, exactly like the live layout
  // the export captured (a collapsed panel's stay shut). A chat
  // task's orphan children (sub-agents no fan-out panel claims in
  // the export) open alongside, there being no panel entry to expand
  // for them here.
  (function initShareSubagentTabs() {
    const roots = document.querySelectorAll(
      '.share-task:not(.share-subagent) .tc-run-parallel[data-rp-subagents]',
    );
    for (let i = 0; i < roots.length; i++) {
      if (!roots[i].classList.contains('collapsed')) syncSubagentTabs(roots[i]);
    }
    const orphans = document.querySelectorAll(
      '.share-subagent[data-sub-orphan]',
    );
    for (let i = 0; i < orphans.length; i++) {
      const pid = orphans[i].getAttribute('data-parent-task-id');
      if (subagentSection(pid)) continue; // opens with its parent's tab
      openSubagentTab(orphans[i].getAttribute('data-task-id'));
    }
  })();
  // sharesub-coverage:end

  // sharetheme-coverage:start
  // Light / dark mode for the shared page.  The page ships dark (the
  // inlined :root palette); the floating #share-theme-btn switches the
  // whole document to the Light Modern palette by toggling the
  // `light-theme` class on <html> (which the inlined
  // `html.light-theme` variable overrides key off) and flips the
  // active highlight.js theme through the two inlined style elements'
  // media attributes.  The choice persists in localStorage.
  const SHARE_THEME_KEY = 'kissShareTheme';

  const THEME_SUN_SVG =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<circle cx="12" cy="12" r="5"/>' +
    '<line x1="12" y1="1" x2="12" y2="3"/>' +
    '<line x1="12" y1="21" x2="12" y2="23"/>' +
    '<line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>' +
    '<line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>' +
    '<line x1="1" y1="12" x2="3" y2="12"/>' +
    '<line x1="21" y1="12" x2="23" y2="12"/>' +
    '<line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>' +
    '<line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>';

  const THEME_MOON_SVG =
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  /**
   * Apply *theme* ('light' or 'dark') to the shared page: toggle the
   * root element's `light-theme` class, enable the matching inlined
   * highlight.js stylesheet, and repaint the toggle button (which
   * shows the theme it switches TO: a sun in dark mode, a moon in
   * light mode).
   *
   * @param {string} theme 'light' or 'dark'.
   */
  function applyShareTheme(theme) {
    const light = theme === 'light';
    document.documentElement.classList.toggle('light-theme', light);
    const darkStyle = document.getElementById('hljs-style-dark');
    const lightStyle = document.getElementById('hljs-style-light');
    if (darkStyle) darkStyle.setAttribute('media', light ? 'not all' : 'all');
    if (lightStyle) lightStyle.setAttribute('media', light ? 'all' : 'not all');
    const btn = document.getElementById('share-theme-btn');
    if (btn) {
      btn.innerHTML = light ? THEME_MOON_SVG : THEME_SUN_SVG;
      const label = light ? 'Switch to dark mode' : 'Switch to light mode';
      btn.title = label;
      btn.setAttribute('aria-label', label);
    }
  }

  (function initShareTheme() {
    let saved = 'dark';
    try {
      if (localStorage.getItem(SHARE_THEME_KEY) === 'light') saved = 'light';
    } catch (_e) {
      /* file:// or private browsing: default to dark */
    }
    applyShareTheme(saved);
    const btn = document.getElementById('share-theme-btn');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const next = document.documentElement.classList.contains('light-theme')
        ? 'dark'
        : 'light';
      try {
        localStorage.setItem(SHARE_THEME_KEY, next);
      } catch (_e) {
        /* the theme simply won't persist */
      }
      applyShareTheme(next);
    });
  })();
  // sharetheme-coverage:end

  document.addEventListener('click', e => {
    const target = e.target;
    if (!target || typeof target.closest !== 'function') return;

    // sharesub-coverage:start
    const closeEl = target.closest('.chat-tab-close');
    if (closeEl) {
      e.stopPropagation();
      const closeTabEl = closeEl.closest('.chat-tab');
      const closeId = closeTabEl && closeTabEl.getAttribute('data-sub-tab-id');
      if (closeId) closeSubagentTab(closeId, true);
      return;
    }
    const tabEl = target.closest('.chat-tab');
    if (tabEl) {
      activateShareTab(tabEl);
      return;
    }
    // sharesub-coverage:end

    // resultimages-coverage:start
    // Tool-result images: the live webview installs a per-element
    // click-to-zoom listener (appendResultImages in main.js), which
    // outerHTML serialization cannot carry over — re-wire it here.
    const trImg = target.closest('.tr-img');
    if (trImg) {
      e.stopPropagation();
      trImg.classList.toggle('tr-img-full');
      return;
    }
    // resultimages-coverage:end

    const drawerBtn = target.closest('#task-panel-drawer-btn');
    if (drawerBtn) {
      // The page holds one #task-panel per task of the chat, so the
      // toggled panel must be the clicked button's own ancestor —
      // getElementById would always fold the first task's panel.
      const panel = drawerBtn.closest('#task-panel');
      if (!panel) return;
      const collapsed = panel.classList.toggle('drawer-collapsed');
      drawerBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
      drawerBtn.setAttribute(
        'aria-label',
        collapsed ? 'Expand task panel' : 'Collapse task panel',
      );
      return;
    }

    const header = target.closest('.collapse-header');
    if (!header) return;
    const panelEl = header.closest('.collapsible');
    if (!panelEl) return;
    e.stopPropagation();
    panelEl.classList.toggle('collapsed');
    if (panelEl.classList.contains('collapsed')) {
      panelEl.classList.remove('user-pinned');
    } else {
      panelEl.classList.add('user-pinned');
    }
    collapsePreview(panelEl);
    // sharesub-coverage:start
    // A fan-out panel's sub-agent tabs follow its collapsed state,
    // exactly like the live webview's syncRunParallelPanel.
    if (panelEl.classList.contains('tc-run-parallel'))
      syncSubagentTabs(panelEl);
    // sharesub-coverage:end
    if (panelEl.classList.contains('collapsed')) {
      collapseNestedRunParallel(panelEl);
    }
  });
})();

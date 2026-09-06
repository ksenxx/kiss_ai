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
   * Collapse every run_parallel panel inside *root*, mirroring the
   * visual half of collapseNestedRunParallel in media/main.js: a
   * collapsed panel hides its children, so a fan-out panel it
   * swallowed must show as collapsed too when it is expanded again.
   * (The live webview also closes the fan-out's sub-agent tabs; a
   * static page has none.)
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
    }
  }

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
    if (panelEl.classList.contains('collapsed')) {
      collapseNestedRunParallel(panelEl);
    }
  });
})();

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

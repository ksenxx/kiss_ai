// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/* global module */
/**
 * Per-panel "Copy to clipboard" helper for the KISS Sorcar chat webview.
 *
 * Exposes ``getRawText(panel)`` and ``addCopyButton(panel)`` on
 * ``window.PanelCopy`` for the browser-loaded ``main.js`` and also as a
 * CommonJS module so a Node + jsdom test can exercise the same code
 * paths without a VS Code host.
 *
 * The Copy button reads the panel's ``data-raw-text`` attribute when
 * present so panels that were rendered via ``marked.parse()`` (Result,
 * Prompt, System Prompt, streamed Thought text, ...) copy the agent's
 * original markdown source instead of the rendered HTML's
 * markdown-stripped ``textContent``.  Panels without a
 * ``data-raw-text`` override (Thought container, tool_call args, bash
 * output, ...) fall back to a DOM walk that still skips the button /
 * chevron / collapse-preview chrome.
 */
'use strict';

(function (root) {
  // Outline + check SVG icons shown on the Copy button.
  const PANEL_COPY_SVG =
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<rect x="9" y="9" width="13" height="13" rx="2"/>' +
    '<path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/></svg>';
  const PANEL_CHECK_SVG =
    '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" ' +
    'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
    'stroke-linejoin="round" aria-hidden="true">' +
    '<polyline points="20 6 9 17 4 12"/></svg>';

  // CSS classes for purely visual sub-nodes that must not leak into the
  // clipboard payload.
  const SKIP_CLASSES = ['panel-copy-btn', 'collapse-chv', 'collapse-preview'];

  function shouldSkip(el) {
    if (!el || !el.classList) return false;
    for (let i = 0; i < SKIP_CLASSES.length; i++) {
      if (el.classList.contains(SKIP_CLASSES[i])) return true;
    }
    return false;
  }

  /**
   * Collect the raw (un-rendered) text payload of ``node`` for the Copy
   * button.
   *
   * Walks ``node``'s subtree, but when an element carries a
   * ``data-raw-text`` attribute returns that attribute's value verbatim
   * instead of recursing into its formatted children.  Element children
   * are joined with ``\n`` so panels that stack several sub-blocks
   * (Thought panel: think + txt + tool_call + ...) round-trip without
   * losing block separation.
   */
  function getRawText(node) {
    if (!node) return '';
    if (node.nodeType === 3) return node.textContent || '';
    if (node.nodeType !== 1) return '';
    if (shouldSkip(node)) return '';
    if (
      node.dataset &&
      Object.prototype.hasOwnProperty.call(node.dataset, 'rawText')
    ) {
      return node.dataset.rawText || '';
    }
    let out = '';
    for (let i = 0; i < node.childNodes.length; i++) {
      const child = node.childNodes[i];
      const t = getRawText(child);
      if (!t) continue;
      if (
        out.length > 0 &&
        child.nodeType === 1 &&
        !out.endsWith('\n') &&
        !t.startsWith('\n')
      ) {
        out += '\n';
      }
      out += t;
    }
    return out;
  }

  /**
   * Normalise the raw text returned by ``getRawText`` so the clipboard
   * receives clean markdown: collapse trailing spaces before newlines,
   * cap runs of blank lines at one, and trim leading / trailing
   * newlines.  Indentation, code-block whitespace, and embedded single
   * spaces are preserved.
   */
  function normalise(text) {
    return String(text == null ? '' : text)
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^\n+|\n+$/g, '');
  }

  /**
   * Attach a Copy-to-clipboard button to a chat panel.
   *
   * The button sits absolutely in the top-right corner of ``panelEl``
   * (the helper adds the ``copyable`` class which sets
   * ``position: relative`` on the panel) and copies the panel's raw
   * text via ``navigator.clipboard.writeText`` (or a textarea +
   * ``document.execCommand('copy')`` fallback).  Clicking the button
   * never collapses the panel: the handler stops propagation before
   * the collapsible-header listener runs.  After a successful copy the
   * icon swaps to a check mark for 1.5 s as feedback.
   *
   * Idempotent: calling twice on the same panel is a no-op.
   *
   * @param {HTMLElement} panelEl - panel container to attach the
   *   button to.
   */
  function addCopyButton(panelEl) {
    if (!panelEl || panelEl.querySelector(':scope > .panel-copy-btn')) return;
    panelEl.classList.add('copyable');
    const doc = panelEl.ownerDocument || document;
    const btn = doc.createElement('button');
    btn.type = 'button';
    btn.className = 'panel-copy-btn';
    btn.title = 'Copy panel text';
    btn.setAttribute('aria-label', 'Copy panel text');
    btn.innerHTML = PANEL_COPY_SVG;
    btn.addEventListener('click', e => {
      e.stopPropagation();
      e.preventDefault();
      const text = normalise(getRawText(panelEl));
      const done = () => {
        btn.innerHTML = PANEL_CHECK_SVG;
        btn.classList.add('copied');
        setTimeout(() => {
          btn.innerHTML = PANEL_COPY_SVG;
          btn.classList.remove('copied');
        }, 1500);
      };
      const win =
        doc.defaultView || (typeof window !== 'undefined' ? window : null);
      const nav = win ? win.navigator : null;
      if (nav && nav.clipboard && nav.clipboard.writeText) {
        nav.clipboard.writeText(text).then(done, () => {});
      } else {
        const ta = doc.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        doc.body.appendChild(ta);
        ta.select();
        try {
          doc.execCommand('copy');
          done();
        } finally {
          doc.body.removeChild(ta);
        }
      }
    });
    panelEl.appendChild(btn);
  }

  const api = {
    getRawText: getRawText,
    addCopyButton: addCopyButton,
    normalise: normalise,
    PANEL_COPY_SVG: PANEL_COPY_SVG,
    PANEL_CHECK_SVG: PANEL_CHECK_SVG,
  };

  if (root && typeof root === 'object') {
    root.PanelCopy = api;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : null);

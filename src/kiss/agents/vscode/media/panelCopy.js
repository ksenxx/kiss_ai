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
  const SKIP_CLASSES = [
    'panel-copy-btn',
    'collapse-chv',
    'collapse-preview',
    'panel-ts',
    'panel-time',
  ];

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

  // panelts-coverage:start
  /**
   * Format an event timestamp (ms since epoch) as a human-readable
   * label in the user's locale.
   *
   * Every label carries the FULL date and a seconds-precision time
   * of day ("Mar 5, 2021 2:07:33 PM") so an event's exact moment is
   * always visible, even for same-day events.  Invalid / missing /
   * non-positive inputs format as the empty string so callers can
   * skip the badge.
   *
   * @param {number} ts - event time in ms since the epoch.
   * @returns {string} the "date time" label, or '' when *ts* is
   *   unusable.
   */
  function formatEventTs(ts) {
    const n = Number(ts);
    if (!isFinite(n) || n <= 0) return '';
    const d = new Date(n);
    // A finite ms value can still overflow the ECMAScript Date range
    // (±8.64e15); such a Date formats as "Invalid Date" — skip it.
    if (!isFinite(d.getTime())) return '';
    const day = d.toLocaleDateString([], {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
    const time = d.toLocaleTimeString([], {
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    });
    return day + ' ' + time;
  }

  /**
   * Return (creating on demand) the panel's bottom footer bar — the
   * ``div.panel-time`` element that renders as the LAST child of every
   * event panel.  The bar is shared by the event-timestamp badge
   * (``span.panel-ts``, left side) and the live "time spent" label
   * (``span.panel-elapsed``, right side, written by ``main.js``).
   *
   * The first call also attaches a one-time ``MutationObserver`` that
   * re-anchors the bar as the panel's last child whenever later
   * content (streamed thoughts, a tool_result output panel, ...) is
   * appended after it, so the bar always renders at the bottom — even
   * for replayed panels that never enter the live 1-second ticker.
   *
   * ``panel-time`` is in ``SKIP_CLASSES`` so nothing in the bar leaks
   * into the clipboard payload.
   *
   * @param {HTMLElement} panelEl - panel container to stamp.
   * @returns {HTMLElement} the panel's ``div.panel-time`` footer bar.
   */
  function ensurePanelFoot(panelEl) {
    const doc = panelEl.ownerDocument || document;
    // Find an existing direct-child bar (scan from the end; avoid
    // matching bars inside nested panels).
    let bar = null;
    for (let i = panelEl.children.length - 1; i >= 0; i--) {
      const c = panelEl.children[i];
      if (c.classList && c.classList.contains('panel-time')) {
        bar = c;
        break;
      }
    }
    if (!bar) {
      bar = doc.createElement('div');
      bar.className = 'panel-time';
      panelEl.appendChild(bar);
    }
    if (!panelEl._kissPanelFootObs) {
      const found = bar;
      const obs = new doc.defaultView.MutationObserver(() => {
        if (
          found.parentNode === panelEl &&
          found !== panelEl.lastElementChild
        ) {
          panelEl.appendChild(found);
        }
      });
      obs.observe(panelEl, {childList: true});
      panelEl._kissPanelFootObs = obs;
    }
    return bar;
  }

  /**
   * Attach a date + seconds event-timestamp badge to a chat panel's bottom
   * footer bar.
   *
   * The badge (``span.panel-ts``) is inserted as the FIRST child of
   * the panel's ``div.panel-time`` footer bar (see
   * ``ensurePanelFoot``) so it renders bottom-LEFT — in the same bar
   * as, and to the left of, the right-aligned "time spent" label
   * (``span.panel-elapsed``).  Its full ``toLocaleString`` form is
   * exposed as a hover tooltip.  ``panel-time`` is in
   * ``SKIP_CLASSES`` so the badge never leaks into the clipboard
   * payload.
   *
   * Idempotent: a panel keeps its FIRST badge; later calls return it
   * unchanged.  Returns ``null`` (and adds nothing) when *panelEl* is
   * falsy or *ts* does not format to a label.
   *
   * @param {HTMLElement} panelEl - panel container to stamp.
   * @param {number} ts - event time in ms since the epoch.
   * @returns {HTMLElement|null} the badge element, or null.
   */
  function addPanelTimestamp(panelEl, ts) {
    if (!panelEl) return null;
    const label = formatEventTs(ts);
    if (!label) return null;
    const existing = panelEl.querySelector(':scope > .panel-time > .panel-ts');
    if (existing) return existing;
    const bar = ensurePanelFoot(panelEl);
    const doc = panelEl.ownerDocument || document;
    const span = doc.createElement('span');
    span.className = 'panel-ts';
    span.textContent = label;
    span.title = new Date(Number(ts)).toLocaleString();
    bar.insertBefore(span, bar.firstChild);
    return span;
  }
  // panelts-coverage:end

  const api = {
    getRawText: getRawText,
    addCopyButton: addCopyButton,
    normalise: normalise,
    formatEventTs: formatEventTs,
    ensurePanelFoot: ensurePanelFoot,
    addPanelTimestamp: addPanelTimestamp,
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

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/* global module */
'use strict';

(function (root) {
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

  function normalise(text) {
    return String(text == null ? '' : text)
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^\n+|\n+$/g, '');
  }

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
  function formatEventTs(ts) {
    const n = Number(ts);
    if (!isFinite(n) || n <= 0) return '';
    const d = new Date(n);
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

  function ensurePanelFoot(panelEl) {
    const doc = panelEl.ownerDocument || document;
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

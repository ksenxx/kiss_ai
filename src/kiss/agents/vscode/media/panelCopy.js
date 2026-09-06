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

  // fmtcopy-coverage:start
  // Tags that read as a paragraph-level block: a blank line separates
  // them in the copied text.
  const PARA_TAGS = {
    P: 1,
    H1: 1,
    H2: 1,
    H3: 1,
    H4: 1,
    H5: 1,
    H6: 1,
    UL: 1,
    OL: 1,
    PRE: 1,
    BLOCKQUOTE: 1,
    TABLE: 1,
  };
  // Tags that read as one line of their own: a single newline
  // separates them.
  const LINE_TAGS = {
    DIV: 1,
    LI: 1,
    TR: 1,
    DT: 1,
    DD: 1,
    DL: 1,
    SECTION: 1,
    ARTICLE: 1,
    HEADER: 1,
    FOOTER: 1,
    ASIDE: 1,
    MAIN: 1,
    FIGURE: 1,
    FIGCAPTION: 1,
    DETAILS: 1,
    SUMMARY: 1,
    ADDRESS: 1,
  };

  /**
   * Convert a rendered DOM subtree into readable plain text — what
   * the panel LOOKS like, not its HTML source.  Block elements become
   * line / paragraph breaks, `<li>` items get "- " or "N. " markers
   * (nested lists indent two spaces per level), `<pre>` text keeps
   * its own line breaks and inner spacing (only whitespace at block
   * edges is trimmed; the copy path additionally collapses runs of
   * 3+ newlines), images reduce to their alt text, table cells are
   * joined with " | ", and the chrome elements copy always skips
   * (SKIP_CLASSES) are left out.  Used by the Result panel so its
   * copy button copies formatted text instead of the summary's raw
   * HTML.
   *
   * @param {Element} root The rendered element to serialize.
   * @returns {string} The formatted plain text (untrimmed; callers
   *     normalise it).
   */
  function formattedTextFromNode(root) {
    const st = {out: ''};

    function breakLine(want) {
      if (!st.out) return;
      st.out = st.out.replace(/[ \t]+$/, '');
      let have = 0;
      for (
        let i = st.out.length - 1;
        i >= 0 && st.out.charAt(i) === '\n';
        i--
      ) {
        have++;
      }
      while (have < want) {
        st.out += '\n';
        have++;
      }
    }

    function addText(text, pre) {
      if (pre) {
        st.out += text;
        return;
      }
      let t = String(text).replace(/\s+/g, ' ');
      if (!t) return;
      if (st.out === '' || /\s$/.test(st.out)) t = t.replace(/^ /, '');
      st.out += t;
    }

    function walk(node, ctx) {
      if (node.nodeType === 3) {
        addText(node.textContent || '', ctx.pre);
        return;
      }
      if (node.nodeType !== 1) return;
      if (shouldSkip(node)) return;
      const tag = node.tagName;
      if (tag === 'SCRIPT' || tag === 'STYLE' || tag === 'TEMPLATE') return;
      if (tag === 'BR') {
        st.out += '\n';
        return;
      }
      if (tag === 'IMG') {
        addText(node.getAttribute('alt') || '', false);
        return;
      }
      if (tag === 'HR') {
        breakLine(1);
        st.out += '---';
        breakLine(1);
        return;
      }
      const para = PARA_TAGS[tag] === 1;
      const line = LINE_TAGS[tag] === 1;
      if (para) breakLine(2);
      else if (line) breakLine(1);
      let childCtx = ctx;
      if (tag === 'PRE') {
        childCtx = {pre: true, lists: ctx.lists};
      } else if (tag === 'UL') {
        childCtx = {pre: ctx.pre, lists: ctx.lists.concat([{ordered: false}])};
      } else if (tag === 'OL') {
        let start = parseInt(node.getAttribute('start') || '1', 10);
        if (!isFinite(start)) start = 1;
        childCtx = {
          pre: ctx.pre,
          lists: ctx.lists.concat([{ordered: true, n: start}]),
        };
      } else if (tag === 'LI') {
        const depth = ctx.lists.length;
        let indent = '';
        for (let i = 1; i < depth; i++) indent += '  ';
        const top = depth > 0 ? ctx.lists[depth - 1] : null;
        let marker = '- ';
        if (top && top.ordered) {
          marker = top.n + '. ';
          top.n++;
        }
        st.out += indent + marker;
      } else if (
        (tag === 'TD' || tag === 'TH') &&
        node.previousElementSibling
      ) {
        st.out = st.out.replace(/[ \t]+$/, '');
        st.out += ' | ';
      }
      for (let i = 0; i < node.childNodes.length; i++) {
        walk(node.childNodes[i], childCtx);
      }
      if (para) breakLine(2);
      else if (line) breakLine(1);
    }

    walk(root, {pre: false, lists: []});
    return st.out;
  }
  // fmtcopy-coverage:end

  function normalise(text) {
    return String(text == null ? '' : text)
      .replace(/[ \t]+\n/g, '\n')
      .replace(/\n{3,}/g, '\n\n')
      .replace(/^\n+|\n+$/g, '');
  }

  /**
   * Copy `text` to the clipboard through a hidden textarea and
   * document.execCommand('copy') — the fallback for contexts where the
   * async clipboard API is unavailable or rejected. Shared by every
   * same-page copy control (main.js, tips.js and this module).
   *
   * @param {string} text What to place on the clipboard.
   * @param {Document} [doc] Owner document; defaults to the page's.
   * @returns {boolean} True when the copy command reported success.
   */
  function fallbackCopyText(text, doc) {
    const d = doc || document;
    const ta = d.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    d.body.appendChild(ta);
    ta.select();
    let ok = false;
    try {
      ok = d.execCommand('copy');
    } catch (_err) {
      ok = false;
    } finally {
      d.body.removeChild(ta);
    }
    return ok;
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
    // copyflash0903-coverage:start
    // The revert timer is tracked per button and restarted on every
    // copy: a bare setTimeout let a rapid second click's flash be cut
    // short by the first click's stale timer.
    let flashTimer = null;
    btn.addEventListener('click', e => {
      e.stopPropagation();
      e.preventDefault();
      const text = normalise(getRawText(panelEl));
      const done = () => {
        btn.innerHTML = PANEL_CHECK_SVG;
        btn.classList.add('copied');
        if (flashTimer) clearTimeout(flashTimer);
        flashTimer = setTimeout(() => {
          flashTimer = null;
          btn.innerHTML = PANEL_COPY_SVG;
          btn.classList.remove('copied');
        }, 1500);
      };
      // copyflash0903-coverage:end
      const win =
        doc.defaultView || (typeof window !== 'undefined' ? window : null);
      const nav = win ? win.navigator : null;
      if (nav && nav.clipboard && nav.clipboard.writeText) {
        nav.clipboard.writeText(text).then(done, () => {});
      } else if (fallbackCopyText(text, doc)) {
        done();
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
    formattedTextFromNode: formattedTextFromNode,
    addCopyButton: addCopyButton,
    fallbackCopyText: fallbackCopyText,
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

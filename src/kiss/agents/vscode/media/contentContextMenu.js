// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Right-click ("contextual") menu for read-only content surfaces.
//
// VS Code webviews suppress Chromium's native context menu, so a user who
// opens an .html file (or a generated report) in a Sorcar content tab gets
// nothing on right-click — no Copy, no Select All.  The HTML is rendered in
// an iframe sandboxed with `allow-scripts` only, i.e. an opaque origin the
// parent document cannot reach into, so the menu has to live *inside* that
// document.  This module is therefore written to be usable two ways:
//
//   * required/loaded normally by the chat webview (parent document), and
//   * serialised via `contentContextMenuBootstrapHtml()` into the iframe
//     `srcdoc` so the sandboxed page installs the very same menu on itself.

/* global module */
'use strict';

(function (root) {
  const MENU_ID = 'sorcar-content-context-menu';

  const MENU_CSS =
    '#' +
    MENU_ID +
    '{position:fixed;z-index:2147483647;min-width:180px;padding:4px 0;' +
    'margin:0;background:#252526;color:#ccc;border:1px solid #454545;' +
    'border-radius:6px;box-shadow:0 4px 16px rgb(0 0 0 / 50%);' +
    "font:13px -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica," +
    'Arial,sans-serif;user-select:none;-webkit-user-select:none}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-item{display:flex;justify-content:space-between;' +
    'gap:24px;padding:5px 14px;cursor:pointer;white-space:nowrap}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-item:hover{background:#04395e;color:#fff}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-item.disabled{opacity:.4;cursor:default;' +
    'pointer-events:none}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-key{opacity:.6}';

  // A VS Code webview serves `script-src 'nonce-<nonce>'`, and an
  // `about:srcdoc` iframe *inherits* the CSP of the document that created
  // it — `sandbox="allow-scripts"` does not lift that.  The bootstrap the
  // iframe receives must therefore carry the very nonce this file was
  // loaded with, which is only readable while the module body runs.
  const SCRIPT_NONCE = readOwnNonce();

  function readOwnNonce() {
    if (typeof document === 'undefined') return '';
    const self = document.currentScript;
    if (!self) return '';
    return String(self.nonce || self.getAttribute('nonce') || '');
  }

  function escapeAttr(value) {
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/"/g, '&quot;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /** True when *el* is an `<input>`/`<textarea>` with a text cursor. */
  function isTextField(el) {
    const tag = String((el && el.tagName) || '').toLowerCase();
    if (tag === 'textarea') return true;
    if (tag !== 'input') return false;
    const type = String(el.getAttribute('type') || 'text').toLowerCase();
    return (
      type === 'text' ||
      type === 'search' ||
      type === 'url' ||
      type === 'tel' ||
      type === 'password' ||
      type === 'email' ||
      type === 'number'
    );
  }

  /** True when text may actually be typed into *el*. */
  function isWritable(el) {
    if (!el) return false;
    if (el.isContentEditable) return true;
    if (!isTextField(el)) return false;
    return !el.disabled && !el.readOnly;
  }

  /**
   * Return the text currently selected in *doc*, or ''.
   *
   * A focused `<input>`/`<textarea>` keeps its selection in its own value
   * rather than in the document Selection, so those are read through
   * `selectionStart`/`selectionEnd`; otherwise Copy would be dead whenever
   * the user highlighted text inside a field.
   */
  function selectedText(doc, target) {
    const field = target && isTextField(target) ? target : null;
    if (field) {
      const start = field.selectionStart;
      const end = field.selectionEnd;
      if (typeof start === 'number' && typeof end === 'number' && end > start) {
        return String(field.value || '').slice(start, end);
      }
      return '';
    }
    const win = doc.defaultView;
    if (!win || typeof win.getSelection !== 'function') return '';
    const sel = win.getSelection();
    if (!sel) return '';
    return selectionText(sel);
  }

  /**
   * Read every range of *sel* as text.
   *
   * `Selection.toString()` reports only the first range in some engines, so
   * a multi-range Select All would otherwise be truncated to its first
   * fragment the moment the user chose Copy.
   */
  function selectionText(sel) {
    const parts = [];
    for (let i = 0; i < sel.rangeCount; i++) {
      const text = String(sel.getRangeAt(i).toString() || '').trim();
      if (text) parts.push(text);
    }
    return parts.join('\n');
  }

  /** True for nodes whose text must never land in a selection. */
  function isHiddenSource(el) {
    const tag = String(el.tagName || '').toLowerCase();
    if (tag === 'script' || tag === 'style') return true;
    return el.hasAttribute('data-sorcar-ctx');
  }

  /** True when *node* has a `<script>`/`<style>`/menu ancestor. */
  function inHiddenSource(node, root) {
    let el = node;
    while (el && el !== root) {
      if (el.nodeType === 1 && isHiddenSource(el)) return true;
      el = el.parentNode;
    }
    return false;
  }

  function copyViaTextarea(doc, text) {
    const ta = doc.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.left = '0';
    ta.style.opacity = '0';
    (doc.body || doc.documentElement).appendChild(ta);
    ta.select();
    ta.setSelectionRange(0, ta.value.length);
    let ok = false;
    try {
      ok = !!doc.execCommand('copy');
    } catch (_e) {
      ok = false;
    }
    ta.parentNode.removeChild(ta);
    return ok;
  }

  /**
   * Copy *text* to the clipboard from *doc*.
   *
   * The asynchronous Clipboard API is unavailable inside an opaque-origin
   * sandboxed iframe, so a hidden-textarea `execCommand('copy')` is always
   * used as the fallback and as the sole path when the API rejects.
   */
  function copyText(doc, text) {
    if (!text) return Promise.resolve(false);
    const win = doc.defaultView;
    const nav = win && win.navigator;
    if (nav && nav.clipboard && typeof nav.clipboard.writeText === 'function') {
      try {
        return Promise.resolve(nav.clipboard.writeText(text)).then(
          () => true,
          () => copyViaTextarea(doc, text),
        );
      } catch (_e) {
        return Promise.resolve(copyViaTextarea(doc, text));
      }
    }
    return Promise.resolve(copyViaTextarea(doc, text));
  }

  /**
   * Select every visible text node of *doc*'s body and return its text.
   *
   * `<script>`/`<style>` text and the menu's own markup live in that same
   * body — at any depth — so the document is walked text node by text node
   * and each run of visible nodes becomes one `Range`.  Selecting the body
   * wholesale would instead hand the user the page's inline JavaScript the
   * moment they followed Select All with Copy.
   *
   * Chromium and WebKit keep at most one `Range` per `Selection`, so the
   * returned text is assembled from the ranges as they are built; the
   * caller must use it rather than re-reading the selection.
   */
  function selectAll(doc, scope) {
    const win = doc.defaultView;
    const body = scope || doc.body || doc.documentElement;
    if (!win || !body || typeof win.getSelection !== 'function') return '';
    const sel = win.getSelection();
    if (!sel) return '';
    sel.removeAllRanges();
    const walker = doc.createTreeWalker(body, 4 /* SHOW_TEXT */, null, false);
    const parts = [];
    let range = null;
    let node = walker.nextNode();
    while (node) {
      if (inHiddenSource(node, body)) {
        range = addRange(sel, range, parts);
      } else {
        if (!range) {
          range = doc.createRange();
          range.setStart(node, 0);
        }
        range.setEnd(node, node.length);
      }
      node = walker.nextNode();
    }
    addRange(sel, range, parts);
    return parts.join('\n');
  }

  /** Commit *range* to *sel*, recording its text, and reset it. */
  function addRange(sel, range, parts) {
    if (!range || range.collapsed) return null;
    sel.addRange(range);
    const text = String(range.toString() || '').trim();
    if (text) parts.push(text);
    return null;
  }

  function closestTag(node, tag) {
    let el = node;
    while (el && el.nodeType !== 1) el = el.parentNode;
    while (el && el.nodeType === 1) {
      if (String(el.tagName || '').toLowerCase() === tag) return el;
      el = el.parentNode;
    }
    return null;
  }

  /** True when *node* sits inside an editable field. */
  function editableTarget(node) {
    let el = node;
    while (el && el.nodeType !== 1) el = el.parentNode;
    while (el && el.nodeType === 1) {
      const tag = String(el.tagName || '').toLowerCase();
      if (tag === 'textarea') return el;
      if (tag === 'input') return el;
      if (el.isContentEditable) return el;
      el = el.parentNode;
    }
    return null;
  }

  function isMac(doc) {
    const win = doc.defaultView;
    const nav = win && win.navigator;
    const plat = String((nav && (nav.platform || nav.userAgent)) || '');
    return /Mac|iPhone|iPad/i.test(plat);
  }

  /**
   * Build the ordered menu model for a right-click on *target* in *doc*.
   *
   * Exported so tests can assert the model without touching the DOM.
   */
  function buildMenuModel(doc, target) {
    const mod = isMac(doc) ? '\u2318' : 'Ctrl+';
    const editable = editableTarget(target);
    const sel = selectedText(doc, editable);
    const link = closestTag(target, 'a');
    const img = closestTag(target, 'img');
    const items = [];
    items.push({
      id: 'copy',
      label: 'Copy',
      key: mod + 'C',
      enabled: sel.length > 0,
      text: sel,
    });
    if (isWritable(editable)) {
      items.push({id: 'paste', label: 'Paste', key: mod + 'V', enabled: true});
    }
    if (link) {
      // The resolved property, not the raw attribute: a relative href would
      // otherwise be copied as an unusable fragment such as "../a.html".
      const href = String(link.href || '');
      items.push({
        id: 'copy-link',
        label: 'Copy Link Address',
        key: '',
        enabled: href.length > 0,
        text: href,
      });
    }
    if (img) {
      // currentSrc is what the browser actually fetched (srcset/picture).
      const src = String(img.currentSrc || img.src || '');
      items.push({
        id: 'copy-image-link',
        label: 'Copy Image Address',
        key: '',
        enabled: src.length > 0,
        text: src,
      });
    }
    items.push({
      id: 'select-all',
      label: 'Select All',
      key: mod + 'A',
      enabled: true,
    });
    return items;
  }

  /** Insert *text* into the contenteditable *el* at its caret. */
  function insertIntoContentEditable(doc, el, text) {
    const win = doc.defaultView;
    const sel = win && win.getSelection && win.getSelection();
    const node = doc.createTextNode(text);
    let range = null;
    if (sel && sel.rangeCount) {
      const live = sel.getRangeAt(0);
      if (el.contains(live.commonAncestorContainer)) range = live;
    }
    if (range) {
      range.deleteContents();
      range.insertNode(node);
      range.setStartAfter(node);
      range.collapse(true);
      sel.removeAllRanges();
      sel.addRange(range);
      return;
    }
    el.appendChild(node);
  }

  function pasteInto(doc, el) {
    const win = doc.defaultView;
    const nav = win && win.navigator;
    if (
      !nav ||
      !nav.clipboard ||
      typeof nav.clipboard.readText !== 'function'
    ) {
      return Promise.resolve(false);
    }
    // readText() throws synchronously when the document is not allowed to
    // read the clipboard, so the call itself has to be guarded.
    let pending;
    try {
      pending = Promise.resolve(nav.clipboard.readText());
    } catch (_e) {
      return Promise.resolve(false);
    }
    return pending.then(
      text => {
        const value = String(text || '');
        if (el.isContentEditable) insertIntoContentEditable(doc, el, value);
        else if (typeof el.setRangeText === 'function') {
          el.setRangeText(value, el.selectionStart, el.selectionEnd, 'end');
        } else {
          el.value = String(el.value || '') + value;
        }
        el.dispatchEvent(new win.Event('input', {bubbles: true}));
        return true;
      },
      () => false,
    );
  }

  /**
   * Install a Copy / Select All context menu on *doc*.
   *
   * Returns a handle with `close()` and `dispose()` so a caller can tear the
   * menu down; `dispose()` also removes every listener it registered.
   */
  function installContentContextMenu(doc, opts) {
    const options = opts || {};
    const win = doc.defaultView;
    let menu = null;
    // Chromium keeps a single Range per Selection, so a multi-range Select
    // All cannot be re-read from the DOM and its text has to be remembered.
    // The Range the menu itself installed is remembered alongside it: the
    // cache is only honoured while the live selection is still exactly that
    // Range, so any selection the user makes afterwards — narrower, wider
    // or elsewhere — discards it.  `selectionchange` is dispatched
    // asynchronously, so it cannot be relied on for this.
    let selectAllText = '';
    let selectAllRange = null;

    function cachedSelectAllText() {
      if (!selectAllRange) return '';
      const sel = win && win.getSelection && win.getSelection();
      if (!sel || sel.rangeCount !== 1) return '';
      const live = sel.getRangeAt(0);
      const same =
        live === selectAllRange ||
        (live.compareBoundaryPoints(live.START_TO_START, selectAllRange) ===
          0 &&
          live.compareBoundaryPoints(live.END_TO_END, selectAllRange) === 0);
      return same ? selectAllText : '';
    }

    function close() {
      if (menu && menu.parentNode) menu.parentNode.removeChild(menu);
      menu = null;
    }

    function activate(item) {
      close();
      if (item.id === 'select-all') {
        // Select All leaves a real DOM selection behind, so a following
        // Copy simply reads it — no cached text that can go stale when the
        // user narrows the selection afterwards.  Over a field it selects
        // that field's contents, exactly like the native menu does.
        if (item.editable && isTextField(item.editable)) {
          item.editable.focus();
          item.editable.setSelectionRange(
            0,
            String(item.editable.value || '').length,
          );
        } else {
          selectAllText = selectAll(doc, item.editable);
          const sel = win && win.getSelection && win.getSelection();
          selectAllRange = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
        }
        return;
      }
      if (item.id === 'paste') {
        pasteInto(doc, item.editable);
        return;
      }
      copyText(doc, item.text || '');
    }

    function ensureStyle() {
      if (doc.getElementById(MENU_ID + '-style')) return;
      const style = doc.createElement('style');
      style.id = MENU_ID + '-style';
      style.setAttribute('data-sorcar-ctx', '');
      style.textContent = MENU_CSS;
      (doc.head || doc.documentElement).appendChild(style);
    }

    function open(x, y, target) {
      close();
      ensureStyle();
      const model = buildMenuModel(doc, target);
      const editable = editableTarget(target);
      // Chromium truncates a multi-range Selection to its first range, so
      // an untouched Select All is copied from the captured text instead.
      const cached = cachedSelectAllText();
      if (cached && model[0].id === 'copy' && model[0].enabled) {
        model[0].text = cached;
      }
      menu = doc.createElement('div');
      menu.id = MENU_ID;
      menu.setAttribute('role', 'menu');
      menu.setAttribute('data-sorcar-ctx', '');
      model.forEach(item => {
        const el = doc.createElement('div');
        el.className = 'sorcar-ctx-item' + (item.enabled ? '' : ' disabled');
        el.setAttribute('role', 'menuitem');
        el.dataset.action = item.id;
        const label = doc.createElement('span');
        label.className = 'sorcar-ctx-label';
        label.textContent = item.label;
        el.appendChild(label);
        if (item.key) {
          const key = doc.createElement('span');
          key.className = 'sorcar-ctx-key';
          key.textContent = item.key;
          el.appendChild(key);
        }
        el.addEventListener('mousedown', ev => {
          ev.preventDefault();
        });
        el.addEventListener('click', ev => {
          ev.preventDefault();
          ev.stopPropagation();
          if (!item.enabled) return;
          activate({id: item.id, text: item.text, editable: editable});
        });
        menu.appendChild(el);
      });
      (doc.body || doc.documentElement).appendChild(menu);
      const vw = (win && win.innerWidth) || 0;
      const vh = (win && win.innerHeight) || 0;
      const mw = menu.offsetWidth || 180;
      const mh = menu.offsetHeight || model.length * 24 + 8;
      const px = vw ? Math.min(x, vw - mw - 4) : x;
      const py = vh ? Math.min(y, vh - mh - 4) : y;
      menu.style.left = Math.max(0, px) + 'px';
      menu.style.top = Math.max(0, py) + 'px';
      return menu;
    }

    function onContextMenu(e) {
      if (typeof options.shouldOpen === 'function' && !options.shouldOpen(e)) {
        close();
        return;
      }
      e.preventDefault();
      e.stopPropagation();
      open(e.clientX || 0, e.clientY || 0, e.target);
    }

    function onDocClick(e) {
      if (menu && e.target && menu.contains(e.target)) return;
      close();
    }

    function onKeyDown(e) {
      if (e.key === 'Escape') close();
    }

    function onScrollOrBlur() {
      close();
    }

    doc.addEventListener('contextmenu', onContextMenu);
    doc.addEventListener('click', onDocClick);
    doc.addEventListener('keydown', onKeyDown);
    if (win) {
      win.addEventListener('blur', onScrollOrBlur);
      win.addEventListener('resize', onScrollOrBlur);
    }

    return {
      close: close,
      open: open,
      dispose: function () {
        close();
        doc.removeEventListener('contextmenu', onContextMenu);
        doc.removeEventListener('click', onDocClick);
        doc.removeEventListener('keydown', onKeyDown);
        if (win) {
          win.removeEventListener('blur', onScrollOrBlur);
          win.removeEventListener('resize', onScrollOrBlur);
        }
      },
    };
  }

  /**
   * Return `<script>` markup that installs this menu on whatever document
   * it is parsed into.
   *
   * `main.js` appends the result to the `srcdoc` of the sandboxed iframe
   * that shows an opened .html file, which is the only way to give that
   * opaque origin a working Copy / Select All menu.  The script carries the
   * webview's CSP nonce because `about:srcdoc` inherits the creator's CSP.
   */
  function contentContextMenuBootstrapHtml() {
    const nonce = SCRIPT_NONCE
      ? ' nonce="' + escapeAttr(SCRIPT_NONCE) + '"'
      : '';
    return (
      '\n<script' +
      nonce +
      ' data-sorcar-ctx>' +
      installContentContextMenu.SOURCE +
      '<' +
      '/script>\n'
    );
  }

  // The exact source text shipped into the sandboxed iframe.  It is built
  // from the very functions above via Function.prototype.toString so the
  // iframe and the parent can never drift apart.
  installContentContextMenu.SOURCE =
    '(function(){' +
    'var MENU_ID=' +
    JSON.stringify(MENU_ID) +
    ';' +
    'var MENU_CSS=' +
    JSON.stringify(MENU_CSS) +
    ';' +
    isTextField.toString() +
    isWritable.toString() +
    selectionText.toString() +
    selectedText.toString() +
    copyViaTextarea.toString() +
    copyText.toString() +
    isHiddenSource.toString() +
    inHiddenSource.toString() +
    addRange.toString() +
    selectAll.toString() +
    closestTag.toString() +
    editableTarget.toString() +
    isMac.toString() +
    buildMenuModel.toString() +
    insertIntoContentEditable.toString() +
    pasteInto.toString() +
    installContentContextMenu.toString() +
    // The handle is published so the page (or a test) can tear the menu
    // down again; installing twice would otherwise be undetectable.
    'window.__sorcarContentContextMenu=' +
    'installContentContextMenu(document,{});' +
    '})();';

  const api = {
    MENU_ID: MENU_ID,
    MENU_CSS: MENU_CSS,
    SCRIPT_NONCE: SCRIPT_NONCE,
    buildMenuModel: buildMenuModel,
    selectAll: selectAll,
    copyText: copyText,
    installContentContextMenu: installContentContextMenu,
    contentContextMenuBootstrapHtml: contentContextMenuBootstrapHtml,
  };

  if (root && typeof root === 'object') {
    root.ContentContextMenu = api;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : null);

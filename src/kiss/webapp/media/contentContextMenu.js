// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Right-click ("contextual") menu for read-only content surfaces.
//
// A user who opens an .html file (or a generated report) in a Sorcar
// content tab gets the same Copy / Select All menu as everywhere else in
// the page.  The HTML is rendered in an iframe sandboxed with
// `allow-scripts` only, i.e. an opaque origin the parent document cannot
// reach into, so the menu has to live *inside* that document.  This
// module is therefore written to be usable two ways:
//
//   * required/loaded normally by the chat page (parent document), and
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
    ' .sorcar-ctx-item:hover,#' +
    MENU_ID +
    ' .sorcar-ctx-item:focus{background:#04395e;color:#fff;outline:none}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-item.disabled{opacity:.4;cursor:default;' +
    'pointer-events:none}' +
    '#' +
    MENU_ID +
    ' .sorcar-ctx-key{opacity:.6}';

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
   * the user highlighted text inside a field.  A `number` / `email` input
   * has no such offsets (both null), but the document Selection reads
   * what is selected in it while it holds the focus.
   */
  function selectedText(doc, target) {
    const field = target && isTextField(target) ? target : null;
    const win = doc.defaultView;
    const sel =
      win && typeof win.getSelection === 'function' ? win.getSelection() : null;
    if (field) {
      const start = field.selectionStart;
      const end = field.selectionEnd;
      if (typeof start === 'number' && typeof end === 'number') {
        return end > start ? String(field.value || '').slice(start, end) : '';
      }
      return sel && doc.activeElement === field ? selectionText(sel) : '';
    }
    return sel ? selectionText(sel) : '';
  }

  /**
   * The text of *sel* as rendered: what the user sees highlighted and the
   * native Copy copies.  A Range's `toString()` would serialise the
   * source instead, so a native Select All followed by Copy would hand
   * over the page's hidden elements and inline `<script>` text.
   */
  function selectionText(sel) {
    return String(sel.toString() || '').trim();
  }

  /**
   * True for nodes whose text must never land in a selection: a
   * `<script>` / `<style>`, the menu's own markup, and an element the
   * page does not render (`hidden`, display:none), whose text the native
   * Select All leaves out too.
   */
  function isHiddenSource(el) {
    const tag = String(el.tagName || '').toLowerCase();
    if (tag === 'script' || tag === 'style') return true;
    if (el.hasAttribute('data-sorcar-ctx')) return true;
    const win = el.ownerDocument.defaultView;
    return !!win && win.getComputedStyle(el).display === 'none';
  }

  /**
   * True when *node* has a `<script>`/`<style>`/menu/unrendered ancestor
   * below *root*, or is invisible (`visibility` is inherited, so its
   * parent's computed value is the text's own).
   */
  function inHiddenSource(node, root) {
    let el = node;
    while (el && el !== root) {
      if (el.nodeType === 1 && isHiddenSource(el)) return true;
      el = el.parentNode;
    }
    const parent = node.parentNode;
    const win = node.ownerDocument.defaultView;
    if (!win || !parent || parent.nodeType !== 1) return false;
    const visibility = win.getComputedStyle(parent).visibility;
    return visibility === 'hidden' || visibility === 'collapse';
  }

  function copyViaTextarea(doc, text) {
    const ta = doc.createElement('textarea');
    ta.value = text;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.top = '0';
    ta.style.left = '0';
    ta.style.opacity = '0';
    // Selecting the textarea focuses it and takes the document
    // selection: both go back to what they are now (the control close()
    // handed focus back to, with a text field's own selection) once the
    // command has run, else a copy from a preview's contenteditable
    // drops its highlighted text and leaves focus on the body.
    let focused = doc.activeElement;
    while (focused && focused.shadowRoot && focused.shadowRoot.activeElement) {
      focused = focused.shadowRoot.activeElement;
    }
    const tag = String((focused && focused.tagName) || '').toLowerCase();
    const field = tag === 'input' || tag === 'textarea';
    const win = doc.defaultView;
    const sel = !field && win && win.getSelection ? win.getSelection() : null;
    const ranges = [];
    for (let i = 0; sel && i < sel.rangeCount; i++) {
      ranges.push(sel.getRangeAt(i).cloneRange());
    }
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
    if (focused && focused !== doc.body && focused.isConnected) {
      const start = focused.selectionStart;
      const end = focused.selectionEnd;
      const direction = focused.selectionDirection;
      focused.focus({preventScroll: true});
      if (field) {
        try {
          focused.setSelectionRange(start, end, direction);
        } catch (_e) {
          // number / email inputs have no selection API; theirs is kept
        }
      }
    }
    if (sel && ranges.length) {
      sel.removeAllRanges();
      ranges.forEach(r => sel.addRange(r));
    }
    return ok;
  }

  /**
   * Copy *text* to the clipboard from *doc*.
   *
   * The asynchronous Clipboard API is unavailable inside an opaque-origin
   * sandboxed iframe, so a hidden-textarea `execCommand('copy')` is always
   * used as the fallback and as the sole path when the API rejects — unless
   * *current* (when given) says a newer copy was asked for meanwhile: then
   * the rejected copy resolves false instead of falling back over the
   * newer clipboard contents.
   */
  function copyText(doc, text, current) {
    if (!text) return Promise.resolve(false);
    const win = doc.defaultView;
    const nav = win && win.navigator;
    if (nav && nav.clipboard && typeof nav.clipboard.writeText === 'function') {
      try {
        return Promise.resolve(nav.clipboard.writeText(text)).then(
          () => true,
          () => (!current || current() ? copyViaTextarea(doc, text) : false),
        );
      } catch (_e) {
        return Promise.resolve(copyViaTextarea(doc, text));
      }
    }
    return Promise.resolve(copyViaTextarea(doc, text));
  }

  /**
   * Select every visible text node of *doc*'s body and return the live
   * Ranges installed, one per stretch.
   *
   * `<script>`/`<style>` text and the menu's own markup live in that same
   * body — at any depth — so the document is walked text node by text node
   * and each run of visible nodes becomes one `Range`.  Selecting the body
   * wholesale would instead hand the user the page's inline JavaScript the
   * moment they followed Select All with Copy.
   *
   * Chromium and WebKit keep at most one `Range` per `Selection`, so the
   * caller keeps the returned Ranges and a later Copy reads them with
   * `rangesText` rather than the selection.
   */
  function selectAll(doc, scope) {
    const win = doc.defaultView;
    const body = scope || doc.body || doc.documentElement;
    if (!win || !body || typeof win.getSelection !== 'function') return [];
    const sel = win.getSelection();
    if (!sel) return [];
    sel.removeAllRanges();
    const walker = doc.createTreeWalker(body, 4 /* SHOW_TEXT */, null, false);
    const ranges = [];
    let range = null;
    let node = walker.nextNode();
    while (node) {
      if (inHiddenSource(node, body)) {
        range = addRange(sel, range, ranges);
      } else if (range) {
        range.setEnd(node, node.length);
      } else if (node.data.trim()) {
        // A stretch starts at visible text: whitespace alone first (the
        // newline before an inline <script>) would be the one Range
        // Chromium keeps, whose rendered text is empty.
        range = doc.createRange();
        range.setStart(node, 0);
        range.setEnd(node, node.length);
      }
      node = walker.nextNode();
    }
    addRange(sel, range, ranges);
    return ranges;
  }

  /** Commit *range* to *sel*, recording it in *ranges*, and reset it. */
  function addRange(sel, range, ranges) {
    if (!range || range.collapsed) return null;
    sel.addRange(range);
    ranges.push(range);
    return null;
  }

  /**
   * The text the live *ranges* (Select All's stretches) cover now, one
   * entry per stretch: a Range follows the page's edits of the text it
   * selected, as the selection's highlight does, so an insertion inside
   * it is read and an addition past its end is not.
   */
  function rangesText(doc, ranges) {
    const parts = [];
    for (let i = 0; i < ranges.length; i++) {
      const text = rangeText(doc, ranges[i]).trim();
      if (text) parts.push(text);
    }
    return parts;
  }

  /**
   * The text *range* covers now, as rendered (what the native Copy of
   * the selection gives): a block boundary is a line break, two for a
   * paragraph, a <br> one, table cells are tab-separated and source
   * whitespace collapses as the CSS `white-space` it renders under
   * says (kept whole under `pre` / `pre-wrap` / `break-spaces`, its
   * line breaks alone under `pre-line`).  Read text node by text node
   * so a hidden source inserted into it since Select All stays out, as
   * it would have then, and so does text the page has hidden since, by
   * any ancestor up to the document.
   */
  function rangeText(doc, range) {
    const out = {text: '', breaks: 0, edge: 'line'};
    renderRangeNode(doc, range, range.commonAncestorContainer, out);
    return out.text;
  }

  /**
   * Append *node*'s rendered text within *range* to `out.text`.
   * `out.breaks` counts the line breaks block boundaries owe since the
   * last text (paid before the next); `out.edge` says how `out.text`
   * ends: 'line' at a line start (nothing yet, a line break, a cell's
   * tab), 'space' after a collapsible space, '' otherwise.
   */
  function renderRangeNode(doc, range, node, out) {
    if (node.nodeType === 3 /* TEXT_NODE */) {
      if (!range.intersectsNode(node) || inHiddenSource(node, doc)) return;
      let t = node.data.slice(
        node === range.startContainer ? range.startOffset : 0,
        node === range.endContainer ? range.endOffset : node.length,
      );
      const ws = whiteSpaceOf(doc, node);
      const kept = ws === 'pre' || ws === 'pre-wrap' || ws === 'break-spaces';
      if (ws === 'pre-line') {
        t = t
          .replace(/[\t ]*(?:\r\n|[\n\r])[\t ]*/g, '\n')
          .replace(/[\t ]+/g, ' ');
      } else if (!kept) {
        t = t.replace(/[\t\n\r ]+/g, ' ');
      }
      // Collapsing crosses text nodes: a collapsible space is dropped
      // at a line start and after another collapsible space, and one
      // before a line break goes with it; a preserved space stays.
      if (t[0] === '\n') endLine(out);
      else if (!kept && t[0] === ' ' && out.edge) t = t.slice(1);
      if (!t) return;
      // Line breaks owed by block boundaries are paid before text only,
      // so a stretch never starts or ends with them.
      if (out.text) {
        for (let i = 0; i < out.breaks; i++) out.text += '\n';
      }
      out.breaks = 0;
      out.text += t;
      const last = t[t.length - 1];
      out.edge = last === '\n' ? 'line' : last === ' ' && !kept ? 'space' : '';
      return;
    }
    const win = doc.defaultView;
    const isEl = node.nodeType === 1;
    if (isEl && (!range.intersectsNode(node) || isHiddenSource(node))) return;
    const tag = isEl ? String(node.tagName || '').toLowerCase() : '';
    if (tag === 'br') {
      if (out.text) {
        endLine(out);
        out.text += '\n';
      }
      return;
    }
    const display = isEl && win ? win.getComputedStyle(node).display : '';
    const block =
      display !== '' && !/^inline/.test(display) && display !== 'contents';
    if (display === 'table-cell' && out.text && out.breaks === 0) {
      endLine(out);
      out.text += '\t';
    } else if (block) {
      endLine(out);
      out.breaks = Math.max(out.breaks, tag === 'p' ? 2 : 1);
    }
    for (let i = 0; i < node.childNodes.length; i++) {
      renderRangeNode(doc, range, node.childNodes[i], out);
    }
    if (block && display !== 'table-cell') {
      endLine(out);
      out.breaks = Math.max(out.breaks, tag === 'p' ? 2 : 1);
    }
  }

  /** A line of `out.text` ends here: its collapsible space, if any, goes. */
  function endLine(out) {
    if (out.edge === 'space') out.text = out.text.slice(0, -1);
    out.edge = 'line';
  }

  /**
   * The CSS `white-space` the text node *node* renders under: the
   * computed value of the nearest ancestor that has one (a browser
   * answers at the parent, the value being inherited; jsdom reports
   * the declaring element only), 'normal' when none does.
   */
  function whiteSpaceOf(doc, node) {
    const win = doc.defaultView;
    let el = node.parentNode;
    while (win && el && el.nodeType === 1) {
      const ws = win.getComputedStyle(el).whiteSpace;
      if (ws) return ws;
      el = el.parentNode;
    }
    return 'normal';
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
      if (el.isContentEditable) {
        // The editing host, not a formatting child that only inherits
        // the editability: Select All and Paste act on the whole field.
        while (el.parentNode && el.parentNode.isContentEditable) {
          el = el.parentNode;
        }
        return el;
      }
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
   * Build the ordered menu model for a right-click on *target* in *doc*:
   * one item per menu row, each carrying the editable element under the
   * click (`editable`, null outside a field) its action applies to.
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
      // An opaque origin (the sandboxed preview iframe) may not read the
      // clipboard: the row is shown disabled, its shortcut still pastes.
      const loc = doc.location;
      items.push({
        id: 'paste',
        label: 'Paste',
        key: mod + 'V',
        enabled: !loc || String(loc.origin) !== 'null',
      });
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
    items.forEach(item => {
      item.editable = editable;
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
        // The read was asynchronous (the browser may have asked the user
        // first): a field the page made read-only, disabled or removed
        // meanwhile is left alone, as the native paste would leave it.
        if (!isWritable(el) || !doc.contains(el)) return false;
        const value = String(text || '');
        // `number` / `email` inputs expose no selection API (selectionStart
        // is null, setRangeText throws InvalidStateError) but they do keep
        // a native selection, which execCommand('insertText') honours; it
        // fires the field's own `input` event.  Appending is the last
        // resort when the command is unavailable (jsdom) or refused.
        let fired = false;
        if (el.isContentEditable) insertIntoContentEditable(doc, el, value);
        else if (
          typeof el.setRangeText === 'function' &&
          el.selectionStart !== null
        ) {
          el.setRangeText(value, el.selectionStart, el.selectionEnd, 'end');
        } else {
          el.focus();
          fired =
            typeof doc.execCommand === 'function' &&
            doc.execCommand('insertText', false, value);
          if (!fired) el.value = String(el.value || '') + value;
        }
        if (!fired) el.dispatchEvent(new win.Event('input', {bubbles: true}));
        return true;
      },
      () => false,
    );
  }

  /**
   * The keyboard model of every menu (this context menu, the sidebar
   * trees' treeContextMenu.js, and main.js's tab menu and content menu
   * bar): Up / Down move between the enabled *items* (wrapping), Home /
   * End jump to the ends, Enter / Space run the focused item, Escape
   * closes through *close* and Tab leaves the menu through *onLeave*
   * with the browser's own focus move.  A caller that handles Escape or
   * Tab elsewhere passes no *close* / *onLeave*, and the key passes
   * through untouched.
   *
   * @param {KeyboardEvent} e The keydown event.
   * @param {Array<HTMLElement>} items The menu's enabled items.
   * @param {?Function} close Closes the menu on Escape.
   * @param {?Function} [onLeave] Closes the menu on Tab.
   * @returns {boolean} Whether the key belonged to the model.
   */
  function menuKeyNav(e, items, close, onLeave) {
    if (e.key === 'Escape' && close) {
      e.preventDefault();
      close();
      return true;
    }
    if (!items.length) return false;
    if (e.key === 'Tab' && onLeave) {
      onLeave();
      return true;
    }
    const idx = items.indexOf(items[0].ownerDocument.activeElement);
    let next = -1;
    if (e.key === 'ArrowDown') next = idx < 0 ? 0 : (idx + 1) % items.length;
    else if (e.key === 'ArrowUp') next = idx <= 0 ? items.length - 1 : idx - 1;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = items.length - 1;
    if (next >= 0) {
      e.preventDefault();
      // The menus are position: fixed and clamped into the viewport, so
      // this scrolls nothing but a dropdown's own overflow (the content
      // menu bar's, which brings the focused row into view).
      items[next].focus();
      return true;
    }
    if ((e.key === 'Enter' || e.key === ' ') && idx >= 0) {
      e.preventDefault();
      items[idx].click();
      return true;
    }
    return false;
  }

  /**
   * Place the menu element *el* at (x, y), kept inside *win*'s viewport.
   *
   * @param {Window} win The menu's window (its viewport size).
   * @param {HTMLElement} el The open menu.
   * @param {number} x Pointer x (client coordinates).
   * @param {number} y Pointer y (client coordinates).
   * @param {number} fallbackW Width of a menu not laid out (offsetWidth 0).
   * @param {number} fallbackH Height of a menu not laid out.
   */
  function clampMenu(win, el, x, y, fallbackW, fallbackH) {
    const vw = (win && win.innerWidth) || 0;
    const vh = (win && win.innerHeight) || 0;
    const mw = el.offsetWidth || fallbackW;
    const mh = el.offsetHeight || fallbackH;
    const px = vw ? Math.min(x, vw - mw - 4) : x;
    const py = vh ? Math.min(y, vh - mh - 4) : y;
    el.style.left = Math.max(0, px) + 'px';
    el.style.top = Math.max(0, py) + 'px';
  }

  /**
   * Install a Copy / Select All context menu on *doc*.
   *
   * Returns a handle with `close()` so a caller can dismiss an open menu
   * (main.js closes it when a tree menu opens over it).  The listeners live
   * as long as *doc* does.
   */
  function installContentContextMenu(doc, opts) {
    const options = opts || {};
    const win = doc.defaultView;
    let menu = null;
    // The element focused before the menu opened; close() hands focus back
    // to it so a keyboard user does not lose their place.
    let opener = null;
    // Chromium keeps a single Range per Selection, so a multi-range Select
    // All cannot be re-read from the selection: the live Ranges it
    // installed (every stretch) are remembered and their text read
    // afresh at Copy time, since the page (a preview's script, typing
    // into a contenteditable) may have changed it meanwhile, and a Range
    // is live: it follows such edits as the highlight does, so a text
    // cached at Select All time would be stale, and the scope re-read
    // whole would take in text added past the selection's end.  The
    // Range the selection itself kept is remembered alongside: the
    // cache is only honoured while the live selection is still exactly
    // that Range, so any selection the user makes afterwards — narrower,
    // wider or elsewhere — discards it.  `selectionchange` is dispatched
    // asynchronously, so it cannot be relied on for this.
    let selectAllRanges = [];
    let selectAllRange = null;
    // The menu copy asked for most recently: an earlier one whose async
    // write rejects after it must not fall back over it (see copyText).
    let copySeq = 0;

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
      return same ? rangesText(doc, selectAllRanges).join('\n') : '';
    }

    // True while focus is still on the menu (or nowhere in particular):
    // Escape, Enter / Space and the menu's own clicks leave it there.  A
    // pointer click on another control has already moved focus to that
    // control by the time the document click handler closes the menu,
    // and that choice is the user's to keep.
    function focusStillOnMenu() {
      const active = doc.activeElement;
      return (
        !active ||
        active === doc.body ||
        (menu !== null && menu.contains(active))
      );
    }

    function close() {
      const restore = focusStillOnMenu();
      if (menu && menu.parentNode) menu.parentNode.removeChild(menu);
      menu = null;
      const prev = opener;
      opener = null;
      if (
        restore &&
        prev &&
        prev !== doc.body &&
        typeof prev.focus === 'function' &&
        doc.contains(prev)
      ) {
        prev.focus({preventScroll: true});
      }
    }

    function focusableItems() {
      return menu
        ? Array.from(menu.querySelectorAll('.sorcar-ctx-item:not(.disabled)'))
        : [];
    }

    function activate(item) {
      close();
      if (item.id === 'select-all') {
        // Over a text field Select All selects that field's contents,
        // exactly like the native menu does, and a following Copy reads
        // that selection.  select() works on every input type;
        // setSelectionRange throws on `number` / `email` inputs.
        // Elsewhere it selects the visible text as one Range per stretch
        // between hidden sources.  Chromium keeps only the first Range, so
        // the live Ranges are kept with that one left in place, and the
        // next Copy reads them only while the live selection is still
        // exactly that Range (cachedSelectAllText).
        if (item.editable && isTextField(item.editable)) {
          item.editable.focus();
          item.editable.select();
        } else {
          selectAllRanges = selectAll(doc, item.editable);
          const sel = win && win.getSelection && win.getSelection();
          selectAllRange = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
        }
        return;
      }
      if (item.id === 'paste') {
        pasteInto(doc, item.editable);
        return;
      }
      // The parent document's copy controls all go through PanelCopy,
      // which retires an older pending copy of any of them when a newer
      // one is asked for: a menu copy there is one of them.  The
      // sandboxed preview iframe has no PanelCopy and keeps its own.
      const panelCopy = win && win.PanelCopy;
      if (panelCopy && typeof panelCopy.copyText === 'function') {
        panelCopy.copyText(item.text || '', doc);
        return;
      }
      const seq = ++copySeq;
      copyText(doc, item.text || '', () => seq === copySeq);
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
      // Chromium truncates a multi-range Selection to its first range, so
      // an untouched Select All is copied from the captured text instead
      // (buildMenuModel always puts the Copy item first).
      const cached = cachedSelectAllText();
      if (cached && model[0].enabled) {
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
        el.tabIndex = item.enabled ? 0 : -1;
        if (!item.enabled) el.setAttribute('aria-disabled', 'true');
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
          activate({id: item.id, text: item.text, editable: item.editable});
        });
        menu.appendChild(el);
      });
      (doc.body || doc.documentElement).appendChild(menu);
      clampMenu(win, menu, x, y, 180, model.length * 24 + 8);
      opener = doc.activeElement;
      const first = focusableItems()[0];
      if (first) first.focus({preventScroll: true});
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
      if (menu) menuKeyNav(e, focusableItems(), close);
    }

    doc.addEventListener('contextmenu', onContextMenu);
    doc.addEventListener('click', onDocClick);
    doc.addEventListener('keydown', onKeyDown);
    if (win) {
      win.addEventListener('blur', close);
      win.addEventListener('resize', close);
    }

    return {close: close};
  }

  /**
   * Return `<script>` markup that installs this menu on whatever document
   * it is parsed into.
   *
   * `main.js` appends the result to the `srcdoc` of the sandboxed iframe
   * that shows an opened .html file, which is the only way to give that
   * opaque origin a working Copy / Select All menu.
   */
  function contentContextMenuBootstrapHtml() {
    return (
      '\n<script data-sorcar-ctx>' +
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
    rangesText.toString() +
    rangeText.toString() +
    renderRangeNode.toString() +
    endLine.toString() +
    whiteSpaceOf.toString() +
    closestTag.toString() +
    editableTarget.toString() +
    isMac.toString() +
    buildMenuModel.toString() +
    insertIntoContentEditable.toString() +
    pasteInto.toString() +
    menuKeyNav.toString() +
    clampMenu.toString() +
    installContentContextMenu.toString() +
    'installContentContextMenu(document,{});' +
    '})();';

  const api = {
    installContentContextMenu: installContentContextMenu,
    contentContextMenuBootstrapHtml: contentContextMenuBootstrapHtml,
    menuKeyNav: menuKeyNav,
    clampMenu: clampMenu,
  };

  if (root && typeof root === 'object') {
    root.ContentContextMenu = api;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : null);

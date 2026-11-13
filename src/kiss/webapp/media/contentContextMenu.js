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
//   * loaded by the chat page (parent document), and
//   * serialised via `contentContextMenuBootstrapHtml()` into the iframe
//     `srcdoc` so the sandboxed page installs the very same menu on itself.

'use strict';

(function () {
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
    // The state, not the attribute: an unknown `type` is a text field.
    const type = el.type;
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
    // A field first: one in a contenteditable host is not made writable.
    if (isTextField(el)) return !el.disabled && !el.readOnly;
    return !!el.isContentEditable;
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
   *
   * The Selection's own `toString()` gives the text as rendered, selected
   * whitespace (a code line's indentation) included: what the native Copy
   * copies.  A Range's would serialise the source instead, the page's
   * hidden elements and inline `<script>` text included.
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
      if (doc.activeElement !== field) return '';
    }
    return sel ? String(sel.toString() || '') : '';
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
   * below *root*, or is invisible.
   */
  function inHiddenSource(node, root) {
    let el = node;
    while (el && el !== root) {
      if (el.nodeType === 1 && isHiddenSource(el)) return true;
      el = el.parentNode;
    }
    return invisible(node);
  }

  /**
   * True when the text node *node* is invisible: `visibility` is
   * inherited, so its parent's computed value is the text's own.
   */
  function invisible(node) {
    const parent = node.parentNode;
    const win = node.ownerDocument.defaultView;
    if (!win || !parent || parent.nodeType !== 1) return false;
    const visibility = win.getComputedStyle(parent).visibility;
    return visibility === 'hidden' || visibility === 'collapse';
  }

  /** The element focused in *doc*, one inside an open shadow root included. */
  function focusedElement(doc) {
    let focused = doc.activeElement;
    while (focused && focused.shadowRoot && focused.shadowRoot.activeElement) {
      focused = focused.shadowRoot.activeElement;
    }
    return focused;
  }

  /** True while the page renders *el*: no ancestor-or-self is display:none. */
  function rendered(el) {
    const win = el.ownerDocument.defaultView;
    let node = el;
    while (win && node && node.nodeType === 1) {
      if (win.getComputedStyle(node).display === 'none') return false;
      node = node.parentNode;
    }
    return true;
  }

  /**
   * Copy *text* to the clipboard through a hidden textarea and
   * `execCommand('copy')`, the way that works where the async clipboard
   * is missing or refuses, and hand the focus and the selection back.
   *
   * @param {Document} doc The document to copy from.
   * @param {string} text What to place on the clipboard.
   * @returns {boolean} Whether the copy command reported success.
   */
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
    const focused = focusedElement(doc);
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
      // a refused copy command: ok stays false
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
   * Select the visible text of *scope* (*doc*'s body by default) as one
   * live Range, from its first visible text node to its last.
   *
   * One selection, as the native Select All installs: the keyboard Copy
   * (Ctrl/Cmd+C) and the menu Copy both copy what it covers as rendered
   * (the document Selection's text), leaving out the `<script>`/`<style>`
   * text, the menu's own markup and the elements the page does not
   * render, as the user never sees them.  It starts and ends at visible
   * text: a Range beginning in the whitespace before an inline `<script>`
   * would select nothing rendered at its edge.  An end inside an editing
   * host the scope is not part of goes before / after the host instead:
   * Chromium cuts a selection with one end in a host and the other
   * outside at the host, as the native Select All of the page takes the
   * host in whole.
   */
  function selectAll(doc, scope) {
    const win = doc.defaultView;
    const body = scope || doc.body || doc.documentElement;
    if (!win || !body || typeof win.getSelection !== 'function') return;
    const sel = win.getSelection();
    if (!sel) return;
    sel.removeAllRanges();
    const walker = doc.createTreeWalker(body, 4 /* SHOW_TEXT */, null, false);
    let first = null;
    let last = null;
    let node = walker.nextNode();
    while (node) {
      if (node.data.trim() && !inHiddenSource(node, body)) {
        if (!first) first = node;
        last = node;
      }
      node = walker.nextNode();
    }
    if (!first) return;
    const range = doc.createRange();
    const from = editableTarget(first);
    if (from && from !== body && !from.contains(body)) {
      range.setStartBefore(from);
    } else {
      range.setStart(first, 0);
    }
    const to = editableTarget(last);
    if (to && to !== body && !to.contains(body)) range.setEndAfter(to);
    else range.setEnd(last, last.length);
    sel.addRange(range);
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
      if (isTextField(el)) return el;
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

  /**
   * What a paste into *el* acts on: the field's text and its selection
   * (a contenteditable's as the live Range the document selection keeps
   * in it, cloned).
   */
  function editState(doc, el) {
    if (isTextField(el)) {
      return {
        text: String(el.value || ''),
        start: el.selectionStart,
        end: el.selectionEnd,
        // All a `number` / `email` input (null offsets) tells of its
        // selection: the text selected in it, while it holds the focus.
        selected: doc.activeElement === el ? selectedText(doc, el) : null,
        range: null,
      };
    }
    const win = doc.defaultView;
    const sel = win && win.getSelection && win.getSelection();
    const live = sel && sel.rangeCount ? sel.getRangeAt(0) : null;
    const inside = live && el.contains(live.commonAncestorContainer);
    return {
      text: el.textContent,
      start: null,
      end: null,
      selected: null,
      range: inside ? live.cloneRange() : null,
    };
  }

  /** True when the edit states *a* and *b* (editState) are the same. */
  function sameEditState(a, b) {
    if (a.text !== b.text || a.start !== b.start || a.end !== b.end) {
      return false;
    }
    const read = a.selected !== null && b.selected !== null;
    if (read && a.selected !== b.selected) return false;
    if (!a.range || !b.range) return !a.range && !b.range;
    return (
      a.range.compareBoundaryPoints(a.range.START_TO_START, b.range) === 0 &&
      a.range.compareBoundaryPoints(a.range.END_TO_END, b.range) === 0
    );
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
    const asked = editState(doc, el);
    // A `number` / `email` input does not tell where its selection is
    // (the offsets are null): for one, a key or a pointer the user
    // presses while the read is pending counts as having moved it.
    // (Not `selectionchange`: it is queued, and Firefox sends one for the
    // focus the closing menu handed back to the field after this.)
    let pressed = false;
    function unwatch() {
      doc.removeEventListener('keydown', onPress, true);
      doc.removeEventListener('pointerdown', onPress, true);
    }
    function onPress() {
      pressed = true;
      unwatch();
    }
    if (asked.start === null && isTextField(el)) {
      doc.addEventListener('keydown', onPress, true);
      doc.addEventListener('pointerdown', onPress, true);
      pending.then(unwatch, unwatch);
    }
    return pending.then(
      text => {
        // The read was asynchronous (the browser may have asked the user
        // first): a field the page made read-only, disabled or removed
        // meanwhile is left alone, as the native paste would leave it.
        // So is one whose text or selection moved on meanwhile: the
        // paste was asked for at the selection of that moment, which a
        // selection the user made since is not, and the old offsets are
        // nowhere over text the field has changed.
        if (!isWritable(el) || !doc.contains(el)) return false;
        if (pressed || !sameEditState(asked, editState(doc, el))) return false;
        // Nor is one the user has moved on from: a control focused since
        // (another content tab's chat composer) is where the insertText
        // command would write once focus() on a field the page hides
        // (the tab switched away from: display:none, still mounted) has
        // failed silently.  The paste is dropped then, never moved to
        // that control nor the focus taken back from it.
        const focused = focusedElement(doc);
        if (
          focused &&
          focused !== doc.body &&
          focused !== el &&
          !el.contains(focused)
        ) {
          return false;
        }
        if (!rendered(el)) return false;
        const value = String(text || '');
        // The browser's own insertText command first: it acts on the
        // focused field, replaces the native selection (the only one a
        // `number` / `email` input has: selectionStart is null,
        // setRangeText throws InvalidStateError), fires the field's
        // `input` event itself and is a step of the field's undo
        // history, which text written in directly is not.  Without the
        // command (jsdom) or when it is refused, the text is written in
        // at the selection, appended as the last resort.
        if (typeof doc.execCommand === 'function') {
          el.focus();
          // A field that cannot take the focus would leave the command
          // to whatever has it.
          if (focusedElement(doc) !== el) return false;
          if (doc.execCommand('insertText', false, value)) return true;
        }
        if (el.isContentEditable) insertIntoContentEditable(doc, el, value);
        else if (
          typeof el.setRangeText === 'function' &&
          el.selectionStart !== null
        ) {
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
   * `opts.shouldOpen(event)` says whether a contextmenu event opens it;
   * `opts.scope(target)` names the element whose text Select All selects
   * for a right-click on *target* outside a field (the body when absent
   * or when it returns nothing).
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
    // The menu copy asked for most recently: an earlier one whose async
    // write rejects after it must not fall back over it (see copyText).
    let copySeq = 0;

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
        // Elsewhere it selects the visible text of the editing host, of
        // the scope the caller names for the clicked element, or of the
        // body, as one live Range the next Copy reads as the browser
        // renders it then (the document Selection's text).
        if (item.editable && isTextField(item.editable)) {
          item.editable.focus();
          item.editable.select();
        } else {
          selectAll(doc, item.editable || item.scope);
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
      const scope =
        (typeof options.scope === 'function' && options.scope(target)) || null;
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
          activate({
            id: item.id,
            text: item.text,
            editable: item.editable,
            scope: scope,
          });
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

    // Tab leaves the menu by the browser's own focus move (there is no
    // focus trap), so it closes then, else it would go on taking the
    // Arrow / Home / End / Enter keys of the control the focus lands on.
    function onKeyDown(e) {
      if (menu) menuKeyNav(e, focusableItems(), close, close);
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
    selectedText.toString() +
    focusedElement.toString() +
    rendered.toString() +
    copyViaTextarea.toString() +
    copyText.toString() +
    isHiddenSource.toString() +
    inHiddenSource.toString() +
    invisible.toString() +
    selectAll.toString() +
    closestTag.toString() +
    editableTarget.toString() +
    isMac.toString() +
    buildMenuModel.toString() +
    insertIntoContentEditable.toString() +
    editState.toString() +
    sameEditState.toString() +
    pasteInto.toString() +
    menuKeyNav.toString() +
    clampMenu.toString() +
    installContentContextMenu.toString() +
    'installContentContextMenu(document,{});' +
    '})();';

  const api = {
    installContentContextMenu: installContentContextMenu,
    contentContextMenuBootstrapHtml: contentContextMenuBootstrapHtml,
    copyViaTextarea: copyViaTextarea,
    menuKeyNav: menuKeyNav,
    clampMenu: clampMenu,
  };

  window.ContentContextMenu = api;
})();

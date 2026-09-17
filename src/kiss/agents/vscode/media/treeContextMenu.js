// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A VS Code-style context menu for the sidebar trees (Explorer rows,
// Source Control commits and files).
//
// One menu element (#sidebar-context-menu) is shared by every tree; a
// caller hands `show()` the items of the menu it wants and the menu
// runs the chosen item's action.  Items are plain objects:
//
//   {label: 'Rename...', key: 'F2', enabled: true, run: () => {...}}
//   {separator: true}
//
// The menu behaves like VS Code's: it opens at the pointer (kept inside
// the viewport), Up / Down move between enabled items, Enter / Space run
// the focused one, Escape or a click anywhere else closes it, and a
// disabled item is shown greyed out and cannot be run.

/* global module */
'use strict';

(function (root) {
  const MENU_ID = 'sidebar-context-menu';

  let menu = null;
  let onClosed = null;

  function menuElement(doc) {
    let el = doc.getElementById(MENU_ID);
    if (el) return el;
    el = doc.createElement('div');
    el.id = MENU_ID;
    el.className = 'tree-ctx-menu';
    el.setAttribute('role', 'menu');
    el.hidden = true;
    doc.body.appendChild(el);
    return el;
  }

  /** Close the open menu (a no-op when none is open). */
  function close() {
    if (!menu) return;
    const el = menu;
    menu = null;
    el.hidden = true;
    el.textContent = '';
    const cb = onClosed;
    onClosed = null;
    if (typeof cb === 'function') cb();
  }

  function focusableItems(el) {
    return Array.from(el.querySelectorAll('.tree-ctx-item:not(.disabled)'));
  }

  function onKeyDown(e) {
    if (!menu) return;
    if (e.key === 'Escape') {
      e.preventDefault();
      close();
      return;
    }
    const items = focusableItems(menu);
    if (!items.length) return;
    const idx = items.indexOf(menu.ownerDocument.activeElement);
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      let next;
      if (e.key === 'ArrowDown') next = idx < 0 ? 0 : (idx + 1) % items.length;
      else next = idx <= 0 ? items.length - 1 : idx - 1;
      items[next].focus();
      return;
    }
    if (e.key === 'Home' || e.key === 'End') {
      e.preventDefault();
      items[e.key === 'Home' ? 0 : items.length - 1].focus();
      return;
    }
    if ((e.key === 'Enter' || e.key === ' ') && idx >= 0) {
      e.preventDefault();
      items[idx].click();
    }
  }

  function onDocPointerDown(e) {
    if (menu && e.target && menu.contains(e.target)) return;
    close();
  }

  function onWindowChange() {
    close();
  }

  let installedOn = null;

  function install(doc) {
    if (installedOn === doc) return;
    installedOn = doc;
    doc.addEventListener('keydown', onKeyDown, true);
    doc.addEventListener('mousedown', onDocPointerDown, true);
    doc.addEventListener('contextmenu', onDocPointerDown, true);
    const win = doc.defaultView;
    if (win) {
      win.addEventListener('blur', onWindowChange);
      win.addEventListener('resize', onWindowChange);
    }
  }

  /**
   * Open the menu at viewport position (x, y) with *items*.
   *
   * @param {Document} doc The document to show the menu in.
   * @param {number} x Pointer x (client coordinates).
   * @param {number} y Pointer y (client coordinates).
   * @param {Array<object>} items Menu items (see the file comment).
   * @param {Function} [closed] Called once the menu has closed.
   * @returns {HTMLElement} The menu element.
   */
  function show(doc, x, y, items, closed) {
    close();
    install(doc);
    const el = menuElement(doc);
    menu = el;
    onClosed = typeof closed === 'function' ? closed : null;
    let lastWasSeparator = true;
    items.forEach(item => {
      if (!item) return;
      if (item.separator) {
        // No leading, trailing or doubled separators, like VS Code.
        if (lastWasSeparator) return;
        const sep = doc.createElement('div');
        sep.className = 'tree-ctx-sep';
        sep.setAttribute('role', 'separator');
        el.appendChild(sep);
        lastWasSeparator = true;
        return;
      }
      lastWasSeparator = false;
      const row = doc.createElement('div');
      const enabled = item.enabled !== false;
      row.className = 'tree-ctx-item' + (enabled ? '' : ' disabled');
      row.setAttribute('role', 'menuitem');
      row.tabIndex = enabled ? 0 : -1;
      if (!enabled) row.setAttribute('aria-disabled', 'true');
      if (item.id) row.dataset.action = item.id;
      const label = doc.createElement('span');
      label.className = 'tree-ctx-label';
      label.textContent = item.label;
      row.appendChild(label);
      if (item.key) {
        const key = doc.createElement('span');
        key.className = 'tree-ctx-key';
        key.textContent = item.key;
        row.appendChild(key);
      }
      row.addEventListener('click', ev => {
        ev.preventDefault();
        ev.stopPropagation();
        if (!enabled) return;
        close();
        if (typeof item.run === 'function') item.run();
      });
      el.appendChild(row);
    });
    const last = el.lastElementChild;
    if (last && last.classList.contains('tree-ctx-sep')) last.remove();
    el.hidden = false;
    const win = doc.defaultView;
    const vw = (win && win.innerWidth) || 0;
    const vh = (win && win.innerHeight) || 0;
    const mw = el.offsetWidth || 200;
    const mh = el.offsetHeight || 24 * items.length;
    const px = vw ? Math.min(x, vw - mw - 4) : x;
    const py = vh ? Math.min(y, vh - mh - 4) : y;
    el.style.left = Math.max(0, px) + 'px';
    el.style.top = Math.max(0, py) + 'px';
    const first = focusableItems(el)[0];
    if (first) first.focus({preventScroll: true});
    return el;
  }

  function isOpen() {
    return !!menu;
  }

  const api = {MENU_ID: MENU_ID, show: show, close: close, isOpen: isOpen};

  if (root && typeof root === 'object') {
    root.TreeContextMenu = api;
  }
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : this);

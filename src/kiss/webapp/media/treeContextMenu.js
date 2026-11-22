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
//   {label: 'Sort', submenu: [{label: 'Name', checked: true, run: ...}]}
//
// The menu behaves like VS Code's: it opens at the pointer (kept inside
// the viewport), Up / Down move between enabled items, Enter / Space run
// the focused one, Escape, Tab (the focus moves on) or a click anywhere
// else closes it, and a disabled item is shown greyed out and cannot be
// run.  The keyboard model and the viewport clamp are
// contentContextMenu.js's (`menuKeyNav`, `clampMenu`), so that file must
// be loaded first.
//
// An item with `submenu` (a list of items, which open no submenu of
// their own) ends with a chevron and opens that list as a second menu
// (#sidebar-context-submenu) beside it: under the mouse pointer, and on
// a click or tap, Enter, Space or Right, which also move the focus to
// its first item.  Left or Escape closes the submenu and hands the focus
// back to its item; running one of its items closes the whole menu.  An
// item with a boolean `checked` is a radio item, drawn with a check mark
// when it is the one that is on.

'use strict';

(function (root) {
  const MENU_ID = 'sidebar-context-menu';
  const SUBMENU_ID = 'sidebar-context-submenu';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  // Icon paths on a 16-unit grid, the ones main.js draws in its trees.
  const ICON_CHECK = 'M3 8.5l3.2 3.2L13 4.5';
  const ICON_CHEVRON = 'M6 4l4 4-4 4';

  let menu = null;
  let onClosed = null;
  // The open submenu and the item it belongs to (null when none is open).
  let submenu = null;
  let submenuOwner = null;

  function menuElement(doc, id) {
    let el = doc.getElementById(id);
    if (el) return el;
    el = doc.createElement('div');
    el.id = id;
    el.className = 'tree-ctx-menu';
    el.setAttribute('role', 'menu');
    el.hidden = true;
    doc.body.appendChild(el);
    return el;
  }

  /**
   * Close the open submenu (a no-op when none is open).
   *
   * @param {boolean} refocus Hand the focus back to the submenu's item.
   */
  function closeSubmenu(refocus) {
    if (!submenu) return;
    const owner = submenuOwner;
    submenu.hidden = true;
    submenu.textContent = '';
    submenu = null;
    submenuOwner = null;
    owner.setAttribute('aria-expanded', 'false');
    if (refocus) owner.focus({preventScroll: true});
  }

  /** Close the open menu (a no-op when none is open). */
  function close() {
    if (!menu) return;
    closeSubmenu(false);
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
    const nav = root.ContentContextMenu.menuKeyNav;
    const active = menu.ownerDocument.activeElement;
    if (submenu && (e.key === 'ArrowLeft' || e.key === 'Escape')) {
      // The submenu alone closes; its item takes the focus back.
      e.preventDefault();
      closeSubmenu(true);
    } else if (submenu && submenu.contains(active)) {
      nav(e, focusableItems(submenu), null, close);
    } else if (
      e.key === 'ArrowRight' &&
      menu.contains(active) &&
      active.getAttribute('aria-haspopup')
    ) {
      // Right opens the focused item's submenu, as Enter and Space do
      // (menuKeyNav turns those two into this click).
      e.preventDefault();
      active.click();
    } else {
      nav(e, focusableItems(menu), close, close);
    }
  }

  function onDocPointerDown(e) {
    const target = e.target;
    if (menu && target && menu.contains(target)) return;
    if (submenu && target && submenu.contains(target)) return;
    close();
  }

  // The focus moving to another item of the menu (Up / Down while the
  // pointer holds a submenu open) closes the submenu.
  function onFocusIn(e) {
    if (submenu && menu.contains(e.target) && e.target !== submenuOwner)
      closeSubmenu(false);
  }

  let installedOn = null;

  function install(doc) {
    if (installedOn === doc) return;
    installedOn = doc;
    doc.addEventListener('keydown', onKeyDown, true);
    doc.addEventListener('mousedown', onDocPointerDown, true);
    doc.addEventListener('contextmenu', onDocPointerDown, true);
    doc.addEventListener('focusin', onFocusIn, true);
    const win = doc.defaultView;
    if (win) {
      win.addEventListener('blur', close);
      win.addEventListener('resize', close);
    }
  }

  /** An icon of the one path *d*; main.css sizes and strokes it. */
  function icon(doc, className, d) {
    const svg = doc.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('class', className);
    svg.setAttribute('viewBox', '0 0 16 16');
    svg.setAttribute('aria-hidden', 'true');
    const path = doc.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', d);
    svg.appendChild(path);
    return svg;
  }

  /**
   * Build the row of *item*.  *nested* says the row goes into the
   * submenu, whose items open no submenu of their own.
   */
  function itemRow(doc, item, nested) {
    const row = doc.createElement('div');
    const enabled = item.enabled !== false;
    const radio = typeof item.checked === 'boolean';
    const opensSubmenu = !nested && Array.isArray(item.submenu);
    row.className = 'tree-ctx-item' + (enabled ? '' : ' disabled');
    row.setAttribute('role', radio ? 'menuitemradio' : 'menuitem');
    row.tabIndex = enabled ? 0 : -1;
    if (!enabled) row.setAttribute('aria-disabled', 'true');
    if (item.id) row.dataset.action = item.id;
    if (radio) {
      row.setAttribute('aria-checked', String(item.checked));
      if (item.checked)
        row.appendChild(icon(doc, 'tree-ctx-check', ICON_CHECK));
    }
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
    if (opensSubmenu) {
      row.setAttribute('aria-haspopup', 'menu');
      row.setAttribute('aria-expanded', 'false');
      row.appendChild(icon(doc, 'tree-ctx-chevron', ICON_CHEVRON));
      // Only a mouse opens the submenu by resting on its item.  A tap
      // opens it through its click: opened under the finger first, the
      // submenu would take the rest of the tap as a pick of the item
      // that landed there.
      if (enabled)
        row.addEventListener('pointerenter', ev => {
          if (ev.pointerType === 'mouse') openSubmenu(row, item, false);
        });
    }
    row.addEventListener('click', ev => {
      ev.preventDefault();
      ev.stopPropagation();
      if (!enabled) return;
      if (opensSubmenu) {
        openSubmenu(row, item, true);
        return;
      }
      close();
      if (typeof item.run === 'function') item.run();
    });
    return row;
  }

  /**
   * Fill the empty menu element *el* with the rows of *items* and show
   * it.  *nested* says *el* is the submenu.
   */
  function fillMenu(doc, el, items, nested) {
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
      el.appendChild(itemRow(doc, item, nested));
    });
    const last = el.lastElementChild;
    if (last && last.classList.contains('tree-ctx-sep')) last.remove();
    el.hidden = false;
  }

  /**
   * Place the open submenu beside its item *owner*: against the item's
   * right edge; against its left edge when the viewport ends before the
   * submenu does and that side has the room; inside the viewport either
   * way (clampMenu).  Its first item is level with *owner*: both menus
   * have the same border and padding.
   *
   * @param {HTMLElement} owner The item the submenu belongs to.
   * @param {number} count The submenu's items (its height when the
   *   page is not laid out).
   */
  function placeSubmenu(owner, count) {
    const win = owner.ownerDocument.defaultView;
    const at = owner.getBoundingClientRect();
    const width = submenu.offsetWidth || 200;
    const flip = at.right + width > win.innerWidth - 4 && at.left >= width;
    const inset = menu.clientTop + menu.firstElementChild.offsetTop;
    root.ContentContextMenu.clampMenu(
      win,
      submenu,
      flip ? at.left - width : at.right,
      at.top - inset,
      200,
      24 * count,
    );
  }

  /**
   * Open the submenu of *item* beside its row *owner*; one that is open
   * already stays as it is.
   *
   * @param {HTMLElement} owner The row of *item*.
   * @param {object} item The item carrying `submenu`.
   * @param {boolean} focusFirst Move the focus to the submenu's first
   *   enabled item (a pointer merely over *owner* leaves the focus).
   */
  function openSubmenu(owner, item, focusFirst) {
    const doc = owner.ownerDocument;
    if (submenuOwner !== owner) {
      closeSubmenu(false);
      submenu = menuElement(doc, SUBMENU_ID);
      submenuOwner = owner;
      submenu.setAttribute('aria-label', item.label);
      fillMenu(doc, submenu, item.submenu, true);
      owner.setAttribute('aria-expanded', 'true');
      placeSubmenu(owner, item.submenu.length);
    }
    const first = focusFirst ? focusableItems(submenu)[0] : null;
    if (first) first.focus({preventScroll: true});
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
    const el = menuElement(doc, MENU_ID);
    menu = el;
    onClosed = typeof closed === 'function' ? closed : null;
    fillMenu(doc, el, items, false);
    root.ContentContextMenu.clampMenu(
      doc.defaultView,
      el,
      x,
      y,
      200,
      24 * items.length,
    );
    const first = focusableItems(el)[0];
    if (first) first.focus({preventScroll: true});
    return el;
  }

  root.TreeContextMenu = {show: show};
})(window);

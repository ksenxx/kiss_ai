// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The streamed browser tab: a page of the browser running on the
// daemon's machine, shown on this surface.
//
// The daemon (kiss/server/browser_tab.py) launches that machine's
// browser, screencasts every page as JPEG frames and replays our
// pointer, wheel, keyboard and paste events on it.  This module builds
// the view for one such page: an address bar (back, forward, reload,
// URL) over a <img> that shows the latest frame and captures input.
// A finger's drag scrolls the page; the daemon is asked, each time one
// goes down, whether a drag to the right from there has anything to
// scroll, so that the phone layout can take the others for its swipe
// back to the chat.
//
//   const view = BrowserTabView.create(tabId, {
//     send: msg => api[msg.type](msg),   // browserNavigate / browserInput /
//                                        // browserViewport / browserOpen
//   });
//   view.el                 -> element to mount in the content area
//   view.frame(ev)          -> paint a browserFrame event
//   view.touched(ev)        -> take a browserTouch event (the daemon's
//                              answer about a finger)
//   view.whenDragRightIsFree(callback)
//                           -> call back if the last finger's drag to
//                              the right had nothing to scroll
//   view.state(ev)          -> apply a browserState event (url, title, history)
//   view.error(text)        -> show a navigation error in the address bar
//   view.setBadge(name, isDefault, note) -> name the browser in the bar
//   view.setVisible(bool)   -> start/stop streaming for this surface
//   view.resubscribe()      -> report the viewport again after a reconnect
//   view.dispose()          -> stop observers, tell the daemon we left
//
// Coordinates: a frame is at most as large as the viewer, so it is
// drawn with object-fit: contain; pointer positions are mapped through
// the letterboxed image rectangle back to the page's CSS pixels
// (ev.width x ev.height of the last frame).

(function (global) {
  'use strict';

  const BUTTON_NAMES = ['left', 'middle', 'right', 'back', 'forward'];
  // The `buttons` bit of each button above.
  const BUTTON_BITS = [1, 4, 2, 8, 16];
  // All of them.  A pointer has buttons the remote page has no name
  // for (a pen's eraser, 32): those are no press there.
  const BUTTONS_MASK = 31;
  const RESIZE_DEBOUNCE_MS = 150;
  // Presses closer than this in time and space count up (double,
  // triple click) - pointer events carry no click count of their own.
  const MULTI_CLICK_MS = 400;
  const MULTI_CLICK_PX = 5;
  // How long after a swipe the daemon's answer about its finger still
  // counts: well above a mobile round trip, short enough for the
  // screen to turn as the swipe's effect.
  const TOUCH_ANSWER_MS = 600;

  function modifiers(e) {
    return {
      alt: !!e.altKey,
      ctrl: !!e.ctrlKey,
      meta: !!e.metaKey,
      shift: !!e.shiftKey,
    };
  }

  function svgButton(title, path) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'browser-nav-btn';
    btn.title = title;
    btn.setAttribute('aria-label', title);
    btn.innerHTML =
      '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
      'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
      'stroke-linejoin="round">' +
      path +
      '</svg>';
    return btn;
  }

  function create(tabId, opts) {
    const send = opts.send;
    const el = document.createElement('div');
    el.className = 'browser-view';

    const bar = document.createElement('div');
    bar.className = 'browser-toolbar';
    const backBtn = svgButton('Back', '<path d="M15 18l-6-6 6-6"/>');
    const fwdBtn = svgButton('Forward', '<path d="M9 18l6-6-6-6"/>');
    const reloadBtn = svgButton(
      'Reload',
      '<path d="M21 12a9 9 0 1 1-3-6.7"/><path d="M21 3v6h-6"/>',
    );
    const urlInput = document.createElement('input');
    urlInput.type = 'text';
    urlInput.className = 'browser-url';
    urlInput.placeholder = 'Search or enter address';
    urlInput.setAttribute('aria-label', 'Address');
    urlInput.spellcheck = false;
    const newTabBtn = svgButton(
      'New browser tab',
      '<path d="M12 5v14M5 12h14"/>',
    );
    const badge = document.createElement('span');
    badge.className = 'browser-badge';
    bar.appendChild(backBtn);
    bar.appendChild(fwdBtn);
    bar.appendChild(reloadBtn);
    bar.appendChild(urlInput);
    bar.appendChild(newTabBtn);
    bar.appendChild(badge);

    const screen = document.createElement('div');
    screen.className = 'browser-screen';
    screen.tabIndex = 0;
    screen.setAttribute('role', 'application');
    screen.setAttribute(
      'aria-label',
      'Remote browser page; click to focus and type',
    );
    const img = document.createElement('img');
    img.className = 'browser-frame';
    img.alt = '';
    img.draggable = false;
    const placeholder = document.createElement('div');
    placeholder.className = 'browser-placeholder';
    placeholder.textContent = 'Connecting to the browser\u2026';
    const errorBar = document.createElement('div');
    errorBar.className = 'browser-error';
    errorBar.style.display = 'none';
    screen.appendChild(img);
    screen.appendChild(placeholder);
    el.appendChild(bar);
    el.appendChild(errorBar);
    el.appendChild(screen);

    const view = {
      el,
      // CSS-pixel size of the page behind the last frame.
      pageWidth: 0,
      pageHeight: 0,
      url: '',
      visible: false,
      disposed: false,
    };

    let urlEditing = false;
    let resizeTimer = null;
    let lastSent = {width: 0, height: 0, visible: false};

    function navigate(action, url) {
      send({type: 'browserNavigate', tab_id: tabId, action, url: url || ''});
    }
    backBtn.addEventListener('click', () => navigate('back'));
    fwdBtn.addEventListener('click', () => navigate('forward'));
    reloadBtn.addEventListener('click', () => navigate('reload'));
    newTabBtn.addEventListener('click', () => {
      send({type: 'browserOpen', url: ''});
    });
    urlInput.addEventListener('focus', () => {
      urlEditing = true;
      urlInput.select();
    });
    urlInput.addEventListener('blur', () => {
      urlEditing = false;
      urlInput.value = view.url;
    });
    urlInput.addEventListener('keydown', e => {
      // An IME's Enter / Escape commits or cancels the composition, not
      // the address (Chromium reports keyCode 229 for keys an IME takes).
      if (e.isComposing || e.keyCode === 229) return;
      if (e.key === 'Enter') {
        e.preventDefault();
        navigate('go', urlInput.value.trim());
        screen.focus();
      } else if (e.key === 'Escape') {
        urlInput.value = view.url;
        screen.focus();
      }
    });

    // ---- frame -> page coordinate mapping -------------------------------

    function pagePoint(e) {
      const box = screen.getBoundingClientRect();
      const natW = img.naturalWidth || view.pageWidth;
      const natH = img.naturalHeight || view.pageHeight;
      if (
        !box.width ||
        !box.height ||
        !natW ||
        !natH ||
        !view.pageWidth ||
        !view.pageHeight
      ) {
        return {x: e.clientX - box.left, y: e.clientY - box.top};
      }
      // object-fit: contain -> the image is letterboxed and centred.
      const scale = Math.min(box.width / natW, box.height / natH);
      const dispW = natW * scale;
      const dispH = natH * scale;
      const offX = box.left + (box.width - dispW) / 2;
      const offY = box.top + (box.height - dispH) / 2;
      return {
        x: ((e.clientX - offX) / dispW) * view.pageWidth,
        y: ((e.clientY - offY) / dispH) * view.pageHeight,
      };
    }

    function inputEvent(event) {
      send({type: 'browserInput', tab_id: tabId, event});
    }

    // The pointer event forwarded last and the page point it mapped to:
    // where the remote pointer is.
    let lastPointer = null;
    let lastPoint = null;
    function mouse(action, e, extra, point) {
      lastPointer = e;
      lastPoint = point || pagePoint(e);
      const event = Object.assign(
        {
          kind: 'mouse',
          action,
          x: Math.round(lastPoint.x * 100) / 100,
          y: Math.round(lastPoint.y * 100) / 100,
          button:
            action === 'mouseMoved' || action === 'mouseWheel'
              ? 'none'
              : BUTTON_NAMES[e.button] || 'left',
          buttons: e.buttons || 0,
          clickCount:
            action === 'mouseMoved' || action === 'mouseWheel' ? 0 : 1,
        },
        modifiers(e),
        extra || {},
      );
      inputEvent(event);
    }

    let moveFrame = null;
    let pendingMove = null;
    /**
     * Send the move waiting for its animation frame now (and drop the
     * frame): a press, release or wheel must reach the daemon after the
     * move that came before it, in the user's order, else the remote
     * page sees the release before the drag and a buttons-down move
     * after it.
     */
    function flushMove() {
      if (moveFrame !== null) {
        cancelAnimationFrame(moveFrame);
        moveFrame = null;
      }
      const move = pendingMove;
      pendingMove = null;
      if (move && !view.disposed) mouse('mouseMoved', move);
    }
    let lastPress = {t: 0, x: 0, y: 0, button: -1, count: 0};
    function clickCountFor(e) {
      const now = Date.now();
      const near =
        now - lastPress.t < MULTI_CLICK_MS &&
        lastPress.button === e.button &&
        Math.abs(e.clientX - lastPress.x) <= MULTI_CLICK_PX &&
        Math.abs(e.clientY - lastPress.y) <= MULTI_CLICK_PX;
      lastPress = {
        t: now,
        x: e.clientX,
        y: e.clientY,
        button: e.button,
        count: near ? lastPress.count + 1 : 1,
      };
      return lastPress.count;
    }
    // The buttons down on the remote page, a `buttons` bitmask.
    let held = 0;
    // The screen lost focus with a button down and no pointer event
    // has come since.
    let heldAtBlur = false;
    // The view was hidden with a button down: that press ended on the
    // remote page then, and its pointer is ignored until it lets go.
    let heldAtHide = false;
    /**
     * Forward the button transitions from *held* to *e*'s `buttons`: a
     * press or a release per changed button, in button order.  Pointer
     * Events fire pointerdown for the first button pressed and
     * pointerup for the last released; a button pressed or released
     * while another stays down (a chord) comes as a pointermove whose
     * `buttons` changed, which forwarded as a move would leave the
     * remote page with that press or release missing.
     */
    function syncButtons(e) {
      const buttons = e.buttons & BUTTONS_MASK;
      for (let i = 0; i < BUTTON_BITS.length; i++) {
        const bit = BUTTON_BITS[i];
        if ((held & bit) === (buttons & bit)) continue;
        const down = (buttons & bit) !== 0;
        mouse(down ? 'mousePressed' : 'mouseReleased', e, {
          button: BUTTON_NAMES[i],
          clickCount: down ? clickCountFor(e) : lastPress.count || 1,
        });
      }
      held = buttons;
    }
    screen.addEventListener('pointerdown', e => {
      if (e.pointerType === 'touch') return;
      heldAtBlur = false;
      heldAtHide = false;
      screen.focus();
      if (screen.setPointerCapture && e.pointerId !== undefined) {
        try {
          screen.setPointerCapture(e.pointerId);
        } catch (_e) {}
      }
      flushMove();
      syncButtons(e);
      e.preventDefault();
    });
    screen.addEventListener('pointerup', e => {
      if (e.pointerType === 'touch') return;
      heldAtBlur = false;
      heldAtHide = false;
      flushMove();
      syncButtons(e);
      e.preventDefault();
    });
    // A pointer the browser cancels after a press (the OS took the
    // device, a native drag began) gets no pointerup.  A release sent
    // instead would click the control under the press (Chromium fires
    // click on a release at the press point whatever its click count),
    // and a cancelled gesture must activate nothing: a move with no
    // buttons down ends the press on the remote page (its pressed
    // state, :active) without mouseup or click, as the cancel ended it
    // here.
    function cancelPress() {
      if (!held) return;
      flushMove();
      held = 0;
      mouse('mouseMoved', lastPointer, {buttons: 0}, lastPoint);
    }
    screen.addEventListener('pointercancel', e => {
      if (e.pointerType !== 'touch') cancelPress();
    });
    screen.addEventListener('pointermove', e => {
      if (e.pointerType === 'touch') return;
      if (heldAtHide && e.buttons) return;
      heldAtHide = false;
      // A button let go in another window or tab gets no pointerup
      // either: the screen lost focus with it down, and the first move
      // back shows none.  That press ends as a cancelled one does.  A
      // button still down then (the focus only moved, or the pointer
      // came back holding it) keeps its press.
      if (heldAtBlur && !e.buttons) cancelPress();
      heldAtBlur = false;
      if ((e.buttons & BUTTONS_MASK) !== held) {
        flushMove();
        syncButtons(e);
      }
      // Coalesce to one move per animation frame.
      pendingMove = e;
      if (moveFrame === null) moveFrame = requestAnimationFrame(flushMove);
    });
    screen.addEventListener(
      'wheel',
      e => {
        flushMove();
        mouse('mouseWheel', e, {
          deltaX: e.deltaMode === 0 ? e.deltaX : e.deltaX * 40,
          deltaY: e.deltaMode === 0 ? e.deltaY : e.deltaY * 40,
        });
        e.preventDefault();
      },
      {passive: false},
    );
    screen.addEventListener('contextmenu', e => e.preventDefault());

    // Touch: a tap is a click, a drag scrolls (as on a phone browser).
    // A second finger (a pinch) or a cancelled touch ends the gesture:
    // its final touchend must not click where the first finger landed.
    let touchStart = null;
    // What the finger last put down asked the daemon: a drag scrolls
    // the remote page, and only the daemon can tell whether a drag to
    // the right from that point has anything to scroll there (the
    // page, a wide table in it, a frame).  `scrollsLeft` is its answer
    // (null until it comes, see view.touched); `free` waits for it
    // (see view.whenDragRightIsFree).
    let asked = null;
    let fingers = 0;
    const viewKey = Math.random().toString(36).slice(2);
    function forgetAsked() {
      if (asked) clearTimeout(asked.timer);
      asked = null;
    }
    screen.addEventListener(
      'touchstart',
      e => {
        forgetAsked();
        if (e.touches.length !== 1) {
          touchStart = null;
          return;
        }
        const t = e.touches[0];
        touchStart = {x: t.clientX, y: t.clientY, moved: false};
        const point = pagePoint(t);
        // Every surface showing the page gets the answer: the id says
        // whose finger it is about.
        asked = {
          id: viewKey + '-' + ++fingers,
          scrollsLeft: null,
          free: null,
          timer: null,
        };
        inputEvent({
          kind: 'touch',
          id: asked.id,
          x: Math.round(point.x * 100) / 100,
          y: Math.round(point.y * 100) / 100,
        });
      },
      {passive: true},
    );
    screen.addEventListener(
      'touchmove',
      e => {
        if (!touchStart) return;
        if (e.touches.length !== 1) {
          touchStart = null;
          return;
        }
        const t = e.touches[0];
        const dx = touchStart.x - t.clientX;
        const dy = touchStart.y - t.clientY;
        if (!touchStart.moved && Math.abs(dx) + Math.abs(dy) < 6) return;
        touchStart.moved = true;
        // The remote pointer follows the finger.  Chromium takes wheels
        // at a page edge for a history swipe, which lasts until the
        // pointer moves: without the move a drag past the edge (the
        // phone's swipe back to the chat) left the page deaf to every
        // later drag.
        mouse('mouseMoved', t, {buttons: held});
        mouse('mouseWheel', t, {deltaX: dx, deltaY: dy});
        touchStart.x = t.clientX;
        touchStart.y = t.clientY;
        e.preventDefault();
      },
      {passive: false},
    );
    screen.addEventListener('touchend', e => {
      if (!touchStart) return;
      const t = e.changedTouches[0];
      if (!touchStart.moved && t) {
        screen.focus();
        const fake = {clientX: t.clientX, clientY: t.clientY, button: 0};
        mouse('mousePressed', fake, {buttons: 1});
        mouse('mouseReleased', fake, {buttons: 0});
      }
      touchStart = null;
    });
    screen.addEventListener('touchcancel', () => {
      touchStart = null;
    });

    function key(action, e) {
      inputEvent(
        Object.assign(
          {
            kind: 'key',
            action,
            key: e.key,
            code: e.code,
            keyCode: e.keyCode,
            repeat: !!e.repeat,
          },
          modifiers(e),
        ),
      );
    }
    // The keys forwarded down and not up yet, by code.  A keyup goes to
    // wherever the focus is then: one that leaves the screen (the
    // address field, another tab, another application) never reaches
    // it, and the remote page would keep the key pressed.  They are
    // released when the screen loses focus, when the view is hidden
    // and on dispose.
    const heldKeys = new Map();
    function releaseKeys() {
      heldKeys.forEach(e => key('up', e));
      heldKeys.clear();
    }
    screen.addEventListener('keydown', e => {
      // Paste comes through the paste event with the LOCAL clipboard's
      // text; the remote page never sees this Ctrl/Cmd+V.
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') return;
      heldKeys.set(e.code || e.key, e);
      key('down', e);
      e.preventDefault();
    });
    screen.addEventListener('keyup', e => {
      // Only a key forwarded down is forwarded up: not the paste's V, nor
      // a key pressed elsewhere or released already (the blur, a hide).
      if (!heldKeys.delete(e.code || e.key)) return;
      key('up', e);
      e.preventDefault();
    });
    screen.addEventListener('blur', () => {
      heldAtBlur = held !== 0;
      releaseKeys();
    });
    screen.addEventListener('paste', e => {
      const text =
        e.clipboardData && e.clipboardData.getData
          ? e.clipboardData.getData('text')
          : '';
      if (text) inputEvent({kind: 'text', text});
      e.preventDefault();
    });

    // ---- viewport reporting ---------------------------------------------

    function reportViewport() {
      if (view.disposed) return;
      const w = Math.round(screen.clientWidth);
      const h = Math.round(screen.clientHeight);
      const visible = view.visible && w > 0 && h > 0;
      if (
        visible === lastSent.visible &&
        (!visible || (w === lastSent.width && h === lastSent.height))
      ) {
        return;
      }
      lastSent = {width: w, height: h, visible};
      send({
        type: 'browserViewport',
        tab_id: tabId,
        width: w,
        height: h,
        visible,
      });
    }
    function scheduleViewport() {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(reportViewport, RESIZE_DEBOUNCE_MS);
    }
    let observer = null;
    if (typeof ResizeObserver === 'function') {
      observer = new ResizeObserver(() => {
        if (view.visible) scheduleViewport();
      });
      observer.observe(screen);
    }

    // After a reconnect the daemon has no viewer for this surface any
    // more: forget what was last reported so the next report goes out.
    view.resubscribe = function () {
      lastSent = {width: 0, height: 0, visible: false};
      if (view.visible) reportViewport();
    };

    view.setVisible = function (visible) {
      view.visible = !!visible;
      clearTimeout(resizeTimer);
      if (view.visible) {
        // Layout has to settle before the size is meaningful.
        resizeTimer = setTimeout(reportViewport, 0);
      } else {
        // The caller hid the screen: with no box left, neither the move
        // waiting for its frame nor what a pointer or finger that is
        // down does next can be mapped to the page.  Their gestures end.
        pendingMove = null;
        if (held) heldAtHide = true;
        cancelPress();
        touchStart = null;
        forgetAsked();
        releaseKeys();
        reportViewport();
      }
    };

    /**
     * The daemon's answer (a `browserTouch` event) about a finger put
     * down on the page.  One about the finger of another surface, or
     * about a finger this view no longer waits for, changes nothing.
     */
    view.touched = function (ev) {
      if (!asked || ev.id !== asked.id) return;
      asked.scrollsLeft = !!ev.scrollsLeft;
      clearTimeout(asked.timer);
      const free = asked.free;
      asked.free = null;
      if (free && !asked.scrollsLeft) free();
    };

    /**
     * Call *callback* if a drag to the right from where the last finger
     * went down had nothing to scroll on the remote page, so that the
     * drag can count as a swipe of the surface instead: now when the
     * daemon has said so, else when it says so within TOUCH_ANSWER_MS.
     * Without its word (an older daemon, a dead link) the drag stays
     * the page's; the next finger, hiding the view and disposing it
     * end the wait.
     */
    view.whenDragRightIsFree = function (callback) {
      if (!asked || asked.scrollsLeft) return;
      if (asked.scrollsLeft === false) {
        callback();
        return;
      }
      asked.free = callback;
      clearTimeout(asked.timer);
      asked.timer = setTimeout(forgetAsked, TOUCH_ANSWER_MS);
    };

    view.frame = function (ev) {
      if (view.disposed || !ev.data) return;
      view.pageWidth = ev.width || view.pageWidth;
      view.pageHeight = ev.height || view.pageHeight;
      img.src = 'data:image/jpeg;base64,' + ev.data;
      if (placeholder.parentNode)
        placeholder.parentNode.removeChild(placeholder);
    };

    view.state = function (ev) {
      view.url = ev.url || '';
      if (!urlEditing) urlInput.value = view.url;
      backBtn.disabled = !ev.canGoBack;
      fwdBtn.disabled = !ev.canGoForward;
      errorBar.style.display = 'none';
    };

    view.error = function (text) {
      errorBar.textContent = text || '';
      errorBar.style.display = text ? '' : 'none';
    };

    view.setBadge = function (browserName, isDefault, note) {
      const product =
        (window.__BRAND__ && window.__BRAND__.productName) || 'KISS Sorcar';
      badge.textContent = browserName || '';
      badge.title = note
        ? note + ' Showing ' + browserName + ' instead.'
        : isDefault
          ? browserName +
            ' (default browser of the machine running ' +
            product +
            ')'
          : browserName + ' (browser on the machine running ' + product + ')';
    };

    view.dispose = function () {
      if (view.disposed) return;
      releaseKeys();
      forgetAsked();
      view.disposed = true;
      clearTimeout(resizeTimer);
      if (moveFrame !== null) cancelAnimationFrame(moveFrame);
      if (observer) observer.disconnect();
      if (lastSent.visible) {
        send({
          type: 'browserViewport',
          tab_id: tabId,
          width: 0,
          height: 0,
          visible: false,
        });
      }
    };

    return view;
  }

  global.BrowserTabView = {create};
})(typeof window !== 'undefined' ? window : globalThis);

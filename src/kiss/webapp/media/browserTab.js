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
//
//   const view = BrowserTabView.create(tabId, {
//     send: msg => api[msg.type](msg),   // browserNavigate / browserInput /
//                                        // browserViewport / browserOpen
//   });
//   view.el                 -> element to mount in the content area
//   view.frame(ev)          -> paint a browserFrame event
//   view.state(ev)          -> apply a browserState event (url, title, history)
//   view.error(text)        -> show a navigation error in the address bar
//   view.setVisible(bool)   -> start/stop streaming for this surface
//   view.dispose()          -> stop observers, tell the daemon we left
//
// Coordinates: a frame is at most as large as the viewer, so it is
// drawn with object-fit: contain; pointer positions are mapped through
// the letterboxed image rectangle back to the page's CSS pixels
// (ev.width x ev.height of the last frame).

(function (global) {
  'use strict';

  const BUTTON_NAMES = ['left', 'middle', 'right', 'back', 'forward'];
  const RESIZE_DEBOUNCE_MS = 150;
  // Presses closer than this in time and space count up (double,
  // triple click) - pointer events carry no click count of their own.
  const MULTI_CLICK_MS = 400;
  const MULTI_CLICK_PX = 5;

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
      tabId,
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

    function mouse(action, e, extra) {
      const p = pagePoint(e);
      const event = Object.assign(
        {
          kind: 'mouse',
          action,
          x: Math.round(p.x * 100) / 100,
          y: Math.round(p.y * 100) / 100,
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
    screen.addEventListener('pointerdown', e => {
      if (e.pointerType === 'touch') return;
      screen.focus();
      if (screen.setPointerCapture && e.pointerId !== undefined) {
        try {
          screen.setPointerCapture(e.pointerId);
        } catch (_e) {}
      }
      mouse('mousePressed', e, {clickCount: clickCountFor(e)});
      e.preventDefault();
    });
    screen.addEventListener('pointerup', e => {
      if (e.pointerType === 'touch') return;
      mouse('mouseReleased', e, {clickCount: lastPress.count || 1});
      e.preventDefault();
    });
    screen.addEventListener('pointermove', e => {
      if (e.pointerType === 'touch') return;
      // Coalesce to one move per animation frame.
      pendingMove = e;
      if (moveFrame === null) {
        moveFrame = requestAnimationFrame(() => {
          moveFrame = null;
          if (pendingMove && !view.disposed) mouse('mouseMoved', pendingMove);
          pendingMove = null;
        });
      }
    });
    screen.addEventListener(
      'wheel',
      e => {
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
    let touchStart = null;
    screen.addEventListener(
      'touchstart',
      e => {
        if (e.touches.length !== 1) return;
        const t = e.touches[0];
        touchStart = {x: t.clientX, y: t.clientY, moved: false};
      },
      {passive: true},
    );
    screen.addEventListener(
      'touchmove',
      e => {
        if (!touchStart || e.touches.length !== 1) return;
        const t = e.touches[0];
        const dx = touchStart.x - t.clientX;
        const dy = touchStart.y - t.clientY;
        if (!touchStart.moved && Math.abs(dx) + Math.abs(dy) < 6) return;
        touchStart.moved = true;
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
    screen.addEventListener('keydown', e => {
      // Paste comes through the paste event with the LOCAL clipboard's
      // text; the remote page never sees this Ctrl/Cmd+V.
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') return;
      key('down', e);
      e.preventDefault();
    });
    screen.addEventListener('keyup', e => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'v') return;
      key('up', e);
      e.preventDefault();
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
        reportViewport();
      }
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
      badge.textContent = browserName || '';
      badge.title = note
        ? note + ' Showing ' + browserName + ' instead.'
        : isDefault
          ? browserName +
            ' (default browser of the machine running KISS Sorcar)'
          : browserName + ' (browser on the machine running KISS Sorcar)';
    };

    view.dispose = function () {
      if (view.disposed) return;
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

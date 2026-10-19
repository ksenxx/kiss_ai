// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Window bounds of the standalone app window on macOS (kiss-app).
// There kiss-app opens media/appWindow.html as the browser's "app"
// window, and that page opens this webapp as a popup window named
// "kiss-app-window" (why: see appWindow.html).  Chromium remembers
// the bounds of app windows between launches but not those of
// popups, so this script keeps the popup's bounds in localStorage
// under kissAppWindowBounds, the key appWindow.html reads on the
// next launch: screen position and outer size, the values
// window.open takes.  They are saved when they change: on resize
// events, when the page is hidden (the window closing), and, as a
// move raises no event, by looking once a second.  Outside that
// popup (a browser tab, the Linux app window) nothing happens: the
// browser keeps the bounds itself.
(function () {
  'use strict';

  const KEY = 'kissAppWindowBounds';
  let last = '';

  /**
   * Save the window's current bounds when they differ from the last
   * saved ones.  Storage that is unavailable (denied, full) is left
   * alone: the next launch then opens the window where the app page
   * was, as before.
   */
  function save() {
    if (!(window.outerWidth > 0) || !(window.outerHeight > 0)) return;
    const bounds = JSON.stringify({
      left: window.screenX,
      top: window.screenY,
      width: window.outerWidth,
      height: window.outerHeight,
    });
    if (bounds === last) return;
    last = bounds;
    try {
      window.localStorage.setItem(KEY, bounds);
    } catch {
      // no storage: see above
    }
  }

  if (window.name !== 'kiss-app-window') return;
  save();
  window.addEventListener('resize', save);
  window.addEventListener('pagehide', save);
  window.setInterval(save, 1000);
})();

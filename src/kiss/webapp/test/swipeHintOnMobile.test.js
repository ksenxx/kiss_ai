// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// The phone swipe hint and the body.remote-mobile class (main.js):
//
// * A phone (mobile user agent) marks <body> `remote-mobile`, which
//   remote-codex.css uses to drop the composer's chats and task-info
//   buttons; a desktop browser never gets the class.
// * On the phone layout (< 900px) a phone shows one sticky
//   "swipe-hint" notice telling the user to swipe between the panels,
//   with a "Do not show this again" box.  Ticking the box removes the
//   toast and records the choice in localStorage without telling the
//   daemon (no `dismissNotice`); a page that finds the record shows no
//   hint.  Closing the toast with its × records nothing.
// * No hint on the desktop layout, for a desktop browser, or when
//   mobileSwipe.js is not loaded (nothing to swipe with).  A phone that
//   starts wide and narrows later gets the hint then, once per page.
// * A localStorage that throws (private browsing) neither blocks the
//   hint nor the tick.

'use strict';

const assert = require('assert');
const {makePage, waitFor, fakeMatchMedia} = require('./simplify2_harness');

const KEY = 'kiss-swipe-hint-dismissed';
const TOAST = '.kiss-notification[data-notification-id="swipe-hint"]';
const PHONE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1';
const DESKTOP_UA =
  'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36';

/**
 * A chat page under the given user agent and window width.
 *
 * @param {{ua: string, desktop: boolean, swipe?: boolean, dismissed?: boolean,
 *   brokenStorage?: boolean}} opts `swipe` (default true) loads
 *   mobileSwipe.js; `dismissed` pre-records the opt-out; `brokenStorage`
 *   makes every localStorage access throw.
 * @returns {{win: object, posted: Array<object>, setDesktop: Function}}
 *   `setDesktop(bool)` flips the (min-width: 900px) media query.
 */
function phonePage(opts) {
  let setDesktop;
  const {win, posted} = makePage({
    userAgent: opts.ua,
    extraScripts: opts.swipe !== false ? ['mobileSwipe.js'] : [],
    beforeScripts(w) {
      Object.defineProperty(w.navigator, 'userAgentData', {
        value: undefined,
        configurable: true,
      });
      setDesktop = fakeMatchMedia(w, opts.desktop);
      if (opts.dismissed) w.localStorage.setItem(KEY, '1');
      if (opts.brokenStorage) {
        Object.defineProperty(w, 'localStorage', {
          configurable: true,
          get() {
            throw new Error('storage disabled');
          },
        });
      }
    },
  });
  return {win, posted, setDesktop};
}

function dismissals(posted) {
  return posted.filter(m => m && m.type === 'dismissNotice');
}

async function runTests() {
  {
    // A phone on the phone layout: the class, the hint, the box.
    const {win, posted} = phonePage({ua: PHONE_UA, desktop: false});
    try {
      assert.ok(win.document.body.classList.contains('remote-mobile'));
      const toast = await waitFor(
        () => win.document.querySelector(TOAST),
        'hint renders',
      );
      assert.ok(/swipe left or right/i.test(toast.textContent));
      assert.ok(/task info/i.test(toast.textContent));
      const label = toast.querySelector('label.kiss-notification-opt-out');
      assert.strictEqual(label.textContent.trim(), 'Do not show this again');
      const box = toast.querySelector('.kiss-notification-opt-out-box');
      assert.strictEqual(box.checked, false);
      assert.strictEqual(win.localStorage.getItem(KEY), null);

      // Unticking (a change event with the box clear) does nothing.
      box.dispatchEvent(new win.Event('change', {bubbles: true}));
      assert.ok(win.document.querySelector(TOAST));

      box.click();
      await waitFor(
        () => !win.document.querySelector(TOAST),
        'tick removes the toast',
      );
      assert.strictEqual(win.localStorage.getItem(KEY), '1');
      assert.strictEqual(
        dismissals(posted).length,
        0,
        'the daemon is not told',
      );
    } finally {
      win.close();
    }
  }

  {
    // The recorded choice: a fresh phone page shows no hint.
    const {win} = phonePage({ua: PHONE_UA, desktop: false, dismissed: true});
    try {
      assert.strictEqual(win.document.querySelector(TOAST), null);
      assert.ok(win.document.body.classList.contains('remote-mobile'));
    } finally {
      win.close();
    }
  }

  {
    // Closing with × records nothing.
    const {win} = phonePage({ua: PHONE_UA, desktop: false});
    try {
      const toast = await waitFor(
        () => win.document.querySelector(TOAST),
        'hint renders',
      );
      toast.querySelector('.kiss-notification-close').click();
      await waitFor(
        () => !win.document.querySelector(TOAST),
        'close removes the toast',
      );
      assert.strictEqual(win.localStorage.getItem(KEY), null);
    } finally {
      win.close();
    }
  }

  {
    // A phone that starts wide: no hint until the window narrows, then
    // one hint for the page however often the layout flips.
    const {win, setDesktop} = phonePage({ua: PHONE_UA, desktop: true});
    try {
      assert.strictEqual(win.document.querySelector(TOAST), null);
      setDesktop(false);
      const toast = await waitFor(
        () => win.document.querySelector(TOAST),
        'hint renders once narrow',
      );
      toast.querySelector('.kiss-notification-close').click();
      await waitFor(
        () => !win.document.querySelector(TOAST),
        'close removes the toast',
      );
      setDesktop(true);
      setDesktop(false);
      assert.strictEqual(
        win.document.querySelector(TOAST),
        null,
        'no second hint this page',
      );
    } finally {
      win.close();
    }
  }

  {
    // Closing the task-info drawer on a phone: the toggle is hidden, so
    // focus goes to the footer's New chat button instead.
    const {win} = phonePage({ua: PHONE_UA, desktop: false});
    try {
      const doc = win.document;
      doc.getElementById('meta-drawer-btn').click();
      await waitFor(
        () => doc.getElementById('meta-panel').classList.contains('open'),
        'drawer opens',
      );
      assert.strictEqual(doc.activeElement.id, 'meta-close');
      doc.getElementById('meta-close').click();
      await waitFor(
        () => !doc.getElementById('meta-panel').classList.contains('open'),
        'drawer closes',
      );
      assert.strictEqual(doc.activeElement.id, 'new-chat-btn');
    } finally {
      win.close();
    }
  }

  {
    // A desktop browser keeps handing focus back to the toggle.
    const {win} = phonePage({ua: DESKTOP_UA, desktop: false});
    try {
      const doc = win.document;
      doc.getElementById('meta-drawer-btn').click();
      await waitFor(
        () => doc.getElementById('meta-panel').classList.contains('open'),
        'drawer opens',
      );
      doc.getElementById('meta-close').click();
      await waitFor(
        () => !doc.getElementById('meta-panel').classList.contains('open'),
        'drawer closes',
      );
      assert.strictEqual(doc.activeElement.id, 'meta-drawer-btn');
    } finally {
      win.close();
    }
  }

  {
    // A desktop browser at phone width: no class, no hint.
    const {win} = phonePage({ua: DESKTOP_UA, desktop: false});
    try {
      assert.ok(!win.document.body.classList.contains('remote-mobile'));
      assert.strictEqual(win.document.querySelector(TOAST), null);
    } finally {
      win.close();
    }
  }

  {
    // A phone without mobileSwipe.js: nothing to swipe with, no hint.
    const {win} = phonePage({ua: PHONE_UA, desktop: false, swipe: false});
    try {
      assert.strictEqual(win.document.querySelector(TOAST), null);
    } finally {
      win.close();
    }
  }

  {
    // A throwing localStorage: the hint still shows and the tick still
    // removes it.
    const {win, posted} = phonePage({
      ua: PHONE_UA,
      desktop: false,
      brokenStorage: true,
    });
    try {
      const toast = await waitFor(
        () => win.document.querySelector(TOAST),
        'hint renders',
      );
      toast.querySelector('.kiss-notification-opt-out-box').click();
      await waitFor(
        () => !win.document.querySelector(TOAST),
        'tick removes the toast',
      );
      assert.strictEqual(dismissals(posted).length, 0);
    } finally {
      win.close();
    }
  }
}

runTests().then(
  () => {
    console.log('\nAll tests passed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  },
);

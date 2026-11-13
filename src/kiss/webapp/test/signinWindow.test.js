// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The sign-in window (chat.html#signin-modal, main.js "Sign-in window").
//
// The daemon sends `signin` while the brand's sign-in hook says the user
// is not logged in; the page shows a modal with one button per provider.
// A click opens a tab at once (so a popup blocker lets it through) and
// sends `signinLogin`; `signinUrl` loads the sign-in page in that tab and
// shows the paste row; `signinDone` closes the window; `signinError`
// keeps it open with the reason; "Not now" sends `signinCancel`.

/* global require, console */

'use strict';

const assert = require('assert');
const {makePage, send, captureTimers} = require('./simplify2_harness');

const SIGNIN = {
  type: 'signin',
  title: 'Sign in to the s10s hub',
  providers: [
    {id: 'google', label: 'Login with Google'},
    {id: 'apple', label: 'Login with Apple'},
  ],
};

/**
 * A fake tab for window.open: records the page it is sent to as
 * `navigatedTo`, and, like a real browser once the tab is on another
 * origin, refuses to tell its location afterwards.
 */
function fakeTab() {
  const tab = {closed: false, navigatedTo: 'about:blank'};
  tab.location = {
    get href() {
      if (tab.navigatedTo !== 'about:blank') {
        throw new Error('SecurityError: cross-origin location');
      }
      return tab.navigatedTo;
    },
    set href(url) {
      tab.navigatedTo = url;
    },
  };
  tab.close = () => {
    tab.closed = true;
  };
  return tab;
}

/**
 * A page whose window.open hands out fake tabs.
 * @returns {{win: object, posted: Array<object>, tabs: Array<object>}}
 */
function signinPage() {
  const tabs = [];
  const page = makePage({
    beforeScripts: win => {
      win.open = (url, target) => {
        const tab = fakeTab();
        tab.navigatedTo = url;
        tab.target = target;
        tabs.push(tab);
        return tab;
      };
    },
  });
  return {win: page.win, posted: page.posted, tabs};
}

/**
 * The messages of *type* the page posted, as plain objects (jsdom's
 * objects come from another realm, which deepStrictEqual minds).
 */
function sent(posted, type) {
  return JSON.parse(JSON.stringify(posted.filter(m => m.type === type)));
}

function modal(win) {
  return win.document.getElementById('signin-modal');
}

function buttons(win) {
  return Array.from(
    win.document.querySelectorAll('#signin-modal-buttons button'),
  );
}

function note(win) {
  return win.document.getElementById('signin-modal-note');
}

function pasteRow(win) {
  return win.document.getElementById('signin-modal-paste');
}

// The sign-in window path in main.js (showSigninWindow, signinPageArrived,
// signinFinished, signinFailed) is synchronous: no timer, promise or frame
// sits between an inbound event and the DOM it paints, so every
// assertion below follows its `send` directly.
function runTests() {
  // The window is hidden until a `signin` event, then shows the title and
  // one button per provider, in order.
  {
    const {win, posted} = signinPage();
    try {
      assert.strictEqual(
        modal(win).style.display,
        'none',
        'hidden before any signin event',
      );
      send(win, SIGNIN);
      assert.strictEqual(
        modal(win).style.display,
        'flex',
        'the signin event shows the window',
      );
      assert.strictEqual(
        win.document.getElementById('signin-modal-title').textContent,
        'Sign in to the s10s hub',
      );
      assert.deepStrictEqual(
        buttons(win).map(b => b.textContent),
        ['Login with Google', 'Login with Apple'],
      );
      assert.deepStrictEqual(
        buttons(win).map(b => b.dataset.provider),
        ['google', 'apple'],
      );
      assert.strictEqual(
        pasteRow(win).style.display,
        'none',
        'no paste row before a click',
      );
      assert.ok(
        !posted.some(m => m.type && m.type.startsWith('signin')),
        'showing posts nothing',
      );
      // A `signin` repeated while the window is up (the daemon sends one
      // on every reconnect) leaves it alone: same title, same buttons.
      send(win, {
        type: 'signin',
        title: 'Other',
        providers: [{id: 'x', label: 'X'}],
      });
      assert.strictEqual(
        win.document.getElementById('signin-modal-title').textContent,
        'Sign in to the s10s hub',
        'a reconnect does not repaint the open window',
      );
      assert.strictEqual(buttons(win).length, 2);
      // The daemon's focusInput (sent with the ready sequence, so right
      // around the signin) must not pull the keyboard from the window
      // to the composer behind it, now or through its 100 / 300 ms
      // retries (focusInputWithRetry in main.js), which it must not arm.
      buttons(win)[0].focus();
      const timers = captureTimers(win);
      send(win, {type: 'focusInput'});
      assert.strictEqual(
        win.document.activeElement,
        buttons(win)[0],
        'focusInput leaves the keyboard on the sign-in window',
      );
      assert.deepStrictEqual(
        timers.filter(t => t.ms === 100 || t.ms === 300),
        [],
        'focusInput arms no focus retry while the window is up',
      );
      // Once dismissed, a missing title falls back to "Sign in".
      win.document.getElementById('signin-modal-later').click();
      send(win, {type: 'signin', providers: [{id: 'x', label: 'X'}]});
      assert.strictEqual(
        win.document.getElementById('signin-modal-title').textContent,
        'Sign in',
        'default title',
      );
    } finally {
      win.close();
    }
  }

  // Clicking "Login with Google" opens a tab at once and asks the daemon
  // for the page; `signinUrl` loads the page in that tab, links it in the
  // note and shows the paste row.
  {
    const {win, posted, tabs} = signinPage();
    try {
      send(win, SIGNIN);
      assert.strictEqual(buttons(win).length, 2, 'buttons');
      buttons(win)[0].click();
      assert.strictEqual(
        tabs.length,
        1,
        'the click opens a tab before the URL is known',
      );
      assert.strictEqual(tabs[0].navigatedTo, 'about:blank');
      // The click carries a token of its own (A18-3) that the daemon
      // echoes on the click's answers.
      const attempt = sent(posted, 'signinLogin')[0].attempt;
      assert.ok(attempt, 'the click is sent with an attempt token');
      assert.deepStrictEqual(sent(posted, 'signinLogin'), [
        {type: 'signinLogin', provider: 'google', attempt},
      ]);
      assert.ok(note(win).textContent.includes('Starting the sign-in'));

      // A URL of an earlier, replaced click (another provider) is ignored.
      send(win, {
        type: 'signinUrl',
        provider: 'apple',
        url: 'https://stale.example/',
      });
      assert.strictEqual(tabs[0].navigatedTo, 'about:blank');
      assert.strictEqual(pasteRow(win).style.display, 'none');

      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/auth?x=1',
      });
      assert.strictEqual(
        tabs[0].navigatedTo,
        'https://accounts.example/auth?x=1',
        'the sign-in page loads in the tab the click opened',
      );
      const link = note(win).querySelector('a');
      assert.ok(link, 'the note links the page in case the tab is gone');
      assert.strictEqual(link.href, 'https://accounts.example/auth?x=1');
      assert.strictEqual(link.target, '_blank');
      assert.ok(note(win).textContent.includes('page that just opened'));
      assert.notStrictEqual(
        pasteRow(win).style.display,
        'none',
        'the paste row is offered',
      );

      // The pasted redirect address goes to the daemon (Enter or the button).
      const input = win.document.getElementById('signin-modal-paste-input');
      input.value = '  http://127.0.0.1:5555/?code=abc&state=xyz ';
      input.dispatchEvent(
        new win.KeyboardEvent('keydown', {key: 'Enter', bubbles: true}),
      );
      assert.deepStrictEqual(sent(posted, 'signinPaste'), [
        {
          type: 'signinPaste',
          text: 'http://127.0.0.1:5555/?code=abc&state=xyz',
          attempt,
        },
      ]);
      assert.ok(note(win).textContent.includes('Finishing the sign-in'));
      input.value = '   ';
      win.document.getElementById('signin-modal-paste-btn').click();
      assert.strictEqual(
        sent(posted, 'signinPaste').length,
        1,
        'an empty paste is not sent',
      );
      input.value = 'ABCD-1234';
      win.document.getElementById('signin-modal-paste-btn').click();
      assert.strictEqual(sent(posted, 'signinPaste').length, 2);

      // `signinDone` closes the window and says who signed in.
      send(win, {type: 'signinDone', email: 'me@example.com'});
      assert.strictEqual(
        modal(win).style.display,
        'none',
        'signinDone closes the window',
      );
      const toast = win.document.querySelector(
        '.kiss-notification[data-notification-id="signin"]',
      );
      assert.ok(toast, 'a toast names the account');
      assert.ok(toast.textContent.includes('Signed in as me@example.com'));
      assert.ok(
        !tabs[0].closed,
        'the tab showing the sign-in page is left alone',
      );
      assert.ok(
        !posted.some(m => m.type === 'signinCancel'),
        'a finished sign-in is not cancelled',
      );
      // Late events for a closed window are ignored.
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://late.example/',
      });
      send(win, {type: 'signinError', provider: 'google', error: 'late'});
      send(win, {type: 'signinDone'});
      assert.strictEqual(modal(win).style.display, 'none');
      assert.strictEqual(tabs.length, 1);
      // So are the two that answer any click: a failure naming no
      // provider (a pasted code's) and a second "done".
      const noteBefore = note(win).textContent;
      send(win, {type: 'signinError', error: 'late'});
      assert.strictEqual(
        note(win).textContent,
        noteBefore,
        'a late failure is not shown',
      );
      toast.remove();
      send(win, {type: 'signinDone', email: 'other@example.com'});
      assert.strictEqual(
        win.document.querySelector(
          '.kiss-notification[data-notification-id="signin"]',
        ),
        null,
        'a late signinDone raises no toast',
      );
    } finally {
      win.close();
    }
  }

  // `signinError` keeps the window open, closes the still-blank tab and
  // says why; the buttons work again.  A window.open that yields nothing
  // (blocked) falls back to the link in the note.
  {
    const {win, posted, tabs} = signinPage();
    try {
      send(win, SIGNIN);
      assert.strictEqual(buttons(win).length, 2, 'buttons');
      buttons(win)[1].click();
      const apple = sent(posted, 'signinLogin').pop();
      assert.strictEqual(apple.provider, 'apple');
      assert.ok(apple.attempt, 'the click carries an attempt token');
      send(win, {
        type: 'signinError',
        provider: 'apple',
        error: 'the hub is unreachable',
      });
      assert.strictEqual(
        note(win).textContent,
        'Sign-in failed: the hub is unreachable',
        'the error is shown',
      );
      assert.ok(note(win).classList.contains('signin-modal-error'));
      assert.strictEqual(
        modal(win).style.display,
        'flex',
        'the window stays open',
      );
      assert.ok(tabs[0].closed, 'the blank tab is closed');

      // A second click (the other provider) opens a fresh tab.
      buttons(win)[0].click();
      assert.strictEqual(tabs.length, 2);
      assert.ok(!note(win).classList.contains('signin-modal-error'));
      send(win, {type: 'signinError', provider: 'google'});
      assert.strictEqual(
        note(win).textContent,
        'Sign-in failed: unknown error',
        'an error without a reason',
      );

      // A tab already on the sign-in page (another origin: its location
      // cannot be read) is left open by a failure and by a retry, which
      // opens a fresh tab.
      buttons(win)[0].click();
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/2',
      });
      assert.strictEqual(
        tabs[2].navigatedTo,
        'https://accounts.example/2',
        'navigated',
      );
      assert.throws(() => tabs[2].location.href, /SecurityError/);
      send(win, {type: 'signinError', provider: 'google', error: 'expired'});
      assert.strictEqual(
        note(win).textContent,
        'Sign-in failed: expired',
        'error shown',
      );
      assert.ok(!tabs[2].closed, 'the navigated tab is not touched');
      buttons(win)[1].click();
      assert.strictEqual(tabs.length, 4, 'the retry opens a fresh tab');
      assert.strictEqual(sent(posted, 'signinLogin').pop().provider, 'apple');
      assert.strictEqual(
        new Set(sent(posted, 'signinLogin').map(m => m.attempt)).size,
        sent(posted, 'signinLogin').length,
        'every click has a token of its own',
      );

      // window.open blocked: the note's link is the way to the page.
      win.open = () => null;
      buttons(win)[0].click();
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/',
      });
      assert.ok(
        note(win).textContent.includes('page this link opens'),
        'the link-only note',
      );
      assert.strictEqual(
        note(win).querySelector('a').href,
        'https://accounts.example/',
      );

      // window.open throwing is survived too.
      win.open = () => {
        throw new Error('blocked');
      };
      buttons(win)[1].click();
      assert.strictEqual(sent(posted, 'signinLogin').pop().provider, 'apple');
    } finally {
      win.close();
    }
  }

  // "Not now" hides the window, closes the blank tab and ends the
  // daemon's sign-in; a paste then has nowhere to go (the daemon says so
  // with a `signinError`, which the closed window ignores).
  {
    const {win, posted, tabs} = signinPage();
    try {
      send(win, SIGNIN);
      assert.strictEqual(buttons(win).length, 2, 'buttons');
      buttons(win)[0].click();
      win.document.getElementById('signin-modal-later').click();
      assert.strictEqual(modal(win).style.display, 'none');
      assert.deepStrictEqual(sent(posted, 'signinCancel'), [
        {type: 'signinCancel'},
      ]);
      assert.ok(tabs[0].closed, 'the blank tab is closed');
      // A later `signin` (the next page load) shows the window afresh.
      send(win, SIGNIN);
      assert.strictEqual(modal(win).style.display, 'flex', 'shown again');
      assert.strictEqual(note(win).textContent, '');
      assert.strictEqual(pasteRow(win).style.display, 'none');
      assert.strictEqual(
        buttons(win).length,
        2,
        'the buttons are rebuilt, not stacked',
      );
    } finally {
      win.close();
    }
  }

  // 2026-11-04 W1 F1: a failure of the Apple click still in transit when
  // the Google button is clicked belongs to the ended attempt: it must
  // not close the new blank tab or hide Google's page when it arrives.
  {
    const {win, posted, tabs} = signinPage();
    try {
      send(win, SIGNIN);
      buttons(win)[1].click();
      buttons(win)[0].click();
      assert.strictEqual(sent(posted, 'signinLogin').length, 2);
      send(win, {type: 'signinError', provider: 'apple', error: 'stale'});
      assert.ok(!tabs[1].closed, "the newer click's tab stays open");
      assert.ok(
        !note(win).classList.contains('signin-modal-error'),
        'the replaced attempt has no failure to report',
      );
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/g',
      });
      assert.strictEqual(tabs[1].navigatedTo, 'https://accounts.example/g');
      assert.strictEqual(pasteRow(win).style.display, '', 'paste row shown');
      // The live attempt's own failure is still reported.
      send(win, {type: 'signinError', provider: 'google', error: 'denied'});
      assert.strictEqual(note(win).textContent, 'Sign-in failed: denied');
    } finally {
      win.close();
    }
  }

  // 2026-11-05 A18-3: a retry of the SAME provider.  The first click's
  // failure, already on its way when the user clicked again, carries the
  // first click's attempt token: it must not close the retry's blank tab
  // or make the retry's page, when it arrives, go unshown.  The retry's
  // own answers (its token) are acted on; a failure echoing no token (an
  // older daemon, a paste's failure) still goes by the provider.
  {
    const {win, posted, tabs} = signinPage();
    try {
      send(win, SIGNIN);
      buttons(win)[0].click();
      buttons(win)[0].click();
      const [first, retry] = sent(posted, 'signinLogin');
      assert.strictEqual(first.provider, 'google');
      assert.strictEqual(retry.provider, 'google');
      assert.ok(first.attempt && retry.attempt, 'both clicks carry a token');
      assert.notStrictEqual(first.attempt, retry.attempt, 'distinct tokens');
      assert.ok(tabs[0].closed, "the first click's blank tab is closed");
      assert.ok(!tabs[1].closed);

      send(win, {
        type: 'signinError',
        provider: 'google',
        attempt: first.attempt,
        error: 'stale',
      });
      assert.ok(!tabs[1].closed, "the retry's tab stays open");
      assert.ok(
        !note(win).classList.contains('signin-modal-error'),
        'the replaced attempt has no failure to report',
      );
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        attempt: first.attempt,
        url: 'https://accounts.example/stale',
      });
      assert.strictEqual(tabs[1].navigatedTo, 'about:blank', 'stale URL ignored');
      assert.strictEqual(pasteRow(win).style.display, 'none');

      send(win, {
        type: 'signinUrl',
        provider: 'google',
        attempt: retry.attempt,
        url: 'https://accounts.example/retry',
      });
      assert.strictEqual(tabs[1].navigatedTo, 'https://accounts.example/retry');
      assert.strictEqual(pasteRow(win).style.display, '', 'paste row shown');
      const input = win.document.getElementById('signin-modal-paste-input');
      input.value = 'ABCD-1234';
      win.document.getElementById('signin-modal-paste-btn').click();
      assert.deepStrictEqual(sent(posted, 'signinPaste'), [
        {type: 'signinPaste', text: 'ABCD-1234', attempt: retry.attempt},
      ]);
      // The paste's failure names no provider and echoes the retry's token.
      send(win, {type: 'signinError', attempt: retry.attempt, error: 'denied'});
      assert.strictEqual(note(win).textContent, 'Sign-in failed: denied');
      assert.ok(!tabs[1].closed, 'the navigated tab is not touched');

      // An older daemon echoes no token: the provider decides, as before.
      buttons(win)[0].click();
      send(win, {type: 'signinError', provider: 'apple', error: 'other'});
      assert.ok(!note(win).classList.contains('signin-modal-error'));
      send(win, {type: 'signinError', provider: 'google', error: 'mine'});
      assert.strictEqual(note(win).textContent, 'Sign-in failed: mine');
      assert.ok(tabs[2].closed, 'the blank tab of the failed click is closed');
    } finally {
      win.close();
    }
  }

  console.log('signinWindow.test.js passed');
}

runTests();

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

/* global require, console, process, setTimeout */

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness');

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

async function runTests() {
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
      await waitFor(
        () => modal(win).style.display === 'flex',
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
      // to the composer behind it.
      buttons(win)[0].focus();
      send(win, {type: 'focusInput'});
      await new Promise(r => setTimeout(r, 450));
      assert.strictEqual(
        win.document.activeElement,
        buttons(win)[0],
        'focusInput leaves the keyboard on the sign-in window',
      );
      // Once dismissed, a missing title falls back to "Sign in".
      win.document.getElementById('signin-modal-later').click();
      send(win, {type: 'signin', providers: [{id: 'x', label: 'X'}]});
      await waitFor(
        () =>
          win.document.getElementById('signin-modal-title').textContent ===
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
      await waitFor(() => buttons(win).length === 2, 'buttons');
      buttons(win)[0].click();
      assert.strictEqual(
        tabs.length,
        1,
        'the click opens a tab before the URL is known',
      );
      assert.strictEqual(tabs[0].navigatedTo, 'about:blank');
      assert.deepStrictEqual(sent(posted, 'signinLogin'), [
        {type: 'signinLogin', provider: 'google'},
      ]);
      assert.ok(note(win).textContent.includes('Starting the sign-in'));

      // A URL of an earlier, replaced click (another provider) is ignored.
      send(win, {
        type: 'signinUrl',
        provider: 'apple',
        url: 'https://stale.example/',
      });
      await new Promise(r => setTimeout(r, 20));
      assert.strictEqual(tabs[0].navigatedTo, 'about:blank');
      assert.strictEqual(pasteRow(win).style.display, 'none');

      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/auth?x=1',
      });
      await waitFor(
        () => tabs[0].navigatedTo === 'https://accounts.example/auth?x=1',
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
      await waitFor(
        () => modal(win).style.display === 'none',
        'signinDone closes the window',
      );
      await waitFor(
        () =>
          win.document.querySelector(
            '.kiss-notification[data-notification-id="signin"]',
          ),
        'a toast names the account',
      );
      assert.ok(
        win.document
          .querySelector('.kiss-notification[data-notification-id="signin"]')
          .textContent.includes('Signed in as me@example.com'),
      );
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
      await new Promise(r => setTimeout(r, 20));
      assert.strictEqual(modal(win).style.display, 'none');
      assert.strictEqual(tabs.length, 1);
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
      await waitFor(() => buttons(win).length === 2, 'buttons');
      buttons(win)[1].click();
      assert.deepStrictEqual(sent(posted, 'signinLogin').pop(), {
        type: 'signinLogin',
        provider: 'apple',
      });
      send(win, {
        type: 'signinError',
        provider: 'apple',
        error: 'the hub is unreachable',
      });
      await waitFor(
        () =>
          note(win).textContent === 'Sign-in failed: the hub is unreachable',
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
      await waitFor(
        () => note(win).textContent === 'Sign-in failed: unknown error',
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
      await waitFor(
        () => tabs[2].navigatedTo === 'https://accounts.example/2',
        'navigated',
      );
      assert.throws(() => tabs[2].location.href, /SecurityError/);
      send(win, {type: 'signinError', provider: 'google', error: 'expired'});
      await waitFor(
        () => note(win).textContent === 'Sign-in failed: expired',
        'error shown',
      );
      assert.ok(!tabs[2].closed, 'the navigated tab is not touched');
      buttons(win)[1].click();
      assert.strictEqual(tabs.length, 4, 'the retry opens a fresh tab');
      assert.deepStrictEqual(sent(posted, 'signinLogin').pop(), {
        type: 'signinLogin',
        provider: 'apple',
      });

      // window.open blocked: the note's link is the way to the page.
      win.open = () => null;
      buttons(win)[0].click();
      send(win, {
        type: 'signinUrl',
        provider: 'google',
        url: 'https://accounts.example/',
      });
      await waitFor(
        () => note(win).textContent.includes('page this link opens'),
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
      assert.deepStrictEqual(sent(posted, 'signinLogin').pop(), {
        type: 'signinLogin',
        provider: 'apple',
      });
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
      await waitFor(() => buttons(win).length === 2, 'buttons');
      buttons(win)[0].click();
      win.document.getElementById('signin-modal-later').click();
      assert.strictEqual(modal(win).style.display, 'none');
      assert.deepStrictEqual(sent(posted, 'signinCancel'), [
        {type: 'signinCancel'},
      ]);
      assert.ok(tabs[0].closed, 'the blank tab is closed');
      // A later `signin` (the next page load) shows the window afresh.
      send(win, SIGNIN);
      await waitFor(() => modal(win).style.display === 'flex', 'shown again');
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

  console.log('signinWindow.test.js passed');
}

runTests().catch(err => {
  console.error(err);
  process.exit(1);
});

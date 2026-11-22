// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests, against the real chat.html + api.js +
// main.js, of the Create pull request button of the task-info strip
// (#create-pr-btn): a press asks the daemon to push the branch of the
// chat's folder and open a pull request for it (`createPrAction`, on
// the machine and in the folder the chat header shows); the button is
// disabled and says why until the press ends (the daemon's `pr_done`,
// a dropped connection, or its failsafe); the outcome is a toast with
// a button that opens the link when there is one.  The daemon's side
// is pinned by tests/server/test_create_pr_action.py.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  byId,
  ofType,
  plain,
  openChat,
  captureTimers,
  fireCapturedTimers,
} = require('./simplify2_harness');
const {makeRunner, toast, toastButton} = require('./ui_antipattern_harness');
const {bootOnGpuBox} = require('./picked_machine_helpers');

const windows = [];
const {test, report} = makeRunner({windows});

const IDLE = 'Create pull request';
const BUSY = 'A pull request is being created';
const PR_URL = 'https://github.com/acme/app/pull/7';
const FAILSAFE_MS = 300000;

/**
 * A remote chat page; `chat` shows the idle chat `a1` of the folder
 * /ws on it.  `timers` are the timeouts the page armed, `opened` what
 * it handed to window.open.
 */
function makePage({chat = true} = {}) {
  const timers = [];
  const opened = [];
  const page = makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      captureTimers(win, timers);
      win.open = (...args) => opened.push(args);
    },
  });
  windows.push(page.win);
  if (chat) openChat(page.win);
  return Object.assign(page, {timers, opened});
}

/** Press the button as a pointer does. */
function press(win) {
  click(win, byId(win, 'create-pr-btn'));
}

/** Assert the button is armed, or (`busy`) waiting, and says which. */
function assertButton(win, busy) {
  const btn = byId(win, 'create-pr-btn');
  const name = busy ? BUSY : IDLE;
  assert.strictEqual(btn.disabled, busy);
  assert.strictEqual(btn.classList.contains('is-busy'), busy);
  assert.strictEqual(btn.getAttribute('aria-label'), name);
  assert.strictEqual(btn.dataset.tooltip, name);
  // The browser's own tooltip while disabled only: the page's opens on
  // pointer events, which a disabled button may not get.
  assert.strictEqual(btn.getAttribute('title'), busy ? BUSY : null);
}

/** The toasts of the notification `id` on the page. */
function toastsOf(win, id) {
  return Array.from(
    win.document.querySelectorAll('.kiss-notification'),
  ).filter(el => el.dataset.notificationId === id);
}

async function main() {
  await test('a press asks for the chat folder and waits for pr_done', () => {
    const {win, posted, opened, timers} = makePage();
    assertButton(win, false);
    const armed = timers.length;
    press(win);
    assert.deepStrictEqual(plain(ofType(posted, 'createPrAction')), [
      {type: 'createPrAction', tabId: 'a1', workDir: '/ws'},
    ]);
    assertButton(win, true);
    const failsafe = timers.slice(armed).filter(t => t.ms === FAILSAFE_MS);
    assert.strictEqual(failsafe.length, 1, 'one failsafe per press');

    // The daemon's progress toast, then the outcome in its place.
    send(win, {
      type: 'notification',
      id: 'create-pr:a1',
      message: 'Pushing feature\u2026',
      severity: 'info',
    });
    assert.match(toast(win, 'create-pr:a1').textContent, /Pushing feature/);
    send(win, {
      type: 'pr_done',
      tabId: 'a1',
      success: true,
      created: true,
      url: PR_URL,
      message: 'Pull request created: ' + PR_URL,
    });
    assertButton(win, false);
    assert.ok(failsafe[0].cleared, 'the end of the press cancels the failsafe');
    const shown = toastsOf(win, 'create-pr:a1');
    assert.strictEqual(shown.length, 1, 'the outcome replaces the progress');
    assert.ok(shown[0].classList.contains('kiss-notification-info'));
    assert.match(shown[0].textContent, /Pull request created/);
    assert.doesNotMatch(shown[0].textContent, /Pushing feature/);
    // A click is what lets the browser open a new tab.
    assert.deepStrictEqual(opened, []);
    click(win, toastButton(shown[0], 'Open pull request'));
    assert.deepStrictEqual(opened, [[PR_URL, '_blank', 'noopener']]);
  });

  await test('the outcome: a page to finish on, or an error without a link', () => {
    const {win, opened} = makePage();
    const compare = 'https://github.com/acme/app/compare/main...feature';
    send(win, {
      type: 'pr_done',
      tabId: 'a1',
      success: true,
      created: false,
      url: compare,
      message: 'Pushed feature. Open the pull request page to finish.',
    });
    const pushed = toast(win, 'create-pr:a1');
    assert.strictEqual(toastButton(pushed, 'Open pull request'), null);
    click(win, toastButton(pushed, 'Open page'));
    assert.deepStrictEqual(opened, [[compare, '_blank', 'noopener']]);

    send(win, {
      type: 'pr_done',
      tabId: 'a1',
      success: false,
      created: false,
      url: '',
      message: 'This repository has no remote to push to.',
    });
    const failed = toast(win, 'create-pr:a1');
    assert.ok(failed.classList.contains('kiss-notification-error'));
    assert.strictEqual(failed.dataset.notificationSticky, 'true');
    assert.match(failed.textContent, /no remote to push to/);
    assert.strictEqual(
      failed.querySelectorAll('.kiss-notification-action').length,
      0,
      'nothing to open',
    );
  });

  await test('only the end of this press re-arms the button', () => {
    const {win, posted, timers} = makePage();
    press(win);
    // Another chat's pull request, asked for on another surface, ends
    // nothing here; its outcome is shown all the same.
    send(win, {
      type: 'pr_done',
      tabId: 'b2',
      success: true,
      created: false,
      url: '',
      message: 'Nothing to open a pull request for.',
    });
    assertButton(win, true);
    assert.ok(toast(win, 'create-pr:b2'));
    // An answer that names no tab ends the press in flight.
    send(win, {type: 'pr_done', success: false, message: 'Refused.'});
    assertButton(win, false);
    assert.ok(toast(win, 'create-pr:'));

    // A dropped connection never brings the daemon's pr_done.
    press(win);
    assertButton(win, true);
    send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    assertButton(win, false);
    send(win, {type: 'daemonStatus', connected: true, reconnecting: true});

    // Nor does a daemon that died: the button re-arms on its own.
    const armed = timers.length;
    press(win);
    assertButton(win, true);
    assert.strictEqual(ofType(posted, 'createPrAction').length, 3);
    fireCapturedTimers(
      win,
      timers.slice(armed).filter(t => t.ms === FAILSAFE_MS),
    );
    assertButton(win, false);
  });

  await test('a chat on a registered machine opens the pull request there', () => {
    const {win, posted} = bootOnGpuBox();
    windows.push(win);
    press(win);
    assert.deepStrictEqual(plain(ofType(posted, 'createPrAction')), [
      {
        type: 'createPrAction',
        tabId: win._testApi.getActiveTabId(),
        workDir: '/srv/proj',
        machine: 'gpu-box',
      },
    ]);
  });

  await test('without a chat tab the press says so and asks nothing', () => {
    const {win, posted} = makePage({chat: false});
    // A report stays on screen while its chat is replaced, and the
    // replacement closed, on another surface: no chat is left to open
    // a pull request for.
    send(win, {
      type: 'fileContent',
      path: '/ws/report.html',
      name: 'report.html',
      content: '<!DOCTYPE html><html><body>report</body></html>',
    });
    const report = win._testApi.getActiveTabId();
    send(win, {
      type: 'tabs_state',
      tabs: [{tabId: 'b2', chatId: 'c2', title: 'b2', workDir: '/ws'}],
    });
    send(win, {type: 'tabs_state', tabs: []});
    assert.strictEqual(win._testApi.getActiveTabId(), report);
    press(win);
    assert.deepStrictEqual(ofType(posted, 'createPrAction'), []);
    assertButton(win, false);
    const warnings = Array.from(
      win.document.querySelectorAll('.kiss-notification-warning'),
      el => el.textContent,
    );
    assert.strictEqual(warnings.length, 1);
    assert.match(warnings[0], /Create pull request needs a chat tab/);
  });

  report('createPrButton');
}

main();

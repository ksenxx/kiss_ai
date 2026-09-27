// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for anti-pattern fix A3 in media/main.js:
// error notifications never dismiss themselves (the close button still
// works), a textless Explorer / Git failure names the action and its
// target, and a dropped server connection reports the in-flight
// Explorer / Git action it discarded.
'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

const FILE_ENTRIES = [
  {name: 'foo.py', path: h.WD + '/foo.py', isDir: false},
  {name: 'src', path: h.WD + '/src', isDir: true},
];

function messageOf(t) {
  return t.querySelector('.kiss-notification-message').textContent;
}

async function main() {
  await test('an error notification is sticky; info and warning are not', () => {
    const {win} = h.makeWebview();
    h.send(win, {
      type: 'notification',
      id: 'err-1',
      message: 'Something broke',
      severity: 'error',
    });
    const err = h.toast(win, 'err-1');
    assert.ok(err, 'the error toast renders');
    assert.strictEqual(
      err.getAttribute('data-notification-sticky'),
      'true',
      'an error must not auto-dismiss',
    );
    h.send(win, {
      type: 'notification',
      id: 'info-1',
      message: 'FYI',
      severity: 'info',
    });
    assert.strictEqual(
      h.toast(win, 'info-1').getAttribute('data-notification-sticky'),
      'false',
      'an info toast still dismisses itself',
    );
    h.send(win, {
      type: 'notification',
      id: 'warn-1',
      message: 'Hm',
      severity: 'warning',
    });
    assert.strictEqual(
      h.toast(win, 'warn-1').getAttribute('data-notification-sticky'),
      'false',
    );
    win.close();
  });

  await test('an error toast survives the old 7.5 s timeout and closes on its X', async () => {
    const {win} = h.makeWebview();
    // Speed the auto-dismiss clock up so the wait is short: a fake
    // setTimeout that fires everything 1000x sooner.
    const realSetTimeout = win.setTimeout;
    win.setTimeout = function (fn, ms) {
      return realSetTimeout.call(win, fn, Math.max(1, (ms || 0) / 1000));
    };
    h.send(win, {
      type: 'notification',
      id: 'err-2',
      message: 'Disk full',
      severity: 'error',
    });
    h.send(win, {
      type: 'notification',
      id: 'info-2',
      message: 'ok',
      severity: 'info',
    });
    await h.sleep(60);
    assert.strictEqual(
      h.toast(win, 'info-2'),
      null,
      'info went away on its own',
    );
    const err = h.toast(win, 'err-2');
    assert.ok(err, 'the error is still on screen');
    const close = err.querySelector('.kiss-notification-close');
    assert.ok(close, 'it keeps a close button');
    h.click(win, close);
    assert.strictEqual(h.toast(win, 'err-2'), null, 'the X closes it');
    win.close();
  });

  await test('a textless Delete failure reads "Could not delete \'foo.py\'."', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    h.click(win, h.toastButton(h.toast(win, 'fs-delete'), 'Delete'));
    const req = h.ofType(posted, 'fsAction')[0];
    h.send(win, {type: 'fsResult', token: req.token, error: true});
    const t = h.toast(win, 'sidebar-action-error');
    assert.ok(t, 'the failure is reported');
    assert.strictEqual(messageOf(t), "Could not delete 'foo.py'.");
    assert.strictEqual(t.getAttribute('data-notification-sticky'), 'true');
    win.close();
  });

  await test('a daemon error text is shown verbatim when present', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    h.click(win, h.toastButton(h.toast(win, 'fs-delete'), 'Delete'));
    const req = h.ofType(posted, 'fsAction')[0];
    h.send(win, {
      type: 'fsResult',
      token: req.token,
      error: 'Permission denied',
    });
    assert.strictEqual(
      messageOf(h.toast(win, 'sidebar-action-error')),
      'Permission denied',
    );
    win.close();
  });

  await test('a blank Find in Folder failure names the folder searched', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/src'), 'Find in Folder...');
    const input = h.toastInput(h.toast(win, 'find-in-folder'));
    input.value = 'needle';
    h.key(win, input, 'Enter');
    const req = h.ofType(posted, 'fsAction')[0];
    h.send(win, {type: 'fsResult', token: req.token, error: '   '});
    assert.strictEqual(
      messageOf(h.toast(win, 'sidebar-action-error')),
      "Could not search 'src'.",
    );
    win.close();
  });

  await test('a textless Create Branch failure names the branch', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Create Branch...');
    const t = h.toast(win, 'git-create-branch');
    h.toastInput(t).value = 'feat';
    h.key(win, h.toastInput(t), 'Enter');
    const req = h.ofType(posted, 'gitAction')[0];
    h.send(win, {
      type: 'gitActionResult',
      token: req.token,
      action: 'createBranch',
      error: true,
    });
    assert.strictEqual(
      messageOf(h.toast(win, 'sidebar-action-error')),
      "Could not create the branch 'feat'.",
    );
    win.close();
  });

  await test('a textless Compare failure names the commit shown', () => {
    const {win, posted} = h.makeWebview();
    const row = h.openScm(win, posted);
    h.runMenuItem(win, row, 'Compare with...');
    const t = h.toast(win, 'git-compare-with');
    h.click(win, h.toastButton(t, 'Compare'));
    const req = h.ofType(posted, 'gitShow')[0];
    h.send(win, {type: 'gitShow', token: req.token, error: true});
    assert.strictEqual(
      messageOf(h.toast(win, 'sidebar-action-error')),
      "Could not show '" + h.SHA_A.slice(0, 7) + "'.",
    );
    win.close();
  });

  await test('a dropped connection reports the pending Explorer action once', () => {
    const {win, posted} = h.makeWebview();
    h.openExplorer(win, posted, FILE_ENTRIES);
    h.runMenuItem(win, h.explorerRow(win, h.WD + '/foo.py'), 'Delete');
    h.click(win, h.toastButton(h.toast(win, 'fs-delete'), 'Delete'));
    assert.strictEqual(h.ofType(posted, 'fsAction').length, 1, 'in flight');
    h.send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
    const t = h.toast(win, 'sidebar-action-error');
    assert.ok(t, 'the dropped request is reported');
    assert.strictEqual(
      messageOf(t),
      'The server connection dropped; the pending Explorer/Git action was ' +
        'not completed. Try again once it reconnects.',
    );
    assert.strictEqual(t.getAttribute('data-notification-sticky'), 'true');
    assert.strictEqual(
      h.all(win, '[data-notification-id="sidebar-action-error"]').length,
      1,
      'exactly one toast',
    );
    // The late reply to the dropped request must find no taker: no
    // second toast, no stale handling.
    const req = h.ofType(posted, 'fsAction')[0];
    h.send(win, {type: 'fsResult', token: req.token, error: 'late'});
    assert.strictEqual(
      messageOf(h.toast(win, 'sidebar-action-error')).indexOf('late'),
      -1,
    );
    win.close();

    // With nothing pending, a drop says nothing about the sidebar.
    const quiet = h.makeWebview();
    h.openExplorer(quiet.win, quiet.posted, FILE_ENTRIES);
    h.send(quiet.win, {
      type: 'daemonStatus',
      connected: false,
      reconnecting: true,
    });
    assert.strictEqual(h.toast(quiet.win, 'sidebar-action-error'), null);
    quiet.win.close();
  });

  report('ui_antipattern_notifications');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

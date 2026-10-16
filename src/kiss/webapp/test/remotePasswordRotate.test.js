// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';
// Settings > Remote password > "Rotate" (main.js against the real
// chat.html):
//   * the button sits beside the password box inside the Settings
//     panel and asks before doing anything (an in-page confirm toast);
//   * confirming posts `rotateRemotePassword`; cancelling posts nothing;
//   * the daemon's `remotePasswordRotated` puts the new password in the
//     box, reveals it (the eye toggle flips to "hide") and writes the
//     note under it; the `configData` that follows keeps the value;
//   * an `error` with code `rotate_remote_password_failed` paints the
//     note red with the daemon's text;
//   * reopening Settings clears the note.
// The transport's adoption of the new password for its own frames is
// covered by transportFramePassword.test.js; the daemon side by
// src/kiss/tests/server/test_rotate_remote_password.py.

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');
const {JSDOM} = require('jsdom');

const TAB_ID = 'tab-rotate-password-test';

function makeDomPage() {
  const mediaDir = path.join(__dirname, '..', 'media');
  let html = fs.readFileSync(path.join(mediaDir, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  const dom = new JSDOM(html, {
    runScripts: 'outside-only',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  const posted = [];
  let state = {
    tabs: [{title: 'new chat', chatId: TAB_ID, backendChatId: '', parentTabId: ''}],
    activeTabIndex: 0,
    chatId: TAB_ID,
  };
  win.acquireTransport = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  for (const name of ['panelCopy.js', 'api.js', 'main.js']) {
    vm.runInContext(
      fs.readFileSync(path.join(mediaDir, name), 'utf8'),
      dom.getInternalVMContext(),
    );
  }
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: TAB_ID, chatId: '', title: 'new chat', workDir: ''}],
  });
  return {win, posted, close: () => win.close()};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

function byId(win, id) {
  const el = win.document.getElementById(id);
  assert.ok(el, '#' + id + ' must exist');
  return el;
}

function confirmToast(win) {
  return win.document.querySelector(
    '.kiss-notification[data-notification-id="rotate-remote-password"]',
  );
}

function toastButton(toast, label) {
  const buttons = Array.from(toast.querySelectorAll('.kiss-notification-action'));
  const btn = buttons.find(b => b.textContent.trim() === label);
  assert.ok(btn, 'the confirm toast must offer "' + label + '"');
  return btn;
}

function openSettings(wv) {
  click(byId(wv.win, 'settings-btn'));
  assert.ok(
    wv.posted.some(m => m.type === 'getConfig'),
    'opening Settings must request the config',
  );
  send(wv.win, {
    type: 'configData',
    config: {remote_password: 'Old-Password-1', max_budget: 5},
  });
}

function rotatePosts(wv) {
  return wv.posted.filter(m => m.type === 'rotateRemotePassword').length;
}

async function runTests() {
  // Layout: the button is beside the box, inside the Settings panel;
  // the note starts hidden; the box shows the stored password masked.
  {
    const wv = makeDomPage();
    try {
      openSettings(wv);
      const btn = byId(wv.win, 'cfg-rotate-password-btn');
      const box = byId(wv.win, 'cfg-remote-password');
      const note = byId(wv.win, 'cfg-remote-password-note');
      assert.ok(byId(wv.win, 'settings-panel').contains(btn));
      assert.strictEqual(
        btn.closest('.config-remote-password-row'),
        box.closest('.config-remote-password-row'),
        'button and box share the row',
      );
      assert.strictEqual(btn.textContent.trim(), 'Rotate');
      assert.strictEqual(box.value, 'Old-Password-1');
      assert.strictEqual(box.type, 'password');
      assert.ok(box.readOnly, 'the box stays read-only');
      assert.strictEqual(note.hidden, true);
      assert.strictEqual(confirmToast(wv.win), null);
    } finally {
      wv.close();
    }
  }

  // Click asks first; Cancel posts nothing; confirming posts the command
  // once and shows "Rotating…".
  {
    const wv = makeDomPage();
    try {
      openSettings(wv);
      const btn = byId(wv.win, 'cfg-rotate-password-btn');
      click(btn);
      let toast = confirmToast(wv.win);
      assert.ok(toast, 'clicking Rotate opens the confirm toast');
      assert.ok(/sign in again/.test(toast.textContent), toast.textContent);
      assert.strictEqual(rotatePosts(wv), 0, 'asking posts nothing');
      click(toastButton(toast, 'Cancel'));
      assert.strictEqual(confirmToast(wv.win), null, 'Cancel closes the toast');
      assert.strictEqual(rotatePosts(wv), 0, 'Cancel posts nothing');
      assert.strictEqual(byId(wv.win, 'cfg-remote-password-note').hidden, true);

      click(btn);
      toast = confirmToast(wv.win);
      click(toastButton(toast, 'Rotate password'));
      assert.strictEqual(rotatePosts(wv), 1, 'confirming posts once');
      assert.strictEqual(confirmToast(wv.win), null);
      const note = byId(wv.win, 'cfg-remote-password-note');
      assert.strictEqual(note.hidden, false);
      assert.strictEqual(note.textContent, 'Rotating…');
      assert.ok(!note.classList.contains('config-field-note-error'));

      // The daemon's answer: new password shown in the clear + note.
      send(wv.win, {type: 'remotePasswordRotated', password: 'New-Password-2'});
      const box = byId(wv.win, 'cfg-remote-password');
      assert.strictEqual(box.value, 'New-Password-2');
      assert.strictEqual(box.type, 'text', 'the new password is revealed');
      const eye = byId(wv.win, 'cfg-remote-password-toggle');
      assert.ok(/hide/i.test(eye.getAttribute('aria-label') || ''), 'eye says hide');
      assert.ok(/New password set/.test(note.textContent), note.textContent);
      assert.ok(/shown above/.test(note.textContent), note.textContent);
      assert.ok(!note.classList.contains('config-field-note-error'));
      // The refreshed configData that follows keeps it.
      send(wv.win, {
        type: 'configData',
        config: {remote_password: 'New-Password-2', max_budget: 5},
      });
      assert.strictEqual(box.value, 'New-Password-2');
      assert.strictEqual(box.type, 'text');
      assert.strictEqual(note.hidden, false);

      // A second rotation while already revealed does not flip the eye back.
      send(wv.win, {type: 'remotePasswordRotated', password: 'New-Password-3'});
      assert.strictEqual(box.value, 'New-Password-3');
      assert.strictEqual(box.type, 'text');

      // Reopening Settings clears the note (the box is repainted by
      // the configData the open requests).
      click(byId(wv.win, 'settings-panel-close'));
      openSettings(wv);
      assert.strictEqual(note.hidden, true);
      assert.strictEqual(note.textContent, '');
    } finally {
      wv.close();
    }
  }

  // The daemon could not store a new password: red note, box unchanged.
  {
    const wv = makeDomPage();
    try {
      openSettings(wv);
      click(byId(wv.win, 'cfg-rotate-password-btn'));
      click(toastButton(confirmToast(wv.win), 'Rotate password'));
      send(wv.win, {
        type: 'error',
        code: 'rotate_remote_password_failed',
        text: 'Could not rotate the remote password: config.json cannot be read',
      });
      const note = byId(wv.win, 'cfg-remote-password-note');
      assert.strictEqual(note.hidden, false);
      assert.ok(/cannot be read/.test(note.textContent), note.textContent);
      assert.ok(note.classList.contains('config-field-note-error'));
      const box = byId(wv.win, 'cfg-remote-password');
      assert.strictEqual(box.value, 'Old-Password-1');
      assert.strictEqual(box.type, 'password');
      // An unrelated error leaves the note alone.
      send(wv.win, {type: 'error', code: 'other', text: 'something else'});
      assert.ok(/cannot be read/.test(note.textContent));
    } finally {
      wv.close();
    }
  }

  // An event without a usable password still reports, with an empty box.
  {
    const wv = makeDomPage();
    try {
      openSettings(wv);
      send(wv.win, {type: 'remotePasswordRotated'});
      const box = byId(wv.win, 'cfg-remote-password');
      assert.strictEqual(box.value, '');
      assert.strictEqual(box.type, 'text');
      assert.strictEqual(byId(wv.win, 'cfg-remote-password-note').hidden, false);
    } finally {
      wv.close();
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

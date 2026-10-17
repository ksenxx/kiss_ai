// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const {makePreloadedPage, send, waitFor} = require('./simplify2_harness');

const PRELOADED_TAB_ID = 'tab-server-reset-test';

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {bubbles: true}),
  );
}

function isFloatingModalOpen(win) {
  const modal = win.document.getElementById('server-reset-confirm-modal');
  return !!(modal && modal.classList.contains('open'));
}

async function runTests() {
  {
    const wv = makePreloadedPage(PRELOADED_TAB_ID);
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      assert.ok(btn, 'settings panel must expose #cfg-server-reset-btn');

      click(btn);
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'no-agent click must post {type: "serverReset"} to the server',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'no-agent click must NOT open the floating confirmation box',
      );
      assert.strictEqual(
        wv.win.document.querySelectorAll('.kiss-notification').length,
        0,
        'the page must NOT render any in-page confirmation toast',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makePreloadedPage(PRELOADED_TAB_ID);
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });

      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'agent-running click must open the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'no serverReset post must be made until the user clicks OK',
      );

      click(wv.win.document.getElementById('server-reset-confirm-cancel'));
      await waitFor(
        () => !isFloatingModalOpen(wv.win),
        'Cancel must close the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'Cancel must NOT post serverReset',
      );

      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'a fresh click after Cancel must re-open the dialog',
      );
      click(wv.win.document.getElementById('server-reset-confirm-ok'));
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'OK must post serverReset',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'OK must close the floating confirmation box',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        1,
        'OK must produce exactly ONE serverReset post',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makePreloadedPage(PRELOADED_TAB_ID);
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: false,
      });

      click(btn);
      const msg = await waitFor(
        () => wv.posted.find(m => m.type === 'serverReset'),
        'click after the agent stops must fast-path serverReset',
      );
      assert.strictEqual(msg.type, 'serverReset');
      assert.strictEqual(
        isFloatingModalOpen(wv.win),
        false,
        'fast path must NOT open the floating dialog',
      );
    } finally {
      wv.close();
    }
  }

  {
    const wv = makePreloadedPage(PRELOADED_TAB_ID);
    try {
      const btn = wv.win.document.getElementById('cfg-server-reset-btn');
      send(wv.win, {
        type: 'status',
        tabId: PRELOADED_TAB_ID,
        running: true,
        startTs: Date.now(),
      });
      click(btn);
      await waitFor(
        () => isFloatingModalOpen(wv.win),
        'click must open the floating dialog',
      );
      wv.win.document.dispatchEvent(
        new wv.win.KeyboardEvent('keydown', {key: 'Escape', bubbles: true}),
      );
      await waitFor(
        () => !isFloatingModalOpen(wv.win),
        'Escape must close the floating dialog',
      );
      assert.strictEqual(
        wv.posted.filter(m => m.type === 'serverReset').length,
        0,
        'Escape must NOT post serverReset',
      );
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

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Two clients (two browser windows / VS Code webviews) showing the same
// chat must show the same blocking UI.  Tab ids are minted per client,
// so the daemon mirrors every ask-user / merge-review / auto-commit
// event: the owner tab gets the original and each other tab gets a copy
// stamped with ITS OWN tab id plus `mirrorOf`.  These tests feed two
// independent webviews exactly that pair of copies and check both
// windows open — and close — the UI together.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;

  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};
  win.requestAnimationFrame = function (cb) {
    cb();
    return 0;
  };

  const posted = [];
  win.acquireVsCodeApi = function () {
    let state;
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));

  return {win, posted, tabId: win._testApi.getActiveTabId()};
}

// The daemon's JsonPrinter.broadcast_tab_ui: one copy per tab showing
// the UI, each stamped with that tab's own id.
function mirror(event, clients) {
  const owner = clients[0];
  clients.forEach(client => {
    const copy = Object.assign({}, event, {tabId: client.tabId});
    if (client !== owner) copy.mirrorOf = owner.tabId;
    client.win.dispatchEvent(
      new client.win.MessageEvent('message', {data: copy}),
    );
  });
}

function mergeToolbar(client) {
  return client.win.document.getElementById('merge-toolbar');
}

function actionBarLabel(client) {
  const bar = client.win.document.querySelector('#input-area .wt-bar');
  return bar ? bar.querySelector('.wt-label').textContent : '';
}

function askText(client) {
  const modal = client.win.document.getElementById('ask-user-modal');
  if (!modal || modal.style.display !== 'flex') return '';
  return modal.textContent || '';
}

function mergeDataEvent() {
  return {
    type: 'merge_data',
    hunk_count: 1,
    data: {
      work_dir: '/repo',
      files: [
        {
          name: 'README.md',
          base: '/tmp/base/README.md',
          current: '/repo/README.md',
          base_text: 'old\n',
          current_text: 'new\n',
          hunks: [{baseStart: 1, baseCount: 1, curStart: 1, curCount: 1}],
        },
      ],
    },
  };
}

function withTwoClients(body) {
  const clients = [makeWebview(), makeWebview()];
  try {
    body(clients);
  } finally {
    clients.forEach(client => client.win.close());
  }
}

function testMergeReviewOpensAndClosesOnBothClients() {
  withTwoClients(clients => {
    mirror(mergeDataEvent(), clients);
    mirror({type: 'merge_started'}, clients);

    clients.forEach((client, i) => {
      assert.ok(
        mergeToolbar(client),
        `client ${i} must show the merge toolbar`,
      );
    });

    mirror({type: 'merge_ended'}, clients);
    clients.forEach((client, i) => {
      assert.strictEqual(
        mergeToolbar(client),
        null,
        `client ${i} must hide the merge toolbar when the review ends`,
      );
    });
  });
  console.log('  ok - merge review opens and closes on every client');
}

function testMergeActionFromTheMirroringClientIsAddressedToItself() {
  withTwoClients(clients => {
    mirror(mergeDataEvent(), clients);
    mirror({type: 'merge_started'}, clients);

    const viewer = clients[1];
    mergeToolbar(viewer).querySelector('#merge-accept-all-btn').click();

    const action = viewer.posted.find(msg => msg.type === 'mergeAction');
    assert.ok(action, 'the mirroring client must be able to act');
    assert.strictEqual(
      action.tabId,
      viewer.tabId,
      'the client addresses its own tab; the daemon maps it to the owner',
    );
  });
  console.log('  ok - a mirroring client can drive the review');
}

function testAutocommitPromptOpensAndClosesOnBothClients() {
  withTwoClients(clients => {
    mirror(
      {type: 'autocommit_prompt', changedFiles: ['README.md']},
      clients,
    );
    clients.forEach((client, i) => {
      assert.strictEqual(
        actionBarLabel(client),
        '1 uncommitted change on main. Auto commit?',
        `client ${i} must show the auto-commit prompt`,
      );
    });

    mirror(
      {
        type: 'autocommit_done',
        success: true,
        committed: true,
        message: 'Committed: chore',
      },
      clients,
    );
    clients.forEach((client, i) => {
      assert.strictEqual(
        actionBarLabel(client),
        '',
        `client ${i} must drop the auto-commit prompt once answered`,
      );
    });
  });
  console.log('  ok - auto-commit prompt opens and closes on every client');
}

function testAskUserOpensAndClosesOnBothClients() {
  withTwoClients(clients => {
    mirror({type: 'askUser', question: 'Proceed?'}, clients);
    clients.forEach((client, i) => {
      assert.ok(
        askText(client).includes('Proceed?'),
        `client ${i} must show the ask-user question`,
      );
    });

    mirror({type: 'askUserDone'}, clients);
    clients.forEach((client, i) => {
      assert.strictEqual(
        askText(client),
        '',
        `client ${i} must close the ask-user window once answered`,
      );
    });
  });
  console.log('  ok - ask-user window opens and closes on every client');
}

function runTests() {
  testMergeReviewOpensAndClosesOnBothClients();
  testMergeActionFromTheMirroringClientIsAddressedToItself();
  testAutocommitPromptOpensAndClosesOnBothClients();
  testAskUserOpensAndClosesOnBothClients();
}

try {
  runTests();
  console.log('4 passed, 0 failed');
} catch (err) {
  console.error(err && err.stack ? err.stack : String(err));
  console.error('failed');
  process.exit(1);
}

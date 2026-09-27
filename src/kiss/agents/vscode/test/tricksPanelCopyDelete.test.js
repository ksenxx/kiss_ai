// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
'use strict';

// End-to-end (JSDOM) tests for the Inject promptlet panel's per-row
// buttons: every row ends with a copy button that puts the promptlet on
// the clipboard, and the first `window.__MY_TRICKS_COUNT__` rows (the
// user's own, from ~/.kiss/MY_INJECTION.md) also carry a delete button
// that posts `deleteTrick` and drops the row at once.  Also covers the
// `tricksData` reply's `userCount` and the compiled `getTricksData()`
// that freezes the count into the VS Code page.

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');
const {makeWebview, send} = require('./simplify2_harness');

function makeUri(fsPath) {
  return {
    fsPath,
    toString() {
      return 'vscode-webview://kiss' + fsPath;
    },
  };
}

// `out/SorcarTab.js` imports `vscode`; the stub supplies the two members
// buildChatHtml touches (see buildChatHtmlTricksEscape.test.js).
global.__kissVscodeStub = {
  Uri: {
    joinPath(base, ...parts) {
      return makeUri(path.join(base.fsPath, ...parts));
    },
  },
  workspace: {
    isTrusted: false,
    getConfiguration() {
      return {get: () => undefined};
    },
  },
};
const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

function rows(doc) {
  return Array.from(doc.getElementById('tricks-list').children);
}

function texts(doc) {
  return rows(doc).map(el => el.querySelector('.sidebar-item-text').textContent);
}

function click(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true, cancelable: true}));
}

function input(win, el, value) {
  el.value = value;
  el.dispatchEvent(new win.Event('input', {bubbles: true}));
}

function flush() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

async function runPanel() {
  const {win, posted} = makeWebview();
  const doc = win.document;
  const clipboardWrites = [];
  Object.defineProperty(win.navigator, 'clipboard', {
    configurable: true,
    value: {
      writeText: text => {
        clipboardWrites.push(String(text));
        return Promise.resolve();
      },
    },
  });
  win.__TRICKS__ = ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta'];
  win.__MY_TRICKS_COUNT__ = 2;

  doc.getElementById('tricks-btn').click();
  const list = doc.getElementById('tricks-list');
  const composer = doc.getElementById('task-input');
  assert.deepStrictEqual(texts(doc), win.__TRICKS__, 'all promptlets listed');

  // --- every row has a copy button, at its right edge (after the text)
  for (const row of rows(doc)) {
    const copy = row.querySelectorAll('.sidebar-item-copy');
    assert.strictEqual(copy.length, 1, 'exactly one copy button per row');
    assert.strictEqual(copy[0].getAttribute('aria-label'), 'Copy promptlet to clipboard');
    assert.strictEqual(copy[0].dataset.tooltip, 'Copy promptlet');
    const text = row.querySelector('.sidebar-item-text');
    assert.ok(
      text.compareDocumentPosition(copy[0]) & win.Node.DOCUMENT_POSITION_FOLLOWING,
      'the copy button follows (sits right of) the promptlet text',
    );
  }

  // --- only the user-owned rows have a delete button, after the copy one
  const deletes = rows(doc).map(r => r.querySelectorAll('.sidebar-item-delete').length);
  assert.deepStrictEqual(deletes, [1, 1, 0, 0], 'delete only on the first userCount rows');
  const firstCopy = rows(doc)[0].querySelector('.sidebar-item-copy');
  const firstDelete = rows(doc)[0].querySelector('.sidebar-item-delete');
  assert.strictEqual(firstDelete.getAttribute('aria-label'), 'Delete promptlet');
  assert.strictEqual(firstDelete.dataset.tooltip, 'Delete promptlet');
  assert.ok(firstDelete.querySelector('svg'), 'delete button shows the trash icon');
  assert.ok(
    firstCopy.compareDocumentPosition(firstDelete) & win.Node.DOCUMENT_POSITION_FOLLOWING,
    'the delete button sits right of the copy button',
  );

  // --- copy puts the promptlet on the clipboard without injecting it
  click(win, rows(doc)[2].querySelector('.sidebar-item-copy'));
  await flush();
  assert.deepStrictEqual(clipboardWrites, ['Bundled alpha']);
  assert.strictEqual(composer.value, '', 'copy does not inject into the composer');
  assert.ok(
    doc.getElementById('tricks-panel').classList.contains('open'),
    'copy keeps the panel open',
  );
  assert.ok(
    rows(doc)[2].querySelector('.sidebar-item-copy').classList.contains('copied'),
    'copy button flashes the copied state',
  );
  assert.strictEqual(posted.filter(m => m.type === 'deleteTrick').length, 0);

  // --- delete posts deleteTrick with the row's text and drops the row now
  // (after the inline Delete confirm the trash icon reveals; see
  // ui_antipattern_destructive_confirm.test.js)
  click(win, rows(doc)[1].querySelector('.sidebar-item-delete'));
  assert.strictEqual(posted.filter(m => m.type === 'deleteTrick').length, 0);
  click(win, rows(doc)[1].querySelector('.sidebar-confirm-yes'));
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(posted.filter(m => m.type === 'deleteTrick'))),
    [{type: 'deleteTrick', text: 'Mine two'}],
  );
  assert.deepStrictEqual(texts(doc), ['Mine one', 'Bundled alpha', 'Bundled beta']);
  assert.deepStrictEqual(
    JSON.parse(JSON.stringify(win.__TRICKS__)),
    ['Mine one', 'Bundled alpha', 'Bundled beta'],
    'the page list drops the promptlet too',
  );
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 1, 'one user row remains');
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-delete').length),
    [1, 0, 0],
    'the bundled rows do not inherit a delete button',
  );
  assert.strictEqual(composer.value, '', 'delete does not inject into the composer');
  assert.ok(
    doc.getElementById('tricks-panel').classList.contains('open'),
    'delete keeps the panel open',
  );

  // --- a failed delete: the daemon answers `error` + the list on disk,
  // stamped for this window, and the row comes back
  send(win, {type: 'error', text: 'Could not write ~/.kiss/MY_INJECTION.md', connId: 'c1'});
  send(win, {
    type: 'tricksData',
    tricks: ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta'],
    userCount: 2,
    connId: 'c1',
  });
  assert.deepStrictEqual(texts(doc), ['Mine one', 'Mine two', 'Bundled alpha', 'Bundled beta']);
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 2, 'the resync restores the user count');
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-delete').length),
    [1, 1, 0, 0],
  );
  // ...and the row can be deleted again once the daemon accepts it
  click(win, rows(doc)[1].querySelector('.sidebar-item-delete'));
  click(win, rows(doc)[1].querySelector('.sidebar-confirm-yes'));
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'deleteTrick').map(m => m.text),
    ['Mine two', 'Mine two'],
  );
  assert.deepStrictEqual(texts(doc), ['Mine one', 'Bundled alpha', 'Bundled beta']);

  // --- a filtered list deletes the right promptlet (index into the full list)
  input(win, doc.getElementById('tricks-search'), 'alpha');
  assert.deepStrictEqual(texts(doc), ['Bundled alpha']);
  assert.strictEqual(rows(doc)[0].querySelectorAll('.sidebar-item-delete').length, 0);
  input(win, doc.getElementById('tricks-search'), 'mine');
  assert.deepStrictEqual(texts(doc), ['Mine one']);
  click(win, rows(doc)[0].querySelector('.sidebar-item-delete'));
  click(win, rows(doc)[0].querySelector('.sidebar-confirm-yes'));
  assert.deepStrictEqual(
    posted.filter(m => m.type === 'deleteTrick').map(m => m.text),
    ['Mine two', 'Mine two', 'Mine one'],
  );
  assert.strictEqual(list.textContent.trim(), 'No matching promptlets');
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 0);
  doc.getElementById('tricks-search-clear').click();
  assert.deepStrictEqual(texts(doc), ['Bundled alpha', 'Bundled beta']);
  assert.strictEqual(list.querySelectorAll('.sidebar-item-delete').length, 0);

  // --- clicking the row itself still injects the promptlet
  click(win, rows(doc)[1]);
  assert.strictEqual(composer.value, 'Bundled beta');
  assert.ok(!doc.getElementById('tricks-panel').classList.contains('open'));

  // --- tricksData carries userCount: the daemon's answer repaints
  send(win, {
    type: 'tricksData',
    tricks: ['Fresh mine', 'Bundled alpha', 'Bundled beta'],
    userCount: 1,
  });
  doc.getElementById('tricks-btn').click();
  assert.deepStrictEqual(texts(doc), ['Fresh mine', 'Bundled alpha', 'Bundled beta']);
  assert.deepStrictEqual(
    rows(doc).map(r => r.querySelectorAll('.sidebar-item-delete').length),
    [1, 0, 0],
  );
  assert.strictEqual(rows(doc).length, list.querySelectorAll('.sidebar-item-copy').length);

  // --- a tricksData without userCount (or a bad one) treats every row as bundled
  send(win, {type: 'tricksData', tricks: ['Only bundled']});
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 0);
  assert.strictEqual(list.querySelectorAll('.sidebar-item-delete').length, 0);
  assert.strictEqual(list.querySelectorAll('.sidebar-item-copy').length, 1);
  send(win, {type: 'tricksData', tricks: 'garbage', userCount: '3'});
  assert.strictEqual(list.textContent.trim(), 'No tricks available');
  assert.strictEqual(win.__MY_TRICKS_COUNT__, 0);
  send(win, {type: 'tricksData', tricks: ['Only bundled'], userCount: 0});

  // --- a page loaded without the count global shows no delete buttons
  const bare = makeWebview();
  bare.win.__TRICKS__ = ['a', 'b'];
  bare.win.document.getElementById('tricks-btn').click();
  const bareList = bare.win.document.getElementById('tricks-list');
  assert.strictEqual(bareList.querySelectorAll('.sidebar-item-copy').length, 2);
  assert.strictEqual(bareList.querySelectorAll('.sidebar-item-delete').length, 0);

  // --- no clipboard API: the copy button falls back to execCommand
  Object.defineProperty(win.navigator, 'clipboard', {configurable: true, value: undefined});
  const execCommands = [];
  doc.execCommand = cmd => {
    execCommands.push(cmd);
    return true;
  };
  click(win, list.querySelector('.sidebar-item-copy'));
  assert.deepStrictEqual(execCommands, ['copy']);
  assert.strictEqual(composer.value, 'Bundled beta', 'fallback copy does not inject');

  console.log('tricksPanelCopyDelete.test.js: panel passed');
}

function runGetTricksData() {
  const sourcePath = path.join(__dirname, '..', 'out', 'SorcarTab.js');
  assert.ok(fs.existsSync(sourcePath), `compiled extension missing: ${sourcePath}`);
  delete require.cache[require.resolve(sourcePath)];
  const {getTricksData, getTricks, buildChatHtml} = require(sourcePath);
  assert.strictEqual(typeof getTricksData, 'function');

  const kissHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-tricks-count-'));
  const bundledFile = path.join(kissHome, 'INJECTIONS.md');
  const prevHome = process.env.KISS_HOME;
  const prevInj = process.env.KISS_INJECTIONS_PATH;
  process.env.KISS_HOME = kissHome;
  process.env.KISS_INJECTIONS_PATH = bundledFile;
  try {
    fs.writeFileSync(bundledFile, '## Trick\n\nbundled one\n\n## Trick\n\nbundled two\n');
    fs.writeFileSync(
      path.join(kissHome, 'MY_INJECTION.md'),
      '## Trick\n\nmine one\n\n## Other\n\nskip\n\n## Trick\n\nmine two\n',
    );
    assert.deepStrictEqual(getTricksData(), {
      tricks: ['mine one', 'mine two', 'bundled one', 'bundled two'],
      userCount: 2,
    });
    assert.deepStrictEqual(getTricks(), getTricksData().tricks);

    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
    assert.deepStrictEqual(getTricksData(), {
      tricks: ['bundled one', 'bundled two'],
      userCount: 0,
    });

    // The page freezes the count next to the list.
    fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '## Trick\n\nmine\n');
    const html = buildChatHtml(
      {cspSource: 'vscode-resource:', asWebviewUri: uri => uri},
      makeUri(path.join(__dirname, '..')),
      'test-model',
    );
    assert.ok(!html.includes('{{MY_TRICKS_COUNT}}'), 'placeholder substituted');
    const m = html.match(/window\.__MY_TRICKS_COUNT__\s*=\s*(\d+);/);
    assert.ok(m, '__MY_TRICKS_COUNT__ assignment present');
    assert.strictEqual(m[1], '1');
    assert.ok(html.includes('window.__TRICKS__ = ["mine","bundled one","bundled two"];'));
  } finally {
    if (prevHome === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prevHome;
    if (prevInj === undefined) delete process.env.KISS_INJECTIONS_PATH;
    else process.env.KISS_INJECTIONS_PATH = prevInj;
    fs.rmSync(kissHome, {recursive: true, force: true});
  }
  console.log('tricksPanelCopyDelete.test.js: getTricksData passed');
}

runPanel()
  .then(runGetTricksData)
  .then(() => console.log('tricksPanelCopyDelete.test.js passed'))
  .catch(err => {
    console.error(err);
    process.exit(1);
  });

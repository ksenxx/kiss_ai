// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom page) test of the checkPaths -> pathsExist -> click
// -> openFile chain in media/main.js, over the page's direct transport:
// a prompt that mentions file paths posts one checkPaths for them
// (stamped with the tab and its work dir); the daemon's pathsExist reply
// turns the existing ones into links and leaves the missing ones plain;
// clicking a link posts openFile (path, workDir, tabId) and clicking a
// missing path posts nothing.  The daemon end of the chain is
// src/kiss/tests/server/test_file_link_resolution.py.

'use strict';

const assert = require('assert');
const {makePage, send, waitFor} = require('./simplify2_harness.js');

function findLinks(win, p) {
  return Array.from(
    win.document.querySelectorAll('#output .kiss-filelink[data-path]'),
  ).filter(el => el.dataset.path === p);
}

function clickEl(win, el) {
  el.dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

async function main() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'configData', config: {work_dir: '/ws/proj'}, apiKeys: {}});
  const realFile = '/ws/proj/src/app.py';
  const missingFile = '/ws/proj/src/gone.py';

  // An agent panel mentions one existing and one missing path.
  send(win, {
    type: 'prompt',
    text: 'compare src/app.py with ' + realFile + ' and ' + missingFile,
    tabId,
  });
  const check = await waitFor(
    () => posted.find(m => m.type === 'checkPaths'),
    'the page must send checkPaths for the mentioned paths',
  );
  assert.strictEqual(check.tabId, tabId, 'checkPaths names its tab');
  assert.strictEqual(check.workDir, '/ws/proj', 'checkPaths carries the work dir');
  for (const p of ['src/app.py', realFile, missingFile]) {
    assert.ok(check.paths.includes(p), `checkPaths asks about ${p}`);
  }
  assert.strictEqual(
    findLinks(win, realFile).length,
    0,
    'nothing is clickable before the daemon answers',
  );
  console.log('  ok - mentioned paths are sent to the daemon in one checkPaths');

  // The daemon's reply gates clickability.
  const results = {};
  for (const p of check.paths) results[p] = p !== missingFile;
  send(win, {type: 'pathsExist', results, tabId, workDir: '/ws/proj'});
  assert.strictEqual(
    findLinks(win, realFile).length,
    1,
    'absolute existing path must become clickable after the reply',
  );
  assert.strictEqual(
    findLinks(win, 'src/app.py').length,
    1,
    'relative existing path must become clickable after the reply',
  );
  assert.strictEqual(
    findLinks(win, missingFile).length,
    0,
    'missing path must NOT be clickable after the reply',
  );
  const missing = Array.from(
    win.document.querySelectorAll('#output [data-path-missing]'),
  ).filter(el => el.textContent === missingFile);
  assert.strictEqual(missing.length, 1, 'missing path stays plain text');
  console.log('  ok - the daemon reply gates clickability');

  // Clicking the promoted link posts openFile for the daemon.
  const before = posted.length;
  clickEl(win, findLinks(win, realFile)[0]);
  const open = posted.slice(before).find(m => m.type === 'openFile');
  assert.ok(open, 'clicking the verified link must post openFile');
  assert.strictEqual(open.path, realFile);
  assert.strictEqual(open.workDir, '/ws/proj');
  assert.strictEqual(open.tabId, tabId);
  console.log('  ok - clicking a verified link posts openFile');

  // Clicking the missing-path text posts nothing.
  const count = posted.length;
  clickEl(win, missing[0]);
  await new Promise(r => setTimeout(r, 100));
  assert.strictEqual(
    posted.slice(count).filter(m => m.type === 'openFile').length,
    0,
    'clicking a missing path must not post openFile',
  );
  console.log('  ok - missing path click opens nothing');
  win.close();
}

main()
  .then(() => console.log('fileLinkResolution.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });

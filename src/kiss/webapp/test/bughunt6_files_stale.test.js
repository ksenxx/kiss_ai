// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/* global require, console, process */
'use strict';

const assert = require('assert');
const {makePage, send, stubRaf} = require('./simplify2_harness');

function setInput(win, text) {
  const inp = win.document.getElementById('task-input');
  inp.value = text;
  inp.setSelectionRange(text.length, text.length);
  return inp;
}

function pickerDisplay(win) {
  return win.document.getElementById('autocomplete').style.display;
}

const FILES = [
  {type: 'file', text: 'src/util.py'},
  {type: 'file', text: 'src/utils/helpers.py'},
];

function testLateFilesEventWithoutAtContextStaysHidden() {
  const {win} = makePage({beforeScripts: stubRaf});
  setInput(win, 'fix the bug');

  send(win, {type: 'files', files: FILES});

  assert.notStrictEqual(
    pickerDisplay(win),
    'block',
    'a late files event must NOT open the picker when no @-mention ' +
      'is being typed (the phantom picker swallows the next Enter)',
  );
}

function testStalePrefixFilesEventIgnored() {
  const {win} = makePage({beforeScripts: stubRaf});
  setInput(win, '@util');

  send(win, {type: 'files', prefix: 'u', files: FILES});

  assert.notStrictEqual(
    pickerDisplay(win),
    'block',
    'a files reply for a stale prefix must not open the picker',
  );
}

function testMatchingPrefixFilesEventRenders() {
  const {win} = makePage({beforeScripts: stubRaf});
  setInput(win, '@util');

  send(win, {type: 'files', prefix: 'util', files: FILES});

  assert.strictEqual(
    pickerDisplay(win),
    'block',
    'a files reply matching the typed @-prefix must open the picker',
  );
  const text = win.document.getElementById('autocomplete').textContent;
  assert.ok(text.includes('util.py'), 'picker must list the files');
}

function testPrefixlessFilesEventStillRendersWithAtContext() {
  const {win} = makePage({beforeScripts: stubRaf});
  setInput(win, '@ut');

  send(win, {type: 'files', files: FILES});

  assert.strictEqual(
    pickerDisplay(win),
    'block',
    'a prefix-less files reply must render while an @-mention is active',
  );
}

// A16-3: the daemon's global workspace moved (a pick on another surface)
// while this surface's ``@`` query was still building the old
// workspace's index.  The reply echoes the workDir it was asked for: one
// for a directory the tab no longer browses is dropped, and adopting the
// new workspace asks again for it, so the picker never lists the old
// workspace's files under the new one.
function testFilesReplyForTheOldWorkspaceIsDroppedAndTheQueryReissued() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  send(win, {type: 'workDirChanged', workDir: '/old-project'});
  const inp = setInput(win, '@ut');
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  const asked = posted.filter(m => m.type === 'getFiles');
  assert.strictEqual(asked.length, 1, 'the mention asks for the files');
  assert.strictEqual(asked[0].workDir, '/old-project');

  send(win, {type: 'workDirChanged', workDir: '/new-project'});
  const reasked = posted.filter(m => m.type === 'getFiles');
  assert.strictEqual(
    reasked.length,
    2,
    'BUG - adopting the new workspace did not ask again for the mention',
  );
  assert.strictEqual(reasked[1].workDir, '/new-project');
  assert.strictEqual(reasked[1].prefix, 'ut');

  // The old workspace's reply, emitted before the new request landed.
  send(win, {
    type: 'files',
    prefix: 'ut',
    workDir: '/old-project',
    files: [{type: 'file', text: './old-project-only.py'}],
  });
  assert.notStrictEqual(
    pickerDisplay(win),
    'block',
    "BUG - the old workspace's file list was rendered under the new one",
  );
  send(win, {
    type: 'files',
    prefix: 'ut',
    workDir: '/new-project',
    files: FILES,
  });
  assert.strictEqual(pickerDisplay(win), 'block', 'the new reply renders');
  const text = win.document.getElementById('autocomplete').textContent;
  assert.ok(text.includes('util.py'), "with the new workspace's files");
  assert.ok(!text.includes('old-project-only'), 'and none of the old one');
}

// A mention typed before the daemon's configData arrived asked with no
// directory; the reply is stamped with the directory the daemon
// resolved it to.  The config then names that directory: the picker
// asks again (the reply of the first request may have been dropped as
// one for a directory the tab does not browse), and renders the reply
// stamped with the directory it browses now.
function testMentionTypedBeforeTheConfigArrivedIsAskedAgain() {
  const {win, posted} = makePage({beforeScripts: stubRaf});
  const inp = setInput(win, '@ut');
  inp.dispatchEvent(new win.Event('input', {bubbles: true}));
  const asked = posted.filter(m => m.type === 'getFiles');
  assert.strictEqual(asked.length, 1, 'the mention asks for the files');
  assert.strictEqual(asked[0].workDir, '', 'no directory known yet');

  send(win, {type: 'configData', config: {work_dir: '/the-project'}});
  const reasked = posted.filter(m => m.type === 'getFiles');
  assert.strictEqual(
    reasked.length,
    2,
    'BUG - the config naming the directory did not ask again for the mention',
  );
  assert.strictEqual(reasked[1].workDir, '/the-project');
  send(win, {type: 'files', prefix: 'ut', workDir: '/the-project', files: FILES});
  assert.strictEqual(pickerDisplay(win), 'block', 'the stamped reply renders');
}

// A reply without a workDir stamp (a machine running an older daemon)
// still renders, as a prefix-less one does.
function testUnstampedFilesReplyStillRenders() {
  const {win} = makePage({beforeScripts: stubRaf});
  send(win, {type: 'workDirChanged', workDir: '/new-project'});
  setInput(win, '@ut');
  send(win, {type: 'files', prefix: 'ut', files: FILES});
  assert.strictEqual(pickerDisplay(win), 'block');
}

const tests = [
  testLateFilesEventWithoutAtContextStaysHidden,
  testStalePrefixFilesEventIgnored,
  testMatchingPrefixFilesEventRenders,
  testPrefixlessFilesEventStillRendersWithAtContext,
  testFilesReplyForTheOldWorkspaceIsDroppedAndTheQueryReissued,
  testMentionTypedBeforeTheConfigArrivedIsAskedAgain,
  testUnstampedFilesReplyStillRenders,
];

let failed = 0;
for (const t of tests) {
  try {
    t();
    console.log('PASS', t.name);
  } catch (err) {
    failed += 1;
    console.error('FAIL', t.name);
    console.error(err && err.stack ? err.stack : err);
  }
}
if (failed) {
  console.error(failed + ' test(s) failed');
  process.exit(1);
}
console.log('All tests passed');

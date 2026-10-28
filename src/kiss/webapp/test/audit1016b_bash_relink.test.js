// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// A running Bash tool's streamed output (system_output) is flushed once
// per animation frame.  Each flush must append the new text only
// (audit 2026-10-16 mainjs-2 F1): rewriting the whole panel through
// textContent flattened every file link made of the earlier chunks, so
// each flush asked the daemon about every path seen so far (O(n^2)
// checkPaths) and every clickable link went inert until the reply.
// A path cut by the frame boundary -- inside a name, or right after a
// "." or ":" the matcher leaves outside the link (review-mainjs I1) --
// must still come out as one link to the whole path.
// Runs with: node test/audit1016b_bash_relink.test.js

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  fakeMatchMedia,
  queueRaf,
} = require('./simplify2_harness');

function makePage() {
  const {win, posted} = makeChatPage({
    bodyClass: 'remote-chat',
    extraScripts: ['marked.min.js'],
    beforeScripts(win) {
      // Frames are queued and flushed by hand (win.flushRaf).
      win.flushRaf = queueRaf(win).run;
      fakeMatchMedia(win, true);
    },
  });
  return {win, posted};
}

function checks(posted) {
  return posted.filter(m => m && m.type === 'checkPaths');
}

function links(win) {
  return win.document.querySelectorAll(
    '#output .bash-panel-content .kiss-filelink',
  ).length;
}

/** Answer *check* as the daemon would: every path exists. */
function answer(win, check, tabId) {
  const results = {};
  check.paths.forEach(p => {
    results[p] = true;
  });
  send(win, {type: 'pathsExist', results, tabId, workDir: check.workDir});
}

/** A page with a running task whose Bash tool (callId 1) is streaming. */
function startBash() {
  const {win, posted} = makePage();
  const tabId = win._testApi.getActiveTabId();
  send(win, {type: 'daemonStatus', connected: true});
  send(win, {type: 'configData', config: {work_dir: '/ws/proj'}, apiKeys: {}});
  send(win, {type: 'status', running: true, tabId, startTs: Date.now()});
  send(win, {
    type: 'tool_call',
    name: 'Bash',
    command: 'make',
    description: 'build',
    tabId,
    ts: Date.now(),
    callId: 1,
  });
  return {win, posted, tabId};
}

function endBash(win, tabId) {
  send(win, {
    type: 'tool_result',
    content: 'ok',
    tabId,
    ts: Date.now(),
    callId: 1,
  });
}

function testFlushKeepsLinksAndAsksEachPathOnce() {
  const {win, posted, tabId} = startBash();
  const before = checks(posted).length;
  const N = 20;
  let asked = 0;
  for (let i = 0; i < N; i++) {
    const linksBefore = links(win);
    send(win, {
      type: 'system_output',
      text: 'built /ws/proj/out/f' + i + '.o\n',
      tabId,
    });
    win.flushRaf();
    assert.ok(
      links(win) >= linksBefore,
      'flush ' + i + ' kept the ' + linksBefore + ' resolved links',
    );
    const batch = checks(posted).slice(before + i);
    assert.strictEqual(batch.length, 1, 'flush ' + i + ' sends one checkPaths');
    asked += batch[0].paths.length;
    answer(win, batch[0], tabId);
    assert.strictEqual(
      links(win),
      i + 1,
      'flush ' + i + ': every path is a link',
    );
  }
  assert.strictEqual(asked, N, 'each path was asked about exactly once');
  // A chunk without any path leaves the links alone and asks nothing.
  send(win, {type: 'system_output', text: 'done\n', tabId});
  win.flushRaf();
  assert.strictEqual(links(win), N, 'a path-free flush keeps every link');
  assert.strictEqual(checks(posted).length, before + N, 'and asks nothing');
  // A path split across two frames still becomes one link, and the
  // panel text reads as the chunks arrived.
  send(win, {type: 'system_output', text: 'see /ws/proj/out/', tabId});
  win.flushRaf();
  send(win, {type: 'system_output', text: 'split.o\n', tabId});
  win.flushRaf();
  const last = checks(posted).pop();
  assert.deepStrictEqual([...last.paths], ['/ws/proj/out/split.o']);
  answer(win, last, tabId);
  assert.strictEqual(links(win), N + 1);
  const panel = win.document.querySelector('#output .bash-panel-content');
  assert.ok(panel.textContent.endsWith('done\nsee /ws/proj/out/split.o\n'));
  // ... also when the cut falls inside a name: the head asked about on
  // its own ("/ws/proj/ou", missing) is re-matched as part of the whole.
  send(win, {type: 'system_output', text: 'ls /ws/proj/ou', tabId});
  win.flushRaf();
  const head = checks(posted).pop();
  assert.deepStrictEqual([...head.paths], ['/ws/proj/ou']);
  send(win, {
    type: 'pathsExist',
    results: {'/ws/proj/ou': false},
    tabId,
    workDir: head.workDir,
  });
  send(win, {type: 'system_output', text: 't/two.o\n', tabId});
  win.flushRaf();
  const whole = checks(posted).pop();
  assert.deepStrictEqual([...whole.paths], ['/ws/proj/out/two.o']);
  answer(win, whole, tabId);
  assert.strictEqual(links(win), N + 2);
  assert.strictEqual(
    panel.querySelectorAll('[data-path-missing], [data-path-candidate]').length,
    0,
    'no stray head or tail span is left',
  );
  assert.ok(panel.textContent.endsWith('ls /ws/proj/out/two.o\n'));
  // A cut right after the "." of a name (review-mainjs I1): the matcher
  // leaves a terminal "." outside the span, and the next chunk must
  // rejoin the span, the "." and itself into the whole file name.
  send(win, {type: 'system_output', text: 'see /ws/proj/file.', tabId});
  win.flushRaf();
  const stem = checks(posted).pop();
  assert.deepStrictEqual([...stem.paths], ['/ws/proj/file']);
  send(win, {
    type: 'pathsExist',
    results: {'/ws/proj/file': false},
    tabId,
    workDir: stem.workDir,
  });
  send(win, {type: 'system_output', text: 'txt\n', tabId});
  win.flushRaf();
  const dotted = checks(posted).pop();
  assert.deepStrictEqual(
    [...dotted.paths],
    ['/ws/proj/file.txt'],
    'the completed file name is asked about',
  );
  answer(win, dotted, tabId);
  assert.strictEqual(links(win), N + 3, 'and linked');
  assert.strictEqual(
    panel.querySelectorAll('[data-path-missing], [data-path-candidate]').length,
    0,
    'the missing stem span is gone',
  );
  assert.ok(panel.textContent.endsWith('see /ws/proj/file.txt\n'));
  // A cut right after the ":" of a path:line reference: the link must
  // end up at the line, not at the bare file.
  send(win, {type: 'system_output', text: 'at /ws/proj/file.py:', tabId});
  win.flushRaf();
  const bare = checks(posted).pop();
  assert.deepStrictEqual([...bare.paths], ['/ws/proj/file.py']);
  answer(win, bare, tabId);
  assert.strictEqual(links(win), N + 4);
  send(win, {type: 'system_output', text: '12\n', tabId});
  win.flushRaf();
  const lined = checks(posted).pop();
  assert.notStrictEqual(lined, bare, 'the whole reference is checked again');
  assert.deepStrictEqual(
    [...lined.paths],
    ['/ws/proj/file.py'],
    'the daemon is asked about the file (the line suffix is stripped)',
  );
  answer(win, lined, tabId);
  const all = panel.querySelectorAll('.kiss-filelink');
  assert.strictEqual(all.length, N + 4, 'the bare link became the lined one');
  assert.strictEqual(all[all.length - 1].textContent, '/ws/proj/file.py:12');
  assert.strictEqual(
    all[all.length - 1].dataset.path,
    '/ws/proj/file.py:12',
    'the link targets the line',
  );
  assert.ok(panel.textContent.endsWith('at /ws/proj/file.py:12\n'));
  // A "." that ends a sentence stays plain text: a chunk that does not
  // continue the path leaves the link alone and asks nothing new.
  send(win, {type: 'system_output', text: 'read /ws/proj/out/f0.o.', tabId});
  win.flushRaf();
  answer(win, checks(posted).pop(), tabId);
  const asked2 = checks(posted).length;
  send(win, {type: 'system_output', text: ' Done\n', tabId});
  win.flushRaf();
  assert.strictEqual(checks(posted).length, asked2, 'nothing re-asked');
  assert.strictEqual(links(win), N + 5);
  assert.ok(panel.textContent.endsWith('read /ws/proj/out/f0.o. Done\n'));
  // The tool's end flushes what is still buffered the same way.
  send(win, {type: 'system_output', text: 'end /ws/proj/out/last.o\n', tabId});
  endBash(win, tabId);
  assert.strictEqual(links(win), N + 5, 'the close flush kept every link');
  assert.deepStrictEqual(
    [...checks(posted).pop().paths],
    ['/ws/proj/out/last.o'],
  );
  win.close();
  console.log(
    '  ok - a Bash flush keeps resolved links and asks each path once',
  );
}

/**
 * The close flush (tool_result with text still buffered) rejoins a
 * path the last frame cut after "." or ":" the same way as a frame
 * flush does.
 */
function testCloseFlushRejoinsPunctuationSplit() {
  // [head chunk, tail chunk, file asked about, link target]
  for (const [head, tail, file, whole] of [
    ['see /ws/proj/file.', 'txt\n', '/ws/proj/file.txt', '/ws/proj/file.txt'],
    ['at /ws/proj/file.py:', '12\n', '/ws/proj/file.py', '/ws/proj/file.py:12'],
  ]) {
    const {win, posted, tabId} = startBash();
    send(win, {type: 'system_output', text: head, tabId});
    win.flushRaf();
    const first = checks(posted).pop();
    answer(win, first, tabId);
    assert.strictEqual(links(win), 1, 'the head is linked on its own');
    send(win, {type: 'system_output', text: tail, tabId});
    endBash(win, tabId);
    const last = checks(posted).pop();
    assert.notStrictEqual(last, first, 'the close flush checks again');
    assert.deepStrictEqual(
      [...last.paths],
      [file],
      'the close flush asks about the whole path',
    );
    answer(win, last, tabId);
    const panel = win.document.querySelector('#output .bash-panel-content');
    const linkEls = panel.querySelectorAll('.kiss-filelink');
    assert.strictEqual(linkEls.length, 1, 'one link, to the whole path');
    assert.strictEqual(linkEls[0].dataset.path, whole);
    assert.strictEqual(panel.textContent, head + tail);
    win.close();
  }
  console.log('  ok - the close flush rejoins a path cut after . or :');
}

testFlushKeepsLinksAndAsksEachPathOnce();
testCloseFlushRejoinsPunctuationSplit();
console.log('audit1016b_bash_relink.test.js: all tests passed');

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the share export's replay of persisted
// tasks in media/main.js (buildShareableHtml):
//
// * the pictures of a replayed result are not fetched for the detached
//   copy (no openFile command); the exported tag keeps its alt text;
// * a task that ended without its closing events (a stopped Bash call)
//   still exports what it streamed, though the frames that would paint
//   it have not run when the copy is serialized.

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  stubRaf,
  queueRaf,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');
const {shareWithTasks} = require('./share_helpers');

const {test, report} = makeRunner();

function makePage() {
  return makeChatPage({
    beforeScripts: stubRaf,
    extraScripts: ['marked.min.js'],
  });
}

async function main() {
  await test('a shared chat fetches none of its pictures', () => {
    const wv = makePage();
    const summary = (n, k) =>
      Array.from(
        {length: n},
        (_, i) => `![p${k}${i}](tmp/plot${k}${i}.png)`,
      ).join('\n\n');
    const msg = shareWithTasks(wv, [
      {
        task: 'plot',
        task_id: 't1',
        events: [{type: 'result', summary: summary(3, 0), success: true}],
      },
      {
        task: 'plot more',
        task_id: 't2',
        events: [{type: 'result', summary: summary(2, 1), success: true}],
      },
    ]);
    assert.ok(msg, 'the chat was exported');
    assert.deepStrictEqual(
      wv.posted.filter(m => m.type === 'openFile'),
      [],
      'no picture is read for the copy',
    );
    assert.ok(msg.html.includes('alt="p00"'), 'the tag keeps its alt text');
    assert.ok(msg.html.includes('data-file-src="tmp/plot10.png"'));
    wv.win.close();
  });

  await test('a stopped Bash call exports what it streamed', () => {
    const wv = makePage();
    const raf = queueRaf(wv.win);
    const msg = shareWithTasks(wv, [
      {
        task: 'run the tests',
        task_id: 't1',
        events: [
          {
            type: 'tool_call',
            name: 'Bash',
            command: 'make test',
            description: 'Run the tests',
          },
          {type: 'system_output', text: 'MARKER_LINE_ONE\n'},
          {type: 'system_output', text: 'MARKER_LINE_TWO\n'},
          {type: 'text_delta', text: 'MARKER_TEXT_DELTA'},
          {type: 'result', summary: 'stopped', success: false},
        ],
      },
    ]);
    assert.ok(msg, 'the chat was exported');
    assert.ok(msg.html.includes('make test'));
    assert.ok(msg.html.includes('MARKER_LINE_ONE'), 'the command output');
    assert.ok(msg.html.includes('MARKER_LINE_TWO'));
    assert.ok(msg.html.includes('MARKER_TEXT_DELTA'), 'the trailing text');
    assert.strictEqual(raf.frames.length, 0, 'no frame is left to paint it');
    wv.win.close();
  });

  report('shareExportReplay_sweep1030');
}

main();

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const DEMO_PATH = path.join(__dirname, '..', 'media', 'demo.js');

function makeDemoWindow(events, opts) {
  const withHooks = !opts || opts.withHooks !== false;
  const dom = new JSDOM(
    '<!DOCTYPE html><html><body><div id="output"></div></body></html>',
    {runScripts: 'dangerously', pretendToBeVisual: true},
  );
  const win = dom.window;
  let active = false;
  const calls = [];
  const api = {
    get active() {
      return active;
    },
    set active(v) {
      active = !!v;
    },
    resolveEvents: null,
    setInput() {},
    clearInput() {},
    clearForReplay() {},
    resetOutputState() {},
    processEvent(ev) {
      calls.push({fn: 'processEvent', ev});
    },
    setTaskText() {},
    updateTabTitle() {},
    hideWelcome() {},
    scrollToBottom() {},
    getActiveTabId() {
      return 'demo-tab';
    },
    sendMessage(msg) {
      if (msg && msg.type === 'resumeSession') {
        setTimeout(() => {
          if (api.resolveEvents) api.resolveEvents(events);
        }, 10);
      }
    },
    collapsePanels() {},
    setRunningState() {},
    showSpinner() {},
    removeSpinner() {},
  };
  if (withHooks) {
    api.speakText = function (text, language) {
      calls.push({fn: 'speakText', text, language});
    };
    api.playTalkEvent = function (ev) {
      calls.push({fn: 'playTalkEvent', ev});
    };
    api.openSubagentTab = function (ev) {
      calls.push({fn: 'openSubagentTab', ev});
    };
    api.stopSpeech = function () {
      calls.push({fn: 'stopSpeech'});
    };
  }
  win._demoApi = api;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  return {win, api, calls};
}

async function runReplay(win, api, preview) {
  const replay = win._startDemoReplay([
    {id: 1, has_events: true, preview: preview, timestamp: 1},
  ]);
  await replay;
  assert.strictEqual(api.active, false, 'replay must clear active flag');
}

function sampleEvents() {
  return [
    {type: 'text_delta', text: 'Working on it.'},
    {
      type: 'tool_call',
      name: 'talk',
      extras: {
        text: 'Hello there, I am on it.',
        language: 'en-US',
        emotion: 'cheerful',
      },
    },
    {type: 'tool_result', content: 'Playing audio', tool_name: 'talk'},
    {
      type: 'tool_call',
      name: 'run_parallel',
      extras: {
        tasks: JSON.stringify(['research topic A', 'summarize topic B']),
        max_workers: '2',
      },
    },
    {type: 'tool_result', content: 'done', tool_name: 'run_parallel'},
    {
      type: 'result',
      success: true,
      summary: 'All done.',
      total_tokens: 10,
      cost: '$0.01',
    },
  ];
}

async function testNoPromptNarrationAndExecutesToolCalls() {
  const {win, api, calls} = makeDemoWindow(sampleEvents());
  await runReplay(win, api, 'plan my trip');

  const speaks = calls.filter(c => c.fn === 'speakText');
  assert.strictEqual(speaks.length, 0, 'no prompt narration');

  const talks = calls.filter(c => c.fn === 'playTalkEvent');
  assert.strictEqual(talks.length, 1, 'exactly one talk playback');
  assert.strictEqual(talks[0].ev.text, 'Hello there, I am on it.');
  assert.strictEqual(talks[0].ev.language, 'en-US');
  assert.strictEqual(talks[0].ev.emotion, 'cheerful');

  const tabs = calls.filter(c => c.fn === 'openSubagentTab');
  assert.strictEqual(tabs.length, 2, 'one sub-agent tab per task');
  tabs.forEach((c, i) => {
    assert.strictEqual(c.ev.type, 'openSubagentTab');
    assert.strictEqual(c.ev.parent_tab_id, 'demo-tab');
    assert.strictEqual(c.ev.taskIndex, i);
    assert.strictEqual(c.ev.isDone, false);
    assert.ok(c.ev.tab_id, 'sub-agent tab id must be set');
  });
  assert.strictEqual(tabs[0].ev.description, 'research topic A');
  assert.strictEqual(tabs[1].ev.description, 'summarize topic B');
  assert.notStrictEqual(
    tabs[0].ev.tab_id,
    tabs[1].ev.tab_id,
    'sub-agent tab ids must be unique',
  );

  const idxOf = pred => calls.findIndex(pred);
  const talkPanelIdx = idxOf(
    c =>
      c.fn === 'processEvent' &&
      c.ev.type === 'tool_call' &&
      c.ev.name === 'talk',
  );
  const talkPlayIdx = idxOf(c => c.fn === 'playTalkEvent');
  assert.ok(
    talkPanelIdx !== -1 && talkPlayIdx > talkPanelIdx,
    'talk playback must follow its rendered panel',
  );
  const rpPanelIdx = idxOf(
    c =>
      c.fn === 'processEvent' &&
      c.ev.type === 'tool_call' &&
      c.ev.name === 'run_parallel',
  );
  const firstTabIdx = idxOf(c => c.fn === 'openSubagentTab');
  assert.ok(
    rpPanelIdx !== -1 && firstTabIdx > rpPanelIdx,
    'sub-agent tabs must be created after the fan-out panel rendered',
  );

  win.close();
  console.log('  ok - no prompt narration, talk played, sub-agent tabs created');
}

async function testBadTasksJsonDegradesGracefully() {
  const events = [
    {
      type: 'tool_call',
      name: 'run_parallel',
      extras: {tasks: '[truncated garba'},
    },
    {type: 'tool_result', content: 'done', tool_name: 'run_parallel'},
    {
      type: 'result',
      success: true,
      summary: 'Done.',
      total_tokens: 1,
      cost: '$0.00',
    },
  ];
  const {win, api, calls} = makeDemoWindow(events);
  await runReplay(win, api, 'broken fanout');
  assert.strictEqual(
    calls.filter(c => c.fn === 'openSubagentTab').length,
    0,
    'unparseable tasks must not create sub-agent tabs',
  );
  win.close();
  console.log('  ok - unparseable run_parallel tasks degrade gracefully');
}

async function testOldHostApiWithoutHooksStillReplays() {
  const {win, api} = makeDemoWindow(sampleEvents(), {withHooks: false});
  await runReplay(win, api, 'legacy host');
  const text = win.document.getElementById('output').textContent;
  assert.ok(text.includes('All done.'), 'replay must still render the result');
  win.close();
  console.log('  ok - host api without new hooks still replays cleanly');
}

async function testCancelStopsSpeech() {
  const {win, api, calls} = makeDemoWindow(sampleEvents());
  const replay = win._startDemoReplay([
    {id: 1, has_events: true, preview: 'cancel me', timestamp: 1},
  ]);
  await new Promise(r => setTimeout(r, 50));
  win._cancelDemoReplay();
  await replay;
  assert.ok(
    calls.some(c => c.fn === 'stopSpeech'),
    'cancelling the demo must stop queued speech',
  );
  assert.strictEqual(api.active, false);
  win.close();
  console.log('  ok - cancel stops queued demo speech');
}

function testParseDemoTasks() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
  });
  const win = dom.window;
  win._demoApi = null;
  win.eval(fs.readFileSync(DEMO_PATH, 'utf8'));
  const parse = raw => JSON.stringify(win._parseDemoTasks(raw));
  assert.strictEqual(parse('["a","b"]'), '["a","b"]');
  assert.strictEqual(parse(['x', 2]), '["x","2"]');
  assert.strictEqual(parse('{"not":"a list"}'), '[]');
  assert.strictEqual(parse('nonsense'), '[]');
  assert.strictEqual(parse(''), '[]');
  assert.strictEqual(parse(undefined), '[]');
  win.close();
  console.log('  ok - parseDemoTasks handles lists, JSON and garbage');
}

async function runTests() {
  testParseDemoTasks();
  await testNoPromptNarrationAndExecutesToolCalls();
  await testBadTasksJsonDegradesGracefully();
  await testOldHostApiWithoutHooksStillReplays();
  await testCancelStopsSpeech();
}

runTests().then(
  () => {
    console.log('\n5 passed, 0 failed');
    process.exit(0);
  },
  err => {
    console.error('FAIL:', err && err.message ? err.message : err);
    process.exit(1);
  },
);

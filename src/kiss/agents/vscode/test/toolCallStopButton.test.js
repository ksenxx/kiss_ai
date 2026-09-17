// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The per-panel Stop button of tool-call panels: every `.tc` panel gets
// a `.panel-stop-btn` immediately left of its copy button; the
// stylesheet shows it only while the tool call runs; a click posts
// `interruptTool` for the panel's tab and tool; the `tool_interrupt_ack`
// receipt resets a rejected click; the arriving tool_result hides it.

/* global require, __dirname, console, process */

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {JSDOM} = require('jsdom');

const MEDIA = path.join(__dirname, '..', 'media');

let passed = 0;
const failures = [];

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  \u2713 ${name}`);
  } catch (e) {
    failures.push({name, error: e});
    console.log(`  \u2717 ${name}`);
    console.log(`      ${e.stack || e.message}`);
  }
}

function makeWebview() {
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  // The real stylesheet, so the running/finished visibility of the
  // button is checked through the cascade, not through class names.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  html = html.replace('</head>', '<style>' + css + '</style></head>');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
    url: 'https://localhost/',
  });
  const win = dom.window;
  win.Element.prototype.scrollIntoView = function () {};
  win.Element.prototype.scrollTo = function () {};
  win.HTMLElement.prototype.scrollTo = function () {};

  const posted = [];
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => undefined,
      setState: () => {},
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function click(el) {
  el.dispatchEvent(
    new el.ownerDocument.defaultView.MouseEvent('click', {
      bubbles: true,
      cancelable: true,
    }),
  );
}

function display(win, el) {
  return win.getComputedStyle(el).getPropertyValue('display');
}

function bootRunningTab(wv) {
  const ready = wv.posted.find(m => m.type === 'ready');
  assert.ok(ready && ready.tabId, 'webview must post ready with a tabId');
  const TAB = ready.tabId;
  send(wv.win, {type: 'clear', chat_id: 'chat-toolstop', tabId: TAB});
  send(wv.win, {
    type: 'status',
    running: true,
    tabId: TAB,
    startTs: Date.now(),
  });
  return TAB;
}

async function webviewTests() {
  const wv = makeWebview();
  const win = wv.win;
  const TAB = bootRunningTab(wv);
  const out = win.document.getElementById('output');

  await test('a running tool-call panel shows Stop left of Copy', () => {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'sleep 30',
      description: 'wait',
      callId: 17,
      tabId: TAB,
      ts: Date.now(),
    });
    const tc = out.querySelector('.ev.tc');
    assert.ok(tc, 'tool_call renders a .tc panel');
    assert.strictEqual(
      tc.dataset.callId,
      '17',
      'the panel remembers its call id',
    );
    const stop = tc.querySelector(':scope > .panel-stop-btn');
    const copy = tc.querySelector(':scope > .panel-copy-btn');
    assert.ok(stop, 'the panel has a stop button');
    assert.ok(copy, 'the panel has a copy button');
    assert.ok(tc.dataset.startMs, 'a live tool_call panel is stamped running');
    assert.strictEqual(display(win, stop), 'flex', 'visible while running');
    assert.strictEqual(display(win, copy), 'flex');
    // Both are absolutely positioned against the panel's top-right
    // corner; the stop button's larger right offset puts it to the
    // left of the copy button.
    const stopRight = parseFloat(win.getComputedStyle(stop).right);
    const copyRight = parseFloat(win.getComputedStyle(copy).right);
    assert.ok(
      stopRight > copyRight,
      `stop (right ${stopRight}px) must sit left of copy (right ${copyRight}px)`,
    );
    assert.strictEqual(
      win.getComputedStyle(stop).top,
      win.getComputedStyle(copy).top,
    );
  });

  await test('collapsing the panel keeps Stop and Copy visible', () => {
    const tc = out.querySelector('.ev.tc');
    click(tc.querySelector('.tc-h'));
    assert.ok(tc.classList.contains('collapsed'));
    assert.strictEqual(
      display(win, tc.querySelector(':scope > .panel-stop-btn')),
      'flex',
    );
    assert.strictEqual(
      display(win, tc.querySelector(':scope > .tc-b')),
      'none',
    );
    click(tc.querySelector('.tc-h'));
  });

  await test('clicking Stop posts interruptTool for the tab and tool', () => {
    const tc = out.querySelector('.ev.tc');
    const stop = tc.querySelector(':scope > .panel-stop-btn');
    const before = wv.posted.length;
    click(stop);
    const msg = wv.posted.slice(before).find(m => m.type === 'interruptTool');
    assert.ok(msg, 'interruptTool is posted');
    assert.strictEqual(msg.tabId, TAB);
    assert.strictEqual(msg.toolName, 'Bash');
    assert.strictEqual(msg.callId, 17, 'the click names exactly this call');
    assert.ok(stop.classList.contains('stopping'));
    assert.ok(
      !tc.classList.contains('collapsed'),
      'the click does not toggle the panel',
    );
    assert.strictEqual(
      wv.posted.slice(before).filter(m => m.type === 'stop').length,
      0,
      'the whole-task Stop is not sent',
    );
  });

  await test('a rejected tool_interrupt_ack resets the button', () => {
    const stop = out.querySelector('.ev.tc > .panel-stop-btn');
    send(win, {type: 'tool_interrupt_ack', accepted: true, tabId: TAB});
    assert.ok(
      stop.classList.contains('stopping'),
      'an accepted ack keeps stopping',
    );
    send(win, {
      type: 'tool_interrupt_ack',
      accepted: false,
      tabId: 'no-such-tab',
    });
    assert.ok(
      stop.classList.contains('stopping'),
      "another tab's ack is ignored",
    );
    send(win, {type: 'tool_interrupt_ack', accepted: false, tabId: TAB});
    assert.ok(!stop.classList.contains('stopping'), 'a rejected ack resets');
    assert.strictEqual(stop.disabled, false);
    click(stop);
    assert.ok(stop.classList.contains('stopping'));
    send(win, {type: 'tool_interrupt_ack', accepted: false});
    assert.ok(
      !stop.classList.contains('stopping'),
      'a tab-less ack targets the active tab',
    );
  });

  await test('the tool_result hides Stop and shows the interrupt message', () => {
    const tc = out.querySelector('.ev.tc');
    const stop = tc.querySelector(':scope > .panel-stop-btn');
    click(stop);
    // Output streamed before the interrupt, without a trailing newline.
    send(win, {type: 'system_output', text: 'partial output', tabId: TAB});
    send(win, {
      type: 'tool_result',
      tool_name: 'Bash',
      content: 'User interrupted the tool call.',
      is_error: false,
      interrupted: true,
      tabId: TAB,
      ts: Date.now(),
    });
    assert.ok(tc.dataset.timeDone, 'the panel is sealed');
    const bashOut = tc.querySelector(
      ':scope > .bash-panel > .bash-panel-content',
    );
    assert.strictEqual(
      bashOut.textContent,
      'partial output\nUser interrupted the tool call.',
      'the interrupt message ends the streamed Bash output on its own line',
    );
    assert.strictEqual(
      display(win, stop),
      'none',
      'hidden once the tool returned',
    );
    assert.strictEqual(
      display(win, tc.querySelector(':scope > .panel-copy-btn')),
      'flex',
    );
    assert.ok(
      tc.textContent.indexOf('User interrupted the tool call.') >= 0,
      'the result text is shown in the panel',
    );
    assert.strictEqual(
      win.PanelCopy.getRawText(tc).indexOf('Stop this tool call'),
      -1,
      'the copy text never contains the button',
    );
  });

  await test('Bash interrupt message: empty and newline-terminated streams', () => {
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'true',
      tabId: TAB,
      ts: Date.now(),
    });
    let tcs = out.querySelectorAll('.ev.tc');
    let tc = tcs[tcs.length - 1];
    send(win, {
      type: 'tool_result',
      tool_name: 'Bash',
      content: 'User interrupted the tool call.',
      interrupted: true,
      tabId: TAB,
      ts: Date.now(),
    });
    assert.strictEqual(
      tc.querySelector(':scope > .bash-panel > .bash-panel-content')
        .textContent,
      'User interrupted the tool call.',
      'no separator when nothing was streamed',
    );
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'echo a',
      tabId: TAB,
      ts: Date.now(),
    });
    tcs = out.querySelectorAll('.ev.tc');
    tc = tcs[tcs.length - 1];
    send(win, {type: 'system_output', text: 'a\n', tabId: TAB});
    send(win, {
      type: 'tool_result',
      tool_name: 'Bash',
      content: 'User interrupted the tool call.',
      interrupted: true,
      tabId: TAB,
      ts: Date.now(),
    });
    assert.strictEqual(
      tc.querySelector(':scope > .bash-panel > .bash-panel-content')
        .textContent,
      'a\nUser interrupted the tool call.',
      'no double newline after a newline-terminated stream',
    );
    send(win, {
      type: 'tool_call',
      name: 'Bash',
      command: 'echo b',
      tabId: TAB,
      ts: Date.now(),
    });
    tcs = out.querySelectorAll('.ev.tc');
    tc = tcs[tcs.length - 1];
    send(win, {type: 'system_output', text: 'b\n', tabId: TAB});
    send(win, {
      type: 'tool_result',
      tool_name: 'Bash',
      content: '',
      tabId: TAB,
      ts: Date.now(),
    });
    assert.strictEqual(
      tc.querySelector(':scope > .bash-panel > .bash-panel-content')
        .textContent,
      'b\n',
      'an ordinary Bash result adds nothing',
    );
  });

  await test('a non-Bash tool panel has a working Stop too', () => {
    send(win, {
      type: 'tool_call',
      name: 'ask_user_question',
      description: 'Which one?',
      tabId: TAB,
      ts: Date.now(),
    });
    const tcs = out.querySelectorAll('.ev.tc');
    const tc = tcs[tcs.length - 1];
    const stop = tc.querySelector(':scope > .panel-stop-btn');
    assert.strictEqual(display(win, stop), 'flex');
    const before = wv.posted.length;
    click(stop);
    const msg = wv.posted.slice(before).find(m => m.type === 'interruptTool');
    assert.strictEqual(msg.toolName, 'ask_user_question');
    assert.ok(!('callId' in msg), 'an event without a callId sends none');
    assert.strictEqual(tc.dataset.callId, undefined);
    send(win, {
      type: 'tool_result',
      content: 'User interrupted the tool call.',
      tabId: TAB,
      ts: Date.now(),
    });
    assert.strictEqual(display(win, stop), 'none');
  });

  await test('the finish panel has no Stop button', () => {
    send(win, {
      type: 'tool_call',
      name: 'finish',
      description: 'done',
      callId: 99,
      tabId: TAB,
      ts: Date.now(),
    });
    const tcs = out.querySelectorAll('.ev.tc');
    const tc = tcs[tcs.length - 1];
    assert.strictEqual(tc.dataset.callId, '99');
    assert.strictEqual(tc.querySelector(':scope > .panel-stop-btn'), null);
    assert.ok(tc.querySelector(':scope > .panel-copy-btn'));
  });

  await test('a replayed still-running tool call shows Stop; finished ones do not', () => {
    const wv2 = makeWebview();
    const win2 = wv2.win;
    const TAB2 = bootRunningTab(wv2);
    const t0 = Date.now() - 60000;
    send(win2, {
      type: 'task_events',
      tabId: TAB2,
      task: 'replayed',
      task_id: 'task-replay-1',
      chat_id: 'chat-toolstop',
      extra: '',
      events: [
        {type: 'prompt', text: 'do things', ts: t0},
        {type: 'tool_call', name: 'Bash', command: 'echo one', ts: t0 + 1000},
        {type: 'tool_result', content: 'one', is_error: false, ts: t0 + 2000},
        {type: 'tool_call', name: 'Read', path: '/tmp/x', ts: t0 + 3000},
      ],
    });
    const out2 = win2.document.getElementById('output');
    const tcs = out2.querySelectorAll('.ev.tc');
    assert.strictEqual(tcs.length, 2);
    assert.strictEqual(
      display(win2, tcs[0].querySelector(':scope > .panel-stop-btn')),
      'none',
      'a finished replayed tool call has no visible Stop',
    );
    const openStop = tcs[1].querySelector(':scope > .panel-stop-btn');
    assert.strictEqual(
      display(win2, openStop),
      'flex',
      'the open replayed call can be stopped',
    );
    const before = wv2.posted.length;
    click(openStop);
    const msg = wv2.posted.slice(before).find(m => m.type === 'interruptTool');
    assert.ok(msg && msg.tabId === TAB2 && msg.toolName === 'Read');
    // The task ending seals every open panel: the button goes away.
    send(win2, {type: 'task_stopped', tabId: TAB2, endTs: Date.now()});
    assert.strictEqual(display(win2, openStop), 'none');
  });
}

async function run() {
  await test(
    'panelCopy.addStopButton: button, click state, reset and guards',
    panelStopBody,
  );
  await webviewTests();
  console.log(`\n${passed} passed, ${failures.length} failed`);
  if (failures.length) {
    failures.forEach(f => console.error(`FAILED: ${f.name}\n${f.error.stack}`));
    process.exit(1);
  }
  // The webviews keep their panel tickers and stop-button reset timers
  // alive; the suite is done, so end the process explicitly.
  process.exit(0);
}

function panelStopBody() {
  const dom = new JSDOM('<!DOCTYPE html><html><body></body></html>', {
    runScripts: 'dangerously',
    pretendToBeVisual: true,
  });
  const win = dom.window;
  const doc = win.document;
  const s = doc.createElement('script');
  s.textContent = fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8');
  doc.head.appendChild(s);
  const PanelCopy = win.PanelCopy;
  assert.strictEqual(typeof PanelCopy.addStopButton, 'function');
  assert.strictEqual(
    PanelCopy.addStopButton(null, () => {}),
    null,
  );

  const panel = doc.createElement('div');
  panel.className = 'ev tc';
  doc.body.appendChild(panel);
  let stops = 0;
  const btn = PanelCopy.addStopButton(panel, () => stops++, 30);
  assert.ok(btn && btn.classList.contains('panel-stop-btn'));
  assert.strictEqual(btn.getAttribute('aria-label'), 'Stop this tool call');
  assert.strictEqual(
    PanelCopy.addStopButton(panel, () => stops++),
    null,
    'a panel gets one stop button',
  );
  assert.strictEqual(
    panel.querySelectorAll(':scope > .panel-stop-btn').length,
    1,
  );
  // The button is chrome: copy must not read it.
  btn.textContent = 'STOP';
  assert.strictEqual(PanelCopy.getRawText(panel), '');
  assert.strictEqual(PanelCopy.formattedTextFromNode(panel), '');

  click(btn);
  assert.strictEqual(stops, 1);
  assert.ok(btn.classList.contains('stopping'));
  assert.strictEqual(btn.disabled, true);
  click(btn);
  assert.strictEqual(stops, 1, 'a click while stopping is ignored');

  // Explicit clear (a rejected ack) cancels the pending reset timer;
  // a second clear with no timer pending is harmless.
  btn._kissClearStopping();
  assert.ok(!btn.classList.contains('stopping'));
  assert.strictEqual(btn.disabled, false);
  btn._kissClearStopping();

  click(btn);
  assert.strictEqual(stops, 2);
  return new Promise(resolve => {
    win.setTimeout(() => {
      assert.ok(
        !btn.classList.contains('stopping'),
        'the stopping state clears by itself after resetMs',
      );
      assert.strictEqual(btn.disabled, false);
      click(btn);
      assert.strictEqual(stops, 3, 'the button can be used again');
      // Default reset delay (no resetMs given).
      const panel2 = doc.createElement('div');
      doc.body.appendChild(panel2);
      const btn2 = PanelCopy.addStopButton(panel2, () => {});
      click(btn2);
      assert.ok(btn2.classList.contains('stopping'));
      btn2._kissClearStopping();
      resolve();
    }, 80);
  });
}

run().catch(e => {
  console.error(e.stack || e);
  process.exit(1);
});

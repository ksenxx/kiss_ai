// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Validates that the demo replay renders result markdown through the SAME
// HTML sanitizer as live chat rendering (main.js kissSanitize).  demo.js
// historically carried its own weaker copy that did not strip custom
// elements, so markup like <x-evil> survived demo replay but not live chat.

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
      postMessage: msg => {
        posted.push(msg);
        if (win._onPosted) win._onPosted(msg);
      },
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };

  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'demo.js'), 'utf8'));

  return {win, posted};
}

function dispatch(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => {
    setTimeout(resolve, ms);
  });
}

const MALICIOUS_SUMMARY =
  'done <x-evil onclick="window.__pwned=1">boom</x-evil> ' +
  '<a href="javascript:window.__pwned=1">link</a> ' +
  '<h2>HtmlHeading</h2> keep **stars** literal end';

async function runDemoReplay(win) {
  dispatch(win, {type: 'configData', config: {demo_mode: true}, apiKeys: {}});
  dispatch(win, {
    type: 'history',
    offset: 0,
    generation: 0,
    sessions: [
      {
        id: 'chat-1',
        preview: 'Do the demo task',
        title: 'Do the demo task',
        has_events: true,
        ts: Date.now() / 1000,
      },
    ],
  });
  win._onPosted = msg => {
    if (msg.type !== 'resumeSession') return;
    setTimeout(() => {
      dispatch(win, {
        type: 'task_events',
        tabId: msg.tabId,
        events: [{type: 'result', summary: MALICIOUS_SUMMARY}],
        task: 'Do the demo task',
        chat_id: 'chat-1',
        extra: '',
      });
    }, 10);
  };
  const row = win.document.querySelector('#history-list > div');
  assert.ok(row, 'history row rendered');
  row.click();
  const t0 = Date.now();
  while (Date.now() - t0 < 30000) {
    await sleep(50);
    if (!win._demoApi.active && Date.now() - t0 > 500) return;
  }
  throw new Error('demo replay did not finish within 30s');
}

async function main() {
  const {win} = makeWebview();
  assert.strictEqual(
    typeof win._demoApi.kissSanitize,
    'function',
    '_demoApi must expose the shared kissSanitize used by live chat',
  );

  // Prove demo replay delegates to the exact shared function (not a
  // coincidentally-equivalent copy): wrap _demoApi.kissSanitize with a
  // sentinel and require the replay to call through it.
  const sharedSanitize = win._demoApi.kissSanitize;
  let sanitizeCalls = 0;
  win._demoApi.kissSanitize = function (html) {
    sanitizeCalls += 1;
    return sharedSanitize(html);
  };

  await runDemoReplay(win);

  assert.ok(
    sanitizeCalls > 0,
    'demo replay must call the shared _demoApi.kissSanitize (delegation), ' +
      'not a private sanitizer copy',
  );

  const body = win.document.querySelector('.rc-body');
  assert.ok(body, 'demo result panel body rendered');
  assert.ok(
    /done/.test(body.textContent) && /end/.test(body.textContent),
    'result text rendered',
  );
  // The summary wire format is HTML: demo replay must render HTML tags as
  // elements and must NOT run the summary through a Markdown parser.
  assert.ok(
    body.querySelector('h2'),
    'BUG: <h2> in the HTML summary must render as a heading in demo replay',
  );
  assert.ok(
    body.textContent.includes('**stars**') && !body.querySelector('strong'),
    'BUG: demo replay must not parse the HTML summary as Markdown',
  );
  assert.strictEqual(
    body.querySelector('x-evil'),
    null,
    'demo replay must strip custom elements exactly like live chat',
  );
  for (const el of body.querySelectorAll('*')) {
    for (const attr of Array.from(el.attributes)) {
      assert.ok(
        !attr.name.toLowerCase().startsWith('on'),
        'no inline event handlers may survive demo replay',
      );
      if (attr.name.toLowerCase() === 'href') {
        assert.ok(
          !/^javascript:/i.test(attr.value.trim()),
          'javascript: URLs must be stripped in demo replay',
        );
      }
    }
  }

  const live = win._demoApi.kissSanitize(
    '<x-evil onclick="1">a</x-evil><p onmouseover="1">b</p>',
  );
  assert.ok(!/x-evil/.test(live), 'shared sanitizer strips custom elements');
  assert.ok(!/onmouseover/.test(live), 'shared sanitizer strips on* attrs');

  console.log('  ok - demo replay sanitizer matches live chat sanitizer');
}

main().then(
  () => process.exit(0),
  err => {
    console.error(err);
    process.exit(1);
  },
);

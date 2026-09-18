// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end tests for the mobile remote webapp's task-info drawer
// (media/chat.html + media/main.js, remote mode below 900px):
//
// * #meta-drawer-btn (the tab bar's right edge) opens the #meta-panel
//   drawer with its #meta-overlay backdrop and flips aria-expanded;
//   the close button and the backdrop both dismiss it;
// * the 1s getInfoFile poll runs only while the drawer is OPEN (and a
//   task runs): opening polls immediately, closing stops the timer;
// * the status-bar mirror still fills the drawer's #meta-list rows on
//   mobile (the hidden #tab-status-bar keeps updating underneath);
// * switching to the desktop layout (matchMedia flip) closes the
//   drawer state so the docked panel never sits over a live backdrop.

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

function makeWebview(opts) {
  const {desktopMatches = false} = opts || {};
  let html = fs.readFileSync(path.join(MEDIA, 'chat.html'), 'utf8');
  html = html.replace(/\{\{MODEL_NAME\}\}/g, 'test-model');
  html = html.replace(/\{\{[A-Z_]+\}\}/g, '');
  html = html.replace(/<script[^>]*>[\s\S]*?<\/script>/g, '');
  html = html.replace('<body', '<body class="remote-chat"');
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
  win.cancelAnimationFrame = function () {};
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
  const mqlListeners = [];
  const mql = {
    matches: desktopMatches,
    media: '(min-width: 900px)',
    addEventListener: (_type, cb) => mqlListeners.push(cb),
    removeEventListener: () => {},
    addListener: cb => mqlListeners.push(cb),
    removeListener: () => {},
  };
  win.matchMedia = query =>
    query === '(min-width: 900px)'
      ? mql
      : {
          matches: false,
          media: query,
          addEventListener: () => {},
          removeEventListener: () => {},
          addListener: () => {},
          removeListener: () => {},
        };
  win.eval(fs.readFileSync(path.join(MEDIA, 'marked.min.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=metadrawer-main.js',
  );
  const setDesktop = matches => {
    mql.matches = matches;
    for (const cb of mqlListeners.slice()) cb({matches});
  };
  return {win, posted, setDesktop};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function click(win, id) {
  win.document
    .getElementById(id)
    .dispatchEvent(new win.MouseEvent('click', {bubbles: true}));
}

function drawerOpen(win) {
  return win.document.getElementById('meta-panel').classList.contains('open');
}

function overlayOpen(win) {
  return win.document
    .getElementById('meta-overlay')
    .classList.contains('open');
}

function pollCount(wv) {
  return wv.posted.filter(m => m.type === 'getInfoFile').length;
}

async function main() {
  await test(
    'the tab-bar button toggles the drawer; close button and backdrop ' +
      'dismiss it',
    () => {
      const wv = makeWebview();
      const win = wv.win;
      const btn = win.document.getElementById('meta-drawer-btn');
      assert.ok(btn, 'the toggle exists in the tab bar');
      assert.strictEqual(
        btn.parentElement.id,
        'tab-bar',
        'the toggle rides the tab bar, not the composer footer',
      );
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');
      assert.ok(!drawerOpen(win));

      click(win, 'meta-drawer-btn');
      assert.ok(drawerOpen(win), 'button opens the drawer');
      assert.ok(overlayOpen(win), 'the backdrop opens with the drawer');
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'true');

      click(win, 'meta-close');
      assert.ok(!drawerOpen(win), 'the close button dismisses the drawer');
      assert.ok(!overlayOpen(win));
      assert.strictEqual(btn.getAttribute('aria-expanded'), 'false');

      click(win, 'meta-drawer-btn');
      assert.ok(drawerOpen(win));
      click(win, 'meta-overlay');
      assert.ok(!drawerOpen(win), 'the backdrop dismisses the drawer');
      assert.ok(!overlayOpen(win));
    },
  );

  await test(
    'the drawer mirrors the (hidden) status-bar values on mobile',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
        machine: 'phone-host',
      });
      send(win, {type: 'usage_info', total_tokens: 2000, cost: 0.25});
      await sleep(50);
      const text = id => win.document.getElementById(id).textContent;
      assert.strictEqual(text('meta-tokens'), '2.00K');
      assert.strictEqual(text('meta-cost'), '$0.25');
      assert.strictEqual(text('meta-machine'), 'phone-host');
      assert.strictEqual(text('meta-workdir'), '/cfg/dir');
      assert.strictEqual(text('meta-max-budget'), '$42.00');
    },
  );

  await test(
    'getInfoFile polls only while the drawer is open, and the reply ' +
      'renders in the drawer',
    async () => {
      const wv = makeWebview();
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      await sleep(1300);
      assert.strictEqual(
        pollCount(wv),
        0,
        'a closed drawer must not poll (mobile network, every second)',
      );

      click(win, 'meta-drawer-btn');
      await sleep(1300);
      const polls = wv.posted.filter(m => m.type === 'getInfoFile');
      assert.ok(polls.length >= 1, 'opening the drawer starts the poll');
      const poll = polls[polls.length - 1];
      assert.strictEqual(poll.workDir, '/cfg/dir');

      send(win, {
        type: 'infoFile',
        workDir: '/cfg/dir',
        tabId: poll.tabId,
        token: poll.token,
        exists: true,
        sig: '5:9',
        content: 'working on **it**\n',
      });
      const info = win.document.getElementById('meta-info');
      assert.ok(info.classList.contains('visible'));
      assert.ok(
        win.document
          .getElementById('meta-info-content')
          .innerHTML.includes('<strong>it</strong>'),
      );

      click(win, 'meta-close');
      const count = pollCount(wv);
      await sleep(1500);
      assert.strictEqual(
        pollCount(wv),
        count,
        'closing the drawer stops the poll',
      );
    },
  );

  await test('switching to the desktop layout closes the drawer', () => {
    const wv = makeWebview();
    const win = wv.win;
    click(win, 'meta-drawer-btn');
    assert.ok(drawerOpen(win) && overlayOpen(win));
    wv.setDesktop(true);
    assert.ok(
      win.document.body.classList.contains('remote-desktop'),
      'the desktop class follows the media query',
    );
    assert.ok(!drawerOpen(win), 'desktop docking drops the drawer state');
    assert.ok(!overlayOpen(win), 'the backdrop goes with it');
  });

  await test(
    'desktop remote keeps polling without the drawer (docked panel)',
    async () => {
      const wv = makeWebview({desktopMatches: true});
      const win = wv.win;
      send(win, {
        type: 'configData',
        config: {work_dir: '/cfg/dir', max_budget: 42},
        apiKeys: {},
      });
      send(win, {type: 'status', running: true});
      await sleep(1300);
      assert.ok(
        pollCount(wv) >= 1,
        'the docked desktop panel polls with no drawer involved',
      );
    },
  );

  console.log(`\n${passed} passed, ${failures.length} failed`);
  process.exit(failures.length > 0 ? 1 : 0);
}

main().catch(e => {
  console.error(e);
  process.exit(1);
});

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (jsdom) tests for the "Open in Browser" button of an
// .html/.htm content tab (media/main.js against the real chat.html):
//  1. the button sits right before Save in preview and in source mode,
//     a read-only .html preview gets a bar for it alone, and Markdown
//     and other files have none;
//  2. a page on the daemon's own machine has the daemon open the file
//     (`openInBrowser`), and shows the document itself when the daemon
//     answers that nothing opened: the answer to the tab's latest
//     click only, never to a tab that was closed since;
//  3. everywhere else the page uploads the document under a random
//     token (`stageHtmlView`) and opens /view/<token>/<name> in a new
//     tab: the text the tab would save, its pictures inlined, without
//     the preview's own scripts;
//  4. what cannot be sent is said in the bar's status instead.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  sleep,
  click,
  ofType,
  plain,
  stubMonaco,
  captureTimers,
  htmlFrames,
} = require('./simplify2_harness.js');

const WD = '/ws/proj';
const PAGE =
  '<!DOCTYPE html><html><head><title>t</title></head>' +
  '<body><h1>Hello page</h1></body></html>';
const WITH_PICTURE =
  '<html><head></head><body><img src="pic.png"></body></html>';
const PNG = 'iVBORw0KGgo=';
const REMOTE = 'https://box.example/';
// Past the 56 MiB a stageHtmlView frame may carry (SHARE_MAX_HTML_JSON_BYTES).
const TOO_LARGE = 57 * 1024 * 1024;
const STATUS = 'Open in Browser: ';

/**
 * A connected page at *url* whose daemon said whether it can start a
 * browser, with Monaco stubbed, timers and window.open recorded.
 */
function page(url, opensBrowser) {
  let created;
  let timers;
  const made = makePage({
    url,
    beforeScripts: win => {
      created = stubMonaco(win);
      timers = captureTimers(win);
    },
  });
  const opened = [];
  made.win.open = (address, target, features) => {
    opened.push({url: address, target, features});
    return null;
  };
  send(made.win, {type: 'daemonStatus', connected: true});
  send(made.win, {
    type: 'configData',
    config: {work_dir: WD},
    apiKeys: {},
    opensBrowser: !!opensBrowser,
  });
  const tab = made.win._testApi.getActiveTabId();
  return Object.assign(made, {created, timers, opened, tab});
}

/** Open *path* as the daemon's fileContent reply does; the tab's view. */
function openFile(ctx, path, content, extra) {
  ctx.win._testApi.switchToTab(ctx.tab);
  send(ctx.win, {
    type: 'fileContent',
    path,
    name: path.split('/').pop(),
    tabId: ctx.tab,
    content,
    version: '1:1',
    ...extra,
  });
  const views = ctx.win.document.querySelectorAll('.content-tab-view');
  return views[views.length - 1];
}

function browserBtn(view) {
  return view.querySelector('.content-browser-btn');
}

function statusOf(view) {
  const status = view.querySelector('.content-save-status');
  return [status.textContent, status.classList.contains('error')];
}

function staged(ctx) {
  return ofType(ctx.posted, 'stageHtmlView');
}

/** The page's last picture request, answered with a PNG of *base64*. */
function answerPicture(ctx, request, base64) {
  send(ctx.win, {
    type: 'fileContent',
    path: request.path,
    name: 'pic.png',
    tabId: request.tabId,
    token: request.token,
    binary: true,
    mime: 'image/png',
    base64,
  });
}

function lastPictureRequest(ctx) {
  return ofType(ctx.posted, 'openFile')
    .filter(m => /^img-\d+$/.test(m.token))
    .pop();
}

async function testWhichTabsHaveTheButton() {
  const ctx = page(REMOTE, false);
  let view = openFile(ctx, WD + '/site/index.html', PAGE);
  const btn = browserBtn(view);
  const toggle = view.querySelector('.content-mode-btn');
  assert.strictEqual(btn.textContent, 'Open in Browser');
  assert.strictEqual(btn.previousElementSibling, toggle);
  assert.strictEqual(
    btn.nextElementSibling,
    view.querySelector('.content-save-btn'),
    'right before Save',
  );
  click(ctx.win, toggle);
  await sleep(0);
  assert.strictEqual(ctx.created.length, 1, 'the source editor is open');
  assert.strictEqual(browserBtn(view), btn, 'still there in source mode');
  assert.ok(browserBtn(openFile(ctx, WD + '/site/old.htm', '<p>old</p>')));

  view = openFile(ctx, WD + '/notes.md', '# Notes');
  assert.ok(view.querySelector('.content-mode-btn'), 'Markdown has its toggle');
  assert.strictEqual(browserBtn(view), null, 'and no Open in Browser');
  view = openFile(ctx, WD + '/app.py', 'print(1)\n');
  assert.ok(view.querySelector('.content-save-btn'));
  assert.strictEqual(browserBtn(view), null);

  // A read-only .html preview: the path, the button and the reload
  // button, nothing to save or to edit.
  view = openFile(ctx, WD + '/big.html', PAGE, {version: undefined});
  const bar = view.querySelector('.content-save-bar');
  assert.ok(
    bar.querySelector('.content-save-path').textContent.includes('big.html'),
  );
  assert.deepStrictEqual(
    Array.from(bar.querySelectorAll('button'), b => b.className),
    ['content-browser-btn', 'content-refresh-btn'],
  );
  view = openFile(ctx, WD + '/big.md', '# Big', {version: undefined});
  assert.strictEqual(view.querySelector('.content-save-bar'), null);
  ctx.win.close();
  console.log('  ok - .html/.htm tabs have the button, other files do not');
}

function testDaemonMachineOpensTheFile() {
  const path = WD + '/site/index.html';
  for (const url of ['https://localhost:8787/', 'https://127.0.0.1/']) {
    const ctx = page(url, true);
    click(ctx.win, browserBtn(openFile(ctx, path, PAGE)));
    assert.deepStrictEqual(plain(ofType(ctx.posted, 'openInBrowser')), [
      {
        path,
        workDir: WD,
        tabId: ctx.tab,
        token: 'browser:1',
        type: 'openInBrowser',
      },
    ]);
    assert.deepStrictEqual([staged(ctx), ctx.opened], [[], []], url);
    ctx.win.close();
  }
  const ctx = page('https://[::1]:8787/', true);
  const answer = (opened, token) =>
    send(ctx.win, {type: 'openInBrowserResult', opened, path, token});
  let view = openFile(ctx, path, PAGE);
  click(ctx.win, browserBtn(view));
  answer(true, 'browser:1');
  // Answers no click waits for: to a click already answered, to a
  // token never issued, without a token, and to the first of two
  // clicks.
  answer(false, 'browser:1');
  answer(false, 'browser:9');
  answer(false, undefined);
  click(ctx.win, browserBtn(view));
  click(ctx.win, browserBtn(view));
  answer(false, 'browser:2');
  assert.deepStrictEqual([staged(ctx), ctx.opened], [[], []]);
  // The daemon opened nothing after all: the page's browser shows it.
  answer(false, 'browser:3');
  assert.strictEqual(staged(ctx).length, 1);
  assert.strictEqual(staged(ctx)[0].html, PAGE);
  assert.strictEqual(
    ctx.opened[0].url,
    '/view/' + staged(ctx)[0].token + '/index.html',
  );
  // The tab that asked was closed, and its file opened again in a new
  // tab nobody clicked: the late answer opens nothing.
  click(ctx.win, browserBtn(view));
  click(
    ctx.win,
    ctx.win.document.querySelector(
      `#tab-list .chat-tab[data-tab-id="${ctx.win._testApi.getActiveTabId()}"] .chat-tab-close`,
    ),
  );
  view = openFile(ctx, path, '<p>opened again</p>');
  answer(false, 'browser:4');
  assert.strictEqual(staged(ctx).length, 1);
  // So does the answer to a click on a view that was reloaded since.
  click(ctx.win, browserBtn(view));
  openFile(ctx, path, '<p>reloaded</p>', {version: '2:2'});
  answer(false, 'browser:5');
  assert.deepStrictEqual([staged(ctx).length, ctx.opened.length], [1, 1]);
  assert.strictEqual(ofType(ctx.posted, 'openInBrowser').length, 5);
  ctx.win.close();
  console.log('  ok - on its own machine the daemon opens the file');
}

function testWhereTheDaemonCannotOpenIt() {
  // A daemon without a browser to start, and a file of a registered
  // machine (whose browser would open on that machine).
  const headless = page('https://localhost/', false);
  click(headless.win, browserBtn(openFile(headless, WD + '/a.html', PAGE)));
  const machine = page('https://localhost/', true);
  const view = openFile(machine, '/box/a.html', PAGE, {machine: 'gpu-box'});
  click(machine.win, browserBtn(view));
  for (const ctx of [headless, machine]) {
    assert.strictEqual(ofType(ctx.posted, 'openInBrowser').length, 0);
    assert.strictEqual(staged(ctx).length, 1);
    assert.strictEqual(
      staged(ctx)[0].machine,
      undefined,
      'staged on this daemon',
    );
    assert.strictEqual(ctx.opened.length, 1);
    ctx.win.close();
  }
  console.log('  ok - a headless daemon or a machine file is staged');
}

async function testRemotePageStagesTheDocument() {
  const ctx = page(REMOTE, true);
  const view = openFile(ctx, WD + '/site/my page.html', PAGE);
  click(ctx.win, browserBtn(view));
  const [first] = staged(ctx);
  assert.match(first.token, /^[0-9a-f]{32}$/);
  assert.deepStrictEqual(plain(first), {
    token: first.token,
    name: 'my page.html',
    html: PAGE,
    type: 'stageHtmlView',
  });
  assert.deepStrictEqual(ctx.opened, [
    {
      url: '/view/' + first.token + '/my%20page.html',
      target: '_blank',
      features: 'noopener',
    },
  ]);
  assert.strictEqual(ofType(ctx.posted, 'openInBrowser').length, 0);
  assert.ok(htmlFrames(ctx.win)[0].srcdoc.includes('<script'));
  assert.ok(!first.html.includes('<script'), 'no preview-only script');

  // The source editor's unsaved text is what the browser shows; every
  // click has a token of its own.
  click(ctx.win, view.querySelector('.content-mode-btn'));
  await sleep(0);
  ctx.created[0].editor._type('<p>edited, not saved</p>');
  click(ctx.win, browserBtn(view));
  const second = staged(ctx)[1];
  assert.strictEqual(second.html, '<p>edited, not saved</p>');
  assert.notStrictEqual(second.token, first.token);
  assert.deepStrictEqual(statusOf(view), ['Unsaved changes', false]);

  // Too large for one frame: said in the bar, and no empty tab opens.
  ctx.created[0].editor._type('<p>' + 'x'.repeat(TOO_LARGE) + '</p>');
  click(ctx.win, browserBtn(view));
  assert.deepStrictEqual(statusOf(view), [
    STATUS + 'this page is too large to send to the browser',
    true,
  ]);
  assert.deepStrictEqual([staged(ctx).length, ctx.opened.length], [2, 2]);
  ctx.win.close();
  console.log('  ok - a remote page stages the text the tab would save');
}

function testPicturesAreInlined() {
  const ctx = page(REMOTE, false);
  const view = openFile(ctx, WD + '/r/report.html', WITH_PICTURE);
  click(ctx.win, browserBtn(view));
  // The tab opens with the click; the upload waits for the picture.
  assert.strictEqual(ctx.opened.length, 1);
  assert.strictEqual(staged(ctx).length, 0);
  assert.strictEqual(lastPictureRequest(ctx).path, WD + '/r/pic.png');
  answerPicture(ctx, lastPictureRequest(ctx), PNG);
  assert.strictEqual(staged(ctx).length, 1);
  assert.ok(
    staged(ctx)[0].html.includes('src="data:image/png;base64,' + PNG + '"'),
  );
  assert.ok(!staged(ctx)[0].html.includes('<script'));
  assert.ok(ctx.opened[0].url.includes(staged(ctx)[0].token));

  // A picture that does not come: the document goes after the wait
  // with the src as written, and a late answer stages nothing more.
  click(ctx.win, browserBtn(view));
  const late = lastPictureRequest(ctx);
  ctx.timers
    .filter(t => t.ms === 4000)
    .pop()
    .cb();
  assert.strictEqual(staged(ctx).length, 2);
  assert.ok(staged(ctx)[1].html.includes('src="pic.png"'));
  answerPicture(ctx, late, PNG);
  assert.strictEqual(staged(ctx).length, 2);

  // Pictures that make the document too large are left out, and said.
  click(ctx.win, browserBtn(view));
  answerPicture(ctx, lastPictureRequest(ctx), 'A'.repeat(TOO_LARGE));
  assert.strictEqual(staged(ctx)[2].html, WITH_PICTURE);
  assert.deepStrictEqual(statusOf(view), [
    STATUS + 'the pictures were left out, the page is too large with them',
    true,
  ]);
  ctx.win.close();
  console.log('  ok - pictures are inlined, late or oversized ones are not');
}

function testNotConnected() {
  // On a read-only preview: its bar has a status and no Save.
  const ctx = page(REMOTE, false);
  const view = openFile(ctx, WD + '/big.html', PAGE, {version: undefined});
  send(ctx.win, {type: 'daemonStatus', connected: false});
  click(ctx.win, browserBtn(view));
  assert.deepStrictEqual(statusOf(view), [
    STATUS + 'not connected, try again when reconnected',
    true,
  ]);
  assert.deepStrictEqual([staged(ctx), ctx.opened], [[], []]);
  // The next click takes its message back and goes through.
  send(ctx.win, {type: 'daemonStatus', connected: true});
  click(ctx.win, browserBtn(view));
  assert.deepStrictEqual(statusOf(view), ['', false]);
  assert.strictEqual(staged(ctx)[0].html, PAGE);
  assert.strictEqual(ctx.opened.length, 1);
  ctx.win.close();
  console.log('  ok - a click while disconnected says so and sends nothing');
}

async function main() {
  await testWhichTabsHaveTheButton();
  testDaemonMachineOpensTheFile();
  testWhereTheDaemonCannotOpenIt();
  await testRemotePageStagesTheDocument();
  testPicturesAreInlined();
  testNotConnected();
}

main()
  .then(() => console.log('openInBrowser.test.js: all tests passed'))
  .catch(err => {
    console.error('FAIL:', err && err.stack ? err.stack : err);
    process.exit(1);
  });

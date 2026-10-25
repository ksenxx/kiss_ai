// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// A result, an answer, a Markdown panel or a previewed .html/.md file
// names its pictures by file path (<img src="tmp/plot.png">).  The
// page's origin serves no files, so main.js fetches each one with a
// tokened openFile (web_server._handle_open_file echoes the token) and
// shows it as a data: URI; the reply settles the picture and never
// opens a content tab.  This drives the page (jsdom, media/main.js)
// with those replies over the direct transport:
//  1. a result's file-path pictures are requested and shown; a URL,
//     data:, blob: or #fragment src is left alone;
//  2. an SVG (served as text) becomes a base64 SVG data: URI;
//  3. a reply the daemon cannot serve marks the picture, no tab opens;
//  4. a fileContent without a known token still opens a tab;
//  5. answer, prompt, streamed text and Markdown panels load theirs too;
//  6. a previewed .html file's pictures resolve against its own
//     directory and the frame renders once they are all in;
//  7. a Markdown preview and a report do the same; a picture that
//     fails keeps its src; the safety timer renders a stalled preview.

'use strict';

const assert = require('assert');
const {
  makePage,
  send,
  ofType,
  stubRaf,
  captureTimers,
  fakeMatchMedia,
} = require('./simplify2_harness.js');

const PNG = 'iVBORw0KGgo=';

function page(beforeScripts) {
  const made = makePage({
    extraScripts: ['marked.min.js'],
    beforeScripts: win => {
      stubRaf(win);
      if (beforeScripts) beforeScripts(win);
    },
  });
  send(made.win, {
    type: 'configData',
    config: {work_dir: '/ws/proj'},
    apiKeys: {},
  });
  made.tab = made.win._testApi.getActiveTabId();
  return made;
}

/** The openFile commands posted so far that carry an image token. */
function imageRequests(posted) {
  return ofType(posted, 'openFile').filter(m => /^img-\d+$/.test(m.token));
}

function contentTabs(win) {
  return win._testApi.openTabs().filter(t => t.isContentTab);
}

function imgs(win, sel) {
  return Array.from(win.document.querySelectorAll(sel + ' img'));
}

function pngReply(req, extra) {
  return Object.assign(
    {
      type: 'fileContent',
      path: '/ws/proj/' + req.path,
      name: req.path.split('/').pop(),
      tabId: req.tabId,
      token: req.token,
      binary: true,
      mime: 'image/png',
      base64: PNG,
    },
    extra || {},
  );
}

function testResultPanel() {
  const {win, posted, tab} = page();
  send(win, {
    type: 'result',
    tabId: tab,
    text: 'done',
    success: true,
    summary:
      '<p><img src="tmp/a.png" alt="Plot A">' +
      '<img src="https://cdn.example/x.png" alt="web">' +
      '<img src="data:image/png;base64,AAAA" alt="inline">' +
      '<img src="blob:http://localhost/1" alt="blob">' +
      '<img src="#frag" alt="frag">' +
      '<img src="C:\\shots\\win.png">' +
      '<img src="my%20plot.png"></p>',
  });
  const reqs = imageRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => r.path),
    ['tmp/a.png', 'C:\\shots\\win.png', 'my plot.png'],
    'only file-path srcs are requested, percent-decoded',
  );
  assert.ok(
    reqs.every(r => r.workDir === '/ws/proj' && r.tabId === tab),
    'requests name the task work dir and tab',
  );
  const all = imgs(win, '.rc-body');
  assert.strictEqual(all.length, 7);
  assert.strictEqual(all[0].getAttribute('src'), null, 'file src taken off');
  assert.strictEqual(all[0].dataset.fileSrc, 'tmp/a.png');
  assert.strictEqual(all[0].alt, 'Plot A', 'alt kept');
  assert.strictEqual(all[5].alt, 'C:\\shots\\win.png', 'alt defaults to src');
  assert.strictEqual(all[1].getAttribute('src'), 'https://cdn.example/x.png');
  assert.strictEqual(all[2].getAttribute('src'), null, 'sanitizer drops data:');
  assert.strictEqual(all[3].getAttribute('src'), 'blob:http://localhost/1');
  assert.strictEqual(all[4].getAttribute('src'), '#frag');

  send(win, pngReply(reqs[0]));
  assert.strictEqual(all[0].src, 'data:image/png;base64,' + PNG);
  assert.strictEqual(contentTabs(win).length, 0, 'no tab opened');
  console.log("  ok - a result's file-path pictures load as data: URIs");

  // 2. An SVG comes back as text (like any UTF-8 file).
  send(win, {
    type: 'result',
    tabId: tab,
    text: 'done',
    success: true,
    summary: '<img src="/ws/proj/reports/d.svg" alt="D">',
  });
  const svgReq = imageRequests(posted).pop();
  assert.strictEqual(svgReq.path, '/ws/proj/reports/d.svg');
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/reports/d.svg',
    name: 'd.svg',
    tabId: tab,
    token: svgReq.token,
    content: '<svg xmlns="http://www.w3.org/2000/svg">\u00e9</svg>',
    version: '1:1',
  });
  const svgImg = imgs(win, '.rc-body').pop();
  const prefix = 'data:image/svg+xml;base64,';
  assert.ok(svgImg.src.startsWith(prefix));
  assert.strictEqual(
    Buffer.from(svgImg.src.slice(prefix.length), 'base64').toString('utf8'),
    '<svg xmlns="http://www.w3.org/2000/svg">\u00e9</svg>',
  );
  assert.strictEqual(contentTabs(win).length, 0);
  console.log('  ok - an SVG served as text shows as a base64 data: URI');

  // Two pictures of one file cost one request; the reply settles both.
  const before = imageRequests(posted).length;
  send(win, {
    type: 'result',
    tabId: tab,
    text: 'done',
    success: true,
    summary:
      '<img src="tmp/twice.png" alt="1"><img src="tmp/twice.png" alt="2">',
  });
  const twiceReqs = imageRequests(posted).slice(before);
  assert.deepStrictEqual(
    twiceReqs.map(r => r.path),
    ['tmp/twice.png'],
    'one request per distinct src',
  );
  send(win, pngReply(twiceReqs[0]));
  const twice = imgs(win, '.rc-body').slice(-2);
  assert.ok(
    twice.every(img => img.src === 'data:image/png;base64,' + PNG),
    'both pictures show',
  );
  console.log('  ok - pictures sharing a src share one request');

  // 3. A picture the daemon cannot serve: marked, no tab, no src.
  send(win, {
    type: 'result',
    tabId: tab,
    text: 'done',
    success: true,
    summary: '<img src="tmp/missing.png" alt="gone"><img src="notes.txt">',
  });
  const [missing, textFile] = imageRequests(posted).slice(-2);
  send(win, {
    type: 'fileContent',
    path: 'tmp/missing.png',
    name: 'missing.png',
    tabId: tab,
    token: missing.token,
    error: 'No such file',
  });
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/notes.txt',
    name: 'notes.txt',
    tabId: tab,
    token: textFile.token,
    content: 'plain text',
    version: '1:1',
  });
  const [gone, txt] = imgs(win, '.rc-body').slice(-2);
  assert.strictEqual(gone.getAttribute('src'), null);
  assert.strictEqual(gone.title, 'Cannot show tmp/missing.png');
  assert.ok(gone.classList.contains('inline-img-missing'));
  assert.strictEqual(txt.title, 'Cannot show notes.txt', 'a text file too');
  assert.strictEqual(
    contentTabs(win).length,
    0,
    'a failed picture opens no tab',
  );
  console.log('  ok - a picture the daemon cannot serve is marked, no tab');

  // The same reply a second time is nobody's picture any more: it
  // opens a tab like any other fileContent (4), as does an unknown
  // token.
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/notes.txt',
    name: 'notes.txt',
    tabId: tab,
    token: 'img-999',
    content: 'plain text',
    version: '1:1',
  });
  assert.strictEqual(contentTabs(win).length, 1, 'unknown token opens a tab');
  console.log('  ok - a fileContent without a pending token opens a tab');
  win.close();
}

function testOtherPanels() {
  const {win, posted, tab} = page();
  send(win, {
    type: 'ask_answer',
    tabId: tab,
    question: 'q',
    text: '<p><img src="tmp/answer.png" alt="ans"></p>',
    success: true,
  });
  send(win, {type: 'prompt', tabId: tab, text: '![shot](tmp/prompt.png)'});
  send(win, {type: 'text_delta', tabId: tab, text: '![t](tmp/text.png)'});
  send(win, {type: 'text_end', tabId: tab});
  const reqs = imageRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => r.path).sort(),
    ['tmp/answer.png', 'tmp/prompt.png', 'tmp/text.png'],
    'answer, prompt and streamed-text panels request their pictures',
  );
  for (const req of reqs) send(win, pngReply(req));
  const loaded = imgs(win, '#output').filter(
    i => i.src === 'data:image/png;base64,' + PNG,
  );
  assert.strictEqual(loaded.length, 3, 'every panel shows its picture');
  assert.strictEqual(contentTabs(win).length, 0);
  console.log('  ok - answer, prompt and text panels load their pictures');
  win.close();
}

function frameOf(win) {
  const frames = Array.from(
    win.document.querySelectorAll('.content-html-frame'),
  ).filter(f => f.closest('.content-tab-view').style.display !== 'none');
  assert.strictEqual(frames.length, 1, 'one html frame on screen');
  return frames[0];
}

function testHtmlPreview() {
  let timers = [];
  const {win, posted, tab} = page(w => {
    timers = captureTimers(w);
  });
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/reports/r.html',
    name: 'r.html',
    tabId: tab,
    content:
      '<html><body><img src="diagram.png"><img src=\'../tmp/x.png\'>' +
      '<img src="a&amp;b.png"><img src="diagram.png">' +
      '<img src="https://cdn.example/x.png"></body></html>',
  });
  assert.strictEqual(contentTabs(win).length, 1, 'the preview tab opened');
  const frame = frameOf(win);
  assert.strictEqual(frame.srcdoc, '', 'the frame waits for its pictures');
  const reqs = imageRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => r.path),
    [
      '/ws/proj/reports/diagram.png',
      '/ws/proj/reports/../tmp/x.png',
      '/ws/proj/reports/a&b.png',
    ],
    'paths resolve against the file directory, entities decoded, no repeats',
  );
  assert.ok(
    reqs.every(r => r.workDir === '/ws/proj' && r.tabId === tab),
    'requests name the owner tab and its work dir',
  );
  send(win, pngReply(reqs[0]));
  send(win, pngReply(reqs[1]));
  assert.strictEqual(frame.srcdoc, '', 'still waiting for the last one');
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/reports/a&b.png',
    name: 'a&b.png',
    tabId: tab,
    token: reqs[2].token,
    error: 'No such file',
  });
  const doc = frame.srcdoc;
  const uri = 'data:image/png;base64,' + PNG;
  assert.ok(doc.includes('<img src="' + uri + '"><img src="' + uri + '">'));
  assert.ok(doc.includes('<img src="a&amp;b.png">'), 'a failed one keeps src');
  assert.ok(doc.includes('<img src="' + uri + '">' + '<img src="https://cdn'));
  assert.strictEqual(contentTabs(win).length, 1, 'no extra tab');
  const wait = timers.filter(t => t.ms === 10000);
  assert.strictEqual(wait.length, 1, 'one safety timer armed');
  wait[0].cb();
  assert.strictEqual(frame.srcdoc, doc, 'the late timer changes nothing');
  console.log("  ok - an .html preview's pictures resolve against its folder");

  // A preview without file-path pictures renders at once.
  win._testApi.switchToTab(tab);
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/reports/plain.html',
    name: 'plain.html',
    tabId: tab,
    content: '<html><body><img src="https://cdn.example/x.png"></body></html>',
  });
  assert.ok(frameOf(win).srcdoc.includes('https://cdn.example/x.png'));
  assert.strictEqual(imageRequests(posted).length, 3, 'nothing requested');
  console.log('  ok - a preview without file pictures renders at once');
  win.close();
}

function testMarkdownPreviewAndReport() {
  let timers = [];
  const {win, posted, tab} = page(w => {
    timers = captureTimers(w);
  });
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/notes/readme.md',
    name: 'readme.md',
    tabId: tab,
    content: '# Notes\n\n![chart](img/chart.png)\n',
  });
  const frame = frameOf(win);
  assert.strictEqual(frame.srcdoc, '');
  const req = imageRequests(posted).pop();
  assert.strictEqual(req.path, '/ws/proj/notes/img/chart.png');
  send(win, pngReply(req));
  assert.ok(
    frame.srcdoc.includes('src="data:image/png;base64,' + PNG + '"'),
    'the Markdown preview shows the picture',
  );
  console.log("  ok - a Markdown preview's pictures load");

  // A stalled picture (its machine never answers): the safety timer
  // renders the preview with the pictures it has.
  win._testApi.switchToTab(tab);
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/notes/other.md',
    name: 'other.md',
    tabId: tab,
    content: '![a](a.png) ![b](b.png)',
  });
  const stalled = frameOf(win);
  const [a] = imageRequests(posted).slice(-2);
  send(win, pngReply(a));
  assert.strictEqual(stalled.srcdoc, '', 'b.png still pending');
  const wait = timers.filter(t => t.ms === 10000).pop();
  wait.cb();
  assert.ok(stalled.srcdoc.includes('data:image/png;base64,' + PNG));
  assert.ok(
    stalled.srcdoc.includes('src="b.png"'),
    'the stalled one keeps src',
  );
  // The late picture is painted in by one more render.
  const [, b] = imageRequests(posted).slice(-2);
  send(win, pngReply(b));
  assert.ok(
    !stalled.srcdoc.includes('src="b.png"') &&
      stalled.srcdoc.split('data:image/png;base64,').length === 3,
    'a picture answered after the timer is shown too',
  );
  console.log('  ok - a stalled preview renders after the safety timer');

  // A finished task's report (the synthetic fileContent of
  // openReadyReportTabs) resolves its pictures the same way.
  win._testApi.switchToTab(tab);
  const report = '<html><body><img src="fig1.png"></body></html>';
  send(win, {
    type: 'tool_call',
    name: 'Write',
    path: 'reports/summary.html',
    content: report,
    tabId: tab,
  });
  send(win, {
    type: 'tool_result',
    content: 'Successfully wrote 1 characters to reports/summary.html',
    is_error: false,
    tool_name: 'Write',
    path: 'reports/summary.html',
    tabId: tab,
  });
  send(win, {type: 'task_done', success: true, tabId: tab});
  const rep = contentTabs(win).find(t => t.title.includes('summary.html'));
  assert.ok(rep, 'the report tab opened at task_done');
  const reportReq = imageRequests(posted).pop();
  assert.strictEqual(reportReq.path, 'reports/fig1.png');
  assert.strictEqual(reportReq.tabId, tab, 'asked for the task tab');
  assert.strictEqual(reportReq.workDir, '/ws/proj');
  send(win, pngReply(reportReq));
  assert.ok(
    frameOf(win).srcdoc.includes('src="data:image/png;base64,' + PNG + '"'),
    'the report shows its picture',
  );
  console.log("  ok - a report's pictures resolve against the report folder");
  win.close();
}

function testEdgeCases() {
  // The desktop layout: its docked task-info panel polls for updates.
  const {win, posted, tab} = page(w => fakeMatchMedia(w, true));
  // 8. The document is parsed, not searched as text: `<img` inside
  // a script is left alone; an unquoted src and an attribute holding
  // `>` are real images.  A query or fragment is not part of the
  // file name; a file of the root directory keeps its directory.
  send(win, {
    type: 'fileContent',
    path: '/index.html',
    name: 'index.html',
    tabId: tab,
    content:
      '<!DOCTYPE html><html><head><script>const h = "<img src=\'a.png\'>";' +
      '</script></head><body><img src=pic.png><img alt="a > b" src="c.png">' +
      '<img src="chart.png?v=2"><img src="sprite.svg#view"></body></html>',
  });
  const reqs = imageRequests(posted);
  assert.deepStrictEqual(
    reqs.map(r => r.path),
    ['/pic.png', '/c.png', '/chart.png', '/sprite.svg'],
    'script text skipped; unquoted src, query and fragment handled',
  );
  const frame = frameOf(win);
  reqs.forEach(r => send(win, pngReply(r)));
  const doc = frame.srcdoc;
  assert.ok(doc.startsWith('<!DOCTYPE html>'), 'doctype kept');
  assert.ok(
    doc.includes('const h = "<img src=\'a.png\'>";'),
    'the script is untouched',
  );
  assert.strictEqual(
    doc.split('data:image/png;base64,').length,
    5,
    'all four pictures painted',
  );
  console.log('  ok - previews are parsed: scripts kept, odd srcs found');

  // 9. A dropped connection settles every pending picture: the panel
  // shows alt text, a waiting preview renders with what it has.
  win._testApi.switchToTab(tab);
  send(win, {
    type: 'result',
    tabId: tab,
    text: 'done',
    success: true,
    summary: '<img src="tmp/late.png" alt="late">',
  });
  const late = imgs(win, '.rc-body').pop();
  assert.ok(late && late.dataset.fileSrc === 'tmp/late.png');
  send(win, {
    type: 'fileContent',
    path: '/ws/proj/reports/w.html',
    name: 'w.html',
    tabId: tab,
    content: '<html><body><img src="never.png"></body></html>',
  });
  const waiting = frameOf(win);
  assert.strictEqual(waiting.srcdoc, '');
  send(win, {type: 'daemonStatus', connected: false, reconnecting: true});
  assert.ok(late.classList.contains('inline-img-missing'));
  assert.ok(waiting.srcdoc.includes('src="never.png"'), 'preview rendered');
  const before = imageRequests(posted).length;
  send(win, pngReply(imageRequests(posted).pop()));
  assert.ok(late.classList.contains('inline-img-missing'), 'reply forgotten');
  assert.strictEqual(imageRequests(posted).length, before);
  assert.strictEqual(contentTabs(win).length, 3, 'the stale reply opens a tab');
  console.log('  ok - a dropped connection settles pending pictures');

  // 10. The task-update info panel loads its pictures as the polled
  // task's files.
  send(win, {type: 'daemonStatus', connected: true});
  win._testApi.switchToTab(tab);
  send(win, {type: 'status', running: true, tabId: tab});
  const poll = ofType(posted, 'getTaskUpdate').pop();
  assert.ok(poll, 'a getTaskUpdate poll fired');
  send(win, {
    type: 'taskUpdate',
    tabId: poll.tabId,
    token: poll.token,
    taskId: 'task-1',
    exists: true,
    sig: 'sig-1',
    content: '<p><img src="tmp/update.png" alt="upd"></p>',
    error: '',
    running: true,
    cost: 0,
    updatedAt: 0,
  });
  const upd = imageRequests(posted).pop();
  assert.strictEqual(upd.path, 'tmp/update.png');
  assert.strictEqual(upd.tabId, poll.tabId);
  send(win, pngReply(upd));
  const updImg = win.document.querySelector('#meta-info-content img');
  assert.strictEqual(updImg.src, 'data:image/png;base64,' + PNG);
  console.log('  ok - the task-update panel loads its pictures');
  win.close();
}

testResultPanel();
testOtherPanels();
testHtmlPreview();
testMarkdownPreviewAndReport();
testEdgeCases();
console.log('inlineImages.test.js passed');

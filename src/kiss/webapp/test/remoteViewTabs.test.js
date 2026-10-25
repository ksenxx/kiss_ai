// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

// The Terminal and Browser buttons follow the machine picked in the
// chat header's drop-down (kiss/server/remote_views.py on the daemon):
//
//   * a chat bound to `box` opens its terminal there: terminalOpen
//     carries machine 'box' and the chat's folder on that machine, and
//     the tab is titled with the machine's label;
//   * its Browser button sends browserOpen {url:'', machine:'box'};
//   * a page the daemon announces with `machine` is a browser tab
//     labelled with the machine, whose "+" opens the next page on the
//     same machine; the `remote` browserTabs snapshot reconciles the
//     remote tabs alone, the daemon's own the local ones;
//   * an unbound chat behaves as before: no machine field anywhere.

const assert = require('assert');
const {
  makePage: makeChatPage,
  send,
  click,
  stubRaf,
  ofType,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

/** A chat page with browserTab.js / terminalTab.js over stubbed xterm. */
function makePage() {
  return makeChatPage({
    beforeScripts: withTabViews,
    extraScripts: ['browserTab.js', 'terminalTab.js'],
  });
}

function withTabViews(win) {
  stubRaf(win);
  win.Terminal = class {
    constructor() {
      this.cols = 80;
      this.rows = 24;
      this.options = {};
    }
    open() {}
    loadAddon() {}
    attachCustomKeyEventHandler() {}
    write() {}
    onData() {}
    onResize() {}
    focus() {}
    getSelection() {
      return '';
    }
    dispose() {}
  };
  win.FitAddon = {
    FitAddon: class {
      fit() {}
    },
  };
  win.ResizeObserver = class {
    observe() {}
    disconnect() {}
  };
}

/** Structural equality across the jsdom realm (its arrays are not ours), key order aside. */
function same(actual, expected) {
  const sorted = (k, v) =>
    v && typeof v === 'object' && !Array.isArray(v)
      ? Object.fromEntries(
          Object.keys(v)
            .sort()
            .map(key => [key, v[key]]),
        )
      : v;
  assert.strictEqual(
    JSON.stringify(actual, sorted),
    JSON.stringify(expected, sorted),
  );
}

function openChat(win) {
  send(win, {
    type: 'tabs_state',
    tabs: [{tabId: 'a1', chatId: 'chat-1', title: 'a1', workDir: '/ws'}],
  });
  send(win, {type: 'task_events', tabId: 'a1', task: 'the task', events: []});
}

function bindToBox(win) {
  send(win, {
    type: 'machines',
    machines: [
      {name: 'here', local: true, status: 'reachable'},
      {name: 'box', label: 'Box', status: 'reachable', address: 'ssh me@box'},
    ],
    checkedAt: 1,
  });
  send(win, {type: 'tabMachine', tabId: 'a1', machine: 'box'});
  send(win, {
    type: 'machineWorkDirs',
    machine: 'box',
    workDir: '/remote/ws',
    recentWorkDirs: [],
  });
}

async function settle() {
  for (let i = 0; i < 4; i++) await Promise.resolve();
}

const {test, report} = makeRunner();

async function main() {
  await test('the terminal of a chat bound to a machine opens there', async () => {
    const {win, posted} = makePage();
    openChat(win);
    bindToBox(win);
    click(win, win.document.getElementById('terminal-btn'));
    await settle();
    const tab = win._testApi.openTabs().find(t => t.isContentTab);
    assert.ok(tab, 'the terminal tab opened');
    assert.strictEqual(tab.title, 'Terminal (Box)');
    const opens = ofType(posted, 'terminalOpen');
    assert.strictEqual(opens.length, 1);
    assert.strictEqual(opens[0].machine, 'box');
    assert.strictEqual(opens[0].workDir, '/remote/ws');
    assert.strictEqual(opens[0].tab_id, tab.id);
    win.close();
  });

  await test('the Browser button opens a page on the bound machine', () => {
    const {win, posted} = makePage();
    openChat(win);
    bindToBox(win);
    click(win, win.document.getElementById('browser-btn'));
    same(ofType(posted, 'browserOpen'), [
      {type: 'browserOpen', url: '', machine: 'box'},
    ]);
    win.close();
  });

  await test('a remote page is a browser tab labelled with its machine', () => {
    const {win, posted} = makePage();
    openChat(win);
    bindToBox(win);
    send(win, {
      type: 'openBrowserTab',
      tab_id: 'browser__1@box',
      machine: 'box',
      url: 'https://example.com/',
      title: 'Example',
      focus: true,
    });
    send(win, {
      type: 'openBrowserTab',
      tab_id: 'browser__1',
      url: 'https://local.test/',
      title: 'Local',
      focus: false,
    });
    const titles = () =>
      win._testApi
        .openTabs()
        .filter(t => t.isContentTab)
        .map(t => t.id + '=' + t.title);
    same(titles(), ['browser__1@box=Example (Box)', 'browser__1=Local']);

    // The page's title follows (browserState) and keeps the label.
    send(win, {
      type: 'browserState',
      tab_id: 'browser__1@box',
      machine: 'box',
      url: 'https://example.com/x',
      title: 'Moved',
    });
    assert.ok(titles().includes('browser__1@box=Moved (Box)'));

    // The "+" of the remote tab opens the next page on the same machine;
    // the local tab's "+" on this machine.
    const views = win.document.querySelectorAll('.browser-tab-view');
    assert.strictEqual(views.length, 2);
    const plusOf = holder =>
      holder.querySelector('.browser-nav-btn[title="New browser tab"]');
    click(win, plusOf(views[0]));
    click(win, plusOf(views[1]));
    same(ofType(posted, 'browserOpen'), [
      {type: 'browserOpen', url: '', machine: 'box'},
      {type: 'browserOpen', url: ''},
    ]);

    // The remote snapshot reconciles the remote tabs alone: the remote
    // tab closes, the local one stays; the daemon's own snapshot then
    // closes the local one.
    send(win, {type: 'browserTabs', remote: true, tabs: []});
    same(titles(), ['browser__1=Local']);
    send(win, {
      type: 'browserTabs',
      remote: true,
      tabs: [
        {
          tab_id: 'browser__2@box',
          url: 'https://example.com/2',
          title: 'Two',
          focus: false,
          machine: 'box',
        },
      ],
    });
    same(titles(), ['browser__1=Local', 'browser__2@box=Two (Box)']);
    send(win, {type: 'browserTabs', tabs: []});
    same(titles(), ['browser__2@box=Two (Box)']);
    // The daemon closing the page (the stream lost, the page gone) drops the tab.
    send(win, {
      type: 'closeBrowserTab',
      tab_id: 'browser__2@box',
      machine: 'box',
    });
    same(titles(), []);
    win.close();
  });

  await test('an unbound chat keeps the terminal and the browser on this machine', async () => {
    const {win, posted} = makePage();
    openChat(win);
    click(win, win.document.getElementById('terminal-btn'));
    await settle();
    const tab = win._testApi.openTabs().find(t => t.isContentTab);
    assert.strictEqual(tab.title, 'Terminal');
    const open = ofType(posted, 'terminalOpen')[0];
    assert.ok(open, 'terminalOpen was sent');
    assert.ok(!('machine' in open), 'no machine field on a local terminal');
    assert.strictEqual(open.workDir, '/ws');
    click(win, win.document.getElementById('browser-btn'));
    same(ofType(posted, 'browserOpen'), [{type: 'browserOpen', url: ''}]);
    win.close();
  });

  report('remoteViewTabs');
}

main();

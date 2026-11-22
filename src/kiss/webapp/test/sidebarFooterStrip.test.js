// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// End-to-end (JSDOM) tests, driving the real chat.html + api.js +
// terminalTab.js + main.js, for the strip under the Chats list
// (#sidebar-footer) and the one under the task-info panel
// (#meta-footer):
// * the Chats strip holds the user's name, the three appearance
//   buttons, the two text size buttons and Settings; the appearance
//   buttons say what a press switches to and show their state;
// * the task-info strip holds Terminal, Browser, Git commit and Create
//   pull request;
// * a button leaves the drawer that would cover what it opens (Settings
//   the Chats drawer; Terminal, Browser and Git commit the task-info
//   drawer), while a docked desktop panel stays; closing the Settings
//   sheet hands the focus to the button that opened it, or to the
//   drawer's way back when that button went off screen with its drawer;
// * the two font buttons move the size of all text 1px at a time from
//   10px to 20px, remembered in localStorage, and the Monaco editors
//   and the terminals take it;
// * a rendered Markdown preview is built at that size and is told
//   every later one (jsdom does not load the preview's document:
//   tests/webapp/test_sidebar_footer_strip.py shows its text resize).

'use strict';

const assert = require('assert');
const {VirtualConsole} = require('jsdom');
const {
  makePage,
  send,
  click,
  byId,
  ofType,
  plain,
  settle,
  openChat,
  stubRaf,
  stubDesktop,
  stubMonaco,
  installXtermStub,
} = require('./simplify2_harness');
const {makeRunner} = require('./ui_antipattern_harness');

// The buttons of the two strips, left to right: each by its id, a font
// button, which has none, by its step.
const CHATS_STRIP = [
  'theme-btn',
  'contrast-btn',
  'theme-auto-btn',
  'font-step -1',
  'font-step 1',
  'settings-btn',
];
const INFO_STRIP = [
  'terminal-btn',
  'browser-btn',
  'autocommit-btn',
  'create-pr-btn',
];
const PHONE_UA =
  'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) Mobile/15E148';

const windows = [];
const {test, report} = makeRunner({windows});

/**
 * A remote chat page showing one idle chat, over the harness's xterm
 * and Monaco stand-ins (`xterm`, `editors`: what each was given).
 * `desktop` docks the two panels; `fontSize` is the saved text size;
 * `virtualConsole` receives the page's jsdom errors; `userAgent` is
 * the browser the page takes itself to run in.
 */
function makeStripPage({
  desktop = false,
  fontSize,
  virtualConsole,
  userAgent,
} = {}) {
  const made = {};
  const page = makePage({
    bodyClass: 'remote-chat',
    virtualConsole: virtualConsole,
    userAgent: userAgent,
    extraScripts: ['terminalTab.js', 'contentContextMenu.js'],
    beforeScripts: win => {
      stubRaf(win);
      made.xterm = installXtermStub(win);
      made.editors = stubMonaco(win);
      if (desktop) stubDesktop(win);
      if (fontSize) win.localStorage.setItem('kissRemoteFontSize', fontSize);
    },
  });
  windows.push(page.win);
  openChat(page.win);
  return Object.assign(page, made);
}

/** The buttons of the strip `id`, in document order. */
function stripButtons(win, id) {
  return Array.from(byId(win, id).querySelectorAll('button'));
}

/** What CHATS_STRIP and INFO_STRIP call the strip button `btn`. */
function buttonName(btn) {
  return btn.id || 'font-step ' + btn.dataset.fontStep;
}

/** The button of the Chats strip that moves the text size by `step` px. */
function fontButton(win, step) {
  return win.document.querySelector('[data-font-step="' + step + '"]');
}

/** Assert the strip button `btn` is an icon button that says what it
 * does: to a screen reader (its name) and on hover (its tooltip). */
function assertNamed(btn) {
  const name = buttonName(btn);
  assert.ok(btn.classList.contains('sidebar-tool-btn'), name);
  assert.ok(btn.querySelector('svg'), name + ' shows an icon');
  assert.ok(btn.getAttribute('aria-label'), name + ' has a name');
  assert.ok(btn.dataset.tooltip, name + ' has a tooltip');
}

/** Click a button as a pointer does: it takes the focus first. */
function press(win, btn) {
  btn.focus();
  click(win, btn);
}

/** The text size the page shows, as the root's `--theme-font-size`. */
function rootSize(win) {
  const root = win.document.documentElement;
  return root.style.getPropertyValue('--theme-font-size');
}

/** Open a file in a content tab of the chat `a1` (a Monaco editor,
 * or the preview of a `.md` file). */
async function openFile(win, name) {
  send(win, {
    type: 'fileContent',
    path: '/ws/' + name,
    name: name,
    content: 'text',
    version: 'v1',
    tabId: 'a1',
  });
  await settle();
}

async function main() {
  await test('the Chats strip: user, appearance, text size, Settings', async () => {
    const errors = [];
    const virtualConsole = new VirtualConsole();
    virtualConsole.on('jsdomError', err => errors.push(String(err)));
    const {win} = makeStripPage({virtualConsole});
    const body = win.document.body;
    const footer = byId(win, 'sidebar-footer');
    const theme = byId(win, 'theme-btn');
    const contrast = byId(win, 'contrast-btn');
    const auto = byId(win, 'theme-auto-btn');
    send(win, {type: 'configData', config: {}, apiKeys: {}, user: 'alice'});
    const user = footer.firstElementChild;
    assert.strictEqual(user.id, 'sidebar-user');
    assert.strictEqual(user.textContent, 'alice');
    assert.strictEqual(user.title, 'alice', 'the tooltip of a cut name');
    assert.deepStrictEqual(
      stripButtons(win, 'sidebar-footer').map(buttonName),
      CHATS_STRIP,
    );
    stripButtons(win, 'sidebar-footer').forEach(assertNamed);

    // The theme button names the palette a press switches to, under
    // that palette's icon (a moon while light, a sun while dark).
    for (let i = 0; i < 2; i++) {
      const light = body.classList.contains('light-theme');
      const icon = theme.innerHTML;
      const name = light ? 'Dark mode' : 'Light mode';
      assert.strictEqual(theme.getAttribute('aria-label'), name);
      assert.strictEqual(theme.dataset.tooltip, name);
      click(win, theme);
      assert.strictEqual(body.classList.contains('light-theme'), !light);
      assert.strictEqual(
        win.localStorage.getItem('kissRemoteTheme'),
        light ? 'dark' : 'light',
      );
      assert.notStrictEqual(theme.innerHTML, icon, 'sun and moon swap');
    }

    // So does the contrast button.
    assert.strictEqual(contrast.getAttribute('aria-label'), 'High contrast');
    click(win, contrast);
    assert.ok(body.classList.contains('high-contrast'));
    assert.strictEqual(contrast.getAttribute('aria-label'), 'Low contrast');
    assert.strictEqual(contrast.dataset.tooltip, 'Low contrast');
    click(win, contrast);
    assert.ok(!body.classList.contains('high-contrast'));
    assert.strictEqual(contrast.dataset.tooltip, 'High contrast');

    // Auto mode is pressed while on; a manual pick leaves it.
    assert.strictEqual(auto.getAttribute('aria-pressed'), 'false');
    click(win, auto);
    assert.strictEqual(win.localStorage.getItem('kissRemoteTheme'), 'auto');
    assert.strictEqual(auto.getAttribute('aria-pressed'), 'true');
    click(win, theme);
    assert.strictEqual(auto.getAttribute('aria-pressed'), 'false');
    stripButtons(win, 'sidebar-footer').forEach(assertNamed);

    win.close();
    await settle();
    assert.deepStrictEqual(errors, []);
  });

  await test('a button leaves the drawer that would cover what it opens', async () => {
    const {win, posted, xterm} = makeStripPage();
    const sidebar = byId(win, 'sidebar');
    const info = byId(win, 'meta-panel');
    const sheet = byId(win, 'settings-panel');
    assert.deepStrictEqual(
      stripButtons(win, 'meta-footer').map(buttonName),
      INFO_STRIP,
    );
    stripButtons(win, 'meta-footer').forEach(assertNamed);
    // Each strip ends its panel: only the panel's resize handle follows.
    for (const [strip, handle] of [
      ['sidebar-footer', 'sidebar-resizer'],
      ['meta-footer', 'meta-resizer'],
    ]) {
      const next = byId(win, strip).nextElementSibling;
      assert.strictEqual(next, byId(win, handle), strip);
      assert.strictEqual(next.nextElementSibling, null, strip);
    }
    /** Click the button `id` of the open task-info drawer. */
    const clickInInfoDrawer = id => {
      if (!info.classList.contains('open')) {
        click(win, byId(win, 'meta-drawer-btn'));
      }
      assert.ok(info.classList.contains('open'), 'the drawer is open');
      click(win, byId(win, id));
    };

    // (The request names the chat's machine; the handler's fallback for
    // a page without a chat tab is unreachable: a page always has one.)
    clickInInfoDrawer('browser-btn');
    assert.ok(!info.classList.contains('open'), 'Browser closed it');
    assert.deepStrictEqual(plain(ofType(posted, 'browserOpen')), [
      {type: 'browserOpen', url: ''},
    ]);

    clickInInfoDrawer('terminal-btn');
    assert.ok(!info.classList.contains('open'), 'Terminal closed it');
    await settle();
    assert.strictEqual(xterm.terminals.length, 1);
    assert.strictEqual(ofType(posted, 'terminalOpen').length, 1);

    // A commit reports in the transcript, which the drawer would cover;
    // a pull request reports in a toast, which shows over the drawer.
    win._testApi.switchToTab('a1');
    clickInInfoDrawer('create-pr-btn');
    assert.ok(info.classList.contains('open'), 'Create PR leaves it open');
    assert.strictEqual(ofType(posted, 'createPrAction').length, 1);
    clickInInfoDrawer('autocommit-btn');
    assert.ok(!info.classList.contains('open'), 'Git commit closed it');
    assert.strictEqual(ofType(posted, 'autocommitAction').length, 1);

    // Settings leaves the Chats drawer.  Its button goes off screen
    // with the drawer, so closing the sheet hands the focus to the
    // drawer's toggle.
    click(win, byId(win, 'menu-btn'));
    assert.ok(sidebar.classList.contains('open'), 'the drawer is open');
    const asked = ofType(posted, 'getConfig').length;
    press(win, byId(win, 'settings-btn'));
    assert.ok(!sidebar.classList.contains('open'), 'Settings closed it');
    assert.ok(sheet.classList.contains('open'), 'the sheet is open');
    assert.strictEqual(ofType(posted, 'getConfig').length, asked + 1);
    press(win, byId(win, 'settings-panel-close'));
    assert.ok(!sheet.classList.contains('open'));
    assert.strictEqual(win.document.activeElement, byId(win, 'menu-btn'));
  });

  await test('a phone has no drawer toggle: the closed sheet focuses New chat', async () => {
    const {win} = makeStripPage({userAgent: PHONE_UA});
    assert.ok(win.document.body.classList.contains('remote-mobile'));
    // A phone shows the Chats by a swipe; the toggle stands in for it
    // here (the stylesheet hides it on a phone).
    click(win, byId(win, 'menu-btn'));
    press(win, byId(win, 'settings-btn'));
    assert.ok(!byId(win, 'sidebar').classList.contains('open'));
    press(win, byId(win, 'settings-panel-close'));
    assert.strictEqual(win.document.activeElement, byId(win, 'new-chat-btn'));
  });

  await test('the docked desktop panels stay open', async () => {
    const {win, posted, xterm} = makeStripPage({desktop: true});
    const sidebar = byId(win, 'sidebar');
    const info = byId(win, 'meta-panel');
    const settings = byId(win, 'settings-btn');
    assert.ok(win.document.body.classList.contains('remote-desktop'));
    const docked = info.classList.contains('open');
    click(win, byId(win, 'browser-btn'));
    assert.strictEqual(ofType(posted, 'browserOpen').length, 1);
    click(win, byId(win, 'terminal-btn'));
    await settle();
    assert.strictEqual(xterm.terminals.length, 1);
    win._testApi.switchToTab('a1');
    click(win, byId(win, 'autocommit-btn'));
    assert.strictEqual(ofType(posted, 'autocommitAction').length, 1);
    assert.strictEqual(info.classList.contains('open'), docked, 'unchanged');
    press(win, settings);
    assert.ok(byId(win, 'settings-panel').classList.contains('open'));
    assert.ok(sidebar.classList.contains('open'), 'still docked');
    // Closing the sheet hands the focus back to the button that opened it.
    press(win, byId(win, 'settings-panel-close'));
    assert.strictEqual(win.document.activeElement, settings);
  });

  await test('the font buttons resize all text from 10px to 20px', async () => {
    const {win, xterm, editors} = makeStripPage();
    const dec = fontButton(win, -1);
    const inc = fontButton(win, 1);
    const step = (btn, times) => {
      for (let i = 0; i < times; i++) click(win, btn);
    };
    assert.strictEqual(rootSize(win), '13px');
    assert.ok(!dec.disabled && !inc.disabled);
    await openFile(win, 'a.txt');
    assert.strictEqual(editors[0].opts.fontSize, undefined, "Monaco's default");

    // A terminal still loading xterm starts at the size picked meanwhile.
    click(win, byId(win, 'terminal-btn'));
    click(win, inc);
    await settle();
    const term = xterm.terminals[0];
    assert.strictEqual(term.options.fontSize, 14);
    assert.strictEqual(rootSize(win), '14px');
    assert.strictEqual(win.localStorage.getItem('kissRemoteFontSize'), '14');
    assert.strictEqual(editors[0].opts.fontSize, 15, 'the open editor follows');

    // A terminal on screen is refitted to its new cell size; a hidden
    // one is when it is shown again.
    const fits = xterm.fits.length;
    click(win, inc);
    assert.strictEqual(term.options.fontSize, 15);
    assert.strictEqual(xterm.fits.length, fits + 1);
    win._testApi.switchToTab('a1');
    click(win, dec);
    assert.strictEqual(term.options.fontSize, 14);
    assert.strictEqual(xterm.fits.length, fits + 1);

    step(inc, 6);
    assert.strictEqual(rootSize(win), '20px');
    assert.ok(inc.disabled && !dec.disabled, 'no step above 20px');
    step(dec, 10);
    assert.strictEqual(rootSize(win), '10px');
    assert.ok(dec.disabled && !inc.disabled, 'no step below 10px');
    assert.strictEqual(editors[0].opts.fontSize, 11);
    step(inc, 3);
    assert.strictEqual(editors[0].opts.fontSize, undefined, 'default again');

    // The size comes back on the next load, for what opens there too.
    step(inc, 3);
    const saved = win.localStorage.getItem('kissRemoteFontSize');
    assert.strictEqual(saved, '16');
    const again = makeStripPage({fontSize: saved});
    assert.strictEqual(rootSize(again.win), '16px');
    await openFile(again.win, 'b.txt');
    assert.strictEqual(again.editors[0].opts.fontSize, 17);
    click(again.win, byId(again.win, 'terminal-btn'));
    await settle();
    assert.strictEqual(again.xterm.terminals[0].options.fontSize, 16);
    // A terminal view made without a size has xterm's 13px.
    const view = again.win.TerminalTabView.create('plain', {send: () => {}});
    view.setVisible(true);
    await settle();
    assert.strictEqual(again.xterm.terminals[1].options.fontSize, 13);

    for (const bad of ['9', '21', '13.5', 'large']) {
      const page = makeStripPage({fontSize: bad});
      assert.strictEqual(rootSize(page.win), '13px', bad + ' is no size');
    }
  });

  await test('a Markdown preview is told the text size', async () => {
    const {win} = makeStripPage({fontSize: '14'});
    const inc = fontButton(win, 1);
    openFile(win, 'notes.md');
    const frame = win.document.querySelector('iframe.content-html-frame');
    // The document is built at the size of the moment: the browser's
    // 16px moved by as many px as the text is from its 13px.
    assert.ok(frame.srcdoc.includes('<style>html{font-size:106.25%}'));
    const told = [];
    frame.contentWindow.addEventListener('message', e => {
      if (e.data.kissFontSize) told.push(e.data.kissFontSize);
    });
    // A size picked while the document loads reaches it once it has
    // loaded, and so does every size picked later.
    click(win, inc);
    await new Promise(resolve => frame.addEventListener('load', resolve));
    await settle();
    assert.deepStrictEqual(told, ['112.5%', '112.5%'], 'the click, the load');
    click(win, fontButton(win, -1));
    await settle();
    assert.deepStrictEqual(told.slice(2), ['106.25%']);
  });

  report('sidebarFooterStrip');
}

main();

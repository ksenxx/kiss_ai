// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// A working-directory change broadcast by the daemon (`workDirChanged`,
// a pick made on ANOTHER surface or a settings save) must not close the
// folder picker this surface has open: a user browsing "Add Folder to
// Explorer" or typing a path in "Open Folder" keeps the dialog, and the
// "Recently opened" list of the add dialog shows the new directory.
// This client's own pick still closes the dialog (sidebarMenuFixes.test.js).

'use strict';

const assert = require('assert');
const {
  makePage: makeChatPage,
  stubRaf,
  fakeMatchMedia,
} = require('./simplify2_harness');
const {
  send,
  click,
  byId,
  showSection,
  ofType,
  makeRunner,
} = require('./ui_antipattern_harness');

const {test, report} = makeRunner();

const WD = '/ws/repo';

function makePage() {
  return makeChatPage({
    bodyClass: 'remote-chat',
    beforeScripts: win => {
      stubRaf(win);
      fakeMatchMedia(win, true);
    },
    extraScripts: ['treeContextMenu.js'],
    sourceUrl: 'workdir-picker-main.js',
  });
}

/** Open the Explorer on WD and answer its root listing. */
function openExplorer(win, posted) {
  send(win, {type: 'configData', config: {work_dir: WD}});
  showSection(win, 'meta-explorer');
  const list = ofType(posted, 'listDir');
  const req = list[list.length - 1];
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.path,
    entries: [],
  });
}

/** Answer the folder picker's latest listing with no sub-folders. */
function answerPickerListing(win, posted) {
  const list = ofType(posted, 'listDir').filter(m =>
    String(m.token).startsWith('picker:'),
  );
  const req = list[list.length - 1];
  send(win, {
    type: 'dirListing',
    token: req.token,
    path: req.path,
    root: req.path,
    entries: [],
  });
}

async function run() {
  await test("another surface's pick keeps the Add Folder dialog open and lists it", async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted);
    send(win, {
      type: 'configData',
      config: {
        work_dir: WD,
        recent_work_dirs: [
          {path: '/old', ts: 1},
          {path: '/elsewhere', ts: 2},
        ],
      },
    });
    click(win, byId(win, 'explorer-add-folder'));
    const picker = byId(win, 'folder-picker');
    assert.ok(!picker.hidden, 'the add dialog is open');
    answerPickerListing(win, posted);
    const recent = picker.querySelector('.folder-picker-recent');
    const offered = () =>
      Array.from(
        recent.querySelectorAll('.workdir-item'),
        el => el.dataset.path,
      ).sort();
    assert.deepStrictEqual(offered(), ['/elsewhere', '/old']);
    send(win, {type: 'workDirChanged', workDir: '/elsewhere'});
    assert.ok(
      !picker.hidden,
      "BUG: another surface's working-directory pick closed the dialog",
    );
    // The list is repainted: the new directory is the Explorer's root
    // now, so it is no longer offered.
    assert.ok(!recent.hidden, 'the Recently opened section is shown');
    assert.deepStrictEqual(offered(), ['/old']);
    // The daemon's directory was adopted all the same.
    const list = ofType(posted, 'listDir').filter(
      m => !String(m.token).startsWith('picker:'),
    );
    assert.strictEqual(list[list.length - 1].path, '/elsewhere');
    win.close();
  });

  await test("another surface's pick keeps the Open Folder dialog and its listing", async () => {
    const {win, posted} = makePage();
    openExplorer(win, posted);
    click(win, byId(win, 'explorer-pick-folder'));
    const picker = byId(win, 'folder-picker');
    assert.ok(!picker.hidden, 'the open-folder dialog is open');
    const before = ofType(posted, 'listDir').filter(m =>
      String(m.token).startsWith('picker:'),
    ).length;
    send(win, {type: 'workDirChanged', workDir: '/elsewhere'});
    assert.ok(!picker.hidden, 'BUG: the broadcast closed the dialog');
    // The listing the dialog was waiting for still lands in it.
    answerPickerListing(win, posted);
    assert.ok(
      picker.querySelector('.folder-picker-note'),
      'the pending listing was rendered into the open dialog',
    );
    assert.strictEqual(
      ofType(posted, 'listDir').filter(m =>
        String(m.token).startsWith('picker:'),
      ).length,
      before,
      'the broadcast did not re-list the dialog',
    );
    // This client's own pick closes it.
    click(win, picker.querySelector('.folder-picker-select'));
    assert.ok(picker.hidden, 'the own pick closes the dialog');
    win.close();
  });
}

run().then(
  () => report('audit1020_workdir_picker'),
  err => {
    console.error(err);
    process.exit(1);
  },
);

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Audit 2026-10-12 (mainjs-3): the host's `gitCommit` message (the
// editor-title Git Commit button) runs triggerManualGitCommit(), which
// calls closeSettingsPanel() unconditionally -- even while the settings
// sheet is closed.  closeSettingsPanel() flushes the form through
// saveSettingsIfPopulated(), and `configFormPopulated` is set by EVERY
// configData reply (the host re-pushes one every 2 s), not only by the
// one the sheet asked for.  So every Git Commit click posts a full
// `saveConfig` (API keys included) nobody asked for, and a stale form
// can overwrite a change another window saved since this window's last
// poll.
//
// Expected: a gitCommit with the sheet closed posts autocommitAction and
// nothing else config-related.

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
  const posted = [];
  let state;
  win.acquireVsCodeApi = function () {
    return {
      postMessage: msg => posted.push(msg),
      getState: () => state,
      setState: s => {
        state = s;
      },
    };
  };
  win.eval(fs.readFileSync(path.join(MEDIA, 'panelCopy.js'), 'utf8'));
  win.eval(fs.readFileSync(path.join(MEDIA, 'api.js'), 'utf8'));
  win.eval(
    fs.readFileSync(path.join(MEDIA, 'main.js'), 'utf8') +
      '\n//# sourceURL=audit1012-mainjs3-gitcommit-main.js',
  );
  return {win, posted};
}

function send(win, data) {
  win.dispatchEvent(new win.MessageEvent('message', {data}));
}

function main() {
  const {win, posted} = makeWebview();
  // The routine configData poll lands while the settings sheet is
  // closed (it was never opened in this session).
  send(win, {
    type: 'configData',
    config: {max_budget: 42, use_web_browser: true},
    apiKeys: {OPENAI_API_KEY: 'sk-test'},
  });
  assert.ok(
    !win.document.getElementById('settings-panel').classList.contains('open'),
    'precondition: the settings sheet is closed',
  );
  posted.length = 0;
  send(win, {type: 'gitCommit'});
  const commit = posted.find(m => m && m.type === 'autocommitAction');
  assert.ok(commit, 'the host gitCommit posts autocommitAction');
  const save = posted.find(m => m && m.type === 'saveConfig');
  assert.strictEqual(
    save,
    undefined,
    'a Git Commit with the settings sheet closed must not post saveConfig ' +
      '(got ' + JSON.stringify(save) + ')',
  );
  win.close();
  console.log(
    'audit1012_mainjs3_gitcommit_spurious_saveconfig.test.js: all passed.',
  );
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}

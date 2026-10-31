// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end (jsdom) tests for the editor language a file tab picks from
// the file's name (languageFromPath in media/main.js): a file with no
// extension that the map names (Dockerfile) is highlighted as such, and a
// name that is an Object.prototype key is plain text, not a function.

'use strict';

const assert = require('assert');
const h = require('./ui_antipattern_harness');

const {test, report} = h.makeRunner();

/** Open a file named *name* in a content tab; resolves to the language
 *  the editor was created with. */
async function editorLanguage(name) {
  const {win, created} = h.makePage();
  h.send(win, {
    type: 'fileContent',
    tabId: win._testApi.getActiveTabId(),
    path: '/ws/' + name,
    name,
    content: 'FROM python:3.12\n',
    version: 'v1',
  });
  for (let i = 0; i < 100 && created.length < 1; i++) await h.sleep(20);
  assert.strictEqual(created.length, 1, 'one editor was created');
  const language = created[0].language;
  win.close();
  return language;
}

async function main() {
  await test('a Dockerfile opens with Dockerfile highlighting', async () => {
    assert.strictEqual(await editorLanguage('Dockerfile'), 'dockerfile');
  });

  await test('an extension still picks the language', async () => {
    assert.strictEqual(await editorLanguage('api.Dockerfile'), 'dockerfile');
    assert.strictEqual(await editorLanguage('a.py'), 'python');
  });

  await test('a prototype-key name opens as plain text', async () => {
    assert.strictEqual(await editorLanguage('constructor'), 'plaintext');
    assert.strictEqual(await editorLanguage('__proto__'), 'plaintext');
    assert.strictEqual(await editorLanguage('x.constructor'), 'plaintext');
    assert.strictEqual(await editorLanguage('README'), 'plaintext');
  });

  report('contentLanguage_sweep1030');
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});

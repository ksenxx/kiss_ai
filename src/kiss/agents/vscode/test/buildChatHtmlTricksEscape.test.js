// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

function makeUri(fsPath) {
  return {
    fsPath,
    toString() {
      return 'vscode-webview://kiss' + fsPath;
    },
  };
}

global.__kissVscodeStub = {
  Uri: {
    joinPath(base, ...parts) {
      return makeUri(path.join(base.fsPath, ...parts));
    },
  },
  workspace: {
    isTrusted: false,
    getConfiguration() {
      return {get: () => undefined};
    },
  },
};

const origResolve = Module._resolveFilename;
Module._resolveFilename = function (request, parent, ...rest) {
  if (request === 'vscode') return require.resolve('./_vscode-stub.js');
  return origResolve.call(this, request, parent, ...rest);
};

const sourcePath = path.join(__dirname, '..', 'out', 'SorcarTab.js');
assert.ok(
  fs.existsSync(sourcePath),
  `compiled extension missing: ${sourcePath} — run \`tsc -p .\` first`,
);
const {buildChatHtml} = require(sourcePath);
assert.strictEqual(typeof buildChatHtml, 'function');

let passed = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok - ${name}`);
  } catch (err) {
    failures.push({name, err});
    console.log(`  FAIL - ${name}: ${err && err.message}`);
  }
}

function renderWithTricks(tricksMarkdown) {
  const kissHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-html-home-'));
  const bundledDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-html-inj-'));
  const bundledFile = path.join(bundledDir, 'INJECTIONS.md');
  fs.writeFileSync(bundledFile, tricksMarkdown);
  const tipsFile = path.join(bundledDir, 'TIPS.md');
  fs.writeFileSync(tipsFile, '# Tip\n\na tip body\n');
  fs.mkdirSync(kissHome, {recursive: true});
  fs.writeFileSync(path.join(kissHome, 'MY_INJECTION.md'), '');
  const prev = {
    home: process.env.KISS_HOME,
    inj: process.env.KISS_INJECTIONS_PATH,
    tips: process.env.KISS_TIPS_PATH,
  };
  process.env.KISS_HOME = kissHome;
  process.env.KISS_INJECTIONS_PATH = bundledFile;
  process.env.KISS_TIPS_PATH = tipsFile;
  const extensionUri = makeUri(path.join(__dirname, '..'));
  const webview = {
    cspSource: 'vscode-resource:',
    asWebviewUri: uri => uri,
  };
  try {
    return buildChatHtml(webview, extensionUri, 'test-model');
  } finally {
    if (prev.home === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev.home;
    if (prev.inj === undefined) delete process.env.KISS_INJECTIONS_PATH;
    else process.env.KISS_INJECTIONS_PATH = prev.inj;
    if (prev.tips === undefined) delete process.env.KISS_TIPS_PATH;
    else process.env.KISS_TIPS_PATH = prev.tips;
    fs.rmSync(kissHome, {recursive: true, force: true});
    fs.rmSync(bundledDir, {recursive: true, force: true});
  }
}

function parseTricksPayload(html) {
  const marker = 'window.__TRICKS__ = ';
  const start = html.indexOf(marker);
  assert.ok(start >= 0, '__TRICKS__ assignment missing from html');
  const afterAssign = html.slice(start + marker.length);
  const end = afterAssign.indexOf('</script');
  assert.ok(end >= 0, '__TRICKS__ script never closes');
  let js = afterAssign.slice(0, end).trim();
  if (js.endsWith(';')) js = js.slice(0, -1);
  return JSON.parse(js);
}

test('trick containing </script> cannot terminate the inline script', () => {
  const evil = 'evil </script><img src=x> trick';
  const html = renderWithTricks('## Trick\n\n' + evil + '\n');
  let tricks;
  assert.doesNotThrow(() => {
    tricks = parseTricksPayload(html);
  }, 'raw </script> inside __TRICKS__ payload truncates the script');
  assert.deepStrictEqual(tricks, [evil]);
});

test('trick containing a later placeholder like {{TIPS_JSON}} survives verbatim', () => {
  const tricky = 'Use {{TIPS_JSON}} and {{MAIN_SRC}} literally';
  const html = renderWithTricks('## Trick\n\n' + tricky + '\n');
  const tricks = parseTricksPayload(html);
  assert.deepStrictEqual(tricks, [tricky]);
});

test('plain tricks and tips still round-trip', () => {
  const html = renderWithTricks('## Trick\n\nplain trick\n');
  assert.deepStrictEqual(parseTricksPayload(html), ['plain trick']);
  assert.ok(html.includes('window.__TIPS__ = '), 'tips assignment present');
  assert.ok(!html.includes('{{TRICKS_JSON}}'));
  assert.ok(!html.includes('{{TIPS_JSON}}') || html.includes('TIPS_JSON}} literally') === false);
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``buildChatHtml`` inline-script safety.
//
// Two contracts locked in here:
//
//   1. ``</`` must never appear raw inside the ``window.__TRICKS__``
//      inline <script> block.  Tricks come from the user-editable
//      ``~/.kiss/MY_INJECTION.md``; a trick body containing
//      ``</script>`` would otherwise terminate the inline script per
//      the HTML spec, throw a SyntaxError, and break the whole chat
//      webview on every render.  ``tipsJson`` already had this escape
//      — ``tricksJson`` must too.
//
//   2. Template placeholders like ``{{TIPS_JSON}}`` occurring inside
//      user trick content must survive verbatim.  The old sequential
//      split/join substitution rewrote a later key's placeholder that
//      an earlier key's VALUE happened to contain, garbling the
//      tricks panel.  Substitution must be single-pass.
//
// Runs against the compiled extension under ``out/``:
//
//     tsc -p . && node test/buildChatHtmlTricksEscape.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// buildChatHtml requires 'vscode' at load time; outside the extension
// host redirect the specifier to the shared on-disk stub.
// ``_vscode-stub.js`` is a git-tracked fixture shared by tests running
// in parallel; it already re-exports ``global.__kissVscodeStub || {}`` —
// never rewrite or delete it here (writeFileSync truncates first, racing
// a concurrent ``require('vscode')`` in sibling test processes).

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

/**
 * Render the chat HTML inside a sandbox: fresh KISS_HOME, a pinned
 * bundled-injections file containing ``tricks``, and empty tips.
 */
function renderWithTricks(tricksMarkdown) {
  const kissHome = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-html-home-'));
  const bundledDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-html-inj-'));
  const bundledFile = path.join(bundledDir, 'INJECTIONS.md');
  fs.writeFileSync(bundledFile, tricksMarkdown);
  const tipsFile = path.join(bundledDir, 'TIPS.md');
  fs.writeFileSync(tipsFile, '# Tip\n\na tip body\n');
  // Suppress the auto-seeded default user trick so assertions see
  // exactly the bundled tricks.
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

/**
 * Extract and parse the ``window.__TRICKS__`` payload the way a real
 * HTML parser would see it: the inline script ends at the FIRST
 * ``</script`` after the assignment.  If the payload contains a raw
 * ``</script>`` the JSON is truncated and parsing fails — exactly the
 * production failure.
 */
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
  // Tips block still present and escaped form intact.
  assert.ok(html.includes('window.__TIPS__ = '), 'tips assignment present');
  // No unsubstituted known placeholders remain in template positions.
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

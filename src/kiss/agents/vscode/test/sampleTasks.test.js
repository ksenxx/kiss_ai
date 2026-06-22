// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// End-to-end test for ``readSampleTasks`` — the helper in
// ``SorcarSidebarView`` that parses ``SAMPLE_TASKS.md`` into the
// ``[{text}]`` array consumed by ``renderWelcomeSuggestions`` in
// ``media/main.js``.
//
// Regression locked in:
//
//   SAMPLE_TASKS used to be a JSON array.  Editing it required quoting
//   every line and escaping every embedded quote.  We switched to a
//   Markdown file with ``## Task`` sections (mirroring
//   ``src/kiss/INJECTIONS.md``'s ``## Trick`` sections).  The loader
//   must:
//     * read ``~/.kiss/SAMPLE_TASKS.md`` (seeded from
//       ``extensionRoot/SAMPLE_TASKS.md`` by ``ensureUserAsset``),
//     * return one ``{text}`` per ``## Task`` section, with body
//       whitespace trimmed,
//     * skip non-``Task`` sections,
//     * tolerate a missing or unparseable file by returning ``[]``,
//     * preserve the order of the sections in the file (so the
//       welcome screen chip ordering is editor-controlled).
//
// Runs against the compiled extension under ``out/`` — same pattern
// as ``bughunt_reopen_running_tab.test.js`` — exercised directly with
// ``node``:
//
//     node test/sampleTasks.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const Module = require('module');

// SorcarTab.js requires 'vscode' at load time.  Outside the extension
// host the module is unavailable, so we redirect that specifier to a
// tiny generated stub on disk — same pattern as the other bughunt
// tests in this directory.  ``readSampleTasks`` itself never touches
// any vscode API, so an empty object suffices.
const stubPath = path.join(__dirname, '_vscode-stub.js');
fs.writeFileSync(
  stubPath,
  `'use strict';\nmodule.exports = global.__kissVscodeStub || {};\n`,
);
global.__kissVscodeStub = {};
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
delete require.cache[require.resolve(sourcePath)];
const {readSampleTasks} = require(sourcePath);
assert.strictEqual(
  typeof readSampleTasks,
  'function',
  'readSampleTasks must be exported from the compiled SorcarTab',
);

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

function mkExt() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'kiss-sample-tasks-'));
}

/**
 * Run ``fn`` with ``KISS_HOME`` redirected to a fresh temp directory
 * so the lazy ``~/.kiss/`` seed in ``ensureUserAsset`` operates on
 * disposable state.  ``readSampleTasks`` resolves the user copy at
 * ``KISS_HOME/SAMPLE_TASKS.md`` and seeds it from the package copy at
 * ``extensionRoot/SAMPLE_TASKS.md`` only when the user copy is
 * missing — so each test pre-populates the extension root and
 * asserts on what the user-copy parser returns.
 */
function withTempKissHome(fn) {
  const ext = mkExt();
  const kissHome = mkExt();
  const prev = process.env.KISS_HOME;
  process.env.KISS_HOME = kissHome;
  try {
    fn(ext, kissHome);
  } finally {
    if (prev === undefined) delete process.env.KISS_HOME;
    else process.env.KISS_HOME = prev;
    fs.rmSync(ext, {recursive: true, force: true});
    fs.rmSync(kissHome, {recursive: true, force: true});
  }
}

test('returns [] when SAMPLE_TASKS.md is missing in both locations', () => {
  withTempKissHome(ext => {
    assert.deepStrictEqual(readSampleTasks(ext), []);
  });
});

test('parses one ## Task section into [{text}]', () => {
  withTempKissHome(ext => {
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\nDo the thing\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Do the thing'}]);
  });
});

test('seeds ~/.kiss/SAMPLE_TASKS.md from the package copy on first read', () => {
  withTempKissHome((ext, kissHome) => {
    const pkg = path.join(ext, 'SAMPLE_TASKS.md');
    fs.writeFileSync(pkg, '## Task\n\nSeeded body\n');
    const userCopy = path.join(kissHome, 'SAMPLE_TASKS.md');
    assert.ok(!fs.existsSync(userCopy), 'user copy must start absent');
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'Seeded body'}]);
    assert.ok(
      fs.existsSync(userCopy),
      'reader must seed ~/.kiss/SAMPLE_TASKS.md from the package copy',
    );
    assert.strictEqual(
      fs.readFileSync(userCopy, 'utf-8'),
      fs.readFileSync(pkg, 'utf-8'),
    );
  });
});

test('reads user-edited ~/.kiss/SAMPLE_TASKS.md in preference to the package copy', () => {
  withTempKissHome((ext, kissHome) => {
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\nPackage default\n',
    );
    fs.writeFileSync(
      path.join(kissHome, 'SAMPLE_TASKS.md'),
      '## Task\n\nUser override\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'User override'}]);
  });
});

test('preserves user edits even when the package copy is newer', () => {
  // Locks in the no-clobber contract: ``ensureUserAsset`` never
  // silently overwrites an existing user copy from the package copy,
  // even when the package copy has been freshly bumped (e.g. by a
  // ``git pull`` that updated ``SAMPLE_TASKS.md``).  User edits
  // survive every read; the user explicitly removes
  // ``~/.kiss/SAMPLE_TASKS.md`` to regenerate from defaults.
  withTempKissHome((ext, kissHome) => {
    fs.writeFileSync(
      path.join(kissHome, 'SAMPLE_TASKS.md'),
      '## Task\n\nUser-curated chip\n',
    );
    const past = Date.now() / 1000 - 7200;
    fs.utimesSync(path.join(kissHome, 'SAMPLE_TASKS.md'), past, past);
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\nFresh package copy\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'User-curated chip'},
    ]);
  });
});

test('parses multiple ## Task sections in source order', () => {
  withTempKissHome(ext => {
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\nFirst task\n\n## Task\n\nSecond task\n\n## Task\n\nThird\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'First task'},
      {text: 'Second task'},
      {text: 'Third'},
    ]);
  });
});

test('preserves multi-line bodies and inline punctuation', () => {
  withTempKissHome(ext => {
    const body =
      'Line one with **bold**, "quotes", and a <<placeholder>>.\n' +
      'Line two continues the same task.';
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      `## Task\n\n${body}\n`,
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: body}]);
  });
});

test('skips sections whose heading is not Task', () => {
  withTempKissHome(ext => {
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Intro\n\nignored\n\n## Task\n\nkept\n\n## Notes\n\nalso ignored\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'kept'}]);
  });
});

test('skips empty-bodied ## Task sections', () => {
  withTempKissHome(ext => {
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\n\n## Task\n\nreal body\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [{text: 'real body'}]);
  });
});

test('unescapes mdformat backslash escapes (\\<< -> <<)', () => {
  withTempKissHome(ext => {
    // ``mdformat`` rewrites ``<<x>>`` to ``\<<x>>`` on save.  The chip
    // renders ``s.text`` literally (only HTML-escaped) so the parser
    // MUST strip CommonMark backslash escapes — otherwise the user
    // sees a literal backslash on the welcome screen.
    fs.writeFileSync(
      path.join(ext, 'SAMPLE_TASKS.md'),
      '## Task\n\nRun on \\<<your dataset>> with **bold** \\*literal\\*\n',
    );
    assert.deepStrictEqual(readSampleTasks(ext), [
      {text: 'Run on <<your dataset>> with **bold** *literal*'},
    ]);
  });
});

test('shipped SAMPLE_TASKS.md tasks never contain a leading backslash before <<', () => {
  withTempKissHome(() => {
    const ext = path.join(__dirname, '..');
    const tasks = readSampleTasks(ext);
    for (const t of tasks) {
      assert.ok(
        !/\\</.test(t.text),
        `task should not retain mdformat escapes: ${JSON.stringify(t.text)}`,
      );
    }
  });
});

test('parses the shipped SAMPLE_TASKS.md (sanity)', () => {
  withTempKissHome(() => {
    // The real file lives next to the extension's package.json.  This
    // also guards against accidentally checking in a malformed file.
    const ext = path.join(__dirname, '..');
    const shipped = path.join(ext, 'SAMPLE_TASKS.md');
    assert.ok(
      fs.existsSync(shipped),
      `shipped SAMPLE_TASKS.md missing at ${shipped}`,
    );
    const tasks = readSampleTasks(ext);
    assert.ok(Array.isArray(tasks), 'must return an array');
    assert.ok(tasks.length > 0, 'shipped file must contain at least one task');
    for (const t of tasks) {
      assert.strictEqual(typeof t.text, 'string');
      assert.ok(t.text.length > 0, 'each task body must be non-empty');
      // Each chip is displayed on the welcome screen — make sure the
      // text is trimmed so the chip label doesn't carry stray newlines.
      assert.strictEqual(t.text, t.text.trim());
    }
  });
});

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length > 0) {
  for (const f of failures) {
    console.error(`\n${f.name}:\n`, f.err);
  }
  process.exit(1);
}

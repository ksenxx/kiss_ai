// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
//
// Integration test for the reduced left whitespace on each
// History sidebar task row.
//
// Requirement driven by this test:
//
//   The left whitespace in front of each task panel in the
//   History sidebar must be reduced by HALF compared to the
//   original layout.  The original layout used:
//
//     .running-item                       { padding-left: 26px; }
//     .running-item > .sidebar-item-*     { left: 10px; }
//
//   Halving both values keeps the visual proportions intact —
//   the absolutely-positioned status indicator (running pulse /
//   failed red dot / completed green dot) still sits inside the
//   reserved left padding column, but the column itself is now
//   half as wide:
//
//     .running-item                       { padding-left: 13px; }
//     .running-item > .sidebar-item-*     { left: 5px;  }
//
//   The 8px-wide status dot at ``left: 5px`` occupies pixels
//   5..13, which exactly meets the new 13px content edge, so
//   text content never overlaps the indicator.
//
// jsdom never loads the external ``main.css`` stylesheet
// referenced via ``{{STYLE_HREF}}``, so we read the CSS file
// directly and assert the property values with regex — the same
// pattern used by ``historyTaskIds.test.js``.
//
// Run directly with ``node``:
//
//     node src/kiss/agents/vscode/test/historyTaskLeftPadding.test.js

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');

// Pick the FIRST declaration of ``prop`` inside the body of the
// given selector rule.  Returns the trimmed value string, or
// ``null`` when neither the rule nor the property is present.
function pickDeclaration(css, selector, prop) {
  // Match a top-level rule whose selector list contains the
  // exact ``selector`` token (allowing leading/trailing
  // whitespace or a sibling selector via comma).  Avoids
  // matching ``selector`` as a prefix of an unrelated class.
  const ruleRe = new RegExp(
    '(?:^|[\\s,}])' +
      selector.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&') +
      '\\s*(?:,\\s*[^{]+)?\\{([^}]*)\\}',
    'g',
  );
  let m;
  while ((m = ruleRe.exec(css)) !== null) {
    const body = m[1];
    const declRe = new RegExp(
      '(?:^|;)\\s*' +
        prop.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&') +
        '\\s*:\\s*([^;]+)',
    );
    const dm = declRe.exec(body);
    if (dm) {
      return dm[1].trim();
    }
  }
  return null;
}

function testRunningItemPaddingLeftHalved() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const value = pickDeclaration(css, '.running-item', 'padding-left');
  assert.ok(
    value,
    'main.css must declare padding-left on .running-item ' +
      '(the left whitespace column reserved for the status dot)',
  );
  assert.strictEqual(
    value,
    '13px',
    `expected .running-item padding-left to be halved from 26px to 13px, got "${value}"`,
  );
  console.log('  ok - .running-item padding-left is 13px (halved from 26px)');
}

function testStatusIndicatorLeftHalved() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  // The CSS file declares the absolutely-positioned status dots
  // via a compound selector list:
  //
  //   .running-item > .sidebar-item-failed,
  //   .running-item > .sidebar-item-running,
  //   .running-item > .sidebar-item-completed { left: 10px; ... }
  //
  // Grab the body of the rule whose selector list ends with
  // ``.sidebar-item-completed`` (the last token in the list) and
  // assert ``left`` is now 5px so the dot stays centred inside
  // the new 13px padding column.
  const ruleRe =
    /\.running-item\s*>\s*\.sidebar-item-failed\s*,\s*\.running-item\s*>\s*\.sidebar-item-running\s*,\s*\.running-item\s*>\s*\.sidebar-item-completed\s*\{([^}]*)\}/;
  const m = ruleRe.exec(css);
  assert.ok(
    m,
    'main.css must define the compound rule positioning the ' +
      'three .sidebar-item-* status dots inside .running-item',
  );
  const body = m[1];
  const leftRe = /(?:^|;)\s*left\s*:\s*([^;]+)/;
  const lm = leftRe.exec(body);
  assert.ok(
    lm,
    'the .sidebar-item-* status-dot rule must declare a left offset',
  );
  assert.strictEqual(
    lm[1].trim(),
    '5px',
    `expected status dot left offset to be halved from 10px to 5px, got "${lm[1].trim()}"`,
  );
  console.log('  ok - status dot left offset is 5px (halved from 10px)');
}

function testDotFitsInsidePadding() {
  // Sanity check tying both halved values together: the 8px-wide
  // status dot at ``left: 5px`` must end at or before the new
  // 13px padding edge so the dot does not overlap text content.
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
  const paddingLeft = pickDeclaration(css, '.running-item', 'padding-left');
  const dotWidth = pickDeclaration(css, '.sidebar-item-completed', 'width');
  const dotLeftRe =
    /\.running-item\s*>\s*\.sidebar-item-failed\s*,\s*\.running-item\s*>\s*\.sidebar-item-running\s*,\s*\.running-item\s*>\s*\.sidebar-item-completed\s*\{([^}]*)\}/;
  const dotLeftMatch = dotLeftRe.exec(css);
  assert.ok(dotLeftMatch, 'compound status-dot rule must exist');
  const leftRe = /(?:^|;)\s*left\s*:\s*([^;]+)/;
  const lm = leftRe.exec(dotLeftMatch[1]);
  assert.ok(lm, 'compound status-dot rule must set "left"');

  const padPx = Number.parseInt(paddingLeft, 10);
  const widthPx = Number.parseInt(dotWidth, 10);
  const leftPx = Number.parseInt(lm[1], 10);

  assert.ok(
    Number.isFinite(padPx) && Number.isFinite(widthPx) && Number.isFinite(leftPx),
    'padding-left / dot width / dot left must all be integer px values',
  );
  assert.ok(
    leftPx + widthPx <= padPx,
    `status dot (left=${leftPx}px + width=${widthPx}px = ${
      leftPx + widthPx
    }px) must fit inside the .running-item padding column (${padPx}px) ` +
      'so it never overlaps text content',
  );
  console.log(
    `  ok - dot (${leftPx}px + ${widthPx}px) fits inside ${padPx}px padding column`,
  );
}

function main() {
  testRunningItemPaddingLeftHalved();
  testStatusIndicatorLeftHalved();
  testDotFitsInsidePadding();
  console.log('historyTaskLeftPadding.test.js: all assertions passed.');
}

main();

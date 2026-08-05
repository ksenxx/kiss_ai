// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');

const MEDIA = path.join(__dirname, '..', 'media');

function pickDeclaration(css, selector, prop) {
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
    '19px',
    `expected .running-item padding-left to be 19px (dot column of 13px + 6px gap to text), got "${value}"`,
  );
  console.log('  ok - .running-item padding-left is 19px (13px dot column + 6px gap)');
}

function testStatusIndicatorLeftHalved() {
  const css = fs.readFileSync(path.join(MEDIA, 'main.css'), 'utf8');
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

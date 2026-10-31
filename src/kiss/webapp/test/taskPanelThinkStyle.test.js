// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The task panel that opens every task of the chat thread paints the
// user bubble's accent wash under a 1px accent hairline.  Everything
// about that panel but the hairline is pinned through computed style
// in transcriptPanelAlignment.test.js; the `border` shorthand with a
// var() colour is the one declaration jsdom's cssstyle cannot read
// back (it reports `medium none`), so this test reads it from the
// stylesheet text.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {lastCssRule} = require('./designTokens');

const CSS = fs.readFileSync(
  path.join(__dirname, '..', 'media', 'main.css'),
  'utf8',
);

assert.match(
  lastCssRule(CSS, '.ev.task-panel'),
  /(?:^|[;\s])border:\s*1px solid var\(--accent-line\)\s*(?:;|$)/,
  'BUG: .ev.task-panel border must be the accent hairline',
);
console.log('  ok - .ev.task-panel draws the accent hairline');

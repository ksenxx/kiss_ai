// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// The task panel that opens every task of the chat thread is the user
// bubble: a neutral wash with no outline.  Everything about that panel
// but the missing outline is pinned through computed style in
// transcriptPanelAlignment.test.js; a `border` shorthand with a var()
// colour is the one declaration jsdom's cssstyle cannot read back (it
// reports `medium none`, as for no border at all), so this test reads
// the stylesheet text.

'use strict';

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const {lastCssRule} = require('./designTokens');

const CSS = fs.readFileSync(
  path.join(__dirname, '..', 'media', 'main.css'),
  'utf8',
);

assert.doesNotMatch(
  lastCssRule(CSS, '.ev.task-panel'),
  /(?:^|[;\s])(?:border|outline)(?:-(?:top|right|bottom|left|width|style|color))?\s*:/,
  'BUG: .ev.task-panel must draw no outline round the user bubble',
);
console.log('  ok - .ev.task-panel draws no outline');

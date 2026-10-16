// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

// Stylelint rule `kiss/design-tokens`: raw design values may only be
// written as custom properties inside a token block (`:root` in
// main.css, `body.remote-chat` in remote-codex.css).  Everywhere else a
// declaration must go through a token, so the rule rejects:
//
//   - hex colours (`#1e1e1e`, including `var(--x, #fff)` fallbacks),
//   - px border radii (`border-radius: 6px`; unitless 0 is allowed),
//   - numeric z-index values (`z-index: 10`, `calc(var(--z-a) + 1)`),
//   - font sizes with a unit (`font-size: 12px`, `font: 12px/1.4 sans-serif`,
//     `1.15em`) instead of the `--fs-*` scale,
//   - redefinitions of the tokens themselves in component rules
//     (`.a { --radius-md: 6px }`).
//
// Secondary option `tokenSelectors` lists the selectors whose custom
// properties form the token block (default `[":root"]`).

import stylelint from 'stylelint';
import valueParser from 'postcss-value-parser';

const {createPlugin, utils} = stylelint;

const ruleName = 'kiss/design-tokens';

const messages = utils.ruleMessages(ruleName, {
  hex: value => `Hex colour "${value}" outside the token block; use a colour token`,
  radius: value => `Raw radius "${value}" outside the token block; use a --radius-* token`,
  zIndex: value => `Numeric z-index "${value}" outside the token block; use a --z-* token`,
  fontSize: value => `Raw font size "${value}" outside the token block; use a --fs-* token`,
});

const HEX = /^#[\da-f]{3,8}$/i;
const BORDER_RADIUS = /^border(-(top|bottom)-(left|right)|-(start|end)-(start|end))?-radius$/;
// Units a font size can carry; `font: oblique 10deg ...` is an angle.
const SIZE_UNIT = /^(px|em|rem|%|pt|pc|ch|ex|vw|vh|vmin|vmax)$/i;

function isTokenDeclaration(decl, tokenSelectors) {
  const parent = decl.parent;
  return (
    decl.prop.startsWith('--') &&
    parent.type === 'rule' &&
    tokenSelectors.includes(parent.selector.trim())
  );
}

// Which token family a property belongs to: the longhand/shorthand
// properties and, outside the token block, a redefinition of the token
// itself (`.a { --radius-md: 6px }` is as raw as `border-radius: 6px`).
function family(prop) {
  if (BORDER_RADIUS.test(prop) || prop.startsWith('--radius-')) return 'radius';
  if (prop === 'z-index' || prop.startsWith('--z-')) return 'zIndex';
  if (prop === 'font-size' || prop === 'font' || prop.startsWith('--fs-')) return 'fontSize';
  return null;
}

// Returns the message key a word node violates, or null.  `kind` is the
// property's token family; `lineHeight` is true after the `/` of the
// `font` shorthand, where a unit is a line height, not a font size.
function violation(kind, word, lineHeight) {
  if (HEX.test(word)) return 'hex';
  const num = valueParser.unit(word);
  if (!num || !kind) return null;
  if (kind === 'zIndex') return 'zIndex';
  if (Number(num.number) === 0) return null;
  if (kind === 'radius') return num.unit.toLowerCase() === 'px' ? 'radius' : null;
  return SIZE_UNIT.test(num.unit) && !lineHeight ? 'fontSize' : null;
}

function checkDeclaration(decl, result) {
  const kind = family(decl.prop.toLowerCase());
  let lineHeight = false;
  valueParser(decl.value).walk(node => {
    if (node.type === 'function' && node.value.toLowerCase() === 'url') return false;
    if (node.type === 'div' && node.value === '/') lineHeight = true;
    if (node.type !== 'word') return undefined;
    const key = violation(kind, node.value, lineHeight);
    if (key) {
      utils.report({
        result,
        ruleName,
        message: messages[key](node.value),
        node: decl,
        word: node.value,
      });
    }
    return undefined;
  });
}

/**
 * Stylelint rule function for `kiss/design-tokens`.
 *
 * @param {boolean} primary `true` enables the rule.
 * @param {{tokenSelectors?: string[]}} [secondary] Selectors whose custom
 *   properties are the token block and may hold raw values.
 * @returns {Function} PostCSS walker that reports each raw value.
 */
function rule(primary, secondary) {
  return (root, result) => {
    const valid = utils.validateOptions(
      result,
      ruleName,
      {actual: primary, possible: [true]},
      {
        actual: secondary,
        possible: {tokenSelectors: [value => typeof value === 'string']},
        optional: true,
      },
    );
    if (!valid) return;
    const tokenSelectors = (secondary && secondary.tokenSelectors) || [':root'];
    root.walkDecls(decl => {
      if (!isTokenDeclaration(decl, tokenSelectors)) checkDeclaration(decl, result);
    });
  };
}

rule.ruleName = ruleName;
rule.messages = messages;

export default createPlugin(ruleName, rule);

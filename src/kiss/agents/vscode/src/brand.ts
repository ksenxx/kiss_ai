// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

/**
 * Product branding for the extension host.
 *
 * Mirrors `kiss.core.brand` on the Python side: the strings come from
 * `media/brand.json`, the one file a custom distribution replaces to
 * re-brand the product.  Read synchronously at module load because the
 * consumers are module-level constants (panel titles, notification
 * prefixes); the stock names are the fallback when the file is missing
 * or malformed, so the extension never fails to activate over branding.
 */

import * as fs from 'fs';
import * as path from 'path';

export interface Brand {
  productName: string;
  shortName: string;
  tagline: string;
}

const DEFAULT_BRAND: Brand = {
  productName: 'KISS Sorcar',
  shortName: 'KISS',
  tagline: 'Your AI assistant. Ask me anything!',
};

/** `media/brand.json`, resolved from `out/brand.js` in both dev and VSIX layouts. */
export const BRAND_FILE = path.join(__dirname, '..', 'media', 'brand.json');

/**
 * Read the brand strings from `file`, falling back key-by-key to the
 * stock KISS Sorcar names for anything missing, empty or not a string.
 */
export function loadBrand(file: string = BRAND_FILE): Brand {
  let raw: unknown;
  try {
    raw = JSON.parse(fs.readFileSync(file, 'utf-8'));
  } catch {
    return {...DEFAULT_BRAND};
  }
  const obj = (raw && typeof raw === 'object' ? raw : {}) as Record<
    string,
    unknown
  >;
  return {
    productName: pickString(obj, 'product_name', DEFAULT_BRAND.productName),
    shortName: pickString(obj, 'short_name', DEFAULT_BRAND.shortName),
    tagline: pickString(obj, 'tagline', DEFAULT_BRAND.tagline),
  };
}

function pickString(
  obj: Record<string, unknown>,
  key: string,
  fallback: string,
): string {
  const v = obj[key];
  return typeof v === 'string' && v ? v : fallback;
}

export const BRAND: Brand = loadBrand();
export const PRODUCT_NAME = BRAND.productName;
export const SHORT_NAME = BRAND.shortName;

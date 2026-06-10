// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here
/**
 * Shared utilities for Sorcar webview HTML rendering.
 * Used by SorcarSidebarView.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {findKissProject} from './kissPaths';

/** Read the KISS project version from ``_version.py`` on disk. */
export function getVersion(): string {
  try {
    const kissRoot = findKissProject();
    if (kissRoot) {
      const versionFile = path.join(kissRoot, 'src', 'kiss', '_version.py');
      const content = fs.readFileSync(versionFile, 'utf-8');
      const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
      if (match) return match[1];
    }
  } catch {
    /* ignore */
  }
  return '';
}

/**
 * Read ``src/kiss/INJECTIONS.md`` and return one entry per ``## Trick``
 * section.  Returns an empty list if the file is missing — the Tricks
 * button still renders, just with an empty list.
 */
export function getTricks(): string[] {
  try {
    const kissRoot = findKissProject();
    if (!kissRoot) return [];
    const tricksFile = path.join(kissRoot, 'src', 'kiss', 'INJECTIONS.md');
    const text = fs.readFileSync(tricksFile, 'utf-8');
    const tricks: string[] = [];
    const sections = text.split(/^##\s+/m);
    for (let i = 1; i < sections.length; i++) {
      const section = sections[i];
      const newline = section.indexOf('\n');
      if (newline < 0) continue;
      const title = section.slice(0, newline).trim();
      if (title !== 'Trick') continue;
      const body = section.slice(newline + 1).trim();
      if (body) tricks.push(body);
    }
    return tricks;
  } catch {
    return [];
  }
}

/**
 * Generate a cryptographically random nonce string for Content Security
 * Policy.  Uses Node's CSPRNG (``crypto.randomBytes``) — never
 * ``Math.random``, which is predictable.
 */
export function getNonce(): string {
  // 24 random bytes -> 32 base64url chars; restrict to [A-Za-z0-9] for
  // CSP-safe nonce values.
  return crypto
    .randomBytes(24)
    .toString('base64')
    .replace(/[^A-Za-z0-9]/g, '')
    .slice(0, 32);
}

/**
 * Build the full chat HTML for a Sorcar webview.
 *
 * The HTML body is loaded from ``media/chat.html`` — the same file the
 * remote web server uses — so the extension and the remote webapp can
 * never drift in markup or script ordering.  Only mode-specific values
 * (CSP, nonces, webview URIs, input placeholder with platform modifier
 * keys) are substituted in here.
 */
export function buildChatHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  selectedModel: string,
): string {
  const nonce = getNonce();
  const version = getVersion();
  const tricksJson = JSON.stringify(getTricks());
  const mod = process.platform === 'darwin' ? '⌘' : 'Ctrl+';

  const tplPath = vscode.Uri.joinPath(
    extensionUri,
    'media',
    'chat.html',
  ).fsPath;
  const tpl = fs.readFileSync(tplPath, 'utf-8');

  const u = (name: string): string =>
    webview
      .asWebviewUri(vscode.Uri.joinPath(extensionUri, 'media', name))
      .toString();

  /* eslint-disable quotes */
  const csp =
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none';` +
    ` style-src ${webview.cspSource} 'unsafe-inline';` +
    ` script-src 'nonce-${nonce}';` +
    ` img-src ${webview.cspSource} data: https:;` +
    ` font-src ${webview.cspSource};` +
    ` form-action 'none'; frame-src 'none'; object-src 'none'; base-uri 'none';">`;

  const placeholder =
    `Ask anything... (@ for files,` +
    ` ${mod}D toggle between editor and chat,` +
    ` ${mod}T new chat,` +
    ` ${mod}E run selected text as task,` +
    ` ${mod}L copy text to chat)`;

  const subs: Record<string, string> = {
    VIEWPORT: 'width=device-width, initial-scale=1.0',
    CSP_META: csp,
    STYLE_HREF: u('main.css'),
    HLJS_CSS_HREF: u('highlight-github-dark.min.css'),
    HEAD_STYLE: '',
    BODY_CLASS_ATTR: '',
    INPUT_PLACEHOLDER: placeholder,
    ENTERKEYHINT: '',
    MODEL_NAME: selectedModel,
    VERSION_SUFFIX: version ? ' ' + version : '',
    AUTH_MODAL: '',
    NONCE_ATTR: ` nonce="${nonce}"`,
    HLJS_SRC: u('highlight.min.js'),
    MARKED_SRC: u('marked.min.js'),
    PANEL_COPY_SRC: u('panelCopy.js'),
    MAIN_SRC: u('main.js'),
    DEMO_SRC: u('demo.js'),
    SHIM_SCRIPT: '',
    TRICKS_JSON: tricksJson,
  };

  let html = tpl;
  for (const [key, value] of Object.entries(subs)) {
    html = html.split(`{{${key}}}`).join(value);
  }
  return html;
}

// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';
import {findKissProject} from './kissPaths';
import {ensureUserAssetFromDefault, kissHomeDir} from './userAssets';

export const MY_INJECTION_DEFAULT_BODY =
  'Write end-to-end 100% coverage tests for the feature first.' +
  '  Then implement the feature.';

export const DEFAULT_MY_INJECTION =
  '## Trick\n\n' + MY_INJECTION_DEFAULT_BODY + '\n';

export function getVersion(): string {
  const kissRoot = findKissProject();
  if (!kissRoot) return '';
  try {
    const content = fs.readFileSync(
      path.join(kissRoot, 'src', 'kiss', 'core', '_version.py'),
      'utf-8',
    );
    const match = content.match(/__version__\s*=\s*["']([^"']+)["']/);
    if (match) return match[1];
  } catch {}
  return '';
}

function unescapeMarkdown(s: string): string {
  return s.replace(/\\([\\`*_{}[\]()#+\-.!<>|~"'$%&,/:;=?@^])/g, '$1');
}

function readMarkdownSections(markdownFile: string, heading: string): string[] {
  let text: string;
  try {
    text = fs.readFileSync(markdownFile, 'utf-8');
  } catch {
    return [];
  }
  const items: string[] = [];
  const sections = text.split(/^##\s+/m);
  for (let i = 1; i < sections.length; i++) {
    const section = sections[i];
    const newline = section.indexOf('\n');
    if (newline < 0) continue;
    const title = section.slice(0, newline).trim();
    if (title !== heading) continue;
    const body = unescapeMarkdown(section.slice(newline + 1).trim());
    if (body) items.push(body);
  }
  return items;
}

export function getTricks(): string[] {
  const items: string[] = [];

  const myInjectionPath = ensureUserAssetFromDefault(
    'MY_INJECTION.md',
    DEFAULT_MY_INJECTION,
  );
  if (myInjectionPath !== null) {
    items.push(...readMarkdownSections(myInjectionPath, 'Trick'));
  }

  const bundledOverride = process.env.KISS_INJECTIONS_PATH;
  let bundledPath: string | null = bundledOverride || null;
  if (!bundledPath) {
    const kissRoot = findKissProject();
    if (kissRoot) {
      bundledPath = path.join(kissRoot, 'src', 'kiss', 'INJECTIONS.md');
    }
  }
  if (bundledPath) {
    items.push(...readMarkdownSections(bundledPath, 'Trick'));
  }

  return items;
}

function parseTipSections(text: string): string[] {
  const tips: string[] = [];
  const sections = text.split(/^# Tip.*$/m);
  for (let i = 1; i < sections.length; i++) {
    const body = sections[i].trim();
    if (body) tips.push(body);
  }
  return tips;
}

export function getTips(): string[] {
  let tipsPath: string | null = process.env.KISS_TIPS_PATH || null;
  if (!tipsPath) {
    const kissRoot = findKissProject();
    if (kissRoot) tipsPath = path.join(kissRoot, 'src', 'kiss', 'TIPS.md');
  }
  if (!tipsPath) return [];
  let text: string;
  try {
    text = fs.readFileSync(tipsPath, 'utf-8');
  } catch {
    return [];
  }
  return parseTipSections(text);
}

export function consumeTipsFirstRun(): boolean {
  const marker = path.join(kissHomeDir(), 'TIPS_SHOWN');
  try {
    if (fs.existsSync(marker)) return false;
    fs.mkdirSync(path.dirname(marker), {recursive: true});
    fs.writeFileSync(marker, new Date().toISOString() + '\n');
    return true;
  } catch {
    return false;
  }
}

export function resetTipsOnExtensionUpdate(): void {
  const home = kissHomeDir();
  try {
    if (fs.existsSync(path.join(home, '.extension-updated'))) {
      fs.rmSync(path.join(home, 'TIPS_SHOWN'), {force: true});
    }
  } catch {}
}

export function readSampleTasks(extensionRoot: string): Array<{text: string}> {
  const items: Array<{text: string}> = [];

  const myTasksPath = ensureUserAssetFromDefault(
    'MY_TASK_TEMPLATES.md',
    '## Task\n\nHi!\n',
  );
  if (myTasksPath !== null) {
    for (const text of readMarkdownSections(myTasksPath, 'Task')) {
      items.push({text});
    }
  }

  const packagePath = path.join(
    extensionRoot,
    'kiss_project',
    'src',
    'kiss',
    'SAMPLE_TASKS.md',
  );
  const sourcePath = path.join(extensionRoot, '..', '..', 'SAMPLE_TASKS.md');
  const bundledPath = fs.existsSync(packagePath) ? packagePath : sourcePath;
  for (const text of readMarkdownSections(bundledPath, 'Task')) {
    items.push({text});
  }

  return items;
}

export function getNonce(): string {
  return crypto
    .randomBytes(24)
    .toString('base64')
    .replace(/[^A-Za-z0-9]/g, '')
    .slice(0, 32);
}

function mediaAssetVersion(extensionUri: vscode.Uri, name: string): string {
  const file = vscode.Uri.joinPath(extensionUri, 'media', name).fsPath;
  const bytes = fs.readFileSync(file);
  return crypto.createHash('sha256').update(bytes).digest('hex').slice(0, 16);
}

/** Escape a string for interpolation into an HTML text position. */
function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function buildChatHtml(
  webview: vscode.Webview,
  extensionUri: vscode.Uri,
  selectedModel: string,
): string {
  const nonce = getNonce();
  const version = getVersion();
  const tricksJson = JSON.stringify(getTricks()).replace(/<\//g, '<\\/');
  const tips = getTips();
  const tipsJson = JSON.stringify({
    tips,
    show: tips.length > 0 && consumeTipsFirstRun(),
  }).replace(/<\//g, '<\\/');
  const mod = process.platform === 'darwin' ? '⌘' : 'Ctrl+';

  const tplPath = vscode.Uri.joinPath(
    extensionUri,
    'media',
    'chat.html',
  ).fsPath;
  const tpl = fs.readFileSync(tplPath, 'utf-8');

  const u = (name: string): string => {
    const uri = webview.asWebviewUri(
      vscode.Uri.joinPath(extensionUri, 'media', name),
    );
    const sep = uri.toString().includes('?') ? '&' : '?';
    return uri.toString() + sep + 'v=' + mediaAssetVersion(extensionUri, name);
  };

  /* eslint-disable quotes */
  const csp =
    `<meta http-equiv="Content-Security-Policy" content="default-src 'none';` +
    ` style-src ${webview.cspSource} 'unsafe-inline';` +
    ` script-src 'nonce-${nonce}';` +
    ` img-src ${webview.cspSource} data: https:;` +
    ` font-src ${webview.cspSource};` +
    ` media-src data: ${webview.cspSource};` +
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
    // The model name can come from user settings or the daemon; escape it
    // so a crafted value cannot inject markup into the privileged webview.
    MODEL_NAME: escapeHtml(selectedModel),
    VERSION_SUFFIX: version ? ' ' + version : '',
    AUTH_MODAL: '',
    NONCE_ATTR: ` nonce="${nonce}"`,
    HLJS_SRC: u('highlight.min.js'),
    MARKED_SRC: u('marked.min.js'),
    API_SRC: u('api.js'),
    PANEL_COPY_SRC: u('panelCopy.js'),
    CTX_MENU_SRC: u('contentContextMenu.js'),
    MAIN_SRC: u('main.js'),
    SHIM_SCRIPT: '',
    TRICKS_JSON: tricksJson,
    TIPS_JSON: tipsJson,
    TIPS_SRC: u('tips.js'),
    VOICE_SRC: u('voice.js'),
    VOICE_CONFIG: JSON.stringify({
      mode: 'webview',
      ackAudioUrl: u('working-on-it.mp3'),
    }),
  };

  return tpl.replace(/\{\{([A-Z_]+)\}\}/g, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(subs, key) ? subs[key] : match,
  );
}

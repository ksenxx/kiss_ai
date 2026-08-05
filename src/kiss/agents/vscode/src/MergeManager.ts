// Author: Koushik Sen (ksen@berkeley.edu)
// Contributors:
// Koushik Sen (ksen@berkeley.edu)
// add your name here

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {EventEmitter} from 'events';
import {
  showInformationNotification,
  showWarningNotification,
} from './WebviewNotifications';

interface ProcessedHunk {
  os: number;
  oc: number;
  ns: number;
  nc: number;
  baseLines: string[];
}

interface MergeFileState {
  basePath: string;
  hunks: ProcessedHunk[];
  isNewFile: boolean;
  isBinary: boolean;
  targetPath: string;
  linkTarget?: string;
  // Tri-state pre-task executable mode: true = base was executable,
  // false = base was not, undefined = unknown (leave the mode alone).
  exec: boolean | undefined;
}

export interface MergeFileData {
  name: string;
  base: string;
  current: string;
  hunks: Array<{bs: number; bc: number; cs: number; cc: number}>;
  binary?: boolean;
  target?: string;
  link_target?: string;
  exec?: boolean;
}

export interface MergeData {
  files: MergeFileData[];
}

/**
 * Whether a merge may bring an editor in front of the user.
 *
 * A plain boolean is a snapshot; a predicate is re-read at the moment the
 * editor would actually be opened, which is the only correct answer when
 * the decision depends on which chat tab the user is looking at right now.
 */
export type RevealPermission = boolean | (() => boolean);

const ALWAYS_REVEAL = (): boolean => true;

function toRevealPredicate(reveal: RevealPermission): () => boolean {
  if (typeof reveal === 'function') return reveal;
  return reveal ? ALWAYS_REVEAL : () => false;
}

export class MergeManager extends EventEmitter {
  private _ms: Record<string, MergeFileState> = {};
  private _curHunk: {fp: string; idx: number} | null = null;
  private _redDeco: vscode.TextEditorDecorationType;
  private _blueDeco: vscode.TextEditorDecorationType;
  private _disposables: vscode.Disposable[] = [];
  private _hunkOpInProgress: boolean = false;
  private _mergeInProgress: boolean = false;
  private _pendingMerge: MergeData | null = null;
  private _pendingReveal: () => boolean = ALWAYS_REVEAL;
  private _navSeq: number = 0;
  private _reinsertingFiles = new Set<string>();
  // Files this manager opened in the editor itself (i.e. that were not
  // already open when it needed them). Post-merge editor restoration
  // must only close these, never editors the user opened.
  public readonly openedFiles = new Set<string>();

  constructor() {
    super();
    this._redDeco = vscode.window.createTextEditorDecorationType({
      backgroundColor: 'rgba(248,81,73,0.15)',
      isWholeLine: true,
    });
    this._blueDeco = vscode.window.createTextEditorDecorationType({
      backgroundColor: 'rgba(46,160,67,0.15)',
      isWholeLine: true,
    });

    const visibleSub = vscode.window.onDidChangeVisibleTextEditors(() => {
      for (const fp of Object.keys(this._ms)) {
        this._refreshDeco(fp);
      }
    });
    this._disposables.push(visibleSub);

    const willSaveSub = vscode.workspace.onWillSaveTextDocument(e => {
      this._onWillSave(e);
    });
    this._disposables.push(willSaveSub);

    const didSaveSub = vscode.workspace.onDidSaveTextDocument(doc => {
      void this._onDidSave(doc);
    });
    this._disposables.push(didSaveSub);
  }

  private _onWillSave(e: vscode.TextDocumentWillSaveEvent): void {
    const fp = e.document.uri.fsPath;
    const s = this._ms[fp];
    if (!s || s.hunks.length === 0 || s.isBinary) return;
    this._reinsertingFiles.add(fp);
    e.waitUntil(
      (async () => {
        const ed = await this._getOrOpenEditor(fp);
        for (let i = s.hunks.length - 1; i >= 0; i--) {
          const h = s.hunks[i];
          if (h.oc > 0) {
            const ok = await this._delLines(ed, h.os, h.oc);
            // Only advance hunk state when the edit actually happened;
            // otherwise the old lines are still in the document and the
            // recorded coordinates must keep describing them.
            if (!ok) continue;
            for (let j = i; j < s.hunks.length; j++) {
              if (j === i) {
                s.hunks[j].ns -= h.oc;
              } else {
                s.hunks[j].os -= h.oc;
                s.hunks[j].ns -= h.oc;
              }
            }
            h.oc = 0;
          }
        }
      })(),
    );
  }

  private async _onDidSave(doc: vscode.TextDocument): Promise<void> {
    const fp = doc.uri.fsPath;
    if (!this._reinsertingFiles.delete(fp)) return;
    const s = this._ms[fp];
    if (!s) return;
    const ed = await this._getOrOpenEditor(fp);
    let offset = 0;
    for (const h of s.hunks) {
      const old = h.baseLines;
      if (old.length > 0 && h.oc === 0) {
        const insertLine = h.os + offset;
        const txt = old.join('\n') + '\n';
        let ok = await ed.edit(eb => {
          eb.insert(new vscode.Position(insertLine, 0), txt);
        });
        if (!ok) {
          ok = await ed.edit(eb => {
            eb.insert(new vscode.Position(insertLine, 0), txt);
          });
        }
        if (!ok) {
          console.error(
            `[MergeManager] ed.edit failed re-inserting base lines in ${fp}`,
          );
          continue;
        }
        h.os = insertLine;
        h.oc = old.length;
        h.ns = insertLine + old.length;
        offset += old.length;
      }
    }
    this._refreshDeco(fp);
  }

  private _refreshDeco(fp: string): void {
    for (const ed of vscode.window.visibleTextEditors) {
      if (ed.document.uri.fsPath !== fp) continue;
      const s = this._ms[fp];
      const reds: vscode.Range[] = [];
      const blues: vscode.Range[] = [];
      if (s) {
        for (const h of s.hunks) {
          if (h.oc > 0) {
            reds.push(new vscode.Range(h.os, 0, h.os + h.oc - 1, 99999));
          }
          if (h.nc > 0) {
            blues.push(new vscode.Range(h.ns, 0, h.ns + h.nc - 1, 99999));
          }
        }
      }
      ed.setDecorations(this._redDeco, reds);
      ed.setDecorations(this._blueDeco, blues);
    }
  }

  private async _delLines(
    ed: vscode.TextEditor,
    start: number,
    count: number,
  ): Promise<boolean> {
    if (count <= 0) return true;
    const end = start + count;
    const doc = ed.document;
    let ok: boolean;
    if (end < doc.lineCount) {
      ok = await ed.edit(eb => {
        eb.delete(new vscode.Range(start, 0, end, 0));
      });
    } else if (start > 0) {
      const prevLine = doc.lineAt(start - 1);
      const lastLine = doc.lineAt(doc.lineCount - 1);
      ok = await ed.edit(eb => {
        eb.delete(
          new vscode.Range(
            start - 1,
            prevLine.text.length,
            lastLine.range.end.line,
            lastLine.text.length,
          ),
        );
      });
    } else {
      const lastLine = doc.lineAt(doc.lineCount - 1);
      ok = await ed.edit(eb => {
        eb.replace(
          new vscode.Range(0, 0, lastLine.range.end.line, lastLine.text.length),
          '',
        );
      });
    }
    if (!ok) {
      console.error(
        `[MergeManager] ed.edit failed in _delLines (start=${start}, count=${count})`,
      );
    }
    return ok;
  }

  private async _getOrOpenEditor(fp: string): Promise<vscode.TextEditor> {
    const existing = vscode.window.visibleTextEditors.find(
      e => e.document.uri.fsPath === fp,
    );
    if (existing) return existing;
    this.openedFiles.add(fp);
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
    return vscode.window.showTextDocument(doc, {
      preview: false,
      viewColumn: vscode.ViewColumn.One,
    });
  }

  private _afterHunkAction(fp: string): void {
    this._refreshDeco(fp);
    if (Object.keys(this._ms).length > 0) {
      this.nextChange();
    } else {
      this._checkAllDone();
    }
  }

  private async _delLinesWithRetry(
    ed: vscode.TextEditor,
    start: number,
    count: number,
  ): Promise<boolean> {
    let ok = await this._delLines(ed, start, count);
    if (!ok) {
      ok = await this._delLines(ed, start, count);
    }
    return ok;
  }

  private async _applyHunkAction(
    fp: string,
    idx: number,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
  ): Promise<void> {
    const s = this._ms[fp];
    if (!s) return;
    const h = s.hunks[idx];
    if (s.isBinary) {
      const wasNew = s.isNewFile;
      const restore = {
        targetPath: s.targetPath,
        basePath: s.basePath,
        linkTarget: s.linkTarget,
        exec: s.exec,
      };
      s.hunks.splice(idx, 1);
      if (!s.hunks.length) {
        delete this._ms[fp];
        if (countProp === 'nc') {
          if (wasNew) {
            await this._deleteNewFile(fp);
          } else {
            this._restoreBase(restore);
          }
        }
      }
      this._afterHunkAction(fp);
      return;
    }
    if (h[countProp] > 0) {
      const ed = await this._getOrOpenEditor(fp);
      const ok = await this._delLinesWithRetry(ed, h[startProp], h[countProp]);
      if (!ok) {
        showWarningNotification('Failed to apply change. Please try again.');
        return;
      }
      const rm = h[countProp];
      s.hunks.splice(idx, 1);
      for (let i = idx; i < s.hunks.length; i++) {
        s.hunks[i].os -= rm;
        s.hunks[i].ns -= rm;
      }
    } else {
      s.hunks.splice(idx, 1);
    }
    if (!s.hunks.length) {
      const wasNew = s.isNewFile;
      delete this._ms[fp];
      if (wasNew && countProp === 'nc') {
        await this._deleteNewFile(fp);
      } else if (countProp === 'nc' && s.targetPath !== fp) {
        this._restoreBase(s);
      } else if (countProp === 'nc') {
        this._applyExecState(s.targetPath, s.exec);
      }
    }
    this._afterHunkAction(fp);
  }

  private async _withHunkGuard(fn: () => Promise<void>): Promise<void> {
    if (this._hunkOpInProgress) return;
    this._hunkOpInProgress = true;
    try {
      await fn();
    } finally {
      this._hunkOpInProgress = false;
    }
  }

  async acceptChange(fp?: string, idx?: number): Promise<void> {
    await this._resolveHunk(fp, idx, 'oc', 'os');
  }

  async rejectChange(fp?: string, idx?: number): Promise<void> {
    await this._resolveHunk(fp, idx, 'nc', 'ns');
  }

  private async _resolveHunk(
    fp: string | undefined,
    idx: number | undefined,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
  ): Promise<void> {
    return this._withHunkGuard(async () => {
      const target = fp && idx !== undefined ? {fp, idx} : this._curHunk;
      if (!target || !this._ms[target.fp]) return;
      await this._applyHunkAction(target.fp, target.idx, countProp, startProp);
    });
  }

  private _hunkLine(h: ProcessedHunk): number {
    return h.nc > 0 ? h.ns : h.os;
  }

  prevChange(): void {
    void this._navigateHunk(-1);
  }

  nextChange(): void {
    void this._navigateHunk(1);
  }

  private async _navigateHunk(dir: number): Promise<void> {
    const seq = ++this._navSeq;
    const allH: Array<{fp: string; h: ProcessedHunk}> = [];
    for (const fp of Object.keys(this._ms)) {
      for (const h of this._ms[fp].hunks) {
        allH.push({fp, h});
      }
    }
    if (!allH.length) {
      this._curHunk = null;
      return;
    }

    const ae = vscode.window.activeTextEditor;
    const cf = ae ? ae.document.uri.fsPath : '';
    const cl = ae ? ae.selection.active.line : dir < 0 ? 999999 : -1;
    const cmp =
      dir < 0
        ? (a: number, b: number) => a < b
        : (a: number, b: number) => a > b;
    const start = dir < 0 ? allH.length - 1 : 0;
    const end = dir < 0 ? -1 : allH.length;
    const step = dir < 0 ? -1 : 1;

    let found: (typeof allH)[number] | null = null;
    for (let j = start; j !== end; j += step) {
      const ln = this._hunkLine(allH[j].h);
      if (allH[j].fp === cf && cmp(ln, cl)) {
        found = allH[j];
        break;
      }
    }
    if (!found) {
      for (let j = start; j !== end; j += step) {
        if (allH[j].fp !== cf) {
          found = allH[j];
          break;
        }
      }
    }
    if (!found) found = allH[dir < 0 ? allH.length - 1 : 0];

    this._curHunk = {
      fp: found.fp,
      idx: this._ms[found.fp].hunks.indexOf(found.h),
    };

    this._recordOpen(found.fp);
    if (this._ms[found.fp]?.isBinary) {
      await vscode.commands.executeCommand(
        'vscode.open',
        vscode.Uri.file(found.fp),
        {viewColumn: vscode.ViewColumn.One, preview: false},
      );
      return;
    }

    const doc = await vscode.workspace.openTextDocument(
      vscode.Uri.file(found.fp),
    );
    if (this._navSeq !== seq) return;
    const ed = await vscode.window.showTextDocument(doc, {
      preview: false,
      viewColumn: vscode.ViewColumn.One,
    });
    if (this._navSeq !== seq) return;
    const ln = this._hunkLine(found.h);
    ed.revealRange(
      new vscode.Range(ln, 0, ln, 0),
      vscode.TextEditorRevealType.InCenter,
    );
    ed.selection = new vscode.Selection(ln, 0, ln, 0);
  }

  private async _deleteFileHunks(
    fp: string,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
  ): Promise<boolean> {
    const s = this._ms[fp];
    if (!s) return true;
    let allOk = true;
    const ed = await this._getOrOpenEditor(fp);
    for (let i = s.hunks.length - 1; i >= 0; i--) {
      if (s.hunks[i][countProp] > 0) {
        const ok = await this._delLinesWithRetry(
          ed,
          s.hunks[i][startProp],
          s.hunks[i][countProp],
        );
        if (!ok) {
          allOk = false;
          console.error(
            `[MergeManager] Failed to delete hunk ${i} lines in ${fp}`,
          );
        }
      }
    }
    return allOk;
  }

  /** Record that this manager is about to open *fp* in the editor. */
  private _recordOpen(fp: string): void {
    const alreadyVisible = vscode.window.visibleTextEditors.some(
      e => e.document.uri.fsPath === fp,
    );
    if (!alreadyVisible) this.openedFiles.add(fp);
  }

  /** Restore the pre-task executable mode recorded in the manifest. */
  private _applyExecState(targetPath: string, exec: boolean | undefined): void {
    if (exec === undefined) return;
    try {
      const mode = fs.statSync(targetPath).mode;
      fs.chmodSync(targetPath, exec ? mode | 0o111 : mode & ~0o111);
    } catch {}
  }

  private _restoreBase(s: {
    targetPath: string;
    basePath: string;
    linkTarget?: string;
    exec: boolean | undefined;
  }): void {
    const {targetPath, basePath, linkTarget, exec} = s;
    try {
      try {
        if (fs.lstatSync(targetPath).isSymbolicLink()) {
          fs.unlinkSync(targetPath);
        }
      } catch {}
      fs.mkdirSync(path.dirname(targetPath), {recursive: true});
      if (linkTarget !== undefined) {
        try {
          fs.unlinkSync(targetPath);
        } catch {}
        fs.symlinkSync(linkTarget, targetPath);
        return;
      }
      fs.copyFileSync(basePath, targetPath);
      this._applyExecState(targetPath, exec);
    } catch {
      console.error(`[MergeManager] failed to restore base for ${targetPath}`);
    }
  }

  private async _deleteNewFile(fp: string): Promise<void> {
    let deleted = false;
    try {
      const edit = new vscode.WorkspaceEdit();
      edit.deleteFile(vscode.Uri.file(fp), {ignoreIfNotExists: true});
      deleted = await vscode.workspace.applyEdit(edit);
    } catch {
      deleted = false;
    }
    if (!deleted) {
      try {
        fs.unlinkSync(fp);
      } catch {}
    }
  }

  private async _resolveAll(
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
    label: string,
  ): Promise<void> {
    const fps = Object.keys(this._ms);
    let anyFailed = false;
    for (const fp of fps) {
      const s = this._ms[fp];
      if (!s) continue;
      if (!s.isBinary) {
        const ok = await this._deleteFileHunks(fp, countProp, startProp);
        if (!ok) {
          // Keep the file's merge state so the user can retry; a failed
          // edit must not be reported as a completed review.
          anyFailed = true;
          continue;
        }
      }
      const wasNew = s.isNewFile;
      delete this._ms[fp];
      this._refreshDeco(fp);
      if (countProp === 'nc') {
        if (wasNew) {
          await this._deleteNewFile(fp);
        } else if (s.isBinary || s.targetPath !== fp) {
          this._restoreBase(s);
        } else {
          this._applyExecState(s.targetPath, s.exec);
        }
      }
    }
    this._curHunk = null;
    await vscode.workspace.saveAll(false);
    if (anyFailed) {
      showWarningNotification(
        'Some changes could not be applied. Please try again.',
      );
      this.nextChange();
      return;
    }
    showInformationNotification(label);
    this.emit('allDone');
  }

  private async _resolveFile(
    fp: string,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
  ): Promise<void> {
    const s = this._ms[fp];
    const wasNew = s?.isNewFile ?? false;
    const wasBinary = s?.isBinary ?? false;
    const restore = s
      ? {
          targetPath: s.targetPath,
          basePath: s.basePath,
          linkTarget: s.linkTarget,
          exec: s.exec,
        }
      : null;
    if (!wasBinary) {
      const ok = await this._deleteFileHunks(fp, countProp, startProp);
      if (!ok) {
        // The document still holds the undeleted lines; keep the merge
        // state so the user can retry instead of losing the review.
        showWarningNotification('Failed to apply change. Please try again.');
        this._refreshDeco(fp);
        return;
      }
    }
    delete this._ms[fp];
    if (countProp === 'nc') {
      if (wasNew) {
        await this._deleteNewFile(fp);
      } else if (restore && (wasBinary || restore.targetPath !== fp)) {
        this._restoreBase(restore);
      } else if (restore) {
        this._applyExecState(restore.targetPath, restore.exec);
      }
    }
    this._curHunk = null;
    this._afterHunkAction(fp);
  }

  async acceptFile(): Promise<void> {
    return this._withHunkGuard(async () => {
      if (!this._curHunk || !this._ms[this._curHunk.fp]) return;
      await this._resolveFile(this._curHunk.fp, 'oc', 'os');
    });
  }

  async rejectFile(): Promise<void> {
    return this._withHunkGuard(async () => {
      if (!this._curHunk || !this._ms[this._curHunk.fp]) return;
      await this._resolveFile(this._curHunk.fp, 'nc', 'ns');
    });
  }

  async acceptAll(): Promise<void> {
    return this._withHunkGuard(() =>
      this._resolveAll('oc', 'os', 'All changes accepted.'),
    );
  }

  async rejectAll(): Promise<void> {
    return this._withHunkGuard(() =>
      this._resolveAll('nc', 'ns', 'All changes rejected.'),
    );
  }

  private _checkAllDone(): void {
    if (Object.keys(this._ms).length > 0) return;
    this._curHunk = null;
    vscode.workspace.saveAll(false).then(
      () => {
        showInformationNotification('All changes reviewed.');
        this.emit('allDone');
      },
      () => {
        showInformationNotification('All changes reviewed.');
        this.emit('allDone');
      },
    );
  }

  /**
   * Load a merge payload and, unless suppressed, show its first hunk.
   *
   * Args:
   *   data: The merge payload to review.
   *   reveal: Whether the first changed file may be brought in front of
   *     the user. Pass ``false`` -- or a predicate that will return
   *     ``false`` -- for a merge produced by a chat tab the user is not
   *     looking at: the hunk state and decorations are still prepared,
   *     but no editor is opened or scrolled. The user sees the diff when
   *     they switch to that tab and navigate a hunk. Prefer a predicate:
   *     preparing a merge awaits several host operations, and the user
   *     can leave the tab while they run, so the permission is only
   *     meaningful when it is read just before the editor is opened.
   */
  async openMerge(
    data: MergeData,
    reveal: RevealPermission = true,
  ): Promise<void> {
    const mayReveal = toRevealPredicate(reveal);
    if (this._mergeInProgress) {
      this._pendingMerge = data;
      this._pendingReveal = mayReveal;
      return;
    }
    this._mergeInProgress = true;
    try {
      let next: MergeData | null = data;
      let nextReveal = mayReveal;
      while (next) {
        const cur: MergeData = next;
        const curReveal = nextReveal;
        next = null;
        try {
          await this._doOpenMerge(cur, curReveal);
        } catch (err) {
          // A failed open must not leave a stale _pendingMerge behind:
          // it would replay an outdated payload over a newer merge.
          console.error('[MergeManager] openMerge failed:', err);
        }
        next = this._pendingMerge;
        nextReveal = this._pendingReveal;
        this._pendingMerge = null;
        this._pendingReveal = ALWAYS_REVEAL;
      }
    } finally {
      this._mergeInProgress = false;
    }
  }

  private async _doOpenMerge(
    data: MergeData,
    mayReveal: () => boolean = ALWAYS_REVEAL,
  ): Promise<void> {
    try {
      await vscode.workspace.saveAll(false);
    } catch {}

    for (const fp of Object.keys(this._ms)) {
      for (const ed of vscode.window.visibleTextEditors) {
        if (ed.document.uri.fsPath === fp) {
          ed.setDecorations(this._redDeco, []);
          ed.setDecorations(this._blueDeco, []);
        }
      }
    }
    this._ms = {};
    this._reinsertingFiles.clear();

    let firstFileFp: string | null = null;

    for (const f of data.files || []) {
      if (!firstFileFp) {
        firstFileFp = f.current;
      }

      if (f.binary) {
        const dummyHunk: ProcessedHunk = {
          os: 0,
          oc: 0,
          ns: 0,
          nc: 0,
          baseLines: [],
        };
        const hasBase = (() => {
          try {
            return fs.statSync(f.base).size > 0;
          } catch {
            return false;
          }
        })();
        this._ms[f.current] = {
          basePath: f.base,
          hunks: [dummyHunk],
          // A pre-existing file whose pre-task content was empty also has
          // a zero-byte base snapshot; the manifest carries `exec` (or
          // `link_target`) exactly when the pre-task file existed, so use
          // that to avoid deleting a pre-existing file on reject.
          isNewFile:
            !hasBase && f.exec === undefined && f.link_target === undefined,
          isBinary: true,
          targetPath: f.target || f.current,
          linkTarget: f.link_target,
          exec: f.exec,
        };
        continue;
      }

      const currentUri = vscode.Uri.file(f.current);
      const doc = await vscode.workspace.openTextDocument(currentUri);

      if (doc.isDirty) {
        try {
          const diskContent = fs.readFileSync(f.current, 'utf8');
          const lastLine = doc.lineAt(doc.lineCount - 1);
          const revertEdit = new vscode.WorkspaceEdit();
          revertEdit.replace(
            doc.uri,
            new vscode.Range(
              0,
              0,
              lastLine.range.end.line,
              lastLine.range.end.character,
            ),
            diskContent,
          );
          await vscode.workspace.applyEdit(revertEdit);
        } catch {}
      }

      let baseLines: string[] = [];
      try {
        baseLines = fs.readFileSync(f.base, 'utf8').split('\n');
      } catch {}

      const hunks = (f.hunks || [])
        .map(h => ({cs: h.cs, cc: h.cc, bs: h.bs, bc: h.bc}))
        .sort((a, b) => a.cs - b.cs);

      const wsEdit = new vscode.WorkspaceEdit();
      let offset = 0;
      const processed: ProcessedHunk[] = [];

      for (const h of hunks) {
        const old = h.bc > 0 ? baseLines.slice(h.bs, h.bs + h.bc) : [];
        if (old.length > 0) {
          const txt = old.join('\n') + '\n';
          wsEdit.insert(doc.uri, new vscode.Position(h.cs, 0), txt);
        }
        processed.push({
          os: h.cs + offset,
          oc: old.length,
          ns: h.cs + offset + old.length,
          nc: h.cc,
          baseLines: old,
        });
        offset += old.length;
      }

      if (wsEdit.size > 0) {
        const ok = await vscode.workspace.applyEdit(wsEdit);
        if (!ok) {
          // The base lines were never inserted, so the computed hunk
          // coordinates do not describe the document. Tracking them
          // anyway would let accept/reject delete unrelated lines.
          console.error(`[MergeManager] applyEdit failed for ${f.current}`);
          showWarningNotification(
            `Could not open the review for ${f.name}; ` +
              'its changes are left applied.',
          );
          if (firstFileFp === f.current) firstFileFp = null;
          continue;
        }
      }

      const hasTextBase = (() => {
        try {
          return fs.statSync(f.base).size > 0;
        } catch {
          return false;
        }
      })();
      // See the binary case above: `exec` present means the pre-task file
      // existed, so an empty base snapshot alone does not make it new.
      const isNewFile =
        processed.length > 0 && !hasTextBase && f.exec === undefined;
      this._ms[f.current] = {
        basePath: f.base,
        hunks: processed,
        isNewFile,
        isBinary: false,
        targetPath: f.target || f.current,
        linkTarget: f.link_target,
        exec: f.exec,
      };
    }

    if (firstFileFp && this._ms[firstFileFp]?.hunks.length) {
      this._curHunk = {fp: firstFileFp, idx: 0};
      // A merge from a chat tab the user is not looking at keeps its hunk
      // state but must not pull an editor in front of them. Nothing was
      // opened, so there is also nothing for the restore pass to close.
      // The permission is read here, after every await above, because the
      // user may have switched or closed the chat tab while they ran.
      if (!mayReveal()) {
        this._refreshDeco(firstFileFp);
      } else {
        this._recordOpen(firstFileFp);
        if (this._ms[firstFileFp].isBinary) {
          await vscode.commands.executeCommand(
            'vscode.open',
            vscode.Uri.file(firstFileFp),
            {viewColumn: vscode.ViewColumn.One, preview: false},
          );
        } else {
          const firstDoc = await vscode.workspace.openTextDocument(
            vscode.Uri.file(firstFileFp),
          );
          const firstEd = await vscode.window.showTextDocument(firstDoc, {
            preview: false,
            viewColumn: vscode.ViewColumn.One,
          });
          const fh = this._ms[firstFileFp].hunks[0];
          const fl = fh.nc > 0 ? fh.ns : fh.os;
          firstEd.revealRange(
            new vscode.Range(fl, 0, fl, 0),
            vscode.TextEditorRevealType.InCenter,
          );
          firstEd.selection = new vscode.Selection(fl, 0, fl, 0);
          this._refreshDeco(firstFileFp);
        }
      }
    } else {
      this._curHunk = null;
    }

    const fileCount = (data.files || []).length;
    showInformationNotification(
      `Reviewing ${fileCount} file(s). Red = old, Green = new. Use Accept / Reject.`,
    );
  }

  dispose(): void {
    this._redDeco.dispose();
    this._blueDeco.dispose();
    for (const d of this._disposables) {
      d.dispose();
    }
    this._disposables = [];
    this.removeAllListeners();
  }
}

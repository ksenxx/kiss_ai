/**
 * Merge view manager for reviewing agent file changes.
 *
 * Ports the inline merge/decoration logic from the code-server extension
 * (code_server.py _CS_EXTENSION_JS) to native VS Code APIs, allowing
 * accept/reject of individual hunks after the agent modifies files.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import { EventEmitter } from 'events';

interface ProcessedHunk {
  /** Old-lines start (0-based, in the merged document) */
  os: number;
  /** Old-lines count */
  oc: number;
  /** New-lines start (0-based, in the merged document) */
  ns: number;
  /** New-lines count */
  nc: number;
  /** Stored base (old) lines for re-insertion after save */
  baseLines: string[];
}

interface MergeFileState {
  basePath: string;
  hunks: ProcessedHunk[];
}

interface MergeFileData {
  name: string;
  base: string;
  current: string;
  hunks: Array<{ bs: number; bc: number; cs: number; cc: number }>;
}

interface MergeData {
  branch?: string;
  files: MergeFileData[];
}

export class MergeManager extends EventEmitter {
  private _ms: Record<string, MergeFileState> = {};
  private _curHunk: { fp: string; idx: number } | null = null;
  private _redDeco: vscode.TextEditorDecorationType;
  private _blueDeco: vscode.TextEditorDecorationType;
  private _disposables: vscode.Disposable[] = [];
  private _hunkOpInProgress: boolean = false;
  private _mergeInProgress: boolean = false;
  private _pendingMerge: MergeData | null = null;
  private _navSeq: number = 0;
  private _reinsertingFiles = new Set<string>();

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

    const willSaveSub = vscode.workspace.onWillSaveTextDocument((e) => {
      this._onWillSave(e);
    });
    this._disposables.push(willSaveSub);

    const didSaveSub = vscode.workspace.onDidSaveTextDocument((doc) => {
      this._onDidSave(doc);
    });
    this._disposables.push(didSaveSub);
  }

  get isActive(): boolean {
    return Object.keys(this._ms).length > 0;
  }

  get totalHunks(): number {
    let count = 0;
    for (const fp of Object.keys(this._ms)) {
      count += this._ms[fp].hunks.length;
    }
    return count;
  }

  /**
   * Before save: strip base (old) lines so saved content is clean.
   * Uses waitUntil() to defer the save until edits complete.
   */
  private _onWillSave(e: vscode.TextDocumentWillSaveEvent): void {
    const fp = e.document.uri.fsPath;
    const s = this._ms[fp];
    if (!s || s.hunks.length === 0) return;
    this._reinsertingFiles.add(fp);
    e.waitUntil(
      (async () => {
        const ed = await this._getOrOpenEditor(fp);
        // Remove old-lines in reverse order to preserve earlier indices
        for (let i = s.hunks.length - 1; i >= 0; i--) {
          const h = s.hunks[i];
          if (h.oc > 0) {
            await this._delLines(ed, h.os, h.oc);
            // Shift this and all later hunks
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
      })()
    );
  }

  /**
   * After save: re-insert base lines so merge decorations reappear.
   */
  private async _onDidSave(doc: vscode.TextDocument): Promise<void> {
    const fp = doc.uri.fsPath;
    if (!this._reinsertingFiles.delete(fp)) return;
    const s = this._ms[fp];
    if (!s) return;
    const ed = await this._getOrOpenEditor(fp);
    let offset = 0;
    for (const h of s.hunks) {
      const old = h.baseLines;
      if (old.length > 0) {
        const il = h.ns + offset - old.length; // ns was shifted; insert before new lines
        const insertLine = h.os + offset;
        const txt = old.join('\n') + '\n';
        await ed.edit((eb) => {
          eb.insert(new vscode.Position(insertLine, 0), txt);
        });
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
    count: number
  ): Promise<boolean> {
    if (count <= 0) return true;
    const end = start + count;
    const doc = ed.document;
    let ok: boolean;
    if (end < doc.lineCount) {
      ok = await ed.edit((eb) => {
        eb.delete(new vscode.Range(start, 0, end, 0));
      });
    } else if (start > 0) {
      const prevLine = doc.lineAt(start - 1);
      const lastLine = doc.lineAt(doc.lineCount - 1);
      ok = await ed.edit((eb) => {
        eb.delete(
          new vscode.Range(
            start - 1,
            prevLine.text.length,
            lastLine.range.end.line,
            lastLine.text.length
          )
        );
      });
    } else {
      const lastLine = doc.lineAt(doc.lineCount - 1);
      ok = await ed.edit((eb) => {
        eb.replace(
          new vscode.Range(0, 0, lastLine.range.end.line, lastLine.text.length),
          ''
        );
      });
    }
    if (!ok) {
      console.error(`[MergeManager] ed.edit failed in _delLines (start=${start}, count=${count})`);
    }
    return ok;
  }

  private async _getOrOpenEditor(
    fp: string
  ): Promise<vscode.TextEditor> {
    const existing = vscode.window.visibleTextEditors.find(
      (e) => e.document.uri.fsPath === fp
    );
    if (existing) return existing;
    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
    return vscode.window.showTextDocument(doc, { preview: false });
  }

  private _afterHunkAction(fp: string): void {
    this._refreshDeco(fp);
    if (Object.keys(this._ms).length > 0) {
      this.nextChange();
    } else {
      this._checkAllDone();
    }
  }

  private async _applyHunkAction(
    fp: string,
    idx: number,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns'
  ): Promise<void> {
    const s = this._ms[fp];
    if (!s) return;
    const h = s.hunks[idx];
    if (h[countProp] > 0) {
      const ed = await this._getOrOpenEditor(fp);
      let ok = await this._delLines(ed, h[startProp], h[countProp]);
      if (!ok) {
        ok = await this._delLines(ed, h[startProp], h[countProp]);
      }
      if (!ok) {
        vscode.window.showWarningMessage('Failed to apply change. Please try again.');
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
    if (!s.hunks.length) delete this._ms[fp];
    this.emit('hunkProcessed');
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
    startProp: 'os' | 'ns'
  ): Promise<void> {
    return this._withHunkGuard(async () => {
      const target = fp && idx !== undefined
        ? { fp, idx }
        : this._curHunk;
      if (!target || !this._ms[target.fp]) return;
      await this._applyHunkAction(target.fp, target.idx, countProp, startProp);
    });
  }

  private _hunkLine(h: ProcessedHunk): number {
    return h.nc > 0 ? h.ns : h.os;
  }

  prevChange(): void {
    this._navigateHunk(-1);
  }

  nextChange(): void {
    this._navigateHunk(1);
  }

  private async _navigateHunk(dir: number): Promise<void> {
    const seq = ++this._navSeq;
    const allH: Array<{ fp: string; h: ProcessedHunk }> = [];
    for (const fp of Object.keys(this._ms)) {
      for (const h of this._ms[fp].hunks) {
        allH.push({ fp, h });
      }
    }
    if (!allH.length) {
      this._curHunk = null;
      return;
    }

    const ae = vscode.window.activeTextEditor;
    const cf = ae ? ae.document.uri.fsPath : '';
    const cl = ae ? ae.selection.active.line : dir < 0 ? 999999 : -1;
    const cmp = dir < 0
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

    const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(found.fp));
    if (this._navSeq !== seq) return;  // Superseded by newer navigation
    const ed = await vscode.window.showTextDocument(doc, { preview: false });
    if (this._navSeq !== seq) return;  // Superseded by newer navigation
    const ln = this._hunkLine(found.h);
    ed.revealRange(
      new vscode.Range(ln, 0, ln, 0),
      vscode.TextEditorRevealType.InCenter
    );
    ed.selection = new vscode.Selection(ln, 0, ln, 0);
  }

  private async _deleteFileHunks(
    fp: string,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns'
  ): Promise<void> {
    const s = this._ms[fp];
    if (!s) return;
    const ed = await this._getOrOpenEditor(fp);
    for (let i = s.hunks.length - 1; i >= 0; i--) {
      if (s.hunks[i][countProp] > 0) {
        let ok = await this._delLines(ed, s.hunks[i][startProp], s.hunks[i][countProp]);
        if (!ok) {
          ok = await this._delLines(ed, s.hunks[i][startProp], s.hunks[i][countProp]);
          if (!ok) {
            console.error(`[MergeManager] Failed to delete hunk ${i} lines in ${fp}`);
          }
        }
      }
    }
  }

  private async _resolveAll(
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns',
    label: string
  ): Promise<void> {
    const fps = Object.keys(this._ms);
    try {
      for (const fp of fps) {
        await this._deleteFileHunks(fp, countProp, startProp);
      }
    } finally {
      this._ms = {};
      this._curHunk = null;
      for (const fp of fps) {
        this._refreshDeco(fp);
      }
      await vscode.workspace.saveAll(false);
      vscode.window.showInformationMessage(label);
      this.emit('allDone');
    }
  }

  private async _resolveFile(
    fp: string,
    countProp: 'oc' | 'nc',
    startProp: 'os' | 'ns'
  ): Promise<void> {
    await this._deleteFileHunks(fp, countProp, startProp);
    delete this._ms[fp];
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
    return this._withHunkGuard(() => this._resolveAll('oc', 'os', 'All changes accepted.'));
  }

  async rejectAll(): Promise<void> {
    return this._withHunkGuard(() => this._resolveAll('nc', 'ns', 'All changes rejected.'));
  }

  private _checkAllDone(): void {
    if (Object.keys(this._ms).length > 0) return;
    this._curHunk = null;
    vscode.workspace.saveAll(false).then(
      () => {
        vscode.window.showInformationMessage('All changes reviewed.');
        this.emit('allDone');
      },
      () => {
        vscode.window.showInformationMessage('All changes reviewed.');
        this.emit('allDone');
      }
    );
  }

  /**
   * Open merge view: insert old lines, apply decorations, navigate to first hunk.
   */
  async openMerge(data: MergeData): Promise<void> {
    if (this._mergeInProgress) {
      this._pendingMerge = data;
      return;
    }
    this._mergeInProgress = true;
    try {
      await this._doOpenMerge(data);
      while (this._pendingMerge) {
        const next = this._pendingMerge;
        this._pendingMerge = null;
        await this._doOpenMerge(next);
      }
    } finally {
      this._mergeInProgress = false;
    }
  }

  private async _doOpenMerge(data: MergeData): Promise<void> {
    try {
      await vscode.workspace.saveAll(false);
    } catch { /* ignore */ }

    // Clear previous decorations
    for (const fp of Object.keys(this._ms)) {
      for (const ed of vscode.window.visibleTextEditors) {
        if (ed.document.uri.fsPath === fp) {
          ed.setDecorations(this._redDeco, []);
          ed.setDecorations(this._blueDeco, []);
        }
      }
    }
    this._ms = {};

    for (const f of data.files || []) {
      const currentUri = vscode.Uri.file(f.current);
      const doc = await vscode.workspace.openTextDocument(currentUri);
      const ed = await vscode.window.showTextDocument(doc, { preview: false });

      if (doc.isDirty) {
        try {
          await vscode.commands.executeCommand(
            'workbench.action.files.revert'
          );
        } catch { /* ignore */ }
      }

      let baseLines: string[] = [];
      try {
        baseLines = fs.readFileSync(f.base, 'utf8').split('\n');
      } catch { /* ignore */ }

      const hunks = (f.hunks || [])
        .map((h) => ({ cs: h.cs, cc: h.cc, bs: h.bs, bc: h.bc }))
        .sort((a, b) => a.cs - b.cs);

      let offset = 0;
      const processed: ProcessedHunk[] = [];

      for (const h of hunks) {
        const old = h.bc > 0 ? baseLines.slice(h.bs, h.bs + h.bc) : [];
        if (old.length > 0) {
          const il = h.cs + offset;
          const txt = old.join('\n') + '\n';
          await ed.edit((eb) => {
            eb.insert(new vscode.Position(il, 0), txt);
          });
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

      this._ms[f.current] = { basePath: f.base, hunks: processed };
      this._refreshDeco(f.current);

      if (processed.length > 0) {
        ed.revealRange(
          new vscode.Range(processed[0].os, 0, processed[0].os, 0),
          vscode.TextEditorRevealType.InCenter
        );
      }
    }

    // Navigate to first hunk
    const firstFp = Object.keys(this._ms)[0];
    if (firstFp && this._ms[firstFp].hunks.length) {
      this._curHunk = { fp: firstFp, idx: 0 };
      const firstDoc = await vscode.workspace.openTextDocument(
        vscode.Uri.file(firstFp)
      );
      const firstEd = await vscode.window.showTextDocument(firstDoc, {
        preview: false,
      });
      const fh = this._ms[firstFp].hunks[0];
      const fl = fh.nc > 0 ? fh.ns : fh.os;
      firstEd.revealRange(
        new vscode.Range(fl, 0, fl, 0),
        vscode.TextEditorRevealType.InCenter
      );
      firstEd.selection = new vscode.Selection(fl, 0, fl, 0);
    } else {
      this._curHunk = null;
    }

    const fileCount = (data.files || []).length;
    vscode.window.showInformationMessage(
      `Reviewing ${fileCount} file(s). Red = old, Blue = new. Use Accept / Reject.`
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

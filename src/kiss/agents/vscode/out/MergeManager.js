"use strict";
/**
 * Merge view manager for reviewing agent file changes.
 *
 * Ports the inline merge/decoration logic from the code-server extension
 * (code_server.py _CS_EXTENSION_JS) to native VS Code APIs, allowing
 * accept/reject of individual hunks after the agent modifies files.
 */
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.MergeManager = void 0;
const vscode = __importStar(require("vscode"));
const fs = __importStar(require("fs"));
const events_1 = require("events");
class MergeManager extends events_1.EventEmitter {
    _ms = {};
    _curHunk = null;
    _redDeco;
    _blueDeco;
    _disposables = [];
    constructor() {
        super();
        this._redDeco = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(248,81,73,0.15)',
            isWholeLine: true,
        });
        this._blueDeco = vscode.window.createTextEditorDecorationType({
            backgroundColor: 'rgba(59,130,246,0.15)',
            isWholeLine: true,
        });
        const visibleSub = vscode.window.onDidChangeVisibleTextEditors(() => {
            for (const fp of Object.keys(this._ms)) {
                this._refreshDeco(fp);
            }
        });
        this._disposables.push(visibleSub);
    }
    get isActive() {
        return Object.keys(this._ms).length > 0;
    }
    get totalHunks() {
        let count = 0;
        for (const fp of Object.keys(this._ms)) {
            count += this._ms[fp].hunks.length;
        }
        return count;
    }
    _refreshDeco(fp) {
        for (const ed of vscode.window.visibleTextEditors) {
            if (ed.document.uri.fsPath !== fp)
                continue;
            const s = this._ms[fp];
            const reds = [];
            const blues = [];
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
    async _delLines(ed, start, count) {
        if (count <= 0)
            return;
        const end = start + count;
        const doc = ed.document;
        if (end < doc.lineCount) {
            await ed.edit((eb) => {
                eb.delete(new vscode.Range(start, 0, end, 0));
            });
        }
        else if (start > 0) {
            const prevLine = doc.lineAt(start - 1);
            const lastLine = doc.lineAt(doc.lineCount - 1);
            await ed.edit((eb) => {
                eb.delete(new vscode.Range(start - 1, prevLine.text.length, lastLine.range.end.line, lastLine.text.length));
            });
        }
        else {
            const lastLine = doc.lineAt(doc.lineCount - 1);
            await ed.edit((eb) => {
                eb.replace(new vscode.Range(0, 0, lastLine.range.end.line, lastLine.text.length), '');
            });
        }
    }
    async _getOrOpenEditor(fp) {
        const existing = vscode.window.visibleTextEditors.find((e) => e.document.uri.fsPath === fp);
        if (existing)
            return existing;
        const doc = await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
        return vscode.window.showTextDocument(doc, { preview: false });
    }
    _afterHunkAction(fp) {
        this._refreshDeco(fp);
        if (Object.keys(this._ms).length > 0) {
            this.nextChange();
        }
        else {
            this._checkAllDone();
        }
    }
    async _applyHunkAction(fp, idx, countProp, startProp) {
        const s = this._ms[fp];
        if (!s)
            return;
        const h = s.hunks[idx];
        if (h[countProp] > 0) {
            const ed = await this._getOrOpenEditor(fp);
            await this._delLines(ed, h[startProp], h[countProp]);
            const rm = h[countProp];
            s.hunks.splice(idx, 1);
            for (let i = idx; i < s.hunks.length; i++) {
                s.hunks[i].os -= rm;
                s.hunks[i].ns -= rm;
            }
        }
        else {
            s.hunks.splice(idx, 1);
        }
        if (!s.hunks.length)
            delete this._ms[fp];
        this.emit('hunkProcessed');
        this._afterHunkAction(fp);
    }
    async acceptChange(fp, idx) {
        const target = fp && idx !== undefined
            ? { fp, idx }
            : this._curHunk;
        if (!target || !this._ms[target.fp])
            return;
        await this._applyHunkAction(target.fp, target.idx, 'oc', 'os');
    }
    async rejectChange(fp, idx) {
        const target = fp && idx !== undefined
            ? { fp, idx }
            : this._curHunk;
        if (!target || !this._ms[target.fp])
            return;
        await this._applyHunkAction(target.fp, target.idx, 'nc', 'ns');
    }
    _hunkLine(h) {
        return h.nc > 0 ? h.ns : h.os;
    }
    prevChange() {
        this._navigateHunk(-1);
    }
    nextChange() {
        this._navigateHunk(1);
    }
    _navigateHunk(dir) {
        const allH = [];
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
            ? (a, b) => a < b
            : (a, b) => a > b;
        const start = dir < 0 ? allH.length - 1 : 0;
        const end = dir < 0 ? -1 : allH.length;
        const step = dir < 0 ? -1 : 1;
        let found = null;
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
        if (!found)
            found = allH[dir < 0 ? allH.length - 1 : 0];
        this._curHunk = {
            fp: found.fp,
            idx: this._ms[found.fp].hunks.indexOf(found.h),
        };
        vscode.workspace
            .openTextDocument(vscode.Uri.file(found.fp))
            .then((doc) => {
            vscode.window
                .showTextDocument(doc, { preview: false })
                .then((ed) => {
                const ln = this._hunkLine(found.h);
                ed.revealRange(new vscode.Range(ln, 0, ln, 0), vscode.TextEditorRevealType.InCenter);
                ed.selection = new vscode.Selection(ln, 0, ln, 0);
            });
        });
    }
    async acceptAll() {
        try {
            for (const fp of Object.keys(this._ms)) {
                const s = this._ms[fp];
                const ed = await this._getOrOpenEditor(fp);
                for (let i = s.hunks.length - 1; i >= 0; i--) {
                    if (s.hunks[i].oc > 0) {
                        await this._delLines(ed, s.hunks[i].os, s.hunks[i].oc);
                    }
                }
                ed.setDecorations(this._redDeco, []);
                ed.setDecorations(this._blueDeco, []);
            }
        }
        finally {
            this._ms = {};
            this._curHunk = null;
            await vscode.workspace.saveAll(false);
            vscode.window.showInformationMessage('All changes accepted.');
            this.emit('allDone');
        }
    }
    async rejectAll() {
        try {
            for (const fp of Object.keys(this._ms)) {
                const s = this._ms[fp];
                const ed = await this._getOrOpenEditor(fp);
                for (let i = s.hunks.length - 1; i >= 0; i--) {
                    if (s.hunks[i].nc > 0) {
                        await this._delLines(ed, s.hunks[i].ns, s.hunks[i].nc);
                    }
                }
                ed.setDecorations(this._redDeco, []);
                ed.setDecorations(this._blueDeco, []);
            }
        }
        finally {
            this._ms = {};
            this._curHunk = null;
            await vscode.workspace.saveAll(false);
            vscode.window.showInformationMessage('All changes rejected.');
            this.emit('allDone');
        }
    }
    _checkAllDone() {
        if (Object.keys(this._ms).length > 0)
            return;
        this._curHunk = null;
        vscode.workspace.saveAll(false).then(() => {
            vscode.window.showInformationMessage('All changes reviewed.');
            this.emit('allDone');
        }, () => {
            vscode.window.showInformationMessage('All changes reviewed.');
            this.emit('allDone');
        });
    }
    /**
     * Open merge view: insert old lines, apply decorations, navigate to first hunk.
     */
    async openMerge(data) {
        try {
            await vscode.workspace.saveAll(false);
        }
        catch { /* ignore */ }
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
                    await vscode.commands.executeCommand('workbench.action.files.revert');
                }
                catch { /* ignore */ }
            }
            let baseLines = [];
            try {
                baseLines = fs.readFileSync(f.base, 'utf8').split('\n');
            }
            catch { /* ignore */ }
            const hunks = (f.hunks || [])
                .map((h) => ({ cs: h.cs, cc: h.cc, bs: h.bs, bc: h.bc }))
                .sort((a, b) => a.cs - b.cs);
            let offset = 0;
            const processed = [];
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
                });
                offset += old.length;
            }
            this._ms[f.current] = { basePath: f.base, hunks: processed };
            this._refreshDeco(f.current);
            if (processed.length > 0) {
                ed.revealRange(new vscode.Range(processed[0].os, 0, processed[0].os, 0), vscode.TextEditorRevealType.InCenter);
            }
        }
        // Navigate to first hunk
        const firstFp = Object.keys(this._ms)[0];
        if (firstFp && this._ms[firstFp].hunks.length) {
            this._curHunk = { fp: firstFp, idx: 0 };
            const firstDoc = await vscode.workspace.openTextDocument(vscode.Uri.file(firstFp));
            const firstEd = await vscode.window.showTextDocument(firstDoc, {
                preview: false,
            });
            const fh = this._ms[firstFp].hunks[0];
            const fl = fh.nc > 0 ? fh.ns : fh.os;
            firstEd.revealRange(new vscode.Range(fl, 0, fl, 0), vscode.TextEditorRevealType.InCenter);
            firstEd.selection = new vscode.Selection(fl, 0, fl, 0);
        }
        else {
            this._curHunk = null;
        }
        const fileCount = (data.files || []).length;
        vscode.window.showInformationMessage(`Reviewing ${fileCount} file(s). Red = old, Blue = new. Use Accept / Reject.`);
    }
    dispose() {
        this._redDeco.dispose();
        this._blueDeco.dispose();
        for (const d of this._disposables) {
            d.dispose();
        }
        this._disposables = [];
        this.removeAllListeners();
    }
}
exports.MergeManager = MergeManager;
//# sourceMappingURL=MergeManager.js.map
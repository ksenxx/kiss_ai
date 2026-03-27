# Plan: Add "Accept File" and "Reject File" Buttons to Merge Diff Toolbar

## Goal

Add two new buttons ("Accept File" and "Reject File") to the merge diff toolbar, rearrange button order (Previous/Next before Accept/Reject), add explicit "Previous"/"Next" text labels, and arrange all buttons in 2 rows.

## Current State

The merge toolbar in `media/main.js` (`showMergeToolbar()`) renders a single row with:

1. **Accept** / **Reject** (single hunk)
1. **← / →** nav arrows (no text labels)
1. **Accept All** / **Reject All**

The `MergeManager.ts` has methods: `acceptChange`, `rejectChange`, `prevChange`, `nextChange`, `acceptAll`, `rejectAll`. It tracks per-file hunks in `_ms` (keyed by file path) and the current hunk in `_curHunk: { fp, idx }`.

## Changes

### 1. `MergeManager.ts` — Add `acceptFile`/`rejectFile` + extract shared helper

**Extract `_deleteFileHunks`** — a shared helper that processes all hunks for a single file in reverse order and deletes the relevant lines. This eliminates code duplication between the new `_resolveFile` and the existing `_resolveAll`:

```typescript
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
      await this._delLines(ed, s.hunks[i][startProp], s.hunks[i][countProp]);
    }
  }
}
```

**Refactor `_resolveAll`** to use the new helper:

```typescript
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
    // _refreshDeco sees fp missing from _ms and clears both decoration
    // arrays, removing stale red/blue highlights from all visible editors.
    for (const fp of fps) {
      this._refreshDeco(fp);
    }
    await vscode.workspace.saveAll(false);
    vscode.window.showInformationMessage(label);
    this.emit('allDone');
  }
}
```

**Add `acceptFile`, `rejectFile`, and `_resolveFile`:**

```typescript
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

private async _resolveFile(
  fp: string,
  countProp: 'oc' | 'nc',
  startProp: 'os' | 'ns'
): Promise<void> {
  await this._deleteFileHunks(fp, countProp, startProp);
  delete this._ms[fp];
  // Null out _curHunk before navigation — prevents stale hunk access
  // during the race window between _hunkOpInProgress release and
  // nextChange() completing asynchronously.
  this._curHunk = null;
  // Reuse _afterHunkAction: refreshes decorations on ALL visible editors
  // for this file (handles split views), then navigates to next hunk or
  // calls _checkAllDone if no hunks remain.
  this._afterHunkAction(fp);
}
```

### 2. `extension.ts` — Register new commands

Add `'acceptFile'` and `'rejectFile'` to the command registration loop:

```typescript
for (const cmd of ['acceptChange', 'rejectChange', 'prevChange', 'nextChange',
                    'acceptAll', 'rejectAll', 'acceptFile', 'rejectFile'] as const) {
```

### 3. `package.json` — Add command entries

Add two new command entries:

```json
{ "command": "kissSorcar.acceptFile", "title": "KISS: Accept File Changes" },
{ "command": "kissSorcar.rejectFile", "title": "KISS: Reject File Changes" }
```

### 4. `SorcarPanel.ts` — Handle new merge actions

Add to the `mergeAction` handler in `_handleMessage`:

```typescript
} else if (action === 'accept-file') {
  await this._mergeManager.acceptFile();
} else if (action === 'reject-file') {
  await this._mergeManager.rejectFile();
}
```

### 5. `media/main.js` — Rebuild toolbar layout (2 rows, reordered buttons)

Replace the `showMergeToolbar()` function's `bar.innerHTML` to arrange buttons in 2 rows:

**Row 1:** Previous | Next | Accept | Reject
**Row 2:** Accept File | Reject File | Accept All | Reject All

```javascript
bar.innerHTML =
  '<div class="merge-toolbar-header">'
  + '<span class="merge-toolbar-title">Review Changes</span>'
  + '<span class="merge-toolbar-hint">Red = old · Blue = new</span>'
  + '</div>'
  + '<div class="merge-toolbar-actions">'
  // Row 1: Navigation + single hunk actions
  + '<div class="merge-toolbar-row">'
  + '<div class="merge-toolbar-group">'
  + '<button class="merge-btn merge-nav" id="merge-prev-btn" data-tooltip="Previous change">' + svgUp + ' Previous</button>'
  + '<button class="merge-btn merge-nav" id="merge-next-btn" data-tooltip="Next change">' + svgDown + ' Next</button>'
  + '</div>'
  + '<div class="merge-toolbar-sep"></div>'
  + '<div class="merge-toolbar-group">'
  + '<button class="merge-btn merge-accept" id="merge-accept-btn" data-tooltip="Accept change">' + svgCheck + ' Accept</button>'
  + '<button class="merge-btn merge-reject" id="merge-reject-btn" data-tooltip="Reject change">' + svgX + ' Reject</button>'
  + '</div>'
  + '</div>'
  // Row 2: File-level + all actions
  + '<div class="merge-toolbar-row">'
  + '<div class="merge-toolbar-group">'
  + '<button class="merge-btn merge-accept-file" id="merge-accept-file-btn" data-tooltip="Accept all changes in current file">' + svgCheck + ' Accept File</button>'
  + '<button class="merge-btn merge-reject-file" id="merge-reject-file-btn" data-tooltip="Reject all changes in current file">' + svgX + ' Reject File</button>'
  + '</div>'
  + '<div class="merge-toolbar-sep"></div>'
  + '<div class="merge-toolbar-group">'
  + '<button class="merge-btn merge-accept-all" id="merge-accept-all-btn" data-tooltip="Accept all changes">' + svgCheckAll + ' Accept All</button>'
  + '<button class="merge-btn merge-reject-all" id="merge-reject-all-btn" data-tooltip="Reject all changes">' + svgXAll + ' Reject All</button>'
  + '</div>'
  + '</div>'
  + '</div>';
```

Add event listeners for the new buttons:

```javascript
document.getElementById('merge-accept-file-btn').addEventListener('click', function() {
  vscode.postMessage({ type: 'mergeAction', action: 'accept-file' });
});
document.getElementById('merge-reject-file-btn').addEventListener('click', function() {
  vscode.postMessage({ type: 'mergeAction', action: 'reject-file' });
});
```

### 6. `media/main.css` — Add row layout and new button styles

Add a `.merge-toolbar-row` class for 2-row layout:

```css
.merge-toolbar-row {
  display: flex; align-items: center; gap: 6px; width: 100%;
}
```

Change `.merge-toolbar-actions` to column layout:

```css
.merge-toolbar-actions {
  display: flex; flex-direction: column; gap: 6px;
}
```

Add styles for the new file-level buttons:

```css
.merge-btn.merge-accept-file {
  color: color-mix(in srgb, var(--green) 80%, var(--dim));
}
.merge-btn.merge-accept-file:hover {
  background: color-mix(in srgb, var(--green) 16%, transparent); color: var(--green);
}
.merge-btn.merge-reject-file {
  color: color-mix(in srgb, var(--red) 80%, var(--dim));
}
.merge-btn.merge-reject-file:hover {
  background: color-mix(in srgb, var(--red) 14%, transparent); color: var(--red);
}
```

### 7. `types.ts` — No changes needed

The `mergeAction` type already uses `action: string`, so `'accept-file'` and `'reject-file'` are covered.

## File Summary

| File | Change |
|------|--------|
| `src/MergeManager.ts` | Extract `_deleteFileHunks()` helper, refactor `_resolveAll()` to use it, add `acceptFile()`, `rejectFile()`, `_resolveFile()` |
| `src/extension.ts` | Register `acceptFile` and `rejectFile` commands |
| `package.json` | Add command entries for new commands |
| `src/SorcarPanel.ts` | Handle `'accept-file'` and `'reject-file'` merge actions |
| `media/main.js` | Rearrange toolbar: 2 rows, Previous/Next with text before Accept/Reject, add Accept File/Reject File buttons |
| `media/main.css` | Add `.merge-toolbar-row` layout, `.merge-accept-file` / `.merge-reject-file` styles, change `.merge-toolbar-actions` to column |

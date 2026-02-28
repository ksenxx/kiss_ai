"""Code-server setup, file scanning, and git diff/merge utilities."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

_CS_SETTINGS = {
    "workbench.startupEditor": "none",
    "workbench.tips.enabled": False,
    "workbench.welcomePage.walkthroughs.openOnInstall": False,
    "security.workspace.trust.enabled": False,
    "update.showReleaseNotes": False,
    "workbench.panel.defaultLocation": "bottom",
    "editor.fontSize": 13,
    "terminal.integrated.fontSize": 13,
    "scm.inputFontSize": 13,
    "debug.console.fontSize": 13,
    "window.restoreWindows": "all",
    "workbench.editor.restoreViewState": True,
    "files.hotExit": "onExitAndWindowClose",
}

_CS_STATE_ENTRIES = [
    ("workbench.activity.pinnedViewlets2", "[]"),
    ("workbench.welcomePage.walkthroughMetadata", "[]"),
    ("coderGettingStarted/v1", "installed"),
    ("workbench.panel.pinnedPanels", "[]"),
    ("memento/gettingStartedService", '{"installed":true}'),
    ("profileAssociations", '{"workspaces":{}}'),
    ("userDataProfiles", '[]'),
    ("welcomePage.gettingStartedTabs", '[]'),
    ("workbench.welcomePage.opened", "true"),
    ("chat.setupCompleted", "true"),
    ("chat.panelVisible", "false"),
    ("workbench.panel.chat.hidden", "true"),
    ("workbench.panel.chatSidebar.hidden", "true"),
]

_CS_EXTENSION_JS = """\
const vscode=require("vscode");
const fs=require("fs");
const path=require("path");
function activate(ctx){
  function cleanup(){
    for(const g of vscode.window.tabGroups.all){
      for(const t of g.tabs){
        if(!t.input||!t.input.uri){
          vscode.window.tabGroups.close(t).then(()=>{},()=>{});
        }
      }
    }
    vscode.commands.executeCommand('workbench.action.closePanel');
    vscode.commands.executeCommand('workbench.action.closeAuxiliaryBar');
  }
  cleanup();
  setTimeout(cleanup,1500);
  setTimeout(cleanup,4000);
  setTimeout(cleanup,8000);
  var home=process.env.HOME||process.env.USERPROFILE||'';
  function writeTheme(){
    var k=vscode.window.activeColorTheme.kind;
    var s=k===1?'light':k===3?'hcDark':k===4?'hcLight':'dark';
    try{
      var d=path.join(home,'.kiss');
      if(!fs.existsSync(d))fs.mkdirSync(d,{recursive:true});
      fs.writeFileSync(path.join(d,'vscode-theme.json'),JSON.stringify({kind:s}));
    }catch(e){}
  }
  writeTheme();
  ctx.subscriptions.push(vscode.window.onDidChangeActiveColorTheme(function(){writeTheme()}));
  var redDeco=vscode.window.createTextEditorDecorationType({
    backgroundColor:'rgba(248,81,73,0.15)',
    isWholeLine:true
  });
  var greenDeco=vscode.window.createTextEditorDecorationType({
    backgroundColor:'rgba(34,197,94,0.13)',
    isWholeLine:true
  });
  var ms={};
  var clFire=new vscode.EventEmitter();
  var nextSB=vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left,104);
  nextSB.text='$(arrow-right) Next';nextSB.command='kiss.nextChange';
  var acceptAllSB=vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left,103);
  acceptAllSB.text='$(check-all) Accept All';acceptAllSB.command='kiss.acceptAll';
  var rejectAllSB=vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left,102);
  rejectAllSB.text='$(close-all) Reject All';rejectAllSB.command='kiss.rejectAll';
  var commitSB=vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left,101);
  commitSB.text='$(git-commit) Commit';commitSB.command='kiss.commitChanges';
  ctx.subscriptions.push(nextSB,acceptAllSB,rejectAllSB,commitSB);
  function showMergeButtons(v){
    [nextSB,acceptAllSB,rejectAllSB,commitSB].forEach(function(b){v?b.show():b.hide()});
  }
  ctx.subscriptions.push(vscode.languages.registerCodeLensProvider({scheme:'file'},{
    onDidChangeCodeLenses:clFire.event,
    provideCodeLenses:function(doc){
      var s=ms[doc.uri.fsPath];
      if(!s||!s.hunks.length)return[];
      var L=[];
      for(var i=0;i<s.hunks.length;i++){
        var h=s.hunks[i];
        var endLn=h.nc>0?h.ns+h.nc:h.os+h.oc;
        var ln=Math.min(endLn,doc.lineCount-1);
        var r=new vscode.Range(ln,0,ln,0);
        var fp=doc.uri.fsPath;
        L.push(new vscode.CodeLens(r,{title:'\\u2705 Accept',
          command:'kiss.acceptChange',arguments:[fp,i]}));
        L.push(new vscode.CodeLens(r,{title:'\\u274c Reject',
          command:'kiss.rejectChange',arguments:[fp,i]}));
      }
      return L;
    }
  }));
  function refreshDeco(fp){
    vscode.window.visibleTextEditors.forEach(function(ed){
      if(ed.document.uri.fsPath!==fp)return;
      var s=ms[fp],reds=[],greens=[];
      if(s)s.hunks.forEach(function(h){
        if(h.oc>0)reds.push(new vscode.Range(h.os,0,h.os+h.oc-1,99999));
        if(h.nc>0)greens.push(new vscode.Range(h.ns,0,h.ns+h.nc-1,99999));
      });
      ed.setDecorations(redDeco,reds);
      ed.setDecorations(greenDeco,greens);
    });
  }
  async function delLines(ed,start,count){
    if(count<=0)return;
    var end=start+count;
    if(end<ed.document.lineCount){
      await ed.edit(function(eb){eb.delete(new vscode.Range(start,0,end,0));});
    }else if(start>0){
      var p=ed.document.lineAt(start-1),l=ed.document.lineAt(ed.document.lineCount-1);
      await ed.edit(function(eb){eb.delete(new vscode.Range(
        start-1,p.text.length,l.range.end.line,l.text.length));});
    }else{
      var ll=ed.document.lineAt(ed.document.lineCount-1);
      await ed.edit(function(eb){eb.replace(new vscode.Range(
        0,0,ll.range.end.line,ll.text.length),'');});
    }
  }
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.acceptChange',async function(fp,idx){
    var s=ms[fp];if(!s)return;
    var h=s.hunks[idx];
    if(h.oc>0){
      var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
      if(!ed)return;
      await delLines(ed,h.os,h.oc);
      var rm=h.oc;
      s.hunks.splice(idx,1);
      for(var i=idx;i<s.hunks.length;i++){s.hunks[i].os-=rm;s.hunks[i].ns-=rm;}
    }else{s.hunks.splice(idx,1);}
    if(!s.hunks.length)delete ms[fp];
    refreshDeco(fp);clFire.fire();checkAllDone();
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.rejectChange',async function(fp,idx){
    var s=ms[fp];if(!s)return;
    var h=s.hunks[idx];
    if(h.nc>0){
      var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
      if(!ed)return;
      await delLines(ed,h.ns,h.nc);
      var rm=h.nc;
      s.hunks.splice(idx,1);
      for(var i=idx;i<s.hunks.length;i++){s.hunks[i].os-=rm;s.hunks[i].ns-=rm;}
    }else{s.hunks.splice(idx,1);}
    if(!s.hunks.length)delete ms[fp];
    refreshDeco(fp);clFire.fire();checkAllDone();
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.nextChange',function(){
    var allH=[];
    for(var fp in ms)ms[fp].hunks.forEach(function(h){allH.push({fp:fp,h:h})});
    if(!allH.length)return;
    var ae=vscode.window.activeTextEditor;
    var cf=ae?ae.document.uri.fsPath:'',cl=ae?ae.selection.active.line:-1;
    var found=null;
    for(var j=0;j<allH.length;j++){
      var ln=allH[j].h.nc>0?allH[j].h.ns:allH[j].h.os;
      if(allH[j].fp===cf&&ln>cl){found=allH[j];break;}
    }
    if(!found)for(var j=0;j<allH.length;j++){
      if(allH[j].fp!==cf){found=allH[j];break;}
    }
    if(!found)found=allH[0];
    vscode.workspace.openTextDocument(vscode.Uri.file(found.fp)).then(function(doc){
      vscode.window.showTextDocument(doc,{preview:false}).then(function(ed){
        var ln=found.h.nc>0?found.h.ns:found.h.os;
        ed.revealRange(new vscode.Range(ln,0,ln,0),vscode.TextEditorRevealType.InCenter);
        ed.selection=new vscode.Selection(ln,0,ln,0);
      });
    });
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.acceptAll',async function(){
    for(var fp of Object.keys(ms)){
      var s=ms[fp];
      var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
      if(!ed){
        var doc=await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
        ed=await vscode.window.showTextDocument(doc,{preview:false});
      }
      for(var i=s.hunks.length-1;i>=0;i--){
        if(s.hunks[i].oc>0)await delLines(ed,s.hunks[i].os,s.hunks[i].oc);
      }
      ed.setDecorations(redDeco,[]);ed.setDecorations(greenDeco,[]);
    }
    ms={};clFire.fire();
    await vscode.workspace.saveAll(false);
    showMergeButtons(false);
    vscode.window.showInformationMessage('All changes accepted.');
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.rejectAll',async function(){
    for(var fp of Object.keys(ms)){
      var s=ms[fp];
      var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
      if(!ed){
        var doc=await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
        ed=await vscode.window.showTextDocument(doc,{preview:false});
      }
      for(var i=s.hunks.length-1;i>=0;i--){
        if(s.hunks[i].nc>0)await delLines(ed,s.hunks[i].ns,s.hunks[i].nc);
      }
      ed.setDecorations(redDeco,[]);ed.setDecorations(greenDeco,[]);
    }
    ms={};clFire.fire();
    await vscode.workspace.saveAll(false);
    showMergeButtons(false);
    vscode.window.showInformationMessage('All changes rejected.');
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.commitChanges',async function(){
    var portFile=path.join(home,'.kiss','assistant-port');
    var port='';
    try{port=fs.readFileSync(portFile,'utf8').trim();}catch(e){}
    if(!port){vscode.window.showErrorMessage('Assistant server not found');return;}
    commitSB.text='$(loading~spin) Committing...';
    try{
      var http=require('http');
      var body=await new Promise(function(resolve,reject){
        var req=http.request({hostname:'127.0.0.1',port:parseInt(port),path:'/commit',method:'POST',
          headers:{'Content-Type':'application/json'}},function(res){
          var d='';res.on('data',function(c){d+=c});
          res.on('end',function(){resolve(JSON.parse(d))});
        });
        req.on('error',reject);
        req.write('{}');req.end();
      });
      if(body.error)vscode.window.showErrorMessage('Commit failed: '+body.error);
      else vscode.window.showInformationMessage('Committed: '+body.message);
    }catch(e){vscode.window.showErrorMessage('Commit error: '+e.message);}
    commitSB.text='$(git-commit) Commit';
  }));
  function checkAllDone(){
    if(Object.keys(ms).length>0)return;
    vscode.workspace.saveAll(false).then(function(){
      vscode.window.showInformationMessage('All changes reviewed.');
      showMergeButtons(false);
    });
  }
  ctx.subscriptions.push(vscode.window.onDidChangeVisibleTextEditors(function(){
    for(var fp in ms)refreshDeco(fp);
  }));
  var mp=path.join(home,'.kiss','code-server-data','pending-merge.json');
  var op=path.join(home,'.kiss','code-server-data','pending-open.json');
  var ap=path.join(home,'.kiss','code-server-data','pending-action.json');
  var iv=setInterval(function(){
    try{
      if(fs.existsSync(op)){
        var od=JSON.parse(fs.readFileSync(op,'utf8'));
        fs.unlinkSync(op);
        var uri=vscode.Uri.file(od.path);
        vscode.workspace.openTextDocument(uri).then(function(doc){
          vscode.window.showTextDocument(doc,{preview:false});
        });
      }
      if(fs.existsSync(ap)){
        var ad=JSON.parse(fs.readFileSync(ap,'utf8'));
        fs.unlinkSync(ap);
        if(ad.action==='next')vscode.commands.executeCommand('kiss.nextChange');
        else if(ad.action==='accept-all')vscode.commands.executeCommand('kiss.acceptAll');
        else if(ad.action==='reject-all')vscode.commands.executeCommand('kiss.rejectAll');
      }
      if(!fs.existsSync(mp))return;
      var data=JSON.parse(fs.readFileSync(mp,'utf8'));
      fs.unlinkSync(mp);
      openMerge(data);
    }catch(e){}
  },800);
  ctx.subscriptions.push({dispose:function(){clearInterval(iv)}});
  async function openMerge(data){
    for(var fp in ms){
      vscode.window.visibleTextEditors.forEach(function(ed){
        if(ed.document.uri.fsPath===fp){
          ed.setDecorations(redDeco,[]);
          ed.setDecorations(greenDeco,[]);
        }
      });
    }
    ms={};
    for(var f of(data.files||[])){
      var currentUri=vscode.Uri.file(f.current);
      var doc=await vscode.workspace.openTextDocument(currentUri);
      var ed=await vscode.window.showTextDocument(doc,{preview:false});
      var baseLines=fs.readFileSync(f.base,'utf8').split('\\n');
      var hunks=(f.hunks||[]).map(function(h){
        return{cs:h.cs,cc:h.cc,bs:h.bs,bc:h.bc};
      });
      hunks.sort(function(a,b){return a.cs-b.cs});
      var offset=0,processed=[];
      for(var i=0;i<hunks.length;i++){
        var h=hunks[i];
        var old=h.bc>0?baseLines.slice(h.bs,h.bs+h.bc):[];
        if(old.length>0){
          var il=h.cs+offset;
          var txt=old.join('\\n')+'\\n';
          await ed.edit(function(eb){eb.insert(new vscode.Position(il,0),txt);});
        }
        processed.push({os:h.cs+offset,oc:old.length,ns:h.cs+offset+old.length,nc:h.cc});
        offset+=old.length;
      }
      ms[f.current]={basePath:f.base,hunks:processed};
      refreshDeco(f.current);
      if(processed.length>0){
        ed.revealRange(new vscode.Range(processed[0].os,0,processed[0].os,0),
          vscode.TextEditorRevealType.InCenter);
      }
    }
    clFire.fire();
    showMergeButtons(true);
    vscode.window.showInformationMessage(
      'Reviewing '+data.files.length+' file(s). '
      +'Red = old, Green = new. Use \\u2705 Accept / \\u274c Reject on each change.');
  }
}
module.exports={activate};
"""


def _setup_code_server(data_dir: str) -> bool:
    """Pre-configure code-server user data: settings, state DB, and cleanup extension.

    Returns True if the extension.js was updated (code-server needs restart).
    """
    user_dir = Path(data_dir) / "User"
    user_dir.mkdir(parents=True, exist_ok=True)

    settings_file = user_dir / "settings.json"
    try:
        existing = json.loads(settings_file.read_text()) if settings_file.exists() else {}
    except (json.JSONDecodeError, OSError):
        existing = {}
    if "workbench.colorTheme" not in existing:
        existing["workbench.colorTheme"] = "Default Dark Modern"
    existing.update(_CS_SETTINGS)
    settings_file.write_text(json.dumps(existing, indent=2))

    state_db = user_dir / "globalStorage" / "state.vscdb"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(state_db)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ItemTable"
            " (key TEXT UNIQUE ON CONFLICT REPLACE, value TEXT)"
        )
        for key, value in _CS_STATE_ENTRIES:
            conn.execute(
                "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)", (key, value),
            )
        conn.commit()

    ws_storage = user_dir / "workspaceStorage"
    if ws_storage.exists():
        for ws_dir in ws_storage.iterdir():
            for sub in ("chatSessions", "chatEditingSessions"):
                chat_dir = ws_dir / sub
                if chat_dir.exists():
                    shutil.rmtree(chat_dir, ignore_errors=True)

    ext_dir = Path(data_dir) / "extensions" / "kiss-init"
    ext_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "package.json").write_text(json.dumps({
        "name": "kiss-init", "version": "0.0.1", "publisher": "kiss",
        "engines": {"vscode": "^1.80.0"},
        "activationEvents": ["onStartupFinished"],
        "main": "./extension.js",
        "contributes": {
            "commands": [
                {"command": "kiss.acceptChange", "title": "Accept Change"},
                {"command": "kiss.rejectChange", "title": "Reject Change"},
                {"command": "kiss.nextChange", "title": "Next Change"},
                {"command": "kiss.acceptAll", "title": "Accept All Changes"},
                {"command": "kiss.rejectAll", "title": "Reject All Changes"},
                {"command": "kiss.commitChanges", "title": "Commit Changes"},
            ],
        },
    }))
    ext_file = ext_dir / "extension.js"
    old_content = ext_file.read_text() if ext_file.exists() else ""
    ext_file.write_text(_CS_EXTENSION_JS)
    return old_content != _CS_EXTENSION_JS


def _scan_files(work_dir: str) -> list[str]:
    paths: list[str] = []
    skip = {
        ".git", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    }
    try:
        for root, dirs, files in os.walk(work_dir):
            depth = os.path.relpath(root, work_dir).count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
            for d in dirs:
                paths.append(os.path.relpath(os.path.join(root, d), work_dir) + "/")
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 2000:
                    return paths
    except OSError:
        pass
    return paths


def _parse_diff_hunks(work_dir: str) -> dict[str, list[tuple[int, int, int, int]]]:
    result = subprocess.run(
        ["git", "diff", "-U0", "HEAD", "--no-color"],
        capture_output=True, text=True, cwd=work_dir,
    )
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        dm = re.match(r"^diff --git a/.* b/(.*)", line)
        if dm:
            current_file = dm.group(1)
            continue
        hm = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if hm and current_file:
            hunks.setdefault(current_file, []).append((
                int(hm.group(1)),
                int(hm.group(2)) if hm.group(2) is not None else 1,
                int(hm.group(3)),
                int(hm.group(4)) if hm.group(4) is not None else 1,
            ))
    return hunks


def _capture_untracked(work_dir: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True, text=True, cwd=work_dir,
    )
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


def _prepare_merge_view(
    work_dir: str,
    data_dir: str,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    pre_untracked: set[str],
) -> dict[str, Any]:
    post_hunks = _parse_diff_hunks(work_dir)
    file_hunks: dict[str, list[dict[str, int]]] = {}
    for fname, hunks in post_hunks.items():
        pre = {(bs, bc) for bs, bc, _, _ in pre_hunks.get(fname, [])}
        filtered = [
            {"bs": bs - 1, "bc": bc, "cs": cs - 1, "cc": cc}
            for bs, bc, cs, cc in hunks
            if (bs, bc) not in pre
        ]
        if filtered:
            file_hunks[fname] = filtered
    new_files = _capture_untracked(work_dir) - pre_untracked
    for fname in new_files:
        fpath = Path(work_dir) / fname
        try:
            if not fpath.is_file() or fpath.stat().st_size > 2_000_000:
                continue
            line_count = len(fpath.read_text().splitlines())
            if line_count:
                file_hunks[fname] = [{"bs": 0, "bc": 0, "cs": 0, "cc": line_count}]
        except (OSError, UnicodeDecodeError):
            pass
    if not file_hunks:
        return {"error": "No changes"}
    merge_dir = Path(data_dir) / "merge-temp"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    manifest_files = []
    for fname, fh in file_hunks.items():
        current_path = Path(work_dir) / fname
        base_path = merge_dir / fname
        base_path.parent.mkdir(parents=True, exist_ok=True)
        base_result = subprocess.run(
            ["git", "show", f"HEAD:{fname}"],
            capture_output=True, text=True, cwd=work_dir,
        )
        base_path.write_text(base_result.stdout if base_result.returncode == 0 else "")
        manifest_files.append({
            "name": fname,
            "base": str(base_path),
            "current": str(current_path),
            "hunks": fh,
        })
    manifest = Path(data_dir) / "pending-merge.json"
    manifest.write_text(json.dumps({
        "branch": "HEAD", "files": manifest_files,
    }))
    return {"status": "opened", "count": len(manifest_files)}

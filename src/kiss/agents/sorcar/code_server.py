"""Code-server setup, file scanning, and git diff/merge utilities."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Any

from kiss.core import config as config_module

logger = logging.getLogger(__name__)

_CS_SETTINGS = {
    "workbench.startupEditor": "none",
    "workbench.tips.enabled": False,
    "workbench.welcomePage.walkthroughs.openOnInstall": False,
    "security.workspace.trust.enabled": False,
    "update.showReleaseNotes": False,
    "workbench.panel.defaultLocation": "bottom",
    "editor.fontSize": None,
    "terminal.integrated.fontSize": 13,
    "scm.inputFontSize": 13,
    "debug.console.fontSize": 13,
    "window.restoreWindows": "all",
    "workbench.editor.restoreViewState": True,
    "files.hotExit": "onExitAndWindowClose",
    "git.repositoryScanMaxDepth": 1,
    "git.autoRepositoryDetection": True,
    "git.openRepositoryInParentFolders": "always",
    "files.saveConflictResolution": "overwriteFileOnDisk",
    "github.copilot.enable": {"*": True},
    "github.copilot.editor.enableAutoCompletions": True,
}

_CS_STATE_ENTRIES = [
    ("workbench.activity.pinnedViewlets2", "[]"),
    ("workbench.welcomePage.walkthroughMetadata", "[]"),
    ("coderGettingStarted/v1", "installed"),
    ("workbench.panel.pinnedPanels", "[]"),
    ("memento/gettingStartedService", '{"installed":true}'),
    ("profileAssociations", '{"workspaces":{}}'),
    ("userDataProfiles", "[]"),
    ("welcomePage.gettingStartedTabs", "[]"),
    ("workbench.welcomePage.opened", "true"),
    ("chat.setupCompleted", "true"),
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
  var dataDir=path.resolve(ctx.globalStorageUri.fsPath,'..','..','..');
  var editorStateFile=path.join(dataDir,'editor-state.json');
  function saveEditorState(){
    try{
      var tabs=[];
      for(var g of vscode.window.tabGroups.all){
        for(var t of g.tabs){
          if(t.input&&t.input.uri&&t.input.uri.scheme==='file'){
            tabs.push({path:t.input.uri.fsPath,viewColumn:g.viewColumn,isActive:t.isActive});
          }
        }
      }
      var ae=vscode.window.activeTextEditor;
      var cursors={};
      for(var ed of vscode.window.visibleTextEditors){
        if(ed.document.uri.scheme==='file'){
          cursors[ed.document.uri.fsPath]={
            line:ed.selection.active.line,
            character:ed.selection.active.character
          };
        }
      }
      var st={tabs:tabs,activeFile:ae&&ae.document?ae.document.uri.fsPath:'',cursors:cursors};
      if(!fs.existsSync(dataDir))fs.mkdirSync(dataDir,{recursive:true});
      fs.writeFileSync(editorStateFile,JSON.stringify(st));
    }catch(e){}
  }
  async function restoreEditorState(){
    try{
      if(!fs.existsSync(editorStateFile))return;
      var st=JSON.parse(fs.readFileSync(editorStateFile,'utf8'));
      if(!st.tabs||!st.tabs.length)return;
      var currentPaths=new Set();
      for(var g of vscode.window.tabGroups.all){
        for(var t of g.tabs){
          if(t.input&&t.input.uri)currentPaths.add(t.input.uri.fsPath);
        }
      }
      for(var tab of st.tabs){
        if(!currentPaths.has(tab.path)&&fs.existsSync(tab.path)){
          try{
            var doc=await vscode.workspace.openTextDocument(vscode.Uri.file(tab.path));
            var opts={preview:false,viewColumn:tab.viewColumn||1,preserveFocus:true};
            await vscode.window.showTextDocument(doc,opts);
          }catch(e){}
        }
      }
      if(st.activeFile&&fs.existsSync(st.activeFile)){
        try{
          var doc=await vscode.workspace.openTextDocument(vscode.Uri.file(st.activeFile));
          var ed=await vscode.window.showTextDocument(doc,{preview:false});
          var c=st.cursors&&st.cursors[st.activeFile];
          if(c){
            var pos=new vscode.Position(c.line,c.character);
            ed.selection=new vscode.Selection(pos,pos);
            ed.revealRange(new vscode.Range(pos,pos),vscode.TextEditorRevealType.InCenter);
          }
        }catch(e){}
      }
    }catch(e){}
  }
  setTimeout(function(){restoreEditorState();},3000);
  var saveTimer=null;
  function debouncedSaveState(){
    if(saveTimer)clearTimeout(saveTimer);
    saveTimer=setTimeout(saveEditorState,500);
  }
  ctx.subscriptions.push(vscode.window.tabGroups.onDidChangeTabs(function(){debouncedSaveState();}));
  var saveStateInterval=setInterval(saveEditorState,30000);
  ctx.subscriptions.push({dispose:function(){clearInterval(saveStateInterval);}});
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
  var blueDeco=vscode.window.createTextEditorDecorationType({
    backgroundColor:'rgba(59,130,246,0.15)',
    isWholeLine:true
  });
  var ms={};
  var curHunk=null;
  function refreshDeco(fp){
    vscode.window.visibleTextEditors.forEach(function(ed){
      if(ed.document.uri.fsPath!==fp)return;
      var s=ms[fp],reds=[],blues=[];
      if(s)s.hunks.forEach(function(h){
        if(h.oc>0)reds.push(new vscode.Range(h.os,0,h.os+h.oc-1,99999));
        if(h.nc>0)blues.push(new vscode.Range(h.ns,0,h.ns+h.nc-1,99999));
      });
      ed.setDecorations(redDeco,reds);
      ed.setDecorations(blueDeco,blues);
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
  function afterHunkAction(fp){
    refreshDeco(fp);
    if(Object.keys(ms).length>0)vscode.commands.executeCommand('kiss.nextChange');
    else checkAllDone();
  }
  async function getOrOpenEditor(fp){
    var ed=vscode.window.visibleTextEditors.find(function(e){return e.document.uri.fsPath===fp;});
    if(!ed){
      var doc=await vscode.workspace.openTextDocument(vscode.Uri.file(fp));
      ed=await vscode.window.showTextDocument(doc,{preview:false});
    }
    return ed;
  }
  async function applyHunkAction(fp,idx,countProp,startProp){
    var s=ms[fp];if(!s)return;
    var h=s.hunks[idx];
    if(h[countProp]>0){
      var ed=await getOrOpenEditor(fp);
      await delLines(ed,h[startProp],h[countProp]);
      var rm=h[countProp];
      s.hunks.splice(idx,1);
      for(var i=idx;i<s.hunks.length;i++){s.hunks[i].os-=rm;s.hunks[i].ns-=rm;}
    }else{s.hunks.splice(idx,1);}
    if(!s.hunks.length)delete ms[fp];
    afterHunkAction(fp);
  }
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.acceptChange',async function(fp,idx){
    await applyHunkAction(fp,idx,'oc','os');
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.rejectChange',async function(fp,idx){
    await applyHunkAction(fp,idx,'nc','ns');
  }));
  function hunkLine(h){return h.nc>0?h.ns:h.os;}
  function navigateHunk(dir){
    var allH=[];
    for(var fp in ms)ms[fp].hunks.forEach(function(h){allH.push({fp:fp,h:h})});
    if(!allH.length){curHunk=null;return;}
    var ae=vscode.window.activeTextEditor;
    var cf=ae?ae.document.uri.fsPath:'',cl=ae?ae.selection.active.line:(dir<0?999999:-1);
    var found=null,cmp=dir<0?function(a,b){return a<b}:function(a,b){return a>b};
    var start=dir<0?allH.length-1:0,end=dir<0?-1:allH.length,step=dir<0?-1:1;
    for(var j=start;j!==end;j+=step){
      var ln=hunkLine(allH[j].h);
      if(allH[j].fp===cf&&cmp(ln,cl)){found=allH[j];break;}
    }
    if(!found)for(var j=start;j!==end;j+=step){
      if(allH[j].fp!==cf){found=allH[j];break;}
    }
    if(!found)found=allH[dir<0?allH.length-1:0];
    curHunk={fp:found.fp,idx:ms[found.fp].hunks.indexOf(found.h)};
    vscode.workspace.openTextDocument(vscode.Uri.file(found.fp)).then(function(doc){
      vscode.window.showTextDocument(doc,{preview:false}).then(function(ed){
        var ln=hunkLine(found.h);
        ed.revealRange(new vscode.Range(ln,0,ln,0),vscode.TextEditorRevealType.InCenter);
        ed.selection=new vscode.Selection(ln,0,ln,0);
      });
    });
  }
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.prevChange',function(){navigateHunk(-1)}));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.nextChange',function(){navigateHunk(1)}));
  async function applyAll(countProp,startProp,msg){
    for(var fp of Object.keys(ms)){
      var s=ms[fp];
      var ed=await getOrOpenEditor(fp);
      for(var i=s.hunks.length-1;i>=0;i--){
        if(s.hunks[i][countProp]>0)await delLines(ed,s.hunks[i][startProp],s.hunks[i][countProp]);
      }
      ed.setDecorations(redDeco,[]);ed.setDecorations(blueDeco,[]);
    }
    ms={};curHunk=null;
    await vscode.workspace.saveAll(false);
    vscode.window.showInformationMessage(msg);
    firePost('/merge-action',{action:'all-done'});
  }
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.acceptAll',async function(){
    await applyAll('oc','os','All changes accepted.');
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.rejectAll',async function(){
    await applyAll('nc','ns','All changes rejected.');
  }));
  function readPort(){
    try{return fs.readFileSync(path.join(dataDir,'assistant-port'),'utf8').trim();}
    catch(e){return '';}
  }
  function postAssistant(p,body){
    var http=require('http');
    return new Promise(function(resolve,reject){
      var req=http.request({hostname:'127.0.0.1',port:parseInt(readPort()),
        path:p,method:'POST',headers:{'Content-Type':'application/json'},
        timeout:60000},function(res){
        var d='';res.on('data',function(c){d+=c});
        res.on('end',function(){
          try{resolve(JSON.parse(d))}
          catch(e){reject(new Error('Bad response: '+d.substring(0,100)))}
        });
      });
      req.on('timeout',function(){req.destroy();reject(new Error('Timed out'))});
      req.on('error',reject);
      req.write(JSON.stringify(body||{}));req.end();
    });
  }
  function firePost(p,body){
    var port=readPort();if(!port)return;
    var http=require('http');
    var req=http.request({hostname:'127.0.0.1',port:parseInt(port),
      path:p,method:'POST',headers:{'Content-Type':'application/json'}},function(){});
    req.on('error',function(){});
    req.write(JSON.stringify(body||{}));req.end();
  }
  ctx.subscriptions.push(vscode.commands.registerCommand(
    'kiss.generateCommitMessage',async function(){
    if(!readPort()){vscode.window.showErrorMessage('Assistant server not found');return;}
    var gitExt=vscode.extensions.getExtension('vscode.git');
    if(!gitExt){vscode.window.showErrorMessage('Git extension not found');return;}
    var git=gitExt.exports.getAPI(1);
    if(!git.repositories.length){vscode.window.showErrorMessage('No git repository found');return;}
    git.repositories[0].inputBox.value='Generating commit message...';
    try{
      var body=await postAssistant('/generate-commit-message',{});
      if(body.error){
        git.repositories[0].inputBox.value='';
        vscode.window.showErrorMessage('Generate failed: '+body.error);
      }else{
        git.repositories[0].inputBox.value=body.message;
        vscode.commands.executeCommand('workbench.view.scm');
      }
    }catch(e){
      git.repositories[0].inputBox.value='';
      vscode.window.showErrorMessage('Generate error: '+e.message);
    }
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.commitChanges',async function(){
    if(!readPort()){vscode.window.showErrorMessage('Assistant server not found');return;}
    try{
      var body=await postAssistant('/commit',{});
      if(body.error)vscode.window.showErrorMessage('Commit failed: '+body.error);
      else vscode.window.showInformationMessage('Committed: '+body.message);
    }catch(e){vscode.window.showErrorMessage('Commit error: '+e.message);}
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.toggleFocus',function(){
    firePost('/focus-chatbox',{});
  }));
  ctx.subscriptions.push(vscode.commands.registerCommand('kiss.runSelection',function(){
    var ed=vscode.window.activeTextEditor;
    if(!ed)return;
    var sel=ed.document.getText(ed.selection);
    if(!sel||!sel.trim()){vscode.window.showInformationMessage('No text selected');return;}
    if(!readPort()){vscode.window.showErrorMessage('Assistant server not found');return;}
    postAssistant('/run-selection',{text:sel.trim()}).then(function(body){
      if(body.error)vscode.window.showErrorMessage('Run selection failed: '+body.error);
    }).catch(function(e){
      vscode.window.showErrorMessage('Run selection error: '+e.message);
    });
  }));
  function checkAllDone(){
    if(Object.keys(ms).length>0)return;
    curHunk=null;
    function notifyDone(){
      vscode.window.showInformationMessage('All changes reviewed.');
      firePost('/merge-action',{action:'all-done'});
    }
    vscode.workspace.saveAll(false).then(notifyDone,notifyDone);
  }
  ctx.subscriptions.push(vscode.window.onDidChangeVisibleTextEditors(function(){
    for(var fp in ms)refreshDeco(fp);
  }));
  function writeActiveFile(){
    var ed=vscode.window.activeTextEditor;
    var fp=ed&&ed.document?ed.document.uri.fsPath:'';
    try{
      if(!fs.existsSync(dataDir))fs.mkdirSync(dataDir,{recursive:true});
      fs.writeFileSync(path.join(dataDir,'active-file.json'),JSON.stringify({path:fp}));
    }catch(e){}
  }
  writeActiveFile();
  ctx.subscriptions.push(vscode.window.onDidChangeActiveTextEditor(function(){writeActiveFile();debouncedSaveState();}));
  var mp=path.join(dataDir,'pending-merge.json');
  var op=path.join(dataDir,'pending-open.json');
  var ap=path.join(dataDir,'pending-action.json');
  var sp=path.join(dataDir,'pending-scm-message.json');
  var iv=setInterval(function(){
    try{
      var fep=path.join(dataDir,'pending-focus-editor.json');
      if(fs.existsSync(fep)){
        fs.unlinkSync(fep);
        vscode.commands.executeCommand('workbench.action.focusActiveEditorGroup');
      }
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
        if(ad.action==='prev')vscode.commands.executeCommand('kiss.prevChange');
        else if(ad.action==='next')vscode.commands.executeCommand('kiss.nextChange');
        else if(ad.action==='accept-all')vscode.commands.executeCommand('kiss.acceptAll');
        else if(ad.action==='reject-all')vscode.commands.executeCommand('kiss.rejectAll');
        else if(ad.action==='accept'){
          if(curHunk&&ms[curHunk.fp])vscode.commands.executeCommand('kiss.acceptChange',curHunk.fp,curHunk.idx);
        }
        else if(ad.action==='reject'){
          if(curHunk&&ms[curHunk.fp])vscode.commands.executeCommand('kiss.rejectChange',curHunk.fp,curHunk.idx);
        }
      }
      if(fs.existsSync(sp)){
        var sd=JSON.parse(fs.readFileSync(sp,'utf8'));
        fs.unlinkSync(sp);
        var gitExt=vscode.extensions.getExtension('vscode.git');
        if(gitExt){
          var git=gitExt.exports.getAPI(1);
          if(git.repositories.length>0){
            git.repositories[0].inputBox.value=sd.message;
            vscode.commands.executeCommand('workbench.view.scm');
          }
        }
      }
      if(!fs.existsSync(mp))return;
      var data=JSON.parse(fs.readFileSync(mp,'utf8'));
      fs.unlinkSync(mp);
      openMerge(data).catch(function(e){
        console.error('openMerge failed:',e);
        vscode.window.showErrorMessage('Merge view setup failed: '+e.message);
      });
    }catch(e){}
  },800);
  ctx.subscriptions.push({dispose:function(){clearInterval(iv)}});
  async function openMerge(data){
    try{await vscode.workspace.saveAll(false);}catch(e){}
    for(var fp in ms){
      vscode.window.visibleTextEditors.forEach(function(ed){
        if(ed.document.uri.fsPath===fp){
          ed.setDecorations(redDeco,[]);
          ed.setDecorations(blueDeco,[]);
        }
      });
    }
    ms={};
    for(var f of(data.files||[])){
      var currentUri=vscode.Uri.file(f.current);
      var doc=await vscode.workspace.openTextDocument(currentUri);
      var ed=await vscode.window.showTextDocument(doc,{preview:false});
      if(doc.isDirty){try{await vscode.commands.executeCommand(
        'workbench.action.files.revert');}catch(e){}}
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
    var firstFp=Object.keys(ms)[0];
    if(firstFp&&ms[firstFp].hunks.length){
      curHunk={fp:firstFp,idx:0};
      var firstDoc=await vscode.workspace.openTextDocument(vscode.Uri.file(firstFp));
      var firstEd=await vscode.window.showTextDocument(firstDoc,{preview:false});
      var fh=ms[firstFp].hunks[0];
      var fl=fh.nc>0?fh.ns:fh.os;
      firstEd.revealRange(new vscode.Range(fl,0,fl,0),vscode.TextEditorRevealType.InCenter);
      firstEd.selection=new vscode.Selection(fl,0,fl,0);
    }
    else curHunk=null;
    vscode.window.showInformationMessage(
      'Reviewing '+data.files.length+' file(s). '
      +'Red = old, Blue = new. Use Accept / Reject on toolbar.');
  }
}
module.exports={activate};
"""


_MS_GALLERY = (
    '{"serviceUrl":"https://marketplace.visualstudio.com/_apis/public/gallery",'
    '"itemUrl":"https://marketplace.visualstudio.com/items"}'
)


def _disable_copilot_scm_button(data_dir: str) -> None:
    """Remove Copilot's 'generate commit message' button from the SCM input box.

    Modifies the Copilot Chat extension's package.json to set the scm/inputBox
    menu entry's ``when`` clause to ``"false"``, preventing it from appearing.
    This ensures only the KISSAgent's generate button is visible.
    """
    ext_base = Path(data_dir) / "extensions"
    if not ext_base.is_dir():
        return
    for ext_dir in ext_base.iterdir():
        if not ext_dir.is_dir() or not ext_dir.name.startswith("github.copilot-chat-"):
            continue
        pkg_path = ext_dir / "package.json"
        if not pkg_path.exists():
            continue
        try:
            pkg = json.loads(pkg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scm_items = pkg.get("contributes", {}).get("menus", {}).get("scm/inputBox", [])
        modified = False
        for item in scm_items:
            if item.get("command") == "github.copilot.git.generateCommitMessage":
                if item.get("when") != "false":
                    item["when"] = "false"
                    modified = True
        if modified:
            try:
                pkg_path.write_text(json.dumps(pkg))
            except OSError:
                logger.debug("Exception caught", exc_info=True)


def _install_copilot_extension(data_dir: str) -> None:
    """Install GitHub Copilot extension if not already present."""
    ext_base = Path(data_dir) / "extensions"
    if ext_base.is_dir() and any(
        d.name.startswith("github.copilot-") for d in ext_base.iterdir() if d.is_dir()
    ):
        return
    cs_binary = shutil.which("code-server")
    if not cs_binary:
        return
    env = {**os.environ, "EXTENSIONS_GALLERY": _MS_GALLERY}
    try:
        subprocess.run(
            [cs_binary, "--install-extension", "github.copilot", "--extensions-dir", str(ext_base)],
            env=env,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError):
        logger.debug("Exception caught", exc_info=True)
        pass
    _disable_copilot_scm_button(data_dir)


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
        logger.debug("Exception caught", exc_info=True)
        existing = {}
    if "workbench.colorTheme" not in existing:
        existing["workbench.colorTheme"] = "Default Dark Modern"
    for key in (
        "chat.editor.enabled",
        "chat.commandCenter.enabled",
        "chat.experimental.offerSetup",
        "workbench.chat.experimental.autoDetectLanguageModels",
    ):
        existing.pop(key, None)
    existing.update(_CS_SETTINGS)
    settings_file.write_text(json.dumps(existing, indent=2))

    state_db = user_dir / "globalStorage" / "state.vscdb"
    state_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(state_db))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value TEXT)"
        )
        for key, value in _CS_STATE_ENTRIES:
            conn.execute(
                "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.commit()
    finally:
        conn.close()

    ws_storage = user_dir / "workspaceStorage"
    if ws_storage.exists():
        for ws_dir in ws_storage.iterdir():
            for sub in ("chatSessions", "chatEditingSessions"):
                chat_dir = ws_dir / sub
                if chat_dir.exists():
                    shutil.rmtree(chat_dir, ignore_errors=True)

    ext_dir = Path(data_dir) / "extensions" / "kiss-init"
    ext_dir.mkdir(parents=True, exist_ok=True)
    (ext_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "kiss-init",
                "version": "0.0.1",
                "publisher": "kiss",
                "engines": {"vscode": "^1.80.0"},
                "activationEvents": ["onStartupFinished"],
                "extensionDependencies": ["vscode.git"],
                "main": "./extension.js",
                "contributes": {
                    "commands": [
                        {"command": "kiss.acceptChange", "title": "Accept Change"},
                        {"command": "kiss.rejectChange", "title": "Reject Change"},
                        {"command": "kiss.prevChange", "title": "Previous Change"},
                        {"command": "kiss.nextChange", "title": "Next Change"},
                        {"command": "kiss.acceptAll", "title": "Accept All Changes"},
                        {"command": "kiss.rejectAll", "title": "Reject All Changes"},
                        {"command": "kiss.commitChanges", "title": "Commit Changes"},
                        {
                            "command": "kiss.generateCommitMessage",
                            "title": "Generate Commit Message",
                            "icon": "$(sparkle)",
                        },
                        {"command": "kiss.toggleFocus", "title": "Toggle Focus to Chatbox"},
                        {"command": "kiss.runSelection", "title": "Run Selection in Chatbox"},
                    ],
                    "keybindings": [
                        {
                            "command": "kiss.toggleFocus",
                            "key": "ctrl+k",
                            "mac": "cmd+k",
                        },
                        {
                            "command": "kiss.runSelection",
                            "key": "ctrl+l",
                            "mac": "cmd+l",
                        },
                    ],
                    "menus": {
                        "scm/inputBox": [
                            {
                                "command": "kiss.generateCommitMessage",
                                "group": "navigation",
                                "when": "scmProvider == git",
                            },
                        ],
                    },
                },
            }
        )
    )
    ext_file = ext_dir / "extension.js"
    old_content = ext_file.read_text() if ext_file.exists() else ""
    ext_file.write_text(_CS_EXTENSION_JS)

    _disable_copilot_scm_button(data_dir)
    threading.Thread(target=_install_copilot_extension, args=(data_dir,), daemon=True).start()

    return old_content != _CS_EXTENSION_JS


def _scan_files(work_dir: str) -> list[str]:
    paths: list[str] = []
    skip = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        ".tox",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
    }
    try:
        for root, dirs, files in os.walk(work_dir):
            depth = os.path.relpath(root, work_dir).count(os.sep)
            if depth > 3:
                dirs.clear()
                continue
            dirs[:] = sorted(d for d in dirs if d not in skip and not d.startswith("."))
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 2000:
                    return paths
            for d in dirs:
                paths.append(os.path.relpath(os.path.join(root, d), work_dir) + "/")
    except OSError:  # pragma: no cover — os.walk swallows all OSErrors internally
        logger.debug("Exception caught", exc_info=True)
        pass
    return paths


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _parse_hunk_line(line: str) -> tuple[int, int, int, int] | None:
    """Parse a unified-diff @@ hunk header line.

    Returns:
        (old_start, old_count, new_start, new_count) or None if not a hunk header.
    """
    hm = _HUNK_RE.match(line)
    if not hm:
        return None
    return (
        int(hm.group(1)),
        int(hm.group(2)) if hm.group(2) is not None else 1,
        int(hm.group(3)),
        int(hm.group(4)) if hm.group(4) is not None else 1,
    )


def _parse_diff_hunks(work_dir: str) -> dict[str, list[tuple[int, int, int, int]]]:
    result = subprocess.run(
        ["git", "diff", "-U0", "HEAD", "--no-color"],
        capture_output=True,
        text=True,
        cwd=work_dir,
    )
    hunks: dict[str, list[tuple[int, int, int, int]]] = {}
    current_file = ""
    for line in result.stdout.split("\n"):
        dm = re.match(r"^diff --git a/.* b/(.*)", line)
        if dm:
            current_file = dm.group(1)
            continue
        hunk = _parse_hunk_line(line)
        if hunk and current_file:
            hunks.setdefault(current_file, []).append(hunk)
    return hunks


def _capture_untracked(work_dir: str) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        cwd=work_dir,
    )
    return {line.strip() for line in result.stdout.split("\n") if line.strip()}


def _snapshot_files(work_dir: str, fnames: set[str]) -> dict[str, str]:
    """Return MD5 hex digests for filenames (relative to work_dir) that exist on disk.

    Args:
        work_dir: Root directory.
        fnames: Set of relative file paths to snapshot.

    Returns:
        Dict mapping filename to hex digest of its content.
    """
    result: dict[str, str] = {}
    for fname in fnames:
        fpath = Path(work_dir) / fname
        try:
            result[fname] = hashlib.md5(fpath.read_bytes()).hexdigest()
        except OSError:
            logger.debug("Exception caught", exc_info=True)
            pass
    return result


def _untracked_base_dir() -> Path:
    """Return the directory for storing untracked file base copies.

    Uses ``{artifact_dir.parent}/data_dir/untracked-base/`` so copies
    live alongside other artifacts rather than inside the code-server
    data directory.

    Returns:
        Path to the untracked-base directory.
    """
    artifact_dir = Path(config_module.DEFAULT_CONFIG.agent.artifact_dir)
    return artifact_dir.parent / "data_dir" / "untracked-base"


def _save_untracked_base(
    work_dir: str, data_dir: str, untracked: set[str]
) -> None:
    """Save copies of untracked files before a task runs.

    These copies serve as the "base" for merge-view diffs when an agent
    modifies a pre-existing untracked file.

    Args:
        work_dir: Repository root.
        data_dir: Code-server data directory (unused, kept for API compat).
        untracked: Set of untracked file paths (relative to work_dir).
    """
    base_dir = _untracked_base_dir()
    if base_dir.exists():
        shutil.rmtree(base_dir)
    for fname in untracked:
        fpath = Path(work_dir) / fname
        try:
            if not fpath.is_file() or fpath.stat().st_size > 2_000_000:
                continue
            dest = base_dir / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fpath, dest)
        except OSError:
            logger.debug("Exception caught", exc_info=True)


def _cleanup_merge_data(data_dir: str) -> None:
    """Remove temporary merge directories and manifest after merge completes.

    Cleans up merge-temp, merge-current, untracked-base, and pending-merge.json.

    Args:
        data_dir: Code-server data directory (merge-temp lives here).
    """
    for dirname in ("merge-temp", "merge-current"):
        d = Path(data_dir) / dirname
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    base_dir = _untracked_base_dir()
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    manifest = Path(data_dir) / "pending-merge.json"
    if manifest.exists():
        try:
            manifest.unlink()
        except OSError:
            logger.debug("Exception caught", exc_info=True)


def _restore_merge_files(data_dir: str, work_dir: str) -> None:
    """Restore files to their new-lines-only state and cleanup all merge data.

    Called when Sorcar closes while hunks remain unreviewed. Copies the
    pre-merge-view file versions (containing only the agent's new lines)
    back to the work directory, then removes all temporary merge data.

    Args:
        data_dir: Code-server data directory.
        work_dir: Repository root.
    """
    current_dir = Path(data_dir) / "merge-current"
    if not current_dir.is_dir():
        return
    for src in current_dir.rglob("*"):
        if not src.is_file():
            continue
        rel = src.relative_to(current_dir)
        dest = Path(work_dir) / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    _cleanup_merge_data(data_dir)


def _diff_files(base_path: str, current_path: str) -> list[tuple[int, int, int, int]]:
    """Compute diff hunks between two files using diff -U0.

    Args:
        base_path: Path to the base (pre-task) file.
        current_path: Path to the current (post-task) file.

    Returns:
        List of (base_start, base_count, current_start, current_count) tuples.
    """
    result = subprocess.run(
        ["diff", "-U0", base_path, current_path],
        capture_output=True,
        text=True,
    )
    return [h for line in result.stdout.split("\n") if (h := _parse_hunk_line(line))]


def _prepare_merge_view(
    work_dir: str,
    data_dir: str,
    pre_hunks: dict[str, list[tuple[int, int, int, int]]],
    pre_untracked: set[str],
    pre_file_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    post_hunks = _parse_diff_hunks(work_dir)
    ub_dir = _untracked_base_dir()
    file_hunks: dict[str, list[dict[str, int]]] = {}
    for fname, hunks in post_hunks.items():
        if pre_file_hashes is not None and fname in pre_file_hashes:
            # File had pre-existing changes — check if agent actually modified it
            fpath = Path(work_dir) / fname
            try:
                current_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
            except OSError:
                logger.debug("Exception caught", exc_info=True)
                continue
            if current_hash == pre_file_hashes[fname]:
                # Content unchanged by agent — skip this file
                continue
        saved_base = ub_dir / fname
        if saved_base.is_file():
            # Diff the saved pre-task copy against current to get only agent's changes
            agent_hunks = _diff_files(str(saved_base), str(Path(work_dir) / fname))
            filtered = [
                {"bs": bs - 1, "bc": bc, "cs": cs if cc == 0 else cs - 1, "cc": cc}
                for bs, bc, cs, cc in agent_hunks
            ]
        else:
            pre = {(bs, bc) for bs, bc, _, _ in pre_hunks.get(fname, [])}
            filtered = [
                {"bs": bs - 1, "bc": bc, "cs": cs if cc == 0 else cs - 1, "cc": cc}
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
            logger.debug("Exception caught", exc_info=True)
            pass
    # Detect modified pre-existing untracked files
    if pre_file_hashes:
        for fname in pre_untracked:
            if fname in file_hunks:
                continue
            if fname not in pre_file_hashes:
                continue
            fpath = Path(work_dir) / fname
            try:
                current_hash = hashlib.md5(fpath.read_bytes()).hexdigest()
            except OSError:
                continue
            if current_hash == pre_file_hashes[fname]:
                continue
            saved_base = ub_dir / fname
            if saved_base.is_file():
                agent_hunks = _diff_files(str(saved_base), str(fpath))
                filtered = [
                    {"bs": bs - 1, "bc": bc, "cs": cs if cc == 0 else cs - 1, "cc": cc}
                    for bs, bc, cs, cc in agent_hunks
                ]
                if filtered:
                    file_hunks[fname] = filtered
            else:
                try:
                    if not fpath.is_file() or fpath.stat().st_size > 2_000_000:
                        continue
                    line_count = len(fpath.read_text().splitlines())
                    if line_count:
                        file_hunks[fname] = [
                            {"bs": 0, "bc": 0, "cs": 0, "cc": line_count}
                        ]
                except (OSError, UnicodeDecodeError):
                    logger.debug("Exception caught", exc_info=True)
    if not file_hunks:
        return {"error": "No changes"}
    merge_dir = Path(data_dir) / "merge-temp"
    if merge_dir.exists():
        shutil.rmtree(merge_dir)
    manifest_files: list[dict[str, Any]] = []
    for fname, fh in file_hunks.items():
        current_path = Path(work_dir) / fname
        base_path = merge_dir / fname
        base_path.parent.mkdir(parents=True, exist_ok=True)
        # For untracked files, use saved pre-task copy as base if available
        saved_base = ub_dir / fname
        if saved_base.is_file():
            shutil.copy2(saved_base, base_path)
        else:
            base_result = subprocess.run(
                ["git", "show", f"HEAD:{fname}"],
                capture_output=True,
                text=True,
                cwd=work_dir,
            )
            base_path.write_text(
                base_result.stdout if base_result.returncode == 0 else ""
            )
        manifest_files.append(
            {
                "name": fname,
                "base": str(base_path),
                "current": str(current_path),
                "hunks": fh,
            }
        )
    # Save current file copies (new lines only) for restoration on ungraceful close
    current_dir = Path(data_dir) / "merge-current"
    if current_dir.exists():
        shutil.rmtree(current_dir)
    for mf in manifest_files:
        src = Path(mf["current"])
        if src.is_file():
            dest = current_dir / mf["name"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    manifest = Path(data_dir) / "pending-merge.json"
    manifest.write_text(
        json.dumps(
            {
                "branch": "HEAD",
                "files": manifest_files,
            }
        )
    )
    return {"status": "opened", "count": len(manifest_files)}

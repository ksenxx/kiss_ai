"""Browser-based chatbot for RelentlessAgent-based agents."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sys
import threading
import webbrowser
from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from kiss.agents.assistant.relentless_agent import RelentlessAgent
from kiss.core.browser_ui import (
    BASE_CSS,
    EVENT_HANDLER_JS,
    HTML_HEAD,
    OUTPUT_CSS,
    BaseBrowserPrinter,
    find_free_port,
)
from kiss.core.kiss_agent import KISSAgent
from kiss.core.models.model_info import MODEL_INFO, get_available_models

_KISS_DIR = Path.home() / ".kiss"
HISTORY_FILE = _KISS_DIR / "task_history.json"
PROPOSALS_FILE = _KISS_DIR / "proposed_tasks.json"
MAX_HISTORY = 1000


def _normalize_history_entry(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict) and "task" in raw:
        return {"task": str(raw["task"]), "result": str(raw.get("result", ""))}
    return {"task": str(raw), "result": ""}


def _load_history() -> list[dict[str, str]]:
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            if isinstance(data, list):
                seen: set[str] = set()
                result: list[dict[str, str]] = []
                for t in data[:MAX_HISTORY]:
                    entry = _normalize_history_entry(t)
                    task_str = entry["task"]
                    if task_str not in seen:
                        seen.add(task_str)
                        result.append(entry)
                return result
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_history(entries: list[dict[str, str]]) -> None:
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(entries[:MAX_HISTORY]))
    except OSError:
        pass


def _set_latest_result(result: str) -> None:
    history = _load_history()
    if history:
        history[0]["result"] = result
        _save_history(history)


def _load_proposals() -> list[str]:
    if PROPOSALS_FILE.exists():
        try:
            data = json.loads(PROPOSALS_FILE.read_text())
            if isinstance(data, list):
                return [str(t) for t in data if isinstance(t, str) and t.strip()][:5]
        except (json.JSONDecodeError, OSError):
            pass
    return []


def _save_proposals(proposals: list[str]) -> None:
    try:
        _KISS_DIR.mkdir(parents=True, exist_ok=True)
        PROPOSALS_FILE.write_text(json.dumps(proposals))
    except OSError:
        pass


def _add_task(task: str) -> None:
    history = _load_history()
    task_strings = [e["task"] for e in history]
    if task in task_strings:
        idx = next(i for i, e in enumerate(history) if e["task"] == task)
        history.pop(idx)
    history.insert(0, {"task": task, "result": ""})
    _save_history(history[:MAX_HISTORY])


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
            for name in sorted(files):
                paths.append(os.path.relpath(os.path.join(root, name), work_dir))
                if len(paths) >= 1000:
                    return paths
    except OSError:
        pass
    return paths


class _StopRequested(BaseException):
    pass


CHATBOT_CSS = r"""
body{
  font-family:'Inter',system-ui,-apple-system,BlinkMacSystemFont,sans-serif;
  background:#0a0a0c;
}
header{
  background:rgba(10,10,12,0.85);backdrop-filter:blur(24px);
  -webkit-backdrop-filter:blur(24px);
  border-bottom:1px solid rgba(255,255,255,0.06);padding:14px 24px;z-index:50;
  box-shadow:0 1px 12px rgba(0,0,0,0.3);
}
.logo{font-size:15px;color:rgba(255,255,255,0.9);font-weight:600;letter-spacing:-0.2px}
.logo span{
  color:rgba(255,255,255,0.25);font-weight:400;font-size:12px;margin-left:10px;
  max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
  display:inline-block;vertical-align:middle;
}
.status{font-size:12px;color:rgba(255,255,255,0.35)}
.dot{width:7px;height:7px;background:rgba(255,255,255,0.2)}
.dot.running{background:#22c55e}
#output{
  flex:1;overflow-y:auto;padding:32px 24px 24px;
  scroll-behavior:smooth;min-height:0;
}
.ev,.txt,.spinner,.empty-msg,.user-msg{max-width:820px;margin-left:auto;margin-right:auto}
.user-msg{
  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);
  border-radius:14px;padding:14px 20px;margin:20px auto 16px;
  font-size:14.5px;line-height:1.6;color:rgba(255,255,255,0.88);
}
.txt{font-size:14.5px;line-height:1.75;color:rgba(255,255,255,0.82);padding:2px 0}
.think{
  border-left:2px solid rgba(120,180,255,0.25);
  background:rgba(120,180,255,0.02);border-radius:0 12px 12px 0;margin:12px auto;
}
.think .lbl{color:rgba(120,180,255,0.6)}
.think .cnt{color:rgba(255,255,255,0.4)}
.tc{
  border:1px solid rgba(255,255,255,0.06);border-radius:12px;
  margin:12px auto;background:rgba(255,255,255,0.015);
  transition:border-color 0.2s,box-shadow 0.2s;
}
.tc:hover{box-shadow:0 2px 16px rgba(0,0,0,0.15);border-color:rgba(255,255,255,0.1)}
.tc-h{padding:10px 16px;background:rgba(255,255,255,0.02);border-radius:12px 12px 0 0}
.tc-h:hover{background:rgba(255,255,255,0.035)}
.tn{color:rgba(88,166,255,0.9);font-size:13px}
.tp{font-size:12px;color:rgba(255,255,255,0.3)}
.td{color:rgba(255,255,255,0.3)}
.tr{
  border-left:2px solid rgba(34,197,94,0.35);
  background:rgba(34,197,94,0.02);border-radius:0 10px 10px 0;
}
.tr.err{border-left-color:rgba(248,81,73,0.35);background:rgba(248,81,73,0.02)}
.rc{
  border:1px solid rgba(34,197,94,0.15);border-radius:14px;
  background:rgba(34,197,94,0.02);
}
.rc-h{padding:16px 24px;background:rgba(34,197,94,0.04)}
.usage{border-color:rgba(255,255,255,0.05);background:rgba(255,255,255,0.02);color:rgba(255,255,255,0.3)}
.spinner{color:rgba(255,255,255,0.35)}
.spinner::before{border-color:rgba(255,255,255,0.08);border-top-color:rgba(88,166,255,0.7)}
#input-area{
  flex-shrink:0;padding:0 24px 24px;position:relative;
  background:linear-gradient(transparent,rgba(10,10,12,0.9) 50%);
  padding-top:24px;
}
#input-container{
  max-width:820px;margin:0 auto;position:relative;
  background:rgba(255,255,255,0.035);
  border:1px solid rgba(255,255,255,0.08);
  border-radius:16px;padding:14px 18px;
  box-shadow:0 0 0 1px rgba(255,255,255,0.02),0 8px 40px rgba(0,0,0,0.35);
  transition:border-color 0.3s,box-shadow 0.3s;
}
#input-container:focus-within{
  border-color:rgba(88,166,255,0.4);
  box-shadow:0 0 0 1px rgba(88,166,255,0.12),0 0 30px rgba(88,166,255,0.1),0 8px 40px rgba(0,0,0,0.35);
}
#task-input{
  width:100%;background:transparent;border:none;
  color:rgba(255,255,255,0.88);font-size:15px;font-family:inherit;
  resize:none;outline:none;line-height:1.5;
  max-height:200px;min-height:24px;
}
#task-input::placeholder{color:rgba(255,255,255,0.28)}
#task-input:disabled{opacity:0.35;cursor:not-allowed}
#input-footer{
  display:flex;justify-content:space-between;align-items:center;
  margin-top:10px;padding-top:10px;
  border-top:1px solid rgba(255,255,255,0.04);
}
#model-select{
  background:rgba(255,255,255,0.03);color:rgba(255,255,255,0.5);
  border:1px solid rgba(255,255,255,0.08);border-radius:8px;
  padding:6px 12px;font-size:12px;font-family:inherit;
  outline:none;cursor:pointer;max-width:240px;transition:border-color 0.2s;
}
#model-select:hover{border-color:rgba(255,255,255,0.16);color:rgba(255,255,255,0.65)}
#model-select:focus{border-color:rgba(88,166,255,0.4)}
#model-select option{background:#141416;color:rgba(255,255,255,0.75)}
#model-select option.no-fc{color:var(--yellow)}
#input-actions{display:flex;gap:8px;align-items:center}
#send-btn{
  background:rgba(88,166,255,0.15);color:rgba(88,166,255,0.9);border:none;
  border-radius:50%;width:36px;height:36px;cursor:pointer;
  transition:all 0.2s;display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
}
#send-btn:hover{background:rgba(88,166,255,0.3);color:#fff;box-shadow:0 0 16px rgba(88,166,255,0.2)}
#send-btn:disabled{opacity:0.2;cursor:not-allowed;box-shadow:none}
#send-btn svg{width:16px;height:16px}
#stop-btn{
  background:rgba(248,81,73,0.1);color:#f85149;
  border:1px solid rgba(248,81,73,0.15);
  border-radius:50%;width:36px;height:36px;
  cursor:pointer;transition:all 0.2s;display:none;
  align-items:center;justify-content:center;flex-shrink:0;
}
#stop-btn:hover{background:rgba(248,81,73,0.2);box-shadow:0 0 16px rgba(248,81,73,0.15)}
#stop-btn svg{width:14px;height:14px}
#autocomplete{
  position:absolute;bottom:100%;left:0;right:0;
  max-width:820px;margin:0 auto;
  background:rgba(20,20,22,0.95);backdrop-filter:blur(20px);
  border:1px solid rgba(255,255,255,0.08);border-radius:14px;
  max-height:240px;overflow-y:auto;display:none;z-index:10;
  box-shadow:0 -8px 32px rgba(0,0,0,0.5);
}
.ac-item{
  padding:10px 16px;cursor:pointer;font-size:13px;
  border-bottom:1px solid rgba(255,255,255,0.04);
  display:flex;align-items:center;gap:10px;transition:background 0.1s;
}
.ac-item:last-child{border-bottom:none}
.ac-item:hover,.ac-item.sel{background:rgba(88,166,255,0.07)}
.ac-type{
  font-size:10px;font-weight:600;text-transform:uppercase;
  padding:3px 7px;border-radius:5px;flex-shrink:0;letter-spacing:0.03em;
}
.ac-type.task{background:rgba(188,140,255,0.08);color:var(--purple)}
.ac-type.file{background:rgba(63,185,80,0.08);color:var(--green)}
.ac-text{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:rgba(255,255,255,0.55)}
#welcome{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:100%;padding:40px 20px;text-align:center;max-width:820px;margin:0 auto;
}
#welcome h2{
  font-size:28px;font-weight:700;color:rgba(255,255,255,0.92);
  margin-bottom:8px;letter-spacing:-0.5px;
  animation:fadeUp 0.5s ease;
}
@keyframes fadeUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:none}}
#welcome p{color:rgba(255,255,255,0.4);font-size:14px;margin-bottom:36px;animation:fadeUp 0.5s ease 0.1s both}
#suggestions{animation:fadeUp 0.5s ease 0.2s both;
  display:grid;grid-template-columns:1fr 1fr;gap:12px;width:100%;max-width:760px;
}
.suggestion-chip{
  background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);
  border-radius:12px;padding:14px 18px;cursor:pointer;text-align:left;
  font-size:13px;color:rgba(255,255,255,0.7);line-height:1.5;
  transition:all 0.2s ease;
}
.suggestion-chip:hover{
  background:rgba(255,255,255,0.055);border-color:rgba(255,255,255,0.14);
  color:rgba(255,255,255,0.9);transform:translateY(-2px);
  box-shadow:0 4px 24px rgba(0,0,0,0.35),0 0 0 1px rgba(255,255,255,0.05);
}
.suggestion-chip:active{transform:translateY(0);transition-duration:0.05s}
.chip-label{
  font-size:10px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.04em;margin-bottom:5px;display:block;
}
.chip-label.recent{color:rgba(88,166,255,0.7)}
.chip-label.suggested{color:rgba(188,140,255,0.7)}
#sidebar{
  position:fixed;right:0;top:0;bottom:0;width:340px;
  background:rgba(12,12,14,0.95);backdrop-filter:blur(24px);
  border-left:1px solid rgba(255,255,255,0.06);
  transform:translateX(100%);transition:transform 0.3s cubic-bezier(0.4,0,0.2,1);
  z-index:200;overflow-y:auto;padding:24px;
}
#sidebar.open{transform:translateX(0)}
#sidebar-overlay{
  position:fixed;inset:0;background:rgba(0,0,0,0.4);
  z-index:199;opacity:0;pointer-events:none;transition:opacity 0.3s;
}
#sidebar-overlay.open{opacity:1;pointer-events:auto}
#sidebar-close{
  position:absolute;top:16px;right:16px;background:none;border:none;
  color:rgba(255,255,255,0.3);font-size:20px;cursor:pointer;
  padding:4px 8px;border-radius:6px;transition:all 0.15s;
}
#sidebar-close:hover{color:rgba(255,255,255,0.7);background:rgba(255,255,255,0.05)}
.sidebar-section{margin-bottom:28px}
.sidebar-hdr{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:0.06em;color:rgba(255,255,255,0.25);margin-bottom:12px;
}
.sidebar-item{
  padding:10px 14px;background:rgba(255,255,255,0.02);
  border:1px solid rgba(255,255,255,0.04);border-radius:10px;
  margin-bottom:6px;cursor:pointer;font-size:13px;
  color:rgba(255,255,255,0.5);transition:all 0.15s;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
}
.sidebar-item:hover{
  border-color:rgba(255,255,255,0.1);background:rgba(255,255,255,0.04);
  color:rgba(255,255,255,0.8);
}
.sidebar-empty{color:rgba(255,255,255,0.2);font-size:13px;padding:8px 0}
#history-btn{
  background:none;border:1px solid rgba(255,255,255,0.07);border-radius:8px;
  color:rgba(255,255,255,0.35);font-size:12px;cursor:pointer;
  padding:5px 12px;transition:all 0.15s;display:flex;align-items:center;gap:6px;
}
#history-btn:hover{color:rgba(255,255,255,0.6);border-color:rgba(255,255,255,0.14)}
#history-btn svg{opacity:0.7}
"""

CHATBOT_JS = r"""
var O=document.getElementById('output');
var D=document.getElementById('dot');
var ST=document.getElementById('stxt');
var inp=document.getElementById('task-input');
var btn=document.getElementById('send-btn');
var stopBtn=document.getElementById('stop-btn');
var ac=document.getElementById('autocomplete');
var rl=document.getElementById('recent-list');
var pl=document.getElementById('proposed-list');
var modelSel=document.getElementById('model-select');
var sidebar=document.getElementById('sidebar');
var sidebarOverlay=document.getElementById('sidebar-overlay');
var suggestionsEl=document.getElementById('suggestions');
var running=false,autoScroll=true,userScrolled=false;
var scrollRaf=0,state={thinkEl:null,txtEl:null};
var acIdx=-1,t0=null,timerIv=null,evtSrc=null;
inp.addEventListener('input',function(){
  this.style.height='auto';
  this.style.height=Math.min(this.scrollHeight,200)+'px';
});
function toggleSidebar(){
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('open');
}
O.addEventListener('scroll',function(){
  var atBottom=O.scrollTop+O.clientHeight>=O.scrollHeight-80;
  if(!atBottom&&running)userScrolled=true;
  if(atBottom)userScrolled=false;
  autoScroll=!userScrolled;
});
function sb(){
  if(autoScroll&&!scrollRaf){scrollRaf=requestAnimationFrame(function(){
    O.scrollTop=O.scrollHeight;scrollRaf=0;
  });}
}
function startTimer(){
  t0=Date.now();
  if(timerIv)clearInterval(timerIv);
  timerIv=setInterval(function(){
    var s=Math.floor((Date.now()-t0)/1000);
    var m=Math.floor(s/60);
    ST.textContent='Running '+(m>0?m+'m ':'')+s%60+'s';
  },1000);
}
function stopTimer(){if(timerIv){clearInterval(timerIv);timerIv=null;}}
function removeSpinner(){
  var sp=document.getElementById('wait-spinner');
  if(sp)sp.remove();
}
function showSpinner(msg){
  removeSpinner();
  var sp=mkEl('div','spinner');
  sp.id='wait-spinner';
  sp.textContent=msg||'Working\u2026';
  O.appendChild(sp);sb();
}
function setReady(label){
  running=false;D.classList.remove('running');
  stopTimer();removeSpinner();
  ST.textContent=label||'Ready';
  inp.disabled=false;
  btn.style.display='';
  stopBtn.style.display='none';
  inp.focus();
}
function connectSSE(){
  if(evtSrc)evtSrc.close();
  evtSrc=new EventSource('/events');
  evtSrc.onopen=function(){};
  evtSrc.onmessage=function(e){
    var ev;try{ev=JSON.parse(e.data);}catch(x){return;}
    try{handleEvent(ev);}catch(err){console.error('Event error:',err,ev);}
  };
  evtSrc.onerror=function(){};
}
function handleEvent(ev){
  var t=ev.type;
  if(t==='thinking_start'||t==='text_delta'||t==='tool_call'
    ||t==='task_done'||t==='task_error'||t==='task_stopped')removeSpinner();
  switch(t){
  case'tasks_updated':loadTasks();loadWelcome();break;
  case'proposed_updated':loadProposed();loadWelcome();break;
  case'clear':
    O.innerHTML='';state.thinkEl=null;state.txtEl=null;
    autoScroll=true;userScrolled=false;showSpinner();break;
  case'task_done':{
    var el=t0?Math.floor((Date.now()-t0)/1000):0;
    var em=Math.floor(el/60);
    setReady('Done ('+(em>0?em+'m ':'')+el%60+'s)');
    loadTasks();loadProposed();break}
  case'task_error':{
    var err=mkEl('div','ev tr err');
    err.innerHTML='<div class="rl fail">ERROR</div>'+esc(ev.text||'Unknown error');
    O.appendChild(err);
    setReady('Error');loadTasks();loadProposed();break}
  case'task_stopped':{
    var stEl=mkEl('div','ev tr err');
    stEl.innerHTML='<div class="rl fail">STOPPED</div>Agent execution stopped by user';
    O.appendChild(stEl);
    setReady('Stopped');loadTasks();loadProposed();break}
  default:
    handleOutputEvent(ev,O,state);
    if(t==='tool_call'&&running)showSpinner('Running tool\u2026');
    if(t==='tool_result'&&running)showSpinner();
    if(t==='thinking_end'&&running)showSpinner();
  }
  sb();
}
function loadModels(){
  fetch('/models').then(function(r){return r.json();})
  .then(function(d){
    modelSel.innerHTML='';
    d.models.forEach(function(m){
      var o=document.createElement('option');
      o.value=m.name;
      o.textContent=m.name+(m.fc?'':' [no FC]');
      if(!m.fc)o.className='no-fc';
      if(m.name===d.selected)o.selected=true;
      modelSel.appendChild(o);
    });
  }).catch(function(){});
}
function submitTask(){
  var task=inp.value.trim();
  if(!task||running)return;
  running=true;inp.disabled=true;
  btn.style.display='none';
  stopBtn.style.display='inline-flex';
  D.classList.add('running');hideAC();startTimer();
  inp.style.height='auto';
  fetch('/run',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:task,model:modelSel.value})
  }).then(function(r){
    if(!r.ok){r.json().then(function(d){setReady('Error');alert(d.error||'Failed')});return;}
    inp.value='';
  }).catch(function(){setReady('Error');alert('Network error')});
}
btn.addEventListener('click',submitTask);
stopBtn.addEventListener('click',function(){fetch('/stop',{method:'POST'}).catch(function(){})});
inp.addEventListener('keydown',function(e){
  if(ac.style.display==='block'){
    var items=ac.querySelectorAll('.ac-item');
    if(e.key==='ArrowDown'){e.preventDefault();acIdx=Math.min(acIdx+1,items.length-1);updateACSel(items);return}
    if(e.key==='ArrowUp'){e.preventDefault();acIdx=Math.max(acIdx-1,-1);updateACSel(items);return}
    if(e.key==='Enter'&&acIdx>=0){e.preventDefault();items[acIdx].click();return}
    if(e.key==='Escape'){hideAC();return}
  }
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitTask()}
});
var acTimer=null;
inp.addEventListener('input',function(){if(acTimer)clearTimeout(acTimer);acTimer=setTimeout(fetchAC,200)});
function fetchAC(){
  var q=inp.value.trim();
  if(!q){hideAC();return}
  fetch('/suggestions?q='+encodeURIComponent(q))
  .then(function(r){return r.json()})
  .then(function(data){
    if(!data.length){hideAC();return}
    ac.innerHTML='';acIdx=-1;
    data.forEach(function(item){
      var d=mkEl('div','ac-item');
      d.innerHTML='<span class="ac-type '+item.type+'">'+item.type+'</span>'
        +'<span class="ac-text">'+esc(item.text)+'</span>';
      d.addEventListener('click',function(){
        if(item.type==='task'){inp.value=item.text}
        else{var p=inp.value.split(/\s/);p[p.length-1]=item.text;inp.value=p.join(' ')}
        hideAC();inp.focus();
      });
      ac.appendChild(d);
    });
    ac.style.display='block';
  }).catch(function(){hideAC()});
}
function hideAC(){ac.style.display='none';acIdx=-1}
function updateACSel(items){
  items.forEach(function(it,i){it.classList.toggle('sel',i===acIdx)});
  if(acIdx>=0)items[acIdx].scrollIntoView({block:'nearest'});
}
document.addEventListener('click',function(e){if(!ac.contains(e.target)&&e.target!==inp)hideAC()});
function loadTasks(){
  fetch('/tasks').then(function(r){return r.json()}).then(function(tasks){
    rl.innerHTML='';
    if(!tasks.length){rl.innerHTML='<div class="sidebar-empty">No recent tasks</div>';return}
    tasks.forEach(function(t){
      var taskText=typeof t==='string'?t:(t.task||'');
      var d=mkEl('div','sidebar-item');
      d.textContent=taskText;d.title=taskText;
      d.addEventListener('click',function(){inp.value=taskText;inp.focus();toggleSidebar()});
      rl.appendChild(d);
    });
  }).catch(function(){});
}
function loadProposed(){
  fetch('/proposed_tasks').then(function(r){return r.json()}).then(function(tasks){
    pl.innerHTML='';
    if(!tasks.length){pl.innerHTML='<div class="sidebar-empty">No suggestions yet</div>';return}
    tasks.forEach(function(t){
      var d=mkEl('div','sidebar-item');
      d.textContent=t;d.title=t;
      d.addEventListener('click',function(){inp.value=t;inp.focus();toggleSidebar()});
      pl.appendChild(d);
    });
  }).catch(function(){});
}
function loadWelcome(){
  if(!suggestionsEl)return;
  Promise.all([
    fetch('/tasks').then(function(r){return r.json()}).catch(function(){return []}),
    fetch('/proposed_tasks').then(function(r){return r.json()}).catch(function(){return []})
  ]).then(function(res){
    var tasks=res[0],proposed=res[1];
    suggestionsEl.innerHTML='';
    var items=[];
    proposed.slice(0,3).forEach(function(t){items.push({text:t,type:'suggested'})});
    tasks.slice(0,3).forEach(function(t){
      items.push({text:typeof t==='string'?t:(t.task||''),type:'recent'});
    });
    items.slice(0,6).forEach(function(item){
      var chip=mkEl('div','suggestion-chip');
      chip.innerHTML='<span class="chip-label '+item.type+'">'
        +(item.type==='recent'?'Recent':'Suggested')+'</span>'
        +esc(item.text);
      chip.addEventListener('click',function(){inp.value=item.text;inp.focus()});
      suggestionsEl.appendChild(chip);
    });
  });
}
connectSSE();loadModels();loadTasks();loadProposed();loadWelcome();inp.focus();
"""


def _build_html(title: str, subtitle: str) -> str:
    font_import = "@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');\n"
    css = font_import + BASE_CSS + OUTPUT_CSS + CHATBOT_CSS
    return HTML_HEAD.format(title=title, css=css) + f"""<body>
<div id="sidebar-overlay" onclick="toggleSidebar()"></div>
<div id="sidebar">
  <button id="sidebar-close" onclick="toggleSidebar()">&times;</button>
  <div class="sidebar-section">
    <div class="sidebar-hdr">Recent Tasks</div>
    <div id="recent-list"></div>
  </div>
  <div class="sidebar-section">
    <div class="sidebar-hdr">Suggested Tasks</div>
    <div id="proposed-list"></div>
  </div>
</div>
<header>
  <div class="logo">{title}<span>{subtitle}</span></div>
  <div style="display:flex;align-items:center;gap:14px">
    <div class="status"><div class="dot" id="dot"></div><span id="stxt">Ready</span></div>
    <button id="history-btn" onclick="toggleSidebar()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
      </svg>
      History
    </button>
  </div>
</header>
<div id="output">
  <div id="welcome">
    <h2>What can I help you with?</h2>
    <p>Describe a task and the agent will work on it</p>
    <div id="suggestions"></div>
  </div>
</div>
<div id="input-area">
  <div id="autocomplete"></div>
  <div id="input-container">
    <textarea id="task-input" placeholder="Ask anything\u2026" rows="1"
      autocomplete="off"></textarea>
    <div id="input-footer">
      <select id="model-select"></select>
      <div id="input-actions">
        <button id="send-btn"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg></button>
        <button id="stop-btn"><svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg></button>
      </div>
    </div>
  </div>
</div>
<script>
{EVENT_HANDLER_JS}
{CHATBOT_JS}
</script>
</body>
</html>"""


def run_chatbot(
    agent_factory: Callable[[str], RelentlessAgent],
    title: str = "KISS Assistant",
    subtitle: str = "Interactive Agent",
    work_dir: str | None = None,
    default_model: str = "claude-sonnet-4-5",
    agent_kwargs: dict[str, Any] | None = None,
) -> None:
    """Run a browser-based chatbot UI for any RelentlessAgent-based agent.

    Starts a Starlette web server with SSE streaming, task history, autocomplete,
    model selection, and proposed task suggestions.

    Args:
        agent_factory: Callable that takes a name string and returns a RelentlessAgent instance.
        title: Title displayed in the browser UI header.
        subtitle: Subtitle displayed in the browser UI header.
        work_dir: Working directory for the agent. Defaults to current directory.
        default_model: Default LLM model name for the model selector.
        agent_kwargs: Additional keyword arguments passed to agent.run().
    """
    import uvicorn
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
    from starlette.routing import Route

    printer = BaseBrowserPrinter()
    running = False
    running_lock = threading.Lock()
    actual_work_dir = work_dir or os.getcwd()
    file_cache: list[str] = _scan_files(actual_work_dir)
    agent_thread: threading.Thread | None = None
    proposed_tasks: list[str] = _load_proposals()
    proposed_lock = threading.Lock()
    selected_model = default_model
    extra_kwargs = agent_kwargs or {}
    html_page = _build_html(title, subtitle)

    def refresh_file_cache() -> None:
        nonlocal file_cache
        file_cache = _scan_files(actual_work_dir)

    def refresh_proposed_tasks() -> None:
        nonlocal proposed_tasks
        history = _load_history()
        if not history:
            with proposed_lock:
                proposed_tasks = []
            printer.broadcast({"type": "proposed_updated"})
            return
        task_list = "\n".join(f"- {e['task']}" for e in history[:20])
        agent = KISSAgent("Task Proposer")
        try:
            result = agent.run(
                model_name="gemini-2.0-flash",
                prompt_template=(
                    "Based on these past tasks a developer has worked on, suggest 5 new "
                    "tasks they might want to do next. Tasks should be natural follow-ups, "
                    "related improvements, or complementary work.\n\n"
                    "Past tasks:\n{task_list}\n\n"
                    "Return ONLY a JSON array of 5 short task description strings. "
                    'Example: ["Add unit tests for X", "Refactor Y module"]'
                ),
                arguments={"task_list": task_list},
                is_agentic=False,
            )
            start = result.index("[")
            end = result.index("]", start) + 1
            proposals = json.loads(result[start:end])
            proposals = [str(p) for p in proposals if isinstance(p, str) and p.strip()][:5]
        except Exception:
            proposals = []
        with proposed_lock:
            proposed_tasks = proposals
        _save_proposals(proposals)
        printer.broadcast({"type": "proposed_updated"})

    def run_agent_thread(task: str, model_name: str) -> None:
        nonlocal running, agent_thread
        try:
            _add_task(task)
            printer.broadcast({"type": "tasks_updated"})
            printer.broadcast({"type": "clear"})
            agent = agent_factory("Chatbot")
            result = agent.run(
                prompt_template=task,
                work_dir=actual_work_dir,
                printer=printer,
                model_name=model_name,
                **extra_kwargs,
            )
            _set_latest_result(result or "")
            printer.broadcast({"type": "task_done"})
        except _StopRequested:
            _set_latest_result("(stopped)")
            printer.broadcast({"type": "task_stopped"})
        except Exception as e:
            _set_latest_result(f"(error: {e})")
            printer.broadcast({"type": "task_error", "text": str(e)})
        finally:
            with running_lock:
                running = False
                agent_thread = None
            refresh_file_cache()
            try:
                refresh_proposed_tasks()
            except Exception:
                pass

    def stop_agent() -> bool:
        nonlocal agent_thread
        with running_lock:
            thread = agent_thread
        if thread is None or not thread.is_alive():
            return False
        import ctypes

        tid = thread.ident
        if tid is None:
            return False
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(_StopRequested),
        )
        return True

    async def index(request: Request) -> HTMLResponse:
        return HTMLResponse(html_page)

    async def events(request: Request) -> StreamingResponse:
        cq = printer.add_client()

        async def generate() -> AsyncGenerator[str]:
            try:
                while True:
                    try:
                        event = cq.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.05)
                        continue
                    yield f"data: {json.dumps(event)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                printer.remove_client(cq)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def run_task(request: Request) -> JSONResponse:
        nonlocal running, agent_thread, selected_model
        with running_lock:
            if running:
                return JSONResponse({"error": "Agent is already running"}, status_code=409)
            running = True
        body = await request.json()
        task = body.get("task", "").strip()
        model = body.get("model", "").strip() or selected_model
        selected_model = model
        if not task:
            with running_lock:
                running = False
            return JSONResponse({"error": "Empty task"}, status_code=400)
        t = threading.Thread(target=run_agent_thread, args=(task, model), daemon=True)
        with running_lock:
            agent_thread = t
        t.start()
        return JSONResponse({"status": "started"})

    async def stop_task(request: Request) -> JSONResponse:
        if stop_agent():
            return JSONResponse({"status": "stopping"})
        return JSONResponse({"error": "No running task"}, status_code=404)

    async def suggestions(request: Request) -> JSONResponse:
        query = request.query_params.get("q", "").strip()
        if not query:
            return JSONResponse([])
        q_lower = query.lower()
        results: list[dict[str, str]] = []
        for entry in _load_history():
            task = entry["task"]
            if q_lower in task.lower():
                results.append({"type": "task", "text": task})
                if len(results) >= 5:
                    break
        last_word = query.split()[-1].lower() if query.split() else q_lower
        if last_word:
            count = 0
            for path in file_cache:
                if last_word in path.lower():
                    results.append({"type": "file", "text": path})
                    count += 1
                    if count >= 10:
                        break
        return JSONResponse(results)

    async def tasks(request: Request) -> JSONResponse:
        return JSONResponse(_load_history())

    async def proposed_tasks_endpoint(request: Request) -> JSONResponse:
        with proposed_lock:
            return JSONResponse(list(proposed_tasks))

    async def models_endpoint(request: Request) -> JSONResponse:
        available = get_available_models()
        models_list = []
        for name in available:
            info = MODEL_INFO.get(name)
            if info:
                models_list.append({"name": name, "fc": info.is_function_calling_supported})
        return JSONResponse({"models": models_list, "selected": selected_model})

    app = Starlette(routes=[
        Route("/", index),
        Route("/events", events),
        Route("/run", run_task, methods=["POST"]),
        Route("/stop", stop_task, methods=["POST"]),
        Route("/suggestions", suggestions),
        Route("/tasks", tasks),
        Route("/proposed_tasks", proposed_tasks_endpoint),
        Route("/models", models_endpoint),
    ])

    threading.Thread(target=refresh_proposed_tasks, daemon=True).start()

    port = find_free_port()
    url = f"http://127.0.0.1:{port}"
    print(f"{title} running at {url}")
    print(f"Work directory: {actual_work_dir}")
    webbrowser.open(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


def main() -> None:
    """Launch the KISS chatbot UI in assistant or coding mode based on KISS_MODE env var."""
    from kiss.agents.assistant.assistant_agent import AssistantAgent
    from kiss.agents.coding_agents.relentless_coding_agent import RelentlessCodingAgent

    work_dir = str(Path(sys.argv[1]).resolve()) if len(sys.argv) > 1 else os.getcwd()

    mode = os.environ.get("KISS_MODE", "assistant").lower()
    if mode == "assistant":
        run_chatbot(
            agent_factory=AssistantAgent,
            title="KISS General Assistant: SWE, browsing, agent creation, optimization and more",
            subtitle=f"Working Directory: {work_dir}",
            work_dir=work_dir,
            agent_kwargs={"headless": False},
        )
    else:
        run_chatbot(
            agent_factory=RelentlessCodingAgent,
            title="KISS Relentless Coding Assistant",
            subtitle=f"Working Directory: {work_dir}",
            work_dir=work_dir,
        )


if __name__ == "__main__":
    main()

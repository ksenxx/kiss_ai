"""Shared browser UI components for KISS agent viewers."""

import json
import queue
import socket
import threading
import time
from typing import Any

import yaml

from kiss.core.printer import Printer, extract_extras, extract_path_and_lang, truncate_result


def find_free_port() -> int:
    """Find and return an available TCP port on localhost.

    Returns:
        int: A free port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return int(s.getsockname()[1])


BASE_CSS = r"""
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#0d1117;--surface:#161b22;--surface2:#1c2128;--border:#30363d;
  --text:#e6edf3;--dim:#8b949e;--accent:#58a6ff;--green:#3fb950;
  --red:#f85149;--yellow:#d29922;--cyan:#79c0ff;--purple:#bc8cff;
}
body{
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  background:var(--bg);color:var(--text);line-height:1.6;
  height:100vh;display:flex;flex-direction:column;overflow:hidden;
}
header{
  background:linear-gradient(135deg,var(--surface) 0%,var(--surface2) 100%);
  border-bottom:1px solid var(--border);padding:12px 24px;
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.logo{font-size:18px;font-weight:700;color:var(--accent);letter-spacing:-.3px}
.logo span{color:var(--dim);font-weight:400;font-size:14px;margin-left:8px}
.status{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--dim)}
.dot{width:8px;height:8px;border-radius:50%;background:var(--dim)}
.dot.running{background:var(--green);animation:pulse 2s infinite}
.dot.done{animation:none;background:var(--dim)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
"""

OUTPUT_CSS = r"""
.ev{margin-bottom:6px;animation:fadeIn .15s ease}
@keyframes fadeIn{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}
.think{
  padding:10px 16px;margin:10px 0;
  border:1px solid var(--border);border-radius:8px;
  background:rgba(121,192,255,.04);max-height:200px;overflow-y:auto;
}
.think .lbl{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;color:var(--cyan);margin-bottom:4px;
  display:flex;align-items:center;gap:6px;cursor:pointer;user-select:none;
}
.think .lbl .arrow{transition:transform .2s;display:inline-block}
.think .lbl .arrow.collapsed{transform:rotate(-90deg)}
.think .cnt{
  font-size:11px;color:var(--dim);font-style:italic;
  white-space:pre-wrap;word-break:break-word;
}
.think .cnt.hidden{display:none}
.txt{font-size:8px;white-space:pre-wrap;word-break:break-word;padding:2px 0;line-height:1.4}
.tc{
  border:1px solid var(--border);border-radius:8px;margin:10px 0;
  overflow:hidden;background:var(--surface);transition:box-shadow .2s;
}
.tc:hover{box-shadow:0 2px 12px rgba(0,0,0,.3)}
.tc-h{
  padding:9px 14px;background:var(--surface2);
  display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;
}
.tc-h:hover{background:rgba(48,54,61,.8)}
.tc-h .chv{color:var(--dim);transition:transform .2s;font-size:11px;flex-shrink:0}
.tc-h .chv.open{transform:rotate(90deg)}
.tn{font-weight:600;font-size:13px;color:var(--accent)}
.tp{font-size:12px;color:var(--cyan);font-family:'SF Mono','Fira Code',monospace}
.td{font-size:11px;color:var(--dim);font-style:italic}
.tc-b{
  padding:10px 14px;max-height:300px;overflow-y:auto;
  font-family:'SF Mono','Fira Code',monospace;font-size:11px;line-height:1.5;
}
.tc-b.hide{display:none}
.tc-b pre{margin:4px 0;white-space:pre-wrap;word-break:break-word}
.diff-old{
  color:var(--red);background:rgba(248,81,73,.08);
  padding:1px 6px;display:block;white-space:pre-wrap;word-break:break-word;
}
.diff-new{
  color:var(--green);background:rgba(63,185,80,.08);
  padding:1px 6px;display:block;white-space:pre-wrap;word-break:break-word;
}
.diff-ctx{
  color:var(--dim);padding:1px 6px;display:block;opacity:0.5;
  white-space:pre-wrap;word-break:break-word;
}
.diff-hl-del{background:rgba(248,81,73,.3);border-radius:2px;padding:0 1px}
.diff-hl-add{background:rgba(63,185,80,.3);border-radius:2px;padding:0 1px}
.extra{color:var(--dim);margin:2px 0}
.tr{
  padding:8px 14px;margin:6px 0;
  border:1px solid var(--border);border-radius:8px;
  font-family:'SF Mono','Fira Code',monospace;
  font-size:11px;max-height:200px;overflow-y:auto;
  white-space:pre-wrap;word-break:break-word;background:rgba(63,185,80,.04);
}
.tr.err{background:rgba(248,81,73,.04);border-color:rgba(248,81,73,.2)}
.tr .rl{
  font-size:11px;font-weight:600;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:4px;
}
.tr .rl.ok{color:var(--green)}
.tr .rl.fail{color:var(--red)}
.rc{
  border:1px solid var(--green);border-radius:8px;
  margin:20px 0;overflow:hidden;background:var(--surface);
}
.rc-h{
  padding:14px 20px;background:rgba(63,185,80,.08);
  display:flex;align-items:center;justify-content:space-between;
}
.rc-h h3{color:var(--green);font-size:14px;font-weight:600}
.rs{font-size:12px;color:var(--dim);display:flex;gap:18px}
.rs b{color:var(--text);font-weight:500}
.rc-body{
  padding:16px 20px;font-size:13px;max-height:400px;overflow-y:auto;
  word-break:break-word;line-height:1.7;
}
.rc-body.pre{white-space:pre-wrap}
.prompt{
  border:1px solid var(--border);border-radius:8px;margin:10px 0;
  overflow:hidden;background:var(--surface);
}
.prompt-h{
  padding:8px 16px;background:rgba(121,192,255,.08);
  font-size:11px;font-weight:600;color:var(--cyan);
  text-transform:uppercase;letter-spacing:.04em;
}
.prompt-body{
  padding:12px 16px;font-size:13px;white-space:pre-wrap;
  word-break:break-word;line-height:1.6;max-height:400px;overflow-y:auto;
}
.sys{
  font-size:13px;color:var(--dim);font-family:'SF Mono','Fira Code',monospace;
  white-space:pre-wrap;word-break:break-word;padding:2px 0;
}
.bash-panel{
  background:#08080a;border:1px solid var(--border);
  border-radius:8px;margin:2px 0 8px;padding:10px 12px;
  max-height:300px;overflow-y:auto;
  font-family:'SF Mono','Fira Code','Cascadia Code',monospace;
  font-size:8px;line-height:1.5;color:rgba(255,255,255,0.65);
  white-space:pre-wrap;word-break:break-word;
}
.usage{
  border:1px solid var(--border);border-radius:8px;margin:6px 0;
  padding:4px 12px;background:var(--surface);font-size:11px;
  color:var(--dim);font-style:italic;
  font-family:'SF Mono','Fira Code',monospace;
  white-space:nowrap;overflow-x:auto;
}
.empty-msg{
  text-align:center;color:var(--dim);
  padding:80px 20px;font-size:15px;line-height:2;
}
.spinner{
  display:flex;align-items:center;gap:10px;
  padding:16px 0;color:var(--dim);font-size:13px;
  animation:fadeIn .3s ease;
}
.spinner::before{
  content:'';width:16px;height:16px;
  border:2px solid var(--border);border-top-color:var(--accent);
  border-radius:50%;flex-shrink:0;animation:spin .8s linear infinite;
}
@keyframes spin{to{transform:rotate(360deg)}}
::-webkit-scrollbar{width:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--dim)}
code{font-family:'SF Mono','Fira Code',monospace}
.hljs{background:var(--surface2)!important;border-radius:6px;padding:10px!important}
"""

EVENT_HANDLER_JS = r"""
function esc(t){var d=document.createElement('div');d.textContent=t;return d.innerHTML}
function mkEl(tag,cls){var e=document.createElement(tag);if(cls)e.className=cls;return e}
function toggleTC(el){
  el.nextElementSibling.classList.toggle('hide');
  el.querySelector('.chv').classList.toggle('open');
}
function toggleThink(el){
  var p=el.parentElement;
  p.querySelector('.cnt').classList.toggle('hidden');
  el.querySelector('.arrow').classList.toggle('collapsed');
}
function lineDiff(a,b){
  var al=a.split('\n'),bl=b.split('\n'),m=al.length,n=bl.length;
  var dp=[];
  for(var i=0;i<=m;i++){dp[i]=new Array(n+1);dp[i][0]=0;}
  for(var j=0;j<=n;j++)dp[0][j]=0;
  for(var i=1;i<=m;i++)for(var j=1;j<=n;j++)
    dp[i][j]=al[i-1]===bl[j-1]?dp[i-1][j-1]+1:Math.max(dp[i-1][j],dp[i][j-1]);
  var ops=[],i=m,j=n;
  while(i>0||j>0){
    if(i>0&&j>0&&al[i-1]===bl[j-1]){ops.unshift({t:'=',o:al[--i],n:bl[--j]});}
    else if(j>0&&(i===0||dp[i][j-1]>=dp[i-1][j])){ops.unshift({t:'+',n:bl[--j]});}
    else{ops.unshift({t:'-',o:al[--i]});}
  }
  return ops;
}
function hlInline(oldL,newL){
  var mn=Math.min(oldL.length,newL.length),pre=0,suf=0;
  while(pre<mn&&oldL[pre]===newL[pre])pre++;
  while(suf<mn-pre&&oldL[oldL.length-1-suf]===newL[newL.length-1-suf])suf++;
  var pf=oldL.substring(0,pre),sf=suf?oldL.substring(oldL.length-suf):'';
  return{
    o:esc(pf)+'<span class="diff-hl-del">'+esc(oldL.substring(pre,oldL.length-suf))+'</span>'+esc(sf),
    n:esc(pf)+'<span class="diff-hl-add">'+esc(newL.substring(pre,newL.length-suf))+'</span>'+esc(sf)
  };
}
function renderDiff(oldStr,newStr){
  var ops=lineDiff(oldStr,newStr),html='',i=0;
  while(i<ops.length){
    var dels=[],adds=[];
    while(i<ops.length&&ops[i].t==='-'){dels.push(ops[i++]);}
    while(i<ops.length&&ops[i].t==='+'){adds.push(ops[i++]);}
    if(dels.length||adds.length){
      var pairs=Math.min(dels.length,adds.length);
      for(var p=0;p<pairs;p++){
        var h=hlInline(dels[p].o,adds[p].n);
        html+='<div class="diff-old">- '+h.o+'</div>';
        html+='<div class="diff-new">+ '+h.n+'</div>';
      }
      for(var p=pairs;p<dels.length;p++)
        html+='<div class="diff-old">- '+esc(dels[p].o)+'</div>';
      for(var p=pairs;p<adds.length;p++)
        html+='<div class="diff-new">+ '+esc(adds[p].n)+'</div>';
      continue;
    }
    html+='<div class="diff-ctx">  '+esc(ops[i].o)+'</div>';i++;
  }
  return html;
}
function handleOutputEvent(ev,O,state){
  var t=ev.type;
  switch(t){
  case'thinking_start':
    state.thinkEl=mkEl('div','ev think');
    state.thinkEl.innerHTML=
      '<div class="lbl" onclick="toggleThink(this)">'
      +'<span class="arrow">\u25BE</span> Thinking</div>'
      +'<div class="cnt"></div>';
    O.appendChild(state.thinkEl);break;
  case'thinking_delta':
    if(state.thinkEl){
      var tc=state.thinkEl.querySelector('.cnt');
      tc.textContent+=(ev.text||'').replace(/\n\n+/g,'\n');
      state.thinkEl.scrollTop=state.thinkEl.scrollHeight;
    }break;
  case'thinking_end':
    if(state.thinkEl){
      state.thinkEl.querySelector('.lbl').innerHTML=
        '<span class="arrow collapsed">\u25BE</span> Thinking (click to expand)';
      state.thinkEl.querySelector('.cnt').classList.add('hidden');
    }
    state.thinkEl=null;break;
  case'text_delta':
    if(!state.txtEl){state.txtEl=mkEl('div','txt');O.appendChild(state.txtEl)}
    state.txtEl.textContent+=(ev.text||'').replace(/\n\n+/g,'\n');break;
  case'text_end':state.txtEl=null;break;
  case'tool_call':{
    if(state.bashPanel&&state.bashBuf){state.bashPanel.textContent+=state.bashBuf;state.bashBuf=''}
    state.bashPanel=null;state.bashRaf=0;
    var c=mkEl('div','ev tc');
    var h='<span class="chv open">\u25B6</span><span class="tn">'+esc(ev.name)+'</span>';
    if(ev.path){var ep=esc(ev.path).replace(/"/g,'&quot;');
      h+='<span class="tp" data-path="'+ep+'"> '+esc(ev.path)+'</span>';}
    if(ev.description)h+='<span class="td"> '+esc(ev.description)+'</span>';
    var b='';
    if(ev.command)b+='<pre><code class="language-bash">'+esc(ev.command)+'</code></pre>';
    if(ev.content){
      var lc=ev.lang?'language-'+esc(ev.lang):'';
      b+='<pre><code class="'+lc+'">'+esc(ev.content)+'</code></pre>';
    }
    if(ev.old_string!==undefined&&ev.new_string!==undefined){
      b+=renderDiff(ev.old_string,ev.new_string);
    }else{
      if(ev.old_string!==undefined)b+='<div class="diff-old">- '+esc(ev.old_string)+'</div>';
      if(ev.new_string!==undefined)b+='<div class="diff-new">+ '+esc(ev.new_string)+'</div>';
    }
    if(ev.extras){for(var k in ev.extras)
      b+='<div class="extra">'+esc(k)+': '+esc(ev.extras[k])+'</div>'}
    var body=b||'<em style="color:var(--dim)">No arguments</em>';
    c.innerHTML='<div class="tc-h" onclick="toggleTC(this)">'+h+'</div>'
      +'<div class="tc-b'+(b?'':' hide')+'">'+body+'</div>';
    O.appendChild(c);
    if(ev.command){var bp=mkEl('div','bash-panel');O.appendChild(bp);state.bashPanel=bp}
    if(typeof hljs!=='undefined')c.querySelectorAll('pre code').forEach(
      function(bl){hljs.highlightElement(bl)});
    break}
  case'tool_result':{
    if(state.bashPanel&&state.bashBuf){state.bashPanel.textContent+=state.bashBuf;state.bashBuf=''}
    var hadBash=!!state.bashPanel;
    state.bashPanel=null;state.bashRaf=0;
    if(hadBash&&!ev.is_error)break;
    var r=mkEl('div','ev tr'+(ev.is_error?' err':''));
    var lb=ev.is_error?'FAILED':'OK';
    var lc2=ev.is_error?'fail':'ok';
    r.innerHTML='<div class="rl '+lc2+'">'+lb+'</div>'+esc(ev.content);
    O.appendChild(r);break}
  case'system_output':{
    if(state.bashPanel){
      if(!state.bashBuf)state.bashBuf='';
      state.bashBuf+=(ev.text||'');
      if(!state.bashRaf){state.bashRaf=requestAnimationFrame(function(){
        if(state.bashPanel)state.bashPanel.textContent+=state.bashBuf;
        state.bashBuf='';state.bashRaf=0;
        if(state.bashPanel)state.bashPanel.scrollTop=state.bashPanel.scrollHeight;
      })}
    }else{
      var s=mkEl('div','ev sys');
      s.textContent=(ev.text||'').replace(/\n\n+/g,'\n');
      O.appendChild(s);
    }break}
  case'result':{
    var rc=mkEl('div','ev rc');
    var rb='';
    if(ev.success===false){
      var fs='color:var(--red);font-weight:700;font-size:16px;margin-bottom:12px';
      rb+='<div style="'+fs+'">Status: FAILED</div>';
    }
    var usePre=true;
    if(ev.summary){
      var sum=(ev.summary||'').replace(/\n{3,}/g,'\n\n').trim();
      if(typeof marked!=='undefined'){rb+=marked.parse(sum);usePre=false;}
      else{rb+=esc(sum);}
    }else{
      rb+=esc((ev.text||'(no result)').replace(/\n{3,}/g,'\n\n').trim());
    }
    rc.innerHTML='<div class="rc-h"><h3>Result</h3><div class="rs">'
      +'<span>Steps <b>'+(ev.step_count||0)+'</b></span>'
      +'<span>Tokens <b>'+(ev.total_tokens||0)+'</b></span>'
      +'<span>Cost <b>'+(ev.cost||'N/A')+'</b></span>'
      +'</div></div><div class="rc-body'
      +(usePre?' pre':'')+'">'+rb+'</div>';
    if(typeof hljs!=='undefined')rc.querySelectorAll('pre code').forEach(
      function(bl){hljs.highlightElement(bl)});
    O.appendChild(rc);break}
  case'prompt':{
    var p=mkEl('div','ev prompt');
    p.innerHTML='<div class="prompt-h">Prompt</div>'
      +'<div class="prompt-body">'+esc(ev.text||'')+'</div>';
    O.appendChild(p);break}
  case'usage_info':{
    var u=mkEl('div','ev usage');
    u.textContent=ev.text||'';
    O.appendChild(u);break}
  }
}
"""

HTML_HEAD = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.1/marked.min.js"></script>
<style>
{css}
</style>
</head>
"""


class BaseBrowserPrinter(Printer):
    def __init__(self) -> None:
        self._clients: list[queue.Queue[dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""
        self._bash_buffer: list[str] = []
        self._bash_last_flush = 0.0

    def reset(self) -> None:
        """Reset internal streaming and tool-parsing state for a new turn."""
        self._current_block_type = ""
        self._tool_name = ""
        self._tool_json_buffer = ""
        self._bash_buffer.clear()

    def _flush_bash(self) -> None:
        if self._bash_buffer:
            text = "".join(self._bash_buffer)
            self._bash_buffer.clear()
            self._bash_last_flush = time.monotonic()
            self.broadcast({"type": "system_output", "text": text})

    @staticmethod
    def _parse_result_yaml(raw: str) -> dict[str, Any] | None:
        try:
            data = yaml.safe_load(raw)
        except Exception:
            return None
        if isinstance(data, dict) and "summary" in data:
            return data
        return None

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send an SSE event dict to all connected clients.

        Args:
            event: The event dictionary to broadcast.
        """
        with self._lock:
            for cq in self._clients:
                cq.put(event)

    def add_client(self) -> queue.Queue[dict[str, Any]]:
        """Register a new SSE client and return its event queue.

        Returns:
            queue.Queue[dict[str, Any]]: A queue that will receive broadcast events.
        """
        cq: queue.Queue[dict[str, Any]] = queue.Queue()
        with self._lock:
            self._clients.append(cq)
        return cq

    def remove_client(self, cq: queue.Queue[dict[str, Any]]) -> None:
        """Unregister an SSE client's event queue.

        Args:
            cq: The client queue to remove.
        """
        with self._lock:
            try:
                self._clients.remove(cq)
            except ValueError:
                pass

    def _broadcast_result(
        self, text: str, step_count: int = 0, total_tokens: int = 0, cost: str = "N/A",
    ) -> None:
        event: dict[str, Any] = {
            "type": "result",
            "text": text or "(no result)",
            "step_count": step_count,
            "total_tokens": total_tokens,
            "cost": cost,
        }
        parsed = self._parse_result_yaml(text) if text else None
        if parsed:
            event["success"] = parsed.get("success")
            event["summary"] = str(parsed["summary"])
        self.broadcast(event)

    def print(self, content: Any, type: str = "text", **kwargs: Any) -> str:
        """Render content by broadcasting SSE events to connected browser clients.

        Args:
            content: The content to display.
            type: Content type (e.g. "text", "prompt", "stream_event",
                "tool_call", "tool_result", "result", "usage_info", "message").
            **kwargs: Additional options such as tool_input, is_error, cost,
                step_count, total_tokens.

        Returns:
            str: Extracted text from stream events, or empty string.
        """
        if type == "text":
            from io import StringIO

            from rich.console import Console

            buf = StringIO()
            Console(file=buf, highlight=False, width=120, no_color=True).print(content)
            text = buf.getvalue()
            if text.strip():
                self.broadcast({"type": "text_delta", "text": text})
            return ""
        if type == "prompt":
            self.broadcast({"type": "prompt", "text": str(content)})
            return ""
        if type == "stream_event":
            return self._handle_stream_event(content)
        if type == "message":
            self._handle_message(content, **kwargs)
            return ""
        if type == "usage_info":
            self.broadcast({"type": "usage_info", "text": str(content).strip()})
            return ""
        if type == "bash_stream":
            self._bash_buffer.append(str(content))
            if time.monotonic() - self._bash_last_flush >= 0.1:
                self._flush_bash()
            return ""
        if type == "tool_call":
            self._flush_bash()
            self.broadcast({"type": "text_end"})
            self._format_tool_call(str(content), kwargs.get("tool_input", {}))
            return ""
        if type == "tool_result":
            self._flush_bash()
            self.broadcast({
                "type": "tool_result",
                "content": truncate_result(str(content)),
                "is_error": kwargs.get("is_error", False),
            })
            return ""
        if type == "result":
            self.broadcast({"type": "text_end"})
            self._broadcast_result(
                str(content), kwargs.get("step_count", 0),
                kwargs.get("total_tokens", 0), kwargs.get("cost", "N/A"),
            )
            return ""
        return ""

    async def token_callback(self, token: str) -> None:
        """Broadcast a streamed token as an SSE delta event to browser clients.

        Args:
            token: The text token to broadcast.
        """
        if token:
            delta_type = (
                "thinking_delta"
                if self._current_block_type == "thinking"
                else "text_delta"
            )
            self.broadcast({"type": delta_type, "text": token})

    def _format_tool_call(self, name: str, tool_input: dict[str, Any]) -> None:
        file_path, lang = extract_path_and_lang(tool_input)
        event: dict[str, Any] = {"type": "tool_call", "name": name}
        if file_path:
            event["path"] = file_path
            event["lang"] = lang
        if desc := tool_input.get("description"):
            event["description"] = str(desc)
        if command := tool_input.get("command"):
            event["command"] = str(command)
        if content := tool_input.get("content"):
            event["content"] = str(content)
        old_string = tool_input.get("old_string")
        new_string = tool_input.get("new_string")
        if old_string is not None:
            event["old_string"] = str(old_string)
        if new_string is not None:
            event["new_string"] = str(new_string)
        extras = extract_extras(tool_input)
        if extras:
            event["extras"] = extras
        self.broadcast(event)

    def _handle_stream_event(self, event: Any) -> str:
        evt = event.event
        evt_type = evt.get("type", "")
        text = ""

        if evt_type == "content_block_start":
            block = evt.get("content_block", {})
            block_type = block.get("type", "")
            self._current_block_type = block_type
            if block_type == "thinking":
                self.broadcast({"type": "thinking_start"})
            elif block_type == "tool_use":
                self._tool_name = block.get("name", "?")
                self._tool_json_buffer = ""

        elif evt_type == "content_block_delta":
            delta = evt.get("delta", {})
            delta_type = delta.get("type", "")
            if delta_type == "thinking_delta":
                text = delta.get("thinking", "")
            elif delta_type == "text_delta":
                text = delta.get("text", "")
            elif delta_type == "input_json_delta":
                self._tool_json_buffer += delta.get("partial_json", "")

        elif evt_type == "content_block_stop":
            block_type = self._current_block_type
            if block_type == "thinking":
                self.broadcast({"type": "thinking_end"})
            elif block_type == "tool_use":
                try:
                    tool_input = json.loads(self._tool_json_buffer)
                except (json.JSONDecodeError, ValueError):
                    tool_input = {"_raw": self._tool_json_buffer}
                self._format_tool_call(self._tool_name, tool_input)
            else:
                self.broadcast({"type": "text_end"})
            self._current_block_type = ""

        return text

    def _handle_message(self, message: Any, **kwargs: Any) -> None:
        if hasattr(message, "subtype") and hasattr(message, "data"):
            if message.subtype == "tool_output":
                text = message.data.get("content", "")
                if text:
                    self.broadcast({"type": "system_output", "text": text})
        elif hasattr(message, "result"):
            budget_used = kwargs.get("budget_used", 0.0)
            self._broadcast_result(
                message.result,
                kwargs.get("step_count", 0),
                kwargs.get("total_tokens_used", 0),
                f"${budget_used:.4f}" if budget_used else "N/A",
            )
        elif hasattr(message, "content"):
            for block in message.content:
                if hasattr(block, "is_error") and hasattr(block, "content"):
                    self.broadcast({
                        "type": "tool_result",
                        "content": truncate_result(str(block.content)),
                        "is_error": bool(block.is_error),
                    })

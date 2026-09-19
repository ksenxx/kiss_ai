"""Second-pass speed report over /tmp/speed7a.db + /tmp/speed7d.db (built by README.md + speed_analysis.py).

Adds to speed_analysis.py: outcome/quality linkage, tool-error waste, repeated commands,
review-round convergence, per-model fixed latency, and lever coverage vs. WP0-WP7.
Prints a plain-text report; nothing is written except /tmp/speed7_report.txt.
"""
import collections
import json
import re
import sqlite3
import statistics as st

A = sqlite3.connect('/tmp/speed7a.db'); A.row_factory = sqlite3.Row
D = sqlite3.connect('/tmp/speed7d.db'); D.row_factory = sqlite3.Row
out_lines = []


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s); out_lines.append(s)


def h(sec):
    return f'{sec/3600:.1f} h'


def pct(a, b):
    return f'{100*a/b:.0f}%' if b else 'n/a'


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0
    return xs[min(len(xs)-1, int(p*len(xs)))]


tasks = {r['id']: dict(r) for r in A.execute('select * from tasks')}
for t in tasks.values():
    t['tool_by'] = json.loads(t['tool_by'] or '{}')
top = [t for t in tasks.values() if not t['parent']]
subs = [t for t in tasks.values() if t['parent']]
kids = collections.defaultdict(list)
for t in subs:
    kids[t['parent']].append(t)

P('=' * 80)
P(f'Tasks {len(tasks)}  top-level {len(top)}  sub-agents {len(subs)}')
lo = min(t['first_ts'] for t in tasks.values()); hi = max(t['last_ts'] for t in tasks.values())
import datetime as dt
P('window', dt.datetime.fromtimestamp(lo, dt.timezone.utc), '->', dt.datetime.fromtimestamp(hi, dt.timezone.utc))

# ---------------------------------------------------------------- A. top-level wall decomposition
wall = sum(t['wall'] for t in top)
llm = sum(t['llm'] for t in top)
tool_by = collections.Counter()
for t in top:
    for k, v in t['tool_by'].items():
        tool_by[k] += v
P('\n## A. Top-level wall', h(wall), ' median', f"{st.median(t['wall'] for t in top):.0f}s")
P('  own LLM round trips', h(llm), pct(llm, wall), ' steps', sum(t['steps'] for t in top))
for k, v in tool_by.most_common(8):
    P(f'  tool {k:22s} {h(v):>8s} {pct(v, wall)}')
other = wall - llm - sum(tool_by.values())
P('  unaccounted (idle/startup/stop)', h(other), pct(other, wall))

# startup: task_settings -> prompt
starts = [t['startup'] for t in tasks.values() if t['startup'] is not None]
P('  startup (first event -> prompt) top-level avg', f"{st.mean([t['startup'] for t in top if t['startup'] is not None]):.1f}s",
  ' sub-agent avg', f"{st.mean([t['startup'] for t in subs if t['startup'] is not None]):.1f}s",
  ' p90', f"{q([t['startup'] for t in subs if t['startup'] is not None], .9):.1f}s")

# ---------------------------------------------------------------- B. outcome / quality signals
res = {}
for r in D.execute("select task_id, snippet, ts from ev where type in ('result','task_stopped')"):
    res[r['task_id']] = r
succ = collections.Counter()
for t in top:
    r = res.get(t['id'])
    if r is None:
        succ['no result'] += 1
    elif 'task_stopped' in r['snippet'][:30]:
        succ['stopped'] += 1
    elif 'success: true' in r['snippet']:
        succ['success'] += 1
    elif 'success: false' in r['snippet']:
        succ['failed'] += 1
    else:
        succ['other'] += 1
P('\n## B. Outcomes (top-level):', dict(succ))
# is_continue restarts (context hand-off)
cont = [t for t in tasks.values() if res.get(t['id']) and 'is_continue: true' in res[t['id']]['snippet']]
P('  is_continue=true finishes:', len(cont), ' wall', h(sum(t['wall'] for t in cont)))
# multiple prompts (restart at context limit)
multi = [r['task_id'] for r in D.execute("select task_id, count(*) c from ev where type='prompt' group by task_id having c>1")]
P('  tasks with >1 prompt event (context-limit restart):', len(multi), ' wall', h(sum(tasks[i]['wall'] for i in multi if i in tasks)))

# quality proxy: user follow-up in same chat that signals dissatisfaction
neg = re.compile(r"\b(still|didn'?t|did not|does not|doesn'?t|not (?:see|work|fixed|done)|wrong|broken|fail|again|regress|revert|you missed|incorrect)\b", re.I)
by_chat = collections.defaultdict(list)
for r in D.execute("select id, chat_id, timestamp, task, cost from th where parent_task_id=''"):
    if r['chat_id']:
        by_chat[r['chat_id']].append(dict(r))
neg_follow = 0; pairs = 0; neg_examples = []
for c, ts in by_chat.items():
    ts.sort(key=lambda x: x['timestamp'])
    for a, b in zip(ts, ts[1:]):
        pairs += 1
        if neg.search(b['task'][:300]) and len(b['task']) < 600:
            neg_follow += 1
            neg_examples.append((b['task'][:110].replace('\n', ' '), a['id']))
P(f'  consecutive same-chat prompt pairs {pairs}; follow-up prompt reads as a complaint/redo: {neg_follow} ({pct(neg_follow, pairs)})')
for e in neg_examples[:12]:
    P('     -', e[0])

# ---------------------------------------------------------------- C. tool errors -> wasted round trips
err = collections.Counter(); tot = collections.Counter(); err_ex = collections.defaultdict(list)
for r in A.execute('select tool, is_error, res_snip, call_snip, task_id from calls'):
    tot[r['tool']] += 1
    if r['is_error']:
        err[r['tool']] += 1
        if len(err_ex[r['tool']]) < 400:
            err_ex[r['tool']].append((r['res_snip'] or '')[:160])
P('\n## C. Tool-call errors (each costs one extra LLM round trip)')
P('  total calls', sum(tot.values()), ' errors', sum(err.values()), pct(sum(err.values()), sum(tot.values())))
for k, v in err.most_common(10):
    P(f'  {k:24s} {v:5d} / {tot[k]:6d} ({pct(v, tot[k])})')
# categorize Bash errors
cat = collections.Counter()
for s in err_ex['Bash']:
    s2 = s.lower()
    if 'timed out' in s2 or 'timeout' in s2:
        cat['timeout'] += 1
    elif 'worktree' in s2 or 'guard' in s2 or 'refus' in s2:
        cat['guard/refused'] += 1
    elif 'no such file' in s2 or 'not found' in s2:
        cat['not found'] += 1
    elif 'failed' in s2 or 'error' in s2:
        cat['cmd failed'] += 1
    else:
        cat['other'] += 1
P('  Bash error categories (sample of', len(err_ex['Bash']), '):', dict(cat))
cat = collections.Counter()
for s in err_ex['Edit']:
    s2 = s.lower()
    if 'not found' in s2 or 'no match' in s2 or 'does not' in s2:
        cat['old_string not found'] += 1
    elif 'multiple' in s2 or 'occurr' in s2:
        cat['ambiguous'] += 1
    elif 'unchanged' in s2 or 'read' in s2:
        cat['must read first / unchanged'] += 1
    else:
        cat['other'] += 1
P('  Edit error categories:', dict(cat))
cat = collections.Counter()
for s in err_ex['run_parallel'] + err_ex['run_agent']:
    s2 = s.lower()
    if 'budget' in s2:
        cat['budget refused'] += 1
    elif 'unknown' in s2 or 'no such agent' in s2 or 'not found' in s2:
        cat['unknown agent'] += 1
    elif 'timed out' in s2 or 'timeout' in s2:
        cat['timeout'] += 1
    else:
        cat['other'] += 1
P('  run_parallel/run_agent error categories:', dict(cat))
for s in (err_ex['Read'] + err_ex['Write'])[:6]:
    P('     Read/Write err e.g.:', s[:120])

# ---------------------------------------------------------------- D. per-step latency vs context per model
P('\n## D. Fixed LLM latency of trivial 1-tool steps (out_chars<600) by model and context')
rows = A.execute('select s.llm_time, s.ctx, s.out_chars, s.ntools, t.model from steps s join tasks t on t.id=s.task_id where s.ntools=1 and s.out_chars<600 and s.ctx is not null').fetchall()
buck = collections.defaultdict(list)
def b_of(c):
    return '<25k' if c < 25000 else '25-50k' if c < 50000 else '50-100k' if c < 100000 else '100-200k' if c < 200000 else '200-300k' if c < 300000 else '>300k'
for r in rows:
    buck[(r['model'], b_of(r['ctx']))].append(r['llm_time'])
models = collections.Counter(r['model'] for r in rows)
order = ['<25k', '25-50k', '50-100k', '100-200k', '200-300k', '>300k']
P('  model'.ljust(28), ''.join(f'{o:>12s}' for o in order))
for m, _ in models.most_common(7):
    P(f'  {m[:26]:26s}', ''.join(f"{(str(round(st.median(buck[(m,o)]),1))+'s/'+str(len(buck[(m,o)]))) if len(buck[(m,o)])>=15 else '-':>12s}" for o in order))
# share of steps by ctx bucket and total llm time
allsteps = A.execute('select llm_time, ctx, out_chars from steps where ctx is not null').fetchall()
cb = collections.Counter(); ct = collections.Counter()
for r in allsteps:
    cb[b_of(r['ctx'])] += 1; ct[b_of(r['ctx'])] += r['llm_time']
P('  steps by ctx bucket:', {o: f'{cb[o]} ({h(ct[o])})' for o in order})
# output throughput
big = A.execute('select llm_time, out_chars from steps where out_chars>3000 and llm_time>5').fetchall()
if big:
    P('  output throughput (steps >3k chars): median', f"{st.median(r['out_chars']/r['llm_time'] for r in big):.0f} chars/s")

# ---------------------------------------------------------------- E. overhead-only steps
P('\n## E. Overhead-only LLM round trips')
def steps_where(cond):
    return A.execute(f'select count(*), sum(llm_time) from steps where {cond}').fetchone()
for label, cond in [
    ('summary alone', "tools='summary'"),
    ('memory_search/pull alone', "tools in ('memory_search','memory_pull','memory_list','memory_read')"),
    ('set_model alone', "tools='set_model'"),
    ('Read SORCAR.md alone (any step)', "tools='Read' and task_id in (select task_id from calls where tool='Read' and call_snip like '%SORCAR.md%') and step_no=1"),
    ('single Read step', "tools='Read'"),
    ('single Bash step', "tools='Bash'"),
    ('0-tool steps (text only, not last)', "ntools=0 and step_no < (select max(step_no) from steps s2 where s2.task_id=steps.task_id)"),
]:
    c, s = steps_where(cond)
    P(f'  {label:36s} {c:6d} steps  {h(s or 0)}')
# consecutive read-only single-tool steps
ro = {'Read', 'memory_search', 'memory_pull', 'memory_read', 'memory_list'}
cons = 0; cons_t = 0.0
prev = None
for r in A.execute('select task_id, step_no, tools, llm_time from steps order by task_id, step_no'):
    cur = (r['task_id'], r['tools'] in ro or (r['tools'] or '').startswith('Read,') and set(r['tools'].split(',')) <= ro)
    if prev and prev[0] == cur[0] and prev[1] and cur[1]:
        cons += 1; cons_t += r['llm_time']
    prev = cur
P(f'  consecutive read-only steps (could be batched) {cons} steps {h(cons_t)}')
# Write steps
c, s = steps_where("tools like '%Write%'")
P(f'  steps containing Write {c} {h(s)}  avg {s/c if c else 0:.1f}s')
# summary tool: how often per steps
sm = A.execute("select count(*) from calls where tool='summary'").fetchone()[0]
P(f'  summary calls {sm} over {len(A.execute("select 1 from steps").fetchall())} steps = one per {len(allsteps)/max(sm,1):.1f} steps')

# ---------------------------------------------------------------- F. Bash
P('\n## F. Bash')
bash = A.execute("select task_id, dur, call_snip from calls where tool='Bash'").fetchall()
def cmd_of(snip):
    m = re.search(r'"command":\s*"((?:[^"\\]|\\.)*)"', snip or '')
    return (m.group(1) if m else '').encode().decode('unicode_escape', 'ignore') if m else ''
cats = collections.Counter(); catn = collections.Counter()
for r in bash:
    c = cmd_of(r['call_snip']).lower()
    k = ('pytest' if 'pytest' in c else 'uv run check' if 'uv run check' in c else 'sleep/poll' if re.search(r'\bsleep\b', c) else
         'npm/node' if re.search(r'\b(npm|npx|node|tsc)\b', c) else 'git' if c.startswith('git') or ' git ' in c[:30] else 'ruff/lint' if 'ruff' in c or 'mypy' in c or 'lint' in c else 'python' if 'python' in c else 'other')
    cats[k] += r['dur']; catn[k] += 1
P('  total', len(bash), h(sum(r['dur'] for r in bash)), ' >=60s:', sum(1 for r in bash if r['dur'] >= 60), h(sum(r['dur'] for r in bash if r['dur'] >= 60)))
for k, v in cats.most_common():
    P(f'  {k:14s} {catn[k]:6d} calls {h(v):>8s}')
# repeated identical commands within one task
rep = collections.Counter(); rep_t = 0.0
seen = collections.defaultdict(collections.Counter)
for r in bash:
    c = cmd_of(r['call_snip'])
    if len(c) < 8:
        continue
    seen[r['task_id']][c] += 1
dups = 0; dup_calls = 0
for tid, cnt in seen.items():
    for c, n in cnt.items():
        if n > 1:
            dups += 1; dup_calls += n - 1
P(f'  identical command re-run inside same task: {dups} distinct cmds, {dup_calls} repeat executions')
# full test suite / check runs per top-level task tree
tree = collections.defaultdict(lambda: collections.Counter())
for r in bash:
    c = cmd_of(r['call_snip'])
    root = tasks[r['task_id']]['root'] if r['task_id'] in tasks else r['task_id']
    if 'uv run check' in c:
        tree[root]['check'] += 1; tree[root]['check_t'] += r['dur']
    if 'pytest' in c and r['dur'] >= 120:
        tree[root]['long_pytest'] += 1; tree[root]['long_pytest_t'] += r['dur']
multi_check = [(k, v['check'], v['check_t']) for k, v in tree.items() if v['check'] >= 3]
P(f"  task trees running `uv run check` >=3x: {len(multi_check)}, total {h(sum(x[2] for x in multi_check))}; max {max((x[1] for x in multi_check), default=0)} runs")
# timeouts
to = [r for r in A.execute("select dur, res_snip, call_snip from calls where tool='Bash' and is_error and (lower(res_snip) like '%timed out%' or lower(res_snip) like '%timeout%')")]
P(f'  Bash timeouts: {len(to)} calls, {h(sum(r["dur"] for r in to))} spent before timing out')

# ---------------------------------------------------------------- G. fan-out
P('\n## G. run_parallel fan-out')
rp = A.execute("select task_id, step_no, dur, call_snip from calls where tool='run_parallel'").fetchall()
P('  calls', len(rp), ' parent wait', h(sum(r['dur'] for r in rp)))
# match batches: children of parent whose first_ts within call window
batch_sizes = []; straggle = 0.0; med_wait = 0.0; single = 0; single_t = 0.0
rounds = collections.Counter()
for r in rp:
    pid = r['task_id']
    rounds[pid] += 1
    ch = [c for c in kids.get(pid, [])]
    # crude: children started within the call's window
    call_ts = D.execute('select ts from ev where task_id=? and type=\'tool_call\' and name=\'run_parallel\' order by seq limit 1 offset ?', (pid, rounds[pid]-1)).fetchone()
    if call_ts is None:
        continue
    cts = call_ts[0]
    inb = [c for c in ch if cts - 5 <= c['first_ts'] <= cts + r['dur'] + 5 and c['first_ts'] - cts < 60]
    n = len(inb)
    batch_sizes.append(n)
    if n == 1:
        single += 1; single_t += r['dur']
    if n >= 3:
        ws = sorted(c['wall'] for c in inb)
        straggle += ws[-1] - ws[len(ws)//2]; med_wait += ws[-1]
P(f'  single-child batches {single} ({h(single_t)});  batches>=3: straggler (max-median) {h(straggle)} of {h(med_wait)} waited')
P('  parents with >=3 rounds:', sum(1 for v in rounds.values() if v >= 3), ' rounds', sum(v for v in rounds.values() if v >= 3))
# review convergence: reviewer children results 'None'/'no issues'
rev_kids = [c for c in subs if re.search(r'\breview', (c['task'] or '')[:400], re.I)]
clean = 0
for c in rev_kids:
    r = res.get(c['id'])
    if r and re.search(r'summary: (<p>)?(None|No (issues|findings|problems)|LGTM)', r['snippet'] or ''):
        clean += 1
P(f'  reviewer sub-agents {len(rev_kids)}  wall {h(sum(c["wall"] for c in rev_kids))}  LLM {h(sum(c["llm"] for c in rev_kids))}  clean verdict ("None"/no issues) {clean}')
# sub-agent step distribution for trivial tasks
sh = [c for c in subs if re.search(r'pytest|shard|split', (c['task'] or '')[:300], re.I)]
if sh:
    P(f'  test-shard sub-agents {len(sh)}: steps median {st.median(c["steps"] for c in sh)}, p90 {q([c["steps"] for c in sh], .9)}, wall median {st.median(c["wall"] for c in sh):.0f}s, LLM share {pct(sum(c["llm"] for c in sh), sum(c["wall"] for c in sh))}')
    P(f'    shards with >4 steps: {sum(1 for c in sh if c["steps"]>4)}  (extra LLM time {h(sum(c["llm"] for c in sh if c["steps"]>4))})')
# LLM bash wrappers: sub-agents whose task is just a shell command
wrap = [c for c in subs if re.match(r'\s*(Run|Execute|run|execute)\b.{0,40}(`|command|pytest|uv run|npm)', (c['task'] or '')[:200])]
P(f'  sub-agents that are shell-command wrappers {len(wrap)}: wall {h(sum(c["wall"] for c in wrap))}, of which LLM {h(sum(c["llm"] for c in wrap))} (pure overhead vs run_commands_parallel)')

# ---------------------------------------------------------------- H. run_agent
ra = A.execute("select dur, call_snip from calls where tool='run_agent'").fetchall()
agents = collections.Counter(); agents_t = collections.Counter()
for r in ra:
    m = re.search(r'"agent":\s*"([^"]+)"', r['call_snip'] or '')
    a = m.group(1) if m else '?'
    agents[a] += 1; agents_t[a] += r['dur']
P('\n## H. run_agent', len(ra), h(sum(r['dur'] for r in ra)))
for a, n in agents.most_common(8):
    P(f'  {a[:40]:40s} {n:4d} {h(agents_t[a])}')

# ---------------------------------------------------------------- I. model mix
P('\n## I. Model mix (steps, LLM hours, median step)')
mm = collections.defaultdict(list)
for r in A.execute('select s.llm_time, t.model from steps s join tasks t on t.id=s.task_id'):
    mm[r['model']].append(r['llm_time'])
for m, xs in sorted(mm.items(), key=lambda kv: -sum(kv[1]))[:8]:
    P(f'  {m[:30]:30s} steps {len(xs):6d}  {h(sum(xs)):>8s}  median {st.median(xs):.1f}s  p90 {q(xs,.9):.1f}s')

# ---------------------------------------------------------------- J. biggest top-level tasks
P('\n## J. Longest top-level tasks')
for t in sorted(top, key=lambda t: -t['wall'])[:10]:
    rp_t = t['tool_by'].get('run_parallel', 0); ra_t = t['tool_by'].get('run_agent', 0); b = t['tool_by'].get('Bash', 0)
    P(f"  {h(t['wall']):>7s} steps {t['steps']:4d} LLM {h(t['llm']):>6s} rp {h(rp_t):>6s} ra {h(ra_t):>6s} bash {h(b):>6s} ${t['cost']:.0f} {t['model'][:18]:18s} {(t['task'] or '')[:60].replace(chr(10),' ')}")

open('/tmp/speed7_report.txt', 'w').write('\n'.join(out_lines))

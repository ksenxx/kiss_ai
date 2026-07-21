# Hacker News post — HydraKV / KISS Sorcar

A "Show HN" submission based on the HydraKV paper
(`papers/kvstorepaper/hydra_kv.tex`). Every number below is taken from the
paper or from the released artifact README; nothing is claimed beyond what
those documents claim. The text is meant to be pasted as-is into the Show HN
text box (HN text posts are plain text; blank lines separate paragraphs).

Per HN's own guidance (newsguidelines.html and dang's Show HN tips), the text
avoids marketing language, states scope limits explicitly, and leads with the
artifact rather than the AI.

______________________________________________________________________

## Title (recommended, 70 chars)

Show HN: HydraKV – an AI-built KV store, 5.9x FASTER on skewed YCSB-A

## Title (alternative, 78 chars)

Show HN: An AI agent wrote a 4K-line KV store that outruns FASTER on YCSB-A

______________________________________________________________________

## Post text

I posted on LinkedIn recently that the way I try to stay relevant in the age
of AI is to push it to its limits, because that's the only way I've found to
see what it genuinely cannot do
(https://www.linkedin.com/feed/update/urn:li:activity:7485166552271495168/).
This project is what that pushing looked like in practice.

The problem is narrow and fully specified: YCSB-A (50% reads, 50% blind
upserts, Zipfian theta=0.95) over 250M keys with 100-byte values, about 25 GB
of data, under a hard 8 GiB memory budget, on a 64-vCPU machine with eight
local NVMe SSDs in RAID0. Microsoft's FASTER, which was engineered for exactly
this kind of larger-than-memory point-operation workload, sustains 0.93 Mops/s
on that machine with that harness. HydraKV, the engine in this post, sustains
5.51 Mops/s on the same setup. That's 5.9x. A roofline argument in the paper
puts it at about 89% of what the measured device IOPS and cache hit rate allow
for this class of design, which is also why two mid-project stretch goals (10,
then 7 Mops/s) were never met and the paper says so with arithmetic instead of
excuses.

The part I find more interesting than the number: I didn't write the engine.
It was written by KISS Sorcar, an open-source agent framework I've been
building (and building with) for the past several months. My entire
contribution to HydraKV was six task prompts and two short steering messages.
One steering message, quoted verbatim in the paper, was: "Did you start with
the most performant variant of the engine?" (It hadn't. It had layered
correctness fixes onto an already-rejected experimental branch, and only
bisecting archived revisions exposed that.)

Because an agent optimizing against a fixed benchmark will overfit it, the
prompts forced an adversarial loop: the agent generated workload variants
designed to break its own engine (sparse key spaces, clustered and drifting
hot sets, grounded in the Facebook and Twitter workload studies it went and
read), repaired the engine until it met the goal on all of them, and finished
with a held-out variant generated after all engine work stopped. Development
and review were split across models from two different vendors: one wrote
code, the other did strictly read-only reviews and kept flagging concurrency
hazards the first one had rationalized away. Two independent audits followed;
the second re-ran the engine on the reference hardware with its own oracles
and called the result genuine, "not reward-hacked, not copied". API spend for
the first three tasks was under $200.

What HydraKV actually is: ~3,970 lines of dependency-free C++17. Append-only
O_DIRECT slot log with CRC-sealed records, a fingerprint hash index, an
admission-controlled write-back cache (doorkeeper plus segmented
second-chance), a hand-rolled io_uring read-miss path, scan-based crash
recovery, background compaction, fail-soft handling of disk-full and torn
writes, and a non-blocking delete path. None of the mechanisms is individually
new; the paper is explicit about that. The claim is only about the
composition, on this one configuration, and the evidence behind it.

Honest limits, straight from the paper: one machine, one harness, 30-second
cold-cache windows, and the FASTER comparison exists only on the original
workload. Read-heavy mixes are much slower (2.78 Mops/s read-only),
delete-heavy ones are slow by design, and there has been no long-duration
endurance run. This is not "a better KV store". It's evidence about what a
conservative composition of known ideas can do under a hard memory budget, and
about how far an agent can be pushed when you make it attack its own work.

About the framework, briefly, since people will ask. KISS Sorcar is Apache-2.0,
bring-your-own-key, and deliberately small: about 2,850 lines for the five
core agent classes. It runs as a CLI, a VS Code extension, and a web app. The
two properties that mattered here were long-horizon execution (it ran
unattended for multi-hour sessions against a remote benchmark box over SSH,
continuing across context windows by summarizing itself) and mixing models
from different vendors inside one task with a sentence in the prompt. It also
built itself: the framework, the extension, and the system prompt were all
written by the agent operating on its own repo. On Terminal Bench 2.0 it
scores 62.2%, next to 58% for Claude Code and 61.7% for Cursor's agent, with
no benchmark-specific tuning.

Everything is public: the engine, the 28 end-to-end tests, the sanitizer
matrix, the variant generator, both audits, the ideas ledger of accepted and
rejected optimizations, and the raw benchmark logs are at
https://github.com/ksenxx/kiss_ai/tree/main/projects/kv_adversarial (it builds
with a single g++ line, no dependencies). The paper, with all six prompts
reproduced word for word, typos included, is at
https://github.com/ksenxx/kiss_ai/blob/main/papers/kvstorepaper/hydra_kv.pdf.
The framework paper is at https://arxiv.org/abs/2604.23822.

Happy to answer questions, including the uncomfortable ones.

______________________________________________________________________

## Submission notes

- Submit as a Show HN text post (title + text), since the artifact is
  runnable and public. If HN drops the text on a URL submission, add it as
  the first comment instead (dang's guidance allows either).
- The URL field, if used instead of a text post, should point to the
  artifact: https://github.com/ksenxx/kiss_ai/tree/main/projects/kv_adversarial
- Post from a personal account (not a project-named one) and stay in the
  thread to answer questions.
- Do not ask anyone to upvote or comment; do not reply to your own thread
  with booster comments.

# Hacker News post — HydraKV / KISS Sorcar

A "Show HN" submission based on the HydraKV paper
(`papers/kvstorepaper/hydra_kv.tex`). Every number below is taken from the
paper or from the released artifact README; nothing is claimed beyond what
those documents claim. The text is meant to be pasted as-is into the Show HN
text box (HN text posts are plain text; blank lines separate paragraphs).

Per HN's own guidance (newsguidelines.html and dang's Show HN tips), the text
avoids marketing language, states scope limits explicitly, and leads with the
artifact rather than the AI. HN limits text submissions to 4,000 characters;
the body below is 3,888 characters (3,952 even if every newline counts as
two), verified by script.

______________________________________________________________________

## Title (recommended, 69 chars)

Show HN: HydraKV – an AI-built KV store, 5.9x FASTER on skewed YCSB-A

## Title (alternative, 78 chars)

Show HN: An AI agent wrote a 4K-line KV store that outruns FASTER on YCSB-A

______________________________________________________________________

## Post text

I posted on LinkedIn recently that the way I try to stay relevant in the age
of AI is to push it to its limits, because that's the only way to see what it
genuinely cannot do
(https://www.linkedin.com/feed/update/urn:li:activity:7485166552271495168/).
This project is what that pushing looked like in practice.

The problem is narrow and fully specified: YCSB-A (50% reads, 50% blind
upserts, Zipfian theta=0.95) over 250M keys with 100-byte values, about 25 GB
of data under a hard 8 GiB memory budget, on a 64-vCPU box with eight local
NVMe SSDs in RAID0. Microsoft's FASTER, which was built for exactly this
kind of larger-than-memory point workload, sustains 0.93 Mops/s with the
same harness. HydraKV sustains 5.51 Mops/s on the same setup. That's 5.9x. A
roofline argument in the paper puts that at about 89% of what measured device
IOPS and cache hit rate allow; two stretch goals (10, then 7 Mops/s) were
never met, and the paper shows the arithmetic for why.

The part I find more interesting than the number: I didn't write the engine.
It was written by KISS Sorcar (https://kisssorcar.github.io/), an open-source 
agent framework I've been building for the past several months. My entire 
contribution was six task prompts and two short steering messages. 

Because an agent optimizing against a fixed benchmark will overfit it, the
prompts forced an adversarial loop: the agent generated workload variants
designed to break its own engine (sparse key spaces, clustered and drifting
hot sets), repaired it until it met the goal on all of them, and finished
with a held-out variant generated after all engine work stopped. One model
wrote code; a second model from a different vendor did strictly read-only
reviews and kept flagging concurrency hazards the first had rationalized
away. A second independent audit re-ran the engine on the reference hardware
and called the result genuine, "not reward-hacked, not copied". API spend for
the first three tasks was under $200.

What HydraKV actually is: ~3,970 lines of dependency-free C++17. Append-only
O_DIRECT slot log with CRC-sealed records, a fingerprint hash index, an
admission-controlled write-back cache, an io_uring read-miss path, scan-based
crash recovery, background compaction, and fail-soft handling of disk-full
and torn writes. None of the mechanisms is individually new; the claim is
only about the composition, on this one configuration.

Honest limits, straight from the paper: one machine, one harness, 30-second
cold-cache windows, and the FASTER comparison exists only on the original
workload. Read-only runs are much slower (2.78 Mops/s), delete-heavy ones are
slow by design, and there's been no long endurance run.

About the framework, since people will ask: KISS Sorcar is Apache-2.0,
bring-your-own-key, and small, about 2,850 lines for the five core agent
classes. What mattered here was long-horizon execution (multi-hour unattended
sessions against a remote benchmark box, continuing across context windows by
summarizing itself) and mixing models from different vendors inside one task.
It also built itself: the framework, its VS Code extension, and its system
prompt were written by the agent operating on its own repo. 

Everything is public: engine, 28 end-to-end tests, sanitizer matrix, variant
generator, audits, ideas ledger, and raw benchmark logs at
https://github.com/ksenxx/kiss_ai/tree/main/projects/kv_adversarial (builds
with one g++ line). The paper, with all six prompts reproduced verbatim,
typos included:
https://github.com/ksenxx/kiss_ai/blob/main/papers/kvstorepaper/hydra_kv.pdf.
The framework paper: https://arxiv.org/abs/2604.23822.

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

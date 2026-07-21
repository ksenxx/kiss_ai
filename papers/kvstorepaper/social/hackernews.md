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

______________________________________________________________________

# LinkedIn post — HydraKV / KISS Sorcar

The same story, rewritten for LinkedIn. Every number is taken from the
HydraKV paper or the artifact README, identical to the Show HN text above.
LinkedIn constraints respected: plain text (LinkedIn renders no markdown),
under the 3,000-character post limit (the body below is 2,565 characters,
verified by script; the opening hook is 176 characters), short paragraphs,
and a hook in the first two lines because LinkedIn truncates the post at
roughly 200 characters behind a "...more" fold. It reads as a follow-up to
the earlier "stay relevant by pushing AI to its limits" post
(https://www.linkedin.com/feed/update/urn:li:activity:7485166552271495168/),
so it opens by calling back to it rather than linking it.

______________________________________________________________________

## Post text

An AI agent just wrote a key-value store that runs 5.9x faster than
Microsoft's FASTER — on the exact workload FASTER was built for. I didn't
write a single line of the engine.

A while back I posted here that the way I try to stay relevant in the age of
AI is to push it to its limits, because that's the only way to see what it
genuinely cannot do. This is what that pushing produced.

The problem was narrow and fully specified: YCSB-A (50% reads, 50% blind
upserts, Zipfian theta=0.95) over 250M keys, about 25 GB of data under a hard
8 GiB memory budget, on a 64-vCPU box with eight local NVMe SSDs in RAID0.
FASTER sustains 0.93 Mops/s with the same harness. HydraKV — about 3,970
lines of dependency-free C++17 — sustains 5.51 Mops/s. A roofline analysis in
the paper puts that at about 89% of what the measured hardware allows.

The engine was written end to end by KISS Sorcar, the open-source agent
framework I've been building. My entire contribution: six task prompts and
two short steering messages. API spend for the first three tasks: under $200.

What made me trust the result more than the number itself:

1. Adversarial self-testing. An agent optimizing a fixed benchmark will
overfit it, so the prompts forced the agent to generate workload variants
designed to break its own engine, repair it until it passed all of them, and
finish on a held-out variant generated after all engine work stopped.

2. Cross-vendor review. One model wrote the code; a second model from a
different vendor did strictly read-only reviews and kept flagging concurrency
hazards the first had rationalized away.

3. Independent audit. A separate audit re-ran the engine on the reference
hardware and called the result genuine — "not reward-hacked, not copied."

The honest limits, straight from the paper: one machine, one harness,
30-second cold-cache windows; read-only runs are much slower (2.78 Mops/s);
two stretch goals (10, then 7 Mops/s) were never met, and the paper shows the
arithmetic for why. None of HydraKV's mechanisms is individually new — the
claim is only about the composition.

Everything is public: the engine, 28 end-to-end tests, the audits, the raw
benchmark logs, and the paper with all six prompts reproduced verbatim, typos
included.

Paper: https://github.com/ksenxx/kiss_ai/blob/main/papers/kvstorepaper/hydra_kv.pdf
Artifact: https://github.com/ksenxx/kiss_ai/tree/main/projects/kv_adversarial
Framework (Apache-2.0, ~2,850 lines of core agent code): https://kisssorcar.github.io/

#AI #AIAgents #SystemsProgramming #Databases #OpenSource

______________________________________________________________________

## Posting notes (LinkedIn)

- Paste as plain text; LinkedIn does not render markdown, so the post uses
  numbered lines instead of bullets and no bold/italics.
- The first two lines are the hook shown above the "...more" fold; do not
  prepend anything to them.
- Attach a visual if desired (e.g., the throughput figure from the paper);
  posts with an image or document generally travel further, but the text
  stands alone.
- Links in the body can suppress reach; if that matters, move the three
  links to the first comment and say "links in the first comment" at the
  end of the post instead.
- Reply to comments from the personal account; do not ask for reposts.

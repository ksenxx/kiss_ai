# 48-Hour Show HN Launch Checklist — KISS Sorcar

## Chosen slot (picked 2026-07-12)

- **Primary: Wednesday 2026-07-15, 9:30am ET** (6:30am PT). Tue 7/14 is too
  soon to complete the human pre-flight (hand-rewrite, karma warm-up,
  public-repo publication of this directory); Wednesday keeps a full buffer
  day inside the strongest Tue–Thu band.
- Fallback 1: Thursday 2026-07-16, 9:30am ET.
- Fallback 2: Tuesday 2026-07-21, 9:30am ET (if pre-flight slips).

## T-7 to T-1 days: pre-flight

- [ ] **Hand-rewrite the first comment.** HUMAN-ONLY step: follow
  HAND_REWRITE_GUIDE.md (beat checklist + verified-facts table; contains
  no prose to copy). Write from scratch in a plain text editor; do not
  paste or LLM-edit any sentence (HN rule, dang 2026-03-28). The agent
  cannot do this step by definition — any agent-written text is
  LLM-generated.
- [x] Quickstart works first-try on a clean machine in ≤ 5 minutes —
  **VERIFIED 2026-07-12** on clean Docker containers:
  `python:3.13-slim` → `pip install pipx && pipx install kiss-agent-framework` installs 2026.7.18 and `sorcar --help` runs;
  timed end-to-end **34 seconds**. `python:3.14-slim` also passes.
  `python:3.12-slim` fails gracefully ("No matching distribution
  found"), validating the "Python 3.13+ only" limitation claim. Only
  caveat: pipx PATH warning (`pipx ensurepath` fixes). Still to do by
  hand: spot-check on a clean macOS user account.
- [ ] README: hero GIF visible in the first screen.
- [ ] Repo loads fast; no signup anywhere in the try path.
- [ ] HN account: personal username (a person, not a brand), email set in
  profile (repost/second-chance invites go there), some comment karma
  from genuine participation in the prior week.
- [x] Verify links — **VERIFIED 2026-07-12**, all 200:
  https://github.com/ksenxx/kiss_ai,
  https://arxiv.org/abs/2604.23822,
  https://kisssorcar.github.io/docs/index.md,
  https://kisssorcar.github.io/llms.txt.
- [ ] **BLOCKER — must clear before T-0.** Ensure
  marketing/agent-markets-itself/episode-01/ is pushed and PUBLIC:
  as of 2026-07-12,
  https://github.com/ksenxx/kiss_ai/tree/main/marketing returns **404**
  (the episode lives in the private dev repo only). Either publish the
  marketing/ directory to the public ksenxx/kiss_ai repo, or mirror the
  episode to kisssorcar.github.io and point the meta-disclosure link
  there. The hand-written comment's meta paragraph depends on this link
  resolving.
- [ ] Prepare (do not pre-post) tailored Reddit drafts: r/selfhosted,
  r/LocalLLaMA, r/ChatGPTCoding, r/opensource — first-person, karma > 80.
- [ ] Clear your calendar for the submission day + next morning.

## T-0: submission (Wed 2026-07-15, 9:30am ET — fallbacks above)

- [ ] Submit at https://news.ycombinator.com/submit
  \- Title: chosen candidate from SHOW_HN_DRAFT.md (must start `Show HN:`).
  \- URL: https://github.com/ksenxx/kiss_ai (the repo, not the site).
- [ ] Immediately post your hand-written first comment on the thread.
- [ ] Do NOT: ask anyone to upvote or comment (voting-ring detection), share
  direct upvote links (those votes don't count), post booster comments,
  delete-and-repost.

## T+0 to T+48h: the window

- [ ] Reply to **every** comment, personally, in your own words. Never paste
  AI-generated or AI-edited replies (explicit HN kill-trigger).
- [ ] Skeptical/technical comments are the good ones — answer with specifics
  (worktree mechanics, security model, LoC breakdown), the Superset way.
- [ ] Price/monetization complainers: it's free and Apache-2.0; say so once,
  politely.
- [ ] If the post stalls under ~30 points and slides off /shownew: it may be
  picked for the second-chance pool; you can also email
  hn@ycombinator.com once, politely.
- [ ] After HN traction is visible (not before), stagger the wave:
  \- [ ] Reddit posts, hours apart, tailored per subreddit.
  \- [ ] Newsletter submissions: TLDR, Console (OSS-specific), Techpresso.
  \- [ ] PR into awesome-ai-agents / awesome-llm lists.
  \- [ ] Publish EPISODE.md as a blog post (Dev.to/Hashnode canonical) and
  link the live HN thread in it.

## Metrics to log (fills Episode 2)

| Metric | Where | Target |
|---|---|---|
| HN points / comments at 24h, 48h | thread | front page of /show |
| GitHub stars/day during window | repo insights | ~100/day → Trending |
| pipx/PyPI installs delta | pypistats | baseline ×3 |
| VS Code Marketplace installs delta | marketplace | baseline ×2 |
| Referral traffic to kisssorcar.github.io | Pages analytics | log all |
| Episode-directory visits | GitHub traffic | log all |
| Feedback themes (security model, small-core claim) | thread notes | ≥ 5 actionable items |

## Post-window

- [ ] Reply to stragglers for another week (threads have long tails).
- [ ] Write Episode 2 from the metrics table + best thread exchanges.
- [ ] File issues for every actionable piece of feedback; tag `from-hn`.
- [ ] If it flopped: wait, ship something significantly new, and repost
  months later linking the previous thread (dang-sanctioned).

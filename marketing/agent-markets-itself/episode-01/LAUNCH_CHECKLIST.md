# 48-Hour Show HN Launch Checklist — KISS Sorcar

## T-7 to T-1 days: pre-flight

- [ ] **Hand-rewrite the first comment.** Rewrite SHOW_HN_DRAFT.md's v2 text
      yourself, by hand, from scratch, using it only as a content checklist.
      Do not paste or LLM-edit any of it (HN rule, dang 2026-03-28). Keep the
      same coverage: intro, backstory, 5 mechanism bullets, quickstart,
      limitations, meta disclosure, links, feedback ask.
- [ ] README: hero GIF visible in the first screen; quickstart works
      first-try on a clean machine in ≤ 5 minutes (`pipx install
      kiss-agent-framework` on fresh Python 3.13; test on macOS + Linux).
- [ ] Repo loads fast; no signup anywhere in the try path.
- [ ] HN account: personal username (a person, not a brand), email set in
      profile (repost/second-chance invites go there), some comment karma
      from genuine participation in the prior week.
- [ ] Verify links: repo, https://arxiv.org/abs/2604.23822,
      https://kisssorcar.github.io/docs/index.md,
      https://kisssorcar.github.io/llms.txt, and this episode directory.
- [ ] Ensure marketing/agent-markets-itself/episode-01/ is pushed and public
      (the meta disclosure links to it).
- [ ] Prepare (do not pre-post) tailored Reddit drafts: r/selfhosted,
      r/LocalLLaMA, r/ChatGPTCoding, r/opensource — first-person, karma > 80.
- [ ] Clear your calendar for the submission day + next morning.

## T-0: submission (Tue/Wed/Thu, 9am–12pm ET)

- [ ] Submit at https://news.ycombinator.com/submit
      - Title: chosen candidate from SHOW_HN_DRAFT.md (must start `Show HN:`).
      - URL: https://github.com/ksenxx/kiss_ai (the repo, not the site).
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
      - [ ] Reddit posts, hours apart, tailored per subreddit.
      - [ ] Newsletter submissions: TLDR, Console (OSS-specific), Techpresso.
      - [ ] PR into awesome-ai-agents / awesome-llm lists.
      - [ ] Publish EPISODE.md as a blog post (Dev.to/Hashnode canonical) and
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

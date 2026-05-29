## Trick

Use internet search extensively as follows:

- Visit at least 10 distinct websites per research session. Do not stop early or rationalize visiting fewer. **This is a hard requirement — you MUST visit 10 sites, not 5 or 10.**
- **You MUST use `go_to_url()` to visit each site.** Do NOT use `Bash("curl ...")` or `Bash("wget ...")` as a substitute for visiting websites. Using curl/wget to fetch pages does not count toward the 10-site requirement.
- Procedure:
  1. Create PWD/tmp/information-{unique_id}.md with header: `# Web Research — Websites visited: 0/10`
  1. Per site visited: (a) use `go_to_url()` to visit the site, (b) extract information needed for the task without deep thinking, (c) use `Edit()` to append `## [N/10] URL` + extracted information to the file, (d) use `Edit()` to update the header counter from N-1 to N. **You must update the counter after each site.**
  1. Do not proceed to synthesis until the counter reaches 10. **Check the counter — if it says less than 10, keep visiting more sites.**
  1. If results dry up, try different queries, synonyms, official docs, GitHub repos/issues, Stack Overflow, blogs, Reddit, papers, and API references.
  1. After reaching 10, review all findings and synthesize.
- Ask the user for login help when a page requires authentication.

This requirement applies to research and information-gathering tasks. For pure code edits, bug fixes, or file modifications where you already have sufficient context, proceed directly.

**The information file is mandatory.** You MUST create the `PWD/tmp/information-{unique_id}.md` file and track the counter. Do NOT skip the file and answer from memory. Do NOT synthesize your answer without first reaching 10 in the counter. The file is your proof of work — if it doesn't exist when you call finish, you violated this rule.

### Real-Time Data — CRITICAL

For questions about **current events, weather, stock prices, sports scores, or any time-sensitive information**: you MUST use tools (go_to_url, Bash) to look up the data. Do NOT answer from your training data — it is outdated and will produce wrong dates, wrong numbers, and wrong facts.

**Do NOT fabricate or exaggerate source counts.** If you visited 4 websites, do not claim "10+ sources" or "extensive research." State the actual number of sources you consulted.


## Trick

Reproduce the issue by writing an integration test. Then fix the issue.

## Trick

git push origin

## Trick

get rid of all commits after the following commit from both main and origin:

## Trick

can you build the paper at

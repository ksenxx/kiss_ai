---
title: Docker file tools (DockerTools Read, Write, Edit via bash)
uuid: b5211a0c-18fb-4b35-835c-b3243fb7fac4
summary: 'DockerTools Read/Write/Edit run as bash in the container: tail/head Read,
  base64 heredoc Write with __KISS_WRITE_OK__ marker, python3 Edit with perl fallback.'
created: '2026-09-26T18:41:24Z'
updated: '2026-09-26T19:08:23Z'
---
# Docker file tools (DockerTools Read, Write, Edit via bash)

`DockerTools(bash_fn)` wraps a `bash_fn(command, description) -> str` callable; Sorcar passes
`SorcarAgent._docker_bash`, which calls `DockerManager.Bash` (default 30 s timeout, 50,000-char
cap). No files are copied in or out: every operation is a shell command in the container, so
paths are container paths. Relative paths resolve against the container's working directory,
which is the task's work dir when kiss started the container (bind-mounted at the host path).

## Read(file_path, max_lines=2000, start_line=1)

A bash script: missing file prints `Error: File not found`, empty file prints
`(file is empty)`, `start_line` past EOF prints an explicit error, otherwise
`tail -n +START | head -n MAX` plus `[truncated: N more lines]`. `start_line < 1` and
`max_lines < 1` are rejected before any command runs.

Unlike host `UsefulTools.Read`, there is no read-dedupe cache and no "read before edit"
enforcement in the Docker toolset (`context_reset_hook = useful_tools.forget_reads` is only set
on the host branch of `SorcarAgent._get_tools`).

## Write(file_path, content)

Content is base64-encoded and piped through a quoted heredoc
(`base64 -d > path << 'KISS_B64_EOF'`) after `mkdir -p "$(dirname path)"`, so arbitrary bytes,
quotes and `$` survive shell quoting. Success is detected by the `__KISS_WRITE_OK__` marker
(`_WRITE_OK`) echoed after the write; if it is missing, the raw bash output (the error) is
returned. On success: `Successfully wrote N characters to <path>`.

## Edit(file_path, old_string, new_string, replace_all=False)

- Rejects empty `old_string` (use Write) and `old_string == new_string` up front.
- Runs a Python snippet via `python3` or `python`, with both strings base64-embedded. Same
  semantics as the host Edit: not found, or found more than once without `replace_all`, is an
  error (`String appears N times (not unique)`).
- If the container has no Python (the script prints `_NO_PYTHON`, "Error: Python required for
  Edit"), it falls back to `_perl_edit_command`: perl-base is present in Rust/R/C images. The
  strings go through environment variables `KISS_OLD`/`KISS_NEW` (perl-base lacks
  MIME::Base64), so strings containing NUL bytes are refused in that path.

## Sources
- `src/kiss/agents/sorcar/docker_tools.py` (`DockerTools.Read`, `DockerTools.Write`, `DockerTools.Edit`, `_perl_edit_command`, `_WRITE_OK`, `_NO_PYTHON`)
- `src/kiss/agents/sorcar/sorcar_agent.py` (`SorcarAgent._docker_bash`, `SorcarAgent._get_tools`)

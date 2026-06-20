# PROGRESS

## Task

> why codex and claude are not in my PATH?

## Status: DONE

## Work completed

1. Read `SORCAR.md` first as required; it is empty.
2. Checked the current Sorcar process environment. In this shell, `PATH` is:

   ```text
   /opt/homebrew/bin:/Users/ksen/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
   ```

   and both commands resolve:

   ```text
   /opt/homebrew/bin/codex
   /Users/ksen/.local/bin/claude
   ```

3. Checked common shells. `zsh -lc`, `zsh -ic`, `bash -lc`, and `sh -lc` all resolve both `codex` and `claude` here. Bash/sh do print a separate warning because `~/.profile` and `~/.bash_profile` source missing `/Users/ksen/.local/bin/env`, but their PATH still includes both required directories.
4. Inspected the KISS daemon LaunchAgent. `~/Library/LaunchAgents/com.kiss.web-server.plist` explicitly sets:

   ```text
   PATH=/opt/homebrew/bin:/Users/ksen/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
   ```

   `launchctl print gui/$(id -u)/com.kiss.web-server` reports the daemon is running as PID 66936 with the same PATH. This explains why the current backend can resolve both commands.
5. Found the relevant code paths:
   - `src/kiss/core/models/codex_model.py` locates `codex` with `shutil.which("codex")`, falling back to known desktop-app paths.
   - `src/kiss/core/models/claude_code_model.py` locates `claude` with `shutil.which("claude")` only.
   - `src/kiss/agents/vscode/src/DependencyInstaller.ts` writes the LaunchAgent PATH as `/opt/homebrew/bin:${HOME}/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin` on macOS.

## Conclusion

On this machine/session, `codex` and `claude` are in PATH for Sorcar and for normal shell probes. If the UI previously reported they were missing, the likely causes are:

1. the daemon was started before the current LaunchAgent PATH fix and needs a safe restart/reload;
2. a different process/session is being checked than PID 66936;
3. VS Code or an already-open terminal has an older inherited PATH;
4. `claude` depends on `/Users/ksen/.local/bin`, and that directory is added in `.zshrc` rather than `.zprofile`, so login/non-interactive contexts that do not use the KISS LaunchAgent PATH could miss it.

## Useful facts

```text
codex  -> /opt/homebrew/bin/codex
claude -> /Users/ksen/.local/bin/claude
```

The installed symlinks are:

```text
/opt/homebrew/bin/codex -> /opt/homebrew/Caskroom/codex/0.128.0/codex-aarch64-apple-darwin
/Users/ksen/.local/bin/claude -> /Users/ksen/.local/share/claude/versions/2.1.114
```

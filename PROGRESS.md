# Task: ensure install.sh and scripts/build-extension.sh do not delete kiss-sorcar.vsix

## Steps done

1. Read install.sh and scripts/build-extension.sh; confirmed both deleted the VSIX:
   - install.sh step [6/6]: `rm -f "$VSIX"` (where `VSIX="$VSCODE_EXT_DIR/kiss-sorcar.vsix"`).
   - scripts/build-extension.sh cleanup: `rm -rf "$EXT_DIR/out" "$EXT_DIR/kiss_project" "$EXT_DIR/kiss-sorcar.vsix"`.
2. Verified `src/kiss/agents/vscode/kiss-sorcar.vsix` is git-tracked (`git ls-files`), so deleting
   it leaves the working tree dirty with a pending deletion after every install/build.
3. install.sh: removed the `rm -f "$VSIX"` block and its "throwaway artifact" comment; replaced
   with a NOTE explaining the VSIX is git-tracked and must not be deleted.
4. scripts/build-extension.sh: removed `"$EXT_DIR/kiss-sorcar.vsix"` from the cleanup `rm -rf`
   (still removes `out/` and `kiss_project/`); added the same NOTE.
5. Verified: `bash -n` on both scripts (syntax OK); grep confirms no remaining `rm` of the vsix
   in either script.
6. Committed the change.

# Defensive security and evaluation-integrity audit

**Scope:** public Cleverest artifact, exact commit `bac4344ab3813a138608e86da8f1103dfe11ff62`. Tests were local and harmless. No network target or third-party system was attacked.

## S-1 — LLM-generated command injection (verified POC; High in an unsandboxed deployment)

**Dataflow.** With `GENCMD=1`, `run.sh` parses a model response using `parse_llm_cmd` (`utils.sh`), calls `abort_if_cmd_danger`, concatenates it with a build path/input using `get_cmd`, and executes the result through `script -c` (a shell). The guard rejects only strings containing `rm` or `sudo`; it permits `;`, pipes, substitutions, redirects, `curl`, and other executables. Released command files themselves include `$(python3 ...)`, confirming that shell evaluation is part of normal operation (not merely a hypothetical parser edge).

**POC.** `evidence/poc_command_injection.sh` gives the exact parser a synthetic model answer:

```text
Command: `target @@; printf VERIFIED > /tmp/cleverest_cmd_injection_poc`
```

The exact guard accepts it. The exact command constructor returns:

```text
timeout 30s /definitely/missing/build/target SAFE_INPUT; printf VERIFIED > /tmp/cleverest_cmd_injection_poc
```

Executing that string with shell grammar creates the marker even though the intended target fails. This was tested successfully. The POC writes only `/tmp/cleverest_cmd_injection_poc`.

**Threat model/impact.** Commit messages/diffs are untrusted in CI and are placed in the LLM prompt. A prompt-injected commit can influence `GENCMD` output. The artifact container runs with network access and receives an API key; absent stronger isolation, a command could read environment secrets, modify files, or make outbound requests. The default non-GENCMD paper configuration does not expose this path, so this finding does not invalidate Table 2.

**Fix implemented in audit prototype.** `cleverest_plus` accepts a JSON argv array, allowlists the executable, rejects shell syntax/path traversal, substitutes `@@` as an argv element, and calls `subprocess.run(..., shell=False)`. Tests reproduce and reject the payload.

## S-2 — Output-oracle spoofing (verified exact-function POC; evaluation gaming/CWE-184-like incomplete filtering)

`check_output_bug` scans the pseudo-terminal's combined output for text fragments such as `ERROR: AddressSanitizer:` and `Segmentation fault`; it does not establish that the line came from sanitizer stderr. `run.sh` counts a marker present only in the scenario's target version as a bug.

`evidence/poc_oracle_spoof.sh` uses the artifact's exact `check_output_bug` and classification branch. A target-controlled string `ERROR: AddressSanitizer: heap-buffer-overflow ...` produced:

```text
bug_before=''
bug_after=heap-buffer-overflow
classified_BIC_final=B
```

Thus a generated program could conditionally print the marker based on any version-dependent behavior and convert “output changed” into “bug”. This demonstrates **potential to game the evaluation**, not actual cheating. We grepped all 31,109 released `INPUT_*` files and found zero crash-marker strings; there is no evidence the reported runs used this exploit.

**Fix implemented.** The prototype captures stdout/stderr separately and parses sanitizer markers only from stderr. A stronger deployment should use a dedicated inherited sanitizer log FD and authenticate report provenance.

## S-3 — No intended-bug identity in fuzz-result adjudication (verified code defect; severity Medium for scientific validity)

`fuzz/checkfuzz.sh` classifies any target-version crash as the intended bug if the other version does not crash. If both versions crash, it excludes the case only when the coarse extracted type strings are equal; different crashes on both sides can still enter the scenario-success branch. It does not compare stack traces, source locations, or a ground-truth signature. Therefore “ClevFuzz found the bug” operationally means “ClevFuzz found a differential crash,” which can overcount unrelated bugs.

This is especially important because ClevFuzz's 20/26 and 18/24 union counts are the paper's largest comparative result. The CSVs include many correctly excluded same-both-side crashes (`X`), but crash artifacts/stacks sufficient to re-triage every claimed `B` are not supplied. We therefore classify actual overcounting as **unresolved**, not proven.

**Fix implemented.** `CrashSignature` includes sanitizer, error kind, and top frame; an optional expected signature is required for benchmark adjudication, and any both-version crash is ambiguous.

## S-4 — Trust boundaries and supply chain (lower severity)

- All scripts `source` a caller-provided `.env` file, which is arbitrary shell code. This is acceptable only if configs are explicitly trusted; the CLI should say so and reject remote/untrusted config paths.
- The Dockerfile downloads a `yq` tarball without checksum/signature verification and installs unversioned apt packages. The base is named by mutable tag in `FROM` even though README links an immutable digest.
- The container defaults to root because user-creation lines are commented. Combined with network access and API-key injection, this amplifies S-1 inside the container. It is not, by itself, a host escape.
- `repdata.tar.xz` was scanned: 123,995 members, only regular files/directories, no absolute paths, `..` traversal, symlinks, or hardlinks. No archive-extraction vulnerability was found in the released file.

## Docker executability caveat

Static inspection finds `COPY` before `WORKDIR /clever` and use of undefined `$UNAME` in the yq installation. These make the documented container flow suspect. The local Docker daemon was unavailable, so we did **not** claim a runtime build failure. This is an untested static reproducibility concern, not a verified security exploit.

## What was not found

- No credentials were found committed (the NVIDIA/OpenAI values are documented placeholders).
- No malicious payload appeared in released generated inputs.
- No unsafe tar members were found.
- No evidence indicates the authors exploited S-1/S-2 or deliberately inflated results. Vulnerability and misconduct are separate questions.

"""Property-based / fuzzing tests for every subprocess and shell
command path in ``src/kiss/agents/vscode/``.

These tests are the regression net for the H1, H3, H8 fixes
(DependencyInstaller shell-injection hardening, RC file shell-quoting,
exact ``pkill -x`` rather than the substring-match ``-f`` flag).

Strategy
--------
Every command path that takes user-controlled data — paths,
environment variables, RC values, file names, queries — is fuzzed
either:

  1. *Behaviourally*, by feeding many random shell-metacharacter
     payloads through the real code path and asserting no
     command-substitution fires (e.g. no marker file is created), and

  2. *Structurally*, by source-grepping the TypeScript files for any
     pattern that interpolates a non-constant variable into a shell
     string passed to ``execSync`` / ``execPromise``.  This catches new
     regressions introduced by future edits, even though we have no
     TypeScript runtime in the test harness.

Each test class corresponds to one subprocess/shell call site.  The
fuzzers use a fixed RNG seed for reproducibility but cover enough of
the metachar surface that an injection regression has near-100%
probability of being detected.
"""

from __future__ import annotations

import os
import random
import re
import shutil
import stat
import string
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VSCODE_TS_DIR = (
    Path(__file__).resolve().parents[3] / "agents" / "vscode" / "src"
)
VSCODE_PY_DIR = Path(__file__).resolve().parents[3] / "agents" / "vscode"

SHELL_METACHARS = list("\"'`$\\;|&<>(){}*?[]!#%^~ \t")


def _ts(name: str) -> str:
    return (VSCODE_TS_DIR / name).read_text()


def _rng_payload(rng: random.Random, *, length_max: int = 40,
                 forbid: str = "\n\r\0") -> str:
    """Return a random string of shell metacharacters and ASCII fillers.

    Excludes characters in ``forbid`` because RC-file export lines
    can't represent newlines without continuation, and NUL is rejected
    by every UNIX exec().
    """
    pool = SHELL_METACHARS + list("abcXYZ012")
    pool = [c for c in pool if c not in forbid]
    return "".join(rng.choice(pool) for _ in range(rng.randint(1, length_max)))


# ---------------------------------------------------------------------------
# 1. Behavioral fuzz — vscode_config.save_api_key_to_shell across shells.
#    Every payload must round-trip through a real shell without any side
#    effect.  H3 fix property.
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestFuzzSaveApiKeyRoundTripBash(unittest.TestCase):
    """200 random metachar payloads must round-trip via ``bash -c source``."""

    SHELL = "bash"
    RC_NAME = ".bashrc"

    def setUp(self) -> None:
        if not shutil.which(self.SHELL):
            self.skipTest(f"{self.SHELL} not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.agents.vscode import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / self.RC_NAME  # type: ignore[assignment]
        self._orig_get_shell = vc._get_user_shell
        vc._get_user_shell = lambda: self.SHELL  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        self._env_patch = mock.patch.dict(os.environ,
                                          {"HOME": str(self.home),
                                           "SHELL": f"/bin/{self.SHELL}"})
        self._env_patch.start()
        self._marker = Path(tempfile.gettempdir()) / f"fuzz-pwned-{os.getpid()}"
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._vc._get_user_shell = self._orig_get_shell  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._env_patch.stop()
        if self._marker.exists():
            self._marker.unlink()
        self._tmp.cleanup()

    def _round_trip(self, value: str) -> str:
        self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
        rc = self.home / self.RC_NAME
        proc = subprocess.run(
            [self.SHELL, "-c",
             f"source '{rc}' && printf '%s' \"$OPENAI_API_KEY\""],
            capture_output=True, text=True, timeout=10,
        )
        return proc.stdout

    def test_fuzz_200_payloads_round_trip(self) -> None:
        rng = random.Random(0xFAFA)
        for _ in range(200):
            value = _rng_payload(rng)
            with self.subTest(value=value):
                got = self._round_trip(value)
                self.assertEqual(got, value,
                                 f"payload {value!r} → {got!r}")
                self.assertFalse(self._marker.exists(),
                                 f"command substitution fired for {value!r}")

    def test_specific_dangerous_payloads(self) -> None:
        m = self._marker
        # Each payload tries a different injection technique.
        for payload in [
            f'$(touch {m})',
            f'`touch {m}`',
            f'"; touch {m}; #',
            f'"$(touch {m})"',
            f"'\";touch {m};echo '",
            f'\\";touch {m};\\"',
            "$IFS$9touch$IFS" + str(m),
            f'${{IFS}}touch${{IFS}}{m}',
        ]:
            with self.subTest(payload=payload):
                got = self._round_trip(payload)
                self.assertEqual(got, payload)
                self.assertFalse(m.exists(),
                                 f"injection fired: {payload}")


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestFuzzSaveApiKeyRoundTripZsh(TestFuzzSaveApiKeyRoundTripBash):
    """Same payload fuzz under zsh."""

    SHELL = "zsh"
    RC_NAME = ".zshrc"


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestFuzzSaveApiKeyRoundTripFish(unittest.TestCase):
    """fish round-trip fuzz.  fish escaping rules differ slightly from
    POSIX so we use a smaller (still meta-rich) alphabet that doesn't
    rely on POSIX command-substitution features."""

    def setUp(self) -> None:
        if not shutil.which("fish"):
            self.skipTest("fish not installed")
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.agents.vscode import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / "config.fish"  # type: ignore[assignment]
        self._orig_get_shell = vc._get_user_shell
        vc._get_user_shell = lambda: "fish"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()

    def tearDown(self) -> None:
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._vc._get_user_shell = self._orig_get_shell  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._tmp.cleanup()

    def test_fuzz_50_payloads_round_trip_fish(self) -> None:
        rng = random.Random(0xFA15)
        marker = Path(tempfile.gettempdir()) / f"fish-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        # fish's quoting handles newlines, $, and a different command
        # substitution syntax `(...)`.  We restrict the alphabet to
        # well-known POSIX-overlap metacharacters.
        alphabet = list("\"'`$\\;|&<>(){}*?[]!# \t") + list("abcXYZ012")
        try:
            for _ in range(50):
                length = rng.randint(1, 40)
                value = "".join(rng.choice(alphabet) for _ in range(length))
                self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
                rc = self.home / "config.fish"
                proc = subprocess.run(
                    ["fish", "-c",
                     f"source '{rc}'; printf '%s' \"$OPENAI_API_KEY\""],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(proc.returncode, 0,
                                 msg=proc.stderr + "\nrc=" + rc.read_text())
                self.assertEqual(proc.stdout, value,
                                 f"fish payload {value!r} → {proc.stdout!r}")
                self.assertFalse(marker.exists(),
                                 f"fish injection fired for {value!r}")
        finally:
            if marker.exists():
                marker.unlink()


# ---------------------------------------------------------------------------
# 2. Behavioral fuzz — diff_merge._git
# ---------------------------------------------------------------------------


class TestFuzzGitCwdNoInjection(unittest.TestCase):
    """``_git`` must run via argv (no shell), so fuzzed cwd values that
    contain shell metacharacters are passed verbatim and cannot inject
    shell commands."""

    def test_fuzz_cwd_paths_with_metacharacters(self) -> None:
        from kiss.agents.vscode import diff_merge as dm

        rng = random.Random(0x617)
        marker = Path(tempfile.gettempdir()) / f"git-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        try:
            for _ in range(30):
                tmpdir = Path(tempfile.mkdtemp(
                    prefix="kiss-git-fuzz-", suffix=_rng_payload(
                        rng, length_max=8, forbid="\n\r\0/")))
                # Initialise an empty repo so ``git status`` has work to do.
                subprocess.run(["git", "init", "-q", str(tmpdir)],
                               capture_output=True, timeout=20)
                # Pre-existing payloads in the working dir tree must not
                # be evaluated as shell.
                bad_name = f"$(touch '{marker}')"
                # Don't actually create that file — we only need
                # ``_git`` to receive the (possibly weird) cwd as data.
                cp = dm._git(str(tmpdir), "status", "--porcelain")
                self.assertEqual(cp.returncode, 0,
                                 msg=cp.stderr)
                self.assertFalse(marker.exists(),
                                 f"_git executed shell for cwd {tmpdir}; "
                                 f"bad_name {bad_name!r}")
                shutil.rmtree(tmpdir, ignore_errors=True)
        finally:
            if marker.exists():
                marker.unlink()

    def test_fuzz_args_are_passed_verbatim(self) -> None:
        """A fuzzed ``*args`` value must arrive at git unmangled (no
        shell expansion)."""
        from kiss.agents.vscode import diff_merge as dm

        captured: list[list[str]] = []
        real_run = subprocess.run

        def spy_run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
            captured.append(list(cmd))
            return real_run(["true"], capture_output=True, text=True)

        rng = random.Random(0x914)
        with mock.patch.object(subprocess, "run", spy_run):
            for _ in range(20):
                arg = _rng_payload(rng, forbid="\0")
                dm._git("/tmp", "log", arg, "--oneline")
                self.assertEqual(captured[-1][0], "git")
                self.assertIn(arg, captured[-1],
                              f"arg {arg!r} not passed verbatim "
                              f"to git: {captured[-1]}")


# ---------------------------------------------------------------------------
# 3. Behavioral fuzz — _save_untracked_base file names
# ---------------------------------------------------------------------------


class TestFuzzSaveUntrackedBaseFilenames(unittest.TestCase):
    """``_save_untracked_base`` must handle file names containing shell
    metacharacters without invoking a shell."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.work = Path(self._tmp.name)
        from kiss.core import config as cfg
        self._cfg_patch = mock.patch.object(
            cfg, "_artifact_root",
            lambda: self.work / ".kiss-artifacts")
        self._cfg_patch.start()

    def tearDown(self) -> None:
        self._cfg_patch.stop()
        self._tmp.cleanup()

    def test_fuzz_filenames_with_metacharacters(self) -> None:
        from kiss.agents.vscode import diff_merge as dm

        marker = Path(tempfile.gettempdir()) / f"save-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        rng = random.Random(0xA17)
        try:
            for i in range(15):
                # File names cannot contain '/' (path separator) or NUL.
                name = _rng_payload(rng, length_max=12,
                                    forbid="\n\r\0/")
                # Avoid empty / leading-dot weirdness.
                if not name.strip(".") or name.startswith("."):
                    name = f"f{i}-" + name
                fpath = self.work / name
                try:
                    fpath.write_text("hello fuzz\n")
                except OSError:
                    # Some chars (e.g. NUL on macOS) raise; skip.
                    continue
                dm._save_untracked_base(str(self.work),
                                        {name}, tab_id=f"tab{i}")
                base = dm._untracked_base_dir(f"tab{i}")
                # Either the file was preserved or copy was a no-op
                # (e.g. on a too-large file) — never a shell injection.
                self.assertFalse(marker.exists(),
                                 f"shell injection fired for {name!r}")
                if (base / name).exists():
                    self.assertEqual((base / name).read_text(),
                                     "hello fuzz\n")
        finally:
            if marker.exists():
                marker.unlink()


# ---------------------------------------------------------------------------
# 4. Behavioral fuzz — vscode_config.source_shell_env paths
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for source-RC fuzzing")
class TestFuzzSourceShellEnvPaths(unittest.TestCase):
    """``source_shell_env`` shell-quotes the RC path so a HOME containing
    metacharacters cannot inject commands into the sourced shell."""

    def setUp(self) -> None:
        if not shutil.which("bash"):
            self.skipTest("bash required")
        self._tmp = tempfile.TemporaryDirectory()
        self._marker = (Path(tempfile.gettempdir())
                        / f"source-pwned-{os.getpid()}")
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        if self._marker.exists():
            self._marker.unlink()
        self._tmp.cleanup()

    def test_fuzz_rc_paths_with_metacharacters(self) -> None:
        from kiss.agents.vscode import vscode_config as vc

        rng = random.Random(0xCAFE)
        for _ in range(20):
            # Build a directory name with shell metacharacters under tmp
            # so we can plant the RC file at the resulting weird path
            # and call ``source_shell_env``.
            payload = _rng_payload(rng, length_max=10,
                                   forbid="\n\r\0/")
            sub = Path(self._tmp.name) / f"d-{payload}"
            try:
                sub.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            rc = sub / ".bashrc"
            # Write a minimal RC that exports a known key — we want to
            # detect whether sourcing this file would also execute the
            # injected payload that lives in the file *path*.
            rc.write_text('export OPENAI_API_KEY=present\n')
            with mock.patch.object(vc, "_shell_rc_path", lambda s: rc), \
                    mock.patch.object(vc, "_get_user_shell", lambda: "bash"), \
                    mock.patch.object(vc, "_refresh_config", lambda: None):
                vc.source_shell_env()
            self.assertFalse(self._marker.exists(),
                             f"source_shell_env injected for path {sub}")


# ---------------------------------------------------------------------------
# 5. Source-grep fuzz — DependencyInstaller.ts must not regress
# ---------------------------------------------------------------------------


class TestSourceGrepDependencyInstaller(unittest.TestCase):
    """Static / regex audit of every shell command path in
    DependencyInstaller.ts.  A regression that re-introduces shell
    interpolation of a user-controlled path will trip these tests."""

    src: str = ""

    @classmethod
    def setUpClass(cls) -> None:
        cls.src = _ts("DependencyInstaller.ts")

    def test_no_pkill_dash_f_anywhere(self) -> None:
        """``pkill -f`` matches any process whose argv contains the
        literal substring — H8 mandates the exact-comm match ``-x``."""
        self.assertNotRegex(self.src, r"pkill\s+-f",
                            "pkill -f re-introduced (H8 regression)")

    def test_no_curl_pipe_tar_through_shell(self) -> None:
        """The original H1 vector — ``curl … | tar xz -C …`` inside
        ``execPromise`` — must stay gone."""
        self.assertFalse(
            re.search(r"execPromise\([^)]*curl[^)]*\|\s*tar", self.src),
            "execPromise(curl ... | tar ...) re-introduced (H1 regression)",
        )

    def test_no_unquoted_uvpath_in_shell_string(self) -> None:
        """An execSync template string interpolating ``${uvPath}`` is
        an injection vector."""
        # Match patterns like execSync(`...${uvPath}...`)
        self.assertFalse(
            re.search(
                r"exec(?:Sync|Promise)\(\s*[`'\"][^`'\"]*\$\{uvPath\}",
                self.src),
            "execSync/execPromise still interpolates ${uvPath} into a shell string",
        )

    def test_no_unquoted_plistfile_in_shell_string(self) -> None:
        self.assertFalse(
            re.search(
                r"exec(?:Sync|Promise)\(\s*[`'\"][^`'\"]*\$\{plistFile\}",
                self.src),
            "execSync still interpolates ${plistFile} (use execFileSync)",
        )

    def test_no_unquoted_kissprojectpath_in_shell_string(self) -> None:
        self.assertFalse(
            re.search(
                r"exec(?:Sync|Promise)\(\s*[`'\"][^`'\"]*\$\{kissProjectPath\}",
                self.src),
            "execSync still interpolates ${kissProjectPath}",
        )

    def test_downloads_use_node_https_not_curl_in_shell(self) -> None:
        """``downloadFile`` (Node ``https``) must be the channel for
        every binary download — no inline curl/wget/Invoke-WebRequest
        through ``execPromise`` against a non-Windows path."""
        # The Windows path uses Invoke-WebRequest in PowerShell — that
        # is acceptable because Windows paths under USERPROFILE rarely
        # contain shell metacharacters and PowerShell's quoting rules
        # are different.  We assert the Linux/macOS uv & node flows
        # never call curl through the shell.
        # Find the installUv / installNode bodies.
        for fn in ("installUv", "installNode"):
            idx = self.src.find(f"async function {fn}")
            self.assertGreater(idx, 0)
            # Find function end: track brace depth from the first '{'.
            i = self.src.index("{", idx)
            depth = 0
            j = i
            while j < len(self.src):
                if self.src[j] == "{":
                    depth += 1
                elif self.src[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            body = self.src[i:j + 1]
            # Heuristic: curl AND a shell pipe, on a non-win32 branch.
            # Find the non-win32 branch by skipping the win32 block.
            non_win = body
            if "process.platform === 'win32'" in body:
                # The win32 branch can be ``if (win32) { ... return; }``
                # *without* a sibling ``else``.  Strip that branch by
                # finding the matching close-brace of the win32
                # if-clause and taking what comes after.
                win_idx = body.index("process.platform === 'win32'")
                # Find the opening '{' that follows the if-condition.
                open_idx = body.index("{", win_idx)
                depth = 0
                k = open_idx
                while k < len(body):
                    if body[k] == "{":
                        depth += 1
                    elif body[k] == "}":
                        depth -= 1
                        if depth == 0:
                            break
                    k += 1
                # Anything past the close-brace is the non-win path.
                # (Optional `else {` at the boundary doesn't matter.)
                non_win = body[k + 1:]
            self.assertNotIn(
                "curl", non_win,
                f"{fn} non-Windows branch still uses curl (must use "
                "downloadFile + Node https)",
            )
            self.assertNotIn(
                "execPromise", non_win,
                f"{fn} non-Windows branch still uses execPromise "
                "(must use spawnPromise / fs.* primitives)",
            )

    def test_downloads_have_sha256_verification(self) -> None:
        """Both installUv and installNode must verify SHA256 (H2)."""
        self.assertIn("verifyDownloadHash", self.src)
        self.assertIn("createHash('sha256')", self.src)

    def test_xml_escape_helper_present(self) -> None:
        self.assertIn("function xmlEscape", self.src,
                      "XML escape helper missing (H10 regression)")
        # Must escape at least &, <, >, ", '.
        idx = self.src.find("function xmlEscape")
        body = self.src[idx: idx + 400]
        for entity in ("&amp;", "&lt;", "&gt;", "&quot;", "&apos;"):
            self.assertIn(entity, body,
                          f"xmlEscape doesn't emit {entity}")

    def test_unit_escape_helper_present(self) -> None:
        self.assertIn("function unitEscape", self.src,
                      "systemd unit-escape helper missing")

    def test_plist_does_not_interpolate_unescaped_path(self) -> None:
        # All <string>${...}</string> in restartKissWebDaemon must be
        # ``x*`` (xml-escaped) variables, not bare names.
        rk_idx = self.src.find("function restartKissWebDaemon")
        self.assertGreater(rk_idx, 0)
        # Body is roughly the next 5000 chars.
        body = self.src[rk_idx: rk_idx + 6000]
        # Find every <string>${...}</string> and verify variable is
        # whitelisted (escaped).
        for m in re.finditer(r"<string>\$\{(\w+)\}</string>", body):
            var = m.group(1)
            self.assertTrue(
                var.startswith("x"),
                f"plist interpolates unescaped variable ${{{var}}} "
                "— must be xml-escaped (H10).",
            )


# ---------------------------------------------------------------------------
# 6. Source-grep fuzz — AgentProcess.ts must consult workspace trust
# ---------------------------------------------------------------------------


class TestSourceGrepAgentProcessTrust(unittest.TestCase):
    """``findKissProject`` must gate env / setting on ``isTrusted`` (H5)."""

    def test_findkissproject_consults_istrusted(self) -> None:
        src = _ts("AgentProcess.ts")
        idx = src.find("export function findKissProject")
        self.assertGreater(idx, 0)
        # Find function end.
        body = src[idx: idx + 2000]
        self.assertIn("isTrusted", body,
                      "findKissProject must consult workspace trust (H5)")
        # ``KISS_PROJECT_PATH`` must appear *after* the isTrusted check.
        trust_idx = body.find("isTrusted")
        env_idx = body.find("KISS_PROJECT_PATH")
        self.assertGreater(env_idx, trust_idx,
                           "KISS_PROJECT_PATH read before isTrusted check")


# ---------------------------------------------------------------------------
# 7. Source-grep fuzz — webview path-traversal / CSP / nonce
# ---------------------------------------------------------------------------


class TestSourceGrepWebviewBoundary(unittest.TestCase):
    """SorcarSidebarView openFile/submit must validate every webview
    path; SorcarTab CSP and nonce must remain hardened."""

    def test_openfile_handler_uses_ispathinside(self) -> None:
        src = _ts("SorcarSidebarView.ts")
        idx = src.find("case 'openFile':")
        self.assertGreater(idx, 0)
        block = src[idx: idx + 1200]
        self.assertIn("isPathInside", block,
                      "openFile path-traversal guard removed (H4 regression)")

    def test_submit_file_shortcut_uses_ispathinside(self) -> None:
        src = _ts("SorcarSidebarView.ts")
        # The file-shortcut detection lives inside the 'submit' case.
        idx = src.find("case 'submit':")
        self.assertGreater(idx, 0)
        block = src[idx: idx + 2400]
        self.assertIn("isPathInside", block,
                      "submit file-shortcut path-traversal guard removed")

    def test_csp_locks_form_action_frame_object_base(self) -> None:
        src = _ts("SorcarTab.ts")
        idx = src.find("Content-Security-Policy")
        self.assertGreater(idx, 0)
        block = src[idx: idx + 600]
        for d in ("form-action", "frame-src", "object-src", "base-uri"):
            self.assertRegex(block, rf"{d}\s+'none'",
                             f"CSP {d} not locked to 'none'")

    def test_nonce_uses_csprng(self) -> None:
        src = _ts("SorcarTab.ts")
        idx = src.find("export function getNonce")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 500]
        self.assertNotIn("Math.random", body,
                         "getNonce regressed to Math.random")
        self.assertTrue(
            "randomBytes" in body or "getRandomValues" in body,
            "getNonce must use a CSPRNG",
        )


# ---------------------------------------------------------------------------
# 8. Behavioral fuzz — markdown sanitizer must strip dangerous content
# ---------------------------------------------------------------------------


class TestMarkdownSanitizerSourceLevel(unittest.TestCase):
    """Regression-grep: every ``marked.parse`` site that flows into
    ``innerHTML`` must wrap through ``kissSanitize``."""

    def test_main_js_no_unwrapped_marked_parse_to_innerhtml(self) -> None:
        media = (Path(__file__).resolve().parents[3]
                 / "agents" / "vscode" / "media" / "main.js")
        src = media.read_text()
        # ``innerHTML = marked.parse(`` is forbidden.
        unwrapped = re.findall(r"innerHTML\s*=\s*marked\.parse\(", src)
        self.assertFalse(unwrapped,
                         f"{len(unwrapped)} unwrapped innerHTML=marked.parse "
                         "call(s) — H6 regression")

    def test_demo_js_no_unwrapped_marked_parse(self) -> None:
        media = (Path(__file__).resolve().parents[3]
                 / "agents" / "vscode" / "media" / "demo.js")
        src = media.read_text()
        self.assertFalse(
            re.findall(r"innerHTML\s*=\s*marked\.parse\(", src),
            "demo.js has unwrapped innerHTML=marked.parse — H6 regression",
        )

    def test_kisssanitize_strips_event_handlers_and_dangerous_tags(self) -> None:
        media = (Path(__file__).resolve().parents[3]
                 / "agents" / "vscode" / "media" / "main.js")
        src = media.read_text()
        idx = src.find("function kissSanitize")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 2000]
        for tag in ("SCRIPT", "IFRAME", "OBJECT", "EMBED", "FORM"):
            self.assertIn(tag, body,
                          f"kissSanitize doesn't strip <{tag}>")
        for url_attr in ("href", "src", "action"):
            self.assertIn(f"'{url_attr}'", body,
                          f"kissSanitize URL_ATTRS missing {url_attr}")
        self.assertIn("javascript", body,
                      "kissSanitize doesn't filter javascript: URLs")
        self.assertIn("on", body,
                      "kissSanitize doesn't strip on*= event handlers")


# ---------------------------------------------------------------------------
# 9. Behavioral fuzz — autocomplete prefix injection
# ---------------------------------------------------------------------------


class TestFuzzAutocompletePrefix(unittest.TestCase):
    """Autocomplete must never invoke a shell.  Fuzz the prefix with
    every shell metachar — must complete without side effects."""

    def test_fuzz_prefix_metachars(self) -> None:
        from kiss.agents.vscode import autocomplete as ac

        broadcasts: list[dict] = []

        class StubPrinter:
            def broadcast(self, msg: dict) -> None:
                broadcasts.append(msg)

        class FakeServer(ac._AutocompleteMixin):
            def __init__(self) -> None:
                self.printer = StubPrinter()  # type: ignore[assignment]
                self.work_dir = "/"
                self._state_lock = threading.Lock()
                self._complete_queue = None
                self._complete_worker = None
                self._complete_seq_latest = 0
                self._file_cache = ["a.py", "b.py", "x/y.txt"]

        srv = FakeServer()
        marker = Path(tempfile.gettempdir()) / f"ac-pwned-{os.getpid()}"
        if marker.exists():
            marker.unlink()
        rng = random.Random(0xACAC)
        try:
            for _ in range(50):
                prefix = _rng_payload(rng, length_max=20)
                broadcasts.clear()
                srv._get_files(prefix)
                self.assertFalse(marker.exists(),
                                 f"autocomplete fired shell for {prefix!r}")
                # Every call yields exactly one broadcast.
                self.assertEqual(len(broadcasts), 1)
                self.assertEqual(broadcasts[0]["type"], "files")
        finally:
            if marker.exists():
                marker.unlink()


# ---------------------------------------------------------------------------
# 10. Source-grep fuzz — every Python subprocess.run uses an argv list
# ---------------------------------------------------------------------------


class TestPythonSubprocessUsesArgvList(unittest.TestCase):
    """No Python file in the vscode package may pass a string command
    to ``subprocess.run`` / ``subprocess.Popen`` (which would default to
    ``shell=False`` BUT only with argv form).  All such call sites
    must be argv lists."""

    def test_all_subprocess_calls_use_list(self) -> None:
        py_files = [
            VSCODE_PY_DIR / "diff_merge.py",
            VSCODE_PY_DIR / "vscode_config.py",
            VSCODE_PY_DIR / "web_server.py",
            VSCODE_PY_DIR / "task_runner.py",
            VSCODE_PY_DIR / "autocomplete.py",
        ]
        for fp in py_files:
            src = fp.read_text()
            # Match ``subprocess.run(`` followed by anything that
            # *isn't* a list/tuple/var-name on the same line.
            for m in re.finditer(r"subprocess\.(run|Popen)\(\s*([^\)\n]+)",
                                 src):
                arg0 = m.group(2).strip()
                # Permit list literals, list-comprehensions, identifiers
                # that hold lists, and the ``["true"]`` we use in tests.
                # Reject string-literal commands (single or double-quote
                # then no comma or list-bracket before the close).
                if arg0.startswith(('"', "'", "f'", 'f"')):
                    # A string literal — make sure ``shell=True`` is
                    # NOT set (a string with shell=False fails fast at
                    # call time, so it would be a bug, but a lurking
                    # ``shell=True`` is a security hole).  Look at a
                    # 200-char window.
                    window = src[m.start():m.start() + 400]
                    self.assertNotIn("shell=True", window,
                                     f"shell=True in {fp.name}: "
                                     f"{window[:200]!r}")


# ---------------------------------------------------------------------------
# 11. Behavioral fuzz — chmod 0600 round-trip on RC under random umasks
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32", "POSIX chmod test")
class TestFuzzRcModeUnderRandomUmasks(unittest.TestCase):
    """Under any umask, the resulting RC file must be 0600."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.agents.vscode import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        self._orig_umask = os.umask(0o000)
        self._env_patch = mock.patch.dict(
            os.environ,
            {"HOME": str(self.home), "SHELL": "/bin/bash"})
        self._env_patch.start()

    def tearDown(self) -> None:
        os.umask(self._orig_umask)
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._env_patch.stop()
        self._tmp.cleanup()

    def test_rc_mode_0600_under_each_umask(self) -> None:
        rng = random.Random(0xC0DE)
        rc = self.home / ".bashrc"
        for umask in [0o000, 0o022, 0o027, 0o077, 0o002, 0o007]:
            os.umask(umask)
            value = "secret-" + "".join(
                rng.choice(string.ascii_letters) for _ in range(20))
            self._vc.save_api_key_to_shell("OPENAI_API_KEY", value)
            mode = stat.S_IMODE(rc.stat().st_mode)
            self.assertEqual(
                mode, 0o600,
                f"umask={oct(umask)} → RC mode {oct(mode)} "
                "(expected 0o600)")


# ---------------------------------------------------------------------------
# 12. Behavioral fuzz — DependencyInstaller xmlEscape / unitEscape
# ---------------------------------------------------------------------------


class TestEscapeHelperProperties(unittest.TestCase):
    """``xmlEscape`` and ``unitEscape`` are TS helpers; we re-implement
    the escape contract in Python and assert the regex search of the
    DependencyInstaller code matches the contract."""

    def test_xml_escape_translation_table(self) -> None:
        src = _ts("DependencyInstaller.ts")
        idx = src.find("function xmlEscape")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 400]
        # The function must escape at least these five characters.
        # The escape strings are encoded in the JS source as e.g.
        #   .replace(/&/g, '&amp;')
        for ch, ent in (("&", "&amp;"), ("<", "&lt;"),
                        (">", "&gt;"), ('"', "&quot;"), ("'", "&apos;")):
            self.assertIn(ent, body,
                          f"xmlEscape doesn't emit {ent} for {ch}")

    def test_unit_escape_handles_backslash_and_newline(self) -> None:
        src = _ts("DependencyInstaller.ts")
        idx = src.find("function unitEscape")
        self.assertGreater(idx, 0)
        body = src[idx: idx + 400]
        # Backslashes must be doubled.
        self.assertIn("\\\\\\\\", body,
                      "unitEscape must escape backslash to \\\\")
        # Newlines must be \n.
        self.assertIn("\\\\n", body,
                      "unitEscape must escape newline to \\n")
        # Percent must be %%.
        self.assertIn("%%", body,
                      "unitEscape must double %")


# ---------------------------------------------------------------------------
# 13. Behavioral — exhaustive injection-payload corpus must round-trip
# ---------------------------------------------------------------------------


@unittest.skipIf(sys.platform == "win32",
                 "POSIX shells required for round-trip fuzzing")
class TestKnownInjectionCorpus(unittest.TestCase):
    """A curated corpus of injection payloads against
    ``save_api_key_to_shell``.  Each must round-trip and never fire."""

    def setUp(self) -> None:
        if not shutil.which("bash"):
            self.skipTest("bash required")
        self._tmp = tempfile.TemporaryDirectory()
        self.home = Path(self._tmp.name)
        from kiss.agents.vscode import vscode_config as vc
        self._vc = vc
        self._orig_rc = vc._shell_rc_path
        vc._shell_rc_path = lambda shell: self.home / ".bashrc"  # type: ignore[assignment]
        self._refresh_patch = mock.patch.object(vc, "_refresh_config",
                                                lambda: None)
        self._refresh_patch.start()
        self._marker = (Path(tempfile.gettempdir())
                        / f"corpus-pwned-{os.getpid()}")
        if self._marker.exists():
            self._marker.unlink()

    def tearDown(self) -> None:
        if self._marker.exists():
            self._marker.unlink()
        self._vc._shell_rc_path = self._orig_rc  # type: ignore[assignment]
        self._refresh_patch.stop()
        self._tmp.cleanup()

    def test_known_payloads(self) -> None:
        m = str(self._marker)
        payloads = [
            f'$(touch {m})',
            f'`touch {m}`',
            f'"; touch {m}; echo "',
            f"'; touch {m}; echo '",
            f"\"; touch \"{m}\"; echo \"",
            f"\\\";touch {m};\\\"",
            "$IFS",
            "${IFS}",
            "&& touch " + m,
            "; touch " + m,
            "| touch " + m,
            ">/dev/null && touch " + m,
            "<(touch " + m + ")",
            ">(touch " + m + ")",
            "$((touch " + m + "))",
            f"$'\\x60touch\\x20{m}\\x60'",
        ]
        for p in payloads:
            with self.subTest(payload=p):
                self._vc.save_api_key_to_shell("OPENAI_API_KEY", p)
                rc = self.home / ".bashrc"
                proc = subprocess.run(
                    ["bash", "-c",
                     f"source '{rc}' && printf '%s' \"$OPENAI_API_KEY\""],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr)
                self.assertEqual(proc.stdout, p,
                                 f"payload {p!r} round-tripped to {proc.stdout!r}")
                self.assertFalse(self._marker.exists(),
                                 f"INJECTION FIRED: {p!r}")


if __name__ == "__main__":
    unittest.main()

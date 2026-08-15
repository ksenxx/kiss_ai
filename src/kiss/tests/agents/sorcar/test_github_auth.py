# Author: Koushik Sen (ksen@berkeley.edu)
# Contributors:
# Koushik Sen (ksen@berkeley.edu)
# add your name here
"""End-to-end tests for copying the GitHub.com credentials to a server.

Step 8 of ``./sorcar-cloud`` makes the agent on the remote machine the same
person on GitHub as the one who deployed it.  Two scripts do it, and the
tests here drive both of them for real -- real files, real ``git``, and
stand-in ``gh`` and ``curl`` programs on ``PATH`` that answer the way the
real ones do:

* ``scripts/collect-github-auth.sh``  -- finds the credentials on this
                                        machine, checks them against the
                                        GitHub API, and prints them where
                                        nothing else can see them;
* ``scripts/install-github-auth.sh``  -- logs those accounts in on the
                                        remote without losing the accounts
                                        it already had.

The two properties worth more than the features are asserted directly: a
token never becomes a command-line argument (which every account on a
machine can read), and nothing that was already on the remote is thrown
away.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[5]
_COLLECT = _ROOT / "scripts" / "collect-github-auth.sh"
_INSTALL = _ROOT / "scripts" / "install-github-auth.sh"

# A PATH without the machine's own gh or git: a test must never read the
# developer's real credentials, nor reach the real GitHub.
_BARE_PATH = "/usr/bin:/bin:/usr/sbin:/sbin"

# gh, as far as these tests need it to behave.  GH_FAKE_STATUS is the output
# of ``gh auth status``; GH_FAKE_TOKENS maps an account to its token; every
# call is appended to GH_FAKE_LOG as "<subcommand> | <arguments> | <stdin>",
# which is what lets a test see that the token arrived on standard input.
_FAKE_GH = r"""#!/bin/bash
sub="$1 $2"
shift 2 || true
case "$sub" in
    "auth login")
        if [ "${1:-}" = "--help" ]; then
            echo "      --insecure-storage   Save authentication credentials in plain text"
            exit 0
        fi
        token="$(cat)"
        if [ -n "${GH_TOKEN:-}${GITHUB_TOKEN:-}" ]; then
            echo "The value of the GH_TOKEN environment variable is being used." >&2
            exit 1
        fi
        echo "login | $* | $token" >> "$GH_FAKE_LOG"
        [ -n "${GH_FAKE_LOGIN_FAILS:-}" ] && { echo "HTTP 401: Bad credentials" >&2; exit 1; }
        [ "$token" = "${GH_FAKE_BAD_TOKEN:-}" ] && { echo "HTTP 401: Bad credentials" >&2; exit 1; }
        # The real gh writes the token it was given into hosts.yml, so the
        # file a second deploy finds is not the one the machine had of its own.
        mkdir -p "$HOME/.config/gh"
        printf 'github.com:\n    oauth_token: %s\n' "$token" > "$HOME/.config/gh/hosts.yml"
        exit 0
        ;;
    "auth setup-git")
        echo "setup-git | $* |" >> "$GH_FAKE_LOG"
        exit 0
        ;;
    "auth status")
        echo "status | $* |" >> "$GH_FAKE_LOG"
        [ -n "${GH_FAKE_STATUS:-}" ] && cat "$GH_FAKE_STATUS"
        exit 0
        ;;
    "auth token")
        echo "token | $* |" >> "$GH_FAKE_LOG"
        user=""
        while [ $# -gt 0 ]; do
            [ "$1" = "--user" ] && user="$2"
            shift
        done
        [ -n "${GH_FAKE_TOKENS:-}" ] || exit 1
        if [ -z "$user" ]; then
            head -1 "$GH_FAKE_TOKENS" | awk '{print $2}'
            exit 0
        fi
        line="$(awk -v u="$user" '$1 == u {print $2}' "$GH_FAKE_TOKENS")"
        [ -n "$line" ] || exit 1
        echo "$line"
        ;;
esac
"""

# curl, for the collector: the token arrives in a configuration file on
# standard input, and CURL_FAKE_ANSWERS says what GitHub replies to it
# ("<token> <http-code> <login>" per line).  Every argument of every call is
# recorded, so a test can insist the token is not among them.
_FAKE_CURL = r"""#!/bin/bash
echo "$*" >> "$CURL_FAKE_ARGS"
out=""
args=("$@")
i=0
while [ $i -lt ${#args[@]} ]; do
    [ "${args[$i]}" = "-o" ] && out="${args[$((i + 1))]}"
    i=$((i + 1))
done
config="$(cat)"
token="$(printf '%s' "$config" | sed -n 's/.*Bearer \([^"]*\)".*/\1/p')"
answer="$(awk -v t="$token" '$1 == t {print $2, $3}' "$CURL_FAKE_ANSWERS")"
code="${answer%% *}"
login="${answer#* }"
[ -n "$code" ] || code=000
[ -n "$out" ] && printf '{"login": "%s"}' "$login" > "$out"
printf '%s' "$code"
"""

# curl for the installer tests: there is no gh here and none can be fetched.
_FAILING_CURL = "#!/bin/bash\nexit 1\n"

# A chmod that cannot change the mode of hosts.yml, which is what that file
# being owned by somebody else -- root, say -- amounts to.
_CHMOD_REFUSING_HOSTS_YML = r"""#!/bin/bash
for arg in "$@"; do
    case "$arg" in *hosts.yml) exit 1 ;; esac
done
exec /bin/chmod "$@"
"""


def _run(
    *command: str, env: dict[str, str] | None = None, stdin: str = ""
) -> subprocess.CompletedProcess[str]:
    """Run a command and capture its output as text.

    Args:
        command: Program and arguments.
        env: Environment for the command; the caller's when None.
        stdin: What the command reads on standard input.

    Returns:
        The completed process.
    """
    return subprocess.run(
        list(command), capture_output=True, text=True, timeout=300, env=env,
        input=stdin, check=False)


def _write_program(path: Path, text: str) -> None:
    """Put an executable stand-in program in place.

    Args:
        path: Where the program goes.
        text: Its source.
    """
    path.write_text(text)
    path.chmod(0o755)


class CollectGithubAuthTest(unittest.TestCase):
    """What this machine hands over, and what it refuses to hand over."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "laptop"
        self.home.mkdir()
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.gh_log = self.tmp / "gh.log"
        self.curl_args = self.tmp / "curl.args"
        self.status = self.tmp / "status.txt"
        self.tokens = self.tmp / "tokens.txt"
        self.answers = self.tmp / "answers.txt"
        for path in (self.gh_log, self.curl_args, self.status, self.tokens,
                     self.answers):
            path.write_text("")
        _write_program(self.bin / "gh", _FAKE_GH)
        _write_program(self.bin / "curl", _FAKE_CURL)
        # git is shadowed as well: ``git credential fill`` on the developer's
        # own machine would ask the real keychain for a real token.
        _write_program(self.bin / "git", "#!/bin/bash\nexit 1\n")
        self.env = {
            "PATH": f"{self.bin}:{_BARE_PATH}",
            "HOME": str(self.home),
            "GH_FAKE_LOG": str(self.gh_log),
            "GH_FAKE_STATUS": str(self.status),
            "GH_FAKE_TOKENS": str(self.tokens),
            "CURL_FAKE_ARGS": str(self.curl_args),
            "CURL_FAKE_ANSWERS": str(self.answers),
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _logged_in(self, *accounts: tuple[str, str, bool]) -> None:
        """Say which accounts gh is logged in to here.

        Args:
            accounts: One (login, token, active) triple per account.
        """
        lines = []
        for login, _token, active in accounts:
            lines.append(f"  \u2713 Logged in to github.com account {login} (keyring)")
            lines.append(f"  - Active account: {'true' if active else 'false'}")
            lines.append("  - Git operations protocol: https")
        self.status.write_text("github.com\n" + "\n".join(lines) + "\n")
        self.tokens.write_text(
            "".join(f"{login} {token}\n" for login, token, _ in accounts))

    def _github_says(self, *answers: tuple[str, str, str]) -> None:
        """Say what the GitHub API answers about each token.

        Args:
            answers: One (token, http status, login) triple per token.
        """
        self.answers.write_text(
            "".join(f"{token} {code} {login}\n" for token, code, login in answers))

    def _collect(self, **overrides: str) -> subprocess.CompletedProcess[str]:
        """Run the real collector against the sandbox."""
        return _run("bash", str(_COLLECT), env={**self.env, **overrides})

    def test_every_account_is_collected_with_the_active_one_last(self) -> None:
        """The installer logs them in in order, so the order is the meaning."""
        self._logged_in(("ksenxx", "gho_one", True), ("other", "gho_two", False))
        self._github_says(("gho_one", "200", "ksenxx"), ("gho_two", "200", "other"))

        done = self._collect()

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout,
                         "account other gho_two\naccount ksenxx gho_one\n")

    def test_the_login_github_knows_is_the_one_that_travels(self) -> None:
        """gh's account name and the GitHub login are not always the same."""
        self._logged_in(("kiss-sorcar", "gho_one", True))
        self._github_says(("gho_one", "200", "kisssorcar"))

        done = self._collect()

        self.assertEqual(done.stdout, "account kisssorcar gho_one\n")

    def test_a_token_github_no_longer_accepts_is_left_behind(self) -> None:
        """Shipping it would leave the server failing at its first push."""
        self._logged_in(("ksenxx", "gho_dead", True), ("other", "gho_two", False))
        self._github_says(("gho_dead", "401", ""), ("gho_two", "200", "other"))

        done = self._collect()

        self.assertEqual(done.stdout, "account other gho_two\n")
        self.assertIn("no longer accepts", done.stderr)

    def test_nothing_usable_is_reported_as_a_failure(self) -> None:
        """A silent empty payload would look like a successful copy."""
        self._logged_in(("ksenxx", "gho_dead", True))
        self._github_says(("gho_dead", "401", ""))

        done = self._collect()

        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")

    def test_a_token_github_will_not_confirm_is_copied_anyway(self) -> None:
        """403 is what a rate limit answers, not what a dead token answers."""
        self._logged_in(("ksenxx", "gho_one", True))
        self._github_says(("gho_one", "403", ""))

        done = self._collect()

        self.assertEqual(done.stdout, "account ksenxx gho_one\n")
        self.assertIn("did not confirm", done.stderr)

    def test_the_accounts_that_are_fine_survive_one_that_is_not(self) -> None:
        """gh prints the whole report to stderr and fails when any account does."""
        self._logged_in(("broken", "gho_gone", False), ("ksenxx", "gho_one", True))
        self._github_says(("gho_gone", "401", ""), ("gho_one", "200", "ksenxx"))
        gh = self.bin / "gh"
        gh.write_text(_FAKE_GH.replace(
            '[ -n "${GH_FAKE_STATUS:-}" ] && cat "$GH_FAKE_STATUS"\n        exit 0',
            '[ -n "${GH_FAKE_STATUS:-}" ] && cat "$GH_FAKE_STATUS" >&2\n        exit 1'))
        gh.chmod(0o755)

        done = self._collect()

        self.assertEqual(done.stdout, "account ksenxx gho_one\n")

    def test_the_machines_own_curlrc_cannot_print_the_token(self) -> None:
        """A 'verbose' line in it would put the Authorization header in the log."""
        self._logged_in(("ksenxx", "gho_one", True))
        self._github_says(("gho_one", "200", "ksenxx"))

        self._collect()

        for call in self.curl_args.read_text().splitlines():
            self.assertEqual(call.split()[0], "-q", call)

    def test_a_token_travels_unchecked_when_github_cannot_be_asked(self) -> None:
        """A deploy from a train is still a deploy."""
        self._logged_in(("ksenxx", "gho_one", True))
        self.answers.write_text("")

        done = self._collect()

        self.assertEqual(done.stdout, "account ksenxx gho_one\n")
        self.assertIn("unchecked", done.stderr)

    def test_the_same_token_is_not_copied_twice(self) -> None:
        """Two gh accounts can hold one token; the remote needs it once."""
        self._logged_in(("ksenxx", "gho_one", True), ("alias", "gho_one", False))
        self._github_says(("gho_one", "200", "ksenxx"))

        done = self._collect()

        self.assertEqual(done.stdout, "account ksenxx gho_one\n")

    def test_gits_own_credential_helper_is_the_next_place_to_look(self) -> None:
        """A machine can have the credentials without ever having run gh."""
        _write_program(self.bin / "gh", "#!/bin/bash\nexit 1\n")
        _write_program(self.bin / "git", "#!/bin/bash\n"
                       "[ \"$1 $2\" = 'credential fill' ] || exit 1\n"
                       "cat > /dev/null\n"
                       "printf 'protocol=https\\nhost=github.com\\n"
                       "username=ksenxx\\npassword=ghp_stored\\n'\n")
        self._github_says(("ghp_stored", "200", "ksenxx"))

        done = self._collect()

        self.assertEqual(done.stdout, "account ksenxx ghp_stored\n")

    def test_the_environment_is_the_last_place_to_look(self) -> None:
        """A machine that runs automation has its token in a variable."""
        _write_program(self.bin / "gh", "#!/bin/bash\nexit 1\n")
        self._github_says(("ghp_env", "200", "robot"))

        done = self._collect(GH_TOKEN="ghp_env")

        self.assertEqual(done.stdout, "account robot ghp_env\n")

    def test_a_machine_with_no_credentials_says_so(self) -> None:
        """The deploy turns this into a warning; silence would not do."""
        _write_program(self.bin / "gh", "#!/bin/bash\nexit 1\n")

        done = self._collect()

        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")
        self.assertIn("no GitHub.com credentials", done.stderr)

    def test_the_token_is_never_an_argument_of_a_command(self) -> None:
        """Every account on a machine can read the arguments of its processes."""
        self._logged_in(("ksenxx", "gho_secret", True))
        self._github_says(("gho_secret", "200", "ksenxx"))

        self._collect()

        self.assertNotIn("gho_secret", self.curl_args.read_text())
        self.assertNotIn("gho_secret", self.gh_log.read_text())

    def test_standard_output_carries_the_payload_and_nothing_else(self) -> None:
        """It is piped into ssh; a stray word would be read as an account."""
        self._logged_in(("ksenxx", "gho_one", True))
        self._github_says(("gho_one", "200", "ksenxx"))

        done = self._collect()

        self.assertNotEqual(done.stdout, "")
        for line in done.stdout.splitlines():
            self.assertTrue(line.startswith("account "), line)
        self.assertIn("found the credentials of ksenxx", done.stderr)


class InstallGithubAuthTest(unittest.TestCase):
    """What arrives on the server, and what the server keeps."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "server"
        self.kiss = self.home / ".kiss"
        self.kiss.mkdir(parents=True)
        self.gh_config = self.home / ".config" / "gh"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        self.gh_log = self.tmp / "gh.log"
        self.gh_log.write_text("")
        _write_program(self.bin / "gh", _FAKE_GH)
        self.env = {
            "PATH": f"{self.bin}:{_BARE_PATH}",
            "HOME": str(self.home),
            "GH_FAKE_LOG": str(self.gh_log),
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, payload: str, **overrides: str
                 ) -> subprocess.CompletedProcess[str]:
        """Run the real installer with a payload on standard input."""
        return _run("bash", str(_INSTALL), env={**self.env, **overrides},
                    stdin=payload)

    def _calls(self, kind: str) -> list[tuple[str, str]]:
        """Return the (arguments, standard input) of each recorded gh call.

        Args:
            kind: The first word of the recorded line, such as "login".
        """
        calls = []
        for line in self.gh_log.read_text().splitlines():
            name, args, stdin = (part.strip() for part in line.split("|", 2))
            if name == kind:
                calls.append((args, stdin))
        return calls

    def test_each_account_is_logged_in_with_the_token_on_standard_input(self) -> None:
        """The arguments of a process are readable by everyone on the machine."""
        done = self._install("account other gho_two\naccount ksenxx gho_one\n")

        self.assertEqual(done.returncode, 0, done.stderr)
        logins = self._calls("login")
        self.assertEqual([stdin for _args, stdin in logins], ["gho_two", "gho_one"])
        for args, _stdin in logins:
            self.assertNotIn("gho_", args)
            self.assertIn("--with-token", args)
            self.assertIn("--insecure-storage", args)
            self.assertIn("--hostname github.com", args)

    def test_the_account_that_is_active_here_is_active_there(self) -> None:
        """gh makes the last login the active one, so it goes last."""
        self._install("account other gho_two\naccount ksenxx gho_one\n")

        self.assertEqual(self._calls("login")[-1][1], "gho_one")

    def test_git_is_pointed_at_gh(self) -> None:
        """Without this a push over https still asks for a password."""
        self._install("account ksenxx gho_one\n")

        self.assertEqual(len(self._calls("setup-git")), 1)
        self.assertIn("--hostname github.com", self._calls("setup-git")[0][0])

    def test_an_empty_payload_is_a_failure(self) -> None:
        """Doing nothing quietly would be reported as a successful copy."""
        done = self._install("")

        self.assertEqual(done.returncode, 1)
        self.assertIn("no github.com credentials", done.stderr)

    def test_the_accounts_this_machine_already_had_are_kept(self) -> None:
        """They are the only record of what it was logged in to."""
        self.gh_config.mkdir(parents=True)
        (self.gh_config / "hosts.yml").write_text("github.com:\n    user: theirs\n")

        self._install("account ksenxx gho_one\n")

        kept = list(self.kiss.glob("gh-hosts-before-sorcar-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(), "github.com:\n    user: theirs\n")

    def test_what_the_machine_had_survives_every_later_deploy(self) -> None:
        """It is the file somebody reaches for; a later copy must not bury it.

        gh rewrites hosts.yml with the token, so the second deploy does
        find a different file and keeps it -- what must not happen is the
        copy from before the first deploy being overwritten, or a further
        copy being taken on every deploy after that.
        """
        self.gh_config.mkdir(parents=True)
        (self.gh_config / "hosts.yml").write_text("github.com:\n    user: theirs\n")

        self._install("account ksenxx gho_one\n")
        self._install("account ksenxx gho_one\n")
        after_two = sorted(p.name for p in self.kiss.glob("gh-hosts-before-sorcar-*"))
        self._install("account ksenxx gho_one\n")
        self._install("account ksenxx gho_one\n")

        kept = sorted(self.kiss.glob("gh-hosts-before-sorcar-*"))
        self.assertEqual([p.name for p in kept], after_two)
        self.assertEqual(kept[0].read_text(), "github.com:\n    user: theirs\n")

    def test_the_gitconfig_is_kept_before_gh_takes_github_off_its_helpers(self) -> None:
        """setup-git empties that list; the copy is how it comes back."""
        (self.home / ".gitconfig").write_text(
            "[credential \"https://github.com\"]\n\thelper = their-helper\n")

        self._install("account ksenxx gho_one\n")

        kept = list(self.kiss.glob("gitconfig-before-sorcar-*"))
        self.assertEqual(len(kept), 1)
        self.assertIn("their-helper", kept[0].read_text())

    def test_a_gitconfig_changed_since_the_last_deploy_is_kept_again(self) -> None:
        """The helper somebody added last week is about to be cleared.

        Keeping a copy only on the very first deploy leaves every change
        made afterwards unprotected: ``setup-git`` empties the list of
        credential helpers for github.com each time it runs, and the
        pristine copy from the first deploy does not have the one that was
        added since.
        """
        gitconfig = self.home / ".gitconfig"
        gitconfig.write_text("[user]\n\tname = Them\n")

        self._install("account ksenxx gho_one\n")
        gitconfig.write_text(
            "[user]\n\tname = Them\n"
            "[credential \"https://github.com\"]\n\thelper = their-new-helper\n")
        self._install("account ksenxx gho_one\n")

        kept = sorted(p.read_text() for p in
                      self.kiss.glob("gitconfig-before-sorcar-*"))
        self.assertTrue(any("their-new-helper" in text for text in kept), kept)
        # And what the machine had before any of this is still kept as well.
        self.assertTrue(any("their-new-helper" not in text for text in kept), kept)

    def test_re_deploying_an_untouched_machine_keeps_no_second_copy(self) -> None:
        """A copy of a file whose content is already kept is only noise."""
        gitconfig = self.home / ".gitconfig"
        gitconfig.write_text("[user]\n\tname = Them\n")

        self._install("account ksenxx gho_one\n")
        before = sorted(p.name for p in self.kiss.glob("gitconfig-before-sorcar-*"))
        self._install("account ksenxx gho_one\n")
        self._install("account ksenxx gho_one\n")

        self.assertEqual(
            sorted(p.name for p in self.kiss.glob("gitconfig-before-sorcar-*")), before)

    def test_a_hosts_file_anybody_could_read_is_locked_down(self) -> None:
        """gh writes the tokens into it in plain text and keeps its mode."""
        self.gh_config.mkdir(parents=True)
        hosts = self.gh_config / "hosts.yml"
        hosts.write_text("github.com:\n")
        hosts.chmod(0o644)

        self._install("account ksenxx gho_one\n")

        self.assertEqual(hosts.stat().st_mode & 0o077, 0)

    def test_the_hosts_file_is_locked_down_even_when_an_account_is_refused(self) -> None:
        """One account logging in is enough to put a token in it."""
        self.gh_config.mkdir(parents=True)
        hosts = self.gh_config / "hosts.yml"
        hosts.write_text("github.com:\n")
        hosts.chmod(0o644)

        self._install("account other gho_two\naccount ksenxx gho_one\n",
                      GH_FAKE_BAD_TOKEN="gho_one")

        self.assertEqual(len(self._calls("login")), 2)
        self.assertEqual(hosts.stat().st_mode & 0o077, 0)

    def test_a_copy_that_cannot_be_made_stops_the_change(self) -> None:
        """"Nothing is overwritten without a copy" is only true if it stops."""
        gitconfig = self.home / ".gitconfig"
        gitconfig.write_text("[credential \"https://github.com\"]\n\thelper = theirs\n")
        gitconfig.chmod(0o000)
        try:
            done = self._install("account ksenxx gho_one\n")
        finally:
            gitconfig.chmod(0o600)

        self.assertNotEqual(done.returncode, 0)
        self.assertEqual(len(self._calls("setup-git")), 0)
        self.assertIn("could not keep a copy", done.stdout)
        self.assertEqual(list(self.kiss.glob("gitconfig-before-sorcar-*")), [])
        # Nor does the fallback touch it: it needs the same copy.
        self.assertFalse((self.home / ".git-credentials").exists())

    def test_a_hosts_file_that_cannot_be_locked_down_gets_no_token(self) -> None:
        """A secret written where somebody else can read it cannot be taken back."""
        self.gh_config.mkdir(parents=True)
        hosts = self.gh_config / "hosts.yml"
        hosts.write_text("github.com:\n")
        hosts.chmod(0o644)
        # A file somebody else owns is one this account cannot chmod, which is
        # what a root-owned hosts.yml is; here that is arranged rather than
        # waited for.
        _write_program(self.bin / "chmod", _CHMOD_REFUSING_HOSTS_YML)

        done = self._install("account ksenxx gho_one\n")

        self.assertEqual(len(self._calls("login")), 0)
        self.assertIn("can read", done.stdout)
        # The credentials still arrive, the way git alone understands them.
        self.assertIn("gho_one", (self.home / ".git-credentials").read_text())

    def test_a_token_in_the_environment_does_not_stop_the_login(self) -> None:
        """gh prefers $GH_TOKEN and refuses to store anything while it is set."""
        done = self._install("account ksenxx gho_one\n",
                             GH_TOKEN="ghp_from_the_api_keys_file")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(len(self._calls("login")), 1)

    def test_a_login_gh_refuses_falls_back_to_git(self) -> None:
        """The credentials are worth having even when gh will not hold them."""
        done = self._install("account ksenxx gho_one\n", GH_FAKE_LOGIN_FAILS="1")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("gho_one", (self.home / ".git-credentials").read_text())


class InstallWithoutGhTest(unittest.TestCase):
    """A server with no gh, and no way to fetch one, still gets the credentials.

    ``git`` here is the real one, pointed at a home directory of its own.
    """

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.home = self.tmp / "server"
        (self.home / ".kiss").mkdir(parents=True)
        self.credentials = self.home / ".git-credentials"
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        # No gh on PATH, and the tarball cannot be fetched either.
        _write_program(self.bin / "curl", _FAILING_CURL)
        self.env = {
            "PATH": f"{self.bin}:{_BARE_PATH}",
            "HOME": str(self.home),
        }

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _install(self, payload: str) -> subprocess.CompletedProcess[str]:
        """Run the real installer with a payload on standard input."""
        return _run("bash", str(_INSTALL), env=self.env, stdin=payload)

    def _helpers(self) -> list[str]:
        """Return the credential helpers configured for github.com."""
        done = _run("git", "config", "--global", "--get-all",
                    "credential.https://github.com.helper", env=self.env)
        return done.stdout.split()

    def _fill(self) -> dict[str, str]:
        """Ask the real git which credential it would use for github.com."""
        done = _run("git", "credential", "fill",
                    env={**self.env, "GIT_TERMINAL_PROMPT": "0"},
                    stdin="protocol=https\nhost=github.com\n\n")
        answer = {}
        for line in done.stdout.splitlines():
            key, _, value = line.partition("=")
            answer[key] = value
        return answer

    def test_the_credentials_are_installed_for_git_alone(self) -> None:
        """gh is a convenience; pushing is not."""
        done = self._install("account other gho_two\naccount ksenxx gho_one\n")

        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(self.credentials.read_text(),
                         "https://ksenxx:gho_one@github.com\n"
                         "https://other:gho_two@github.com\n")
        self.assertEqual(self._helpers(), ["store"])

    def test_git_hands_back_the_account_that_is_active_here(self) -> None:
        """It answers with the first line that matches, so the order is the answer."""
        self._install("account other gho_two\naccount ksenxx gho_one\n")

        self.assertEqual(self._fill(),
                         {"protocol": "https", "host": "github.com",
                          "username": "ksenxx", "password": "gho_one"})

    def test_the_credentials_this_machine_had_are_kept(self) -> None:
        """The file is rewritten; what was in it is somebody's only copy."""
        self.credentials.write_text("https://a-colleague:gho_theirs@github.com\n")

        self._install("account ksenxx gho_one\n")

        kept = list((self.home / ".kiss").glob("git-credentials-before-sorcar-*"))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].read_text(),
                         "https://a-colleague:gho_theirs@github.com\n")

    def test_a_credentials_file_that_is_a_link_stays_a_link(self) -> None:
        """Replacing it would detach it from the store that maintains it."""
        real = self.tmp / "dotfiles" / "credentials"
        real.parent.mkdir()
        real.write_text("https://someone:glpat_x@gitlab.com\n")
        self.credentials.symlink_to(real)

        self._install("account ksenxx gho_one\n")

        self.assertTrue(self.credentials.is_symlink())
        self.assertEqual(real.read_text(),
                         "https://ksenxx:gho_one@github.com\n"
                         "https://someone:glpat_x@gitlab.com\n")

    def test_only_this_account_can_read_them(self) -> None:
        """The file is a password in plain text."""
        self._install("account ksenxx gho_one\n")

        self.assertEqual(self.credentials.stat().st_mode & 0o077, 0)

    def test_the_credentials_of_other_hosts_and_accounts_are_kept(self) -> None:
        """They belong to whoever put them there."""
        self.credentials.write_text(
            "https://someone:glpat_x@gitlab.com\n"
            "https://a-colleague:gho_theirs@github.com\n"
            "https://ksenxx:gho_stale@github.com\n")

        self._install("account ksenxx gho_fresh\n")

        self.assertEqual(self.credentials.read_text(),
                         "https://ksenxx:gho_fresh@github.com\n"
                         "https://someone:glpat_x@gitlab.com\n"
                         "https://a-colleague:gho_theirs@github.com\n")

    def test_the_account_that_arrived_answers_before_one_that_was_here(self) -> None:
        """git takes the first line that matches, so it has to be ours."""
        self.credentials.write_text("https://a-colleague:gho_theirs@github.com\n")

        self._install("account ksenxx gho_one\n")

        self.assertEqual(self._fill()["username"], "ksenxx")

    def test_a_second_deploy_leaves_one_helper_and_one_line(self) -> None:
        """Deploys are repeated; the file and the config must not grow."""
        self._install("account ksenxx gho_one\n")
        self._install("account ksenxx gho_one\n")
        self._install("account ksenxx gho_one\n")

        self.assertEqual(self.credentials.read_text(),
                         "https://ksenxx:gho_one@github.com\n")
        self.assertEqual(self._helpers(), ["store"])

    def test_a_helper_this_machine_already_configured_keeps_working(self) -> None:
        """git asks the helpers in turn; the one that was here is still asked."""
        _run("git", "config", "--global", "--add",
             "credential.https://github.com.helper", "cache", env=self.env)

        self._install("account ksenxx gho_one\n")

        self.assertEqual(self._helpers(), ["cache", "store"])


if __name__ == "__main__":
    unittest.main()

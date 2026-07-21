#!/bin/bash
# kiss-slack-sorcar-poller — cron job wrapper.
#
# Invoked once per minute from cron.  Runs the KISS Slack channel
# poller, which holds an fcntl lock (so overlapping cron ticks exit
# immediately) and polls the #sorcar Slack channel every 3 seconds for
# 57 seconds per invocation — giving effective 3-second polling under
# cron's one-minute granularity.
#
# For each new unanswered top-level message from ksen (oldest first),
# the poller runs the message text as a Sorcar task and posts the
# result back as a threaded Slack reply formatted with Slack mrkdwn.
#
# Install (idempotent):
#   scripts/kiss-slack-sorcar-cron.sh --install
# Uninstall:
#   scripts/kiss-slack-sorcar-cron.sh --uninstall
# Logs: ~/.kiss/slack_channel_sorcar_poller/poller.log

set -euo pipefail

JOB_NAME="kiss-slack-sorcar-poller"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Resolve the venv python of the newest installed KISS Sorcar extension
# at run time.  A hardcoded extension path breaks silently on every
# extension upgrade (the old version's .venv is deleted), so pick the
# highest-versioned ksenxx.kiss-sorcar-* directory that actually has a
# working venv python.
find_venv_python() {
    local ext
    for ext in $(ls -d "$HOME"/.vscode/extensions/ksenxx.kiss-sorcar-* 2>/dev/null \
                 | sort -t- -k2,2V -r); do
        if [ -x "$ext/kiss_project/.venv/bin/python" ]; then
            echo "$ext/kiss_project/.venv/bin/python"
            return 0
        fi
    done
    return 1
}
CRON_LINE="* * * * * $SCRIPT_PATH >> $HOME/.kiss/slack_channel_sorcar_poller/cron.log 2>&1 # $JOB_NAME"

install_cron() {
    mkdir -p "$HOME/.kiss/slack_channel_sorcar_poller"
    { crontab -l 2>/dev/null | grep -v "# $JOB_NAME\$" || true
      echo "$CRON_LINE"
    } | crontab -
    echo "Installed cron job '$JOB_NAME':"
    crontab -l | grep "# $JOB_NAME\$"
}

uninstall_cron() {
    crontab -l 2>/dev/null | grep -v "# $JOB_NAME\$" | crontab -
    echo "Removed cron job '$JOB_NAME'."
}

run_poller() {
    export PATH="/opt/homebrew/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    # cron does not set SHELL to the user's login shell (it uses /bin/sh
    # and leaves $SHELL unset in the child env), which made the poller's
    # source_shell_env() fall back to sourcing ~/.bashrc instead of
    # ~/.zshrc — so no API keys were imported, get_available_models()
    # lacked the default model, and every Sorcar task aborted with
    # "No model available" (empty Slack replies).  Export the user's
    # real login shell so RC sourcing picks up the API keys.
    SHELL="$(dscl . -read "/Users/$(id -un)" UserShell 2>/dev/null | awk '{print $2}')"
    export SHELL="${SHELL:-/bin/zsh}"
    export KISS_SLACK_WORKSPACE="${KISS_SLACK_WORKSPACE:-learningsystems}"
    export KISS_SLACK_USER="${KISS_SLACK_USER:-ksen}"
    export KISS_SLACK_CHANNEL="${KISS_SLACK_CHANNEL:-sorcar}"
    VENV_PYTHON="$(find_venv_python)" || {
        echo "$(date '+%Y-%m-%d %H:%M:%S') ERROR: no KISS Sorcar extension venv python found" >&2
        exit 1
    }
    exec "$VENV_PYTHON" -m kiss.agents.third_party_agents.slack_channel_sorcar_poller
}

case "${1:-}" in
    --install) install_cron ;;
    --uninstall) uninstall_cron ;;
    *) run_poller ;;
esac

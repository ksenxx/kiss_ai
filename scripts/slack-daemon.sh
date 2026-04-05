#!/bin/bash
# Wrapper script for the KISS Slack daemon.
# Sourced by launchd on login to pick up API keys from ~/.zshrc.

# launchd doesn't inherit shell env vars, so source them
source ~/.zshrc 2>/dev/null

cd /Users/ksen/work/kiss
exec /opt/homebrew/bin/uv run kiss-slack \
    --daemon \
    --daemon-channel all-kisssorcar \
    --allow-users ksen

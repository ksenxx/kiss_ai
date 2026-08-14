#!/usr/bin/env bash
# Author: Koushik Sen (ksen@berkeley.edu)
# OpenAI organization cost report via the official Admin Costs API.
# One curl from your machine to api.openai.com, nothing in between.
#
# Requires: OPENAI_ADMIN_KEY (created by an org Owner at
# platform.openai.com -> Settings -> Organization -> Admin keys).
#
# Usage: openai_costs.sh [days-back]   (default 30)
set -euo pipefail

: "${OPENAI_ADMIN_KEY:?export OPENAI_ADMIN_KEY=sk-admin... first}"
DAYS="${1:-30}"

if date -v-1d >/dev/null 2>&1; then
  START=$(date -u -v-"${DAYS}"d +%s)          # BSD/macOS date
else
  START=$(date -u -d "${DAYS} days ago" +%s)  # GNU date
fi

curl -sS "https://api.openai.com/v1/organization/costs?start_time=${START}&limit=180&group_by=project_id" \
  --header "Authorization: Bearer ${OPENAI_ADMIN_KEY}"

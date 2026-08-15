#!/usr/bin/env bash
# Author: Koushik Sen (ksen@berkeley.edu)
# Anthropic organization cost report via the official Admin API.
# The most private connector possible: one curl from your machine to
# api.anthropic.com, nothing in between.
#
# Requires: ANTHROPIC_ADMIN_KEY (sk-ant-admin..., created in the Anthropic
# Console by an org admin; needs an Organization, not an individual account).
#
# Usage: anthropic_costs.sh [days-back]   (default 30)
set -euo pipefail

: "${ANTHROPIC_ADMIN_KEY:?export ANTHROPIC_ADMIN_KEY=sk-ant-admin... first}"
DAYS="${1:-30}"

if date -v-1d >/dev/null 2>&1; then
  START=$(date -u -v-"${DAYS}"d +%Y-%m-%dT00:00:00Z)   # BSD/macOS date
else
  START=$(date -u -d "${DAYS} days ago" +%Y-%m-%dT00:00:00Z)  # GNU date
fi

curl -sS "https://api.anthropic.com/v1/organizations/cost_report?starting_at=${START}&group_by[]=workspace_id&group_by[]=description" \
  --header "anthropic-version: 2023-06-01" \
  --header "x-api-key: ${ANTHROPIC_ADMIN_KEY}"

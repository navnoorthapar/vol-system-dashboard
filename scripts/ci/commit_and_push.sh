#!/usr/bin/env bash
# Commit the given paths and push the commit to main (the daily bot's commit steps).
#
#   scripts/ci/commit_and_push.sh "<title>" "<body>" PATH [PATH ...]
#
# A PATH that does not exist is skipped. A PATH starting with ':' is a git pathspec
# (e.g. ':(exclude)data_store/calibrations/diagnostics') and is passed through; at
# least one real path is required, because a list of exclusions alone would stage
# the whole tree. Nothing staged: no commit. A rejected push rebases onto
# origin/main and retries (4 attempts); --autostash keeps the run's other unstaged
# changes (vol_system.db, a later step's artifacts) out of the rebase.
# Needs GH_TOKEN and REPO; writes sha=<HEAD> to $GITHUB_OUTPUT when that is set.
# PUSH_RETRY_SLEEP (seconds per attempt, default 10) is for the test of this script.
set -euo pipefail

title=$1
body=$2
shift 2

specs=()
real=0
for p in "$@"; do
  case "$p" in
    :*) specs+=("$p") ;;
    *) if [ -e "$p" ]; then specs+=("$p"); real=$((real + 1)); fi ;;
  esac
done

git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

if [ "$real" -gt 0 ]; then
  git add -- "${specs[@]}"
fi
if [ "$real" -eq 0 ] || git diff --staged --quiet; then
  echo "No changes to commit."
else
  git commit -q -m "$title" -m "$body" -m "Run: ${GITHUB_SERVER_URL:-}/${REPO:-}/actions/runs/${GITHUB_RUN_ID:-}"
  auth="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$GH_TOKEN" | base64 -w0)"
  pushed=0
  for i in 1 2 3 4; do
    if git -c http.https://github.com/.extraheader="$auth" push origin HEAD:main; then
      pushed=1; break
    fi
    echo "Push rejected (attempt $i); rebasing onto origin/main"
    sleep $((i * ${PUSH_RETRY_SLEEP:-10}))
    git -c http.https://github.com/.extraheader="$auth" pull --rebase --autostash origin main || {
      git rebase --abort || true; echo "::error::rebase onto main failed"; exit 1; }
  done
  [ "$pushed" = 1 ] || { echo "::error::push to main failed after 4 attempts"; exit 1; }
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"
fi

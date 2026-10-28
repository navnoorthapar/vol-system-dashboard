#!/usr/bin/env bash
# Commit the given paths and push the commit to main (the daily bot's commit steps).
#
#   scripts/ci/commit_and_push.sh "<title>" "<body>" PATH [PATH ...]
#
# A PATH that does not exist is skipped. A PATH starting with ':' is a git pathspec
# (e.g. ':(exclude)data_store/calibrations/diagnostics') and is passed through; at
# least one real path is required, because a list of exclusions alone would stage
# the whole tree. Nothing staged: no commit.
#
# Push: up to PUSH_ATTEMPTS (default 5). After a rejected or failed push the script
# fetches origin/main, itself retried FETCH_ATTEMPTS times (default 4) with a
# growing pause, because a GitHub 5xx or a network blip is transient: a fetch that
# keeps failing costs one push attempt, it does not end the loop (it used to: one
# failed `git pull` after a failed push exited 1 at once, "rebase onto main failed",
# and the run's captures went with the runner). Then the commit is moved onto
# origin/main (sync_onto_main).
#
# git cannot merge the binary files the bot rewrites (the yearly market partitions,
# regime_labels.parquet), so any commit to main that touched one of them (a PR merge,
# a manual data fix) made the rebase fail and the whole commit was dropped, or made
# --autostash's pop leave main's copy where the run's was. sync_onto_main therefore
# first compares what main changed with what this run changed, committed or not:
#   * a re-derivable file (REDERIVABLE: space-separated path prefixes; default
#     data_store/market/ and data_store/signals/regime_labels.parquet) both changed
#     is reset to main's version and rebuilt from this run's database by REDERIVE
#     (`python scripts/cloud_refresh.py --rederive ...`): main's rows and the run's
#     both survive. If it was in the commit, the commit is made again on top of main;
#   * any other uncommitted file both changed (vol_system.db, a later step's
#     artifacts) keeps this run's version;
#   * any other COMMITTED file both changed is a real conflict: exit 1, naming it
#     (nothing is touched). So is a re-derivable one in the commit without REDERIVE.
# The capture commit holds only files this run created, which main does not have,
# so it lands first and on its own (.github/workflows/daily_refresh.yml).
#
# Needs GH_TOKEN and REPO; writes sha=<HEAD> to $GITHUB_OUTPUT when that is set.
# PUSH_RETRY_SLEEP (seconds per attempt, default 10) is for the tests of this script.
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

attempts=${PUSH_ATTEMPTS:-5}
fetch_attempts=${FETCH_ATTEMPTS:-4}
pause=${PUSH_RETRY_SLEEP:-10}
rederivable=${REDERIVABLE:-"data_store/market/ data_store/signals/regime_labels.parquet"}

git config user.name  "github-actions[bot]"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
auth="AUTHORIZATION: basic $(printf 'x-access-token:%s' "${GH_TOKEN:-}" | base64 -w0)"

gitc() { git -c http.https://github.com/.extraheader="$auth" "$@"; }

# stage the paths; 0 when there is something to commit
stage() {
  [ "$real" -gt 0 ] || return 1
  git add -- "${specs[@]}"
  ! git diff --staged --quiet
}

commit() {
  git commit -q -m "$title" -m "$body" -m "Run: ${GITHUB_SERVER_URL:-}/${REPO:-}/actions/runs/${GITHUB_RUN_ID:-}"
}

fetch_main() {
  local j
  for j in $(seq 1 "$fetch_attempts"); do
    if gitc fetch -q origin main; then return 0; fi
    echo "Fetching origin/main failed (try $j of $fetch_attempts); retrying"
    sleep $((j * pause))
  done
  return 1
}

is_rederivable() {
  local f=$1 pre
  for pre in $rederivable; do
    case "$f" in "$pre"*) return 0 ;; esac
  done
  return 1
}

# Move this run's commit (if any) and its uncommitted changes onto origin/main.
# Returns 0 when there is a commit to push, 3 when origin/main already holds
# everything, and 2 when a step failed. The caller tests the status, and bash
# ignores `set -e` inside a function whose status is tested (even in a subshell
# that sets it again), so every step that can fail returns 2 explicitly: a
# failed reset, rebase or REDERIVE used to fall through to "Nothing left to
# commit" and exit 0 with the run's rows missing from main (CODE-01).
sync_onto_main() {
  local mb f save entry recommit=0
  local -A mainchg=() mine=() dirty=()
  local clash_committed=() clash_dirty=() rederive=() keep=() bad=()
  mb=$(git merge-base HEAD origin/main) || return 2
  while IFS= read -r -d '' f; do mainchg["$f"]=1; done < <(git diff -z --name-only "$mb" origin/main)
  while IFS= read -r -d '' f; do mine["$f"]=1; done < <(git diff -z --name-only "$mb" HEAD)
  while IFS= read -r -d '' entry; do dirty["${entry:3}"]=1; done \
    < <(git status -z --porcelain --untracked-files=all)
  for f in "${!mine[@]}"; do
    if [ -n "${mainchg[$f]:-}" ]; then clash_committed+=("$f"); fi
  done
  for f in "${!dirty[@]}"; do
    if [ -n "${mainchg[$f]:-}" ] && [ -z "${mine[$f]:-}" ]; then clash_dirty+=("$f"); fi
  done
  for f in "${clash_committed[@]}"; do
    if is_rederivable "$f" && [ -n "${REDERIVE:-}" ]; then rederive+=("$f"); else bad+=("$f"); fi
  done
  if [ "${#bad[@]}" -gt 0 ]; then
    echo "::error::main changed what this commit changes: ${bad[*]} (re-derivable only with REDERIVE set; REDERIVE ${REDERIVE:+set}${REDERIVE:-unset})"
    return 4
  fi
  for f in "${clash_dirty[@]}"; do
    if is_rederivable "$f" && [ -n "${REDERIVE:-}" ]; then rederive+=("$f"); else keep+=("$f"); fi
  done
  if [ "${#rederive[@]}" -gt 0 ]; then
    echo "Main also changed re-derivable file(s): ${rederive[*]}; re-deriving them over main's"
  fi
  if [ "${#keep[@]}" -gt 0 ]; then
    echo "Main also changed uncommitted file(s) of this run; keeping this run's: ${keep[*]}"
  fi
  if [ "${#clash_committed[@]}" -gt 0 ]; then
    git reset -q --mixed HEAD~1 || return 2      # the commit is made again on top of main
    recommit=1
  fi
  save=$(mktemp -d) || return 2
  for f in "${keep[@]}"; do mkdir -p "$save/$(dirname "$f")" && cp -p -- "$f" "$save/$f" || return 2; done
  for f in "${rederive[@]}" "${keep[@]}"; do     # to the base version, so the move cannot clash
    if git cat-file -e "HEAD:$f" 2>/dev/null; then git checkout -q -- "$f" || return 2; else rm -f -- "$f"; fi
  done
  if [ "$recommit" = 1 ] || [ "$(git rev-list --count origin/main..HEAD)" = 0 ]; then
    git reset -q --keep origin/main || return 2
  else
    git rebase -q --autostash origin/main || return 2
  fi
  for f in "${keep[@]}"; do mkdir -p "$(dirname "$f")" && cp -p -- "$save/$f" "$f" || return 2; done
  rm -rf "$save"
  if [ "${#rederive[@]}" -gt 0 ]; then
    echo "Re-deriving: $REDERIVE"
    bash -c "$REDERIVE" || { echo "::error::re-deriving ${rederive[*]} over main's failed"; return 2; }
  fi
  if [ "$recommit" = 1 ]; then
    if stage; then
      commit || return 2
    else
      echo "Nothing left to commit: origin/main already holds it."
      return 3
    fi
  fi
  return 0
}

if ! stage; then
  echo "No changes to commit."
else
  commit
  pushed=0
  for i in $(seq 1 "$attempts"); do
    if gitc push origin HEAD:main; then
      pushed=1; break
    fi
    if [ "$i" -eq "$attempts" ]; then break; fi
    echo "Push failed (attempt $i of $attempts); fetching origin/main to rebase onto it"
    sleep $((i * pause))
    if ! fetch_main; then
      echo "origin/main could not be fetched after $fetch_attempts tries; next push attempt"
      continue
    fi
    rc=0
    sync_onto_main || rc=$?
    case "$rc" in
      0) ;;
      3) pushed=1; break ;;                      # nothing of ours is left to push
      4) exit 1 ;;                               # a real conflict, named above
      *) echo "::error::moving this run's commit onto origin/main failed (status $rc); nothing pushed"
         exit 1 ;;
    esac
  done
  [ "$pushed" = 1 ] || { echo "::error::push to main failed after $attempts attempts"; exit 1; }
fi
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "sha=$(git rev-parse HEAD)" >> "$GITHUB_OUTPUT"
fi

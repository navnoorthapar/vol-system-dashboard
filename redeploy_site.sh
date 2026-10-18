#!/usr/bin/env bash
# redeploy_site.sh — re-render navnoorbawa.me THROUGH THE DAILY WORKFLOW, or preview locally.
#
# The live site is published only by .github/workflows/daily_refresh.yml (the
# "Daily Market Refresh" bot), which restores the committed data captures, runs
# the publish-path tests and appends one commit to the gh-pages history. This
# script never pushes from your machine: a local render uses your local
# data_store/ (on a clean clone the committed DB, months stale) and the old
# version of this script force-pushed an orphan gh-pages that erased the bot's
# deploy history.
#
#   ./redeploy_site.sh                 dispatch the workflow as a pure re-render
#                                      (steps=freeze; needs the GitHub CLI `gh`)
#   ./redeploy_site.sh --steps S       dispatch with other cloud_refresh steps,
#                                      e.g. market,calibrate,freeze
#   ./redeploy_site.sh --preview [DIR] render locally into DIR (default
#                                      .site-build/) and publish nothing
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
WORKFLOW="daily_refresh.yml"

case "${1:-}" in
  --preview)
    out="${2:-$ROOT/.site-build}"
    echo "==> Rendering a local preview into $out (nothing is published)"
    python "$ROOT/dashboard/freeze_site.py" "$out"
    echo "==> Open $out/index.html"
    exit 0
    ;;
  --steps)
    steps="${2:?--steps needs a value, e.g. market,calibrate,freeze}"
    ;;
  "")
    steps="freeze"
    ;;
  -h|--help)
    sed -n '2,17p' "$0"
    exit 0
    ;;
  *)
    echo "unknown option: $1 (see --help)" >&2
    exit 2
    ;;
esac

if ! [[ "$steps" =~ ^[a-z,]+$ ]]; then
  echo "invalid steps: $steps" >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "The GitHub CLI (gh) is not installed. Start the workflow from the Actions tab:" >&2
  echo "  Actions -> Daily Market Refresh -> Run workflow (steps: $steps)" >&2
  exit 1
fi

echo "==> Dispatching $WORKFLOW on main with steps=$steps"
gh workflow run "$WORKFLOW" --ref main -f steps="$steps"
echo "==> Follow it with: gh run watch \"\$(gh run list --workflow $WORKFLOW --limit 1 --json databaseId -q '.[0].databaseId')\""

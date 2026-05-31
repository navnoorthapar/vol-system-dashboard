#!/bin/bash
# redeploy_site.sh — Re-render the dashboard and push to GitHub Pages.
#
# The site is a STATIC SNAPSHOT (GitHub Pages can't run Flask). It does not
# auto-update. Run this whenever you've refreshed data_store/ and want the live
# site at navnoorbawa.me to reflect it.
#
#   ./redeploy_site.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/.site-build"
REPO="https://github.com/navnoorthapar/vol-system-dashboard.git"

echo "==> Rendering static site..."
python "$ROOT/dashboard/freeze_site.py" "$BUILD"

echo "==> Publishing to gh-pages..."
cd "$BUILD"
git init -q 2>/dev/null || true
git checkout -q -B gh-pages
git add -A
if git -c user.name="Navnoor Bawa" -c user.email="navnoorbawa@gmail.com" \
      commit -q -m "Redeploy dashboard snapshot $(date '+%Y-%m-%d %H:%M')"; then
    :
else
    echo "    (no content changes to commit)"
fi
git remote add origin "$REPO" 2>/dev/null || git remote set-url origin "$REPO"
git push -f origin gh-pages

echo "==> Done. Live in ~1-2 min at https://navnoorbawa.me"

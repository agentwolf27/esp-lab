#!/usr/bin/env bash
# Produce an anonymised copy of the repo for double-blind submission.
# ICBINB-BIO anonymity extends to LINKED MATERIAL, so the public repo cannot be cited.
# Usage:  bash paper/anonymise.sh  ->  ../esp-lab-anon/
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"; DST="${SRC}-anon"
rm -rf "$DST"; mkdir -p "$DST"
# copy tracked files only, minus the private and identifying ones
cd "$SRC"
git ls-files | grep -vE '^(paper/anonymise\.sh|outreach/)' | while read -r f; do
  mkdir -p "$DST/$(dirname "$f")"; cp "$f" "$DST/$f"
done
cd "$DST"
# scrub identifying strings from all text files
find . -type f \( -name '*.md' -o -name '*.py' -o -name '*.tex' -o -name '*.txt' -o -name '*.json' \
     -o -name 'LICENSE' -o -name '*.sty' -o -name '*.cfg' -o -name '*.yml' -o -name '*.yaml' \) -print0 |
  xargs -0 sed -i '' \
    -e 's|github\.com/agentwolf27/esp-lab|ANONYMISED-REPO|g' \
    -e 's|agentwolf27|anon|g' \
    -e 's|Vishrut Malhotra|Anonymous Author|g' \
    -e 's|vishrutmalhotra[0-9]*@gmail\.com|anon@example.com|g' \
    -e 's|/Users/vish|/home/anon|g'
# no git history (it carries the author name and email)
rm -rf .git
echo "=== residual identifying strings (should be empty) ==="
grep -rniE 'agentwolf|vishrut|malhotra|/Users/vish' . || echo "  clean"
echo "=== wrote $DST ($(find . -type f | wc -l | tr -d ' ') files) ==="
echo "Next: upload to anonymous.4open.science (or a fresh anonymous GitHub account),"
echo "then replace [ANONYMISED MIRROR URL] in paper_a.tex."

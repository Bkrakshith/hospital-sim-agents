#!/usr/bin/env bash
# Pre-commit scan: refuse staged files that contain API-key-shaped strings.
# Prints the file name only, never the match.
set -u
hit=0
while IFS= read -r f; do
  [ -f "$f" ] || continue
  if git show ":$f" 2>/dev/null | grep -aEq 'sk-ant-[A-Za-z0-9_-]{8,}|sk-[A-Za-z0-9_-]{16,}'; then
    echo "check_secrets: possible API key in staged file: $f" >&2
    hit=1
  fi
done < <(git diff --cached --name-only --diff-filter=ACM)
if [ "$hit" -ne 0 ]; then
  echo "check_secrets: commit blocked. Remove the key (keys belong in the keyring)." >&2
  exit 1
fi
exit 0

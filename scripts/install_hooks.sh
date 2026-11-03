#!/usr/bin/env bash
# Install the secret-scan pre-commit hook for this repository.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
chmod +x scripts/check_secrets.sh
hooks="$(cd "$(git rev-parse --git-path hooks)" && pwd)"
target="$hooks/pre-commit"
src="$root/scripts/check_secrets.sh"
if [ -L "$target" ] && [ "$(readlink -f "$target")" = "$(readlink -f "$src")" ]; then
  echo "pre-commit hook already installed"
  exit 0
fi
if [ -e "$target" ] || [ -L "$target" ]; then
  mv "$target" "$target.bak"
  echo "existing pre-commit hook backed up to pre-commit.bak"
fi
ln -s "$src" "$target"
echo "installed pre-commit hook -> scripts/check_secrets.sh"

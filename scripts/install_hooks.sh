#!/usr/bin/env bash
# Install the secret-scan pre-commit hook for this repository.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
chmod +x "$root/scripts/check_secrets.sh"
ln -sf ../../scripts/check_secrets.sh "$root/.git/hooks/pre-commit"
echo "installed pre-commit hook -> scripts/check_secrets.sh"

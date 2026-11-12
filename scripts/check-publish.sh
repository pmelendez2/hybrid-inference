#!/usr/bin/env bash
# Fail closed: no publication without local security and native ARM64 checks.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

for tool in git cargo gitleaks ssh; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool not found"
done

printf '[check] clean working tree\n'
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || \
    fail 'Commit or remove untracked files first'

printf '[check] Git whitespace\n'
git diff HEAD^ HEAD --check || fail 'Commit whitespace check failed'

printf '[check] Rust formatting (local)\n'
cargo fmt --all --check || fail 'Rust formatting failed'

printf '[check] Git history: Gitleaks\n'
gitleaks git --log-opts='--all' --redact . || \
    fail 'Git history secret scan failed'

printf '[check] working directory: Gitleaks\n'
gitleaks dir --redact . || fail 'Working tree secret scan failed'

printf '[check] Git history: private infrastructure identifiers\n'
if git grep -q -I -E \
    'tecocllc|\.home\.arpa|\.local|192\.168\.|10\.[0-9]+\.[0-9]+\.' \
    $(git rev-list --all) -- ':!Cargo.lock'; then
    fail 'Private infrastructure references remain in history'
fi

printf '[check] ARM64 compilation and tests on picam\n'
"$(git rev-parse --show-toplevel)/scripts/test-arm64.sh" || \
    fail 'ARM64 validation failed'

printf 'PASS: all publication checks complete\n'

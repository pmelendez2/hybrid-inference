#!/usr/bin/env bash
# Native Raspberry Pi ARM64 build/tests against this exact Git checkout.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

command -v ssh >/dev/null 2>&1 || fail 'ssh not installed'

host="$(git config --local --get hybrid.arm64Host || true)"
container="$(git config --local --get hybrid.arm64Container || true)"
[[ -n "$host" ]] || fail \
    'Set git config --local hybrid.arm64Host picam'
[[ -n "$container" ]] || fail \
    'Set git config --local hybrid.arm64Container CONTAINER_NAME'

# Restrict input to simple host aliases/container names; no shell metacharacters.
[[ "$host" =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'Unsafe host alias'
[[ "$container" =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'Unsafe container name'

expected="$(git rev-parse HEAD)" || exit 1
printf '[arm64] host=%s container=%s commit=%s\n' \
    "$host" "$container" "$expected"

ssh -T -o BatchMode=yes "$host" bash -s -- \
    "$container" "$expected" <<'REMOTE'
set -uo pipefail
container="$1"
expected="$2"
fail() { printf 'FAIL (picam): %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == aarch64 ]] || fail 'Host is not ARM64'

# NFS-mounted repository must refer to the SAME commit as on sparta.
repo="$HOME/src/ai-rust/hybrid_inference"
[[ -d "$repo/.git" ]] || fail "Missing host checkout: $repo"
[[ "$(git -C "$repo" rev-parse HEAD)" == "$expected" ]] || \
    fail 'Host checkout differs from sparta'

# Container mount path and image are established by the aihailo setup.
docker exec -i "$container" bash -s -- "$expected" <<'CONTAINER'
set -uo pipefail
expected="$1"
fail() { printf 'FAIL (aihailo): %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == aarch64 ]] || fail 'Container is not ARM64'
[[ "$(rustc -vV | sed -n 's/^host: //p')" == aarch64-unknown-linux-gnu ]] || \
    fail 'Rust toolchain is not native aarch64-unknown-linux-gnu'

cd /workspace/hybrid_inference || fail 'Missing container checkout'
[[ "$(git rev-parse HEAD)" == "$expected" ]] || \
    fail 'Container checkout differs from sparta'

cargo fmt --all --check || fail 'Formatting failed'
cargo build --all-targets || fail 'ARM64 build failed'
cargo test --all-targets || fail 'ARM64 tests failed'
CONTAINER
REMOTE

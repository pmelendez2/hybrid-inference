#!/usr/bin/env bash
# Native ARM64 build/tests against the exact Git commit being checked.
set -uo pipefail

cd "$(git rev-parse --show-toplevel)" || exit 1
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

for tool in git ssh; do
    command -v "$tool" >/dev/null 2>&1 || fail "$tool not found"
done

host="$(git config --local --get hybrid.arm64Host || true)"
container="$(git config --local --get hybrid.arm64Container || true)"
host_repo="$(git config --local --get hybrid.arm64HostRepo || true)"
container_repo="$(git config --local --get hybrid.arm64ContainerRepo || true)"

[[ -n "$host" ]] || fail 'Set hybrid.arm64Host in local Git config'
[[ -n "$container" ]] || fail 'Set hybrid.arm64Container in local Git config'
[[ -n "$host_repo" ]] || fail 'Set hybrid.arm64HostRepo in local Git config'
[[ -n "$container_repo" ]] || fail 'Set hybrid.arm64ContainerRepo in local Git config'

# SSH sends remote arguments through a shell. Restrict values to safe characters.
[[ "$host" =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'Unsafe SSH host alias'
[[ "$container" =~ ^[A-Za-z0-9_.-]+$ ]] || fail 'Unsafe container name'
for repo in "$host_repo" "$container_repo"; do
    [[ "$repo" == /* ]] || fail 'Repository paths must be absolute'
    [[ "$repo" =~ ^/[A-Za-z0-9_./-]+$ ]] || \
        fail 'Repository paths contain unsupported characters'
done

expected="$(git rev-parse HEAD)" || exit 1
printf '[arm64] host=%s container=%s commit=%s\n' \
    "$host" "$container" "$expected"

ssh -T -o BatchMode=yes "$host" bash -s -- \
    "$container" "$expected" "$host_repo" "$container_repo" <<'REMOTE'
set -uo pipefail
container="$1"
expected="$2"
host_repo="$3"
container_repo="$4"
fail() { printf 'FAIL (ARM64 host): %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == aarch64 ]] || fail 'Host is not ARM64'
[[ -d "$host_repo/.git" ]] || fail 'Host checkout is missing'
actual="$(git -c "safe.directory=$host_repo" \
    -C "$host_repo" rev-parse HEAD)" || exit 1
[[ "$actual" == "$expected" ]] || fail 'Host checkout commit mismatch'

docker exec -i "$container" bash -s -- \
    "$expected" "$container_repo" <<'CONTAINER'
set -uo pipefail
expected="$1"
repo="$2"
fail() { printf 'FAIL (ARM64 container): %s\n' "$*" >&2; exit 1; }

[[ "$(uname -m)" == aarch64 ]] || fail 'Container is not ARM64'
[[ "$(rustc -vV | sed -n 's/^host: //p')" == \
    aarch64-unknown-linux-gnu ]] || fail 'Rust target is not native ARM64'
cd "$repo" || fail 'Container checkout is missing'
actual="$(git -c "safe.directory=$repo" rev-parse HEAD)" || exit 1
[[ "$actual" == "$expected" ]] || fail 'Container checkout commit mismatch'

cargo fmt --all --check || fail 'Rust formatting failed'
cargo build --all-targets || fail 'ARM64 build failed'
cargo test --all-targets || fail 'ARM64 tests failed'
CONTAINER
REMOTE

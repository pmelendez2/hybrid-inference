# Testing and verified environments

Keep verified environments and test results here rather than making broad
operating-system compatibility claims in the README.

## Local checks

Run from the repository root:

```bash
cargo fmt --all --check
cargo build --all-targets
cargo test --all-targets
git diff --check
```

Bash script syntax:

```bash
bash -n \
    .githooks/pre-commit \
    .githooks/pre-push \
    scripts/check-publish.sh \
    scripts/test-arm64.sh
```

Security checks:

```bash
gitleaks git --log-opts='--all' --redact .
gitleaks dir --redact .
```

## Publication preflight

From a clean checkout, if you have configured the optional ARM64
maintainer test environment:

```bash
./scripts/check-publish.sh
```

The preflight requires a clean checkout and runs whitespace, formatting,
Gitleaks history/directory scans, a check for private infrastructure
identifiers, and the native ARM64 build/test script. It never publishes by
itself. See [docs/PUBLISHING.md](docs/PUBLISHING.md).

## Verified on November 11, 2026

| Configuration | Result |
| --- | --- |
| Target platform | Raspberry Pi 5, native Linux ARM64 |
| Development container | ARM64, `rust-ai-dev:1.99` |
| HailoRT environment | HailoRT 4.23; shared MPS service available |
| `master` | Native ARM64 build and 30 Rust tests passed |
| `feature/hailort-mps-support` | Native ARM64 build and 30 Rust tests passed |
| `scripts/check-publish.sh` | Passed for both branches |
| Gitleaks | No leaks detected in scans of both branches |

These are specific verification results, not a guarantee of compatibility
with every operating system, hardware model, or HailoRT release.

## Hardware testing

Unit tests and compilation are separate from a real HEF inference run. A
previous MPS run completed alongside another inference client using the
shared Hailo-8 service,
but the final publication checks above did **not** repeat that hardware
inference run. Do not claim device concurrency was revalidated by the unit
suite alone.

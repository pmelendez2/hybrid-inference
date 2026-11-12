# Secure publication workflow

Public GitHub publication is deliberate, never automatic.
This repository retains `master` (non-MPS) and
`feature/hailort-mps-support` (MPS) until reviewed through a PR.

## Setup on sparta

Configure this clone (not global Git configuration):

```bash
git config --local core.hooksPath .githooks
git config --local hybrid.arm64Host picam
```

Identify the running ARM64 `aihailo` container on picam:

```bash
ssh picam 'docker ps --format "{{.Names}} {{.Image}}"'
```

Set its exact name (replace the example value):

```bash
git config --local hybrid.arm64Container YOUR_AIHAILO_CONTAINER
```

The ARM64 test script verifies host architecture, the container architecture,
Rust's native target, and that the checkout commit on picam and inside the
container matches sparta. It then runs `cargo fmt --all --check`,
`cargo build --all-targets` and `cargo test --all-targets` inside that
container. No x86 fallback or no-op command is available.

The NFS checkout is expected at `$HOME/src/ai-rust/hybrid_inference` on
picam, mounted in the container as `/workspace/hybrid_inference`.

## Hooks and checks

- Pre-commit rejects staged whitespace, Rust formatting errors, and staged
  Gitleaks findings.
- Pre-push refuses deletions, tag pushes, non-current branches, and direct
  updates to an existing `master` branch.
- The initial `master` push is allowed only if the remote lacks `master`.
- Pre-push runs `scripts/check-publish.sh` before asking for the exact
  interactive word `PUBLISH`; anything else rejects the push.
- The full check includes Git-history and working-tree Gitleaks scans,
  a targeted private infrastructure search, and required native ARM64
  compilation and tests.

Run checks manually from the repository root:

```bash
./scripts/check-publish.sh
```

To publish both branches, **switch to each branch before its push**, so that
ARM64 checks run against the exact commit being published. The safeguards
must be committed on both branches before publishing either branch.

## Signed commits, PRs, and CI

Commit using signed Git commits on sparta. After the first remote bootstrap,
all changes to `master` must go through a pull request. Configure GitHub
branch protection and required CI before accepting outside contributions.

Local hooks only protect clones that enable `core.hooksPath`; they can be
bypassed and are not a substitute for GitHub branch protection.

Git history rewriting invalidates old commit signatures. Verify rewritten
commit signatures separately. Never commit local `.env` files or reports.

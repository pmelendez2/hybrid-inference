# Contributing

Contributions are welcome. See the project's
[Code of Conduct](CODE_OF_CONDUCT.md).

## Before starting

- Open an issue or discuss larger changes before implementing them.
- Work from a feature branch; do not push directly to `master` after the
  repository's initial bootstrap.
- Keep non-MPS behavior working. MPS changes should remain opt-in until
  reviewed and merged.
- Do not add secrets, private hostnames, local credentials, sample images
  containing private information, or machine-specific infrastructure details.
- Keep Rust formatting consistent with the repository's configuration.

## Verify your change

```bash
cargo fmt --all --check
cargo build --all-targets
cargo test --all-targets
git diff --check
```

For changes that require HailoRT, run tests on a host with compatible hardware
and runtime libraries. The maintainer-specific `scripts/test-arm64.sh` requires
local configuration and is not a prerequisite for external contributors.
Explain what checks were possible on your own system.

## Submit changes

Create a pull request with a description, test evidence, and any known
hardware or runtime limitations. Do not upload private inference data in PR
logs. Maintainer commits are signed; maintainers follow the publication
procedures in [docs/PUBLISHING.md](docs/PUBLISHING.md).

The `PUBLISH` gate is a local maintainer control, not a substitute for remote
branch protection or required CI checks.

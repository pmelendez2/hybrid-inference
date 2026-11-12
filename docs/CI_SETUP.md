# Continuous integration and publication

The GitHub Actions workflow `.github/workflows/ci.yml` provides a public,
portable `isolated` check: shell syntax, Rust formatting, committed whitespace,
Gitleaks scanning and a private-infrastructure reference scan.

CI **does not** compile or test HailoRT-dependent binaries on the hosted runner.
The HailoRT SDK and device are not available there. Maintainers must run the
native ARM64 preflight independently before publishing. A successful hosted
CI job is not evidence of an ARM64 compile, tests, or device inference.

## Configure native ARM64 preflight

The default publication checks are fail-closed. Configure these keys in the
*local repository's Git configuration*; do not commit the values:

```bash
git config --local core.hooksPath .githooks
git config --local hybrid.arm64Host YOUR_SSH_HOST
git config --local hybrid.arm64Container YOUR_CONTAINER
git config --local hybrid.arm64HostRepo /absolute/path/on/arm64/host
git config --local hybrid.arm64ContainerRepo /absolute/path/in/container
```

Both the ARM64 host and container must have the same Git commit checked out
as the workstation. Their paths can be different; no shared filesystem or
specific mount layout is required, but the selected sources must match the
commit. The container must contain Rust and HailoRT development dependencies.

Run from the Git checkout:

```bash
./scripts/test-arm64.sh
./scripts/check-publish.sh
```

The host and container architecture must both be `aarch64`, and Rust must
report `aarch64-unknown-linux-gnu`. A scoped `safe.directory` option is used
for the inspected checkouts; no global trust exception is needed.

## Protecting the default branch

After the initial repository bootstrap, enable a GitHub ruleset or branch
protection rule for `master`: require pull requests, require successful
`isolated` status checks, block force pushes and deletions. Local hooks are
not a substitute for server-side protection.

`pre-push` requires an interactive terminal and the exact `PUBLISH` response.
It rejects direct updates of an existing remote `master`, deletions, and tags.
Do not bypass the hooks to publish.

For security, never expose privately networked HailoRT hardware as an
unrestricted GitHub Actions self-hosted runner for untrusted pull requests.

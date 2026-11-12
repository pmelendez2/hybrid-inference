# Publishing workflow

Public publication is deliberate, never automatic. The `master` branch
contains the non-MPS baseline; `feature/hailort-mps-support` adds MPS support.
Use signed commits, pull requests and server-side branch protection.

## Local Git configuration

Enable repository hooks and configure a reachable native ARM64 test host and
an ARM64 Rust/HailoRT development container using the local Git configuration.
The host and container checkout locations are independent and configurable;
no specific machine names, filesystem paths or shared mounts are assumed.

```bash
git config --local core.hooksPath .githooks
git config --local hybrid.arm64Host YOUR_SSH_HOST
git config --local hybrid.arm64Container YOUR_CONTAINER
git config --local hybrid.arm64HostRepo /absolute/path/on/host
git config --local hybrid.arm64ContainerRepo /absolute/path/in/container
```

Run `./scripts/check-publish.sh` from the repository root. The script
requires a clean working tree, committed whitespace and formatting checks,
Git-history and working-directory Gitleaks scans, a private-infrastructure
scan and successful native ARM64 compilation and unit tests. It verifies
the exact commit in both checkouts. See [CI setup](CI_SETUP.md).

## Hooks

- `pre-commit` validates staged changes before permitting a signed commit.
- `pre-push` refuses deletions, tags, non-current-branch pushes and direct
  updates to an existing remote `master`.
- Initial publication of `master` is allowed only if the remote does not
  yet have that branch.
- Every push requires preflight checks and the exact interactive word
  `PUBLISH`; otherwise it is rejected.

Switch to each branch before its first push so the exact pushed revision is
verified. Never bypass the hooks merely to publish. After bootstrapping the
remote, use a PR to update the default branch.

## GitHub CI and branch protection

The `isolated` GitHub Actions job provides portable syntax, formatting and
security checks. It does **not** verify compilation against the HailoRT SDK
or live device inference. ARM64 compilation/tests are required separately by
the local publication preflight.

After bootstrapping the repository, configure a ruleset for `master` requiring
pull requests and the successful `isolated` check. Disable force pushes and
branch deletion. Never attach private hardware as an unrestricted runner for
untrusted public pull requests.

Git history rewriting invalidates commit signatures. Verify newly written
commit signatures. Do not commit private environment settings or reports.

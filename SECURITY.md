# Security policy

## Reporting security issues

Do not disclose working credentials, private keys, sensitive files, or exploit
proofs in public GitHub issues or pull requests.

If the repository's GitHub **Report a vulnerability** / private security
advisory feature is available, use that channel. If it is unavailable, contact
the maintainer through an established private channel before sharing details.
No unpublished email address or response-time guarantee is implied here.

## Expectations

- Keep secrets and private infrastructure references out of the repository,
  including Git history and build logs.
- Run Gitleaks against history and the working tree before publication.
- Validate dependencies and hardware-specific runtime changes independently.
- Treat a clean secret scan as a useful check, not proof that no sensitive
  information exists.

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for the enforced local workflow.

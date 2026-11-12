# AI-assisted development

AI tools may assist with planning, documentation, code review, test design,
and implementation. The human contributor remains responsible for every
change and its effects.

## Rules

- Review generated code line by line before committing it.
- Verify build, formatting, tests, security scans, and runtime behavior in the
  appropriate environment. AI output is not verification evidence.
- Never supply production credentials, private keys, customer data, private
  inference images, or internal infrastructure configuration to external AI
  services.
- Check generated code and text for license obligations, provenance concerns,
  and unsupported assertions.
- Make small-enough changes to review while preferring coherent, complete
  implementation units over untested fragments.
- Preserve explicitly chosen architecture and compatibility constraints.
- Do not let an AI tool circumvent hooks, the `PUBLISH` requirement, branch
  protections, signed commits, or manual deployment approval.

## Evidence and disclosure

Provide reproducible verification commands and results in pull requests.
Describe material AI assistance when it helps reviewers understand the
change, but do not present untested AI suggestions as completed work.

All published source changes belong in Git; no separate source-code upload is
part of the supported release workflow.

See [CONTRIBUTING.md](CONTRIBUTING.md), [TESTING.md](TESTING.md), and
[docs/PUBLISHING.md](docs/PUBLISHING.md).

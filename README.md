# Hybrid Inference

Rust experiments and examples for local AI inference, including Hailo-8
inference through HailoRT and interaction with an Ollama server.

The `master` branch supports both direct-device inference and optional HailoRT
multi-process service (MPS) inference for shared Hailo-8 environments.

Direct-device mode remains the default. The HEF example supports
`--multi-process-service` and `--group-id` (default `SHARED`).
See [Testing](TESTING.md) for verified hardware results.

## Requirements

Install only the components needed for the workflows you intend to run:

- [Git](https://git-scm.com/downloads) for source control.
- [Rust and Cargo](https://rust-lang.org/tools/install/) via `rustup`, with a
  toolchain supporting the edition in `Cargo.toml`.
- [C/C++ build tools](https://ubuntu.com/developers/docs/howto/rust-setup/)
  and [CMake](https://cmake.org/download/) for native dependencies.
- [HailoRT](https://github.com/hailo-ai/hailort) and the compatible
  [Hailo PCIe driver](https://github.com/hailo-ai/hailort) for Hailo-8
  examples. Obtain the device-specific installation materials from the
  [Hailo Developer Zone](https://hailo.ai/developer-zone/).
- [Raspberry Pi AI hardware documentation](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
  when using a Raspberry Pi AI accelerator.
- [HailoRT multi-process service](INSTALLATION.md#2-build-and-install-hailort-4230-with-mps-new-host)
  and a permitted Unix socket connection when using optional MPS mode.
- [Ollama installation](https://ollama.com/download/linux) for workflows
  that call a local Ollama server; see its
  [Linux installation guide](https://docs.ollama.com/linux). Configure the
  server address with `OLLAMA_HOST` (default `http://localhost:11434`).
- [Docker Engine](https://docs.docker.com/engine/install/) for the optional
  containerized ARM64 development and testing setup.
- [Gitleaks](https://github.com/gitleaks/gitleaks) for the repository's
  local security and publication checks.

See [INSTALLATION.md](INSTALLATION.md) for setup procedures and
[TESTING.md](TESTING.md) for the tested environment. These requirements are
not a restriction to one Ubuntu release.

## Get started

See [INSTALLATION.md](INSTALLATION.md) for setup and build prerequisites.
The main CLI and example arguments can be inspected after the dependencies are
available:

```bash
cargo run -- --help
cargo run --example hailort_hef -- --help
```

The HailoRT example requires HailoRT development libraries at build/link time;
`cargo run` is not expected to work on an unconfigured host.

## Project documentation

- [Installation](INSTALLATION.md)
- [Testing and compatibility](TESTING.md)
- [Contributing](CONTRIBUTING.md)
- [AI-assisted development](AIUSAGE.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
- [Security](SECURITY.md)
- [Publishing](docs/PUBLISHING.md)

## Scope

This repository is for inference development and examples. It does not deploy
or reconfigure another application, the HailoRT host service, or the device itself.

## License

No license is asserted by this document. Review the repository's `LICENSE`
file, if present, before copying or redistributing code. Adding an explicit
license is a separate maintainer decision.

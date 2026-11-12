# Installation

This guide covers the Rust application and installing/configuring the HailoRT
multi-process service (MPS) on a Linux host with a Hailo-8 device. A tested configuration uses a Raspberry Pi 5 with HailoRT **4.23.0**.
Other hardware and HailoRT releases may require different driver packages.

> **Existing shared device:** If another client is already using the
> Hailo-8, **do not install over, restart, or reconfigure its live HailoRT
> service**. Review the inspection commands first, and schedule any service
> changes for a maintenance window. Installation commands below are for a new
> or intentionally maintained host.

## 1. Prerequisites and device driver

Install [Git](https://git-scm.com/downloads),
[Rust/Cargo](https://rust-lang.org/tools/install/), a C/C++ compiler,
[CMake](https://cmake.org/download/), and the dependencies required by the
selected HailoRT source release. For Hailo-8 PCIe devices, the matching
Hailo PCIe kernel driver must
be installed and loaded separately from the user-space HailoRT library.
Obtain the driver package appropriate for your kernel and platform from the
[Hailo developer resources](https://github.com/hailo-ai/hailort) and follow its
packaged installation/DKMS instructions; this project does not provide a
driver or firmware installer.

Before changing anything on an existing host:

```bash
uname -m
ls -l /dev/hailo* 2>/dev/null || true
lsmod | grep '^hailo_pci' || true
command -v hailortcli || true
hailortcli --version 2>/dev/null || true
hailortcli scan 2>/dev/null || true
systemctl status hailort.service --no-pager || true
systemctl cat hailort.service || true
```

If the device node is missing, resolve the driver/PCIe issue before proceeding.
`hailortcli scan` should discover the Hailo device when the runtime and driver
are working. Avoid mixing HailoRT library, CLI, and service versions.

## 2. Build and install HailoRT 4.23.0 with MPS (new host)

These commands compile HailoRT and its multi-process service natively on the
Linux target. The essential CMake setting is `HAILO_BUILD_SERVICE=ON`, as declared in
the HailoRT v4.23.0 CMake configuration. Use a
source directory **outside** this project's checkout.

```bash
sudo apt update
sudo apt install -y \
    git build-essential cmake pkg-config

mkdir -p ~/src
cd ~/src
git clone https://github.com/hailo-ai/hailort.git
cd hailort
git checkout v4.23.0

cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DHAILO_BUILD_SERVICE=ON

cmake --build build --parallel "$(nproc)"
sudo cmake --install build
sudo ldconfig
```

CMake may report additional version-specific dependencies; install the
prerequisites documented by the pinned HailoRT source instead of substituting
incompatible system libraries. The HailoRT installation may also install a
systemd unit. Check what was installed before adding any unit manually:

```bash
command -v hailort_service
command -v hailortcli
ldconfig -p | grep libhailort
systemctl cat hailort.service
```

If `systemctl cat hailort.service` reports that no unit exists, verify that the
binary was installed and install the unit supplied by this **same** HailoRT
source checkout:

```bash
cd ~/src/hailort
find . -type f -name 'hailort.service' -print

# After checking the vendor unit location and ExecStart, install it:
# sudo install -m 0644 PATH_FROM_FIND \
#     /etc/systemd/system/hailort.service
# sudo systemctl daemon-reload
# systemctl cat hailort.service
```

Review `ExecStart` in the unit and confirm it points to the installed
`hailort_service` binary before enabling the service. Do not overwrite an
existing unit simply because its location differs.

## 3. Configure the Unix socket for shared clients

The established Linux setup exposes the MPS service socket at
`/run/hailort/hailort_uds.sock`. Using a dedicated group permits authorized
non-root host clients and containers to connect without making the socket
world-writable.

For a **new** host, create the group if necessary and configure a systemd
drop-in:

```bash
getent group hailort || sudo groupadd --system hailort

sudo install -d -m 0755 \
    /etc/systemd/system/hailort.service.d
sudo vi /etc/systemd/system/hailort.service.d/override.conf
```

Use this drop-in, matching the established working configuration:

```ini
[Service]
RuntimeDirectory=hailort
RuntimeDirectoryMode=0755
ExecStartPost=/usr/bin/chgrp hailort /run/hailort/hailort_uds.sock
ExecStartPost=/usr/bin/chmod 0660 /run/hailort/hailort_uds.sock
```

The service must create its socket before the `ExecStartPost` commands run;
check the vendor unit's startup ordering and the service journal if those
commands fail. The established setup also has a short vendor-supplied
post-start delay. Do not assume this drop-in is portable to a service with a
different socket path or startup model.

After reviewing the unit, on a **new/inactive** installation only:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now hailort.service
systemctl is-active hailort.service
sudo systemctl status hailort.service --no-pager
sudo journalctl -u hailort.service -n 50 --no-pager
stat -c '%A %a %U %G %n' \
    /run/hailort/hailort_uds.sock
```

Expected socket access in the established configuration is mode `660` and
group `hailort`. The user account running inference must be authorized for
that group. To authorize a **specific trusted local user**, an administrator
may use `sudo usermod -aG hailort USERNAME`; the user must start a new login
session before the new membership normally takes effect.

Check the exact group number rather than assuming it is identical on every
host:

```bash
getent group hailort
id
```

## 4. Configure clients and Docker containers

MPS clients need the same socket address and access to the host socket:

```bash
export HAILORT_SERVICE_ADDRESS=unix:///run/hailort/hailort_uds.sock
```

For a Docker client on the **same host**, the relevant Compose service
configuration is:

```yaml
services:
  inference:
    environment:
      HAILORT_SERVICE_ADDRESS: unix:///run/hailort/hailort_uds.sock
    volumes:
      - /run/hailort:/run/hailort
    group_add:
      - "${HAILORT_GID:?Set HAILORT_GID from the host hailort group}"
```

Provide `HAILORT_GID` from `getent group hailort | cut -d: -f3` through the
appropriate Compose environment/configuration. The container image also
needs matching HailoRT client libraries and any hardware access required by
its workload; the socket mount by itself does not install those dependencies.

Verify access using your container name (replace `INFERENCE_CONTAINER`):

```bash
docker exec INFERENCE_CONTAINER id
docker exec INFERENCE_CONTAINER \
    stat -c '%A %a %U %G %n' \
    /run/hailort/hailort_uds.sock
```

When using MPS with the Rust HEF example on `master`, pass
`--multi-process-service` and use the same `--group-id SHARED` across
clients intended to share a device. Inspect
`cargo run --example hailort_hef -- --help` for model, input, and other
required options. Direct-device mode remains the default; it is not
interchangeable with shared MPS operation while another process owns the
device.

## 5. Obtain and build hybrid_inference

After the GitHub repository is published:

```bash
git clone https://github.com/pmelendez2/hybrid-inference.git
cd hybrid-inference
```

For a pre-existing checkout, navigate to its repository root and select
the current mainline branch:

```bash
git switch master
```

Both direct-device and optional MPS inference are available on `master`.

On a host/container with matching HailoRT headers and libraries:

```bash
cargo fmt --all --check
cargo build --all-targets
cargo test --all-targets
cargo run --example hailort_hef -- --help
```

Building and unit tests do not themselves confirm that a real inference ran
or that multiple clients safely share the accelerator.

## 6. Build and test on your own system

Run the build and tests on a Linux host or development container with an
appropriate Rust toolchain, compiler, HailoRT headers, and compatible HailoRT
libraries. ARM64 builds should execute on an ARM64 host or use a correctly
configured cross-compilation toolchain. This project does not require a
particular hostname, shared filesystem, mount point, or container name.

From the repository root:

```bash
cargo fmt --all --check
cargo build --all-targets
cargo test --all-targets
```

For the optional container setup, map the repository into a path of your
choosing and make the required HailoRT libraries and device access available.
Use the `HAILORT_SERVICE_ADDRESS` and socket permissions described above if
using MPS. Do not copy another developer's mount layout or group IDs.

The repository also contains maintainer-specific automation in
`scripts/test-arm64.sh` and `scripts/check-publish.sh`. These scripts currently
assume a configured remote ARM64 host and development container. They are not
required for ordinary installation or local testing; adapt the automation to
your own environment before using it. For publication policy, see
[docs/PUBLISHING.md](docs/PUBLISHING.md).

## 7. Troubleshooting and operational safety

- **`dubious ownership` in a container:** verify that the checkout is trusted
  and use a narrowly scoped `git -c safe.directory=PATH` for specific Git
  commands rather than trusting every directory globally.
- **Cannot connect to MPS:** inspect `systemctl status hailort.service`,
  `journalctl -u hailort.service`, the socket's owner/mode/group, container
  mount, `HAILORT_SERVICE_ADDRESS`, and the client's supplemental groups.
- **Mismatched library/runtime:** use the same compatible HailoRT version for
  host service, client library, and CLI; verify with the installed tooling.
- **Socket disappears after reboot:** verify the service is enabled and the
  `RuntimeDirectory` and drop-in settings are loaded; verify permissions
  after an **authorized maintenance restart**, not by disrupting live users.
- **Active inference clients:** do not restart the service, alter socket
  ownership, replace driver/runtime libraries, or change device access in a
  production session merely to satisfy a development check.

Upstream references: [HailoRT source](https://github.com/hailo-ai/hailort),
[HailoRT v4.23.0 source tree](https://github.com/hailo-ai/hailort/tree/v4.23.0),
and [Hailo documentation](https://hailo.ai/developer-zone/).

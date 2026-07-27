# uv-torch-compass

[日本語](https://github.com/Onely7/uv_torch_compass/blob/main/README_ja.md)

**By using `uv-torch-compass`, you can test official PyTorch package indexes against both your version requirements and the current Linux machine, then safely write the first verified choice to the target project's `pyproject.toml`.**

An index is a package download location. PyTorch publishes separate official indexes for CPU and NVIDIA CUDA builds. This tool checks more than whether a package can be installed. By default, it rejects CUDA builds newer than the selected NVIDIA driver normally supports, then runs PyTorch, NumPy, the selected GPU, cuBLAS, cuDNN, and optional `torchvision` or `torchaudio` checks before applying a choice.

## Quick start

You need Linux, a recent [uv](https://docs.astral.sh/uv/), internet access, and Python 3.10–3.14. PyTorch may be declared directly or introduced by another selected package such as `vllm`.

For a minimal project:

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5"]
```

For a framework that depends on PyTorch, keep its real dependency in the project:

```toml
dependencies = ["vllm==0.19.1"]
```

The candidate environment resolves the complete selected dependency graph. If `vllm` requires a particular `torch`, `torchvision`, or `torchaudio` version, that constraint participates in backend selection. The applied configuration adds only the direct source anchors needed for uv's explicit PyTorch index, records those anchors as tool-managed state, and preserves the framework requirement.

If no allowed CUDA index contains the required PyTorch build, the command fails before changing the project. The summary identifies the package and requirement, the package that introduced it, the attempted index, and practical next steps; complete redacted uv output remains in the private log.

From the target project, run a version published on PyPI to verify a candidate and preview the change:

```bash
uvx uv-torch-compass plan
```

To try a local checkout or wheel instead, select it explicitly:

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass plan
```

If the plan is suitable, apply it. This updates `pyproject.toml`, locks the workspace, synchronizes the selected project environment, and verifies the result again:

```bash
uvx uv-torch-compass apply
```

Later, validate the recorded source, lockfile, synchronized environment, and installed runtime without changing them:

```bash
uvx uv-torch-compass check
```

When using a local checkout, replace `/path/to/uv_torch_compass` with this repository's path and keep the same `--from` prefix for `apply` and `check`. Add `--pyproject /path/to/project/pyproject.toml` when running from another directory.

## Choosing what to test

The default `--backend auto --cuda-compatibility strict` policy behaves as follows:

- When an NVIDIA GPU is visible, it tests only concrete CUDA builds that the driver normally supports, from newer to older. It does not silently switch to CPU when those candidates fail.
- When no NVIDIA GPU is visible, it tests the official CPU build.

The command stops at the first candidate that passes. It does not benchmark every candidate or claim to choose the fastest build. This also means that a machine whose `nvidia-smi` output says `CUDA Version: 12.4` will not accept `cu129` under the default policy.

You can narrow the policy:

```bash
uv-torch-compass plan --backend cpu
uv-torch-compass plan --backend cuda
uv-torch-compass plan --backend cu128
uv-torch-compass plan --channel nightly
uv-torch-compass plan --probe-profile compile
```

`stable` is the default channel. `nightly` is used only when explicitly selected. CUDA minor-version compatibility is also opt-in with `--cuda-compatibility minor`; it can use a newer CUDA runtime within the same major family, but a successful result is reported with a warning. See [backend and runtime selection](https://github.com/Onely7/uv_torch_compass/blob/main/docs/how-it-works.md) for the exact order and checks.

## Safety at a glance

- `plan` installs and tests candidates in temporary environments but does not change the target `pyproject.toml`, `uv.lock`, or project environment.
- `apply` creates timestamped backups and treats a workspace member's `pyproject.toml` and the shared root `uv.lock` as one transaction.
- Before changing the project environment, `apply` locks the complete graph and performs a locked sync dry run. It also records the current Linux architecture as a required uv environment, so unavailable wheels fail before installation starts.
- Writes use same-directory temporary files and atomic replacement. A workspace lock prevents two `apply` processes from updating together.
- Lock, sync, final validation, timeout, SIGINT, and SIGTERM failures trigger file rollback and an environment recovery attempt.
- Logs and JSON reports redact common credential forms and are created with private file permissions.

Review `git diff` after `plan` and `apply`. Backups remain after success; [recovery and troubleshooting](https://github.com/Onely7/uv_torch_compass/blob/main/docs/recovery.md) explains their names and limitations.

## Supported scope

- `plan`, `apply`, and `check` run on Linux. `--help` and `--version` work on other systems.
- CPU and NVIDIA CUDA builds are supported. AMD ROCm and Intel XPU are rejected.
- Stable and nightly official PyTorch indexes are supported; stable never falls back to nightly automatically.
- Base dependencies, selected extras, selected dependency groups, uv workspaces, `torchvision`, and `torchaudio` are supported.
- CUDA success requires GPU tensor, cuBLAS, cuDNN, architecture, and selected companion-package checks. `--probe-profile compile` additionally tests `torch.compile`.
- NVIDIA drivers are never installed or updated. Re-run `plan` and `apply` after updating a driver.
- Exit codes are `0` for success, `1` for configuration or operational failure, and `2` for invalid command syntax.

## Documentation

| Goal | Guide |
| --- | --- |
| Learn the commands and options | [CLI usage](https://github.com/Onely7/uv_torch_compass/blob/main/docs/usage.md) |
| Configure project defaults and environment variables | [Configuration](https://github.com/Onely7/uv_torch_compass/blob/main/docs/configuration.md) |
| Understand the process flow, backend, channel, GPU, Python, and runtime checks | [How selection works](https://github.com/Onely7/uv_torch_compass/blob/main/docs/how-it-works.md) |
| Use extras, groups, and workspaces | [Projects and dependency scopes](https://github.com/Onely7/uv_torch_compass/blob/main/docs/projects-and-scopes.md) |
| Consume text and JSON results | [Reports and automation](https://github.com/Onely7/uv_torch_compass/blob/main/docs/reports.md) |
| Recover files or diagnose a failure | [Recovery and troubleshooting](https://github.com/Onely7/uv_torch_compass/blob/main/docs/recovery.md) |
| Test, build, and prepare artifacts | [Development](https://github.com/Onely7/uv_torch_compass/blob/main/docs/development.md) |

See the [documentation index](https://github.com/Onely7/uv_torch_compass/blob/main/docs/README.md) for the complete map.

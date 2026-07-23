# uv-torch-compass

[日本語](README_ja.md)

**By using `uv-torch-compass`, you can test official PyTorch package indexes against both your version requirements and the current Linux machine, then safely write the first verified choice to the target project's `pyproject.toml`.**

An index is a package download location. PyTorch publishes separate official indexes for CPU and NVIDIA CUDA builds. This tool checks more than whether a package can be installed: it runs PyTorch, NumPy, the selected GPU, and optional `torchvision` or `torchaudio` checks before applying a choice.

> [!IMPORTANT]
> This project is not published to PyPI yet. Use `uvx --from <wheel-or-source> uv-torch-compass` with a local checkout or wheel. The shorter `uvx uv-torch-compass` command is reserved for a future PyPI release.

## Quick start

You need Linux, a recent [uv](https://docs.astral.sh/uv/), internet access, and Python 3.10–3.14. The target `pyproject.toml` must contain `torch`, `torchvision`, or `torchaudio` in base dependencies or in a selected extra or dependency group.

For a minimal project:

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5"]
```

From the target project, first verify a candidate and preview the change:

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass plan
```

If the plan is suitable, apply it. This updates `pyproject.toml`, locks the workspace, synchronizes the selected project environment, and verifies the result again:

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass apply
```

Later, validate the recorded source, lockfile, synchronized environment, and installed runtime without changing them:

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass check
```

Replace `/path/to/uv_torch_compass` with this repository's path. Add `--pyproject /path/to/project/pyproject.toml` when running from another directory.

## Choosing what to test

The default `--backend auto` policy tries uv's automatic selection, compatible CUDA candidates advertised by the installed uv, and CPU in that order. It stops at the first candidate that passes; it does not benchmark all candidates or claim to choose the fastest one.

You can narrow the policy:

```bash
uv-torch-compass plan --backend cpu
uv-torch-compass plan --backend cuda
uv-torch-compass plan --backend cu128
uv-torch-compass plan --channel nightly
```

`stable` is the default channel. `nightly` is used only when explicitly selected. See [backend and runtime selection](docs/how-it-works.md) for the exact order and checks.

## Safety at a glance

- `plan` installs and tests candidates in temporary environments but does not change the target `pyproject.toml`, `uv.lock`, or project environment.
- `apply` creates timestamped backups and treats a workspace member's `pyproject.toml` and the shared root `uv.lock` as one transaction.
- Writes use same-directory temporary files and atomic replacement. A workspace lock prevents two `apply` processes from updating together.
- Lock, sync, final validation, timeout, SIGINT, and SIGTERM failures trigger file rollback and an environment recovery attempt.
- Logs and JSON reports redact common credential forms and are created with private file permissions.

Review `git diff` after `plan` and `apply`. Backups remain after success; [recovery and troubleshooting](docs/recovery.md) explains their names and limitations.

## Supported scope

- `plan`, `apply`, and `check` run on Linux. `--help` and `--version` work on other systems.
- CPU and NVIDIA CUDA builds are supported. AMD ROCm and Intel XPU are rejected.
- Stable and nightly official PyTorch indexes are supported; stable never falls back to nightly automatically.
- Base dependencies, selected extras, selected dependency groups, uv workspaces, `torchvision`, and `torchaudio` are supported.
- CUDA success requires a real tensor calculation on the selected GPU. NVIDIA drivers are never installed or updated.
- Exit codes are `0` for success, `1` for configuration or operational failure, and `2` for invalid command syntax.

## Documentation

| Goal | Guide |
| --- | --- |
| Learn the commands and options | [CLI usage](docs/usage.md) |
| Configure project defaults and environment variables | [Configuration](docs/configuration.md) |
| Understand the process flow, backend, channel, GPU, Python, and runtime checks | [How selection works](docs/how-it-works.md) |
| Use extras, groups, and workspaces | [Projects and dependency scopes](docs/projects-and-scopes.md) |
| Consume text and JSON results | [Reports and automation](docs/reports.md) |
| Recover files or diagnose a failure | [Recovery and troubleshooting](docs/recovery.md) |
| Test, build, and prepare artifacts | [Development](docs/development.md) |
| Publish verified distributions | [Publishing](docs/publishing.md) |

See the [documentation index](docs/README.md) for the complete map.

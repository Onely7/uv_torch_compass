# Projects and dependency scopes

[日本語](projects-and-scopes_ja.md) · [Documentation index](README.md) · [CLI usage](usage.md)

The base dependency list is always selected. Repeated `--extra` and `--group` options add optional dependency scopes for candidate and final verification.

## Base dependencies

Declare ordinary PEP 508 requirements in `[project].dependencies`:

```toml
[project]
name = "trainer"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5,<3"]
```

`--torch`, `--torchvision`, and `--torchaudio` add or replace requirements in this base list. PyTorch direct URL and Git requirements are rejected because they bypass the official index that the tool is meant to verify.

## Extras and dependency groups

Extras and groups are selected by name:

```toml
[project.optional-dependencies]
vision = ["torchvision>=0.20"]

[dependency-groups]
audio = ["torchaudio>=2.5"]
training = [
    { include-group = "audio" },
    "pytest>=8",
]
```

```bash
uv-torch-compass plan --extra vision --group training
```

An unknown scope, unsupported group entry, include cycle, malformed requirement, conflicting exact PyTorch version, or a selection with no applicable PyTorch package is rejected before target files are changed.

If a selected scope effectively contains `torchvision` or `torchaudio` but no `torch`, the tool adds `torch` to that same scope in the proposed update. Include-group expansion is considered, so `training` in the example receives its own `torch` declaration.

Requirements whose PEP 508 markers do not match the resolved Linux Python version and implementation are excluded from the probe. At least one applicable PyTorch requirement must remain.

Unselected extras and groups are not rewritten. They still participate when uv resolves the complete lockfile, so an incompatibility elsewhere in the project can make `uv lock` fail and trigger rollback.

## Source update

After `cpu` succeeds, selected PyTorch packages receive the same Linux-only source:

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

CUDA and nightly names follow `pytorch-cu128` and `pytorch-nightly-cu128`. A same-name index with a different URL is never overwritten. Old official PyTorch index entries are removed only when no source refers to them.

Existing non-Linux source behavior is retained by adding `and sys_platform != 'linux'` to its marker. The Linux source is replaced rather than accumulated, and comments and ordering are preserved where the TOML structure permits. Applying the same verified input again is idempotent.

When a NumPy bridge failure is repaired by `numpy<2`, the requirement is added to every selected PyTorch scope that needs it:

```toml
dependencies = [
    "torch>=2.5,<3",
    "numpy<2; sys_platform == 'linux'",
]
```

An existing applicable NumPy requirement that requires version 2 or newer causes a conflict instead of being silently weakened.

## uv workspaces

The tool uses uv's workspace metadata to resolve the target member, workspace root, member package name, and shared root `uv.lock`. If an older uv cannot inspect a detected workspace, the command asks for an uv update and stops before editing.

For a member target:

- only that member's `pyproject.toml` is edited;
- the root `uv.lock` is updated for the entire workspace;
- sync and final runtime execution use the target package and selected extras/groups;
- the member metadata and shared lockfile are backed up and restored as one transaction;
- a root advisory lock prevents concurrent uv-torch-compass updates.

`plan` and `check` snapshot both files. If an editor or another process changes either file during the command, the result is rejected instead of being reported as current.

Continue with [Recovery and troubleshooting](recovery.md) for transaction details.

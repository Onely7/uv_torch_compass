# CLI usage

[日本語](usage_ja.md) · [Documentation index](README.md) · [Project README](../README.md)

uv-torch-compass has three commands. `plan` and `check` are read-only for the target project; `apply` is the only command that updates it.

## Run the command

For a version published on PyPI, run:

```bash
uvx uv-torch-compass --version
```

To test a local checkout, wheel, or unreleased version, point `uvx` at it with `--from`:

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass --version
```

You can also build a wheel and run that exact artifact:

```bash
cd /path/to/uv_torch_compass
uv build --no-sources
uvx --from dist/uv_torch_compass-<version>-py3-none-any.whl \
  uv-torch-compass --help
```

## `plan`: verify and preview

`plan` locks each candidate for the selected Python minor version and Linux architecture, installs that exact lock in a temporary environment, runs runtime and detected-framework checks, and prints the proposed `pyproject.toml` diff. It does not change the target `pyproject.toml`, `uv.lock`, or synchronized project environment.

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass plan \
  --pyproject /work/my-project/pyproject.toml \
  --torch '>=2.5,<3' \
  --backend auto
```

Candidate downloads, uv's cache, a managed Python installation, and the run log are outside that read-only target-file contract.

## `apply`: update and verify

`apply` repeats candidate verification, creates backups, updates the project, runs a locked sync, and validates the installed runtime with locked/no-sync execution.

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass apply \
  --pyproject /work/my-project/pyproject.toml \
  --backend cuda \
  --cuda-device GPU-01234567-89ab-cdef-0123-456789abcdef
```

On a machine with a visible NVIDIA GPU, both `auto` and `cuda` avoid CPU fallback. `auto` uses CPU only when no NVIDIA GPU is visible. Use a concrete value such as `cu128` only when that exact official index is required and permitted by the compatibility policy.

## `check`: validate the recorded state

`check` reads the requirements and Linux sources already recorded in the project. It checks lockfile freshness, synchronization, package versions, backend identity, and runtime behavior without relocking or syncing.

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass check \
  --pyproject /work/my-project/pyproject.toml \
  --extra vision \
  --group training
```

`check` intentionally has no requirement, Python, backend, or channel override options. A different requested state should be evaluated with `plan` and applied with `apply`.

## Options shared by all commands

| Option | Meaning |
| --- | --- |
| `--pyproject PATH` | Target an existing file named `pyproject.toml`; default: `./pyproject.toml`. |
| `--extra NAME` | Include a project extra; repeat for more than one. |
| `--group NAME` | Include a dependency group; repeat for more than one. |
| `--cuda-device INDEX_OR_UUID` | Select a GPU known to `nvidia-smi`. |
| `--cuda-compatibility strict\|minor` | Require normal driver support, or explicitly allow limited same-major CUDA compatibility; default: `strict`. |
| `--probe-profile standard\|compile` | Run standard library checks, or also test `torch.compile`; default: `standard`. |
| `--framework-probe vllm` | Explicitly request the bounded vLLM integration check; installed vLLM is also detected and checked automatically. |
| `--log-dir PATH` | Store the unique private run log in this directory. |
| `--timeout SECONDS` | Set the positive timeout for installation, project checks, runtime probes, lock, and sync; default: 1800. |
| `--output-format text\|json` | Choose human-readable output or one final JSON object. |
| `--report-file PATH` | Atomically write the final JSON document to a private file. |

## Options for `plan` and `apply`

| Option | Meaning |
| --- | --- |
| `--python REQUEST` | Pass a uv Python request, specifier, download key, implementation, variant, or executable path. |
| `--torch REQUIREMENT` | Add or replace the base `torch` requirement. A shorthand such as `'>=2.5'` is accepted. |
| `--torchvision REQUIREMENT` | Add or replace the base `torchvision` requirement. |
| `--torchaudio REQUIREMENT` | Add or replace the base `torchaudio` requirement. |
| `--backend auto\|cpu\|cuda\|cuNNN` | Select the candidate policy; default: `auto`. |
| `--channel stable\|nightly` | Select the official release channel; default: `stable`. |
| `--link-mode clone\|copy\|hardlink\|symlink` | Select uv's package installation link mode; default: `copy`. |

The three requirement options update base `[project].dependencies` when `apply` succeeds. They are not stored in `[tool.uv-torch-compass]`.

For example, preview minor-version compatibility and the compiler path without changing the target:

```bash
uv-torch-compass plan \
  --cuda-compatibility minor \
  --probe-profile compile
```

Minor mode is not an automatic fallback. If it selects a runtime newer than the driver's normal support level, the successful result includes a warning.

When the candidate lock contains vLLM, the framework probe runs automatically. The option records an explicit request but does not duplicate the check. It checks installed metadata, importability, the native extension, and the selected CPU or CUDA platform. It does not download a model, allocate a model cache, or start workers.

## Understanding candidate failures

Candidate work is split into `lock`, `install`, `runtime`, and `framework` phases. A failure after `lock` retains the resolved PyTorch versions. This means an unavailable wheel for another dependency is reported as that package's blocker, with its dependency path and platform, instead of being described as a missing PyTorch backend.

For a transitive setup, this is enough:

```toml
dependencies = ["vllm==0.19.1"]
```

uv-torch-compass discovers the PyTorch packages in the lock graph and proposes the bare source anchors needed to keep them on the selected official index.

## Help and exit codes

```bash
uv-torch-compass --help
uv-torch-compass plan --help
uv-torch-compass --version
```

These informational commands work outside Linux. A command without `plan`, `apply`, or `check` is invalid.

| Code | Meaning |
| --- | --- |
| `0` | Successful, including success with warnings. |
| `1` | Configuration, command, resolution, validation, or recovery failure. |
| `2` | Command-line syntax error reported by argparse. |

Continue with [Configuration](configuration.md) for persistent defaults or [How selection works](how-it-works.md) for backend behavior.

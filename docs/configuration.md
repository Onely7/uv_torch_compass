# Configuration

[日本語](configuration_ja.md) · [Documentation index](README.md) · [CLI usage](usage.md)

Settings are resolved in this order: an explicit CLI option, a namespaced environment variable, `[tool.uv-torch-compass]`, then the built-in default. A higher layer replaces the lower value; list values are not merged across layers.

Dependency versions remain in the standard project dependency tables. They are never duplicated in the tool table.

## Project settings

Add persistent defaults to the target `pyproject.toml`:

```toml
[tool.uv-torch-compass]
python = "3.12"
backend = "auto"
channel = "stable"
cuda-compatibility = "strict"
probe-profile = "standard"
framework-probes = ["vllm"]
extras = ["vision"]
groups = ["training"]
cuda-device = "0"
link-mode = "copy"
log-dir = ".uv-torch-compass/logs"
timeout = 1800
output-format = "text"
```

All keys are optional. Unknown keys, wrong value types, explicit empty strings, invalid backend or channel values, and non-positive timeouts are rejected before candidate verification or file updates begin.

`report-file` and requirement overrides are deliberately absent from this table. A report is usually run-specific, while package requirements belong in `[project].dependencies`, `[project.optional-dependencies]`, or `[dependency-groups]`.

## Environment variables

| Variable | Setting |
| --- | --- |
| `UV_TORCH_COMPASS_PYPROJECT` | Target `pyproject.toml`. |
| `UV_TORCH_COMPASS_PYTHON` | uv Python request. |
| `UV_TORCH_COMPASS_TORCH` | Base `torch` requirement override. |
| `UV_TORCH_COMPASS_TORCHVISION` | Base `torchvision` requirement override. |
| `UV_TORCH_COMPASS_TORCHAUDIO` | Base `torchaudio` requirement override. |
| `UV_TORCH_COMPASS_BACKEND` | `auto`, `cpu`, `cuda`, or `cuNNN`. |
| `UV_TORCH_COMPASS_CHANNEL` | `stable` or `nightly`. |
| `UV_TORCH_COMPASS_CUDA_COMPATIBILITY` | `strict` or explicitly permitted `minor`. |
| `UV_TORCH_COMPASS_PROBE_PROFILE` | `standard` or `compile`. |
| `UV_TORCH_COMPASS_FRAMEWORK_PROBES` | Comma-separated explicitly requested framework checks; currently `vllm`. Installed vLLM is checked automatically. |
| `UV_TORCH_COMPASS_EXTRAS` | Comma-separated extras. |
| `UV_TORCH_COMPASS_GROUPS` | Comma-separated dependency groups. |
| `UV_TORCH_COMPASS_CUDA_DEVICE` | NVIDIA index or UUID. |
| `UV_TORCH_COMPASS_LINK_MODE` | `clone`, `copy`, `hardlink`, or `symlink`. |
| `UV_TORCH_COMPASS_LOG_DIR` | Log directory relative to the target project, or an absolute path. |
| `UV_TORCH_COMPASS_TIMEOUT` | Positive heavy-command timeout in seconds. |
| `UV_TORCH_COMPASS_OUTPUT_FORMAT` | `text` or `json`. |
| `UV_TORCH_COMPASS_REPORT_FILE` | Path for the final JSON report. |

Empty and duplicate items in comma-separated extra or group lists are removed while preserving the first occurrence:

```bash
export UV_TORCH_COMPASS_EXTRAS='vision,audio,vision,'
```

This resolves to `vision` followed by `audio`.

## Built-in defaults

| Setting | Default |
| --- | --- |
| Target | `./pyproject.toml` |
| Python | `.python-version`, then `[project].requires-python` |
| Backend | `auto` |
| Channel | `stable` |
| CUDA compatibility | `strict` |
| Probe profile | `standard` |
| Explicit framework probes | none; installed vLLM is detected automatically |
| Extras and groups | none |
| CUDA device | visible device with the most free memory, unless the current CUDA selection or `--cuda-device` chooses one |
| Link mode | `copy` |
| Log directory | `.uv-torch-compass/logs` below the target project |
| Project-operation timeout | 1800 seconds |
| Output | `text` |

The configurable timeout covers candidate installation, project checks, runtime probes, lock, and sync. Short metadata commands, such as reading the uv version or available backend names, retain a separate 30-second limit.

`strict` rejects a CUDA runtime newer than the level normally supported by the selected driver. `minor` opts into limited same-major CUDA compatibility and records a warning if used. `standard` runs tensor, NumPy, cuBLAS, cuDNN, architecture, and selected companion-package checks; `compile` adds `torch.compile`.

## Environment passed to uv

Network, proxy, certificate, index authentication, cache, keyring, and offline settings used by uv are preserved for child commands. Values are not copied into diagnostic logs.

Variables that could silently change the selected project, Python, backend, lock policy, or virtual environment are removed and replaced only with validated values. This includes `VIRTUAL_ENV`, `UV_PROJECT`, `UV_PYTHON`, `UV_TORCH_BACKEND`, `UV_LOCKED`, `UV_FROZEN`, `UV_NO_SYNC`, and every `UV_TORCH_COMPASS_*` variable.

An existing `CUDA_VISIBLE_DEVICES` is respected. Without `--cuda-device`, its first visible index or UUID is selected; an empty value or `-1` means that CUDA is hidden. An explicit `--cuda-device` takes priority.

Continue with [Projects and dependency scopes](projects-and-scopes.md) to place package requirements correctly.

# Recovery and troubleshooting

[日本語](recovery_ja.md) · [Documentation index](README.md) · [Project README](../README.md)

`apply` is transactional for project metadata and the lockfile. Environment recovery is attempted separately because an interrupted package installation cannot be restored byte for byte.

## Files involved

| Path | Role |
| --- | --- |
| Target `pyproject.toml` | Receives verified Linux PyTorch sources and necessary dependency updates. |
| Workspace-root `uv.lock` | Receives the complete resolved workspace lock. |
| Target environment, usually `.venv` | Receives the locked selected package and scopes. |
| `.uv-torch-compass/logs/<command>-<time>-<pid>-<suffix>.log` | Private run log; directory is relative to the target project by default. |
| Workspace-root `.uv-torch-compass.lock` | Reusable advisory-lock inode for concurrent apply protection. |
| `pyproject.toml.bak.<time>[.<suffix>]` | Durable pre-apply metadata backup. |
| `uv.lock.bak.<time>[.<suffix>]` | Durable lock backup when a lockfile existed. |

Backups remain after success. A numeric suffix prevents an existing backup from being overwritten.

## Automatic rollback

After locking, `apply` first runs `uv sync --locked --dry-run`. A preflight failure restores the files but does not run environment recovery because the environment has not been changed. If actual synchronization has started, a sync, final probe, timeout, SIGINT, or SIGTERM failure starts this sequence:

1. stop an interrupted child process group before recovery;
2. restore `pyproject.toml` and the previous `uv.lock` with atomic replacement;
3. if the original lockfile existed, run sync against the restored files;
4. if it did not exist, create a temporary recovery lock, attempt sync, then remove the lock again;
5. report file restoration and environment recovery as separate outcomes.

Before replacing a file, the transaction compares its expected content hash. If an editor or another process made an unrecognized change, uv-torch-compass does not overwrite that content; it reports the conflict and retains the backups.

Transaction targets and report destinations must not be symbolic links. After acquiring the workspace lock, `apply` rechecks both `pyproject.toml` and `uv.lock` before writing, so a change made while waiting for the lock is rejected instead of overwritten.

A failed recovery sync means the files may be restored while `.venv` still needs manual repair. The warning says which part failed.

## Manual recovery

Read the run log and inspect backups before copying. From the directory containing the affected file:

```bash
cp pyproject.toml.bak.YYYYMMDD-HHMMSS pyproject.toml
cp uv.lock.bak.YYYYMMDD-HHMMSS uv.lock
uv sync --locked
```

Use the second copy only when a lock backup exists. In a workspace, the `uv.lock` backup is under the workspace root, not necessarily beside the member `pyproject.toml`.

After recovery:

```bash
git status
git diff -- pyproject.toml uv.lock
uv-torch-compass check
```

`check` is useful only after a valid uv-torch-compass source configuration exists. Otherwise, inspect and repair the TOML first.

## Logs

Each command creates one unique mode-`0600` log. `--log-dir` changes its directory; there is no option that reuses or overwrites a chosen log filename.

The log contains phases, redacted subprocess output, package versions, local paths, and GPU information. Common credential patterns are removed, but review the file before sharing it. `--report-file` creates a separate mode-`0600` JSON result.

## Common failures

| Message or symptom | What to check |
| --- | --- |
| `uv was not found in PATH` | Run `uv --version` in the same shell. |
| backend selection is unsupported | Update uv and check that `uv pip install --help` lists `--torch-backend`. |
| Linux-only command failure | Run `plan`, `apply`, or `check` on Linux; only help/version are cross-platform. |
| no selected PyTorch requirement applies | Check selected extras/groups and PEP 508 Python or implementation markers. |
| `nvidia-smi` failure or missing CUDA version | Check NVIDIA driver installation, device visibility, and `CUDA_VISIBLE_DEVICES`. |
| no CUDA backend satisfies strict compatibility | Update the NVIDIA driver, relax the PyTorch version requirement, explicitly select `--backend cpu`, or review whether `--cuda-compatibility minor` is acceptable. |
| configured backend is not allowed by strict compatibility | The recorded source may be newer than the driver's normal support. Run `plan` to preview a supported source; `check` does not modify it. |
| CUDA runtime component does not match the backend | Recreate the synchronized environment from the lockfile and inspect index configuration; the installed CUDA major/minor must match `cuNNN`. |
| native architecture is missing in minor mode | The wheel lacks native machine code for the selected GPU. Choose another backend or update the driver and return to strict mode. |
| compile probe failed | Re-run with `--probe-profile standard` to separate normal CUDA operation from the optional Inductor/Triton path. |
| vLLM framework probe failed | Check the installed `vllm` metadata, native extension, and reported platform. The probe does not load a model or start workers. |
| candidate source policy could not be prepared | Inspect path, Git, URL, workspace, constraint, override, and index sources used by the target project. Candidate verification preserves relevant sources instead of silently replacing them with PyPI. |
| no usable backend | Read each failed candidate's package, requirement, dependency path, and index. Use its suggestions; inspect the private log when the failure kind is `unknown`. |
| `uv lock` failure | Read uv's resolver explanation, including unselected scopes and other workspace members. |
| `uv sync preflight failed` and a package has no wheel | Inspect the package and platform in uv's message. The tool records the current Linux architecture in `tool.uv.required-environments`, allowing uv to choose a compatible version when one exists. |
| environment is not synchronized | Run the intended `apply`, or inspect `uv sync --locked --check` output. |
| final runtime validation failure | The synchronized project did not reproduce the temporary candidate result; rollback should have started. |
| report could not be written after apply | The project update remains applied because report persistence occurs after the transaction. The command exits with `1`; use the private log and retry with a safe regular-file report path. |
| changed while plan/check was running | Retry after the editor or other dependency process has finished. |
| another process is updating the workspace | Wait for the other `apply` to finish; do not delete an active lock to force concurrency. |

To isolate CPU behavior, use `plan --backend cpu`. On a visible NVIDIA GPU, default `auto` and `cuda` both avoid CPU fallback. If `nvidia-smi` exists but cannot be inspected reliably, fix that error before retrying; the tool will not assume the machine is CPU-only.

## When to keep backups

Keep backups until the application has run its own tests and `uv-torch-compass check` succeeds. After that, version control is usually the better long-term history. Remove only explicitly reviewed backup files; never use a broad wildcard from an uncertain directory.

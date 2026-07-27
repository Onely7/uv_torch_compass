# Reports and automation

[日本語](reports_ja.md) · [Documentation index](README.md) · [CLI usage](usage.md)

Text is the default for interactive use. JSON provides one stable final object for scripts and CI.

## Text output

Progress uses the `INSPECT`, `RESOLVE`, `VERIFY`, `APPLY`, and `RESTORE` phase names. Supporting information, warnings, and the final summary identify the selected backend, compatibility decision, skipped newer candidates, official index, changes, backups, and log.

Under `plan`, the summary includes a zero-context unified diff for `pyproject.toml`. Review it before running `apply`.

## JSON output

Use `--output-format json`:

```bash
uv-torch-compass plan --output-format json > result.json
```

stdout contains exactly one final JSON object. Progress and warnings go to stderr, so redirecting stdout produces a parseable document.

The schema version is `3` and includes these top-level fields:

```json
{
  "schema_version": 3,
  "operation": "plan",
  "status": "planned",
  "exit_code": 0,
  "applied": false,
  "target": "/work/app/pyproject.toml",
  "workspace": "/work/app",
  "request": {
    "backend": "auto",
    "cuda_compatibility": "strict",
    "probe_profile": "standard"
  },
  "python": {},
  "candidate_attempts": [
    {
      "backend": "cu129",
      "stage": "policy",
      "status": "skipped",
      "reason": "driver does not provide strict support",
      "compatibility": "unsupported"
    }
  ],
  "selected_backend": "cu124",
  "selected_index": "https://download.pytorch.org/whl/cu124",
  "selected_gpu": {},
  "resolved_packages": {},
  "dependency_roots": [],
  "source_anchors": [],
  "required_environment": "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "validation": {},
  "changes": [],
  "backups": [],
  "warnings": [],
  "errors": [],
  "timing": {}
}
```

The actual document also includes the target package, proposed diff, run log, and non-secret diagnostic metadata. GPU metadata contains the selected device, NVIDIA driver version, and the CUDA maximum printed by `nvidia-smi`.

Each `candidate_attempts` entry has `backend`, `stage`, `status`, `reason`, and `compatibility`. Policy-rejected candidates are recorded as `skipped` before installation. A successful CUDA `validation` includes:

- resolved PyTorch CUDA runtime and runtime-component version;
- required minimum driver and `strict`, `minor`, or `unsupported` classification;
- GPU compute capability and PyTorch compiled architectures;
- CUDA tensor, cuBLAS, cuDNN, native architecture, NumPy, `torchvision`, `torchaudio`, and optional compile results.

Candidate details use redacted summaries rather than raw command data. Consumers should check `schema_version` before depending on fields.

Possible status values are:

| Status | Meaning |
| --- | --- |
| `planned` | Candidate verification passed and a read-only change was prepared. |
| `success` | The change was applied and final validation passed. |
| `success_with_warnings` | Apply and validation passed with non-fatal warnings. |
| `valid` | `check` found the recorded state current and runnable. |
| `failed` | Configuration, command, verification, or recovery failed. |

When JSON output was recognized, configuration and runtime failures use the same schema with exit code `1`. Syntax errors that argparse cannot parse are written to stderr and exit with code `2` before a JSON contract can be established.

## Report files

`--report-file` writes the same final JSON independently of the terminal output format:

```bash
uv-torch-compass apply \
  --output-format text \
  --report-file artifacts/torch-compass.json
```

The destination is resolved from the target project when relative, written through atomic replacement, and set to mode `0600`. A report write failure makes the command fail rather than silently omitting the requested artifact.

## Credential handling

Before text reaches logs, diffs, JSON, or reports, a shared redactor removes common URL user information, token/key/password/secret/signature query values, authorization headers, and secret-like variable assignments. Child-environment values are never logged; only names of removed control variables may be recorded.

Redaction is a safety boundary, not permission to publish logs unchanged. Local paths, package versions, GPU names, and nonstandard credential formats can still be sensitive, so review artifacts before sharing them.

## CI example

```bash
set -o pipefail
uv-torch-compass check --output-format json --report-file compass.json \
  | jq -e '.status == "valid"'
```

Use the process exit code as the primary success signal. Inspect `status`, `warnings`, and `errors` for structured reporting.

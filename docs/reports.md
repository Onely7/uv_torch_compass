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

The schema version is `5` and includes these top-level fields:

```json
{
  "schema_version": 5,
  "operation": "plan",
  "status": "failed",
  "exit_code": 1,
  "applied": false,
  "target": "/work/app/pyproject.toml",
  "workspace": "/work/app",
  "request": {
    "backend": "auto",
    "cuda_compatibility": "strict",
    "probe_profile": "standard",
    "framework_probes": []
  },
  "python": {},
  "candidate_attempts": [
    {
      "backend": "cu121",
      "stage": "install",
      "status": "failed",
      "reason": "The required package build is unavailable from this index.",
      "compatibility": "strict",
      "failure": {
        "kind": "no-compatible-distribution",
        "summary": "The required package build is unavailable from this index.",
        "package": {
          "name": "torch",
          "version": "2.10.0",
          "requirement": "torch==2.10.0"
        },
        "required_by": ["vllm>=0.25.0", "torch==2.10.0"],
        "index": {
          "name": "pytorch-cu121",
          "url": "https://download.pytorch.org/whl/cu121"
        },
        "platform": null,
        "suggestions": ["Select a dependency version compatible with a published PyTorch build."]
      }
    }
  ],
  "resolution_failure": {},
  "candidate_failure_summary": {},
  "selected_backend": "",
  "selected_index": "",
  "selected_gpu": {},
  "resolved_packages": {},
  "dependency_roots": [],
  "source_anchors": [
    {"package": "torch", "scope": "base"}
  ],
  "required_environment": "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "probe_contract": {},
  "framework_validation": [],
  "operation_state": {
    "applied": false,
    "report_written": false
  },
  "environment_policy": {},
  "validation": {},
  "changes": [],
  "backups": [],
  "warnings": [],
  "errors": [],
  "timing": {}
}
```

The actual document also includes the target package, proposed diff, run log, and non-secret diagnostic metadata. GPU metadata contains the selected device, NVIDIA driver version, and the CUDA maximum printed by `nvidia-smi`.

Each `candidate_attempts` entry has `backend`, `stage`, `status`, `reason`, `compatibility`, and an optional `failure`. Policy-rejected candidates are recorded as `skipped` before installation. `candidate_failure_summary` aggregates failed packages, indexes, and de-duplicated suggestions across all candidates. `resolution_failure` contains that terminal aggregate only when the command itself fails; an earlier rejected candidate does not make a successful result look failed. Missing facts are JSON `null`; they are never guessed.

Schema 5 also records the executed `probe_contract`, opt-in `framework_validation`, filtered child-environment policy, scoped source anchors, and final operation state. A successful CUDA `validation` includes:

- resolved PyTorch CUDA runtime and runtime-component version;
- required minimum driver and `strict`, `minor`, or `unsupported` classification;
- GPU compute capability and PyTorch compiled architectures;
- CUDA tensor, cuBLAS, cuDNN, native architecture, NumPy, `torchvision`, `torchaudio`, and optional compile results.

Candidate details use bounded, redacted summaries rather than raw command data. Full redacted uv output remains in the private log. Consumers should check `schema_version` before depending on fields.

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

The destination is resolved from the target project when relative, written through atomic replacement, and set to mode `0600`. The report is persisted before the final stdout summary. If project application succeeds but report persistence fails, the command exits with `1`, writes a structured failure to stderr, and records `applied: true` with `operation_state.report_written: false`. It does not roll back a valid project solely because the separate report artifact failed.

## Credential handling

Before text reaches logs, diffs, JSON, or reports, a shared redactor removes common URL user information, token/key/password/secret/signature query values, authentication and cookie headers, JSON secret fields, secret-bearing command options, and secret-like variable assignments. Child-environment values are never logged; only names of removed control variables may be recorded.

Redaction is a safety boundary, not permission to publish logs unchanged. Local paths, package versions, GPU names, and nonstandard credential formats can still be sensitive, so review artifacts before sharing them.

## CI example

```bash
set -o pipefail
uv-torch-compass check --output-format json --report-file compass.json \
  | jq -e '.status == "valid"'
```

Use the process exit code as the primary success signal. Inspect `status`, `warnings`, and `errors` for structured reporting.

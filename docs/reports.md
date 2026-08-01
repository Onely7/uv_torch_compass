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

The schema version is `8`. Candidate resolution, metadata evidence, version search, and later failures are represented separately:

```json
{
  "schema_version": 8,
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
      "backend": "cu126",
      "stage": "install",
      "status": "failed",
      "reason": "A required wheel is unavailable for the selected platform.",
      "compatibility": "strict",
      "framework": {
        "requested": ["vllm==0.19.1"],
        "resolved": {"vllm": "0.19.1"}
      },
      "framework_compatibility": null,
      "resolution": {
        "status": "resolved",
        "environment": {
          "implementation": "cpython",
          "python_version": "3.12.12",
          "python_minor": "3.12",
          "sys_platform": "linux",
          "platform_machine": "x86_64",
          "required_marker": "sys_platform == 'linux' and platform_machine == 'x86_64'"
        },
        "pytorch": {
          "torch": {
            "version": "2.10.0+cu126",
            "index": "https://download.pytorch.org/whl/cu126"
          }
        },
        "package_count": 182,
        "evidence_source": "workspace-metadata",
        "lock_schema": null
      },
      "phases": {
        "lock": "passed",
        "artifact": "passed",
        "install": "failed",
        "runtime": "not-run",
        "framework": "not-run"
      },
      "failure": {
        "kind": "wheel-unavailable",
        "summary": "xgrammar has no wheel for the selected platform.",
        "package": {
          "name": "xgrammar",
          "version": "0.2.4",
          "requirement": "xgrammar==0.2.4"
        },
        "required_by": ["vllm==0.19.1", "xgrammar==0.2.4"],
        "index": null,
        "platform": "linux-x86_64",
        "dependency_paths": [
          ["uv-torch-compass-candidate==0", "vllm==0.19.1", "xgrammar==0.2.4"]
        ],
        "available_wheel_platforms": ["linux-aarch64", "macos-x86_64"],
        "suggestions": ["Select a dependency version with a wheel for linux-x86_64."]
      }
    }
  ],
  "blocking_summary": {
    "summary": "Compatible PyTorch builds were resolved, but a later candidate phase failed.",
    "pytorch_builds_found": [
      {
        "backend": "cu126",
        "index": "https://download.pytorch.org/whl/cu126",
        "packages": {"torch": "2.10.0+cu126"}
      }
    ],
    "common_blockers": [],
    "suggestions": []
  },
  "failure_category": "dependency-unsatisfiable",
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
  "framework_version_selection": null,
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

Each `candidate_attempts` entry has `backend`, `stage`, `status`, `reason`, `compatibility`, `phases`, `framework`, and optional `resolution`, `framework_compatibility`, and `failure` objects. The `framework` object keeps the requested requirement separate from the version actually resolved for that attempt. The five phases are `lock`, `artifact`, `install`, `runtime`, and `framework`. Once locking succeeds, `resolution` retains the exact execution environment, metadata source, fallback lock schema, PyTorch packages, and framework-related package versions even if a later phase fails.

`blocking_summary` groups the same blocker across candidates while preserving every attempt, including candidates skipped after a conclusive failure. It distinguishes “no candidate resolved” from “PyTorch resolved but a later package or validation failed.” Missing facts are JSON `null`; they are never guessed.

Schema 8 adds `failure_category` and `framework_version_selection`. The category distinguishes `dependency-unsatisfiable`, `framework-cuda-incompatible`, `framework-api-incompatible`, `lock-schema-unsupported`, and `tool-validation-error` without discarding candidate-level detail. A successful bounded vLLM search records the original request, rejected releases, and verified release. The report also records the executed `probe_contract`, automatic or explicit `framework_validation` trigger, filtered child-environment policy, scoped source anchors, final operation state, and reviewed framework-catalog provenance. A successful CUDA `validation` includes:

- resolved PyTorch CUDA runtime and runtime-component version;
- required minimum driver and `strict`, `minor`, or `unsupported` classification;
- GPU compute capability and PyTorch compiled architectures;
- CUDA tensor, cuBLAS, cuDNN, native architecture, NumPy, `torchvision`, `torchaudio`, and optional compile results.

Candidate details use bounded, redacted summaries rather than raw command data. Full redacted uv output remains in the private log. Consumers should check `schema_version` before depending on fields.

A framework failure uses its own structure instead of resolver fields. It can include `binary_requirement` (`required_cuda_variant`, `required_cuda_major`, needed libraries, and `catalog|elf|metadata|runtime` evidence), resolved framework `packages`, a bounded `exception`, `dependency_paths`, and `backend_independent`. For example, a missing `DTensor` is `framework-api-incompatibility`, while `libcudart.so.13` on `cu129` is `framework-cuda-abi`. Public exception data is limited to the type, redacted message, symbol or module names, consumer/provider packages, and at most 12 frames with basename-only paths.

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

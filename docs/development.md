# Development

[日本語](development_ja.md) · [Documentation index](README.md) · [Project README](../README.md)

This guide is for contributors. Python 3.10–3.14 are supported; PyTorch itself is intentionally not a dependency of the tool package.

## Set up

From the repository root:

```bash
uv sync --locked --all-groups
uv run pre-commit install
```

The runtime dependencies are `packaging`, `tomlkit`, and `tomli` on Python 3.10. Tests use controlled uv, NVIDIA, and PyTorch boundaries so the normal suite does not download PyTorch or require a GPU.

## Run checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit run --all-files --show-diff-on-failure
```

Ruff applies PEP 257 `D` rules with the Google pydocstyle convention to production code. Type details belong in annotations; docstrings describe meaning, constraints, side effects, and relevant exceptions. Tests alone omit docstring rules and allow test assertions.

pytest measures branch coverage and fails below 85%. Transaction, process, workflow, rollback, JSON, marker, and source-update failures deserve direct tests when changed.

pre-commit also checks TOML, YAML, trailing whitespace, ShellCheck inputs when present, and committed secrets with gitleaks.

## Build and smoke-test

Build standards-compliant distributions without local source overrides:

```bash
uv build --no-sources
```

Test the wheel artifact, not the editable source environment:

```bash
uvx --refresh --isolated \
  --from dist/uv_torch_compass-0.1.0-py3-none-any.whl \
  uv-torch-compass --version

uvx --refresh --isolated \
  --from dist/uv_torch_compass-0.1.0-py3-none-any.whl \
  uv-torch-compass plan --help
```

Normal CI tests Python 3.10, 3.11, 3.12, 3.13, and 3.14 on Ubuntu. Separate wheel smoke jobs verify isolated installation on 3.10 and 3.14.

## GitHub automation

| Workflow or service | Purpose |
| --- | --- |
| `CI` | Run the version matrix, pre-commit, ty, build, and wheel smoke tests. |
| `CodeQL` | Analyze Python and Actions with `security-extended` queries and minimal permissions. |
| `Real PyTorch CPU smoke test` | Manually install and validate the real CPU build in a temporary target project. |
| `Build publication artifacts` | Manually build, smoke-test, and upload preparation artifacts without publishing. |
| Dependabot | Check uv, GitHub Actions, and pre-commit dependencies weekly in separate groups. |

Action references use full commit SHAs. Dependabot's `github-actions` ecosystem updates those pins. CUDA runtime validation still requires a Linux host or runner with an NVIDIA GPU.

## Publication-preparation artifacts

The manual artifact workflow produces:

- wheel and source distribution;
- CycloneDX 1.5 SBOM exported from the locked runtime dependencies;
- `SHA256SUMS` for generated files;
- a JSON provenance manifest with commit, Python, uv, and smoke-test results;
- the plain-text smoke-test output.

It uploads them only as a GitHub Actions artifact with finite retention. The workflow does not create a tag or GitHub Release, upload to PyPI, request `id-token`, configure Trusted Publishing, or use release credentials.

Before any future public release, choose and add authorship, license, and repository metadata, confirm the package name, configure publishing outside this workflow, and repeat the wheel checks. Only then should documentation introduce bare `uvx uv-torch-compass` as an available command.

## Security boundaries

- External executables are resolved to absolute paths and launched as argument vectors without a shell.
- Timeout and termination stop the child process group before rollback.
- Control environment variables are removed; credential-bearing values are not logged.
- Temporary environments use secure temporary directories and are removed after verification.
- Project writes use content hashes, backups, atomic replacement, file/directory fsync, and a Linux advisory lock.
- CodeQL findings should be fixed at their cause. A suppression is acceptable only for an unavoidable line with a specific reason.

## References

- [uv: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv: Tools](https://docs.astral.sh/uv/concepts/tools/)
- [uv: Building distributions](https://docs.astral.sh/uv/concepts/projects/build/)
- [GitHub: CodeQL workflow configuration](https://docs.github.com/en/code-security/reference/code-scanning/workflow-configuration-options)
- [GitHub: Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)

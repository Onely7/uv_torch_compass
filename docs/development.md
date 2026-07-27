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
version="$(uv version --short)"
wheel="dist/uv_torch_compass-${version}-py3-none-any.whl"

uvx --refresh --isolated \
  --from "$wheel" \
  uv-torch-compass --version

uvx --refresh --isolated \
  --from "$wheel" \
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
| `Publish Python package` | Publish a manual build to TestPyPI or a published GitHub Release to PyPI through Trusted Publishing. |
| `Release Please` | Maintain a Release PR, then create a version tag and GitHub Release when a human merges it. |
| `CUDA compatibility catalog audit` | Check the NVIDIA source weekly and fail when the bundled catalog has not received a recent human review. |
| Dependabot | Check uv, GitHub Actions, and pre-commit dependencies weekly in separate groups. |

Action references use full commit SHAs. Dependabot's `github-actions` ecosystem updates those pins. CUDA runtime validation still requires a Linux host or runner with an NVIDIA GPU.

## Release management

Release Please reads Conventional Commit messages on `main` and keeps one Release PR current. The Python release strategy updates `CHANGELOG.md`, `pyproject.toml`, `src/uv_torch_compass/__init__.py`, `uv.lock`, and the release manifest together. Before 1.0, the configured policy raises the minor version for breaking changes.

Use these commit prefixes for changes that should determine a version:

| Commit | Version effect before 1.0 |
| --- | --- |
| `fix: ...` | Patch, such as `0.1.0` to `0.1.1`. |
| `feat: ...` | Minor, such as `0.1.0` to `0.2.0`. |
| `feat!: ...` or a `BREAKING CHANGE:` footer | Minor while the project remains below 1.0. |

Other recognized commits can appear in the changelog but do not necessarily request a new release. Prefer squash-merging ordinary pull requests with a Conventional Commit title so `main` contains one clear release entry per change.

Merging the Release PR creates the `vX.Y.Z` tag and a published GitHub Release. The separate publishing workflow then builds that tag and waits for approval in the `pypi` Environment before uploading through Trusted Publishing.

The workflow works immediately with the built-in `GITHUB_TOKEN`. GitHub does not start another workflow for a tag or release created with that token. Before adding the separate PyPI workflow, create a repository secret named `RELEASE_PLEASE_TOKEN` containing a repository-scoped GitHub App token or fine-grained personal access token with Contents, Pull requests, and Issues write access. The existing workflow automatically prefers that secret when present.

## Publication-preparation artifacts

The manual artifact workflow produces:

- wheel and source distribution;
- CycloneDX 1.5 SBOM exported from the locked runtime dependencies;
- `SHA256SUMS` for generated files;
- a JSON provenance manifest with commit, Python, uv, and smoke-test results;
- the plain-text smoke-test output.

It uploads them only as a GitHub Actions artifact with finite retention. This manual workflow does not create a tag or GitHub Release, upload to PyPI, request `id-token`, or use release credentials. Tag and GitHub Release creation belong to Release Please; package upload belongs to the separate publishing workflow.

For each release, confirm the TestPyPI and PyPI Trusted Publisher settings, test the candidate on TestPyPI when appropriate, and approve the protected `pypi` Environment only after reviewing the tag and artifacts.

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
- [Release Please](https://github.com/googleapis/release-please)

# Publishing

[日本語](publishing_ja.md) · [Documentation index](README.md) · [Project README](../README.md)

The `Publish Python package` workflow uses PyPI Trusted Publishing. GitHub Actions exchanges an OpenID Connect identity for a short-lived upload credential, so this repository does not store PyPI API tokens.

## Publication paths

The two indexes use deliberately different triggers:

- A manual workflow run builds the selected Git ref and publishes it only to TestPyPI.
- A published GitHub Release builds its tag and requests publication to PyPI.

Both paths build a wheel and source distribution, run strict metadata checks, smoke-test the wheel, and pass only those two distribution files to the publishing job. A Release tag must exactly equal `v` followed by the package version.

The PyPI job waits for approval in the `pypi` GitHub Environment. Approving a job is the final production-publication decision; published files cannot be replaced with different contents under the same version.

## GitHub Environments

Create `testpypi` and `pypi` under **Settings → Environments**.

For `testpypi`, do not require a reviewer and allow only the `main` branch. For `pypi`, require `Onely7` as a reviewer, leave **Prevent self-review** disabled while the repository has only one maintainer, and allow only tags matching `v*`. These policies match the workflow triggers without granting unused deployment paths.

No PyPI password or API token belongs in either environment or in repository secrets. `RELEASE_PLEASE_TOKEN` is separate: Release Please uses it to create release events that can start this publishing workflow.

## Trusted Publishers

Configure a pending GitHub publisher separately on [PyPI](https://pypi.org/manage/account/publishing/) and [TestPyPI](https://test.pypi.org/manage/account/publishing/).

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| PyPI project name | `uv-torch-compass` | `uv-torch-compass` |
| Owner | `Onely7` | `Onely7` |
| Repository | `uv_torch_compass` | `uv_torch_compass` |
| Workflow | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

The workflow filename and Environment name must match exactly. PyPI and TestPyPI use separate accounts and separate Trusted Publisher registrations.

## TestPyPI

After the workflow and Environment configuration are present on `main`, open **Actions → Publish Python package → Run workflow**. Select `main` and run it once. The build job must pass before the TestPyPI upload receives an OIDC credential.

TestPyPI does not allow replacing an existing file or version. If the version has already been uploaded, publish a later package version rather than enabling `skip-existing`, which could hide that the tested artifact differs from the intended one.

## PyPI

Merging a Release Please PR creates a version tag and a published GitHub Release. That event starts the publication workflow against the release tag. Review the build logs and artifact, then approve the pending `pypi` deployment in GitHub.

The `v0.1.0` GitHub Release predates this workflow and will not trigger it retroactively. Automatic production publication starts with the next published GitHub Release.

## References

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PyPI Trusted Publishing security model](https://docs.pypi.org/trusted-publishers/security-model/)
- [Python Packaging User Guide: publishing from GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub deployment Environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)

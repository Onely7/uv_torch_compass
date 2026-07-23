# 公開

[English](publishing.md) · [ドキュメント一覧](README_ja.md) · [プロジェクト README](../README_ja.md)

`Publish Python package` workflow は、PyPI Trusted Publishing を使います。GitHub Actions の OpenID Connect（OIDC）identity を有効期間の短い upload credential と交換するため、PyPI API token をこの repository へ保存しません。

## 公開経路

二つの index では、意図的に異なる起動条件を使います。

- workflow を手動実行すると、選択した Git ref を build し、TestPyPI だけへ公開する
- GitHub Release が公開されると、その tag を build し、PyPI への公開を要求する

どちらも wheel と source distribution を build し、厳密な metadata 検査と wheel の smoke test を行います。公開 job へ渡すのは、この二つの配布ファイルだけです。Release tag は、package version の先頭へ `v` を付けた値と完全に一致する必要があります。

PyPI job は、GitHub の `pypi` Environment で承認を待ちます。この承認が本番公開の最終判断です。同じ version で公開済みのファイルを、異なる内容へ置き換えることはできません。

## GitHub Environments

**Settings → Environments** で `testpypi` と `pypi` を作成します。

`testpypi` では reviewer を要求せず、`main` branch だけを許可します。`pypi` では `Onely7` を reviewer に指定し、`v*` に一致する tag だけを許可します。maintainer が一人の間は、**Prevent self-review** を無効にしてください。この制限は workflow の起動条件に対応し、使わない公開経路を許可しません。

どちらの Environment にも、PyPI password や API token は登録しません。repository secret の `RELEASE_PLEASE_TOKEN` は別の目的で使います。Release Please が、この公開 workflow を起動できる release event を作るための token です。

## Trusted Publishers

[PyPI](https://pypi.org/manage/account/publishing/) と [TestPyPI](https://test.pypi.org/manage/account/publishing/) で、pending GitHub publisher を個別に設定します。

| 項目 | PyPI | TestPyPI |
| --- | --- | --- |
| PyPI project name | `uv-torch-compass` | `uv-torch-compass` |
| Owner | `Onely7` | `Onely7` |
| Repository | `uv_torch_compass` | `uv_torch_compass` |
| Workflow | `publish.yml` | `publish.yml` |
| Environment | `pypi` | `testpypi` |

workflow filename と Environment name は、文字列が完全に一致する必要があります。PyPI と TestPyPI では別々の account と Trusted Publisher 設定を使います。

## TestPyPI

workflow と Environment の設定が `main` に入ったら、**Actions → Publish Python package → Run workflow** を開きます。`main` を選んで一度実行してください。build job が成功してから、TestPyPI upload が OIDC credential を取得します。

TestPyPI でも、既存のファイルや version は置き換えられません。すでに同じ version を公開している場合は、`skip-existing` で差異を隠さず、次の package version を公開してください。

## PyPI

Release Please PR を merge すると、version tag と公開済み GitHub Release が作られます。この event により、release tag を対象とした公開 workflow が始まります。build log と artifact を確認し、GitHub で待機中の `pypi` deployment を承認してください。

`v0.1.0` GitHub Release は、この workflow より先に作られたため、あとから自動実行されません。本番への自動公開は、次に作成する GitHub Release から始まります。

## 参考資料

- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PyPI Trusted Publishing の security model](https://docs.pypi.org/trusted-publishers/security-model/)
- [Python Packaging User Guide: GitHub Actions からの公開](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [GitHub deployment Environments](https://docs.github.com/en/actions/reference/workflows-and-actions/deployments-and-environments)

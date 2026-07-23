# 開発

[English](development.md) · [ドキュメント一覧](README_ja.md) · [プロジェクト README](../README_ja.md)

この文書は開発者向けです。Python 3.10–3.14 に対応し、PyTorch 自体は意図的に tool package の依存へ含めていません。

## 開発環境

repository root で次を実行します。

```bash
uv sync --locked --all-groups
uv run pre-commit install
```

runtime 依存は `packaging`、`tomlkit`、Python 3.10 での `tomli` です。通常テストは制御可能な uv、NVIDIA、PyTorch の境界を使うため、PyTorch の download や GPU を必要としません。

## 検査

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run pytest
uv run pre-commit run --all-files --show-diff-on-failure
```

production code には PEP 257 の `D` rule と Google pydocstyle convention を適用します。型情報は annotation に任せ、docstring には意味、制約、副作用、利用者が処理すべき例外を記述します。テストだけは docstring rule を除外し、test assertion を許可します。

pytest は branch coverage を計測し、85% 未満を失敗にします。transaction、process、workflow、rollback、JSON、marker、source 更新の失敗分岐を変更した場合は、直接テストを追加してください。

pre-commit は TOML、YAML、末尾空白、対象がある場合の ShellCheck、gitleaks による commit 済み秘密情報も確認します。

## build と smoke test

ローカル source override を使わず、標準の配布 package を作ります。

```bash
uv build --no-sources
```

editable source 環境ではなく wheel artifact を検証します。

```bash
uvx --refresh --isolated \
  --from dist/uv_torch_compass-0.1.0-py3-none-any.whl \
  uv-torch-compass --version

uvx --refresh --isolated \
  --from dist/uv_torch_compass-0.1.0-py3-none-any.whl \
  uv-torch-compass plan --help
```

通常 CI は Ubuntu 上の Python 3.10、3.11、3.12、3.13、3.14 でテストします。別の wheel smoke job が 3.10 と 3.14 で隔離 install を確認します。

## GitHub 自動化

| workflow または service | 目的 |
| --- | --- |
| `CI` | version matrix、pre-commit、ty、build、wheel smoke test を行う |
| `CodeQL` | 最小権限で Python と Actions を `security-extended` 解析する |
| `Real PyTorch CPU smoke test` | 一時対象 project へ実 CPU build を install し、手動で検証する |
| `Build publication artifacts` | 公開せず、build・smoke test・準備 artifact の upload を手動実行する |
| `Release Please` | Release PR を更新し、人が merge したときに version tag と GitHub Release を作る |
| Dependabot | uv、GitHub Actions、pre-commit の依存を別 group で週次確認する |

Action の参照は完全な commit SHA で固定します。Dependabot の `github-actions` ecosystem が固定値を更新します。CUDA runtime の確認には、NVIDIA GPU を持つ Linux host または runner が別途必要です。

## release 管理

Release Please は `main` の Conventional Commit を読み、一つの Release PR を最新の状態に保ちます。Python 用の release 方針により、`CHANGELOG.md`、`pyproject.toml`、`src/uv_torch_compass/__init__.py`、`uv.lock`、release manifest を同じ PR で更新します。最初に提案する release は `v0.1.0` です。

version を決める変更には、次の commit prefix を使います。

| commit | 1.0 より前の version への影響 |
| --- | --- |
| `fix: ...` | patch。例: `0.1.0` から `0.1.1` |
| `feat: ...` | minor。例: `0.1.0` から `0.2.0` |
| `feat!: ...` または `BREAKING CHANGE:` footer | 1.0 未満では minor |

ほかの認識可能な commit も changelog へ入る場合がありますが、必ず新しい release を要求するわけではありません。通常の PR は Conventional Commit 形式の title で squash merge し、`main` 上の一変更を明確な一 commit にする運用を推奨します。

Release PR を merge すると、`vX.Y.Z` tag と公開済み GitHub Release を作ります。PyPI への公開は行いません。将来の PyPI workflow は分離し、GitHub Release の公開時だけ実行する構成にしてください。

workflow は組み込みの `GITHUB_TOKEN` だけでも動作します。ただし、この token が作った tag や release から別の workflow は起動しません。PyPI workflow を追加する前に、Contents・Pull requests・Issues の書き込み権限を持つ、対象 repository 限定の GitHub App token または fine-grained personal access token を `RELEASE_PLEASE_TOKEN` という repository secret へ登録してください。登録後は既存 workflow がその secret を優先します。

## 公開準備 artifact

手動 artifact workflow は次を作ります。

- wheel と source distribution
- lock 済み runtime 依存から export した CycloneDX 1.5 SBOM
- 生成ファイルの `SHA256SUMS`
- commit、Python、uv、smoke test 結果を持つ JSON provenance manifest
- text の smoke test 出力

有限の保持期間を持つ GitHub Actions artifact としてだけ upload します。この手動 workflow は tag、GitHub Release、PyPI upload、`id-token` 要求、Trusted Publishing、公開用 credentials を追加しません。tag と GitHub Release の作成は Release Please が担当します。

将来の PyPI 公開前には、残っている著者・repository の package metadata を追加し、package 名を確認し、この workflow とは別に publishing を設定して wheel 検証を繰り返します。それらが終わるまで、`uvx uv-torch-compass` を利用可能な command として案内しません。

## security boundary

- 外部実行ファイルは絶対パスへ解決し、shell を介さない引数 list で起動する
- timeout と終了 signal では、rollback 前に子 process group を停止する
- 制御用環境変数を除外し、認証情報を含む値を log へ出さない
- 安全な一時 directory に候補環境を作り、検証後に削除する
- content hash、backup、atomic replace、file/directory fsync、Linux advisory lock で project を更新する
- CodeQL の指摘は原因から修正する。抑制は避けられない行に具体的な理由を付ける場合だけ使う

## 参考資料

- [uv: Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [uv: Tools](https://docs.astral.sh/uv/concepts/tools/)
- [uv: Building distributions](https://docs.astral.sh/uv/concepts/projects/build/)
- [GitHub: CodeQL workflow configuration](https://docs.github.com/en/code-security/reference/code-scanning/workflow-configuration-options)
- [GitHub: Dependabot options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference)
- [Release Please](https://github.com/googleapis/release-please)

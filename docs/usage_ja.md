# CLI の使い方

[English](usage.md) · [ドキュメント一覧](README_ja.md) · [プロジェクト README](../README_ja.md)

uv-torch-compass には三つの command があります。対象プロジェクトを変更しないのは `plan` と `check`、更新するのは `apply` だけです。

## コマンドを実行する

PyPI に公開された版は、次のように実行できます。

```bash
uvx uv-torch-compass --version
```

ローカルのリポジトリ、wheel、未公開版を試す場合は、`uvx` の `--from` で取得元を指定します。

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass --version
```

wheel を作り、その artifact を直接検証することもできます。

```bash
cd /path/to/uv_torch_compass
uv build --no-sources
uvx --from dist/uv_torch_compass-0.1.0-py3-none-any.whl \
  uv-torch-compass --help
```

## `plan`: 検証して変更案を見る

`plan` は一時環境を作り、PyTorch の候補をインストールし、実行 probe を行ってから `pyproject.toml` の変更案を表示します。対象の `pyproject.toml`、`uv.lock`、同期済みプロジェクト環境は変更しません。

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass plan \
  --pyproject /work/my-project/pyproject.toml \
  --torch '>=2.5,<3' \
  --backend auto
```

候補の download、uv の cache、uv が管理する Python の追加、実行ログは、この「対象ファイルを変更しない」という契約には含まれません。

## `apply`: 反映して再検証する

`apply` は候補の検証後にバックアップを作り、プロジェクトを更新して locked sync を行います。最後に locked/no-sync でインストール済み環境を検証します。

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass apply \
  --pyproject /work/my-project/pyproject.toml \
  --backend cuda \
  --cuda-device GPU-01234567-89ab-cdef-0123-456789abcdef
```

CPU への fallback を許可しない場合は `--backend cuda` を使います。特定の公式 index だけが必要な場合は、`cu128` のような具体値を指定します。

## `check`: 記録済みの状態を確認する

`check` はプロジェクトに記録済みの依存条件と Linux 用 source を読みます。relock や sync を行わず、lockfile の鮮度、同期状態、package version、backend、実行結果を確認します。

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass check \
  --pyproject /work/my-project/pyproject.toml \
  --extra vision \
  --group training
```

`check` では依存条件、Python、backend、channel を上書きできません。別の条件は `plan` で確認し、`apply` で反映してください。

## すべての command で使える option

| option | 意味 |
| --- | --- |
| `--pyproject PATH` | `pyproject.toml` という名前の既存ファイルを選ぶ。初期値は `./pyproject.toml` |
| `--extra NAME` | project extra を追加する。複数指定では繰り返す |
| `--group NAME` | 依存グループを追加する。複数指定では繰り返す |
| `--cuda-device INDEX_OR_UUID` | `nvidia-smi` が認識する GPU を選ぶ |
| `--log-dir PATH` | 重複しない非公開の実行ログを保存する |
| `--timeout SECONDS` | install、project 検査、runtime probe、lock、sync の正の timeout を指定する。初期値は 1800 秒 |
| `--output-format text\|json` | 人向け出力、または最後の JSON object 一つを選ぶ |
| `--report-file PATH` | 最終 JSON を、所有者だけが読めるファイルへ書きかけを残さず保存する |

## `plan` と `apply` で使える option

| option | 意味 |
| --- | --- |
| `--python REQUEST` | uv の Python request、specifier、download key、implementation、variant、実行ファイルのパスを渡す |
| `--torch REQUIREMENT` | 基本依存の `torch` を追加・置換する。`'>=2.5'` の省略形も使える |
| `--torchvision REQUIREMENT` | 基本依存の `torchvision` を追加・置換する |
| `--torchaudio REQUIREMENT` | 基本依存の `torchaudio` を追加・置換する |
| `--backend auto\|cpu\|cuda\|cuNNN` | 候補の方針を選ぶ。初期値は `auto` |
| `--channel stable\|nightly` | 公式 release channel を選ぶ。初期値は `stable` |
| `--link-mode clone\|copy\|hardlink\|symlink` | uv の package 配置方法を選ぶ。初期値は `copy` |

三つの依存 option は、`apply` が成功すると基本 `[project].dependencies` を更新します。`[tool.uv-torch-compass]` には保存しません。

## help と終了コード

```bash
uv-torch-compass --help
uv-torch-compass plan --help
uv-torch-compass --version
```

これらの情報表示は Linux 以外でも動作します。`plan`、`apply`、`check` のない呼び出しは構文エラーです。

| code | 意味 |
| --- | --- |
| `0` | 警告付きの成功を含む正常終了 |
| `1` | 設定、外部 command、依存解決、検証、復旧の失敗 |
| `2` | argparse が報告する command line の構文エラー |

永続的な初期値は[設定](configuration_ja.md)、backend の挙動は[選択の仕組み](how-it-works_ja.md)へ進んでください。

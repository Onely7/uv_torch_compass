# uv-torch-compass

[English](https://github.com/Onely7/uv_torch_compass/blob/main/README.md)

**`uv-torch-compass` を使うことで、指定した PyTorch のバージョン条件と現在の Linux 実行環境の両方に対して公式 PyTorch index を検証し、最初に検証を通過した取得先を対象プロジェクトの `pyproject.toml` へ安全に設定できます。**

index はパッケージの取得先です。PyTorch は CPU 版と NVIDIA CUDA 版を別々の公式 index で配布しています。このツールはインストールできるかだけでなく、初期設定では NVIDIA driver が通常サポートする範囲より新しい CUDA build を除外します。そのうえで PyTorch、NumPy、選択した GPU、cuBLAS、cuDNN、必要に応じて `torchvision` や `torchaudio` を実行してから取得先を反映します。

## クイックスタート

Linux、新しい [uv](https://docs.astral.sh/uv/)、インターネット接続、Python 3.10–3.14 が必要です。対象 `pyproject.toml` の基本依存、または選択する extra・依存グループに `torch`、`torchvision`、`torchaudio` のいずれかを含めます。

最小構成は次のとおりです。

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5"]
```

対象プロジェクトで、PyPI に公開された版を使い、まず候補を検証して変更案を確認します。

```bash
uvx uv-torch-compass plan
```

ローカルのリポジトリや wheel を試す場合は、取得元を明示します。

```bash
uvx --from /path/to/uv_torch_compass uv-torch-compass plan
```

変更案に問題がなければ反映します。`pyproject.toml` を更新し、workspace を lock し、対象環境を同期したあと、もう一度実行検証します。

```bash
uvx uv-torch-compass apply
```

あとから、記録済みの取得先、lockfile、同期状態、インストール済みの実行環境を変更せず確認できます。

```bash
uvx uv-torch-compass check
```

ローカル版を使う場合は、`/path/to/uv_torch_compass` をこのリポジトリのパスへ置き換え、`apply` と `check` にも同じ `--from` を付けてください。別のディレクトリから対象を指定する場合は、`--pyproject /path/to/project/pyproject.toml` を追加します。

## 検証対象の選び方

初期値の `--backend auto --cuda-compatibility strict` は、次のように動作します。

- NVIDIA GPU が見える場合、driver が通常サポートする具体的な CUDA build だけを新しい順に試します。すべて失敗しても、CPU 版へ暗黙に切り替えません。
- NVIDIA GPU が見えない場合、公式 CPU 版を試します。

最初に検証を通過した候補で終了し、すべてをベンチマークして最速の候補を選ぶわけではありません。たとえば、`nvidia-smi` に `CUDA Version: 12.4` と表示される環境では、初期設定で `cu129` を採用しません。

検証対象は次のように限定できます。

```bash
uv-torch-compass plan --backend cpu
uv-torch-compass plan --backend cuda
uv-torch-compass plan --backend cu128
uv-torch-compass plan --channel nightly
uv-torch-compass plan --probe-profile compile
```

channel の初期値は安定版を示す `stable` です。開発版の `nightly` は、明示した場合だけ使います。CUDA minor-version compatibility も `--cuda-compatibility minor` で明示した場合だけ使い、同じ major 系列の新しい CUDA runtime を利用できますが、成功時にも警告が残ります。順番と検証内容は[バックエンドと実行環境の選択](https://github.com/Onely7/uv_torch_compass/blob/main/docs/how-it-works_ja.md)を参照してください。

## 安全性の概要

- `plan` は一時環境へ候補をインストールして検証しますが、対象の `pyproject.toml`、`uv.lock`、プロジェクト環境を変更しません。
- `apply` は日時付きバックアップを作り、workspace member の `pyproject.toml` と root の共有 `uv.lock` を一つの更新単位として扱います。
- 同じディレクトリの一時ファイルを使い、書きかけを見せずに一括置換します。workspace lock により、二つの `apply` が同時に更新することも防ぎます。
- lock、sync、最終検証、timeout、SIGINT、SIGTERM の失敗時は、ファイルを復元し、環境の復旧も試みます。
- ログと JSON report は一般的な認証情報をマスクし、所有者だけが読める権限で作ります。

`plan` と `apply` のあとには `git diff` を確認してください。成功後もバックアップは残ります。名前と復旧上の制約は[復旧とトラブル対応](https://github.com/Onely7/uv_torch_compass/blob/main/docs/recovery_ja.md)で説明します。

## 対応範囲

- `plan`、`apply`、`check` は Linux 専用です。`--help` と `--version` はほかの OS でも動作します。
- CPU と NVIDIA CUDA に対応します。AMD ROCm と Intel XPU は非対応です。
- PyTorch の公式 stable・nightly index に対応します。stable の失敗から nightly へ自動移行しません。
- 基本依存、選択した extra・依存グループ、uv workspace、`torchvision`、`torchaudio` に対応します。
- CUDA の成功には、GPU tensor、cuBLAS、cuDNN、architecture、選択した関連 package の検証が必要です。`--probe-profile compile` では `torch.compile` も確認します。
- NVIDIA driver のインストールや更新は行いません。driver 更新後は `plan` と `apply` を再実行してください。
- 終了コードは、成功が `0`、設定・実行の失敗が `1`、コマンド構文の誤りが `2` です。

## ドキュメント

| 目的 | 文書 |
| --- | --- |
| コマンドとオプションを確認する | [CLI の使い方](https://github.com/Onely7/uv_torch_compass/blob/main/docs/usage_ja.md) |
| プロジェクト設定と環境変数を指定する | [設定](https://github.com/Onely7/uv_torch_compass/blob/main/docs/configuration_ja.md) |
| 処理の流れ、backend、channel、GPU、Python、実行検証を理解する | [選択の仕組み](https://github.com/Onely7/uv_torch_compass/blob/main/docs/how-it-works_ja.md) |
| extra、依存グループ、workspace を使う | [プロジェクトと依存範囲](https://github.com/Onely7/uv_torch_compass/blob/main/docs/projects-and-scopes_ja.md) |
| text・JSON の結果を利用する | [report と自動化](https://github.com/Onely7/uv_torch_compass/blob/main/docs/reports_ja.md) |
| ファイルを復旧し、失敗を調べる | [復旧とトラブル対応](https://github.com/Onely7/uv_torch_compass/blob/main/docs/recovery_ja.md) |
| テスト、build、artifact 準備を行う | [開発](https://github.com/Onely7/uv_torch_compass/blob/main/docs/development_ja.md) |

文書全体は[ドキュメント一覧](https://github.com/Onely7/uv_torch_compass/blob/main/docs/README_ja.md)から確認できます。

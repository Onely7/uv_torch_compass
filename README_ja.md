# uv-torch-compass

[English](https://github.com/Onely7/uv_torch_compass/blob/main/README.md)

**`uv-torch-compass` を使うことで、指定した PyTorch のバージョン条件と現在の Linux 実行環境の両方に対して公式 PyTorch index を検証し、最初に検証を通過した取得先を対象プロジェクトの `pyproject.toml` へ安全に設定できます。**

index はパッケージの取得先です。PyTorch は CPU 版と NVIDIA CUDA 版を別々の公式 index で配布しています。このツールはインストールできるかだけでなく、初期設定では NVIDIA driver が通常サポートする範囲より新しい CUDA build を除外します。そのうえで PyTorch、NumPy、選択した GPU、cuBLAS、cuDNN、必要に応じて `torchvision` や `torchaudio` を実行してから取得先を反映します。

## クイックスタート

Linux、新しい [uv](https://docs.astral.sh/uv/)、インターネット接続、Python 3.10–3.14 が必要です。PyTorch は直接指定しても、`vllm` のような別の選択済み package から導入しても構いません。

最小構成は次のとおりです。

```toml
[project]
name = "my-project"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5"]
```

PyTorch に依存する framework を使う場合は、実際に使う依存だけを記述できます。

```toml
dependencies = ["vllm==0.19.1"]
```

候補環境では、選択した Python の minor version、Linux、CPU architecture に対象を絞り、最初に依存グラフ全体を lock します。`vllm` が特定の `torch`、`torchvision`、`torchaudio` version を要求する場合、その制約を backend 選択にも反映します。推移的に導入された PyTorch package を同じ公式 index へ向け、取得先がそろうまで再度 lock します。元の framework 依存は維持されます。

`vllm==0.6.0` のように確認済みの version を完全指定した場合は、lock より前に CUDA 候補を絞ります。`vllm>=0.19.1` のような範囲では、解決された vLLM が backend と合わなければ、同じ backend で一つ前の version を上限付きで検証できます。`apply` が代替 version を採用した場合も、元の範囲指定は残し、ツール管理の uv constraint で検証済みの結果を再現します。

候補解決では、関連する uv の constraint、override、private index、選択した path・Git・URL・workspace source も引き継ぎます。検証中の公式 index へ切り替えるのは PyTorch package だけです。

lock に成功したあと別の package をインストールできなかった場合も、解決済みの PyTorch version を保持し、backend がないと誤って報告しません。たとえば、「`torch==2.10.0+cu126` は解決済み」と「`vllm` が必要とする `xgrammar` の Linux x86_64 用 wheel がない」を区別できます。uv の完全な出力は、認証情報を除去してprivate logだけに残します。

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
uv-torch-compass plan --framework-probe vllm
```

channel の初期値は安定版を示す `stable` です。開発版の `nightly` は、明示した場合だけ使います。CUDA minor-version compatibility も `--cuda-compatibility minor` で明示した場合だけ使い、同じ major 系列の新しい CUDA runtime を利用できますが、成功時にも警告が残ります。順番と検証内容は[バックエンドと実行環境の選択](https://github.com/Onely7/uv_torch_compass/blob/main/docs/how-it-works_ja.md)を参照してください。

解決結果に `vllm` が含まれる場合、範囲を限定した vLLM 検証を自動実行します。`--framework-probe vllm` は明示的に要求したい場合にも使えます。model を取得したり worker を起動したりせず、metadata、import、native extension、選択された実行 platform を確認します。

利用中の uv が package の選択 install に対応している場合、依存一式を入れる前に lock 済み vLLM wheel だけを展開します。wheel の import や実行は行いません。ELF metadata から `libcudart.so.13` などの必要 library を読み取り、PyTorch backend と照合します。公式 wheel について確認済みの情報も補助的に使うため、「vLLM は CUDA 13 を必要とするが、`cu129` は CUDA 12.9」のような不一致を、大容量 install の繰り返しより前に報告できます。`DTensor` がないといった Python API の失敗は、CUDA ABI の失敗とは分けて表示します。

## 安全性の概要

- `plan` は一時環境へ候補をインストールして検証しますが、対象の `pyproject.toml`、`uv.lock`、プロジェクト環境を変更しません。
- `apply` は日時付きバックアップを作り、workspace member の `pyproject.toml` と root の共有 `uv.lock` を一つの更新単位として扱います。
- 候補検証を lock、lock 済み install、runtime 検証、framework 検証に分けます。一時 lock の対象を選択した Python minor、Linux、CPU architecture に絞るため、利用できる wheel を持つ version があれば uv がそこまで戻って選び直せます。
- 候補の依存グラフは、利用できる場合は uv の JSON workspace metadata から読みます。互換用の lockfile parser も残し、任意項目の wheel `size` がなくても受理します。
- project 環境を変更する前に、依存グラフ全体を lock し、locked sync の dry run を行います。現在の Linux architecture も uv の必須環境として記録するため、利用できない wheel はインストール開始前に検出されます。
- 同じディレクトリの一時ファイルを使い、書きかけを見せずに一括置換します。workspace lock により、二つの `apply` が同時に更新することも防ぎます。
- lock、sync、最終検証、timeout、SIGINT、SIGTERM の失敗時は、ファイルを復元し、環境の復旧も試みます。
- ログと JSON report は一般的な認証情報をマスクし、所有者だけが読める権限で作ります。`apply` の成功後に指定した report だけを書き込めなかった場合、project は適用済みのまま、`applied: true` を報告して終了コード `1` になります。
- uv が 0.11.28 より古い場合は更新を勧める警告を表示します。metadata と選択 install の機能は個別に確認し、任意機能がなければ検証済みの fallback を使います。完全 install と runtime 検証は省略しません。

`plan` と `apply` のあとには `git diff` を確認してください。成功後もバックアップは残ります。名前と復旧上の制約は[復旧とトラブル対応](https://github.com/Onely7/uv_torch_compass/blob/main/docs/recovery_ja.md)で説明します。

## 対応範囲

- `plan`、`apply`、`check` は Linux 専用です。`--help` と `--version` はほかの OS でも動作します。
- CPU と NVIDIA CUDA に対応します。AMD ROCm と Intel XPU は非対応です。
- PyTorch の公式 stable・nightly index に対応します。stable の失敗から nightly へ自動移行しません。
- 基本依存、選択した extra・依存グループ、uv workspace、`torchvision`、`torchaudio` に対応します。
- CUDA の成功には、GPU tensor、cuBLAS、cuDNN、architecture、選択した関連 package の検証が必要です。`--probe-profile compile` では `torch.compile` も確認します。
- GPU を指定しない場合、見えている device のうち空き memory が最も多いものを選びます。選択を固定するには `--cuda-device` を使います。
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

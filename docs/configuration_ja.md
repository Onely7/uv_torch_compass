# 設定

[English](configuration.md) · [ドキュメント一覧](README_ja.md) · [CLI の使い方](usage_ja.md)

設定は、明示した CLI option、名前空間付き環境変数、`[tool.uv-torch-compass]`、組み込みの初期値の順で解決します。上位の値が下位の値を置き換え、複数の層にある list を結合することはありません。

依存 package の version は標準の project 依存 table にだけ記録します。tool table へ重複保存しません。

## プロジェクト設定

継続して使う初期値は、対象 `pyproject.toml` に追加します。

```toml
[tool.uv-torch-compass]
python = "3.12"
backend = "auto"
channel = "stable"
cuda-compatibility = "strict"
probe-profile = "standard"
framework-probes = ["vllm"]
extras = ["vision"]
groups = ["training"]
cuda-device = "0"
link-mode = "copy"
log-dir = ".uv-torch-compass/logs"
timeout = 1800
output-format = "text"
```

すべて省略できます。未知の key、誤った型、明示的な空文字列、不正な backend・channel、0 以下の timeout は、候補検証やファイル更新を始める前に拒否します。

`report-file` と依存条件の上書きは、この table に置きません。report は実行ごとの情報であり、package の依存条件は `[project].dependencies`、`[project.optional-dependencies]`、`[dependency-groups]` に記録するためです。

## 環境変数

| 環境変数 | 設定内容 |
| --- | --- |
| `UV_TORCH_COMPASS_PYPROJECT` | 対象 `pyproject.toml` |
| `UV_TORCH_COMPASS_PYTHON` | uv の Python request |
| `UV_TORCH_COMPASS_TORCH` | 基本依存の `torch` 上書き |
| `UV_TORCH_COMPASS_TORCHVISION` | 基本依存の `torchvision` 上書き |
| `UV_TORCH_COMPASS_TORCHAUDIO` | 基本依存の `torchaudio` 上書き |
| `UV_TORCH_COMPASS_BACKEND` | `auto`、`cpu`、`cuda`、`cuNNN` |
| `UV_TORCH_COMPASS_CHANNEL` | `stable` または `nightly` |
| `UV_TORCH_COMPASS_CUDA_COMPATIBILITY` | `strict`、または明示的に許可する `minor` |
| `UV_TORCH_COMPASS_PROBE_PROFILE` | `standard` または `compile` |
| `UV_TORCH_COMPASS_FRAMEWORK_PROBES` | comma 区切りの明示的な framework 検証。現在は `vllm`。インストール済み vLLM は自動検証する |
| `UV_TORCH_COMPASS_EXTRAS` | comma 区切りの extra |
| `UV_TORCH_COMPASS_GROUPS` | comma 区切りの依存グループ |
| `UV_TORCH_COMPASS_CUDA_DEVICE` | NVIDIA index または UUID |
| `UV_TORCH_COMPASS_LINK_MODE` | `clone`、`copy`、`hardlink`、`symlink` |
| `UV_TORCH_COMPASS_LOG_DIR` | 対象 project からの相対、または絶対 log directory |
| `UV_TORCH_COMPASS_TIMEOUT` | 重い command の正の timeout 秒数 |
| `UV_TORCH_COMPASS_OUTPUT_FORMAT` | `text` または `json` |
| `UV_TORCH_COMPASS_REPORT_FILE` | 最終 JSON report のパス |

comma 区切りの extra・group に含まれる空要素と重複は、最初の出現順を保って取り除きます。

```bash
export UV_TORCH_COMPASS_EXTRAS='vision,audio,vision,'
```

この例は `vision`、`audio` の順に解決されます。

## 組み込みの初期値

| 設定 | 初期値 |
| --- | --- |
| 対象 | `./pyproject.toml` |
| Python | `.python-version`、次に `[project].requires-python` |
| backend | `auto` |
| channel | `stable` |
| CUDA compatibility | `strict` |
| probe profile | `standard` |
| 明示的な framework probe | なし。インストール済み vLLM は自動検出する |
| extra と group | なし |
| CUDA device | 現在の CUDA 選択や `--cuda-device` で固定しない場合、見えている device のうち空き memory が最大のもの |
| link mode | `copy` |
| log directory | 対象 project 以下の `.uv-torch-compass/logs` |
| project 操作の timeout | 1800 秒 |
| 出力 | `text` |

設定可能な timeout は、候補の install、project 検査、runtime probe、lock、sync に使います。uv の version や利用可能な backend 名を読む短い metadata command には、別の 30 秒制限を残します。

検証済みの uv 最低 version は 0.11.28 です。それより古いという理由だけでは拒否しません。警告を出し、metadata と選択 install の機能を個別に確認します。workspace metadata が使えなければ上限付き lockfile parser を使い、選択 install に対応しなければ vLLM wheel の事前検査だけを省略します。完全 install と runtime 検証は引き続き実行します。

`strict` は、選択した driver が通常サポートする範囲より新しい CUDA runtime を拒否します。`minor` は同じ CUDA major 内の制限付き互換性を明示的に許可し、採用時は警告を残します。`standard` は tensor、NumPy、cuBLAS、cuDNN、architecture、選択した関連 package を確認し、`compile` は `torch.compile` も追加します。

## uv へ渡す環境

uv が使う network、proxy、証明書、index 認証、cache、keyring、offline の設定は子 process に引き継ぎます。値そのものを診断ログへ記録しません。

対象 project、Python、backend、lock 方針、virtual environment を暗黙に変えられる変数は除外し、検証済みの値だけを設定し直します。対象には `VIRTUAL_ENV`、`UV_PROJECT`、`UV_PYTHON`、`UV_TORCH_BACKEND`、`UV_LOCKED`、`UV_FROZEN`、`UV_NO_SYNC`、すべての `UV_TORCH_COMPASS_*` を含みます。

既存の `CUDA_VISIBLE_DEVICES` は保持します。`--cuda-device` がなければ、そこにある先頭の index または UUID を選びます。空値または `-1` は CUDA を非表示にします。明示した `--cuda-device` が最優先です。

依存条件の置き場所は[プロジェクトと依存範囲](projects-and-scopes_ja.md)へ進んでください。

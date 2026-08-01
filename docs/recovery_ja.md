# 復旧とトラブル対応

[English](recovery.md) · [ドキュメント一覧](README_ja.md) · [プロジェクト README](../README_ja.md)

`apply` は project metadata と lockfile を一つの更新単位として扱います。中断された package install を byte 単位で戻すことはできないため、環境の復旧はファイルとは分けて試行・報告します。

## 関係するファイル

| パス | 役割 |
| --- | --- |
| 対象 `pyproject.toml` | 検証済みの Linux 用 PyTorch source と必要な依存更新を受け取る |
| workspace root の `uv.lock` | workspace 全体の解決結果を受け取る |
| 通常は `.venv` の対象環境 | lock 済みの対象 package と scope を受け取る |
| `.uv-torch-compass/logs/<command>-<時刻>-<pid>-<suffix>.log` | private な実行 log。初期値では対象 project からの相対パス |
| workspace root の `.uv-torch-compass.lock` | 同時 apply を防ぐ再利用可能な advisory lock inode |
| `pyproject.toml.bak.<時刻>[.<suffix>]` | apply 前 metadata の永続 backup |
| `uv.lock.bak.<時刻>[.<suffix>]` | 実行前に lockfile があった場合の永続 backup |

backup は成功後も残ります。既存の backup がある場合は数値 suffix を加え、上書きしません。

## 自動 rollback

lock 後、`apply` は最初に `uv sync --locked --dry-run` を実行します。事前検査に失敗した場合は、環境がまだ変わっていないため、ファイルだけを復元して環境復旧は行いません。実際の同期開始後に sync、最終 probe、timeout、SIGINT、SIGTERM のいずれかが失敗すると、次の順で処理します。

1. 中断された子 process group を復旧前に停止する
2. `pyproject.toml` と以前の `uv.lock` を、書きかけを見せない一括置換で復元する
3. 元の lockfile があった場合は、復元したファイルで sync する
4. 元の lockfile がなければ、復旧用の一時 lock を作って sync を試し、lock を再び削除する
5. ファイル復元と環境復旧を別々の結果として報告する

置換前には期待している content hash と比較します。editor や別 process による未知の変更があれば、その内容を上書きせず、競合を報告して backup を残します。

transaction の対象と report の保存先に symbolic link は指定できません。`apply` は workspace lock の取得後に `pyproject.toml` と `uv.lock` の両方を再確認するため、lock 待機中の変更も上書きせず拒否します。

復旧 sync が失敗した場合、ファイルは復元済みでも `.venv` を手動で直す必要があります。どの復旧が失敗したかを警告で確認できます。

## 手動で復旧する

実行 log を読み、backup の内容を確認してから copy します。対象ファイルがある directory で次を実行します。

```bash
cp pyproject.toml.bak.YYYYMMDD-HHMMSS pyproject.toml
cp uv.lock.bak.YYYYMMDD-HHMMSS uv.lock
uv sync --locked
```

二つ目の copy は lock backup がある場合だけ使います。workspace では、`uv.lock` の backup が member の `pyproject.toml` と同じ場所ではなく workspace root にあります。

復旧後は次を確認します。

```bash
git status
git diff -- pyproject.toml uv.lock
uv-torch-compass check
```

`check` を利用できるのは、有効な uv-torch-compass source 設定が残っている場合です。なければ先に TOML を確認・修復してください。

## ログ

各 command は重複しない mode `0600` の log を一つ作ります。`--log-dir` で directory を変更できますが、選んだファイル名を再利用・上書きする option はありません。

log には phase、マスク済み subprocess 出力、package version、ローカルパス、GPU 情報を含みます。一般的な認証情報は除去しますが、共有前に内容を確認してください。`--report-file` は別の mode `0600` JSON を作ります。

## 主な失敗

| message または状態 | 確認内容 |
| --- | --- |
| `uv was not found in PATH` | 同じ shell で `uv --version` を実行する |
| backend 選択が未対応 | uv を更新し、`uv pip install --help` に `--torch-backend` があるか確認する |
| uv が 0.11.28 より古いという警告 | 可能なら uv を更新する。metadata と選択 install は別々に確認し、fallback を使う場合も完全検証は省略しない |
| Linux 専用 command の失敗 | `plan`、`apply`、`check` は Linux で実行する。help/version だけはほかの OS でも動く |
| 選択 Python に適用できる PyTorch requirement がない | extra・group と PEP 508 の Python・implementation marker を確認する |
| `nvidia-smi` の失敗または CUDA version 不在 | NVIDIA driver、device の可視性、`CUDA_VISIBLE_DEVICES` を確認する |
| strict compatibility を満たす CUDA backend がない | NVIDIA driver を更新する、PyTorch version 条件を緩める、`--backend cpu` を明示する、または `--cuda-compatibility minor` を許可できるか確認する |
| 設定済み backend が strict compatibility で許可されない | 記録済み source が driver の通常サポートより新しい可能性がある。`plan` で対応 source を確認する。`check` 自体は変更しない |
| CUDA runtime component が backend と一致しない | lockfile から同期環境を作り直し、index 設定を確認する。インストール済み CUDA major・minor は `cuNNN` と一致する必要がある |
| minor で native architecture がない | 選択 GPU 用の native machine code が wheel にない。別 backend を選ぶか、driver 更新後に strict へ戻す |
| compile probe の失敗 | `--probe-profile standard` で再実行し、通常の CUDA 動作と任意の Inductor／Triton 経路を切り分ける |
| vLLM framework probe の失敗 | install 済み `vllm` の metadata、native extension、報告された platform を確認する。vLLM 解決時は自動実行され、model の読込みや worker の起動は行わない |
| `framework-cuda-abi` | 必要な CUDA variant または `libcudart.so.N` の major を候補と driver に照合する。対応 vLLM wheel を選ぶ、必要 backend に対応する driver へ更新する、または vLLM を source build する |
| `DTensor` を含む `framework-api-incompatibility` | 表示した依存経路を確認する。確認済みの vLLM 0.6.0 の場合、Transformers を 4.44.2 に制約する、`cu121` を選ぶ、または vLLM を更新する。backend 非依存の失敗では後続候補の install を止める |
| 候補の source 方針を準備できない | 対象 project が使う path、Git、URL、workspace、constraint、override、index source を確認する。候補検証では関連 source を PyPI に暗黙置換せず保持する |
| `lock-schema-unsupported` | fallback lock parser が生成済み schema を認識できない。uv-torch-compass を更新するか、対応済み uv を使う。backend がないことを示す失敗ではない |
| `tool-validation-error` | 拒否された metadata field を private log で確認する。metadata の境界を通るまで候補を検証済みとして扱わない |
| vLLM の範囲指定が非互換な最新版を選ぶ | Requested、Resolved、Rejected を確認する。同じ backend で最大16 releaseを検証し、全検証を通った release だけを管理 constraint として提案する |
| 候補の `lock` 失敗 | 候補の failure と private log で package、requirement、index、platform を確認する。有効な lock を解析できるまでは PyTorch の解決成功として表示しない |
| PyTorch 解決後の候補 `install` 失敗 | 最初に `Resolved PyTorch` を確認し、次に原因 package と依存経路を調べる。すべての候補で wheel がない package は CUDA backend を変えても解決しない |
| project の `uv lock` 失敗 | 選択していない scope や別 workspace member を含む uv の resolver 説明を読む |
| 利用可能な backend がない | 候補ごとのpackage、requirement、依存経路、indexと対応案を確認する。failure kindが`unknown`の場合はprivate logを確認する |
| `uv sync preflight failed` と利用可能な wheel がない旨の表示 | uv が示す package と platform を確認する。現在の Linux architecture は `tool.uv.required-environments` に記録されるため、互換 version があれば uv が lock 時に選択できる |
| 環境が同期されていない | 意図した `apply` を実行するか、`uv sync --locked --check` の出力を調べる |
| 最終 runtime 検証の失敗 | 同期後 project で一時候補の結果を再現できていない。rollback の結果を確認する |
| apply 後に report を書き込めない | report 保存は transaction 完了後に行うため、project 更新は適用済みのままになる。終了 code は `1`。private log を確認し、安全な通常ファイルの保存先へ変更して再実行する |
| plan/check 中にファイルが変わった | editor や別の依存更新 process が終わってから再実行する |
| 別 process が workspace を更新中 | ほかの `apply` が終わるまで待つ。強制的な同時実行のため active lock を消さない |

CPU の動作だけを切り分ける場合は `plan --backend cpu` を使います。NVIDIA GPU が見える場合、初期値の `auto` と `cuda` はどちらも CPU へ fallback しません。`nvidia-smi` が存在するのに正しく検査できない場合は、CPU 専用と仮定せず、そのエラーを直してから再実行してください。

## backup を残す期間

application 自身のテストと `uv-torch-compass check` が成功するまで backup を残してください。その後の長期履歴には通常 version control が適しています。明示的に内容を確認した backup だけを削除し、不確かな directory から広い wildcard を使わないでください。

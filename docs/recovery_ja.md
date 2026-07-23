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

backup 作成後に lock、sync、最終 probe、timeout、SIGINT、SIGTERM のいずれかが失敗すると、次の順で処理します。

1. 中断された子 process group を復旧前に停止する
2. `pyproject.toml` と以前の `uv.lock` を、書きかけを見せない一括置換で復元する
3. 元の lockfile があった場合は、復元したファイルで sync する
4. 元の lockfile がなければ、復旧用の一時 lock を作って sync を試し、lock を再び削除する
5. ファイル復元と環境復旧を別々の結果として報告する

置換前には期待している content hash と比較します。editor や別 process による未知の変更があれば、その内容を上書きせず、競合を報告して backup を残します。

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
| Linux 専用 command の失敗 | `plan`、`apply`、`check` は Linux で実行する。help/version だけはほかの OS でも動く |
| 選択 Python に適用できる PyTorch requirement がない | extra・group と PEP 508 の Python・implementation marker を確認する |
| `nvidia-smi` の失敗または CUDA version 不在 | NVIDIA driver、device の可視性、`CUDA_VISIBLE_DEVICES` を確認する |
| 利用できる backend がない | 候補の警告、version 条件、network/index、disk 容量、GPU runtime error を確認する |
| `uv lock` の失敗 | 選択していない scope や別 workspace member を含む uv の resolver 説明を読む |
| 環境が同期されていない | 意図した `apply` を実行するか、`uv sync --locked --check` の出力を調べる |
| 最終 runtime 検証の失敗 | 同期後 project で一時候補の結果を再現できていない。rollback の結果を確認する |
| plan/check 中にファイルが変わった | editor や別の依存更新 process が終わってから再実行する |
| 別 process が workspace を更新中 | ほかの `apply` が終わるまで待つ。強制的な同時実行のため active lock を消さない |

CPU の動作だけを切り分ける場合は `plan --backend cpu`、CPU fallback なしで GPU を必須にする場合は `plan --backend cuda` を使います。

## backup を残す期間

application 自身のテストと `uv-torch-compass check` が成功するまで backup を残してください。その後の長期履歴には通常 version control が適しています。明示的に内容を確認した backup だけを削除し、不確かな directory から広い wildcard を使わないでください。

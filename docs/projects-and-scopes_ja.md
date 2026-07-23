# プロジェクトと依存範囲

[English](projects-and-scopes.md) · [ドキュメント一覧](README_ja.md) · [CLI の使い方](usage_ja.md)

ここでは、依存関係を記述する基本依存、extra、依存グループの各場所を「依存範囲」と呼びます。基本の依存 list は常に選択し、`--extra` と `--group` を繰り返すと検証対象を追加できます。

## 基本依存

通常の PEP 508 依存条件を `[project].dependencies` に記述します。

```toml
[project]
name = "trainer"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["torch>=2.5,<3"]
```

`--torch`、`--torchvision`、`--torchaudio` は、この基本 list の依存条件を追加・置換します。PyTorch の直接 URL と Git requirement は、検証対象の公式 index を迂回するため拒否します。

## extra と依存グループ

extra と group は名前で選びます。

```toml
[project.optional-dependencies]
vision = ["torchvision>=0.20"]

[dependency-groups]
audio = ["torchaudio>=2.5"]
training = [
    { include-group = "audio" },
    "pytest>=8",
]
```

```bash
uv-torch-compass plan --extra vision --group training
```

存在しない依存範囲、未対応の group 要素、include の循環、不正な requirement、PyTorch の矛盾する完全一致 version、適用可能な PyTorch がない選択は、対象ファイルを変更する前に拒否します。

選択範囲が実質的に `torchvision` または `torchaudio` を含み、`torch` を含まない場合は、同じ場所の変更案へ `torch` を追加します。include-group の展開も考慮するため、この例では `training` 自身に `torch` が入ります。

PEP 508 marker が、解決済み Linux Python の version と implementation に一致しない requirement は probe から外します。適用可能な PyTorch requirement が一つ以上残る必要があります。

選択していない extra と group の宣言は書き換えません。ただし uv が lockfile 全体を解決するときには影響するため、project 内の別の非互換性で `uv lock` が失敗し、rollback する場合があります。

## source の更新

`cpu` が成功した場合、選択した PyTorch package に同じ Linux 限定 source を設定します。

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]
torchvision = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

CUDA と nightly の名前は `pytorch-cu128`、`pytorch-nightly-cu128` の形式です。同じ名前で別 URL の index があっても上書きしません。古い公式 PyTorch index は、どの source からも参照されていない場合だけ削除します。

既存の Linux 以外の source には marker と `and sys_platform != 'linux'` を組み合わせ、元の動作を保ちます。Linux source は追加し続けず置換し、TOML の構造が許す範囲でコメントと順序を保持します。同じ検証結果を再適用しても内容は変わりません。

NumPy bridge の失敗が `numpy<2` で直った場合は、必要な各 PyTorch 依存範囲へ次のような条件を追加します。

```toml
dependencies = [
    "torch>=2.5,<3",
    "numpy<2; sys_platform == 'linux'",
]
```

既存の適用対象 NumPy requirement が version 2 以上を必須にする場合は、暗黙に弱めず矛盾として報告します。

## uv workspace

uv の workspace metadata を使って、対象 member、workspace root、member package 名、root の共有 `uv.lock` を解決します。古い uv が検出済み workspace を調べられない場合は、uv の更新を案内して編集前に終了します。

member を対象にした場合の動作は次のとおりです。

- 対象 member の `pyproject.toml` だけを編集する
- root の `uv.lock` を workspace 全体について更新する
- sync と最終 runtime は対象 package と選択した extra・group に限定する
- member metadata と共有 lockfile を一つの更新単位として backup・restore する
- root の advisory lock で uv-torch-compass 同士の同時更新を防ぐ

`plan` と `check` は両ファイルの snapshot を取ります。実行中に editor や別 process がどちらかを変更すると、現在の結果として報告せず失敗させます。

更新と復元の詳細は[復旧とトラブル対応](recovery_ja.md)へ進んでください。

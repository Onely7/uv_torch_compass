# report と自動化

[English](reports.md) · [ドキュメント一覧](README_ja.md) · [CLI の使い方](usage_ja.md)

対話的に使う場合の初期値は text です。JSON では script や CI が扱える最終 object を一つ返します。

## text 出力

進捗は `INSPECT`、`RESOLVE`、`VERIFY`、`APPLY`、`RESTORE` という phase 名で表示します。補足情報、警告、最終 summary から、選択 backend、互換性判定、除外した新しい候補、公式 index、変更、backup、log を確認できます。

`plan` の summary には、`pyproject.toml` の前後行を含まない unified diff を表示します。`apply` の前に確認してください。

## JSON 出力

`--output-format json` を指定します。

```bash
uv-torch-compass plan --output-format json > result.json
```

stdout には最後の JSON object 一つだけを出します。進捗と警告は stderr へ送るため、stdout の redirect から解析可能な文書を得られます。

schema version は `4` で、次の top-level field を含みます。

```json
{
  "schema_version": 4,
  "operation": "plan",
  "status": "failed",
  "exit_code": 1,
  "applied": false,
  "target": "/work/app/pyproject.toml",
  "workspace": "/work/app",
  "request": {
    "backend": "auto",
    "cuda_compatibility": "strict",
    "probe_profile": "standard"
  },
  "python": {},
  "candidate_attempts": [
    {
      "backend": "cu121",
      "stage": "install",
      "status": "failed",
      "reason": "The required package build is unavailable from this index.",
      "compatibility": "strict",
      "failure": {
        "kind": "no-compatible-distribution",
        "summary": "The required package build is unavailable from this index.",
        "package": {
          "name": "torch",
          "version": "2.10.0",
          "requirement": "torch==2.10.0"
        },
        "required_by": ["vllm>=0.25.0", "torch==2.10.0"],
        "index": {
          "name": "pytorch-cu121",
          "url": "https://download.pytorch.org/whl/cu121"
        },
        "platform": null,
        "suggestions": ["Select a dependency version compatible with a published PyTorch build."]
      }
    }
  ],
  "resolution_failure": {},
  "selected_backend": "",
  "selected_index": "",
  "selected_gpu": {},
  "resolved_packages": {},
  "dependency_roots": [],
  "source_anchors": [],
  "required_environment": "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "validation": {},
  "changes": [],
  "backups": [],
  "warnings": [],
  "errors": [],
  "timing": {}
}
```

実際の文書には、対象 package、変更案の diff、実行 log、秘密情報を含まない診断 metadata も入ります。GPU metadata には選択 device、NVIDIA driver version、`nvidia-smi` が示す CUDA 上限を記録します。

各 `candidate_attempts` は `backend`、`stage`、`status`、`reason`、`compatibility` と、省略可能な `failure` を持ちます。方針で除外した候補はinstall前に `skipped` として記録します。`resolution_failure`は、全候補の失敗package、index、重複除去した対応案を集約します。uv出力から確定できないfieldは推測せずJSON `null`にします。CUDA の `validation` には次を含みます。

- 解決した PyTorch CUDA runtime と runtime component version
- 必要な最低 driver と、`strict`、`minor`、`unsupported` の判定
- GPU compute capability と PyTorch の compiled architecture
- CUDA tensor、cuBLAS、cuDNN、native architecture、NumPy、`torchvision`、`torchaudio`、必要に応じた compile の結果

候補の詳細には長さを制限したredaction済み要約を記録し、生の command data は残しません。完全なredaction済みuv出力はprivate logに残ります。fieldを利用する側は、先に `schema_version` を確認してください。

status は次のいずれかです。

| status | 意味 |
| --- | --- |
| `planned` | 候補検証に成功し、読み取り専用で変更案を作った |
| `success` | 変更と最終検証に成功した |
| `success_with_warnings` | 致命的ではない警告を伴って変更と検証に成功した |
| `valid` | `check` で記録済み状態が新しく、実行可能だった |
| `failed` | 設定、command、検証、復旧のいずれかに失敗した |

JSON 指定を認識できた場合、設定・runtime の失敗も同じ schema と終了コード `1` で返します。argparse が解析できない構文エラーは JSON 契約を確立できないため、stderr と終了コード `2` で報告します。

## report file

`--report-file` は、terminal の出力形式と関係なく同じ最終 JSON を保存します。

```bash
uv-torch-compass apply \
  --output-format text \
  --report-file artifacts/torch-compass.json
```

相対パスは対象 project から解決します。同じ directory の一時ファイルを使って書きかけを残さず置換し、mode `0600` を設定します。指定した report を書けない場合は、欠落したまま成功にせず command 自体を失敗させます。

## 認証情報の扱い

text が log、diff、JSON、report へ入る前に、共通 redactor が URL の user 情報、query 内の token・key・password・secret・signature、authorization header、秘密値らしい変数代入を除去します。子 process の環境変数は値を記録せず、除外した制御変数の名前だけを記録する場合があります。

redaction があるからといって、log を無確認で公開してよいわけではありません。ローカルパス、package version、GPU 名、一般的ではない形式の認証情報が機密になる場合があるため、共有前に内容を確認してください。

## CI の例

```bash
set -o pipefail
uv-torch-compass check --output-format json --report-file compass.json \
  | jq -e '.status == "valid"'
```

成否の第一判断には process の終了コードを使い、構造化した報告には `status`、`warnings`、`errors` を確認します。

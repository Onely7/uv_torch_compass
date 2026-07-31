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

schema version は `7` です。候補の解決結果、artifact の根拠、その後の失敗を分けて表します。

```json
{
  "schema_version": 7,
  "operation": "plan",
  "status": "failed",
  "exit_code": 1,
  "applied": false,
  "target": "/work/app/pyproject.toml",
  "workspace": "/work/app",
  "request": {
    "backend": "auto",
    "cuda_compatibility": "strict",
    "probe_profile": "standard",
    "framework_probes": []
  },
  "python": {},
  "candidate_attempts": [
    {
      "backend": "cu126",
      "stage": "install",
      "status": "failed",
      "reason": "A required wheel is unavailable for the selected platform.",
      "compatibility": "strict",
      "framework_compatibility": null,
      "resolution": {
        "status": "resolved",
        "environment": {
          "implementation": "cpython",
          "python_version": "3.12.12",
          "python_minor": "3.12",
          "sys_platform": "linux",
          "platform_machine": "x86_64",
          "required_marker": "sys_platform == 'linux' and platform_machine == 'x86_64'"
        },
        "pytorch": {
          "torch": {
            "version": "2.10.0+cu126",
            "index": "https://download.pytorch.org/whl/cu126"
          }
        },
        "package_count": 182
      },
      "phases": {
        "lock": "passed",
        "artifact": "passed",
        "install": "failed",
        "runtime": "not-run",
        "framework": "not-run"
      },
      "failure": {
        "kind": "wheel-unavailable",
        "summary": "xgrammar has no wheel for the selected platform.",
        "package": {
          "name": "xgrammar",
          "version": "0.2.4",
          "requirement": "xgrammar==0.2.4"
        },
        "required_by": ["vllm==0.19.1", "xgrammar==0.2.4"],
        "index": null,
        "platform": "linux-x86_64",
        "dependency_paths": [
          ["uv-torch-compass-candidate==0", "vllm==0.19.1", "xgrammar==0.2.4"]
        ],
        "available_wheel_platforms": ["linux-aarch64", "macos-x86_64"],
        "suggestions": ["Select a dependency version with a wheel for linux-x86_64."]
      }
    }
  ],
  "blocking_summary": {
    "summary": "Compatible PyTorch builds were resolved, but a later candidate phase failed.",
    "pytorch_builds_found": [
      {
        "backend": "cu126",
        "index": "https://download.pytorch.org/whl/cu126",
        "packages": {"torch": "2.10.0+cu126"}
      }
    ],
    "common_blockers": [],
    "suggestions": []
  },
  "selected_backend": "",
  "selected_index": "",
  "selected_gpu": {},
  "resolved_packages": {},
  "dependency_roots": [],
  "source_anchors": [
    {"package": "torch", "scope": "base"}
  ],
  "required_environment": "sys_platform == 'linux' and platform_machine == 'x86_64'",
  "probe_contract": {},
  "framework_validation": [],
  "operation_state": {
    "applied": false,
    "report_written": false
  },
  "environment_policy": {},
  "validation": {},
  "changes": [],
  "backups": [],
  "warnings": [],
  "errors": [],
  "timing": {}
}
```

実際の文書には、対象 package、変更案の diff、実行 log、秘密情報を含まない診断 metadata も入ります。GPU metadata には選択 device、NVIDIA driver version、`nvidia-smi` が示す CUDA 上限を記録します。

各 `candidate_attempts` は `backend`、`stage`、`status`、`reason`、`compatibility`、`phases` と、省略可能な `resolution`、`framework_compatibility`、`failure` を持ちます。五つの phase は `lock`、`artifact`、`install`、`runtime`、`framework` です。lock に成功したあとは、後続 phase が失敗しても、実際の実行環境、解決済み PyTorch package、framework 関連 package version を `resolution` に保持します。

`blocking_summary` は、同じ原因を複数候補の間でまとめつつ、確定的な失敗後に省略した候補も含め、個別 attempt をすべて残します。「どの候補も解決できなかった」と「PyTorch は解決したが後続 package または検証が失敗した」を区別します。uv 出力から確定できない field は推測せず JSON `null` にします。

schema 7 では、実行した `probe_contract`、自動または明示指定を示す `framework_validation` trigger、子 process の環境変数方針、scope 付き source anchor、最終操作状態、確認済み framework catalog の根拠も記録します。CUDA の `validation` には次を含みます。

- 解決した PyTorch CUDA runtime と runtime component version
- 必要な最低 driver と、`strict`、`minor`、`unsupported` の判定
- GPU compute capability と PyTorch の compiled architecture
- CUDA tensor、cuBLAS、cuDNN、native architecture、NumPy、`torchvision`、`torchaudio`、必要に応じた compile の結果

候補の詳細には長さを制限したredaction済み要約を記録し、生の command data は残しません。完全なredaction済みuv出力はprivate logに残ります。fieldを利用する側は、先に `schema_version` を確認してください。

framework の失敗は resolver field へ押し込まず、専用構造で返します。`binary_requirement` には必要な CUDA variant・major、必要 library、`catalog|elf|metadata|runtime` の判定根拠を含めます。さらに、解決済み framework `packages`、上限付き `exception`、`dependency_paths`、`backend_independent` を記録できます。たとえば、`DTensor` 不足は `framework-api-incompatibility`、`cu129` での `libcudart.so.13` 要求は `framework-cuda-abi` です。公開する例外情報は、型、認証情報を除去した message、symbol・module、consumer・provider package、basename だけの最大 12 frame に限定します。

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

相対パスは対象 project から解決します。同じ directory の一時ファイルを使って書きかけを残さず置換し、mode `0600` を設定します。report は最終 stdout より先に保存します。project への適用に成功したあと report だけを保存できなかった場合は、project を戻さず終了コード `1` とし、stderr の構造化結果へ `applied: true` と `operation_state.report_written: false` を記録します。

## 認証情報の扱い

text が log、diff、JSON、report へ入る前に、共通 redactor が URL の user 情報、query 内の token・key・password・secret・signature、認証・cookie header、JSON の秘密 field、秘密値を伴う command option、秘密値らしい変数代入を除去します。子 process の環境変数は値を記録せず、除外した制御変数の名前だけを記録する場合があります。

redaction があるからといって、log を無確認で公開してよいわけではありません。ローカルパス、package version、GPU 名、一般的ではない形式の認証情報が機密になる場合があるため、共有前に内容を確認してください。

## CI の例

```bash
set -o pipefail
uv-torch-compass check --output-format json --report-file compass.json \
  | jq -e '.status == "valid"'
```

成否の第一判断には process の終了コードを使い、構造化した報告には `status`、`warnings`、`errors` を確認します。

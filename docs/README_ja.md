# uv-torch-compass ドキュメント

[English](README.md) · [プロジェクト README](../README_ja.md)

最初にルートの README を読んでください。詳しい情報が必要になったら、この一覧から目的に合う文書を選べます。上から順に読む必要はありません。

| 目的 | 文書 |
| --- | --- |
| `plan`、`apply`、`check` と CLI option を使う | [CLI の使い方](usage_ja.md) |
| 初期値、環境変数、設定の優先順位を指定する | [設定](configuration_ja.md) |
| Python、backend、channel、GPU、実行検証を理解する | [選択の仕組み](how-it-works_ja.md) |
| 基本依存、extra、依存グループ、workspace member を選ぶ | [プロジェクトと依存範囲](projects-and-scopes_ja.md) |
| text 出力、JSON、report file、終了コードを使う | [report と自動化](reports_ja.md) |
| バックアップ、rollback、ログ、主な失敗を確認する | [復旧とトラブル対応](recovery_ja.md) |
| 検査、配布 package の build、公開準備 artifact を作る | [開発](development_ja.md) |

バージョン 0.1.0 で公開インターフェースとして扱うのは CLI です。`uv_torch_compass` 以下の Python module は内部実装であり、予告なく変更する場合があります。

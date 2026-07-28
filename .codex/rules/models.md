## モデル保存場所

別サーバーでも同じ手順で動くよう、パスはリポジトリ相対をデフォルトにすること。

- モデル: `./model/`
- データセット: `./data/`
- 実験出力: `./outputs/`
- キャッシュ: `./.cache/`
- ランタイムや生成したvenv: `./.runtime/`

容量やディスク都合で外部ストレージへ逃がす場合は、コード中に絶対パスを直書きせず、`TEST_DSPY_MODEL_DIR`、`TEST_DSPY_DATA_DIR`、`TEST_DSPY_OUTPUT_DIR`、`TEST_DSPY_CACHE_DIR`、`TEST_DSPY_RUNTIME_DIR` の環境変数で上書きすること。

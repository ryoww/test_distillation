# Python ルール

Python 関連の作業では必ず `uv` を使う。`uv` で代替できる場合、`pip install`・`python -m venv`・素の `python` コマンドは使わない。

| 用途 | コマンド |
|---|---|
| 依存関係のインストール | `uv sync` |
| コマンド・スクリプトの実行 | `uv run <command>` |
| 依存関係の追加 | `uv add <package>` |

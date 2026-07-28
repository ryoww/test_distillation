# Node.js ルール

- このリポジトリで Node.js 関連の作業を行う場合は、ランタイム管理に必ず `volta` を使う。
- プロジェクトで使用する Node.js のバージョンは `volta pin node@<version>` で固定する。
- このリポジトリのパッケージマネージャとしては `pnpm` を優先する。
- Node.js のワークフローでは、`pnpm install`、`pnpm add`、`pnpm remove`、`pnpm run <script>`、`pnpm exec <tool>` を使う。
- 文書化された例外がない限り、このリポジトリでは `npm`、`yarn`、`nvm`、`fnm`、`asdf` を使わない。
- `pnpm` を Volta 経由で管理する場合、Volta の `pnpm` サポートには回避策が必要になることがある。その場合は、回避策を使う前に理由を説明すること。


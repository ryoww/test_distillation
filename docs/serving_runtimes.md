# ローカル推論サーバー運用手順（vLLM / llama.cpp / Ollama）

このドキュメントは、`~/test_DSPy` の運用スクリプトとハンドオフ資料から、
このリポジトリの `test_distillation` 用に運用手順を抽出・要約したものです。
学習で得た adapter を配信するときや、外部で蒸留教師応答の再生成を試すときの
下敷きとして使います。

出典:

- `~/test_DSPy/scripts/serve_vllm.py`
- `~/test_DSPy/scripts/bootstrap_vllm_env.py`
- `~/test_DSPy/scripts/serve_llamacpp.py`
- `~/test_DSPy/scripts/serve_ollama.py`
- `~/test_DSPy/scripts/create_ollama_from_gguf.py`
- `~/test_DSPy/scripts/download_gguf.py`
- `~/test_DSPy/docs/cuda13-migration-handoff.md`
- `~/test_DSPy/qwen36-local-starter/qwen36_27b_vllm_video_segmentation_gpu_server(1).md`
- `~/test_DSPy/README.md`

## 方針

- ローカル LLM/VLM は Python プロセスに直接ロードせず、
  **OpenAI 互換 API** として起動してクライアントから HTTP で叩きます。
- サーバー環境と学習・クライアント環境は必ず venv を分けます。
  vLLM は CUDA driver / Torch wheel との相性が強いためです。
- モデル、キャッシュ、ランタイムは全てリポジトリ相対をデフォルトにします。
  外部ディスクを使う場合だけ `DISTILL_*` 環境変数で上書きします。
- Hugging Face token などの秘密情報はコマンドやログに書かず、環境変数で渡します。

## サーバー・ポートの規約

| ランタイム | エンドポイント | API key ヘッダ |
| --- | --- | --- |
| vLLM | `http://127.0.0.1:7501/v1` | `Authorization: Bearer local` |
| llama.cpp (`llama-server`) | `http://127.0.0.1:7501/v1` | `Authorization: Bearer local` |
| Ollama | `http://127.0.0.1:11434/v1` | 不要（OpenAI 互換は空でも通る） |

vLLM と llama.cpp は同じ 7501 を使うため、**同時起動しません**。
既存プロセスが 7501 を掴んでいる場合は先に停止します。

疎通確認:

```bash
curl http://127.0.0.1:7501/v1/models -H "Authorization: Bearer local"
curl http://127.0.0.1:11434/api/version
```

## vLLM

### 環境構築

このリポジトリでは [scripts/bootstrap_vllm_env.py](../scripts/bootstrap_vllm_env.py)
が `./.runtime/vllm/` に `vllm==0.20.0` を含む Python 3.13 の venv を作ります。
`--torch-backend auto` によって現在のドライバに合う wheel を uv が選びます。

```bash
uv run scripts/bootstrap_vllm_env.py
```

再構築する場合は `--rebuild` を渡します。乾式実行は `--dry-run` です。

CUDA driver とバイナリの適合性は必ず先に確認します。

```bash
./.runtime/vllm/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
PY
```

`cuda available True` にならない場合、vLLM を起動しても失敗します。
先にドライバと Torch wheel の組み合わせを直します。

### CUDA 12.2 サーバーへ持ち出す場合

`~/test_DSPy/scripts/bootstrap_vllm_env.py` は `--profile cuda122` を持ち、
Python 3.12 + Torch 2.4.0+cu121 + vLLM 0.6.3.post1 を導入します。
CUDA 13 用の現行 `bootstrap_vllm_env.py` はそのプロファイルを持たないため、
CUDA 12 系のサーバーへ移す場合は test_DSPy 側のスクリプトを流用するか、
このリポジトリの `bootstrap_vllm_env.py` に同じプロファイル分岐を追加します。

vLLM 0.6.3 は Qwen3 系（`Qwen3_5ForConditionalGeneration` など）に未対応です。
Qwen3.5/3.6 を CUDA 12.2 で使う場合は llama.cpp か Ollama に切り替えます。

### 起動（ベースモデル配信）

[scripts/serve_vllm.py](../scripts/serve_vllm.py) は `InternScience/Agents-A1-4B` を
デフォルトで配信します。単一 GPU では `CUDA_VISIBLE_DEVICES=0` を明示します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_vllm.py \
  --served-model-name agents-a1-4b \
  --max-model-len 4096
```

このスクリプトの既定では以下の 2 点を明示的に設定しています。

- `--gdn-prefill-backend triton`
  FlashInfer の初回 JIT は nvcc を要求するため、`triton` を既定にして
  `bootstrap_cuda_env.py` 未実行の状態でも起動できるようにしています。
- `VLLM_USE_DEEP_GEMM=0`
  現行サーバーで DeepGEMM warmup が失敗する回避策です。

### 起動（LoRA adapter を追加配信）

pilot 学習の adapter を配信して base と比較します。
`--adapter-path` は `outputs/<run-name>/adapter/` を指します。

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_vllm.py \
  --adapter-path outputs/agents-a1-4b-sft-final-pilot/adapter \
  --served-model-name agents-a1-4b-distilled \
  --max-model-len 4096
```

`serve_vllm.py` は `--enable-lora --max-lora-rank 64 --max-loras 1` を自動で足し、
`--lora-modules <served-model-name>=<adapter-path>` を組み立てます。
initial CUDA graph capture と `torch.compile` で初回は数分かかります。

### 疎通確認

```bash
curl http://127.0.0.1:7501/v1/chat/completions \
  -H 'Authorization: Bearer local' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "agents-a1-4b-distilled",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

Python から OpenAI SDK で叩く場合。

```bash
./.runtime/vllm/bin/python - <<'PY'
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:7501/v1", api_key="local")
r = c.chat.completions.create(
    model="agents-a1-4b-distilled",
    messages=[{"role": "user", "content": "Reply with exactly: ok"}],
    max_tokens=8,
    temperature=0,
)
print(r.choices[0].message.content)
PY
```

### 追加オプションの目安

vLLM CLI に未定義の flag は `serve_vllm.py` の末尾に位置引数として渡すと
そのまま `vllm serve` に転送されます。

| 目的 | 追加フラグ |
| --- | --- |
| tensor parallel（H200 x2） | `--tensor-parallel-size 2` |
| コンテキスト拡張 | `--max-model-len 131072`（安定後に `262144`） |
| VRAM 上限調整 | `--gpu-memory-utilization 0.85` |
| ローカル画像を許可 | `--allowed-local-media-path /abs/path` |
| Qwen3 系の reasoning parser | `--reasoning-parser qwen3` |
| マルチモーダルの画像上限 | `--limit-mm-per-prompt.image 96` |

`--host 0.0.0.0` は SSH トンネルまたは VPN 内でだけ使い、
外部からアクセス可能なネットワークに露出させません。

## llama.cpp（`llama-server`）

`~/test_DSPy/scripts/serve_llamacpp.py` は `llama-server` を OpenAI 互換で
起動する薄いラッパです。このリポジトリには複製していないため、
以下いずれかで運用します。

- test_DSPy のスクリプトを `PYTHONPATH=~/test_DSPy/scripts` から実行する。
- 同等の起動コマンドを直接叩く（後述）。
- `scripts/serve_llamacpp.py` を移植する（`project_paths` の差異に注意）。

`llama-server` は事前にビルド済みバイナリを PATH に置きます。
sudo が必要な導入は避け、ユーザー権限で入る配布を使います。
`LLAMA_SERVER_BIN` で明示パスも指定できます。

### GGUF のダウンロード

`~/test_DSPy/scripts/download_gguf.py` の等価コマンドは以下です。
Hugging Face のトークンは環境変数 `HF_TOKEN` に置き、コマンドには書きません。

```bash
export HF_HOME="$PWD/model/hf_home"
hf download Jackrong/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-Q4_K_M.gguf \
  --local-dir ./model/Jackrong/Qwen3.6-27B-GGUF
```

VLM の場合は `mmproj.gguf` も同じ手順で取得します。
A6000 48GB クラスでは `Q4_K_M` / `Q5_K_M` から試します。

### 起動

`llama-server` を OpenAI 互換 API として起動します。

```bash
CUDA_VISIBLE_DEVICES=0 llama-server \
  -m ./model/Jackrong/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 7501 \
  --api-key local \
  --alias Qwen/Qwen3.6-27B-GGUF \
  -c 8192 \
  -ngl 999
```

VLM を配信する場合は `--mmproj` に `mmproj.gguf` を渡します。

```bash
CUDA_VISIBLE_DEVICES=0 llama-server \
  -m ./model/Jackrong/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf \
  --mmproj ./model/Jackrong/Qwen3.6-27B-GGUF/mmproj.gguf \
  --host 127.0.0.1 --port 7501 --api-key local \
  --alias Qwen/Qwen3.6-27B-GGUF \
  -c 8192 -ngl 999
```

`-ngl 999` は「全レイヤーを GPU に載せる」意味です。
VRAM が足りない場合は具体数を指定します。

## Ollama

`~/test_DSPy/scripts/serve_ollama.py` はモデルストアに `./model/ollama` を、
バイナリに `./.runtime/ollama/ollama-0.22.0/bin/ollama` を既定として使います。

同等の直接コマンドは以下です。

```bash
export OLLAMA_MODELS="$PWD/model/ollama"
export OLLAMA_HOST="127.0.0.1:11434"
ollama serve
```

既に別の Ollama サーバーが 11434 を占有している場合は、先に停止します。

### GGUF を Ollama に登録

`~/test_DSPy/scripts/create_ollama_from_gguf.py` に相当する手順です。

`Modelfile` を作成します。

```text
FROM ./model/Jackrong/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf
PARAMETER num_ctx 8192
```

登録します。

```bash
ollama create qwen3.6-27b:q4_k_m -f Modelfile
```

疎通確認します。

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b:q4_k_m",
    "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    "temperature": 0,
    "max_tokens": 8
  }'
```

### DSPy / OpenAI SDK からの接続

`base_url` を Ollama の OpenAI 互換エンドポイントに向けます。
モデル名には Ollama のタグ名をそのまま渡します。

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:11434/v1", api_key="ollama")
```

DSPy 側は `openai/<name>` プレフィックスで指定します
（例: `openai/qwen3.6-27b:q4_k_m`）。

VLM の画像入力を DSPy 経由の OpenAI 互換 API で流したとき、
Ollama 側が受けきれない場合があります。その場合は Ollama native
`/api/chat` に base64 画像を直接送る実装へ切り替えます。

## 手順選択の目安

| 状況 | 推奨経路 | 理由 |
| --- | --- | --- |
| adapter 配信、ベース比較 | vLLM | LoRA を hot-swap しやすい |
| Qwen3 系 GGUF を CUDA 12.2 で試す | Ollama or llama.cpp | vLLM 0.6 系は未対応 |
| 単発疎通確認だけしたい | Ollama | 起動と登録が最軽量 |
| 長コンテキスト・高スループット | vLLM (CUDA 13) | continuous batching と KV cache |
| sudo 不可のサーバー | Ollama（既存導入）または llama-server user 権限版 | vLLM も動くが CUDA 13 driver 必須 |

## 保管と再構築

移設時に持ち出すと再ダウンロードが減るもの:

- `./model/ollama`
- `./model/<hf-org>/<gguf-repo>/*.gguf`
- `./model/hf_home`
- `./outputs`

移設先で作り直すもの:

- `./.runtime/vllm/`（CUDA / driver / Python 依存）
- `./.venv`

## 障害切り分け

### vLLM が起動しない

- `model_type` / `AutoConfig` エラー → vLLM / Transformers のバージョン不足。
  該当モデルの必要バージョンを確認し、必要なら `bootstrap_vllm_env.py` を
  改修して再構築します。
- CUDA initialize 失敗 → PyTorch wheel と driver の不整合。
  `./.runtime/vllm/bin/python` の `torch.version.cuda` と
  `nvidia-smi` の CUDA Version を照合します。
- OOM → `--max-model-len` を下げる、`--gpu-memory-utilization` を下げる、
  量子化モデルへ切り替える。
- Tokenizer / config 取得失敗 → `HF_HOME` のキャッシュとネットワーク、
  トークン、モデル名を確認します。
- 初回 JIT で nvcc が要求される → `--gdn-prefill-backend triton` を維持し、
  必要なら `bootstrap_cuda_env.py` でローカル `nvcc` を導入します。

### 初回の起動が遅い

`torch.compile` と CUDA graph capture で初回は数分かかります。
2 回目以降は短縮されます。

### 短答評価にゴミが混じる

Qwen3.6 GGUF は短答指示でも `<think>` や説明を出しやすいので、
評価スクリプト側で未完了 `<think>` を採点対象から除外するか、
プロンプト・テンプレート・採点処理を固定して比較条件を揃えます。

### Ollama 画像入力が通らない

DSPy の OpenAI 互換画像入力が Ollama で通らない場合は、
Ollama native `/api/chat` に base64 画像を直接送る実装に切り替えます。

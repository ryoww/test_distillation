---
name: ml-python-env
description: PyTorch、Transformers、TensorFlow を使う Python 機械学習環境を、CPU/GPU の目的に合わせて構築・再構築・診断する。uv / pip / conda など既存の package manager と lock file を尊重し、CUDA / ROCm / Apple Silicon / CPU の backend 選択、モデル読み込み、GPU 可視性、再現可能なセットアップ手順の文書化が必要なときに使う。
---

# ML Python Env

## 目的

PyTorch、Transformers、TensorFlow の Python 環境を、プロジェクトに合わせて小さく安定した形で構築・修復・検証する。場当たり的に package を足す前に、実際に使う Python、package manager、lock file、GPU backend、対象ワークロードを確認する。

## 最初に守ること

- 先にプロジェクトの指示ファイル、`pyproject.toml` / `requirements*.txt` / `environment.yml` / `uv.lock` / `poetry.lock` / `conda-lock.yml` / `.python-version` / 既存の実行スクリプトを読む。
- 既存の package manager を尊重する。`uv` project なら `uv sync` / `uv add` / `uv run`、pip project なら project venv 内の `python -m pip`、conda project なら `conda` / `mamba` を使う。
- 一時 shell や別 venv で通った結果を、project 環境で通ったことにしない。
- lock file や大量 dependency を変えるときは、どの失敗を直すためかを説明し、無関係な package update を混ぜない。
- 具体的な CUDA / TensorFlow / PyTorch 対応表は変わるため、version pin を決める直前に公式 docs または既存 lock を確認する。

## どの Python を直すか

作業前に、失敗している処理がどの Python で動くべきかを決める。

- application client が外部 inference server へ HTTP 接続するだけなら、client venv に GPU 版 Torch / TensorFlow を無理に入れない。
- training、local inference、evaluation、notebook kernel、worker process が直接 model を load するなら、その Python に ML stack が必要。
- serving runtime、batch worker、notebook kernel、project `.venv` が分かれている場合は、実際に model を読む Python を検証対象にする。
- `scripts/check_ml_stack.py` は、検証対象の Python から実行する。別 Python の結果で結論づけない。

## Backend を選ぶ

まず目的を分ける。

- CPU only: 再現性と install の軽さを優先する。GPU package を入れない。
- NVIDIA GPU: host driver、CUDA runtime、framework wheel の対応を確認する。`nvidia-smi` は host が見えているかの入口であり、Python package の成功条件ではない。
- AMD GPU: ROCm 対応 wheel と OS / GPU 対応を確認する。
- Apple Silicon: PyTorch は MPS を確認する。TensorFlow は公式 GPU サポートの扱いが OS ごとに異なるため、現行 docs を確認する。
- Windows native / WSL: TensorFlow GPU は native Windows と WSL で扱いが違う。GPU が必要なら WSL2 や Docker も選択肢に入れる。

## インストール方針

### 1. 既存 lock から再現する

既知の環境を戻す目的なら、dependency を変える前に lock から再構築する。

```bash
uv sync --locked
```

または project の既存手順に従う。

### 2. PyTorch

- PyTorch は公式 install selector か既存 lock の backend 方針を優先する。
- `uv pip` interface では `--torch-backend auto` または `--torch-backend cuXXX` を使える。
- `uv sync` / `uv add` の project workflow では、`--torch-backend` が効く前提で書かない。必要なら `pyproject.toml` の `[[tool.uv.index]]` / `[tool.uv.sources]` で PyTorch index を明示するか、runtime venv を分離して `uv pip` で入れる。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend auto torch torchvision
```

### 3. Transformers

- 先に backend framework を決める。PyTorch 用なら Torch、TensorFlow 用なら TensorFlow を入れてから `transformers` を入れる。
- stable release を優先する。model の `config.json` / `model_type` を release が読めないときだけ、newer release や source install を検討する。
- Hugging Face cache は、project が指定していなければ `HF_HOME` / `HF_HUB_CACHE` / `TRANSFORMERS_CACHE` で明示してよい。

```bash
uv pip install --python .venv/bin/python transformers huggingface-hub
```

### 4. TensorFlow

- TensorFlow は PyPI の公式 package を優先する。Linux / WSL の NVIDIA GPU では `tensorflow[and-cuda]` が使える場合がある。
- Python version、OS、GPU support の条件が PyTorch と違うため、PyTorch と同じ venv に無理に入れず、必要なら TensorFlow 専用 venv を作る。
- macOS や Windows native の GPU 対応は制約が強い。GPU が必須なら公式 docs の現行条件を確認してから進める。

```bash
python -m pip install "tensorflow[and-cuda]"
```

CPU only:

```bash
python -m pip install tensorflow
```

## 検証

import 成功だけで終えない。検証対象 Python で次を確認する。

```bash
python path/to/scripts/check_ml_stack.py --json
```

必要な stack を strict にする。

```bash
python path/to/scripts/check_ml_stack.py \
  --strict \
  --require torch-gpu \
  --require transformers \
  --model-dir /path/to/model \
  --expected-model-type <model_type>
```

TensorFlow GPU を確認する場合:

```bash
python path/to/scripts/check_ml_stack.py --strict --require tensorflow-gpu
```

確認する項目:

- `python.executable` が期待する環境を指している。
- host GPU command (`nvidia-smi` など) と framework の GPU 可視性を分けて見ている。
- `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` または MPS / ROCm 相当の状態。
- `transformers.__version__` と、必要なら `AutoConfig.from_pretrained(...)`。
- `tensorflow.__version__`, build 情報、`tf.config.list_physical_devices("GPU")`。

## Smoke test

環境チェックのあと、実ワークロードの最小経路を 1 本走らせる。

- PyTorch: 小さな tensor を GPU に載せる、または最小 forward pass。
- Transformers: 小さな tokenizer / model config / pipeline / local model load。
- TensorFlow: 小さな tensor op と GPU device placement。
- notebook や service で使うなら、その kernel / service command から同じ検証を実行する。

## 記録すること

最終報告には最低限これを書く。

- OS / Python version / `python.executable`
- package manager と実行した setup command
- PyTorch version、CUDA / ROCm / MPS 状態
- Transformers version、model `config.json` / `model_type` の確認結果
- TensorFlow version、GPU device の有無
- driver version、GPU 名、backend / index / extra の選択理由
- lock file を変えたか、変えたなら理由
- 実ワークロード smoke test の command と結果

## 失敗時の地図

- Torch から GPU が見えない: [references/failure-patterns.md](references/failure-patterns.md) の「PyTorch GPU が見えない」を読む。
- TensorFlow から GPU が見えない: 同ファイルの「TensorFlow GPU が見えない」を読む。
- Transformers が model を読めない: model directory の `config.json` と `model_type` を確認する。
- exact rebuild recipe が必要: [references/rebuild-playbook.md](references/rebuild-playbook.md) を package manager と backend に合わせて使う。

## Resources

- `scripts/check_ml_stack.py`
  Python / GPU command / PyTorch / Transformers / TensorFlow / model config の健康状態を text または JSON で出す。`--strict --require ...` で必須 stack を非ゼロ終了にできる。
- `references/rebuild-playbook.md`
  lock file からの再構築、backend 選択、runtime 分離、検証、最終記録の短い手順。
- `references/failure-patterns.md`
  wrong interpreter、PyTorch GPU、TensorFlow GPU、Transformers model loading、環境 drift の切り分け。

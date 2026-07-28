# Rebuild Playbook

## 目的

PyTorch、Transformers、TensorFlow の Python 環境を、最小変更で再構築し、実際に動くことを確認し、別マシンでも再現できるコマンド列を残す。

## 手順

### 1. プロジェクト制約を読む

- 指示ファイル、`pyproject.toml`、`requirements*.txt`、`environment.yml`、lock file、`.python-version` を読む。
- 対象 workload を動かす entrypoint を探す。notebook なら kernel の Python も確認する。
- cache、model、data、output、runtime の置き場所が project で決まっているか確認する。

### 2. 検証対象 Python を決める

- project `.venv`
- conda env
- notebook kernel
- service / worker runtime
- one-off isolated runtime

`python -c "import sys; print(sys.executable)"` で対象 Python を明示する。別環境で通った結果を流用しない。

### 3. lock から再現する

lock があり、既知の環境を戻す目的なら、まず lock から再構築する。

```bash
uv sync --locked
```

pip / requirements project:

```bash
python -m pip install -r requirements.txt
```

conda project:

```bash
conda env update -f environment.yml
```

### 4. 必要な stack だけ追加する

PyTorch GPU runtime を分離する例:

```bash
uv venv --python 3.12 .venv-torch
uv pip install --python .venv-torch/bin/python --torch-backend auto torch torchvision
```

Transformers:

```bash
uv pip install --python .venv-torch/bin/python transformers huggingface-hub
```

TensorFlow Linux / WSL GPU:

```bash
python -m pip install "tensorflow[and-cuda]"
```

TensorFlow CPU:

```bash
python -m pip install tensorflow
```

Rules:

- PyTorch backend / index を変えるときは、Torch 関連 package だけを動かす。
- `uv sync` / `uv add` に `--torch-backend` が効く前提で書かない。project workflow では `pyproject.toml` の index / sources を使うか、runtime 分離して `uv pip --torch-backend` を使う。
- TensorFlow は Python / OS / GPU 条件が PyTorch と違う。衝突しそうなら専用 venv に分ける。
- Transformers source install は、stable release が対象 model を読めないときだけ使う。

### 5. library stack を検証する

```bash
python scripts/check_ml_stack.py --json
```

必須 stack を指定する。

```bash
python scripts/check_ml_stack.py \
  --strict \
  --require torch-gpu \
  --require transformers \
  --model-dir /path/to/model
```

TensorFlow:

```bash
python scripts/check_ml_stack.py --strict --require tensorflow-gpu
```

### 6. 実 workload で検証する

- framework import だけで終わらせない。
- training / inference / evaluation / notebook / service の最小 command を 1 本走らせる。
- GPU を使う予定なら、framework が実際に GPU device を使ったことを記録する。

## 記録チェックリスト

- OS / Python version / `python.executable`
- package manager と setup command
- PyTorch version / CUDA or ROCm runtime / GPU availability
- Transformers version / model config result
- TensorFlow version / GPU device result
- driver version / GPU model
- backend / index / source install を選んだ理由
- lock file を変更したか
- smoke-test command と結果

## 判断規則

- 再現が目的なら lock file を優先する。
- 失敗している Python と修理している Python を一致させる。
- PyTorch、TensorFlow、serving runtime は必要なら venv を分ける。
- framework install 成功ではなく、最小 workload 成功を完了条件にする。

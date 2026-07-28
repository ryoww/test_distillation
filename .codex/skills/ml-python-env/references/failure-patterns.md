# Failure Patterns

## 間違った Python を見ている

最初に確認する。

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

よくある原因:

- shell の `python` と `uv run python` と notebook kernel が違う。
- service / worker は別 venv で動いている。
- 一時環境で package を入れたが、project lock に反映していない。
- Docker / WSL / host の境界をまたいで結果を混同している。

## PyTorch GPU が見えない

1. host 側で `nvidia-smi` または ROCm の確認 command を実行する。ここで失敗するなら Python package より driver / container / device visibility を見る。
2. 検証対象 Python で `torch.__version__`, `torch.version.cuda`, `torch.cuda.is_available()` を出す。
3. CPU wheel を入れていないか、backend / index が目的の CUDA / ROCm と合っているか確認する。
4. `CUDA_VISIBLE_DEVICES`、container runtime、scheduler の GPU assignment を確認する。
5. version pin を変える前に lock から再構築し、必要な場合だけ PyTorch 関連 package を動かす。

## TensorFlow GPU が見えない

1. 対象 OS / Python version が公式 TensorFlow GPU 条件に入っているか確認する。
2. Linux / WSL で NVIDIA GPU を使う場合、driver と CUDA/cuDNN 依存が wheel の想定と合うか確認する。
3. 検証対象 Python で `tf.config.list_physical_devices("GPU")` を出す。
4. native Windows や macOS では GPU support の扱いが制限されるため、CPU package、WSL2、Docker、専用 runtime のどれにするか決め直す。
5. PyTorch と同じ venv に入れて依存衝突が起きるなら、TensorFlow 専用 venv に分ける。

## Transformers がモデルを読めない

model directory を直接確認する。

1. `config.json` を読む。
2. `model_type` と `architectures` を記録する。
3. `AutoConfig.from_pretrained(..., trust_remote_code=True)` を検証対象 Python で実行する。

`model_type` を installed Transformers release が知らない場合は、新しい release を試す。source install は、release が必要な model support を持っていないと確認できたときだけ使う。

## Native package build が壊れる

- build isolation が原因か、system library 不足か、Python version 非対応かを分ける。
- `flash-attn`、`xformers`、`bitsandbytes`、`tensorflow-io`、動画 binding などは framework version と compiler / CUDA 条件が強い。
- まず prebuilt wheel で済む構成を探す。source build は最後にする。

## 一度動いた環境が再実行で壊れる

drift を疑う。

- lock file と実環境が違う。
- notebook kernel が古い。
- cache / model path が実行ごとに違う。
- dependency graph を変えたのに rebuild 手順を更新していない。

迷ったら、検証対象 Python を決め直し、lock から再構築し、`check_ml_stack.py --strict --require ...` と最小 workload を再実行する。

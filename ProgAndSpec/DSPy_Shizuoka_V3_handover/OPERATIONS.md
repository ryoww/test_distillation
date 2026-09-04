# DSPy Shizuoka V3 運用手順

このディレクトリはアプリケーションではなく、DSPy + GEPA で
`solve(instance)` の生成指示を最適化する実験一式です。
root の sequence distillation とは別ワークロードですが、OpenAI互換のローカルLLM
配信基盤を共有します。

## 構成

| 層 | Python / runtime | 役割 |
|---|---|---|
| root `.runtime/train` | Python 3.13 / Torch CUDA 13 | Agents-A1-4B のLoRA蒸留 |
| root `.runtime/vllm` または既存vLLM環境 | Python 3.13 / vLLM 0.20.0 | QwenをOpenAI互換APIで配信 |
| 本ディレクトリ `.venv` | Python 3.11 / DSPy 3.3.0b1 | GEPA学習、生成、評価 |

モデルをDSPyプロセスへ直接ロードしません。生成LMとreflection LMを別ポートで起動し、
HTTP経由で接続します。既定の役割とrevisionは
[`model_manifest.json`](model_manifest.json) に固定しています。

- generation: `Qwen/Qwen3.6-27B`、port 7501
- GEPA reflection: `Qwen/Qwen3.8-27B`、port 7502

## 1. DSPyクライアント環境

```bash
cd ProgAndSpec/DSPy_Shizuoka_V3_handover
uv sync --locked
uv run python scripts/preflight.py --offline
```

offline preflight は次を検査します。

- Python 3.11、DSPy 3.3.0b1、GEPA 0.1.1
- `data/problems/` の100問と必須フィールド
- Phase E / F の保存済みDSPyプログラムとdemos 2件
- `safe_exec` のspawn実行

100問中95問は数値目的値を選択できます。残り5問は数値目的が定義されていないため、
数値比較の対象外です。

## 2. ローカルQwenサーバー

このサーバーでは、両モデルが次の共有cacheにあります。

```bash
cd /home/yy-lab/test_distillation
export QWEN_HF_HOME=/home/yy-lab/test_DSPy/model/hf_home
export QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13
```

別ホストでは同じモデルrevisionを取得し、2変数だけ差し替えます。root側にvLLM環境を
新設する場合は、先に次を実行します。

```bash
uv run scripts/bootstrap_vllm_env.py
```

root から2つのterminalで起動します。H200を1枚ずつ使うため、事前に
`nvidia-smi` で両GPUが空いていることを確認します。

generation server:

```bash
HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_vllm.py \
  --vllm-env-dir "$QWEN_VLLM_ENV" \
  --model-path Qwen/Qwen3.6-27B \
  --model-revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --served-model-name Qwen/Qwen3.6-27B \
  --port 7501 \
  --max-model-len 131072 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization 0.85 \
  --no-enable-log-requests --disable-log-stats \
  --disable-uvicorn-access-log --generation-config vllm \
  --reasoning-parser qwen3
```

reflection server:

```bash
HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
uv run scripts/serve_vllm.py \
  --vllm-env-dir "$QWEN_VLLM_ENV" \
  --model-path Qwen/Qwen3.8-27B \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name Qwen/Qwen3.8-27B \
  --port 7502 \
  --max-model-len 131072 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --no-enable-log-requests --disable-log-stats \
  --disable-uvicorn-access-log --generation-config vllm \
  --reasoning-parser qwen3
```

初回は重みロード後に `torch.compile` とCUDA graph warmupが走ります。この環境では
両モデルをGPU 1で個別起動し、`/v1/models` とDSPy chat completionを確認済みです。
Qwen3.6-27Bは起動完了まで約4分36秒、Qwen3.8-27Bは約4分半で、どちらも
モデル重みのGPU使用量は51.1 GiBでした。

### thinkingの扱い

Qwen3系のchat templateは`enable_thinking`が未指定なら思考を開きます。本実験は
指定を送らないので、**thinkingは有効のまま**です。切る場合は
`train_gepa_v3.py --no-thinking` か `DSPY_GENERATION_ENABLE_THINKING=0` を使います。

`--reasoning-parser qwen3` を付けると、思考文は`content`ではなく
`reasoning_content`へ分離されます。付けないと`<think>`が本文に混ざったまま
DSPyのJSONAdapterへ渡り、構造化出力のパースを妨げます。

出力枠は思考文と最終出力の合計です。既定を32,768 tokensにしており、上限は
`max_model_len` から入力長を引いた値です（Phase Eの先頭問題で入力8,905 tokens）。
timeoutは1800秒を維持します。

保存済みPhase Eプログラムの先頭問題は、Qwen3.6 tokenizerで入力8,905 tokensでした。
context長は入力と最大出力の合計を収める必要があるため、比較時は131072を指定します。
MTPは`num_speculative_tokens=1`で固定し、推論ログはquietオプションで抑制します。

実 workload smokeでは、保存済みPhase Eと`prob_001`から1,731文字の
`solve(instance)`を生成しました。AST検査と`safe_run`を通過し、
`objective_value`、`optimal_sequence`、`note`を返しています。

サーバー起動後、handover側からonline preflightを実行します。

```bash
cd ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/preflight.py
```

1枚しか空いていない場合は、両方の `DSPY_*_MODEL` と `DSPY_*_API_BASE` を同じ
起動済みモデルへ向ければ機能確認はできます。ただし、元実験の2モデル構成とは
異なる条件です。

## 3. 保存済みプログラムの再評価

参照あり:

```bash
uv run python train_gepa_v3.py --eval-only --run-name phaseE-recheck
```

参照フリー:

```bash
uv run python train_gepa_v3.py --eval-only --no-reference \
  --run-name phaseF-recheck
```

出力は `outputs/<run-name>/` に保存します。引き継ぎ時の
`compiled_program_v3_gepa_*.json`、`evaluation_results_v3_gepa_*.json`、
`data/best_known*.jsonl` は上書きしません。

## 4. Qwen3.6 / Qwen3.8の同条件スコア比較

port 7501と7502の両サーバーを起動した状態で、次を実行します。

```bash
uv run python scripts/score_qwen_models.py --run-name phaseE-qwen-comparison
```

既定では保存済みPhase Eプログラム、seed 42、temperature 0.0を使い、同じテスト20問を
Qwen3.6-27B、Qwen3.8-27Bの順に評価します。学習やreflectionは実行しません。
各モデルの生の評価結果と集計結果を次へ保存します。

```text
outputs/model_scores/phaseE-qwen-comparison/
├── qwen3_6_27b/evaluation_results_v3_gepa_phaseE.json
├── qwen3_8_27b/evaluation_results_v3_gepa_phaseE.json
└── comparison.json
```

`comparison.json`には平均スコア、valid件数と率、status内訳、reference到達件数、順位、
モデルrevision、プログラムとデータのSHA256、実際のテスト問題IDを記録します。
問題IDが両モデルで一致しなければ比較不能として終了します。同点には同じ順位を付けます。
実行前に条件だけ確認する場合は、ファイルを作らないdry-runを使います。

```bash
uv run python scripts/score_qwen_models.py --dry-run
```

GPU容量に余裕がある場合だけ `--parallel` で同時評価できます。別のモデル名、revision、
endpointは `--qwen36-*` / `--qwen38-*` で上書きできます。revisionは期待値の記録であり、
OpenAI互換APIからロード済みrevisionを検証するものではありません。

この比較は同一プログラムでの推論スコアです。Phase Eプログラムは元のQwen3.6生成・
Qwen3.8 reflection構成で最適化されているため、モデル固有に再最適化した最高性能の比較では
ありません。

`--no-reference` は保存済みPhase Fプログラムを再現評価するためのオプションです。
同梱のPhase F成果物は歴史的成果物であり、修正後に再学習したPhase Fプログラムとの
比較には使わないでください。

### 改善前後プロンプト × 2モデルの100問比較

GPUを1枚だけ使う場合は、Qwen3.6とQwen3.8を同じGPU 1で順番に起動します。
各モデルについて、現在の未コンパイル`AlgorithmGenerator`を改善前、保存済みPhase Eを
改善後として、100問を評価します。
`--only-model`を指定すると、そのモデルの改善前後2条件だけを実行します。
`--parallel`は同じモデルへ4つの子評価（改善前後 × 2 shard）を同時に送ります。
GPUメモリに余裕が必要なため、vLLMの割当は`0.85`に固定します。

まず、次の環境変数を設定した端末を用意します。

```bash
cd /home/yy-lab/test_distillation
export QWEN_HF_HOME=/home/yy-lab/test_DSPy/model/hf_home
export QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13
```

#### Qwen3.6の部分実行

端末AでQwen3.6をGPU 1へ割り当てます。

```bash
HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 VLLM_DEEP_GEMM_WARMUP=skip \
uv run scripts/serve_vllm.py \
  --vllm-env-dir "$QWEN_VLLM_ENV" \
  --model-path Qwen/Qwen3.6-27B \
  --model-revision 6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
  --served-model-name Qwen/Qwen3.6-27B \
  --port 7501 \
  --max-model-len 131072 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --no-enable-log-requests --disable-log-stats \
  --disable-uvicorn-access-log --generation-config vllm \
  --reasoning-parser qwen3
```

端末Bで、サーバーの起動完了後にQwen3.6の改善前後を評価します。

```bash
cd /home/yy-lab/test_distillation/ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/compare_prompt_models.py \
  --only-model qwen3_6_27b \
  --run-name prompt-model-qwen36 \
  --parallel --shards 2
```

評価が終わったら、端末Aで`Ctrl-C`を入力してQwen3.6を停止します。
`nvidia-smi`でGPU 1のプロセスがなくなったことを確認してから、Qwen3.8を起動します。
この部分実行は2条件を保存し、`run_complete=true`、`partial=true`、
`comparable=false`、`effects={}`になります。

#### Qwen3.8の部分実行

Qwen3.6を停止した後、端末AでQwen3.8を同じGPU 1へ割り当てます。

```bash
HF_HOME="$QWEN_HF_HOME" CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_vllm.py \
  --vllm-env-dir "$QWEN_VLLM_ENV" \
  --model-path Qwen/Qwen3.8-27B \
  --model-revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  --served-model-name Qwen/Qwen3.8-27B \
  --port 7502 \
  --max-model-len 131072 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --gpu-memory-utilization 0.85 \
  --no-enable-log-requests --disable-log-stats \
  --disable-uvicorn-access-log --generation-config vllm \
  --reasoning-parser qwen3
```

端末Bで、サーバーの起動完了後にQwen3.8の改善前後を評価します。

```bash
cd /home/yy-lab/test_distillation/ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/compare_prompt_models.py \
  --only-model qwen3_8_27b \
  --run-name prompt-model-qwen38 \
  --parallel --shards 2
```

評価が終わったら、端末AでQwen3.8を停止します。
`nvidia-smi`でGPU 1のプロセスがなくなったことを確認し、GPU 1を解放します。

#### 2つの部分成果物を統合

両モデルの評価が成功したら、次のコマンドで4条件を統合します。

```bash
cd /home/yy-lab/test_distillation/ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/compare_prompt_models.py \
  --output-dir outputs/prompt_model_comparisons \
  --run-name prompt-model-full100-merged \
  --merge-runs \
  outputs/prompt_model_comparisons/prompt-model-qwen36 \
  outputs/prompt_model_comparisons/prompt-model-qwen38
```

統合時には、seed、temperature、max tokens、データマニフェスト、問題ID、
改善前後プログラムのSHA256が一致するかを検査します。
不一致の成果物は統合しません。
成功時の`factorial_comparison.json`には、改善前後 × Qwen3.6/Qwen3.8の4条件と、
`untouched40`、`legacy_test20`、`train40`、`all100`ごとの効果を保存します。

```bash
uv run python scripts/compare_prompt_models.py --dry-run
```

dry-runはサーバーへ接続せず、選択したモデルの子コマンドだけを表示します。

#### 保存済みコードの再採点

feasibilityチェッカーを追加・変更するとスコアの意味が変わるため、過去の
`factorial_comparison.json`と新しい結果は直接比較できません。評価結果JSONには
各問題の生成コードが保存されているので、LLMもGPUも使わずに再採点できます。

```bash
cd /home/yy-lab/test_distillation/ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/rescore_with_checkers.py \
  outputs/prompt_model_comparisons/prompt-model-qwen36 \
  outputs/prompt_model_comparisons/prompt-model-qwen38 \
  --subsets-from outputs/prompt_model_comparisons/prompt-model-full100-merged/factorial_comparison.json \
  --output outputs/prompt_model_comparisons/rescored.json
```

各shardディレクトリを1つの評価単位として扱い、元実行と同じ順序で
`BestKnownRegistry`を参照値からseedし直します。再実行したコストが保存値と
食い違う問題は`cost_mismatches`へ記録します。ソルバーの時間制限や乱数を含む
解法が該当するので、件数が多い場合は結果の解釈に注意してください。

生成段階で落ちた`gen_error`のレコードにはコードが無いため、そのまま引き継ぎます。

2026-09-01 に20 core_type 分のチェッカーを追加して4条件を再採点した結果と、
その過程で見つかった欠陥は `RESCORE_REPORT.md` にまとめてあります。

## 5. GEPA再学習

参照あり:

```bash
uv run python train_gepa_v3.py --breadth 6 --depth 8 \
  --run-name phaseE-train
```

参照フリー:

```bash
uv run python train_gepa_v3.py --no-reference --breadth 6 --depth 8 \
  --run-name phaseF-train
```

参照フリーモードでは、reference値と参照解の具体値をrequirementへ入れません。
返却スキーマとして必要なトップレベルキーと型だけを残します。未登録scorerの目的値も
候補解だけから正規化し、参照値は報酬・セルフベースラインに使用しません。

別endpointを使う場合はCLIまたは環境変数で指定できます。

```bash
export DSPY_GENERATION_MODEL=Qwen/Qwen3.6-27B
export DSPY_GENERATION_API_BASE=http://127.0.0.1:7501/v1
export DSPY_REFLECTION_MODEL=Qwen/Qwen3.8-27B
export DSPY_REFLECTION_API_BASE=http://127.0.0.1:7502/v1
```

認証が必要なendpointでは、credentialをコマンドへ書かず
`DSPY_GENERATION_API_KEY` / `DSPY_REFLECTION_API_KEY` から渡します。

## 6. 検証

```bash
uv run pytest -q
uv run ruff check --select E9,F,I src scripts train_gepa_v3.py tests
uv run python scripts/preflight.py --offline
```

学習済みPhase E/Fプログラムを同梱しない再学習用パッケージでは、成果物検査だけを
`uv run python scripts/preflight.py --offline --skip-programs` で省略できます。

保存済み評価結果は、旧メトリックの最大化誤判定と参照フリーprompt漏洩を含む
歴史的成果物です。新しい実験結果との直接比較には、修正版で再評価した結果を使います。

## 7. 雛形からの問題生成（検証済みホールドアウトの追加）

既存の100問は前部門の生成器で作られ、生成器は同梱されていません。参照解も最適とは
限らず、`beat_reference` が起きます。`src/datagen/` は既存問題を雛形にして、
同じ形状の instance を乱数で作り、厳密ソルバーで参照解を付けます。

- 問題文と requirements は雛形の文章をそのまま使うため、文章に書かれている件数・容量・
  予算は雛形の値を保ち、それ以外の数値だけを引き直します。
- 参照解は既存の feasibility チェッカーを `verified` かつ違反ゼロで通ったものだけを
  書き出します。通らなければ生成自体が失敗します。
- 各ソルバーは雛形 instance で元の参照値を再現することをテストで固定しています。
- 生成 instance は世界に存在しなかったものなので、モデルが答えを記憶している可能性が
  ありません。GEPA の候補選択にも demo にも使っていない集合として扱えます。

```bash
uv run python scripts/generate_problems.py --list
uv run python scripts/generate_problems.py --per-template 5 --seed 20260904
```

出力は `data/problems_generated/prob_1001.json` 以降と `manifest.json` です。
既存の `data/problems/` と `data_manifest.json` は変更しません。乱数は
`seed:雛形ID:通番` で決まり、CP-SAT は 1 worker 固定なので、同じ引数なら別プロセスでも
同じファイルになります。出力先に前回の `prob_*.json` が残っている場合は止まるので、
入れ替えるときは `--force` を付けます。

2026-09-04 時点の雛形は28問で、27 core_type のうち24を覆います。`--per-template 5` で
140問を生成し、参照解をそのまま返す `solve()` を `metrics_v3` に通すと140問すべてが
`exact_match` になります。雛形の一覧は `--list` で表示できます。

雛形にできなかった core_type は3つです。

| core_type | 問題 | 理由 |
|---|---|---|
| スケジューリング_確率最適化 | prob_018 | 参照解に数値の目的値がなく、期待コスト率 7.82 を再現する式も特定できない |
| 配送・輸送_確率最適化 | prob_026 | 問題文は初期顧客5人だが instance は4人で、参照解も近似解 |
| 生産・在庫計画_確率最適化 | prob_080 | 問題文が価格・需要分布のすべての数値を固定しており、乱数で変える余地がない |

同梱参照解が厳密最適でない雛形は `shipped_reference_optimal=False` で登録し、テストは
参照値の再現ではなく「厳密解が参照値より良い」ことを確認します。該当は prob_021（基本
CVRP）で、同梱参照解は容量15の車両1台に需要25を積む単一経路（総距離245）でした。
厳密最適は約144です。配送・輸送_混合整数計画 の他の問題も note が「ortools routing近似解」
なので、この core_type の `beat_reference` は参照値が近似解であることを踏まえて読みます。

prob_082（キャッシュフローマッチング）は問題文にも instance にも債券の額面がなく、雛形の
参照解は額面 14 以上なら再現できます。雛形は市場慣行の額面 100 を前提に参照解を作るので、
この雛形の `exact_match` 率はモデルが同じ前提を置くかどうかに左右されます。

**生成集合は最終ホールドアウトとして扱います。** GEPA の train/val、demo、候補選択、
停止判断のどれにも使いません。評価に使う集合は seed とテンプレート一覧を manifest に
記録し、比較する条件間で同じ manifest を使います。

生成した集合を保存済みプログラムで評価するには、全問をテスト側に置きます。

```bash
uv run python train_gepa_v3.py --eval-only --data-dir data/problems_generated \
  --n-train 0 --n-test 60 --run-name phaseE-generated60
```

雛形を増やすときは `src/datagen/templates.py` に `(generate, solve)` の組を登録し、
`tests/test_datagen.py` が雛形の参照値を再現できることと、生成問題がチェッカーを
通ることを自動で確認します。

## 8. 既知の限界

- 95問は数値目的を選択できますが、5問は数値目的がありません。
- 旧チェッカー（feasibility.py 直登録の8 core_type）は `jobs` や `customers` キーのない
  instance を検証せずに feasible と返します。構造検査だけで目的値の再計算はしません。
- 配送・輸送 系の同梱参照解は近似解で、prob_021 は容量制約に違反しています。
  この core_type の参照値は上限の目安にすぎません。
- feasibility checker未登録の問題は `unverified` とし、参照一致・参照超えに数えません。
- `safe_exec` は既知のdunder/import迂回を拒否しますが、Python sandboxを安全境界とは
  みなしません。信頼できない生成コードは、権限を落としたコンテナ等で実行してください。
- 既定のbreadth 6 / depth 8は長時間実行です。接続確認では小さい値を使います。

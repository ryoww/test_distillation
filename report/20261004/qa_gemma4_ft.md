# Gemma 4 12B の特化学習（LoRA / FFT）質疑応答

下沢 亮太郎（2026-10-05 作成）

これまで集めた検証済みの正解コードで、Gemma 4 12B を LoRA と全層学習（FFT）の 2 通りで学習しました。
その内容と結果、聞かれそうな疑問への答えをまとめます。数値は
`ProgAndSpec/DSPy_Shizuoka_V3_handover/outputs/prompt_model_comparisons/gemma4-ft-20261004-*` の評価結果を
`scripts/summarize_solver_runs.py` で数えたものです。

## 0. ひとことで（専門外の人への説明）

配送ルートや勤務表のような「最適化問題」を解くプログラムを書く AI を、手元の GPU で動く中くらいの大きさのモデル
（Gemma 4 12B、120 億パラメータ）で作りました。

作り方は「お手本を見せて覚えさせる」です。お手本は、もっと大きな AI（Claude Opus・Fable や Qwen）が書いたプログラムの
うち、**実際に動かして正しい答えが出ると確かめたものだけ**を使いました。約 1 万 6 千組の「問題文 → 正解プログラム」です。

覚えさせ方は 2 通り試しました。

- **LoRA**: モデル本体はそのままにして、薄い追加部品（全体の 1%）だけを学習する。「教科書に付箋を貼る」やり方。
  GPU 1 枚で 5 時間半。
- **FFT（全層学習）**: モデルの 120 億個の数値すべてを書き換える。「教科書そのものを書き直す」やり方。
  GPU 2 枚で 6 時間。

どちらも大きく伸び、FFT がわずかに上でした（差は統計的に有意ではない）。

| 問題 | 学習前 | LoRA | FFT |
|---|---:|---:|---:|
| 学習で見た種類の小〜中規模問題（445 問） | 345 | 442 | **445** |
| 大規模問題（120 問、厳密解が出せない規模） | 0 | 88 | **93** |
| 大規模の元問題（28 問） | 3 | 12 | **14** |

大規模問題では、FFT の 81 問が「お手本の基準解（主に Claude Fable が 10 分かけて出した解）と同等以上」でした。
12B のモデルが、思考なしで直接コードを書いてこの水準に届いています。

## 1. 何をしたか

### 1.1 学習データ（`data/sft_merged`、16,348 行 → 長さ制限後 16,228 行）

| 出典 | 中身 | 学習に入れた行数 |
|---|---|---:|
| 雛形 89 問（`data/sft`） | 89 雛形 × 40 instance に、Fable・Qwen・Gemma・Ministral・手書きの教師コード 412 本を当て、新 instance で正解（参照解と一致または上回る）を確かめた 15,088 対 | 15,088 |
| 大規模問題（`data/sft_hard_opus`） | 大規模 20 種別の生成 instance を Opus 5.5 に「書いて試して直す」形で解かせ、検証器で違反 0・参照解から +10% 以内だった 630 対（19 種別） | 630 × 2 = 1,260 |
| 長さ超過で除外 | ポートフォリオ 2 種別は問題文に収益率の行列がそのまま入り 36k〜42k token。上限 16,384 token を超えて落ちた | −120 |

- 入力は system = 既定の指示文、user = 参照値を含まない問題文、出力 = `solve()` のコードだけ（思考文なし）。
- 統合は `scripts/merge_sft_datasets.py --input templates=data/sft --input hard_opus=data/sft_hard_opus*2`。

### 1.2 学習の設定

| 項目 | LoRA | FFT（全層学習） |
|---|---|---|
| 土台 | Gemma 4 12B（`google/gemma-4-12B-it`） | 同じ |
| 更新するパラメータ | 1.31 億（全体の 1.1%、言語部の全線形層に rank 32 / α 64） | 119.6 億（全部） |
| 重みの精度 | bf16（土台は凍結） | fp32 の主重み、計算は bf16 |
| 学習率 / optimizer | 1e-4 / AdamW | 1e-5 / 8-bit AdamW（bitsandbytes） |
| epoch / step | 1 / 2,029（実効バッチ 8） | 同じ |
| 最大長 | 16,384 token | 同じ |
| GPU | H200 1 枚（約 82 GB） | H200 2 枚に層を分割（約 110 GB + 101 GB） |
| 所要 | 5 時間 38 分 | 6 時間 7 分 |
| 学習損失（平均）/ 検証損失 | 0.051 / 0.0024 | 0.039 / 0.0023 |

学習・推論とも chat template に `enable_thinking=true` を渡す（22 章と同じ。学習で見た列と推論の入力を token 単位で
一致させるため）。

### 1.3 評価

素の Gemma 4 12B（思考なし）、LoRA、FFT を同じ 3 集合で解かせた。vLLM 0.28、温度 0、出力枠 8,192 token、実行上限 900 秒。

| 集合 | 問題数 | 学習との関係 |
|---|---:|---|
| 雛形テスト `data/sft/problems_test` | 445 | 学習した 89 雛形の未知 instance（seed 別） |
| 大規模生成テスト `data/problems_hard_gen/test` | 120 | 学習した大規模 20 種別の未知 instance（20 種別 × 6） |
| 大規模の元問題 `data/problems_hard` | 28 | 生成 instance の元になった問題。種別は学習済み、instance は未学習 |

「正解」は学習データを選んだときと同じく、検証器が解を読めて違反 0、参照解からの gap が +10% 以内。
「参照以上」は参照解と一致または上回ったもの。

## 2. 結果

### 2.1 3 集合での正解数

右端は素の Gemma と比べた問題単位の入れ替わり（両方正解 / 素の Gemma だけ正解 / 学習後だけ正解）。

**雛形テスト 445 問**

| 条件 | 正解 | 参照以上 | 平均スコア | 平均出力 token | 入れ替わり |
|---|---:|---:|---:|---:|---|
| 素の Gemma 4 12B（思考なし） | 345 | 337 | 1.166 | 1,253 | — |
| LoRA | 442 | 438 | 1.502 | 821 | 345 / 0 / 97 |
| **FFT** | **445** | **443** | **1.513** | 754 | 345 / 0 / 100 |
| FFT 1 回目（bf16 重み、失敗。4 節） | 352 | 351 | 1.188 | 978 | 307 / 38 / 45 |

**大規模生成テスト 120 問**

| 条件 | 正解 | 参照以上 | 可行 | 平均スコア | 平均出力 token |
|---|---:|---:|---:|---:|---:|
| 素の Gemma 4 12B | 0 | 0 | 12 | 0.148 | 2,770 |
| LoRA | 88 | 74 | 95 | 1.164 | 3,960 |
| **FFT** | **93** | **81** | **99** | **1.250** | 3,667 |
| FFT 1 回目（失敗） | 0 | 0 | 1 | −0.369 | 6,324 |

**大規模の元問題 28 問**

| 条件 | 正解 | 参照以上 |
|---|---:|---:|
| 素の Gemma 4 12B | 3 | 2 |
| LoRA | 12 | 12 |
| **FFT** | **14** | **14** |
| FFT 1 回目（失敗） | 0 | 0 |

参考（24 章。条件が少し違う: 出力枠 12k・実行 600 秒・指標は「可行（厳密）」）: Qwen3.6 27B 3 問、Qwen3.8 27B 11 問、
Claude Fable 22 問。12B の LoRA / FFT が 27B の Qwen3.8 と同じかそれ以上の水準に来た。

**LoRA と FFT の差**: 問題単位で FFT だけ正解 / LoRA だけ正解は、雛形 3 / 0、大規模生成 8 / 3、元問題 3 / 1。
符号検定の p は 0.25、0.23、0.63 で、どれも有意ではない。

### 2.2 大規模生成テストを種別で見ると（各 6 問）

| 種別 | LoRA | FFT |
|---|---:|---:|
| clsp, crew_pairing_seniority, cutting_2d, facility_2ech, facility_multi, fjsp_setup, mcnd_surv, pdptw, prp_tw, role_roster | 各 6 | 各 6 |
| cutting_1d | 5 | 6 |
| prp | 3 | 6 |
| vrptw_md | 3 | 6 |
| fjsp | 3 | 4 |
| facility_robust, mcnd | 各 4 | 各 4 |
| nurse_roster | 6 | 3 |
| **crew_pairing** | 0 | 0 |
| **portfolio, portfolio_cvar** | 0 | 0 |

学習データにある 17 種別に限ると、LoRA 88 / 102、FFT 93 / 102。0 問の 3 種別は学習データに入っていない（3 節 Q7・Q8）。
学習済み種別での失敗は、長いコードの括弧の対応崩れ（構文エラー、FFT 16 件・LoRA 14 件）と、参照より 10% 超悪い解
（FFT 6 件・LoRA 7 件）がほとんど。出力枠 8,192 token に達したのは FFT 3 件、LoRA 11 件。

## 3. 聞かれそうな疑問

**Q1. 「もろもろのデータ」とは具体的に何か。**
2 つです。雛形 89 問を 40 instance ずつ生成し、教師コード 412 本を当てて正解を確かめた 15,088 対（RESCORE_REPORT 23 章）と、
大規模 20 種別の生成 instance に Opus 5.5 が書いて検証器を通した 630 対（同 27 章）。どちらも「新しい instance で実際に
動かして正解した」ものだけで、モデルの出力を確かめずに学習に使ったものはありません。

**Q2. LoRA と FFT は何が違うのか。どちらを使えばよいか。**
LoRA は元の重みを凍結し、各線形層に小さな行列（rank 32）を足して、その 1.31 億パラメータだけを学習します。FFT は 119.6 億
パラメータすべてを更新します。今回は FFT が 3 集合すべてで少し上でしたが、差は有意ではありません。一方、FFT は GPU 2 枚・
fp32 の主重みが要り、成果物も 23 GB の完全なモデルになります（LoRA の adapter は 0.5 GB）。手軽さで LoRA、最後の数問を取りに
行くなら FFT、という使い分けです。どちらも見たことのない種別での劣化（Q4）はまだ測っていません。

**Q3. 正解の基準「参照解から +10% 以内」は甘くないか。**
大規模問題は厳密解が出せない規模なので、参照解は「既存の教師コード（主に Fable が 600 秒で出した解）の最良」で、最適解では
ありません。参照解ちょうどを要求すると良い近似解まで落ちるため、学習データを選んだときと同じ基準にしました。「参照以上」の
列も併記していて、FFT は大規模生成テストで 81 問が参照解と同等以上です。雛形 445 問は厳密解が参照なので、正解と参照以上の
差は 2〜4 問だけです。

**Q4. テストは学習データと重なっていないか。**
instance は重なっていません（雛形は seed を、大規模は train / validation / test の分割を分けて生成）。ただし種別は学習と同じ
なので、測っているのは「見た種別の新しい instance を解けるか」です。見たことのない種別への汎化は今回は測っておらず、22 章では
12B の LoRA が雛形外で素のモデルより悪化しました（48 問 → 28 問）。大規模の元問題 28 問は生成 instance の元で、問題文の書きぶりは
学習データとほぼ同じです。

**Q5. GLM 5.3 Flash の解答は入っていないのか。**
入っていません。llama.cpp で配信した GLM（320B / 活性 18B）は約 14 token/秒で、1 問目に 31 分かけても思考が 25,000 token を
超えて終わらず、試行 6 回で 4 時間以上かかる見込みだったため後回しにしました。GEPA の後に流すよう予約してあり
（Slurm job 802）、正解が十分取れるなら同じ手順（`build_hard_sft_dataset.py --runs glm=...`）で加えて学習し直せます。

**Q6. 大規模問題のデータを 2 回繰り返したのはなぜか。過学習しないか。**
件数で 15,088 対 630 と 4% しかなく、そのままだと雛形問題に埋もれるためです。大規模のコードは長く（中央値 3,400 token、雛形は
約 900 token）、2 回入れると損失を計算する token の約 25% になります。1 epoch なので、各大規模対を見るのは 2 回です。ただし 630 対の
中身は 21 種類のコードを別 instance で再生したもので、コードの多様性は小さい。未知 instance の大規模テストで 9 割前後解けている
ので、instance への過学習は見えていません。

**Q7. ポートフォリオ 2 種別はなぜ 0 問なのか。**
問題文に収益率の行列がそのまま入っていて 36,000〜42,000 token あり、学習の最大長 16,384 token を超えて学習データから落ちました
（60 対 × 2 回 = 120 行）。最大長を 45k にすると FFT のメモリが足りません。問題文から行列を外して instance から読む形に書き換えるか、
LoRA だけ長い最大長で学習するのが対策です。

**Q8. 乗務員ペアリングはなぜ 0 問なのか。**
学習データが 0 件だからです。元問題 302 と 312 で規則が違う（各便ちょうど 1 回か、1 回以上か）のに、検証器が 302 系にも重複を
許しているため、302 系で問題文どおりに解いた Opus の解が参照より 45〜80% 悪いと判定され、正解から外れました（27.3 節）。
検証器を core_type で分けて参照解を付け直せば学習に入れられます。

**Q9. system prompt の既定の指示文には「インスタンスは小さい、30 秒以内」とあるが、大規模問題と矛盾しないか。**
矛盾しています。今回は雛形問題と同じ指示文に統一しました（学習と推論で同じ文字列なら、モデルは指示文より問題文と正解コードの
対応から学ぶため）。大規模問題の問題文には実行上限 1,800 秒が書かれていて、学習したコードは 300 秒前後の時間制限付きソルバー
です。大規模用の指示文（`prompts/seed_instruction_hard.md`）に差し替えて学習し直す選択肢はあります。

**Q10. 思考（reasoning）はさせているのか。**
させていません。学習データの出力はコードだけで、推論も同じ形で出させます。Gemma のテンプレートに `enable_thinking=true` を
渡していますが、これは学習時と推論時の描画（token 列）を一致させるためで、モデルは思考を書かずに直接コードを出します（22 章）。

**Q11. 残った失敗は何が原因か。もっと伸ばせるか。**
大規模生成テストで FFT が外した 27 問のうち 18 問は学習データにない 3 種別（Q7、Q8）。残る 9 問は、長いコードの括弧の対応崩れ
（構文エラー）と、参照より 10% 超悪い解です。伸ばし方は、(1) 欠けている 3 種別を学習に入れる、(2) 構文エラーなら 1 回書き直させる
（25 章の修復ループ）、(3) 出力枠を 16k に広げる（LoRA は 11 件が 8,192 token で打ち切られた）、の順に効くと見ています。

**Q12. 素の Gemma が大規模問題で 0 問なのはなぜか。**
120 問の内訳は、制約の一部違反 72、実行時エラー 15、検証器が読めない形 14、制約違反 6、解の形の不正 1、可行だが参照より
10% 超悪い 12 でした。大規模の制約をすべて満たす解を出すこと自体ができていません。学習データのコードは「まず可行解を作り、
時間制限付きのソルバーか局所探索で改善する」形で、学習後のモデルはその型を身につけています。

**Q13. FFT の 1 回目はなぜ失敗したのか。**
4 節にまとめました。重みを bf16 のまま更新したため、学習率 1e-5 の小さな更新が丸めで消えたのが原因です。

**Q14. 学習したモデルはどこにあるか。**
ホームのディスクが割当て 1,024 GB に対して 969 GB 使用中だったため、大きいファイルは `/var/tmp/yy-lab-ft/`（別ディスク、
空き約 400 GB）に置きました。`/var/tmp` は 30 日触らないと消える場所なので、残すものは容量を空けてから移してください。

| もの | 場所 | 大きさ |
|---|---|---:|
| LoRA adapter | `outputs/gemma4-12b-merged-lora-20261004/adapter` | 0.5 GB |
| LoRA 焼き込み済み（vLLM で配信可） | `/var/tmp/yy-lab-ft/gemma4-12b-merged-lora-20261004-merged` | 23 GB |
| FFT（bf16 で保存、vLLM で配信可） | `/var/tmp/yy-lab-ft/gemma4-12b-merged-fft-fp32-20261004/adapter` | 23 GB |
| FFT の途中保存（fp32 + optimizer、2 個） | `/var/tmp/yy-lab-ft/gemma4-12b-merged-fft-fp32-20261004/checkpoints` | 146 GB |
| FFT 1 回目（失敗、比較用） | `/var/tmp/yy-lab-ft/gemma4-12b-merged-fft-20261004/adapter` | 23 GB |

**Q15. 次に何をするか。**
1. 欠けている 3 種別（乗務員ペアリング、ポートフォリオ 2 種）を学習データに入れる（検証器の修正、問題文の短縮）。
2. 修復ループと出力枠 16k で測り直す。
3. GLM の解答を加え、教師の多様性を上げる。
4. 学習で見ない種別を分けて、汎化の劣化を測る（22 章の懸念。FFT のほうが元の能力を壊しやすいと言われるので特に）。

## 4. 補足: FFT 1 回目の失敗と原因

**症状**: bf16 の重みのまま 8-bit paged AdamW で全層学習した 1 回目（GPU 1 枚）は、検証損失 0.0305（やり直し後は 0.0023）に
留まった。大規模問題では 120 問中 48 問が出力枠を使い切り（問題文のデータをコードに書き写し続ける）、53 問が構文エラー。
引用符の閉じ忘れなど文字単位の崩れが目立ち、雛形 445 問でも素の Gemma とほぼ同じ（352 問）だった。

**原因**: 学習率 1e-5 で Adam が出す 1 回の更新量はおよそ 1e-5 で、bf16 で表した重み（大きさ 0.01〜0.05）の丸め幅
（約 6e-5〜2e-4）より小さい。重みを bf16 のまま持つと更新の大半が丸めで消える。勾配のノルムも学習中に 500 を超える場面が
何度もあった（やり直し後は 2〜3）。

**やり直し**: 重みを fp32 で持ち（主重み）、計算だけ bf16 で行う標準的な混合精度に変えた。fp32 の 12B は 1 枚に載らないので
層を GPU 2 枚に分けて置く。埋め込みと出力層（語彙 26 万）が GPU 0 に乗り、長い系列では logits だけで 12 GB を超えるため、
GPU 0 に置く重みを 16 GiB に絞った。保存時に bf16 へ戻す（vLLM は fp32 のモデルを fp16 で配信し、Gemma は fp16 で溢れるため）。
最長の系列で 2 step の試験を通してから本番を流した。

## 5. 再現

```bash
cd ProgAndSpec/DSPy_Shizuoka_V3_handover
export HF_HOME=/home/yy-lab/test_DSPy/model/hf_home HF_HUB_OFFLINE=1
uv run python scripts/merge_sft_datasets.py --input templates=data/sft \
  --input hard_opus=data/sft_hard_opus*2 --output-dir data/sft_merged
COMMON='--model google/gemma-4-12B-it --model-revision main --chat-template-kwargs {"enable_thinking":true} --save-steps 500'
# LoRA（GPU 1 枚）
RUN_NAME=gemma4-12b-merged-lora-20261004 DATASET_DIR=$PWD/data/sft_merged EPOCHS=1 LORA_R=32 LR=1e-4 MAX_LENGTH=16384 \
  EXTRA_ARGS="$COMMON" sbatch --time=12:00:00 --export=ALL scripts/slurm_train_solver.sbatch
# FFT（GPU 2 枚、fp32 主重み）
RUN_NAME=gemma4-12b-merged-fft-fp32-20261004 DATASET_DIR=$PWD/data/sft_merged EPOCHS=1 LR=1e-5 MAX_LENGTH=16384 \
  EXTRA_ARGS="$COMMON --full-finetune --fp32-weights --model-parallel --max-memory {\"0\":\"16GiB\",\"1\":\"60GiB\"} --optim adamw_bnb_8bit --output-dir /var/tmp/yy-lab-ft/gemma4-12b-merged-fft-fp32-20261004" \
  sbatch --gres=gpu:2 --cpus-per-task=16 --mem=200G --time=24:00:00 --export=ALL scripts/slurm_train_solver.sbatch
# 評価（素の Gemma は EXTRA_BODY を enable_thinking=false、MODEL_PATH=google/gemma-4-12B-it）
export QWEN_VLLM_ENV=/home/yy-lab/test_DSPy/.runtime/vllm/vllm-cu13 VLLM_ENV=$PWD/../../.runtime/vllm-0.28 MODEL_HF_HOME=$HF_HOME
RUN_NAME=gemma4-ft-20261004 DATA_DIRS="data/sft/problems_test data/problems_hard_gen/test data/problems_hard" \
  MAX_TOKENS=8192 MAX_MODEL_LEN=53248 EXEC_TIMEOUT=900 EVAL_TIMEOUT=1800 SERVED=gemma4-12b REASONING_PARSER=gemma4 \
  EXTRA_BODY='{"chat_template_kwargs": {"enable_thinking": true}}' LABEL=sft_gemma4_12b_fft_fp32 \
  MODEL_PATH=/var/tmp/yy-lab-ft/gemma4-12b-merged-fft-fp32-20261004/adapter \
  sbatch --time=08:00:00 --export=ALL scripts/slurm_eval_solver.sbatch
uv run python scripts/summarize_solver_runs.py --run base=<shard dir> --run lora=<shard dir> --run fft=<shard dir>
```

LoRA は焼き込み（ルートの `scripts/merge_adapter.py --model google/gemma-4-12B-it --model-revision main`）してから評価する。
FFT の出力には `processor_config.json` が保存されないので、評価前に土台のスナップショットから写す。
Slurm job: 学習 788（LoRA）、789（FFT 1 回目）、798（FFT やり直し）、評価 790 / 794 / 795 / 800。

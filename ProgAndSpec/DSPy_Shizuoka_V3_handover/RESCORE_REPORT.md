# 制約チェッカー追加と4条件の再採点（2026-09-01）

27 ある core_type のうち制約チェッカーが登録されていたのは 8 つだけで、100問中70問が
`unverified` として一律に減点されていました。そのうち約9割は参照値と完全一致した正解です。
残り20 core_type 分のチェッカーを実装し、保存済みの生成コードから4条件を再採点した記録です。

| 項目 | 内容 |
|---|---|
| 実施 | 2026-09-01 |
| 対象 | 改善前後プロンプト × Qwen3.6-27B / Qwen3.8-27B の4条件 × 各100問 |
| ブランチ | `feat/feasibility-checkers` |
| 再採点 | GPU・LLM 不使用（保存済み生成コードの再実行） |
| 手順 | `OPERATIONS.md` 4章「保存済みコードの再採点」 |

---

## 1. 要点

| 指標（400レコード合計） | 再採点前 | 再採点後 |
|---|---:|---:|
| `unverified` | 198 | **4** |
| `exact_match` | 17 | **189** |
| `exec_error` | 93 | 93 |

`exec_error` が変化しないことは、実行失敗が feasibility と無関係である以上あるべき挙動で、
再採点が既存の判定を壊していないことの確認にもなります。

残る `unverified` 4件は、チェッカーが形状を解析できず正直に未検証としたものです。

---

## 2. 原因

`check_feasibility_detailed` は core_type 文字列の完全一致でチェッカーを引きます。
未登録なら `verified: false` を返し、`metrics_v3` はそれを `status="unverified"` として
スコアを 1.0 に頭打ちにします。つまり**モデルが何を出力しても、その分類にチェッカーが
無ければ減点される**構造でした。

core_type 別に見ると、チェッカーの有無で `unverified` が完全に決まっています
（改善後 Qwen3.6・再採点前）。

| core_type | チェッカー | unverified / 問題数 |
|---|---|---:|
| スケジューリング_混合整数計画 | 登録済み | 0 / 10 |
| 配送・輸送_混合整数計画 | 登録済み | 0 / 9 |
| スケジューリング_整数計画 | 登録済み | 0 / 7 |
| ナップサック・パッキング_整数計画 | なし | 10 / 11 |
| ネットワークフロー_線形計画 | なし | 6 / 7 |
| 施設配置・被覆_整数計画 | なし | 6 / 6 |
| 割当・マッチング_整数計画 | なし | 6 / 6 |
| 複合・グラフ最適化_整数計画 | なし | 5 / 5 |

未登録は 20 core_type・70問でした。

---

## 3. 実装方針

`src/utils/feasibility_v3_ext.py` に20チェッカーを実装し、`feasibility.py` から一括登録します。

- 同じ core_type でも instance の形状が複数あるため、各チェッカーは instance のキーを見て
  サブ形状へ振り分けます。
- **解析してから検証する。** 構造を解析できなかった場合は違反を積み上げず `verified: false`
  を返して `unverified` に戻します。読めなかった構造について違反を報告するのは筋が通りません。
- 構造から目的値が一意に決まる問題では、再計算値との一致も検証します。構造だけを見る
  チェッカーは、最小化問題での過少申告——不当に高いスコアに直結する主張——を素通りさせます。
- 費用モデルは実装前に参照解で検算し、一致したものだけ組み込みました。

---

## 4. 再採点後のスコア

平均スコア（再採点前 → 再採点後）。`untouched40` が `factorial_comparison.json` の
`primary_subset` です。

| 条件 | untouched40（主指標） | legacy20 | train40 | all100 |
|---|---|---|---|---|
| before · Qwen3.6 | 0.774 → **0.960** | 0.628 → 0.875 | 0.567 → 0.820 | 0.662 → 0.887 |
| after · Qwen3.6 | 0.714 → **0.907** | 0.840 → 1.146 | 0.842 → 1.180 | 0.791 → 1.064 |
| before · Qwen3.8 | 0.482 → **0.563** | 0.335 → 0.535 | 0.275 → 0.455 | 0.369 → 0.514 |
| after · Qwen3.8 | 0.439 → **0.576** | 0.345 → 0.520 | 0.568 → 0.797 | 0.472 → 0.653 |

---

## 5. 解釈

**先に結論を述べます。信頼区間を出すと、プロンプト効果はほぼすべての分割で 0 と区別できません。**
唯一の例外が、GEPA が候補選択に使った13問での Qwen3.6 の伸びです。

### 5.1 露出段階で分けると別の姿になる

`train40` は均質な集合ではありません。GEPA は27問で反省し、残り13問で候補を選びます
（[train_gepa_v3.py](train_gepa_v3.py) の `n_val = max(3, len(train_raw) // 3)`）。
この2つは露出の質が違うので、混ぜた平均は「訓練例の記憶か汎化か」を判別できません。

paired bootstrap（10,000回、seed=42）による効果と 95% 信頼区間です。
`scripts/diagnose_effects.py` で再現できます。

| 分割 | n | Qwen3.6 | Qwen3.8 |
|---|---:|---|---|
| train27（反省に使用） | 27 | +0.238 [−0.087, +0.580] | +0.512 [−0.044, +1.054] |
| **val13（候補選択に使用）** | 13 | **+0.613 [+0.219, +1.054]** ✱ | −0.010 [−0.625, +0.606] |
| legacy20（旧テスト） | 20 | +0.271 [−0.033, +0.634] | −0.015 [−0.580, +0.530] |
| untouched40（主指標・未使用） | 40 | −0.053 [−0.439, +0.318] | +0.013 [−0.388, +0.419] |
| all100（全体） | 100 | +0.177 [−0.024, +0.374] | +0.139 [−0.124, +0.401] |

✱ = 95% 信頼区間が 0 を含まない

**10個の対比のうち、0 を跨がないのは1つだけ**です。しかもそれは、
GEPA が候補を選ぶのに使った集合での効果でした。これは選択過学習の兆候としては
整合的ですが、n=13 の単一の対比であり、それ以上のことは言えません。

### 5.2 untouched40 は厳密には未使用ではない

Phase E は instruction だけでなく few-shot demo も含むコンパイル済みプログラムです。
その demo 2問は `prob_001` と `prob_021` で、それぞれ **legacy20** と
**untouched40** に属します。主指標の40問のうち1問は、after プロンプトへ
焼き込まれた実例そのものです。

demo を除いた効果は次のとおりで、Qwen3.6 はほぼ不変、Qwen3.8 は上向きます。

| 分割 | Qwen3.6 | Qwen3.8 |
|---|---:|---:|
| legacy20（−prob_001） | +0.271 → +0.286 | −0.015 → +0.089 |
| untouched40（−prob_021） | −0.053 → −0.054 | +0.013 → +0.082 |

つまり demo 問題では、Qwen3.8 の after 条件はむしろ悪化していました。
混入の影響は小さく、方向も Qwen3.8 に不利に働いています。

### 5.3 過学習と言い切れるか

言い切れません。所見としては次までです。

- 唯一有意な効果が、候補選択に使った集合に出ている
- 未使用40問の効果は 0 を中心に広く分布し、劣化とも改善とも言えない
- train40 の +0.36 は、性質の違う2集合を平均した値であり、単独では読めない

### 5.4 過去の記述の訂正

このレポートの初版では「train40 と untouched40 の乖離という過学習の所見は維持される」と
記述しましたが、**取り下げます**。点推定の差は残るものの、信頼区間を出すと
train27 も untouched40 も 0 と区別できず、「乖離」を統計的に主張できません。

また「改善プロンプトは untouched40 で両モデルともマイナス」も誤りでした。
Qwen3.8 は +0.013、demo を除けば +0.082 です。

### 5.5 モデル差の解釈について

「Phase E は Qwen3.8 を reflection 側に置いて最適化されたので Qwen3.8 に効くはず」という
読みは誤りです。reflection LM の役割は候補 instruction を提案することで、
候補の良し悪しを決めたのは実際にコードを生成・評価された Qwen3.6 です。

Phase E は「**Qwen3.8 が書いた、Qwen3.6 向けのプロンプト**」であり、
Qwen3.8 自身に効く理由は設計上ありません。移植性を測るには、task model を
入れ替えた転移行列（3.6最適化→3.6/3.8評価、3.8最適化→3.6/3.8評価）が要ります。

---

## 6. 再採点で見つかった欠陥

チェッカーを足すだけでは終わりませんでした。以下は実データで再採点を2周する過程で
順に露見したもので、いずれも単体テストと参照解の検証だけでは表面化しませんでした。

### 6.1 `verified` が無条件に上書きされていた（満点の誤付与）

`check_feasibility_detailed` が登録済みチェッカーの返り値に対して `result["verified"] = True`
を常に実行しており、チェッカー側が返した `verified: false` が潰されていました。
新チェッカーが判定できない形状にまで満点が出る状態です。`setdefault` に変更しました。

### 6.2 参照解の形状しか読めていなかった（正解の取りこぼし）

prob_100（発電機起動停止計画）は**4条件すべてが最適値 5102 を出している**のに、
全条件で「10/10 全制約違反」になりました。

モデルは自然に書ける形で返します。

| 問題 | 参照解の形 | モデルが返した形 |
|---|---|---|
| prob_100 | `{on:[...], output:[...]}` | `[80,80,...]` / `[{period,output}]` |
| prob_070 / 076 | `[{left,right}]` | `[[0,0],[2,3]]` |
| prob_094 | `{plant:{customer:qty}}` | 工場×顧客の行列 / `[{plant,customer,amount}]` |
| prob_085 | `[{period,workforce,hire,fire}]` | `[14,14,14,18,19]` / `hires`・`fires`（複数形・1始まり） |
| prob_082 | `{id: qty}` | `[{id,quantity}]` |
| prob_075 | `{course:{room,slot}}` | `{course:[room,slot]}` |

未検証のまま 1.0 だったものを 0.0 に落とすのは、元の問題より悪い結果です。
これを受けて「解析してから検証する」規律に統一しました。

### 6.3 `[{"id": i, "count": n}]` を位置リストとして誤読

有界ナップサック（prob_056）の正解が「全要素が数値でない」判定になり、
`exact_match` 相当の解が 0.03 点に落ちていました。

### 6.4 欠品に在庫費のマイナスを付けていた

需要をまったく満たさない計画（prob_086）で在庫が負になり、再計算費用が **−1218** に
なりました。供給不足で弾かれる点は変わりませんが、欠品は値引きではありません。

観測した形状ゆれはすべて `tests/test_feasibility_v3_ext.py` に回帰テストとして固定しました。

---

## 7. 検証状態

| 項目 | 結果 |
|---|---|
| 参照解の通過 | **70 / 70**（未登録だった全 core_type、違反ゼロ） |
| 目的値改ざんの検出 | **66 / 69** |
| pytest | **235 passed**（作業開始時 47） |
| ruff | 新規ファイルは全てクリーン。継承コードの既存指摘は HEAD 時点から増加なし |

真の検出も残っています。二次割当（prob_071）でモデルは配置 `[1,3,4,2,0]` を出しながら
**費用 252 と申告**しましたが、参照解 457 を通すのと同じ式で計算すると 506 になります。
これは正しく弾いています。

---

## 8. 残していること

- **目的値を確定できない3問** — 最大流の過少申告は制約違反ではない（過大申告はカット上界で弾く）。
  単一割当ハブ配置は参照解の費用モデルを再現できず構造のみ検証。発注ロット最適化は解に
  発注回数が含まれず供給量を復元できない。理由は
  `src/utils/feasibility_v3_ext.py` の docstring に記載。
- **未検証のまま残る4レコード** — prob_062・prob_088・prob_093 の形状を解析できなかったもの。
  いずれも参照値と一致しているため、わずかな減点が残ります。
- **非決定な再実行が4件** — うち3件（prob_071 / 082 / 086）はチェッカーが過少申告を検出した
  真の差分、1件（prob_083 MADポートフォリオ）はソルバーの非決定です。
- **unverified の解消はスコアの意味を変える** — 再採点前の `factorial_comparison.json` と
  新しい結果を直接並べてはいけません。比較する場合は4条件すべてを同じチェッカー構成で
  採点し直します。
- **GEPA の train/val 分割は修正済みだが、既存結果は旧分割で作られている** — 旧実装は
  ソート済み train を末尾13問で切っており、val 側が3 domain に偏っていました。
  現在は core_type で層化します（`split_train_val_stratified`）。過去の結果を読むときの
  境界は `scripts/diagnose_effects.py` の `historical_train_val` が再現します。
- **untouched40 に demo が1問混入している** — 新規の最終ホールドアウトを用意するか、
  untouched40 を開発用と明記するかの判断が要ります。停止判断に使えばその時点で
  「未使用」ではなくなります。

---

## 9. 再現

評価結果 JSON には各問題の生成コードが保存されているため、再採点に GPU も LLM も要りません。
shard ディレクトリを1つの評価単位として扱い、元実行と同じ順序で `BestKnownRegistry` を
参照値から seed し直します。

```bash
cd /home/yy-lab/test_distillation/ProgAndSpec/DSPy_Shizuoka_V3_handover
uv run python scripts/rescore_with_checkers.py \
  outputs/prompt_model_comparisons/prompt-model-qwen36 \
  outputs/prompt_model_comparisons/prompt-model-qwen38 \
  --subsets-from outputs/prompt_model_comparisons/prompt-model-full100-merged/factorial_comparison.json \
  --output outputs/prompt_model_comparisons/rescored.json
```

再実行したコストが保存値と食い違う問題は `cost_mismatches` に記録されます。

続けて、露出段階別の効果と信頼区間を出します。

```bash
uv run python scripts/diagnose_effects.py \
  outputs/prompt_model_comparisons/rescored.json \
  --comparison outputs/prompt_model_comparisons/prompt-model-full100-merged/factorial_comparison.json \
  --output outputs/prompt_model_comparisons/effect-diagnosis.json
```

---

## 10. 関連ファイル

| パス | 役割 |
|---|---|
| `src/utils/feasibility_v3_ext.py` | 追加した20 core_type 分のチェッカー |
| `src/utils/feasibility.py` | チェッカー登録と `verified` の扱い |
| `scripts/rescore_with_checkers.py` | 保存済みコードの再採点 CLI |
| `tests/test_feasibility_v3_ext.py` | 参照解通過・改ざん検出・形状ゆれの回帰テスト |
| `tests/test_rescore_with_checkers.py` | 再採点 CLI のテスト |
| `scripts/diagnose_effects.py` | 露出段階別の効果分解と信頼区間 |
| `src/data_loader.py` | `split_train_val_stratified`（GEPA の train/val 層化） |
| `tests/test_train_val_split.py` | 層化分割の回帰テスト |
| `tests/test_diagnose_effects.py` | 効果分解のテスト |
| `OPERATIONS.md` 4章 | 再採点の運用手順 |

---

## 10. 追補（2026-09-04）: 旧チェッカーが参照解を弾いていた4問

feasibility.py に直接登録された旧チェッカー（8 core_type）は、prob_002・003・012・027 の
同梱参照解そのものを infeasible と判定していました。`schedule` / `assignments` / `assignment`
以外のキー（`machine_assignment`、`optimal_sequence`、`node_assignment`）と、`routes` 以外の
経路キー（`day1_routes`、`initial_routes`）を読めなかったためです。参照解と同じ形で返した
正答が 0 点になる構造でした。

キー候補を広げ、makespan が無い解では他の数値目的を objective として受けるようにしました。
100問すべての参照解が自分のチェッカーを通ることは `tests/test_legacy_checkers.py` で固定しています。

同じ保存済みコードを再採点した結果（`rescored-legacy-checkers-20260904.json`）:

| 条件 | 問題 | 修正前 | 修正後 |
|---|---|---|---|
| after · Qwen3.6 | prob_002 / 003 | infeasible 0.00 | exact_match 1.50 |
| after · Qwen3.6 | prob_012 / 027 | infeasible 0.00 | partial_feasible 0.28 / 0.16 |
| after · Qwen3.8 | prob_002 | infeasible 0.00 | exact_match 1.50 |
| before · Qwen3.6 | prob_003 | infeasible 0.00 | exact_match 1.50 |
| その他 | exec_error / gen_error の記録 | 変化なし | 変化なし |

分割平均への影響は最大でも +0.045（untouched40、Qwen3.6 の before / after とも）で、
before と after に同じ幅で乗るため、プロンプト効果の推定はほぼ動きません。

| 条件 | untouched40 | train40 | all100 |
|---|---|---|---|
| before · Qwen3.6 | 0.960 → 1.004 | 0.820 → 0.826 | 0.887 → 0.907 |
| after · Qwen3.6 | 0.907 → 0.951 | 1.180 → 1.221 | 1.064 → 1.098 |
| before · Qwen3.8 | 0.563 → 0.570 | 0.455 → 0.459 | 0.514 → 0.518 |
| after · Qwen3.8 | 0.576 → 0.583 | 0.797 → 0.835 | 0.653 → 0.671 |

partial_feasible に留まる prob_012 / 027 は、チェッカーが構造しか見ないためモデル解の
目的値が検証されず、旧メトリックの部分点になっています。

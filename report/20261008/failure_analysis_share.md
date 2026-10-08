# 未正解問題の失敗原因と方式比較（10/7 会議の決定事項への回答）

下沢 亮太郎（2026-10-08）。詳細は `ProgAndSpec/DSPy_Shizuoka_V3_handover/RESCORE_REPORT.md` 34 章。

## 要点

- 12B FFT が解けなかった 27 問（大規模テスト 120 問中）の内訳は、**学習データにない種別 18 問、評価時の CPU の混み具合 6 問、
  構文エラー 3 問**。「問題が数学的に難しくて解けない」ものは、学習した種別には 1 問も残っていない。
- CPU の混み具合: 時間制限付きのソルバー（CP-SAT、HiGHS、局所探索）は使える CPU 時間で解の質が変わる。評価は 8 問並列で
  他のジョブとも重なっていたため、**教師（Opus）と 1 文字も違わないコードが +19〜+187% と判定されていた**。同じコードを
  4 並列で再実行すると −9〜+4% で正解。評価の実行は今後 4 本ずつに絞る（`EXEC_CONCURRENCY`、既定 4）。
- 再実行後の正解数（120 問中）: 4B 91、12B LoRA 95、12B FFT 99、**4B + 検証 + 12B への引き継ぎ 101**。学習した 17 種別に
  限ると FFT 99 / 102、4B + 引き継ぎ 101 / 102。
- 学習前に解けなかった理由はモデルで違う: **素の 4B は推論が終わらない**（120 問中 103 問が出力上限）、**素の 12B は定式化と
  制約の扱いを誤る**（制約違反 64、返り値の形・目的値の申告ずれ 31）。

## 方式の比較（大規模テスト 120 問、正解 = 違反 0 かつ参照解から +10% 以内）

| 方式 | 条件 | 正解 |
|---|---|---:|
| なし | 素の 12B | 0 |
| プロンプト最適化のみ | 素の 12B + GEPA の指示文 | 2 |
| 学習のみ | 4B LoRA / 12B LoRA / 12B FFT | 91 / 95 / 99 |
| Better Together | 12B LoRA → 自己蒸留 / 素 → GEPA → 学習 | 98 / 97 |
| ハーネス | 4B + 検証 + 修復 + 12B への引き継ぎ | **101** |

- 効いているのは学習（教師の正解コード）。Better Together は学習のみと差がない（入れ替わり 5 対 2、5 対 3、有意でない）。
- ハーネスで効くのは「検証器で失敗を見つけて別のモデルに回す」部分。モデル自身に直させる修復は 88 回中 1 回しか成功しない。

## 残っている失敗（共有して一緒に見てほしいもの）

**構文エラー（12B FFT、nurse_roster の 3 問）**。どれも括弧の対応崩れで、同じ種別に固まっている。

| 問題 | エラー | 該当行 |
|---|---|---|
| prob_4516 | `'(' was never closed` | `need = int(instance.get("daily_requirements", [{}].get(str(d), {}).get(k, 0))` |
| prob_4518 | `')' does not match '['` | `model.AddExactlyOne([x[i, d, s] for s in ["early", "late", "night", None])` |
| prob_4520 | `unmatched ')'` | `sh = m.NewIntVar(0, max(0, int(r.get(s, 0)))), "")` |

**4B のデータ読み取り・構文（fjsp_setup の 6 問）**、**12B LoRA の出力上限（prp の 3 問、コードが 8,192 token に収まらない）**。

**学習データにない 3 種別（crew_pairing、portfolio、portfolio_cvar の 18 問）**は、検証器と問題文を直した 20 種別データ（v2）で
学習し直すと対象に入る。今のエージェントは、対応外の種別を「unsupported」と明示して返す。

## 再現

```bash
cd ProgAndSpec/DSPy_Shizuoka_V3_handover
P=outputs/prompt_model_comparisons
# 失敗の分類（保存済みの判定ログから）
uv run python scripts/classify_failures.py --problem-dir data/problems_hard_gen/test \
  --supported-kinds-file prompts/supported_kinds/sft_merged_v1.json --examples 10 \
  --run 12B_FFT=$P/gemma4-ft-20261004-test/sft_gemma4_12b_fft_fp32__shard01of01
# 未正解の行を 4 並列で再実行（LLM は呼ばない）
uv run python scripts/replay_failed.py --data-dir data/problems_hard_gen/test --out-run replay-20261008-test \
  --workers 4 --run 12B_FFT=$P/gemma4-ft-20261004-test/sft_gemma4_12b_fft_fp32__shard01of01
```

生成コードとエラーの全文は、各 shard の `evaluation_results_v3_gepa_phaseE.json`（`code`、`detail`）にある。

## 進行中

- 複数パターンの指示文で学習した 12B（会議の「学習データ改善」）。学習後に指示文 5 本で評価し、GEPA で指示文が動くかを見る。
- 論文どおりの Better Together（教師データなし、素の 12B + GEPA の指示文で解いた正解 2,667 対だけで学習 → GEPA）。

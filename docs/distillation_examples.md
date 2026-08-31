# 蒸留の効果と学習データの実例

このドキュメントは、`InternScience/Agents-A1-4B` をベースに、
`r0b0tlab/qwen3.8-max-glm5.2-distillation-51389` の教師応答で
sequence-level distillation を行った 2 種の学習結果（LoRA / Full FT）と、
蒸留前のベースモデルとで、同一プロンプトに対する応答がどう変わるかを示します。
加えて、学習に使ったデータセットの構造と、train / validation / test 各 split の
実例を掲載します。

## 1. 実験セットアップ

| 項目 | 値 |
| --- | --- |
| ベースモデル | `InternScience/Agents-A1-4B` @ `945c40a4aa6f534d434a353207b8d42ecf7a5293` |
| データセット | `r0b0tlab/qwen3.8-max-glm5.2-distillation-51389` (config `sft_final`, revision `2415a1…dfb103`) |
| LoRA チェックポイント | `outputs/agents-a1-4b-lora-3epoch/adapter/` |
| Full FT チェックポイント | `outputs/agents-a1-4b-fullft-3epoch/adapter/` |
| 学習エポック | 3.0（両方とも） |
| max_length | 2,048 |
| lr / warmup / accum | `2e-5` / 100 / 16（両方とも） |
| GPU | H200 NVL 141 GB × 1（Full FT は `visible_devices=1`、LoRA は `visible_devices=0`） |

デコード条件は 3 モデル共通で以下です。
- greedy（`do_sample=False`, `temperature=1.0`, `repetition_penalty=1.0`）
- `max_new_tokens=1536`（第 2 節の比較）
- チャットテンプレートで `add_generation_prompt=True`

学習ログから抽出した最終評価の loss は次のとおりです。
Full FT の方がベースモデルの表現力を全体で使えるぶん、
validation loss が下がっています。

| 変種 | validation loss | ソース |
| --- | --- | --- |
| LoRA (3 epoch) | 0.5995 | `outputs/agents-a1-4b-lora-3epoch/eval_metrics.json` |
| Full FT (3 epoch) | 0.4883 | `outputs/agents-a1-4b-fullft-3epoch/eval_metrics.json` |

再現するには次を実行します。

```bash
CUDA_VISIBLE_DEVICES=0 .runtime/train/bin/python scripts/compare_variants.py \
  --lora-adapter outputs/agents-a1-4b-lora-3epoch/adapter \
  --fullft-model outputs/agents-a1-4b-fullft-3epoch/adapter \
  --test-index 49 \
  --max-new-tokens 1536 \
  --output-json outputs/compare-test49.json
```

## 2. 顕著に差が出るプロンプトの例

test split の `index=49`（`domain=instruction`、`source=Dolly`）を使います。
この system プロンプトは `<think>...</think>` タグの中で推論を行い、
外側に簡潔な回答を書くことを要求します。この形式追従は、蒸留前後で
最も差が出やすい部分です。

### 2.1 プロンプト

**system:**

```text
You are a helpful, precise assistant. Follow instructions exactly.

Format your response as:
<think>
[Understand what is being asked. Identify constraints and requirements.
Plan your response structure.]
</think>

[Your complete, well-structured response following all given constraints]
```

**日本語訳:**

```text
あなたは有能で正確なアシスタントです。指示に厳密に従ってください。

応答は以下の形式で出力してください:
<think>
[何が問われているかを理解する。制約と要件を特定する。
応答の構成を計画する。]
</think>

[すべての制約に従った、完全でよく構成された応答]
```

**user:**

```text
What is the purpose of operating a motor vehicle?
```

**日本語訳:**

```text
自動車を運転する目的は何ですか?
```

### 2.2 教師（teacher）の応答

学習データに保存されている、Qwen3.8-Max / GLM5.2 系の教師応答です。

```markdown
## Purpose of Operating a Motor Vehicle

The purpose of operating a motor vehicle is to **transport people or goods
from one location to another** efficiently and conveniently.

More specifically, motor vehicles serve several key purposes:

- **Personal transportation** – commuting to work, running errands, traveling for leisure
- **Commercial transport** – delivering goods, providing services (taxis, rideshares, trucks)
- **Emergency response** – ambulances, fire trucks, police vehicles
- **Public transit** – buses and shuttles moving groups of people
- **Specialized tasks** – construction equipment, agricultural vehicles, military operations

At its core, a motor vehicle exists to overcome the limitations of human
physical movement — allowing faster travel over greater distances while
carrying heavier loads than a person could manage on foot.
```

**日本語訳:**

```markdown
## 自動車を運転する目的

自動車を運転する目的は、**人や物をある地点から別の地点へ**
効率的かつ便利に**輸送する**ことである。

より具体的には、自動車は以下のような主要な目的を果たす。

- **個人の移動** – 通勤、用事、レジャー目的の移動
- **商用輸送** – 貨物の配送、サービス提供（タクシー、ライドシェア、トラック）
- **緊急対応** – 救急車、消防車、警察車両
- **公共交通** – バスやシャトルによる集団輸送
- **専門用途** – 建設機械、農業車両、軍用車両

本質的には、自動車は人間の身体的移動能力の限界を克服するために存在し、
徒歩では扱えない重量物を運びながら、より速く、より遠くへ
移動することを可能にする。
```

### 2.3 蒸留前（`InternScience/Agents-A1-4B` ベース）の応答

1,536 tokens を消費しても最終回答に到達しません。system プロンプトの
`Format your response as: [A] [B]` の角括弧を、テンプレートなのか、
そのまま出力するべき文字列なのかを繰り返し自問する **メタ解釈ループ** に入り、
質問への回答そのものは書き始めません（抜粋、7,110 文字のうち冒頭）。

```text
Thinking Process:

1.  **Analyze the Request:**
    *   Question: "What is the purpose of operating a motor vehicle?"
    *   Constraint: "Format your response as: [Understand what is being asked. ...
    *   Wait, the user instruction says: "Format your response as: [Understand what is being asked. ...
    *   This looks like a system instruction or a specific formatting requirement...
    ...
    *   Wait, is this a trick? "Format your response as: [Understand what is being asked. ...
    *   It might mean I should literally output those phrases.
    *   Let's assume the user wants the response to *contain* these sections.
    ...
    *   Wait, if I just output the headers, it's not helpful. I need to output the content *within* those sections or *after* them.
    ...
```

**日本語訳:**

```text
思考プロセス:

1.  **リクエストを分析:**
    *   質問: 「自動車を運転する目的は何か?」
    *   制約: 「以下の形式で応答せよ: [何が問われているかを理解する。...
    *   待って、ユーザーの指示は次のように言っている: 「以下の形式で応答せよ: [何が問われているかを理解する。...
    *   これはシステム命令、あるいは特定の出力形式要件のようにも見える...
    ...
    *   待って、これは罠だろうか? 「以下の形式で応答せよ: [何が問われているかを理解する。...
    *   このフレーズを文字通り出力するべき、という意味かもしれない。
    *   ユーザーは応答がこれらのセクションを*含む*ことを望んでいる、と仮定しよう。
    ...
    *   待って、ヘッダーだけを出力しても役に立たない。それらのセクションの
        *中*か*後*に内容を出力する必要がある。
    ...
```

観察できる差分:

- `<think>...</think>` の形式を全く守れない（`Thinking Process:` という
  自前の枠を使う）。
- 「これは指示か、雛形か、リテラル出力か」を自問して発散し、
  1,536 tokens 予算を使い切っても回答本体に到達しない。

### 2.4 LoRA (3 epoch) の応答

同じ 1,536 tokens 予算で自然終端し、教師応答の骨組み
（transportation を大分類にして、通勤/仕事/レジャー/緊急を細分化）に沿った
回答を出します。ただし、思考パートは **`Thinking Process:` の自前枠のまま**
（`<think>` は開かない）で、指示された形式に完全には従っていません。
出力全長は 3,384 文字。抜粋:

```text
Thinking Process:

1.  **Analyze the Request:**
    *   **Question:** "What is the purpose of operating a motor vehicle?"
    *   **Intent:** The user is asking for the reasons or goals behind
        driving/operating a motor vehicle.
    ...

7.  **Final Polish:** (Self-Correction during drafting)
    *   *Initial thought:* Maybe list specific car models? *Correction:* No,
        the question is about the *purpose*, not the vehicle itself.
    ...
</think>

The primary purpose of operating a motor vehicle is **transportation**—moving
people and goods from one location to another. However, this broad purpose
breaks down into several specific categories:

**1. Commuting and Daily Life**
*   Getting to work, school, or other daily destinations.
...
**5. Personal Freedom and Independence**
*   The ability to choose one's own route, schedule, and destination.
*   Mobility for those who cannot use other forms of transport.

In summary, while the fundamental goal is **mobility**, the specific purpose
depends on the context—whether it is for survival, work, pleasure, or social
connection. Safety and legal compliance are also integral parts of operating
a motor vehicle responsibly.
```

**日本語訳:**

```text
思考プロセス:

1.  **リクエストを分析:**
    *   **質問:** 「自動車を運転する目的は何か?」
    *   **意図:** ユーザーは自動車を運転・運行することの理由や目標を尋ねている。
    ...

7.  **最終仕上げ:** (執筆中の自己修正)
    *   *初期案:* 特定の車種を列挙するべきか? *修正:* いや、質問は自動車自体では
        なく*目的*についてである。
    ...
</think>

自動車を運転する主な目的は**輸送**——人や物をある場所から別の場所へ
移動させることである。しかし、この広義の目的はいくつかの具体的な
カテゴリに分けられる。

**1. 通勤と日常生活**
*   仕事、学校、その他の日常の目的地へ行く。
...
**5. 個人の自由と自立**
*   自分のルート、スケジュール、目的地を選ぶ能力。
*   他の交通手段を利用できない人のための移動性。

要約すると、根本的な目標は**移動性**にあるが、具体的な目的は文脈——
生存のためか、仕事、娯楽、社会的つながりのためか——に依存する。
自動車を責任を持って運転するためには、安全と法令遵守も不可欠な要素である。
```

観察できる差分:

- 回答本体は教師とほぼ同じ構成に到達している。
- 思考パートは `Thinking Process:` の numbered outline のまま
  （teacher と一致していない）。
- `</think>` の直前で長い自前 outline を挟むため、全長が Full FT の
  約 2.3 倍。

### 2.5 Full FT (3 epoch) の応答

デコード開始からわずか 13.4 秒（LoRA の 58.1 秒、Base の 139.0 秒に対し）で、
1,495 文字の応答を書き終えます。system 指示どおり
`<think>...</think>` の内側に推論を置き、外側に教師と同じ形の回答を出します。
全長を掲載します。

```text
The user is asking about the purpose of operating a motor vehicle. This is a
straightforward question about the fundamental reason people drive cars,
trucks, motorcycles, etc.

The primary purpose is transportation — moving people and goods from one
location to another. However, there are several related purposes:

1. **Personal transportation** – commuting to work, school, running errands, traveling for leisure.
2. **Cargo/logistics** – transporting goods for business and commerce.
3. **Emergency response** – ambulances, fire trucks, police cars.
4. **Accessibility** – providing mobility to those who cannot walk or use other forms of transport.
5. **Economic activity** – enabling trade, travel, and business operations.

I should keep the answer clear and concise, focusing on the main purpose while
acknowledging the broader context.
</think>

The primary purpose of operating a motor vehicle is **transportation** —
moving people and goods from one location to another.

This includes:

- **Personal travel** – commuting, going to work or school, running errands, and recreational activities.
- **Cargo and logistics** – delivering goods for commerce and supply chains.
- **Emergency services** – transporting patients, responding to incidents, and law enforcement.
- **Accessibility** – providing mobility to those who cannot use other means of transport.

In short, motor vehicles exist to enable movement and connectivity between
places, supporting daily life, work, and commerce.
```

**日本語訳:**

```text
ユーザーは自動車を運転する目的について尋ねている。これは、人々が自動車、
トラック、バイクなどを運転する根本的な理由についての率直な質問である。

主な目的は輸送——人や物をある場所から別の場所へ移動させることである。
ただし、関連するいくつかの目的がある。

1. **個人の輸送** – 通勤、通学、用事、レジャー目的の移動。
2. **貨物/物流** – ビジネスや商業のための物資輸送。
3. **緊急対応** – 救急車、消防車、警察車両。
4. **アクセシビリティ** – 徒歩や他の交通手段を利用できない人への移動性の提供。
5. **経済活動** – 貿易、旅行、ビジネス運営の実現。

主な目的に焦点を当てつつ、より広い文脈も認識しながら、明確で簡潔な答えを
維持すべきである。
</think>

自動車を運転する主な目的は、**輸送**——人や物をある場所から別の場所へ
移動させることである。

これには以下が含まれる。

- **個人の移動** – 通勤、通学、用事、レクリエーション活動。
- **貨物と物流** – 商業とサプライチェーンのための物資配送。
- **緊急サービス** – 患者の輸送、事故対応、法執行。
- **アクセシビリティ** – 他の交通手段を利用できない人への移動性の提供。

要するに、自動車は場所と場所の間の移動と接続を可能にし、日常生活、
仕事、商業を支えるために存在する。
```

備考: 出力の decode 上では `<think>` の開きタグが省略されて見えます。
これは、Agents-A1-4B のチャットテンプレートが assistant の先頭を
`<think>` で自動的にプリフィルし、モデルは `</think>` 側だけを出力する
仕様のためです。トークン列としては規定どおりの形式です。

### 2.6 まとめ

| 変種 | 生成時間 | 出力長 | `<think>` タグ | 教師形式への追従 | 最終回答に到達 |
| --- | ---: | ---: | :---: | :---: | :---: |
| Base（蒸留前） | 139.0 s | 7,110 chars | × | × | × (1,536 tokens で未完) |
| LoRA (3 epoch) | 58.1 s | 3,384 chars | 部分的（`</think>` のみ） | △（本体は近い） | ○ |
| Full FT (3 epoch) | 13.4 s | 1,495 chars | ○ | ○ | ○ |

この 1 プロンプトで見える定性的な違いは 3 点です。

1. **形式追従**: base は system 指示の解釈で発散し、`<think>` を書けない。
   LoRA は本体は書けるが `<think>` を出せない。Full FT は書ける。
2. **簡潔さ**: 同一プロンプト・同一 decode で、Full FT は base の 1/5、
   LoRA の 1/2 弱の長さで完了する。
3. **完了速度**: greedy decode 時間は Full FT が base の約 1/10。
   モデルサイズは同一なので、これは 1 応答あたりのトークン数が
   減った直接の効果です。

これは 1 サンプルの観察であり、正確な性能比較にはテストセット全体の
自動評価（loss だけでなく domain 別の完了率や形式一致率）が必要です。
定量比較のフックは pilot 用の `train.log`（tensorboard）と
`eval_metrics.json` に残しています。

## 3. 蒸留データの構造

`load_dataset("r0b0tlab/qwen3.8-max-glm5.2-distillation-51389", "sft_final",
revision="2415a156cfaec47cab320d559dc4df2df0dfb103")` で読み込むと、
以下の 3 split と 8 カラムが返ります。

| split | 行数 |
| --- | ---: |
| train | 46,250 |
| validation | 2,569 |
| test | 2,570 |

各行のカラム:

| カラム | 型 | 意味 |
| --- | --- | --- |
| `parent_id` | string | 教師応答の生成前の元行 ID |
| `id` | string | 蒸留後の行 ID |
| `split` | string | 元テーブルにおける split ラベル |
| `messages` | JSON文字列またはlist[dict] | `role`/`content` を持つ会話履歴。system → user → assistant の 3 メッセージが典型 |
| `tools` | JSON文字列または空 | agent_tool ドメイン用のツール定義（存在するときのみ） |
| `sampling_weight` | float | 教師応答生成時のサンプリング重み |
| `source` | string | 元ソース（`Evol-Code`, `MetaMathQA`, `Dolly` など） |
| `domain` | string | 上位カテゴリ |

Domain 分布（train / validation / test は同じ比率）:

| domain | train | val | test |
| --- | ---: | ---: | ---: |
| math | 12,914 | 719 | 719 |
| code | 12,596 | 699 | 699 |
| reasoning | 9,415 | 524 | 524 |
| instruction | 6,545 | 364 | 364 |
| agent_tool | 4,780 | 263 | 264 |

主な `source` は `Evol-Code` / `MetaMathQA` / `CodeAlpaca` / `tulu-3` /
`glm5.2-agent-tool-synthetic` / `SciQ` / `NuminaMath-CoT` / `CommonsenseQA` /
`OrcaMath` / `Dolly` などです。

### 3.1 学習時のフィルタリング

`scripts/train_distillation.py` は次を落とします（初回学習の
`dataset_stats.json` より）。

- `tools` を持つ行は `--include-tools` を指定しないと落とす
  （train で 4,780 行、validation で 263 行）。
- assistant-only ラベルを付けた入力トークン列が
  `--max-length` を超える行は落とす（train で 1,682 行、validation で 7 行）。

結果として、初期レシピが実際に学習する行数は次です。

- train: `46,250 → 4,780 (tools) → 1,682 (overlength) → 39,788` 行
- validation: `2,569 → 263 (tools) → 7 (overlength) → 221` 行

test は現行スクリプトでは学習・評価に使っていません。定性/定量の
最終評価にだけ使う想定です。

## 4. 各 split の代表例

以下は「system + user + assistant」の 3 メッセージ、tool 呼び出しなし、
本文長が中程度の代表行を各 split から 1 例ずつ選んだものです。
`messages[-1]` の `assistant` が蒸留の学習ターゲット
（`InternScience/Agents-A1-4B` に模倣させたい教師応答）です。

### 4.1 train split（index=2, domain=reasoning, source=CommonsenseQA）

**system:**

```text
You are an expert logical reasoner. Think through the problem carefully.

Format your response EXACTLY as:
<think>
[Break down the problem into components.
Consider multiple hypotheses if applicable.
Eliminate incorrect paths explicitly.
Build toward the conclusion step by step.]
</think>

[Clear, concise final answer with brief justification]
```

**日本語訳:**

```text
あなたは論理的推論の専門家です。問題を注意深く考えてください。

応答は以下の形式に**厳密に**従ってください:
<think>
[問題を構成要素に分解する。
必要に応じて複数の仮説を検討する。
誤った経路を明示的に排除する。
段階的に結論へ組み立てる。]
</think>

[明快で簡潔な最終回答と、短い根拠]
```

**user:**

```text
Where could you find a library that people of any age can visit?

Choices:
  A) house
  B) school
  C) college
  D) inside
  E) downtown

Select the correct answer and explain your reasoning step by step.
```

**日本語訳:**

```text
年齢を問わず誰でも訪れることができる図書館は、どこにあるでしょうか?

選択肢:
  A) house（家）
  B) school（学校）
  C) college（大学）
  D) inside（屋内）
  E) downtown（繁華街）

正解を選び、段階的に推論を説明してください。
```

**assistant（教師応答＝学習ターゲット）:**

```markdown
**The correct answer is E) downtown.**

A public library, which is open to people of any age, is typically located in
a downtown area. Unlike school or college libraries (which restrict access
to students/staff) or a house library (which is private), a public library
in a downtown location welcomes visitors of all ages.
```

**日本語訳:**

```markdown
**正解は E) downtown（繁華街）です。**

年齢を問わず誰でも訪れることができる公立図書館は、通常、繁華街に位置する。
学校や大学の図書館（学生・職員に利用が制限される）や、家の書斎（私的なもの）
とは異なり、繁華街にある公立図書館はあらゆる年齢の訪問者を歓迎する。
```

### 4.2 validation split（index=3, domain=reasoning, source=ARC-Easy）

**system:**（train と同一の reasoning テンプレート。和訳は 4.1 を参照）

**user:**

```text
Which of these moves in a regular path around the Sun?

Choices:
  A) a planet
  B) a constellation
  C) a meteor
  D) a star

Select the correct answer and explain your reasoning.
```

**日本語訳:**

```text
以下のうち、太陽の周りを規則的な経路で動くのはどれですか?

選択肢:
  A) a planet（惑星）
  B) a constellation（星座）
  C) a meteor（流星）
  D) a star（恒星）

正解を選び、推論を説明してください。
```

**assistant（教師応答＝学習ターゲット）:**

```markdown
**Answer: A) a planet**

A planet moves in a regular, predictable path (called an orbit) around the
Sun. This is a defining characteristic of planets — they are celestial bodies
that orbit a star in a stable, elliptical path. The other options do not
orbit the Sun: constellations are fixed patterns of distant stars, meteors
are brief flashes of light in Earth's atmosphere, and stars are distant suns
that do not orbit our Sun.
```

**日本語訳:**

```markdown
**回答: A) a planet（惑星）**

惑星は、太陽の周りを規則的で予測可能な経路（軌道と呼ばれる）で移動する。
これは惑星を定義づける特徴である——惑星は、恒星の周りを安定した楕円軌道
で公転する天体である。他の選択肢は太陽を公転していない: 星座（constellation）
は遠くの恒星の固定パターンであり、流星（meteor）は地球大気中の一瞬の閃光で
あり、恒星（star）は太陽を公転しない別の恒星である。
```

### 4.3 test split（index=7, domain=code, source=CodeAlpaca）

**system:**

```text
You are an expert software engineer. Write complete, correct, runnable code
in the language requested by the user. Follow the requested language,
interface, and constraints. Explain the approach briefly when requested.
```

**日本語訳:**

```text
あなたは熟練したソフトウェアエンジニアです。ユーザーが指定した言語で、
完全で、正しく、実行可能なコードを書いてください。指定された言語、
インターフェース、制約に従ってください。求められた場合は、アプローチを
簡潔に説明してください。
```

**user:**

```text
Create a program in C that outputs all perfect square numbers between 1 and 100.
```

**日本語訳:**

```text
1 から 100 までのすべての完全平方数を出力する C 言語のプログラムを作成してください。
```

**assistant（教師応答＝学習ターゲット。test 行は現行スクリプトでは学習に使いません）:**

````markdown
```c
#include <stdio.h>

int main(void)
{
    printf("Perfect squares between 1 and 100:\n");

    for (int i = 1; i * i <= 100; i++)
    {
        printf("%d ", i * i);
    }

    printf("\n");
    return 0;
}
```

**Output:**
```
Perfect squares between 1 and 100:
1 4 9 16 25 36 49 64 81 100
```

**How it works:**
- The loop variable `i` starts at 1 and increments by 1.
- The condition `i * i <= 100` ensures we stop once the square exceeds 100
  (i.e., when `i` reaches 11, since 11² = 121 > 100).
- Each iteration prints `i * i`, which is the perfect square.

**Time complexity:** O(√n) where n = 100, so just 10 iterations.
**Space complexity:** O(1) — no extra data structures used.
````

**日本語訳（コードは共通言語のためそのまま、解説部分のみ和訳）:**

````markdown
```c
#include <stdio.h>

int main(void)
{
    printf("Perfect squares between 1 and 100:\n");

    for (int i = 1; i * i <= 100; i++)
    {
        printf("%d ", i * i);
    }

    printf("\n");
    return 0;
}
```

**出力:**
```
Perfect squares between 1 and 100:
1 4 9 16 25 36 49 64 81 100
```

**動作の仕組み:**
- ループ変数 `i` は 1 から始まり 1 ずつ増加する。
- 条件 `i * i <= 100` は、平方が 100 を超えたら停止することを保証する
  （`i` が 11 に達すると 11² = 121 > 100 となるため）。
- 各反復で `i * i`（完全平方数）を出力する。

**時間計算量:** O(√n) （n = 100 なので、反復は 10 回のみ）。
**空間計算量:** O(1) （追加のデータ構造を使わない）。
````

### 4.4 assistant-only ラベルの付き方（イメージ）

`distillation/semantic_labels.py` は、上の `messages` にチャットテンプレートを
適用したうえで、**system と user 相当のトークンには `-100`**、
**assistant の生成部分にだけ実トークン ID** を入れます。
損失は assistant 側のトークンでしか動きません。以下は概念図です。

```text
input_ids : [<sys>...system tokens...</sys> <user>...user tokens...</user> <asst>...assistant tokens...</asst>]
labels    : [-100     -100 ...          -100   -100   -100 ...        -100    <asst>...assistant tokens...</asst>]
```

このラベル付けは prefix-differential 方式で行われ、`apply_chat_template` の
出力が prefix-preserving でない場合のみ redaction 方式にフォールバックします。
入力長が `--max-length` を超える行は落とし、
supervised token が 0 の行も落とします（`encode_example` の実装）。

## 5. 参考リンク

- 学習スクリプト: [scripts/train_distillation.py](../scripts/train_distillation.py)
- ラベル生成: [distillation/semantic_labels.py](../distillation/semantic_labels.py)
- コレーター: [distillation/collator.py](../distillation/collator.py)
- 変種比較スクリプト: [scripts/compare_variants.py](../scripts/compare_variants.py)
- サーバー運用手順: [docs/serving_runtimes.md](serving_runtimes.md)

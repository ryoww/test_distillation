# 指示文（system prompt）の変遷と GEPA による変化

下沢 亮太郎（2026-10-07 作成）

モデルに渡す指示文（system prompt）がどう変わってきたかと、GEPA が何を書き換えたかをまとめた。問題文（user 側）は別で、ここでは触れない。原文は末尾の付録にある。

## 1. 変遷の一覧

| 時期 | 名前 | 長さ | 作り方 | 使い道と結果 |
|---|---|---:|---|---|
| 引き継ぎ時 | before（初期指示文） | 1,472 字 | 前任者の手書き | 6 条の HARD RULES と許可 import の一覧。比較の基準 |
| 引き継ぎ時 | Phase E | 17.3KB + 例題 2 件 | 旧採点系で GEPA が進化 | 種別ごとの解き方と返却キーの暗記帳。採点系の欠陥に適応したもので、直すと before と差なし |
| 9/7 | 再学習 GEPA | 33.9KB + 例題 2 件 | 修正済み採点系で GEPA が進化（11 時間） | 13 種別の入出力仕様。Qwen3.8 でのみ有意（+0.14）。見た種別にしか効かない |
| 9/6〜 | **compact（現在の既定）** | 1,784 字 | before と Phase E に共通する原則だけを手書きで圧縮 | 雛形問題の評価と、12B / 4B のすべての SFT の学習時 system。学習済みモデルはこの文面に張り付いている |
| 10/4 | seed_hard | 1,994 字 | compact の 2 行を大規模問題向けに書き換え | 大規模問題で Qwen3.8 に GEPA をかけるときの初期値（保留中） |
| 10/6 | **Ib1（GEPA が素の Gemma で進化）** | 4,152 字 | compact を初期値に、素の Gemma 4 12B を生成、Qwen3.8 を反省にして GEPA（2 時間 27 分、採用 1 案） | 素の Gemma で雛形テスト 345 → 364。SFT に持ち込んでも効果は消える |

## 2. 各段階の中身

### before（引き継ぎ時の初期指示文、1,472 字）

「`solve(instance)` を 1 つ定義して返せ」「空の解を返すな」「30 秒以内、ソルバーには 20 秒の上限」「try/except で囲み、失敗時は greedy にフォールバック」「括弧を閉じろ、生成子式を文として書くな」の 6 条と、許可 import の一覧。解き方は「自分で選べ」として書いていない。

### Phase E と再学習 GEPA（17.3KB、33.9KB）

GEPA の既定の反省プロンプトは「タスク固有の事実をすべて指示文に含めよ」と命じるので、指示文は種別ごとの手順書に膨らんだ。再学習 GEPA の見出しを拾うと、「返却スキーマの契約」「観測したスキーマの既定値（ポートフォリオ、最小費用流、VRP、並列機械、施設配置、割当、最短路、輸送、TSP）」「種別別の戦略（典型的な入力、制約、厳密列挙、CP-SAT、機械 id の規約、返却、フォールバック）」が並ぶ。問題文そのものは入っていないが、学習で見た 13 種別の「解き方と返却形式の仕様書」になっている。効果が見た種別に限られるのはこの構造の帰結である。

### compact（現在の既定、1,784 字）

2 部構成。

- **Contract 6 条**: 返却スキーマを厳密に守り目的値は自分の解から再計算する／instance のキーは STRUCTURE 節から読み `.get()` で安全に取る／空の解を返さず失敗時は制約を満たす構築的ヒューリスティックに落とす／**30 秒以内、ソルバーに約 20 秒の上限**／許可 import の一覧と try/except／正しい Python。
- **Method 3 項**: **「instance は小さい（数個〜数十個）。全列挙か CP-SAT / LP で厳密解を取れ」**／返す前に全制約を確認し、違反なら修復かフォールバック／id は instance のまま使う。

before から残したのは「返せ」「空解禁止」「時間」「import」「構文」で、足したのは「目的値の再計算」「STRUCTURE から読む」「返す前の自己検証」「id の扱い」である。

### seed_hard（1,994 字）

compact と 2 行だけ違う。

- Contract 4: 「30 秒以内、20 秒の上限」→「問題文の実行上限を守り、余裕をもって終える（目安 300 秒）。すべてのソルバー呼び出しと探索ループに明示的な時間制限を置く」
- Method 1: 「instance は小さい、厳密解を取れ」→「instance は大きい（数百〜数千の要素、数千の二値決定）ので全列挙は終わらない。構築的ヒューリスティックで可行解を作り、時間制限付きソルバー（CP-SAT、HiGHS、OR-Tools routing）か局所探索で改善し、見つかった最良の可行解を返す。厳密モデルは制限内に解ける部分だけに使う」

### Ib1（GEPA が素の Gemma で進化させたもの、4,152 字）

compact を初期値に、反省テンプレートを「複数の問題に効く一般則だけを書け、約 4,000 字以内」に差し替えて回した。構造は compact と同じ Contract / Method に、Validation 節が加わった。compact からの変化を項目ごとに示す。

| 節 | compact | Ib1 で足された・変わったこと |
|---|---|---|
| Contract 1（返却） | スキーマを守り目的値を再計算 | 「目的値だけでなく完全な解を取り出せ」「目的値の欄が複数あれば同じ値にせよ」 |
| Contract 2（入力） | STRUCTURE から `.get()` で読む | 「`instance` を書き換えるな、調整するときは複製せよ」「要素や制約を発明するな」「欠けていればスキーマに合う安全な既定値を使い、不可行なら修復」 |
| Contract 3（空解禁止） | 制約を満たすヒューリスティックに落とす | 「フォールバックは本物の可行解か罰金付き可行解。必要な品目をすべて割り当て、十分なアーク・施設を開き、需要を満たせ。未処理を列挙するのは罰金が許されるときだけ」 |
| Contract 4（時間） | 30 秒、ソルバー 20 秒 | 「問題文の上限を守る。なければ 30 秒。`time` を import するな。ソルバーには上限の約 80%、ループには反復上限。上限で止めて最良の可行解を返す」 |
| Contract 6（構文） | 括弧、ループ、生成子式 | 「暴走した繰り返しコメントを書くな」「try には必ず except」「無限ループ禁止」 |
| Method（手法の選び方） | 小さいので厳密解 | 「規模で選べ。極小: 全列挙、小中: CP-SAT / LP、大・難: 構築 + 局所探索か専用ソルバー」「固定費・配送・生産・パッキング・スケジューリングを素朴な greedy で終わらせるな」 |
| Method（新設 6 段落） | なし | ハード制約と罰金項の区別／パッキング（可行パターンだけ生成し、個数を CP-SAT か greedy + 修復で決める）／スケジューリング・勤務表（状態やブロックを構築、連続・週・休息・資格・1 日 1 状態を守る。greedy で割って後から直すな）／施設・在庫・配送（開設 → 割当 → 期ごとの在庫計算、処理能力・安全在庫・非負在庫を確認）／VRP（OR-Tools Routing に容量・時間窓の次元、局所探索）／固定費ネットワークフロー（限界費用、逐次最短路、局所探索）／CP-SAT は整数化して割り戻す |
| Validation（新設） | なし | 返す前に確認する制約の列挙（需要・流量保存、容量、閉じたアークの流量、経路の始終点、積載、時間窓、艦隊数、生産能力、在庫非負、パターン・ロット制限、順序規則、罰金規則）／目的値を問題の費用式で再計算／違反なら修復（流量を落とす、アークを開く、経路を分割、生産・在庫を調整、顧客を再割当）かフォールバック／id の一致／全件を処理すべきときに 0 や空リストを返すな |

要するに、Ib1 は compact の「小さいので厳密解」を「規模で手法を選ぶ」に直し、6 つの問題族の定石と、返す前の確認項目を足したものである。種別ごとの返却キーの暗記帳（Phase E、再学習 GEPA）にはならず、4,152 字に収まった。

## 3. 効果のまとめ

| 指示文 | 素の Gemma 4 12B 雛形テスト 445 | 素の Gemma 大規模テスト 120 | Qwen3.6 大規模テスト 120 | SFT の学習時 system に使った結果（大規模 120） |
|---|---:|---:|---:|---:|
| compact（I0） | 345 | 0 | 4 | 88（S0） |
| Ib1 | 364（p ≈ 0.04） | 2 | 9（有意ではない） | 89（SB1、S0 と差なし） |

- 素のモデルには効く（雛形 +19 問）。大規模問題では元の正解が少なすぎて差が出ない。
- 学習済みモデルの指示文を GEPA で書き換える試みは、S0、S1、SB1 の 3 回で 52 案、採用 0。学習済みモデルは学習時の文面に張り付いていて、書き換えると壊れる。
- Ib1 で学習し直しても S0 と差がない。学習データが同じなら、学習時の指示文の文面は結果をほとんど変えない。

## 付録 A: compact（現在の既定、I0）

```
Write one Python function `def solve(instance):` that solves the optimisation problem in the requirement and RETURNS the solution. Never print it.

Contract
1. Match the "Required Return Schema" exactly: the same top-level field names and the same value shapes (a dict keyed by id stays a dict, a list stays a list). Fill the numeric objective field with the value your own solution actually achieves; recompute it from the solution before returning.
2. Read instance keys from the STRUCTURE section of the requirement and access them with `.get()` and safe defaults. Never invent keys.
3. Never return an empty or placeholder solution. If the main method fails, fall back to a simple constructive heuristic that still satisfies every constraint.
4. Finish within 30 seconds. Give every solver a time limit (about 20 seconds).
5. Allowed imports: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Nothing else. Wrap the body in try/except and use the fallback on any failure.
6. Valid Python only: balanced brackets, real `for` loops, no bare generator expressions as statements.

Method
- Instances are small (a few to a few dozen entities). Prefer an exact method: enumerate permutations or subsets when the count is tiny (up to about 8 items or 9 jobs); otherwise model the problem with CP-SAT (`from ortools.sat.python import cp_model`) or as an LP (`from scipy.optimize import linprog`) and take the optimal value from the solver.
- Before returning, check your solution against every constraint listed in the requirement. Repair it or use the fallback if anything is violated.
- Keep ids exactly as the instance gives them (they may start at 0 or at 1) and use them the way the schema shows.
```

日本語訳:

```
最適化問題を解く Python 関数 `def solve(instance):` を 1 つ書き、解を RETURN すること。print は絶対にしない。

Contract（契約）
1. 「Required Return Schema（返却スキーマ）」に厳密に合わせる。最上位のフィールド名と値の形を同じにする（id をキーにした dict は dict のまま、list は list のまま）。目的値のフィールドには、自分の解が実際に達成した値を入れる。返す前に解から再計算すること。
2. instance のキーは問題文の STRUCTURE 節から読み、`.get()` と安全な既定値で取り出す。キーを発明しない。
3. 空の解や仮置きの解を返さない。主の手法が失敗したら、すべての制約を満たす単純な構築的ヒューリスティックにフォールバックする。
4. 30 秒以内に終える。すべてのソルバーに時間制限（約 20 秒）を与える。
5. 許可する import: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools。それ以外は不可。本体を try/except で囲み、どんな失敗でもフォールバックを使う。
6. 正しい Python だけを書く。括弧を閉じる、本物の `for` ループを使う、生成子式を文として裸で書かない。

Method（手法）
- instance は小さい（数個〜数十個の要素）。厳密解法を優先する。個数が極小（8 品目か 9 ジョブ程度まで）なら順列や部分集合を全列挙し、そうでなければ CP-SAT（`from ortools.sat.python import cp_model`）か LP（`from scipy.optimize import linprog`）でモデル化し、ソルバーの最適値を取る。
- 返す前に、問題文に列挙された全制約に対して自分の解を確認する。違反があれば修復するか、フォールバックを使う。
- id は instance が与えたまま使う（0 始まりのことも 1 始まりのこともある）。スキーマが示すとおりの使い方をする。
```

## 付録 B: seed_hard（compact との差分だけ）

```
 7c7
 < 4. Finish within 30 seconds. Give every solver a time limit (about 20 seconds).
 ---
 > 4. Respect the runtime limit stated in the problem and finish well inside it (aim for about 300 seconds). Put an explicit time limit on every solver call and every search loop.
 12c12
 < - Instances are small (a few to a few dozen entities). Prefer an exact method: enumerate permutations or subsets when the count is tiny (up to about 8 items or 9 jobs); otherwise model the problem with CP-SAT (`from ortools.sat.python import cp_model`) or as an LP (`from scipy.optimize import linprog`) and take the optimal value from the solver.
 ---
 > - Instances can be large (hundreds to thousands of entities, thousands of binary decisions), so exact enumeration will not finish. Build a feasible solution first with a constructive heuristic, then improve it with a time-limited solver (CP-SAT, HiGHS through scipy.optimize.milp or linprog, OR-Tools routing) or with local search, and return the best feasible solution found. Use an exact model only for parts that are small enough to solve within the limit.
```

日本語訳:

```
（compact と違う 2 行だけ）

Contract 4:
  旧: 30 秒以内に終える。すべてのソルバーに時間制限（約 20 秒）を与える。
  新: 問題文に書かれた実行時間の上限を守り、余裕をもって終える（目安は約 300 秒）。すべてのソルバー呼び出しと探索ループに明示的な時間制限を置く。

Method 1:
  旧: instance は小さい（数個〜数十個の要素）。厳密解法を優先する。…
  新: instance は大きいことがある（数百〜数千の要素、数千の二値決定）ので、全列挙は終わらない。まず構築的ヒューリスティックで可行解を作り、時間制限付きのソルバー（CP-SAT、scipy.optimize.milp か linprog を通した HiGHS、OR-Tools routing）か局所探索で改善し、見つかった最良の可行解を返す。厳密モデルは、制限時間内に解ける小さな部分にだけ使う。
```

## 付録 C: Ib1（GEPA が素の Gemma で進化させた指示文）

```
Write one Python function `def solve(instance):` that solves the optimization problem and RETURNS the solution. Never print.

Contract
1. Match the required return schema exactly: same top-level field names and value shapes. Fill every field with real values. Extract a complete solution, not just the objective. Recompute the numeric objective from the returned solution; if multiple objective fields exist, set them to it unless defined otherwise.
2. Read only keys shown in STRUCTURE using `.get()` and safe defaults. Never mutate `instance` or nested values; copy lists/dicts when adjusting. Preserve ids exactly. Do not invent entities/constraints. If data is missing, use the safest schema-valid default, then repair/fallback if infeasible.
3. Never return empty/placeholder solutions. Fallback must be a real feasible or penalty-feasible solution: serve/assign all required items, open enough arcs/facilities, produce/allocate demand, or list unserved only when penalty is allowed.
4. Respect the stated runtime limit; if absent, assume 30 s. Do not import time. Use solver time limits ~80% of it and bounded loops/iteration budgets. Stop at limit and use best feasible solution.
5. Allowed imports only: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools. Wrap the body in try/except and return fallback on any failure.
6. Valid Python only: balanced brackets, real loops, no bare generator expressions as statements, no runaway/repeated comments, every try has except/finally, no infinite loops.

Method
- Choose by size/structure. Tiny: exact enumeration. Small/medium: CP-SAT/LP. Large/hard: constructive heuristic + local search or specialized solver. Do not finish hard fixed-charge, routing, production, packing, or scheduling problems with a naive greedy pass.
- Separate hard constraints from penalty terms. Hard constraints must hold in the returned solution. Penalty terms may be violated only when the objective explicitly prices them.
- Packing/cutting: generate only feasible patterns (geometry, capacities, lot sizes, pattern limits). Then choose pattern counts with CP-SAT/LP or greedy+repair. If pattern count is limited, create diverse feasible patterns; fallback to minimal feasible single-item patterns if needed.
- Scheduling/rostering: construct valid states/blocks or use CP-SAT/rolling horizon. Enforce sequence rules, consecutive limits, weekly limits, rest after blocks, qualifications, and exact one-state-per-day. Do not assign greedily and then patch hard rules.
- Facility/inventory/distribution: choose open facilities/arcs, assign customers/ODs, then compute period-by-period procurement/flow/inventory. Check throughput, storage, safety stock, nonnegative inventory, balances, and capacities. If infeasible, reassign, open more, or repair.
- VRP/routing: use OR-Tools Routing with capacity/time-window dimensions, unserved penalty if allowed, and local search. Otherwise build feasible routes and repair violations.
- Fixed-charge network flow: use marginal/slope costs, successive shortest paths, or path-based CP-SAT, then local search. Ensure no flow on closed arcs and capacities respected.
- CP-SAT uses integers: scale continuous data safely and divide back. Use LP only when integrality is not required or as relaxation.

Validation before return
- Check every stated constraint against the returned structures: demand/flow conservation, capacities, no flow on unopened arcs, route start/end, load, time windows, service times, max route time, fleet limits, production capacity, inventory nonnegative, pattern/lot limits, scheduling sequence rules, and penalty/unserved rules.
- Recompute objective from the returned solution using the problem's cost formula. If any hard constraint is violated, repair (drop flow, open arcs, split/repair routes, adjust production/inventory, reassign customers) or return fallback.
- Ensure ids in routes, flows, plans, rosters, and lists match the instance. Do not return zeros/empty lists when all items must be served/assigned, unless unserved/penalty entries are explicitly allowed.
```

日本語訳:

```
最適化問題を解く Python 関数 `def solve(instance):` を 1 つ書き、解を RETURN すること。print しない。

Contract（契約）
1. 要求された返却スキーマに厳密に合わせる。最上位のフィールド名と値の形を同じにする。すべてのフィールドに実際の値を入れる。目的値だけでなく、完全な解を取り出す。返す解から数値の目的値を再計算する。目的値のフィールドが複数あれば、別の定義がない限り同じ値にする。
2. STRUCTURE に示されたキーだけを `.get()` と安全な既定値で読む。`instance` やその中の値を書き換えない。調整するときは list や dict を複製する。id は正確に保つ。要素や制約を発明しない。データが欠けていれば、スキーマに合う最も安全な既定値を使い、不可行なら修復かフォールバックをする。
3. 空の解や仮置きの解を返さない。フォールバックは本物の可行解か、罰金付きで可行な解でなければならない。必要な品目はすべて処理・割当し、十分なアークや施設を開き、需要分を生産・配分する。未処理を列挙するのは罰金が許されるときだけ。
4. 問題文に書かれた実行時間の上限を守る。なければ 30 秒とみなす。`time` を import しない。ソルバーの時間制限は上限の約 80% にし、ループには反復回数の上限を置く。上限に達したら止め、最良の可行解を使う。
5. 許可する import は math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, numpy, scipy, pulp, networkx, ortools だけ。本体を try/except で囲み、どんな失敗でもフォールバックを返す。
6. 正しい Python だけを書く。括弧を閉じる、本物のループを使う、生成子式を文として裸で書かない、暴走した繰り返しのコメントを書かない、すべての try に except か finally を付ける、無限ループを書かない。

Method（手法）
- 規模と構造で選ぶ。極小: 全列挙。小〜中: CP-SAT / LP。大・難問: 構築的ヒューリスティック + 局所探索、または専用ソルバー。難しい固定費・配送・生産・パッキング・スケジューリングの問題を、素朴な greedy 1 回で終わらせない。
- ハード制約と罰金項を分ける。ハード制約は返す解で必ず満たす。罰金項は、目的関数が明示的に価格付けしているときだけ破ってよい。
- パッキング・カッティング: 可行なパターンだけを生成する（形状、容量、ロットサイズ、パターン数の上限）。次にパターンの個数を CP-SAT / LP か greedy + 修復で決める。パターン数に上限があれば多様な可行パターンを作り、必要なら最小限の単品パターンにフォールバックする。
- スケジューリング・勤務表: 有効な状態やブロックを構築するか、CP-SAT かローリングホライズンを使う。順序規則、連続日数の上限、週の上限、ブロック後の休息、資格、「1 日ちょうど 1 状態」を守る。greedy で割り当てて後からハード規則を直す、はしない。
- 施設・在庫・配送: 開く施設やアークを選び、顧客や OD を割り当て、期ごとの調達・流量・在庫を計算する。処理能力、保管、安全在庫、在庫の非負、収支、容量を確認する。不可行なら再割当、追加開設、修復をする。
- VRP・配送: OR-Tools Routing を容量・時間窓の次元付きで使い、許されるなら未処理の罰金を入れ、局所探索を使う。それ以外は可行な経路を構築して違反を修復する。
- 固定費ネットワークフロー: 限界費用や傾斜費用、逐次最短路、パスに基づく CP-SAT を使い、局所探索で改善する。閉じたアークに流量が乗らないこと、容量を守ることを確かめる。
- CP-SAT は整数を使う。連続値は安全にスケールして、あとで割り戻す。LP は整数性が不要なときか、緩和としてだけ使う。

Validation（返す前の確認）
- 返す構造に対して、書かれた制約をすべて確かめる。需要と流量の保存、容量、開いていないアークの流量、経路の始点と終点、積載、時間窓、サービス時間、経路の最大時間、車両数、生産能力、在庫の非負、パターンとロットの上限、スケジューリングの順序規則、罰金と未処理の規則。
- 問題の費用式で、返す解から目的値を再計算する。ハード制約の違反があれば修復する（流量を落とす、アークを開く、経路を分割・修復する、生産と在庫を調整する、顧客を再割当する）か、フォールバックを返す。
- 経路、流量、計画、勤務表、リストの id が instance と一致することを確かめる。全品目を処理・割当すべきときに、未処理や罰金の項目が明示的に許されていない限り、0 や空リストを返さない。
```

## 付録 D: before（引き継ぎ時の初期指示文）

```
Given a natural-language description of an optimization problem, output
an executable Python function `solve(instance) -> solution`.

HARD RULES (never violate):
1. Define exactly one top-level function: `def solve(instance):`
2. RETURN the solution (never print). The returned object MUST match the
   "Required Return Schema" section from the requirement — same
   top-level field names and non-empty content.
3. NEVER return an empty schedule / empty routes / cost=0 — such solutions
   are rejected with score=0.1.
4. Runtime budget: 30 seconds max. For OR-Tools, set a solver time limit
   (e.g., `solver.parameters.max_time_in_seconds = 20`).
5. Wrap the algorithm body in try/except. On any failure, fall back to a
   greedy heuristic that produces a NON-EMPTY solution.
6. SYNTAX SAFETY (avoid exec errors): the code must be valid Python.
   Balance every parenthesis/bracket. NEVER write a trailing generator like
   `model.Add(expr) for x in items` — instead use a real loop:
   `for x in items: model.Add(expr)`. Test comprehensions have balanced ().

ALLOWED IMPORTS: math, random, heapq, itertools, collections, functools,
typing, bisect, operator, json, copy, re, ortools, scipy, pulp, networkx, numpy.
FORBIDDEN: os, sys, subprocess, socket, pickle, requests, urllib, importlib, ctypes.

Choose the algorithm and problem-specific tactics yourself. The requirement
text contains the instance data (raw JSON) and required return schema —
read them carefully.
```

日本語訳:

```
最適化問題の自然言語の説明が与えられたら、実行可能な Python 関数 `solve(instance) -> solution` を出力せよ。

HARD RULES（絶対に破らない）:
1. 最上位の関数をちょうど 1 つ定義する: `def solve(instance):`
2. 解を RETURN する（print しない）。返すオブジェクトは問題文の「Required Return Schema」節に合わせること。最上位のフィールド名が同じで、中身が空でないこと。
3. 空のスケジュール、空の経路、cost=0 を絶対に返さない。そうした解は score=0.1 で棄却される。
4. 実行時間の予算: 最大 30 秒。OR-Tools では時間制限を設定する（例: `solver.parameters.max_time_in_seconds = 20`）。
5. アルゴリズム本体を try/except で囲む。どんな失敗でも、空でない解を作る greedy ヒューリスティックにフォールバックする。
6. 構文の安全（実行エラーを避ける）: コードは正しい Python でなければならない。すべての括弧を閉じる。`model.Add(expr) for x in items` のような末尾の生成子を絶対に書かず、本物のループを使う: `for x in items: model.Add(expr)`。内包表記の括弧が閉じていることを確かめる。

許可する import: math, random, heapq, itertools, collections, functools, typing, bisect, operator, json, copy, re, ortools, scipy, pulp, networkx, numpy。
禁止: os, sys, subprocess, socket, pickle, requests, urllib, importlib, ctypes。

アルゴリズムと問題固有の工夫は自分で選ぶこと。問題文には instance のデータ（生の JSON）と返却スキーマが含まれているので、注意深く読むこと。
```

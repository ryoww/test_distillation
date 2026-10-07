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
- Ensure ids in routes, flows, plans, rosters, and lists match the instance. Do not return zeros/empty lists when all items must be served/assigned, unless unserved/penalty entries are explicitly allowed.```

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

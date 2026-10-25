# Held-out test: the learner on ten formulas none of its rules were written for

<!-- PLAIN -->
## In plain words

**The question.** The bar-3 learner was built and tuned on a handful of formulas. Does it invent programs for
formulas none of its rules were written for? It was frozen with every rule that held up: the depth-2 law book, the
new sources, the guided second search, clean routines and the subgoal. Ten new formulas, with a prediction for each,
were written down (`PREREG.md`) before any run on them. Both of its rules for choosing practice were run:
- hardest first;
- anchored pairs.

Each run got at most 5 rounds, from scratch, as in training. The ten formulas: (x² + x), (x + 1)², 3·x², x⁴, (x³ + x)
and 2^x, all mod N; Lucas and Tribonacci mod N; floor(x²/N); and the triangular numbers x·(x + 1)/2 mod N.

**What came.** Five of the ten formulas, in 9 of the 20 runs, every one in round 0 or 1. The search sees only N ≤ 7,
and a program is accepted once it is right up to N = 40. Every program found is right on all 7,260 pairs x < N ≤ 120.

| formula | hardest first | anchored pairs |
|---|---|---|
| (x² + x) mod N | round 1 | round 0 |
| (x + 1)² mod N | round 1 | round 1 |
| 3·x² mod N | round 1 | round 1 |
| x⁴ mod N | round 1 | round 1 |
| 2^x mod N | no, 5 rounds | round 1 |
| (x³ + x), Lucas, Tribonacci, floor(x²/N), triangular | no, 5 rounds | no, 5 rounds |

**What did the work.** Read from the records: where each routine of each program came from.
- "Subtract N once if you can", from the law book, was practised in round 0 in every run that came. Every program
  reduces mod N with it.
- x² mod N is the piece inside 3·x², x⁴ ("square the square") and (x + 1)². For (x + 1)² it came by a mirror: a near
  miss leaves N − 1 − x, and the rest of the way from there is that number's square mod N, since (N − a)² ≡ a².
- The law book gave (x² + x): x·(x + 1) under anchored pairs, and (x − 1)·x applied to x + 1 under hardest first.
- The recurrence source gave 2^x: its step is "double, subtract N once".
- Routines and merges (the pieces a round's programs share) kept the programs short: 2 to 5 tokens each, 12 to 24
  primitive instructions written out.

**The subgoal.** It changed nothing in 12 of the 20 runs, four of the wins among them. The other eight were run again
without it, as a control added after the test:
- 3·x² under hardest first and x⁴ under anchored pairs come only with it: without it, nothing in 5 rounds.
- x⁴ under hardest first comes in round 1 with it, and in round 3 without it.
- In those three, what the subgoal gave was x² mod N in round 1, which only its deeper probe solved.
- 3·x² under anchored pairs comes in round 1 either way. Without the subgoal it is a shorter program: add 3 and
  subtract N once, x·x times.
- (x² + x) under hardest first comes with the same program either way.
- Lucas and floor(x²/N) do not come either way.

**What did not come, and why.**
- (x³ + x) is two laws away from x² mod N, so the score did not single that out. It picked a mirror, which was never
  solved. Hardest first got x² mod N in round 3, but not the rest of the way (times x, plus x, reduce).
- 2^x under hardest first: its step is cheap, and hardest first takes the costliest problems.
- Lucas: Fibonacci's step was learnable from round 1. Anchored pairs never picked it, and hardest first picked it only
  in its last round, where the target search found nothing. Lucas starts at (2, 1), which needs reducing when N ≤ 2,
  so its program is longer than Fibonacci's.
- Tribonacci: the recurrence source poses only two-number steps.
- floor(x²/N) needs division by repeated subtraction, with a count, which the search did not reach.
- The triangular numbers are the clearest gap. Every piece is there, but no source poses their step, y(x + 1) from
  y(x) and x.

**The predictions.** 16 of the 20 held, including all 8 made with medium or high confidence. The four misses:
- 2^x under hardest first, the risk the prediction named;
- the triangular numbers under both rules, whose expected route was not there;
- Lucas under anchored pairs, which never picked the step.

**In one line.** The frozen learner invents half the held-out formulas, all in its first two rounds, each right far
past the sizes it searched, and the subgoal's deeper probe is needed for three of its nine wins. What it misses needs
either a step no source poses (the triangular numbers' running sum, Tribonacci's three numbers) or a program longer
than its search reaches (Lucas, floor(x²/N), (x³ + x)).

**Cost.** $11.25 in all, against a soft limit of $15 and a cap of $20:
- $7.68 for the test (68.0 search-hours);
- $3.57 for the control (25.5 new search-hours).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md, repeated here unchanged (its sha256 is in §8).

Written 2026-10-25 05:15 PDT, before any search, score or run on these formulas, and not edited after. The only
thing computed on them beforehand is that each can be posed (below). HELDOUT.md §1 repeats this text unchanged.

### The learner, frozen

- `bar3/scale/subgoal/learn.py` with `lp_clean_deep_sub` (hardest first) and `ct2a2_clean_deep_sub` (anchored pairs)
  from `bar3/scale/subgoal/configs.py`. That is every general rule that held up: the depth-2 law book, the new
  sources (recurrences, constants, the rest of the way from each routine), the guided second target search, clean
  routines, and the subgoal (its score, the deeper probe, its ties).
- Not on: the chain (`_subchain`: it changed 3 of 20 runs and brought nothing) and the recursive subgoal (stopped at
  its tiny test, SUBGOAL.md §8; never built into the learner).
- Code as of git `97f30b9` (unchanged at `542b69b`). File hashes (sha256, first 16 hex digits):
  `learn.py` f88a69cd6570e2b8, `configs.py` cc39ab3b431699c0, `score.py` 05220f0c82b984fd,
  `bar3/scale/learn.py` 6adfee79bedf1080, `bar3/scale/sources.py` a33bfb0fdfdbce6b, `bar3/problems.py` bbd1b2f38132023c.
- Unchanged during the test: no rule, config or budget changes. Search at x < N ≤ 7, checked at N ≤ 10, general to
  N = 40, at most 5 rounds, seed 0; every invented program rechecked on every pair up to N = 120 (7,260 pairs).
- The formulas are data: `bar3/scale/heldout/targets.py` adds them to the target table the learner reads.

### The formulas, and whether they can be posed

All ten can be posed: each answer lies below N, so below the machine's caps (98 at N ≤ 7, 200 at N ≤ 10), the answers
vary, and each is a function of (x, N). Checked on the data only (`P.the_target` at both sizes; answers up to N = 120).

| # | name | definition |
|---|---|---|
| 1 | `x2x` | (x² + x) mod N |
| 2 | `x1sq` | (x + 1)² mod N |
| 3 | `three` | 3·x² mod N |
| 4 | `x4` | x⁴ mod N |
| 5 | `x3x` | (x³ + x) mod N |
| 6 | `pow2` | 2^x mod N |
| 7 | `lucas` | L(x) mod N; L(0) = 2, L(1) = 1, L(x + 2) = L(x) + L(x + 1) |
| 8 | `trib` | T(x) mod N; T(0) = T(1) = 0, T(2) = 1, T(x + 3) = T(x) + T(x + 1) + T(x + 2) |
| 9 | `floorsq` | floor(x² / N) |
| 10 | `tri` | x·(x + 1)/2 mod N |

### Predictions

Each is "the round it comes in, or no in 5 rounds", for each rule, with the mechanism expected to do the work and a
confidence. They come from reasoning about the programs and the learner's signals, not from any run. Two facts from
the training targets' records that they lean on: the target search reaches about 6 tokens (x² mod N came in round 0
as `L.X[L.X[INC.A]] L.N[MOV.X.A T.mod(X,N)@x+qN]`), and "subtract N" (`T.mod(X,N)@x+qN`) is usually practised in
round 0. "Reduce fully" below means a routine or merge for `L.N[MOV.X.A T.mod…]` or `L.A[MOV.X.A T.mod…]`.

| # | formula | hardest first | anchored pairs | expected mechanism, or why not | confidence |
|---|---|---|---|---|---|
| 1 | (x² + x) mod N | round 1–2 | round 0–1 | No subgoal: several candidates are one exact law away (x² mod N and x² by "add X, reduce"; x·(x + 1) by "reduce"; x + 1 by "times X, reduce"), all with table 0, so the score ties. Anchored pairs: the book's x·(x + 1) with "subtract N" makes the target 4 tokens, which the look sees, as x² + 1 came in round 0. Hardest first skips x·(x + 1) (cost 4, below its picks) but finds "x·(x + 1), reduce fully" (5 tokens) once that reduction is a routine. | low / medium |
| 2 | (x + 1)² mod N | round 1–3 | round 1–3 | No subgoal: x + 1 and (x + 1) mod N both pass mod(mul(g, g), N), table 0, a tie; x² mod N is not seen. INC.X then x² mod N's program is 7 tokens, one past the target search; 5 once "reduce fully" is a routine. | low |
| 3 | 3·x² mod N | round 1–2 | round 1–4 | The subgoal: x² mod N, the one candidate with an exact law, mod(mul(g, 3), N), with the 3 the data shows (x² itself, times 3, goes over the cap), and table 24. The deeper probe solves it in round 1 once "subtract N" is a routine; then three rounds of "add x² mod N and reduce" (6 tokens), as 10·x² came. | medium / low |
| 4 | x⁴ mod N | round 1 | round 1 | The subgoal: x² mod N, by mod(mul(g, g), N), table 24; deeper probe in round 1; then "square the square", `T.law MOV.X.A CLR.A T.law` (4 tokens), as x³ came. | medium |
| 5 | (x³ + x) mod N | no | no | x·g + x is two laws from g = x² mod N, and no table sees it (x and N − x give opposite answers), so the score does not single out x² mod N. Without x² mod N's routine, the loop needs "add x², reduce" as one routine, which nothing makes. | medium |
| 6 | 2^x mod N | round 1–4 | round 1–3 | Its step, "double mod N", is posed twice: by the book (2X mod N) and by the recurrence source (order 1). It costs about 5 once "subtract N" is a routine; then "x times double", with a final reduction for N = 1 (5–6 tokens). Hardest first takes the step only if it is among its four costliest picks, in pool order. | low |
| 7 | Lucas | no | round 2–4 | Fibonacci's two-number step, posed by the recurrence source (Fibonacci: anchored pairs round 1, hardest first round 4). But the start (2, 1) has to be reduced at N ≤ 2, which makes the program a few tokens longer than Fibonacci's `INC.U L.X[step]`. | low |
| 8 | Tribonacci | no | no | The recurrence source stops at two numbers. The three-number step is never posed, and a program from scratch is far out of reach. | high |
| 9 | floor(x²/N) | no | no | It is itself a depth-2 law (div(mul(X, X), N)), so it is dropped from the pool as the target, and the score's law view is off; no table sees it. Dividing by counting subtractions needs a division routine (the laws' X div N), which comes late if at all. | low |
| 10 | x·(x + 1)/2 mod N | round 1–3 | round 0–2 | The plain sum loop `L.X[L.X[INC.A] DEC.X]` (4 tokens) then "reduce fully": 5 tokens once that is a routine. Anchored pairs: the loop is a likely near miss of the target probe (right wherever the sum is below N), its value has the one exact law (mod(g, N)), and with "subtract N" it makes the target 4 tokens. | low |

In all: hardest first 6 of 10 (1, 2, 3, 4, 6, 10), anchored pairs 7 of 10 (1, 2, 3, 4, 6, 7, 10).
- The subgoal is expected to do the work for 3 and 4.
- Routines carried within a run ("subtract N", "reduce fully") and the target search are expected to do it for 1, 2
  and 10.
- The sources are expected to do it for 6 and 7: the book and the recurrence source.
- 5, 8 and 9 are expected to fail.

### What is reported

For each formula and rule:
- the round it came in, or "no, 5 rounds";
- the program, rechecked on every pair up to N = 120;
- which mechanism did the work, read from the records. That means where each routine of the program came from (the
  source and round of the practised problem), whether the subgoal's deeper probe solved one of them, and a replay of
  the run on its stored searches without the subgoal (`attribute.py`).

Each prediction is then marked right or wrong. Budget: soft $15, cap $20; every launch guarded by `spend.py 4750`.

### 2. The early check: what the score picks in round 0 (`rank.py`, `records/rank0.json`)

Run after the pre-registration and before the runs: round 0's pool for each formula (the learner's own round-0
target probe, run locally), scored by the subgoal score, as SUBGOAL.md §2 did for the training formulas. The runs'
logs show the same subgoals in their round 0.

| formula | the subgoal in round 0 | its law, its table | x² mod N's place | the prediction about the score |
|---|---|---|---|---|
| (x² + x) mod N | (x + N)·x (`law2/mul(add(X,N),X)`) | mod(add(g, X), N); 9 rows by (g, x) | 2nd | wrong: predicted a tie |
| (x + 1)² mod N | none: x + 1, N − (x + 1) and (x + 1) mod N tie | mod(mul(g, g), N); 0 | 4th | right |
| 3·x² mod N | x² mod N | mod(mul(g, 3), N); 24 by (g, N) | 1st | right |
| x⁴ mod N | x² mod N | mod(mul(g, g), N); 24 by (g, N) | 1st | right |
| (x³ + x) mod N | a near miss's remainder, from X = what `L.X[DEC.N MOV.A.N]` leaves (N − x) | mod(absdiff(g, N), N); 21 by (g, N) | 16th | right that x² mod N is not it |
| 2^x mod N | none: 114 tie at 0 | – | (in the tie) | – |
| Lucas | N·\|x − 1\| (`law2/absdiff(mul(X,N),N)`) | no law; 8 by (g, x) | 3rd | – |
| Tribonacci | none: 2 tie at 33 by (g, x) | – | 10th | – |
| floor(x²/N) | (x − 1)·N (`law2/mul(pred(X),N)`) | no law (the view is off: the formula is div(mul(X, X), N)); 17 by (g, x) | 25th | right that the law view is off |
| x·(x + 1)/2 mod N | none: 2 tie at 24 by (g, x) | – | 19th | – |

Two kinds of pick here are not steps toward the formula, and the score cannot tell them apart from one that is:
- *A mirror.* For (x³ + x) mod N, the near miss's remainder posed on X = N − x is the formula reflected: its answer
  is N minus the formula's, mod N.
- *A stand-in for N.* For Lucas and floor(x²/N), the pick is a multiple of N that, grouped with x, pins down N
  itself. It then "determines" the formula on the rows where the formula does not depend on N (x = 0 and x = 1),
  which is the formula's own structure, not the pick's.

The sum loop that prediction 10 counted on, `L.X[L.X[INC.A] DEC.X]`, is not among the target probe's near misses: the
best of them fits 17 of 28 at cost 3.

### 3. The runs

- Launched 2026-10-25 at 05:19 PDT (12:19 UTC), after the pre-registration (05:15) and the early check. The app
  was `ap-GqOjlw7uaIW2SZPUCJNZWw`, on four containers (16 CPUs and 64 GiB each, a worker process per CPU), each running
  five runs as threads that share its searches. Spend before the launch: $3,414.66 (`spend.py 4750`: launch OK).
  - Container 0: hardest first on (x² + x), 3·x², (x³ + x), Lucas and floor(x²/N).
  - Container 1: anchored pairs on the same five.
  - Container 2: hardest first on (x + 1)², x⁴, 2^x, Tribonacci and the triangular numbers.
  - Container 3: anchored pairs on the same five.
- Two restarts, both replayed from the container's store:
  - Container 0 was preempted at 12:35:58 UTC (its log: "Container terminated due to preemption") and started again
    at 12:36:07.
  - Container 1's records start again at 13:00:40 UTC. The logs kept do not say why: for a stopped app, Modal keeps
    only the tail of the log.
  - A restarted container truncates its records and replays its runs from the store. It searches only what the store
    does not hold: the jobs cut off by the restart. Each container had one retry; both finished.
- All four containers returned normally, each with its five runs; the largest worker reached 10.0–11.9 GB. The last
  finished at 13:58 UTC, and the app stopped by itself. The spend watcher (`watch_spend.py`, stop at $18) never had to
  stop it.
- Every run, replayed on the stores (`attribute.py`, §5.2), gives back its record exactly, the restarted containers'
  runs included.
- 68.0 search-hours (hardest first 28.6, anchored pairs 39.4) and $7.68 billed (the hours 12:00 and 13:00 UTC).
- The control (§5.3) was launched at 07:15 PDT (14:15 UTC), after the outcomes were read. It ran as two apps of one
  container each (16 CPUs, 64 GiB). Each was launched after `spend.py 4750` (launch OK) and had a spend watcher at
  $3.50, which never had to stop it.
  - `ap-LcCT6mNn4rLbbHHhUmsOHr`, hardest first: preempted at 15:05 UTC (its log: "Container terminated due to
    preemption"). It started again, replayed from its store and finished at 15:27.
  - `ap-Q5nWfEKrXrB2mgbAGjmp61`, anchored pairs: no restart; it finished at 15:53.
  - 25.5 new search-hours (8.3 and 17.2) and $3.57 billed ($1.52 and $2.05), all of both apps' hours
    (`cost.py`; `logs/cost_ctl1.txt`). The records count 36.4 search-hours, because they include the test's stored
    searches, which the control reused.
- The test and the control cost $11.25 together, against a soft limit of $15 and a cap of $20.

### 4. Outcomes (`verify.py`; `logs/verify_ho1.txt`)

| formula | hardest first | anchored pairs |
|---|---|---|
| (x² + x) mod N | round 1: `INC.X T.law8 L.X[MOV.X.A T.mod]` (14) | round 0: `M0.4 L.N[MOV.X.A T.mod]` (12) |
| (x + 1)² mod N | round 1: `INC.X T.res5` (12) | round 1: `INC.X M1.2` (12) |
| 3·x² mod N | round 1: `T.law5 L.X[INC.A INC.A] M1.1` (22) | round 1: `M1.3 L.X[T.res8 MOV.X.A]` (21) |
| x⁴ mod N | round 1: `T.law5 T.law8 M1.1` (24) | round 1: `M1.2 CLR.A M1.2` (23) |
| (x³ + x) mod N | no, 5 rounds | no, 5 rounds |
| 2^x mod N | no, 5 rounds | round 1: `INC.A L.X[T.rec6] MOV.X.A M0.2` (20) |
| Lucas | no, 5 rounds | no, 5 rounds |
| Tribonacci | no, 5 rounds | no, 5 rounds |
| floor(x²/N) | no, 5 rounds | no, 5 rounds |
| x·(x + 1)/2 mod N | no, 5 rounds | no, 5 rounds |

- `T.mod` stands for `T.mod(X,N)@x+qN`, "subtract N once if you can", from X into A.
- In brackets: the program's size in primitive instructions, with its routines expanded.
- Every program found is right on all 7,260 pairs up to N = 120 (cap 57,600, 10⁷ steps).
- All nine came from the plain target search. The guided second search ran 63 times, in the rounds where the plain
  one found nothing, and never found a program.

Search-hours per run:

| | (x² + x) | (x + 1)² | 3·x² | x⁴ | (x³ + x) | 2^x | Lucas | Tribonacci | floor(x²/N) | triangular |
|---|---|---|---|---|---|---|---|---|---|---|
| hardest first | 0.37 | 0.75 | 0.48 | 0.86 | 5.64 | 4.04 | 5.41 | 4.52 | 2.81 | 3.77 |
| anchored pairs | 0.28 | 1.10 | 0.71 | 1.18 | 8.06 | 2.06 | 7.04 | 5.98 | 5.30 | 7.66 |

### 5. What did the work

#### 5.1 The programs, read from the records (`mechanism.py`; `logs/mechanism_ho1.txt`)

Below, g is x² mod N. For each routine: what it computes, the problem it was practised as (its source is the prefix of
the name), and the round.

- **(x² + x), hardest first, round 1**: `INC.X T.law8 L.X[MOV.X.A T.mod]`.
  - It computes x + 1, times x, then x + 1 rounds of `T.mod`.
  - T.law8 = (X − 1)·X is the law book's `law2/mul(pred(X),X)`, practised in round 1. `T.mod` is the book's
    `law/mod(X,N)/x+qN`, practised in round 0.
  - The reduction is in the program, not the "reduce fully" routine the prediction expected.
- **(x² + x), anchored pairs, round 0**: `M0.4 L.N[MOV.X.A T.mod]`.
  - It computes x + x·x, then N rounds of `T.mod`.
  - M0.4 is a merge of round 0, from the book's x·(x + 1) (`law2/mul(succ(X),X)`, practised in round 0 as
    `MOV.A.X L.X[L.X[INC.A]]`).
  - These are the 4 tokens the prediction expected.
- **(x + 1)², hardest first, round 1**: `INC.X T.res5`.
  - It computes x + 1, then g's program: T.res5 = `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod]`.
  - T.res5 was practised in round 1 as `restrict/remainder[M0.4 -> N]/3<X`. M0.4, a near miss of the target probe,
    leaves N − 1 − x in N. The rest of the way from that value is its square mod N, since (N − a)² ≡ a² (mod N). The
    problem is restricted to inputs above 3.
  - So x² mod N came as the remainder of a mirror. That is the same kind of pick that was no use as (x³ + x)'s
    subgoal (§2).
- **(x + 1)², anchored pairs, round 1**: `INC.X M1.2`.
  - M1.2 = `L.X[M1.1] L.A[MOV.X.A T.mod]` computes x², then x² rounds of `T.mod`. It is a merge of round 1.
  - The same mirror did it. `restrict/remainder[M0.3 -> N]/2<X` (M0.3 also leaves N − 1 − x) was practised in
    round 1 as T.res6, g's program. The target search found `INC.X T.res6` at the same cost.
- **3·x², hardest first, round 1**: `T.law5 L.X[INC.A INC.A] M1.1`.
  - It computes g, adds twice g, then reduces fully.
  - T.law5 is the subgoal `law2/mod(mul(X,X),N)`, solved only by its deeper probe in round 1. M1.1 =
    `L.A[MOV.X.A T.mod]` is a merge of round 1.
  - As predicted, except the last step: "add twice g, reduce" is one program here, not three rounds.
- **3·x², anchored pairs, round 1**: `M1.3 L.X[T.res8 MOV.X.A]`.
  - It computes g, then repeats "add 2, subtract N once" g times: g + 2g.
  - M1.3 (x², reduce fully) is a merge of round 1. T.res8 = `INC.X INC.X T.mod` was practised in round 1 as
    `restrict/target/X·X<X+X`: the target on x = 1, which is 3 mod N.
  - The subgoal x² mod N was practised in round 1: the usual probe solved it, and it was picked. The target search
    found `T.law5 L.X[T.res8 MOV.X.A]`, with the subgoal's routine in place of M1.3, at the same cost.
- **x⁴, hardest first, round 1**: `T.law5 T.law8 M1.1`.
  - It computes g, then g + (g − 1)·g = g², then reduces fully: "square the square", as predicted.
  - T.law5 is the subgoal, solved by its deeper probe. T.law8 = (X − 1)·X is from the book, practised in round 1.
- **x⁴, anchored pairs, round 1**: `M1.2 CLR.A M1.2`.
  - It computes g, then g's square mod N. M1.2 (x², reduce fully) is a merge of round 1.
  - The subgoal x² mod N was solved by its deeper probe in round 1 and practised as T.law6. The target search found
    `T.law6 CLR.A M1.2` at the same cost.
- **2^x, anchored pairs, round 1**: `INC.A L.X[T.rec6] MOV.X.A M0.2`.
  - It starts at 1, repeats "double, subtract N once" x times, then subtracts N once more for N = 1. As predicted, in
    4 tokens.
  - T.rec6 = `MOV.X.A M0.4 T.mod` is the recurrence source's `recur1/A` (y(x + 1) from y(x)), practised in round 1
    at cost 3.
  - M0.4 = `INC.A L.X[INC.X MOV.A.X]` doubles, and M0.2 makes the last subtraction. Both are merges of round 0.

"Subtract N once" was practised in round 0 in all nine runs that came, as the book's `law/mod(X,N)/x+qN`.

#### 5.2 The replays without the subgoal (`attribute.py`; `logs/attribute_ho1.txt`)

Each run was replayed on its container's stored searches twice:
- as it ran: all 20 give back their records;
- without the subgoal: no score, no deeper probe, and pool order for every tie. The replay stops at the first search
  no store holds, and that round counts as different.

Without the subgoal:
- 12 runs are the same in every round:
  - under anchored pairs: (x² + x), (x + 1)², 2^x, (x³ + x), Tribonacci and the triangular numbers;
  - under hardest first: (x + 1)², 2^x, (x³ + x), Tribonacci, the triangular numbers and floor(x²/N).
  - Four of the nine wins are among them. The subgoal had nothing to do with (x² + x), (x + 1)² or 2^x under
    anchored pairs, nor with (x + 1)² under hardest first.
- 5 runs differ from round 1, the round their target came in: 3·x² and x⁴ under both rules, and (x² + x) under
  hardest first.
- Lucas differs from round 0 under both rules: its subgoal is the stand-in for N (§2).
- floor(x²/N) under anchored pairs differs from round 3.

What the subgoal changed in that round (`picks_without.py`; `logs/picks_without_ho1.txt`). This is the same replay,
except that a target search no store holds finds nothing, so the round reaches its picks.

| run, round | picked with the subgoal | without it |
|---|---|---|
| (x² + x), hardest first, 1 | **(x + N)·x**, \|x − N\|, N mod x, (x − 1)·x | \|x − N\|, N mod x, (x − 1)·x, N div (x − 1) |
| 3·x² and x⁴, hardest first, 1 | **x² mod N**, \|x − N\|, N mod x, (x − 1)·x | \|x − N\|, N mod x, (x − 1)·x, N div (x − 1) |
| Lucas, hardest first, 0 | **N·\|x − 1\|**, x mod N, [x + 1 < N], min(2x, N) | x mod N, [x + 1 < N], min(2x, N), [2x < N] |
| floor(x²/N), anchored pairs, 3 | **(x − 1)·N**, gcd(2x, N), x² mod N, x + N | x + N, gcd(2x, N), x² mod N, 2x |

- The subgoal is in bold. In the second row, x² mod N was solved only by its deeper probe. Without the subgoal it
  stays unsolved, and hardest first takes the next cost-5 problem in pool order.
- Under anchored pairs, the picks for 3·x², x⁴ and Lucas cannot be read this way. Without the subgoal the learner
  asks to look at pairs it never looked at with it, and no store holds those searches. The control (§5.3) runs them.

#### 5.3 The control: the same runs without the subgoal (`records/*_ctl1.jsonl`; `logs/*_ctl1.txt`)

This was added after the test and was not pre-registered. The replays (§5.2) stop at the first search the subgoal
changed, so they cannot say what the learner does without it from there. The control runs it. It used the frozen
configs less their `subgoal`: `lp_clean_deep` and `ct2a2_clean_deep`, which `modal_heldout.py --control` checks are
exactly that. Each started from scratch, on the eight runs the subgoal changed:
- hardest first on (x² + x), 3·x², x⁴ and Lucas;
- anchored pairs on 3·x², x⁴, Lucas and floor(x²/N).

Everything else was the test's: the budgets, 5 rounds, seed 0 and the check to N = 120. Each rule's four runs shared a
container, which started from that rule's test stores joined, so it searched only what the test had not.

| formula, rule | with the subgoal (§4) | without it | the subgoal's part |
|---|---|---|---|
| (x² + x), hardest first | round 1 (14) | round 1, the same program (14) | none |
| 3·x², hardest first | round 1 (22) | no, 5 rounds | it made the formula come |
| x⁴, hardest first | round 1 (24) | round 3: `M3.3 M3.3` (36) | two rounds sooner |
| Lucas, hardest first | no, 5 rounds | no, 5 rounds | none |
| 3·x², anchored pairs | round 1 (21) | round 1: `MOV.U.X L.X[L.U[T.val8]]` (13) | another program, the same round |
| x⁴, anchored pairs | round 1 (23) | no, 5 rounds | it made the formula come |
| Lucas, anchored pairs | no, 5 rounds | no, 5 rounds | none |
| floor(x²/N), anchored pairs | no, 5 rounds | no, 5 rounds | none |

- In brackets: the program's size in primitive instructions, as in §4.
- Every program is right on all 7,260 pairs up to N = 120 (`logs/verify_ctl1.txt`).
- Every run, replayed on its stores, gives back its record (`logs/attribute_ctl1.txt`). That log's other half,
  "without the subgoal", says nothing here: these configs have no subgoal.

What happened without the subgoal (`logs/mechanism_ctl1.txt`):
- **(x² + x), hardest first.** Round 1 picks |x − N|, N mod x, (x − 1)·x and N div (x − 1), as the replay showed
  (§5.2). The target search finds `INC.X T.law7 L.X[MOV.X.A T.mod]`. That is the test's program; T.law7 is the same
  routine, (X − 1)·X, under another number.
- **3·x², hardest first.**
  - x² mod N becomes learnable in round 2, at cost 5, once (X − 1)·X is a routine:
    `MOV.A.X T.law7 L.X[MOV.X.A T.mod]`. In round 2 it loses on pool order to four other cost-5 problems. Hardest
    first picks it in round 3, among its four costliest.
  - No program comes in rounds 3–4. The test's program, `T.law5 L.X[INC.A INC.A] M1.1`, used two things from round 1:
    - x² mod N, from the deeper probe;
    - the merge M1.1 = `L.A[MOV.X.A T.mod]`, "reduce fully". The merge step drew it from round 1's programs, which
      included x² mod N's.
  - Without the subgoal, the merge step took N mod x's whole program instead: M1.3 =
    `MOV.A.N MOV.N.X L.A[MOV.X.A T.mod]`. Its first moves overwrite the sum.
  - With the control's routines, the test's program is 7 tokens long. The target search covers 5.
- **x⁴, hardest first.**
  - The same x² mod N, picked in round 3, brings x⁴ in that round as `M3.3 M3.3`: "square the square".
  - M3.3 = `MOV.A.X T.law7 MOV.X.N MOV.N.A M1.3` is a merge of round 3, from x² mod N's programs. It computes x²,
    moves it into N and N into X, and lets N mod x's merge M1.3 reduce it.
  - The target search also found `T.law16 T.law16`, with the practised x² mod N, at the same cost.
- **Lucas, hardest first.** No two-number step is picked in 5 rounds; with the subgoal, two were, in round 4. Nothing
  comes either way.
- **3·x², anchored pairs.**
  - x² mod N is learnable in round 1 here too (cost 6, by the usual probe), but without the tie-break it is not
    picked.
  - Among the picks is the value source's `value/T.con2 MOV.X.A M0.1 -> A`: the value A holds after the start of a
    near miss, which is 3 less N once. T.con2 (3 → A) is a constant of round 0.
  - It is practised as T.val8 = `T.con2 MOV.X.A T.mod`. The target search then finds "add 3, subtract N once, x·x
    times": `MOV.U.X L.X[L.U[T.val8]]`, at cost 4. That is 13 primitive instructions, against the test's 21.
- **x⁴, anchored pairs.**
  - x² mod N stays unsolved in all 5 rounds. Each usual probe stops at its level 5 (over 1M states), one level short
    of x² mod N's 6 tokens. With the subgoal, the deeper probe (10M states) solved it in round 1.
  - In the 3·x² run, the usual probe did solve it in round 1. Round 0 had left 5 routines there and 7 here, and with
    fewer routines the probe's levels are smaller: it reached level 6.
- **Lucas, anchored pairs.** Fibonacci's two-number steps (`recur2/stepUA`, …) are learnable at cost 5 from round 1,
  as with the subgoal. They are never picked, and nothing comes.
- **floor(x²/N), anchored pairs.** x² mod N is practised in round 3, as with the subgoal, and nothing comes.

In all, of the nine runs that came (§4):
- Without the subgoal, two do not come in 5 rounds: 3·x² under hardest first and x⁴ under anchored pairs.
- One comes two rounds later: x⁴ under hardest first, in round 3 instead of round 1.
- In these three, the subgoal's part was x² mod N in round 1, which only its deeper probe solved.
- It changed the program, not the round, of 3·x² under anchored pairs.
- It had no part in the other five:
  - (x² + x) under hardest first, by this control;
  - (x² + x), (x + 1)² and 2^x under anchored pairs and (x + 1)² under hardest first, by the replays (§5.2).
- It brought nothing where nothing came: Lucas under both rules and floor(x²/N) under anchored pairs do not come
  without it either.

### 6. Predictions against outcomes

| # | formula | predicted | came | right | the mechanism: predicted, and found |
|---|---|---|---|---|---|
| 1 | (x² + x) mod N | 1–2 / 0–1 | 1 / 0 | yes / yes | Anchored pairs: the book's x·(x + 1) with "subtract N", 4 tokens, as found. Hardest first: "x·(x + 1), reduce fully" once the reduction is a routine; found (X − 1)·X on x + 1 instead, with the reduction written out. The prediction that the score ties was wrong (§2). |
| 2 | (x + 1)² mod N | 1–3 / 1–3 | 1 / 1 | yes / yes | "INC.X, then x² mod N's program": found, with that program a routine of round 1. The routine came from a near miss's remainder (the mirror, §5.1), which the prediction did not foresee. |
| 3 | 3·x² mod N | 1–2 / 1–4 | 1 / 1 | yes / yes | The subgoal x² mod N, solved in round 1: as predicted. Hardest first needed it: only its deeper probe solved x² mod N, and without the subgoal nothing came in 5 rounds (§5.3). Anchored pairs did not: without it, 3·x² came in the same round by another program, which adds 3 and subtracts N once, x·x times. |
| 4 | x⁴ mod N | 1 / 1 | 1 / 1 | yes / yes | The subgoal, deeper probe in round 1, then "square the square": as predicted. Without the subgoal, hardest first got x⁴ in round 3, when its own order picked x² mod N, and anchored pairs got nothing in 5 rounds (§5.3). |
| 5 | (x³ + x) mod N | no / no | no / no | yes / yes | x² mod N is not singled out: as predicted. Hardest first still practised x² mod N in round 3 (one of its four costliest); nothing came in rounds 3–4. |
| 6 | 2^x mod N | 1–4 / 1–3 | no / 1 | no / yes | Anchored pairs: the recurrence source's "double mod N", round 1, as predicted. Hardest first: the named risk came true; the step cost 3–4 and was never among its four costliest (4–6). |
| 7 | Lucas | no / 2–4 | no / no | yes / no | The recurrence source's two-number step: posed and learnable from round 1 (cost 5) under both rules. Anchored pairs never picked it. Hardest first picked two two-number steps in round 4, its last, and found no program. Without the subgoal, neither rule picked one (§5.3). |
| 8 | Tribonacci | no / no | no / no | yes / yes | The three-number step is never posed: as predicted. The two-number steps posed in its place are never solved. |
| 9 | floor(x²/N) | no / no | no / no | yes / yes | Its subgoal is a stand-in for N (§2), (x − 1)·N, in rounds 0–4 (hardest first) and 0–3 (anchored pairs); nothing came, nor without the subgoal under anchored pairs (§5.3). |
| 10 | x·(x + 1)/2 mod N | 1–3 / 0–2 | no / no | no / no | It needed the sum loop as a near miss. That loop never was one (§2), and no source poses the running sum. |

In all:
- 16 of the 20 "round, or no" predictions held: 8 of 10 under each rule. Every formula that came, came in its
  predicted range of rounds.
- The 8 predictions made with medium or high confidence all held. Of the 12 made with low confidence, 8 held.
- The four misses:
  - 2^x under hardest first: the risk the prediction named.
  - The triangular numbers, both rules: the route the prediction assumed was not there.
  - Lucas under anchored pairs: the step was there, and the rule did not pick it.
- Of the mechanisms the predictions named:
  - The subgoal for 3·x² and x⁴: it did the work in all four runs, as predicted. The control (§5.3) shows it was
    needed for two of them: 3·x² under hardest first and x⁴ under anchored pairs. It brought a third two rounds
    sooner (x⁴ under hardest first). The fourth, 3·x² under anchored pairs, came in the same round without it.
  - The carried routines and the target search for (x² + x) and (x + 1)²: right, with a different routine each time
    (§5.1). For the triangular numbers: wrong, nothing came.
  - The recurrence source for 2^x (anchored pairs): right; for Lucas: wrong.
  - The three predicted failures: all right.

### 7. What did not come, dead ends and loose ends

- **(x³ + x) mod N.**
  - Its round-0 subgoal is the mirror (§2), which was never solved.
  - Hardest first practised x² mod N in round 3 (cost 5, one of its four costliest). Even with that routine, the
    rest of the way (times x, plus x, reduce) was not found in rounds 3–4.
  - Anchored pairs never solved x² mod N in this run.
- **2^x, hardest first.**
  - The step was learnable from round 1 under both of its names: `recur1/A` at cost 4, then 3, and the book's
    `law2/mod(add(X,X),N)` at cost 3.
  - It was below the four costliest every round: they cost 5 in rounds 1–3 and 4 in round 4.
  - Hardest first takes the most expensive pieces, and this step is cheap.
- **Lucas.**
  - Fibonacci's step under new names (`recur2/stepUA`, `stepTA`, …) was learnable at cost 5 from round 1 under both
    rules.
  - Anchored pairs never picked it.
  - Hardest first picked two of them in round 4. One was T.rec18 = `L.A[INC.U] MOV.X.U MOV.U.A M0.5`: the shape of
    Fibonacci's step `L.A[INC.U] MOV.X.U MOV.U.A T.mod`, with a merge, M0.5, where Fibonacci's has `T.mod`. Its
    target search then found nothing. The start (2, 1) needs reducing at N ≤ 2, so the program is longer than
    Fibonacci's `INC.U L.X[step]`, past what the search reaches.
- **Tribonacci.**
  - The recurrence source poses two-number steps only. Tribonacci's next term is not a function of two terms and N,
    so all of them stay unsolved.
  - The three-number step is never posed.
- **floor(x²/N).**
  - Its subgoal, (x − 1)·N, is a stand-in for N (§2).
  - x² mod N was practised in round 3 under both rules. Anchored pairs also picked the one-step division in round 0
    (`law/div(X,N)/x+qN`: X div N for X < 2N).
  - floor(x²/N) needs division by counting: that step in a loop, with a count. The target search did not reach it.
- **The triangular numbers.** The prediction's route was not there:
  - The sum loop `L.X[L.X[INC.A] DEC.X]` was never a near miss (§2).
  - No source poses a running sum. The recurrence source poses y(x + 1) from y(x), and y(x + 2) from y(x) and
    y(x + 1); it never poses y(x + 1) from y(x) and x, which is the sum's step.
  - The two-number steps it does pose (T(x + 2) = 2T(x + 1) − T(x) + 1) stay unsolved.
  - This is the clearest gap the test found: a step the learner has every piece for, posed by no source.
- **The score's picks that were not steps.**
  - The mirror for (x³ + x) and the stand-ins for N for Lucas and floor(x²/N) (§2) took a pick slot each time they
    were learnable. For Lucas this changed round 0 under both rules (§5.2).
  - None of the three formulas came with them. Without the subgoal, Lucas does not come either, under either rule
    (§5.3).
- **Loose ends.**
  - Container 1's restart at 13:00:40 UTC is not explained by the logs kept (§3). Its records replay exactly from its
    store, so the restart changed no outcome.
  - The replay without the subgoal stops at the first search no store holds, so it cannot say whether a target the
    subgoal changed would have come without it. The control (§5.3) runs those rounds. It was added after the
    outcomes were read and was not pre-registered.
  - The control's hardest-first container was preempted once (§3) and replayed from its store. Its records replay
    exactly from the store, so the restart changed no outcome.

### 8. Files and how to rerun

Everything is in `bar3/scale/heldout/`, which is new; nothing outside it was changed.
- `PREREG.md`: the pre-registration, read-only. Its sha256 is
  `7e669138ab87b15aff8ea5014fdd11e9fe7406b49a5fde4e75c89a592dad2ca8`, the same after the runs. The frozen files'
  hashes it lists were checked again after the runs and match.
- `targets.py`: the ten formulas, added on import to the target table (`bar3/breadth/targets.py`'s `TARGETS`).
- `rank.py`: the early check (§2), writing `records/rank0.json`. It needs numba: it runs the round-0 target probe.
- `searcher.py`: `bar3/scale/pool.Searcher`, with workers that know the held-out targets.
- `modal_heldout.py`: the runs.
  - App and Volume `bar3-heldout`, in `v1/`.
  - `--fetch` brings the records, not the stores.
  - `--control` also accepts the configs without the subgoal (§5.3).
- `watch_spend.py`: a spend limit for one app.
- `verify.py`: the outcomes, each program rechecked on every pair up to N = 120. It is
  `bar3/scale/subgoal/verify.py` on these records.
- `mechanism.py`: where each routine of each program came from (§5.1).
- `attribute.py`: the replays on the stores. It is `bar3/scale/subgoal/attribute.py` with the held-out targets.
- `picks_without.py`: the picks without the subgoal, in the first round that differs (§5.2).
- `cost.py`: the control's cost, billed and in new search-hours (§3).
- `records/`:
  - one `.jsonl` per run: `<config>_<target>_ho1`, and `<config>_<target>_ctl1` for the control;
  - `rank0.json`;
  - `calls_ho1.json` and `calls_ctl1.json`, the Modal call ids.
- `logs/`:
  - the test's `launch_ho1.log` and `watch_ho1.log`;
  - its app log, `app_ap-GqOjlw7uaIW2SZPUCJNZWw.log`, which is the tail Modal keeps. Two earlier captures,
    `…_at_0550.log` and `…_at_0555.log`, hold container 0's preemption;
  - `verify_ho1.txt`, `mechanism_ho1.txt`, `attribute_ho1.txt` and `picks_without_ho1.txt`;
  - the control's `launch_ctl1_lp.log`, `launch_ctl1_ct2a2.log`, `watch_ctl1_lp.log` and `watch_ctl1_ct2a2.log`;
  - its app logs, which are the tails Modal keeps: `app_ap-LcCT6mNn4rLbbHHhUmsOHr.log` (hardest first, with its
    preemption) and `app_ap-Q5nWfEKrXrB2mgbAGjmp61.log` (anchored pairs);
  - `verify_ctl1.txt`, `mechanism_ctl1.txt`, `attribute_ctl1.txt` and `cost_ctl1.txt`.

The search stores stay on the Volume: they are caches, not records. There are the test's four (`store_*_ho1.jsonl`,
6–9 MiB each) and the control's two (`store_ctl_<rule>_ho1.jsonl`: the test's joined per rule, then extended by the
control; 14 and 21 MiB).

```bash
cd cpu_first
NV=<python with numba 0.68 and numpy 2.5.4>                                     # the Modal image's; locally a venv
$NV -m bar3.scale.heldout.rank                                                   # §2, minutes
~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750            # before every launch
HO_CPU=16 HO_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/heldout/modal_heldout.py \
  --fresh lp_clean_deep_sub,ct2a2_clean_deep_sub --targets x2x,x1sq,three,x4,x3x,pow2,lucas,trib,floorsq,tri \
  --tag ho1 --groups 4
~/github/sotaku/venv/bin/python bar3/scale/heldout/watch_spend.py <app id> 18    # beside it
~/github/sotaku/venv/bin/python -m modal run bar3/scale/heldout/modal_heldout.py --fetch
~/github/sotaku/venv/bin/python -m bar3.scale.heldout.verify                      # §4
~/github/sotaku/venv/bin/python -m bar3.scale.heldout.mechanism                   # §5.1
~/github/sotaku/venv/bin/python -m modal volume get bar3-heldout v1/store_lp_clean_deep_sub_x2x_ho1.jsonl stores/  # each of the four
$NV -m bar3.scale.heldout.attribute stores/store_*_ho1.jsonl -- bar3/scale/heldout/records/*_ho1.jsonl     # §5.2, 3 minutes
$NV -m bar3.scale.heldout.picks_without stores/store_*_ho1.jsonl -- lp_clean_deep_sub_x2x_ho1:1 \
  lp_clean_deep_sub_three_ho1:1 lp_clean_deep_sub_x4_ho1:1 ct2a2_clean_deep_sub_three_ho1:1 ct2a2_clean_deep_sub_x4_ho1:1 \
  lp_clean_deep_sub_lucas_ho1:0 ct2a2_clean_deep_sub_lucas_ho1:0 ct2a2_clean_deep_sub_floorsq_ho1:3                # §5.2
cat stores/store_lp_clean_deep_sub_x2x_ho1.jsonl stores/store_lp_clean_deep_sub_x1sq_ho1.jsonl > stores/store_ctl_lp_ho1.jsonl
cat stores/store_ct2a2_clean_deep_sub_x2x_ho1.jsonl stores/store_ct2a2_clean_deep_sub_x1sq_ho1.jsonl > stores/store_ctl_ct2a2_ho1.jsonl
~/github/sotaku/venv/bin/python -m modal volume put bar3-heldout stores/store_ctl_lp_ho1.jsonl v1/store_ctl_lp_ho1.jsonl  # and ct2a2's
HO_CPU=16 HO_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/heldout/modal_heldout.py \
  --control --fresh lp_clean_deep --targets x2x,three,x4,lucas --tag ctl1 --store store_ctl_lp_ho1.jsonl     # §5.3
HO_CPU=16 HO_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/heldout/modal_heldout.py \
  --control --fresh ct2a2_clean_deep --targets three,x4,lucas,floorsq --tag ctl1 --store store_ctl_ct2a2_ho1.jsonl
~/github/sotaku/venv/bin/python bar3/scale/heldout/watch_spend.py <app id> 3.5   # beside each
~/github/sotaku/venv/bin/python -m modal run bar3/scale/heldout/modal_heldout.py --fetch --only ctl1
~/github/sotaku/venv/bin/python -m bar3.scale.heldout.verify bar3/scale/heldout/records/*_ctl1.jsonl
~/github/sotaku/venv/bin/python -m bar3.scale.heldout.mechanism bar3/scale/heldout/records/*_ctl1.jsonl
~/github/sotaku/venv/bin/python -m modal volume get bar3-heldout v1/store_ctl_lp_ho1.jsonl stores/after/  # and ct2a2's
$NV -m bar3.scale.heldout.attribute stores/after/store_ctl_*_ho1.jsonl -- bar3/scale/heldout/records/*_ctl1.jsonl  # 1.5 minutes
~/github/sotaku/venv/bin/python bar3/scale/heldout/cost.py stores stores/after                              # §3
```

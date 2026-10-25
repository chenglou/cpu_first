# Held-out test: pre-registration

Written 2026-10-25 05:15 PDT, before any search, score or run on these formulas, and not edited after. The only
thing computed on them beforehand is that each can be posed (below). HELDOUT.md §1 repeats this text unchanged.

## The learner, frozen

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

## The formulas, and whether they can be posed

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

## Predictions

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

## What is reported

For each formula and rule:
- the round it came in, or "no, 5 rounds";
- the program, rechecked on every pair up to N = 120;
- which mechanism did the work, read from the records. That means where each routine of the program came from (the
  source and round of the practised problem), whether the subgoal's deeper probe solved one of them, and a replay of
  the run on its stored searches without the subgoal (`attribute.py`).

Each prediction is then marked right or wrong. Budget: soft $15, cap $20; every launch guarded by `spend.py 4750`.

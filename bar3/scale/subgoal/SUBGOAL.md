# Bar 3: a signal that sees a subgoal before it is solved

<!-- PLAIN -->
## In plain words

**The question.** Bar 3's learner picks its practice by signals that see a problem only once it is solved:
hardest first takes the costliest solved problems; anchored pairs take the pairs of solved problems that make the
target cheapest. 10·x², 10·x² + 1 and x³ all need x² mod N as a piece. The depth-2 law book poses it, but its
program (6 tokens, once "subtract N if you can" is a routine) is mostly beyond the 1M-state probe, and the one time a
probe solved it, it tied with dozens of other problems at the same cost and lost on pool order (SCALE.md §4.5). Here
the learner gets a signal that sees a piece before it is solved — does the target factor through it? — and uses it to
aim a deeper probe and to break ties.

**The score** (`score.py`). For each problem the learner poses on the target's own instances, from the rows alone:
- *table*: group the target's rows by (the problem's answer, N), by (its answer, x), or by its answer alone. If every
  group wants a single answer, the target is a function of that grouping, and the score is the rows it saves (rows
  minus groups). 10·x² mod N and (10·x² + 1) mod N are functions of (x² mod N, N): x drops out.
- *law*: the target is one law of the book away from the problem: t = op(a, b) or op(op'(a, b), c), with the book's
  own operations and a, b, c among the problem's answer, x, N and the constants the data shows, every value within the
  machine's cap. x³ mod N = (x² mod N)·x mod N; 10·x² mod N = (x² mod N)·10 mod N.

A law counts first, then the rows saved. Left out: a problem that equals the target on all its rows (the target
posed again: a near miss's remainder can be), and any grouping that leaves a group mixed (it credits only the
target's own structure, such as 10·x² being 0 for every x at N = 5 and 10).

**The tiny test** (round 0 with the depth-2 book, seconds, no search; `rank.py`). x² mod N ranks first of about 105
scored problems for 10·x² (law), 10·x² + 1 (table: 24 rows saved, the next problem 20), x³ (law) and Montgomery's
(table: 11 against 10); also for x² + 1 and x² + 3, and second for x² − 1, after x² + N (also a law). For Fibonacci and
x² mod N itself the top is a tie of 4–5 problems: the score singles out nothing.

**The learner** (`learn.py`). The subgoal is the problem the score singles out: the best score, held by no other
problem. It gets one deeper probe when the usual two leave it unsolved (10M states: one level beyond the 1M-state
probe), and it wins the ties that pool order used to break. Everything else is as before: with no subgoal, the
learner runs exactly as the scale study's (checked: the same searches, the same records).

**What came of it** (from scratch, N ≤ 7, checked at N ≤ 10, general to N = 40, at most 5 rounds, the scale study's
depth-2 config; "without" is the same learner and config with no subgoal, run to 5 rounds as the control; every
program found is right on every pair up to N = 120, 7,260 pairs):

| target | hardest first | | anchored pairs | |
|---|---|---|---|---|
| | with the subgoal | without | with the subgoal | without |
| **x³** | **round 1** | no, 5 rounds | **round 1** | no, 5 rounds |
| **10·x²** | **round 1** | no, 5 rounds | round 4 | round 4 |
| 10·x² + 1 | no, 5 rounds | no, 5 rounds | no, 5 rounds | no, 5 rounds |
| Montgomery's x²·2⁻¹ | no, 5 rounds | no, 5 rounds | no, 5 rounds | no, 5 rounds |

So the subgoal brings x³ under both rules and 10·x² under hardest first, each in round 1, where the learner without it
gets neither in 5 rounds. (Under anchored pairs, 10·x² comes in round 4 with or without the subgoal: that is the
depth-2 book's own result, which the scale study did not see because its run was cut after 2 rounds.)
- *x³*: the deeper probe finds x² mod N in round 1 (its 6-token program, one level beyond the usual probe, once the
  "subtract N" link is a routine). It is at the top cost that round, alone or with one or two other problems, and
  both rules take it (§5). That round's target search then finds x³ = "x times: add x² mod N, reduce mod N" (4
  tokens, 14 primitive instructions).
- *10·x², hardest first*: the same x² mod N in round 1, then "g more times: add 9 and reduce", where 9 is 3² and the 3
  comes from the merge of the three "make 10" routines practised in round 0 — the constants the data shows, which
  SCALE.md found practised but never used.
- *10·x² + 1 and Montgomery's*: x² mod N is a routine from round 1 (but for anchored pairs on 10·x² + 1, where it never
  comes), and the rest of the way from it — (10X + 1) mod N, or halving mod N — is posed and never solved. A second
  refinement made that rest the next subgoal, with the deeper probe (§7): still out of reach. For 10·x² + 1 it is a
  token longer than the deeper probe goes (right on 26 of 28 instances); for Montgomery's, N odd and at most 7 gives
  16 instances, too few to pin "halve mod N" down (programs right on all 16 fail at N = 9).

**Nothing regressed, and one target came back.** Under the same config, x² mod N, x² ± 1, x² + 3, 2x + 1 and Fibonacci
all still come from scratch, by both rules, each right on all 7,260 pairs:

| target | hardest first: with / without | anchored pairs: with / without |
|---|---|---|
| x² mod N | round 0 / the same run | round 0 / the same run |
| x² + 1 | round 1 / round 1 | round 0 / the same run |
| x² − 1 | round 4 / the same run | round 0 / the same run |
| x² + 3 | **round 1 / no, 5 rounds** | round 1 / round 2 |
| 2x + 1 | round 0 / the same run | round 0 / the same run |
| Fibonacci | round 4 / the same run | round 1 / the same run |

"The same run": replayed on its stored searches without the subgoal, the run is identical in every round. The
depth-2 book alone costs hardest first x² + 3 (in 5 rounds; SCALE.md's configs had it in round 1), and delays its
x² − 1 and Fibonacci to round 4 (from 0 and 1); the subgoal gives x² + 3 back and does not touch the other two.

**What did the work: the deeper probe, aimed by the score.** Every target the subgoal brings comes in the round the
deeper probe first solves x² mod N; with the tie-break turned off (the deeper probe alone), every one of them still
comes, in the same round (replayed on the stores, and run where the replay could not go on: anchored pairs on x³ and
10·x²). The tie-break changes which problems
are taken in a few rounds (it decides anchored pairs' anchors on x³ and 10·x²) but no outcome. The deeper probe is
cheap: 30 of them, 1.6 of the 47 search-hours.

**Cost.** About $11.90 on Modal, against a soft budget of $50: $11.34 billed to the app over its complete hours, plus
about $0.55 for its last 26 minutes at the same rate. The month's bill is now $3,412. Search time, as the records
count it: 46.8 search-hours for the 20 runs with the subgoal (1.6 of them in the deeper probe), 23.1 for the 4 chain
runs, 5.6 for the 2 runs without the tie-break and 47.1 for the 12 controls, 122.6 in all. Each later run started
from an earlier run's store and replayed the rounds they share instead of searching them again, so less than that
was actually searched.

**What it says about bar 3.** On these targets, what held the learner back was its choice of what to search
deeply, not the speed of its search. The score is worked out from the rows alone, with no search and nothing
specific to the target, and in round 0 it already points at the missing piece. One deeper search aimed there brings
two of the three targets in round 1, for about 3% more search. Two limits remain, and neither is about the choice:
- *The score sees one step.* Once x² mod N is a routine, the rest of the way to 10·x² + 1 is a new problem, posed on
  the routine's values rather than on the target's instances. Aiming the deeper probe at it (the chain) is not enough,
  because its program is a token longer than that probe reaches. The next step would be to pose and score pieces for
  that problem the way the target's own pieces are posed and scored: (10X + 1) mod N factors through 10X mod N by
  one law.
- *Montgomery's at N ≤ 7 is too small to learn from.* Only odd N count, which leaves 16 instances. That is too few
  to tell halving mod N apart from programs that only fit those 16. Here the limit is the size of the problem, not
  what the learner picks.
<!-- /PLAIN -->

## Details

### 1. The score (`score.py`)

**Which problems.** A problem is scored when it is posed with x in X and N in N, nothing else set, its answer in A,
and is defined on every one of the target's instances at both sizes: the 28 search instances (N ≤ 7) and the check
instances (N ≤ 10), 55 rows in all (25 for Montgomery's, odd N only). That is the depth-2 laws, the laws on X = x,
the near misses' values, the dreams and the target's projections; not the laws on X = x + N, x + qN or x + QN, the
restrictions (parts of the rows), the remainders (other inputs), the recurrences or the constants. Rows only: no
search, about 0.3 s for a pool of 230–310 problems.

**The table view.** For each grouping of the target's rows — by (g, N), by (g, x), by g alone, where g is the
problem's answer — the target is a function of the grouping when every group wants one answer; the grouping then
saves rows − groups rows. The view's value is the largest saving among the groupings that determine the target, 0 if
none does. On 10·x² mod N, x² mod N saves 24 of 55 rows by (g, N); the problems that fold x and N − x together
(|2x − N|, (N − x)·x, min and max of N − x and x) save 20, since every target symmetric in x and N − x factors through
them; x itself saves 0.

**The law view.** Over the leaves g, X, N and the constants the data shows (sources.constants: 10 for 10·x², 11 for
10·x² + 1, 3 and 4 for x² + 3), every op(a, b) and op(op'(a, b), c) with the law book's two-argument operations (add,
monus, absdiff, mul, div, mod, min, max, lt, le, eq, gcd) and one-argument ones (succ, pred), g among the leaves used,
every value on the way within [0, cap) on its row (cap 98 at N ≤ 7, 200 at N ≤ 10, as the machine's), and the result
equal to the target on all 55 rows. The array versions of the operations are checked against bar3/problems' on a
grid. The cap matters: x³ mod N = mod(mul(g, X), N) holds for g = x² too, but x²·x reaches 216 > 98 at N = 7, so only
x² mod N passes; 10·x² mod N = mod(mul(g, 10), N) the same way. The view is off for a target that is itself such a
law of X, N and the constants (x² mod N = mod(mul(X, X), N)), since then any copy of x passes.

**The order.** (a law is exact, the table's rows saved), larger first. A problem equal to the target on every row is
not scored.

**Why both views.** x³ mod N is not a function of (x² mod N, N): x and N − x share x² mod N, and their cubes differ
in sign. So no table sees x³'s subgoal; the law does. 10·x² + 1 and Montgomery's need a third step after x² mod N (a
multiply and an add; a halving mod N), beyond depth 2, so no law sees theirs; the table does.

**Two versions dropped on the way** (both in the tiny test, no runs):
- a graded table that counted the rows in one-answer groups and ignored mixed ones: projections of the target (its
  answer divided by 2; is it 0) and coarse problems (gcd(2x, N)) outranked x² mod N on 10·x² and 10·x² + 1, all from the
  rows where the target is the same for every x (N = 2, 5 and 10 for 10·x²);
- no exclusion: on the targets symmetric in x and N − x, a near miss's remainder posed on X = N − x is the target
  permuted onto its own instances, and it topped the table view.

### 2. The tiny test (`rank.py`, `records/rank0.json`)

Round 0's pool with the depth-2 book and every lever (the scale study's `*_deep` sources), rebuilt from the records'
round-0 near misses; for the four missed targets its size matches the record exactly (231, 237, 278, 308; the other
targets' records come from configs without the book). Where x² mod N lands:

| target | scored | x² mod N | its law | its table | the next problem |
|---|---|---|---|---|---|
| 10·x² | 104 | **1st** | mod(mul(g,10),N) | 24 | 20 (no law) |
| 10·x² + 1 | 107 | **1st** | – | 24 | 20 |
| x³ | 107 | **1st** | mod(mul(g,X),N) | 0 | 17 (no law) |
| Montgomery's | 106 | **1st** | – | 11 | 10 |
| x² + 1 | 114 | 1st | mod(succ(g),N) | 24 | law, 1 |
| x² + 3 | 110 | 1st | mod(add(g,3),N) | 24 | law, 1 |
| x² − 1 | 107 | 2nd | – | 24 | (1st: x² + N, law mod(pred(g),N), table 1) |
| 2x + 1 | 113 | (8th) | – | 0 | (1st: 2x mod N, law mod(succ(g),N), table 15) |
| x² mod N | 107 | (the target) | – | – | (4 problems tied at 20: no subgoal) |
| Fibonacci | 111 | – | – | – | (5 problems tied at 29: no subgoal) |

### 3. The learner and its checks (`learn.py`, `make_learner.py`, `configs.py`, `check.py`)

`learn.py` is bar3/scale/learn.run copied by `make_learner.py`, which applies the hooks as exact replacements, each
asserted to match where it should (learn.py could not change while the scale relaunches that import it were live).
The hooks:
- the subgoal (`subgoal_of`): the problem with the best score when no other has the same, scored after the pool is
  posed;
- the deeper probe: when the probes (30k, then 1M states) leave the subgoal unsolved, one more search with 10M states
  (`probe_deep`; a level starts only below the budget, so this streams one level more than the 1M probe), up to an
  hour;
- the ties: in hardest first's order, the anchors' order, the pairs' order and the singles' order, the subgoal comes
  first among equals; everything else keeps pool order;
- options: `chain` (§7, `_subchain`); `ties` off (`_subnoties`: the deeper probe alone, with pool order for the
  subgoal's ties too; §5); `pick` (practise the subgoal as soon as it is learnable), which was not run.

The config is the scale study's `<rule>_deep` (sources: law, restrict, project, value, remainder, dream, the depth-2
laws, recurrences, constants, routine remainders; the guided second target search; clean routines; 5 rounds; N ≤ 7,
checked at N ≤ 10, general to N = 40) with the subgoal: `lp_clean_deep_sub` and `ct2a2_clean_deep_sub`. The controls
are `lp_clean_deep` and `ct2a2_clean_deep`: the same config without the subgoal, which is bar3/scale/learn.run.

Checks, local, k = 5 with tiny budgets, 2 rounds:
- `check.py same`: with no subgoal, the copy asks for exactly the searches learn.run asks for (it runs on learn.run's
  searches only) and writes the same records, on 10·x², x² mod N and Montgomery's under both rules (736–1,321
  searches each); rerun after the chain option was added (10·x², both rules);
- `check.py smoke`: with the subgoal on, x² mod N is the subgoal in both rounds and gets one deeper probe a round.

### 4. The runs (`modal_subgoal.py`, `records/*_s1.jsonl`, `logs/`)

20 runs from scratch, 2 rules × 10 targets, on Modal in 4 containers (16 cores, 64 GiB each; app `bar3-subgoal`,
Volume `bar3-subgoal`), searches of a batch on the same space shared as in the scale study's `--share` (each gets
the record it would alone). One container (10·x² + 1 and Montgomery's) was preempted after an hour and replayed its
runs from its store. Every program is checked by `verify.py` on every pair up to N = 120 (7,260 pairs; 3,600 for
Montgomery's), beyond the learner's own N = 40.

| target | hardest first | | anchored pairs | |
|---|---|---|---|---|
| | outcome | search-hours | outcome | search-hours |
| x³ | **round 1** | 0.65 | **round 1** | 0.87 |
| 10·x² | **round 1** | 0.89 | **round 4** | 4.58 |
| 10·x² + 1 | no, 5 rounds | 7.11 | no, 5 rounds | 6.27 |
| Montgomery's | no, 5 rounds | 3.58 | no, 5 rounds | 5.45 |
| x² mod N | round 0 | 0.58 | round 0 | 0.61 |
| x² + 1 | round 1 | 1.33 | round 0 | 1.02 |
| x² − 1 | round 4 | 4.17 | round 0 | 0.63 |
| x² + 3 | round 1 | 0.89 | round 1 | 1.23 |
| 2x + 1 | round 0 | 0.57 | round 0 | 0.61 |
| Fibonacci | round 4 | 4.48 | round 1 | 1.28 |

46.8 search-hours in all. The deeper probe ran 30 times (1.6 search-hours, 1.6 billion states, up to 14 GB in one
worker) and found the subgoal 9 times, each time x² mod N at cost 6 in round 1 (`L.X[L.X[INC.A]]
L.X[MOV.X.A T.mod(X,N)@x+qN]`: square, then x links), after the 1M-state probe had stopped at level 5.

What the subgoal was, round by round (`verify.py`): x² mod N on the four squaring targets and on x² ± 1 and x² + 3,
until it was practised; x² + N on x² − 1 (learnable at cost 3–4 every round, never among the hardest, never picked);
2x mod N on 2x + 1; none on x² mod N and Fibonacci (ties). After x² mod N was practised, the next unique best, when
there was one, was a dream or a near miss's value, learnable at cost 2–3; it was picked once (anchored pairs,
Montgomery's, round 4), and the target still did not come.

**The other 18 runs.** Each started from the store of the subgoal runs on the same targets (`--store`), so the rounds
it shares with those runs were replayed, not searched again.
- 12 controls: `lp_clean_deep` and `ct2a2_clean_deep`, with no subgoal, on the six targets where the subgoal changed
  a run (§5), each run to 5 rounds or until its target came:

| target | hardest first | | anchored pairs | |
|---|---|---|---|---|
| | outcome | search-hours | outcome | search-hours |
| x³ | no, 5 rounds | 2.85 | no, 5 rounds | 6.70 |
| 10·x² | no, 5 rounds | 4.45 | round 4 | 4.41 |
| 10·x² + 1 | no, 5 rounds | 7.82 | no, 5 rounds | 5.81 |
| Montgomery's | no, 5 rounds | 3.12 | no, 5 rounds | 3.88 |
| x² + 1 | round 1 | 1.26 | round 0 | 0.99 |
| x² + 3 | no, 5 rounds | 3.00 | round 2 | 2.85 |

- 4 chain runs (§7), `*_deep_subchain`, on 10·x² + 1 and Montgomery's: none invented; 23.1 search-hours.
- 2 runs without the tie-break (§5), `ct2a2_clean_deep_subnoties`: x³ in round 1 and 10·x² in round 4; 5.6
  search-hours.

The container with the last four controls was preempted twice, and each time it replayed its runs from its store.
Every program these runs found is also right on all 7,260 pairs.

### 5. What the subgoal changed (`attribute.py`, and bar3/scale/compare.py against the scale study's records)

**Replays.** Each run is replayed on the searches its container stored (the Volume's `store_*.jsonl`, 23,504
distinct searches when the replays ran; no search runs): first as it ran, which must give its records back (it does,
for all 20), then without the subgoal, which is the scale study's `<rule>_deep` exactly: the same learner with no
score, no deeper probe and pool order for every tie. A round is the same when its picks, library, target search and
every probe outcome but the subgoal's own are the same. Where the run without the subgoal first differs, it asks for
searches no store holds, and the replay stops there.

| what the subgoal changed | runs |
|---|---|
| nothing, in any round (identical to `<rule>_deep`) | x² mod N, x² − 1, 2x + 1, Fibonacci (both rules); x² + 1 and 10·x² + 1 (anchored pairs) |
| round 1, the round the target came in | x³, x² + 3 (both rules); 10·x², x² + 1 (hardest first) |
| round 4, the round the target came in | 10·x² (anchored pairs) |
| round 1, the target did not come | 10·x² + 1 (hardest first); Montgomery's (both rules) |

So the subgoal changed 10 of the 20 runs, and 7 of those got their target in the very round it changed them. The
controls (§4) show what happens to those runs without it: x³ (both rules), and 10·x² and x² + 3 under hardest first,
do not come in 5 rounds; anchored pairs' x² + 3 comes a round later; hardest first's x² + 1 and anchored pairs' 10·x²
come in the same round.

The round-1 changes are all the same event. The deeper probe solves x² mod N at cost 6, the top cost that round:
alone in four of the nine runs, and with one or two other problems in the rest.
- *Hardest first* takes it on cost alone: whichever way the ties go, it is among the 4 costliest. Without the
  tie-break, five of its six changed runs replay the same in every round. On Montgomery's the same problems are
  practised into the same routines, but two of them are named the other way round; the next search then asks for
  other names, which no store holds, so the replay stops.
- *Anchored pairs* takes the 2 costliest as anchors. On x³, x² mod N ties with two others for them, and the
  tie-break makes it one. Without the tie-break it is still picked, as an anchor's partner, because any pair that
  contains it makes the target cheapest: `ct2a2_clean_deep_subnoties_cube_s1` gets x³ in round 1 with the same
  program.

Anchored pairs' 10·x² is the same story in round 4. There the usual probes solve x² mod N at cost 5, tied at the top
cost, and the tie-break makes it an anchor. Without the tie-break, or without the subgoal at all, it is picked in that
round anyway, and 10·x² comes from the same program: `ct2a2_clean_deep_subnoties_ten_s1` and the control are the same
in every round. So the tie-break changes which problems are picked, and never an outcome.

Hardest first's x² − 1 and Fibonacci, which come in round 4 against rounds 0 and 1 in SCALE.md's configs, are
identical to `lp_clean_deep` in every round: the delay is the depth-2 book's (about a hundred more problems for
hardest first to take first), not the score's.

**Against the scale study's own `*_deep` records** (`compare.py --old`, cut there after 2–4 rounds): round 0 is the
same in all eight main runs. In the six where the subgoal changed round 1, round 1 differs from the record by exactly
one probe outcome (x² mod N: unsolved there, learnable at cost 6 here) and what follows from it (picks, library, the
target search); anchored pairs' 10·x² and 10·x² + 1 are the same in every round the records hold (2 and 3).

### 6. The programs

**x³ mod N** (both rules, round 1): `MOV.U.X L.X[MOV.X.U T.law5]`, with `T.law5` = x² mod N above. U keeps x; x
times, X ← x and `T.law5` adds x² to A and reduces A mod N (x links suffice, since A < N + x² < (x + 1)·N). After x
passes A = x·x² mod N. 4 tokens, 14 primitive instructions.

**10·x² mod N, hardest first** (round 1): `M0.2 T.law5 L.X[MOV.X.U M1.4]`, 31 primitive instructions.
- `M0.2` = `INC.T INC.U INC.U INC.U L.U[L.U[INC.T]]`: T = 10, U = 3. It is the merge of the three "make 10"
  routines practised in round 0, the constants the data shows (at x = 1 the answer is 10 mod N).
- `T.law5` leaves g = x² mod N in A, and in X the last value its reduction loop moved, g or g + N.
- `L.X[MOV.X.U M1.4]`, X times: X ← U = 3; `M1.4` = `L.X[L.X[INC.A]] M1.1` adds 3² = 9 and reduces
  (`M1.1` = `L.A[MOV.X.A T.mod(X,N)@x+qN]`). So A = g + 9·X ≡ 10·g (mod N), and it is reduced.

The second program the same search found reads more plainly: `M0.2 MOV.U.X L.T[MOV.X.U T.law5]`, T = 10 times "add
x² and reduce".

**10·x² mod N, anchored pairs** (round 4): `T.law18 M0.3 L.A[MOV.X.U T.law18]`, 45 primitive instructions. Here x²
mod N came by another road: `T.law18` = `T.law14 L.A[INC.X T.absdiff(X,N)@x+qN] MOV.A.X`, from x² − x (`T.law14` =
`L.X[L.X[INC.A]] L.X[DEC.A]`) and the absolute difference with N, without the "subtract N" link this run never
practised. `M0.3` is the same constant merge (T = 10, U = 3), and the loop again adds g more passes of a piece built on
U = 3.

### 7. Where it stops: 10·x² + 1 and Montgomery's, and the next subgoal (`_subchain`)

With x² mod N a routine (from round 1 under hardest first; under both rules on Montgomery's), what is missing is the
rest of the way from it: (10X + 1) mod N, or halving mod N, on X = x² mod N. The learner poses it, from round 2, as
"the target's answer from X = what T.law5 leaves in A" (the routine-remainder source, or a near miss's remainder that
is the same problem), and its probes never solve it. The score does not see it, since it is posed on x² mod N's values
rather than on the target's instances.

So one refinement: once a subgoal is practised, the next subgoal is the rest of the way from its routine, when the
pool poses it (`chain`): it gets the deeper probe and the ties. It changes only rounds that come after a subgoal is
practised: replayed on the stores, 17 of the 20 runs are the same under it in every round; the other three (hardest
first on 10·x² + 1, both rules on Montgomery's) differ from round 2, and were run (`*_deep_subchain_*_s1`; rounds
0–1 replayed from the stores), together with anchored pairs on 10·x² + 1, whose run under the chain is identical to
its run without it.

Neither target came. The remainder was the subgoal in every round from 2 to 4, and its deeper probe never solved it:
- *10·x² + 1*: the remainder is right on 26 of its 28 instances by level 5 (16–89M states, 74–508 s); it needs at
  least one more token over that library, and the next level would be 500M states or more. The target search stands
  at the same 26 of 28 from round 1 on.
- *Montgomery's*: programs right on all 16 of the remainder's search instances turn up at levels 4–6 (24–309M
  states), but none passes the check at N ≤ 10. With N odd and at most 7 there are only 16 instances, too few to pin
  "halve mod N" down; the target probe shows the same (by round 3–4 a one-token merge fits all 16 and fails at N = 9,
  the one odd N the check adds).
  Here the size of the problem, not the choice of practice, is what stops it.

### 8. Files and how to rerun

All in `bar3/scale/subgoal/` (new; nothing outside it was changed):
- `score.py`: the score (`python -m bar3.scale.subgoal.score` checks its operations against bar3/problems');
- `rank.py`: the tiny test, `records/rank0.json`;
- `make_learner.py` → `learn.py`: the learner (rerun make_learner.py after any change to bar3/scale/learn.run, then
  check.py); `configs.py`: its configs;
- `check.py`: the copy against bar3/scale/learn.run (`same`) and a smoke run (`smoke`), local;
- `modal_subgoal.py`: the runs (app and Volume `bar3-subgoal`, `v1/`; `--store` starts from a store already there;
  `--fetch` brings the records, not the stores);
- `verify.py`: each record's outcome, its program checked on every pair up to N = 120;
- `attribute.py`: the replays on the stores, with and without the subgoal or under another config (`--as`);
- `records/`: one `.jsonl` per run (`<config>_<target>_s1`), `rank0.json`, `calls_s1.json` (Modal call ids);
- `logs/`: the containers' logs, `app_<app id>.log` (`ap-lNvJAqRy5MrBLucRw2dwMM` the eight main runs,
  `ap-4tXAOD75zIPMojgpTQMJO1` the twelve others, `ap-dcQyCySoU36iiftoGmhSeg` the chain runs,
  `ap-j8lCHxBHviHI7VukxbtBP5` the runs without the tie-break, and the controls: `ap-OYtCYyxITkFWZPnL180zLD` on x³ and
  10·x², `ap-Vne2NeDHtnm2ZgrpjQonYc` on x² + 1 and x² + 3, `ap-QXa3K8OuRBiebjRrXdPDy6` on 10·x² + 1 and Montgomery's).

The search stores (`store_*.jsonl`, 4–9 MiB each) stay on the Volume; they are caches, not records.

```bash
cd cpu_first
~/github/sotaku/venv/bin/python -m bar3.scale.subgoal.rank                      # the tiny test, seconds
NV=<python with numba 0.68 and numpy 2.5.4>                                     # the Modal image's; locally a venv
$NV -m bar3.scale.subgoal.check same ten h1 mont                                 # the copy, about 4 minutes
~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750            # before every launch
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep_sub,ct2a2_clean_deep_sub --targets ten,tenrho,cube,mont --tag s1 --groups 2
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep_sub,ct2a2_clean_deep_sub --targets h1,rho,minus1,plus3,affine,fib --tag s1 --groups 2
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep_subchain,ct2a2_clean_deep_subchain --targets tenrho,mont --tag s1 --store store_lp_clean_deep_sub_tenrho_s1.jsonl
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh ct2a2_clean_deep_subnoties --targets cube,ten --tag s1 --store store_lp_clean_deep_sub_ten_s1.jsonl
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep,ct2a2_clean_deep --targets cube,ten --tag s1 --store store_lp_clean_deep_sub_ten_s1.jsonl
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep,ct2a2_clean_deep --targets rho,plus3 --tag s1 --store store_lp_clean_deep_sub_rho_s1.jsonl
SUB_CPU=16 SUB_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/subgoal/modal_subgoal.py \
  --fresh lp_clean_deep,ct2a2_clean_deep --targets tenrho,mont --tag s1 --store store_lp_clean_deep_sub_tenrho_s1.jsonl
~/github/sotaku/venv/bin/python -m modal run bar3/scale/subgoal/modal_subgoal.py --fetch
~/github/sotaku/venv/bin/python -m bar3.scale.subgoal.verify                      # outcomes, N = 120
~/github/sotaku/venv/bin/python -m modal volume get bar3-subgoal v1/store_lp_clean_deep_sub_ten_s1.jsonl stores/   # each store
$NV -m bar3.scale.subgoal.attribute stores/*.jsonl -- bar3/scale/subgoal/records/*_sub_*_s1.jsonl --as chain
$NV -m bar3.scale.subgoal.attribute stores/*.jsonl -- bar3/scale/subgoal/records/*_sub_*_s1.jsonl --as noties
~/github/sotaku/venv/bin/python -m bar3.scale.compare bar3/scale/subgoal/records/lp_clean_deep_sub_cube_s1.jsonl \
  --old bar3/scale/records/lp_clean_deep_cube_L1.jsonl
```

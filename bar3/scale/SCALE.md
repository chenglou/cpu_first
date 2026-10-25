# Scale: making the bar-3 learner cheaper, then deeper

*Written 2026-10-24. Code in `cpu_first/bar3/scale/`, records in `bar3/scale/records/`. Nothing submitted, no file
outside `bar3/scale/` changed.*

## In plain words

<!-- PLAIN -->
**The question.** The bar-3 learner ([PRACTICE.md](../PRACTICE.md), [BREADTH.md](../BREADTH.md)) poses and picks its
own practice problems and invents formulas by search and compression, on a bare counting machine at x < N ≤ 7. It
invented x² mod N and, changed in nothing, x² + 1, x² − 1, x² + 3 and 2x + 1 from scratch, and x³ and Montgomery's
x²·2⁻¹ once its solved formulas carried over. It missed 10·x², 10·x² + 1 and Fibonacci. Its search reaches about 6
instructions, choosing cost up to 10 times the search of a curated list, and the breadth study took about 520
search-hours. The view tested here: engineering (a cheaper choosing rule, a faster search) can cut the cost about
10 times, but since each extra instruction multiplies the search by dozens, that buys about one more instruction of
reach; real progress needs algorithmic work.

**What got cheaper, with the same choices.**
- *A compiled search.* The same breadth-first search, with the bare machine compiled (numba) and the same hash, so
  every record comes out the same but for the seconds. On Modal's hosts the biggest searches (a million states) run
  4–5 times faster, the small ones 11–22 times, and searches over a carried library of routines 24–45 times (the
  original interpreted every routine call in Python).
- *Shared searches.* Problems posed on the same instances build the same search and differ only in their goal, so
  one search serves them all: another 1.2 times.
- *The cheaper choosing rule* (BREADTH.md's: try only the pairs that contain one of the two hardest problems): 2.5
  times on a round, since lookaheads were two thirds of it.
- *The breadth study run again*: 515 search-hours became 51 (10 times), and 50 of its 54 runs made the same choices
  as before in every round (the same probe outcomes, picks, routines and programs), at 4.5–40 times less search.
  Round 0 on the ten formulas fell from 67.8 search-hours (all pairs, as BREADTH.md ran it) to 3.95 (17 times); the
  carry-over sequence from 52 to 2.2; its branches from 65 to 1.6. The other 4 runs (hardest first on its four
  failures) first differ in round 2 or 3, where the original had stopped a probe at its time limit and the faster
  search finished it: the practised problem kept two more of its shortest programs, one compressed instruction came
  out with a bracket elsewhere, and later rounds posed and picked differently. They end the same: nothing found in 5
  rounds.

**What that buys: about one instruction.** Counted directly: on the 35 primitives, every extra instruction
multiplies the distinct machine states by 9–10 (cost 7 needs 12.9M states, cost 8 about 370M); over a library of 34
carried routines, by about 40. So 17 times buys about 1.3 instructions from scratch, and less than one over a
carried library; the missed formulas need 13–22 instructions written from the learner's first pieces (BREADTH.md's
count). The view holds for the search: more speed alone does not reach them.

**The algorithmic levers, and which missed formulas now come.**
1. *Steps that move two numbers at once.* A new source reads the target's table of answers and asks whether each
   answer is a function of the ones before it. For Fibonacci it is (y(x + 2) = y(x) + y(x + 1) mod N), and the source
   poses the step (a, b) → (b, a + b mod N), with its two inputs and two answers in every placement in the
   registers. **Fibonacci is now invented from scratch**, by every choosing rule: hardest first practises the link
   ("subtract N unless that goes below zero") in round 0 and the step in round 1, then finds `INC.U L.X[step]`: start
   from (F₋₁, F₀) = (1, 0) and step x times. Right on every pair up to N = 120.
2. *Carry-over with the pairs rule.* Run along the sequence of formulas (each solved one kept as a routine), the
   pairs rule anchored on the two hardest problems now invents **10·x², 10·x² + 1 and x³, each in the first round
   of its step**. 10·x² is "x² mod N (carried), then the rest of the way", a problem the existing near-miss source
   poses once x² mod N is a routine; 10·x² + 1 is "10·x², then add one and wrap". BREADTH.md could not afford this
   rule along a sequence (its all-pairs sequence was stopped after 5 hours on its second formula), and its
   hardest-first sequence missed 10·x²; so did hardest first here, in 5 rounds, with the levers. The same sequence
   with only the breadth study's sources invents them too, with the same programs: this came from the cheaper rule
   made affordable, not from the new sources. Hardest first's own sequence, finished after the cut, misses 10·x² + 1
   as well, but then invents x³ (round 0 of its step), **Montgomery's x²·2⁻¹ (round 3)** and **Fibonacci (round 1)**,
   over a library grown to 92–123 routines.
3. *Constants taken from the data.* At a fixed x the answer depends on N alone; when it is c mod N for one small c,
   "make c" is posed (10 for 10·x², read off the remainders at x = 1), searched on one instance and checked on all.
   "Make 10" (7 instructions) is solved and picked, but no program used it, and from scratch neither 10·x² nor
   10·x² + 1 came in 5 rounds.
4. *Targets made short in terms of learned routines* (for each routine, the rest of the way from what it computes)
   and *search guided by what has been learned* (tokens priced in bits by a model of the library's own programs):
   neither changed any outcome, from scratch (40 runs) or along the sequence (the same picks and programs, step by
   step). Guidance reaches about 3 bits further for each tenfold of states, and found x³ over a carried library with
   a third of the states the plain search needed; of its 76 searches in these runs, 2 found a program the plain one
   had not, and neither was general.
5. *Still missing*: Montgomery's x²·2⁻¹ in the anchored-pairs sequences (not in 5 rounds; its programs there pass the
   N ≤ 10 check but are wrong on 40% of the pairs up to N = 40), Fibonacci along them (from round 1 they find a program
   that is right on every pair up to N = 40, but it needs more steps at N = 40 than the general check's machine allows,
   a million, so it is not accepted), and 10·x², 10·x² + 1, x³ from scratch. Those three need x² mod N as a piece, which
   the base law book never poses; the depth-2 law book poses it, but its 6-token program is mostly beyond the probes
   over the bigger vocabulary, and the one time it was learnable it tied with dozens of others and was not picked. A
   follow-up ([subgoal/SUBGOAL.md](subgoal/SUBGOAL.md)) aims one deeper probe at that piece, chosen because the target
   factors through it, and gets x³ (by both rules) and 10·x² (hardest first) from scratch in round 1.

**How far N goes.** At N ≤ 26 the learner, unchanged, invents x² mod N, x² ± 1 and 2x + 1 in round 0, and
Fibonacci in round 1 (by both rules); not x² + 3 in 2 rounds (it needed 1 at N ≤ 7). At N ≤ 50, x² mod N and
2x + 1 in round 0 and Fibonacci in round 1 (hardest first). Each program is the algorithm found at N ≤ 7, right on
every pair up to N = 120. A round costs 5.5–10 times more at N ≤ 26 and 14–15 times more at N ≤ 50, because every
count runs longer.

**What it says about bar 3.**
- The view holds for depth: 17 times cheaper buys about one instruction, and none of the missed formulas was within
  one instruction.
- But cheaper also changed what could be afforded. With the search 10–40 times cheaper, the pairs rule could run
  along a whole sequence of formulas, and there two of the three formulas the breadth study missed, 10·x² and
  10·x² + 1, turned out to be short in terms of earlier answers: each came in the first round of its step (and x³
  too, as it did in BREADTH.md's branch). So "engineering buys one instruction" is true of the search, not of the
  learner: it bought a choosing rule that could not be afforded before, and that rule reached them.
- Along the sequence the two rules reach different formulas: anchored pairs 10·x² and 10·x² + 1, which hardest first
  misses; hardest first Montgomery's and Fibonacci, which anchored pairs misses (for Fibonacci its program is right
  but too slow for the general check). Hardest first pays for its reach with a library of up to 123 routines and
  steps of up to 55 search-hours.
- The new kinds of practice problem mattered where the old sources could not pose the piece at all: the two-number
  step gave Fibonacci. Constants, as posed here, were not needed.
- What still limits it: the probes' reach (a needed piece 6 tokens long over a growing vocabulary is often out of
  it), and the choosing rules, which see nothing when no pair helps and break ties by pool order, so a needed piece
  among dozens of equally hard problems is easily passed over.

**Bounds on the search space** (§8). Asked whether the search could skip regions it treats as unbounded: it already
caps values and steps and merges programs that reach the same state. Of the bounds left, searching on fewer
instances and adding counterexamples spent 2.25–2.6 times more states on the learner's own problems, merging the
T/U swap of the scratch registers saves at most 1.7 times (and only before routines use them), registers nothing
reads later do not exist before the last level, meet in the middle fails because 94% of the items are not one to
one and the goal is a set, and a lower value cap changes nothing. None comes near the 9–10 times each instruction
costs.

**Cut, then finished.** Modal's workspace spend limit stopped every container at about 14:27 UTC on Oct 24, cutting
Montgomery's after 3–4 rounds of each anchored-pairs sequence, Fibonacci at the end of the sequences, hardest first's
sequence at 10·x² + 1, hardest first's Fibonacci at N ≤ 26 after round 0, and a few side runs. Once the limit was
raised, the five groups with a store were relaunched; each replayed its finished rounds exactly and went on (§6). What
they added: hardest first's sequence invents x³, Montgomery's and Fibonacci; the anchored-pairs sequences and both
branches miss Montgomery's and Fibonacci in 5 rounds; hardest first at N ≤ 26 invents Fibonacci in round 1. The depth-2
law-book runs and x² + 3 at N ≤ 26 had no store and were not rerun here. The subgoal study's controls ran the depth-2
runs again, the same in every round these records hold, to 5 rounds ([subgoal/SUBGOAL.md](subgoal/SUBGOAL.md)). In all:
about 562 search-hours and $62 (184 and $22 of them after the cut).
<!-- /PLAIN -->

## Details

### 1. The compiled search (`engine.py`, `jobs.py`)

**Where the time went.** A 250,000-state probe, profiled: three quarters of it in the bare machine's interpreter
(`library/inventing/brm.py`), which walks every item as Python recursion over numpy index arrays (1.65M calls for
that one search), and most of the rest in hashing states row by row.

**What changed.** The search is `bar3/search.bfs` line for line, with three parts compiled (numba):
- each item is compiled once per job into a flat bytecode (library tokens and abstractions inlined, as brm runs
  them), and one kernel runs it on every probe of every state;
- the kernel hashes each new state with brm's own hash (`library/search.hash_rows` of the state's int16 bytes), in
  the same pass; an item that leaves a state unchanged is not hashed, since its source is already in the seen set;
- the item enumeration (`library/inventing/bfs.enumerate_items`) is the same code with the kernel's signature.

The search's bookkeeping is the original's, except that a small source level feeds the kernel many classes at once
(the candidates come in the same order, so no outcome changes).

**Exact.** The same hash, the same order of states, the same parents: every record field but the seconds is the
same.
- `check.py`: 38 search jobs through both (the h1 target probe, every source's candidates, 150k-state probes,
  lookaheads, a target probe over the breadth study's 34-routine library, one at N ≤ 26): 0 records differ.
- `smoke.py`: the learner end to end on tiny configs (fresh and carried, three rules): 20 of 20 round records the
  same, with the original search and with the compiled one, and with the original learner and this folder's copy.
- Every replicated breadth run (§2) compared field by field with its record (`compare.py`).

One change of behaviour, only where the original crashes: when the start state already meets a problem's goal on
its search instances, the original reports the empty program at cost 0 without checking it on the other instances.
At N ≤ 7 every instance is searched and answers must vary, so it never happens; at N ≤ 26 the search sees a sample
of 32, it does, and the learner then crashes choosing a routine among no programs (`ValueError` in `choose`). Here
the empty program goes through the same acceptance test as any other, and the search goes on when it fails. Five
N ≤ 26 runs crashed on the original behaviour before the fix: four were rerun (x² + 1 and x² − 1, both rules), and
the fifth's container was stopped (§6).

**How much faster.** Locally (Apple M-series) 29–30× on the 38 checked jobs (340 s against 11 s), 110× on the
34-routine probe (261 s against 2.4 s), 5× on a 30k-state target probe. On Modal's x86 hosts less on the biggest
searches, where the original's numpy code runs whole 16k-state slices at once: one host, three 1M-state probes of
x² mod N's round 0, seconds:

| probe | original | compiled, each probe alone | compiled, lockstep (kept) |
|---|---|---|---|
| restrict/target/3<X (5.6M states streamed) | 93, 101 | 39 | 24, 24 |
| law/mod(X,N)/x+QN (4.6M streamed) | 120, 120 | 47 | 24, 24 |
| law/gcd(X,N)/x (4.2M streamed) | 48, 49 | 35 | 12, 12 |

Per job kind, for x² mod N's round 0, against the breadth study's records (its hosts; search-hours):

| job kind | jobs | breadth study | compiled | ratio |
|---|---|---|---|---|
| probes at 30k states | 192 | 0.278 | 0.017 | 16× |
| probes at 1M states (the unsolved) | 114 | 2.050 | 0.429 | 4.8× |
| lookaheads, full pairs rule | 1,081 | 3.443 | 0.153 | 22× |
| target search (3M) | 1 | 0.011 | 0.001 | 11× |

The big probes, the share that gains least, are now nine tenths of a round.

**Kernel variants tried** (two 1M-state probes, one Modal host, seconds): running each probe's machine alone 38–46;
straight-line instructions on all probes at once 28–29; that and loops over straight-line bodies in lockstep 25–26
(kept). Running each distinct machine row once (probes of different states often hold the same registers; 1.7–5
fewer rows than probes) made no gain locally: the per-state assembly costs what it saves.

**Batching.** Three kinds, all exact:
- many classes per kernel call when their source level is small (above);
- whole learner runs in one container, a thread each, their searches on a process pool there (`pool.py`): a search
  now often takes milliseconds, so the breadth study's per-job Modal fan-out and Dict claims would dominate;
- problems posed on the same instances share one search (`engine.bfs_multi`, `jobs.search_jobs`, `--share`): their
  searches build the same levels and differ only in the goal test, so one search serves them all, each with the
  record its own search gives (check.py: 93 jobs in 7 shared searches, 0 records differ; 8.9 s alone, 2.7 s shared).
  In a round-0 pool about half the law-book problems share inputs; the saving measured on the breadth formulas is
  in §2.

**Caching.** Identical jobs run once per container, in flight or done (the breadth study's Dict did this across
containers); and every finished search is kept on the Volume, so a container restarted after preemption replays
its runs instead of searching again (before it existed, two preempted groups started over, losing 37 and 55
minutes). Each vocabulary's item classes and their bytecode are built once per worker and reused by every job with
that library.

**Skipping programs that behave exactly like one already tried.** The original already does this twice: items
that do the same to 24 random machines are one class, searched once, and a state reached at a lower cost is never
expanded again. The kernel adds one exact skip: an item that leaves a state as it was gives back a state already
seen, so it is dropped before it is hashed (8% on the check jobs, 25.3× to 27.4×). Skipping one order of two items
that commute was not done (§6).

### 2. The breadth study again, cheaper: same choices?

Every breadth run was run again on the compiled search, with its config unchanged, and compared with its record
round by round (`compare.py`): the pool, every probe's outcome (status, cost, the routine it keeps), the picks, the
library, the target search's programs and the general program. A run counts as identical when all of those match in
every round the original ran. Search-hours are summed as in BREADTH.md (every job's own seconds; a job shared
between runs counts in each).

| runs | runs identical | rounds compared | search-hours, breadth → now | saving | probes the original stopped at its time limit (of them, now different) |
|---|---|---|---|---|---|
| pairs anchored on the 2 hardest (`ct2a2_clean`), round 0, 10 targets | 10 of 10 | 10 | 26.58 → 4.68 | 5.7× | 0 |
| the same with shared searches (`--share`) | 10 of 10 | 10 | 26.58 → 3.95 | 6.7× | 0 |
| all pairs (`ct2_clean`), 1–2 rounds | 10 of 10 | 16 | 137.48 → 13.96 | 9.8× | 5 (0) |
| practice everything (`all_once`), 1 round | 10 of 10 | 10 | 24.01 → 4.78 | 5.0× | 0 |
| our curated list (`curated_clean`), 1–5 rounds | 9 of 9 | 30 | 18.58 → 4.17 | 4.5× | 0 |
| hardest first (`lp_clean`), 1–5 rounds | 6 of 10 | 31 | 191.53 → 19.98 | 9.6× | 261 (21) |
| the carry-over sequence, x² mod N to 10·x² | 1 of 1 | 6 | 52.04 → 2.17 | 24× | 83 (2) |
| its branches from the 34-routine library (x³, Montgomery's, Fibonacci, 10·x² + 1), round 0 | 4 of 4 | 4 | 64.90 → 1.62 | 40× | 65 (0) |

**All of the breadth study again: 515 search-hours became 51** (10×), counting each family once (the shared-search
rerun aside). 50 of the 54 runs made the same choices in every round they ran.

The differences all come from searches the original stopped at its time limit (120 s for a 30k-state probe, 600 s
for a 1M one): given the same state budget, the faster search finished them.
- In the sequence, two such probes found a program where the original had none; neither was picked.
- In four hardest-first runs (x³, Montgomery's, 10·x², 10·x² + 1) the first difference is in round 2 or 3, and it
  is the same one in all four: a practised problem's probe (gcd(X, N) for X = x + QN) had stopped at 120 s midway
  through cost 4 with 6 shortest programs (2 in Montgomery's); finished, it kept 8 (4), with the same routine first.
  The compression step (MDL over the practised programs) then wrote one of its merges with a bracket elsewhere, the
  same tokens grouped differently.
  That round's picks, routines and probe outcomes were the same. With a different library, the next rounds posed
  different problems and picked others (from round 3 or 4; by round 4, 41–43 of their probes had hit the original's
  time limit). The outcome is the same in all four: no general program in 5 rounds, the target searches' best fits
  round by round within one probe of the original's.

The saving grows with the library, because the interpreter was slowest on long routines: 5–6× on a round 0 from
scratch, 24× along the sequence, 40× from the 34-routine library.

**Against our curated list.** Choosing its own practice cost 3.8–7.9 times the search of our curated list where both
invent the formula in round 0 (x² + 1, x² − 1, 2x + 1, anchored pairs); now 1.8–4.4 times, since the many probes
gained more than the curated list's few big practice searches.

**The cheaper choosing rule, on top.** For round 0 on the ten formulas, the breadth study's all-pairs rule cost 67.8
search-hours (lookaheads 46.0 of them). Pairs anchored on the two hardest problems, on the compiled search with
shared searches, cost 3.95: **17× less**, inventing the same four formulas with the same pieces. Of that, the rule
gives 2.5× (BREADTH.md §5's 6–14× on the lookaheads, which were two thirds of the round), the compiled search 5.7×,
shared searches 1.2×. With lookaheads now cheap, the rule matters less than it did: on the compiled search all
pairs cost 1.7× anchored pairs on x² mod N's round 0, against 2.1× before.

### 3. What ten times buys: reach against budget

The target searched alone (no practice), at 300k, 3M and 30M states (`modal_scale.reach`), N ≤ 7. Distinct states
whose cheapest program has each cost (the last level is streamed: every fresh state counted, repeats included):

| vocabulary | cost 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| the 35 primitives (x² mod N, 10·x²: the same states) | 17 | 183 | 1,754 | 16,579 | 154,464 | 1,410,855 | 12,880,497 | 368,668,921 streamed |
| the 34-routine library after 2x + 1 (69 tokens) | 45 | 1,882 | 77,280 | 3,045,079 | 186,594,354 streamed | | | |

- **Primitives**: from cost 2 on, every level is 9.1–9.6 times the one before (cost 2 is 10.8 times cost 1). 300k
  states reach cost 6 (streamed), 3M cost 7, 30M cost 8: ten times the states, one more instruction, almost
  exactly. 30M states took 1,280 s.
- **The 34-routine library**: each level is 39–42 times the one before. 300k and 3M states both stop at cost 4
  (it passes their budgets: 4.7M states streamed); 30M stores cost 4 and streams cost 5, in 2,090–2,200 s. Ten times
  the states buys one instruction only every second time here: cost 6 would need cost 5 kept (on the order of 100M
  states) and cost 6 streamed, some 40 times more.
- So the 17× of §2 buys about 1.3 instructions on the primitives, and less than one over a carried library. The
  missed formulas need 13–22 instructions written from the square and the link, or 8 with Fibonacci's loop body of
  5 (BREADTH.md §1): engineering alone does not reach them; that is what §4 tries.
- Guided by learned costs (§4.4), the same library's search goes by bits: 25 bits at 300k states, 29 at 3M, 32 at
  30M (x³'s 4-token program over that library costs 28 bits): about 3 bits for each tenfold, where a token costs
  4–9 bits.

### 4. The levers

Four changes to what the learner poses and how it searches (§4.1–4.4), each generic (it reads the data's table of
answers, or the library, never the formula), in `sources.py` and `learn.py`; PRACTICE.md's depth-2 law book with them
(§4.5); and the targets in sequence with the anchored pairs rule, which the cheaper search made affordable (§4.6).
`learn.py` is `bar3/learner.run` line for line plus these options; with none of them on it reproduces the original
(smoke.py: 20 of 20 records).

| formula | hardest first, new sources | the same, all levers | anchored pairs, new sources | the same, all levers | BREADTH.md's best from scratch |
|---|---|---|---|---|---|
| x² mod N | round 0 (0.47 h) | round 0 (0.47 h) | round 0 (0.48 h) | round 0 (0.48 h) | round 0 |
| x² + 1 | round 1 (1.52 h) | round 1 (1.53 h) | round 0 (0.82 h) | round 0 (0.82 h) | round 0 (pairs) |
| x² − 1 | round 0 (0.49 h) | round 0 (0.49 h) | round 0 (0.50 h) | round 0 (0.50 h) | round 0 |
| x² + 3 | round 1 (1.28 h) | round 1 (1.30 h) | round 1 (1.30 h) | round 1 (1.32 h) | round 1 |
| 2x + 1 | round 0 (0.66 h) | round 0 (0.66 h) | round 0 (0.68 h) | round 0 (0.68 h) | round 0 |
| **Fibonacci** | **round 1** (1.19 h) | **round 1** (1.21 h) | **round 2** (2.74 h) | **round 2** (2.82 h) | no, 5 rounds |
| 10·x² | no, 5 rounds (7.89 h) | no, 5 rounds (8.28 h) | no, 5 rounds (4.65 h) | no, 5 rounds (4.91 h) | no, 5 rounds |
| 10·x² + 1 | no, 5 rounds (4.08 h) | no, 5 rounds (4.32 h) | no, 5 rounds (4.09 h) | no, 5 rounds (4.40 h) | no, 4 rounds |
| x³ | no, 5 rounds (2.77 h) | no, 5 rounds (2.94 h) | no, 5 rounds (4.65 h) | no, 5 rounds (4.98 h) | no, 5 rounds |
| Montgomery's | no, 5 rounds (3.27 h) | no, 5 rounds (3.39 h) | no, 5 rounds (3.02 h) | no, 5 rounds (3.21 h) | no, 5 rounds |

From scratch at N ≤ 7 (records `*_L1`; search-hours of the whole run).

**4.1 Steps that move two numbers at once (`recur`).** The learner's problems had one shape: x in X, N in N, one
answer in A. A step like Fibonacci's, (a, b) → (b, a + b mod N), has inputs in two registers and an answer in two.
The new source reads the target's table of answers, y(x, N) for every x < N, and asks whether the answer is a
function of the answers before it:
- order 1: is y(x + 1) a function of (y(x), N) on every N the data shows (sizes N ≤ 7 and N ≤ 10)? Then pose
  "y(x + 1) from y(x)", with y(x) in A or in X;
- else order 2: is y(x + 2) a function of (y(x), y(x + 1), N)? Then pose the number it computes, "y(x + 2) from
  y(x) and y(x + 1)", with the two in every ordered pair of X, A, U (6 problems), and the step that moves the pair,
  (y(x), y(x + 1)) → (y(x + 1), y(x + 2)) in place, in every ordered pair of A, U, T, X (12 problems).

Problems of the new shape set registers before the program runs and check several registers after (`RCand`,
`jobs.RProblem`); their routines keep the rule that inputs are left as they were, the inputs now being X, N and the
registers set beforehand, less those that hold the answer.

On Fibonacci (F_x mod N) it is order 2 (a + b mod N). Order 2 holds for every squaring formula too (x² mod N's
second differences are constant: y(x + 2) = 2y(x + 1) − y(x) + 2 mod N), so they get 18 such problems each, none of
them learnable at this size; 2x + 1 is order 1 ("add 2 and wrap").

**Fibonacci is invented from scratch, by every rule** (records `*_fib_L1`):
- In round 0 none of the 18 is learnable: the step needs "subtract N unless that goes below zero" (the link), and
  from primitives it is 10 instructions.
- Hardest first practises the law book's link in round 0, as it always does. In round 1, with the link a routine, 8
  of the 18 are learnable (5 one-number problems at cost 3–4, 3 steps at cost 5); it picks two steps, and the
  target search finds `INC.U L.X[T.rec7]` at cost 3, with `T.rec7` = `L.A[INC.U] MOV.X.U MOV.U.A T.link` (`T.link`
  is the law book's routine, `T.mod(X,N)@x+qN` in the records): the step (U, A) = (a, b) → (b, a + b mod N).
  Starting from (U, A) = (1, 0), that is (F₋₁, F₀), x steps leave Fₓ in A. 12 primitive instructions, a loop body
  of 10; right on every pair up to N = 120 (7,260 pairs).
- Pairs anchored on the two hardest take a round longer: in round 0 the link is one of six problems tied at the top
  cost, and pool order puts two others first, so no pair is tried with it; it is picked in round 1, and the step in
  round 2, from its single lookahead (alone it makes the target 3 tokens). The same program.
- 1.2 search-hours (hardest first), 2.7 (anchored pairs). BREADTH.md's rules never got there in 5 rounds (31.2
  search-hours for hardest first), nor did our curated list.

**4.2 Constants from the data (`const`).** At a fixed x the answer depends on N alone. When it is c mod N on every N
the data shows for one c with 2 ≤ c < the machine's cap, the smallest such c is posed as a problem with no input,
"make c" in A, T or U. It reads c off the remainders (the Chinese remainder theorem's smallest solution): 10·x²
at x = 1 holds 10 mod N, and N ≤ 10 pins c = 10 among 0…2,519; x² + 3 at x = 0 holds 3; 10·x² + 1 at x = 1 holds
11. Only x = 0 and x = 1 are read (x = 2 and 3 give 40 and 90 for 10·x²: constants of the same formula, 9 or more
instructions to build). A problem with no input is searched on one instance (x = 0, N = 1) and checked on every
instance of both sizes, so a program that leans on that instance's x or N is rejected: "make 10" takes 7
instructions and is found among 7,418 states, against 2.5M on two instances and out of reach on the 28 the other
problems use.

What it did: hardest first, on 10·x² and 10·x² + 1, spends three of its four round-0 picks on the same constant in
three registers (cost 7, the costliest problems in the pool), the fourth on the link. Neither formula came in 5
rounds, by either rule, with or without the other levers: the missing piece is x² mod N itself, which the base law
book never poses (§4.5). The constants did not disturb anything else: x² mod N, x² ± 1, x² + 3 and 2x + 1 came in the
same rounds as without them.

**4.3 The rest of the way from each routine (`rrem`, "targets made short in terms of learned routines").**
BREADTH.md's near-miss source poses "the rest of the way" from the target probe's 12 best attempts. This poses it from
every routine the learner has: for each `T.*`, the target's answer from X = the value that routine leaves in A, when
that is a function. With x² mod N carried, 10·x² + 1 is "10·x² mod N, then X + 1 mod N". It changed nothing:
- from scratch (`*_all3` against `*_new`, 40 runs), its 44 problems were never learnable at the probes' budget;
- along the anchored-pairs sequence, 2 of its 42 were learnable and none was picked; the picks and programs were the
  same as without it, step by step (hardest first's sequence learnt 23 of 49 and picked 5, and still missed 10·x²);
- the useful remainders were posed anyway, by BREADTH.md's near-miss source: once a carried routine is the target
  probe's best attempt (x² mod N for 10·x²), "the rest of the way" from it is among the near misses' remainders.
  That is how 10·x² got its piece (§4.6); this source's own "rest of the way from `T.tgt_h1`" was the same problem
  and was merged into the near-miss one.

**4.4 Search guided by what has been learned (`guided`).** When the round's target search finds nothing, a second
one runs with each token's cost in bits under a unigram model of the library's definitions (every routine's and
merge's program, add-½ smoothing; `learn.learned_costs`): tokens the library uses often cost 4–5 bits, primitives it
never uses 9. The search is then breadth-first by description length, so programs made of common pieces come first.
With an empty library every token costs the same and it is the plain search. On the 34-routine library it found x³
mod N's 4-token program at 25 bits after 1.6M states, where the plain search needed 4.8M (§3 for its reach). From
scratch it found nothing the plain search did not (`*_all3`).

**4.5 The depth-2 law book (`deep`).** PRACTICE.md's laws2 (one law applied to another's result) poses x² mod N
itself, the piece 10·x² needs. With every lever and laws2 (`*_deep`, on 10·x², 10·x² + 1, x³, Montgomery's; 2–3
rounds each, both rules): nothing invented. The pool grows to 96–151 learnable problems. x² mod N is posed every
round but mostly out of reach: once the link is a routine it is 6 tokens (square, then x links), and the 1M-state
probe over the bigger vocabulary stops at 5 (unsolved in 8 of the 9 rounds checked). Where it was learnable (hardest
first on x³, round 2, at cost 5), it tied with dozens of other cost-5 problems and was not among the four picked.

**4.6 Carry-over, with the anchored pairs rule.** The ten formulas in sequence (x² mod N, x² + 1, x² − 1, x² + 3, 2x +
1, 10·x², 10·x² + 1, x³, Montgomery's, Fibonacci), each solved one kept as a routine, as BREADTH.md §4 did with
hardest first alone. The anchored pairs rule could not be afforded along a sequence before (BREADTH.md's all-pairs
sequence was stopped after 5 hours on its second formula); now it can.

**10·x² is invented, in round 0 of its step**, by anchored pairs with the new sources and with every lever
(`seq_ct2a2_clean_new_targets_L1`, `seq_ct2a2_clean_all3_targets_L1`): `T.tgt_h1 T.res22 T.mod(X,N)@x+QN`, right on
every pair up to N = 100 (5,050 pairs), 50 primitive instructions.
- `T.tgt_h1`, carried, leaves v = x² mod N in A and X;
- `T.res22` turns v into 10·v mod N. It is the routine of a practised problem, "the rest of the way from what
  `T.tgt_h1` leaves in A, where x > 1" (BREADTH.md's near-miss remainder source: `T.tgt_h1` is now the target probe's
  near miss), solved at cost 5 over the library as `L.X[INC.X] L.X[INC.X T.res15]`, where `T.res15` was practised
  for an earlier formula; no constant 10 appears in it;
- the law book's full reduction ("X mod N for X up to N²"), a loop of links, ends it.

The pairs rule found it: the pair ("the target where x > 3", that remainder) saves 74.6 bits, the most of any pair,
and the round's target search then finds the 3-token program. The posed constant ("make 10 in A", the costliest
learnable problem and so one of the two anchors) was picked but is not used.

Hardest first, in its own sequence with the same levers, missed 10·x² in all 5 rounds, as BREADTH.md's chain did in
its round 0 (and in 5 rounds when rerun here). There the same remainder problem was out of its probe's reach,
because that sequence carries 34 routines where anchored pairs carry 25 (hardest first practises more problems the
formulas never use), and every search over the bigger library is dearer; by 10·x² + 1 it carries 66.

**The same sequence with only the breadth study's sources** (`seq_ct2a2_clean_targets_L1`) invents 10·x², 10·x² + 1
and x³ in round 0 of their steps too, with the same programs. So these three came from the choosing rule, made
affordable along a sequence, and from BREADTH.md's own sources (the near-miss remainder over a carried routine, the
target's restrictions, the law book), not from the new sources.

**10·x² + 1** then comes as `T.res27` = `T.tgt_ten INC.X T.res3` ("10·x² mod N, then add one and wrap"), the
routine of "the target where x < 3", whose 3-token program over the library was already the whole algorithm; right on
every pair up to N = 100. **x³** comes as BREADTH.md's branch had it, x times "add x² and reduce" over the carried
x² mod N.

| target | anchored pairs, new sources | anchored pairs, all levers | anchored pairs, breadth's sources | hardest first, all levers | BREADTH.md (hardest first; † from the branches) |
|---|---|---|---|---|---|
| x² mod N | round 0 (0.9 h) | round 0 (0.9 h) | round 0 (0.4 h) | round 0 (0.8 h) | round 0 (2.4 h) |
| x² + 1 | round 0 (0.3 h) | round 0 (0.3 h) | round 0 (0.1 h) | round 0 (0.2 h) | round 0 (1.9 h) |
| x² − 1 | round 0 (0.6 h) | round 0 (0.7 h) | round 0 (0.2 h) | round 0 (0.6 h) | round 0 (6.0 h) |
| x² + 3 | round 0 (1.4 h) | round 0 (1.5 h) | round 0 (1.2 h) | round 0 (1.3 h) | round 0 (9.7 h) |
| 2x + 1 | round 0 (1.9 h) | round 0 (2.1 h) | round 0 (2.0 h) | round 0 (0.4 h) | round 0 (6.9 h) |
| 10·x² | **round 0** (3.0 h) | **round 0** (3.2 h) | **round 0** (2.8 h) | no, 5 rounds (5.5 h) | no, round 0 (25.2 h) |
| 10·x² + 1 | **round 0** (2.0 h) | **round 0** (2.0 h) | **round 0** (1.8 h) | no, 5 rounds (29.7 h) | no, round 0 † (28.3 h) |
| x³ | round 0 (1.4 h) | round 0 (1.5 h) | round 0 (1.6 h) | round 0 (10.7 h) | round 0 † (15.1 h) |
| Montgomery's | no, 5 rounds (10.2 h) | no, 5 rounds (11.5 h) | no, 5 rounds (8.2 h) | **round 3** (54.9 h) | round 0 † (12.1 h) |
| Fibonacci | no, 5 rounds ‡ (13.5 h); branched after x³: no, 5 rounds ‡ (6.5 h) | no, 5 rounds ‡ (15.8 h) | no, 5 rounds (18.9 h); branched after x³: no, 5 rounds (7.7 h) | **round 1** (17.3 h) | no, round 0 † (9.5 h) |

Search-hours are each step's own. The library each step starts from grows by 4–7 routines a step for anchored pairs
(25 at 10·x², 36 at x³, 43 at Montgomery's, 67 at Fibonacci) and faster for hardest first (34 at 10·x², 66 at
10·x² + 1, 92 at x³, 99 at Montgomery's, 123 at Fibonacci). Montgomery's, Fibonacci, hardest first's last four steps
and the branches were finished after the spend-limit cut, replayed from their stores (§6). ‡: from round 1 the target
search finds a Fibonacci program that is right on every pair up to N = 40, but the general check rejects it (below).

**After the cut.** Every program here is right on every pair up to N = 120 (Montgomery's: all 3,600 odd-N pairs).
- *Hardest first's sequence* gets x³ in round 0 of its step (`M0.21`: x times "add x² and reduce", as in BREADTH.md's
  branch). It then gets **Montgomery's in round 3**: `Z.X[M0.24] T.tgt_minus1 T.res79`, 83 primitive instructions.
  That is a special case for x = 0 (`M0.24`, merged in the step's round 0), the carried x² − 1 mod N, and `T.res79`,
  the routine of "the rest of the way from what `T.tgt_minus1` leaves in X, where x > 3", practised in round 3, the
  round the target search finds the program. It was the study's dearest step: 54.9 search-hours over 99–122
  routines, with target searches of 73–117M states a round. Then comes **Fibonacci in round 1**:
  `INC.U L.X[T.rec86]`, the two-number step as from scratch (§4.1), 16 primitive instructions. 10·x² + 1 it misses in
  5 rounds.
- *Anchored pairs* miss Montgomery's in 5 rounds in all three sequences. From round 1 or 2 their target searches
  find programs that pass the N ≤ 10 check, but these are right on only 240 of the 400 odd-N pairs up to N = 40.
- ‡ *Fibonacci with anchored pairs and the new sources* (both sequences, and the branch after x³): the step is
  posed and practised in rounds 1–2. From round 1 the target search finds programs such as `INC.U L.X[T.rec58]` that
  pass the N ≤ 10 check and are right on all 820 pairs up to N = 40, but at N = 40 they need between 1 and 2 million
  machine steps. The general check runs a program for at most a million (`library/inventing/jobs.general`), so it
  rejects every one of them. They are slow because the practised step adds one number to the other a unit at a time,
  reducing mod N after each unit (`L.U[INC.X T.mod(X,N)@x+QN]`); hardest first's step adds it and reduces once. The
  breadth study's sources do not pose the step, and their sequence and branch find nothing.

### 5. Larger N

The learner at N ≤ 26 (351 pairs, a sample of 32 to search on, checked at N ≤ 30, general when right on every pair up
to N = 52), PRACTICE.md §8's size, with the new sources; and at N ≤ 50 (1,275 pairs, checked at N ≤ 56, general up to
N = 100). Values saturate at 2N² (1,352 and 5,000), so the counting algorithms still fit, but every loop runs longer:
a round costs 5.5–10 times what it does at N ≤ 7, and 14–15 times at N ≤ 50 (x² mod N: 0.47, 4.8 and 7.3
search-hours; 2x + 1: 0.66, 3.7 and 9.4).

| target | N ≤ 7 (anchored pairs / hardest first) | N ≤ 26, anchored pairs | N ≤ 26, hardest first | N ≤ 50, hardest first |
|---|---|---|---|---|
| x² mod N | round 0 / round 0 | round 0 (4.8 h) | round 0 (4.8 h) | round 0 (7.3 h) |
| x² + 1 | round 0 / round 1 | round 0 (6.3 h) | round 1 (24.8 h) | – |
| x² − 1 | round 0 / round 0 | round 0 (3.1 h) | round 0 (3.1 h) | – |
| x² + 3 | round 1 / round 1 | no, 2 rounds (13.1 h; stopped) | no, 2 rounds (14.7 h; stopped) | – |
| 2x + 1 | round 0 / round 0 | round 0 (3.7 h) | round 0 (3.7 h) | round 0 (9.4 h) |
| Fibonacci | round 2 / round 1 | round 1 (9.2 h) | round 1 (9.3 h; finished after the cut, §6) | round 1 (14.9 h) |

Every program found is right on every pair up to N = 120 (7,260 pairs), checked beyond the learner's own bound:
- x² mod N at N ≤ 50: `L.X[L.X[INC.A]] L.X[MOV.X.A T.mod(X,N)@x+qN]`, the same algorithm as at N ≤ 7;
- 2x + 1 at N ≤ 50: `L.X[INC.X] M0.1` (double, then add one and wrap);
- Fibonacci at N ≤ 26: `INC.U L.X[T.rec6]`, the same two-number step as at N ≤ 7 (anchored pairs), and
  `INC.T L.X[T.rec6]` with the step in another placement (hardest first); at N ≤ 50 the step in another
  placement, (X, A) = (a, b) → (b, a + b mod N) with T to hold x's copy: `INC.T L.X[MOV.X.T T.rec8]`;
- x² ± 1 at N ≤ 26: x² ± 1, then N links.

### 6. Dead ends, and how hard each was tried

Engineering:
- **Running each distinct machine row once** (probes of different states often hold the same registers: 1.7–5.1
  times fewer rows than probes on x² mod N's levels 2–4). An exact variant of the kernel; no gain: 216 against
  107 ns a state on one-token items, 338 against 374 on two-token loops, 624 against 514 on three-token ones (local
  benchmark over every class of the base vocabulary). The per-state assembly costs what the fewer runs save.
  Reverted.
- **Lockstep loops, measured on one host.** A first lockstep kernel (straight-line instructions on all probes at
  once) showed no gain locally and 1.4–1.6× on Modal; lockstep loops another 1.1–1.2×. Single-container timings
  were needed: across hosts, one kernel's time on the same probe varied from 19 to 29 s.
- **Packing state rows into fewer bits, partial-order reduction** (skipping one order of two items that commute):
  not tried. The second would change the minimal programs a search reads back, which the routine choice uses, so
  the records would no longer be the original's.
- **Preemptible containers without a store**: two groups were preempted and restarted from scratch (one of them
  twice), 37 and 55 minutes lost; the store of finished searches (`pool.py`) fixed it for later launches.
  Non-preemptible containers would have cost three times as much.

The levers:
- **Constants from the data** pose and solve "make 10", but on 10·x² and 10·x² + 1 hardest first then spends three of
  its four round-0 picks on one constant in three registers, and neither formula came in 5 rounds (four configs each:
  26 search-hours on 10·x², 17 on 10·x² + 1). Along the sequence both came, but with the breadth study's own sources
  as well, and their programs do not use the constant.
- **The rest of the way from each routine** (`rrem`) and **guided target search** (`guided`): the configs with them
  (`*_all3`) invented exactly what the configs without them did, in the same rounds: 20 runs against 20 from
  scratch, and the anchored-pairs sequences step by step. Of 76 guided searches, 2 found a program fitting the
  search instances where the plain search had none (Montgomery's, rounds 1 and 2); neither was general.
- **The depth-2 law book** (`*_deep`, 8 runs, 2–4 rounds each, about 9 search-hours): it poses x² mod N, but that
  piece was out of the probes' reach in 8 of 9 rounds checked, and not picked the one time it was learnable (§4.5).
  The runs were preempted once and repeated their rounds, and were then cut by the spend limit (below).
- **Hardest first along the sequence**: it missed 10·x² and 10·x² + 1 in 5 rounds each, because it practises
  problems the formulas never use, so its library grows fast (34 routines at 10·x², 66 at 10·x² + 1, against 25 and
  31 for anchored pairs) and every search over it gets dearer: the piece 10·x² needed was out of its probe's reach.
  The same sequence then invented x³, Montgomery's and Fibonacci (§4.6), at 10–55 search-hours a step.
- **Montgomery's in the anchored-pairs sequences**: not in 5 rounds of its step, in any of the three, where
  BREADTH.md's hardest-first branch had it in round 0 from its own 34-routine library and hardest first's sequence
  here in round 3 from 99. From round 1 or 2 their target searches find programs that pass the N ≤ 10 check but are
  aliases: right on 240 of the 400 pairs up to N = 40 (odd N), with any step budget. The branch's program leaned on
  a routine hardest first had practised for x² + 1; along these sequences x² + 1 came from other picks, and nothing
  here shows whether an equivalent piece was within reach.
- **Anchored pairs and ties.** On Fibonacci the link is one of six problems tied at the top cost in round 0, and the
  two hardest are taken in pool order, so its pairs are never tried: a round late (BREADTH.md §5 named the risk; not
  fixed here, since no pair helps Fibonacci in round 0 anyway, and when no pair helps the rule falls back on pool
  order).
- **x² + 3 at N ≤ 26**: not in 2 rounds by either rule (at N ≤ 7 it came in round 1); stopped there (below).

**How hard it was tried, in all.** About 562 search-hours (each job's own seconds, as in §2), about $62 of Modal
(16-core preemptible CPU containers, up to about 18 at once): 378 and $40 before the spend-limit cut, 184 and $22 to
finish the cut runs after it.

| part | runs | rounds | search-hours |
|---|---|---|---|
| the breadth study again, and the shared-search rerun (§2) | 64 | 130 | 69 |
| the levers at N ≤ 7: from scratch, the depth-2 law book, the sequences and their branches (§4) | 53 | 186 | 175 |
| larger N (§5) | 15 | 20 | 127 |
| the reach searches (§3), the engine timings (§1) and two test runs | – | – | about 4 |
| the cut runs, finished after the cut (below): the four sequences, the two branches, Fibonacci at N ≤ 26 | 7 | 38 more | 184 |

**What was cut short, and how it ended.** At about 14:27 UTC on Oct 24 Modal stopped every container in the
workspace and refused new ones ("the workspace has exceeded its spend limit": a limit set in Modal itself,
apparently near $3,000 for the month; this study's own share was about $40). Once the limit was raised, the runs with
a store were relaunched (below) and finished:

| run (record) | target | got to before the cut | how it ended |
|---|---|---|---|
| `seq_ct2a2_clean_new_targets_L1` | Montgomery's (step 9 of 10) | rounds 0–2, no program | Montgomery's: no, 5 rounds; Fibonacci: no, 5 rounds ‡ |
| `seq_ct2a2_clean_all3_targets_L1` | Montgomery's | rounds 0–2, no program | Montgomery's: no, 5 rounds; Fibonacci: no, 5 rounds ‡ |
| `seq_ct2a2_clean_targets_L1` (the breadth study's sources) | Montgomery's | rounds 0–3, no program | Montgomery's: no, 5 rounds; Fibonacci: no, 5 rounds (no source poses its step) |
| `seq_lp_clean_all3_targets_L1` | 10·x² + 1 (step 7) | rounds 0–1, no program | 10·x² + 1: no, 5 rounds; x³: round 0; **Montgomery's: round 3**; **Fibonacci: round 1** |
| `branch_ct2a2_clean_targets_L1_after_cube_fib_L1` | Fibonacci, from the state after x³ | rounds 0–2, no program (no source poses its step) | no, 5 rounds |
| `branch_ct2a2_clean_new_targets_L1_after_cube_fib_L1` | Fibonacci, from the state after x³, new sources | round 0's probes (in its store) | no, 5 rounds ‡ |
| `lp_clean_new_k26_fib_N1` | Fibonacci at N ≤ 26, hardest first | round 0, no program | **round 1** |
| `{lp,ct2a2}_clean_deep_{ten,tenrho,cube,mont}_L1` | the depth-2 law book | 2–4 rounds each, nothing | not rerun here (no store); run again as the subgoal study's controls, the same in every round these hold, to 5 rounds: 10·x² in round 4 under anchored pairs, nothing else ([subgoal/SUBGOAL.md](subgoal/SUBGOAL.md)) |
| `{lp,ct2a2}_clean_new_k26_plus3_N1` | x² + 3 at N ≤ 26 | rounds 0–1, no program | not rerun here (stopped by hand: its container's neighbour run crashed on the empty-program bug of §1) |

Every program the finished runs found is right on every pair up to N = 120 (7,260 pairs; Montgomery's 3,600, odd N).
‡ as in §4.6. Every other record was complete before the cut: the breadth study again (`*_s1`, `*_s2`), the levers
from scratch (`*_L1` but the above), N ≤ 26 for x² mod N, x² ± 1, 2x + 1 and Fibonacci (anchored pairs), N ≤ 50,
the reach searches (`reach_*.json`), the bounds (`bounds_*.json`).

**How they were finished.** The five commands below, at 14:43–14:45 PDT on Oct 24, each after
`integration/ablation_s3/spend.py 3750` printed LAUNCH OK. Each group replays its finished searches from its store on
the Volume (`store_<first run of the group>.jsonl`, matched by the job's content), so it was relaunched with exactly
the flags it had; it rewrote its records from round 0 and went on. Every record matches its pre-cut version (git
`5d108a7`) in each round they share (`compare.py --old`). Three of the five containers were preempted (3, 2 and 1
times); each restart replayed from the store. The last finished at 02:51 PDT on Oct 25. Logs: `logs/`. From
`cpu_first`:

```bash
SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --seq lp_clean_all3,ct2a2_clean_all3,ct2a2_clean_new --order h1,rho,minus1,plus3,affine,ten,tenrho,cube,mont,fib --tag L1
```

```bash
SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --seq ct2a2_clean --order h1,rho,minus1,plus3,affine,ten,tenrho,cube,mont,fib --tag L1 --share
```

```bash
SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --branch bar3/scale/records/seq_ct2a2_clean_new_targets_L1.state.cube.json --branch-config ct2a2_clean_new --targets fib --tag L1 --share
```

```bash
SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --branch bar3/scale/records/seq_ct2a2_clean_targets_L1.state.cube.json --branch-config ct2a2_clean --targets fib --tag L1 --share
```

```bash
SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --fresh lp_clean_new_k26 --targets fib --tag N1 --share
```

What it took: 184 search-hours of new rounds (the three sequences 146, of which hardest first's 105 and its
Montgomery's step alone 55; the plain sequence 21; the two branches 11; Fibonacci at N ≤ 26 6) and $21.92 (the app's
billing, 21:00 UTC Oct 24 to its stop), against the estimate of $12–15 and 3–5 hours: hardest first's steps over its
92–123 routines were dearer than its 66 had suggested, and its container ran for 12 hours. The depth-2 law-book runs
and x² + 3 at N ≤ 26 predate the stores and would start over (about 25 and 70 search-hours, $3 and $8); they were
not rerun here.

### 8. Bounds on the search space

Asked: can the search skip whole regions it now treats as unbounded? It already caps values (2k², 98 at N ≤ 7) and
drops a machine after 50k² + 1000 steps; it merges items that do the same to 24 random machines, and programs that
reach the same machine state. So numbers never explode and "skip programs that behave the same" is already in. The
space is big because there are many distinct short computations, each state holding every register on every
instance. Measured locally on the compiled engine (`bounds.py`, records `bounds_*.json`), on x² mod N's target
search over the 35 primitives (N ≤ 7, 28 instances, 300k states) and on the learner's own round-0 problems:

| bound | what it would merge or skip | measured | verdict |
|---|---|---|---|
| fewer instances (counterexample-guided search: search on a few, check on all, add a failing one, repeat) | states that agree on the instances searched | level 5: 154,464 states on 28 instances, 154,055 on 14, 108,365 on 7 (1.4×); but programs fitting 7 instances and failing others already come at cost 5, fitting 4 at cost 3 | **no gain.** On 64 of the learner's problems (30k-state probes) it spent 2.25× more states in all (each restart searches again) and 1.3× more time; it found 3 problems the plain probe missed, with that extra search. On 8 problems at 150k states, 2.6× more states, the same one found. It kept a different routine in 2 of 28 problems both solved |
| swapping the scratch registers T and U | a state and its mirror | 1.50× fewer at cost 3, 1.60× at 4, 1.70× at 5 | **1.7× at most, and only over the primitives**: a routine that uses T as scratch (the link does) breaks the symmetry. Merging would also change which shortest programs are read back, and so the routine kept: one of the 2 routine changes above is exactly the T/U mirror of the plain search's |
| registers no later instruction reads | states that differ only there | every register is read by some instruction (MOV, L, Z), so none is dead before the last level; at the last level only A matters (154,464 states hold 946 distinct answers at cost 5) | **nothing to save**: the last level is streamed, not kept, and generating it is the work |
| meet in the middle | search back from the goal | 381 of 5,988 items (6%) are one to one on a level's states (5 of 35 primitives, 45 of 310 two-token loops, 331 of 5,643 three-token ones); and the goal is a set (A fixed, every other register free), not a state | **not practical** without a symbolic backward search (conditions on registers, not states): the backward frontier is not a list of states |
| a lower value cap | values between the cap and 98 | cap 98: 154,464 at cost 5; 49: 154,483; 20: 154,560 | **no gain, and not sound**: short programs never reach big numbers, and a lower cap changes what programs compute |
| the step limit | machines that run long | dead states are rare at these sizes | already a bound; never the reason a level is big |

So none of the bounds asked about shrinks the space by more than 1.7×, against the 9–10× each instruction costs
(§3). What does shrink it is a smaller vocabulary in effect: fewer, more useful items, which is what the pairs rule
did along the sequence (§4.6: 25 routines carried instead of hardest first's 34, where it found what hardest first
could not), and what guided search does by price (§4.4).

### 7. Files and how to rerun

Everything is in `cpu_first/bar3/scale/`. It imports `bar3/`, `bar3/breadth/`, `library/` and
`library/inventing/` and changes none of them.
- `engine.py`: the compiled search: the bytecode compiler, the kernels (one machine, one state, a batch of states,
  the item signature), `enumerate_items`, `bfs` (bar3/search.bfs with the kernel) and `bfs_multi` (one search for
  problems on the same instances). `MODE` picks the kernel variant (2: lockstep, kept).
- `jobs.py`: `search_job` (bar3/jobs.py's, on the engine; `"engine": "py"` runs the original), `search_jobs` (shared
  searches), problems of other shapes (`RProblem`), learned costs in the item classes.
- `pool.py`: `Searcher`, jobs on a process pool: identical jobs once, shared searches (`share`), and a store of
  finished searches that a restarted container replays.
- `sources.py`: the new sources (`recurrences`, `constants`, `routine_remainders`) and their problem type `RCand`.
- `learn.py`: bar3/learner.run with the carry-over, the anchored pairs rule, the new sources and the guided target
  search built in; `learned_costs`; `sequence`.
- `configs.py`: the breadth study's configs and the levers' (`*_new`, `*_rrem`, `*_guided`, `*_all3`, `*_deep`,
  `*_k26`, `*_k50`). `drive.py`: fresh and carried runs through the breadth study's Tap.
- `modal_scale.py`: Modal. `group` runs many learner runs in one container (16 cores, 64 GB, preemptible, retried
  once); `reach` the budget experiment of §3; `profile` the engine timings of §1. Limits: every search keeps its
  job's state and time budget; containers have a 24-hour timeout and a hard memory limit of 1.5 times the request.
- `check.py` (exactness, about seven minutes locally, mostly the original search), `smoke.py` (the learner end to
  end), `compare.py` (a run against the breadth run it repeats), `report.py` (the tables here), `bounds.py` (§8,
  about two minutes locally).
- `records/<run>.jsonl`: one line per round, the breadth study's fields (plus `guided` where it ran), named
  `<config>_<target>_<tag>`: tag `s1` the breadth study again (§2), `s2` with shared searches, `L1` the levers at
  N ≤ 7 (§4; sequences `seq_*`, branches `branch_*`), `N1` larger N (§5), `t0`/`t1` the first test runs;
  `*.state.<target>.json`: a sequence's carried state after each target (for `--branch`); `reach_*.json` (§3);
  `bounds_*.json` (§8);
  `calls_*.json`: the Modal call ids. The Volume also keeps each group's store of finished searches
  (`store_*.jsonl`), which `--fetch` skips.
- `logs/`: the relaunch of the cut runs (`relaunch.md`) and each relaunched container's log, `app_<app id>.log`.

The compiled engine needs numba (0.68, with numpy 2.5.4; the Modal image pins both). Locally, any Python with them:

```bash
cd cpu_first && <python with numba> -m bar3.scale.check
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.smoke
```

```bash
cd cpu_first && SCALE_CPU=16 SCALE_MEM=65536 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --fresh ct2a2_clean --targets h1,rho --tag rerun --share
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/modal_scale.py --seq lp_clean_all3 --order h1,rho,minus1,plus3,affine,ten,tenrho,cube,mont,fib --tag rerun
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/scale/modal_scale.py --fetch
```

`--fetch` needs a container; the Volume can also be read without one (it still worked while the spend limit
refused every container), into a fresh folder, then the records copied over:

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal volume get --force bar3-scale v1 /tmp/bar3-scale-volume
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.compare bar3/scale/records/ct2a2_clean_*_s2.jsonl
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.report --matrix bar3/scale/records/*_L1.jsonl
```

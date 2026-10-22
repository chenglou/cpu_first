# Inventing without our inputs: does search-and-compress still find x² mod N?

*Written 2026-10-22. Code in `cpu_first/library/inventing/`; records in `library/inventing/records/`. Nothing
committed, nothing submitted, no certified file touched.*

## In plain words

<!-- PLAIN -->
**The question.** [LIBRARY.md](LIBRARY.md) showed that a search-and-compress loop rebuilds my hand-written pieces of
x² mod N: the ripple subtraction, the restoring link, the chain, the division step, the reduction loop, the
multiply rows. But the loop had three things from me:
1. **my instruction set:** 32 digit instructions, among them "subtract with borrow", "multiply-accumulate a digit"
   and "keep the copy if it borrowed";
2. **my practice tasks:** 26 small problems I picked (subtract, compare, one division step, reduce, multiply, ...),
   each a piece of the answer;
3. **my merge rule:** greedy BPE with the tie-breaks and the "solved tasks become routines" step I added along the
   way.

The hypothesis under test: a learner that *invents* algorithms keeps the loop and needs none of the three. So I took
them away, one at a time and together, and ran the loop on the tiniest x² mod N: every N up to 7 (three binary
digits), every x below N.

**The answer.** Without my practice tasks: no, on either instruction set. With them: yes, even on bare instructions,
as long as solved tasks become routines. On a bare machine (five counters; add one, subtract one, clear, copy,
"repeat r times", "if r is 0"), the bare versions of my practice tasks led the loop to invent a restoring link and a
reduction loop, and to put together a correct x² mod N in its second round, after half a search-hour. So of my three
inputs, the practice tasks are the one the loop could not do without, and one part of my merge rule (routines) was
needed too. The digit instructions were not needed at this size; they decide *which* algorithm comes out.

**What it invented with the practice tasks, on bare instructions.** Its x² mod N, 4 tokens:

    T.sq  L.X[ MOV.X.A  T.csub ]          square x; then x times: subtract N unless that overshoots

    T.sq   = L.X[L.X[INC.A]]                            A = x·x, by counting
    T.csub = MOV.A.X  L.N[MOV.T.A DEC.A]  Z.T[MOV.A.X]  the restoring link

`T.csub` is the link from my skeleton, in counters: copy X to A, then N times "remember A in T, take one off A"; if T
ends at 0, A had hit 0 before the N-th step (the subtraction went below zero), so put X back. T plays the borrow. Both
pieces were practice tasks' solutions ("x², by counting", "x mod N for x < 2N"), kept as routines. Running the
link x times is always enough, since x² < x·N. The program is right on all 3,240 pairs up to N = 80.

**Without the practice tasks, in one example.** On the same machine, x² mod N also has an 8-instruction program:
count to x·x, wrapping around to 0 at N.

    MOV.T.N  L.X[ L.X[ INC.A DEC.T Z.T[CLR.A MOV.T.N] ] ]

    T = N; repeat x times: repeat x times: A = A + 1, T = T − 1, and if T is 0: A = 0, T = N.

The inner step, "add one and wrap around at N" (5 instructions), is the whole trick. **Given that one step as a
ready-made instruction, the same search finds a correct x² mod N in about one second** (4 tokens), and the
growing-sizes practice finds it by N ≤ 4. With only the target to practice on (at N ≤ 2, 3, ..., 7), the loop never
invented that step or the link:
- **Small sizes are solved by shortcuts.** For N up to 3 the answers are 0, 0, 1, 0, 1, 1, and "copy x, then count
  down" gets all six right (`L.X[MOV.A.X DEC.X]`). Shortcuts are shorter than the real algorithm, so the search,
  which returns the shortest programs that fit, finds them first. They fit the small pairs and 15–19% of the pairs up
  to N = 40.
- **At N = 5 the shortcuts run out, and the real programs are out of reach.** The counter is 8 instructions with a
  loop body of 5; square-then-links is 11. The search stops at 5 to 7 instructions (the 7th level alone is 39 million
  candidate states), and like the library study's search it builds loop bodies of at most 2 tokens: longer ones have
  to come from the library.
- **Compression can only name what is already in the solved programs.** Those were shortcuts, so the library got
  pieces of shortcuts.
- **Self-made practice did not supply the pieces either.** The DreamCoder-style variant runs random programs from its
  library on the target's own inputs and practices on whatever they compute. That gave reusable pieces (x + N,
  x·(N + 1), "is x zero?"), but the link and the wrap-around step only do something inside a loop, and random
  programs, like the search, build loop bodies of at most 2 tokens. No dream asked for them.

**What it invented without the practice tasks.** Shortcuts for the smallest sizes, one of them an alien one (for
N ≤ 4 it computes x² − (x − 1)·N: a reduction that guesses the quotient); addition (`L.N[INC.X MOV.A.X]`, x + N); a
unary multiply-add (`L.X[L.N[INC.X MOV.A.X]]`, x·(N + 1)), a counting cousin of the skeleton's multiply rows; a zero
test; constants like N − 1 and x + 2. On the digit instructions, the dreams gave subtract-with-borrow loops that stay
at one place (`P[LDA.X SBB.N]`) or shift digits in (`CLA P[SBB.N BRG.X]`, the reduction loop's outer shape
`CLA P[… BRG.X]` with no link inside). None of the reduction pieces.

**Which of my inputs mattered.**

| | my practice tasks | the target only, at growing sizes | + self-made practice (dreams) |
|---|---|---|---|
| **my digit instructions** | found (library study: 22 instructions; 6.6 billion states, 4.5 search-hours) | not found: stuck at 2 digits | not found: stuck at 2 digits |
| **bare instructions** | found with routines (this study: 11 instructions; 132 million states, 0.5 search-hours); not found without | not found: shortcuts up to N = 4, stuck at N = 5 | not found: the same |

- **The practice tasks did the inventing.** Each is a problem whose shortest solution is one piece of the answer
  (square, subtract with undo, reduce). Without them the loop only meets problems whose shortest solutions are
  shortcuts or odd bits.
- **The instruction set decides which algorithm appears.** The bare algorithm counts: about x² + 2·x·N steps,
  nothing for N ≤ 80 (at most 31,759), about 10¹⁶ for H1's 8-digit numbers. The digit algorithm works place by place
  and runs H1 as it stands.
- **The merge rule: routines yes, MDL vs BPE no.** With practice tasks, MDL and BPE both got there, and both only
  with "solved tasks become routines" (the composition step the digit study needed too): without it, the link was
  never one token, the reduction stayed out of reach, and both runs stopped. Without practice tasks, greedy BPE was
  worse than MDL: it adds 12 merges a level whether they help or not, the vocabulary grew to 95 tokens, and the
  search reached one level less.

**What it says about the hypothesis.** The loop survives losing my instruction set and does not survive losing my
practice tasks (or the routines step of my merge rule). It compresses what the search finds, and the search finds
the shortest programs that fit the problems it is given; the target's own bigger sizes and random programs' outputs
are problems whose shortest solutions don't contain the pieces. So "ditch the curated tasks" is the part that fails
here: a learner that invents needs a source of good practice problems of its own, problems whose short solutions are
the pieces. One idea is to pose them from its near misses ("what does my almost-right program get wrong?"). This is
one tiny setup; it does not rule the hypothesis out, it says which piece is missing.
<!-- /PLAIN -->

## Details

### 1. The bare machine (`inventing/brm.py`)

- **Registers.** X = x, N = N (the inputs), A (the answer is read here), T and U (scratch). All but X and N start
  at 0, so a program may assume clean scratch.
- **Instructions (35 tokens).** `INC.r`, `DEC.r` (stops at 0), `CLR.r` for each of the 5 registers; `MOV.d.s` (d = s)
  for every ordered pair. Values saturate at a cap (2k² at size k, enough for x²).
- **Loops (10 kinds).** `L.r[body]`: repeat the body r times, r read once on entry (the LOOP language of Meyer and
  Ritchie). `Z.r[body]`: run the body if r is 0. Nesting is unbounded.
- **Step limit.** A run that takes more than 50k² + 1000 steps is dropped (the counter above needs about 6k²).
  Each probe (x, N) runs as its own machine, so loop counts can depend on the data.
- Nothing here is digit-specific: no places, no borrow, no cells. It is the "handful of bare ops" of the request:
  inc, dec, zero test, move, loop (plus clear).

### 2. Search, sizes, acceptance

- **The search** is the library study's (`inventing/bfs.py`, adapted from `library/search.py`): breadth-first by cost
  (tokens; a loop costs 1 plus its body), one program kept per distinct machine state (on every probe), all minimal
  programs recoverable. Items are tokens and loops whose bodies cost at most 2 (`max_body` 2: so `L.N[L.X[DEC.A]]`,
  cost 3, is one item); longer bodies have to come from library tokens. Budget: 3 million states held (the next
  level is streamed: checked but not kept), 40 minutes per search. State counts below add the held levels' distinct
  states and the streamed level's candidates (new against the lower levels, repeats within the level not removed),
  as in the library study.
- **Sizes (the target's own data, smallest numbers first).** Size k is every pair (x, N) with N ≤ k and x < N: 3
  pairs at k = 2, 28 at k = 7 (every N with three binary digits). A program is accepted at size k when it is right on
  every pair of the size.
- **General** means right on every pair with N ≤ 40 (820 pairs), on a machine big enough for the counting
  algorithm. That is what "reaches x² mod N" means below for the bare machine.
- **Digit arms** use the library study's machine, instructions and search unchanged (`library/dsl.py`,
  `library/search.py`); sizes are digit counts: 1, 2, 3 digits in bases 2 and 3 (settings (2,1)+(3,1), (2,2)+(3,2),
  (2,3)+(3,3)). Accepted when right on 200 random pairs of every setting of the size; general when right on 200
  random pairs of each of the library study's 8 verification settings (bases 2 to 10, up to 4 digits).

### 3. The arms (`inventing/loop.py`, `CONFIGS`)

| arm | instructions | practice | merges | config |
|---|---|---|---|---|
| A, plain search | bare (or digit) | none: the target alone at N ≤ 7 | none | `A_s1` (3M states), `A_s1_big` (16M) |
| B, the library study | digit | my 26 curated tasks | BPE / MDL, routines | `library/records/` (§4.2) |
| B-bare (cross) | bare | the bare machine's analogues of my tasks (17, §4.2) | MDL / BPE, with or without routines | `Bbare_mdl_rt`, `Bbare_bpe_rt`, `Bbare_mdl`, `Bbare_bpe` |
| C, no curriculum | bare | the target only, at N ≤ 2, 3, ..., 7 | MDL (half-price library) / BPE | `C_mdl`, `C_bpe` |
| C-digit (cross) | digit | the target only, at 1, 2, 3 digits | MDL / BPE | `Cdig_mdl`, `Cdig_bpe` |
| D, self-made practice | bare | C + 16 dreams per size | MDL / BPE | `D_mdl`, `D_bpe` |
| D-digit (cross) | digit | C-digit + 16 dreams per size | MDL / BPE | `Ddig_mdl`, `Ddig_bpe` |
| checks | bare | plain search and C, with the wrap step given as a token | none / MDL | `A_oracle`, `C_oracle` (local) |

**The loop (C and D).** For each size in turn: search for the target (accepted = right on every pair of the size);
in D, first make dreams and search for them too; put every accepted program in the corpus; compress (MDL: merge the
run of items that most shortens corpus + library, the library counted at half price, until nothing pays; BPE: up to 12
merges of item pairs that appear in at least two tasks' programs, ties to the larger pair); the new tokens join the
vocabulary of the next size. Stop when a target program is general.

**Dreams (D).** Before each size's search: draw random programs from the current library (1 to 4 items; an item is a
token 60% of the time, otherwise a loop over 1–2 tokens; at most 6 tokens in all), run each on the size's pairs (the
target's inputs), and keep 16 whose output varies, is not just x or N, and is new. Each is a task: find a program
with the same outputs (accepted on every pair of the size), budget capped at the dream's own cost. On the digit
machine a dream must also give the same output from two random starting states (the digit machine starts with
random scratch).

**The loop (B-bare)** is the library study's: every unsolved task searched at each level with the same budget,
solved programs compressed, and, in the `_rt` configs, every newly solved task's simplest program made one routine
token `T.<task>` (as in `library/wake_sleep.py`, whose `promote` it calls). It stops when x² mod N is general or a
level adds nothing.

### 4. Results

Every run in one table (`summarize.py --table`; states and search-hours over all of a run's searches, dreams and
curated tasks included; round n is level n − 1 in the records):

| run | rounds | sizes | x² mod N | states | search-hours |
|---|---|---|---|---|---|
| A_s1_r1 | 1 | N ≤ 7 | no | 40.7M | 0.09 |
| A_s1_big_r1 | 1 | N ≤ 7 | no | 40.7M | 0.11 |
| Bbare_mdl_rt_r2 | 2 | N ≤ 7 | general in round 2: `T.sq L.X[MOV.X.A T.csub]` | 132.1M | 0.51 |
| Bbare_bpe_rt_r3 | 2 | N ≤ 7 | general in round 2: the same | 132.2M | 0.65 |
| Bbare_mdl_r4 | 2 | N ≤ 7 | no | 183.0M | 0.81 |
| Bbare_bpe_r4 | 3 | N ≤ 7 | no | 403.4M | 2.23 |
| C_mdl_r1 | 6 | N ≤ 2 … 7 | no (shortcuts to N ≤ 4) | 70.3M | 0.08 |
| C_bpe_r1 | 6 | N ≤ 2 … 7 | no (shortcuts to N ≤ 4) | 57.2M | 0.08 |
| D_mdl_r1 | 6 | N ≤ 2 … 7 | no (shortcuts to N ≤ 4) | 45.7M | 0.95 |
| D_bpe_r1 | 6 | N ≤ 2 … 7 | no (shortcuts to N ≤ 3) | 131.1M | 3.75 |
| Cdig_mdl_r2 | 3 | 1, 2, 3 digits | no (1 digit only) | 28.1M | 0.02 |
| Cdig_bpe_r3 | 3 | 1, 2, 3 digits | no (1 digit only) | 28.1M | 0.01 |
| Ddig_mdl_r2 | 3 | 1, 2, 3 digits | no (1 digit only) | 101.6M | 0.11 |
| Ddig_bpe_r3 | 3 | 1, 2, 3 digits | no (1 digit only) | 249.5M | 0.20 |
| library study (B, digit; `bpe_big_routines_r11`) | 6 | (2,3) + (3,2) | general in round 6: `LDA.X STA.Y T.mul STA.X T.reduce` | 6,557M | 4.54 |

#### 4.1 Arm A: plain search

| machine | size | budget | result | states | time |
|---|---|---|---|---|---|
| bare | N ≤ 7 (28 pairs) | 3M | no program of 7 or fewer tokens | 40.7M | 5.4 min |
| bare | N ≤ 7 | 16M | the same: level 7 (39M candidates) is streamed either way | 40.7M | 6.7 min |
| digit (library study `bpe_r3` level 0) | (2,3) + (3,2) | 3M | no program of 5 or fewer | 13.4M | 24 s |
| digit (`bpe_big_r5` level 0) | (2,3) + (3,2) | 16M | no program of 6 or fewer | 286M | 5.9 min |

Level sizes on the bare machine grow about 9–10× per token (1, 17, 183, 1,754, 16,579, 154,464, 1.41M; level 7,
streamed, 39.1M), so level 8 would be hundreds of millions of states. The 8-token counter is not even in this
search's space: its loop bodies cost 6 and 5, above `max_body` 2. On the digit machine, the program the library study
found is 22 primitives.

#### 4.2 Arm B: my inputs (the library study, already at this size)

The library study searched at exactly the tiny size: settings (2,3) and (3,2), base 2 with three digits and base 3
with two. Its records are the reference, not rerun:
- 3M states per search, BPE (`bpe_r3`) or half-price MDL (`mdl_half_r6`): 19 of 26 tasks, including the reduction
  loop; not x² mod N.
- 16M states, then loop bodies up to cost 3, then solved tasks as routines (`bpe_big_routines_r11`, the r5 → r8 → r11
  chain): all 26 tasks; x² mod N = `LDA.X STA.Y T.mul STA.X T.reduce` (22 primitives) at level 5, right on all 8
  verification settings. 6.56 billion distinct states, 4.54 search-hours in all.

**B-bare: the same kind of tasks on the bare machine** (`Bbare_mdl_rt_r2`). The 17 tasks, all at N ≤ 7 (28 probes,
one per pair (x, N) of the size, with x redrawn from the task's range): copy x, copy N, x + 1, N + 1, x + N, 2x,
x − N and N − x stopping at 0, [x < N] and [x = N] (x < 2N), x mod N for x < 2N and for x ≤ N, x mod N for x < N²
(two samples), x·N, x², and x² mod N. A program is accepted when it is right on all 28 probes. Half-price MDL, then
routines:

| level | solved (tokens) | not solved |
|---|---|---|
| 0 (35 tokens) | copy, copyN (1); inc, incN (2); add, dbl, monus, monusN, mul, sq (3); csub1 (4); lt, eq (5); csub (6, found while streaming level 6: 4.7M states, 65 s) | mod, modN1 (nothing of ≤ 7 tokens, 42–44M states, ~9 min each); x² mod N (as arm A) |
| 1 (50 tokens: + 4 merges, 11 routines) | mod, modN1 = `L.X[T.csub MOV.X.A]` (3; 14k states, 27 s); x² mod N = `T.sq L.X[MOV.X.A T.csub]` (4; 169k states, 43 s) | — |

The x² mod N is general: 820/820 up to N = 40 in the run, 3,240/3,240 up to N = 80 checked afterwards (at most
31,759 steps). In all 132M states, 0.51 search-hours. Expanded it is 11 instructions:
`L.X[L.X[INC.A]] L.X[MOV.X.A MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]]`. The level-0 merges were `L.X[INC.A]`
("add x"), csub1's program `L.X[DEC.N] L.N[MOV.A.X]`, and two alien runs (`L.N[MOV.N.X DEC.X]`, and it inside
`MOV.A.X L.N[MOV.N.X DEC.X] L.N[MOV.A.X]`); none of them is in the final program, which uses only routines.

BPE with routines (`Bbare_bpe_rt_r3`) did the same: level 0 the same 14 tasks with the same programs, 12 BPE merges
(`L.X[INC.A]`, `L.X[DEC.N]`, `INC.X MOV.A.X`, and alien pairs like `MOV.U.X DEC.X`) and 12 routines; level 1 mod,
modN1 and the same x² mod N (4 tokens, 311k states, 59 s). 132M states, 0.65 search-hours.

**Without routines both fail** (`Bbare_mdl_r4`, `Bbare_bpe_r4`): level 0 is the same (14 tasks, the same merges),
but the link stays a 6-token program used once, so no merge makes it one token, and a loop of links needs a 7-token
body. Level 1: mod, modN1 and x² mod N not found (≤ 6 tokens; MDL 12–20M states each, BPE 34–62M); MDL then adds no
merge and stops; BPE adds 5 (`L.N[MOV.A.X]`, `L.N[DEC.X]`, ...) and fails again at level 2. MDL 183M states and 0.81
search-hours, BPE 403M and 2.23.

#### 4.3 Arm C: the target only, at growing sizes

| size | MDL | BPE | general |
|---|---|---|---|
| N ≤ 2 | `MOV.A.X` (1 token) | same | 15% |
| N ≤ 3 | `L.X[MOV.A.X DEC.X]` (3) | same | 19% |
| N ≤ 4 | `L.X[MOV.A.N] L.X[DEC.N] L.N[L.X[DEC.A]]` (7; 40M states) | same | 18% |
| N ≤ 5, 6, 7 | nothing of ≤ 6 tokens (≈10M states each) | nothing of ≤ 6 (≈6M each) | |

The N ≤ 4 program computes N − x·(N − x) for x > 0 (A = N; N = N − x; then subtract x, N − x times), which is
x² − (x − 1)·N: x² mod N whenever the quotient of x² by N happens to be x − 1. That is true for every pair up to
N = 4 and for 18% of the pairs up to 40: an alien shortcut, a reduction that guesses the quotient. MDL's only
merge was the bulk of that shortcut (`DEC.N L.X[MOV.A.N DEC.N] L.N[L.X[DEC.A]]`, after the search had found a
cost-7 spelling with it); BPE's was `MOV.A.X DEC.X`. Neither helps at N = 5. In all: 70M states (MDL), 57M (BPE),
about 5 search-minutes each.

C-digit: 1 digit `LDA.X` (x² mod N = x when N ≤ 2); 2 digits and 3 digits: nothing of ≤ 5 tokens (14M states
each). One solved 1-token program gives MDL nothing to merge, so the library stayed empty and 2 and 3 digits were
plain search (arm A). BPE (`Cdig_bpe_r3`): the same, state for state (it needs a pair in two tasks' programs).

#### 4.4 Arm D: self-made practice

Bare, MDL (`D_mdl_r1`): 16 dreams posed and solved at every size. Target: N ≤ 2 `MOV.A.X`; N ≤ 3
`L.X[MOV.X.A Z.X[INC.A]]` (3, 19% general); N ≤ 4 the same shortcut as C, one token shorter with a dream's merge
(`L.X[MOV.A.N]`): 6 tokens, 27M states; N ≤ 5, 6, 7: nothing of ≤ 5 tokens. The vocabulary grew from 35 to 52
tokens, which moves the 3M budget's wall from 6 tokens down to 5. 17 merges (§5). 45.7M states, 0.95
search-hours with the dreams.

Bare, BPE (`D_bpe_r1`): 12 merges at every level (pairs of pairs of dream fragments), vocabulary 35 → 71 tokens;
N ≤ 3 the same as MDL; N ≤ 4, 5, 6, 7 not found (5 tokens reached; at N ≤ 7 the 40-minute limit struck inside
level 5). Vocabulary 95 tokens by N ≤ 7. 131M states, 3.75 search-hours.

Digit, MDL (`Ddig_mdl_r2`): 1 digit `LDA.X`, plus 14 dreams (only 14 distinct behaviours turned up in 6,000
draws) and 5 merges: `P[LDA.X SBB.N]`, `P[CLA SBB.N]`, `CLA P[SBB.N BRG.X]`, `CLC LDA.N`, `LDA.N BRG.X`. 2 digits:
16 dreams, 7 more merges (`SHL.X LDA.X`, `CLC LDA.N ADC.N` = 2N, `P[LDA.N BRG.X]`, ...), the target not found (≤ 5
tokens, 20M states). 3 digits: not found (≤ 5 tokens, 37M states). The dreams are digit-level oddities: subtract with
borrow without stepping to the next place, shifts. 102M states, 0.11 search-hours.

Digit, BPE (`Ddig_bpe_r3`): 12 merges a level (`LDA.X SBB.N`, `CLA SBB.N`, `BRG.X SBB.N LDA.N`, ...); 2 digits not
found (≤ 5 tokens, 39M states); 3 digits not found (≤ 5 tokens, 71M states). 36 merges in all, none holding
`SBB.N INC`. 250M states, 0.20 search-hours.

#### 4.5 The missing piece, given by hand

With one extra token `MI = INC.A DEC.T Z.T[CLR.A MOV.T.N]` ("add one, wrap at N"):
- plain search at N ≤ 7 (`A_oracle`): cost 4, 22,909 states, 1.2 s. Programs: `MOV.T.N L.X[L.X[MI]]` and
  `L.X[L.X[MI]] MI` (the first MI from T = 0 wraps at once, so it only sets T = N). Both general (820/820 up to
  N = 40).
- growing sizes (`C_oracle`): N ≤ 2 and N ≤ 3 still the shortcuts (1 and 3 tokens beat 4), then at N ≤ 4 the general
  program, and the run stops.

So on the bare machine the gap between "stuck at N = 5" and "general" is one piece: this 5-instruction step, or the
6-instruction link that B-bare's practice task csub made a routine. Nothing in arms C and D ever needed either at a
cost the search could reach. A dream could not ask for them either: on its own the step leaves A = 0 (a constant,
discarded), and inside a loop either piece is a body of 5 or more tokens, while the dream sampler, like the search,
builds bodies of 1–2 tokens (library tokens count as one, and the library never had the pieces' parts).

### 5. What it invented, against the skeleton's table

| skeleton part (LIBRARY.md) | bare analogue | with my tasks (B-bare) | without them (C, D) |
|---|---|---|---|
| ripple subtraction | x − N, stopping at 0 | `T.monus` = `L.N[DEC.X] MOV.A.X` | only inside the N ≤ 4 shortcut (`L.N[L.X[DEC.A]]`) |
| comparison (the borrow out of the ripple) | [x < N] | `T.lt` = `INC.X L.N[DEC.X] Z.X[INC.A]` (x + 1 − N, stopping at 0, is 0) | no |
| restoring link | x mod N for x < 2N | `T.csub` = `MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]`, T as the borrow | no |
| chain of links, division step | (no places, no analogue) | — | — |
| reduction loop | x mod N by repeated links | `T.mod` = `L.X[T.csub MOV.X.A]` (x links; enough for x < N²) | no |
| multiply rows | x·N, x² by counting | `T.mul` = `L.N[L.X[INC.A]]`, `T.sq` = `L.X[L.X[INC.A]]` | a cousin, x·(N + 1) = `L.X[L.N[INC.X MOV.A.X]]` (D) |
| x² mod N (H1's step: multiply, reduce) | square, then links | `T.sq L.X[MOV.X.A T.csub]` | no |

On the digit instructions without my tasks, nothing of the table either: the dreams' merges hold `SBB.N` (the
subtraction cell) but never `SBB.N INC` (subtract and step: the ripple's body), and `CLA P[SBB.N BRG.X]` has the
reduction loop's outer shape without a link inside.

Alien ones (not in the skeleton): without my tasks, x + N (`L.N[INC.X MOV.A.X]`), "A = N if x is 0" (`Z.X[MOV.A.N]`),
"A = N if x > 0" (`L.X[MOV.A.N]`), the zero test `Z.X[INC.A]`, N ± 1, x ± 1, x + 2, and the bulk of the N ≤ 4
shortcut, the quotient guess x² − (x − 1)·N (both MDL runs); with them, `L.N[MOV.N.X DEC.X]` and the unused second
reductions `L.N[T.csub MOV.X.A]` (N links) and `T.sq L.A[MOV.X.A T.csub]` (x² links).

### 6. Which inputs were necessary

| | my practice tasks | the target only, growing sizes | + dreams |
|---|---|---|---|
| digit instructions | found (library study) | no (stuck at 2 digits; MDL and BPE) | no (stuck at 2 digits; MDL and BPE) |
| bare instructions | found with routines (MDL and BPE, level 1); not without (MDL and BPE) | no (stuck at N = 5; MDL and BPE) | no (stuck at N = 5; MDL and BPE) |

- **Practice tasks: necessary.** No arm without them reached x² mod N, on either machine, with either merge rule.
  With them, both machines got there once solved tasks became routines (the digit machine also needed the bigger
  budget and cost-3 loop bodies). The tasks are what make the pieces worth finding: each is a problem whose shortest
  solution is a piece.
- **Instruction set: not necessary at this size, but it fixes the algorithm.** The bare machine has no places, so
  what it invents counts in unary. That is general (right for every N tried) and useless at H1's size: the steps
  grow with the values, not with their digits.
- **Merge rule: routines necessary, the merge criterion not.** Routines turn every solved task's whole program into
  one token. Merges never made the link one token (in neither run did its 6-token program, used by one task, pay for
  itself as a merge), and the routine is what put the loop of links within a 2-token body. BPE against MDL never
  changed whether x² mod N was found; BPE made the searches slower and, in D, lost the N ≤ 4 shortcut (§7).
- **One more input, the search's own shape.** Loop bodies of at most 2 tokens (the library study's choice) keep the
  search tractable but put the bare machine's shortest general program (loop bodies 6 and 5) out of reach, so long
  bodies can only arrive as library tokens. That is fine when practice tasks supply the tokens, fatal when nothing
  does.

### 7. BPE vs MDL

Without curated tasks the two differ in how much junk they add, not in what they find. Both found the same targets
in C. In D, BPE merged 12 pairs every level (pairs of pairs of dream fragments: `L.X[INC.N] MOV.A.N INC.A ...`),
so the vocabulary grew 35 → 47 → 59 → 71 → 83 → 95 tokens; the same budget then reached 5 tokens instead of 6, the
N ≤ 4 shortcut was lost, and the run took 3.75 search-hours against MDL's 0.95. MDL merged 5, 4, 6 and 2 runs, each
paying for itself, then nothing. On the digit machine both stalled at 2 digits; BPE's 36 merges against MDL's 12
cost it 2.5× the states.

With curated tasks (B-bare) both worked with routines (MDL 0.51 search-hours, BPE 0.65) and both failed without them
(MDL 0.81, BPE 2.23). In the library study, with curated digit tasks, BPE and MDL both reached the reduction at 3M
states (half-price MDL token for token the hand-written loop).

### 8. One size up

The request: if D works at the smallest size, push it one size up. D did not work at N ≤ 7 (it stopped at N = 5), so
there was nothing to push; the next size's configs are in `loop.py` (`S2`: N up to 26, base 3 with three digits;
`A_s2_big`) but were not run. B-bare did work, and its x² mod N is already general (right up to N = 80), so I did
not rerun it at a bigger size.

### 9. Files and how to rerun

In `cpu_first/library/inventing/` (nothing outside it changed except `library/merge.py`'s `mdl` /
`description_length`, which now take the base instruction count, `n_prims` and `n_kinds`, defaulting to the digit
machine's):
- `brm.py`: the bare machine. `target.py`: the sizes, the target's data, the bare curated tasks.
- `bfs.py`: the search on the bare machine. `jobs.py`: one bare search job (digit jobs go to `digit_jobs.py`).
- `digit_jobs.py`: digit target and dream jobs over the library study's machine.
- `loop.py`: the arms (`CONFIGS`), dreams, compression, the loops. `modal_inventing.py`: Modal (one driver per arm,
  searches fanned out). `summarize.py`: a record, level by level.
- `records/`: `<config>_r1.jsonl` (bare A, C, D), `_r2` / `_r3` / `_r4` (cross arms), `calls_r*.json` (Modal call
  ids), `*_local.jsonl` (the local checks: `C_tiny`, `Cdig_tiny`, `Ddig_tiny`, `A_oracle`, `C_oracle`).

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m library.inventing.loop --config A_oracle
```

```bash
cd cpu_first && LIB_MAX=4 ~/github/sotaku/venv/bin/python -m modal run --detach library/inventing/modal_inventing.py --configs C_mdl,D_mdl --tag rerun
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m library.inventing.summarize library/inventing/records/D_mdl_r1.jsonl
```

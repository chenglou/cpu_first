# Carry: what bar 3's learner needs to invent add-with-carry and multiply-accumulate

*2026-10-24/25. Code in `cpu_first/bar3/bridge/` (`dwiden.py`, `dsubgoal.py`; default-off flags in
`dlearner.py`, `djobs.py`, `dsearch.py` and `modal_bridge.py`; checks in `checks/`), records and logs in
`records/carry/` and, for §12 and §13, `records/subgoal/`. Nothing submitted; the certified variant is unchanged.*

## In plain words

<!-- PLAIN -->
**The question.** BRIDGE.md left add-with-carry and multiply-accumulate uninvented after round 0; round 1 was cut
short by Modal's spend limit. It traced the block to one piece the learner poses itself: the target restricted to the
rows where R < C, "a carry of 1 comes in and nothing is added". With the digit link (A mod N for A < 2N, which the
learner learns in round 0), that piece's shortest programs are 4 instructions, and every one of them clears the carry
instead of counting with it. On the piece's own rows both kinds are right. Inside add-with-carry, where C holds how
many additions are still to come, only the counting one works. This study changed what the learner poses so that it
can tell the two apart, and finished the lost round 1 under the unchanged rules as the control.

**The rule: widen the pins.** A problem's rows can hold an input register at one value on every row; they *pin* it.
The piece's rows hold R at 0 and C at 1. Beside every problem that pins a register, the learner now also poses its
*widening*: the same rows, and each again with the pinned registers shifted up by 1 or 2, wanting the same answers;
a shifted register that is also an output is shifted by as much. What a pinned register holds beyond its pin is not
the problem's to use: it must not change the answer, and an output passes it through. The rule is about every
problem and knows nothing of add, carries or C. It adds 21 to 48 problems to a round-0 pool of 365 to 577.

Example: the widened piece wants (a, 0, 3) → ((a + 1) mod B, 2 + [A wrapped]). The clearing piece
`INC.A T.mod L.A[DEC.C]` counts C down once per unit of the new digit, so it leaves 3 − A there (stopping at 0); the
counting piece `INC.A T.mod Z.A[INC.C] DEC.C` leaves 2 + [A wrapped].

**What happened.**
- **Add-with-carry is invented without the rule.** Finished under the unchanged rules, the pairs signal invents it
  in its lost round 1: `T.lawadd(A,R)a MOV.R.C M1.3` adds r and the carry, compares the sum with the base and
  subtracts it once. It never counts with C: a + r + c stays below 2B, so one subtraction is enough. Its pieces are
  the learner's own round-1 practice: the target restricted to the rows where no carry comes in, and a remainder of
  its own near miss. Compiled to a table it is exactly the variant's ADC cell at every base from 2 to 16 (200 of 200
  rows at base 10). Practise-everything invents it in its lost round 1 too, by the same route: `INC.A T.rem126`
  adds the carry and r into A, compares with the base and subtracts once, and leaves N as it was. Hardest first does
  not invent it in 2 rounds. BRIDGE.md §3 was right about the counting route and wrong to treat it as the only one.
- **With the rule, the pairs signal invents add the same way**, in round 1, one instruction shorter
  (`T.wid4 M1.4`): a widened problem it practised in round 0, "add the carry into A and empty C", replaces the first
  two steps. Also exactly the ADC table.
- **Multiply-accumulate is invented once the learner aims a deeper search at its piece (§12).** Its sum
  a + r·h + c reaches B² − 1, so one subtraction is not enough and the carry has to count. Neither the control's
  lost round 1 nor the widening alone finds it. With the second study's additions, hardest first invents it in round
  1: `L.H[L.R[INC.C]] L.C[T.wid7]`, "add r·h to C, then C times add one to A with the carry counting", with
  `T.wid7` the clean counting piece. Compiled to a table it is exactly the variant's MUL cell, 10,000 of 10,000 rows
  at base 10 and every row at every base up to 16. The pairs signal does not invent it in 2 rounds: it never
  practises the mod link, so the clean piece stays past even the deeper probe.
- **The rule does what it is for, but with it alone the learners never use the counting piece.** Posed widened, the
  piece's cheapest answer with the link is the counting one, `INC.A T.mod Z.A[INC.C] DEC.C`, found by the learner's own
  1M-state probe in 29 s. With it the target searches find add (`L.R[INC.C] L.C[T.cnt]`) and multiply-accumulate
  (`L.H[L.R[INC.C]] L.C[T.cnt]`), right at every base up to 16 and exactly stage 3's tables. In the learners' own runs
  it never gets that far. Hardest first's library offers only pieces that count with C but consume N (a merge from its
  absdiff practice), and those break inside the loop. The pairs signal's library cannot reach the piece in round 1, and
  the pairs signal invents add by reducing once anyway. In practise-everything's round 1 the probe finds the clean
  piece, but the same round's practice also yields a reduce-once add at cost 2, which the target search finds first.
- **Multiply-accumulate's own pool needs a second pose rule.** A single comparison cannot pick the piece's rows
  (R = 0, C = 1) out of its rows, where the carry counts to B − 1. Posing the conjunctions of two restrictions that
  pin more (78 for multiply-accumulate) gives it the piece. Every step of the path is then found by the learner's own
  searches given the step before: the piece, "add C into A" as `L.C[T.cnt]`, and the target as
  `L.H[L.R[INC.C]] T.addC` (4 instructions, general). It fails at the first step, for the same reason as add's piece
  under hardest first: the cheapest program counts with C but consumes N, and the clean one is one instruction past
  the probe in multiply-accumulate's six-register machine. The first study checked this step by step; §12 runs it
  and gets past that step.
- **Aiming a deeper search (§12)**, the subgoal study's idea carried over. A score read off the rows alone asks
  whether the target factors through a *loop* of a candidate: load a count into a register the candidate pins, then
  repeat the candidate's step that many times. A widened candidate has a step beyond its rows, by its own
  pass-through rule. In each arithmetic target's round-0 pool exactly one piece has such a loop; for
  multiply-accumulate it is the widened conjunction, "C := R·H + C, then C times the step". That piece, the
  *subgoal*, gets one deeper probe (10M states, one level past the 1M probe). In it, a program that changes an input
  it does not output wins only when no clean one is within one level. The subgoal also wins ties in the choosing
  rule, and once it is practised the target search gets the deeper budget too. None of it names multiply-accumulate,
  carries or C.
- **Montgomery's quotient digit is a one-place cell (§13).** In base B, with N coprime to B, the digit q with
  t + q·N ≡ 0 (mod B) is a function of t's lowest digit and N's last digit at every base from 2 to 16: a table of 40
  rows at base 10. Long division's quotient digit is not a function of the leading digits it reads. Run from that
  table and the invented multiply-accumulate, subtract and select alone, Montgomery's reduction gives t·B⁻ᵏ mod N on
  300 of 300 random cases at every base. So a reduction with no estimated quotient digit needs one new one-place table
  and the cells the learner now invents. Nothing was built or certified from it.
- **Subtract and select are still invented as before**: in round 0 under the pairs signal and practise-everything,
  select also under hardest first; hardest first still does not invent subtract in 2 rounds. With the §12 additions
  the same holds, except that hardest first now invents subtract in round 1. Add is still invented in round 1 under
  the pairs signal, and still not by hardest first.

**What it says about bar 2.** "Add-with-carry is not invented" in BRIDGE.md was the lost round, not the learner:
unchanged, it invents add in round 1, by its own route. Multiply-accumulate needs the counting behaviour. The
widening pins that behaviour down, and a search aimed by a score read off the rows finds it within reach. The
pieces needed, in order:
- a pose rule that pins the piece in multiply-accumulate's own rows (the conjunctions);
- one that says what the piece must do beyond them (the widening);
- a score that names it before it is solved (the loop view);
- one search a level deeper there, where the clean program wins over the cheaper one that consumes N;
- the same depth for the target once the piece is a routine.

So the variant's ADC and MUL cells are now both invented instructions, as are its subtract and its select at the
pointer. Its instruction list does not change, and its certification stands. What remains designed is BRIDGE.md's
list: the law book, the frame, and which slices become cells.

**Cost.** First study: about $28 on Modal (CPU containers), against its $40 soft budget; 225 search-hours counted as
BRIDGE.md counts them, a job the cache answered counted at what it cost when first run. Second study (§12):
about $13.50 against its $25 soft budget; 140 search-hours counted the same way.
<!-- /PLAIN -->

## Details

### 1. The block, measured

The piece is the target add restricted to the rows where R < C: (a, 0, 1) → ((a + 1) mod B, [a + 1 = B]), 27 rows
at bases 2–7 and 54 at bases 2–10. It pins R at 0 and C at 1. With the digit link as a routine (`T.mod` =
`MOV.T.N L.A[DEC.T] Z.T[L.N[DEC.A]]`, the learner's own round-0 program for law/mod(A,N)/a+qN):

| program | the piece's rows | its widening (bases 2–10) | inside add (bases 2–16) | inside multiply-accumulate |
|---|---|---|---|---|
| `INC.A T.mod L.A[DEC.C]`, clearing (C counted down A times) | 54 / 54 | 270 / 486 | 1,552 / 2,968 | 4,006 / 20,488 |
| `INC.A T.mod Z.A[INC.C] DEC.C`, counting (C one less, plus one if A wrapped) | 54 / 54 | 486 / 486 | 2,968 / 2,968 | 20,488 / 20,488 |

(The compositions are `L.R[INC.C] L.C[piece]` and `L.H[L.R[INC.C]] L.C[piece]`.) The learner's 30,000-state probe
finds the piece at cost 4 in 3 seconds: six programs, the clearing one above and five like it (`L.T[DEC.C]`,
`L.A[CLR.C]`, `L.A[MOV.C.R]`, ...), none right on the widening. The learner keeps `INC.A T.mod L.A[DEC.C]`.

**Why not keep one program per behaviour instead** (the coordinator's first option). Those six programs show four
distinct behaviours on the widened rows, and the counting program costs one more. A cap that keeps the cheapest
program of each behaviour would need more than four places for this one problem, and the counting piece would then
compete with every behaviour of cost 5. The widening names the behaviour the composition needs instead of keeping
many. Measured, not built.

### 2. The rules

**Widen the pins** (`dwiden.widen`, flag `widen`). A candidate pins an input register (not the base N) when the
register holds one value on every row at every base. Its widening: every row, and each again with the pinned
registers shifted by 0, 1 or 2 (every combination but all 0), wanting the row's answer with each shifted output
register shifted by as much. Rows of the widened piece include (a, 0, 3) → ((a + 1) mod B, 2 + [a + 1 = B]) and
(a, 2, 1) → ((a + 1) mod B, [a + 1 = B]). It applies to every candidate that pins an input, whatever its source: 31
of subtract's 365 round-0 candidates, 47 of add's 457, 21 of multiply-accumulate's 577 and 48 of select's 501. Most
are restrictions; others are the law div(A,C) (C = 1 on those rows) and the slices succ(R), pred(R) and muld(R,H)
(A = 0).

**Pin by two restrictions** (`dwiden.conjunctions`, flag `conj`). The target's rows where two of its restrictions both
hold, kept when well posed, not already posed, and pinning a register neither part pins alone: 20 for subtract, 18
for add, 78 for multiply-accumulate and 15 for select. Multiply-accumulate needs it: its carry counts to B − 1, so no
single comparison picks out the rows R = 0, C = 1, and `0 < C and R + C < 2` does (100 rows). That is the add piece
inside multiply-accumulate's own rows. Widened, the clearing piece is right on 966 of its 1,800 rows and the counting
piece on all of them.

**Defaults.** Both flags are off unless a config sets them (`w<rule>_<target>` widens; `cw<rule>_<target>` does both).
With them off, round 0 of all twelve BRIDGE.md runs replays from the shared cache field for field
(`checks/reproduce_r0.py`, log in `records/carry/reproduce_r0.log`); the only difference anywhere is the `cached`
flag the cache adds to the jobs it answered. A job whose description had changed would have missed the cache, so the
replay also shows that the jobs are the ones the records were made with.

### 3. The tiny check

Local, seconds (`checks/carry_tiny.py`): the table of §1, the compositions on every row at bases 2–16 and against
stage 3's tables at base 10, multiply-accumulate's conjunction, and the pinned candidates of each pool. On Modal, the
learner's own searches (`checks/carry_jobs.py`, record `records/carry/carry_jobs.json`):

| search | budget | found | programs |
|---|---|---|---|
| the widened piece, library {T.mod} | 30,000 states | no (level 4 streamed) | – |
| the widened piece, library {T.mod} | 1M states | cost 5, 29 s | `INC.A T.mod Z.A[INC.C] DEC.C` and 7 more orders of the same steps; all count, all compose into add, all leave R and N as they were |
| add, library {T.mod, T.cnt} | 3M | cost 4, 20 s | `L.R[INC.C] L.C[T.cnt]`, right at every base up to 16 |
| multiply-accumulate, library {T.mod, T.cnt} | 3M | cost 5, 333 s | `L.H[L.R[INC.C]] L.C[T.cnt]`, right at every base up to 16 |

So the rule does what it is for: posed widened, the piece's cheapest answer is the counting one, and it composes into
both targets. Whether the learner gets there on its own depends on what its library holds when it meets the piece
(§5–§7).

### 4. The control: the lost round 1, finished

BRIDGE.md's configs, flags off, launched again under the tag k1. Round 0, and whatever of round 1 had finished before
Modal's spend limit, replay from the shared cache, so the comparison is like for like. Each run stops after round 1.

| run | round 1 picks | round 1's target search |
|---|---|---|
| add, pairs signal | lt(C,R); add where C < 1 (cost 5); add's digit alone (5); the remainder of its own near miss `L.R[INC.A]` where C < 1 (5) | **invented**: `T.lawadd(A,R)a MOV.R.C M1.3`, cost 3, right at every base up to 16; nine more programs of cost 3, all general |
| add, hardest first | gcd(A,N) on a + N (7); succ(A) (5); add where A < C (5); add's carry alone (5) | not found (level 5 streamed) |
| add, practise-everything | all 96 learnable problems (209 routines and merges after the round) | **invented**: `INC.A T.rem126`, cost 2, right at every base up to 16; fifteen more programs of cost 2, all general |
| multiply-accumulate, pairs signal | monus(A,R); lt(C,R); gcd(A,N) on a + N; monus(A,C) | not found (level 5 streamed) |
| multiply-accumulate, hardest first | gcd(A,N) on a + N (7); eq(R,C) (6); min(R,H) (5); max(R,H) (5) | not found (level 4 streamed) |
| multiply-accumulate, practise-everything | not run (§9) | – |

**How the pairs signal's add works.** In primitives: `L.R[INC.A] MOV.R.C MOV.C.N Z.C[DEC.A] L.R[INC.A] DEC.C INC.C
L.A[DEC.C] Z.C[INC.T] MOV.C.T L.C[MOV.T.A L.N[DEC.A] L.T[DEC.N] L.N[INC.A]]`. It adds r to A, copies the carry into R
and adds it too, so A = a + r + c; then C = N − A (stopping at 0), T = [A ≥ N], C = T; and if C is set, A becomes
|A − N| = A − N. Since a + r + c ≤ 2B − 1, one subtraction is enough. The last step consumes N, which does not matter
for one place (stage 3's cell is a table). Its routines are round 1's practice: `M1.3` is a merge from the programs
for "add where C < 1" (a + r and its carry, by the same compare-and-subtract) and for the remainder of the near miss
`L.R[INC.A]`, and `T.lawadd(A,R)a` = `L.R[INC.A]` is round 0's practice of the law A + R.

**How practise-everything's add works.** In primitives: `INC.A Z.C[DEC.A] L.R[INC.A] DEC.C MOV.T.N L.A[DEC.T]
Z.T[L.N[DEC.A]] Z.T[INC.C]`. It adds one and takes it back unless a carry came in (A = a + c), adds r, clears C,
then T = N − A (stopping at 0), and if T is 0 it subtracts N from A and sets C. `T.rem126` is its round-1 practice of
the remainder of its own near miss `INC.A`: the target's answers from A = a + 1. Its target search held level 2 and
found it there, after 2,156 s, most of it spent building the vocabulary of 209 routines.

BRIDGE.md §3 analysed one route to add, the counting one, and found its piece blocked; the learner's own round 1
takes another, under the pairs signal and under practise-everything alike.

### 5. The learner with the rule

The w-configs: the same choosing rules and budgets, every pinned candidate widened. Subtract and select run their
rounds as BRIDGE.md did (round 0, where they were invented; hardest-first subtract two rounds), add and
multiply-accumulate two rounds like the control.

| run | round 0 | round 1 |
|---|---|---|
| subtract, pairs signal | **invented** `L.R[INC.C] L.C[T.res2]` | – |
| subtract, practise-everything | **invented** `L.R[INC.C] L.C[T.res83]` | – |
| subtract, hardest first | the same picks as BRIDGE.md's | not invented, as before; it picked the widening of "subtract where A < C" |
| select, all three rules | **invented**: `L.C[MOV.A.R]`, `T.lawmul(R,C)a`, `M0.16` | – |
| add, hardest first | the control's picks | the control's picks; not invented |
| add, pairs signal | the widening of "add where A + R < 1" in place of the law A + R | **invented** `T.wid4 M1.4`, cost 2, right at every base up to 16: the control's reduce-once route, one instruction shorter (below) |
| multiply-accumulate, hardest first | the control's picks | the control's picks; not invented. A round 2 practised eq(A,C), gcd(A,R), gcd(C,H) and mod(N,A) on a + qN; its target search was cut when the run was stopped |
| multiply-accumulate, pairs signal | the widening of "multiply-accumulate where A + R < 1" in place of the law lt(A,R) on a + qN | stopped for budget during round 1 (§10) |

**What the rule did for the pairs signal's add.** In round 0 the pairs signal picked the widening of "add where
A + R < 1" (the rows (0, 0, c) → (c, 0) with A and R shifted: "add the carry into A and empty C"), whose program
`L.C[INC.A] DEC.C` became `T.wid4`. In round 1 it picked lt(C,R), add where C < 1, add's digit alone and gcd(A,N)
on a + N: the control's picks, with gcd in place of the remainder of a near miss built on the law A + R, which this
run never practised. The target search found `T.wid4 M1.4`, where `M1.4` is the control's `M1.3` (the same merge:
add R, compare with the base, subtract it once). So the rule gave the reduce-once route a cheaper first step. It did
not bring in the counting piece, which the pairs signal never practised (its library cannot reach it in round 1, §6).

Hardest first never picks a widened problem for add or multiply-accumulate: it takes the four costliest learnable
problems and breaks ties by pool position, and the widenings come last in the pool. With its library the widened add
piece has no composing program anyway (§6).

### 6. The widened piece in the learners' own libraries

Each rule's round-1 library (the one its round-0 record left), with the widened add piece searched as its probes
would, and each program found put into add on every row at bases 2–16 (`checks/carry_jobs.py`):

| library | 30,000-state probe | 1M-state probe | programs found | compose into add | leave R and N as they were |
|---|---|---|---|---|---|
| T.mod alone | no | cost 5 | 8 | 8 | 8 |
| hardest first's (8 entries) | no | cost 5 | 4, e.g. `INC.A M0.2 L.A[MOV.A.T] M0.1` | 0 | 0 |
| the pairs signal's (6) | no | no (level 5 streamed) | – | – | – |
| practise-everything's (105) | no | cost 4 (stopped at its 600 s limit) | 13, e.g. `INC.A T.lawmod(A,N)aqN MOV.R.C T.lawle(C,R)a` | 13 | 1: `INC.A T.lawmod(A,N)aqN T.res86 M0.14`, the one the learner keeps |

Hardest first's programs count with C, but they go through the merge `M0.2` = `MOV.T.A L.N[DEC.A] L.T[DEC.N]
L.N[INC.A]`, an |A − N| that leaves N at N − A. Inside add's loop the next step then wraps at the wrong base. MDL
extracted the merge from absdiff's practice, where consuming N was harmless. The learner's clean choice (bar 3's v2)
prefers programs that leave their inputs as they were, but only among the cheapest, and here every cheapest program
consumes N. So the widening fixes what a routine does with the extra in the registers it pins, and not what it does
to the inputs it does not output.

**Practise-everything's target search** (`checks/carry_reach.py`, record `records/carry/carry_reach.json`). With its
round-0 library plus the clean piece, the target search finds add as `L.R[INC.C] L.C[T.wid]` (cost 4, 1,694 s, level
4 streamed). With its round-1 library (184 entries, rebuilt from the cache by `checks/replay_cache.py`, the probes the
cache lacked counted as failed) plus the piece, it finds add at cost 2 without the piece: `T.val125 T.rem126`, where
`T.val125` = `L.C[INC.A] L.R[INC.A]` (a + c + r) and `T.rem126` sets C to [A ≥ N] and then reduces A mod N with the
link. Both are round-1 practice of a value and a remainder of the learner's own near misses: the same
reduce-once route as the pairs signal's.

### 7. Multiply-accumulate's own path

Its slice is a + r·h + c with c a digit, so the place's sum reaches B² − 1 and wraps many times. One subtraction of
the base, the route the learner found for add, cannot do it: the carry has to count. Step by step, with the learners'
libraries (`checks/carry_mac.py`, record `records/carry/carry_mac.json`):

| step | library | found |
|---|---|---|
| the widened conjunction `0<C & R+C<2` (the counting piece) | hardest first's round 1 | cost 4 at the 1M probe: `INC.A M0.3 Z.N[DEC.C]`. It counts with C but leaves N at 0 or 1 (`M0.3` = `MOV.T.A L.N[MOV.N.A DEC.A] Z.N[MOV.A.T]`), so it does not compose. The clean piece costs 5, past the probe: in multiply-accumulate's six-register machine level 4 alone holds 2.0M states |
| the same | the pairs signal's round 1 | nothing within 1M |
| its restriction R < 1, "add C into A" | hardest first's round 1 + the counting piece | `L.C[T.cnt]`, cost 2, 34 s |
| the target | hardest first's round 2 + the piece + `L.C[T.cnt]` | `L.H[L.R[INC.C]] T.addC`, cost 4, right at every base up to 16, 214 s |

So with both pose rules every step exists in multiply-accumulate's own pool, and the learner's own searches find
each one given the step before. The chain stops at the first step: the cheapest program for the counting piece
consumes N, and the clean one is one instruction past the probe.

### 8. End to end: the variant's cells

`checks/carry_tables.py` compiles a program into a table by running it on every row of the variant's cell (ADC: a, x
and a carry of 0 or 1; MUL: a, x, h and a carry digit). It compares the table with the variant's own `Cells` class
(its source taken from `submission_stage3_bridge.py`, md5 85ff74c5 asserted, run without the rest of the file) at
base 10 and at every base from 2 to 16, and runs the place loop of the base-10 table on 3,000 random whole numbers.

| program | from | base 10 | bases 2–16 | place loop |
|---|---|---|---|---|
| add `T.lawadd(A,R)a MOV.R.C M1.3` | the control, pairs signal, round 1 | 200 / 200 | every row equal | 3,000 / 3,000 |
| add `INC.A T.rem126` | the control, practise-everything, round 1 | 200 / 200 | every row equal | 3,000 / 3,000 |
| add `T.wid4 M1.4` | the rule, pairs signal, round 1 | 200 / 200 | every row equal | 3,000 / 3,000 |
| add `L.R[INC.C] L.C[T.cnt]` | the counting piece given (§3) | 200 / 200 | every row equal | 3,000 / 3,000 |
| multiply-accumulate `L.H[L.R[INC.C]] L.C[T.cnt]` | the counting piece given (§3) | 10,000 / 10,000 | every row equal | 3,000 / 3,000 |

Each invented add computes exactly the place function of the variant's ADC cell, whose table the variant learns from
data. So the variant's instruction list does not change, its records stand, and no new certification is needed.
Multiply-accumulate was not invented by any run, so the variant's MUL cell stays as BRIDGE.md describes it: in the
instruction list, its table learned from data, the piece that makes it a counting program written by hand.

### 9. What was not done, and dead ends

- **Practise-everything for multiply-accumulate**, control and rule. None of its round 1 is in the cache. With 134
  routines its probes run to their time limits, so the round costs about 90 search-hours (about $18), half the
  study's budget for one run. Its target search would be unlikely to reach a 4- or 5-instruction composition: with
  practise-everything's round-1 library for add, level 2 already holds 26,000 to 34,000 states (§4, §6), so level 3
  would be streamed and level 4 out of reach (an estimate; not run).
- **Practise-everything for add with the rule.** Its round 1 would be the control's plus the widened problems, and
  the control already invents add there at cost 2 (§4), before any counting route.
- **The conjunction rule in learner runs.** Built and checked step by step (§7), not run: hardest first breaks ties
  by pool position, so it would reach the cost-5 piece only after every other cost-5 candidate; the pairs signal has
  no mod link before round 1, and its 30,000-state lookahead does not reach multiply-accumulate's composition.
- **A frame for the inputs a routine does not output.** The widening could also want every input that is not an
  output to come out as it went in. That rejects the N-consuming pieces, but in these libraries the clean piece is
  past the probe (cost 7 with hardest first's absdiff routines for add, cost 5 in multiply-accumulate's six-register
  machine), so on its own it would change no outcome here. Posed without the input its answer ignores (H),
  multiply-accumulate's conjunction would have add's five registers, where the 1M probe reaches cost 5.
- **Rounds 2 and later** for hardest first (both targets, control and rule; multiply-accumulate's round 2 practice
  under the rule ran before the stop) and for the pairs signal's multiply-accumulate with the rule: stopped for budget
  (§10).

### 10. How hard each was tried, and the cost

| run | target | rule | pose rule | rounds finished | invented | program | search-hours |
|---|---|---|---|---|---|---|---|
| lp_adc_k1 | add | hardest first | none | 2 | no | - | 11.3 |
| ct2_adc_k1 | add | pairs signal | none | 2 | round 1 | `T.lawadd(A,R)a MOV.R.C M1.3` | 22.8 |
| all_adc_k1 | add | practise-everything | none | 2 | round 1 | `INC.A T.rem126` | 53.9 |
| wlp_adc_k1 | add | hardest first | widen | 2 | no | - | 13.0 |
| wct2_adc_k1 | add | pairs signal | widen | 2 | round 1 | `T.wid4 M1.4` | 23.2 |
| lp_mul_k1 | multiply-accumulate | hardest first | none | 2 | no | - | 14.4 |
| ct2_mul_k1 | multiply-accumulate | pairs signal | none | 2 | no | - | 25.9 |
| wlp_mul_k1 | multiply-accumulate | hardest first | widen | 2 (a third practised) | no | - | 29.6 |
| wct2_mul_k1 | multiply-accumulate | pairs signal | widen | 1 | no | - | 12.5 |
| wlp_sbb_k1 | subtract | hardest first | widen | 2 | no (as before) | - | 6.3 |
| wct2_sbb_k1 | subtract | pairs signal | widen | 1 | round 0 | `L.R[INC.C] L.C[T.res2]` | 3.8 |
| wall_sbb_k1 | subtract | practise-everything | widen | 1 | round 0 | `L.R[INC.C] L.C[T.res83]` | 2.9 |
| wlp_sel_k1 | select | hardest first | widen | 1 | round 0 | `L.C[MOV.A.R]` | 1.0 |
| wct2_sel_k1 | select | pairs signal | widen | 1 | round 0 | `T.lawmul(R,C)a` | 1.6 |
| wall_sel_k1 | select | practise-everything | widen | 1 | round 0 | `M0.16` | 1.1 |

(`checks/carry_report.py` prints this table from the records.)

- **Control** (flags off): add and multiply-accumulate under hardest first and the pairs signal, rounds 0–1, and
  practise-everything's add, rounds 0–1. Hardest first's round 2 was started for both targets and stopped for budget.
- **The rule** (widen): subtract and select under all three rules, as BRIDGE.md ran them; add and multiply-accumulate
  under hardest first (rounds 0–1, multiply-accumulate's round 2 practised) and the pairs signal (add rounds 0–1,
  multiply-accumulate round 0).
- **Checks on Modal**: the widened piece with the link alone and with each rule's round-1 library (8 searches), the
  targets given the piece (2), practise-everything's target search with its round-0 and round-1 libraries plus the
  piece (2), and multiply-accumulate's path with the learners' libraries (7).
- **Search-hours**: 223 over the fifteen runs and 2 for the 19 check searches. They count every job's seconds, a
  cached job at what it cost when first run, as BRIDGE.md does. Most of round 0 and a large part of round 1 were
  cached from BRIDGE.md's runs and are counted again in every run that asked, so these are not the hours bought
  here; the bill is.
- **Money**: $27.37 on the Modal app `bar3-bridge-carry` through 10:00 UTC on 2026-10-25 (Modal's billing report,
  complete hours only), plus about $0.40 for the last 45 minutes before the last apps stopped: about $28 against the
  $40 soft budget. It covers the learners' CPU containers, the drivers and the checks. At 06:30 UTC three apps were
  running about 72 search containers at about $15 an hour, so hardest first's second round and the pairs signal's
  multiply-accumulate with the rule were stopped to stay under the soft budget.

### 11. Files and how to rerun

New files, all in `cpu_first/bar3/bridge/`:

| file | what |
|---|---|
| `dwiden.py` | the rules: `pinned`, `widen`, `widenings`, `conjunctions` |
| `dlearner.py` (edited) | flags `widen`, `conj`, `lib0`/`corpus0`/`prefix`, all off by default; configs `w<rule>_<target>` and `cw<rule>_<target>` |
| `modal_bridge.py` (edited) | `--sub` (a records subfolder), `--rounds`, `--carry`, `BRIDGE_APP` (the app's name, for the bill); defaults unchanged |
| `checks/carry_tiny.py` | the tiny check, local, seconds |
| `checks/carry_jobs.py` | the widened piece with the link and with each rule's round-1 library, and the targets given the piece (Modal) |
| `checks/carry_reach.py` | practise-everything's target search with its round-0 and round-1 libraries plus the piece (Modal) |
| `checks/carry_mac.py` | multiply-accumulate's path step by step with the learners' libraries (Modal) |
| `checks/carry_tables.py` | invented programs against the variant's ADC and MUL tables at bases 2–16 |
| `checks/cache_r1.py`, `checks/replay_cache.py`, `checks/reproduce_r0.py` | what the cache holds of the lost round 1; a run replayed from the cache alone; round 0 of every run reproduced with the flags off |
| `checks/carry_watch.py`, `checks/carry_report.py` | the runs' rounds read straight from the Volume; the run table above |
| `records/carry/` | `<config>_k1.jsonl` (a round a line), `calls_k1.json` (the drivers' calls), `carry_jobs.json`, `carry_reach.json`, `carry_mac.json`, `reproduce_r0.log`, `carry_tables.log` |

```bash
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_tiny
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.reproduce_r0 lp_adc ct2_adc all_adc lp_mul ct2_mul all_mul lp_sbb ct2_sbb all_sbb lp_sel ct2_sel all_sel
cd cpu_first && BRIDGE_APP=bar3-bridge-carry PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_jobs
cd cpu_first && BRIDGE_APP=bar3-bridge-carry PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_reach <round-1 library json from checks/replay_cache.py>
cd cpu_first && BRIDGE_APP=bar3-bridge-carry PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_mac
cd cpu_first && BRIDGE_APP=bar3-bridge-carry BRIDGE_MAX=24 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs lp_adc,ct2_adc,lp_mul,ct2_mul --tag k1 --sub carry --rounds 2
cd cpu_first && BRIDGE_APP=bar3-bridge-carry BRIDGE_MAX=24 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs all_adc --tag k1 --sub carry --rounds 2
cd cpu_first && BRIDGE_APP=bar3-bridge-carry BRIDGE_MAX=24 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs wlp_adc,wct2_adc,wlp_mul,wct2_mul --tag k1 --sub carry --rounds 2
cd cpu_first && BRIDGE_APP=bar3-bridge-carry BRIDGE_MAX=8 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs wlp_sbb --tag k1 --sub carry --rounds 2
cd cpu_first && BRIDGE_APP=bar3-bridge-carry BRIDGE_MAX=16 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs wct2_sbb,wall_sbb,wlp_sel,wct2_sel,wall_sel --tag k1 --sub carry --rounds 1
cd cpu_first && BRIDGE_APP=bar3-bridge-carry ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_bridge.py --fetch --only carry/
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_report
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_tables bar3/bridge/records/carry/ct2_adc_k1.jsonl bar3/bridge/records/carry/all_adc_k1.jsonl bar3/bridge/records/carry/wct2_adc_k1.jsonl
```

Guard each launch first with `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`. The shared
search cache answers every job it has seen, so rerunning a finished run costs little more than its driver. The runs
were launched with `--rounds 3` for add and multiply-accumulate and stopped after round 1 (§10), and two of them
(`ct2_mul`, `wct2_adc`) were relaunched with `--rounds 2`; `--rounds 2` above reproduces what is reported here, from
the cache.

### 12. Aiming a deeper search at multiply-accumulate's piece

*2026-10-25, the second study. Code in `dsubgoal.py` (new) and behind default-off flags in `dlearner.py`, `djobs.py`
and `dsearch.py`; records in `records/subgoal/`; app `bar3-bridge-subgoal`.*

§7 left multiply-accumulate one step short: its own pool poses the piece once the conjunctions are on, and the
learner's searches find every later step, but the cheapest program for the piece consumes N and the clean one is one
instruction past the 1M-state probe. The subgoal study (bar3/scale/subgoal/SUBGOAL.md) met the same shape of block
for x³ and 10·x² and broke it by scoring the posed problems from the rows alone and aiming one deeper probe at the
problem the score singles out. This section carries that over.

**The score: does the target factor through a loop of the candidate?** (`dsubgoal.py`). SUBGOAL's score sees
problems posed on every one of the target's rows, through a function or one law of their answer. The digit pieces are
mostly not posed on the target's rows: they are restrictions (some of its rows) and widenings (rows it never shows),
and what puts them together is the counting machine's other composition, the loop. So the score asks a loop question,
from the rows alone:
- a widened candidate's answers are known beyond its rows by its own rule (a pinned register holding k more gives the
  same answers, a pinned output k more), and an input its answers never depend on is ignored. That gives it a *step*
  on any machine state, the registers it does not output left as they were;
- the target factors through a loop of the candidate when, on every one of the target's rows, loading a count into a
  register the candidate pins and applying the step that many times (`L.r[...]`, r read once) gives the target's
  answers. The count is one a counting machine's loops build: an input or the base, a sum or product of two, or of
  such a sum or product and a third (`L.x[INC.r]` adds x to r, `L.x[L.y[INC.r]]` adds x·y), within the machine's cap;
- the subgoal is the candidate with such a loop, when every candidate with one steps the same way on the states the
  loops visit (two posings of one piece); of those, the one posed on the most rows.

It knows nothing of a law, a carry or a register's role: only widened candidates have a step beyond their rows, and
only the registers they pin can count. On each target's round-0 pool (conjunctions and widening on; local, 0.2–0.6 s a
pool, `checks/subgoal_tiny.py`):

| target | widened candidates | with a loop | subgoal | its loop |
|---|---|---|---|---|
| subtract | 51 | 1 | the widening of "subtract where R < C" | C := R + C, then C times the step |
| add | 65 | 1 | the widening of "add where R < C" | C := R + C, then C times the step |
| multiply-accumulate | 99 | 1 | the widening of `0<C & R+C<2` | C := R·H + C, then C times the step (H ignored) |
| select | 59 | 8, one step ("take R") | the widening of "select where 0 < C" | C := C, then C times the step |

**The deeper probe, and the clean program winning** (`djobs.py` `clean_win`; `dsearch.py`, an accept that defers one
level). The subgoal gets one more search, 10M states and an hour (`PROBE_DEEP`: it stores the level the 1M-state probe
streams, and streams the next), when the usual probes leave it unsolved, or solved only by programs that change an
input they do not output. In that search a right program that changes such an input wins only when no right program
within one level of it leaves them alone: the search looks one level further and keeps the first right programs if
none turns up. It is a rule about any problem that gets the deeper probe. Example, local: mod on a + qN with a routine
that consumes N, `T.x` = `MOV.T.A L.N[DEC.A] L.T[DEC.N] L.N[INC.A]`, and a clean one, `T.c` = `MOV.T.N L.A[DEC.T]`. The
plain search keeps `T.x L.N[MOV.A.T]` (cost 3, N consumed); with clean-win it looks one level further, and the learner
keeps `T.c Z.T[L.N[DEC.A]]` (cost 4).

**The ties and the target search.** As in SUBGOAL, the subgoal comes first among equals in every ordering of the
choosing step: hardest first's costs, the pairs signal's anchors, pairs and singles. And once a subgoal has been
practised, the target search gets the same deeper budget, since the target now factors through a loop of a routine:
the deeper search is aimed at it in turn. All of this is behind the flag `subgoal` (configs `s<rule>_<target>`, with
the conjunctions and the widening). With every flag off, round 0 of all twelve BRIDGE.md runs still replays from the
cache field for field (`records/carry/reproduce_r0_subgoal.log`), and the local search of §1 gives the same levels and
programs as before.

**The tiny check on Modal** (`checks/subgoal_jobs.py`, `records/carry/subgoal_jobs.json`):

| search | library | found |
|---|---|---|
| the piece, the deeper probe with clean-win | hardest first's round 1 | cost 5, 324 s: the cost-4 `INC.A M0.3 Z.N[DEC.C]` first (it consumes N), then 135 programs at cost 5, of which the learner keeps the clean `INC.A T.lawmod(A,N)aqN Z.A[INC.C] DEC.C` |
| the same | the pairs signal's round 1 | nothing within 10M: that library has no mod link |
| multiply-accumulate, with the clean piece a routine | hardest first's round 2, 3M states | not found (level 4 streamed) |
| the same | 10M states | cost 5, `L.H[L.R[INC.C]] L.C[T.cnt]`, right at every base up to 16, 1,580 s |
| the loop's count, C := C + R·H, as a problem | hardest first's round 2 | cost 3, `L.H[L.R[INC.C]]` (not used by the rules) |

**The runs** (`records/subgoal/<config>_k1.jsonl`; multiply-accumulate 3 rounds, the regressions as CARRY.md ran them):

| run | rule | round 0 | round 1 | round 2 |
|---|---|---|---|---|
| multiply-accumulate | hardest first | the control's picks, mod among them; the subgoal's deeper probe finds nothing (no link yet) | the deeper probe finds the clean piece at cost 5; the tie-break makes it the third of four picks (after costs 7 and 6, first of eight at 5); the deeper target search finds `L.H[L.R[INC.C]] L.C[T.wid7]` (cost 5, general, 1,762 s): **invented** | – |
| multiply-accumulate | pairs signal | two widened restrictions among its picks, no mod link; the subgoal's deeper probe finds nothing | lt(C,R), a remainder restriction, gcd(A,N) on a + N and monus(A,R): no mod link again (it is learnable at cost 3, through a merge, and not picked); the deeper probe finds nothing within 10M; the target is not found | not run: with no mod link the clean piece stays past the deeper probe, so the run was stopped to save its cost (about $7) |
| subtract | pairs signal | **invented** `L.C[T.res2] L.R[M0.1]` | – | |
| subtract | practise-everything | **invented** `T.val90 T.res83 L.R[M0.9]` | – | |
| subtract | hardest first | not invented (as before) | **invented** `L.R[INC.C] L.C[M1.2]`: the subgoal (its widened piece, clean at cost 5) is picked on the tie-break; before, hardest first did not invent subtract in 2 rounds | |
| select | all three | **invented**: `L.C[MOV.A.R]`, `T.lawmul(R,C)a`, `M0.16` | – | |
| add | pairs signal | the same picks as with the widening alone | **invented** `T.wid4 M1.4`, as with the widening alone (§5) | |
| add | hardest first | not invented | not invented, as before: its piece is picked on the tie-break, but the deeper probe finds only pieces that consume N at cost 5 (with its absdiff routines the clean one costs 7), and those do not compose | |

Add under practise-everything was not rerun with the subgoal (its round 1 costs about 50 search-hours). Every
invented program is right at every base up to 16.

**What each part did, for multiply-accumulate under hardest first.** The run shows them in turn:
- without the conjunctions, its pool has no piece at all (§7);
- without the deeper probe, the 1M-state probe keeps the cost-4 piece that consumes N, which does not compose (§7);
- without clean-win, the deeper probe would stop at that same cost-4 piece;
- without the tie-break, hardest first takes the eight cost-5 problems in pool order, and the widenings come last;
- without the deeper target search, the 3M-state search stops at level 4, one short of the composition.

**The invented multiply-accumulate.** `L.H[L.R[INC.C]] L.C[T.wid7]`: add r·h to C; then, C times, add one to A,
wrapping at the base, with C one less, plus one if A wrapped. So C counts the units still to add and ends as the
carry. `T.wid7` = `INC.A T.lawmod(A,N)aqN Z.A[INC.C] DEC.C`, with `T.lawmod(A,N)aqN` = `MOV.T.N L.A[DEC.T]
Z.T[L.N[DEC.A]]` (A mod N for A < 2N), round 0's practice. `checks/carry_tables.py` against the variant's MUL cell:
10,000 of 10,000 rows at base 10, every row at every base from 2 to 16, and the place loop right on 3,000 of 3,000
random numbers (`records/subgoal/carry_tables_mul.log`). So the variant's instruction list does not change, its
records stand, and its certification stands.

**Cost.** $12.50 billed to the app `bar3-bridge-subgoal` through 16:00 UTC on 2026-10-25 (complete hours), plus about
$1 for the last 48 minutes before the last app stopped: about $13.50, against the $25 soft budget. Search-hours, counted
as above: 139.9 over the ten runs (the eight deeper probes 0.47 of them) and 0.8 for the six check searches. The pairs
signal's multiply-accumulate is the costliest run (62.5 counted): its round-1 pool was 1,946 problems, since its near
misses' remainders pin registers and each is widened.

**Files and how to rerun** (all in `cpu_first/bar3/bridge/`):

| file | what |
|---|---|
| `dsubgoal.py` (new) | the score: a widened candidate's `Step`, the counts, the loop check, `subgoal_of` |
| `djobs.py`, `dsearch.py` (edited) | `clean_win` in a search job, and the accept that defers one level; off by default |
| `dlearner.py` (edited) | the flag `subgoal` (score, deeper probe, ties, the deeper target search; `pick` built but not run), `PROBE_DEEP`, configs `s<rule>_<target>` and `sp<rule>_<target>` |
| `dwiden.py` (edited) | a widening keeps its pins (`pins`), for the step; not part of its key |
| `checks/subgoal_tiny.py`, `checks/subgoal_jobs.py` (new) | the score on each round-0 pool (local, seconds); the deeper probe, the targets and the count on Modal |
| `checks/carry_watch.py` (edited) | `--sub subgoal` reads this study's records |
| `records/subgoal/` | `<config>_k1.jsonl`, `calls_k1.json`, `carry_tables_mul.log` |
| `records/carry/` | `subgoal_jobs.json`, `reproduce_r0_subgoal.log` |

```bash
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.subgoal_tiny
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.reproduce_r0 lp_adc ct2_adc all_adc lp_mul ct2_mul all_mul lp_sbb ct2_sbb all_sbb lp_sel ct2_sel all_sel
cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.subgoal_jobs
cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal BRIDGE_MAX=24 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs slp_mul,sct2_mul --tag k1 --sub subgoal --rounds 3
cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal BRIDGE_MAX=16 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs slp_adc,sct2_adc,slp_sbb --tag k1 --sub subgoal --rounds 2
cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal BRIDGE_MAX=16 ~/github/sotaku/venv/bin/python -m modal run --detach bar3/bridge/modal_bridge.py --multi --configs sct2_sbb,sall_sbb,slp_sel,sct2_sel,sall_sel --tag k1 --sub subgoal --rounds 1
cd cpu_first && BRIDGE_APP=bar3-bridge-subgoal ~/github/sotaku/venv/bin/python -m modal run bar3/bridge/modal_bridge.py --fetch --only subgoal/
cd cpu_first && PYTHONPATH=. ~/github/sotaku/venv/bin/python -m bar3.bridge.checks.carry_tables bar3/bridge/records/subgoal/slp_mul_k1.jsonl
```

Guard each launch first with `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`.

### 13. Montgomery's quotient digit, a one-place cell (theory/LARGE_NUMBERS.md §2.6, §5.5 test 4)

*Local, seconds, no search, nothing built or certified from it. `checks/montgomery.py`, log
`records/subgoal/montgomery.log`.*

Montgomery's reduction of t by N in base B, with N coprime to B and k digits long, repeats one step k times:
1. read t's lowest digit t₀;
2. take the digit q with t + q·N ≡ 0 (mod B);
3. add q·N, which clears that digit;
4. shift t down a place.

The result is t·B⁻ᵏ mod N, or that plus N (one final subtraction).

**It is a function of one place.** Rows read off random reductions the way the book reads a slice off its law
(about 1,500 steps a base), put to the book's one-place test ("the same inputs never want two answers"):

| base | keyed by (t₀, N₀) | keyed by t₀ alone | q = −t₀·N₀⁻¹ mod B |
|---|---|---|---|
| 2 to 16, each | a function: 0 groups mixed (40 rows at base 10, 128 at base 16) | not a function from base 3 up (8 of 10 groups mixed at base 10) | on every row |

So q is a one-place table of t's lowest digit and N's last digit, the kind the book poses, with no state. It is not
a place-loop slice: it acts once a step, at the lowest place, and the pass that adds q·N does the place loop.

**It lifts.** The reduction run from cells alone, at every base from 2 to 16:
- q from the table the rows give;
- t + q·N from the invented multiply-accumulate (§12), place by place, its carry passed up, with H = q;
- the shift;
- the final subtraction from the invented subtract and select: keep t when t − N borrows, else take the difference.

It is right on 300 of 300 random reductions at every base, against t·B⁻ᵏ mod N.

**Long division's quotient digit is not.** Its first digit (t div N, with t < B·N), keyed by t's top two digits and
N's top digit (the places a division step reads first), is not a function of them: at base 10, 223 of 474 groups
are mixed, the digits in one group differing by up to 5 (up to 3 at base 7, 7 at base 16). That is the estimate
long division corrects (Knuth's algorithm D), and where stage 3's quotient-digit puzzle and its weak reduction check
sit (§5 of BRIDGE.md: repeated subtraction right only for small quotients).

**What it would take for stage 3's reduction to be built from invented one-place cells plus this one** (not built):
- *cells*: multiply-accumulate (one pass of t + q·N a step, H = q, as stage 3's MUL), subtract and select (the
  final conditional subtraction, as s3i's), all invented now; and one new one-place table, q(t₀, N₀), 40 rows at
  base 10. Its rows are read off the law like the other slices, so the learner can pose it as a target. Whether
  bar 3's learner invents it on the counting machine was not tried: it is "the count of N₀ additions mod B that
  bring t₀ to 0";
- *frame*: per step, read place 0 of t and of N, one multiply-accumulate pass with H = q, a shift down (stage 3's
  SHR); k steps (a loop by the operand width, which the frame has); then one conditional subtraction;
- *Montgomery form*: the squaring chain runs on x̄ = x·Bᵏ mod N, since x̄²·B⁻ᵏ = x²·Bᵏ mod N. Each round becomes a
  square, then k reduction steps. Entering needs B²ᵏ mod N once per modulus, a reduction outside the loop over T;
  leaving is one more reduction. Every quotient digit inside the loop over T is then a table lookup, with nothing
  estimated or corrected;
- *the requirement*: N coprime to the base. H1's moduli are products of primes 3 more than a multiple of 4, never of
  2 or 5, so in base 10 every N₀ is 1, 3, 7 or 9.

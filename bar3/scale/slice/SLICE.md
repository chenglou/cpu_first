# Composition round 4: the target's laws cut to one digit place

<!-- PLAIN -->
## In plain words

**The question.**
- Round 3 (bar3/scale/compose3/COMPOSE3.md) found that nothing of bar 3's poses the multiply-add row, a + b·c, the
  multiply's inner loop. Without the book's row the multiply never came.
- In the bridge study, bar 3's learner invented stage 3's digit cells by practising laws cut to one digit place, laws
  from a book.
- This round cuts the target's own laws instead: the subgoal score's law view reads x² mod N as mod(mul(X, X), N).
- Does a source that cuts those laws to one digit place bring the row, and then the round?

**The source** (`hooks.py`, `lawread.py`).
- **It reads the target's law:** the law view taken to depth 5, with constants from the data.
  - x² mod N reads as mod(mul(X,X),N).
  - x³ mod N reads as mod(mul(mod(mul(X,X),N),X),N): reduced first, since x³ overflows the registers.
  - 10·x² + 1 mod N reads as mod(add(mul(mod(mul(X,X),N),10),1),N).
- **For each operation it poses its cut at one digit place:**
  - the multiply at one digit of one operand, the rest carried: the row a + x·d (and x·d);
  - the reduction at one digit of the dividend, from the top: Horner's step (B·r + d) mod N;
  - the addition at one digit: x + d.
- **Each cut** is posed in every binding, answered by Python's integers (the law view's own arithmetic), and accepted
  at the core's verify widths. The source never poses the laws themselves.

**What came** (one B200 each, round 3's settings):

| arm | what | the round |
|---|---|---|
| `s4_x2` | round 3's harness (book_min less its reduction and its row, round 2's fixes) and the cuts | **yes, level 5** (965 s); right on both of H1's ladders, T = 1 to 64 (768/768 on all 14 rungs) |
| `replay_c3_r1`, `replay_c2_b2_rd` | the same harness with the cuts off; round 2's B2rd | their records again, apart from times: no round; B2rd's round at level 5 |
| `s4_x2_no_step` | s4_x2 without Horner's step | **yes, level 5** (920 s), through REST_DEEP as in B2rd; right on both ladders |
| `s4_x3` | s4_x2's setup on x³ mod N | no (10 levels): the cuts and the multiply came, no route |
| `s4_10x2p1` | s4_x2's setup on 10·x² + 1 mod N | no (the deadline at level 7): the same, and the addition from its own cut at level 0 |
| `s4_x2_less_mul` | the book less its multiply too | no (the deadline at level 8): no multiply |
| `s4_x2_lessons` | anchored pairs, SUBGOAL and REST_DEEP off | no (the deadline at level 4): the row never practised |

**How the round came.**
1. Level 0: the row, `LDH.Y CLK P[MUL.X INC]`, from the multiply's cut.
2. Level 1: Horner's step, `B[link]`, from the reduction's cut.
3. Level 4: the book's multiply over the row, `CLA W[row INC SHL.X]`.
4. Level 5: x² (posed by LAWNEAR), and the rest of the way from it, `LDA.N P[BRG.X step]`: 4 tokens over the step,
   found by the regular 1M probe.

So this route needs neither the book's row nor REST_DEEP. With the step off, REST_DEEP's 10M probe found the reduction
over the link instead (5 tokens), and the round came at level 5 again: either the step or REST_DEEP supplies the
reduction, while the row has only the cut.

**Generality.**
- On x³ and 10·x² + 1 the readings, the cuts, the row and the multiply all came. For 10·x² + 1, add's cut also gave
  the addition at level 0, before the book could.
- No round. Both need the reduction as a routine before their own rows show it. Here the reduction comes only as the
  rest of the way from x², which is x²'s alone.

**How far down the book.** Not past the row. Without the book's multiply nothing poses it: the multiply is the row's
loop over the multiplier's digits, not a cut.

**The consolidation's lessons.**
- **Anchored pairs never practised the row.** A 300k-state look at the target with the row alone shows nothing until
  the multiply exists, two levels later: the consolidation study's myopia again. And the lookaheads cost 500–1,000 s a
  level here.
- **The uniform probe:** this harness probes every root at 1M already. With the step, the special deeper probes
  (SUBGOAL's, REST_DEEP) never ran.

**What it means.**
- The row was buried two loops deep inside the multiply (round 3's words). Cutting the target's law at one multiplier
  digit exposes it, with no book. The reduction's step comes out the same way, and with it REST_DEEP is not needed.
- A cut is one step of a loop. The loops themselves, the multiply over the multiplier and the reduction over the
  dividend, still come from the book and from bar 3's rest of the way. That is where x³ and 10·x² + 1 stop, and where
  the next source would have to act.
- **Cost:** about $25 of the soft $30 (§9).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

PREREG.md (frozen 2026-10-26 15:03:57 PDT, read-only, sha256
`05cfd5e5c7da362adef412bd4f66bad2958c09780912f636dd0d67506b563451`, `logs/prereg_sha256.txt`), before any arm ran,
unchanged:

<!-- PREREG -->
# Composition round 4: the target's laws cut to one digit place, pre-registered

Written 2026-10-26, before any run of this study. SLICE.md §1 repeats this text unchanged and records the time and hash
it was frozen with.

## The questions

Round 3 (bar3/scale/compose3/COMPOSE3.md) found that nothing of bar 3's poses the multiply-add row a + b·c. With
book_min less its reduction and its row, the multiply (9 tokens over the primitives) never came. In the bridge study
(bar3/bridge/BRIDGE.md §1, CARRY.md), bar 3's learner invented stage 3's digit cells by practising laws cut to one
digit place, laws from a book. Here the laws come from the target: the subgoal score's law view reads x² mod N as
mod(mul(X, X), N).
1. **The row and the round.** In round 3's harness, does a source that cuts the target's laws to one digit place bring
   the row, and then the round?
2. **Generality, unchanged:** x³ mod N and 10·x² + 1 mod N, whose laws also hold mul.
3. **If it works,** how far down the book can it go, the multiply next?

## Theory, in brief

1. **A slice** (BRIDGE.md §1) maps what one place sees to what it writes and passes on. What it sees is every operand's
   digit there and what the places below pass up.
   - On this machine the slices of add, sub, lt and a + x·d are its cells (ADC, SBB, MUL): primitives.
   - What the climb must learn are their loops over the places: the comparison, the addition, the row (COMPOSE.md
     §2.1's table).
2. **The multiply has no place slice:** a digit of the full product needs every lower digit pair. Cut at one digit of
   one operand, it has one.
   - x·y = Σⱼ (x·Bʲ)·yⱼ, so the step that takes in the multiplier's next digit d is o + x′·d.
   - The accumulator o carries what the lower digits passed on (the partial product); x′ = x·Bʲ is the frame's shift.
   - That step is the row, a + b·c on (o, x, d): the multiply cut at one multiplier digit. With nothing carried it is
     x·d.
   - It is BRIDGE.md's state ("the law on the low parts") with one operand cut instead of all.
3. **The reduction has no place slice from the bottom** (red and div, BRIDGE.md §1). From the top it has one: Horner's
   step, (B·r + d) mod N, with r < N the remainder so far (COMPOSE.md §2.1: the reduction is Horner's automaton).
   That step is red on (Bx+d, n).
4. **add cut at one digit of one operand** is x + d.
5. **What the source poses.** For each operation of the target's law, its cuts: the row and x·d for mul, Horner's
   step for mod, x + d for add.
   - They are posed as problems on stage 3's registers, in every binding, at the core's widths, accepted at the
     verify widths.
   - Their answers come from Python's integers: the law view's own arithmetic, as BRIDGE.md's slices are read off
     their laws and bar 3's law source is computed by numpy.
   - The source poses neither the laws themselves (the multiply, the reduction) nor the target's subexpressions.
6. **The law reader.** The law view reads to depth 2 over X and N.
   - The cube needs depth 4 within the registers: x³ overflows them, so the reading reduces first,
     mod(mul(mod(mul(X,X),N),X),N).
   - 10·x² + 1 needs depth 5 with the data's constants (1 at x = 0, 11 at x = 1, and their difference 10):
     mod(add(mul(mod(mul(X,X),N),10),1),N).
   - The reader extends the law view to left-deep chains of depth ≤ 5 over X, N and those constants (bar 3's constants
     rule), within the registers, by observational equivalence.
7. **What should happen on x².**
   - The row costs 5 over the primitives (`LDH.Y CLK P[MUL.X INC]`), within the probes: book_min's row came at cost
     5 in round 1.
   - With it, the book's multiply is 5 over the row (`CLA W[T.row INC SHL.X]`).
   - With the multiply, x² is a value, and round 2's route takes over: LAWNEAR, SUBGOAL, the rest of the way, the
     reduction.
   - Once the link is a routine, Horner's step is 2 over it (`B[T.red.pair.AN]`). The rest of the way from x² is then
     4 over routines (`LDA.N P[BRG.X T.step]`): inside the regular 1M probe, where round 2 needed REST_DEEP's 10M.
8. **Why not the cube and 10·x² + 1.**
   - The cuts supply the row and the step. Both targets then need the reduction as a routine before their rows show
     it: the cube's round reduces x² before its second multiply, and 10·x² + 1's reduces x² before the shift.
   - The reduction is posed only as the rest of the way from x² (bar 3's rest source). For the cube that rest is no
     function of (x², N) (SUBGOAL.md §1). For 10·x² + 1 it is not the target's.
   - Neither the cuts nor the book (less its reduction) pose the whole reduction: it is the step's loop over the
     dividend's digits, as the multiply is the row's loop over the multiplier's.
   - So both should practise the cuts and the multiply, and get no round.
9. **Removing the multiply from the book.** Nothing then poses it.
   - The cuts are slices, and the multiply is the row's loop.
   - The restriction x² < N needs a two-digit multiply at the acceptance width (COMPOSE3.md).
   - The book's own constructions tie it: its mul's answers need its add (repeated addition), and its add needs red
     and lt.
   - So the source should reach no further down the book than the row and the reduction's step.

## Design (`make_copies4.py`, `hooks.py`, `lawread.py`)

- **The copies:** round 3's, from make_copies3.py's lists, paths moved to this folder, with SLICE's changes:
  - the pose phase takes its (pattern, laws) from `hooks.pose_laws`: the book's, then the target's cuts;
  - `law_target` answers a cut by `hooks.slice_target`;
  - the flag is in the run's config.

  With SLICE off the copies are round 3's.
- **SLICE=1** poses every cut of the target's operations. A comma list of cut names (row, xd, step, add) poses a
  subset, for the flags-off arms. A cut the book poses itself is left to the book.
- **The reader** runs once, at level 0, on the target's probes and acceptance rows (subgoal_scores's rows). Its reading
  is recorded.
- **The consolidation's lessons** (bar3/scale/consolidate/CONSOLIDATE.md), where the harness allows:
  - the harness has anchored pairs as a rule (ct2a2, the cheaper pairs signal with lookaheads);
  - it probes every root at 1M, the uniform probe, with its standard deep probe (5.8M, the 8 closest roots);
  - its special deeper probes are SUBGOAL's and REST_DEEP's 10M.

  The main arms keep round 3's harness (lpca, round 2's flags), so that the cuts are the only change. One arm, S2c,
  takes the lessons: ct2a2, with SUBGOAL and REST_DEEP off. TIEBREAK then has no subgoal to follow.

## Runs

Each on one B200, round 3's settings: TARGET small, ROUND_TOO, NP 8, deep probes over whole concepts, SECONDS 2,400,
LEVELS 10. Round 3's R1 is LAWS lt;red;add;mul and PATTERNS pair;x,o, bar 3's four sources, round 2's flags, REST_DEEP
and TIEFOLLOW.

**Phase 1, in this order, at most 2 B200s at once:**
- `s4_x2`: R1 + SLICE=1;
- `replay_c3_r1`: R1, SLICE off;
- `s4_x3` and `s4_10x2p1`: s4_x2 with TARGET_FN x3 and 10x2p1;
- `s4_x2_lessons`: s4_x2 with RULE ct2a2, SUBGOAL and REST_DEEP off;
- `replay_c2_b2_rd`: round 2's B2rd, SLICE off.

**Phase 2, only if s4_x2 gets its round, in this order:**
1. `s4_x2_no_step`: the step off (SLICE=row,xd,add);
2. `s4_x2_no_rest_deep`: REST_DEEP off;
3. `s4_x2_less_mul`: the book less the multiply too (LAWS lt;red;add, PATTERNS pair);
4. `s4_x2_no_subgoal`, `s4_x2_no_lawnear` and `s4_x2_no_spell`: SUBGOAL, LAWNEAR or SPELL off.

Each is s4_x2 with that one change. The row's cuts off is R1 itself, the replay.

**Any round** is checked on both of H1's ladders to T = 64 (bar3/scale/compose3/verify3.py).

**The tests before the arms:** locally under a memory cap:
- the cuts' answers against known programs (`slice_check.py`);
- the reader on the oracle's own rows;
- the copies with SLICE off against round 3's tiny records;
- a tiny run with SLICE on.

**Money:** soft $30, cap $40, `spend.py 4750` before each launch. One spend watcher, its total the cap less one hour of
the containers' cost: at most 2 B200s at $6.37 an hour, so stop all at $27. Phase 2 runs only while the watcher's
estimate leaves room for it.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the replays | their records again, apart from times: R1 no round; B2rd its round at level 5 | high |
| the readings | x²: mod(mul(X,X),N) (depth 2); x³: mod(mul(mod(mul(X,X),N),X),N) (4); 10·x² + 1: mod(add(mul(mod(mul(X,X),N),10),1),N) (5) | high |
| s4_x2: the row | practised by level 1 | medium-high |
| s4_x2: the multiply | `CLA W[T.row INC SHL.X]`, practised by level 4 | medium-high |
| s4_x2: Horner's step | practised the level after the link, or the next | medium |
| s4_x2: the round | by level 8, right on both ladders to T = 64 | medium |
| s4_x3 | the row and the multiply practised; no round | high (no round) |
| s4_10x2p1 | the row and the multiply practised, x + d posed; no round | high (no round) |
| s4_x2_lessons | the round | low-medium |
| phase 2: the step off | the round, through REST_DEEP as in B2rd | medium |
| phase 2: REST_DEEP off | the round: the step makes the rest of the way 4 tokens, inside the 1M probe | medium |
| phase 2: SUBGOAL off | the round | low-medium |
| phase 2: LAWNEAR off | no round: x² is not posed as a value | medium |
| phase 2: SPELL off | the round | low-medium |
| phase 2: the book less the multiply | no multiply, no round | high |
| the conclusion | the cuts replace the book's row, and its reduction's step, for x². They do not reach the cube or 10·x² + 1, whose rounds need the reduction as a routine first, nor the multiply, which is the row's loop and not a cut | medium |
<!-- /PREREG -->

### 2. Theory and design

#### 2.0 Theory, and what the runs say about it

- **The cuts.** A law cut at one digit place of one operand, the rest of the law carried as the state (what the lower
  digits passed on), is BRIDGE.md's slice with one operand cut instead of all. BRIDGE.md's per-place slices are this
  machine's cells, so posing them teaches nothing here. The one-operand cuts are the pieces above the cells:
  - the multiply at one multiplier digit is the row a + x·d (the multiply's recurrence over its multiplier's digits,
    LARGE_NUMBERS.md);
  - the reduction at one dividend digit, from the top, is Horner's step (B·r + d) mod N (Horner's rule; the
    reduction's automaton, COMPOSE.md §2.1);
  - add at one digit of one operand is x + d.
- **Where they come from.** Round 3 found the row "buried two loops deep inside a piece, the multiply, that the target's
  rows never show alone at the acceptance width" (COMPOSE3.md). The cut does not need the target's rows to show it: it
  needs the target's law to contain mul, which the law view reads off the rows.
- **What the run showed on x²:**
  - the row is the multiply's slice, learnable at level 0, 5 tokens over the primitives;
  - Horner's step is the reduction's slice, 4 tokens over the comparison, practised at level 1 with the link;
  - and each made the next piece cheap: the multiply 5 tokens over the row, the rest of the way 4 tokens over the
    step.
- **What it cannot do:** a cut is one step of a loop. The loops themselves (the multiply over the multiplier's digits,
  the reduction over the dividend's) are not cuts, and here they still came from the book (the multiply) and from bar
  3's rest of the way (the reduction).

#### 2.1 The law reader (`lawread.py`)

- **The law view's operations:** score.py's (round 2's `subgoal_c2.OPS2` and `OPS1`), in the law view's own order:
  add, monus, absdiff, mul, div, mod, min, max, lt, le, eq, gcd; then succ, pred.
- **The chains:**
  - depth 1 is an operation on two leaves;
  - each further depth applies an operation to the chain so far and a leaf, on either side, or succ or pred to the
    chain;
  - at most 5 operations.
- **The leaves:**
  - X and N;
  - the data's constants: for x = 0 and for x = 1 (3 rows or more), the smallest c ≥ 1 below every row's capacity
    with c mod N equal to the answer on those rows (bar3/scale/sources.py's constants), and the difference of the
    two when it is ≥ 1.
- **Pruning:** a chain's value is kept only when it is in [0, the row's capacity) on every row, and only the first
  chain with that value on the rows (observational equivalence).
- **The reading** is the first chain, by depth and then in that order, whose value is the answer on every row; its
  operations are the source's input.

#### 2.2 The cuts (`hooks.py`: CUTS, slice_target)

| operation of the law | its cut at one digit place | as a problem on stage 3's registers | its program, known |
|---|---|---|---|
| mul | one operand at one digit, the rest carried | acc.mul on (o, x, d): o + x·d, every binding | `LDH.Y CLK P[MUL.X INC]` (AXY) |
| mul | the same, nothing carried | mul on (x, d): x·d | `CLA LDH.Y CLK P[MUL.X INC]` (XY) |
| mod | the dividend at one digit, from the top | red on (Bx+d, n): (B·x + d) mod N, x < N | `B[T.red.pair.AN]` (AN) |
| add | one operand at one digit | add on (x, d): x + d | `LDA.X CLC P[ADC.Y INC]` (XY) |

- **The draws** are the book's (spcore's `Draw`): the data's rows' top parts at the pattern's width, another row's x′
  for o, the digits 0 to 9 cycling for d, and Bx + d built by moving x′'s digits up one place.
- **The answers** are Python's integers on those values.
- **Acceptance** is the book's: the same construction on 60 fresh rows a verify setting (widths 1 to 3).
- **Routines and records:** a cut's candidates are named as the book's would be (`acc.mul/o,x,d@AXY`), and its
  routines too (`T.acc.mul.o,x,d.AXY`). In this study's arms the book has none of these patterns, so every such
  candidate is a cut.

#### 2.3 The copies (`make_copies4.py`)

Round 3's five copies and its hooks, from make_copies3.py's lists, the module paths moved to this folder, with three
asserted replacements:
- in `spcore_s.py`, the pose phase's loop over the book's patterns takes `hooks.pose_laws` (the book's laws by pattern,
  then the cuts merged in by pattern);
- in `spcore_s.py`, `law_target` answers a cut by `hooks.slice_target`;
- in `core_s.py`, the run's config records SLICE.

`hooks.py` is round 3's with the slicing source appended. Apart from the paths and these lines, each copy is round 3's
(`logs/diff_*`).

### 3. The tests before the arms (local, under a memory cap)

| test | what it showed |
|---|---|
| `test_lawread.py` (`logs/test_lawread.txt`) | on rows built like the oracle's, the reader reads x² at depth 2 and x³ at depth 4 (reduced first: mod(mul(mod(mul(X,X),N),X),N)), and 10·x² + 1 not by depth 4; at depth 5 it reads mod(add(mul(mod(mul(X,X),N),10),1),N) in 29 s |
| `slice_check.py` (`logs/slice_check.txt`) | 12 of 12 against known programs: the row, x·d, x + d and Horner's step over the link pass their cuts' probes and acceptors at every verify width; the row without its carry cleared, x·d where the row is wanted, one conditional subtraction where the step is wanted, and the like, fail. On the oracle arm's own target rows (157 distinct), the readings are those of the pre-registration |
| the copies with SLICE off (`logs/tiny_off.txt`, `logs/off_c1bookmin.*`, `logs/on_b2.*`) | round 3's two tiny records again (book_min; B2 with bar 3's sources and round 2's flags), apart from times |
| the copies' diffs (`logs/diff_*.txt`) | against round 3's, after the path moves: the three slice lines, and hooks.py's 89 added lines |
| the tiny runs with SLICE on (`logs/tiny_slice.txt`; `logs/tiny_slice_x2.*`, `logs/smoke_slice_*`) | x², 3 levels at tiny budgets: the reading mod(mul(X,X),N); the cuts row, x·d and step posed (82 candidates on 68 roots at level 0, R1's book alone posing fewer); by level 2 the link and Horner's step practised, `T.red.Bx+d.AX = B[T.red.pair.AX]`. x³ and 10·x² + 1, one level each: their readings as pre-registered; 10·x² + 1's add cut posed, and practised at level 0 (`T.add.x,d.AX = CLC P[ADC.X INC]`) |

### 4. The runs (`modal_slice.py`; `records/*.json` and `.log`)

- **Phase 1:** the six arms launched together at 15:04 PDT, one app (`ap-eAgytteJx73N51JrAItShR`), at most 2 B200s at
  once, after `spend.py 4750` (LAUNCH OK). The pre-registration was frozen at 15:03:57 PDT, sha256
  `05cfd5e5c7da362adef412bd4f66bad2958c09780912f636dd0d67506b563451` (`logs/prereg_sha256.txt`).
- **The spend watcher:** `logs/watch_s4_1.txt`, stopping all at $27.
- **Phase 2:** launched at 16:36 PDT, when phase 1 had ended, so that at most 2 B200s ran at once: one app
  (`ap-ylmSJysPFNCBKVfHx41cX3`) with `s4_x2_no_step` and `s4_x2_less_mul`, after `spend.py 4750`. One watcher over
  both apps, stopping all at $27 (`logs/watch_s4_2.txt`).
- **Deviations from the pre-registration:**
  - `s4_x2_no_rest_deep` was not run: REST_DEEP never acted in s4_x2, so with it off the run would be the same.
  - `s4_x2_no_subgoal`, `s4_x2_no_lawnear` and `s4_x2_no_spell` were not run, for money (§9).

### 5. Outcomes (`records/*.json`; `summary.py`)

#### 5.1 s4_x2: the cuts bring the row, the step, the multiply and the round (level 5, 965 s)

| level | practised (lpca) | from |
|---|---|---|
| 0 | the row in four bindings (`LDH.Y CLK P[MUL.X INC]`, cost 5); the comparison in three; a value | the mul cut; the book |
| 1 | Horner's step in three bindings (`B[STA.S T.lt.pair.AX SEL.S]`, cost 4: the link written in, before it is a routine); the link in three; x·d (`CLA T.row`); a value | the mod cut; the book; the mul cut |
| 2 | the addition in three bindings; values | the book |
| 3 | values | bar 3's sources |
| 4 | the multiply, `CLA W[T.acc.mul.o,x,d.AXY INC SHL.X]` (cost 5 over the row); a value | the book (its answers by its own addition) |
| 5 | the rest of the way from x², `LDA.N P[BRG.X T.red.Bx+d.AN]` (cost 4); x² itself, `LDA.X STA.Y T.mul.x,o.XY`; a restriction; a value | bar 3's rest, on LAWNEAR's x² |

- **The round** came in level 5's target search: `T.val.134.tgt.XN STA.X T.target.rest.134.XN` (cost 3), x² and then
  the reduction by Horner's step.
- **The ladders:** 768/768 on all 14 rungs of both, T = 1 to 64 (`logs/ladder_s4_x2.txt`).
- **How the last pieces came:**
  - LAWNEAR found x² in the look's held states at level 5 (fit 3, cost 3), and the subgoal score singled it out
    (key (1, 0), mod(g, N)).
  - Its rest was solved by the regular 1M probe, at cost 4 over the step.
  - Neither the subgoal's deeper probe nor REST_DEEP ever ran: at level 5 both the subgoal's problem and its rest were
    already solved, and before level 5 there was no subgoal.
- **Against round 2's B2rd** (the book's row; its round also at level 5):
  - the row came at level 0 instead of 3: the cut's answers come from Python, the book's from its own addition;
  - the reduction came through the step at 4 tokens, not REST_DEEP's 10M probe of the same at 5;
  - so this route needs neither the book's row nor REST_DEEP.

#### 5.2 The replays: their records again

- `replay_c3_r1` (round 3's R1, SLICE off): the same record as round 3's `c3_r1`, apart from times (`same.py`;
  `logs/replays_same.txt`). It ran slower, 1,824 s against 1,325 s, inside every time limit: no round.
- `replay_c2_b2_rd` (round 2's B2rd, SLICE off): the same record as round 3's replay of it, its round at level 5,
  692 s.

#### 5.3 s4_x3: the cuts, the multiply, and no route (10 levels, 2,042 s)

- **Levels 0–4:** the same pieces as x², at the same levels: the row (0); the step, the link and x·d (1); the
  addition (2); the multiply (4).
- **Levels 6–9:** the row in four more bindings, then values built on it: x plus x times N's low digit
  (`LDA.X T.acc.mul.o,x,d.AXN`), reduced by the step (`T.red.Bx+d.AN`), and others on those.
- **Level 7:** the subgoal score singled one of them out by table alone (key (0, 2): two rows saved, no law). REST_DEEP
  probed its rest at 10M and found nothing.
- **LAWNEAR found no state.** The cube has no value one law of depth 1 from it within the look's reach.
- **The end:** the level cap. No small target, no round.

#### 5.4 s4_10x2p1: the cuts, the addition first, and no route (the deadline at level 7)

- **Level 0:** the row, the comparison, and **the addition from add's cut** (`CLC P[ADC.X INC]`, three bindings), two
  levels before the book could pose its own add (it needs red and lt as routines).
- **Level 1:** the step, the link, x·d.
- **Level 2:** the book's addition, as 2 tokens over the cut's.
- **Level 4:** the multiply.
- **Levels 1–6:** values built from the addition, the step and the multiply; no subgoal with a law, no LAWNEAR state.
- **The end:** its levels grew slow (level 6 took 926 s), and the 2,400 s deadline came at level 7's pose. No round.

#### 5.5 s4_x2_lessons: anchored pairs never practise the row (the deadline at level 4)

- With ct2a2 in place of lpca, and SUBGOAL and REST_DEEP off, the four levels practised the comparison (0), the link
  (1), the addition (2), and other bindings of these, with Horner's step in one binding (`red/Bx+d@NA`, level 3).
- **The row was learnable from level 0 and never picked.** Anchored pairs rank a routine by the gain of a 300k-state
  look at the target with it (and with pairs). The row gains nothing there: its worth shows only once the multiply
  over it exists, two levels later. That is the consolidation study's myopia (CONSOLIDATE.md §2.0) in this harness.
- **The lookaheads cost 500–1,000 s a level** here (against 100–200 s for lpca's whole level), so the deadline came at
  level 4.

#### 5.6 Phase 2

**s4_x2_no_step (the step off): the round, through REST_DEEP (level 5, 920 s).**
- **Levels 0–4:** the row, the comparison, the link, x·d, the addition and the multiply, at s4_x2's levels; no step.
- **Level 5:**
  - LAWNEAR found x² again and the subgoal score singled it out.
  - Its rest needed the reduction over the link, `LDA.N P[BRG.X B[T.red.pair.AN]]` (cost 5). The 1M probe missed it,
    and REST_DEEP's 10M probe found it, as in round 2's B2rd.
- **The round:** `T.val.115.tgt.XN STA.X T.target.rest.115.XN`, 768/768 on all 14 rungs of both ladders
  (`logs/ladder_s4_x2_no_step.txt`).
- **So the reduction has two suppliers:** Horner's step (4 tokens, the 1M probe), or REST_DEEP (5 tokens, 10M). The row
  has one, the mul cut: without it (R1) the multiply never comes.

**s4_x2_less_mul (the book less the multiply): no multiply, no round (the deadline at level 8).**
- **Levels 0–2:** the row, the comparison, the step, the link, x·d and the addition, as in s4_x2.
- **Levels 3–7:** only values, built from x·d, the step and the addition. No multiply was ever practised, x² was never
  a value, and LAWNEAR found nothing.
- **Nothing poses the multiply once the book's mul is gone.** The cuts are slices, and the multiply is the row's loop
  over the multiplier's digits.

**Not run:**
- `s4_x2_no_rest_deep`. REST_DEEP never acted in s4_x2: there the rest was solved by the 1M probe, and the subgoal's
  own problem too. With it off, the run would be s4_x2's.
- `s4_x2_no_subgoal`, `s4_x2_no_lawnear` and `s4_x2_no_spell`, for money. After the two arms above the spend stood at
  about $25 against the watcher's $27, and each arm costs about $2.
- So round 2's other three fixes (SUBGOAL, LAWNEAR, SPELL) are untested here with the cuts. In s4_x2, LAWNEAR posed x²
  and the subgoal score singled it out.

### 6. Predictions against outcomes

| what | prediction | confidence | outcome |
|---|---|---|---|
| the replays | their records again, apart from times: R1 no round; B2rd its round at level 5 | high | **right**: both, B2rd's round at level 5 |
| the readings | x²: mod(mul(X,X),N) (2); x³: mod(mul(mod(mul(X,X),N),X),N) (4); 10·x² + 1: mod(add(mul(mod(mul(X,X),N),10),1),N) (5) | high | **right**: all three, in the harness |
| s4_x2: the row | practised by level 1 | medium-high | **right**: level 0 |
| s4_x2: the multiply | `CLA W[T.row INC SHL.X]`, by level 4 | medium-high | **right**: level 4 |
| s4_x2: Horner's step | the level after the link, or the next | medium | **wrong in its timing**: at the same level as the link (1), written over the comparison |
| s4_x2: the round | by level 8, right on both ladders to T = 64 | medium | **right**: level 5; 768/768 on all 14 rungs |
| s4_x3 | the row and the multiply practised; no round | high (no round) | **right** |
| s4_10x2p1 | the row and the multiply practised, x + d posed; no round | high (no round) | **right**; x + d was practised too, at level 0, before the book's addition |
| s4_x2_lessons | the round | low-medium | **wrong**: the row was never practised; the deadline came at level 4 |
| phase 2: the step off | the round, through REST_DEEP as in B2rd | medium | **right**: level 5, REST_DEEP's 10M found `LDA.N P[BRG.X B[T.red.pair.AN]]`; 768/768 on all 14 rungs |
| phase 2: REST_DEEP off | the round | medium | not run: REST_DEEP never acted in s4_x2, whose round is right, so the run would be s4_x2's |
| phase 2: SUBGOAL off | the round | low-medium | not run (money) |
| phase 2: LAWNEAR off | no round | medium | not run (money) |
| phase 2: SPELL off | the round | low-medium | not run (money) |
| phase 2: the book less the multiply | no multiply, no round | high | **right**: 8 levels, no multiply |
| the conclusion | the cuts replace the book's row, and its reduction's step, for x²; they do not reach the cube or 10·x² + 1, nor the multiply | medium | **right** on each part; and with the step, REST_DEEP was not needed |

### 7. Dead ends, and how hard each was tried

| what | how hard | where it stopped |
|---|---|---|
| the cube with the cuts (`s4_x3`) | 10 levels, 2,042 s on a B200 | the cuts and the multiply came, but no route: LAWNEAR found no value one law from the cube, the one subgoal was a table shortcut whose rest REST_DEEP could not solve, and the reduction was never a routine |
| 10·x² + 1 with the cuts (`s4_10x2p1`) | 7 levels, to the 2,400 s deadline | the cuts, the addition from its own cut, the multiply; no LAWNEAR state, no subgoal; levels slowed to 926 s |
| the consolidation's lessons (`s4_x2_lessons`) | 4 levels, to the deadline | anchored pairs never picked the row (no gain in a 300k look); the lookaheads took 500–1,000 s a level |
| the book less its multiply (`s4_x2_less_mul`) | 8 levels, to the deadline | nothing poses the multiply: it is the row's loop, not a cut |

### 8. What is left

- **The loops, not just their steps.**
  - A cut is one step of a loop. The multiply (the row over the multiplier's digits) still came from the book, and the
    whole reduction (the step over the dividend's) from bar 3's rest of the way from x².
  - Both the cube and 10·x² + 1 need the reduction as a routine before their own rows show it. A source that poses a
    cut's loop as a law on whole numbers would give it to them, and would replace the book's multiply too.
- **For 10·x² + 1, its constants:** + 1 has no cheap program on this machine.
- **Anchored pairs here** need a look that sees past one level, or a prior that a null look cannot erase
  (CONSOLIDATE.md's lesson): the row is worth nothing to a 300k look until the multiply exists.
- **Not run:** SUBGOAL, LAWNEAR and SPELL off with the cuts (§5.6).
- **More operations' cuts:** div (the quotient's digit, on Bx+d), and the comparison's, when a target's law has them.

### 9. Cost (`cost.py`; `logs/cost.txt`, `logs/watch_s4_*.txt`)

- **Billed to 00:00 UTC: $23.33.**
  - Phase 1 $18.92: 22:00 UTC $11.89, 23:00 UTC $7.03.
  - Phase 2 $4.41: 23:00 UTC.
- **Not yet billed:** the hour from 00:00 UTC, the last 16 minutes of s4_x2_less_mul on one B200, about $1.7.
- **About $25 in all:** under the soft $30.
- **B200 time:**
  - phase 1's six runs 10,342 s (965, 1,824, 2,042, 2,417, 2,402 and 692);
  - phase 2's two runs 3,321 s (920 and 2,401);
  - about 3.8 B200-hours with startup, at round 2's $6.37 an hour.
- **The watcher** stopped all at $27 (the cap less one hour of two B200s, at most 2 at once by `max_containers`). It
  never stopped anything: its last estimate was $24.57 at 00:06 UTC. It was then killed (exit 144) while the last arm
  ran its final minutes; that arm ended at its own deadline.
- **Every app of this study is stopped.** Local work ran under 8–10 GB memory caps.

### 10. Files and how to rerun

**The study:**
- `PREREG.md`: read-only, its hash in §1.
- `lawread.py`: the law reader; `test_lawread.py`, its test on rows built like the oracle's.
- `make_copies4.py`: writes `spcore_s.py`, `oracle_s.py`, `core_s.py`, `climb_s.py` and `clean_s.py`, and `hooks.py`
  (round 3's, with the slicing source appended). Round 2's subgoal score is imported from its folder.
- `slice_check.py`: the cuts against known programs, and the reader on the oracle arm's own rows.
- `modal_slice.py`: the arms, one B200 each, at most 2 at once; app and Volume `bar3-slice`.
- `summary.py`: a run's climb in short.
- `cost.py`: the billing report's hours for this study's apps.

**Records and logs:**
- `records/<tag>.json` and `.log`: each arm's record and log.
- `logs/`:
  - the local tests: `test_lawread.txt`, `slice_check.txt`, `tiny_off.txt` (with `off_c1bookmin.*`, `on_b2.*`),
    `tiny_slice.txt` (with `tiny_slice_x2.*`, `smoke_slice_*`);
  - the copies' diffs: `diff_*.txt`;
  - the pre-registration's hash: `prereg_sha256.txt`;
  - the launches and watchers: `launch_phase*.txt`, `watch_s4_*.txt`;
  - the checks: `ladder_s4_x2.txt`, `replays_same.txt`;
  - the cost: `cost.txt`.

**To rerun** (from `cpu_first`):
- **The copies:** `~/github/sotaku/venv/bin/python -m bar3.scale.slice.make_copies4`
- **The local tests:** each under `bar3/scale/rowcheck/memcap.sh 8 ...`, with `H1_DATA` set to the data's folder and
  `PYTHONPATH=$PWD`:
  - `~/github/sotaku/venv/bin/python -m bar3.scale.slice.slice_check`
  - `~/github/sotaku/venv/bin/python -m bar3.scale.slice.test_lawread`
- **An arm:**
  1. `~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750`
  2. `~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/slice/modal_slice.py --arms s4_x2`
  3. `~/github/sotaku/venv/bin/python -m modal run bar3/scale/slice/modal_slice.py --mode fetch`
- **A round on the ladders:** `~/github/sotaku/venv/bin/python -m bar3.scale.compose3.verify3
  bar3/scale/slice/records/s4_x2.json "T.val.134.tgt.XN STA.X T.target.rest.134.XN"`
- **The records' summaries:** `~/github/sotaku/venv/bin/python -m bar3.scale.slice.summary
  bar3/scale/slice/records/*.json`
- **The replays' check:** `~/github/sotaku/venv/bin/python -m bar3.scale.compose.same
  bar3/scale/compose3/records/c3_r1.json bar3/scale/slice/records/replay_c3_r1.json`

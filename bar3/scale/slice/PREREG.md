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

# Composition round 2: REST_DEEP, two more arms, pre-registered before their launch

Written 2026-10-26 08:43 PDT. PREREG.md's arms were running: B1 and B1lp had ended (no round), B2 was at level 7 and
B2lp at level 6. These arms were not launched. COMPOSE2.md §1 repeats this text unchanged.

## Why

B2 (bar 3's full set with book_min less its reduction) took the route as far as its fourth link:
1. the multiply, from the book, at level 4;
2. at level 5, x² posed by LAWNEAR (`LDA.X STA.Y T.mul.x,o.XY`, right on 3 of 16 probes; the 12 by fit, on 5–6),
   singled out by the subgoal score (key (1, 0), mod(g,N)), and practised as `T.val.126.tgt.XN`;
3. the rest of the way from it posed, as `rest.126@XN`.

But it never solved that rest:
- The probe (1M states over the composition vocabulary) does not reach cost 5 with a library like this one. On
  round 1's A2c library, the rest's level 4 alone held 5.6M states.
- The deep probe's focus is the closest attempt's routines. Here those were none, or `T.lt.pair.AX`, never the link
  `T.red.pair.AN` the reduction calls.
- Later levels practised x² less multiples of N (`T.target.rst5.XN B[B[T.red.pair.AN]]`). Those are right on all 16
  practice probes and wrong at width 3.

Two checks on round 1's route setup (A2c's library; on a B200):
- `spell_full`: the composition vocabulary at 10M states with SPELL=16 accepts `CLA P[BRG.X B[T.red.pair.AN]]` at cost
  5, in 71 s.
- `rest_deep_test`: the hook below, end to end. The rest from x², posed with the subgoal x² and left open, gets
  `LDA.N P[BRG.X B[T.red.pair.AN]]` and `CLA P[BRG.X B[T.red.pair.AN]]`, both accepted.

## The change (`hooks.rest_deep`, flag REST_DEEP)

SUBGOAL's deeper probe goes to the subgoal when the probes leave it unsolved. A value subgoal is solved at once (its
own prefix computes it), so with REST_DEEP the deeper probe goes one step on instead:
- to the rest of the way from the subgoal's value, posed with it this level;
- when the probes have left that rest unsolved;
- at the same 10M states, over the composition vocabulary (every routine; the focus off).

With REST_DEEP off the hook returns at once, and the copy is the one the earlier arms ran. The diff is the flag's
lines (`logs/core_c2_main_arms.py.txt`, `logs/spcore_c2_main_arms.py.txt`).

## The arms

- **B2rd** (`c2_b2_rd`): B2 with REST_DEEP.
- **B2lp_rd** (`c2_b2_lp_rd`): B2lp with REST_DEEP.

Everything else as B2 and B2lp. If either gets the round:
- that arm with each flag off in turn, to the level after the round's (LEVELS);
- REST_DEEP, LAWNEAR, SUBGOAL, SPELL and CLEAN_VALUES first, as the flags on the route;
- ROWCHECK, TIEBREAK and RESTRICT_ROWS if the money allows.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| B2rd | levels 0–5 as B2's, up to REST_DEEP's probe at level 5 | high |
| | REST_DEEP finds a reduction of the rest from x², accepted at width 3 | medium-high |
| | it is practised | high, given the line above |
| | the round comes, by level 6 | medium |
| | if it comes, right on every rung of both ladders | high |
| B2lp_rd | the same route, later (lp reached the multiply later) | low-medium for the round |
| one flag off, if the round comes | REST_DEEP, LAWNEAR, SUBGOAL or SPELL off: no round | high |
| | CLEAN_VALUES, ROWCHECK, TIEBREAK or RESTRICT_ROWS off: the round still comes | medium |

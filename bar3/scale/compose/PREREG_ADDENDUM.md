# The composition test: one more arm, pre-registered before its launch

Written 2026-10-26 06:40 PDT. The five arms of PREREG.md (sha256 7b698b48…) were launched at 06:29 and were running;
this arm was not launched. COMPOSE.md §1 repeats this text unchanged.

## Why

The offline one-more-digit test (`digit_test.py`) ran on `cp_bar3`'s record as it stood at level 2. Every one of its
12 practised routines reads a register the target's inputs leave dirty, in every placement. Not one is a function of
x and N:
- the ported value source (integration/self_practice/oracle.py) poses whatever a near miss's prefix leaves in a
  register, whatever it read (the carry, A, S);
- its acceptor recomputes that value on fresh rows with the same prefix, which then always matches.

Bar 3's own machine starts clean (library/inventing/brm.py: A, T and U are 0). There every near miss's value is a
function of the inputs. So the five arms test the sources as ported. This arm tests them closer to bar 3's own.

## The change (`clean.py`, flag CLEAN_VALUES)

- A near miss's value is posed only when it is the same under a second dirty start of the target's root (the same x'
  and N'):
  - as a value problem;
  - and as the rest of the way from it.
- The projections and the restrictions are the target's own functions, unchanged.

With the flag off the driver copy runs as the five arms ran it (the diff against the launched copy,
`logs/core_compose_cp1.py.txt`, is the flag's lines only). A tiny run with the flag off gave the earlier tiny run's
records again.

## The arm

`cp_bar3_clean`: A1 (`cp_bar3`) with CLEAN_VALUES=1. Everything else is the same: one B200, RULE=lpca, NP 8, the deep
probes over whole concepts, LEVELS 10, SECONDS 2,400, the small target and the round's search each level.

## Predictions

| prediction | confidence |
|---|---|
| each level poses fewer value problems than `cp_bar3` at the same level | high |
| its practised routines are functions of x and N in the target's placement (the offline test finds them clean) | medium-high |
| none of the six pieces is practised | high |
| no small target, no round | high |
| the one-more-digit test: some of its routines are states, and none is a piece of the round | medium |

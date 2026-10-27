# A legal climb from scratch: addendum 1

Written 2026-10-26 at about 20:42 PDT, before the two arms it adds were launched. LEGALCLIMB.md §1 repeats it unchanged
and records the time and hash it was frozen with.

## What changed

- **The budget** (the coordinator, relaying the user, at about 20:38 PDT): soft $100, cap $150, from $40 and $60. The
  spend watcher's stop-all moved to $110 (the cap less one hour of six B200s), without stopping the running arms.
- **Two arms added**, in an app of their own, one B200 at a time (six at most, with the first app's five):
  - `g1_b2`: g1 with the full-size vocabulary's bodies up to 2 (FULL_BODY=2). The tiny test rs_b2 built 6,386 item
    classes in 46 s, against rs_b3's 285,819 in 519 s, and took the same route. If the climb gets its round this way,
    this is the configuration a training file can afford.
  - `g1_s1`: g1 with the climb's seed 1 (SEED=1: the practice roots' draws), to see whether the route depends on one
    draw.
- **The code they need** (hashes in `logs/addendum1_sha256.txt`):
  - `core_l.py`: one replacement more (make_copies5.py), SEED as the climb's seed. It is inert unless SEED is set.
  - `modal_lclimb.py`: the two arms; MAX read from LEGALCLIMB_MAX (1 for the added arms' app).

  The running arms run the code frozen with PREREG.md: their image was built when they were launched.

## Seen before this was written

- The tiny tests: rs_b2 and rs_b3 got the round at level 5, right on both ladders.
- The first arms' early levels: five running, two of them posing two candidates at level 5 (the look's value and its
  rest), one at level 6 posing 69. Nothing else.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| `g1_b2` | as g1: the round at level 5 or 6; the full-size vocabulary under 2 minutes; the run under 20 minutes | medium |
| `g1_s1` | as g1: the round at level 5 or 6 (the draws change which bindings are practised, not which concepts) | medium |
| both, given a round | right on both ladders, 768/768 on all 14 rungs | medium-high |

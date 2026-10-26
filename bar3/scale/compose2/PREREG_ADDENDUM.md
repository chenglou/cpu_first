# Composition round 2: the test of the guess, and SPELL, fixed before any arm

Written 2026-10-26 07:53 PDT, after the test of the guess and before any arm of this study. COMPOSE2.md §1 repeats
this text unchanged.

## What the test found (`spell_test.py` on a B200, 07:45 PDT; `records/spell_test.log`)

On round 1's route setup, the committed search over the primitives with T.red.pair.AN and T.lt.pair.AN, at 5.8M
states:
- The two hand-written reductions, `CLA P[BRG.X B[T.red.pair.AN]]` and `LDA.N P[BRG.X B[T.red.pair.AN]]`, end in the
  same machine state on the probes. Both reach the goal on the probes, and both are accepted at width 3.
- **That state is held at level 5**, one of its 22 goal states (level 4 had 1). It has 6 spellings. The one tested
  is the first, `P[BRG.X] B[P[T.red.pair.AN]]`. It fails acceptance. Both reductions are among the other five, and
  both pass.
- The acceptor was handed 23 programs in all, one a goal state. Neither reduction was among them.

So the answer is **(b)**:
- not (a): the state is first reached at cost 5;
- not (c): it is reached;
- not (d): 22 goal states, far under the 50,000 that get a spelling.

The guess holds.

## The fix (flag SPELL, in `spcore_c2.py` only)

With SPELL=k, `bfs_roots` takes up to k spellings of each goal state (`programs(..., limit=k)`) instead of one.
- Every spelling of every goal state is sorted as before, by top-level items and then primitives.
- They are tested in batches of 256 until a batch has one that passes.
- At most 50,000 programs a level.

The arms run SPELL=16. With SPELL=0 (unset) the copy tests one spelling a state, as the committed search does. The
round's own search (libcore.bfs, whose probes are the answers) is untouched.

## Predictions

| what | prediction | confidence |
|---|---|---|
| the test's setup on the copy with SPELL=16 (`spell_fix`) | a reduction accepted at level 5 | certain, given the test |
| the arms | as PREREG.md | |

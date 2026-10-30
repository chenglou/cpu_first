# Addendum 1: the whole runs' second search, pre-registered

Written 2026-10-29 while the corpus captures ran: before any gate was trained, any whole learner run, and any arm was
searched on a test problem. GATE.md records its hash.

## What changes from PREREG.md

PREREG.md measure 4 gives the gate's second search "as many states as the plain search spent". The gated engine keeps
every level it expands until its cap (the plain search keeps one level and streams the next), so its memory grows with
the states it is given: about 1 GB per million states at N ≤ 7. The plain search spends about 4.3M states on a 1M probe
it leaves unsolved over the primitives (level 6 streamed), and more on the 10M deep probe and the 3M target search.

So in the whole runs (`learnrun.py`) the gate's second search is capped at the plain search's own spend or 4M states,
whichever is smaller: about the same as the plain search on the 1M probes, less on the deep probe and the target
search. A gate that cuts the branching factor should still find more with fewer states; the cap makes the outcome test
conservative for the gate, and the offline arms (measure 3) give it the plain search's spend up to 5M.

Everything else is as PREREG.md says: the consolidation arm unchanged, the gate cross-fitted (a run on fold A's formulas
uses the gates trained on fold B), C0's gate on an empty library and C1's over one, at most 5 rounds, stopping when the
target comes, 20 runs: 12 of this arm's 13 wins (binary x mod N, on the digits machine, left out, as PREREG.md says),
its three misses (10·x², x⁴, 10·x² + 1) and the five held-out non-wins. "All 13 still come" in PREREG.md's table reads
as all 12 tested.

## The predictions stand

No prediction of PREREG.md changes.

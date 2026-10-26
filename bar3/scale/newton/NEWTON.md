# Two sources from Newton's differences: the running sum's step and the offset

<!-- PLAIN -->
## In plain words

**The question.** Two earlier tests each named a practice problem that no source poses:
- for the triangular numbers, x·(x + 1)/2 mod N, their step: the next answer from the last one and x, (Y + X + 1)
  mod N (HELDOUT.md §7);
- for 10·x² + 1, the piece 10X mod N: the rest of the way from x² mod N, (10X + 1) mod N, less its value at X = 0
  (SUBGOAL.md §8).

These are the first two moves of Newton's forward differences on a sequence: take away the constant term, and take
differences, whose inverse is the running sum. Each is now a source, behind a flag that is off by default, in a copy
of the learner frozen at 97f30b9. Nothing else changed: with both flags off, the copy replays the earlier runs exactly.
Both formulas were then rerun under both rules, with predictions written down first (`PREREG.md`).

**What each source poses.**
- *The running sum's step*, y(x + 1) from y(x) and x, in 7 placements of the two numbers in the registers. It is
  posed from round 0 on every formula, since with x as an input the rows are always a function.
- *The offset*: each rest of the way less its value at X = 0, mod N. On 10·x² + 1 it poses 10X mod N from round 2
  under hardest first, once x² mod N is a routine. It is not applied to the target itself. The target's own offset,
  10·x² mod N, is one law from the target, so the subgoal's score would rank it above x² mod N and take away the
  deeper probe that x² mod N needs (the tiny test, §3).

**What came.**

| formula | hardest first | anchored pairs |
|---|---|---|
| triangular numbers | no, 5 rounds (before: no) | **round 1** (before: no, 5 rounds) |
| 10·x² + 1 | no, 5 rounds (before: no) | no, 5 rounds (before: no) |

- **The triangular numbers, anchored pairs, round 1.** The program is `L.X[M1.4]`: x times, U ← U + 1, then
  A ← (A + U) mod N. That is the running sum written as a loop. Anchored pairs scores each learnable problem by how
  much its routine shortens the target. The step's routine let that search find the target at cost 3, a gain of
  66.5 bits against 20.8 for the next problem, so the step was picked in round 1. The program is right on all 7,260
  pairs x < N ≤ 120. Its cost grows as x²·N, so the standard machine for N = 120 runs out of steps on 1,262 of them.
- **The triangular numbers, hardest first: no.** Four placements of the step were learnable at cost 5 in rounds 1–3.
  Hardest first takes the four costliest problems, ties in pool order, and 12 to 18 problems at cost 5 stood ahead of
  them. In round 4 the costliest of the others cost 4, so a step at cost 5 would have been picked, but by then the
  library had grown. The 1M-state probe then ran out of states within level 4, so cost 5 was out of reach. Every round's
  picks and library are the earlier run's.
- **10·x² + 1, hardest first: no.** The offset posed 10X mod N in rounds 2–4, and the probe solved it at cost 5.
  But the learner's duplicate check called it a copy of 3X. That check runs each program on 24 random machine
  states. Both programs die on every one of them, so their signatures match, although they differ on the pairs
  (at N = 7, x = 3: 2 against 9). So the piece was never counted learnable, never picked. The run is the earlier one,
  round for round.
- **10·x² + 1, anchored pairs: no.** This rule never practises x² mod N, so the piece is never posed. Again the run
  is the earlier one, round for round.
- **The earlier wins still come.** With both sources on, the nine held-out runs that came still come, each in the
  same round with the same program. Their picks, libraries and target searches match the records exactly, and no new
  problem was picked.

**Against the predictions.** The triangular numbers under anchored pairs (round 1) were seen in a local replay
before the predictions were written, so that was no prediction. Of the four real ones, three held: 10·x² + 1 does not
come under either rule, and the earlier wins stand. The fourth failed: the triangular numbers did not come in round 4
under hardest first. The prediction for hardest first on 10·x² + 1 also got its mechanism wrong. It said the piece
would be learnable and picked in round 4; it was neither, because of the duplicate check.

**Cost.** No Modal: $0. The earlier runs' stores held nearly every search, so the four runs and the check ran here.
The four runs made 273 new searches, 0.79 search-hours, and took 48 minutes, two at a time. The replays' new searches,
all cheap, are not counted.

**What it says.**
- **The running sum's step works where the pick rule asks what a problem does for the target.** Anchored pairs asks that
  through its looks, and the step gets the triangular numbers in round 1. Hardest first asks only how costly a problem
  is. There the step loses to pool order, and then to its own probe's shrinking reach. The probe's state budget is
  fixed, and each new routine widens every level.
- **The offset poses its piece, and a check in the frozen learner throws the piece away.** The alias check by random
  machine states cannot tell apart two programs that die on all of them. Tested on the problem's own rows, the
  piece would count as learnable. Whether 10·x² + 1 then comes is still open. The program PREREG.md wrote down
  for it over those routines is 6 tokens, longer than the target searches here reached.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, written 2026-10-25 13:28 PDT, read-only since (sha256
`803c70358b2c27d293c1bd7b2ac8d7df0722392111cba90041dc63c42c8cde0c`). Its written time first read 13:30. It was corrected
to the true time, then frozen and hashed again, within the minute and before anything else ran. PREREG.md says this
section repeats its text unchanged; it does not, so read PREREG.md itself. It records what ran before it:
- the pose test (no search);
- the replays with the sources off;
- the replays with them on, for the triangular numbers, under hardest first to round 2 and under anchored pairs to
  round 1, where the target came. That one outcome was seen before the file was written, and is marked as such;
- a count from the earlier records (no search).

The replay of 10·x² + 1 with the sources on was started before the file and read only after it.

### 2. The theory, briefly

Newton's forward-difference interpolation writes a sequence from its value at 0 and its differences:
f(x) = f(0) + Σ_{i<x} Δf(i), with Δf(i) = f(i + 1) − f(i). He published it in the *Principia* (1687, Book III,
Lemma V) and set it out in the *Methodus differentialis*, printed by William Jones in 1711 (Cambridge University
Library, Newton's papers, MS Add. 3964). The explicit solution of the interpolation system was first written by de
Moivre (1730), who credited Newton (arXiv:1204.4716).

Summation undoes differencing. Graham, Knuth and Patashnik (*Concrete Mathematics*, 2nd ed. 1994, chapter 2, "Finite
and Infinite Calculus") state it as the finite calculus's fundamental theorem: Σ_{a≤k<b} Δf(k) = f(b) − f(a).
The two moves split a formula:
- taking away f(0), the constant term: 10·x² + 1 less its value at 0 is 10·x²;
- the running sum, f(x + 1) = f(x) + Δf(x): the triangular numbers have Δ = x + 1, so T(x + 1) = T(x) + x + 1.

Mod N the running sum's step is (T + x + 1) mod N, and it needs no division. The closed form x·(x + 1)/2 does need
one, and mod an even N the triangular numbers are not even a function of x mod N (their period is 2N).

What the outcome tests:
- **The running sum: rediscovered, under one rule.** Anchored pairs found the step and the loop over it. The loop
  sums the first differences, as the fundamental theorem says.
- **The constant term: posed, never used.** The piece 10X mod N is posed and solvable, and the duplicate check
  discards it before any rule can pick it.

### 3. The tiny tests (local, no Modal)

#### 3.1 What the sources pose (`tiny.py pose`; `logs/tiny_pose.txt`; before PREREG)

- **The running sum's step** is posed on both formulas from round 0, in 7 placements: y(x) in A, U or T, x in X, U
  or T. It has 21 rows at k = 7 and 45 at k = 10, every x < N − 1.
- **The offset** is posed for 10·x² + 1 once x² mod N is a routine (hardest first's library at round 2): 10X mod N,
  28 rows at k = 7, 55 at k = 10. On the triangular numbers no offset is posed, since their answer at X = 0 is 0.
- **The target's own offset**, 10·x² mod N, is one law from the target, mod(succ(g), N). Its subgoal key is (1, 35),
  against (0, 24) for x² mod N, in every round tested under both rules. So it would take the subgoal from round 0.
  The earlier hardest-first run solved x² mod N only with the subgoal's deeper probe (cost 6, round 1). The source
  was therefore restricted to the rests of the way before any run, as SUBGOAL.md §8 had defined it.

#### 3.2 The copy is the frozen learner (`check.py off`)

With both flags off, all four earlier records replay exactly from their stores (all searches read back): both rules
on the triangular numbers (held-out records) and on 10·x² + 1 (the subgoal study's).

#### 3.3 Replays with the sources on (`check.py on`; logs `check_on_*.txt`)

A replay reads every search from the stores. It runs the ones the stores lack here when they are cheap (the 30k- and
1M-state probes, the looks, the target probe), or of any kind given in `--run`.
- **The triangular numbers, hardest first, to round 2** (before PREREG): the steps learnable from round 1 and not
  picked; every round as before.
- **The triangular numbers, anchored pairs, to round 1, every missing search run** (before PREREG): the step
  rsum/A.U picked in round 1, and the target came in round 1.
- **10·x² + 1, both rules, to round 3** (read after PREREG): every round as before. Under hardest first the piece
  was posed and none of the new problems was learnable. A second replay that also prints every status
  (`logs/check_on_tenrho_lp_status.txt`) showed why: the piece is "duplicate" of law2/add(add(X,X),X), at cost 5.
- **The earlier wins** (§6).

#### 3.4 The piece and the duplicate check (`tiny.py probe`; `logs/tiny_probe.txt`; after PREREG)

On 10X mod N, with hardest first's libraries at rounds 2 and 3, the 30k-state probe finds nothing. The 1M-state probe
finds it at cost 5, `T.con2 MOV.A.X L.X[MOV.X.U M1.4]`: U = 3, A = X, then x times add 3² and reduce. That is the
program SUBGOAL.md §8 found.

The learner's alias check (bar3/learner.Signatures) runs a program on 24 random machine states, with values below 12
in every register, the cap 98 and 3,450 steps (k = 7). A program that dies there maps to one constant
(library/inventing/bfs._sig). This program dies on all 24, and so does `L.X[M0.2]`, the program for 3X. The two get
the same signature, though at N = 7, x = 1..6 one gives 3, 6, 2, 5, 1, 4 and the other 3, 6, 9, 12, 15, 18.

### 4. The runs (`run.py`; `records/*_nw1.jsonl`; `logs/run_*_nw1.log`)

Four runs: both formulas, both rules, configs `_ns` (both sources), 5 rounds, seed 0, the frozen budgets. They ran
here, one process each, two at a time.

Every search the earlier runs made was read back from their stores:
- the held-out test's stores, joined, for the triangular numbers;
- the subgoal study's four stores, for 10·x² + 1.

Every other search ran in the process and went to the run's own store (`logs/store_<run>.jsonl`), so each run
replays from the stores alone.

| run | wall | searches read back | run here | their search-hours |
|---|---|---|---|---|
| hardest first, triangular | 453 s | 1,847 | 70 | 0.12 |
| anchored pairs, triangular | 38 s | 1,479 | 43 | 0.01 |
| hardest first, 10·x² + 1 | 1,590 s | 2,009 | 84 | 0.44 |
| anchored pairs, 10·x² + 1 | 795 s | 3,422 | 76 | 0.22 |

The replays of §3.3 and §6 ran their new searches here too, and kept none. Modal was not used, so the cost is $0.
The run on Modal planned in PREREG.md was not needed: the stores left a few dozen new searches a run.

### 5. Outcomes (`verify.py`; `logs/verify_nw1.txt`)

**The triangular numbers, anchored pairs: round 1.**
- Round 0 is the earlier run's.
- In round 1 the four learnable placements were:

  | placement | cost | best program |
  |---|---|---|
  | rsum/A.X | 4 | |
  | rsum/A.U | 5 | `L.U[INC.A] MOV.X.A INC.X T.mod`, A ← (A + U + 1) mod N |
  | rsum/A.T | 5 | |
  | rsum/U.X | 5 | |
  | rsum/T.X | 5 | |

- rsum/A.U's look scored 66.5 bits; the next problem scored 20.8. Both of the round's best pairs contain it, and
  each finds the target at cost 3.
- Picked: law2/mod(N, pred(X)), rsum/A.U, restrict/remainder[T.lt(N,X)@x+QN → N]/X<3, law2/min(add(X,X),N). The step
  replaced law2/mod(N, monus(N, X)) in the earlier run's picks.
- After the merges, the target search found `L.X[M1.4]` at cost 2, where M1.4 = `INC.U MOV.X.A L.U[INC.X T.mod]`:
  U ← U + 1, then A ← (A + U) mod N.
- In primitives: `L.X[INC.U MOV.X.A L.U[INC.X MOV.A.X L.N[MOV.T.A DEC.A] Z.T[MOV.A.X]]]` (11).
- Checked on every pair x < N ≤ 120:
  - 7,260 of 7,260 right with 10⁹ steps;
  - on machine_for(120) (721,000 steps), 5,998 right and 1,262 out of steps: about x²·N/2 steps at the largest pairs.

**The triangular numbers, hardest first: no.**
- All 5 rounds have the earlier run's picks and library.
- The step's placements:

  | placement | learnable, cost | rounds |
  |---|---|---|
  | rsum/A.X | 4 | 1–4 |
  | rsum/A.U, rsum/A.T | 5 | 1–3 |
  | rsum/U.X, rsum/T.X | 5 | 1–2 |

- **Pool order.** In rounds 1–3 the learnable problems at cost 5 that stand ahead of the new source in pool order
  numbered 18, 14 and 12, with none costlier. All four picks went to them.
- **Probe reach.** In round 4 the costliest earlier problems cost 4, so a cost-5 step would have been picked. But
  rsum/A.U went from learnable in round 3 (a library of 20; the probe streams level 5) to unsolved in round 4 (28
  routines; level 4 alone passes 1M states).

**10·x² + 1, hardest first: no.** All 5 rounds have the earlier run's picks and library. The piece was posed as
"offset/remainder/M1.4 -> A" in round 2, a near miss's remainder that is the same problem as the rest from x² mod N,
and as "offset/rrem/T.law5" in rounds 3–4. It was solved at cost 5 and marked duplicate in rounds 2–3, and was
unsolved in round 4 (the probe's reach again). The running sum's step was never learnable for this formula: its step
is (Y + 10·(2x + 1)) mod N.

**10·x² + 1, anchored pairs: no.** All 5 rounds have the earlier run's picks and library. x² mod N is never
practised, so the rest of the way from it is never posed. Its only offset, of the rest from T.add(X,X) in round 3,
is not learnable.

### 6. The check on the earlier wins (`check.py on`; `logs/check_on_wins.txt`)

The nine held-out runs whose formula came were replayed with both sources up to that round, each search the stores
lack run here (14 to 49 a run, all cheap). The runs: (x² + x), (x + 1)², 3·x² and x⁴ under both rules, and 2^x
under anchored pairs.

All nine have the records' picks, library and target search in every round up to the round they came. So each comes
in the same round with the same program.

The new problems posed there:
- the step's 7 placements, every round;
- offsets of near misses' remainders in five of the runs (up to four in 2^x's).

None of them was picked. On 2^x the step was learnable at cost 3–4 in round 1, three of its placements as
duplicates.

### 7. Predictions against outcomes

| prediction (PREREG.md) | confidence | outcome |
|---|---|---|
| triangular, hardest first: comes in round 4 | low-medium | **failed**: no; in round 4 the step was out of the probe's reach |
| triangular, anchored pairs: round 1 | seen before | came in round 1, as the replay before PREREG showed |
| 10·x² + 1, hardest first: no | medium | **held**. Its mechanism failed: "the piece learnable at cost 5 and picked in round 4"; it was marked a duplicate, then out of reach |
| 10·x² + 1, hardest first: the risk of a near miss's remainder taking the subgoal | low | did not happen; the subgoal was the earlier run's in every round |
| 10·x² + 1, anchored pairs: no, the run is the earlier one | high | **held** |
| the earlier wins: all 9 come in the same round with the same program | medium-high | **held** |

PREREG's caveat for hardest first on the triangular numbers named the wrong direction. It feared a merge would make
the step cheaper, so that cost-4 problems stood ahead of it. What happened is the reverse: the step got harder to
reach, because the probe's budget is fixed while the library grows.

### 8. Dead ends and loose ends

- **The target's own offset** was dropped before any run (§3.1). A source that offsets the target would need the
  subgoal not to prefer it over x² mod N. Kept in `tiny.py` to show why.
- **The duplicate check by random states.** Two programs that die on all 24 random test states share a signature.
  That is how the piece was lost here. It is the frozen learner's check, unchanged in this study. Checking on the
  problem's own rows would tell them apart. Not tried.
- **The probe's reach shrinks as the library grows.** With a fixed state budget, cost-5 problems learnable in round 3
  were unsolved in round 4. It hits every source, not only these.
- **An outcome seen before the predictions.** The triangular numbers under anchored pairs. The local replay meant
  to check what the sources pose ran that run's two rounds in full.

### 9. Files and how to rerun

Python with numba: the scratchpad venv that ran the digits study (`nvenv`); the plain one:
`~/github/sotaku/venv/bin/python`.

- `sources.py`: the two sources; `make_learner.py` writes `learn.py` from the frozen learner (3 asserted
  replacements); `configs.py`: the `_ns` configs.
- `tiny.py`: `pose` and `probe` (§3.1, §3.4).
- `check.py`: replays on the stores: `off` (§3.2) and `on` (§3.3, §6).
- `run.py`: a run here from the stores (§4).
- `verify.py`: the outcomes and the checks to N = 120 (§5).
- `PREREG.md` (read-only); `records/*_nw1.jsonl`; `logs/`: every output above, the runs' logs and their own stores.

The earlier stores are not in the repository: the held-out test's are on the Modal Volume `bar3-heldout` (v1/,
`store_ctl_<rule>_ho1.jsonl` after its control), the subgoal study's on `bar3-subgoal` (v1/, four files).

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/newton/make_learner.py
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.newton.tiny pose
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.newton.check off STORES -- RECORDS
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.newton.check on [--upto=R] [--run=KINDS] STORES -- RECORDS
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.newton.run ct2a2_clean_deep_sub_ns tri nw1 STORES
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.newton.verify
```

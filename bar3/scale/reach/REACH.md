# A deeper look at the posed steps, and the subgoal's twin as a fifth pick

<!-- PLAIN -->
## In plain words

**The question.** PICKS.md §8 named two changes aimed at x² mod N at nine-digit x:
- **A deeper look at the steps the digits source poses.** The five-token program that passes at big x,
  `D.2[step] MOV.X.A CLR.A T.law10`, lay one level past every look, so anchored pairs' big-x score could never see it.
- **The subgoal's twin as a fifth pick.** Practising the twin in one of the four slots had pushed hardest first's
  queue back a round and lost binary x mod N.

Each went behind its own flag, in copies of the learners, with predictions written down first (`PREREG.md`).

**What came: nothing new, and nothing lost.**

| change | arm | before | with the change |
|---|---|---|---|
| deeper look, with the big-x score | x² mod N, base 2, anchored pairs | no | no: every round the earlier run's |
| twin as a fifth pick | x² mod N, base 2, hardest first | no | no |
| twin as a fifth pick | binary x mod N, hardest first | round 3 | round 3 (kept) |

- **The deeper look did its part.**
  - In round 3 it found what the target accepts, `D.2[MOV.U.N T.look0] CLR.A T.law10` and others at cost 5, for
    three placements of the step. The frozen look finds nothing there.
  - With the big-x score each of those steps scored 2,189 bits; nothing else scored more than 12.
- **Anchored pairs does not pick by those scores.**
  - It picks the members of its best pairs, among pairs that include one of the two costliest problems. Single
    scores only fill slots the pairs leave, and the pairs filled all four.
  - The pair looks kept the old 30,000-state budget, found nothing, and the best pair gained 11.9 bits without the
    steps. So the steps were never picked, and every round is the earlier run's.
- **The fifth pick keeps binary x mod N.** It comes in round 3 as before, `D.2[MOV.T.X T.dig16]`, right on all rows,
  at big x, on 1,000 x below 10⁹ and at N ≤ 40. The four picks of each round stay as they were, so the queue does not
  move.
- **The fifth pick does not get x² mod N.**
  - The twin was practised in round 1. Its steps were practised in round 3, sooner than predicted, since the queue now
    reached them.
  - But x² mod N on the pairs (the square) was learnable at cost 4 and never picked. With it as a routine only from
    round 4, the target needed five tokens, and the target search ran out of its 2,400 s at level 5 in rounds 3 and
    4.

**The 12 earlier wins: all still come, each in the same round.**
- **With the deeper look:** all 12 are unchanged: the nine held-out, both binary x mod N runs, and 10·x² + 1.
  Anchored pairs' binary x mod N ran its nine deeper looks and kept its picks and program.
- **With the fifth pick:** 11 are unchanged, since no subgoal of theirs is a duplicate before the target comes.
  Hardest first's binary x mod N changes from round 1 and still comes in round 3, with the same program shape.

**Against the predictions.**
- Held: every "certain" item, the deeper look finding the program (medium-high), the fifth pick keeping binary x mod N
  (medium), and x² mod N not coming under either change.
- Failed, both on how the picks work:
  - "the step is then picked" (high, given the look): anchored pairs picks from pairs, and its pair looks are shallow;
  - "the twin's step is not picked" (medium-high): the queue reached it in round 3.

**Cost.** About $7.4 on Modal: two apps, 1,895 new searches, 20.2 search-hours (12.8 of them in 27 deeper looks).
The replays and the tiny test ran here.

**What it says.**
- **The deeper look reaches the program, and the big-x score then sees it.** What blocks anchored pairs is now its
  own pick rule, which ranks pairs whose looks are still shallow. Giving pair looks that include a step the same
  deeper budget, or letting an accepted single look take a slot, is the next change to test.
- **Under hardest first the fifth pick costs nothing and gets the twin's steps practised.** What remains is the
  square, x² mod N on the pairs, practised in time, and a target search that reaches cost 5. At level 5 it ran out of
  time, and it checks 200 programs a level among hundreds of look-alikes that fit every row.
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, read-only since it was written, sha256 `994a20817335150577329836ca4faac13368232c68f833f4639bc69dac0f8f3f`.
Its time was stamped as it was frozen (01:24 PDT), before anything ran with a change on. Before it: the all-off
replays (§2.3), and the facts settled from the records (its "Settled" section).

PREREG.md's text, unchanged:

````markdown
# A deeper look at the posed steps, and the twin as a fifth pick: pre-registration

Written 2026-10-26 01:24 PDT, before any run with one of the changes on. REACH.md §1 repeats this text unchanged.

## The question

PICKS.md §8 named two changes aimed at x² mod N at nine-digit x:
- **Under anchored pairs**, the five-token program that passes at big x, `D.2[step] MOV.X.A CLR.A T.law10`, is one
  level past every look, so the big-x score never sees it.
- **Under hardest first**, practising the subgoal's twin in one of the four slots delayed the queue and lost binary
  x mod N.

Does a deeper look at the posed steps let anchored pairs pick the step and get x² mod N at big x? Does the twin as
a fifth pick keep binary x mod N, and does it help x² mod N?

## The changes (`reach.py`, `make_learners.py`)

- **`deeplook`.** Anchored pairs' single look at a step the digits source posed gets a deeper budget, as the
  subgoal gets a deeper probe:
  - 1M states, so that it stores level 4 and streams level 5;
  - 3,600 s;
  - 20,000 programs checked per level instead of 200.

  The extra checks are there because at level 4 alone, 602 counting programs fit every row and fail at big x
  (PICKS.md §8). Pair looks keep their budget. The arm runs it with the picks study's big-x score on (`_bigx_deep`),
  since the deeper look is there for that score to act on.
- **`twin5`.** The subgoal's twin (picks.twins: the learnable problem the subgoal is a duplicate of) is practised as a
  fifth pick as soon as it is learnable. It neither replaces one of the four nor joins the subgoal's ties, so the four
  picks are the frozen learner's. It joins the subgoal set, so its routine counts as a practised subgoal's.

The flags of the earlier studies stay available. The arms run only the flags named for them below.

Learner copies: `learn_newton.py` and `learn_digits.py`, each written by `make_learners.py` from the learner frozen at
97f30b9 with that study's asserted replacements, then the row check's 3, the picks study's 7 and this study's 4. With
every flag off, 14 earlier records replay exactly from their stores (`check.py off`; `logs/check_off_*.txt`):
- the nine held-out wins;
- the picks study's 10·x² + 1 (its config `lp_clean_deep_sub_ns_rc_tie`);
- the digits study's four base-2 runs with the source.

## Settled from the records before this file (no run)

- **`deeplook` acts only on looks at the digits source's steps, which only anchored pairs runs, and only in the
  digits runs.** So it cannot change:
  - the nine held-out wins, 10·x² + 1, or hardest first's binary x mod N.

  Where it does act:
  - **anchored pairs' binary x mod N (an earlier win):** in round 1, 9 steps are learnable: the target's 5 and
    T.add(X,N)'s 4;
  - **the arm, anchored pairs' x² mod N:** no step is learnable in rounds 0–1, then 5, 10 and 12 in rounds 2–4.
- **`twin5` acts only where the subgoal is a duplicate of a learnable problem:**
  - x² mod N, hardest first, rounds 1–4;
  - binary x mod N, hardest first, round 1;
  - x² mod N, anchored pairs, round 1.

  No subgoal in the nine held-out wins, 10·x² + 1, or anchored pairs' binary x mod N is a duplicate before its target
  came.

## Runs

Each arm from the earlier runs' stores, 5 rounds, seed 0, the frozen budgets, with its flags:
- **`ct2a2_clean_b_src_bigx_deep`:** x² mod N, base 2, anchored pairs. Flags: big-x score, deeper look.
- **`lp_clean_b_src_twin5`:** x² mod N, base 2, hardest first; and binary x mod N, hardest first, which must still
  come in round 3. Flag: twin5.
- **The 12 earlier wins,** replayed with each change on, up to the round each came:
  - the nine held-out wins (`<rule>_deep_sub_deep`, `_twin5`);
  - the two binary x mod N wins (`<rule>_b_src_bigx_deep`, `_twin5`);
  - 10·x² + 1 (`lp_clean_deep_sub_ns_rc_tie_deep`, `_twin5`).

The tiny test, first: the deeper look at round 3's twin step, on the earlier anchored-pairs run's library, locally.
Searches the stores lack run locally under a memory cap when few, else on Modal from the stores. If x² mod N comes,
its program is checked as the digits study checked binary x mod N:
- on the rows;
- at big x;
- on 1,000 x below 10⁹ at N ≤ 10;
- on 1,000 x at N ≤ 40.

Money: soft $20, cap $25, launches guarded by `spend.py 4750`, one spend watcher.

## Predictions

| arm | prediction | confidence |
|---|---|---|
| deeper look, x² mod N base 2, anchored pairs | rounds 0–1 the earlier run's | certain |
| | round 2: no deeper look finds an accepted program, and the picks are the earlier run's | high |
| | round 3: the twin step's deeper look finds `D.2[step] MOV.X.A CLR.A T.law10` | medium-high |
| | then that step is picked | high, given the look |
| | x² mod N comes, in round 3 or 4 | low-medium |
| | if it comes, its program is right on the 1,000 and 1,000 x | high |
| twin5, x² mod N base 2, hardest first | the twin practised as a fifth pick in round 1; its step learnable from round 2, not picked; no | high; medium-high; medium-high |
| twin5, binary x mod N, hardest first | comes in round 3 | medium |
| the earlier wins, deeper look | 11 unchanged | certain |
| | anchored pairs' binary x mod N comes in round 1 | high |
| | with the same program | medium |
| the earlier wins, twin5 | 11 unchanged | certain |
| | hardest first's binary x mod N: the arm above | |

Why:
- **The deeper look, round 2.** x² mod N on the pairs (T.law10) is practised in round 2 and is not yet a routine,
  so the program is longer than five tokens.
- **The deeper look, round 3.** The program is right on all rows and at big x on round 3's library (PICKS.md §8). A
  1M-state look stores level 4 (about 0.8M states) and streams level 5. Checking 20,000 programs per level should
  reach it among the look-alikes.
- **The deeper look, the pick.** Found, it gains the big-x charge, about 2,193 bits, which no other candidate gets.
- **The target.** The target search (3M states, 200 programs per level) must then find the same five tokens with
  the step as a routine. At level 5 the look-alikes may fill its 200 programs, unless a merge shortens the program.
  The guided search is the other way.
- **The fifth pick, x² mod N.** The twin is practised as before (PICKS.md §5), and its step, posed from round 2,
  waits behind the cost-5 problems ahead of it in pool order. In round 4 it is out of the probe's reach.
- **The fifth pick, binary x mod N.** The four picks of round 1 are the earlier run's, plus the twin. The queue of
  cost-5 problems is the earlier run's, so the steps' turn comes in round 3 again. But one more routine widens every
  level, and round 3 was the last round in which the steps' cost-5 placements were in the probe's reach.
- **The wins under the deeper look.** The target's steps' looks find the target at cost 2–3, and a deeper look stops
  at the same level, so their gains are the same. T.add(X,N)'s steps get deeper looks too. One of those could find an
  accepted program the shallow look missed and change the picks.

What is reported for each run: whether and in which round its target came, the program and its checks; what the
changes did each round (the deeper looks' results, the fifth picks); and the first round that differs from the
earlier run.
````

### 2. The changes

#### 2.1 What they borrow

- **The deeper look** is the subgoal study's deeper probe (SUBGOAL.md), aimed at the looks of the posed steps. A
  budget that ends one level short of a program cannot score it, however that program is scored. The deeper look's
  budget:
  - stores level 4 and streams level 5 (1M states, 3,600 s);
  - checks 20,000 programs a level instead of 200. At level 4 alone, 602 programs fit every row and fail at big x
    (PICKS.md §8).
- **The fifth pick** keeps priority inheritance (Sha, Rajkumar and Lehoczky 1990, §III; PICKS.md §2.1) and drops what
  cost the earlier win, the slot taken from the queue.

#### 2.2 The changes (`reach.py`)

- **`deeplook`:** anchored pairs' single look at a step the digits source posed (source "digit") gets `DEEP_LOOK`.
  Pair looks keep the frozen budget. The arm runs it with the picks study's big-x score (`_bigx_deep`).
- **`twin5`:** the subgoal's twin (picks.twins) is practised as an extra pick as soon as it is learnable. It does not
  join the subgoal's ties, and it joins the subgoal set.

#### 2.3 The copies (`make_learners.py`, `check.py off`)

`make_learners.py` writes two copies from the learner frozen at 97f30b9, each by asserted exact replacements: that
study's, the row check's 3, the picks study's 7, then this study's 4. With every flag off, 14 earlier records replay
exactly from their stores (`logs/check_off_*.txt`):
- the nine held-out wins;
- the picks study's 10·x² + 1;
- the digits study's four base-2 runs.

### 3. The tiny test (`tiny.py`; `logs/tiny_deeplook.txt`)

On round 3 of the earlier anchored-pairs x² mod N run, with that round's library and the twin's step
(`L.A[INC.A] T.law10`) as T.look0:
- **The frozen look:** nothing accepted. Levels [1, 33, 823, 18,577, 784,720], level 4 streamed, 26 s, 212 programs
  right on the rows only.
- **The deeper look:** cost 5 accepted, e.g. `D.2[MOV.U.N T.look0] CLR.A T.law10`. Levels [1, 33, 823, 18,577,
  394,282, 16,911,152], level 5 streamed, 901 s. It checked 17,955 programs right on the rows only before the
  accepted ones.

### 4. Replays and runs (`check.py on`; `run` on Modal: `modal_reach.py`; `records/*_rh1.jsonl`; `logs/`)

**Replays of the 12 earlier wins with each change on** (`logs/check_on_*`):
- the ten in the two-source copy (nine held-out, 10·x² + 1): unchanged under both changes, with no new search;
- the two binary x mod N runs, under the deeper look: unchanged. Anchored pairs' ran its nine deeper looks here and
  kept its picks;
- the two binary x mod N runs, under the fifth pick: anchored pairs' unchanged; hardest first's diverges from round 1,
  so it ran as an arm.

**On Modal, from the earlier stores** (the digits study's base-2 stores and the picks study's twin runs, joined, on
Volume `bar3-reach`):

| app | runs | ran | preempted | billed | new searches |
|---|---|---|---|---|---|
| `ap-rNgHUTxhhnTgESIgEv9T3Z` | twin5: x² mod N and binary x mod N, hardest first | 08:41–12:46 UTC | once, 10:58 | $4.22, and the 12:00 hour still to come (about $1) | 1,868, 7.41 search-hours |
| `ap-viC4DxhdxbwaS7GfqL3S33` | deeper look and big-x score: x² mod N, anchored pairs | 08:42–10:25 UTC | no | $2.21 | 27, 12.81 search-hours |

`spend.py 4750` printed LAUNCH OK before each launch ($3,589.36 billed this month). One spend watcher covered both apps
(`logs/watch_rh1.log`) and never had to stop one.

### 5. Outcomes (`verify.py`; `logs/verify_rh1.txt`)

**x² mod N, base 2, anchored pairs, the deeper look and the big-x score: no.**
- Every round's picks, library and target search are the earlier run's.
- **Round 3.** Three deeper looks found accepted programs at cost 5, for steps A.X, A.U and U.X. Each step's single
  score was 2,188.9 bits; the next was 11.9.
  - The picks came from pairs: law2/mul(add(X,N),N), restrict/target/X+X<N·N, law2/mul(add(X,X),N), and value/T.mod
    (X,N)@x+QN → T, the best pairs gaining 11.9 and 5.9 bits.
  - The rule pairs only problems with one of the two costliest (`anchor_cost`), and the pairs filled all four slots,
    so single scores were never read.
- **Round 4.** Six looks found accepted programs, at costs 3–4 (e.g. `T.law15 T.law10`), with the steps at 2,183–2,189
  bits. The pairs, with gains of 0, again filled the four slots.

**x² mod N, base 2, hardest first, the fifth pick: no.**
- **Round 1:** the twin, law/mod(X,N)/x+QN, was practised as a fifth pick beside the earlier run's four.
- **Round 3:** the twin's steps A.U and A.T were picked.
- **x² mod N on the pairs** (law2/mod(mul(X,X),N)) was learnable at cost 4 in rounds 2–3, never picked, and trivial
  over a routine in round 4.
- **The target search** checked 582 (round 3) and 590 (round 4) programs right on the rows only, and stopped on its
  time limit at level 5 both times.

**Binary x mod N, hardest first, the fifth pick: round 3, kept.**
- Round 1 practised five problems: the earlier run's four, and the twin law2/div(succ(X),N).
- Rounds 2–3 picked what the earlier run picked. Round 3 practised steps A.U and A.T, and the target search found
  `D.2[MOV.T.X T.dig16]`.
- The program is 12 primitive instructions. It is right on 490/490 and 1,000/1,000 rows and at big x (100%), on
  1,000/1,000 x below 10⁹ at N ≤ 10 (100,000 steps), and on 1,000/1,000 at N ≤ 40 (10⁷ steps).

### 6. Predictions against outcomes

| prediction (PREREG.md) | confidence | outcome |
|---|---|---|
| deeper look: rounds 0–1 the earlier run's | certain | **held** |
| round 2: no deeper look finds an accepted program; the earlier picks | high | **held** |
| round 3: the twin step's deeper look finds the five-token program | medium-high | **held**: three placements, cost 5 |
| then that step is picked | high, given the look | **failed**: anchored pairs picks from pairs, whose looks are shallow |
| x² mod N comes, in round 3 or 4 | low-medium | no |
| twin5, x² mod N: the twin practised as a fifth pick in round 1 | high | **held** |
| its step learnable from round 2 | medium-high | **held** |
| its step not picked | medium-high | **failed**: picked in round 3 |
| x² mod N does not come | medium-high | **held** |
| twin5, binary x mod N comes in round 3 | medium | **held** |
| the wins, deeper look: 11 unchanged | certain | **held** |
| anchored pairs' binary x mod N in round 1 | high | **held** |
| with the same program | medium | **held** |
| the wins, twin5: 11 unchanged | certain | **held** |

### 7. What is left

- **Anchored pairs' pick rule.** The deeper look and the big-x score find and value the step, but picks come from
  pairs anchored on the two costliest problems, and pair looks are shallow. To test:
  - give a pair look the deeper budget when it includes a posed step;
  - or let a single look that finds an accepted program take a slot ahead of the pairs.
- **The target search at cost 5.** With the step practised (hardest first, fifth pick) the target search ran out of
  time at level 5 and checks 200 programs a level among hundreds of look-alikes. When a look has found an accepted
  program with a routine since practised, that program is a candidate for the target. Passing it on, or a deeper,
  wider target search, would test it.
- **The square in time.** Under hardest first, x² mod N on the pairs waits at cost 4 behind cost-5 problems. With the
  step practised in round 3, the square's absence was the missing piece.

### 8. Files and how to rerun

Python with numba: the scratchpad venv that ran the digits study (`nvenv`). The plain one:
`~/github/sotaku/venv/bin/python`.

- **The changes:** `reach.py`; `make_learners.py` writes `learn_newton.py` and `learn_digits.py`; `configs.py`.
- **Tests and runs:** `tiny.py`; `check.py` (`off`, `on --variant=_deep|_bigx_deep|_twin5`); `modal_reach.py`.
- **Results:** `verify.py`.
- **Records:** `PREREG.md` (read-only); `records/*_rh1.jsonl`; `logs/`: every output above, the apps' captured logs and
  the watcher's log.

The earlier stores are on Modal Volumes (`bar3-heldout`, `bar3-subgoal`, `bar3-digits`, `bar3-picks`; this study's
joined ones on `bar3-reach`), and the two-source and picks studies' own stores in their `logs/`.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/reach/make_learners.py
```

```bash
cd cpu_first && bar3/scale/rowcheck/memcap.sh 12 <python with numba> -m bar3.scale.reach.tiny
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.reach.check on --variant=_twin5 [--digits] STORES -- RECORDS
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/scale/reach/modal_reach.py --fresh lp_clean_b_src_twin5 --targets dxsq,dxmod --tag rh1 --store store_rh1_twin5.jsonl
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.reach.verify
```

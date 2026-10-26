# The duplicate check on a problem's own rows

<!-- PLAIN -->
## In plain words

**The question.** The learner throws a solved practice problem away when another problem's program looks the same.
"Looks the same" means doing the same thing to 24 random machine states. Every program that runs out of steps on all
24 looks like every other such program. That is how 10X mod N, the piece the offset source posed for 10·x² + 1, was
thrown away as a copy of 3X (NEWTON.md).

Program synthesis has a standard answer: compare programs by their outputs on the task's own examples. Escher calls
this observational equivalence. This study puts that check in the learner, behind a flag: a problem is a copy only
when the other program is right on the problem's own rows. Then it reruns what the earlier studies said the fix
should unlock:
- 10·x² + 1 under hardest first;
- x² mod N in base 2 with the digits source, under both rules.

Predictions were written down first (`PREREG.md`).

**What the check changed.** In the three reruns it freed 9 problems that the old check had thrown away:

| run | problems freed | rounds | cost |
|---|---|---|---|
| 10·x² + 1, hardest first | 7, including the piece 10X mod N | the piece in rounds 2 and 3 | 5 |
| x² mod N, base 2, hardest first | 1 (X mod X·X) | 2–4 | 3 |
| x² mod N, base 2, anchored pairs | the same 1 | 3–4 | 3 |

The same alias was freed in hardest first's binary x mod N run. In the nine held-out wins and anchored pairs' binary
x mod N run it changed nothing.

**What came: nothing new.** Every run's picks, library and target search are the earlier run's, round for round.

| run | before | with the check |
|---|---|---|
| 10·x² + 1, hardest first | no, 5 rounds | no, 5 rounds |
| x² mod N, base 2, hardest first | no, 5 rounds | no, 5 rounds |
| x² mod N, base 2, anchored pairs | no, 5 rounds | no, 5 rounds |

All 11 earlier wins come in the same round with the same program: the nine held-out wins, and binary x mod N under
both rules in the digits study.

**Why.**
- **10·x² + 1.** The piece is now learnable, but hardest first never picks it. It takes the four costliest problems,
  ties in pool order, and the piece comes last. 24 problems at cost 5 stood ahead of it in round 2, and 12 in round
  3. In round 4 the library had grown past the probe's reach.
- **x² mod N.** Its hidden state's copy is a true one. The problem the subgoal found, x mod N on long x, and the law
  book's x mod N on short pairs are solved by the same program, so the new check keeps them together. Under anchored
  pairs the blocker was never this check, but the pick score: counting programs already fit every row.
- **The freed problems.** None was picked.

**Against the predictions.** As pre-registered, the fix changed none of the three runs' outcomes, and the wins stand.
- Held: the piece is learnable in rounds 2 and 3 and still not picked; x mod N on long x stays a duplicate; some
  statuses change.
- Failed: the medium-confidence prediction that some pick would change somewhere. None did.

**Cost.** $0: nothing ran on Modal. The earlier stores held every search but 6 looks, which ran locally under a
memory cap (0.03 search-hours).

**What it says.** The old check was wrong for 10X mod N, and the new one is right and changes nothing that was right.
But the earlier misses each have another blocker:
- hardest first's pool order and the probe's shrinking reach, for 10·x² + 1;
- a subgoal whose priority does not pass to its twin, and a pick score that sees only the rows, for x² mod N.

Those are the next fixes to test (§7).
<!-- /PLAIN -->

## Details

### 1. Pre-registered

`PREREG.md`, read-only since it was written, sha256 `227fbca76691b3b78478d2b52e9a14a0890d0f898430f237964436686f2ed218`.
- Its written time first read 20:05 PDT. It was frozen at 20:04, corrected to that true time, then frozen and hashed
  again within the minute, before anything ran with the check on.
- Before it: the flag-off replays (§2.3), and two facts settled from the records with no run (PREREG.md, "The two
  pieces").

PREREG.md's text, unchanged:

````markdown
# The duplicate check on a problem's own rows: pre-registration

Written 2026-10-25 20:04 PDT, before any run with the new check on. ROWCHECK.md §1 repeats this text unchanged.

## The question

The frozen learner calls a solved problem a duplicate when its best program has the same signature as an earlier
learnable problem's, and an alias when it has a routine's or a primitive's. A signature is what a program does to 24
random machine states (bar3/learner.Signatures). Every program that dies on all 24 shares one signature. That is how
10X mod N, the piece the offset posed for 10·x² + 1, was called a duplicate of 3X (NEWTON.md §3.4).

Enumerative synthesis tells programs apart by their outputs on the task's examples: TRANSIT (Udupa et al., PLDI
2013, §1) prunes "by considering two expressions equivalent if they evaluate to the same values for the given concrete
examples". Escher (Albarghouthi, Gulwani and Kincaid, CAV 2013, §3.3) names it observational equivalence reduction,
and proves its search stays complete under it. The new check applies that to each problem on its own rows. Does it
unlock what the earlier studies named?

## The change (`rowcheck.py`, `make_learners.py`)

With cfg["rowcheck"], a shared signature only proposes. A problem is an alias when a primitive or routine with its
signature is right on all its own rows, and a duplicate when an earlier learnable problem's best program with its
signature is. Its own rows are the instances it is accepted on and its check instances, its registers set as its job
sets them. Otherwise it stays learnable. Nothing else changes.

The flag is added to copies of the two learners the reruns need, each written by `make_learners.py` from the learner
frozen at 97f30b9 with that study's asserted replacements, then this study's three:
- `learn_newton.py`, the two-source study's learner: for 10·x² + 1, and with no source on, the held-out test's;
- `learn_digits.py`, the digits study's.

Run before this file (`check.py off`; `logs/check_off_*.txt`): with the flag off, 14 earlier records replay exactly
from their stores:
- the nine held-out wins and the two-source study's hardest-first run on 10·x² + 1, on `learn_newton.py`;
- the digits study's four base-2 runs with the source, x² mod N and x mod N under both rules, on `learn_digits.py`.

## The two pieces, settled from the records (no run)

- **10·x² + 1: the fix separates them.** On its own rows 10X mod N differs from 3X (at N = 7, x = 1..6: 3, 6, 2, 5, 1, 4
  against 3, 6, 9, 12, 15, 18; NEWTON.md §3.4).
- **x² mod N: the fix does not.** The problem the digits study's subgoal found, x mod N on long x
  (`law/mod(X,N)/rows`), is a duplicate of the law book's x mod N on short pairs (`law/mod(X,N)/x+QN`). In every
  round of the records the two problems' best programs are the same program (`L.X[T.mod MOV.X.A]`, later
  `MOV.A.X M1.1`), so that program is right on the twin's own rows too. The check at big x is the target's acceptance,
  not part of a practice problem's rows.

## Runs

The arms, from scratch, 5 rounds, seed 0, the frozen budgets, with the check on:
- 10·x² + 1 under hardest first, both sources (`lp_clean_deep_sub_ns_rc`);
- x² mod N in base 2 with the digits source, under both rules (`lp_clean_b_src_rc`, `ct2a2_clean_b_src_rc`).

The earlier wins, replayed with the check on up to the round each came:
- the nine held-out wins (`<rule>_deep_sub_rc`, no source on);
- binary x mod N in the digits study, under both rules (`<rule>_b_src_rc`).

Each run starts from the earlier runs' stores, and reads a stored search back instead of repeating it. Searches the
stores lack run locally under a memory cap (`memcap.sh`). Modal is used only for what is too big to run locally.
Money: soft $20, cap $25.

## Predictions

| arm | outcome | confidence |
|---|---|---|
| 10·x² + 1, hardest first | no, 5 rounds | medium |
| x² mod N, base 2, hardest first | no, 5 rounds | high |
| x² mod N, base 2, anchored pairs | no, 5 rounds | high |
| the 11 earlier wins | each in the same round with the same program | medium |

Why:
- **10·x² + 1, hardest first.** If rounds 0–1 keep the earlier picks, the piece is posed in round 2 and is learnable
  at cost 5 in rounds 2 and 3 (medium). It is still not picked. Hardest first takes the four costliest problems, ties
  in pool order. The offsets come last in the pool, and 24 (round 2) and 12 (round 3) problems learnable at cost 5
  stood ahead of them in the earlier run. In round 4 the piece was out of the 1M-state probe's reach.
- **x² mod N, base 2, hardest first.** x mod N on long x stays a duplicate of its twin (high), and that twin, at cost
  2–3, is never among hardest first's four costliest. So no routine holds x mod N on long x, and the source poses no
  step for x² mod N, as before.
- **x² mod N, base 2, anchored pairs.** The twin was practised in round 1 and the step was learnable from round 2.
  The blocker was the pick score: a look with the step gained −0.1 bits, since counting programs already fit every
  row. The check does not touch that score.
- **Elsewhere.** The check also reaches every other problem whose best program shares a signature with a program
  that is wrong on its rows. Loop-heavy programs die on the random states (3,450 steps at k = 7), so in at least one
  of the 14 runs some status changes from duplicate or alias to learnable (high). Picks change in at least one round
  of at least one run (medium). A changed pick in rounds 0–1 can change an earlier win. That is why the wins are
  rechecked.

So I expect the fix to change none of the three arms' outcomes. The local replays settle that at no cost. If they do
not change the arms, nothing runs on Modal, and that is the report.

What is reported for each run: whether and in which round its target came, the program and its check; every status
the check changed, by round; and the first round whose picks, library or target search differ from the earlier
run's.
````

### 2. The theory and the change

#### 2.1 Observational equivalence

Enumerative synthesizers tell programs apart by their outputs on the task's examples, and keep one program per class.
- TRANSIT (Udupa, Raghavan, Deshmukh, Mador-Haim, Martin and Alur, PLDI 2013, §1) prunes its space of expressions
  "by considering two expressions equivalent if they evaluate to the same values for the given concrete examples".
- Escher (Albarghouthi, Gulwani and Kincaid, "Recursive Program Synthesis", CAV 2013, §3.3, "Search guidance")
  names it observational equivalence reduction. P ≡ Q iff they give the same outputs on the task's inputs. A newly
  synthesized P is discarded when an equivalent Q exists, and its Proposition 1 shows that its completeness theorem
  (Theorem 1) still holds.

The frozen learner's alias and duplicate check is an equivalence test on random states, independent of the problem:
the 24 states of bar3/learner.Signatures, through library/inventing/bfs._sig. Two things set it apart:
- **It is not on the problem's examples.** Two programs can differ on a problem's rows and agree on the states.
- **Death is a class.** A program that runs out of steps on a state maps to one constant there, so all programs that
  die on all 24 states share one signature.

The new check uses the signature only to propose a candidate. It then applies Escher's relation per problem: c is a
copy of c′ when c′'s best program is right on c's own examples. That is exactly when practising c′ already gives a
routine that solves c.

#### 2.2 The change (`rowcheck.py`)

With cfg["rowcheck"] a solved problem is:
- **an alias** when a primitive or routine with the signature of its best program is right on all its own rows;
- **a duplicate** when an earlier learnable problem's best program with that signature is;
- **learnable** otherwise, and it joins that signature's list.

Its own rows are the instances it is accepted on and its check instances, with its registers set as its job sets
them (bar3/scale/jobs.run_on). The check at big x is the target's acceptance, not part of a practice problem's rows.
Without the flag, the frozen code runs.

#### 2.3 The copies, and the check that they are the earlier learners (`make_learners.py`, `check.py off`)

`make_learners.py` writes two copies from the learner frozen at 97f30b9, each by asserted exact replacements:
- `learn_newton.py`: the two-source study's 3, then this study's 3;
- `learn_digits.py`: the digits study's 11 (and its DigitsTap), then this study's 3.

Configs: the earlier studies' configs, and each with `_rc` (`configs.py`). With the flag off, 14 earlier records
replay exactly from their stores (`logs/check_off_newton.txt`, `logs/check_off_digits.txt`):
- the nine held-out wins, on `learn_newton.py` with no source;
- the two-source study's hardest-first run on 10·x² + 1;
- the digits study's four base-2 runs with the source (x² mod N and x mod N, both rules).

### 3. The tiny test: replays with the check on (`check.py on`; `logs/check_on_*.txt`)

Each of the 14 runs was replayed with the check on, up to the round its target came (all 5 rounds for the three
reruns), every search read from the stores. Cheap searches the stores lacked were run here. The replays needed 6:
looks under anchored pairs on x² mod N, for the freed alias in rounds 3 and 4. No round needed any other search, so
every replay ran to its end.

### 4. The runs (`run.py`, under `memcap.sh` at 10 GB; `records/*_rc1.jsonl`; `logs/run_*_rc1.log`)

The three reruns, as runs that write records: 5 rounds, seed 0, the frozen budgets, from the earlier stores, every
other search run here into the run's own store.

| run | wall | searches read back | run here |
|---|---|---|---|
| 10·x² + 1, hardest first (`lp_clean_deep_sub_ns_rc`) | 15 s | 2,093 | 0 |
| x² mod N, base 2, hardest first (`lp_clean_b_src_rc`) | 20 s | 1,761 | 0 |
| x² mod N, base 2, anchored pairs (`ct2a2_clean_b_src_rc`) | 159 s | 4,126 | 6 looks (0.03 h) |

No Modal: $0.

### 5. Outcomes (`verify.py`; `logs/verify_rc1.txt`)

In all three runs, all 5 rounds have the earlier run's picks, library and target search, and the target does not
come.

The statuses the check changed, all from "duplicate" or "alias" to "learnable":
- **10·x² + 1, hardest first:**
  - round 1: dream/CLR.T L.X[M0.4 CLR.U] DEC.A (cost 3);
  - round 2:
    - restrict/remainder[MOV.A.X INC.X → X]/X·X<X+N (cost 5);
    - restrict/remainder[M0.5 T.law5 → A]/N+N<X (cost 5);
    - **offset/remainder/M1.4 → A**: the piece 10X mod N, cost 5;
  - round 3: restrict/remainder[MOV.U.X M1.2 → U]/X·X<X+N (cost 5), and **offset/rrem/T.law5**, the piece again
    (cost 5);
  - round 4: law/lt(N,X)/x+QN (cost 3).
- **x² mod N, base 2, both rules:** law2/mod(X,mul(X,X)), X mod X·X, an alias that is now learnable at cost 3. It
  is freed in rounds 2–4 under hardest first and rounds 3–4 under anchored pairs. Its program has a primitive's
  signature but is wrong on its rows.
- **Not changed: x mod N on long x** (law/mod(X,N)/rows). It stays a duplicate of law/mod(X,N)/x+QN in every round:
  their best programs are the same program.
- **The earlier wins:**
  - in the nine held-out wins and anchored pairs' binary x mod N run, no status changed;
  - in hardest first's binary x mod N run, the same alias was freed in rounds 2–3;
  - every win comes in the same round with the same program.

### 6. Predictions against outcomes

| prediction (PREREG.md) | confidence | outcome |
|---|---|---|
| 10·x² + 1, hardest first: no | medium | **held** |
| the piece learnable at cost 5 in rounds 2–3, not picked | medium | **held** |
| x² mod N, base 2, hardest first: no; x mod N on long x stays a duplicate | high | **held** |
| x² mod N, base 2, anchored pairs: no | high | **held** |
| some status changes in at least one of the 14 runs | high | **held**: 14 problem-rounds in 4 runs |
| picks change in at least one round of at least one run | medium | **failed**: none did |
| the 11 earlier wins: same round, same program | medium | **held** |

### 7. Dead ends, loose ends, and what would unlock them

- **10·x² + 1 under hardest first** needs its piece picked. Hardest first breaks ties by pool order, and the offsets
  come last. SUBGOAL.md §8's own-subgoal score singles 10X mod N out "first and alone" when it scores the rest of the
  way's candidates. Aiming the ties at that would test it. By round 4 the 1M-state probe no longer reaches cost 5; that
  limit is the same for every source.
- **x² mod N under hardest first** needs x mod N on long x practised. Its twin is a true copy, and cheap, so hardest
  first never takes it. Letting the subgoal's priority pass to its twin, and practising it when learnable (the
  subgoal study's `_subpick`), would give the source its state.
- **x² mod N under anchored pairs** needs a pick score that sees the target's acceptance. Its looks score how short
  the rows become, and counting programs fit every row.
- **The search's own item classes use the same signature.** The engine's enumeration (bar3/scale/engine.
  enumerate_items) groups items by what they do to random states too, so items that die on every state fall into one
  class there as well. That was not touched here.

### 8. Files and how to rerun

Python with numba: the scratchpad venv that ran the digits study (`nvenv`). The plain one:
`~/github/sotaku/venv/bin/python`.

- **The change:**
  - `rowcheck.py`: the check;
  - `make_learners.py` writes `learn_newton.py` and `learn_digits.py`;
  - `configs.py`.
- **Replays and runs:**
  - `check.py`: replays, `off` and `on`;
  - `run.py`: a run from the stores;
  - `memcap.sh`: the memory cap for local runs.
- **Results:** `verify.py`.
- **Records:** `PREREG.md` (read-only); `records/*_rc1.jsonl`; `logs/`: every output above, and the anchored-pairs
  run's own store.

The earlier stores are not in the repository:
- the held-out test's: Modal Volume `bar3-heldout`, v1/, `store_ctl_<rule>_ho1.jsonl`;
- the subgoal study's: Volume `bar3-subgoal`, v1/, four files;
- the digits study's base-2 ones: Volume `bar3-digits`, v1/, `store_<rule>_clean_b_src_dxmod_dg1.jsonl`;
- the two-source study's hardest-first store: `bar3/scale/newton/logs/`.

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python bar3/scale/rowcheck/make_learners.py
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.rowcheck.check off [--digits] STORES -- RECORDS
```

```bash
cd cpu_first && <python with numba> -m bar3.scale.rowcheck.check on [--digits] STORES -- RECORDS
```

```bash
cd cpu_first && bar3/scale/rowcheck/memcap.sh 10 <python with numba> -m bar3.scale.rowcheck.run lp_clean_deep_sub_ns_rc tenrho rc1 STORES
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.scale.rowcheck.verify
```

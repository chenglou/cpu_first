# The five-piece file, fast: the same climb in less time

*Written 2026-10-25/26. Folder `cpu_first/integration/combined_fast/`. The input is round 2's file,
`integration/combined_sp2/submission_stage3_combined2.py`, read from its committed bytes (commit 914952f, md5
`4e0e3f09`, asserted); nothing in `combined_sp2/` was written.*

## In plain words

<!-- PLAIN -->
**What was asked.** Round 2's file has all five certified variants of the self-practising file and finds its program,
but too late for the hour: in the tiny core the round came at 3,206 s of climb, about 3,400-3,700 s on the official
H100. The brief: make it fit by pure engineering, changing nothing it learns, then certify it.

**Where the seconds went.** A profile of the climb on a B200 (§1) shows almost all of it in two kinds of overhead, not
in arithmetic: the program machine ran one instruction at a time for every row of a search, about 60 small GPU kernels
an instruction (62% of the climb), or one instruction at a time per program (19%); and listing each goal state's program
read the GPU several times per state and level (9%).

**What changed** (§2), in one new file, [`submission_stage3_fast.py`](submission_stage3_fast.py), built from round 2's
file by exact, asserted edits ([`make_fast.py`](make_fast.py)):
1. *One kernel per lockstep.* A Triton kernel runs every row's whole instruction sequence in a single launch, doing each
   step exactly as the old machine did (integers only). Triton ships with the evaluator's torch.
2. *The item-by-item rows go through it too,* and their machine work is counted exactly as the item-by-item path counted
   it, step by step, the same floats. So the work, and every budget read from it, is unchanged.
3. *All goal states listed in one walk* on the GPU instead of one at a time.
4. *Room for the new code* (the file had 390 bytes to spare): code the file can never run with its settings (three
   other choosing rules, the tiny core's oracle arm, an unused listing mode, a search order it does not use) and the
   CUDA-graph path the kernel replaces. 258,310 bytes, 3,834 under the limit.

**Nothing learned changed** (§3, §4). The new tiny core gives round 2's record unit for unit: the same 66 log lines, the
same 47 searches (roots, programs found, level sizes, stops), every unit's machine work as the same float bit for bit,
the same history, library, rounds and answer program. Underneath: the kernel equals the old machine byte for byte on
random programs of every instruction and on a real 10,751-item vocabulary, at every size the climb uses, in the tiny
core's environment and in the evaluator's own (torch 2.12.1, Triton 3.7.1, H100); the work count equals the old one
bit for bit; combined_sp2's local checks give round 2's results.

**How much faster.** In the tiny core on a B200 the climb reaches the round at **416 s instead of 3,206 s**
(7.7 times faster); level 4, round 2's cost, takes 204 s instead of 2,499 s, and the answer program's search 7 s
instead of 102 s. The same tiny core on an H100 gives the same climb and reaches the round at 459 s.

**It certifies** (§5). The official run (one H100, the manifest untouched): T = 64 on both ladders, every split exact
(test 9,999 of 9,999, unseen T and unseen moduli 10,002 of 10,002 each); `tools/verify_records.py` checks it out.
The round came at about 460 s of training, the answer program at about 480 s, the freeze at about 525 s; training
ended at the step cap at 751 s of the 3,600. Its climb is round 2's own on the evaluator line for line: the same 72
climb lines as round 2's file run with a two-hour budget (combined_sp2's relaxed run, same seed), where the round came
at about 3,792 s.
<!-- /PLAIN -->

## 1. Where the climb's seconds went

[`prof_c.py`](prof_c.py) is the tiny core ([`core_c.py`](core_c.py), combined_sp2's, copied byte for byte: md5
`2e9c6a52`) with timers: every function named in its list is wrapped, its calls and its time with and without the
timed functions it calls, by call path, and the lockstep by rows and steps; with `SYNC=1` each timed function waits for
the GPU at its entry and exit, so a kernel's time counts where it was launched. It ran round 2's measuring build
(budgets 0, md5 `41077007`, the build of `records/core_combined2_measure`) on one B200 from the climb's start through
level 4's first deep probe, 1,761 s (`records/prof_base_sync_prof.json`; a run without the syncs, through level 3,
`prof_base_nosync_prof.json`, kept round 2's unit times to within 10% and gave the same work counts).

| where | seconds | share |
|---|---|---|
| the lockstep, replayed from CUDA graphs (`_graph_steps`: long sequences, a graph captured anew each call) | 845 | 48% |
| the lockstep, step by step (`_lockstep`'s own steps, short sequences; `_columns`) | 243 | 14% |
| item by item (`_apply_grouped`'s `run_items`, a few kernels per primitive: 4.7 million `step`s) | 342 | 19% |
| the listing of goal states' programs (`programs`, one GPU walk per state: 471,504 of them) | 167 | 9% |
| the searches' bookkeeping (`bfs_roots`, `apply_items`, goal tests, keys, acceptance setup) | 156 | 9% |

By task: the probes' searches proper 1,599 s, their acceptance checks 172 s, the new vocabulary's signatures each level
125 s, the round's searches 11 s. Level 4 is where it adds up: its first probe group spent 112 s in graph replays, 99 s
of it building the new routines' vocabulary.

The lockstep's cost is launches, not arithmetic: one step was about 60 kernels (`_ustep`), so a sequence of a few
thousand instructions on a few hundred rows (an acceptance check) took seconds, and a single instruction on 8 million
rows 14 ms. The graph path saves the launches but not the kernels: a replayed step still costs about 60 tiny kernels
of GPU time.

## 2. The changes

`make_fast.py` reads round 2's file from git (`914952f`), asserts its md5, and applies four groups of edits in order,
each asserted to match exactly once (a cut between two anchors is pinned by the md5 of the text it removes). It then
checks that the file parses, compiles, reads no undefined global and passes the contest's source policy at 256 KiB.
`NAME=VALUE` writes a research build to `research/`.

| group | edits | what |
|---|---|---|
| listing | 2 | `first_programs`: every goal state's first parent, level by level, all states at once on the GPU; `bfs_roots` lists with it |
| fused | 4 | the Triton import (none: the old path), `_Ops.table`, `_lock_fused` and the kernel `_lock_k`, the dispatch in `_lockstep` |
| grouped | 4 | `_fused`, `_steps_of`, `_grouped_work`; `apply_items` sends the buckets the cost model gives to `_apply_grouped` to the lockstep too, counting their work as `_apply_grouped` would |
| room | 23 | the dead code below |

**The kernel** (`_lock_k`). One program handles a few rows for the whole sequence: per step it reads each row's
instruction, gathers the row's places by the instruction's move (the same table, `_Ops.rm`), reads the cell's inputs at
their places after the move, looks up the cell (`Cells.alu`), writes the place at the pointer, the flags, the shift
through k, the C blocks' off counts and the routines' pointer stack, all as `_ustep` computes them, then stores the
row in place (barriers order a step's reads before its writes). Integers only, so the state is the same, not close.
Each instruction's flags and numbers come from one int32 table (`_Ops.table`). The probe count is a compile-time
constant, so a row's places load 8 bytes at a time. Where there is no GPU or no Triton, or `LOCK_FUSED` is False, the
old `_ustep` runs step by step (the graph path is gone; it was only ever a speedup of that).

**The item-by-item buckets.** `apply_items` gives a bucket of short sequences to `_apply_grouped` when its cost model
says few distinct items share the steps; that path counted its work step by step (`PRIM_STEP + rows PRIM_ROW` per
primitive, per group). With the kernel those rows run in the lockstep (apply_items gives run_items' result either way)
and `_grouped_work` adds to `WORK` exactly what `_apply_grouped` would have: each group's increments in its order, the
same Python floats added one at a time. The cost model and the lockstep buckets' work are untouched.

**The listing.** `bfs_roots` listed each goal state's cheapest program with `programs(..., limit=1)`: a recursive walk
with a device lookup and several host reads per level, per state. `first_programs` takes every state's first parent at
once, level by level: the same programs in the same order. The work budget was read every 256 states; the listing does
no machine work, so it is read once.

**Room.** With rule `lpca`, `deep_by` "bindings", `calls_first` True and no small target fixed by the file's constants,
these never run, and are cut: the rules `lp`, `all`, `ct2a2`, `lpu` (`look_jobs`, `look`, `mdl`, `unlockers`, their
phases and scores), the tiny core's small target (`target`, `_near`, `near_log` and its branches), `deep_roots`' other
modes, `bfs`'s "goals" listing, the round's search by calls after the flat one, and the CUDA-graph path. A research
build that asks for any of them now raises instead of running it.

## 3. Checks

| check | where | result |
|---|---|---|
| [`checks/check_fused.py`](checks/check_fused.py), random | B200, torch 2.14.1, Triton 3.8.0 | 300 random sequences of every instruction (primitives, NOP, routine saves and restores at every depth, C blocks) on random machines, P 2-16, 1-60 probes, up to 4,000 rows and 700 steps: the state after `_lockstep` the same byte for byte with the kernel and with `_ustep` |
| check_fused.py, vocabulary | the same | `apply_items` on 10,751 items (`enumerate_items` over the primitives and the round-2 tiny core's 53 routines, bodies up to cost 2), the same: the final kernel at W 1-4, 8 (2,000 items, 60 probes) and the signatures' 6 probes; the first kernel at W 1-4 (stopped in W 8, where the reference ran slowly) |
| check_fused.py | the evaluator's environment (`modal_fast.py`'s `official_image`: torch 2.12.1, Triton 3.7.1, H100) | the final kernel: the 300 random cases and the vocabulary at W 1, 2, 4, 8 (1,000 items) and the signatures' 6 probes, the same; the first kernel: random and W 1, 2, 4 |
| [`checks/speed_fused.py`](checks/speed_fused.py) | B200 and H100 | both kernels and `_ustep` give the same state on 7 climb-shaped workloads (below) |
| [`checks/check_grouped_work.py`](checks/check_grouped_work.py) | the laptop's CPU | `_grouped_work`'s `WORK` equals `_apply_grouped`'s as a float, bit for bit, on 200 item lists (loops, C blocks, routines, ONE bodies; 0-7 rows each) from three starting values up to 3.1e9 |
| [`checks/check_listing.py`](checks/check_listing.py) | the laptop's CPU, small budgets | every call of `first_programs` in a climb's first units against the per-state walk: 78 calls, 1,524 goal states, the same programs |
| combined_sp2's `checks/local_combined.py` (its working copy, md5 `6f98f631`: round 3's wording, the same checks for this instruction set) | the laptop's CPU, 319 s, peak 3.2 GiB (`checks/logs/local_fast.log`) | every part as in round 2's `local2.log`: policy (258,310 bytes), machines, isa, designed, buffers, grad (14 of 15 tensors), fix, book (12,382 targets), the tiny-budget climb (the same work, 1.298e6) and the replay (the same losses, frozen at step 21 with the same four answer programs and weights). On a CPU the kernel never runs, so this checks the rest of the file |

The kernel's speed on climb-shaped workloads (`records/speed_v1v2_b200.log`, B200; seconds a call after the first,
which compiles; the H100's within 25%, `speed_v1v2_h100.log`):

| rows x steps (places P, probes; C blocks) | first kernel | final kernel |
|---|---|---|
| 8.4M x 1 (4, 8) | 0.024 | 0.0041 |
| 4.2M x 16 (4, 8; C) | 0.157 | 0.020 |
| 131k x 2,048 (4, 8) | 0.632 | 0.065 |
| 2.1M x 100 (16, 8; C: the round's) | 0.956 | 1.005 |
| 262k x 2,048 (4, 6; C: the signatures') | 1.255 | 0.474 |
| 256 x 4,000 (8, 60; C: acceptance) | 0.021 | 0.0082 |
| 64 x 16,000 (16, 60: full-size acceptance) | 0.079 | 0.029 |

Against `_ustep` step by step (`check_fused2.log`, B200; round 2's graph path is no longer in these files): 4.2M rows x
1 step 14 ms against 2.3 ms; 131k x 2,048 1.14 s against 0.066 s; 256 x 4,000 at 60 probes 1.8 s against 7 ms; 64 x
16,000 at 16 places 7.0 s against 29 ms.

## 4. The tiny core

`core_c.py` (combined_sp2's, byte for byte) on one B200 through [`modal_fast.py`](modal_fast.py) (app and Volume
`combined-fast`; the image is combined_sp2's plus gcc, which Triton needs to build its launcher), on the research build
with both budgets 0 (`research/submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py`), as round 2's record
`core_combined2_measure` ran its own. [`checks/compare_more.py`](checks/compare_more.py) adds what
`compare_cores.py` masks: the work as floats, history, library, rounds, routines by law, the answer program.

**The final file's climb is round 2's, unit for unit** (`records/core_fast_v2_measure.*`, md5 of the build
`39e00753`; `compare_cores.py` and `compare_more.py` against `combined_sp2/records/core_combined2_measure.json`):

```
lines: the first 66 units are the same (of 66 and 66)
searches: the first 47 are the same (of 47 and 47)
units: 66 and 66; work and running total equal as floats on 66 of 66
total work: 2651201081.3771276 and 2651201081.3771276: equal
history: equal
lib: equal
rounds: equal
ops: equal
outer (but its seconds and work): equal; work 117027282.40724993 and 117027282.40724993; wall 102.0 s and 7.1 s
seconds: 3321.3 and 436.8 (7.60 times)
```

The round is the same, `LDA.X STA.Y LDA.N T.add.mul.o,x,o.AXY STA.Y T.red.x+on.YN` and three like it, and so is
the answer program: `C.S[T.round1 STA.X]` and `C.S[T.round3 STA.X]`, right on all 224 unseen rows (T up to 64).

| | round 2 (`core_combined2_measure`) | first kernel (`core_fast_v1_measure`) | final, B200 (`core_fast_v2_measure`) | final, H100 (`core_fast_v2_h100`) |
|---|---|---|---|---|
| level 0 | 51 s | 24 s | 24 s | 26 s |
| level 1 | 111 s | 65 s | 42 s | 46 s |
| level 2 | 104 s | 48 s | 32 s | 37 s |
| level 3 | 440 s | 201 s | 113 s | 124 s |
| level 4's 11 probe groups | 974 s | 388 s | 117 s | 134 s |
| level 4's 6 deep-probe groups | 1,504 s | 263 s | 69 s | 76 s |
| **the round** | **3,206 s** | **1,003 s** | **416 s** | **459 s** |
| the answer program's search | 102 s | 7.0 s | 7.1 s | 7.6 s |
| in all | 3,321 s | 1,021 s | 437 s | 481 s |

All four climbs are the same: the first kernel's compares with round 2's, and the H100's with the final B200's, as
above, every line, search and work float equal. The H100 tiny core ran the same image (torch 2.14.1) as the B200's;
the climb took it 1.10 times as long. The rounds' own searches (36 s in all levels) did not get faster: their machine
has 16 places at 8 probes on up to 2 million rows, the one shape the final kernel does not speed up (§3).

**Projected before the official run.** Earlier pairs put the official run's climb at 1.05-1.15 times the tiny core's
(COMBINED.md §8), measured when the climb was bound by launching kernels on the host. Here it is bound by the GPU's
work, and the H100's tiny core took 1.10 times the B200's. So the round should come about 460-530 s into the climb, the
climb starting about 60 s into training: by roughly 600 s of the hour, and the answer program soon after.

## 5. The official run

`records/combined_fast_official.json`: the file byte for byte (md5 `91bb30b9`), the untouched official manifest, one
H100, through the shared runner (`--memory-gb 48`, as round 1's run). **It certifies**: T = 64 on both ladders, every
rung 768 of 768; test 9,999 of 9,999, unseen T 10,002 of 10,002, unseen moduli 10,002 of 10,002.
`tools/verify_records.py`: the embedded source's md5 is the file's, the manifest the official one, return code 0, the
RESULT_JSON line equal to the result; "1 of 1 records check out".

| step | training clock | what |
|---|---|---|
| 563 | about 60 s | the climb starts, every cell exact |
| 563-629 | 389 s of climb units, to about 460 s | levels 0-4; the round, `LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+xn.XN` and three like it |
| 630 | about 480 s | the answer program's search (19.7 s): `C.S[T.round1 STA.X]`, `C.S[T.round2 STA.X]`, `C.S[T.round3 STA.X]` |
| 829 | about 525 s | frozen (phase D's 200 exact steps) |
| 20,000 | 751 s | training ends at the step cap |

Evaluation 362 s; peak host memory 13.8 GB of the 48 GB cap.

**Round 2's own climb, on the evaluator, step for step.** combined_sp2's round 3 ran round 2's file (`4e0e3f09`) in
the same evaluator, seed 74, with a 7,200 s budget (`combined_sp2/records/combined2_h100_relaxed7200.json`, a research
run). [`checks/compare_official.py`](checks/compare_official.py) finds its 72 climb lines (`LR climb`, `LR outer`,
`LR round_found`, `LR frozen`) and this run's the same, seconds masked: the round at step 629, the answer program at
step 630, the freeze at step 829. There the round came at about 3,792 s of training and the freeze at about 4,131 s;
here at about 460 s and 525 s, 8 times sooner. Both runs end the answer program's search at 3.254e9 units of work,
0.1% past `CLIMB_WORK` (3.25e9): its last batch of acceptance, the one that holds the programs, crossed the budget
(combined_sp2's note on the relaxed run). The work is counted as before, so that is round 2's behaviour too, and
harmless here.

The official climb took 389 s of units to the round, 0.85 times the H100 tiny core's 459 s (its rows differ from the
tiny core's, so its searches do too).

## 6. Cost, dead ends, how hard it was tried

**Money**, at Modal's list prices (B200 $6.25 an hour, H100 $3.95): about $19 of the soft $40 (the billing report had
not yet billed the last hour).

| what | hardware, wall | about |
|---|---|---|
| the two profiles of round 2's build (with and without syncs) | B200, 31 + 13 min | $4.6 |
| the first kernel: its byte-for-byte check, its tiny core, its profile | B200, 21 + 18 + 8 min | $4.9 |
| the final kernel: its check, the speed comparison (both GPUs), its tiny cores | B200 31 + 3 + 8 min, H100 3 + 9 min | $5.2 |
| the checks in the evaluator's environment (both kernels) | H100, 12 + 30 min | $2.8 |
| the official run | H100, 19 min | $1.26 |

`integration/ablation_s3/spend.py 4750` ran before every launch: $3,522.55 billed this month before the first,
$3,542.28 before the official run (the billing report counts complete hours only). Everything else ran on the laptop's
CPU under combined_sp2's `checks/capped.py`.

**How hard it was tried.** One profile (and its no-sync twin), four groups of edits, two versions of the kernel, three
tiny cores of the fast file (first kernel on a B200, final kernel on a B200 and on an H100), one official run. No seeds
(they need the coordinator's go-ahead).

**Dead ends and corrections.**
- *The first kernel* (kept in the records as `core_fast_v1_measure`, `check_fused1`, `check_fused_official_env`,
  `prof_fast_sync`; its source is not kept: the builder now writes the final one) read the cell's inputs by summing over
  every place of the row, byte by byte, with the probe count unknown at compile time. It was right (the same checks) and
  gave the tiny core 3.25 times, but on many rows it was no faster than the old machine, sometimes slower (8.4M rows,
  one step: 25 ms; the old machine's step on 4-8M rows took about 14 ms in §1's profile). Reading the inputs at their
  places after the move, with the probe count a compile-time constant (8-byte loads) and 32-bit index arithmetic, made
  it 5-10 times faster on large row counts and 2.7 times on small ones. Tried once each.
- *The round's shape* (16 places, 8 probes, 2 million rows, 100 steps) takes 1 s with either kernel, 5 ns a row-step,
  probably the 128-wide padded tile; not pursued (36 s of the climb).
- *The profile with syncs ran faster* than round 2's record and than the run without them (1,761 s against 1,962 s to
  the same unit): the same work counts, so it is the hosts or the syncs, not the climb. Not followed up.
- *The CUDA-graph path is gone* (it paid for the room). On a GPU without Triton the old machine now runs step by step,
  slower than round 2's graphs; the evaluator's environment has Triton (§3).
- *A local listing check at a 60k budget* hit its 1 GiB cap and was killed; the check had passed at a 150k budget
  (1.7 GiB, 114 s) on the first build, whose listing code is the final one's.

**What this leaves.** The climb now takes about 420 s on a B200. What is left is mostly the searches' bookkeeping on
the host (in the first kernel's profile, `bfs_roots` and `apply_items` took 83 s of the 406 s through level 4's first
probe) and the round's 16-place shape. Neither is needed for the hour.

## 7. Files and how to rerun

Everything is in `cpu_first/integration/combined_fast/`; nothing outside it was written.

| file | what |
|---|---|
| `submission_stage3_fast.py` | the file, md5 `91bb30b9d131791d165025d682b3f458`, 258,310 bytes (3,834 under the limit), written by `make_fast.py` |
| `make_fast.py` | the builder (§2): round 2's file from git at `914952f`, md5 asserted, then the listing, fused, grouped and room edits |
| `research/` | `submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py` (md5 `39e00753`, the tiny cores' build) and round 2's own measuring build (`41077007`, copied from git, for the profile) |
| `core_c.py` | combined_sp2's tiny core, copied byte for byte (md5 `2e9c6a52`) |
| `prof_c.py` | the profile (§1): core_c.py with timers |
| `modal_fast.py` | the Modal launcher (app and Volume `combined-fast`): B200, H100, and an H100 in the evaluator's own environment for checks |
| `checks/` | `check_fused.py`, `speed_fused.py`, `check_grouped_work.py`, `check_listing.py`, `compare_more.py`, `compare_official.py`; `logs/local_fast.log` (combined_sp2's local checks on this file) |
| `records/` | the profiles (`prof_base_*`, `prof_fast_sync*`), the tiny cores (`core_fast_v1_measure`, `core_fast_v2_measure`, `core_fast_v2_h100`), the checks' logs, the official record (`combined_fast_official.json`) |

From `cpu_first/`, with the sotaku venv's Python:

```bash
~/github/sotaku/venv/bin/python integration/combined_fast/make_fast.py
```

```bash
~/github/sotaku/venv/bin/python integration/combined_fast/make_fast.py SEARCH_WORK=0 CLIMB_WORK=0
```

```bash
~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/combined_fast/modal_fast.py --envs '[{"TAG": "core_fast_v2_measure", "SCRIPT": "core_c.py", "SECONDS": "3900", "FILE": "/root/cpu_first/integration/combined_fast/research/submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py"}]'
```

```bash
~/github/sotaku/venv/bin/python -m modal run integration/combined_fast/modal_fast.py --mode fetch
```

```bash
python3 integration/combined_sp2/checks/compare_cores.py integration/combined_sp2/records/core_combined2_measure.json integration/combined_fast/records/core_fast_v2_measure.json
```

```bash
python3 integration/combined_fast/checks/compare_more.py integration/combined_sp2/records/core_combined2_measure.json integration/combined_fast/records/core_fast_v2_measure.json
```

The profile of round 2's build (add `"SYNC": "0"` for the run without syncs):

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/combined_fast/modal_fast.py --envs '[{"TAG": "prof_base_sync", "SCRIPT": "prof_c.py", "SYNC": "1", "STOP": "^L4 deep \\[", "SECONDS": "4200", "FILE": "/root/cpu_first/integration/combined_fast/research/submission_stage3_combined2_SEARCH_WORK-0_CLIMB_WORK-0.py"}]'
```

The kernel's checks, in the tiny core's environment and in the evaluator's:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/combined_fast/modal_fast.py --envs '[{"TAG": "check_fused", "SCRIPT": "checks/check_fused.py"}, {"TAG": "check_fused_official_env", "SCRIPT": "checks/check_fused.py", "GPU": "h100_official_env", "VOCAB": "1,8,2;2,8,2;4,60,1;8,60,1,1000;2,6,24"}]'
```

On the laptop:

```bash
PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_sp2/checks/capped.py 150 1 -- ~/github/sotaku/venv/bin/python integration/combined_fast/checks/check_grouped_work.py
```

```bash
COMBINED_FILE=integration/combined_fast/submission_stage3_fast.py PYTHONPATH=. ~/github/sotaku/venv/bin/python integration/combined_sp2/checks/capped.py 1500 6 -- ~/github/sotaku/venv/bin/python integration/combined_sp2/checks/local_combined.py policy machines isa designed buffers grad fix book climb_tiny replay
```

The official run, its record and its check:

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/combined_fast/submission_stage3_fast.py --memory-gb 48 --name combined_fast_official --out integration/combined_fast/records --run combined-fast-official
```

```bash
~/github/sotaku/venv/bin/python tools/modal_runner.py --mode fetch --run combined-fast-official --out integration/combined_fast/records --wait
```

```bash
python3 tools/verify_records.py integration/combined_fast/records/combined_fast_official.json=integration/combined_fast/submission_stage3_fast.py
```

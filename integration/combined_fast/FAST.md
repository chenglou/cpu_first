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

**Seeds 1-7 certify too** (§8): eight of eight seeds on the official H100, the manifest changed only in its seed, all
checked by `tools/verify_records.py`. Every seed finds its answer program by 471-504 s of training and ends training
by 838 s. The thin margin was the climb's work budget, round 2's: on seeds 1, 4 and 6 the climb's count passed it in
the answer program's last search level, where it happened to cut nothing (§9).

**A budget with margin, certified** (§9): [`submission_stage3_fast2.py`](submission_stage3_fast2.py) (md5 `41b9a013`)
is the fast file with its two work budgets set again by STRICT.md §2's rule, from the eight climbs measured with no
budget on the official manifest: `CLIMB_WORK` 4.3e9 (1.22 times the largest climb, 3.49e9) and `SEARCH_WORK` 8e8
(twice the largest search). Three lines change. Its climb is unchanged (the tiny core unit for unit, and each official
seed's climb line for line its budget-free run's), the largest climb uses 81% of the budget, and it certifies on all
eight seeds, the official run included, training in 759-883 s.
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
| `submission_stage3_fast2.py` | the same with the budgets of §9, md5 `41b9a013ae9436d7b7a53e9e594a18db`, 258,349 bytes, written by `make_fast.py --fast2` |
| `make_fast.py` | the builder (§2): round 2's file from git at `914952f`, md5 asserted, then the listing, fused, grouped and room edits |
| `research/` | `submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py` (md5 `39e00753`, the tiny cores' build) and round 2's own measuring build (`41077007`, copied from git, for the profile) |
| `core_c.py` | combined_sp2's tiny core, copied byte for byte (md5 `2e9c6a52`) |
| `prof_c.py` | the profile (§1): core_c.py with timers |
| `modal_fast.py` | the Modal launcher (app and Volume `combined-fast`): B200, H100, and an H100 in the evaluator's own environment for checks |
| `checks/` | `check_fused.py`, `speed_fused.py`, `check_grouped_work.py`, `check_listing.py`, `compare_more.py`, `compare_official.py`, `summarize_seeds.py`, `measure_work.py`; `logs/` (combined_sp2's local checks on the fast file, and §9's logs) |
| `records/` | the profiles (`prof_base_*`, `prof_fast_sync*`), the tiny cores (`core_fast_v1_measure`, `core_fast_v2_measure`, `core_fast_v2_h100`), the checks' logs, the official record (`combined_fast_official.json`) and seeds 1-7's (`combined_fast_h100_seed<N>.json`, §8); the second file's (`core_fast2`, `combined_fast2_official`, `combined_fast2_h100_seed<N>`, §9); `research/`: the budget-free runs (§9) |

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

## 8. Seeds 1-7

*Added 2026-10-26, on the coordinator's go-ahead (after commit 5f78072).* The file byte for byte (`91bb30b9`), once for
each of seeds 1 to 7, the official manifest with only `runtime.seeds` changed, one H100 each, all seven at once,
through the shared runner as s3i's seven H100 seeds were run (INTEGRATION.md: `--mode seeds --hardware h100`); the
memory cap 48 GB, as this file's official run (s3i used the default 24 GB; these peaked at 14.7 GB). Records
`records/combined_fast_h100_seed<N>.json`.

**Every seed certifies.** `tools/verify_records.py` on the seven: "7 of 7 records check out", each embedding the
file's md5, its manifest the official one but the seed, return code 0, the RESULT_JSON line equal to the result, T =
64 on both ladders with every rung 768 of 768, and test 9,999 of 9,999, unseen T 10,002 of 10,002, unseen moduli 10,002
of 10,002. With seed 74's official run, eight of eight.

[`checks/summarize_seeds.py`](checks/summarize_seeds.py) reads each record's climb (the clock at a step is the
evaluator's progress line before it plus the climb's own unit seconds, so "about"):

| seed | climb starts | the answer program | frozen | training | evaluation | work at the round | work after the answer search |
|---|---|---|---|---|---|---|---|
| 74 (official) | step 563, about 54 s | step 630, about 476 s | step 829, about 520 s | 751 s | 362 s | 2.512e9 | 3.254e9 |
| 1 | 584, 45 s | 653, 502 s | 852, 542 s | 733 s | 321 s | 2.686e9 | 3.332e9 (budget, its level 3) |
| 2 | 584, 47 s | 653, 491 s | 852, 620 s | 838 s | 346 s | 2.747e9 | 2.864e9 |
| 3 | 612, 57 s | 681, 480 s | 880, 602 s | 803 s | 321 s | 2.678e9 | 3.057e9 |
| 4 | 653, 56 s | 719, 504 s | 918, 555 s | 744 s | 340 s | 2.751e9 | 3.321e9 (budget, its level 3) |
| 5 | 609, 56 s | 677, 471 s | 876, 593 s | 796 s | 304 s | 2.643e9 | 2.772e9 |
| 6 | 623, 57 s | 691, 503 s | 890, 540 s | 763 s | 350 s | 2.608e9 | 3.492e9 (budget, its level 3) |
| 7 | 565, 54 s | 632, 498 s | 831, 544 s | 770 s | 376 s | 2.652e9 | 2.900e9 |

On every seed the climb finds its round at level 4, the generated multiply-accumulate then a reduction
(`LDA.X STA.Y CLA T.add.mul.o,x,o.AXY STA.X T.red.x+on.XN` and others like it, on x + o·N or x + x·N), and the answer
program is `C.S[T.round<i> STA.X]` over three or four of its rounds (seed 4 also keeps `C.S[T.round3 T.eq.pair.AX]`).
The answer program comes by 471-504 s of training and the freeze by 520-620 s; training ends at the step cap by 733-838
s, under a quarter of the hour.

**The work budget is the thin margin, and it is round 2's.** The climb reaches its round with 2.51-2.75e9 units of
`CLIMB_WORK`'s 3.25e9 (77-85%). The answer program's search then costs 1.2e8 to 8.8e8, far more spread than the tiny
core's 1.2e8, and on seeds 1, 4 and 6 the climb's count passed the budget inside that search's last level (cost 3; its
record's stop reads "work budget at level 3"; the totals 3.33e9, 3.32e9 and 3.49e9). The budget is read only between
blocks of a level, and it was passed in the last one, so it cut nothing: the same seeds with no budget do the same work
and find the same level sizes and programs (§9). A search that passed it one block sooner would have lost the rest of
its level. The speedups change no work count, so round 2's file meets the budget at exactly the same points; it was
set in round 2 from one tiny-core climb (1.22 times its 2.651e9, COMBINED.md §8). §9 sets it again, by the same rule,
from these eight climbs.

*(Corrected after §9's runs: this paragraph first said the budget cut that level short after its goal states were
found; the runs with no budget show it cut nothing.)*

**Cost.** The seven runs' list-price estimate is $8.84 (walls 1,059-1,189 s; $4.73 billed so far, complete hours
only), of the soft $15. `spend.py 4750` read $3,542.28 before the launch.

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/combined_fast/submission_stage3_fast.py --seeds 1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name combined_fast --out integration/combined_fast/records --run combined-fast-h100-seeds
```

```bash
~/github/sotaku/venv/bin/python tools/modal_runner.py --mode fetch --run combined-fast-h100-seeds --out integration/combined_fast/records --wait
```

```bash
python3 tools/verify_records.py integration/combined_fast/records/combined_fast_h100_seed{1,2,3,4,5,6,7}.json=integration/combined_fast/submission_stage3_fast.py
```


```bash
python3 integration/combined_fast/checks/summarize_seeds.py integration/combined_fast/records/combined_fast_official.json integration/combined_fast/records/combined_fast_h100_seed{1,2,3,4,5,6,7}.json
```

## 9. A budget with margin: `submission_stage3_fast2.py`

*Added 2026-10-26, on the coordinator's decision (after commit a2e4bac): set the two work budgets again by STRICT.md
§2's rule, from the climbs on the official manifest, in a new file. The fast file and its records stay as they are.*

**The true work, measured with no budget.** The fast file's budget-free build
(`research/submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py`, md5 `39e00753`, the tiny cores' build) ran on the
official manifest for seeds 74 and 1-7, one H100 each, as research runs
(`records/research/fast_budget0_h100_seed<N>.json`; all eight certify too). Each climb is its certified run's line for
line ([`checks/compare_official.py`](checks/compare_official.py), seconds masked; `checks/logs/compare_budget0.log`):
every line on seeds 74, 2, 3, 5 and 7; on seeds 1, 4 and 6 every line but the answer program's search, which differs
only in its stop ("work budget at level 3" there, none here), with the same level sizes, programs and work. So the old
budget cut nothing, and the counts below are the true ones ([`checks/measure_work.py`](checks/measure_work.py),
`checks/logs/measure_work.log`):

| seed | work at the round | the answer program's search | the climb in all | largest search (a probe unit) | its largest run |
|---|---|---|---|---|---|
| 74 | 2.512e9 | 7.43e8 | 3.254e9 | 3.19e8 | 1.17e8 |
| 1 | 2.686e9 | 6.46e8 | 3.332e9 | 3.07e8 | 2.14e8 |
| 2 | 2.747e9 | 1.17e8 | 2.864e9 | 3.25e8 | 1.45e8 |
| 3 | 2.678e9 | 3.79e8 | 3.057e9 | **3.91e8** | 1.34e8 |
| 4 | 2.751e9 | 5.70e8 | 3.321e9 | 3.39e8 | 1.50e8 |
| 5 | 2.643e9 | 1.29e8 | 2.772e9 | 3.84e8 | 1.46e8 |
| 6 | 2.608e9 | 8.83e8 | **3.492e9** | 3.60e8 | 1.22e8 |
| 7 | 2.652e9 | 2.48e8 | 2.900e9 | 3.85e8 | 1.19e8 |
| the tiny core (B200, §4) | 2.534e9 | 1.17e8 | 2.651e9 | 3.34e8 | 1.27e8 |

**The rule.** As STRICT.md §2 set the budgets and round 2 applied it (COMBINED.md §8):
- `CLIMB_WORK` = 1.22 times the largest climb, the answer program's search in it as the file counts it: 1.22 x 3.492e9
  (seed 6) = 4.26e9, rounded up to **4.3e9** (round 2: 3.25e9).
- `SEARCH_WORK` = twice the largest search, measured as round 2 and STRICT.md measured it, a probe unit's work (its
  log line): 2 x 3.913e8 (seed 3's level-4 deep probe on x + x·N) = 7.83e8, rounded up to **8e8** (round 2: 7e8). A
  unit is one or more runs and the budget bounds each run; the largest run is 2.14e8, so the unit is the safer reading.

**The margin.** The largest climb uses 81% of the new `CLIMB_WORK`. To reach it after the latest round seen (2.75e9), an
answer program's search would have to spend 1.55e9, 1.75 times the most any seed spent. On time (STRICT.md's other
condition: a climb that spends the whole budget still ends inside the hour): the eight research runs climbed at
0.13-0.16 s per million units, so 4.3e9 units is at most about 710 s; with the climb's start (about 70 s) and phase
D (at most 132 s), even a climb that spent it all would freeze by about 910 s.

**No climb budget at all?** Considered, as the coordinator suggested: the evaluator owns the deadline and the strict
build reads no clock, so `CLIMB_WORK` = 0 would remove the cut entirely. Not taken. It changes how a certified piece
ends its climb (at a fixed amount of work, the same on every host, rather than wherever the evaluator's deadline
falls), while the rule's value already leaves every measured climb at least 19% below it and is still far inside the
hour if spent whole. The rule is the established method; that alternative would be a new one.

**The file.** `make_fast.py --fast2` writes [`submission_stage3_fast2.py`](submission_stage3_fast2.py) (md5
`41b9a013ae9436d7b7a53e9e594a18db`, 258,349 bytes, 3,795 under the limit): the fast file with three lines changed,
`SEARCH_WORK = 800_000_000`, `CLIMB_WORK = 4_300_000_000` and the docstring's pointer here. `make_fast.py` alone
still writes the fast file byte for byte (`91bb30b9`).

**The climb is unchanged.** Its tiny core with its own budgets (`records/core_fast2.*`, B200) against the fast file's
budget-free one (`core_fast_v2_measure`): compare_cores "the first 66 units are the same (of 66 and 66)", "the first
47 [searches] are the same"; compare_more every unit's work and the total equal as floats, the history, library,
rounds, routines and answer program equal. Neither budget binds there, nor in any climb measured above.

**It certifies on all eight seeds.** The official run (`records/combined_fast2_official.json`, the manifest untouched)
and seeds 1-7 (`records/combined_fast2_h100_seed<N>.json`, only the seed changed), one H100 each, through the shared
runner as before (`--memory-gb 48`). `tools/verify_records.py`: "8 of 8 records check out"
(`checks/logs/verify_fast2.log`), each embedding the file's md5, T = 64 on both ladders with every rung 768 of 768, test
9,999 of 9,999 and both unseen splits 10,002 of 10,002. Each climb is its budget-free research run's line for line,
every line on every seed (`checks/logs/compare_fast2.log`), with no work-budget stop anywhere: the new budgets bind
nowhere on these seeds.

| seed | the answer program | frozen | training | evaluation | the climb in all, of 4.3e9 |
|---|---|---|---|---|---|
| 74 (official) | step 630, about 492 s | step 829, about 538 s | 759 s | 377 s | 3.254e9 (76%) |
| 1 | 653, 555 s | 852, 600 s | 838 s | 366 s | 3.332e9 (77%) |
| 2 | 653, 499 s | 852, 631 s | 861 s | 334 s | 2.864e9 (67%) |
| 3 | 681, 509 s | 880, 633 s | 883 s | 384 s | 3.057e9 (71%) |
| 4 | 719, 535 s | 918, 586 s | 782 s | 360 s | 3.321e9 (77%) |
| 5 | 677, 513 s | 876, 641 s | 870 s | 395 s | 2.772e9 (64%) |
| 6 | 691, 499 s | 890, 538 s | 781 s | 365 s | 3.492e9 (81%) |
| 7 | 632, 501 s | 831, 549 s | 776 s | 407 s | 2.900e9 (67%) |

**Cost.** List prices: the eight research runs $10.66, the official run $1.29, seeds 1-7 $9.52, the tiny core about
$0.85: about $22 of the soft $25 ($16.41 billed so far, complete hours only). `spend.py 4750` ran before each launch:
$3,558.61 before the first, the last read $3,581.84.

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/combined_fast/research/submission_stage3_fast_SEARCH_WORK-0_CLIMB_WORK-0.py --seeds 74,1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name fast_budget0 --out integration/combined_fast/records/research --run combined-fast-budget0
```

```bash
python3 integration/combined_fast/checks/measure_work.py integration/combined_fast/records/research/fast_budget0_h100_seed{74,1,2,3,4,5,6,7}.json --extra "tiny core=3.342e8,2.651e9"
```

```bash
~/github/sotaku/venv/bin/python integration/combined_fast/make_fast.py --fast2
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach integration/combined_fast/modal_fast.py --envs '[{"TAG": "core_fast2", "SCRIPT": "core_c.py", "SECONDS": "3900", "FILE": "/root/cpu_first/integration/combined_fast/submission_stage3_fast2.py"}]'
```

```bash
python3 integration/combined_sp2/checks/compare_cores.py integration/combined_fast/records/core_fast_v2_measure.json integration/combined_fast/records/core_fast2.json
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/combined_fast/submission_stage3_fast2.py --memory-gb 48 --name combined_fast2_official --out integration/combined_fast/records --run combined-fast2-official
```

```bash
~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/combined_fast/submission_stage3_fast2.py --seeds 1,2,3,4,5,6,7 --hardware h100 --memory-gb 48 --name combined_fast2 --out integration/combined_fast/records --run combined-fast2-h100-seeds
```

```bash
python3 tools/verify_records.py integration/combined_fast/records/combined_fast2_official.json=integration/combined_fast/submission_stage3_fast2.py integration/combined_fast/records/combined_fast2_h100_seed{1,2,3,4,5,6,7}.json=integration/combined_fast/submission_stage3_fast2.py
```

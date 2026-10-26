# The bar-3 search on the GPU: the same programs, faster and a little cheaper

*Written 2026-10-26. Code in `cpu_first/bar3/gpu_search/`, records in `bar3/gpu_search/records/`, logs in
`bar3/gpu_search/logs/`. No file outside that folder changed; nothing committed.*

## In plain words

<!-- PLAIN -->
**The question.** Bar 3's learner ([../scale/SCALE.md](../scale/SCALE.md)) finds short programs on a bare counting
machine by breadth-first search, and every extra instruction multiplies that search by about 10 (by about 40 over a
carried library of routines). SCALE.md compiled the search for the CPU and ran it on Modal's CPU containers. Here the
same search runs on a GPU, under one bar: identical results. Then: how much faster is it, and how much further does
the same money reach?

**What was built.** A GPU backend for the learner's search jobs, behind a flag (`engine.use(True)` points
`bar3/scale/jobs.py` at it, in-process; no existing file changed). A level of the search applies every item to every
state of the level below; here each (item, state) pair runs on the GPU as one thread per probe, in a Triton kernel
that executes the CPU kernel's machine step for step, and a second kernel hashes each new state with the CPU's own
64-bit hash, bit for bit. The level's bookkeeping (which states are new, which reach the goal, which occurrence of
each state comes first, and the parents from which programs are read back) runs on the GPU too, in the CPU's order.
The item enumeration (which items count as distinct) and the checking of found programs stay on the CPU, unchanged.

**The same results.** 18,719 stored searches from the held-out, subgoal and newton studies (with the picks and
rowcheck studies' newton-learner runs) were run again on GPUs: every big search those 55 runs made (12,696
million-state probes, 153 target, 124 guided and 21 deep searches, 123 target probes) and samples of the small ones.
Each record was compared field by field with the stored one: the programs and their order, the level sizes, the best
fits, the stops, the aliases, the near misses. **All identical**, but for the 11 the CPU had cut at its own time
limit, which the GPU finishes. Two whole learner runs (1,522 and 3,498 searches, one of them the newton study's
invention of the triangular numbers) were **reproduced from scratch on the GPU, record for record**. The kernels were
also checked against the CPU's on random programs (nesting 8 deep, dying machines, saturated values) and on real
vocabularies, and SCALE.md's biggest searches (up to 383 million states) came out with the same level sizes.

**How much faster.** Per search, one GPU against one CPU core: the big searches, where the CPU time goes (74% of all
stored search time was in million-state probes), run **about 95 times faster** on an L40S or an H100, and 36–60 times
on an L4, A10G or A100. The larger the level, the larger the gain: none on levels under a thousand states, about 30
times at ten thousand, about 150 times from a million on. SCALE.md's 30-million-state searches took 10.3 and 10.4 s on
an L40S against 1,280 and 2,087 s on a CPU core. Small searches (30,000 states) gain little: they are a few
milliseconds of search beside CPU work (the item enumeration, the learner). The two replayed runs took 77 and 492 s on
one L40S, where their searches alone had taken 3,528 and 23,364 s.

**What the same money buys.** A GPU hour costs 20–40 times a CPU core-hour, so speed is not money. Against the CPU
containers' billed rate ($0.113 a search-hour), the L40S, the cheapest per search of the five GPUs tried, does the big
searches **about 3–5 times cheaper** (3.1 times as the learner ran them, startup and CPU work included; 4.6 times for
the search alone), SCALE.md's 3M–30M-state searches 6–10 times cheaper, and a whole learner run 1.6–1.9 times (it
waits on the CPU a third of the time). Each instruction multiplies the search by about 9.5 over the primitives and 40
over a carried library, so **the same budget buys about 0.5–1.0 more instruction over the primitives and 0.3–0.6
over a library**. Past SCALE.md's deepest search, the next level must also be stored: about 370 million states, some
100 GB, more than one GPU holds.

**What it says.** As SCALE.md found for the compiled search, engineering buys about one instruction, not the 8–22 the
missed formulas need. What the GPU changes is time more than money: a 10-minute search takes 5 seconds and a learner's
round a minute or two, so studies that waited hours for their rounds can run in minutes, at about the same cost.
<!-- /PLAIN -->

## Details

### 1. Where the time goes (the hot path)

`bar3/scale/engine.bfs` (the compiled search) builds level d from every item of cost c applied to every state of level
d − c, in item order, in slices; it drops states that died or that the item left unchanged, hashes the rest, keeps
those not seen at an earlier level, tests each against the goal, and at the level's end keeps one state per new hash
(its first occurrence) and every (item, source) pair that reached it, so that all shortest programs can be read back.
A level that would pass 1.5 times the state budget is streamed: only counted and goal-tested.

Profiled locally on a 1M-state target search (x² mod N, 6 levels, 4.2M states streamed): 86% in the numba kernel
(`_apply_many`: the machine and the hash), the rest in the seen test (`searchsorted`), the goal fits and the level's
bookkeeping. Across the stores (§3), the million-state probes (`probe_b`) were 74% of all search time (178 of 240
search-hours on the held-out and subgoal Volumes), lookaheads 13%, target searches 6%.

### 2. The GPU search

`engine.py` is `bar3/scale/engine.py`'s `bfs` and `bfs_multi` with the same arguments and results; `kernel.py` holds
the kernels.

- **Bytecode.** Each item class is compiled by the CPU engine's own compiler (library routines inlined), then packed
  one int32 per instruction. Inlined, every loop has a fixed nesting depth, so a machine needs no stack: one remaining
  count per depth (up to 8, which every vocabulary here fit; a deeper class would be refused, not run wrong). An
  END points back to its LOOP; each class ends with a HALT.
- **The interpreter** (`_interp_k`): one lane per (candidate, probe), a candidate being item c0 + g / S on source state
  g mod S of the chunk's segment. Each step reads the lane's instruction and applies it with selects: values saturate
  at the cap, DEC stops at 0, every primitive and every loop iteration counts a step, a machine dies only when its count
  passes the limit at an iteration's start. One warp per program and 2 steps between checks of the loop condition were
  the fastest of 21 settings (`kernel_bench.py`).
- **The hash** (`_hash_k`): `library/search.hash_rows` of the state's registers as little-endian int16 words, the same
  uint64 arithmetic; with whether a probe died and whether the item left the state as it was. Keys are kept as int64
  with the top bit flipped, so that int64 order is the uint64 order numpy sorts by (the goals' order, and so the
  programs' order, follows it).
- **The bookkeeping** (torch, on the GPU): the seen test is a binary search in the sorted keys of the earlier levels;
  the goal fits read the answer registers in place; at the level's end a stable sort gives each new key's first
  occurrence, the level's states in first-occurrence order, and the parents sorted by child as numpy's stable argsort
  sorts them. A streamed level gathers nothing but its goal states. Chunks are up to 1 GB of output rows. Whether the
  budget is passed at one chunk or another changes nothing: the CPU's test is on the running count of fresh states,
  and every goal state is kept either way.
- **Read-back.** Parents and held levels stay on the GPU; `programs()` and `near_misses()` (unchanged) read them through
  lists that copy an entry to numpy on first use.
- **What can differ:** only a search that stops on its time limit (the GPU checks the clock after each chunk, the CPU
  after each class), and the seconds.
- **The flag.** `engine.use(True)` sets `bar3.scale.jobs.engine` to this module (and `use(False)` back), so
  `jobs.search_job` and `jobs.search_jobs` (shared searches) run here unchanged. For learner runs, `searcher.py` is the
  mapper (as `pool.Searcher` is on the CPU containers): searches one at a time on the GPU, each batch's vocabularies
  prepared ahead on CPU worker processes by the same `classes_for`.

### 3. The same results

| check | what | result |
|---|---|---|
| `kernel_check.py` (L4, on the first kernel and again on the final one) | the Triton kernel against the CPU kernel on 200 chunks of random programs (every instruction, L and Z blocks on every register, nesting to depth 8, step limits low enough that machines die), at 1–32 probes and caps of k = 3–26; then every class of the 35 primitives at k = 7 and 26, and of the 34-routine library (27,774 classes, depth 5), in chunks that start and end mid-class | 0 of 245 chunks differ (drop flags, rows, keys) |
| `check.py` (H100) | `bar3/scale/check.py`'s 38 jobs, 2 million-state probes and a 3M target search, through `jobs.search_job` on both engines; and the h1 pool's 93 probes in 7 shared searches | 41 of 41 and 93 of 93 records identical |
| `equiv.py` (10 L40S) | stored searches (below) | 18,719 stored searches: 18,708 identical, 0 differing, 11 cut by the CPU's time limit (below) |
| `replay.py` (L40S) | whole learner runs from scratch on the GPU, every round record against the stored run's | two runs (1,522 and 3,498 searches, 2 and 5 rounds): every round identical |
| `reach.py` (L40S) | SCALE.md §3's six reach searches (to 383M states) | the same level sizes |

**The stored searches.** A store holds only (the job's hash, its record), so the jobs were recovered by replaying each
run's learner from its stores (`capture.py`, local, seconds a run): 55 runs (the newton study's 4, picks 2, rowcheck's
newton run, the held-out test's 20 and 8 controls, the subgoal study's 20), every search found in the stores, 63,262
distinct jobs, 131 CPU search-hours. Then on the GPU:

| kind | searches rerun | identical | differing | cut by the CPU's time limit | stored CPU s | GPU s (L40S) |
|---|---|---|---|---|---|---|
| million-state probes (`probe_b`) | 12,696 (all) | 12,685 | 0 | 11 | 345,256 | 4,726 |
| target searches (3M states) | 153 (all) | 153 | 0 | 0 | 26,450 | 602 |
| guided target searches | 124 (all) | 124 | 0 | 0 | 12,005 | 450 |
| deep probes (10M states) | 21 (all) | 21 | 0 | 0 | 4,731 | 34 |
| target probes (with near misses) | 123 (all) | 123 | 0 | 0 | 315 | 9 |
| 30k-state probes (`probe_a`) | 4,002 (of 26,132) | 4,002 | 0 | 0 | 2,325 | 298 |
| lookaheads (`look`) | 1,600 (of 24,013) | 1,600 | 0 | 0 | 4,533 | 427 |
| **all** | **18,719** | **18,708** | **0** | **11** | 395,615 (109.9 h) | 6,546 (1.82 h) |

The seconds are each job's wall time, as stored (the item enumeration, the search, the checks of found programs; a
shared search's time split evenly over its jobs). The searches come from the held-out test (7,413), its controls
(3,393), the subgoal study (5,309) and the newton study (2,601, with the picks and rowcheck studies' newton-learner
runs, which share most of their searches); 8,175 ran as shared searches (`bfs_multi`) and 10,544 alone (`bfs`). The
lookaheads ran through `searcher.py`, each vocabulary prepared in a worker process.

Every million-state probe, target, guided and deep search of the 55 runs and every target probe was run (split over 8
GPUs by vocabulary; 6 parts with shared searches, `bfs_multi`, 2 one search per job), and seeded samples of the small
ones. The 11 searches the CPU had stopped at its 600 s limit midway through their last level are reported apart: each
earlier level is the same, and the GPU finishes the last one in 1.3–7.0 s (one shared group shows the CPU's cut point
varying with the clock, 20.2M–22.2M states, against the GPU's 22,213,182 every time).

**The replay.** Two whole runs of the newton study's learner, from scratch, every search on one L40S (12 cores
preparing vocabularies), shared searches as on the CPU containers, each round record compared with the stored run's in
every field but the times (seconds, walls and elapsed times at any depth):

| run | searches (after sharing) | rounds | identical | GPU run | stored run's search time | billed |
|---|---|---|---|---|---|---|
| anchored pairs on the triangular numbers (`ct2a2_clean_deep_sub_ns_tri_nw1`) | 1,522 (1,020) | 2 | 2 of 2: round 1 finds `L.X[M1.4]`, as stored | 77 s (41 s searching, 30 s waiting for vocabularies) | 3,528 s | $0.068 against $0.111 |
| anchored pairs on 10·x² + 1 (`ct2a2_clean_deep_sub_ns_tenrho_nw1`) | 3,498 (2,244) | 5 | 5 of 5 (no formula in 5 rounds, as stored) | 492 s (286 s searching, 183 s waiting for vocabularies) | 23,364 s | about $0.39 against $0.73 |

The billed GPU cost includes the container's start (about 15 s); the CPU's is the stored search-hours at the CPU
containers' billed rate. Both runs spend a third or more of their time on the CPU: preparing each lookahead's
vocabulary (two new routines, so a new item enumeration) and checking found programs.

**Harness mistakes caught on the way** (each a false alarm or a bug of this folder, not of the search):
- comparing a fresh record (tuples) with a stored one (JSON lists) flags every record with aliases or near misses;
  all comparisons now go through JSON first;
- the first replay differed in four searches' level sizes: the worker processes that prepare vocabularies received
  the library as a key-sorted dict, and the item enumeration depends on the library's order (it picks each class's
  representative) while `classes_for`'s cache key does not, so the reordered vocabulary was accepted silently. With
  the order kept, the worker's vocabularies equal the in-process ones (16 of 16: classes, members, bytecode) and the
  replay is identical. The same key also means that, on the CPU containers, two jobs whose libraries hold the same
  routines in different orders would share whichever vocabulary was built first (not seen to matter in these runs).

### 4. Speed per level, and by GPU

The timing sample: 56 stored searches, 8 of each kind (`timing.py --per=8 --seed=1`), each run on the CPU engine (a
16-core CPU container, 4 searches at once) and on each of five GPUs (one search at a time), with the clock read at each
level's end; every record is the stored one on every engine. Seconds per level, summed by the level's size (its
distinct states; for a streamed level, the fresh states counted), leaving out the first 4 jobs (the first search of
each CPU worker holds numba's compile, and the GPU's first holds Triton's):

| states in the level | levels | CPU core s | H100 s (x) | L40S s (x) | A100 80 GB s (x) | A10G s (x) | L4 s (x) |
|---|---|---|---|---|---|---|---|
| 1–10 | 17 | 0.04 | 0.03 (1x) | 0.05 (1x) | 0.04 (1x) | 0.04 (1x) | 0.07 (1x) |
| 10–100 | 61 | 0.15 | 0.63 (0.2x) | 0.62 (0.2x) | 0.65 (0.2x) | 0.76 (0.2x) | 1.07 (0.1x) |
| 100–1k | 58 | 0.25 | 0.16 (2x) | 0.22 (1x) | 0.20 (1x) | 0.22 (1x) | 0.32 (1x) |
| 1k–10k | 50 | 1.82 | 0.24 (8x) | 0.32 (6x) | 0.34 (5x) | 0.34 (5x) | 0.46 (4x) |
| 10k–100k | 48 | 17.65 | 0.45 (40x) | 0.52 (34x) | 0.73 (24x) | 0.71 (25x) | 0.83 (21x) |
| 100k–1M | 48 | 153.3 | 1.37 (112x) | 1.35 (114x) | 2.46 (62x) | 3.00 (51x) | 3.13 (49x) |
| 1M–10M | 30 | 661.4 | 4.62 (143x) | 4.26 (155x) | 8.31 (80x) | 11.80 (56x) | 12.53 (53x) |
| 10M–100M | 16 | 3,314.7 | 22.1 (150x) | 22.0 (150x) | 39.2 (85x) | 61.2 (54x) | 65.7 (50x) |

Small levels gain nothing: a level of a hundred states is a fraction of a millisecond on the CPU and about ten
milliseconds of kernel launches and synchronizations on the GPU (the 10–100 row also holds a few later compiles, for
vocabularies with deeper loops). From ten thousand states on the GPU pulls ahead, and from a million on the L40S and
the H100 run a level about 150 times faster than a CPU core. The CPU engine here took 4,422.7 s for the 32 big
searches that took 3,566.9 s on the CPU containers that stored them (their cores were 1.24 times faster), so against
the stored times the gains are about a fifth smaller (§5's table uses the stored times).

Where the GPU time goes (`engine.PROF`, H100): in a big search almost all of it is the two kernels (a 10M-state deep
probe: 96%); the seen test, the goal fits and the level's bookkeeping together are a few percent. In a learner run
the rest is the CPU's: the replayed runs spent 30 and 183 s waiting for vocabularies, against 41 and 286 s searching.

### 5. What the same money buys

**Rates, billed.** The CPU containers' searches cost $0.113 a search-hour: the held-out test's $7.68 for 68.0
search-hours, the digits study's $50.69 for 446.4, SCALE.md's $62 for 562 (a 16-core container runs 16 searches at
once beside the learner, and is billed for idle cores too). This study's own bills, per container-hour: an L40S with
4 cores and 32 GiB $2.23–2.34 (taken as $2.30), with 12 cores and 48 GiB $2.67–2.81; an H100 with 4 cores and 32 GiB
$4.25; a 16-core CPU container $1.02. The other GPUs at their list price plus the $0.35 an hour the same cores and
memory add: A100 80 GB $2.85, A10G $1.45, L4 $1.15.

**Per search**, on the timing sample's 32 big searches (8 each of million-state probes, target, guided and deep
searches; search time only, the vocabularies already built), against their stored CPU search time:

| engine | search s | x faster | $ | x cheaper | instructions bought over the primitives (x 9.5 each) | over a library (x 40) |
|---|---|---|---|---|---|---|
| CPU containers (stored) | 3,566.9 | 1 | 0.112 | 1 | 0 | 0 |
| L40S | 38.0 | 94 | 0.024 | 4.6 | 0.68 | 0.41 |
| H100 | 37.0 | 96 | 0.044 | 2.6 | 0.42 | 0.26 |
| L4 | 99.0 | 36 | 0.032 | 3.5 | 0.56 | 0.34 |
| A10G | 88.7 | 40 | 0.036 | 3.1 | 0.51 | 0.31 |
| A100 80 GB | 60.4 | 59 | 0.048 | 2.3 | 0.38 | 0.23 |

The L40S is as fast as the H100 on this integer work, at half the price; the slower GPUs are cheaper than the H100
but not than the L40S.

**As the learner runs them**, billed for billed:
- the equivalence run: every big search of the 55 runs, 108.0 search-hours on the CPU containers ($12.20), cost $3.90
  on eight L40S containers, their start, enumeration and program checks included: **3.1 times cheaper**;
- the two whole runs (§3): $0.068 against $0.111 (1.6 times) and about $0.39 against $0.73 (1.9 times): a run waits on
  the CPU for a third or more of its time (each lookahead's vocabulary, the program checks, the learner itself);
- SCALE.md's reach searches (§3 there; `reach.py`, L40S), where the search is all the work:

| search | states (streamed) | CPU core s | L40S s | x faster | x cheaper |
|---|---|---|---|---|---|
| x² mod N's target, primitives, budget 3M | 40.7M (39.1M) | 118.8 | 0.91 | 131 | 6.4 |
| the same, budget 30M | 383.1M (368.7M) | 1,279.7 | 10.31 | 124 | 6.1 |
| 10·x²'s target over the 34-routine library, budget 3M | 4.8M (4.7M) | 45.8 | 0.23 | 199 | 9.8 |
| the same, budget 30M | 189.7M (186.6M) | 2,086.5 | 10.40 | 201 | 9.9 |

(The 300k-budget searches, 1.7 and 0.5 s, included the kernels' first compile.) The 30M-state search used 23.4 GB of
the GPU's 48.

**What that buys.** The same money buys R times more states, R from about 3 (the learner's big searches as it ran
them) to about 10 (the deepest searches over a library). Each instruction costs about 9.5 times the states over the
primitives and about 40 times over a carried library (SCALE.md §3), so the same budget reaches **log R / log 9.5 =
0.5–1.0 more instructions over the primitives, and log R / log 40 = 0.3–0.6 over a library**: about two thirds of an
instruction, as SCALE.md's compiled search bought about one. Past the 30M-state searches one GPU also runs out of room:
their next level, 368.7M states, would have to be stored (about 100 GB).

**What it buys in time.** Much more: per search 36–200 times faster, so a 10-minute probe takes 5 seconds, and the
replayed runs took 77 and 492 seconds where their searches alone had taken an hour and six and a half. Studies that
waited hours for a learner's rounds (SCALE.md's dearest step, Montgomery's in hardest first's sequence, took 12 hours of
a 16-core container) can run in a fraction of that, at about the same cost.

### 6. Dead ends, and how hard each was tried

- **Kernel settings** (`kernel_bench.py`, one H100, two real 1M-state searches, each setting's record checked equal):
  21 settings of lanes per program (32, 64, 128), warps (1, 2, 4) and machine steps between checks of the loop
  condition (1, 2, 4, 8). Over the primitives every setting took 0.05–0.07 s of kernel (the search is too small to
  tell them apart); over the 34-routine library one warp per program was fastest (0.25 s) and the others took
  0.29–0.50 s; the steps between checks changed little. Kept: 32 lanes, one warp, 2 steps. Not tried: a lane running
  several machines, ordering candidates by expected work, the hash fused into the interpreter.
- **Recompiling.** The first kernels took the row length as a compile-time constant and let Triton specialize integer
  arguments by value (divisible by 16, equal to 1): small searches spent about a second each compiling a new variant.
  Runtime row lengths and `do_not_specialize` fixed it.
- **Gathering every fresh row of a streamed level**, as the CPU's chunks do: dropped; a streamed level now counts,
  fits and keeps only its goal states.
- **Item enumeration on the GPU: not done.** It is 7% of the stored CPU time, but 46% of the lookaheads' (each
  lookahead adds two routines, so each needs a vocabulary of its own: 19,305 of the 24,013 do), and it stays on the
  CPU here. Profiled on the 34-routine library (2.3 s for 27,774 classes), two thirds is the per-item signature call's
  Python and dispatch and a third is compiling items in Python, so a GPU signature kernel would cut it about three
  times at most. Done instead: vocabularies prepared ahead on CPU worker processes (`searcher.py`).
- **CPU timings with 16 searches at once**, as the CPU containers ran them: the timing sample (8 jobs of each kind,
  heavy with 3M- and 10M-state searches) ran a 16-core container out of its 96 GiB three times (the CPU engine holds
  rows as int32); stopped, and rerun with 4 at once.
- **Budgets past 30M states: not run.** The next level of x² mod N's search over the primitives (cost 8, 368.7M
  states) would have to be stored: about 100 GB of rows (half as bytes, which the cap of 98 would allow), more than
  one GPU holds; reaching it needs the stored levels in host memory, streamed through the GPU a chunk at a time.
- **The digit machine** (DIGITS.md's D.B blocks) is not in the kernel, so the digits runs of the rowcheck and picks
  studies were not replayed (their newton-learner runs were).

### 7. Files and how to rerun

Everything is in `cpu_first/bar3/gpu_search/`; it imports `bar3/`, `bar3/scale/` (and its study folders) and
`library/`, and changes none of them.
- `kernel.py`: the packed bytecode, the Triton interpreter and hash, and the CPU kernel standing in where there is no
  GPU (so everything but the kernels can be tested anywhere).
- `engine.py`: `bfs`, `bfs_multi`, `Codes` (the CPU engine's API), the lazily copied parents and levels, `use()` (the
  flag), and an optional phase profiler.
- `searcher.py`: a learner's mapper on one GPU host, vocabularies prepared on CPU workers.
- `kernel_check.py`, `check.py`: the kernel, and whole searches, against the CPU engine.
- `capture.py`: stored searches as jobs, by replaying each run's learner from its stores.
- `equiv.py`: captured jobs on the GPU against their stored records (`--part` by vocabulary, `--share`, `--workers`).
- `replay.py`: a whole learner run from scratch, round records against the stored run's.
- `timing.py` (seconds per level, either engine), `kernel_bench.py` (kernel settings), `reach.py` (SCALE.md §3's
  searches), `report.py` (the tables here).
- `debug_chunks.py`, `debug_ways.py`, `debug_replay.py`: the three diagnoses of §3.
- `modal_gpu.py`: Modal (a function per GPU type, a 12-core L40S for learner runs, a 16-core CPU container); each run
  in-process, with a timeout and a hard memory limit, and `modal run` without `--detach`, so a run stops with its
  client.
- `records/`: `equiv/` (one line per search: its key, the stored and GPU seconds, level sizes, stops, differing
  fields), `timing/`, `reach/`, `replay/` (the round records); `logs/`: each Modal run's printout. The captured jobs
  (46 MB) are on the Volume (`v1/jobs/`), not in the repository; `capture.py` rebuilds them from the stores.

Python with numba 0.68, numpy 2.5.4 and torch 2.14.1 (the Modal image pins the same; Triton 3.8.0 comes with torch).
Every launch was preceded by `integration/ablation_s3/spend.py 4750`. What this study cost: about $8 of Modal (31 runs,
at most 10 GPUs at once; $6.90 billed by 20:19 UTC on Oct 26, the last hour's remainder estimated from the runs'
rates), most of it the equivalence run's eight L40S containers ($3.90); every app is stopped.

```bash
cd cpu_first && <python with numba and torch> -m bar3.gpu_search.check
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L4 --module bar3.gpu_search.kernel_check
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu H100 --module bar3.gpu_search.check --args=--big
```

```bash
cd cpu_first && <python with numba> -m bar3.gpu_search.capture newton ct2a2_clean_deep_sub_ns tri nw1 OUT.jsonl.gz STORES
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L40S --module bar3.gpu_search.equiv --args="/out/v1/equiv/OUT.jsonl /out/v1/jobs/FILE.jsonl.gz --part=0/8 --share"
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L40S-C12 --module bar3.gpu_search.replay --args="newton ct2a2_clean_deep_sub_ns tri nw1 /out/v1/records/ct2a2_clean_deep_sub_ns_tri_nw1.jsonl /out/v1/replay/OUT.jsonl --engine=gpu --share --workers=10"
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L40S --module bar3.gpu_search.reach --args="/out/v1/reach/OUT.jsonl 300000 3000000 30000000"
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m bar3.gpu_search.report
```

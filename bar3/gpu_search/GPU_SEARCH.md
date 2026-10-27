# The bar-3 search on the GPU: the same programs, faster and a little cheaper

*Written 2026-10-26. Code in `cpu_first/bar3/gpu_search/`, records in `bar3/gpu_search/records/`, logs in
`bar3/gpu_search/logs/`. No file outside that folder changed; nothing committed. The last section (§8 on, the same
evening) wires the search into the studies' runners, adds the digits study's loop block, and checks a cache key; it
changes one file outside this folder, behind a flag that is off by default.*

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
  studies were not replayed (their newton-learner runs were). Done since: §10, §11.

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

## The GPU in the studies: a flag in the runners, the digit loop, the cache key

*Written 2026-10-26, evening. New in this folder: `modal_learn.py`, `studies.py`, `pool_gpu.py`, `checks_pool.py`,
`digits_accept.py`, `compare_runs.py`, `kernel_check_digits.py`, `order_check.py`, `capture_many.py`; changed:
`kernel.py` (the digit loop), `engine.py` (`use()` also points the digits jobs at the GPU), `capture.py` (more learners,
`stored_only`), `modal_gpu.py` (more folders, one function). Outside it, one flag in `bar3/scale/jobs.py` (`LIB_ORDER`,
off by default: every key as before). Records in `records/learn/`, logs in `logs/`. Nothing committed.*

### In plain words

**What was asked.** The GPU search above was a replayer of its own. Here it becomes something any bar-3 study can
switch on: a flag in the runner that runs the learners, the digits study's loop block in the GPU kernel, and a check
of a cache key that the GPU work had found suspect. The bar stayed the same: identical records.

**The flag.** One Modal runner now runs any bar-3 study's learners, `modal_learn.py --study S --engine gpu|cpu`. It is
the studies' own runner (a container per group of runs, a thread per run, a store of finished searches that a
restarted container replays), with the searches done where the flag says: on `gpu`, one L40S per group, every search
on the GPU, and the item enumeration and the checks of found programs on the container's other cores; on `cpu`, the
16-core containers and worker pools the studies used. Eight studies are registered (`studies.py`); their own runners
are unchanged.

**The digit loop.** The digits study's block `D.B[...]` (the body once per digit of X in base B, the digit in X) now
runs in the GPU kernel. Its code is compiled in only for vocabularies that use it, so every other vocabulary runs the
kernel as before. Against the CPU kernel, on random programs with digit loops in four bases nested up to 8 deep and on
the digits study's own vocabularies (up to 26,682 item classes): 0 of 373 chunks differ, and the original kernel check
still passes on the changed kernel.

**The cache key.** The vocabulary cache keyed a library by its routines sorted by name, though the routines' order
fixes the order of the search's items. So a job whose library held the same routines in another order would get the
vocabulary of whichever came first. Checked by replaying 234 runs from their stores (every job of the held-out,
subgoal, newton, digits, rowcheck, picks, reach and consolidate studies, 457,651 in all): no library ever appeared in
two orders, and every library a job used was a round's library with routines appended. Every round library in all 388
committed bar-3 record files: none in two orders either. So no committed record was affected. The fix is behind a
flag (`LIB_ORDER`, off by default); turned on, it changes none of these runs.

**The same records.** Whole runs from scratch on the GPU, each round record compared with the stored one field by
field, and each search with the stored search. The digits study's two inventions (binary x mod N, under anchored
pairs in round 1 and under hardest first in round 3) came again, record for record. The whole rowcheck study (three
runs) and the picks study's two Modal runs reproduce record for record once the 1–4 searches a run that the CPU's clock
had cut (it stops a million-state probe at 600 s, a target search at 2,400 s, wherever it is) are read back from their
stored records. Of 14,207 searches, 12,114 of them over digit vocabularies, every one is identical but those 11. The
runner's CPU side, run on the same study, gives the same searches.

**Time and money.** On one GPU a run of these studies takes 5 to 34 minutes, where the CPU containers took 2.5 to 3.8
hours. Its searches run 30 to 60 times faster than on a CPU core; what is left is the learner itself and each
lookahead's item enumeration, which stay on the CPU. A GPU container costs about $3.2 an hour, so a run costs 0.7–0.8
times what its searches cost on the CPU containers. Whole studies, one GPU per run: the picks study's two Modal runs
in 9 minutes for $0.93 (the CPU containers: 3 hours and $5.30); the rowcheck study in 34 minutes for about $3.30 (its
searches: $4.06, in containers that took 3.7 hours); the 36-run digits study, projected, in about 41 minutes on 36
GPUs or 1.3 hours on 10, for about $38 (it billed $50.69 and took 4.8 hours). So a study turns around in minutes
instead of hours, for a little less money.

### 8. The flag in the runners

Every bar-3 study since the subgoal study that ran on Modal ran its learners through a copy of one `group`
(bar3/scale/subgoal/modal_subgoal.py, copied by the held-out, digits, picks, reach and consolidate studies; the newton
and rowcheck studies ran from stores, locally): a container per group, a thread per run, the searches on a process
pool (`pool.Searcher` or a subclass whose workers know the study's targets or the digits jobs), every finished search
appended to the group's store on a Volume, so that a container restarted after preemption replays its runs.
`modal_learn.py` is that `group` once, for every study, with the searcher chosen by `--engine`:

- `cpu`: a 16-core, 64 GiB container and the CPU searcher the study's own runner gave that learner copy
  (`studies.cpu_searcher`: the held-out workers for the newton-type copies, the digits workers for the digits copies);
- `gpu`: an L40S with 16 cores and 64 GiB, and `pool_gpu.Searcher`.

`pool_gpu.Searcher` takes `pool.Searcher`'s arguments and keeps its behaviour: identical jobs run once (in flight or
done); with `share`, a call's jobs that share a search space (`jobs.same_search`) run as one search, each with the
record it would get alone; the store (one `[key, record]` a line) is read at start and extended after each call;
a failed job returns its exception, which is never stored. What changes is where the work runs: every search on the
GPU, one at a time (several learner threads take turns under a lock); each search's vocabulary prepared ahead (two per
worker) on 15 CPU worker processes by `jobs.classes_for` itself and put in jobs.py's cache under its own key just
before the search, as `searcher.py` did; and the checks of found programs on a second pool of 15 (§10). A shared
search whose jobs need a target registered for their N <= 40 check runs with it
set (`pool.work` sets it for a job run alone; `work_many` did not, a latent gap; the runs here had none,
`general_shared` = 0).

`studies.py` names, for each study, its configs and the learner copy of each config: `subgoal`, `heldout`, `newton`,
`digits`, `rowcheck`, `picks`, `reach`, `consolidate`. A group runs one copy. Runs are named
`<config>_<target>_<tag>` as the studies name them, so a rerun's round records compare field by field with the stored
ones; each engine writes under its own folder of the `bar3-learn` Volume (`learn/<study>/<engine>/`, or `--folder`).
`--store NAME` starts each group from a copy of a store put on the Volume (§11 uses it to read back the searches the
CPU's clock had cut). Limits: the GPU container has a 4-hour timeout, the CPU one 24 hours, both a hard memory limit
at 1.5 times the request and one retry after preemption, and at most 10 GPU and 8 CPU containers run at once.
`engine.use(True)` now also points `bar3/scale/digits/jobs.py`'s own `engine` name at the GPU (its target jobs
searched through it, so before this they would have run on the CPU even with the flag on).

`compare_runs.py` does the comparisons: `rounds` (round records, every field but the times, as `replay.py`),
`searches` (every search in a rerun's store against the record the earlier stores hold under the same key, as
`equiv.py`; a stored search the CPU's clock cut is counted apart) and `pins` (§11).

### 9. The cache key

`jobs.classes_for` caches a vocabulary under `(json.dumps({token: program} sorted by token), kinds, k, max_body,
costs)`, but builds it with `tokens = PRIMS + [t for t in lib if t not in kinds]`, in the library's order, and the
item enumeration keeps the first item of each signature as the class's representative, in that order. So two
libraries with the same routines in different orders give different vocabularies (the classes' order, so the order in
which the search applies items and reads programs back, and which 200 programs a search returns) under one key: in a
worker that has built the first, the second gets it. `jobs.same_search` has the same blind spot (it dumps the job
with `sort_keys`), so two such jobs in one call would run as one shared search on the first's vocabulary. A two-
routine library in both orders (`T.a`, `T.b`): the second order is served the first's entry, whose classes come in the
other order.

**Was any committed record affected?** Only if one container saw a library in two orders. Checked by replays
(`order_check.py`):

| check | runs | jobs | result |
|---|---|---|---|
| `orders`: jobs grouped by the cache key, the orders of each key's libraries, over all runs at once (any two runs, the conservative test) | 234 | 457,651 (72,806 vocabulary keys) | no key in more than one order |
| `prefix`: is each job's library a round record's library (or none) with routines appended? | 234 | all | yes, every job of every run |
| `rounds`: the libraries of every round record, over all files at once | 388 record files (1,322 round libraries, 710 distinct) | — | none in more than one order |

The 234 runs' jobs were recovered by replaying each learner from its stores (`capture.py`, `capture_many.py`): the 55
the GPU work had captured (held-out 28, subgoal 20, newton 4, the picks and rowcheck studies' newton-copy runs 3), the
digits study's 36 with the digits-copy runs of rowcheck (2), picks (3) and reach (3), and the consolidate study's 135
(read from its own Volume; 32 of them, runs its spend watcher cut or that were stopped by hand, end at their first
search the stores lack, `stored_only`). Every job of the others came from the stores; nothing was searched. The 388
record files are every round record of bar3/scale/ and its studies, the scale study's 136 included (its runs were not
captured: for them the round-library check stands alone, with the `prefix` result on 234 runs as the reason it
suffices: a job's library is a round's with routines appended). `consolidate/check.py`'s `learner()` reads reach's
LEARNER table, so it cannot name consolidate's own configs; `capture.py` picks the copy itself. The breadth study's
`bar3/jobs.classes_for` has the same key; its records were not checked here.

**The fix**, behind a flag: `LIB_ORDER` in `bar3/scale/jobs.py` (environment `BAR3_LIB_ORDER=1`, read on import, so
the worker processes a container starts share it; `modal_learn.py --lib-order` sets it). On, `classes_for` keys a
library as its (token, program) pairs in order and `same_search` keeps them in order, so each order gets its own
vocabulary and never shares a search. Off (the default), every key is exactly as before, so the studies' runners and
every stored record replay as they did. Since no run had a library in two orders, the ordered keys split these runs'
jobs exactly as the sorted ones did: the flag on changes none of their records.

### 10. The digit loop in the kernel, and the checks of found programs

`bar3/scale/digits/machine.py` adds `D.B[...]` to the CPU engine as opcode DLOOP B e: read X once on entry, run the
body once per digit of that value in base B (B = 1: v ones), from the top, with the digit in X; every iteration counts
a step as a LOOP's do. In the GPU kernel (`kernel.py`):

- **Packing.** A DLOOP is packed as a LOOP on X whose b field (3 bits, otherwise 0 for a LOOP or END) holds its base's
  index + 1 among the vocabulary's bases, and so does its END; an unknown opcode is refused (before this a DLOOP would
  have read as a HALT). `pack_all` (all classes at once) is checked against `pack` (row by row).
- **Tables.** For each base: the number of digits of every value below 2^15 (rows are int16) and the place values
  B^p, p < 16 (`digit_tables`). On entry a lane looks up its count, keeps the value read per depth, and puts the top
  digit, (v // B^(n-1)) mod B, in X; at the END, the next place's digit; in unary, 1.
- **A switch.** The digit code is compiled in (`DIG`, a constexpr) only for a vocabulary with digit loops; without, the
  kernel is the one above, line for line (`cnt` is the loop register's value, as before).

| check (`kernel_check_digits.py`, L4, 316 s) | classes | chunks | differing |
|---|---|---|---|
| packing: `pack_all` against `pack` on 20 vocabularies of 20 random classes | 400 | — | 0 |
| random programs: every instruction, L, Z and D blocks (bases 1, 2, 3, 10) nested up to 8 deep, 1–32 probes, caps up to 32,000 (values of 15 binary digits), step limits 40 to 100,000 | 1,988 in 60 vocabularies (651 with a D block) | 300 | 0 |
| primitives + D.2, k = 7, on the target's machine (20,000 steps), x < 70 | 6,968 (981 with D) | 18 | 0 |
| primitives + D.10, k = 7 | 7,011 (1,023) | 18 | 0 |
| primitives + D.1, k = 7 | 6,858 (871) | 12 | 0 |
| primitives + D.2, k = 10 (100,000 steps) | 6,968 (981) | 12 | 0 |
| a digits run's last library (27 routines, two with D.2) + D.2, k = 7, depth 5 | 26,682 (3,111) | 13 | 0 |
| `kernel_check.py` (the original check) on the changed kernel | | 245 | 0 |

Each chunk compares the drop flags and every kept candidate's registers and hash key with the digits study's CPU
kernel (`machine._apply_many`, which the CPU engine runs once the block is installed). The whole runs of §11 then put
the kernel through 12,114 searches over digit vocabularies, every one identical to its stored record but the 10
that the CPU's clock had cut (§11).

**The checks of found programs** (`digits_accept.py`, `checks_pool.py`). A job on the digits target accepts a program
only when it is
right on the target's rows at both sizes and at big x (200 instances of up to 100,000 steps), and the search checks
each level's programs (up to 200) before going on. On the CPU containers those checks ran inside each search's worker,
beside fifteen other searches; on a GPU host they ran one after another in the process that drives the GPU, which
waited. The first GPU run of the digits study's anchored-pairs win spent 1,091 of its 1,258 s in 702 lookaheads (a
look whose level held 433 programs right on the rows only took 17–20 s; the CPU's looks had taken 3,386 s), while its
million-state probes ran 58 times faster than the CPU's. `digits_accept.search_job` is `digits/jobs.search_job` line
for line but for its acceptance: a level's programs not yet checked go to a second pool at once, are checked by the
same code (`machine.evaluate`, `targets.bigx_share`), and their rows are filled in the order the original fills them,
so `rows`, `right_on_rows_only` and the record are the original's. The same run then took 279 s (lookaheads 169 s),
every search identical. The other jobs' checks (`jobs._posed`'s verify, brm's interpreter on the problem's instances
at both sizes; for the digits learners those are the target's rows, 490 and 1,000 of them) then held the hardest-first
runs: a digits run's 1,019 small probes took about 0.15 s each on the GPU host, against 0.4 s on a CPU core.
`checks_pool.py` is `jobs.search_job` and `search_jobs` line for line but for their acceptance: a level's programs not
yet checked go to the pool, where each worker builds the job's own verify with `jobs._posed` and returns the row it
computes, and the search reads those rows as its own (fewer than 4 programs are checked in place). The picks study's
twin runs went from 717 and 880 s to 306 and 520 s (their million-state probes from 324 to 72 s, their small probes
from 186 to 31 s), every record and search identical.

### 11. Whole runs reproduced on the GPU

Each run from scratch through `modal_learn.py --engine gpu`, one L40S container per run, every search on the GPU but
the few the CPU's clock had cut (below), each round record compared with the stored run's in every field but the
times (`compare_runs.py rounds`), and each search the run made with the stored record of the same job
(`compare_runs.py searches`):

| study | run | rounds | target came | GPU run | stored CPU search | searches: identical / clock-cut | round records |
|---|---|---|---|---|---|---|---|
| digits | `ct2a2_clean_b_src_dxmod_dg1` (anchored pairs, binary x mod N) | 2 | round 1, `D.2[T.dig7]` | 279 s ² | 6,373 s | 1,410 / 0 | identical |
| digits | `lp_clean_b_src_dxmod_dg1` (hardest first, binary x mod N) | 4 | round 3, `D.2[MOV.T.X T.dig15]` | 735 s ¹ | 9,664 s | 1,287 / 0 | identical |
| rowcheck | `lp_clean_deep_sub_ns_rc_tenrho_rc1` (newton copy, 10·x² + 1) | 5 | no, as stored | 566 s ² | 27,170 s | 2,092 / 1 | identical ⁴ |
| rowcheck | `lp_clean_b_src_rc_dxsq_rc1` (digits copy, x² mod N) | 5 | no, as stored | 808 s ² | 25,516 s | 1,759 / 2 | identical ⁴ |
| rowcheck | `ct2a2_clean_b_src_rc_dxsq_rc1` (digits copy, x² mod N) | 5 | no, as stored | 2,024 s ³ | 76,456 s | 4,130 / 2 | identical ⁴ |
| picks | `lp_clean_b_src_twin_dxmod_pk1` (the twin, binary x mod N) | 5 | no: the lost win, as stored | 306 s ³ (717 s ²) | 16,792 s | 1,550 / 2 | identical ⁴ |
| picks | `lp_clean_b_src_twin_dxsq_pk1` (the twin, x² mod N) | 5 | no, as stored | 520 s ³ (880 s ²) | 26,704 s | 1,968 / 4 | identical ⁴ |

The code each was run with: ¹ found programs checked in place (before §10's pools); ² the digits target's checks
pooled, the others in place; ³ both pools, the final code. The picks runs ran both ways, with the same records.
⁴ With the searches the CPU's clock had cut read back from their stored records (`compare_runs.py pins` writes them
as a store; `--store` starts the run from it). The CPU stops a million-state probe at 600 s and a target or guided
search at 2,400 s, wherever its clock puts it; the GPU finishes them (one probe the CPU cut at 29.2M states of its
last level reached 29.9M; two target searches cut at 71.9M and 84.7M states reached 79.0M and 118.0M), so its records
differ there and its rounds after. Run without the pins, the newton-copy run was identical in rounds 0–3 and
differed in round 4 (one problem's status, through the probe), and the rowcheck `lp` digits run in rounds 3–4 (the
target search); the first anchored-pairs run, stopped after round 2, was identical in all three. Every other search of
every run was identical.

So the digits study's two inventions came again on the GPU, in the same round, with the same programs, and the whole
rowcheck study and the picks study's two Modal runs reproduce record for record. Over the seven runs 14,207 searches
were compared, 12,114 of them over digit vocabularies (every run but the newton copy): 14,196 are identical and the
other 11 are the clock-cut ones.

### 12. Time and cost per run, and per whole study

**Rates, billed.** The GPU container here (an L40S with 16 cores and 64 GiB) bills about $3.2 an hour, for the run
and about 110 s of start (Triton's compile, the vocabulary workers' imports); the 16-core CPU containers of the same
runner about $1.25 an hour; the studies' own CPU containers, fully loaded, came to $0.113 a CPU search-hour (§5).

**Per run** (the final code: both pools of §10; the two runs marked * had only the target's checks pooled):

| run | stored CPU search | at $0.113/h | the CPU containers' run (stored) | GPU run | GPU cost ⁰ | search: GPU s against CPU s |
|---|---|---|---|---|---|---|
| picks `lp_clean_b_src_twin_dxmod_pk1` | 4.66 h | $0.53 | 9,070 s ¹ | 306 s | about $0.37 | 216 against 12,241 (57x) |
| picks `lp_clean_b_src_twin_dxsq_pk1` | 7.42 h | $0.84 | 10,946 s ¹ | 520 s | about $0.56 | 345 against 21,065 (61x) |
| rowcheck `ct2a2_clean_b_src_rc_dxsq_rc1` | 21.24 h | $2.40 | 13,359 s ² | 2,024 s | about $1.90 | 1,830 against 71,619 (39x) |
| rowcheck `lp_clean_b_src_rc_dxsq_rc1` * | 7.09 h | $0.80 | 13,606 s ² | 808 s | about $0.82 | 705 against 20,771 (29x) |
| rowcheck `lp_clean_deep_sub_ns_rc_tenrho_rc1` * | 7.55 h | $0.85 | — ³ | 566 s | $0.61 | 444 against 26,570 (60x) |
| digits `ct2a2_clean_b_src_dxmod_dg1` * | 1.77 h | $0.20 | — ³ | 279 s | $0.35 | 212 against 6,373 (30x) |

⁰ Billed where an app ran one run within one billed hour ($0.61, $0.35); else at the container's rate, start
included. ¹ Both in one 16-core container on Modal, from the earlier studies' stores (their rounds 0–1 read back, not
searched); the arm billed $5.3. ² The digits study's runs whose searches these reuse (`ct2a2_clean_b_src_dxsq_dg1`,
`lp_clean_b_src_dxsq_dg1`), six runs to a 16-core container, with preemptions replayed from the store. ³ Replayed
from stores, so their stored times are not a run's.

So on one GPU a run takes 5 to 34 minutes where the CPU containers took 2.5 to 3.8 hours: 7 to 30 times sooner, the
least for the anchored-pairs run with its 1,956 lookaheads. Its searches run 30 to 60 times faster than on a CPU core;
the rest of a run's time is the learner itself and each lookahead's item enumeration, which stay on the host's CPU (the
anchored-pairs run searched 1,830 of its 2,024 s, of which its lookaheads 1,355 s). In money, a run on the final code
costs 0.67 to 0.79 times what its searches cost on the CPU containers (the three runs above); the runs from before
both pools cost 0.7 to 1.75 times (the smallest run is dearest: its two minutes of start are a quarter of its bill).

**The same runner on the CPU** (`--engine cpu`, the rowcheck study's three runs in two 16-core containers, launched
beside the GPU runs). My spend watcher stopped both containers at 18:12 PDT, 72 minutes in (its total, §13), so these
runs are partial. Each round finished before the stop, against the same rounds on the GPU (the pinned runs of §11):

| run | rounds finished | CPU container | GPU | records |
|---|---|---|---|---|
| `ct2a2_clean_b_src_rc_dxsq_rc1` | 0 | 2,701 s | 77 s | identical |
| `lp_clean_b_src_rc_dxsq_rc1` | 0–1 | 3,396 s | 96 s | identical |
| `lp_clean_deep_sub_ns_rc_tenrho_rc1` | 0–2 | 2,135 s ⁴ | 286 s | identical |

⁴ From its restart after a preemption ten minutes in (the replayed part is quick). The two rounds the digits runs
wrote at the stop itself (round 1 and round 2) hold as failed the searches the stop interrupted, where the stored ones
hold them; every search the containers stored before (2,262 and 1,663) is the stored record. So the CPU side of the
flag gives the studies' searches, and the CPU containers took 7 to 35 times as long per round as one GPU.

**Per whole study.** A study's turnaround is its slowest run when every run has its own GPU, and its cost the sum:

| study | runs | CPU containers (stored) | on GPUs, one per run |
|---|---|---|---|
| the picks study's Modal arm (the twin runs) | 2 | 10,946 s, $5.3 billed (from earlier stores) | 520 s, about $0.93 (from scratch) |
| the rowcheck study | 3 | the stored runs' 35.9 search-hours: $4.06 at $0.113, made in containers that took 3.7–3.8 hours | 2,024 s (the anchored-pairs run; the others 808 and 566 s, before both pools), about $3.33 |
| the digits study (projected) | 36 | 446.4 search-hours, $50.69 billed, the slowest container 4.8 hours | about 41 minutes and $38 on 36 GPUs; on 10, about 1.3 hours |

The projection: on the final code the hardest-first runs took 66–70 s of GPU container per CPU search-hour (the two
picks twin runs) and the anchored-pairs run 95 s; the digits study's 18 hardest-first runs hold 121.3 of its CPU
search-hours and its 18 anchored-pairs runs 325.1. With about 110 s of start a run, that is 12.0 GPU-hours, about
$38 against the $50.69 the CPU containers billed (1.3 times cheaper); its slowest run (26.0 CPU search-hours) about 41
minutes on its own GPU, so about 41 minutes for the study on 36 GPUs, and about 1.3 hours on 10 (longest first),
against 4.8 hours. A study of hardest-first runs gains more (the picks arm: 21 times sooner, 5.7 times cheaper, though
its CPU containers read two rounds from stores); one of anchored-pairs runs less, since each lookahead's item
enumeration stays on the host's CPU.

### 13. Files, how to rerun, and what this cost

New in `cpu_first/bar3/gpu_search/`:
- `modal_learn.py`: the runner (`--study`, `--fresh`, `--targets`, `--tag`, `--engine gpu|cpu`, `--groups`, `--store`,
  `--folder`, `--lib-order`; `--fetch` for the records, `--results` for each group's times and searcher counts).
- `studies.py`: the studies the runner knows, their configs and learner copies.
- `pool_gpu.py`: `pool.Searcher` on one GPU; `checks_pool.py`, `digits_accept.py`: the checks of found programs on a
  worker pool, with the original's rows.
- `compare_runs.py`: `rounds`, `searches`, `pins` (§8, §11).
- `kernel_check_digits.py`: the digit loop against the CPU kernel (§10).
- `order_check.py`: the cache-key check (`orders`, `prefix`, `rounds`; `vocab` and `rerun` for a key with two
  orders, which never came up); `capture_many.py`: `capture.py` for many runs at once, each ending at its first job no
  store holds.

Changed: `kernel.py` (the digit loop; an unknown opcode is refused), `engine.py` (`use()` also points the digits
jobs at the GPU), `capture.py` (the digits, reach and consolidate learners; `stored_only`), `modal_gpu.py` (every
study folder in the image, the digits record the kernel check reads, a CPU function that reads the consolidate
study's Volume). Outside this folder: `bar3/scale/jobs.py`'s `LIB_ORDER`, off by default.

Records in `records/learn/<study>/<folder>/`: each run's round records, `calls_*.json` (each run's Modal call) and
`groups_*.json` (what each group returned: its seconds, its searcher's counts); `gpu_first/` and `gpu/` hold the runs
from before the checks were pooled (§10), `gpu_pinned/` the runs with the clock-cut searches read back, `gpu_pooled/`
the final code, `cpu/` the stopped CPU runs; `records/learn/rowcheck/pins_rc1.jsonl` and
`records/learn/picks/pins_twin_pk1.jsonl` are the pinned searches, as stores. The runs' stores (every search, key and
record) are on the `bar3-learn` Volume beside their records, and the captured jobs of §9 on `bar3-gpu-search`
(`v1/jobs_digits/`, `v1/jobs_consolidate/`; the stores they were read from under `v1/stores/`). Logs in `logs/`
(`kcheck_digits1_L4.log`, `kcheck3_L4_dig.log`, `capture_*.log`).

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python integration/ablation_s3/spend.py 4750
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_gpu.py --gpu L4 --module bar3.gpu_search.kernel_check_digits
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/gpu_search/modal_learn.py --study digits --fresh lp_clean_b_src --targets dxmod --tag dg1 --engine gpu
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run bar3/gpu_search/modal_learn.py --fetch --study digits --engine gpu
```

```bash
cd cpu_first && python3 -m bar3.gpu_search.compare_runs rounds bar3/gpu_search/records/learn/digits/gpu/lp_clean_b_src_dxmod_dg1.jsonl bar3/scale/digits/records/lp_clean_b_src_dxmod_dg1.jsonl
```

```bash
cd cpu_first && python3 -m bar3.gpu_search.compare_runs pins CAPTURED.jsonl.gz --old CAPTURED.jsonl.gz --out PINS.jsonl
```

```bash
cd cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach bar3/gpu_search/modal_learn.py --study rowcheck --fresh lp_clean_b_src_rc --targets dxsq --tag rc1 --engine gpu --store pins_rc1.jsonl --folder gpu_pinned
```

```bash
cd cpu_first && <python with numba> -m bar3.gpu_search.order_check orders JOBS.jsonl.gz ...
```

For a pinned rerun, `pins_rc1.jsonl` goes to the Volume first (`modal volume put bar3-learn PINS learn/rowcheck/`).

**What did not go as planned, and what is not done.**
- The first whole runs checked found programs inline, on the GPU host's one process: a digits anchored-pairs run
  took 1,258 s, 1,091 of them in lookaheads waiting on checks at big x (§10). Both pools fixed it with the same
  records; the runs in `gpu/` and `gpu_pinned/` are from before one or both.
- The CPU's clock: 0–4 searches a run (11 of the 7 runs' 14,207) were cut by the CPU's time limit; the GPU finishes
  them, so whole-run comparisons need them read back (`pins`). The three runs rerun without pins were identical up to
  the first.
- Money: about $17.40 in all ($13.58 billed by 18:00 PDT, the hour after estimated from the containers' times),
  under the soft $20 (the cap was $30); most of it many short GPU runs at $3.2 an hour with about two minutes of start
  each, run twice or three times as the checks were pooled, and the CPU runs. The spend watcher
  (`bar3/scale/digits/watch_spend.py`, its total kept one hour of the running containers' cost below the cap)
  stopped every app at 18:12 PDT on its estimate of $16.58 against a total of $15: it extrapolates the hour under way
  at the last complete hour's rate, which after a busy hour runs high. That cut the CPU reruns (§12) and one GPU run,
  which was then run again alone.
- Not done: several runs per GPU (a run's searches take turns on one lock that also covers a search's CPU work, so
  packing runs would gain little until that work leaves the lock); the item enumeration on the GPU (§6); a GPU-side
  check of found programs; the breadth study's `bar3/jobs.classes_for`, which has the same key.

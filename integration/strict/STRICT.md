# Strict builds of stage 3: the gray points that go without changing what is learned

*Started 2026-10-24. Folder `cpu_first/integration/strict/`. The certified files are only read: every change is an
exact edit that [`make_strict.py`](make_strict.py) applies to the certified source and checks.*

**Result.** Both strict files certify on the untouched official H100 manifest and on every seed (sp2: the official
run and eight B200 seeds; s3i: the official run and seven more H100 seeds), and every run's climb is the certified
file's run of the same seed line for line: the same practice, the same round, the same tournament, the same freeze
step (§4). So the changes below remove gray points without changing what is learned.

## In plain words

[REVIEW.md](../../REVIEW.md) §5 lists what a strict judge would question in the stage-3 files. Some of it is the
design itself: the climb is a program search inside training, the instruction set and the law book were written by
hand. Some of it is plumbing that can be done another way without the model learning anything different:

- **The clock.** The certified files looked at the wall clock: each search stopped after 900 seconds, and the climb
  stopped 120 seconds before the evaluator's time budget ran out. So how far the climb got could depend on how fast
  the machine was. The strict files never look at the clock to decide anything. Each search and the climb as a whole
  have a budget of *machine work*, counted by the same cost model the search already uses to choose how to run a
  batch of programs (for example, 200 units for one step of the lockstep machine, plus a small amount per row). The
  count is the same on every host, so how far a search goes no longer depends on the machine's speed.
- **Memory of past rows.** The climb practises on the last 4,096 rows it read and searches the round against the last
  512 answer rows. These were Python lists. Now they are tensors registered in the model (buffers), so they are in the
  state dict, counted against the 500 M limit, and on the GPU.
- **The found program.** The round program and the routines it calls were Python objects the forward pass read. Now
  the model holds them in buffers: the program's instructions as the machine runs them (6,602 instructions for
  sp2's round), and its text. The forward pass reads the instructions from there.
- **The optimizer reset.** At the switch to phase B, Adam starts its moments afresh. The scheduler used to do that;
  now the optimizer does it itself, at the same point, so the updates are the same.
- **The late table lessons**, which included a fact that depends on the register's width ("a wrapped x − N is
  large"), are gone, and so is the check of that fact in the table's gate. The s3i ablation had already shown the
  lessons change nothing; the runs here show the same for the check.

What did not change, and why, is in §1's table. The main one: in training the forward pass still returns zero
logits (rule 8(a)). Computing the model's real answer inside autograd on a training batch does not fit in an H100's
memory (§5).

## 1. REVIEW.md §5, item by item

"Fixed" means the strict files no longer have it; §4 says whether the run still learns the same thing.

| REVIEW.md §5 | what it is | strict files | how, or why not |
|---|---|---|---|
| rule 4: reads the clock | `build_optimizer` sets a deadline from the evaluator's budget; each search stops after `SEARCH_S` = 900 s; the climb stops `CLIMB_MARGIN` = 120 s before the budget ends | **fixed** | no decision reads the clock or the budget. Each search stops after `SEARCH_WORK` units of machine work and the climb after `CLIMB_WORK` (§2). The clock is read only for the seconds printed in the logs (`_clock`) |
| rule 4: memory of past rows | the last `POOL` = 4,096 read rows (the practice problems) and `ANS_POOL` = 512 answer rows (the round's search), Python lists | **fixed** | registered buffers `pool_rows`, `pool_sizes`, `ans_rows`, `ans_sizes` (and `ans_t` in sp2): in the state dict, counted, on the GPU, written in place in training only. The same rows, in the same order (§3, *keep*) |
| rule 4: the schedule in Python | the reader's stages, phases A–D, the freeze (learning rate 0 after 200 exact steps) | partly | the optimizer's reset at the switch to phase B moved from the scheduler into the optimizer (the scheduler had been changing the optimizer's state). The rest is unchanged: it is a curriculum and a learning-rate schedule, which rule 3 gives the participant, kept as Python state the way any scheduler keeps its own |
| rule 7: the instruction set and the curriculum | digit instructions; the law book (sp2) or the hand-picked practice problems (s3i); the countdown over T | unchanged | that is what the model learns from: changing it changes what is learned. ABLATION_S3.md and SELF_PRACTICE.md say which parts carry the result |
| rule 8(a): zero logits in training | the lessons teach, not the final logits | unchanged | the real answer inside autograd on a training batch does not fit in an H100's memory; on the rows where it would fit it is already exact, so it would teach nothing (§5) |
| rule 8(b): the damped gradient | `GRAD_DAMP` = 0.1 per lookup; underflows to 0 on the deepest paths | unchanged | without it the gradient through 2,848 lookups a round overflows; the path is there, and the test shows which tensors it reaches (§3) |
| rule 8(c): the program held in Python | `ctl.lib`, `ctl.progs`; only the choice among the found programs is a parameter | **fixed** (the holding) | the round programs are buffers: `round_ops` and `round_len` (their instructions as both machines run them) and `round_src` (their text and the definitions they use). The forward pass and the training's checks run them from there; there is no Python-side library or program list for them. The climb's own working state while it searches (its routines, merges and probes) stays Python-side, as an optimizer's state would; what the model keeps and answers with is in the buffers. That the program is a discrete result of search, not a parameter, stays: it is how the model learns |
| rule 8(d): the reading hypothesis | chosen by designed run-detection logic | unchanged | stage 1's reader, as designed; changing it changes what is learned |
| rule 14: the climb inside the forward pass | a program search over many steps; practice problems from the rows' digits and the law book; the miner | unchanged | that is the learning method |
| rule 14: the late table lessons | they join `LATE_AFTER` = 100 steps after the digit laws; "a wrapped x − N is large" (`ANCHOR_WRAP`) among them | **fixed** | removed, with their dead constants (`LATE_AFTER`, `STE_CIN`, `ANCHOR_*`, `ID_PLACE0`, `ID_BORROW`). The table's other checks, which gate the next phase, stay |
| (not in §5) the same fact in the gate | one of the table's checks is "a wrapped x − N is large" | **fixed** | removed from the gate too: the climbs are the same without it (§4) |
| rules 5, 6, 9: the found programs | outside the state dict, so not counted; traced on the host from their text | **fixed** | the programs are counted state, on the GPU: sp2's state is 207,374 parameters and 109,347 numbers of buffers once the round is found (316,721 in all; the records say 289,816, the evaluator's count at the start, before the round's buffers grow). Their trace is made once, when the round is found, and kept; the forward pass reads the instructions from the GPU buffer. The host still launches each instruction's kernels in turn, as any eager PyTorch model does. At evaluation the answer is a function of the state dict and the input alone (§3, *statedict*); the only other thing the evaluation forward touches is a Python-side tally of the subtraction rows it used, printed in the log's AUDIT line, which changes no answer |
| bar 2 (not a rule) | the instruction set and the curriculum | unchanged | as rule 7 |

## 2. The work budgets

**What is counted.** The search runs candidate programs on batches of machines in two ways, and it already chose
between them with a cost model measured on an H100 (s3h): `LOCK_STEP` = 200 units per step of the lockstep machine
plus `LOCK_ROW` = 0.0025 per row and step, against `PRIM_STEP` = 25 units per instruction run alone plus
`PRIM_ROW` = 0.004 per row. The strict files add up that same model as they go: `apply_items` adds the model's
count for its lockstep buckets, and `step` (one instruction run alone, which `apply_items`' other path and every
other machine run go through) adds `PRIM_STEP + rows · PRIM_ROW`. That is `WORK`. It counts the machines' work in
the searches, the vocabularies' signatures and the acceptance tests; it does not count hashing, sorting, listing
programs or BPE, which are bounded by the same caps. It is a deterministic function of the programs and row counts,
so a run computes the same `WORK` on any host.

**Where the clock was, the budget is.** Every place that compared the time with a deadline now compares `WORK`
with a stopping point (`stop_at`, a `WORK` value), with the same effect when it passes (the search stops and keeps
what it found; a vocabulary being built raises `OutOfWork`, and the unit returns at once):
- one search's limit (`SEARCH_S` = 900 s) is `SEARCH_WORK` units from the search's start, reset where the
  certified files reset the 900 s (after the shared vocabularies are built);
- the climb's end (`CLIMB_MARGIN` = 120 s before the evaluator's budget) is `CLIMB_WORK` units from the climb's
  start.

**The values.** Round 1 (§4) ran the strict files with no budgets and the work counted. The count is the same on
every host: sp2's seed 74 climb is 6.629e8 units on two H100s and three B200s. The time a unit takes is not: for
sp2's climb 1.25, 1.39 and 1.57 s per million units on three H100s (round 1's official run; the certified run, whose
climb is the same; the final file's official run), 0.88 to 1.22 on the B200s; for s3i's, 0.84 to 0.96 on H100s.
The values were set after round 1, before the last of those hosts was seen:
- `SEARCH_WORK` = 7e8: twice the largest single search in any run (sp2 seed 3's deep probe at level 5, 3.52e8 units,
  258 s on its B200). On the H100s measured, 7e8 units take 590 s (s3i's fastest rate) to 1,100 s (sp2's slowest):
  about the 900 s it replaces.
- `CLIMB_WORK` = 2.2e9: 1.22 times the largest climb (sp2 seed 3, 1.81e9 units; s3i's climbs are about 1.03e9).
  Spent whole, it is 2,070 s of climb at s3i's H100 rate and 2,750 to 3,450 s at sp2's. Phases A and B take about 85
  s before it, so a climb that used it all ends within the hour on every H100 measured; on the slowest, at about
  3,540 s, with no room left for phase D. Seed 3, the heaviest climb, needs 1.81e9 units: 2,840 s on the slowest H100
  seen, so it would still freeze before the hour there.

The budgets fix how much the model searches, not how long that takes. A host much slower than these takes longer for
the same climb, and the evaluator's deadline may then end training first; the certified files would instead have cut
their searches short there.

## 3. Checks before the runs

All on both strict files, against their certified files. Commands in §7; logs of the Modal ones in `checks/logs/`.

**Local, seconds** ([`checks/strict_local.py`](checks/strict_local.py), CPU, under 1 GB):
- *keep*: the buffers' memory of whole batches gives the same rows as the certified file's lists, on random batch
  sizes (capacities 8, 64 and 4,096, 300 batches each, including batches larger than the capacity).
- *adam*: the optimizer's own reset gives the same parameters as the scheduler's, over 40 steps with two resets.
- *ops*: the machine run from the buffer's instructions (`SteMachine.run_ops`) against the machine run from the
  program's text (`run`), on each round program a certified run found (sp2's two, s3i's four): every register,
  flag and pointer equal, and every table's gradient equal.
- *buffers*: the instructions in `round_ops` are exactly what the trace of each program gives, and `round_src` reads
  back as the programs and the definitions they use (sp2: 15 of the library's 81; s3i: 14 of 40).
- *forward*: with the reader and the cells at the truth (MODE "designed") and those programs, the certified file
  (programs in Python) and the strict file (programs in buffers) give bit-identical logits on H1 rows of depth_t_1,
  depth_t_4 and depth_ood_n_t_2, all exact.
- *statedict*: the model is its state dict. A fresh strict model, from another random start, loaded with a model's
  state dict (parameters and buffers) gives that model's logits exactly; a fresh certified model loaded the same way
  does not, since its programs are not in the state dict. (A loader must size the round's buffers first: they grow
  when the round is found.)
- *grad*: rule 8's gradient test (INTEGRATION.md §5: the final logits' loss, backward, which parameter tensors get a
  gradient), on depth_t_1 rows: the strict file's gradients equal the certified file's, tensor by tensor. Details
  below.

**Training path, on Modal CPU containers** ([`checks/smoke_compare.py`](checks/smoke_compare.py) through
[`checks/modal_smoke.py`](checks/modal_smoke.py)): the reader and cells at the truth, the schedule moved to phase C
(the climb, with small budgets) or to phase D (the round programs given), then 40 evaluator-style training steps on
the same batches, in the certified file and in the strict file. Compared line by line without timings: every climb
line, every loss to 6 decimals, the pool and answer counts, the round's fit, T's direction. With the certified file's
late lessons off (`LATE_AFTER=10**9`, ABLATION_S3.md's no_late) all four comparisons (sp2 and s3i, phases C and D)
are identical. With them on, only the losses differ, by the late lessons' terms. And with budgets made tiny
(`STRICT_SEARCH_WORK`, `STRICT_CLIMB_WORK`), the budgets bind as they should: sp2's probes come back empty and its
climb ends ("nothing learnable"); s3i's searches stop with "work budget at level 4", then at level 3 once the climb's
budget is spent.

**The gradient test.** The strict files' prediction path is the certified files' (equal logits, equal gradients), so
the test answers the same way it does for them; what it shows depends on which program the tournament picks:

| file, programs | objective | tensors with a nonzero gradient | none or zero, and why |
|---|---|---|---|
| sp2, its official two | the right tokens' logits | 13 of 16 | the addition table (`aa`, `ab`) is not in the graph: sp2's round calls no addition; the tournament (`prog`) gets exactly 0 because its two programs give the same answer, so moving weight between them changes nothing |
| sp2, its official two | cross-entropy | 13 of 16 | as above; the rest are tiny (1e-27 to 1e-10) at a perfect prediction |
| s3i, its official four | the right tokens' logits | 14 of 16 | the addition table gets exactly 0: the tournament's argmax calls no addition, and the programs that do have weight 0 in value; the tournament gets 3e-7, which is rounding (its four programs give the same answer) |
| s3i, its official four | cross-entropy | 13 of 16 | also the tournament, as for sp2 |

INTEGRATION.md §5's "all 16" (s3g, with s3f's programs) had an argmax program that calls the addition. With s3f's
four programs and this test's seed, the argmax is one that does not, and the answer is s3i's (14 of 16): every table
the chosen program reads gets a gradient.

## 4. Results on H1

Every run here used the untouched official manifest (`h100_hard_h1.json`; the seed runs change only
`runtime.seeds`), through the shared runner. "Climb line for line" compares every climb, phase and reader line of
the log with the certified file's run of the same seed on the same kind of GPU, timings and work counts left out
([`checks/compare_runs.py`](checks/compare_runs.py)). "Frozen by" is the evaluator's progress line just after the
freeze: it depends on the host, since the climb runs mostly on the host's CPU.

### 4.1 Round 1: no budgets, the work counted

The strict builds with both budgets at 0 (none), in `records/r1/`, built by `make_strict.py --r1`: sp2 with the
gate's wrap check kept (`r1/submission_stage3_sp2_strict_r1.py`, md5 `bf083d42`) and without it
(`r1/submission_stage3_sp2_strict_nowrap_r1.py`, `192c6a06`), and s3i with it kept
(`r1/submission_stage3_s3i_strict_r1.py`, `b428f40d`). Every run is exact everywhere: T = 64 on both ladders,
every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002.

| record | GPU | seed | round at step (certified) | frozen at step (certified) | climb line for line | frozen by s (certified) | climb work, and its seconds | largest search |
|---|---|---|---|---|---|---|---|---|
| `sp2s_r1_official` | H100 | 74 | 688 (688) | 887 (887) | identical | 922 (1,077) | 6.63e8 in 828 s | 7.5e7 in 67 s |
| `sp2s_r1_b200_seed74` | B200 | 74 | 688 (688) | 887 (887) | identical | 919 (729) | 6.63e8 in 812 s | 7.5e7 in 61 s |
| `sp2s_r1_b200_seed1` | B200 | 1 | 630 (630) | 829 (829) | identical | 832 (666) | 6.22e8 in 732 s | 7.6e7 in 66 s |
| `sp2s_r1_b200_seed2` | B200 | 2 | 683 (683) | 882 (882) | identical | 968 (770) | 7.17e8 in 867 s | 5.9e7 in 53 s |
| `sp2s_r1_b200_seed3` | B200 | 3 | 610 (610) | 809 (809) | identical | 1,696 (1,804) | 1.81e9 in 1,597 s | 3.5e8 in 258 s |
| `sp2s_nowrap_r1_b200_seed74` | B200 | 74 | 688 (688) | 887 (887) | identical | 705 (729) | 6.63e8 in 641 s | 7.5e7 in 56 s |
| `sp2s_nowrap_r1_b200_seed1` | B200 | 1 | 630 (630) | 829 (829) | identical | 794 (666) | 6.22e8 in 703 s | 7.6e7 in 64 s |
| `sp2s_nowrap_r1_b200_seed3` | B200 | 3 | 610 (610) | 809 (809) | identical | 1,774 (1,804) | 1.81e9 in 1,667 s | 3.5e8 in 261 s |
| `s3is_r1_official` | H100 | 74 | 671 (671) | 870 (870) | identical | 1,077 (1,143) | 1.03e9 in 968 s | 1.3e8 in 92 s |

So the strict changes leave every climb as it was, line for line, with the same round, the same tournament, the same
T resolution and the same freeze step (the frozen line prints the tournament's weights: equal too). Only the losses
printed in the log differ (the late lessons' terms), and the time: these B200 hosts took 0.94 to 1.26 times the
certified runs' time for the same work. The gate without its wrap check was the same on all three seeds it ran, so
the final files drop it too.

### 4.2 The final files

The final builds:
- [`submission_stage3_sp2_strict.py`](submission_stage3_sp2_strict.py), md5 `1a1f15fb231d6da5fd53d530bc9bdcb6`,
  254,804 bytes;
- [`submission_stage3_s3i_strict.py`](submission_stage3_s3i_strict.py), md5 `11baebb4265de98a395033ae90a3cebe`,
  177,657 bytes.

Each has every edit of §1, the gate's wrap check removed and the budgets set (§2). Records in `records/`. All 17
pass `tools/verify_records.py`: the embedded source's md5 against the file, the manifest official except for the
seed, the evaluator's RESULT_JSON line, every rung and split. Every run is exact everywhere: T = 64 on both ladders,
every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002. "Budget stops" counts the log's "work budget"
lines: no search and no climb ran into a budget.

**sp2 strict**, against the certified sp2's runs (the official H100 run and the eight B200 seeds of
SELF_PRACTICE.md §4.2):

| record | GPU | seed | round at step (certified) | frozen at step (certified) | climb line for line | frozen by s (certified) | training s | climb work, and its seconds | largest search | budget stops |
|---|---|---|---|---|---|---|---|---|---|---|
| `sp2s_official` | H100 | 74 | 688 (688) | 887 (887) | identical | 1,174 (1,077) | 1,377 | 6.63e8 in 1,042 s | 7.5e7 in 72 s | 0 |
| `sp2s_b200_seed74` | B200 | 74 | 688 (688) | 887 (887) | identical | 696 (729) | 817 | 6.63e8 in 634 s | 7.5e7 in 56 s | 0 |
| `sp2s_b200_seed1` | B200 | 1 | 630 (630) | 829 (829) | identical | 635 (666) | 745 | 6.22e8 in 577 s | 7.6e7 in 58 s | 0 |
| `sp2s_b200_seed2` | B200 | 2 | 683 (683) | 882 (882) | identical | 743 (770) | 861 | 7.17e8 in 682 s | 5.9e7 in 47 s | 0 |
| `sp2s_b200_seed3` | B200 | 3 | 610 (610) | 809 (809) | identical | 1,826 (1,804) | 2,044 | 1.81e9 in 1,714 s | 3.5e8 in 266 s | 0 |
| `sp2s_b200_seed4` | B200 | 4 | 696 (696) | 895 (895) | identical | 644 (981) | 757 | 6.30e8 in 582 s | 7.6e7 in 58 s | 0 |
| `sp2s_b200_seed5` | B200 | 5 | 637 (637) | 836 (836) | identical | 630 (669) | 751 | 6.14e8 in 570 s | 5.0e7 in 39 s | 0 |
| `sp2s_b200_seed6` | B200 | 6 | 600 (600) | 799 (799) | identical | 730 (897) | 852 | 6.82e8 in 626 s | 5.7e7 in 46 s | 0 |
| `sp2s_b200_seed7` | B200 | 7 | 684 (684) | 883 (883) | identical | 753 (969) | 868 | 7.51e8 in 690 s | 7.7e7 in 58 s | 0 |

The strict sp2 file certifies on the official H100 manifest and on all eight seeds, and on every one of them it
learns what the certified file learned: the same practice problems in the same order, the same routines and merges,
the same round (`LDA.X STA.Y T.mul.x,o.XY STA.X LDA.N T.red.x+on.XA` and its X/Y twin), the same tournament weights,
the same freeze step. The official run's H100 was a slow host (1.57 s per million units, against 1.25 in round 1),
which is why it froze later than the certified run's; the work is the same.

**s3i strict**, against s3i's official run and its seven H100 seeds (`integration/records/s3i_official.json`,
`s3i_h100_seed1` … `seed7`):

| record | GPU | seed | round at step (certified) | frozen at step (certified) | climb line for line | frozen by s (certified) | training s | climb work, and its seconds | largest search | budget stops |
|---|---|---|---|---|---|---|---|---|---|---|
| `s3is_official` | H100 | 74 | 671 (671) | 870 (870) | identical | 1,069 (1,143) | 1,278 | 1.03e9 in 960 s | 1.3e8 in 94 s | 0 |
| `s3is_h100_seed1` | H100 | 1 | 613 (613) | 812 (812) | identical | 954 (1,069) | 1,113 | 1.03e9 in 864 s | 1.4e8 in 92 s | 0 |
| `s3is_h100_seed2` | H100 | 2 | 663 (663) | 862 (862) | identical | 959 (982) | 1,109 | 1.03e9 in 868 s | 1.4e8 in 92 s | 0 |
| `s3is_h100_seed3` | H100 | 3 | 584 (584) | 783 (783) | identical | 939 (1,002) | 1,088 | 1.02e9 in 851 s | 1.4e8 in 93 s | 0 |
| `s3is_h100_seed4` | H100 | 4 | 679 (679) | 878 (878) | identical | 937 (998) | 1,082 | 1.02e9 in 843 s | 1.4e8 in 93 s | 0 |
| `s3is_h100_seed5` | H100 | 5 | 619 (619) | 818 (818) | identical | 960 (1,094) | 1,115 | 1.05e9 in 871 s | 1.5e8 in 97 s | 0 |
| `s3is_h100_seed6` | H100 | 6 | 580 (580) | 779 (779) | identical | 970 (1,149) | 1,118 | 1.01e9 in 840 s | 1.3e8 in 83 s | 0 |
| `s3is_h100_seed7` | H100 | 7 | 665 (665) | 864 (864) | identical | 949 (1,147) | 1,112 | 1.03e9 in 853 s | 1.4e8 in 91 s | 0 |

The strict s3i file certifies on the official manifest and on all seven more H100 seeds, each run its certified run
line for line. These runs froze 2 to 17 % sooner than the certified ones on the same work: the hosts, and the late
lessons' compute, which the strict file no longer does.

### 4.3 How hard it was tried, and what it cost

Two rounds through the shared runner: round 1, nine runs of the round-1 builds (§4.1); round 2, seventeen runs of the
final files (§4.2). No run failed or was repeated. Before them: the local checks (§3), four smoke comparisons and two
budget-binding smokes on Modal CPU containers, and one H100 measurement (§5). Modal's list price for the 26
evaluator runs is about $50, plus a few dollars of checks.

## 5. Real logits in training (rule 8(a)): not done, and why

In training the forward pass returns zero logits; the lessons teach. REVIEW.md §5 asks whether the training forward
could return the model's real answer instead (the current round program's, through the same straight-through path as
evaluation), so that the evaluator's loss reaches the parameters through the final logits.

**What it costs.** [`checks/logit_cost.py`](checks/logit_cost.py) runs the strict sp2 file's prediction path inside
autograd on H1 training rows, then a cross-entropy on the final logits and its backward, on one H100 (the reader
and the cells at the truth, sp2's official round programs: 6,602 instructions a round, 2,848 of them table lookups):

| rows | T | programs run | forward | backward | peak GPU memory |
|---|---|---|---|---|---|
| 48 | 1 | both | 0.7 s | 1.3 s | 1.0 GiB |
| 48 | 1 | the argmax only | 0.3 s | 0.7 s | 0.5 GiB |
| 512 | 1 | the argmax only | 0.4 s | 0.7 s | 5.3 GiB |
| 64 | 16 | the argmax only | 5.9 s | 10.5 s | 10.3 GiB |
| 512 | 16 | the argmax only | out of memory (80 GB) | | |

A training batch is 512 rows with T up to 16, so the real answer on every row does not fit in an H100's memory, and
on 64 rows it already costs 16 s a step, against about 0.25 s for a whole training step of phase D now (sp2's
official run: 199 steps from the round to the freeze in 48 s). Every round keeps its
2,848 lookups' activations for the backward pass. The batch's answer rows alone (the rows of its smallest T, at most
48 at T = 1) would fit: 1 to 2 s a step, so +200 to +400 s over phase D's 200 steps.

**Why that partial version was not built.** From phase D on, the round program reproduces those rows exactly (the
climb accepted it on such rows), so the cross-entropy on them is about 0 and so is its gradient. It would teach
nothing, at a cost of several hundred seconds a run. Before phase D there is no program; the model's answer is x
itself, and a cross-entropy on it would push the reader to misread x. So rule 8(a) stays as it was: the lessons teach,
and the final logits are produced by the model, inside autograd, only at evaluation (where §3's gradient test shows
the path).

## 6. s3i

The same edits apply to s3i ([`submission_stage3_s3i_strict.py`](submission_stage3_s3i_strict.py)): the two files share
libcore and the template, and differ only in the climb (libcore's `Climb` over the nine designed practice problems,
against `spcore.SelfClimb`) and in the answers' T (s3i keeps T = 1; sp2 the smallest T seen, `ans_t`). In
libcore's `Climb` one search covers a task's three tries (routines, library, primitives) under one budget, as the
900 s did, and a search stopped by the climb's budget does not count as a used-up cap, as before. It was cheap: the
same builder, the same checks (§3, all passing), and §4's runs; its records are in §4.2's second table. s3i's climbs
do 1.01e9 to 1.05e9 units each, under half of `CLIMB_WORK`, and its largest search 1.5e8, about a fifth of
`SEARCH_WORK`.

## 6a. Inherited, not changed: the reduction's acceptance check

The strict files keep the certified files' acceptance check for the reduction as it is. That check is weak: it can
pass a repeated subtraction that is right only for quotients up to 600, when none of its 60 three-digit probes has a
larger quotient, about 3% of pools on H1's rows (`integration/ablation_s3/reduce_check.py`, from the s3i
ablation's no_step arm). None of the 26 runs here met it: each one found the certified run's long division and
certified on unseen moduli up to T = 64. A fix for s3i and sp2 is being made elsewhere; adopting it means building
the strict files from the fixed sources with `make_strict.py` (its md5 asserts name the files it reads) and running
§4.2 again.

## 7. Files and how to rerun

Everything is in `cpu_first/integration/strict/`; nothing outside it was changed.
- `make_strict.py`: builds the strict files from the certified ones (asserting their md5s), as named exact edits
  (`clock`, `pools`, `round`, `adam`, `late`, `wrap`; `--keep-wrap` leaves out `wrap`, and `--r1` builds round 1's
  files, without budgets, into `r1/`), then the clock's identifiers renamed
  (`deadline` → `stop_at`, `Deadline` → `OutOfWork`, `_late` → `_spent`, `search_s` → `search_work`,
  `time.time()` → `_clock()`), then checks that nothing of the old mechanism is left, that the file parses and that
  it is within 256 KiB. `NAME=VALUE` sets a constant (the budgets are set in the script's `BUDGETS`).
- `submission_stage3_sp2_strict.py`, `submission_stage3_s3i_strict.py`: the strict files (md5s in §4).
- `checks/strict_local.py`: §3's local checks. `checks/smoke_compare.py`: the training path against the certified
  file; `checks/modal_smoke.py`: runs a check script on a Modal CPU container (or `--gpu`, an H100), logs in
  `checks/logs/`. `checks/logit_cost.py`: §5's measurement. `checks/lib_from_record.py`: a record's library and
  round programs as JSON (`libs/`). `checks/compare_runs.py`: two records, climb line by line.
  `checks/work_table.py`, `checks/results_table.py`: §2's and §4's tables.
- `records/`: the evaluator's records (`r1/`: the first round, budgets off; the rest: the final files).

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python integration/strict/make_strict.py sp2 s3i
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python integration/strict/make_strict.py sp2 s3i --r1 --keep-wrap --out /tmp/r1 && ~/github/sotaku/venv/bin/python integration/strict/make_strict.py sp2 --r1 --out /tmp/r1 && md5 /tmp/r1/*.py integration/strict/r1/*.py
```

The official records against the files (the competition's checks: md5, manifest, RESULT_JSON, every rung and split):

```bash
cd ~/github/one-layer/cpu_first && python3 tools/verify_records.py integration/strict/records/sp2s_official.json=integration/strict/submission_stage3_sp2_strict.py integration/strict/records/s3is_official.json=integration/strict/submission_stage3_s3i_strict.py
```

A strict record against the certified run of the same seed, and the tables of §4:

```bash
cd ~/github/one-layer/cpu_first && python3 integration/strict/checks/compare_runs.py integration/self_practice/records/runner/submission_stage3_sp2_official.json integration/strict/records/sp2s_official.json
```

```bash
cd ~/github/one-layer/cpu_first && python3 integration/strict/checks/results_table.py sp2 integration/strict/records/sp2s_*.json && python3 integration/strict/checks/results_table.py s3i integration/strict/records/s3is_*.json
```

The runs (the shared runner; the seeds' `--cap` keeps this study at 8 GPUs at a time):

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/strict/submission_stage3_sp2_strict.py --name sp2s_official --out integration/strict/records --run strict-sp2-official
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/strict/submission_stage3_sp2_strict.py --seeds 8 --memory-gb 48 --cap 4 --name sp2s --out integration/strict/records --run strict-sp2-seeds
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode official --file integration/strict/submission_stage3_s3i_strict.py --name s3is_official --out integration/strict/records --run strict-s3i-official
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run --detach tools/modal_runner.py --mode seeds --file integration/strict/submission_stage3_s3i_strict.py --seeds 1,2,3,4,5,6,7 --hardware h100 --cap 2 --name s3is --out integration/strict/records --run strict-s3i-h100-seeds
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python integration/strict/checks/strict_local.py KIND=sp2
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run integration/strict/checks/modal_smoke.py --script smoke_compare --args "KIND=sp2 STEPS=40 SP_FALLBACK=30000 SP_DEEP=30000 SP_PROBE=30000 ROUND_STATES=20000 SP_LEVELS=2 CERT_LATE_AFTER=10**9;KIND=sp2 STEPS=40 PHASE=D LIB=integration/strict/libs/sp2_official_lib.json CERT_LATE_AFTER=10**9" --tag smoke
```

```bash
cd ~/github/one-layer/cpu_first && ~/github/sotaku/venv/bin/python -m modal run integration/strict/checks/modal_smoke.py --script logit_cost --args "KIND=sp2 CASES=48x1,48x1a,512x1a,64x16a,512x16a" --tag logit_cost --gpu
```

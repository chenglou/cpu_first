# For a reviewer: what is claimed, the file to judge, and what to rule on

*Written 2026-10-23 for an outside judge of One Layer Deeper H1 (learn x^(2^T) mod N from prompt/answer pairs). It
points into [OVERVIEW.md](OVERVIEW.md) and [INTEGRATION.md](integration/INTEGRATION.md) instead of repeating their
tables. The checks in §3 were rerun on 2026-10-23. Nothing here has been submitted to the hosted evaluator.*

The rules are the competition's own: [one-layer-deeper/README.md §Rules](../one-layer-deeper/README.md#rules), at
commit `87e5c26` ("Releasing H1"). Links into `../one-layer-deeper/` work once that clone sits next to this folder.

## 1. What is claimed

The project measures itself against three bars ([OVERVIEW §4](OVERVIEW.md#4-where-it-stands-three-bars-and-the-models-at-the-frontier)):

1. **Certify H1 with learned parts: done.** Every step of the removal ladder certifies T = 64 on both ladders of the
   untouched official manifest on an H100. From step 11 on, no hand-written arithmetic runs in training or in the
   answer: every table that computes is learned from a random start, though until step 17 the datapath around those
   tables is still written by hand.
2. **Nothing a strict judge would call the hand-written algorithm: partly cleared.** The best file, s3i, has no
   hand-written datapath and no menu of wirings: the program its forward pass runs ("copy x, multiply, reduce") was
   found by search inside the training run. But the instruction set that search runs over (32 digit instructions) and
   the curriculum that walks it up the pieces of long division and the schoolbook multiply (which practice problems,
   in which order) were written by hand, knowing the task family
   ([INTEGRATION §6](integration/INTEGRATION.md#6-is-bar-2-cleared)). Two later studies (§7) narrow this down. An
   ablation on the official manifest shows what the result rests on: two task-shaped instructions, the practice
   problems that define long division's step, how the multiply's row and the registers are posed, and two parts of the
   search. The curriculum's order only saves time. And a follow-up file that poses its own practice problems from a
   general law book, in place of the hand-picked list, also certifies on the official manifest.
3. **A general algorithm learner: results at a tiny size only.** A learner that picks its own practice problems
   invents x² mod N for x < N ≤ 7 (and ≤ 26) on a bare counting machine; unchanged, it invents 4 of 9 other formulas,
   and 2 more once its solved formulas carry over ([PRACTICE.md](bar3/PRACTICE.md), [BREADTH.md](bar3/BREADTH.md)).
   Nothing at H1's size.

Four rules are **gray** for s3i: 4, 7, 8 and 14 (§5). The write-ups flag them; they do not argue them away.

Not claimed:
- nothing was submitted to the hosted evaluator, before or after the deadline;
- the leaderboard count in [REPORT.md](REPORT.md) §0 (3 of 146 entries certified) comes from the public leaderboard at
  the time and has no record behind it;
- runs on B200s, CPUs or relaxed budgets are research; only official-manifest H100 runs count as certification here.

## 2. The file to judge, and the others

**[`integration/submission_stage3_s3i.py`](integration/submission_stage3_s3i.py)**, md5
`0e8131440e1842546c67e7378caf472d` ("s3i", step 17 of the removal ladder). 3,437 lines, 173,817 bytes (the limit is
256 KiB), 207,374 learned numbers in 16 parameter tensors, Adam. It is a generated file:
[`make_stage3.py`](integration/make_stage3.py) builds it from [`stage3_template.py`](integration/stage3_template.py)
(1,008 lines), [`lib/libcore.py`](integration/lib/libcore.py) (1,877 lines) and stage 1's reader, and rebuilding it
from those sources gives the same bytes (§4.3). Its docstring (lines 1–35) is the best one-page summary. Two of its
practice checks are weak: on about 3% of runs, none of its 8 certified seeds among them, the reduction's accepts a
shortcut that fails at full size, and the conditional subtraction's never tests x between N and 2N. A fixed copy,
[`fix/submission_stage3_s3i_fix.py`](integration/ablation_s3/fix/submission_stage3_s3i_fix.py) (md5
`09a8263cf09321fef9ba3d4ffbc2aea6`), only adds the missing probes and replays all 8 certified runs field for field
(§7).

The other two frontier files ([OVERVIEW §4.1](OVERVIEW.md#41-the-models-at-the-frontier) compares all three):
- [`integration/submission_stage1.py`](integration/submission_stage1.py), md5 `cfd7b61d7d3d4b76c606606ba8b8d9db`:
  the tournament and the general lessons in one file. Its forward pass is still the hand-written H1 datapath, with
  menus that hold the right wirings (rule 7 is gray as for its parents).
- [`general_lessons/submission_general_v3.py`](general_lessons/submission_general_v3.py), md5
  `0805b7d38f9fe4fa15b182c1b481886e`: general lessons v3, where the answers choose whole digit codes. Its 8 H1 seeds
  ran an earlier build, `v3f` (md5 `32a1d154`), on B200s; the final file's own H1 run is the official one.

Finished after this file was first written ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md), §7):
- [`integration/self_practice/submission_stage3_sp2.py`](integration/self_practice/submission_stage3_sp2.py), md5
  `afc70fe91069a85c416eba18b48ee30e`: s3i with its curriculum replaced by self-posed practice. 250,643 bytes. It
  certifies on the official manifest and on all 8 B200 seeds. Its parent,
  [`submission_stage3_sp.py`](integration/self_practice/submission_stage3_sp.py) (md5 `f0ef5b069ef2e1bd9dc337798d246ba4`),
  certifies on the official manifest and on 7 of 8 seeds. s3i stays the file to judge here, since it has the most
  evidence (the official run, seven more H100 seeds and the ablation). sp2 is the one to look at for how far bar 2
  goes.

Every earlier certified file is a step of the removal ladder ([OVERVIEW §5](OVERVIEW.md#5-the-removal-ladder)), each
with its md5 and record.

## 3. The official records, and how to check them

| file | official record | other seeds on the official manifest |
|---|---|---|
| s3i | [`integration/records/s3i_official.json`](integration/records/s3i_official.json) | `integration/records/s3i_h100_seed1.json` … `seed7.json` (H100s) |
| stage 1 | [`integration/records/s1_official.json`](integration/records/s1_official.json) | CPU seeds: `integration/records/s1_cpu_seed*.json` |
| general v3 | [`general_lessons/fixed/records/v3k_h1_official.json`](general_lessons/fixed/records/v3k_h1_official.json) | `v3f` build on B200s ([FIXED_MODULUS.md](general_lessons/fixed/FIXED_MODULUS.md)) |

Careful with one name: `runs/learned_remainder/s1_official.json` is a different, failed run (the removal ladder's
first recipe for step 1), not stage 1's.

**What a record holds.** One JSON per evaluator run: `submission` (the exact source sent, as `submission.py`),
`manifest` (as passed; the shared runner's records also name `manifest_file`), the evaluator's full `stdout`, whose
last `RESULT_JSON=` line is the evaluator's own result, `result` (that line, parsed), `gpu`, `returncode`, and for
records written by [`tools/modal_runner.py`](tools/modal_runner.py) also `submission_md5` (checked inside the
container before the run) and `runner.data_md5`, an md5 over the generated H1 folder the run read.

**One command** (standard library only, from this folder, with the competition clone next to it):

```bash
python3 tools/verify_records.py
```

For each of the ten records above (s3i's official run and seeds 1–7, stage 1, general v3) it prints: the md5 of the
embedded source against the file's md5 now; the manifest against the competition's `h100_hard_h1.json` (identical,
or differing only in `runtime.seeds` for the seed runs); whether the `RESULT_JSON=` line in `stdout` equals `result`;
T on both ladders; every rung's count; the three splits; the times. On 2026-10-23 all ten checked out: md5s equal,
T = 64 on both ladders, every rung 768/768, test 9,999/9,999, ood_t and ood_n_t 10,002/10,002. Any other record:
`python3 tools/verify_records.py REC.json=FILE.py`. For example, the self-practice files' official records (both
check out the same way):

```bash
python3 tools/verify_records.py integration/self_practice/records/runner/submission_stage3_sp2_official.json=integration/self_practice/submission_stage3_sp2.py
python3 tools/verify_records.py integration/self_practice/records/runner/submission_stage3_sp_official.json=integration/self_practice/submission_stage3_sp.py
```

Or by hand, for one record:

```bash
python3 -c "import hashlib,json; r=json.load(open('integration/records/s3i_official.json')); print(hashlib.md5(r['submission'].encode()).hexdigest())"
md5 integration/submission_stage3_s3i.py      # Linux: md5sum
```

**What to look for in s3i's log** (`stdout` of the record): `LR climb step=671 L6 round: cost 5` (the round found, as
one of several equivalent spellings), `LR frozen step=870` (the freeze), the `ROUND` line (the four programs the
tournament holds, with their weights), and the `SCHEDULE … history=[…]` line (every phase change, with its step).

Three things that look odd but are not:
- `exact_accuracy` reads 0.9999999892 on ood_t and ood_n_t although `correct_examples` is 10,002 of 10,002. The
  evaluator sums float32 batch means, and the shortfall is exactly one float32 rounding (2⁻²⁴) on the last batch of
  1,810 rows. The counts are what certification uses.
- `final_train_loss` is −0.0: after the freeze the loss is the parameters' sum times 0.0, exactly zero with the sum's
  sign.
- Training reports 1,340.3 s, though the model froze at step 870: the evaluator keeps stepping, at learning rate 0,
  to the file's 20,000 steps. In `structured_metrics` the progress line just after the freeze (step 900) reads
  1,143.0 s.

[OVERVIEW §9](OVERVIEW.md#9-where-docs-and-records-disagree) lists every place a write-up and a record disagree.

## 4. How to reproduce

### 4.1 With the competition's own evaluator (one H100, no Modal)

```bash
git clone https://github.com/tilde-research/one-layer-deeper.git && cd one-layer-deeper && git checkout 87e5c26
uv venv .venv && source .venv/bin/activate && uv sync
bash scripts/generate_datasets.sh        # every tier; the H1 block alone (lines 282-301) is enough
python3 ../cpu_first/tools/verify_records.py --data data/generated/squaring_mod_final_variable_n_t_grouped_bidirectional_plus1k_t1_t2
mkdir -p /tmp/s3i && cp ../cpu_first/integration/submission_stage3_s3i.py /tmp/s3i/submission.py
CUDA_VISIBLE_DEVICES=0 python -m benchmark.runner --manifest benchmark/manifests/h100_hard_h1.json \
  --submission-file /tmp/s3i/submission.py --include-structured-metrics | tee s3i_run.log
python3 ../cpu_first/tools/verify_records.py --log s3i_run.log
```

- The `--data` line should print `a6414ec9fc6a5efce6414895b52d4eb7`, the H1 folder every shared-runner record read
  (this checkout's own H1 folder gives it too).
- The copy only gives the file rule 1's name; the bytes are unchanged.
- The run takes about 26 minutes (the official record: 1,557 s of wall time).
- `--log` compares your `RESULT_JSON=` line with `s3i_official`: the certified T, every rung and the splits. Times
  will differ: the eight H100 runs trained in 1,133–1,350 s.
- The run's path is reproducible in practice: seed 74 on a B200
  (`integration/records/b200/s3i_b200_seed74.json`) found the round at step 671 and froze at step 870, with the same
  four round programs and the same tournament weights as the H100 run.
- But the climb reads the wall clock (§5, rule 4). A much slower host can change how far a search gets before its
  time limit; on an H100 the searches end well inside their limits (the round's took 26.6 s).

### 4.2 With our runner (Modal)

[`tools/modal_runner.py`](tools/modal_runner.py) builds the competition's image (CUDA 12.9.1, Python 3.13,
`uv sync --frozen` on the clone's lock file), regenerates every dataset with `scripts/generate_datasets.sh`, sends the
file as bytes, checks its md5 in the container, and runs the untouched manifest on an H100. It needs a Modal account,
the Modal client 1.2.6, and the competition clone at `../one-layer-deeper` (or `ONE_LAYER_DEEPER=<path>`).

```bash
python -m modal run --detach tools/modal_runner.py --mode official --file integration/submission_stage3_s3i.py --name s3i_judge --out /tmp/judge_records --run s3i-judge
python tools/modal_runner.py --mode fetch --out /tmp/judge_records --run s3i-judge --wait
python tools/modal_runner.py --mode compare integration/records/s3i_official.json /tmp/judge_records/s3i_judge.json
```

[tools/README.md](tools/README.md) has the other modes, the limits, and the runner's own validation against stored
records. The write-ups' commands say `~/github/sotaku/venv/bin/python`: that is the author's virtualenv, and any
Python with torch, numpy and that Modal client does.

### 4.3 Cheap checks (seconds, CPU)

- **s3i is built by its sources.** Without writing anything:

  ```bash
  python3 -c "import importlib.util as u, hashlib; s=u.spec_from_file_location('m','integration/make_stage3.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(hashlib.md5(m.build().encode()).hexdigest())"
  ```

  It prints s3i's md5. (`python3 integration/make_stage3.py s3h` checks the previous build, s3h, the same way.)
- **The competition's static source policy** (the check the CLI and the evaluator run before importing a file):

  ```bash
  python3 -c "import sys; sys.path.insert(0, '../one-layer-deeper'); from submission_validation import validate_submission_source as v; print(v('submission.py', open('integration/submission_stage3_s3i.py').read(), 256 * 1024))"
  ```

  It passes for all three frontier files.
- **Plumbing and gradient checks** (reader and cells at the truth, the prediction path, the gradient test of rule 8):
  [INTEGRATION §7](integration/INTEGRATION.md#7-reproduce).

## 5. The gray areas: what a judge should rule on

[OVERVIEW §6](OVERVIEW.md#6-rules-116-for-the-best-model) goes through rules 1–16 for s3i, and
[INTEGRATION §5](integration/INTEGRATION.md#5-rules-7-8-and-14) sets rules 7, 8 and 14 for stage 1 and stage 3 side by
side. The questions, with where to look in s3i:

1. **Rule 4: the evaluator owns the loop, the cadence and the deadline.** The evaluator is untouched: one forward,
   one loss and one backward per step. But the model keeps its own schedule as Python state (`Control`, line 2349:
   the reader's stages, phases A–D, then a freeze that sets the learning rate to 0 once every check has held for 200
   steps, `FREEZE_WIN`). It also reads the clock: `build_optimizer` (line 3425) sets a deadline from the budget the
   evaluator passes in; each search stops after `SEARCH_S` = 900 s (line 92); the climb stops `CLIMB_MARGIN` = 120 s
   before the budget ends (line 97). And it keeps rows across batches: up to `POOL` = 4,096 read rows for its
   practice problems and `ANS_POOL` = 512 answered rows with T = 1 for the round's search (lines 110–111).
   *Is a model that reads the clock, stops its own work at the deadline, and keeps a memory of past rows within
   rule 4?*
2. **Rule 7: no hard-coded algorithm in the forward pass.** No datapath and no menu is in the file; the forward pass
   runs the program training found, on learned tables. But the instruction set is digit arithmetic
   (subtract-with-borrow, add-with-carry and multiply-accumulate at a pointer, keep-or-discard on the borrow, bring
   the top digit in, loops by the number's length), the practice problems (made from the law library, `LAWS` at
   line 140, in the order `NEEDS` at line 1413 sets) walk the search up long division's and the schoolbook
   multiply's pieces, two of them are definitions of long division's step, and the countdown over T is hand-written.
   *Did the algorithm move into the instruction set and the curriculum, and does that count as hard-coding it?*
3. **Rule 8: final logits from the model, inside autograd, with an unbroken gradient path.** In the gradient test
   every one of the 16 parameter tensors reaches the final logits. That test ran on the earlier build s3g; s3h kept
   s3g's prediction path, and s3i's changes are in phase D's checking machine and the search (INTEGRATION §3, §5).
   But:
   - (a) in training the forward returns zero logits, so the lessons teach, not the final logits;
   - (b) the gradient is damped per lookup (`GRAD_DAMP`, line 53) and underflows to exactly 0 on the deepest paths at
     a perfect prediction;
   - (c) the answering program is a discrete result of search, held in Python (`model.ctl.lib` and
     `model.ctl.progs`), and only the choice among four found programs is a parameter (`prog`, one logit each);
   - (d) which reading hypothesis a prompt uses is decided by designed run-detection logic.

   *Is that "produced entirely by the submitted model from its inputs and learned PyTorch state"?*
4. **Rule 14: no data inspection, task-specific solvers, custom training loops, or hidden training work.** No
   backward or grad call, no custom autograd function, no file read, no number made up. But the climb is a program
   search inside the training forward pass, run over many steps; its practice problems are made from the rows'
   own digits and the law library; a miner computes statistics over the training batches; and the lessons include
   the old table lessons, joining `LATE_AFTER` = 100 steps after the digit laws, with the width-dependent fact "a
   wrapped x − N is large" among them (`ANCHOR_WRAP`, line 70). *Is the curriculum a task-specific solver written as
   lessons, and is an in-forward search "hidden training work"?*
5. **The instruction set and the curriculum, as a matter of bar 2** (the project's own bar, not a rule): which parts
   would you call the algorithm? [INTEGRATION §4](integration/INTEGRATION.md#4-what-is-still-hand-made) lists
   everything hand-made, against stage 1's.

Worth a word too: rules 5, 6 and 9 as they apply to the found programs. The library lives in Python outside the
state dict, so the 207,374 counted numbers do not include it, and each program is traced on the host from its text
into opcode columns on the GPU. OVERVIEW §6 marks these rules "yes"; a strict reader may want to say so explicitly.

For stage 1 and general v3, the differences are listed under OVERVIEW §6's table ("The other frontier files").

## 6. An hour's reading order

| minutes | read | for |
|---|---|---|
| 0–5 | this file, §1–3 | the claims and the evidence |
| 5–15 | OVERVIEW: "In one minute", [§1](OVERVIEW.md#1-the-task-in-plain-words), [§2](OVERVIEW.md#2-why-it-is-hard) | the task, and why exact answers defeat ordinary training |
| 15–25 | OVERVIEW [§4](OVERVIEW.md#4-where-it-stands-three-bars-and-the-models-at-the-frontier) and [§4.7](OVERVIEW.md#47-discovering-the-skeleton-library-learning-then-stage-3) | the bars, the frontier, what s3i still writes by hand |
| 25–40 | INTEGRATION: ["In plain words"](integration/INTEGRATION.md#in-plain-words), [§4](integration/INTEGRATION.md#4-what-is-still-hand-made), [§5](integration/INTEGRATION.md#5-rules-7-8-and-14), [§6](integration/INTEGRATION.md#6-is-bar-2-cleared) | what one training run learns, what is hand-made, the rules, the bar-2 call |
| 40–50 | s3i itself: the docstring (lines 1–35), the constants (51–139), `LAWS` (140), the practice tasks (1412), `Control` (2349), `LibraryMachine` (2394, `forward` at 2749), the loss, the schedule and `SUBMISSION` (3380–3437) | where each gray point lives |
| 50–55 | `python3 tools/verify_records.py`, and the log lines of §3 | the record against the file |
| 55–60 | OVERVIEW [§6](OVERVIEW.md#6-rules-116-for-the-best-model), [§7](OVERVIEW.md#7-status-and-open-questions), [§9](OVERVIEW.md#9-where-docs-and-records-disagree) | every rule, the open questions, the known disagreements |

With more time: OVERVIEW [§5](OVERVIEW.md#5-the-removal-ladder) (each hand-written piece removed in turn, and how hard
stage 3 was pushed), [INTEGRATION §3](integration/INTEGRATION.md#3-stage-3-no-skeleton-no-menus) (every stage-3
build, including the failures), and bar 3's [PRACTICE.md](bar3/PRACTICE.md) and [BREADTH.md](bar3/BREADTH.md).
[README.md](README.md) indexes every file and folder.

## 7. Later studies

Three studies were still running when this file was first written. All three have finished and are committed, and
later ones follow them as they land. None changes a certified file or record.

- **The s3i ablation** ([ABLATION_S3.md](integration/ABLATION_S3.md), code and records in
  [`integration/ablation_s3/`](integration/ablation_s3/)). Each hand-made part was removed in turn and the model
  retrained on the official manifest, one H100 per arm, on seed 74 and on seed 1. The unchanged file, run beside the
  arms, replays s3i's own runs on both seeds exactly.
  - **Needed on both seeds** (without it the round is never found, each time at the same level and missing the same
    piece): SEL.S (keep or discard on the borrow); BRG.X (bring X's next digit in); the two practice problems that
    define long division's step; mac posed with the running sum; the register twins; routines.
  - **Needed only because a check was weak:** Occam's tie-break toward fewer executed steps. It kept the search off
    a wrong conditional subtraction that the weak check let through; with the check fixed, s3i certifies without it
    on both seeds.
  - **Needed on one seed of two:** the chain problem alone. Without it the reduction depends on a tie between two
    equal-cost spellings, which the search's listing order breaks.
  - **Speed only** (it still certifies on both seeds, frozen by 1,584 s at the latest): the curriculum's order
    (+277 s and +321 s on seeds 74 and 1); not re-searching solved twins (+364 s, +416 s); BPE merges (+282 s,
    +291 s; an earlier +546 s had counted a search whose time swings 0.5× to 5.5× between runs); the T = 1 filter on
    the round's answers (+82 steps on seed 74, none on seed 1).
  - **Not needed:** the search's routines-first order; the late table lessons, the wrap-around fact (`ANCHOR_WRAP`)
    among them; the step problem alone; the multiply's row problem (one seed); a round cost limit above 5 (at 4
    there is no round).
  - **How firm:** every failed arm searched its whole cost-5 reach without the missing piece, so more training time
    would not rescue it. A wider search, every program up to cost 6, found nothing for any of the six parts it
    applies to. Occam's tie-break was not a search's reach but the weak check below.
  - **Two weak checks in s3i, and a fix** ([ABLATION_S3.md §8](integration/ABLATION_S3.md#8-fixing-the-two-weak-checks)).
    - The reduction's check can accept a repeated subtraction that is right only for small quotients. It slips
      through when none of the check's 60 three-digit probes has a quotient above 600: about 3% of runs on H1's rows
      ([`reduce_check.py`](integration/ablation_s3/reduce_check.py)). Such a run cannot certify. None of s3i's 8
      certified seeds drew it; the step problem's arm did, on seed 74.
    - The conditional subtraction's check never tests x between N and 2N, so "x if x < N, else 0" passes it. Occam's
      tie-break is what kept s3i off that program.
    - The fix only adds the missing probes: full-size ones for the reduction, built by doubling with the learner's
      own addition, and ones between N and 2N for the conditional subtraction. It is in new copies of s3i and sp2,
      [`integration/ablation_s3/fix/`](integration/ablation_s3/fix/). Both copies certify on the official manifest
      and on their originals' seeds, 17 runs in all, and every record equals the original's field for field. The
      reduction check's 3% falls to about 10^-11. With the fix, s3i needs neither Occam's tie-break nor the step
      problem: both arms certify.
    - The fix is in the copies only. `lib/libcore.py` and `self_practice/spcore.py`, which s3i and sp2 are built
      from, keep the old checks.
- **Stage-3 self-practice** ([SELF_PRACTICE.md](integration/SELF_PRACTICE.md), code and records in
  [`integration/self_practice/`](integration/self_practice/)). The nine hand-picked practice problems, their order,
  the twins and mac's posing are replaced by a book of 11 laws on 8 patterns of the data's numbers. Every law is posed
  in every binding to the registers, and one general rule picks what to practise. The problems the ablation found
  needed come out of that book; nothing marks them.
  - `submission_stage3_sp.py` certifies on the official H100 manifest (round by 746 s, frozen by 785 s) and on 7 of 8
    B200 seeds.
  - `submission_stage3_sp2.py` certifies on the official manifest (round by 1,029 s) and on all 8 seeds.
  - **Still designed:** the law book, the choosing rule (picked after bar 3's own rules failed, by watching whether
    the round came), the instruction set and the search sizes.
  - The other formulas are still not learned.
- **The shader follow-up** ([NCA.md](nca/NCA.md) §9, code and records in [`nca/`](nca/)). The NCA probe is one
  identical tiny cell repeated over a grid, each wired only to its neighbours, trained from random weights on lessons
  alone. Two known cures for memorization were tried on it, far past the point where it fits: long training with
  weight decay ("trimming", the setting in which networks that had memorized modular arithmetic later found the
  rule), and agreement (a weight moves only where groups of questions, or the grid's columns, push it the same way;
  or it speeds up while its gradient keeps its sign). 72 runs from random weights, 4 seeds per arm (50,000 steps at
  (3, 2, 1), 20,000 at (3, 3, 2)), and 20 started at the compiled rule.
  - **Result:** none certifies, none fits the training questions on snapped digits (the best any run reaches is 62
    of the 99 at (3, 2, 1)), and one size up every run stays at chance from start to end: no late jump.
  - **Why:** decay holds the weights small, small weights pass blurred digits, and a blurred code fits the training
    answers on the soft path, unsteadily, without being the machine. The exact rule needs larger weights than decay
    allows. Started at the compiled rule, every arm loses it within a few hundred steps.
- **The loop over T, learned** ([LEARNED_LOOP.md](integration/learned_loop/LEARNED_LOOP.md), code and records in
  [`integration/learned_loop/`](integration/learned_loop/)). sp2's hand-written countdown, its gate, and the round
  search's assumption that the answer at T = t is the round run t times are removed. The machine gets one generic
  count loop, `C.r[body]` (run the body as many times as register r holds), and T goes into a register as x and N
  do. Once the round is found, a second search finds `C.S[T.round1 STA.X]`: run the round T times, feeding each answer
  back as x.
  - `submission_stage3_ll.py` (md5 `419c15e70ccf6483ff68b04b08f52bdc`) certifies on the official H100 manifest
    (frozen by 1,002 s) and on 7 of 8 B200 seeds. Seed 2 runs out of time in sp2's unchanged climb before any loop is
    searched; with the budget doubled (a research run) it certifies. The slow path comes from the random start, not
    the loop: a diagnostic copy that starts from sp2's own random draws replays sp2's climb line for line on seeds 74
    and 2, and the learned loop then certifies both, seed 2 frozen by 1,062 s.
  - **Still designed:** the count loop itself (a strict judge can call it the countdown made into an instruction),
    T's place in the machine and its 2-digit field, the answer program's search space, and what sp2 designs.
- **Strict builds of s3i and sp2** ([STRICT.md](integration/strict/STRICT.md), code and records in
  [`integration/strict/`](integration/strict/)). The gray points of §5 that can go without changing what is learned
  are removed, by exact edits to the certified sources:
  - No decision reads the clock: each search, and the climb as a whole, gets a budget of machine work, counted by the
    search's own cost model, the same on every host.
  - The rows the climb remembers and the found round program are buffers: in the state dict, counted, on the GPU.
    The forward pass runs the round from there.
  - The optimizer resets itself at phase B instead of the scheduler doing it, and the late table lessons and the
    wrap-around check in the table's gate are gone.
  - Both strict files certify on the official H100 manifest and on every seed their originals ran (s3i: seeds 1–7 on
    H100s; sp2: 74 and 1–7 on B200s), 17 runs. Each run's climb and schedule are the original's line for line.
  - **Unchanged, and why:** zero logits in training (the real answer inside autograd on a training batch does not fit
    in an H100's memory); the damped gradient; the reader; the climb, the instruction set and the curriculum, which
    are the learning method. The strict files keep the old weak checks; adopting the fix is a rebuild and 17 reruns.
- **Breadth at full size** ([SP_BREADTH.md](integration/sp_breadth/SP_BREADTH.md), code and records in
  [`integration/sp_breadth/`](integration/sp_breadth/)). Does sp2 learn the family's other formulas at H1's size?
  - Unchanged, 2 of 11 tasks: H1 and H1 with the prompt's fields reordered. The other formulas fail in the round's
    search, which composes at most two routine calls and cannot make a constant.
  - Three general changes, all in the round's search, none naming a formula: three calls instead of two; a constant
    read off the answers when every answer differs from a program's result by the same amount mod N; and a round must
    also reproduce the answers of the other one-digit T's.
  - With all three (`submission_spb_g3.py`, md5 `30efec64d6e4f0a7417938e9370a363e`), 8 of 11 are learned, each in one
    run on one H100 with every split exact: H1 (also on 8 of 8 B200 seeds), the reordered prompt, 10y², y² + 1,
    y² + 3, 10y² + 1, y² − 1, and the competition's Medium data (H1's formula, every T = 8). Not learned: y³ (four
    calls), Montgomery's y²·10⁻⁸ (no law poses its reduction), and a prompt with a fourth number (the reader takes
    three).
  - **How general:** more than before, less than bar 3 asks. Each change is stated without a formula, but each was
    written after watching a formula fail, with the family in view.

## 8. About this checkout

[HANDOVER.md](HANDOVER.md) has the hygiene pass: what is tracked and what is on disk only, the scan for secrets and
private data (no credentials anywhere; the author's home path and Modal workspace name appear in run logs), the link
check, and the files proposed for removal.

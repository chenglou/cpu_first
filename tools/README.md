# tools/: the shared Modal runner

`modal_runner.py` runs any submission file, from any folder, through the unmodified One Layer Deeper evaluator on
Modal. Use it instead of writing another copy of `litmus/modal_litmus.py`; the per-folder runners stay as they are.

## Commands

```bash
PY=~/github/sotaku/venv/bin/python    # the Modal client (1.2.6); run from any directory
R=cpu_first/tools/modal_runner.py     # paths are relative to the current directory or to cpu_first/

# Launches, in the tips-for-running-modal form (`$PY $R ...` with the same arguments does the same):
$PY -m modal run --detach $R --mode official --file X/submission.py --out X/records   # untouched h100_hard_h1.json, H100
$PY -m modal run --detach $R --mode seeds --file X/submission.py --seeds 4 --out X/records   # 74,1,2,3 on b200
$PY -m modal run --detach $R --mode plan --plan X/plans/P.json --out X/records [--hardware cpu] [--cap 20]
$PY -m modal run --detach $R --mode resume --run TAG       # a new driver for a run whose driver ended

# Reads (no app):
$PY $R --mode peek --run TAG [--app-logs 15]     # check-ins: each running job's latest lines and memory
$PY $R --mode status [--run TAG | --out X/records]   # every job's state; latest 10 runs by default
$PY $R --mode fetch --out X/records [--run TAG] [--wait]   # Volume -> X/records
$PY $R --mode cost --run TAG                     # the run's billed $, and list-price estimates
$PY $R --mode compare REF.json NEW.json ...      # a reproduction, field by field
```

Don't use `~/github/tips-for-running-modal/venv/bin/modal`: its interpreter line points to
`~/github/testing-modal/venv/bin/python3`, which no longer exists ("bad interpreter"). Recreate that venv to use it.

Launches print the jobs first: the file, its md5, the task, the hardware, either "official file, untouched" or the
exact overrides, and the limits. `--dry` stops there. `--set '{"MODE": "\"designed\""}'` rewrites `CONSTANT = ...`
lines, as every earlier runner's `--set` does. `--seeds` takes a count from 1 to 8 (the first N of 74, 1, 2, …, 7)
or a list such as `74,5`; for seed 1 alone, write `1,`. Every launch returns within about a minute. The image comes
from Modal's cache unless the competition clone or the task generators changed.

A plan is a JSON list of jobs. Its keys are a superset of the existing plans' keys:

| key | meaning |
|---|---|
| `name` | the record name; unique in the run and in the records folder (the launch refuses duplicates) |
| `file` | the submission (relative to the current directory or cpu_first/); default `--file` |
| `record` | instead of `file`: a stored record whose exact source, and manifest unless `manifest` is given, is run again |
| `task` | `h1` (default), `e1`–`e10`, `m1`–`m10`, or a litmus task (`xfirst mont ten rho cube xy tenrho plus3 minus1`) |
| `set` | `{CONSTANT: python source}` |
| `manifest` | `{section: {key: value}}` overrides, or `null` for the official file |
| `seed` | shorthand for `{"runtime": {"seeds": [seed]}}` |
| `hardware` | `b200` (the default), `h200`, `h100` (pinned `H100!`, never an H200) or `cpu` (8 cores: the earlier runners' CPU container); see the policy below. The old `"gpu": true` means `h100` and `"gpu": false` means `cpu`. Without either key, a job whose own manifest sets `runtime.device` to `cpu` (the old plans' way to ask for CPU) runs on `cpu`; any other job takes the run's `--hardware`, else `b200`. `--mode official` always uses `h100`, the contest's one-H100 evaluation |
| `memory_gb`, `timeout_s` | this job's limits (below); default the run's `--memory-gb` (24) and `--timeout` |

Manifests follow the earlier runners' rule, with one refinement. A job with no overrides runs the official file
untouched, and the record says so in `manifest_file`. Overrides equal to the official value are dropped first, so an
H100 job with seed 74 is the official run. A CPU job sets `runtime.device` to `cpu`. A GPU job keeps the official
`cuda:0`, even when its manifest comes from a CPU record.

## What it guarantees

- **The same image and data as before.** The layers are the ones `general_lessons/` and `integration/` build:
  `scripts/generate_datasets.sh` (Easy, Medium and H1 by the official generator and arguments), then the litmus and
  held-out tasks. Modal served it from cache on the first launch. Its H1 folder (19 files, 57.5 MB) has the same md5,
  `a6414ec9fc6a5efce6414895b52d4eb7`, as the H1-only image of `learned_remainder/` and `anchors/`. Every record
  carries this md5 as `runner.data_md5`.
- **Byte-faithful official runs.** The submission is sent as bytes, the container checks its md5 before the run, and
  the record keeps it as `submission_md5` (`file_md5` is the file's md5 before any `set`). The official manifest is
  passed by path, as before.
- **The tips-for-running-modal conventions.**
  - One `spawn()` per detached launch: the launch spawns only the driver, and the driver spawns the jobs from
    inside the app. Under `modal run`, a launch without `--detach` is refused before anything is written: without
    it, the app and the driver would stop when the entrypoint returns.
  - Separate lifetimes: a launch returns within about a minute. "App completed" from the CLI says nothing about the
    jobs; `peek`, `status` and `fetch` read the Volume, which holds the durable results.
  - App logs are for active debugging only (`peek --app-logs`).
  - A connection error while reading a result never launches a replacement.
  - Launch mode: the file path, never `-m`. Both `modal run --detach $R` and `$PY $R` were tested end to end.
  - See [PROPOSED_TIPS.md](PROPOSED_TIPS.md) for additions to the tips that came out of this work.
- **Preemption.** Each launch is its own detached app (`cpu-first-runner`, tagged `run=TAG`) with one
  non-preemptible driver: a 0.25-core container billed at 3x, about $0.05 an hour. The driver spawns the jobs in plan
  order, at most `--cap` unfinished at a time. It commits each call id to the Volume `cpu-first-runner`
  (`runs/TAG/calls/`) as soon as the call is spawned. A restarted driver re-attaches to those calls with
  `modal.FunctionCall.from_id`, as general_lessons/modal_general.py first did, and spawns only the rest. A worker
  that starts after its record already exists returns at once, so a restarted input never overwrites a finished
  record. A preempted worker still restarts its evaluation from the beginning: the evaluator has no checkpoints.
  `runs/TAG/started/NAME.json` counts attempts, and the record keeps the count in `runner.attempt`. If a driver ends
  early (an error, or the 24 h limit), `--mode resume` starts a new one. It refuses while the old driver is alive.
- **Limits on every job.**
  - Timeout: 1.5 x the training budget per seed, plus 30 minutes; `timeout_s` or `--timeout` sets another.
  - Memory: at most `memory_gb` (default 24) of summed RSS across the evaluator and its DataLoader workers (shared
    pages count twice, so this errs early).
  - The evaluator runs in its own session. Every way out SIGKILLs the whole session, children included: the end of
    the run, a limit, an error, or a cancelled input. The record notes `runner.killed`, with return code -9, and
    `runner.leftover_processes` (0 in every test).
  - Every worker container also has a hard 64 GiB memory limit, `memory=(request, 65536)`. A bare `memory=16384`,
    which the earlier runners use, sets none.
- **One cap.** `--cap` (default 20) bounds a run's unfinished calls, GPUs included, and each function also stops at
  20 containers. Since Oct 22 the workspace is on Modal's Team plan, with 50 GPUs and 5,000 containers, so a run
  rarely waits for capacity. Starter's 10 GPUs and 100 containers needed a GPU cap, a CPU fallback and a workspace
  check; those are gone.
- **Records.** Each record keeps the keys of the litmus, general_lessons and integration records, in the same
  order: `name, task, gpu, returncode, wall_seconds, manifest_file, manifest, stdout, stderr (last 200k), result,
  submission`. After those come `submission_md5, submission_file, file_md5, set, hardware, runner` (run tag, call id,
  container, attempt, data md5, limits, kill reason) and `usage` (CPU seconds and mean cores of the evaluator's
  processes, the session's peak summed RSS, and cores visible). Fetch writes whole files only (`.part`, then rename).
  It never replaces a record that another run wrote.

**Check in early and often.** Every minute, each worker copies its log tail, stderr tail and memory use to the
Volume, and prints a progress line to the app log. `--mode peek --run TAG` prints them for every unfinished job and
marks a copy older than 3 minutes as stale (stalled, preempted or dead). Check every few minutes while a run is new,
for blowups (memory climbing, errors in stderr) or stalls (the same step, a stale copy), then back off once it is
steady. `modal container exec <id> -- tail /tmp/<name>/stdout.log` can't replace this: it does not see /tmp on every
container. To stop a run, stop its app: `$PY -m modal app stop <app id>` (from `--mode status`). Cancelling only the
driver leaves the calls it already spawned running; `--mode resume` picks them up again.

## Validation (2026-10-22, records in `validation/records/`, plans in `validation/plans/`)

- **Official H100 run.** `--mode official --file learned_remainder/submission.py` (md5 `bd477b9a`) against
  `runs/learned_remainder/final_s2_official.json`: `--mode compare` finds no difference in any field. Both runs
  have T = 64 on both ladders, every rung at 768/768, exact splits, the same split losses, 20,000 steps, a freeze at
  step 499, and the same final loss and state sizes. The training traces (201 points) and the logs without timings
  (41 lines) are identical. Training took 181.3 s against 181.9 s, on the same H100 driver (610.57.04).
- **A CPU-seed plan.** `validation/plans/v2_p3_cpu_seeds.json` reruns the source stored in the eight
  `runs/learned_remainder/p3_s1v2_cpu_seed*.json` records (`fd26b584`) with each record's manifest. All 8 match
  their records in every field, traces and logs included (this file's CPU training came out the same on every host;
  wide_anchored's did not, below). Host speed differed: 6 of the 8 trained 1.1–1.7x slower than on Oct 20 (195–357 s
  against 196–242 s). They used about 7 cores, like the fast ones, but needed 1.3–1.7x the CPU-seconds.
- **Re-attach.** In a 2-job plan at `--cap 1`, `modal container stop` on the driver container while job 1 ran
  (Modal then hands the input to a new container, as after a preemption). The new driver logged "1 calls
  re-attached, 1 to spawn", waited for job 1 and spawned only job 2. In a second plan, the driver call was cancelled
  after it spawned job 2. Job 2 finished, and `--mode resume` logged "2 calls re-attached, 0 to spawn".
- **Launch forms and limits** (`validation/plans/limits.json`, launched with `modal run --detach`; without
  `--detach` the same launch was refused before anything was written). The job capped at 1 GiB was killed at 2.9 GiB.
  The job given 45 s was killed at 46.6 s and kept its partial log. Both reported `leftover_processes` 0. The third
  job, unlimited, matched its stored record (`p3_s1v2_cpu_seed2`) in every field. `peek` showed its log tail and
  memory while it ran.

## Seed policy (measured 2026-10-22)

A robustness seed runs the official manifest with another `runtime.seeds` value. The earlier runners ran these on
8-core CPU containers. How much slower a CPU is than an H100 depends on the file. Against each file's official H100
run, 8-core CPU seeds trained 3.9x as long for `wide_anchored`, 2.0x for wide_mlp, 1.9x for general_lessons v1
(half of its CPU seeds stopped at the deadline), and 1.0x for both integration stage 1 and tournament. So two files
were measured: `wider_menus/submission_wide_anchored.py` (`c82b11fa`, seeds 74 and 1) and the heaviest certified
file, tournament's `tourney2` (`7ae547d9`, seed 74, its official record's source). Each hardware ran in its own
app, so Modal's billing report separates the costs (`--mode cost`). Billed costs include the driver, start-up and
idle time.

| hardware | wide_anchored: train s (74, 1) | wall min | billed $ per seed | tourney2: train s | wall min | $ per seed |
|---|---|---|---|---|---|---|
| `b200` | 404, 419 | 7, 7 | 0.86 | 806 | 14 | 1.63 |
| `h200` | 597, 691 | 11, 12 | 1.02 | 1,558 | 27 | 2.21 |
| `h100` | 699, 842 (the stored official run: 558) | 13, 15 | 1.04 | 2,114 (stored) | 37 | ~2.63 est. |
| `a10` | 939, 1,262 | 17, 23 | 0.43 | 2,647 | 46 | 0.97 |
| `t4` | 1,347, 1,626 | 24, 29 | 0.33 | 3,094 | 54 | 0.66 |
| `l4` | 1,493, 1,521 | 27, 27 | 0.44 | | | |
| `cpu` (8 cores, 16 GiB) | 1,731, 2,062 | 36, 43 | 0.37 | 2,977 (stored) | 61 | ~0.57 est. |
| `cpu4` (4 cores, 16 GiB) | 2,575, 2,872 | 60, 52 | 0.33 | | | |

Every other cost is the run's complete bill divided by its seeds, so it includes the driver, start-up and idle
time. The two estimates are the tourney2 runs that exist only as stored records. The H100's is list price for its
wall time, plus about 95 s of H100 time for start-up and idle; that is what the official validation run's bill
showed. The CPU's is its stored record at today's measured CPU bill-to-list ratio (1.10).

Every run certified (T = 64 on both ladders) and froze inside the 3,600 s budget. tourney2 froze by 677 s on the
B200, 1,331 s on the H200, 1,875 s on the H100, 2,356 s on the A10, 2,611 s on the CPU and 2,704 s on the T4.
- The H200 replays the official H100 record exactly on both files: every field, the training trace included. So
  does today's H100 run of wide_anchored (tourney2 was not rerun on an H100).
- On wide_anchored, the A10, T4, L4 and B200 match it too, except one diagnostic log line (a visit tally about
  0.003% apart). On tourney2 their traces diverge from step 200, and they froze at 695 (A10), 704 (T4) and 708
  (B200) against 705.
- CPU seeds depend on the host. Seed 1 matched its Oct 20 record exactly and seed 74 did not, and CPU speed varies
  1.1–1.7x from host to host. The H100's speed varies too: today's seed 74 trained in 699 s against 558 s for the
  official run.

**Policy** (the user's call on 2026-10-22: the iteration left is bounded, so speed beats cost):
- **Everything on `b200`, the default, except official certification, which stays on `h100`**, the contest's
  one-H100 evaluation; `--mode official` refuses other hardware. The B200 was the fastest on both files: 7 and 14
  minutes per seed, 1.4–2.6x the H100's speed. Because it finishes so soon, it is also the cheapest big GPU per
  seed, 0.6–0.85x an H100's or H200's.
- **4 seeds (74, 1, 2, 3) at intermediate stages, 8 (74, 1–7) for finals.** On B200s that is about $0.9–1.6 per
  seed, so $7–13 for 8 seeds, all running at once.
- **`h200`** to check what the H100 itself would compute, faster: it replayed the official H100 runs exactly on
  both files. The B200, like the other GPUs, diverged from the H100's trace on tourney2.
- **`cpu`** for a CPU-numerics check, or for old plans that ask for it.
- The T4, L4, A10 and 4-core CPU were retired from the runner; the table above keeps their numbers.
- **Money:** the workspace's usage limit is $2,000 per billing period (set by the user; $877 used by Oct 22). A B200
  lists at $6.25 an hour, so a full 20-container run of them bills about $125 an hour.

**How many seeds.** The records give the basis for 4 and 8. Seven seed groups had a failure, counted as failed
seeds out of seeds run: `wide_reglu` 3/4, `chain_c1_idonly` 1/4, `p1_s1` 2/5, `io3` 2/8, `tourney` 5/8,
`tourney2_nonorm` 4/8 and `div` 1/8. The first four seeds (74, 1, 2, 3) catch six of them. They miss `div`, the
first learned_wiring stage, whose only failure was seed 7. So run 4 seeds at intermediate stages and 8 for finals.

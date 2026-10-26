# cpu_first — experiments for a CPU-first take on One Layer Deeper

**The entry:** [`integration/combined_fast/submission_stage3_fast2.py`](integration/combined_fast/submission_stage3_fast2.py)
(md5 `41b9a013ae9436d7b7a53e9e594a18db`, "fast2"). It is one training run that finds its own program: every certified
variant of the self-practising stage-3 file combined, then made 7.7 times faster by engineering alone. It certifies
H1 (T = 64 on both ladders, every split exact) on 8 of 8 official-manifest H100 runs, training in 759–883 s of the
3,600; `python3 tools/verify_records.py` checks all eight records.

**Reviewing or judging the project? Start with [REVIEW.md](REVIEW.md)**: what is claimed, the one file to judge and its
md5, how to check the official records and reproduce them, and the gray rules a judge should rule on.

**Otherwise, start with [OVERVIEW.md](OVERVIEW.md)**: where everything stands against three bars, the models at the frontier, the removal ladder, the
rules, and the open questions. This file indexes every file and folder.

The early experiments run on a laptop CPU in seconds to minutes. Later ones, from the grammar search on, run on Modal
(CPU containers and H100s), and every official result is a record in `runs/`. Data: the released H1 set, generated
locally with the competition repo (`python -m data.reversed_n_squaring_mod ...`, the H1 command in
`scripts/generate_datasets.sh`). Point `H1_DATA` at that folder if it moves. Python: torch + numpy (python-flint for
the lattice experiments).

The competition repo sits next to this folder, in `../one-layer-deeper`: a clean clone at `87e5c26` ("Releasing H1")
with the H1 data under `data/generated/`. Its virtualenv was left out of the copy; run `uv sync` there to recreate it.

## Write-ups

| file | what it is |
|---|---|
| `OVERVIEW.md` | the current entry point: the task, why it is hard, the timeline, the best model, the removal ladder, rules 1–16, open questions, a map |
| `README.md` | this index |
| `REVIEW.md` | for an outside reviewer: the claims, the file to judge, the official records and how to check them, how to reproduce, the gray rules and the questions to rule on, an hour's reading order |
| `LEGAL_PLAN.md` | a plan from bar 3 to a contest-legal entry: what has to move inside the training run, what is still designed in the self-practising file, the questions only a judge can settle, and the cheapest test of each |
| `HANDOVER.md` | the hygiene pass before handing the repo over: secrets and private paths, links, checksums, proposed removals, what is tracked and what is on disk only, and the handover commands |
| `REPORT.md` | the first day's write-up (§0–8) and the follow-ups on SpecBit and the grammar search (§9–14) |
| `FINDINGS_substrate.md` | raw numbers for REPORT.md: the substrate experiments, SpecBit through the evaluator, and the grammar-search runs |
| `RELAXED.md` | which single relaxed rule makes the step learnable by a mostly generic network: per-step labels, show-your-work traces (and curricula), periodic units tied to N in fp64, unlimited data, and supervising s². A trace GPT whose every token is local (trial subtractions) certifies T=64 up to 24 bits |
| `UNFOLD.md` | can a model unfold the remainder by itself from answer-only rows? The clock loss, easiest-first ordering, a rung head, and lattice reduction that solves every wrap count at once |
| `UNFOLD_FAMILY.md` | the same when the recurrence itself is unknown: which polynomial step made the labels |
| `panel_synthesis.md` | the five-design panel's synthesis (the panel itself is in `panel/`) |

## Scripts at the top level

| file | what it shows |
|---|---|
| `h1.py` | H1 loader in LSB-first digit form + an evaluator-faithful certifier (T-ladder, both profiles) |
| `cells.py` | learned truth-table cell (soft / straight-through / hard) — the "gate" of a learned machine |
| `exp_substrate.py`, `exp_codes.py`, `exp_wide.py` | answers-only learning of composed digit tables: single table easy; latent 10-way wires stall; wide (24–48-way) wires fix shallow compositions; deep (multiplication) compositions stall even without any modular reduction |
| `diag_mul.py` | a diagnostic for `exp_wide.py`: reruns its 4-digit multiplication and counts the wrong entries of the learned ADD table |
| `machine.py` | decimal Montgomery machine (panel CADET-M datapath): Peano self-test ALU + vectorised microcode |
| `cadet.py` | end to end: self-tests → label speculation for N′ → output map → T=64 certified on both profiles |
| `menu.py` | unknown recurrence: labels select among 24 complete programs; 6/6 hidden recurrences certified |
| `speculate_gate.py` | the binary gate from labels only: 65,536 candidate machines bitsliced, 4 rows → the full adder |
| `speculate_wiring.py` | 31,104 wirings at once; factored search via the label-free round-trip (finds a base-12 impostor) |
| `speculate_full.py` | gate × wiring (≈2³¹) via round-trip then labels → exactly full adder + reference wiring |
| `alias_width_audit.py` | the same netlist at 24 bits passes every training row and the training-pool profile but fails 2,068/5,376 OOD rows: speculation cannot see off-distribution structure |
| `bitcpu_netlist.py` | panel critic's gate-level BitCPU (every gate = one FA table): `python bitcpu_netlist.py $H1_DATA 01101001,00010111` certifies T=64 on both profiles |
| `submission.py` | SpecBit as a competition submission: that netlist in torch, its one cell learned inside the forward pass as a mixture over all 65,536 truth tables (bitsliced, 64 per int64 word). Certifies T=64 on both profiles through the unmodified `benchmark.runner`, also on the official H100 manifest (`runs/h100_h1.json`) |
| `h1_local_mps.json`, `h1_local_cpu.json` | local copies of `h100_hard_h1.json` (device `mps:0` / `cpu`, fp32, amp off, `num_workers` 0, `log_every` 1); `data_root` is the official relative path, so run them from `../one-layer-deeper` as the official manifest is |
| `spec_grammar.py` | bitsliced speculation in torch over a grown grammar of whole machines (gate, converters, entry/exit, a two-operand Montgomery core with radix-2 and radix-4 families): staged search with label-free round trips, label-free core laws (stage C0, `--law`), labels, every training row, then the certification audit. `--families r2 --ppop ab --law none --kw-max 28` reproduces the first 2^54.4 runs; `--core ports` adds carry-save port sources; `--law-synth`, `--count-law` and `--c-check decimal` make the search label-free (REPORT §10); `--cell-law` and `--width-law` add per-cell value laws and a width bound inside the core, and `--cell-audit` checks them (REPORT §11); `--kw-max` (default 32) sets the largest K, W and word width, `--digit-scales` the value laws' per-row scales, `--witness` adds declared-range witness rows to the audit (REPORT §12); `--collapse` runs D and E on one machine per class of law-equivalent survivors (`--collapse C1` already groups stage C's survivors, so the semigroup check also runs once per class), and `--collapse-check` spot-checks other members of each class (REPORT §13–14). `--bench` measures raw throughput, `--fuse-probe` the `torch.compile` headroom of the hot block, `--alias-demo` the T=1 aliases without the entry/exit round trip |
| `modal_h100.py` | the same on a Modal H100, in an image mirroring the competition's `modal_runner.py`: `--task h1` runs the official H1 manifest untouched, `--task bench` / `--task search` run `spec_grammar.py` |
| `relax_common.py`, `relax_step.py`, `relax_trace.py`, `relax_trace_errors.py`, `relax_periodic.py`, `relax_periodic_cert.py`, `relax_square.py`, `relax_table.py` | the RELAXED.md experiments; `relax_table.py` prints every result |
| `modal_relaxed.py` | runs any `relax_*.py` on a Modal H100 (Volume `relaxed-runs`; copies in `runs/modal/`) |
| `unfold.py` | the UNFOLD.md experiments: fp64 models of the value before the remainder, trained on the folded label with the clock loss |
| `unfold_lattice.py` | UNFOLD.md §5: every row's wrap count solved at once by lattice reduction (LLL on Kannan's embedding) |
| `unfold_modal.py` | runs `unfold.py` / `unfold_lattice.py` command lists on Modal CPU containers, one container per line |
| `unfold_family.py` | the UNFOLD_FAMILY.md experiments: an unknown integer polynomial step, by lattice and by gradient |
| `unfold_family_modal.py` | runs `unfold_family.py` command lists on Modal CPU containers |
| `modal_unfold_sub.py` | UnfoldB (`unfold_sub/submission.py`) through the unmodified evaluator on Modal: the official H100 run, CPU plans, the gradient check |
| `.gitignore` | keeps run outputs over 5 MB (search dumps, checkpoints) on disk but out of git |

## Folders

| folder | what it holds |
|---|---|
| `learned_remainder/` | the removal ladder: H1 certified with the digit machine's parts learned one at a time, then all in one model. `submission*.py` (the certified files), their write-ups (`LEARNED_REMAINDER.md`, `CHAIN.md`, `SQUARE.md`, `CONTROL.md`, `ADDER.md`, `IO.md`, `COMBINED.md`), the Modal launchers (`modal_chain.py` runs any of them), `plans/` for CPU seeds, `*_local/` local checks and diagnostics |
| `ablation/` | which ingredients of the learned machine's training recipe matter (`ABLATION.md`), and `submission_simple.py`, the simplified recipe, certified. Records in `ablation/runs/` |
| `anchors/` | digit meanings from the order of the numbers: the subtraction table learned with no hand-written adder at any step (`ANCHORS.md`). `submission_anchors.py` (the single-piece machine) and `submission_combined_anchors.py` (the combined model) both certified; `core.py`, `tiny.py` and `exact.py` are the lab; records in `anchors/records/` |
| `learned_wiring/` | the combined machine learns its own wiring in three certified stages (`WIRING.md`): `submission_div.py` (the division's switches), `submission_sq.py` (+ the squaring's layout), `submission_io3.py` (+ the reader's and writer's shape); the first versions `*_v1.py` are kept with their records; `tiny/` holds the small machines. Then stage 3 with no hand-written adder (`ANCHORED.md`): `submission_anchored.py`, certified, with its diagnostics and variants in `anchored/` |
| `wider_menus/` | the wiring chosen from very wide menus, about 4·10^17 whole machines (`MENUS.md`): `submission_wide.py`, certified; `wide.py` is the grammar, `tiny_wide.py` the tiny machine, `exact_wide.py` the exact counts and `check_grammar.py` the grammar's checks on exact arithmetic. Then the same with no hand-written adder (`WIDE_ANCHORED.md`): `submission_wide_anchored.py`, certified; every later frontier file builds on it. Records in `runs/wider_menus/` |
| `bridge/` | the machine as one weight setting of a generic looped transformer, loosened notch by notch and trained from random (`BRIDGE.md`): `compile.py` and `net.py` build it, `notches.py` and `train.py` loosen and train it, `probe.py` and `diagnose.py` compare trained networks with the compiled one, `blocksearch.py` searches the wiring block by block. `h1/submission_wide_mlp.py` (the wide model with generic ReLU MLP digit cells) certified; records in `records/` and `h1/records/` |
| `handwired_bridge/` | the same machine compiled by the handwired project into affine + ReGLU layers (`js/`), loaded into PyTorch (`load.py`) and loosened the same way (`HANDWIRED_BRIDGE.md`). `h1/submission_wide_reglu.py` did not certify on its official run; records in `records/` and `h1/records/` |
| `tournament/` | the wiring chosen inside training by tournaments over whole block combinations (`TOURNAMENT.md`): `tiny.py` (the tiny machine), `traps.py` and `h1_census.py` (what the checks let through, counted), `submission_tournament.py` (the H1 file, certified), records in `records/` and `h1_records/` |
| `general_lessons/` | a law library, facts the model mines, and lesson sets the answers choose, in one file run unchanged on every task (`LESSONS.md`): `submission_general.py` (certified on H1 and on seven other formulas), `tasks_general.py` (the tasks), `tiny_table.py` and `tiny_general.py` (tiny tests), `variants/` (diagnostics, not submissions), records in `records/`. Then `submission_general_v2.py` (the answers read at the batch's smallest T: learns M4, LESSONS.md §11) and `submission_general_v3.py` (the answers choose whole digit codes: learns all 14 fixed-modulus practice sets; `fixed/FIXED_MODULUS.md`, with its census, tiny tests and records in `fixed/`) |
| `litmus/` | the wide model (the best H1 model at the time) run unchanged on 26 other tasks (`LITMUS.md`): `tasks.py` (the tasks), `audit_data.py` (what each dataset can teach), `variants/` (the per-task changes that fixed some), records in `records/` |
| `library/` | search and compression that rediscovers the hand-written skeleton (`LIBRARY.md`): `dsl.py` (the 32 digit instructions), `search.py`, `merge.py` (BPE and MDL), `wake_sleep.py`, `inrun.py` (the version inside one training run), `certify.py` (the discovered programs on H1's ladders); and the inventing test (`INVENTING.md`, code in `inventing/`). Records in `records/` and `inventing/records/` |
| `integration/` | the tournament, the general lessons and the library in one H1 model (`INTEGRATION.md`): `submission_stage1.py` (stage 1, the tournament and the general lessons in one file, certified); `submission_stage3_s3i.py` (stage 3, no skeleton and no menus: the round discovered in training; certified on all 8 seeds on the official manifest, the current best model), with the earlier stage-3 builds (`submission_stage3*.py`), `make_stage3.py` that builds them, the climb's library code in `lib/`, checks in `checks/`, records in `records/`; then the ablation of s3i's hand-made parts (`ABLATION_S3.md`, code, variants and records in `ablation_s3/`) and stage 3 with self-posed practice (`SELF_PRACTICE.md`, in `self_practice/`: `submission_stage3_sp.py` and `submission_stage3_sp2.py`, both certified on the official manifest, records in `self_practice/records/runner/`) |
| `bar3/` | the self-practising learner on a bare counting machine: it picks its own practice problems and invents x² mod N (`PRACTICE.md`), and, unchanged, other formulas (`BREADTH.md`, code in `breadth/`): `learner.py`, `problems.py` (where practice problems come from), `search.py`; records in `records/` and `breadth/records/` |
| `nca/` | long division as a shader: identical local cells on a grid, compiled exactly and then trained (`NCA.md`) |
| `tinyrouting/` | can a digit machine learn its wiring together with its tables? Tiny machines, computed exactly (`TINYROUTING.md`) |
| `tinycore/` | the quotient-digit chicken-and-egg, shrunk until it can be computed exactly (`TINYCORE.md`) |
| `microlab/` | a harness for one-minute experiments on small versions of the task, and its findings: the generic-network ladders, exact blocks, looping, learning before the remainder, the quotient digit (`README.md` there; `LADDER*.md`, `FINDINGS_*.md`) |
| `unfold_sub/` | UnfoldB: UNFOLD.md's self-unfolding as a submission, certified on the official H100 manifest (`NOTE.md`) |
| `unfold_small/` | UnfoldS: UnfoldB at the manifest's batch size of 512, certified on the official H100 manifest (`NOTE.md`) |
| `panel/` | the design panel: brief, prior-art survey, 5 designs, 5 adversarial critiques (synthesis in `panel_synthesis.md`) |
| `runs/` | logs and JSON records of the runs above (MPS, CPU, H100). Official records: `h100_h1.json` (SpecBit), `unfold_sub/`, `unfold_small/`, `learned_remainder/*_official.json`, `learned_wiring/`, `wider_menus/`; each learned-machine record holds the full log, the result and the exact source. The ablation keeps its own in `ablation/runs/`, the anchors in `anchors/records/` |
| `tools/` | `modal_runner.py`, the shared Modal runner for new runs: any submission file through the unmodified evaluator (official, seeds, plans), re-attaching after a driver restart, at most 20 containers per run, records in the usual format; `README.md` there has the commands, the plan format and the measured seed policy; `validation/` holds its reproductions of certified records; `verify_records.py` checks official records against the files they certify (standard library only) |

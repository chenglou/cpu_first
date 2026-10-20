# Substrate experiments: can learned digit tables be composed into a CPU?

All runs: shared learned truth tables (`cells.Table`), hard straight-through forward, Adam lr 0.05, batch 256,
answers-only supervision (cross-entropy on output digits), no modular reduction anywhere. Width extrapolation is
measured by training on ≤k digits and testing at longer widths. Scripts: `exp_substrate.py`, `exp_codes.py`,
`exp_wide.py`.

| setup | latent wires | result (exact match) |
|---|---|---|
| a+b, one table ADD(a,b,c)→(d,c) in an LSB-first scan | carry (2-way) | 100% at 5, 7, 9 digits (trained ≤5) in ~1k steps |
| a·d (digit × number): MUL→(lo,hi), ADD3(lo,hi_prev,c) | lo 10-way, hi 9-way (exact size) | 73–78% (4/4 seeds stuck) |
| same, MUL feeds the *shared* canonical ADD table | pinned to digit codes | 67–82% (stuck); +multitask on a+b: 43–67% (worse) |
| same, wires 16-way | over-provisioned | 2/4 seeds 100% (incl. 9 digits), others 96–98% |
| same, wires 24-way | over-provisioned | 4/4 seeds ≥99.9%, 100% at 9 digits |
| same, wires 48-way | over-provisioned | 4/4 seeds 100% at 5 and 9 digits |
| a·d, 24-way wires, fixed pool of 50 / 150 / 500 examples | | 12% / 33–43% / 75% test (train loss → 0: memorised) |
| full 4×4-digit schoolbook a·b (≈72 sequential lookups), 24-way wires | | 8–13%; + multiplier-width curriculum 11–16% |
| same, every latent wire widened (incl. pp→acc), carry 2 or 4-way, ± curriculum | | 10–26% after 5k steps |

Takeaways
1. **Exact-size latent wires trap straight-through learning** (TerpreT-style). Two tables talking through a private
   10-way code settle into collisions they cannot undo one entry at a time.
2. **Over-provisioned ("wide") wires fix shallow compositions.** 24–48-way codes make the same circuit learn reliably,
   and the learned circuit is exactly width-generic (trained ≤5 digits, perfect at 9). Redundancy in the encoding plays
   the role that redundant number systems (carry-save, signed digits) play in hardware — here it buys optimisability,
   not speed. Matches TerpreT's "more gates/wires than needed helps" and CGP's neutrality results.
3. **Wide wires cost data efficiency.** Lookup cells have no inductive bias across entries; every input combination
   must be exercised, and extra code space admits spurious fits. With a few hundred examples they memorise.
4. **Credit-path length kills learning even without any modular reduction.** Four sequential partial-product/accumulate
   rounds (~72 lookups deep) do not learn from answers alone, even with unlimited fresh data, wide wires and curriculum.
   The panel shows the mod-N reduction makes this categorically worse: one wrong upstream entry scrambles the residue.

# End-to-end demos (all CPU, M5 Max)

| demo | what is learned, from what | result |
|---|---|---|
| `cadet.py` — decimal Montgomery machine (panel CADET-M datapath) | FA/MUL/COMP tables from Peano self-tests (no H1 rows, 0.5 s); N′ per N mod 10 by label speculation over 10 complete machines (0.3 s; right one 100%, rest 0%); digit→token map by CE | **T=64 certified on both profiles**, 14/14 rungs at 100%, 1.1 s |
| `menu.py` — unknown recurrence | same ALU; labels jointly select program (24-program menu s^e·x0^f + a) and N′ | 6/6 hidden recurrences (square, cube, square+1, square·x0, square+x0, cube+2) identified and **T=64 certified on both profiles**, ~8 s each. Runner-up matches exactly the T=1 share of rows (T=1 aliasing, e.g. s·x0 ≡ s² at T=1); T=2 rows break the tie |
| `speculate_gate.py` — binary netlist (panel BitCPU; every gate = one 3→2 truth table) | the gate, from **labels only**: all 65,536 candidate tables executed at once, bitsliced 64 per uint64 | 1 T=1 row → 2 survivors (0.5 s); **4 rows → exactly the full adder** (1.9 s). No axioms, no gradients |
| `bitcpu_netlist.py` with that gate | — | **5,376/5,376 on both profiles** (T=1..64, mixed-T batch, 5,839 gate levels at T=64), 5 s |

Takeaway: where gradients see a needle, *speculative execution over hypothesis space* with labels as an exact
verifier works — if the hypothesis space is enumerable. Binary cells make it tiny (2^16); bitslicing makes
evaluating it nearly free. Decimal tables (≈10^260 full adders) are not enumerable and need structure (laws/self-tests).
| `speculate_wiring.py` — 8 wiring choices (~15 bits: Horner shifts, Montgomery quotient-bit source, shift amounts, scan order, exit quotient rule, double-dabble rule), gate fixed | labels only (joint), or round-trip (no labels) then labels (factored) | joint: 31,104 machines vs 4 (even 1) T=1 labels in 0.1 s → **exactly the reference wiring**. Factored: round-trip leaves 2 converter pairs — base 10 and a genuine **base-12 pair** (×12 Horner + "≥6 add 2" dabble) — labels reject base 12 |
| `speculate_full.py` — gate × wiring jointly (2^16 × 31,104 ≈ 2^31) | round-trip on 8 prompt numbers (no labels), then 4 T=1 labels | stage A: 2.36M machines, 22 s → 9 survivors; stage B: 7,776 machines, <1 s → **exactly one: full adder (0x96/0xE8) + reference wiring**. 2.4M evaluations instead of 2.04B |
| `alias_width_audit.py` — the BitCPU netlist built 24 bits wide (R=2^22) instead of 28 (R=2^26) | — | training T=1 rows 1,000/1,000; training-pool certification 5,376/5,376; **OOD certification 3,308/5,376**. Off-distribution structure (width, counter range) is invisible to label-only speculation |

# SpecBit through the evaluator; speculation at GPU scale

Scripts: `submission.py`, `h1_local_{mps,cpu}.json`, `spec_grammar.py`, `modal_h100.py`. Logs and JSON: `runs/`.
H1 regenerated with the repo's H1 command (byte-identical to the earlier copy). Torch 2.12.1, Python 3.13.5 (the
repo's `uv.lock`). Laptop = M5 Max; H100 = Modal H100 80GB HBM3.

## SpecBit as a competition submission (`submission.py`)

`bitcpu_netlist.py` in torch. Every gate is one learned 3→2 cell; θ ∈ ℝ^65,536 holds one logit per (sum, carry) truth
table, init N(0, 0.01²). Training forward: the machine runs under every hypothesis with π_h ≥ 1e-9·max π on the
batch's T ≤ 2 rows, bitsliced (lane j of an int64 word = hypothesis 64w + j; the cell is a Shannon expansion against
per-lane table words). `token_training_loss` = −mean_rows logsumexp_h[log π_h − 1e4·1(digits_h ≠ y)]. Prediction:
the argmax cell's machine for every row, logits = log(π*·onehot(digit) + (1 − π*)/V), so argmax is the machine's
digit and the loss reaches θ. Adam lr 5, `max_steps` 20, `eval_batch_size` 1024. All fp32 or int64.

| run | manifest | training | eval | max certified T (seen N / unseen N) | test, ood_t, ood_n_t |
|---|---|---|---|---|---|
| M5 Max MPS | `h1_local_mps.json` = `h100_hard_h1.json` with `mps:0`, fp32, amp off, `num_workers` 0, `log_every` 1 | 20 steps, 13.9 s | 21.8 s | **64 / 64** (every rung 768/768) | 100% / 100% / 100% |
| M5 Max CPU | `h1_local_cpu.json` (same, `cpu`) | 20 steps, 8.4 s | 16.0 s | **64 / 64** | 100% |
| Modal H100 | `benchmark/manifests/h100_hard_h1.json` **untouched** (cuda:0, bf16 autocast on, 3,600 s), image mirroring the repo's `modal_runner.py` | 20 steps, 16.5 s | 26.9 s | **64 / 64** | 100% |

- Learning: argmax θ is the full adder (0xE896 = LOP3 0x96 | 0xE8 << 8) after 1 Adam step on seed 74 and after ≤ 2
  steps on shuffle seeds 1–4. π_FA = 0.06–0.25 after step 1, ≥ 0.988 after step 2, ≥ 0.9999 after step 3.
  - Adam's first step is ±lr for every logit, so every table consistent with *some* row ties briefly with the full
    adder.
  - The forward speculates over all 65,536 tables for 3 steps, then over 1–15.
  - The training accuracy logged by the runner goes 0 → 1.0 at step 2.
- Cost: a training step with the full 65,536-hypothesis pass over a batch's ~10 T ≤ 2 rows is ≈ 0.5 s on MPS. That
  includes the argmax machine on all 512 rows. Model state is 65,536 elements; optimizer state is 131,073.
- Interface gotchas (both rejected the run before any training):
  - The manifest device must be `mps:0`, not `mps`. The runner compares `param.device != torch.device(manifest)`, and
    `mps:0 != mps`.
  - Adam must be `fused=True`, or `capturable` on CUDA. Otherwise its `step` counter lives on the CPU and
    `validate_optimizer` fails (rule 9).
- Compliance: every mechanical check passes. These cover source lint, model/optimizer state on device, the
  differentiable finite loss, logits shape, and state unchanged during eval. Rules 7 and 14 (designed netlist,
  task-specific structure) remain the judgment call of REPORT §6.

## Bitsliced speculation in torch: throughput

Candidate machine-rows per second (one machine on one T=1 row, full machine unless noted), eager, warm, best over
rows ∈ {1, 4, 16, 64}; chunk 2^20 lanes on the laptop, 2^24 on the H100 (`spec_grammar.py --bench`).

| case | numpy CPU (previous session) | torch CPU | MPS | H100 |
|---|---|---|---|---|
| gate: 65,536 per-lane cells, reference wiring | 1.4e5 (`speculate_gate.py`, 4 rows in 1.9 s) | 9.2e5 | 3.4e6 | 2.3e7 |
| wiring: 979,776 core combos, full adder | 1.2e6 (`speculate_wiring.py`, 31,104 × 4 in 0.1 s) | 2.2e6 | 9.1e6 | 5.3e7 |
| joint: per-lane cell × qsrc × l2sh (12.4M) | — | 1.1e6 | 4.0e6 | 2.8e7 |
| round trip only: per-lane cell × io, d2b + b2d | — | 1.6e6 | 6.1e6 | 4.7e7 |

Fusion headroom (`--fuse-probe`, `torch.compile` of the Montgomery square + exit + double-dabble block only, front end
computed once; outputs bit-exact):

| block | torch CPU (2^16 lanes) | MPS (2^20 lanes) | H100 (2^22 lanes; 979,776 for the first row) |
|---|---|---|---|
| core, full adder | 2.5e6 → 8.4e6 (3.3×) | 1.65e7 → 1.45e8 (8.8×) | 3.0e7 → 3.9e8 (12.9×) |
| core + port wiring, full adder | 2.4e6 → 7.9e6 (3.2×) | 1.72e7 → 1.61e8 (9.4×) | 1.05e8 → 6.4e8 (6.1×) |
| core, per-lane cell × qsrc × l2sh | 1.2e6 → 6.1e6 (5.0×) | 8.4e6 → 6.3e7 (7.5×) | 4.8e7 → 4.9e8 (10.1×) |

- Compile time on the H100 host (16 CPUs) was 110–245 s per block.
- A second H100 run of the same blocks, with the select masks rebuilt inside the traced graph, gave 8.6–13.5×
  (`runs/h100_fuse_argform.json`).
- Compiling the whole per-lane-cell machine as one graph did not finish within 15 min on Modal's default CPU
  allocation (Inductor on a 10^4–10^5-node graph).
- On MPS, Inductor's Metal kernels exceed the 31-buffer limit unless the select masks are precomputed in an eager pass
  first.
- So every end-to-end search number below is eager. A fused end-to-end search would be engineering work, not a
  measured number.

## The grown grammar (21 factors)

| block | factor | options | reference |
|---|---|---|---|
| io | gate | 65,536 truth tables | full adder 0xE896 |
| io | d2b_mul (Horner v ← v≪a + v≪b + d) | 15 (multipliers 2–32) | (3,1) = ×10 |
| io | d2b_order | 2 | MSB first |
| io | b2d_thr (add if nibble ≥ t) | 6 (t = 3..8) | 5 |
| io | b2d_add (+c) | 8 (c = 0..7) | 3 |
| io | b2d_order | 2 | MSB first |
| entry/exit | entry_radix | 2 (s ← 2s − [·]N; s ← 4s − kN with 3 borrow-select subtractions) | 4 |
| entry/exit | Ke (entry doublings) | 4 (22, 24, 26, 28) | 26 |
| entry/exit | exq (exit quotient bit) | 4 | Ss0 ⊕ Sc0 |
| entry/exit | Kx (exit halvings) | 4 | 26 |
| core | ppdir | 2 | LSB first |
| core | l1src (which of Ss, Sc, pp feeds each port of carry-save cell 1) | 27 | (Ss, Sc, pp) |
| core | l1sh (shift per port) | 8 | (0,0,0) |
| core | qsrc (Montgomery quotient bit) | 7 | u0 |
| core | l2src (which of u, v, qN feeds each port of cell 2) | 27 | (u, v, qN) |
| core | l2sh | 27 | (0,1,0) |
| core | osh (sum-row hand-off) | 3 | ≫1 |
| core | csh (carry-row hand-off) | 3 | as is |
| core | Km (Montgomery iterations, R = 2^Km) | 4 | 26 |
| core | W (datapath width) | 3 (24, 26, 28) | 28 |
| core | cw (T-counter width) | 3 (5, 6, 7) | 7 |

Sizes: gate × io = 188,743,680 (2^27.5); entry/exit = 128; core without the two port-source factors = 979,776 (2^19.9),
with them 714,256,704 (2^29.4). Joint: **2^54.4** (small core), **2^63.9** (with port wiring). The previous session's
grammar was 2^31.

## Staged search

Each stage screens a lane on 1 row, then a few more, then the rest, dropping it at the first mismatch. The row groups
were 1/1/2/rest on MPS and 1/3/rest in the H100 runs.

| stage | check | machines | survivors | MPS | H100 |
|---|---|---|---|---|---|
| A | b2d(d2b(v)) = v on 16 prompt numbers (no labels) | 188,743,680 (2^27.5) | 7,782 across 382 distinct gates (mostly degenerate cells whose converters just pack digits into nibbles) | 52.5 s | 8.3 s |
| B | exit(entry(x)) = x on 8 prompts (no labels) | 7,782 × 128 = 996,096 | 64, all full adder: {×10/≥5+3, ×12/≥6+2, ×16 with +0 (6 equivalent thresholds)} × radix {2,4} × Ke = Kx ∈ {22, 24, 26, 28} | 0.8 s | 0.4 s |
| C | 64 T=1 labels | 64 × 979,776 = 62,705,664 (2^25.9) | 144 | 14.3 s | 2.4 s |
| D | all 101,999 training rows (T = 1, 2, 4, 8, 16) | 144 | 144 | 10.7 s | 10.2 s |
| E | both certification profiles, T = 1..64 | 144 | 16 certify T=64 on both | 3.8 s | 4.6 s |

- Total screening is 2^27.9 machine evaluations for the 2^54.4 joint space: 11.2 s on the H100 (31 s wall with D, E
  and data loading) and 67.6 s on MPS (85 s wall).
- How many checks it takes:
  - Stage C, 62.7M machines: 3,966 survive 1 T=1 label, 198 survive 2, and **144 survive 4** (unchanged through 64).
  - Stage A, 188.7M machines: 10,448 survive 1 number, 8,509 survive 2, 8,404 survive 4 and 7,782 survive 16.
    Degenerate converters keep failing on rarer digit patterns, so more prompt numbers would shrink A further.
  - Stage B is final (64) after 1 prompt.
- **With port wiring (2^63.9)**, eager on one H100, the whole run takes **19.7 min**: A 7.9 s, B 0.4 s, C 1,104 s,
  D 46 s, E 20 s.
  - Stage C screens 64 × 714,256,704 = 45.7e9 (2^35.4) machines at 4.8e7 machine-rows/s and leaves 5,184.
  - 5,184 = 144 × 36. In every survivor, cell 1's (source, shift) multiset is exactly {(Ss,0), (Sc,0), (pp,0)}. Cell 2's
    is one of the two hand-offs above, {(u,0), (v,1), (qN,0)} or {(u,1), (v,2), (qN,1)}.
  - So the 729 × 729 port-source choices are pinned up to the 3! × 3! port permutations of the two symmetric
    full-adder cells. All 5,184 fit every training row, and 576 = 16 × 36 certify T=64 on both profiles.

## Aliasing audit (stage E)

The 144 machines that fit every training row differ only in these switches. Each entry gives a value, the number of
machines with it, and how many of those certify T=64 on both profiles.

| switch | value: machines → certify both | what the failing values miss |
|---|---|---|
| cw (T counter) | 5: 48 → 0 (seen max T = 16) · 6: 48 → 0 (max T = 32) · 7: 48 → 16 | T range: training T ≤ 16 fits in 5 bits |
| K = Ke = Kx = Km | 22: 36 → 0 · 24: 36 → 0 · 26: 36 → 8 · 28: 36 → 8 | modulus range: Montgomery needs R = 2^K > 4N, i.e. K ≥ bits + 2. Training N ≤ 20 bits only needs K ≥ 22. At T=64, K=22 gets 1/192 of the 22-bit and 0/192 of the 24-bit OOD moduli; K=24 gets 0/192 of the 24-bit ones; all pass 14/18-bit |
| W (datapath) | 24: 48 → 0 · 26: 48 → 8 · 28: 48 → 8 | width: W=24 gets 0/192 of the 24-bit OOD moduli at T=64 even with K=26 (cf. `alias_width_audit.py`) |
| entry_radix | 2: 72 → 8 · 4: 72 → 8 | benign: both entries compute x·2^Ke mod N exactly |
| carry-save hand-off | ((0,1,0), ≫1, as is): 72 → 8 · ((1,2,1), ≫2, ≫1): 72 → 8 | benign: the same datapath scaled by 2; identical on every row of both profiles |

- The 16 certifying machines are exactly radix {2,4} × hand-off {2} × K {26, 28} × W {26, 28} × cw 7. Every one uses
  the spec-sized values of the three off-distribution switches.
- Nothing in the training data separates them from the 128 that fail. Choosing K, W and cw takes a prior from the input
  spec (8-digit N ⇒ 24 bits ⇒ K ≥ 26, W ≥ 26; 2-digit T ⇒ 7 bits), or a test on synthesized inputs.
- With port wiring, the extra factors add only benign aliases: the 36 port permutations. Certification counts are
  identical across permutations.
- Ablation (`--alias-demo`): drop the entry/exit round trip. T=1 labels then keep every (Ke, Km, Kx) with 2Ke = Km + Kx,
  i.e. 8 off-diagonal machines such as Ke=24, Km=26, Kx=22 (T=1 output x²·2^(2Ke−Km−Kx) = x²). All 8 fail the 1,000
  T=2 rows (exponent 4Ke − 3Km − Kx ≠ 0). So either the label-free invariant or the T=2 rows pins K jointly.

## How big a grammar fits in a GPU-hour

- **Unfactored** (every factor per lane, labels only, ≈ 1.0003 rows per machine thanks to early rejection): the joint
  per-lane-cell machine runs 2.1–2.8e7 machine-rows/s on the H100 (eager), so about **2^36.1–2^36.5 machines per
  H100-hour**. MPS gives 2^33.7 per hour and the torch CPU 2^31.9. The previous estimate, "~2^30 in minutes on an
  H100", holds: 2^30 takes about 40–50 s eager. The fused hot block is 6–13× faster on the H100: 4.9e8 machines/s with
  per-lane cells. A fused implementation would therefore land around 2^39–2^40 per H100-hour. That is an estimate,
  because the front end was not fused.
- **Factored** (label-free round trips per block, labels last): costs add instead of multiplying. The cost is
  |gate × io| + |A survivors| · |entry/exit| + |B survivors| · |core|.
  - Measured: the 2^54.4 grammar takes 11.2 s of H100 screening. The **2^63.9** grammar takes 2^35.4 evaluations,
    **19.7 min on one H100 (eager)**, i.e. a third of a GPU-hour.
  - Extrapolated, keeping this three-block structure and its survivor counts (7,782 after A, 64 after B), with the hour
    split evenly at the measured eager rates (A 2.9e7/s, C 4.8e7/s): |gate × io| ≈ 2^35, |entry/exit| ≈ 2^22,
    |core| ≈ 2^29.7. That is about **2^87 joint machines per H100-hour**. This is an extrapolation of grammar size,
    not a measurement.
- What actually limits it:
  - **Survivor multiplicity.** Stage A's 7,782 degenerate gate × converter pairs multiply stage B. Without stage B's
    invariant, stage C would be 7,782 × 128 × 979,776 = 2^39.8 evaluations, about 10 H100-hours eager instead of 2.4 s.
  - **A label-free invariant for every block.** The core has none here, so labels pay for it.
  - **Off-distribution switches**, which no amount of compute on training data can pin.

# The Montgomery core gets its own label-free laws; radix 4; a label-free search

Scripts: `spec_grammar.py` (two-operand MonPro, four multiplier families, MonPro laws, stages C0/C1, `--ablate`),
`modal_h100.py` (detached `--spawn` runs, results on the `specbit-runs` Volume). Logs and JSON: `runs/h100_search_*`,
`runs/law_small_mps.*`. H100 = Modal H100 80GB, eager, 2^22-lane chunks (2^24 for the ports grammar). Every H100 run
reads A and B's survivors from one cached run (A 30.7 s, B 0.5 s on the H100); wall times below add them back.

## What changed in the grammar

- **MonPro(a, b)**, partial products a_i & b; the machine squares with MonPro(s, s). A shared factor `ppop` picks the
  operand roles: a.b, b.a, and the squaring-only a.a, b.b (correct for this task, wrong as a two-operand product).
- **Four multiplier families** (`mult`), each with its own wiring factors:
  - r2: radix 2, two carry-save cells per bit. This is the old core with its port wiring, and it is the reference.
  - r2m: radix 2, one cell per bit. The quotient bit comes first (LSB of Ss + Sc + a_i b), then one row from a
    precomputed {0, b, N, b+N}. Factors: quotient formula (6) × which row each (a_i, q) selects (24).
  - r4: radix 4, Km/2 iterations (13 for R = 2^26). Per 2-bit digit, rows Ss, Sc, a_2i b and a_2i+1 b << 1 meet in two
    carry-save cells. The 2-bit quotient comes from the low two bits of that sum and N mod 4. Rows q0 N and q1 N << 1
    meet in two more cells, then the sum shifts right by 2, leaving a carry for bit 0 of the next iteration. Factors:
    tree shape (which row skips the first cell, 4) × quotient-row shape (3) × quotient formula (8) × carry source (5)
    × shifts.
  - r4m: radix 4 on precomputed 3b and 3N, two cells per digit. Factors: which multiple each digit selects (24) ×
    which multiple each quotient selects (24) × quotient formula × carry source.
  - `--core ports` adds r2's port sources and per-port shifts (27 per cell) on every cell of r2m, r4 and r4m.
- Checked before any search (scratch tests, not in the repo):
  - All four families compute the same integer, (ab + mN)/R, since m = -ab/N mod R is unique.
  - They agree bit-exactly with big-integer Montgomery on 40 random (a, b, N) with 13–24-bit N, including N ≡ 3
    (mod 4).
  - Each family's reference machine certifies T=64 on both profiles.
- **Every modulus in every split is ≡ 1 (mod 4).** The generator draws N = pq with p ≡ q ≡ 3 (mod 4) (Blum integers;
  `factor_congruence` in `dataset_config.json`). So a radix-4 quotient that ignores N mod 4 is exact on the whole
  benchmark. On N ≡ 3 (mod 4) it gives 0/15 exact products, against 15/15 for the general formula.

| grammar | MonPro block per family | core incl. cw | joint (gate × io × entry/exit × core) |
|---|---|---|---|
| small (`--core small`) | r2 1.31M, r2m 124k, r4 3.73M, r4m 19.9M | 75.2M = 2^26.2 | **2^60.7** (first runs: 2^54.4) |
| ports (`--core ports`) | r2 952M, r2m 3.36M, r4 907M, r4m 14.5G | 4.91e10 = 2^35.5 | **2^70.0** (first runs: 2^63.9) |

## Stage C0: the core alone against label-free MonPro laws

- C0 screens the core once per distinct (gate, entry/exit) among B's survivors, which gives 8 contexts.
- It runs on the prompts' own (N, x) in binary. The io block and the T counter are not read, so B's io aliases and the
  cw factor don't multiply this stage.
- Stage C then joins each B survivor with the C0 survivors of its own entry/exit, × cw, against the 64 T=1 labels.
- Without C0, stage C is B's 64 survivors × the whole core × cw, as in the first runs.

| law | both sides (a = entry(x), 1~ = entry(1), a' = entry(N − x)) | synthesizes |
|---|---|---|
| id | exit(MonPro(a, 1~)) = x | 1 |
| comm | MonPro(a, 1~) = MonPro(1~, a), bit-exact | 1 |
| comm_sq | MonPro(a, a²) = MonPro(a², a) | — |
| assoc | exit((a²)²) = exit(a·(a·a²)) | — |
| mirror | exit(a²) = exit(a'²) | N − x |
| mirror2 | exit((a²)²) = exit((a'²)²) | N − x |
| unit | exit(MonPro(1~, 1~)) = 1 | 1 |

What each law can see. P = passes; checked on explicit machines (scratch test) and borne out by the ablation below:

| machine | id | comm | comm_sq | assoc | mirror, mirror2 | unit |
|---|---|---|---|---|---|---|
| correct MonPro, any family, any recoding | P | P | P | P | P | P |
| squaring-only operand roles (a.a, b.b) | – | – | – | – | P | P |
| left projection f(a, b) = a (a degenerate datapath) | P | – | P | P | – | P |
| Km ≠ 2Ke − Kx | – | P | P | P | P | – |

- **The identity law has a projection blind spot.** Any f(a, b) = a satisfies f(a, 1) = a.
  - On the small grammar, 74,940 radix-2 cores pass it, across the 8 contexts. Only 96 of them are correct MonPros.
  - In the reference context, 9,504 of the 9,516 radix-2 cores that pass it are exact left projections: carry-save
    datapaths that return the multiplicand unchanged for an arbitrary second operand. The other 12 are the correct
    ones.
  - For r2m the blind spot is broader: of 324 cores that pass, 24 are projections and 288 are right only with 1~ as an
    operand.
  - Commutativity on (a, 1~) removes all of them and nothing else. Commutativity on (a, a²) and associativity do not,
    because for a projection a² = f(a, a) = a, so both sides agree.
- **id and comm test a two-operand product, which is more than the task needs.** They reject the squaring-only
  multipliers, which labels keep and which certify.
- **unit + mirror are squaring laws.** They keep exactly what the labels keep.

Law ablation (H100). C0 screens the small grammar's 200,558,592 core machines; C then applies the 64 T=1 labels.
"Labels only" is the first runs' pipeline on this grammar.

| law suite | C0 survivors (r2 / r2m / r4 / r4m) | C0 | stage C machines | C survivors |
|---|---|---|---|---|
| labels only (no C0) | — | — | 4,813,406,208 | 40,320 (123.0 s) |
| id | 84,030 (74,940 / 2,562 / 5,376 / 1,152) | 12.1 s | 2,016,720 | 20,160 |
| comm | 1,913,908 | 33.2 s | 45,933,792 | 20,160 |
| comm_sq | 2,295,164 | 36.2 s | 55,083,936 | 20,160 |
| assoc | 2,550,686 | 50.3 s | 61,216,464 | 20,160 |
| mirror | 1,409,264 | 30.7 s | 33,822,336 | 40,320 |
| unit | 166,440 | 8.9 s | 3,994,560 | 40,320 |
| **id, comm** | **6,720 (96 / 96 / 5,376 / 1,152)** | 12.6 s | 161,280 | **20,160** |
| **unit, mirror** | **13,440 (192 / 192 / 10,752 / 2,304)** | 9.9 s | 322,560 | **40,320** |
| id, comm, assoc | 6,720 | 14.1 s | 161,280 | 20,160 |

- **On the core, a law pair is exactly as strong as the labels.**
  - Every id+comm survivor fits all 64 T=1 labels with the ×10 io, and later every training row. The labels then
    remove only the ×12 and ×16 io aliases that stage B leaves (8 io variants per entry/exit, 1 correct).
  - Against labels only, id+comm loses exactly the 20,160 squaring-only machines and nothing else. unit+mirror loses
    nothing.
- assoc adds nothing at training widths. Its use is at the declared width, below.

## Survivors after each stage, with and without the core invariant (small grammar, 2^60.7 joint, H100)

| stage | check | labels only | id + comm | unit + mirror | label-free, two-operand laws | label-free, squaring laws |
|---|---|---|---|---|---|---|
| A | b2d(d2b(v)) = v (no labels) | 188,743,680 → 7,782 | same | same | same | same |
| B | exit(entry(x)) = x (no labels) | 996,096 → 64 | same | same | same | same |
| C0 | MonPro laws on the prompts (+ 16 synthesized 24-bit Blum moduli in the last two) | — | 200,558,592 → 6,720 (14.5 s) | → 13,440 (11.7 s) | → 2,240 (21.2 s): id, comm, assoc | → 4,480 (17.7 s): unit, mirror, mirror2 |
| C | 64 T=1 labels; last two: y(x, 1) prints decimal digits (no labels) | 4,813,406,208 → 40,320 (123.0 s) | 161,280 → 20,160 (3.2 s) | 322,560 → 40,320 (3.2 s) | 53,760 → 6,720 (1.4 s) | 107,520 → 13,440 (1.5 s) |
| C1 | y(x, 50+49) = y(y(x, 50), 49) (no labels) | — | — | — | 6,720 → 2,240 (32.8 s) | 13,440 → 4,480 (32.8 s) |
| D | all 101,999 training rows | 40,320 (186 s) | 20,160 (108 s) | 40,320 (193 s) | 2,240 (39 s) | 4,480 (47 s) |
| E | certify T=64 on both profiles | 4,480 (47 s) | 2,240 (31 s) | 4,480 (53 s) | **2,240 = all** (17 s) | **4,480 = all** (16 s) |
| | machines evaluated, A–C | 5.00e9 = 2^32.2 | 3.91e8 = 2^28.5 | 3.91e8 | 3.90e8 | 3.91e8 |
| | search wall to the last search stage | 154 s | 49 s | 46 s | 87 s | 83 s |

- **Cost.** C0 + C take 17.7 s against 123 s for labels only, with 24× fewer machines evaluated in the core stages.
  - The ratio has two parts: 8 entry/exit contexts in place of B's 64 survivors, and cw (×3) moving out of the screen.
  - Per machine, a law evaluation costs about 2.3× a T=1 label here. Its two operands differ, so operand roles and
    partial products are per lane, while the labelled machine squares one lane-free entry output.
- **Label-free search.** The last two columns read no label until D:
  - io and entry/exit round trips;
  - MonPro laws on the prompts and on synthesized Blum moduli at 24 bits;
  - the decimal format of the answer in place of labels at C;
  - the semigroup law of repeated squaring for the counter.
- **Every machine they return certifies.** The squaring version returns 4,480. That equals the labels-only search's
  certifying set: the label-free survivors fit every training row, so they lie inside it, and the counts match. The
  two-operand version returns the 2,240 of those that are also correct two-operand MonPros.

How the off-distribution switches fall to label-free checks at the declared ranges (labels at C; runs `synth24`,
`synth24a` and `pinned`, where Modal ran `synth24a` on an H200):

| label-free checks | removes | D survivors | certify T=64 both |
|---|---|---|---|
| id, comm on the prompts | — | 20,160 | 2,240 (11%) |
| + id, comm on 16 synthesized 24-bit Blum moduli | Km = 22 (bits of a lost) and W = 24 | 10,080 | 2,240 (22%) |
| + assoc on the prompts and at 24 bits | Km = 24 | 6,720 | 2,240 (33%) |
| + y(x, 50+49) = y(y(x, 50), 49) | cw = 5, 6 | 2,240 | 2,240 (100%) |

- **id, comm and unit don't catch Km = 24 at 24 bits.** Their operands are reduced below N < 2^24, so 24 scanned bits
  suffice. The failure in the machine comes from repeated squaring of unreduced operands below 2N, which have a 25th
  bit. assoc and mirror2 square a square, so they catch it.
- The counter is not part of MonPro, so no MonPro law can see cw. The semigroup law with t1 + t2 = 99, the largest
  2-digit T, needs 7 counter bits. With t1 + t2 = 31, the training range, every cw passes.

## Equivalent variants the search found

Nobody placed these as the reference. All certify T=64 on both profiles (counts from the id+comm run; labels only
doubles each with the squaring-only operand roles):

- **Radix 4 is found, and certifies:**
  - r4: 1,792 of the 2,240 certifying machines.
  - r4m: 384.
  - Next to r2: 32, and r2m: 32.
- **r4's tree shapes are all equivalent:** all 4 choices of the row that skips the first cell, × all 3 places for the
  two quotient rows.
- **r4's carry into bit 0 has five working forms:**
  - u4₁, v4₀, u4₁ ∧ v4₀ and u4₁ ∨ v4₀ all work.
  - Dropping the carry also works, but only for the two quotient-row shapes whose last cell adds a row shifted by 1.
    Bit 0 of that cell then has one nonzero input, so no carry ever forms. It fails for the third shape.
- **Quotient formulas that rely on the Blum moduli** (exact on N ≡ 1 mod 4, wrong on N ≡ 3 mod 4):
  - q1 = S1 ⊕ S0, which ignores N mod 4 (r4 and r4m).
  - In r4m, any bijective recoding of the 2-bit quotient paired with the matching permutation of the {0, N, 2N, 3N}
    table: (S0, S1) with (0, 3N, 2N, N), (S1, S0) with (0, 2N, 3N, N), (¬S0, S1) with (3N, 0, N, 2N). That makes 6
    (formula, table) pairs.
- **r2m:** the inverted quotient bit with the swapped table (N, b+N, 0, b) is exact for every N.
- **Operand roles:** b.a always works; a.a and b.b work for squaring.
- **Widths:** W = 26 is enough for every family, radix 4 included; K ∈ {26, 28}.
- Known from the first runs: the radix-2 hand-off scaled by 2, and the 36 port permutations (ports grammar).

## The largest grammar searched (ports, 2^70.0 joint)

`--core ports --law id,comm`, one H100, 2^24-lane chunks (`runs/h100_search_law_ports.*`): **60.4 min wall**
(+ A/B 31 s from the cache).

| stage | machines → survivors (r2 / r2m / r4 / r4m) | time |
|---|---|---|
| C0, id then comm (no labels) | 130,996,611,072 = 2^36.9 → 15,072 (3,456 / 192 / 9,408 / 2,016) | 3,282 s (54.7 min) |
| C, 64 T=1 labels | 361,728 → 45,216 (10,368 / 576 / 28,224 / 6,048) | 3.7 s |
| D, every training row | 45,216 → 45,216 | 271 s |
| E, certify T=64 on both | 4,016 (1,152 / 64 / 2,224 / 576) | 66 s |

- **Cost.** C0 ran at 4.0e7 machines/s, so cost grows linearly with the core: ×653 in core size took ×226 in C0
  time, the rest being larger chunks.
  - Labels alone would screen 64 × 4.91e10 = 3.1e12 machines at stage C. At the label rate measured on the small
    grammar (3.9e7 machines/s) that is about 22 H100-hours (extrapolated, not run). The law-first search took under an
    hour.
  - The largest core searched is 2^35.5 machines with cw (2^33.9 MonPro block); joint 2^70.0.
- **Survivor multiplicity grows with the wiring factors, as before.** Radix 2 gains its 36 full-adder port
  permutations (96 → 3,456 in C0). The per-port shifts add a ×2 alias in r2m (all three ports shifted by 1, with the
  ×2 hand-off) and new aliases in r4 and r4m.
- **The new radix-4 aliases are off-distribution:**
  - Shifting two ports of the first cell by 2 multiplies the partial products by 4. That is Montgomery with R/4,
    compensated by Km + 2.
  - Such machines fit every training row, then fail on the unseen 24-bit moduli: their effective R is too small, or
    their 4×-scaled datapath overflows W = 26.
  - That is why r4 certifies 2,224 of 28,224 (7.9%), against 1/9 in the small grammar.
- **Post-hoc, the declared-range checks on these 45,216 machines (MPS, `runs/posthoc_*_mps.log`):**
  - The checks: id, comm and assoc on the 16 synthesized 24-bit Blum moduli, then the semigroup law at 50 + 49. They
    leave 5,016 machines: all 4,016 certifying ones, plus 1,000 that do not certify.
  - Edge-of-range inputs remove 536 of those 1,000: 32 Blum N just below 2^24, x near N, through assoc. The other 464
    pass every check. They are shifted-port radix-4 variants (304 at W = 26, 160 at W = 28) whose scaled datapath
    overflows on rare 24-bit inputs; 264 of them fail exactly 1 of the 5,376 unseen-N certification rows.
  - A few dozen synthesized inputs cannot rule out a rare overflow. Ruling it out takes a bound on intermediate
    values: a width law per cell, not more samples.

## MPS

- **The laptop was shared** with about 16 jobs from sibling sessions during these runs (load average 9–45). So these
  wall times are not comparable to the first runs' MPS numbers: stage C of the 2^54.4 grammar took 14.3 s then.
- **Survivors match the H100 at every stage.** A and B come from the H100 cache (`--ab-cache`).

| MPS run (small grammar) | C0 | C | C1 | wall after B |
|---|---|---|---|---|
| id + comm, then labels (`runs/law_small_mps.*`) | 200,558,592 → 6,720 (430 s) | 161,280 → 20,160 (19.3 s) | — | 450 s |
| label-free, squaring laws at prompts and 24 bits, decimal C (`runs/labelfree_sq_small_mps.*`) | → 4,480 (383 s) | 107,520 → 13,440 (9.0 s) | 13,440 → 4,480 (124 s) | 516 s |

- **D and E ran only on the H100.** On the shared laptop, D on the 20,160 equivalent machines was still on the T=4
  rows after 14 minutes, with the T=8 and T=16 rows to go, so I stopped it (`law_small_mps` was rerun with
  `--stop-after C`). The H100 does that D in 108 s.

## Ideas, rated by promise

1. **A label-free law for every block, checked at the declared input range. Promise: high; demonstrated here.**
   - On this grammar the laws replace every label in the search:
     - round trips for io and entry/exit;
     - MonPro laws for the core;
     - the semigroup law for the counter;
     - the answer's decimal format for the io base.
   - Labels only verify, and every machine returned certifies.
   - The ranges are priors from the task spec (24-bit moduli, 2-digit T, Blum moduli, decimal digits), and choosing
     them matters: t1 + t2 = 31 pins nothing, and id + comm drop correct squaring-only designs.
   - Next: a target the grammar was not built around, such as `menu.py`'s other recurrences or another modulus
     family, where the laws must come from the spec rather than from knowing Montgomery.
2. **Generate the laws instead of choosing them. Promise: high; not tried.**
   - The laws here are a handful of algebraic identities. A small grammar of identities over a block's inputs could be
     scored on sampled wrong designs (how many it rejects) and on the reference (it must pass).
   - The projection blind spot says to score suites, not single laws.
3. **Sub-block laws inside the core. Promise: medium-high; tried in the next section.**
   - Each carry-save cell preserves value (u + 2v = a + b + c), the quotient zeroes the low bits, and the hand-off
     divides exactly.
   - Checked per cell on synthesized words, these would factor the core itself. The port wiring would cost
     729 + 729 instead of 729 × 729, and r4m's per-port shifts 27 + 27 instead of 27 × 27.
4. **Cut survivor multiplicity before D and E. Promise: medium; this is the binding cost now.**
   - The certifying sets are genuine equivalents: tree shapes, carries, quotient recodings, operand roles, the ×2
     hand-off, and in the ports grammar 36 port permutations.
   - D costs 108 s for 20,160 machines on the H100 and grows with them.
   - Two remedies: canonicalize known symmetries (full-adder port permutations, operand roles), or keep one machine
     per behavior fingerprint on synthesized inputs.
5. **Fuse C0. Promise: medium.**
   - C0 runs at 1–4e7 machine-rows/s eager. The fuse probe gave 6–13× on the hot block.
   - Here each chunk changes which options are present, and with them the traced graph. A formulation that always
     computes every option with data-driven routing (as `perm_mux` does) would give one graph per family.
6. **More multiplier families. Promise: low for this task, high as a test of the laws.**
   - Candidates: Booth radix 4 (negative digits), MSB-first interleaved (Blakley) multiplication with K = 0 entry and
     exit, and Barrett reduction.
   - Do the laws stay as strong as labels for products without Montgomery's (ab + mN)/R form?

Compliance, neutral and short: the laws synthesize inputs the prompts don't contain: the constant 1, N − x, 24-bit
Blum moduli (a range read from the task spec) and T = 50 + 49. Whether synthesized checks and spec-derived ranges are
admissible belongs to the strict review before a real submission.

# Per-cell laws inside the Montgomery core

Code: `spec_grammar.py` (`CELL_LAWS`, `cell_law_bad`, `cell_stage`, `width_law`; flags `--cell-law`,
`--width-law 24`, `--cell-audit`). Runs: `runs/cell_*`, `runs/posthoc_sets_ports.*`,
`runs/h100_search_cell_ports.*`, `runs/h100_search_cellfree_ports.*`. The four families' iterations are now built
from shared cell functions, bit-exact against the previous code (4,096 random machines per family, both grammars), so
each sub-block law screens exactly the wiring the MonPro runs.

| law | sub-block | check (no label) |
|---|---|---|
| value | each carry-save cell | u + 2v = 2^j × (the cell's rows at their natural weights), one j ∈ {0, 1, 2} per cell, on 32 synthesized rows |
| quotient | each iteration | the lane's quotient digit makes S + d·b + q·N = 0 mod 2^r (r = 1 or 2), on 18-bit Blum N |
| hand-off | each iteration | the new state rows hold (u + 2v) / 2^r exactly |
| width | every port, carry row, register and precomputed multiple | a bound, not samples: no set bit can fall off a 28-bit word or a W-bit register in any iteration, for N < 2^24, operands below the Montgomery closure bound (s' < s²/R + N, R = 2^(2Ke − Kx)), all-ones digits and maximal quotient digits |

The width law bounds every signal twice. The value bound keeps u + 2v = (the cell's sum) and Ss + Sc = (one state
value) when it pushes a bound back through the cells. The width bound follows the bitwise cell: u is as wide as the
cell's widest port, v as its second widest. A signal passes if either bound fits.

## (b) The 464, and what certification misses (post hoc on the ports run, 3 s on MPS)

| checks | all 45,216: certify / not | the 4,480 that pass every current label-free check: certify / not |
|---|---|---|
| without per-cell laws | 4,016 / 41,200 | 4,016 / 464 |
| value laws | 3,392 / 27,136 | 3,392 / 0 |
| width law, 24-bit N | 3,296 / 6,592 (only cw = 5, 6 copies) | 3,296 / 0 |
| both | 3,296 / 6,592 | 3,296 / 0 |

- **Both laws remove all 464.**
  - 160 of the 464 have W = 28. With quotient-row shape (u,v,Q0)(Q1), cell 3's carry row can reach bit 27 of the
    28-bit word.
- **They also remove certifying machines, and every one of them is wrong on declared-range inputs.**
  - The value laws remove the whole ×4 class, 624 certifying. Those machines scale the partial products by 4 against
    the state rows, so cell 1 doesn't conserve value.
  - The width law removes those 624 plus 96 r4m machines at Km = 26, W = 26, whose precomputed 3b needs 27 bits once
    b ≥ 2^26/3.
  - On 8,982 rows (24-bit Blum N, x < N, T ≤ 64), the 4,480 split exactly as the width law says. The 1,184 it rejects
    each get at least 200 rows wrong; the 3,296 it keeps get none wrong.
  - 8,192 of those rows have N near 2^24 (12.3M–16.7M) and T = 64.
  - The other 790 come from an exhaustive search over every x for the four largest 24-bit Blum moduli: chains whose
    unreduced operand reaches s ≥ 1.5·2^24. Two moduli have such chains (max s = 1.53 N); 1.28M random squarings
    never got past 1.40 N.
  - 144 of the ×4 machines at W = 28 fail only on those 790 rows: their cell-2 carry reaches bit 27 only when
    b ≥ 1.5·2^24.
- **Certifying T=64 on both profiles is a sample; with exact cells, the width law is a bound.** The unseen-N profile
  has 190 distinct 24-bit moduli, averaging 11.0M (max 16.44M), and these overflows need N near 2^24 with operands
  near 2N.
- On all 45,216 the width law alone leaves 3,296 × 3 counter widths. Only the semigroup law sees cw.

## (a) Factoring the core: C0 cost on the ports grammar (2^70.0 joint, one H100, eager, A/B cached)

Sub-block screens, machines → survivors:
- r2: cell 1 216 → 12, quotient 84 → 6, cell 2 729 → 12, hand-off 108 → 12; 72 wirings.
- r2m: 3,888 → 6, hand-off 54 → 4; 4 wirings.
- r4: 108 → 12, 108 → 12, quotient 288 → 8, quotient rows 9 → 3, hand-off 135 → 14; 112 wirings.
- r4m: 648 → 3, 27 → 3, quotient 576 → 6, hand-off 135 → 4; 24 wirings.

That is 7,113 machines in 0.3 s, with identical survivors over 5 seeds of synthesized words. They are the equivalents
C0 used to find jointly: port permutations, the ×2 hand-offs, r4's tree shapes and carries, and r4m's 6 quotient
recodings with their tables.

| stage | without (`h100_search_law_ports`) | with per-cell laws (`h100_search_cell_ports`) | label-free + width law (`h100_search_cellfree_ports`) |
|---|---|---|---|
| sub-block screens | — | 7,113 machines, 0.3 s | same |
| C0, MonPro laws | 1.31e11 = 2^36.9 → 15,072, 3,282 s | 162,816 = 2^17.3 → 10,176, 3.4 s | + id, comm at 24 bits: → 5,088, 6.8 s |
| width law | — | — | → 3,296, 1.7 s |
| C | 64 T=1 labels: 361,728 → 45,216, 3.7 s | labels: 244,224 → 30,528, 2.9 s | decimal digits: 79,104 → 9,888, 1.5 s |
| C1, semigroup 50+49 | — | — | → 3,296, 40.7 s |
| D, every training row | 45,216, 271 s | 30,528, 184 s | 3,296, 46 s |
| E, certify T=64 on both | 4,016 | 3,392 (= 4,016 − the 624 ×4) | **3,296 = all** |
| machines evaluated, A–C | 2^36.9 | 2^27.5 (stage A's 188.7M dominates) | 2^27.5 |
| wall | 60.4 min | 4.0 min | 2.0 min |

- C0 screens 800,000× fewer machines, because costs add per cell instead of multiplying across cells. On the shared
  laptop's MPS the same C0 takes 3.2 s.
- The label-free run reads no label until D and returns 3,296 machines, all certifying. That is exactly the set the
  witness rows leave standing.

## Rating and next

- **Promise: high.**
  - On the largest grammar, C0 drops from 55 minutes to seconds, and the grammar can now grow per cell at additive
    cost.
  - The width law is the first check here that is a bound, not samples. On this grammar it agrees exactly with the
    witness search, and it found 720 machines that pass both certification profiles but are wrong on 24-bit inputs.
- **What it costs:**
  - The value laws need a per-cell spec: which rows a cell adds, and at what weights. That is more design knowledge
    than the black-box MonPro laws.
  - One scale per cell rejects the ×4 aliases at any W. At W ≥ 30 those would be correct, so one scale per row class
    (state, digit, quotient rows) is the more general law.
  - The width law assumes exact cells and takes the closure bound from the context's R. It treats every digit and
    quotient digit as maximal at once, which is conservative, though here it was exact.
- **Next:**
  1. Grow the grammar where factoring makes it cheap: per-port shifts on r4's cells 3–4, W and K up to 32, Booth
     digits.
  2. Use the width law as the acceptance check, or add a declared-range certification profile (N near 2^24, chains
     with large operands).
  3. Derive each block's operand range by interval propagation from the input spec (entry, core, exit), instead of
     the hand-derived closure bound.
  4. Check the quotient laws exhaustively over their 2–3 input bits instead of 32 random rows, so each sub-block law is
     a proof.

Compliance, one line: the per-cell laws synthesize words and moduli the prompts don't contain, and the width law reads
the declared modulus range; whether that is admissible is for the strict review.

# Growing the core where per-cell factoring makes it cheap

Code: `spec_grammar.py` (K and W options up to 32 with every wire 32 bits, `--kw-max 28` for the earlier widths;
`r4c3sh`, `r4c4sh`; digit scales in the value laws, `DIGIT_SCALES`, `--digit-scales`; `width_law` reads the operand
bound into the digits; `witness_rows`, `--cell-audit --witness`). Runs, one Modal H100 each, eager, 2^24-lane chunks,
A and B recomputed: `runs/h100_search_cellfree_k32.*` (the requested search), `runs/h100_search_nowidth_k32.*` (the
same without the width law, which gives the witness check its population), `runs/h100_search_witness_k32.*` (the
witness check), `runs/cell_audit_cellfree_k32.*` (the cell audit of the first, on the laptop's CPU).

## What grew

| | before (the 2^70.0 ports grammar) | now |
|---|---|---|
| Ke, Kx, Km | 22–28 | 22–32 |
| W | 24–28; every wire 28 bits | 24–32 as the register width; every wire 32 bits |
| r4 cells 3 and 4 | one row shift, `r4q1sh` (3) | per-port shifts `r4c3sh`, `r4c4sh` (27 each); `r4q1sh` joins `r4p1sh` in the small grammar |
| value law | one scale per cell | one scale per row class: a cell's u + 2v = 2^j (state and quotient rows + 2^e × digit rows), j per cell, e ∈ {0, 1, 2} per iteration |
| Booth radix 4 | — | not added (below) |
| joint, gate included | 2^70.0 | **2^76.3** (core with cw 2^35.5 → 2^40.7; r4's MonPro block 9.1e8 → 5.5e11) |

- **The row classes.** State rows are Ss, Sc, the carry into bit 0 and the running sums of earlier cells. Digit rows are
  the multiples of b. Quotient rows are the multiples of N. The quotient rows get no scale of their own: at another
  scale than the state rows they cannot zero the state's low bits.
- e is shared by the cells of an iteration. So each sub-block law that reads digit rows runs once per e, and a wiring
  must pass every sub-block at the same e.
- **Checked before the runs:**
  - With `--kw-max 28` the file is bit-exact against the previous one: 256 random machines per family, law sides and
    whole machines, ports and small grammars. The first runs' sizes (2^54.4, 2^63.9) and the small grammar (2^60.7)
    come back.
  - The reference design of every family certifies T=64 on both profiles at every K ∈ {26, …, 32} × W ∈ {26, …, 32}.
- **The width law got tighter where the new options needed it.**
  - Operands are below the closure bound s*, so multiplier bits at or above s*'s bit length are 0. The law now uses
    that.
  - From R = 2^28 up, s* ≈ 1.07 N has 25 bits while Km scans up to 32. Treating those top digits as all ones flagged a
    final sum that cannot overflow: 2 r4m machines with no witness, in a scratch check.
  - On the earlier run's 45,216 machines (28-bit wires) the tightened law gives the same verdict for every machine. It
    still matches the earlier witness rows on all 4,480.

## Sub-block screens and C0

| | 2^70.0 ports (`h100_search_cellfree_ports`) | 2^76.3 grown (`h100_search_cellfree_k32`) |
|---|---|---|
| sub-block screens | 7,113 machines, 0.3 s | 20,744 machines, 0.1 s |
| wirings they leave | 212 (r2 72, r2m 4, r4 112, r4m 24) | 1,108 (r2 216, r2m 4, r4 816, r4m 72) |
| C0 machines (wirings × ppdir, ppop, Km, W × entry/exit contexts) | 162,816 = 2^17.3 (8 contexts) | 3,191,040 = 2^21.6 (12 contexts) |
| C0: id, comm on the prompts and at 24 bits | → 5,088, 6.8 s | → 53,440, 7.3 s |
| C0 without per-cell laws | 1.31e11, 55 min (run) | 7.1e12 = 2^42.7: about 49 H100-hours at 4.0e7/s (extrapolated) |
| stage C with labels only, no C0 | 2^41.5 | 2^47.3 |

- **Wirings by digit scale:**
  - e = 0: 372 (r2 72, r2m 4, r4 272, r4m 24).
  - e = 1: 440 (r2 144, r4 272, r4m 24). This is MonPro with R = 2^(Km − 1). 2Ke − Kx is even here, so C0's identity
    law removes every one of them.
  - e = 2, the ×4-scaled datapath: 296 (r4 272, r4m 24).
  - r2's cell 1 shifts by at most 1, so r2 has no e = 2. r2m's single row adds b and N together, so it has only e = 0.
- **r4's back ends go from 14 to 34: a ×2 hand-off for radix 4.**
  - Cells 3–4 double the value (c4sh = (1, 1, 1), or c3sh = (1, 1, 1) with the last row shifted once more), and the
    hand-off divides by 8 (r4osh 3, r4csh 2).
  - Only the two tree shapes whose last cell adds a shifted row have them. There the carry into bit 0 is always 0, so
    every carry option works.
  - With the reference front end, all 34 certify T=64 on both profiles.

## Survivors at each stage (label-free: no label is read until D)

| stage | with the width law (requested) | without it |
|---|---|---|
| A, io round trip | 188,743,680 → 7,782, 7.1 s | same, 7.3 s |
| B, entry/exit round trip | 2,241,216 → 96 (8 io × 12 entry/exit), 0.4 s | same |
| sub-block screens | 20,744 machines → 1,108 wirings, 0.1 s | same |
| C0, MonPro laws | 3,191,040 → 53,440, 7.3 s | same, 7.5 s |
| width law | → 34,560 (r2 4,608, r2m 256, r4 27,200, r4m 2,496), 20.9 s | — |
| C, decimal digits | 829,440 → 103,680, 3.4 s | 1,282,560 → 160,320, 4.5 s |
| C1, semigroup 50+49 | → 34,560, 46 s | → 53,440, 45 s |
| D, every training row | 34,560, 306 s | 53,440, 446 s |
| E, certify T=64 on both | **34,560 = all** | 37,232 |
| machines evaluated, A–C | 2^27.5 | 2^27.5 |
| wall | 7.7 min | 10.1 min |

- **Every machine the requested search returns certifies.** That is 34,560 machines, against 3,296 before:
  - 23,712 at e = 0: every earlier design at 4 values of R (2^26 to 2^32, Km = 2Ke − Kx) × 4 values of W (26 to 32,
    except r4m's 3b, which needs 27 bits at R = 2^26), plus r4's ×2 hand-offs;
  - 10,848 ×4-scaled machines at e = 2, Km = 2Ke − Kx + 2 (r4 9,792, r4m 1,056).
- The width law removes 18,880 machines: 15,424 with R = 2^22 or 2^24 (≤ 4N, never certifying), and 3,456 at W = 26.
  Without it, 2,672 machines certify that the width law rejects. The witness rows below decide them.
- **The cost is now survivor multiplicity, not search.** C0 grew ×20 in machines and still takes 7 s. D and E take
  6.2 of the 7.7 minutes, because each correct design comes back once per (R, W) pair, per ×2 or ×4 scaling, and per
  port permutation.

## The ×4-scaled aliases

- **They pass the value laws now (e = 2), and the width law alone judges them.** Run without the width law, 23,680
  machines pass at e = 2. 13,424 of them certify.
- **With 32-bit wires they are exact at W ≥ 28, not W ≥ 30.** What needs 30 bits is a wire, not a register:
  - in r4, cells 1–4 reach about 4(4 s* + N) ≈ 2^29.2 at R = 2^26, while the state stays below 4 s* + N ≈ 2^27.2;
  - for r4's ×4 machines the width law accepts W = 28 once wires have 30 bits (29 at R ≥ 2^28), and no W with
    28-bit wires (scratch check, wire width varied by hand);
  - r4m's ×4 machines at R ≥ 2^28 even fit W = 26.

  In the earlier grammar W = 28 also meant 28-bit wires, hence "W ≥ 30".
- **They are genuine equivalents:** Montgomery with R = 2^(Km − 2), the digit rows scaled by 4. All 10,848 that the
  width law keeps certify T=64 on both profiles. 608 more, at W = 26, are correct but rejected (next section).

## Width law against the witness rows

The population is the 53,440 machines of the search without the width law: everything that passes every other check
and fits every training row. The rows are built as `runs/cell_witness_ports.py` and `runs/cell_chain_rows.py` built
them, now inside `witness_rows`:
- 8,192 random rows: 24-bit Blum N near 2^24, random x, T = 64;
- 8,452 chain rows: for each R ∈ {2^26, 2^28, 2^30, 2^32}, over every x of the four largest such N, the first 200 x
  whose exact chain passes halfway from 2^24 to s* (the earlier rule), plus the 200 that reach the largest operands.
  Each is taken at the squaring that reads that operand and at T = 64.

They took 12.9 min on the H100 (`runs/h100_search_witness_k32.*`).

| machines (by digit scale e) | count | width law | witness rows |
|---|---|---|---|
| certify, e = 0 | 23,712 | keep | none wrong |
| certify, e = 2 (×4) | 10,848 | keep | none wrong |
| certify, e = 0: r4m at R = 2^26, W = 26 (3b needs 27 bits) | 96 | reject | wrong |
| certify, e = 2 | 1,968 | reject | wrong |
| certify, e = 2 | **608** | reject | **none wrong** |
| don't certify (R ≤ 4N, and ×4 r4 at R = 2^26, W = 26) | 16,208 | reject | wrong |

- **The width law is still sound here.** No machine it keeps gets a single row wrong.
- **It is no longer exact.** It agrees with the witness rows on 52,832 of 53,440. The other 608 are correct machines it
  rejects.
  - All 608 are ×4 r4 at W = 26 with R = 2^28 or 2^30. They hold Ss or Sc back from cell 1, whose ports are shifted
    (0, 2, 2), and use one of the two quotient-tree shapes whose last cell adds a state row.
  - **Bit-level argument.** Follow the top bits through the four cells, with both state registers below 2^26. A
    hand-off row then needs a 27th bit only if the operand b has bit 24 set together with bit 23 (the sum row) or bit
    22 (the carry row), i.e. b ≥ 1.25·2^24. s* rules that out: it is 1.07 N at R = 2^28 and 1.016 N at R = 2^30. The
    argument holds at every iteration, for both held rows, both tree shapes and the ×2 back ends. At R = 2^26, where
    s* ≈ 2N, the same wirings do get witnesses.
  - **Checked directly** (`runs/monpro_scan_k32.*`, laptop CPU, 72 s): 1,048,576 products MonPro(s, s) per machine on
    the four largest 24-bit Blum moduli, with s just below s* (low 4 bits all ones, the worst case for iteration 1) or
    uniform below s*. All 608 are exact on every product. All 304 machines with the same 76 wirings at R = 2^26 get
    some product wrong.
  - **Why the law misses it.** It bounds each signal by a value and a width separately, so it cannot see which bits of
    b can be set together. Making carry rows narrower whenever the cell's sum bound allows (a carry bit j needs a sum
    ≥ 2^(j+1)) is sound, but it changed no verdict on either population, so it is not in the file.
- So the requested search returns 34,560 of the 35,168 correct machines in this population, and nothing incorrect.

## Booth radix 4: not added

- **Signed digits don't fit the unsigned carry-save cells as they are.** Two's-complement rows break the per-row
  hand-off, since each row would need its own sign extension. So the rows need offsets: −b as 3N − b and −2b as
  6N − 2b. That keeps every operand and state below 3N.
- **The offsets break bit-exact commutativity.** MonPro(a, b) becomes the canonical integer plus a multiple of N, and
  that multiple depends on which operand is scanned, because its negative digits pick the offsets. So the comm law in
  C0 would reject correct Booth machines. (Argued, not run.)
- **It would fit after three changes:**
  - comm compares exit outputs instead of raw integers;
  - the digit row gets a value law mod N: row ≡ d b (mod N), with 0 ≤ row < 6N;
  - the width law takes the 3N closure bound instead of the one from R.

  The family would look like r4m, with a 5-entry table and a 3-bit window.

## Rating and next

- **Promise: high, for growing a grammar.**
  - The joint space grew about 80× (2^70.0 → 2^76.3). The sub-block screens still take 0.1 s and C0 7 s, against an
    extrapolated 49 H100-hours unfactored.
  - Every machine the label-free search returns certifies, the ×4 aliases included.
- **The per-row-class law did what it was for.**
  - The ×4 aliases pass the value laws, and the width law alone decides them.
  - The bound placed the line at W = 28 rather than 30 once wires were 32 bits, and the witness rows agree there.
- **What it costs:**
  - Survivor multiplicity: 34,560 returned against 3,296. Every correct design comes back for each of 16 (R, W)
    pairs, each ×2 or ×4 scaling, and each port permutation. D and E are 6.2 of the 7.7 minutes.
  - The width law lost exactness: 608 correct machines rejected, with a proof that they are correct. It needed one
    more piece of the declared range (operand bits above s* are 0) to stay exact on the K = 32 options, and per-signal
    bounds cannot hold the bit correlations behind the 608.
  - Booth radix 4 needs a weaker commutativity law before it can join.
- **Next:**
  1. Collapse equivalents before D and E, using what the laws already know:
     - the scale class (e, j, ×2 hand-offs) and port permutations from the sub-block screens;
     - (R, W) pairs the width law accepts for the same wiring.

     Certify one machine per class. That would cut D and E by one to two orders of magnitude.
  2. Turn each width-law flag into a proof or a witness: ask a SAT or BDD query whether the flagged register bit can be
     set over the declared operand range. That is what the bit-level argument above did by hand.
  3. Give wires their own width factor, so the width law picks wires and registers separately. The ×4 aliases need
     30-bit wires but only 28-bit registers.
  4. Booth radix 4 with offset rows, comm compared after exit, and a mod-N row law. This tests whether the laws stay
     as strong as labels for a product that isn't Montgomery's canonical integer.

Compliance, one line: as before, the laws synthesize inputs and read the declared 24-bit modulus range; whether that is
admissible is for the strict review.

# One machine per equivalence class before D and E

Code: `spec_grammar.py` (`--collapse`, `--collapse-check K`; `design_key`, `collapse`, `collapse_check`, `certify`).
Run: `runs/h100_search_collapse_k32.*`, the requested label-free search of the previous section with `--collapse` (one
Modal H100, eager, 2^24-lane chunks, A and B recomputed). Checks against earlier runs: `runs/collapse_classes_k32.*`
(CPU, about a minute).

## The classes

The survivors after C1 are grouped by what the laws already establish:
- **Shifts are scales.** With every other factor fixed, the shift settings that pass a family's value and hand-off laws
  differ only in scale: the cell scale j, the digit scale e, and the hand-offs that undo a ×2 cell. A value law fixes
  each port's shift once its source and the scales are fixed.
- **Port order.** The full adder is symmetric, so permuting r2's port sources together with their shifts gives the same
  outputs bit for bit.
- **(R, W).** With exact cells and no lost bit (the width law), MonPro is the canonical integer for
  R = 2^(Km − e) = 2^(2Ke − Kx), which the id law pins. So the answer is the same at every (R, W) the width law
  accepts.

A class keeps every factor except the shifts, takes r2's port sources as a multiset, and drops (Ke, Kx, Km, W). The
representative is the member with the fewest factors off the reference. The family, tree shapes, quotient formulas and
tables, carry forms, operand roles and entry radix still separate classes.

| family | classes × members | per class: port orders, scalings, (Ke, Kx, Km, W) | what separates the classes |
|---|---|---|---|
| r2 | 4 × 1,152 | 36, 2, 16 | entry radix, operand roles |
| r2m | 8 × 32 | 1, 2, 16 | those, and the quotient formula with its table (2) |
| r4 | 128 × 25 and 320 × 75 | 1, 2 (e = 0, 2) or 6 (also the two ×2 back ends), 25 (16 at e = 0, 9 at e = 2) | those, q4 (2), tree shape (4), quotient-row shape with carry (14) |
| r4m | 96 × 26 | 1, 2 (e = 0, 2), 26 | those, quotient formula with its table (6), carry (4) |
| all | **556 classes, 34,560 machines** | | |

## Stages, before and after

| stage | before (`h100_search_cellfree_k32`) | with `--collapse` (`h100_search_collapse_k32`) |
|---|---|---|
| A to C1, same survivors at every stage | 86 s | 115 s (a slower host: C1 61 s against 46 s, the width law 27 s against 21 s) |
| collapse | — | 34,560 → 556 classes, 0.6 s |
| D, every training row | 34,560 machines, 306 s | 556 representatives, 4.5 s |
| E, certify T=64 on both | 34,560 = all, 68 s | 556 = all, 15.8 s |
| **D + E** | **374 s** | **20.3 s** (18× less) |
| spot-check: E on 2 random non-representatives per class | — | 1,112, each certifying exactly as its representative, 29.3 s |
| wall | 7.7 min | 2.75 min (2.3 min to the end of E) |

- `evaluate_lanes` now also gives a family with few lanes more rows per pass. That is bit-exact against 64-row passes on
  40 machines, 24 of them wrong on some rows. Its share of the D speedup is not timed separately.
- The time left is in stages before the collapse: C1 (61 s), the width law (27 s, in Python, one machine at a time), C0
  (11 s) and A (9 s).

## Does a representative predict its class?

- **Here, on every member.** The 34,560 members are exactly the machines the earlier run certified one by one. Each
  member's own outcome there equals its representative's here: the same max certified T and the same number of exact
  rows, on both profiles. The 1,112 sampled non-representatives agree as well.
- That population certifies to the last machine, so it cannot show a class that splits. The search without the width
  law can: 53,440 machines, each run through E (37,232 certify) and through the 16,644 witness rows.
  - Merging shifts and port order only, with (Ke, Kx, Km, W) kept, gives 22,000 classes. Within every class the E
    outcomes and the witness outcomes (wrong rows per row source) are identical. That includes the 7,824 classes whose
    members get witness rows wrong. At a fixed Km this merges the ×2 hand-offs and the port orders; the ×4 datapath
    changes Km, so it merges only together with (R, W).
  - Merging (Ke, Kx, Km, W) as well, without the width law, mixes outcomes in all 556 classes: the R = 2^22 and 2^24
    members fail the unseen-N profile.
  - Within the 34,560 that the width law keeps, all 556 classes are uniform again.
- So the (R, W) merge rests on the width law. `--collapse` makes it only when `--width-law` is on.

## Rating and next

- **Promise: high, and cheap.** D and E now cost per design, not per alias, and every class checked so far predicts its
  members exactly.
- **Next:**
  1. Collapse before C1. The semigroup law only sees the counter, so it could run on one machine per (class, cw): 1,668
     instead of 103,680.
  2. Merge what is bit-identical in the machine:
     - operand roles, which a squaring machine never reads;
     - carry forms, since the quotient law makes u1 = v0;
     - q4's gen and n1, which agree on N ≡ 1 mod 4;
     - paired quotient recodings and tables, which select the same multiple of N.

     About 30 classes would remain: entry radix (2) × (r4's 12 tree shapes, plus one design per other family).
     Beyond that, the argument behind the (R, W) merge covers every class here, since all four families compute the
     same integer MonPro. The laws alone would predict the whole set from one machine. Keeping one class per design
     keeps E a test of each design, not only of that argument.
  3. Memoize the width law on the factors it reads (it ignores entry radix, operand roles and the io block), or
     vectorize it.

Compliance, one line: the collapse reads no labels; D and E use labels as before.

# Collapse before C1; the width law once per setting

Code: `spec_grammar.py` (`--collapse C1`: `classes_of`, `semigroup_pass`, `c1_check`; `WIDTH_IGNORES`,
`width_verdicts`). Runs, one Modal H100 each, eager, 2^24-lane chunks, A and B recomputed:
`runs/h100_search_collapse_c1_k32.*` (the requested search, `--collapse C1 --collapse-check 2`) and
`runs/h100_search_collapse_d_k32.*` (the same file with `--collapse` as before, a same-file baseline). Checks:
`runs/width_memo_k32.*` (laptop CPU, 17 s) and `runs/collapse_c1_k32.*` (reads the three runs' JSON).

## What changed

- **C1 on one machine per (class, cw).**
  - design_key keeps cw, and the rest of a class computes the same squarings. Within a class, the semigroup law can
    only tell counter widths apart.
  - `--collapse C1` groups stage C's 103,680 survivors by design_key before C1: 1,668 groups, that is 556 classes ×
    cw ∈ {5, 6, 7}.
  - The law runs on each group's representative, the member nearest the reference. A group keeps all its members if
    its representative passes, and none if it fails.
  - D and E then form their classes from the kept members exactly as before. `--collapse` alone is the earlier path.
  - The representatives take all 8 prompts in one pass per family, instead of row groups of 1, 1 and 6 prompts.
- **The width law once per setting of what it reads.**
  - It never reads the gate and io block, entry radix, exq, ppop, qsrc, r2mq, q4, r4mqn, r4cin or cw (`WIDTH_IGNORES`).
  - Among the 53,440 C0 survivors, the other 27 factors take 2,680 settings. Dropping only the io block, entry radix
    and operand roles would leave 13,360.

## Stages, before and after

| stage | before (`h100_search_collapse_k32`) | `--collapse C1` (`h100_search_collapse_c1_k32`) | `--collapse`, this file (`h100_search_collapse_d_k32`) |
|---|---|---|---|
| A, B, cell laws, C0 | 20.8 s | 17.4 s | 17.9 s |
| CW, width law | 53,440 evaluations, 27.2 s | **2,680 evaluations, 1.2 s** | 2,680 evaluations, 1.4 s |
| C, decimal digits | 4.5 s | 3.6 s | 3.7 s |
| C1, semigroup 50+49 | 103,680 machines, 3 passes per family, 60.9 s | **1,668 representatives, 1 pass, 9.4 s** (0.8 s of it grouping) | 103,680 machines, 3 passes, 55.0 s |
| collapse, D, E | 20.8 s | 18.3 s | 19.0 s |
| **wall to the end of E** | 135.8 s | **50.9 s** | 98.2 s |
| spot-checks: C1, E | —, 29.3 s | 17.3 s, 25.8 s | —, 27.3 s |
| wall | 165.0 s (2.75 min) | **94.1 s (1.6 min)** | 125.5 s |

- The requested run's host was faster than the earlier one on the stages nothing touched: A 7.8 against 8.6 s, C0 8.9
  against 11.3 s, E 13.7 against 15.8 s. The same-file baseline ran on a similar host, and its C1 took 55.0 s.
- On the laptop's CPU, the width law on the same 53,440 machines goes from 10.1 s to 0.59 s.

## Why C1 is 6× faster, not 60×

- **A C1 pass costs about the same at any lane count here.**
  - The spot-check makes one pass of 8 prompts over 3,336 non-representatives: 17.3 s.
  - The earlier C1 makes three passes per family (103,680 lanes on 1 prompt, then 34,560 on 1 and on 6): 55.0 s, or
    18.3 s a pass.
  - So 10 to 30 times more lanes cost no more per pass. At 64 lanes per word, even 103,680 lanes are 1,620 words, far
    too few to fill an H100. The cost is the kernels a pass launches.
- **A pass over the representatives costs half as much: 8.6 s against 17.3 s.**
  - Among the 556 representatives, every shift, port order and each of Ke, Kx, Km and W takes one value, the
    reference's (Km = 26, W = 28). Among the members they take 2 to 6 values.
  - Fewer options present means fewer multiplexer branches. Km = 26 also means 26 radix-2 or 13 radix-4 iterations
    instead of 32 or 16.
- So 55 s → 8.6 s comes from one pass instead of three (×3) and uniform representatives (×2). The 62× fewer machines add
  little on their own.

## Same survivors at every stage

- **Against `h100_search_collapse_k32`, timings aside** (`runs/collapse_c1_k32.*`):
  - A, B, the cell laws, C0, CW and C have identical stage records: survivors per family, free switches, survivors after
    each row group, and the width law's reasons.
  - C1 keeps the same 34,560 machines. The collapse then forms the same 556 classes, with the same members row for row
    and the same representatives. D and E give the same records and the same audit rows.
  - The same-file `--collapse` run matches too, so the earlier path is unchanged.
- **The memoized width law is the width law** (`runs/width_memo_k32.*`):
  - On the 53,440 machines of `h100_search_nowidth_k32`, which are the machines CW sees, it returns the same verdict
    string for every machine.
  - A scan of the source of width_law and _width_iteration finds 27 factors read, none in WIDTH_IGNORES, and none read
    through a computed name.
  - Redrawing every ignored factor at random changes no verdict: 42,720 of 42,720. That is 4 redraws each of one machine
    per setting (5 kinds of verdict) and of 8,000 random grammar machines with R > 4N (54 kinds).

## Does a representative predict its (class, cw)?

- **Here, on every sampled member.** With two random non-representatives per group, 3,336 machines, each passes or
  fails as its representative did: the 1,112 at cw = 7 pass, and the 2,224 at cw = 5 and 6 fail.
- E's spot-check, as before: 1,112 of 1,112 non-representatives certify exactly as their representatives.
- In this search, C1's verdict is cw = 7 alone, for every representative and every sampled member. So the check confirms
  the grouping but cannot tell it apart from grouping by cw alone. The earlier label-free runs are the same: each C1
  kept exactly the 7-bit counters.

## Rating and next

- **Promise: high, and cheap.** The label-free search takes 1.6 min instead of 2.75 (51 s instead of 2.3 min to the end
  of E), and every stage keeps the same survivors.
- The time is now spread out: E 13.7 s, C1 9.4 s, C0 8.9 s, A 7.8 s, D 4.0 s, C 3.6 s. The two spot-checks add 43 s,
  almost as much again.
- **Next:**
  1. C1 now costs per pass and per option present, not per machine. Two levers:
     - fewer kernels per pass, e.g. one squaring step captured as a CUDA graph, or `torch.compile` at a fixed lane
       count;
     - fewer options present: the bit-identical merges of the previous section (operand roles, carry forms, q4's gen
       and n1, paired recodings and tables), which would leave about 30 classes.

     D and E were not measured this way. E on 1,112 varied members takes 1.9× as long as on the 556 representatives,
     with as many passes; twice the machines and more options present would both explain that.
  2. The spot-checks now cost about as much as the search. `--collapse-check 0` skips them. A smaller sample would save
     less than its size suggests, at least for C1, since a pass costs about the same at any lane count.

Compliance, one line: neither change reads a label; D and E use labels as before.

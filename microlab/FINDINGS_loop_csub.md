# Loop bodies without a fused modular add

**Question.** FINDINGS_exact_blocks.md §3 found Blakley's binary modular multiplication as a 3-instruction loop body:
1 label, 2^22 bodies, 5.5 s. Its blocks included addm(a, b) = a + b − N·[a + b ≥ N], which fuses the add with the
conditional subtract. Is that fused block doing the heavy lifting?

Here it is replaced by csub(a) = a − N·[a ≥ N]: one conditional subtract with no add. The body must now be built from
generic blocks {add, sub, mul, csub}, and the shortest correct one is 5 instructions:

    t0=add(s,s); t1=mul(x,d); t2=add(t0,t1); t3=csub(t2); t4=csub(t3)

**Answer.**
- **Speculation still finds it, cheaply.**
  - No body of 4 or fewer instructions is correct.
  - At L=5 the 2^37.3 raw bodies are 2.7e8 non-redundant bodies and 3.4e6 distinct functions.
  - The search takes 8–10 s on 32 Modal cores; running every body with no merging at all takes 24 s.
  - The found body certifies T=64 on both H1 profiles.
- **1–2 labels keep the right function, but usually not alone.** Near-miss bodies reduce too little: s drifts into
  [N, 2N), and the answer comes out congruent but unreduced on some rows. The worst one is right on 86% of rows.
  - In ext_exact_loop's label order, label 1 leaves 2 functions: the right one and csub(2·csub(s) + d·x). Label 4
    leaves only the right one.
  - Over 2,000 random orders of 32 H1 rows: after 1 label a median of 7 functions is left, after 2 labels 3.
  - It takes a median of 5 labels before only correct functions are left (p90 12, worst order 25). L=6 needs exactly
    the same, though more is left early: a median of 19 functions after 1 label.
- **The fused block helped only by leaving no room.**
  - At its minimal length (L=3), the addm grammar needs a median of 1 label (max 5).
  - With one spare instruction (L=4), its label statistics are identical to csub at L=5: median 5, p90 12, max 25.
    addm(a, 0) is csub(a), so the same near-misses fit.
- **Cost per extra instruction is ×40–70.**
  - L=6 (2^45.6 raw bodies, 2.5e10 non-redundant) took 6.2 min on 32 cores.
  - L=7 extrapolates to about 5–7 h, and L=8 to weeks.
  - Loop-body speculation tops out at about 6–7 generic instructions, the same ceiling as straight-line speculation.

## Setup (`ext_loop_csub.py`)
- **The loop.** It scans the 24 bits of x from the top, starts from s = 0, and applies s ← body(s, x, N, d).
  - Registers: [s, x, N, d, 0, 1, 2] plus earlier results.
  - Arithmetic is exact int64.
  - Operands: add and mul take a ≤ b, sub takes ordered pairs, csub takes one operand.
  - Raw bodies: 112·144·180·220·264 = 1.7e11 (2^37.3) at L=5, and ×312 more at L=6.
- **Enumeration.** Bodies are built bottom-up.
  - Partial bodies whose intermediates agree on 32 probe states are merged. The probes have s < 2N (two of them
    s = 0), d alternating 0/1, x < N, and odd N of 12–24 bits.
  - A body is dropped if an instruction equals an existing register on all probes, or if it leaves a result unused.
    Such a body is a shorter body, searched at a smaller L.
- **Merging by function.** Complete bodies with the same outputs on the probes share one loop run (merged within each
  work chunk).
- **Scoring.**
  - Every function runs the loop on label 1; survivors also run labels 2–8.
  - Survivors of label 1 are scored on all 1,000 H1 T=1 rows, in order.
  - Those that fit all 1,000 rows are scored on 2,000 fresh rows each at 13, 16, 20 and 24 bits.
- **Labels.** H1 T=1 training rows in ext_exact_loop's order (permutation seed 0). The first four labels have
  N = 54253 (16-bit), then 5969, 6313 and 7493 (13-bit).
- **The scan** (`--scan 32`). Every function also runs on all of the first 32 rows, so any order of those rows can be
  replayed; I replay 2,000 random orders.
- **Compute.** All L ≥ 5 runs are on one Modal box with 32 CPUs (31 worker processes), via `ext_loop_csub_modal.py`.

## Results
**Search size and cost, csub grammar.** "Loop runs" are the functions after merging within chunks. Labels are in
ext_exact_loop's order.

| L | raw bodies | non-redundant | distinct functions | loop runs | time | functions left after 1 / 2 / 3 / 4 labels |
|---|---|---|---|---|---|---|
| 1 | 112 | 70 | 64 | 64 | <0.1 s, laptop | 0 |
| 2 | 16,128 | 1,714 | 689 | 689 | <0.1 s, laptop | 0 |
| 3 | 2.9e6 (2^21.5) | 67,758 | 9,529 | 9,529 | <0.1 s, laptop | 0 |
| 4 | 6.4e8 (2^29.3) | 3,751,449 | 162,369 | 634,542 | 0.5 s, 3 laptop workers | 0 |
| 5 | 1.7e11 (2^37.3) | 272,637,962 | 3,365,584 | 35.9e6 | 7.6–9.8 s, 32 cores (160–215 CPU-s) | 2 / 2 / 2 / 1 |
| 6 | 5.3e13 (2^45.6) | 24,900,923,417 | ≈8.3e7 (HyperLogLog) | 2.5e9 | 373 s, 32 cores (11,200 CPU-s) | 8 / 5 / 5 / 4 |

- **Growth per instruction.**
  - Non-redundant bodies grow ×24, ×40, ×55, ×73 and ×91 from each L to the next.
  - Distinct functions grow ×11, ×14, ×17, ×21 and ×25.
  - From L=5 to L=6, CPU time grew ×52–69 and wall time ×38–49.
- **What merging buys at L=5.**
  - 272.6M bodies become 112M (state, last instruction) pairs, then 35.9M functions within chunks, then 3.37M
    functions globally.
  - At L=6, merging globally would cut loop runs another ×30, but the enumeration costs as much as the loop does.
- **Compared with straight-line speculation** (FINDINGS_exact_blocks §1a: 2.6 s at 2^38, 83 s at 2^46.7), the loop
  search takes 3–4.5× longer, over raw spaces 1.6–2× smaller.
- **Survivor curve, raw bodies, same order.**
  - L=5: 272,637,962 → 20 → 20 → 20 → 12. From label 4 on, the 12 left are exactly the correct bodies, through label 8
    and all 1,000 rows.
  - L=6: 2.49e10 → 318 → 288 → 288 → 180. From label 4 on, the 180 left are exactly the correct bodies.
- **Every survivor that fits all 1,000 labelled rows is correct** on the fresh rows at all four sizes: 1 of 1
  function at L=5, 4 of 4 at L=6.
  - There are no register-width aliases: no survivor fits the training rows and then fails at 24 bits.
  - The certified bodies (L=5: Blakley; L=6: csub(csub(2s) − (−d·x))) loop to T=64 on depth_t and on depth_ood_n_t.
- **The survivors that are not correct:**
  - The lazy alias csub(2·csub(s) + d·x) (8 bodies at L=5, 108 at L=6). It fails label 4 and is right on 88.5% of
    fresh 24-bit rows.
  - At L=6, three chance survivors that fail label 2. One adds 240 per 0-bit of x and 1260 per 1-bit, which happens to
    equal y = 14,940 on row 1.
- **The 12 correct bodies at L=5 are one function.** They differ in:
  - how 2s is made: add(s,s) or mul(s,2);
  - the order in which 2s and x·d are made;
  - where the first csub sits: csub(csub(2s + dx)), csub(csub(2s) + dx), or csub(s + csub(s + dx)).

**Brute-force check (`--brute`: nothing merged, every non-redundant body runs the loop on its own).**
- **L=3 and L=4** were also counted by a separate scratch enumeration that uses exact vector equality. The counts are
  identical (67,758 bodies and 9,529 functions; 3,751,449 and 162,369), and no L=4 body fits label 1.
- **L=5 on 32 cores:** 272,637,962 loop runs in 24 s. They give the same 3,365,584 functions and exactly the same 20
  bodies after label 1. Merging lost nothing.
- **Probe count.** At L=4, K = 16 / 32 / 64 probes give 162,079 / 162,369 / 163,402 functions. The ±1% are piecewise
  functions that differ on small regions. Survivors are checked on real rows either way.

**Label order (the scan: 2,000 random orders of the first 32 H1 T=1 rows).**

| grammar | L | functions fitting ≥ 1 of 32 rows | left after 1 label (median / p90 / max) | after 2 | only correct left after 1 / 2 / 4 / 8 / 16 labels | labels until only correct (median / p90 / max) |
|---|---|---|---|---|---|---|
| addm (fused) | 3 | 20 | 1 / 3 / 14 | 1 / 1 / 4 | 66% / 96% / 99.9% / 100% / 100% | 1 / 2 / 5 |
| addm (fused) | 4 | 201 | 10 / 21 / 133 | 6 / 10 / 28 | 9% / 24% / 48% / 77% / 97% | 5 / 12 / 25 |
| csub | 5 | 921 | 7 / 46 / 671 | 3 / 6 / 52 | 9% / 26% / 50% / 77% / 97% | 5 / 12 / 25 |
| csub | 6 | 11,000 | 19 / 287 / 8,546 | 7 / 14 / 486 | 0% / 15% / 47% / 77% / 97% | 5 / 12 / 25 |

- **The L=6 scan** ran as a detached Modal app (ap-lqP8uP0Ti6EafWWZtBdE5F). It was launched at 18:50 PDT, retried once
  (the attempt that finished started at 19:14), finished at 20:52 and was then stopped.
  - It took 1.6 h on 32 cores (180,000 CPU-s), versus 6 min for the single-order L=6 search, because every one of the
    2.5e9 L=6 functions runs on all 32 rows.
  - At L=6, 4 functions are correct, so "left" never falls below 4.
  - The labels needed are identical to L=5 because the same worst near-miss decides them; it is right on 86% of rows.
  - What grows is the early crowd: 11,000 functions fit at least one row, against 921 at L=5.

- **The correct function is always left.** It fits every row, and its share of what is left is what improves.
- **The addm L=3 row reproduces ext_exact_loop exactly:** 4,941,216 raw bodies, and 3 correct bodies after its first
  label. That order was a typical one for addm; for csub, the same order is a lucky one.
- **The near-misses that slow it down.** These are the L=5 rates, as the share of the 1,000 rows each one fits:
  - 0.864: csub(2·csub(s) + dx). It keeps s < 2N instead of s < N.
  - 0.744: csub(2·csub(s)) + dx.
  - 0.670: csub(s + csub(s) + dx).
  - 0.586: csub(csub(2s)) + dx.
  - 0.488: csub(s) + csub(csub(s) + dx).
  - 0.26 and below: one conditional subtract where two are needed (58 bodies), and others.
- **What a near-miss is.** Each one is exact on every row where its missing reduction isn't needed. It is the loop's
  analogue of the register-width aliases in FINDINGS_exact_blocks, and the labels catch it because they are fully
  reduced residues.
- **Why it takes several labels.** A random label removes the worst near-miss with probability 0.136. So expect about
  1/0.136 ≈ 7 labels on average, with a median of 5.

## Takeaway
- **A generic small-block loop grammar works.** With no fused modular add, speculation still finds a modular
  multiplier: a 5-instruction body among 2^37 raw bodies (3.4e6 distinct functions), in 8–10 s on 32 cores.
- **1–2 labels find it but do not pin it.**
  - The first label always keeps it, and in ext_exact_loop's order only 1 other function is left with it.
  - Over random orders, a median of 7 functions is left after 1 label and 3 after 2.
  - Isolating it takes a median of 5 labels (90% of orders within 12, worst 25 of 32), at L=5 and at L=6 alike.
  - Every leftover is a partial-reduction near-miss or a chance hit, never a wrong function that fits all 1,000 rows.
- **The fused addm did two jobs.**
  - It made the body shorter: 3 instructions instead of 5, i.e. 2^22 raw bodies instead of 2^37, and 16k distinct
    functions instead of 3.4M.
  - At its minimal length it left no room for near-misses, so 1 label usually sufficed.
  - Without it, the near-misses fit inside the minimal body, and the labels needed go from about 1 to about 5. With one
    spare instruction, the fused grammar is no better.
- **The reach is about 6–7 instructions.** Each extra generic instruction costs ×40–70: L=6 takes 6 min, L=7 about
  5–7 h, and L=8 weeks on 32 cores. A loop grammar removes the length limit of the whole computation, not of the body.

## Files
- **Code:**
  - `microlab/ext_loop_csub.py`: search, `--brute`, `--scan`, `--isa addm` for the control, and certification.
  - `microlab/ext_loop_csub_modal.py`: the 32-core Modal driver.
- **Data:** `microlab/results_loop_csub.jsonl` has one record per run. `microlab/runs_loop_csub/` has the logs and the
  full JSON of each Modal run (`modal_L5*.json`, `modal_L6.json`, `modal_L6_scan32.json`).

```bash
python -m microlab.ext_loop_csub search --L 4 --workers 3
```

```bash
python -m microlab.ext_loop_csub search --isa addm --L 3 --scan 32
```

```bash
~/github/tips-for-running-modal/venv/bin/python -m modal run microlab/ext_loop_csub_modal.py --length 5 --brute
```

```bash
~/github/tips-for-running-modal/venv/bin/python -m modal run --detach microlab/ext_loop_csub_modal.py --length 6 --no-exact --chunk 512 --spawn
```

```bash
python -m microlab.ext_loop_csub certify --body "add,0,0;mul,1,3;add,7,8;csub,9,9;csub,10,10"
```

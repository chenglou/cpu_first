# One Layer Deeper, from the CPU up

> **2026-10-21: the current entry point is [OVERVIEW.md](OVERVIEW.md).** It covers where everything stands: the best
> model (every computing part learned, and its wiring chosen from wide menus), the removal ladder, rules 1–16, and the
> open questions. This report is the first day's analysis (§0–8) and the follow-ups on SpecBit and the grammar search
> (§9–14), kept as written.

**TL;DR.**
- **Labels can check the arithmetic but can't teach it.** Inside a single squaring step, the mod-N reduction turns any
  wrong digit fact into a pseudorandom residue. So gradient descent from labels gets no signal about the arithmetic.
  That is why 143 of 146 leaderboard entries sit near 0%.
- **What works is the CPU's own structure.** Use a designed datapath with discrete, restored signals and one small cell
  replicated per bit or digit. Let the labels *select*: run every candidate machine in parallel and keep the one that
  reproduces the answers. That is branch prediction applied to learning.
- **Bits make the hypothesis space enumerable, and bitslicing makes enumerating it cheap.** On a laptop CPU this
  recovers a whole binary CPU design (gate truth table + wiring, ~31 bits) from a label-free round-trip check plus
  4 labeled rows in 23 s, and the resulting machine certifies T=64 on both profiles.
- **The limits are honest ones.**
  - The algorithm still lives in the designed search space.
  - Structure that only matters off-distribution (datapath width, counter range) is invisible to the labels and must
    come from priors tied to the input spec.

## 0. The problem
Predict y = x^(2^T) mod N (repeated modular squaring: the Rivest–Shamir–Wagner time-lock puzzle, i.e. the
Blum-Blum-Shub recurrence) from decimal tokens: `N <N digits, LSB first> X <x digits> T <T digits>`, answer
right-aligned. N is a 13/16/20-bit Blum semiprime in training (only 430 distinct moduli); certification needs
T = 1,2,4,…,64 all 100% exact, on training-pool moduli and on unseen 14/18/22/24-bit moduli (up to 8-digit N, never
seen). Training has 33k rows at each of T=4/8/16 but only **1,000 rows at T=1 and 1,000 at T=2**. Leaderboard: 3 entries
certified T=64 on both profiles; the other 143 sit at T=1 with ≤12.8% (most <1%). It is all-or-nothing.

## 1. How a CPU does it
`N=atoi(rev(digits)); s=atoi(x); repeat T: s = s*s mod N; print(itoa(s))` — one 24-bit register, one multiply and one
reduction per step. Its parallelism is entirely *inside* a step (all partial products at once, carry-save trees,
carry-lookahead/carry-select, SRT quotient digits guessed from a few top bits and fixed later, cmov for the conditional
subtract). Across steps a CPU's speculation buys nothing: every step needs the previous result. That is the time-lock
assumption. Exactness over billions of cycles comes from signal restoration (every gate snaps to 0/1). Width
independence comes from bit-slicing (one cell, replicated).

## 2. Why stock ML gets ~0
- **The labels are Blum-Blum-Shub output, and that holds inside a single step too.** A wrong upstream digit fact shifts
  the pre-reduction value by (error × place value), which the mod-N reduction maps to a pseudorandom residue. Four
  panel lenses saw this in independent small probes:
  - a shared full adder learns 2/2 inside a plain adder, but 0/4 inside a Montgomery netlist;
  - one wrong product-table entry sends the rows it touches to near-chance digit accuracy;
  - straight-through training cannot repair 2 wrong entries out of 100;
  - an upstream product table has a basin of about 0.3 per entry (64 seeds × 4 learning rates, best 0.2% exact).

  The critics' caveats are fair:
  - The runs are small.
  - Montgomery is the worst case: a top-down datapath keeps 0.29 against 0.13 digit accuracy on failing rows after one
    wrong entry.
  - Graded signal exists *near* the solution: one wrong cell entry can leave 40–92% of rows exact.

  What survives is the part that matters. Every datapath tried is at chance by about 16 wrong entries. So from random
  init there is no gradient to follow, and a wrong *wiring* gives exactly chance-level output.
- So the 1,000 T=1 rows can **check** the arithmetic but cannot **teach** it. The T≥4 rows are useful only once the step
  is already right; then they are a strong verifier.
- **Precision cliff.** Analog representations of a 24-bit residue need learned weights within about 1e-7 relative error
  before any smooth gradient exists.
- **Credit-path length.** In my substrate tests, even without any reduction, 4 rounds of learned partial-product and
  accumulate tables (~72 lookups deep) do not learn from answers, even with unlimited data.

## 3. What labels *can* do
| component | label-learnable? | how |
|---|---|---|
| anything after the last reduction (output map, readout) | yes | plain CE |
| tolerant / sandboxed decisions in a congruent redundant state (quotient guesses, carry amounts) | not from labels, but they don't need to be | errors change only the representative, never the residue, so a generic label-free objective ("keep magnitudes small") suffices. Caution: the panel's "hindsight target (r−y)/N" turned out to equal trunc(r/N) exactly (2.4M/2.4M cases), i.e. an exact divider hidden in the loss, not label information |
| loop counter for unseen T=32/64 | only via shared cells | digits 3,5,7,9 never occur in training T: 432 one-hot decrement tables fit every training halting time, only 6 get T=32 right. It works when the counter reuses cells whose digit meanings are pinned elsewhere (the arithmetic's FA, as in the demos) |
| choice among complete alternative machines | yes, crisply | likelihood mixture over whole machines (right one 100%, wrong ones ≈0%) |
| exact arithmetic cells inside the reduction | **no** (needle) | design them, or calibrate them on short-path self-tests, or enumerate them |
| wiring / dataflow | **no** by gradient (zero partial credit) | design it, or enumerate it and let labels verify |

## 4. Demos (all on a laptop CPU)
- **`cadet.py`**: a decimal Montgomery machine. Its add, multiply and complement tables are learned from Peano
  self-tests in 0.5 s, with no task data. Labels choose N′ among 10 complete machines per N mod 10 class (0.3 s), and the
  output map comes from CE. Result: **T=64 certified on both profiles**, 14/14 rungs at 100%, 1.1 s.
- **`menu.py`**: the recurrence is unknown. Same ALU, a menu of 24 complete programs. Labels pick the program jointly
  with N′. All **6/6 hidden recurrences** (square, cube, square+1, square·x₀, square+x₀, cube+2) are identified and
  **certified T=64 on both profiles**, about 8 s each. T=2 rows matter here: s·x₀ ≡ s² at T=1.
- **`speculate_gate.py`**: the binary netlist (every gate is one 3→2 truth table, 65,536 candidates) runs all candidates at
  once, **bitsliced 64 per uint64**. 4 T=1 labels leave exactly the full adder, in 1.9 s, with no axioms and no gradients.
  `bitcpu_netlist.py` with that gate scores **5,376/5,376 on both profiles** (T=1..64).
- **`speculate_wiring.py` / `speculate_full.py`**: gate × 8 wiring choices, about 2³¹ machines. The label-free round-trip
  b2d(d2b(v))=v on the prompt's numbers leaves 9 of 2.36M (22 s); 4 labels then leave **exactly one: full adder + reference
  wiring**. The round-trip alone also admits a genuine base-12 converter pair, which the labels reject.
- **Panel synthesis, independent confirmations** (`panel_synthesis.md`):
  - Labels pin the binary gate from any 2 rows (120/120 random pairs).
  - In decimal, pure algebraic laws (identity, commutativity, associativity, place value, distributivity) leave exactly
    8! = 40,320 digit relabelings, and 4 labeled rows pick the right one.
  - One shared learned borrow cell, trained by a *sequence-level* likelihood with no oracle, learns canonicalization,
    the sign test and the T counter (correct for T=1..99). It succeeded on 3/3 seeds in 250 steps; a per-digit loss
    fails on 0/3.
- **`alias_width_audit.py`: the aliasing audit.** The same netlist built 24 bits wide (R=2²²) is 1,000/1,000 on training
  T=1 rows and 5,376/5,376 on the training-pool certification profile, but **3,308/5,376 on the OOD profile**. Every
  training modulus fits in 24 bits. A label-only search would keep this alias.

## 5. Your ideas, evaluated
- **Branch prediction / speculation.** Inside a step: yes (quotient candidates, carry-select). Across steps: provably
  futile (time-lock; chaos defeats parallel-in-time methods). **In learning: this is the mechanism that works.** Execute
  every hypothesis in parallel and commit the one the exact verifier (the labels) accepts. Gradient descent sees a needle;
  speculative search does not care.
- **Temporarily weak connections.** Soft mixing *inside* the circuit is destroyed by the avalanche. Mean-field
  execution of even the exact table is chaotic. Soft speculative quotient lanes in a long division leak wrong candidates
  into every later comparison, roughly ×50 per round. There, even at the true tables, the loss is worse than uniform, and
  it *improves* as the tables move away from the truth. Weak connections work *between whole machines* (an output-level mixture that
  hardens in tens of steps) and in the backward pass (straight-through). Separately, over-provisioned "wide wires" make
  shallow discrete compositions trainable: 10-way latent wires stall at ~75%, 24–48-way wires reach 100% and stay
  width-exact.
- **One bit per activation.** Binary cells shrink the hypothesis space to something enumerable (2¹⁶ for a full adder).
  That is what makes label-only speculation possible. Decimal makes I/O trivial, but its tables are not enumerable
  (~10²⁶⁰ full adders), so it needs self-tests.
- **32 bits per word (bitslicing).** It is useless for gradient training, but **it is how speculation over hypotheses
  becomes cheap**: 64 candidate machines per bitwise op. 31k machines evaluate in 0.1 s on CPU, so an H100 should
  verify ~2³⁰ in minutes. That is roughly 30 bits of datapath design per GPU-hour, against roughly 0 for gradients here.
  The learned gate is literally the LOP3 immediates 0x96/0xE8.

## 6. The honest catch
Every certifying system imports the algorithm: as fixed microcode, as a self-test "answer key", or as the grammar of a
search space. Under a strict reading of "no hard-coded algorithm", the panel's critics rated every design fatal
(0.04–0.3). The defensible research position is to shrink what is designed toward *laws and grammars*, and let labels
select. That means algebraic laws such as round-trip and identity, plus a space of candidate circuits.

The kind of "learning" that works here is closer to **superoptimization / exact circuit synthesis** (Massalin's
enumerate-and-test superoptimizer, STOKE, exhaustive minimal-circuit search, AlphaDev) than to SGD. Candidates are
enumerated and the labels serve as test vectors. A wrong machine matches a random residue with probability ~1/N, so a
handful of rows verifies astronomically many hypotheses. The bottleneck is enumeration throughput, and bitslicing
attacks exactly that. On CPU the fixed-gate wiring search runs ~3×10⁵ candidate machines/s (31,104 in 0.1 s);
per-lane gate tables cost ~10× more. A GPU should be 2–3 orders of magnitude faster (untested). Factoring the search
with label-free invariants turns products of spaces into sums: round-trip took gate × wiring from 2³¹ to 2.4M
evaluations. The frontier question is therefore: **how general can the grammar be (e.g. "any systolic array of ≤k shared
cells with local wiring") before the search no longer fits?** Past that point, the search is discovering algorithms
rather than confirming a design.

A second limit is structural, not computational. **Speculation learns exactly the structure the training labels
exercise.** Choices that only bite off-distribution survive every training check:
- the datapath width (the 24-bit alias above);
- the loop-counter range (training T ≤ 16 needs only 5 bits);
- the digit meanings of T's tens place.

These must come from priors tied to the input spec, or from label-free invariance tests. Width can be sized from the
declared max prompt length. Counter range follows from two T digits. A composition check f(N,x,32) = f(N, f(N,x,16), 16)
on training prompts needs no labels, but it synthesizes inputs, so it is gray under the rules. The demos above all use
spec-sized widths, which is why they certify.

## 7. How this meets handwired (read only after the above was written)
handwired compiles ordinary JS into exact, differentiable nets: affine + ReGLU, a fp32-rounding boolean atom
(x + 2²³ + 1 − (2²³ + 1)) that snaps to 0/1, and index reads that act as lookup tables. Seen from this problem, it is
the **designed-microcode half** of the CPU-first picture, already built:
- **Signal restoration.** Its boolean atom is the float-native version of the snapping that makes T=64 exact.
  It uses the fp32 mantissa itself as the quantizer, which is a real instance of "use the bits of the float".
- **Predication.** "Both branches compute, a 0/1 selects" is the cmov/carry-select move. Here, applied across
  hypotheses instead of branches, it is the learning mechanism.
- **Same wall, other side.** Its learned interpreter only succeeds by scoring all 3,600 complete programs (a complete
  loss table); `recovery-lr` reports that gradient training alone does not close the gap while search does. That
  matches this problem's finding: labels select among complete machines; gradients see a needle.
- **What this adds to handwired's open item** ("scaling to longer programs without a complete loss table"):
  (1) **bitslicing** makes complete-program evaluation ~64× cheaper per op, and it is trivially GPU-parallel;
  (2) **label-free invariants factor the search**, here round-trip b2d(d2b(v)) = v, so the full product table is never
  built (2³¹ → 2.4M evaluations); (3) **wide wires** help where a gradient path does exist.
  A natural experiment is to compile the Montgomery step with handwired, mark a few instructions/wires as
  "choices", and search them with bitsliced speculation using H1 rows as test vectors.

## 8. Next
Ordered by information per hour:
1. **SpecBit through the real evaluator, then a GPU-scale grammar search.** The panel's recommended entry is the binary
   netlist with its gate learned from labels as a mixture over all 65,536 hypotheses inside the forward pass. Run it
   through the unmodified benchmark runner.
   - Then grow the grammar beyond the 8 hand-picked switches. Try "any carry-save cell wiring", entry/exit variants and
     radix choices, and measure the largest space an H100 can verify in an hour with bitslicing plus invariant
     factoring.
   - This is the experiment that would turn "~30 bits per GPU-hour" from an extrapolation into a number, and test whether
     the search can find a modular multiplier that was not hand-placed in the grammar.
2. **handwired as the substrate.** Compile the Montgomery step with handwired and expose a few instructions or wires as
   choice points. Search them with bitsliced speculation, using H1 rows as test vectors.
   - This directly attacks handwired's open item, learning without a complete loss table.
   - It also tests whether the boolean atom's fp32 snapping is enough restoration for T=64.
3. **Non-enumerable cells.** With the quotient exposed, can speculative local search over table edits learn a decimal
   ALU where gradients plateau? This is the panel's experiment 2. If yes, speculation extends past tiny binary cells.
4. **An admissible quotient source** for residue-safe designs, e.g. a value-size penalty instead of a hidden divider.

## 9. Follow-up: SpecBit through the evaluator, and speculation at GPU scale
Next item 1, done. Numbers in `FINDINGS_substrate.md`; code in `submission.py`, `spec_grammar.py`, `modal_h100.py`.
- **SpecBit certifies through the real evaluator.** The submission is the netlist above, with its one cell learned inside
  the forward pass as a mixture over all 65,536 truth tables, bitsliced on the batch's T≤2 rows.
  - The unmodified `benchmark.runner` reports **max certified T = 64 on both profiles** in three settings: the
    laptop's MPS and CPU (fp32, amp off) and the official `h100_hard_h1.json` on a Modal H100 (bf16 autocast).
  - Training takes 20 steps and 8–17 s; evaluation takes 16–27 s.
  - Argmax θ is the full adder after 1–2 Adam steps, on every shuffle seed tried.
  - Two interface details were needed: the manifest device must be `mps:0`, and Adam must be fused so its step
    counter stays on the device.
- **Throughput, measured.** The torch port runs 64 candidate machines per int64 op. One candidate on one T=1 row
  (eager, warm) runs at:
  - per-lane cell and per-lane wiring: 4e6/s on MPS and 2.1–2.8e7/s on an H100, i.e. **≈2^36 machines per
    H100-hour unfactored**;
  - fixed cell, wiring only: 9e6/s on MPS and 5.3e7/s on an H100.

  The old extrapolation ("2^30 in minutes on an H100") holds: 2^30 takes about 45 s. Compiling just the hot block with
  `torch.compile` is 7.5–9.4× faster on MPS and 6–13× faster on an H100 (bit-exact), so a fused end-to-end search
  would add about 3 bits.
- **A grown grammar, searched.** The grammar has 21 factors, 2^63.9 machines including the gate. It covers carry-save
  port sources and shifts, quotient source, hand-offs, entry radix, Ke/Km/Kx, exit rule, converter bases, datapath
  width and counter width.
  - It factors into three blocks. The first two have label-free round trips: b2d(d2b(v)) = v and
    exit(entry(x)) = x on the prompts' own numbers. Labels are used only for the core.
  - The 2^54.4 version needs 2^27.9 evaluations: 11 s on an H100, 68 s on MPS.
  - The 2^63.9 version takes **20 min on one H100**: 2^35.4 evaluations, leaving exactly the reference port wiring up to
    the full adder's port symmetry.
  - Costs add rather than multiply. At this grammar's survivor counts, an H100-hour would cover **≈2^87 joint
    machines** (extrapolated).
  - The real limits are not compute. One is survivor multiplicity: stage A keeps 7,782 degenerate gate × converter
    pairs, and without the second round trip stage C would cost 10 H100-hours instead of 2 s. The other is needing a
    label-free invariant for every block.
- **Aliasing audit, at scale.** 144 machines fit all 101,999 training rows, and **16 certify**. The training data cannot
  tell them apart.
  - Three switches matter only off-distribution:
    - counter width: 5 bits fails at T=32, 6 bits at T=64;
    - Montgomery R = 2^K: K=22 fails the unseen 22- and 24-bit moduli and K=24 the 24-bit ones, since the bound
      R > 4N needs K ≥ bits + 2;
    - datapath width: W=24 fails the 24-bit moduli even at K=26.
  - Two are benign equivalences: entry radix 2 vs 4, and a carry-save hand-off scaled by 2. With port wiring, add the
    36 port permutations of the two symmetric cells.
  - Without the entry/exit round trip, T=1 labels also admit 2Ke = Km + Kx aliases; the T=2 rows reject them.
  - So the search can confirm or reject anything the grammar contains, and it found equivalent variants nobody wrote
    down as the reference. It still cannot choose among designs that differ only off-distribution, and it found
    nothing outside the grammar.

## 10. Follow-up: the Montgomery core gets label-free laws, radix 4 joins the grammar
Numbers in `FINDINGS_substrate.md` (last section); code in `spec_grammar.py`; runs in `runs/h100_search_*`.
- **The core became a two-operand MonPro(a, b) with four multiplier families.**
  - The families are the old radix-2 core, radix 2 on a precomputed b + N, radix 4 (13 iterations for R = 2^26, a
    2-bit quotient from the low bits of the carry-save sum and N mod 4), and radix 4 on precomputed 3b and 3N.
  - Each family has its own tree shapes and tables.
  - All four compute the same integer, and each one's reference machine certifies T=64.
  - The small grammar is 2^60.7 joint machines. With port wiring and per-port shifts it is 2^70.0.
- **A stage C0 screens the core alone with label-free laws on the prompts' own (N, x).**
  - It runs once per distinct entry/exit left by stage B, so the cost adds instead of multiplying.
  - Two laws together are exactly as strong as the 64 T=1 labels on the core: identity exit(MonPro(x̃, 1̃)) = x plus
    commutativity, or the squaring laws 1² = 1 plus x² = (N − x)². After them, the labels only remove the base-12/16
    io aliases.
  - On the small grammar, C0 + C take 18 s against 123 s on an H100, with 24× fewer machine evaluations in those
    stages.
  - The ports grammar (2^70.0 joint, core 2^35.5) ran in 60 min on one H100: C0 on 1.31e11 core machines took 55 min,
    then 361,728 label checks. Labels alone would need 3.1e12 checks, about 22 H100-hours (extrapolated).
  - There the per-port shifts add radix-4 aliases that are exact on training and fail on 24-bit moduli: a ×4-scaled
    datapath, equivalent to R/4 with Km + 2.
  - Laws on synthesized and edge-of-range 24-bit inputs remove all but 464 of them. Those overflow a 26- or 28-bit
    word on rare inputs, sometimes on 1 row of 5,376. Ruling that out needs a width bound per cell, not more samples
    (§11).
- **Each law has a blind spot, so suites matter.**
  - The identity law passes degenerate datapaths that return their first operand. Of the 74,940 radix-2 cores it keeps,
    only 96 are correct. In the reference context, 9,504 of 9,516 are exact projections f(a, b) = a.
  - Commutativity on (x̃, 1̃) removes them. Commutativity on (x̃, x̃²) and associativity can't, because there x̃² = x̃.
  - Identity and commutativity also reject squaring-only multipliers. Those are correct for this task; the labels keep
    them.
- **At the declared input ranges, the laws pin the off-distribution switches the first runs could not.**
  - MonPro laws on 16 synthesized 24-bit Blum moduli remove Km = 22 and W = 24. Associativity, or x⁴ = (N − x)⁴, also
    removes Km = 24, because only chained squarings feed the multiplier a 25-bit operand.
  - The semigroup law y(x, 50+49) = y(y(x, 50), 49) removes the 5- and 6-bit counters.
  - Checking that the answer prints decimal digits replaces the labels at stage C.
  - **A search that reads no label returns 4,480 machines, and all 4,480 certify T=64 on both profiles.** That is
    exactly the certifying subset of the 40,320 machines the labels-only search keeps. The labels only verify.
- **The search finds radix 4 and equivalents nobody placed as the reference.**
  - Radix 4 makes up 2,176 of the 2,240 certifying two-operand machines.
  - All 12 radix-4 tree shapes are equivalent.
  - Dropping the radix-4 carry works when the last cell adds a shifted row.
  - In the precomputed-multiples version, any recoding of the 2-bit quotient works when paired with the matching
    {0, N, 2N, 3N} table.
  - A quotient that ignores N mod 4 is exact only because every benchmark modulus is a Blum integer (N ≡ 1 mod 4). On
    N ≡ 3 (mod 4) it fails 15/15.
- **What still limits it:**
  - **Survivor multiplicity.** The certifying sets are thousands of genuine equivalents, and D/E cost scales with them.
  - **The priors.** 24-bit moduli, 2-digit T, Blum moduli and decimal output come from the task spec, not from the
    data.
- **Compliance note:** the laws synthesize inputs (1, N − x, Blum moduli, T values). Whether that is allowed is for the
  strict review before a real submission.

## 11. Follow-up: per-cell laws inside the Montgomery core
Numbers in `FINDINGS_substrate.md` (last section); code in `spec_grammar.py` (`--cell-law`, `--width-law`,
`--cell-audit`).
- **Per-cell laws factor the core.** Each carry-save cell must conserve value (u + 2v = its rows at their natural
  weights, one scale per cell). Per iteration, the quotient zeroes the low bits and the hand-off divides exactly.
  - Checked one sub-block at a time on synthesized words, they cost 7,113 machines where C0 screened 1.31e11 jointly.
  - On the 2^70.0 ports grammar, C0 drops from 55 min to 3.4 s on an H100, and the whole run from 60 min to 4 min.
- **A width law that is a bound, not samples.** Over the declared range (N < 2^24, operands below the Montgomery
  closure bound, all-ones digits), it bounds every port, carry row and register in every iteration.
- **Either law removes all 464 machines that passed every earlier label-free check.**
- **Certification was too lenient.** The width law also removes 720 machines that certify T=64 on both profiles.
  - Each one is wrong on 24-bit inputs the profiles don't contain: N near 2^24, or chains whose operand reaches
    1.5·2^24 (found by exhaustive search over x).
  - On 8,982 such rows, the 4,480 earlier survivors split exactly as the width law says.
- **A search that reads no label until D returns 3,296 machines in 2 min on an H100, and all of them certify.** It uses
  the cell laws, MonPro laws at 24 bits, the width law, the decimal format and the semigroup law.
- **The catch:** the value laws need a per-cell spec (which rows each cell adds). That is more design knowledge than
  the MonPro laws need. One scale per cell also rejects the ×4-scaled aliases, even where a wider W would make them
  correct.

## 12. Follow-up: the core grows where per-cell factoring makes it cheap
Numbers in `FINDINGS_substrate.md` (last section); code in `spec_grammar.py`; runs in `runs/h100_search_*_k32.*`.
- **The grammar grew from 2^70.0 to 2^76.3 joint machines.** K and W now go up to 32, with every wire 32 bits, and
  radix 4's last two cells get per-port shifts.
- **The value law has one scale per row class.** Within an iteration, the digit rows may sit at 2^e times the state and
  quotient rows, with the same e in every cell. The ×4-scaled radix-4 aliases now pass the value laws, and only the
  width law judges them.
- **Factoring holds.** 20,744 sub-block machines (0.1 s) leave 3.2M core machines, which C0 screens in 7 s on an H100.
  Unfactored, C0 would take about 49 H100-hours (extrapolated).
- **The label-free search returns 34,560 machines in 7.7 min, and every one certifies T=64 on both profiles.**
  - 10,848 of them are ×4-scaled. With 32-bit wires they are exact at W ≥ 28: their extra bits are in the wires, not
    the registers.
  - **The width law is still sound, but no longer exact.** Without it the search keeps 53,440 machines, which then
    face 16,644 declared-range witness rows. No machine the law keeps gets a row wrong. 608 that it rejects get none
    wrong either, and they are correct: a bit-level argument and a direct scan of a million products each show it. The
    law bounds each signal on its own, so it cannot see that operands below the closure bound never set bit 24
    together with bit 23 or 22.
- **The binding cost is now survivor multiplicity.** D and E take 6.2 of the 7.7 minutes, because every correct design
  comes back once per (R, W) and per equivalent scaling.
- **Booth radix 4 was not added.** Signed digits need offset rows, and with offsets MonPro(a, b) and MonPro(b, a) differ
  by multiples of N, so the bit-exact commutativity law would reject correct designs.

## 13. Follow-up: one machine per equivalence class
Numbers in `FINDINGS_substrate.md` (last section); code in `spec_grammar.py` (`--collapse`); run in
`runs/h100_search_collapse_k32.*`.
- **The 34,560 survivors fall into 556 classes that the laws already make equivalent.** Shift settings that pass the
  value laws differ only in scale. The full adder's port order doesn't matter. Every (R, W) the width law accepts gives
  the same answer.
- **D and E run on one representative per class: 20 s instead of 374 s.** The whole search takes 2.75 min on an H100
  instead of 7.7 min, on a host that was slower in the other stages.
- **Every representative certifies T=64 on both profiles, and so does every member.**
  - The members are exactly the machines the earlier run certified one by one, with identical outcomes.
  - 1,112 random non-representatives, two per class, certify exactly as their representatives do.
- **The (R, W) merge needs the width law.** In the search without it, merging shifts and port order still predicts every
  member exactly, on certification and on declared-range witness rows. Merging (R, W) as well puts certifying and
  failing machines in every class.

## 14. Follow-up: collapse before C1, the width law once per setting
Numbers in `FINDINGS_substrate.md` (last section); code in `spec_grammar.py` (`--collapse C1`, `width_verdicts`); runs
in `runs/h100_search_collapse_c1_k32.*`, against `runs/h100_search_collapse_k32.*`.
- **The semigroup law runs on one machine per (class, cw): 1,668 instead of 103,680, in 9.4 s instead of 61 s.** Each
  group keeps or drops all its members with its representative. 3,336 sampled other members each pass or fail exactly
  as their representative did.
- **The width law runs once per setting of the 27 factors it reads: 2,680 evaluations instead of 53,440, 1.2 s instead
  of 27 s.** Its verdict is the same on every machine.
- **Same answer, faster.** Every stage keeps the same survivors as before, down to the same 556 classes, the same 34,560
  members, and the same D and E outcomes. The search takes 1.6 min on an H100 instead of 2.75 min, and reaches the end
  of E in 51 s instead of 2.3 min.
- **C1 costs per pass, not per machine.** On the H100, a pass costs about the same at 3,336 lanes as at 34,560 to
  103,680. The gain comes from one pass instead of three and from representatives that all sit at the reference shifts
  and widths.

## Files
`README.md` indexes every script. `FINDINGS_substrate.md` has the raw numbers. `panel/` holds the brief, the prior-art
survey, all five designs and five critiques. `panel_synthesis.md` is the panel's synthesis.

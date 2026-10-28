# Prior art for 1L (CPU-first): what transfers

**Bottom line.** The successes that are exact, generalize to longer inputs, and need little data have three things in common. They use a small local cell copied across positions with tied weights. They snap the state to a discrete code at every step. They supervise or verify one step at a time. The failures learned the wiring, learned the whole map as one function, or relied on analog precision. I found no published neural model that learns BBS/RSW-style modular squaring.

## A. Exact arithmetic that generalizes to longer inputs

- **Neural GPU** ([Neural GPUs Learn Algorithms, 2016](https://arxiv.org/abs/1511.08228); [Extensions and Limitations, 2016](https://arxiv.org/abs/1611.00736)).
  - **Learned:** a conv-GRU cellular automaton on a bit grid learned binary addition and multiplication. It trained on numbers up to 20 bits and made 0 errors at 2,000 bits.
  - **Cost:** about 200k examples, plus a curriculum, parameter-sharing relaxation, dropout and gradient noise. Of 729 grid-search runs, "usually only a few" generalized.
  - **Failure:** decimal needed bigger models. A model that was near-perfect on random 100-digit products failed on 000…002×000…002.
  - **Lesson:** a generic cellular automaton can find multiplication, but it needs about 100× our data plus luck. Random tests don't prove exactness.
- **DNGPU** ([Improving the Neural GPU Architecture, 2018](https://arxiv.org/abs/1702.08727)).
  - **What made it work:**
    - hard tanh/sigmoid units;
    - a saturation cost that penalizes activations beyond |0.9|;
    - diagonal gates, which take their gate from the left or right neighbour and so act as carry wires.
  - **Result:** binary multiplication in about 800 steps instead of 30k. All 5 seeds reached about 100% at 10× the training length; 2 of 5 exceeded 99% at 100×. Decimal worked for 2 of 5 seeds.
  - **Failure:** with soft units it fit the training set but did not generalize.
  - **Lesson:** force saturation during training (signal restoration), and give cells explicit wires to their neighbours.
- **Neural Execution Engines** ([NEE, 2020](https://arxiv.org/abs/2006.08084)).
  - **Learned:** transformer subroutines on 8–12-bit binary numbers. Each number's embedding is the sum of per-bit vectors. Shift-invariant learned masks pick where to work, and each intermediate state was supervised.
  - **Result:** trained at size 8, 100% at size 100. Addition stays at 100% with half the numbers held out.
  - **Lesson:** separate "where" (fixed dataflow) from "what" (a tiny arithmetic cell). Bit-wise codes let the model handle values it never saw.
- **NALU/NAU/NMU** ([NALU, 2018](https://arxiv.org/abs/1808.00508); [Neural Arithmetic Units, 2020](https://arxiv.org/abs/2001.05016)).
  - **Learned:** scalar units that extrapolate +, − and × to larger magnitudes.
  - **Failure:** NALU converges unreliably. NAU/NMU fix that, but they cannot divide or take a modulus, and they are only as exact as floating point.
  - **Lesson:** never hold the residue as a scalar. Analog values are useful only as a rough predictor (such as a quotient estimate) that exact digital logic then corrects.
- **Transformer arithmetic.**
  - **Abacus embeddings** ([2024](https://arxiv.org/abs/2405.17399)): 20M samples and 1 GPU-day gave 99% on 100-digit addition after training on 20 digits. Multiplication was near-perfect only within the training range (up to 15 digits, about 192 GPU-hours, looped model).
  - **Position coupling** ([2024](https://arxiv.org/abs/2405.20671)): addition went from 30 to 200 digits. Multiplication, even with a scratchpad, reached only about 2–3× the training length ([2025](https://arxiv.org/abs/2410.15787)).
  - **Looped transformers** ([2025](https://arxiv.org/abs/2409.15647)) generalize only when given the true step count.
  - **Priming** ([2023](https://arxiv.org/abs/2306.15400)): adding 50 long examples to 5,000 short ones took multiplication from 5×3 digits to 35×3 digits.
  - **4×4-digit multiplication** ([Why Can't Transformers Learn Multiplication?, 2025](https://arxiv.org/abs/2510.00184)), trained on 80.8k samples: standard training reached under 1%, and the middle digits were never learned. An auxiliary loss that probes the running sum reached 99%.
  - **Lessons:**
    - Index digits by significance from each operand's least significant digit. This matters because H1 gives N least-significant-digit first and x most-significant-digit first.
    - Don't count on learned attention routing for multiplication.
    - The 292 seven-digit rows may serve as a primer for 8-digit N.
    - Auxiliary losses on intermediate values work, but computing their targets inside the loss amounts to a solver in the loss. That is a rules gray area.
- **Exact by construction.**
  - **NTK ensembles** ([Learning to Add, Multiply… Exactly, 2025](https://arxiv.org/abs/2502.16763)): training data is built from orthogonal templates that isolate each bit rule (40 examples for 10-bit addition), and outputs are rounded at every step. This gives exact addition and multiplication, but only at a fixed width.
  - **Discrete neural algorithmic reasoning** ([Discrete NAR, 2024](https://arxiv.org/abs/2402.11628)): discretized states, hard attention and supervision of state transitions give perfect out-of-distribution scores with correctness proofs.
  - **Lesson:** rounding at every step turns "nearly right" into "right for every T".

## B. Discrete circuits and programs learned by gradient

- **DiffLogic family** ([2022](https://arxiv.org/abs/2210.08277); [conv, 2024](https://arxiv.org/abs/2411.04732); [Mind the Gap, 2025](https://arxiv.org/abs/2506.07500); [DiffLogic CA, 2025](https://arxiv.org/abs/2506.04912); [recurrent, 2025](https://arxiv.org/abs/2508.06097)).
  - **Learned:** wiring is fixed and random; each neuron learns a softmax over 16 two-input gates. CIFAR-10 reached 62%, and 86.3% with gate-tree convolutions, residual initialization and 61M gates.
  - **Fix for the gap:** a hard forward pass with Gumbel/straight-through gradients cuts the train-to-inference discretization gap by 98%.
  - **DiffLogic CA:** learned Game of Life perfectly from all 512 single-step 3×3 transitions and runs correctly on larger grids. Pattern growth trained with 20-step backprop through time works, but needed heavy tuning amid numerical instability.
  - **Failure:** recurrent logic-gate networks remain weak.
  - **Lesson:** learn a local rule from single transitions that cover its whole input space, then iterate the hardened circuit. In all of this work the gates were learned; the wiring never was.
- **TerpreT** ([2016](https://arxiv.org/abs/1608.04428)).
  - **Failure:** gradient descent over circuit structure succeeded in 0 of 2,400 restarts on a full adder and on a 2-bit adder (3 of 2,400 after 5× longer training). A parity chain of length 128 succeeded 0% of the time. SMT and Sketch solvers took seconds to minutes.
  - **What helped:** providing more gates and wires than needed (NAND synthesis went from 36% to 60% success).
  - **Lesson:** wiring is the hard part for gradients. If you try it, over-provision, add gradient noise and restart many times.
- **∂4** ([Differentiable Forth, 2017](https://arxiv.org/abs/1605.06640)).
  - **Learned:** a program sketch fixes the control flow and dataflow, and only the slots are learned. 256 examples gave perfect sorting (trained on length 2–3, tested on 64) and perfect addition (trained on 8, tested on 64).
  - **Comparison:** seq2seq needed 16k examples and still failed on lengths it hadn't seen.
  - **Lesson:** fixed dataflow plus learned slots is the regime that fits 1k rows.
- **NRAM / NPI** ([NRAM, 2016](https://arxiv.org/abs/1511.06392); [NPI, 2016](https://arxiv.org/abs/1511.06279); [Recursion, 2017](https://arxiv.org/abs/1704.06611)).
  - **NRAM:** hard tasks needed a curriculum, gradient noise and an entropy bonus. Even with the best settings, about 11% of seeds converged (1% without noise).
  - **Recursive NPI:** trained on 100–200 execution traces, it is provably perfect. The proof checks the network on a finite verification set covering all base cases and reduction rules.
  - **Lesson:** make the step's local configurations finite, enumerate all of them after hardening, and plan for many seeds.

## C. Modular arithmetic, crypto recurrences, hardness

- **Grokking** ([Power, 2022](https://arxiv.org/abs/2201.02177); [Nanda, 2023](https://arxiv.org/abs/2301.05217); [Discrete-Log Clock, 2026](https://arxiv.org/abs/2606.17399)).
  - **Learned:** addition and multiplication mod p (p ≈ 100) as Fourier "clocks", with multiplication done through the discrete log. This needed about 30% of the full p×p table.
  - **Lesson:** learning the table cannot scale to N ≈ 2²⁴ from 1k rows. The model has to compose over digits.
- **Learned recurrences.**
  - **Linear congruential generators** ([2025](https://arxiv.org/abs/2502.10390)): learned up to m = 2³² (22 hours on 4 A100s) by exploiting the short periods of low bits when m is a power of two, and residue digits when m is composite. A prime modulus, m = 2039, was not learned.
  - **Collatz** ([2025](https://arxiv.org/abs/2511.10811)): bases 16, 24 and 32 reached 99.7%; odd bases did poorly. Over 90% of errors were wrong loop lengths (control), not wrong arithmetic.
  - **Modular exponentiation** ([2025](https://arxiv.org/abs/2506.23679)): small moduli only.
  - **Speck cipher** ([Gohr, 2019](https://eprint.iacr.org/2019/037)): neural distinguishers stop working after about 8–9 of 22 rounds.
  - **xorshift128** ([NCC Group, 2021](https://www.nccgroup.com/research-blog/cracking-random-number-generators-using-machine-learning-part-1-xorshift128/)): this generator is linear over GF(2) and was learned to 100%.
  - **Lessons:** networks exploit shortcuts from the modulus's factors or the number base. A Blum N (two large primes, coprime to 10) offers none. Signal decays with each round, so only the T=1 and T=2 rows can get learning started.
- **Hardness.**
  - Any ordered binary decision diagram for the middle bit of a product needs at least 2^(n/8) nodes ([Bryant, 1991](https://doi.org/10.1109/12.73590)).
  - Gradient variance vanishes for modular multiplication with large moduli ([Takhanov et al., 2023](https://arxiv.org/abs/2310.12660)).
  - Targets that depend on many input tokens at once ("high globality") need inductive scratchpads to be learnable ([Abbe et al., 2024](https://arxiv.org/abs/2406.06467)).
  - Transformers learn log-depth shortcuts for automata, but those shortcuts break out of distribution ([Liu et al., 2023](https://arxiv.org/abs/2210.10749)).
  - **Lessons:** learned as one function, s² mod N is effectively unlearnable from 1k rows. With per-step accuracy p, only about a p^T fraction of T≥4 examples gives useful signal. Prefer sequential, local (systolic) cells over learned parallel-prefix shortcuts.

## D. Recursion and parallelizing recurrences

- **TRM** ([Less is More, 2025](https://arxiv.org/abs/2510.04871)).
  - **Setup:** a 2-layer network with 5–7M parameters, run for 3 recursions of 6+1 updates each, with gradients through the last recursion only. It uses 16 deep-supervision steps and an exponential moving average (EMA) of the weights.
  - **Result:** 87.4% on Sudoku-Extreme from 1k puzzles, falling to 79.9% without EMA. But it used 1,000 augmentations per puzzle and up to 36 hours on one L40S. A 4-layer version overfit.
  - **Lesson:** EMA and tiny networks transfer. Augmentation is forbidden here, so tying weights across digit positions has to play its role.
- **DEER / ELK / ParaRNN** ([DEER, 2024](https://arxiv.org/abs/2309.12252); [ELK, 2024](https://arxiv.org/abs/2407.19115); [Predictability, 2025](https://arxiv.org/abs/2508.16817); [ParaRNN, 2025](https://arxiv.org/abs/2510.21450)).
  - **Method:** Newton's method over the whole trajectory, using parallel scans. It needs O((log T)²) time when the system is contracting (top Lyapunov exponent negative) and is hopelessly ill-conditioned when it is chaotic.
  - **Why it fails here:** squaring maps s+δ to s² + 2sδ + δ² mod N, so any error explodes. For Blum N, squaring also permutes the quadratic residues, so paths never merge. Speculation that relies on merging paths, as in [Data-Parallel FSMs, 2014](https://www.microsoft.com/en-us/research/publication/data-parallel-finite-state-machines/), gains nothing either. This is what the time-lock assumption predicts ([RSW, 1996](https://people.csail.mit.edu/rivest/pubs/RSW96.pdf)).
  - **Contrast:** carries are a recurrence with a tiny, associative state, so they parallelize well ([Kogge–Stone, 1973](https://www.gwern.net/doc/cs/algorithm/1973-kogge.pdf)).
  - **Lesson:** don't parallelize across T; 64 sequential steps fit easily in the eval budget. Speculate only inside a step.

## E. Hardware for exactly this recurrence

- **Montgomery systolic arrays** ([Montgomery, 1985](https://doi.org/10.1090/S0025-5718-1985-0777282-X); [Walter, 1999](https://doi.org/10.1049/el:19991230); Orup, ARITH-12, 1995; [Tenca & Koç, 1999](https://doi.org/10.1007/3-540-48059-5_10)).
  - **Design:** one processing element per digit, carry-save, least significant digit first. The quotient digit depends only on the lowest digit; Orup pre-scales N so the quotient digit simply is the low digit.
  - **No comparisons:** Walter showed that with R > 4N, values stay below 2N for the whole exponentiation.
  - **Global work:** only the setup (R² mod N) and the final exit step.
  - **Lesson:** this is the dataflow that best fits bit-slicing. It is width-generic, needs no comparisons, and matches H1's least-significant-first N.
- **VDF FPGA squarers** ([Öztürk, 2019](https://eprint.iacr.org/2019/826); [Jane Street, 2020](https://blog.janestreet.com/really-low-latency-multipliers-and-cryptographic-puzzles/)).
  - **Design:** a 1024-bit squaring takes 25 ns. Numbers are stored as 16-bit words plus one redundant carry bit, so carries only reach neighbouring words. High bits are folded back using lookup tables of precomputed residues.
  - **Rival:** a "predictive Montgomery" design used 2.3× less area at 46 ns.
  - **Lesson:** folding with per-N tables is proven in hardware. The tables come from a recurrence that reduces by N with a one-digit quotient.
- **RNS** ([Cox-Rower, 2000](https://www.springerprofessional.de/cox-rower-architecture-for-fast-parallel-montgomery-multiplicati/13540668); [Bajard & Imbert, 2004](https://www.lirmm.fr/~imbert/pdfs/rsa_rns_ieeetc_2004.pdf); [HPS, 2019](https://eprint.iacr.org/2018/117)).
  - **Design:** residue channels need no carries between them. Converting between bases needs the CRT quotient, which is estimated from truncated fractions (Cox-Rower) or floating-point fractions (HPS). The estimate is exact or off by a multiple that RNS-Montgomery tolerates.
  - **Lesson:** this is a proven "analog predictor plus digital corrector": low precision suffices when there is margin. The cost moves to converting into and out of decimal.
- **Cautionary tale** ([Pentium FDIV, 2024](https://www.righto.com/2024/12/this-die-photo-of-pentium-shows.html)).
  - **What happened:** five of the 1,066 entries in the Pentium's SRT division table were wrong. By Intel's estimate, errors hit about 1 in 9 billion random divisions.
  - **Lesson:** random tests can't certify a learned table; enumerating every entry can.

## Five lessons that matter most

1. **Learn the step, not the trajectory.**
   - Train the one-step map on the T=1 rows (plus T=2), then iterate a hardened state, as DiffLogic CA and recursive NPI do.
   - Use the T≥4 rows as a verifier and for selecting models. Use them as gradient signal only once per-step accuracy is high.
   - Skip parallel-in-time methods.
2. **Take the dataflow from hardware and learn only small cell contents.**
   - Use a Montgomery systolic array (least-significant-first, carry-save, with Orup's and Walter's tricks) or Öztürk-style folding.
   - Gradients fill slots well (∂4, DiffLogic) and find wiring badly (TerpreT, NRAM).
   - A fully fixed dataflow invites "hard-coded algorithm" objections. The defensible middle ground is to learn a selection among more candidate wirings than needed, then harden it.
3. **Size cells so that 1k rows cover them many times over.**
   - 1k T=1 rows yield roughly 10⁴–10⁵ digit-cell instances, or 10⁵–10⁶ bit-cell instances.
   - Binary or carry-save cells (32 or fewer input configurations), or digit-pair products (100) plus small adders, are covered many times. Cells that sum whole columns are not.
   - Tie weights by digit significance (Abacus or position coupling). This extrapolates to 8-digit N and multiplies the effective data, taking the place of TRM's augmentation, which is forbidden here.
4. **Enforce discreteness during training.**
   - Use a hard forward pass with straight-through/Gumbel gradients, saturation penalties, per-step rounding and weight EMA.
   - Use analog math only as a predictor with margin (Barrett- or RNS-style), with exact digital selection after it.
5. **Verify exhaustively and train populations.**
   - Per-seed success when learning discrete structure is often only about 10–40%. Train many small models in parallel and select by exactness on the T=4–16 training rows.
   - After hardening, enumerate every cell configuration (compare FDIV, and the Neural GPU's 000…002 failure).
   - Make the T counter an exact decimal cell with tied weights that reuses the N/x digit embeddings. Three reasons:
     - "3" and "6" never appear in T's tens place in training.
     - The Collatz models' errors were control errors.
     - Looped transformers needed oracle step counts.
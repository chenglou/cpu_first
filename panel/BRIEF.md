# Brief: "One Layer Deeper" from a CPU-first perspective

## 0. What we're doing
The user wants a *whole different perspective* on the One Layer Deeper (1L) problem: first ask how a CPU
solves it, then design a learned system informed by that — instead of cramming a stock ML architecture
(transformer, etc.) onto it. Hints from the user (not mandates): CPUs get parallelism from speculation /
branch prediction ("think bigger", "temporarily weak connections"); maybe treat each float activation as
one bit, or pack 32 bits per fp32/int32 word (bitslicing). "Don't over-index; think for yourself."

DO NOT read ~/github/handwired, ~/github/handwired-lab or ~/github/handwired-worktrees (the user wants
independent exploration first). Do not search the web for this competition's solutions or write-ups.

## 1. The task (exact facts, verified from the released repo + locally generated data)
Repo clone: `REPO=/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/repo/one-layer-deeper`
Generated H1 data (JSONL): `DATA=.../scratchpad/data/h1` (same scratchpad root). Python venv with torch (MPS) at `.../scratchpad/.venv`.

- Predict y = x^(2^T) mod N (repeated modular squaring = RSW time-lock puzzle / Blum-Blum-Shub recurrence).
- N = p*q, p ≡ q ≡ 3 (mod 4) (Blum integers), balanced factors. x is a unit mod N.
- Tokens: PAD0 BOS1 N2 X3 T4 ANS5 EOS6, digits 0-9 = 7..16. Prompt = `N <digits of N REVERSED (LSB first)> X <digits of x, MSB first> T <digits of T, MSB first>`.
  Example: `N 1 6 6 4 X 3 0 9 3 T 4` means N=4661, x=3093, T=4 → y=3960.
- Output: separate target tensor = decimal digits of y (MSB first), **right-aligned onto the last len(y) prompt positions**:
  target digit k-from-LSB is read from logits at prompt position (input_len-1-k). Leading positions unscored.
  Model gets input_ids + padding mask (bidirectional attention allowed), returns logits [B, L, 17].
- Exact match per example (all target digits right). No partial credit.
- H1 training split (101,999 rows): T ∈ {4,8,16}: 33,333 each; **T=1: only 1,000 rows; T=2: only 1,000 rows.**
  Moduli 13/16/20-bit; **only 430 distinct training moduli** (11 / 42 / 377). Split is by modulus identity.
- N digit counts: train 4–6 digits (7 digits: 292 rows only). x, y up to 6–7 digits.
- Certification ("Hard" ranking): T = 1,2,4,8,16,32,64 must ALL be 100% exact, consecutively.
  - "Max T": 768 fresh prompts on training-pool moduli (256 each at 13/16/20 bits), same (N,x) reused across rungs.
  - "OOD N Max T": 768 prompts on **unseen moduli at 14, 18, 22, 24 bits** (192 each); N has 4–8 digits
    (8-digit N never appears in training; 7-digit is rare). 505 distinct OOD moduli.
  - T=32 and T=64 never appear in training (training T ≤ 16). Note T's tens digit is only ever "1" in training.
- N mod 10 ∈ {1,3,7,9} always (odd, never divisible by 5) → N is coprime to 10.
- Squaring trajectories have long cycles (thousands) → no cycle shortcut.
- Budget: 3,600 s training on one H100 (model build/compile included), eval budget 1,800 s. bf16 autocast by default
  (model may control precision internally). ≤ 500M parameters. Batch 512 default (participant can change).
- Rules (spirit): learned model from random init; no hard-coded algorithm in the forward pass; end-to-end gradient
  path; no data augmentation / task-specific solver; evaluator owns the loop/data order; participant owns model,
  optimizer (anything incl. curvature methods), LR schedule, loss (custom loss gets logits, labels, valid mask and the
  model's auxiliary output), batch size, up to 8 backward passes per step and up to 8 reuses of a batch.
  Recurrence, adaptive computation, depth curricula, memory tokens, parameter-free work all explicitly allowed.
  (The competition ended Aug 31 2026; the hidden H1 set is now public. We care about the research question, but
  rule-spirit compliance is still a useful discipline. Flag gray areas instead of ignoring them.)
- Leaderboard (146 ranked): exactly 3 entries certified T=64 on both profiles; ALL other 143 are stuck at T=1
  with ≤12.8% (most < 1%). Bimodal: you either have the right structure or you get ~nothing.

## 2. How a CPU solves it
```
N = atoi(rev(Ndigits)); s = atoi(xdigits); T = atoi(Tdigits)
repeat T times: s = s*s mod N        # one 24x24→48-bit MUL + one reduction (DIV, Barrett or Montgomery)
emit itoa(s)
```
- State = one 24-bit register + constant N + loop counter. Per step: multiply + reduce. Nothing else.
- The CPU's speculation buys nothing *across* steps here: every iteration depends on the previous one (true data
  dependence; this is exactly what makes it a time-lock puzzle). The CPU's real parallelism is *inside* a step:
  all n² partial products at once, carry-save (Wallace) trees, carry-lookahead / carry-select (compute both carry
  outcomes, select later), SRT division with redundant quotient digits (guess a digit from a few top bits, fix later),
  cmov/predication for the conditional subtract (compute both r and r−N, pick). Speculation = compute candidates in
  parallel, select by an exact cheap test.
- Exactness over billions of cycles comes from **signal restoration**: every gate snaps its output back to 0/1.
- Width independence comes from **bit-slicing**: an n-bit ALU is n copies of one tiny cell with carry wires.

## 3. My analysis so far (push beyond this; disagree where wrong)
A. **Why stock nets get ~0 (the leaderboard).**
   1. *T≥4 data is Blum-Blum-Shub output.* Unless a model already computes the step nearly exactly, the T-step label
      is pseudorandom w.r.t. what it computes (avalanche/chaos), so the gradient is noise. If p = per-step exact
      accuracy, roughly a fraction p^(T−1) of T-step examples carry useful signal. So **all bootstrapping signal comes
      from 1,000 T=1 (+1,000 T=2) examples**; the 100k T≥4 rows become valuable only once p ≳ 0.9 — then they are an
      enormous *verifier* covering rare configurations (certification needs ~49k step-instances per profile correct).
   2. *Precision cliff for analog representations.* If the residue is a float, y = s² mod N wraps when s moves by
      ~N/(2s) ≈ 0.5, so learned "place-value" weights must already be within ~1e-7 relative of exact before any smooth
      gradient exists. Gradient descent from random init cannot find exact analog arithmetic (bf16 has 8 mantissa bits).
   3. *Non-triangular dependencies.* In s² (no mod) output digit j depends only on input digits ≤ j (LSB-first,
      carries flow up) — that's why reversed-digit addition/multiplication is learnable. In s² mod N the quotient
      floor(s²/N) is set by the TOP digits and feeds back into every low digit: dense, bidirectional coupling.
      Montgomery reduction (LSB-first; quotient digit m = (r₀·N′) mod 10 depends only on the lowest digit and N mod 10)
      restores triangularity inside the loop; only a per-example setup (R² mod N) and nothing per step is top-down.
      Note H1 hands N over LSB-first.
   4. *Exactness to T=64 needs a digital state.* Per-step error must be < ~1/(64·768). Any analog drift dies. The
      recurrent state must be snapped to a discrete code each step; then T-extrapolation is free *iff* the step is exact
      on all reachable states.
   5. *Width extrapolation (to 8-digit N) needs position-generic cells* (bit-slice weight sharing, relative positions).
   6. *T extrapolation needs a loop counter that generalizes* (e.g. decimal decrement/compare by a shared digit ALU),
      since "32"/"64" never appear in training and tens digits other than 1 are unseen in T.
B. **Data budget forces a tiny hypothesis class.** 1,000 T=1 examples ≈ 17k bits of labels. Whatever learns the step
   must have description length ≲ a few kbits and benign optimization — i.e., regular structure with a few small,
   heavily-shared learned cells. That's what a CPU is: a tiny design replicated.
C. **Candidate step algorithms (CPU/hardware lore), by locality:**
   - Decimal or binary **Montgomery** (LSB-first, local, quotient digit from lowest digit; carry-save state; Walter's
     trick: keep values < 2N with R > 4N → no per-step comparison; exit MonPro(z,1) needs no final subtract).
     Cost: per-example R² mod N (a long division) for domain entry.
   - **Barrett** with speculation: q̂ from top digits × per-N constant μ, candidates q̂, q̂+1, q̂+2, select by sign (final
     borrow of an LSB-first subtraction — triangular!). Cost: per-example μ = floor(10^{2k}/N).
   - **Interleaved long division** with speculative quotient digits (all 10–19 candidates in parallel, select by sign).
   - **Folding with powers**: s² ≡ Σ_k c_k·P_k (mod N) with column sums c_k and per-N table P_k = 10^k mod N
     (setup recurrence P_{k+1} = 10·P_k mod N, each a 1-digit-quotient speculative reduction). Matvec-shaped.
   - **RNS** (residue number system, carry-free, per-channel lookup tables) with Bajard/Kawamura reduction: with
     M ≫ v², the approximate-CRT fractional part is near 0, so α = round(Σ ξ_i/m_i) is robust even in bf16; the step
     becomes lookup → low-precision sum/round → per-N K×K matvec mod m_j. Tensor-core shaped. Hard parts: per-N
     constants, final exact reduction + decimal output.
   - **Analog predictor + digital corrector** ("branch prediction"): floats estimate the quotient (smooth, learnable
     approximately), exact LSB-first digit arithmetic evaluates a few speculative candidates, sign selects.
D. **Bits vs digits.** One bit per activation = maximal restoration margin, but needs decimal↔binary conversion
   (double-dabble is a nice *local* hardware algorithm for binary→BCD). Decimal digits (10-way one-hot) make I/O
   trivial (just reorder digits). Packing many bits into one float = efficient but no error correction. Bitslicing
   (32 lanes per int32) is an inference-speed trick, not a learning trick; training needs relaxed (float) bits.
E. **Speculation across steps is futile** (chaotic map; Parareal/Jacobi/DEER-style parallel-in-time converge only after
   ~T sweeps). Within a step it's the main tool to trade width for depth.
F. **The open research question:** how much of the datapath ("wiring"/dataflow) can be *learned* rather than designed,
   while remaining learnable from ~1k T=1 examples and exact? Spectrum:
   (0) fixed dataflow + learned cell tables (≈ hand-designed CPU with unknown gates — surely learnable, but the
   algorithm lives in the wiring); (1) generic bit-slice ALU + learned micro-program (soft instruction/operand selection
   that hardens — "temporarily weak connections"); (2) generic neural CA / looped block with broadcast, dataflow emerges
   (leaderboard suggests this fails in 1 h). Prior results to remember: Neural GPU learned binary multiplication as a
   2D CA but needed millions of examples and luck; TerpreT found gradient descent much worse than discrete search for
   program synthesis; difflogic learns gate types with fixed random wiring.

## 4. Machine
Apple M5 Max (18 CPU cores, 32-core GPU, MPS), 36 GB RAM. Modal (H100) available if truly needed.

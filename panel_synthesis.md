# One Layer Deeper, CPU-first: synthesis of the five-design panel

Paths: `SCRATCH=/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad`, `DATA=$SCRATCH/data/h1`.

## Bottom line

- **Exactness is solved; learning is the open problem.**
  - Four designed datapaths are exact in integer simulation on all 10,752 certification prompts. A fifth (SRT) is exact with an oracle.
  - Every design certifies only because the algorithm is designed, either in the forward pass or in the loss.
  - All five critiques returned "fatal". Four failed on the rules' spirit; learned-wiring failed because it does not learn.
- **The panel's one robust law:** gradient descent on labels gets almost no information about anything upstream of the modular reduction. Labels only teach two kinds of thing:
  - **transparent** components, which sit after the last step that fixes the residue;
  - **enumerable** choices, meaning complete alternatives that the forward pass evaluates side by side.
- **New in this synthesis** (each checked in minutes on CPU; scripts listed at the end):
  1. **Labels alone identify the binary design's single shared gate by exhaustive speculation.**
     - Of the 65,536 possible truth tables, one labeled T=1 row leaves 1–6 survivors (14 of 16 sampled rows leave exactly 1).
     - Any two rows leave only the full adder: 120 of 120 random pairs, with T=2 rows working too. The survivor is exact on all 1,000 T=1 rows.
     - "Labels can check but cannot teach" is true of gradients, not of labels: when the hypothesis space is enumerable, checking *is* teaching.
  2. **In decimal, law-only axioms cannot pin the digit code.**
     - All 8! = 40,320 digit relabelings that fix 0 and 1 satisfy identity, commutativity, associativity, inverse, place value, A·1=A and distributivity.
     - Yet each scores 0–1.5% at T=1 when fed the true digits.
     - Four labeled rows leave exactly one relabeling.
  3. **Pilot on the fold critic's exact-core states.** One shared learned borrow cell, trained without an oracle by a *sequence-level* marginal likelihood, learns three jobs:
     - canonicalization;
     - the sign test that selects the representative;
     - the T counter with halting.
     
     Results:
     - The table is exact, held-out accuracy is 100%, and the counter is correct for every T=1..99, including 32 and 64. This took 250 steps and worked for 3 of 3 seeds.
     - The per-digit mixture loss the critic used fails here too (0 of 3 runs), so its negative replication was an artifact of that loss.
     - Without value-preserving digit spreading, the counter fails (2 of 2 seeds) because the exit never shows the cell the symbols 7–9.
- **Recommendation: SpecBit.** This is the verified bit-sliced Montgomery netlist, with its one gate learned from labels by speculation inside the forward pass. The wiring is designed, the semantics are learned from labels, and there are no designer losses or oracles. My estimate is ≈0.95 that it certifies.
- **Fallback: RFS-X.** A decimal design with a gradient-trained shared cell; my estimate is ≈0.8.

## (a) Comparison

| Design (lens) | Core idea | Designed vs learned | Critic verdict, P(certify T=64, both profiles) | Key risk |
|---|---|---|---|---|
| Self-calibrated BitCPU (binary-cpu) | Bit-sliced carry-save Montgomery netlist with R=2^26, a parity quotient bit, Walter's bound, a 13-step radix-4 speculative entry, a halving exit and double-dabble, all built from **one** shared 3→2 gate. The gate is calibrated without labels on adder axioms (including place value, A+A=A≪1); 16 candidates are chosen by axiom loss. | About 99% designed. The 16-bit gate comes from an axiom loss whose only zero is the full adder (1 of 65,536), so this is hard-coding by another route. A 16×10 readout is the only part labels teach. | **Fatal, 0.04.** As specified ≈0 (0 of 128 candidates calibrate). With fixes ≈0.85–0.9 technically. | The calibration operands come from the gate under test, so it collapses (6,098 wrong tables are self-consistent). With prompt-BCD operands, only ~3% of candidates succeed, so P≥256 is needed. The logit-saturation shield does nothing under Adam. Rule 7. |
| CADET-M (decimal-machine) | Radix-10 Montgomery microcode: R=10^9, binary carry-save carries, a 9-step speculative entry, MonPro(s,1) exit, and a counter that adds 99 through the shared full adder (FA). It calls 4 generic lookup tables: FA (200 entries), MUL (100), COMP (10), N′ (4×10). | Microcode designed. About 1,560 bits of tables come from Peano/successor self-tests, which act as an answer key (18 of 18 seeds exact in 0.5 s). Labels supply about 54 bits: N′ via a likelihood mixture over 10 complete machines, plus the output map. | **Fatal, 0.10** (≈0.95 if the rules are ignored). | Rules 6, 7 and 14: the self-tests' unique fixed point is the true table. Label updates can corrupt exact tables (2→11 wrong entries). |
| Sandboxed carry-save folding, SCF (gpu-native) | Balanced signed-digit carry-save state. A designed anti-diagonal square, a fold through a per-N table P_m≡10^m (mod N), and telescoping carries: 7 kernels per step for any digit count. Every in-loop decision preserves the residue. One final speculative select over S−qN. | Residue engine designed. Learned: magnitude control (via a label-free penalty), the sign FSM, selector, canonicalizer and writer, and a count decoder c=a·tens+b·ones. | **Fatal (rules), 0.30.** About 0.75 with fixes. | The setup quotient reads N's leading digits, which breaks on OOD moduli: an MLP got 480 of 768 OOD T=64 rows wrong. The digit-magnitude penalty cannot see value size. In the critic's replication the output stage did not bootstrap without an oracle (c*, q*). NaN at init. |
| LRDM-QX (learned-wiring) | A designed dataflow with short credit paths: schoolbook square, then a 16-round long division with 10 speculative lanes, then a verify path y=P−qN that is both output and next state. Three shared ROM cells (PROD, ADD, SUB; 503 entries) are learned from labels. | Wiring designed. All arithmetic facts (about 1.7 kbit) are meant to be learned from labels. | **Fatal, 0.02.** | It does not learn. With a learned quotient q, the signal is at chance beyond ~15–30 wrong fields. Mean-field long division is worse than uniform even at the optimum (NLL 6.69 per digit). Learning SUB alone with a learned q makes no progress. Three cells with an oracle q plateau at 1.3%. |
| Congruent SRT (training-dynamics) | Radix-10 SRT on signed columns with digit set q∈[−19,19], so the state stays ≡ s² (mod N) whatever the decisions. One shared log-domain MLP makes all 20 decisions per squaring, trained on "hindsight" targets. Learned exit canonicalizer, select gate and decimal down-counter. | ALU, schedule and parse designed. Learned: the quotient MLP (about 7k parameters), the canonicalizer κ, the gate and the counter tables. | **Fatal, 0.04.** | The hindsight target equals trunc(cur/N) in 2.4M of 2.4M cases, so the loss contains an exact divider (rule 14). The exit candidates {r−N, r, r+N} miss r<−N, so every rung fails. The counter tables are unidentifiable: P(T=32 correct given T=64 correct) = 1/4. The loop runs under no_grad (rule 8). |

## (b) Cross-cutting insights

### Where the designs agree (likely true)

1. **The datapath is not the bottleneck.**
   - Integer simulations pass all certification prompts:
     - the binary netlist at gate level (5,839 FA levels at T=64);
     - decimal Montgomery (74.8M FA lookups);
     - the fold core with an exact P table (all 1,536 T=64 trajectories);
     - QX;
     - SRT with an oracle.
   - All of them rely on two things:
     - **restoration**: every operation snaps its output back to a discrete value;
     - **position-shared cells**: one cell is reused at every digit position.
   - Coverage holds when cells are generic: the rarest FA entry is visited 6,837+ times and the rarest QX entry appears in 88+ rows.
   - Coverage fails when cells are wide: a Montgomery carry-save column ROM leaves 220–374 entries that certification reaches but training never shows.
2. **The transparency law, negative half.** Five independent probes all show labels-by-gradient cannot reach upstream cells:
   - **Binary:** the same full adder is learned 2/2 inside a ripple adder but 0/4 inside an 8-bit Montgomery netlist.
   - **Decimal:**
     - Soft execution with the exact MUL table gives near-uniform outputs.
     - Adam started from the exact table drives it from 100% to 40% correct.
     - Straight-through training cannot repair 2 wrong entries out of 100.
   - **Fold:** the loss stays at chance unless every product-table entry is within 0.3 of the truth. Beyond rms error 1, the gradient's cosine with the solution is 0.00±0.07.
   - **QX:** a wrong wiring gives 9.4–11.3% digit agreement, against 11.3% for a random residue.
   - **SRT:** one ±1 table error drops affected rows to 0.47–0.33 digit accuracy, against 0.10 at chance.
   - **Where signal exists at all:** only within roughly 10–15 wrong entries of the solution.
3. **What labels do teach:**
   - **Components after the reduction**, such as a readout.
   - **Complete alternatives evaluated together.** For N′, one of 10 candidates gives 100% and the others give at most 0.5%.
   - **Repairs very close to the solution.**
4. **Speculation belongs inside a step, never across T.** Inside a step, all the designs use it: carry-select entry, quotient lanes, a final candidate window.
   - **Width replaces prediction.** An exact select over enumerated candidates beats a predictor keyed on N's leading digits.
   - The leading-digit shift makes this fatal OOD: training moduli start with 1 or 2 only 2 times out of 430, versus 370 times out of 505 for OOD moduli. 471 of 505 OOD moduli have a top-3 prefix never seen in training.
5. **Residue-safe redundancy** (fold, SRT).
   - If every in-loop move preserves value mod N, decisions only change magnitude: ±1 errors on 30% of decisions still give exact T=64.
   - All exactness then sits in one exit select, and the brief's p^(T−1) decay disappears.
   - Limits: systematic bias is only tolerated within about −15%/+20%, and ±3 errors at a 0.1% rate overflow.
6. **Hard forward pass.**
   - Never soft-mix integer states:
     - soft mixtures of congruent states stop being congruent after a fold;
     - mean-field long division is worse than uniform at its optimum;
     - deep multilinear circuits wash out.
   - Straight-through gradient norms grow about 1e5× per 50 levels, so the evaluator's global clip at 1 silently kills deep backprop.
7. **Populations plus exact certificates** turn a 3–56% per-candidate success rate into reliability, provided the certificate's inputs do not come from the hypothesis being tested.
8. **The counter must reuse digit semantics anchored elsewhere.**
   - T tokens never contain 0, 3, 5, 7 or 9, since training T ∈ {1, 2, 4, 8, 16}.
   - Independent counter tables are unidentifiable: 432 tables fit training, only 6 get T=32 right and 24 get T=64 right.
   - A linear count decoder extrapolates on only 1/6 of the parameter region that fits training.
   - Decrementing through a shared arithmetic cell extrapolates exactly. This holds for decimal (FA), QX (SUB) and the pilot.
9. **Evaluator facts:**
   - The shuffle uses seed 74, not 45.
   - At batch size 512, a batch holds about 5 T=1 rows (none in 1% of batches).
   - bf16 is exact only up to 256, so integer cores must run in fp32 with autocast off.
   - Evaluation at T=64 takes seconds for every design.

### Where they disagree (the real open questions), with status after this synthesis

1. **Can gradient training learn composed transparent stages without an oracle?** The fold design said yes; its critic's replication stayed at chance. **Resolved by the pilot:** the critic used a per-digit mixture loss, which fails 0 of 3; a sequence-level likelihood succeeds 3 of 3.
2. **Montgomery or top-down?**
   - binary-cpu and decimal-machine chose Montgomery.
   - Others found top-down keeps more partial credit: with one wrong entry, 0.68 of rows pass versus 0.42, and failing rows keep 0.293 digit accuracy versus 0.126.
   - **Resolution: it depends on the role.** Montgomery is the best designed reference; a residue-safe fold or SRT core is the best substrate for learned decisions.
3. **Where should arithmetic knowledge enter?** Options tried: forward-pass structure, loss axioms or self-tests, and labels (QX, which failed). **Partly resolved:**
   - In binary, labels alone suffice by speculation.
   - In decimal, law-only axioms leave a relabeling gauge of size 8!, which labels can resolve by enumeration.
   - It is untested whether law-only training from random init lands inside that gauge family.
4. **Gradient vs. search for cells that cannot be enumerated.** Untested; this is Experiment 2.
5. **Can a quotient predictor be trained without a solver in the loss?** SRT's hindsight target is exact division; the fold design's penalty cannot see value size. A value-size penalty is untested.
6. **How many bits of structure can labels select?** learned-wiring estimated 8–10 bits per H100-hour using population training. Exact-check speculation is much cheaper (16 bits in 15 s on CPU), but see the aliasing caveat in Experiment 1.

### Where this synthesis disagrees with the panel

- **binary-cpu's thesis** "labels can check but not teach heavily reused cells" holds only for gradients (finding 1).
- **The decimal critique's fix** of replacing the successor self-tests with law-only losses does not pin the decimal tables (finding 2): labels must still choose among 8! relabelings.
- **The fold critic's claim** that "the loss-side oracle is required" is wrong (finding 3).

### What the panel showed to be wrong in the brief

1. **§3F(0) "fixed dataflow + learned cell tables — surely learnable."** False for gradient descent on labels. True by enumeration only when the cell space is tiny (the binary gate).
2. **§3A1 "all bootstrapping signal comes from the T=1 rows."** For upstream cells, T=1 rows check but do not teach. The Blum–Blum–Shub avalanche already happens within one step: one wrong decimal table entry leaves failing rows at 0.11–0.13 digit accuracy. Conversely, with an exact designed core all 102k rows are clean signal from step 0.
3. **§3A3/§3C "Montgomery's locality restores triangularity, hence learnability."** Locality inside the dataflow is not locality of credit:
   - Montgomery outputs the high half, so every output digit depends on every cell use.
   - The quotient digit m is an exact inverse with no tolerance.
   - The T=1 label sits behind the setup plus two MonPros.

   Montgomery is the right reference datapath, not the right substrate for learning. Also, R² mod N is unnecessary: a 9- or 13-step speculative entry replaces it.
4. **§3B "description length ≲ a few kbits is the constraint."** The constraint is transparency of the credit path, or enumerability.
   - The same 310 decimal table entries learn from self-tests in 0.6 s and never learn from labels.
   - A 16-bit gate beats gradient descent (0/4) but is pinned by 2 labeled rows through enumeration.
5. **"T's tens digit is only ever 1"** understates the problem: the digits 0, 3, 5, 7 and 9 never appear in any T token.
6. **Missing from the brief:** the OOD leading-digit shift (agreement 4).
7. **§3C "analog predictor plus digital corrector."**
   - This only works with residue-safe redundancy or an exact select, and only if the predictor's target is not a solver in the loss.
   - A learned analog unit is good for about 5–7 bits, which sets the radix: fold first, then estimate.
8. **Batching:** a 512-row batch carries about 5 T=1 rows, and the shuffle seed is 74.

## (c) Recommended architecture: SpecBit (speculate the cell, design the wire)

SpecBit is a hybrid. It combines:
- binary-cpu's netlist;
- decimal-machine's idea of letting labels select among complete machines;
- gpu-native's use of width to enumerate hypotheses;

with the gate obtained by enumeration (finding 1) instead of axioms.

**Datapath.** The wiring is designed. Every gate is the single learned 3→2 cell g = (sum, carry), with:
- AND = carry(a,b,0)
- OR = carry(a,b,1)
- XOR = sum(a,b,0)
- NOT = sum(a,1,0)
- MUX = OR(AND(c,a), AND(NOT c, b))

This is the critic's `gate_sim.py`, which is exact on 10,752 of 10,752 prompts. The blocks are:

1. **Parse.** Find the markers; map each digit token to 4-bit BCD (value = token−7). N arrives LSB-first; x and T are reversed.
2. **Decimal to binary.** Carry-save Horner v←8v+2v+d (3 levels per digit) plus a Kogge–Stone adder. N and x are 28 bits wide; T is 7 bits.
3. **Entry.** Compute 3N = N+2N. Then s = x·2^26 mod N in 13 radix-4 steps: run 4s−kN for k=1..3 in parallel and select by borrow-outs.
4. **Predicated loop.**
   - active = OR-tree over the counter bits.
   - Each squaring is MonPro(s,s) in 26 carry-save iterations:
     - (u,v) = g(Ss, Sc, pp_i)
     - q = u₀
     - (s2, c2) = g(u, v≪1, q∧N)
     - Ss = s2≫1, Sc = c2
   - Then a Kogge–Stone adder, s ← MUX(active, s′, s), and the counter is decremented.
   - Cost: 84–92 levels per squaring. Training runs up to T=16; eval loops until every counter reaches zero.
5. **Exit.** MonPro(s,1) with 26 halvings; Walter's bound guarantees y<N, so no compare is needed.
6. **Output.** Double-dabble produces 8 BCD nibbles; a designed readout writes nibble k to position L−1−k.

**Depth:** about 575 levels at T=1, 1,930 at T=16 and 5,839 at T=64.

**Learning.**
- **Parameters:** θ ∈ ℝ^65536, one logit per (sum, carry) truth table, random init N(0, 0.01²).
- **Forward pass in training:**
  - Take the batch's rows with T≤2. Every 512-row batch has at least 3 such rows, and 2 suffice.
  - Run the netlist under every hypothesis h with π_h = softmax(θ)_h ≥ 1e−9·max π, using hypotheses as the batch dimension. That is all 65,536 at first.
  - aux returns each hypothesis's output digits.
- **Loss:** `L = −mean_rows logsumexp_h[log π_h − 1e4·1(digits_h ≠ y)]`.
- **Optimizer:** Adam at lr≈5 on θ collapses π onto the full adder in ≤3 steps. Pruned hypotheses then leave a single netlist.
- **Eval:** use the argmax table and the hard netlist. Logits are a mixture over the top hypotheses, so a gradient path to θ exists (rule 8).

**Designed vs learned.**

| Component | SpecBit | RFS-X (fallback) |
|---|---|---|
| Wiring/dataflow | designed (whole netlist) | designed (fold engine, candidates) |
| Arithmetic semantics | **learned from labels by speculation** | designed integer operations |
| Comparison, canonicalization, loop control | built from the learned gate | **learned shared cell (CANON), gradient-trained** |
| Digit semantics at input/output | designed BCD | designed token−7 |
| Label-learned content | 16 bits, from 2 rows | ≈164 bits, ~250 steps |
| Designer losses, axioms, oracles | none | none |

**Gray areas.**
- Rule 7: the algorithm is designed, as in all five panel designs.
- Enumeration may read as search. It is gradient descent on a marginal likelihood over a finite hypothesis class evaluated inside the forward pass, with no nested calls.
- I/O digit semantics are designed.

**Risk.** ≈0.95 to certify. What remains is the GPU port and kernel-launch overhead; eval is about 17k sequential kernels per T=64 batch, i.e. seconds.

**Fallback: RFS-X (residue-safe fold-and-select).**
- **Designed:**
  - Parse.
  - Balanced signed-digit state: W = k+5 digits, fp32, autocast off.
  - Setup: P_m ≡ 10^m (mod N), each step choosing among the 10 candidates 10P_m − qN by an exact sign test.
  - Squaring: anti-diagonal einsum, fold by bmm with P, 2 telescoping carry rounds, fold, 2 rounds. Verified exact on all 1,536 T=64 trajectories.
  - Exit candidates y_{c,q} = S_c − qN for q ∈ [−16,16] in training (±64 at eval), after 3 carry rounds.
  - Training-time value-preserving spreading (d_j ±10, d_{j+1} ∓1, probability 0.5), so candidates cover the symbols −9..9.
- **Learned:** one cell CANON, (symbol ∈ [−9,9], borrow in) → (digit, borrow out), stored as 19×2×12 logits. It does three jobs:
  - canonicalizes the selected candidate;
  - its final borrow is the sign test: p(q|c) ∝ P(y_q ≥ 0)·P(y_{q+1} < 0);
  - it runs the T counter (decrement through CANON; a top borrow means halt), giving p(c|T).
- **Loss:** `−log Σ_{c,q} p(c|T) p(q|c) p(y|c,q)`, with p(y|c,q) computed by a 2-state forward algorithm on the zero-extended label. aux carries the candidate symbols, the two log-priors and the CANON tables.
- **Evidence:** the pilot.
- **Version 2:** after training, the setup's sign test can also use CANON's final borrow (weight tying).
- **Risk:** ≈0.8. Untested parts: the core implemented in-model, 8-digit OOD widths, and the evaluator interface.

**Floor.** A fully designed engine (gate fixed to the full adder, or the CADET-M integer machine). It certifies by construction and learns nothing; use it as the test oracle.

## (d) Three decisive experiments, ordered by information value

### 1. SpecBit under the evaluator, then joint gate × wiring speculation with an aliasing audit

**Hypothesis.**
- (a) SpecBit pins the gate from the first batch and certifies T=64 on both profiles through the unmodified evaluator.
- (b) Joint speculation over the gate and 8 wiring switches is identifiable from ≤4 labeled rows, except for switches that only act outside the training distribution. Those leave aliases that pass every training row but fail certification.
- (c) Bit-sliced speculation runs at ≥1e5 hypothesis-rows/s on the M5 Max.

**Spec.**
- **Stage 0 (≈6 min).**
  - Port `$SCRATCH/panel/critique-binary-cpu/gate_sim.py` into a torch `submission.py` implementing (c): uint8 bit tensors, aux, the loss above, Adam lr 5, max_steps 20, batch 512.
  - Local manifest: a copy of `$SCRATCH/repo/one-layer-deeper/benchmark/manifests/h100_hard_h1.json` with `data_root=$DATA`, device `mps`, `amp=false`, `num_workers=0`, `pin_memory=false`, `total_training_time_seconds=600`.
  - Run `python -m benchmark.runner --manifest … --submission-file …`.
  - Metrics: `RESULT_JSON` `max_certified_time_steps` and `ood_n_max_certified_time_steps`; the step at which argmax θ becomes the full adder.
- **Stage 1 (≈15 min).** Add 8 switches per hypothesis, implemented as MUX masks:
  - S1: quotient bit u₀ or v₀
  - S2: Montgomery shift (≫1 or none)
  - S3: carry alignment (v≪1 or v)
  - S4: exit on or off
  - S5: entry on or off
  - S6: double-dabble threshold ≥5 or ≥4
  - S7: datapath width 28 or 24 bits
  - S8: counter width 7 or 5 bits
  
  Together with the gate this is 2^24 hypotheses. Evaluate them on the first `time_steps==1` row of `DATA/train.jsonl` with torch on MPS: either hypotheses in the batch dimension (about 1–3 s per 65,536-hypothesis pass, 256 passes), or bit-sliced, 64 hypotheses per int64 word, with the per-lane gate as 8-minterm logic. Then run the survivors on 63 more rows, then on all 1,000 T=1 and 1,000 T=2 rows.
- **Stage 2 (≈4 min).**
  - Run every survivor on `depth_t_{1..64}` and `depth_ood_n_t_{1..64}`.
  - Optional, label-free and gray because it synthesizes inputs: check f(N,x,32) = f(N, f(N,x,16), 16) on training prompts.

**Metrics:** survivors after r = 1, 2, 4, 64 and 2,000 rows; which switches are left free; certification per survivor; hypothesis-rows/s.

**Success:**
- Stage 0 certifies 64 on both profiles.
- After ≤4 rows exactly 4 survivors remain: the full adder with S1–S6 at their verified settings, and S7, S8 free. Only the survivor with S7=28 and S8=7 certifies both profiles.
- The composition check rejects S8=5.
- Throughput is ≥1e5 hypothesis-rows/s.

**What each outcome changes:**
- **As predicted:**
  - Speculation learns exactly the structure that training exercises.
  - Structure that governs extrapolation (width, T range) needs a designed prior or a label-free invariance test.
  - The research program becomes "speculate structure plus design invariances", and throughput sets its ceiling (log₂ of the number of hypotheses checkable per hour).
- **Unexpected aliases among S1–S6 or non-FA gates:** label consistency is not enough even in-distribution; add Occam tie-breaks and measure how often a random survivor certifies.
- **Stage 0 fails:** treat it as a port or memory problem; speculate on 2 rows only.

### 2. Can labels teach a decimal ALU that cannot be enumerated, once the quotient is exposed? (gradient vs. speculative local search)

**Hypothesis.** When q is supplied, the 1,000 T=1 labels contain enough information to pin the 504 entries of the PROD, ADD and SUB tables. The panel's 1.3% plateau would then be a failure of optimization (soft relaxation, frustration from internal digit relabeling), not missing information. I predict that search succeeds and gradient descent does not.

**Spec.**
- **Data:** the 1,000 `time_steps==1` rows of `DATA/train.jsonl`, split by modulus into 800 training and 200 held-out rows. q = ⌊x²/N⌋ is computed offline; it is a diagnostic oracle and is not admissible in a submission. Validation uses `depth_t_1` and `depth_ood_n_t_1` with offline q.
- **Machine:** the verify path in `$SCRATCH/panel/minimal-prior-learned-wiring/shallow_probe.py`: y = SUBscan(schoolbook(x,x), schoolbook(q,N)), K=7, W=15, 1,000 output fields.
- **Arm A, gradient (10 min, MPS):** start from `pop_probe.py` with:
  - 64 batched seeds: 32 identity-init and 32 random-peaked-init;
  - hard-forward straight-through with annealed Gumbel noise;
  - an LSB-first curriculum (digit j switched on after 100·j steps);
  - Adam lr 0.05;
  - population-based training every 300 steps (the top quarter's tables are copied over the bottom quarter, plus N(0, 0.5) logit noise);
  - verified single-entry repair once a seed reaches 10% exact.
- **Arm B, speculative local search (10 min):** hard integer execution only, with 4 random restarts.
  - Each sweep evaluates all ~5.8k single-field alternatives on the 800 rows in one batched pass (3–10 s).
  - It applies the 16 best non-conflicting moves under the triangular fitness F = Σ_rows (length of the correct LSB-first prefix).
  - When nothing improves, it perturbs 20 fields.
- **Arm C, control (5 min):** arm B's search on the full T=1 QX machine without q (`$SCRATCH/panel/critique-learned-wiring/hardsim.py`), on 200 rows.

**Metrics:** hard exact-match on the training and held-out rows; the number of wrong fields; exact-match on both `depth_*_t_1` sets for any 100% solution.

**Success:** any arm reaches 100% on training, held-out and both `depth_*_t_1` sets.

**What each outcome changes:**
- **B succeeds and A fails:** the information is there and gradients are the obstacle. Speculating over table edits becomes a viable learner when credit paths are short, and the open problem becomes an admissible source for the quotient.
- **A succeeds:** the plateau was fixable by gradient. The open problem is the same.
- **Both fail:** decimal arithmetic must be designed or derived from laws. Label-learned semantics stay limited to enumerable cells (SpecBit) and transparent cells (RFS-X).
- **C succeeds:** this overturns the panel's central negative result and should be pursued above everything else.

### 3. RFS-X end-to-end through the evaluator

**Hypothesis.** The pilot's result holds inside the real model and evaluator, including 8-digit OOD moduli and T=32/64.

**Spec.**
- **Harness replication (≈6 min, CPU).** Run `$SCRATCH/synth/pilot_canon_em_v2.py` on `$SCRATCH/panel/critique-gpu-native/states.npz` for 4 seeds × {sequence loss + spread, sequence loss alone, per-digit loss + spread}, 250 steps each. Expected:
  - sequence loss + spread: 4 of 4 exact, 0 of 76 wrong CANON fields, counter correct for 99 of 99 values of T;
  - sequence loss alone: the counter fails;
  - per-digit loss: 0%.
- **Submission (≈15 min training, ≈3 min eval, MPS).** RFS-X as specified in (c):
  - batch 1024, max_steps 1500;
  - Adam lr 0.03 on the CANON logits, with 4 population copies combined by a learned mixture gate.
  
  Use the Experiment 1 manifest with `total_training_time_seconds=900`.
- **Stage 2 (≈1 min).** Replace the setup's designed sign test with CANON's final borrow and certify again.

**Metrics:** `RESULT_JSON` for both profiles; the number of wrong CANON entries; counter correctness for T=1..99 (offline); in-batch exact-match per T band.

**Success:** certified to T=64 on both profiles, CANON exact, and Stage 2 still certified.

**What each outcome changes:**
- **Success:**
  - One label-trained cell provides comparison, normalization, loop control and halting, with exact extrapolation in T and width.
  - That validates RFS-X as the fallback and the rule "train where transparent, tie into opaque roles only after covering their alphabet".
- **Fails at T=1:** a bug in the in-model core or the interface (the harness already works).
- **Fails only at T=32/64 or OOD:** a coverage gap. Find the CANON entries those paths use that the exit never trained.
- **Stage 2 fails:** weight tying needs more coverage than the exit provides.

## (e) Open questions, and how the user's ideas fared

**Open questions.**
1. **What counts as a learned model?**
   - Every certifying design keeps the algorithm in designed structure or in designer losses. The panel's evidence suggests the data budget forces this.
   - Proposed lines:
     - a designer loss with a unique zero is close to hard-coding;
     - label speculation over a designed finite class is Bayesian model selection;
     - gradient training on transparent cells is ordinary ML.
   - Once the independent phase is over, compare with the 3 certified leaderboard entries.
2. **Structure that governs extrapolation is invisible to the labels.** Width and T range need a designed prior or an invariance test (Experiment 1).
3. **Relabeling symmetries as the cause of plateaus.** The internal digit code has an S₉ gauge, and counter tables can be permuted. Fixing the gauge by tying cells across roles is untested.
4. **An admissible quotient source.** A value-size penalty on sandboxed decisions is untested. The log-domain features were chosen by looking at `depth_ood_n_t_64`, which is data inspection; re-validate them on held-out training moduli.
5. **An information budget.**
   - The labels carry about 17 kbits, which in principle is enough to pin every table.
   - Through the reduction, the usable information is roughly log₂ of the number of hypotheses the forward pass evaluates. So far: gate 16 bits from 2 rows, relabeling gauge 15 bits from 4 rows, N′ 13 bits.
6. **Law-only decimal calibration from random init.** Does it land inside the 8! relabeling family? If so, the labels finish the job.
7. **Spectrum point (1), made safe.** A learned micro-program over residue-preserving moves, rewarded only for value size, cannot break correctness by construction. Untested.
8. **Hygiene.**
   - `srt_sim.py` contains `assert … or True`.
   - The hindsight targets are computed in float64 and overflow beyond 2^53.

**The user's ideas.**
- **Speculation / branch prediction.**
  - Across steps it is useless, unanimously.
  - Inside a step it is the main tool, and "width replaces prediction" fixes the OOD leading-digit problem.
  - Its biggest payoff is as a learning mechanism: speculate over hypotheses and let exact checks pick. It is the only mechanism found that gets information through the reduction:
    - the gate, 1 of 65,536, from 2 rows;
    - the relabeling gauge, 1 of 40,320, from 4 rows;
    - N′;
    - candidate selection by sequence likelihood.
- **Temporarily weak connections.**
  - As soft mixing inside the forward pass they fail: congruence breaks, mean-field long division is worse than uniform at its optimum, and straight-through gradients blow up.
  - As mixtures over whole machines or whole candidates that later harden, they work: SpecBit, N′, RFS-X.
  - As a micro-program over residue-preserving moves they are safe by construction but untested.
- **One bit per activation.** Vindicated, for an unexpected reason: identifiability.
  - One shared 3→2 cell is a 16-bit hypothesis. Labels pin it in 2 rows, and law axioms pin it uniquely too.
  - Decimal tables are about 2^1560 in size, and decimal laws leave an 8! gauge.
  - Costs:
    - netlists are deeper (5,839 vs. 4,466 lookup levels at T=64, about 530 kernels for the fold);
    - the binary-to-decimal output conversion is anti-triangular;
    - maximal cell reuse makes gradient learning hopeless.
- **Bitslicing.** Irrelevant for evaluation, which already takes seconds, and for gradient training. Relevant for speculation: packing 64 hypotheses per word is what makes 2^24–2^30 structure hypotheses checkable. Experiment 1 measures this.

## Files (new in this synthesis)

- `/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/synth/enum_gate_by_labels.py` — all 65,536 gates against labeled T=1 rows through the gate-level netlist; 15 s per pass.
- `/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/synth/enum_gate_robust.py` — 16 single rows and 120 random pairs.
- `/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/synth/gauge_check.py` — the decimal relabeling gauge under law-only axioms, and label selection over the 8! relabelings.
- `/private/tmp/claude-501/-Users-chenglou-Library-Application-Support-Claude-scratch-workspaces-2037fc6c-747d-4cf1-a5f5-8a4697a4e390-5ef58abe-646a-4ac9-9fae-d0ea15914ce2-scratch-2026-10-19-536f87/b9b2c6b4-1440-4f5a-b125-efb0abec6f53/scratchpad/synth/pilot_canon_em_v2.py` — the shared-CANON pilot (sequence vs. per-digit loss, with and without spread). Logs are `pilot_*.log` in the same directory.